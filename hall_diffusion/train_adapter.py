"""Train one portable multimodal condition adapter against a frozen EDM2 base.

Run with ``uv run python -m hall_diffusion.train_adapter CONFIG.toml``.
The condition source is intentionally just preprocessed condition arrays; their
numeric normalization is the responsibility of the data producer.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import tomllib

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hall_diffusion import models
from hall_diffusion.adapter_data import ConditionDataset, collate_condition_batch
from hall_diffusion.configuration import resolve_training_config
from hall_diffusion.loss import EDM2Loss
from hall_diffusion.models.adapter_io import save_adapter
from hall_diffusion.models.conditioning import ConditionAdapter, ConditionedEDM2, build_condition_encoder
from hall_diffusion.models.ema import EMA
from hall_diffusion.utils import thruster_data, utils


def _base_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    return path / "checkpoint.pth.tar" if path.is_dir() else path


def load_frozen_base(path: str | Path, device, weights: str = "ema"):
    checkpoint = utils.load_checkpoint(_base_checkpoint(path), device)
    config = models.resolve_model_config(checkpoint["model_config"])
    base = models.from_config(config.copy(), device=device)
    state = checkpoint.get(weights) if weights == "ema" else None
    base.load_state_dict(state if state is not None else checkpoint["model"], strict=True)
    base.requires_grad_(False).eval()
    return base, config


def _loader(dataset, batch_size, shuffle, workers, device, prefetch_factor=2):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_condition_batch,
        pin_memory=device.type == "cuda",
    )
    if workers:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=True)
    return DataLoader(dataset, **kwargs)


def train(config_path: str | Path, device_name: str = "auto"):
    with open(config_path, "rb") as handle:
        config = tomllib.load(handle)
    if "adapter" not in config or "training" not in config:
        raise ValueError("adapter training config requires [adapter] and [training] tables")
    adapter_config = config["adapter"]
    training = resolve_training_config(config["training"])
    device = utils.get_device(device_name)
    base, base_config = load_frozen_base(adapter_config["base_checkpoint"], device, adapter_config.get("base_weights", "ema"))

    settings = models.dataset_settings(base_config)
    directories = training["directories"]
    dataset_kwargs = dict(
        scalars_in_tensor=settings["scalars_in_tensor"], fourier_features=settings["fourier_features"],
        downsample_res=settings["downsample_res"],
    )
    train_base = thruster_data.ThrusterDataset(directories["train_data_dir"], **dataset_kwargs)
    test_base = thruster_data.ThrusterDataset(directories["test_data_dir"], **dataset_kwargs)
    source = adapter_config["condition"]
    train_source = dict(source)
    test_source = dict(source)
    if source.get("type", "files") == "files":
        train_source["directory"] = source["train_directory"]
        test_source["directory"] = source["test_directory"]
    train_data = ConditionDataset(train_base, train_source)
    test_data = ConditionDataset(test_base, test_source)

    name = adapter_config["name"]
    adapter = ConditionAdapter(base, build_condition_encoder(adapter_config["encoder"]), adapter_config.get("channels_per_head"))
    model = ConditionedEDM2(base, {name: adapter}).to(device)
    ema_adapter = copy.deepcopy(adapter).eval().requires_grad_(False)
    optimizer_args = training["optimizer"]
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(), lr=optimizer_args["lr"], betas=tuple(optimizer_args["adam_betas"])
    )
    batch_size = training["batch_size"]
    ema = EMA(
        EMA.calculate_ema_factor(batch_size, len(train_data), training["epochs"], training["ema_epochs"]),
        step_start=EMA.calculate_start_step(batch_size, len(train_data), training["ema_start_epochs"]),
    )
    loss_fn = EDM2Loss(**training["loss"])
    loader_args = dict(
        workers=training["load_workers"],
        device=device,
        prefetch_factor=training["prefetch_factor"],
    )
    train_loader = _loader(train_data, batch_size, True, **loader_args)
    test_loader = _loader(test_data, batch_size, False, **loader_args)
    output_dir = Path(directories["out_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "adapter.pth.tar"

    training_model = utils.compile_model(model, training["torch_compile"])
    for epoch in range(training["epochs"]):
        training_model.train()
        for _, vector, target, condition in tqdm(train_loader, desc=f"adapter epoch {epoch + 1}"):
            target = target.to(device, non_blocking=device.type == "cuda")
            vector = vector.to(device, non_blocking=device.type == "cuda") if base.condition_dim else None
            conditions = {name: condition.to(device, non_blocking=device.type == "cuda")}
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = loss_fn(target, training_model, condition_vec=vector, conditions=conditions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 100.0)
            optimizer.step()
            ema.step_ema(ema_adapter, adapter)

        # Validation deliberately uses the live adapter: it checks the full
        # frozen-base plus adapter path without consuming condition gradients.
        model.eval()
        losses = []
        with torch.no_grad():
            for _, vector, target, condition in test_loader:
                target = target.to(device, non_blocking=device.type == "cuda")
                vector = vector.to(device, non_blocking=device.type == "cuda") if base.condition_dim else None
                conditions = {name: condition.to(device, non_blocking=device.type == "cuda")}
                _, value, _, _ = loss_fn(target, model, condition_vec=vector, conditions=conditions)
                losses.append(value)
        value = sum(losses) / len(losses)
        print(f"epoch {epoch + 1}: validation loss {value:.6g}")
        save_adapter(
            checkpoint_path,
            name=name,
            adapter=adapter,
            adapter_config={"encoder": adapter_config["encoder"], "channels_per_head": adapter_config.get("channels_per_head")},
            base_model_config=base_config,
            ema_state=ema_adapter.state_dict(),
            optimizer_state=optimizer.state_dict(),
            train_config=config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda", "xpu"))
    args = parser.parse_args()
    train(args.config, args.device)
