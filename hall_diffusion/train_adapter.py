"""Train one portable multimodal condition adapter against a frozen EDM2 base.

Run with ``uv run python -m hall_diffusion.train_adapter CONFIG.toml``.
The condition source is intentionally just preprocessed condition arrays; their
numeric normalization is the responsibility of the data producer.
"""

from __future__ import annotations

import argparse
import csv
import copy
import math
from pathlib import Path
import shutil
import tomllib

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hall_diffusion import models
from hall_diffusion.adapter_data import ConditionDataset, HDF5ConditionBatchSampler, collate_condition_batch
from hall_diffusion.configuration import resolve_training_config
from hall_diffusion.loss import EDM2Loss
from hall_diffusion.models.adapter_io import load_adapter, save_adapter
from hall_diffusion.models.conditioning import (
    ConditionAdapter,
    ConditionedEDM2,
    TLPPVAEConditionEncoder,
    build_condition_encoder,
)
from hall_diffusion.models.ema import EMA
from hall_diffusion.utils import thruster_data, utils, visualization


AMP_DTYPE = torch.float16


def _amp_enabled(device, requested):
    """Use the same CUDA-only mixed-precision policy as base-model training."""
    return bool(requested and device.type == "cuda")


def _create_grad_scaler(device, enabled):
    if not _amp_enabled(device, enabled):
        return None
    return torch.amp.GradScaler("cuda", enabled=True)


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
        num_workers=workers,
        collate_fn=collate_condition_batch,
        pin_memory=device.type == "cuda",
    )
    if workers:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=True)
    kwargs["batch_sampler"] = HDF5ConditionBatchSampler(dataset, batch_size, shuffle=shuffle)
    return DataLoader(dataset, **kwargs)


def _condition_source(source: dict, split: str) -> dict:
    """Resolve the split-specific location without changing source semantics."""
    if source.get("type", "hdf5") != "hdf5":
        raise ValueError("adapter condition source type must be 'hdf5'")
    resolved = dict(source)
    source_key = f"{split}_file"
    sorted_key = f"{split}_sorted_file"

    if source_key in source:
        resolved["path"] = source[source_key]
    elif sorted_key in source:
        # A prebuilt sorted product is already a complete condition source.
        # Do not route it through the cache creation path, which requires an
        # unsorted source solely for creating a missing sorted file.
        resolved["path"] = source[sorted_key]
        resolved.pop("sorted_file", None)
        return resolved
    else:
        raise ValueError(
            f"adapter condition source requires {source_key!r} or {sorted_key!r}"
        )

    if sorted_key in source:
        resolved["sorted_file"] = source[sorted_key]
    return resolved


def _rng_devices(device):
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _interval_due(batch_index: int, interval: int) -> bool:
    """Return whether a positive batch interval is due; -1 disables it."""
    if interval == -1:
        return False
    if interval <= 0:
        raise ValueError("batch intervals must be positive or -1")
    return batch_index % interval == 0


def _progress_postfix(loss: float, grad: float, validation: float) -> str:
    """Format progress metrics without changing the rendered line width."""
    return f"loss={loss:10.3e}, grad={grad:10.3e}, val={validation:10.3e}"


def _cache_frozen_vae_means(dataset, encoder, device, batch_size, use_amp, description):
    if not isinstance(encoder, TLPPVAEConditionEncoder):
        raise ValueError("VAE mean caching requires a tlpp_vae condition encoder")
    if not encoder.freeze_vae:
        raise ValueError("VAE mean caching requires freeze_vae=true")

    def encode(conditions):
        conditions = conditions.to(device, non_blocking=device.type == "cuda")
        with torch.inference_mode(), torch.amp.autocast(
            device.type,
            dtype=AMP_DTYPE,
            enabled=use_amp,
        ):
            return encoder.encode_latent(conditions)

    cache = dataset.cache_condition_vectors(encode, batch_size, description)
    size_gib = cache.numel() * cache.element_size() / 2**30
    print(f"Cached {len(dataset):,} VAE means in memory ({size_gib:.2f} GiB)")


def _save_checkpoint(
    path,
    previous_path,
    name,
    adapter,
    adapter_config,
    base_config,
    ema_adapter,
    optimizer,
    config,
    training_state,
):
    if path.exists():
        shutil.move(path, previous_path)
    save_adapter(
        path,
        name=name,
        adapter=adapter,
        adapter_config=adapter_config,
        base_model_config=base_config,
        ema_state=ema_adapter.state_dict(),
        optimizer_state=optimizer.state_dict(),
        train_config=config,
        training_state=training_state,
    )


def _validation_loss(model, loss_fn, loader, adapter_name, base, device, seed):
    """Evaluate with a reproducible noise draw and a sample-weighted mean."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.random.fork_rng(devices=_rng_devices(device)), torch.no_grad():
        torch.manual_seed(seed)
        for _, vector, target, condition in loader:
            target = target.to(device, non_blocking=device.type == "cuda")
            vector = vector.to(device, non_blocking=device.type == "cuda") if base.condition_dim else None
            conditions = {adapter_name: condition.to(device, non_blocking=device.type == "cuda")}
            _, value, _, _ = loss_fn(target, model, condition_vec=vector, conditions=conditions)
            total_loss += value.item() * target.shape[0]
            total_samples += target.shape[0]
    return total_loss / total_samples


def _plot_validation(
    model,
    loss_fn,
    dataset,
    loader,
    adapter_name,
    base,
    device,
    epoch,
    loss,
    seed,
    output_dir,
    diagnostic_type,
    diagnostic_examples,
):
    """Create the standard denoising plots plus TLPP/source diagnostics."""
    model.eval()
    with torch.random.fork_rng(devices=_rng_devices(device)), torch.no_grad():
        torch.manual_seed(seed)
        _, vector, target, condition = next(iter(loader))
        if target.shape[0] < len(visualization.NOISE_LEVELS_FOR_PLOTTING):
            raise ValueError("adapter validation plotting requires at least four examples per batch")
        target = target.to(device, non_blocking=device.type == "cuda")
        vector = vector.to(device, non_blocking=device.type == "cuda") if base.condition_dim else None
        conditions = {adapter_name: condition.to(device, non_blocking=device.type == "cuda")}
        fixed_noise = torch.tensor(visualization.NOISE_LEVELS_FOR_PLOTTING, device=device)
        noise_std = torch.full((target.shape[0], 1, 1), fixed_noise[-1], device=device)
        noise_std[: len(fixed_noise), 0, 0] = fixed_noise
        _, _, noisy, denoised = loss_fn(
            target,
            model,
            noise_std=noise_std,
            condition_vec=vector,
            conditions=conditions,
        )

    title = f"Epoch: {epoch + 1:04d}, Loss: {loss:.4f}"
    visualization.plot_denoising_2d(
        len(fixed_noise),
        noisy_image=noisy.cpu(),
        denoised_prediction=denoised.cpu(),
        ground_truth=target.cpu(),
        title=title,
        folder=output_dir,
    )
    visualization.plot_denoising_1d(
        noisy.cpu(),
        denoised.cpu(),
        target.cpu(),
        folder=output_dir,
        data_dir=dataset.base.dir,
    )

    if diagnostic_type == "tlpp":
        rng = np.random.default_rng(seed)
        example_count = min(diagnostic_examples, len(dataset))
        logical_indices = rng.choice(len(dataset), size=example_count, replace=False).tolist()
        raw_conditions = np.stack([dataset.raw_condition(index) for index in logical_indices])
        base_handle = dataset.base._hdf5_handle()
        time_names = base_handle["time_names"].asstr()[:].tolist()
        time_traces = np.stack(
            [np.asarray(base_handle["time"][dataset.base._indices[index]]) for index in logical_indices]
        )
        time_s = time_traces[:, :, time_names.index("time_s")]
        discharge_current = time_traces[:, :, time_names.index("discharge_current_A")]
        selected_record_ids = [dataset.base.record_ids[index] for index in logical_indices]
        condition_attrs = dataset._hdf5_handle().attrs
        current_range = (float(condition_attrs.get("current_min_A", 0.0), float(condition_attrs.get("current_max_A", 100.0)))
        visualization.plot_condition_diagnostic(
            raw_conditions,
            time_s,
            discharge_current,
            current_range,
            selected_record_ids,
            title=title,
            folder=output_dir,
        )


def train(config_path: str | Path, device_name: str = "auto", restart: bool = False):
    with open(config_path, "rb") as handle:
        config = tomllib.load(handle)
    if "adapter" not in config or "training" not in config:
        raise ValueError("adapter training config requires [adapter] and [training] tables")
    adapter_config = config["adapter"]
    training = resolve_training_config(config["training"])
    device = utils.get_device(device_name)
    use_amp = _amp_enabled(device, training["use_amp"])
    scaler = _create_grad_scaler(device, use_amp and AMP_DTYPE == torch.float16)
    amp_description = f"enabled ({AMP_DTYPE})" if use_amp else "disabled"
    print(f"Selected device: {device}; AMP: {amp_description}")
    base, base_config = load_frozen_base(
        adapter_config["base_checkpoint"], device, adapter_config.get("base_weights", "ema")
    )

    settings = models.dataset_settings(base_config)
    directories = training["directories"]
    dataset_kwargs = dict(
        scalars_in_tensor=settings["scalars_in_tensor"], fourier_features=settings["fourier_features"],
        downsample_res=settings["downsample_res"],
    )
    train_base = thruster_data.ThrusterDataset(directories["train_data_dir"], **dataset_kwargs)
    test_base = thruster_data.ThrusterDataset(directories["test_data_dir"], **dataset_kwargs)
    source = adapter_config["condition"]
    diagnostic_config = adapter_config.get("diagnostics", {})
    diagnostic_type = diagnostic_config.get("type")
    diagnostic_examples = int(diagnostic_config.get("examples", 6))
    if diagnostic_type not in {None, "tlpp"}:
        raise ValueError("adapter diagnostics type must be 'tlpp' when specified")
    if diagnostic_examples < 1:
        raise ValueError("adapter diagnostics examples must be at least one")
    train_source = _condition_source(source, "train")
    test_source = _condition_source(source, "test")
    train_data = ConditionDataset(train_base, train_source)
    test_data = ConditionDataset(test_base, test_source)

    batch_size = training["batch_size"]
    loss_fn = EDM2Loss(**training["loss"])
    output_dir = Path(directories["out_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "adapter.pth.tar"
    previous_checkpoint_path = output_dir / "adapter_prev.pth.tar"
    log_path = output_dir / training.get("log_file", "training.csv")
    evaluation_interval = int(training["eval_freq"])
    if evaluation_interval <= 0:
        raise ValueError("training eval_freq must be positive")
    checkpoint_config = training["checkpoints"]
    checkpoint_interval = int(checkpoint_config["checkpoint_save_freq"])
    _interval_due(1, checkpoint_interval)
    resume = bool(checkpoint_config["load_checkpoint"] and not restart and checkpoint_path.is_file())

    name = adapter_config["name"]
    saved_adapter_config = {
        "encoder": adapter_config["encoder"],
        "channels_per_head": adapter_config.get("channels_per_head"),
    }
    artifact = None
    if resume:
        loaded_name, adapter, artifact = load_adapter(
            checkpoint_path,
            base,
            base_config,
            weights="model",
        )
        if loaded_name != name:
            raise ValueError(f"checkpoint adapter is named {loaded_name!r}, expected {name!r}")
        if artifact["adapter_config"] != saved_adapter_config:
            raise ValueError("checkpoint adapter configuration does not match the training configuration")
    else:
        adapter = ConditionAdapter(
            base,
            build_condition_encoder(adapter_config["encoder"]),
            adapter_config.get("channels_per_head"),
        )

    model = ConditionedEDM2(base, {name: adapter}).to(device)
    cache_vae_means = bool(source.get("cache_vae_means", False))
    if cache_vae_means:
        _cache_frozen_vae_means(
            train_data,
            adapter.encoder,
            device,
            batch_size,
            use_amp,
            "first epoch: caching train VAE means",
        )
        _cache_frozen_vae_means(
            test_data,
            adapter.encoder,
            device,
            batch_size,
            use_amp,
            "first epoch: caching validation VAE means",
        )

    loader_args = dict(
        workers=training["load_workers"],
        device=device,
        prefetch_factor=training["prefetch_factor"],
    )
    train_loader = _loader(train_data, batch_size, True, **loader_args)
    test_loader = _loader(test_data, batch_size, False, **loader_args)

    ema_adapter = copy.deepcopy(adapter).eval().requires_grad_(False)
    if artifact is not None and artifact.get("ema") is not None:
        ema_adapter.load_state_dict(artifact["ema"], strict=True)
    ema_model = ConditionedEDM2(base, {name: ema_adapter}).to(device).eval()
    optimizer_args = training["optimizer"]
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(), lr=optimizer_args["lr"], betas=tuple(optimizer_args["adam_betas"])
    )
    if artifact is not None and artifact.get("optimizer") is not None:
        optimizer.load_state_dict(artifact["optimizer"])
    ema = EMA(
        EMA.calculate_ema_factor(batch_size, len(train_data), training["epochs"], training["ema_epochs"]),
        step_start=EMA.calculate_start_step(batch_size, len(train_data), training["ema_start_epochs"]),
    )
    restored_state = artifact.get("training_state") if artifact is not None else None
    restored_state = restored_state or {}
    if scaler is not None and restored_state.get("grad_scaler") is not None:
        scaler.load_state_dict(restored_state["grad_scaler"])
    batch_index = int(restored_state.get("batch_idx", 0))
    example_index = int(restored_state.get("example_idx", 0))
    start_epoch = int(restored_state.get("epoch_idx", 0))
    last_validation_loss = float(restored_state.get("val_loss", math.nan))
    last_ema_loss = float(restored_state.get("ema_loss", math.nan))
    ema.restore_state(
        restored_state.get("ema_step", batch_index),
        started=restored_state.get("ema_started", artifact is not None and artifact.get("ema") is not None),
    )
    log_fields = (
        "event",
        "example_idx",
        "batch_idx",
        "epoch_idx",
        "train_loss",
        "val_loss",
        "ema_loss",
        "grad_norm",
        "learning_rate",
    )

    training_model = utils.compile_model(model, training["torch_compile"])
    log_mode = "a" if resume and log_path.is_file() else "w"
    write_log_header = log_mode == "w" or log_path.stat().st_size == 0
    with log_path.open(log_mode, newline="", buffering=1) as log_handle:
        log_writer = csv.DictWriter(log_handle, fieldnames=log_fields)
        if write_log_header:
            log_writer.writeheader()
        for epoch in range(start_epoch, training["epochs"]):
            training_model.train()
            batch_losses = []
            progress = tqdm(train_loader, desc=f"adapter epoch {epoch + 1}")
            for _, vector, target, condition in progress:
                target = target.to(device, non_blocking=device.type == "cuda")
                vector = vector.to(device, non_blocking=device.type == "cuda") if base.condition_dim else None
                conditions = {name: condition.to(device, non_blocking=device.type == "cuda")}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, dtype=AMP_DTYPE, enabled=use_amp):
                    loss, _, _, _ = loss_fn(target, training_model, condition_vec=vector, conditions=conditions)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 100.0)
                loss_value, grad_norm_value = torch.stack(
                    (loss.detach().float(), grad_norm.detach().float())
                ).cpu().tolist()
                if scaler is not None:
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    step_skipped = scaler.get_scale() < previous_scale
                else:
                    step_skipped = not math.isfinite(loss_value) or not math.isfinite(grad_norm_value)
                    if not step_skipped:
                        optimizer.step()
                if step_skipped and scaler is None:
                    raise FloatingPointError(
                        f"non-finite adapter training metric: loss={loss_value}, grad_norm={grad_norm_value}"
                    )
                if not step_skipped:
                    ema.step_ema(ema_adapter, adapter)

                batch_index += 1
                example_index += target.shape[0]
                batch_losses.append(loss_value)
                learning_rate = optimizer.param_groups[0]["lr"]
                log_writer.writerow(
                    {
                        "event": "train",
                        "example_idx": example_index,
                        "batch_idx": batch_index,
                        "epoch_idx": epoch,
                        "train_loss": loss_value,
                        "val_loss": math.nan,
                        "ema_loss": math.nan,
                        "grad_norm": grad_norm_value,
                        "learning_rate": learning_rate,
                    }
                )
                progress.set_postfix_str(
                    _progress_postfix(loss_value, grad_norm_value, last_validation_loss)
                )

                if _interval_due(batch_index, evaluation_interval):
                    validation_seed = torch.randint(2**31, (1,)).item()
                    last_validation_loss = _validation_loss(
                        model, loss_fn, test_loader, name, base, device, validation_seed
                    )
                    if ema.started:
                        last_ema_loss = _validation_loss(
                            ema_model,
                            loss_fn,
                            test_loader,
                            name,
                            base,
                            device,
                            validation_seed,
                        )
                        diagnostic_model = ema_model
                        diagnostic_loss = last_ema_loss
                    else:
                        last_ema_loss = math.nan
                        diagnostic_model = model
                        diagnostic_loss = last_validation_loss
                    _plot_validation(
                        diagnostic_model,
                        loss_fn,
                        test_data,
                        test_loader,
                        name,
                        base,
                        device,
                        epoch,
                        diagnostic_loss,
                        validation_seed,
                        output_dir,
                        diagnostic_type,
                        diagnostic_examples,
                    )
                    log_writer.writerow(
                        {
                            "event": "validation",
                            "example_idx": example_index,
                            "batch_idx": batch_index,
                            "epoch_idx": epoch,
                            "train_loss": loss_value,
                            "val_loss": last_validation_loss,
                            "ema_loss": last_ema_loss,
                            "grad_norm": grad_norm_value,
                            "learning_rate": learning_rate,
                        }
                    )
                    log_handle.flush()
                    visualization.plot_training_progress(
                        log_path,
                        output_dir,
                        evaluation_iters=evaluation_interval,
                        outlier_inds=[],
                    )
                    training_model.train()

                if _interval_due(batch_index, checkpoint_interval):
                    progress.set_description(f"adapter epoch {epoch + 1} (saving)")
                    _save_checkpoint(
                        checkpoint_path,
                        previous_checkpoint_path,
                        name,
                        adapter,
                        saved_adapter_config,
                        base_config,
                        ema_adapter,
                        optimizer,
                        config,
                        {
                            "batch_idx": batch_index,
                            "example_idx": example_index,
                            "epoch_idx": epoch,
                            "ema_step": ema.step,
                            "ema_started": ema.started,
                            "val_loss": last_validation_loss,
                            "ema_loss": last_ema_loss,
                            "grad_scaler": scaler.state_dict() if scaler is not None else None,
                        },
                    )
                    progress.set_description(f"adapter epoch {epoch + 1}")

            mean_train_loss = float(np.mean(batch_losses))
            print(
                f"epoch {epoch + 1}: train loss {mean_train_loss:.6g}, "
                f"validation loss {last_validation_loss:.6g}, EMA loss {last_ema_loss:.6g}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--restart", action="store_true", help="ignore any existing adapter checkpoint")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda", "xpu"))
    args = parser.parse_args()
    train(args.config, args.device, restart=args.restart)
