# Python deps
import argparse
import copy
import itertools
import math
import json
import os
import shutil
import time
import tomllib
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# External deps
import numpy as np
import pandas as pd
from tqdm import tqdm

# Pytorch deps
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Our dependencies
import models
from models.controlnet import ControlNetDenoiser
from models.ema import EMA
from loss import LossFunction
from noise import NoiseSampler
from utils import utils, thruster_data, visualization

DEVICE = utils.get_device()

# Set RNG seed
torch.manual_seed(10)

# Get directory in which script is being run as well as the root directory of the project.
SCRIPT_DIR = utils.get_script_dir()
ROOT_DIR = SCRIPT_DIR / ".."


class StepTimer:
    """Per-section wall-clock profiler with optional CUDA synchronization.

    Usage:
        timer = StepTimer(enabled=True, print_every=50)
        with timer.section("forward"):
            loss = model(x)
        timer.step()   # call once per training step; prints a table every print_every steps

    Each section time is measured with GPU sync so it reflects real compute time, not just
    kernel-dispatch time.  Disable (enabled=False) for zero-overhead operation.
    """

    def __init__(self, enabled: bool = False, print_every: int = 50):
        self.enabled = enabled
        self.print_every = print_every
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._step = 0

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        self._totals[name] += elapsed
        self._counts[name] += 1

    def step(self):
        if not self.enabled:
            return
        self._step += 1
        if self._step % self.print_every == 0:
            self._print_report()

    def _print_report(self):
        if not self._totals:
            return
        accounted = sum(self._totals.values())
        print(f"\n--- StepTimer report  (step {self._step}) ---")
        for name, total in sorted(self._totals.items(), key=lambda kv: -kv[1]):
            avg_ms = 1000.0 * total / self._counts[name]
            pct = 100.0 * total / accounted if accounted > 0 else 0.0
            print(f"  {name:<30s}  {avg_ms:7.2f} ms/step  ({pct:5.1f}%)")
        print(f"  {'total accounted':<30s}  {1000.0 * accounted / self._step:7.2f} ms/step")
        print()


@dataclass
class TrainingState:
    """Bundles all mutable training state: model components and running metrics."""
    model: Any
    ema_model: Any
    optimizer: Any
    ema: Any
    scaler: Any
    val_loss: float = math.inf
    ema_loss: float = math.inf
    best_loss: float = math.inf
    best_params: Any = None
    outlier_inds: list = field(default_factory=list)
    outlier_losses: list = field(default_factory=list)
    outliers: dict = field(default_factory=dict)
    batch_idx: int = -1
    example_idx: int = -1


def learning_rate_schedule(
    cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_batches=0
):
    """Learning rate decay schedule from "Analyzing and Improving the Training Dynamics of Diffusion Models" (EDM2)."""
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_batches > 0:
        lr *= min(cur_nimg / (rampup_batches * batch_size), 1)
    return lr

def validation_loss(
    model,
    loss_fn,
    val_loader,
    visualize=False,
    epoch_idx=0,
    out_folder=Path("."),
    data_dir: Path | str = "data/training",
    ctrl_fn=None,
    seed=None,
) -> float:
    """Evaluate the loss on the validation set. Optionally visualize denoising progress, saving images to out_folder."""
    model.eval()

    # Iterate through dataset and compute the loss on each batch.
    # Fix the RNG for the duration of the loop so the loss is a deterministic function of the
    # model weights for a given seed. Passing the same seed to multiple calls (e.g. EMA model
    # and base model) makes their losses directly comparable.
    rng_state = torch.get_rng_state()
    if seed is not None:
        torch.manual_seed(seed)
    losses = []
    for _, (_, vec, x) in enumerate(val_loader):
        with torch.no_grad():
            x = x.float().to(DEVICE)
            vec = vec.float().to(DEVICE)
            ctrl = ctrl_fn(x) if ctrl_fn is not None else None
            _, loss, *_ = loss_fn(x, model, condition_vec=vec, ctrl=ctrl)
            losses.append(loss)
    torch.set_rng_state(rng_state)
    loss = float(np.mean(losses))

    if visualize:
        # Load first batch with fixed noise to visualize results
        with torch.no_grad():
            _, vec, y = next(iter(val_loader))
            vec = vec.float().to(DEVICE)
            y = y.float().to(DEVICE)
            noise_std = torch.rand((y.shape[0], 1, 1), device=DEVICE)
            fixed_noise = torch.tensor(visualization.NOISE_LEVELS_FOR_PLOTTING, device=DEVICE)
            noise_std[: len(fixed_noise), 0, 0] = fixed_noise

            _, _, noisy_im, denoise_pred = loss_fn(
                y, model, noise_std, condition_vec=vec
            )

            suptitle = f"Epoch: {epoch_idx + 1:04d}, Loss: {loss:.4f}"
            visualization.plot_denoising_2d(
                len(fixed_noise),
                noisy_image=noisy_im.cpu(),
                denoised_prediction=denoise_pred.cpu(),
                ground_truth=y.cpu(),
                title=suptitle,
                folder=out_folder,
            )

            visualization.plot_denoising_1d(
                noisy_im.cpu(),
                denoise_pred.cpu(),
                y.cpu(),
                folder=out_folder,
                data_dir=data_dir,
            )

    return loss

def update_lr(optimizer, batch_idx, batch_size, ref_lr, decay_batches, min_lr):
    """Update the learning rate for all optimizer parameter groups and return the new lr."""
    lr = max(learning_rate_schedule(batch_idx * batch_size, batch_size, ref_lr, decay_batches), min_lr)
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def compute_grad_norm(model):
    """Compute the global gradient norm across all model parameters."""
    grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    norm = torch.concat(grads).norm().item() if grads else 0.0
    if not np.isfinite(norm):
        print(f"Warning: non-finite gradient norm encountered: {norm}")
    return norm


def train_one_batch(y, state, loss_fn, condition_vec=None, use_amp=False, ctrl_fn=None, timer: StepTimer | None = None):
    """Perform one step of the optimization procedure on a batch of data."""
    if timer is None:
        timer = StepTimer(enabled=False)

    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)

    # Transfer conditioning vector and data to device
    with timer.section("data_to_device"):
        y = y.float().to(DEVICE)
        if condition_vec is not None:
            condition_vec = condition_vec.to(DEVICE)

    with timer.section("ctrl_fn"):
        ctrl = ctrl_fn(y) if ctrl_fn is not None else None

    # Compute loss and do backwards pass
    if use_amp:
        with timer.section("forward"):
            with torch.amp.autocast_mode.autocast(DEVICE.type):
                loss, base_loss, *_ = loss_fn(y, state.model, condition_vec=condition_vec, ctrl=ctrl)
        with timer.section("backward"):
            state.scaler.scale(loss).backward()
    else:
        with timer.section("forward"):
            loss, base_loss, *_ = loss_fn(y, state.model, condition_vec=condition_vec, ctrl=ctrl)
        with timer.section("backward"):
            loss.backward()

    with timer.section("grad_norm"):
        grad_norm = compute_grad_norm(state.model)

    with timer.section("optimizer"):
        if use_amp:
            prev_scale = state.scaler.get_scale()
            state.scaler.step(state.optimizer)
            state.scaler.update()
            step_skipped = state.scaler.get_scale() < prev_scale
        else:
            state.optimizer.step()
            step_skipped = False

    if step_skipped:
        print(f"Warning: AMP optimizer step skipped due to inf/nan gradients (grad_norm={grad_norm})")

    with timer.section("ema"):
        state.ema.step_ema(state.ema_model, state.model)

    return loss.item(), base_loss, grad_norm

def save_checkpoint(state, batch_loss, config, train_dataset, checkpoint_file, old_checkpoint, log_file, out_dir, evaluation_iters):
    """Save a training checkpoint and update the diagnostic plot."""
    if batch_loss < state.best_loss:
        state.best_params = state.model.state_dict()
        state.best_loss = batch_loss

    if os.path.exists(checkpoint_file):
        shutil.move(checkpoint_file, old_checkpoint)

    # For ControlNet, save only the adapter weights — the frozen denoiser is
    # re-loaded from base_model at startup, keeping checkpoint sizes small.
    if isinstance(state.model, ControlNetDenoiser):
        model_state = state.model.controlnet.state_dict()
        ema_state = state.ema_model.controlnet.state_dict()
    else:
        model_state = state.model.state_dict()
        ema_state = state.ema_model.state_dict()

    out_dict = dict(
        model=model_state,
        model_config=config["model"],
        train_config=config["training"],
        normalizer=train_dataset.norm,
        optimizer=state.optimizer.state_dict(),
        best=state.best_params,
        ema=ema_state,
    )
    torch.save(utils.paths_to_strings(out_dict), checkpoint_file)
    visualization.plot_training_progress(log_file, out_dir, evaluation_iters, state.outlier_inds, state.outlier_losses)


def train(args):
    config_file = args.config

    # Load config file
    with open(config_file, "rb") as fp:
        config = tomllib.load(fp)

    # ---------------------------------------------
    # Training configuration
    train_args = config["training"]
    max_epochs = train_args["epochs"]
    batch_size = train_args["batch_size"]
    evaluation_iters = train_args["eval_freq"]
    use_amp = train_args.get("use_amp", True)
    ema_factor = train_args.get("ema", None)
    load_workers = train_args.get("load_workers", 2)
    prefetch_factor = train_args.get("prefetch_factor", 4)

    # ---------------------------------------------
    # Directory configuration
    dir_args = train_args["directories"]
    out_dir = Path(dir_args["out_dir"])
    train_data_dir = Path(dir_args["train_data_dir"])
    test_data_dir = Path(dir_args["test_data_dir"])
    os.makedirs(out_dir, exist_ok=True)
    log_file = out_dir / train_args["log_file"]

    # Set up training and test data loaders.
    # For controlnet, dataset settings (scalars_in_tensor, downsample_res) are
    # inherited from the base model's stored config so they don't need to be
    # re-specified in the controlnet toml.
    data_cfg = models.dataset_config(config["model"])
    scalars_in_tensor = data_cfg.get("scalars_in_tensor", False)
    downsample_res = data_cfg.get("downsample_res", None)
    train_dataset = thruster_data.ThrusterDataset(train_data_dir, scalars_in_tensor=scalars_in_tensor, downsample_res=downsample_res)
    test_dataset = thruster_data.ThrusterDataset(test_data_dir, scalars_in_tensor=scalars_in_tensor, downsample_res=downsample_res)

    # Check that normalization is the same between training and test datasets
    assert train_dataset.norm == test_dataset.norm

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
        num_workers=load_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=load_workers,
        prefetch_factor=prefetch_factor,
    )

    # ---------------------------------------------
    # Model configuration
    # Load model from config file and print some summary statistics
    # TODO: is there a need to specify input channels in the TOML file or can they always be inferred?

    config["model"]["in_channels"] = train_dataset.num_fields
    if scalars_in_tensor:
        config["model"]["label_dim"] = 0
    else:
        config["model"]["label_dim"] = train_dataset.num_params

    config["model"]["resolution"] = len(train_dataset.grid)

    model = models.from_config(config["model"], device=DEVICE)

    print(
        "Number of parameters = ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # Set up the exponential moving average
    if ema_factor is None:
        if max_epochs <= 0:
            raise ValueError("Cannot auto-compute EMA beta for infinite training (epochs=0); set ema explicitly in config.")
        total_images = max_epochs * len(train_dataset)
        t_half = 0.05 * total_images  # EDM2 heuristic: half-life = 5% of total training images
        ema_factor = 0.5 ** (batch_size / t_half)
        print(f"Auto-computed EMA beta: {ema_factor:.8f} (t_half={t_half:.3e} images over {max_epochs} epochs)")
    ema = EMA(ema_factor, step_start=2000)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # ---------------------------------------------
    # Optimizer configuration
    opt_args = train_args["optimizer"]
    ref_lr = opt_args["lr"]
    min_lr = opt_args["min_lr"]
    decay_epochs = opt_args["lr_decay_start_epochs"]

    checkpoint_file = out_dir / "checkpoint.pth.tar"
    old_checkpoint = out_dir / "checkpoint_prev.pth.tar"
    betas = tuple(opt_args.get("adam_beta", [0.9, 0.995]))

    decay_batches = decay_epochs * len(train_dataset) // batch_size

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=ref_lr,
        weight_decay=train_args["weight_decay"],
        betas=betas,
    )
    scaler = torch.amp.grad_scaler.GradScaler()

    # ---------------------------------------------
    # Checkpoint configuration
    checkpoint_args = train_args["checkpoints"]
    checkpoint_iters = checkpoint_args["checkpoint_save_freq"]
    load_checkpoint = checkpoint_args["load_checkpoint"] and not args.restart

    # Load checkpoint if found
    if load_checkpoint and os.path.exists(checkpoint_file):
        ckpt = torch.load(checkpoint_file, weights_only=False)

        if isinstance(model, ControlNetDenoiser):
            model.controlnet.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        print("Model loaded from checkpoint")
        optimizer.load_state_dict(ckpt["optimizer"])

        if "ema" in ckpt:
            if isinstance(ema_model, ControlNetDenoiser):
                ema_model.controlnet.load_state_dict(ckpt["ema"])
            else:
                ema_model.load_state_dict(ckpt["ema"])
            ema.step_start = 0
            print("EMA loaded from checkpoint")


    # ---------------------------------------------
    # Print a string of the current epoch, batch, and stage
    def description(epoch_idx, batch_idx, stage):
        return f"Epoch {epoch_idx + 1}, batch {batch_idx + 1} ({stage})"

    # Read log file if it exists, otherwise start fresh
    if log_file.exists() and load_checkpoint:
        df = pd.read_csv(log_file)
    else:
        df = pd.DataFrame()

    if not df.empty:
        start_epoch = int(df['epoch_idx'].iloc[-1])
        state = TrainingState(
            model=model, ema_model=ema_model, optimizer=optimizer, ema=ema, scaler=scaler,
            val_loss=float(df['val_loss'].iloc[-1]),
            ema_loss=float(df['ema_loss'].iloc[-1]),
            batch_idx=int(df['batch_idx'].iloc[-1]),
            example_idx=int(df['example_idx'].iloc[-1]),
        )
    else:
        start_epoch = 0
        state = TrainingState(model=model, ema_model=ema_model, optimizer=optimizer, ema=ema, scaler=scaler)

    ema.step = state.batch_idx  # Ensure EMA step counter is in sync with loaded batch index

    # ---------------------------------------------
    # ControlNet measurement generator (None for vanilla EDM2)
    ctrl_fn = train_dataset.generate_measurements if isinstance(model, ControlNetDenoiser) else None

    # ---------------------------------------------
    # Noise args
    channels = config["model"]["in_channels"]
    resolution = config["model"]["resolution"]
    print(f"{channels=}, {resolution=}")

    noise_sampler = NoiseSampler.from_config(channels, resolution, device=DEVICE, **train_args.get("noise_sampler", {}))
    # ---------------------------------------------
    # Loss function
    loss_fn = LossFunction.from_config(noise_sampler=noise_sampler, **train_args.get("loss", {}))

    # ---------------------------------------------
    # Profiling setup
    timer = StepTimer(enabled=args.profile, print_every=50)

    if args.profile_trace:
        profile_dir = Path(args.profile_trace)
        profile_dir.mkdir(parents=True, exist_ok=True)
        prof_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
            record_shapes=True,
            with_stack=False,
        )
        prof_ctx.__enter__()
        print(f"torch.profiler: trace will be written to {profile_dir}/  (runs for 8 steps then stops)")
    else:
        prof_ctx = None

    # ---------------------------------------------
    # Main training loop
    epoch_range = range(start_epoch, max_epochs + 1) if max_epochs > 0 else itertools.count(start_epoch)
    for epoch_idx in epoch_range:
        progress = tqdm(train_loader)
        data_iter = iter(progress)
        while True:

            with timer.section("data_load"):
                try:
                    filenames, vec, y = next(data_iter)
                except StopIteration:
                    break

            state.batch_idx += 1
            state.example_idx += batch_size

            progress.set_description(description(epoch_idx, state.batch_idx, "Training"))

            _, batch_loss, grad_norm = train_one_batch(y, state, loss_fn, condition_vec=vec, use_amp=use_amp, ctrl_fn=ctrl_fn, timer=timer)
            timer.step()
            if prof_ctx is not None:
                prof_ctx.step()

            lr = update_lr(state.optimizer, state.batch_idx, batch_size, ref_lr, decay_batches, min_lr)

            # Save outliers for later inspection and analysis
            # We check outliers by comparing the batch loss to recent validation losses
            if np.isfinite(state.val_loss) and (batch_loss - state.val_loss) > state.val_loss:
                state.outlier_inds.append(state.batch_idx * batch_size)
                state.outlier_losses.append(batch_loss)
                for f in filenames:
                    state.outliers[f] = state.outliers.get(f, 0) + 1
                sorted_outliers = dict(sorted(state.outliers.items(), key=lambda item: item[1], reverse=True))
                with open(out_dir / "outliers.json", "w") as fd:
                    json.dump(sorted_outliers, fd, indent=4)

            # Evaluate on test set, if it's time to do so
            if state.batch_idx % evaluation_iters == 0:
                progress.set_description(description(epoch_idx, state.batch_idx, "Validating"))
                val_seed = torch.randint(2**31, (1,)).item()
                state.ema_loss = validation_loss(state.ema_model, loss_fn, test_loader, visualize=True, epoch_idx=epoch_idx, out_folder=out_dir, data_dir=test_data_dir, ctrl_fn=ctrl_fn, seed=val_seed)
                state.val_loss = validation_loss(state.model, loss_fn, test_loader, data_dir=test_data_dir, ctrl_fn=ctrl_fn, seed=val_seed)

            progress.set_postfix_str(f"batch_loss={batch_loss:.4f}, val_loss={state.val_loss:.4f}")

            # Update log file
            row = dict(example_idx=state.example_idx, batch_idx=state.batch_idx, epoch_idx=epoch_idx,
                       train_loss=batch_loss, val_loss=state.val_loss, ema_loss=state.ema_loss,
                       grad_norm=grad_norm, learning_rate=lr)
            write_header = not log_file.exists() or log_file.stat().st_size == 0
            pd.DataFrame([row]).to_csv(log_file, mode='a', header=write_header, index=False)

            if state.batch_idx % checkpoint_iters != 0:
                continue

            progress.set_description(description(epoch_idx, state.batch_idx, "Saving"))
            save_checkpoint(state, batch_loss, config, train_dataset, checkpoint_file, old_checkpoint, log_file, out_dir, evaluation_iters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Whether to restart the training procedure, discarding the old checkpoint file. This can also be done by setting load_checkpoint=false in the config file",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable lightweight per-section wall-clock timing. Prints a table every 50 steps.",
    )
    parser.add_argument(
        "--profile-trace",
        metavar="DIR",
        default=None,
        help="Run torch.profiler for ~8 steps and write a TensorBoard/Chrome trace to DIR.",
    )
    args = parser.parse_args()
    train(args)
