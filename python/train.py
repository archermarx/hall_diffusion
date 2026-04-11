# Python deps
import argparse
import copy
import itertools
import math
import json
import os
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# External deps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Pytorch deps
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Our dependencies
import models
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

# Noise levels at which we plot progress during training
NOISE_LEVELS_FOR_PLOTTING = [0.05, 0.1, 0.5, 0.75]


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
):
    """Evaluate the loss on the validation set. Optionally visualize denoising progress, saving images to out_folder."""
    model.eval()

    # Iterate through dataset and compute the loss on each batch
    losses = []
    for _, (_, vec, x) in enumerate(val_loader):
        with torch.no_grad():
            x = x.float().to(DEVICE)
            vec = vec.float().to(DEVICE)
            _, loss, *_ = loss_fn(x, model, condition_vec=vec)
            losses.append(loss)
    loss = np.mean(losses)

    if visualize:
        # Load first batch with fixed noise to visualize results
        with torch.no_grad():
            _, vec, y = next(iter(val_loader))
            vec = vec.float().to(DEVICE)
            y = y.float().to(DEVICE)
            noise_std = torch.rand((y.shape[0], 1, 1), device=DEVICE)
            fixed_noise = torch.tensor(NOISE_LEVELS_FOR_PLOTTING, device=DEVICE)
            noise_std[: len(fixed_noise), 0, 0] = fixed_noise

            _, _, noisy_im, denoise_pred = loss_fn(
                y, model, noise_std, condition_vec=vec
            )

            suptitle = f"Epoch: {epoch_idx + 1:04d}, Loss: {loss:.4f}"
            fig, _ = visualize_denoising(
                len(fixed_noise),
                noisy_image=noisy_im.cpu(),
                denoised_prediction=denoise_pred.cpu(),
                ground_truth=y.cpu(),
                title=suptitle,
            )
            fig.savefig(out_folder / "denoise_2d.png")
            plt.close(fig)

            visualize_denoising_1d(
                noisy_im.cpu(),
                denoise_pred.cpu(),
                y.cpu(),
                folder=out_folder,
                data_dir=data_dir,
            )

    return loss


def visualize_denoising(N, noisy_image, denoised_prediction, ground_truth, title=""):
    """Plot noised and denoised tensors for N noise levels."""
    fig, axes = plt.subplots(3, N, constrained_layout=True, figsize=(7, 5.5))

    data = (noisy_image, denoised_prediction, ground_truth)

    if title:
        fig.suptitle(title)

    fig.supxlabel("Noise std. dev")

    titles = [f"$\\sigma = {sigma}$" for sigma in NOISE_LEVELS_FOR_PLOTTING]
    ylabels = ["Noisy", "Denoised", "Original"]

    for irow, (row, dataset) in enumerate(zip(axes, data)):
        for icol, ax in enumerate(row):
            ax.imshow(
                dataset[icol, ...],
                aspect="auto",
                vmin=-1.5,
                vmax=1.5,
                interpolation="none",
                cmap="gray",
            )
            if icol == 0:
                ax.set_ylabel(
                    ylabels[irow], rotation="horizontal", va="center", ha="right"
                )
            if irow == 2:
                ax.set_xlabel(titles[icol])

            ax.set_xticks([])
            ax.set_yticks([])
            for direction in ["top", "left", "bottom", "right"]:
                ax.spines[direction].set_visible(False)

    return fig, axes


def visualize_denoising_1d(
    noisy_image,
    denoised_prediction,
    ground_truth,
    folder=Path("."),
    data_dir: Path | str = "data/training",
):
    """Plot 1D plasma properties with and without noise for a few example simulations."""
    dataset = thruster_data.ThrusterDataset(Path(data_dir), None, 1)
    colors = ["tab:blue", "tab:blue", "black"]
    alphas = [0.25, 1.0, 1.0]
    for i, sigma in enumerate(NOISE_LEVELS_FOR_PLOTTING):
        plotter = thruster_data.ThrusterPlotter1D(
            dataset,
            [noisy_image[i], denoised_prediction[i], ground_truth[i]],
            colors=colors,
            alphas=alphas,
        )
        fig, _ = plotter.plot(
            ["nu_an", "ui_1", "ni_1", "Tev", "phi", "E"], denormalize=True, nrows=2
        )
        fig.savefig(folder / f"denoise_1d_{sigma}.png")
        plt.close(fig)


def update_lr(optimizer, batch_idx, batch_size, ref_lr, decay_batches, min_lr):
    """Update the learning rate for all optimizer parameter groups and return the new lr."""
    lr = max(learning_rate_schedule(batch_idx * batch_size, batch_size, ref_lr, decay_batches), min_lr)
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def compute_grad_norm(model):
    """Compute the global gradient norm across all model parameters."""
    grads = [p.grad.detach().flatten().cpu() for p in model.parameters() if p.grad is not None]
    norm = torch.concat(grads).norm().item() if grads else 0.0
    if not np.isfinite(norm):
        print(f"Non-finite grad norm: {norm}")
    return norm


def train_one_batch(y, state, loss_fn, condition_vec=None, use_amp=False):
    """Perform one step of the optimization procedure on a batch of data."""
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)

    # Transfer conditioning vector and data to device
    y = y.float().to(DEVICE)
    if condition_vec is not None:
        condition_vec = condition_vec.to(DEVICE)

    # Compute loss and do backwards pass
    if use_amp:
        with torch.amp.autocast_mode.autocast(DEVICE.type):
            loss, base_loss, *_ = loss_fn(y, state.model, condition_vec=condition_vec)
        state.scaler.scale(loss).backward()
    else:
        loss, base_loss, *_ = loss_fn(y, state.model, condition_vec=condition_vec)
        loss.backward()

    # Make sure gradients are finite
    # For some reason, this causes strange training dynamics (stair-stepping, grad norms approach zero)
    # for param in state.model.parameters():
    #     if param.grad is not None:
    #         torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

    if use_amp:
        state.scaler.step(state.optimizer)
        state.scaler.update()
    else:
        state.optimizer.step()

    state.ema.step_ema(state.ema_model, state.model)

    return loss.item(), base_loss

def save_checkpoint(state, batch_loss, config, train_dataset, checkpoint_file, old_checkpoint, log_file, out_dir, evaluation_iters):
    """Save a training checkpoint and update the diagnostic plot."""
    if batch_loss < state.best_loss:
        state.best_params = state.model.state_dict()
        state.best_loss = batch_loss

    if os.path.exists(checkpoint_file):
        shutil.move(checkpoint_file, old_checkpoint)

    out_dict = dict(
        model=state.model.state_dict(),
        model_config=config["model"],
        train_config=config["training"],
        normalizer=train_dataset.norm,
        optimizer=state.optimizer.state_dict(),
        best=state.best_params,
        ema=state.ema_model.state_dict(),
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
    ema_factor = train_args["ema"]
    load_workers = train_args.get("load_workers", 2)

    # ---------------------------------------------
    # Directory configuration
    dir_args = train_args["directories"]
    out_dir = Path(dir_args["out_dir"])
    train_data_dir = Path(dir_args["train_data_dir"])
    test_data_dir = Path(dir_args["test_data_dir"])
    os.makedirs(out_dir, exist_ok=True)
    log_file = out_dir / train_args["log_file"]

    # Set up training and test data loaders
    scalars_in_tensor = config["model"].get("scalars_in_tensor", False)
    downsample_res = config["model"].get("downsample_res", None)
    train_dataset = thruster_data.ThrusterDataset(train_data_dir, scalars_in_tensor=scalars_in_tensor, downsample_res=downsample_res)
    test_dataset = thruster_data.ThrusterDataset(test_data_dir, scalars_in_tensor=scalars_in_tensor, downsample_res=downsample_res)

    # Check that normalization is the same between training and test datasets
    assert train_dataset.norm == test_dataset.norm

    pin = DEVICE.type == "cuda"
    print(f"{pin=}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
        num_workers=load_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=load_workers,
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
        model.parameters(),
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

        model.load_state_dict(ckpt["model"])
        print("Model loaded from checkpoint")
        optimizer.load_state_dict(ckpt["optimizer"])

        if "ema" in ckpt:
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
    # Main training loop
    epoch_range = range(start_epoch, max_epochs + 1) if max_epochs > 0 else itertools.count(start_epoch)
    for epoch_idx in epoch_range:
        progress = tqdm(train_loader)
        for filenames, vec, y in progress:
            state.batch_idx += 1
            state.example_idx += batch_size

            progress.set_description(description(epoch_idx, state.batch_idx, "Training"))

            _, batch_loss = train_one_batch(y, state, loss_fn, condition_vec=vec, use_amp=use_amp)
            lr = update_lr(state.optimizer, state.batch_idx, batch_size, ref_lr, decay_batches, min_lr)
            # Without the grad checks, we still hit non-finite grad norms occasionally.
            # Despite that, things seem to keep chugging along -- these must have been the cause of the stair-steps.
            grad_norm = compute_grad_norm(state.model)

            # Exit on encountering a non-finite batch loss
            if not np.isfinite(batch_loss):
                print("Non-finite batch loss detected. Exiting")
                return

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
                state.ema_loss = validation_loss(state.ema_model, loss_fn, test_loader, visualize=True, epoch_idx=epoch_idx, out_folder=out_dir, data_dir=test_data_dir)
                state.val_loss = validation_loss(state.model, loss_fn, test_loader, data_dir=test_data_dir)

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
    args = parser.parse_args()
    train(args)
