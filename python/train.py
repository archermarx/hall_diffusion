# Python deps
import argparse
import copy
import math
import json
import os
import shutil
import tomllib

# External deps
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Pytorch deps
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Our dependencies
import models.edm2
import models.edm
import noise

from models.ema import EMA
from utils import utils, thruster_data

DEVICE = utils.get_device()

# Set RNG seed and save initial seeded RNG state
torch.manual_seed(10)
start_state = torch.get_rng_state()

# Get directory in which script is being run as well as the root directory of the project.
SCRIPT_DIR = utils.get_script_dir()
ROOT_DIR = SCRIPT_DIR / ".."

# Noise levels at which we plot progress during training
NOISE_LEVELS_FOR_PLOTTING = [0.05, 0.1, 0.5, 1.0]


# ----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving the Training Dynamics of Diffusion Models". (EDM2)
def learning_rate_schedule(
    cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_batches=0
):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_batches > 0:
        lr *= min(cur_nimg / (rampup_batches * batch_size), 1)
    return lr


# ----------------------------------------------------------------------------
# Loss function for EDM2 model
# Modified to include first- and second-deriviative losses to hopefully reduce noise
class EDM2Loss:
    def __init__(
        self,
        noise_sampler,
        P_mean=-0.4,
        P_std=1.0,
        sigma_data=0.5,
        include_logvar=False,
        deriv_h=1.0,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.noise_sampler = noise_sampler
        self.deriv_h = deriv_h
        self.include_logvar = include_logvar

    def __call__(
        self,
        x,
        model,
        noise_std=None,
        condition_vec=None,
    ):
        batch_size, _, _ = x.shape
        rnd_normal = torch.randn([batch_size, 1, 1], device=x.device)

        if noise_std is None:
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        else:
            sigma = noise_std

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        noise = self.noise_sampler.sample(batch_size) * sigma

        noisy_im = x + noise

        if self.include_logvar:
            denoised, logvar = model(noisy_im, sigma, condition_vec, return_logvar=True)
        else:
            denoised = model(noisy_im, sigma, condition_vec)
            logvar = torch.tensor(0.0)

        # Base loss
        base_loss = (denoised - x) ** 2

        # "Step size": scale for diff loss

        # Derivative loss
        diff_denoised = torch.diff(denoised)
        diff_images = torch.diff(x)
        diff_loss_1 = (diff_denoised - diff_images) ** 2
        diff_loss_1 = torch.nn.functional.pad(diff_loss_1, (0, 1))

        # Second derivative loss
        diff2_denoised = torch.diff(diff_denoised)
        diff2_images = torch.diff(diff_images)
        diff_loss_2 = (diff2_denoised - diff2_images) ** 2
        diff_loss_2 = torch.nn.functional.pad(diff_loss_2, (1, 1))

        h = self.deriv_h

        total_weight = 1 + 1 / h + 1 / h**2
        loss = (
            weight * (base_loss + diff_loss_1 / h + diff_loss_2 / h**2) / total_weight
        )
        base_loss = loss.mean().item()

        # Weight by homoscedastic uncertainty
        if self.include_logvar:
            loss = loss / logvar.exp() + logvar

        return loss.mean(), base_loss, noisy_im, denoised


# ----------------------------------------------------------------------------
# Evaluate the loss on the validation set.
# The validation set is loaded by `val_loader`
# Optionally, visualize progress on some validation set examples, with images saved to `out_folder`
def validation_loss(
    model,
    loss_fn,
    val_loader,
    visualize=False,
    epoch_idx=0,
    out_folder=Path("."),
    data_dir: Path | str = "data/training",
):
    # Set model into evaluation mode
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


# ----------------------------------------------------------------------------
# Plot noised and denoised tensors for N noise levels
def visualize_denoising(N, noisy_image, denoised_prediction, ground_truth, title=""):
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


# ----------------------------------------------------------------------------
# Plot 1D plasma properties with and without noise for a few example simulations
def visualize_denoising_1d(
    noisy_image,
    denoised_prediction,
    ground_truth,
    folder=Path("."),
    data_dir: Path | str = "data/training",
):
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
        fig, axes = plotter.plot(
            ["nu_an", "ui_1", "ni_1", "Tev", "phi", "E"], denormalize=True, nrows=2
        )
        fig.savefig(folder / f"denoise_1d_{sigma}.png")
        plt.close(fig)


# ----------------------------------------------------------------------------
# Perform one step of the optimization procedure on a batch of data
def train_one_batch(
    y,
    model,
    loss_fn,
    optimizer,
    ema_model,
    ema,
    scaler,
    condition_vec=None,
    use_amp=False,
):
    # Set model to training mode and zero gradients
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Tranfer conditioning vector and data to device
    y = y.float().to(DEVICE)
    if condition_vec is not None:
        condition_vec = condition_vec.to(DEVICE)

    # Compute loss function and do backwards pass
    if use_amp:
        with torch.amp.autocast_mode.autocast(DEVICE.type):
            loss, base_loss, *_ = loss_fn(y, model, condition_vec=condition_vec)

        scaler.scale(loss).backward()
    else:
        loss, base_loss, *_ = loss_fn(y, model, condition_vec=condition_vec)
        loss.backward()

    # Make sure gradients are finite
    # For some reason, this causes strange training dynamics (stair-stepping, grad norms approach zero)
    # for param in model.parameters():
    #     if param.grad is not None:
    #         torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

    # Step optimizer and EMA
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    ema.step_ema(ema_model, model)

    return loss.item(), base_loss


def train(args):
    config_file = args.config

    # Load config gile
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
    train_dataset = thruster_data.ThrusterDataset(train_data_dir)
    test_dataset = thruster_data.ThrusterDataset(test_data_dir)

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
        pin_memory_device=DEVICE.type,
        num_workers=load_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        pin_memory_device=DEVICE.type,
        num_workers=load_workers,
    )

    # ---------------------------------------------
    # Model configuration
    # Load model from config file and print some summary statistics
    architecture = config["model"].get("architecture", "edm2")
    if architecture == "edm2":
        model = models.edm2.Denoiser.from_config(config["model"]).to(DEVICE)
    else:
        model = models.edm.Denoiser.from_config(config["model"]).to(DEVICE)

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
        state = torch.load(checkpoint_file, weights_only=False)
        model.load_state_dict(state["model"])
        print("Model loaded from checkpoint")
        optimizer.load_state_dict(state["optimizer"])

        if "ema" in state:
            ema_model.load_state_dict(state["ema"])
            ema.step_start = 0
            print("EMA loaded from checkpoint")

    # ---------------------------------------------
    # Loss args
    loss_args = train_args["loss"]

    # ---------------------------------------------
    # Print a string of the current epoch, batch, and stage
    def description(epoch_idx, batch_idx, stage):
        return f"Epoch {epoch_idx + 1}, batch {batch_idx + 1} ({stage})"

    # ---------------------------------------------
    # Training setup
    model.train()
    best_training_params = None
    best_training_loss = math.inf

    # Read loss file if it exists, otherwise create it
    header = ",".join(
        [
            "example_idx",
            "batch_idx",
            "epoch_idx",
            "train_loss",
            "val_loss",
            "ema_loss",
            "grad_norm",
            "learning_rate",
        ]
    )

    if (
        log_file.exists()
        and load_checkpoint
        and os.path.getsize(log_file) > len(header) + 1
    ):
        log_file_contents = np.genfromtxt(log_file, skip_header=1, delimiter=",")
        num_examples = list(log_file_contents[:, 0].astype(int))
        example_idx = num_examples[-1]
        batch_idx = int(log_file_contents[-1, 1])
        epoch_idx = int(log_file_contents[-1, 2]) - 1
        train_losses = list(log_file_contents[:, 3])
        val_losses = list(log_file_contents[:, 4])
        ema_losses = list(log_file_contents[:, 5])
        grad_norms = list(log_file_contents[:, 6])
        learning_rates = list(log_file_contents[:, 7])
    else:
        num_examples = []
        example_idx = -1
        batch_idx = -1
        epoch_idx = -1
        val_losses = []
        train_losses = []
        ema_losses = []
        grad_norms = []
        learning_rates = []

    # Info for saving outliers
    outlier_inds = []
    outlier_losses = []
    outlier_batches = []
    outlier_vecs = []
    outliers = {}

    val_loss = val_losses[-1] if len(val_losses) > 0 else np.inf
    ema_loss = ema_losses[-1] if len(ema_losses) > 0 else np.inf

    start_epoch = max_epochs

    # Loss function
    channels = train_dataset.num_fields
    resolution = len(train_dataset.grid)
    print(f"{channels=}, {resolution=}")

    noise_sampler_args = train_args.get("noise_sampler", dict(type="gaussian"))
    train_args["noise_sampler"] = noise_sampler_args

    if noise_sampler_args["type"] == "gaussian":
        noise_sampler = noise.RandomNoise(channels, resolution, device=DEVICE)
    elif noise_sampler_args["type"] == "rbf":
        noise_sampler = noise.RBFKernel(
            channels, resolution, scale=noise_sampler_args["scale"], device=DEVICE
        )
    else:
        raise NotImplementedError()

    loss_fn = EDM2Loss(noise_sampler, **loss_args)

    # ---------------------------------------------
    # Main training loop
    while True:
        epoch_idx += 1
        progress = tqdm(train_loader)
        for filenames, vec, y in progress:
            batch_idx += 1
            example_idx += batch_size

            progress.set_description(description(epoch_idx, batch_idx, "Training"))

            # Compute batch loss and step optimizer
            _, batch_loss = train_one_batch(
                y,
                model,
                loss_fn,
                optimizer,
                ema_model,
                ema,
                scaler,
                condition_vec=vec,
                use_amp=use_amp,
            )
            num_examples.append(example_idx)
            train_losses.append(batch_loss)

            # Update learning rate
            lr = learning_rate_schedule(
                batch_idx * batch_size, batch_size, ref_lr, decay_batches
            )
            lr = max(lr, min_lr)
            learning_rates.append(lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # Compute gradient norm
            grads = [
                param.grad.detach().flatten().cpu()
                for param in model.parameters()
                if param.grad is not None
            ]
            grad_norm = torch.concat(grads).norm().numpy()
            if not np.isfinite(grad_norm):
                # Without the grad checks, we still hit this occasionally.
                # Despite that things seem to keep chugging along.
                # Encountering these occasional non-finite gradients must have been the cause of the stair-steps
                print(f"Non-finite grad norm: {grad_norm}")

            grad_norms.append(grad_norm)

            # Exit on encountering a non-finite batch loss
            if not np.isfinite(batch_loss):
                print("Non-finite batch loss detected. Exiting")
                return

            # Save outliers for later inspection and analysis
            # We check outliers by comparing the batch loss to recent validation losses
            if val_losses and ((batch_loss - val_losses[-1]) > val_losses[-1]):
                outlier_inds.append(batch_idx * batch_size)
                outlier_losses.append(batch_loss)
                outlier_batches.append(y)
                outlier_vecs.append(vec)

                for f in filenames:
                    total = outliers.get(f, 0)
                    outliers[f] = total + 1

                sorted_outliers = dict(
                    sorted(outliers.items(), key=lambda item: item[1], reverse=True)
                )

                with open(out_dir / "outliers.json", "w") as fd:
                    json.dump(sorted_outliers, fd, indent=4)

            # Evaluate on test set, if it's time to do so
            if batch_idx % evaluation_iters == 0:
                progress.set_description(
                    description(epoch_idx, batch_idx, "Validating")
                )
                ema_loss = validation_loss(
                    ema_model,
                    loss_fn,
                    test_loader,
                    visualize=True,
                    epoch_idx=epoch_idx,
                    out_folder=out_dir,
                    data_dir=test_data_dir,
                )
                val_loss = validation_loss(
                    model, loss_fn, test_loader, data_dir=test_data_dir
                )

            val_losses.append(val_loss)
            ema_losses.append(ema_loss)

            progress.set_postfix_str(
                f"batch_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}"
            )

            # Update log file
            with open(log_file, "a") as fd:
                if len(num_examples) == 1:
                    # Print the header
                    print(header, file=fd)

                row = f"{example_idx},{batch_idx},{epoch_idx},{batch_loss},{val_loss},{ema_loss},{grad_norm},{lr}"
                print(row, file=fd)

            # Exit if it's not time to checkpoint. Otherwise, continue to saving.
            if batch_idx % checkpoint_iters != 0:
                continue

            progress.set_description(description(epoch_idx, batch_idx, "Saving"))

            if batch_loss < best_training_loss:
                best_training_params = model.state_dict()
                best_training_loss = batch_loss

            if os.path.exists(checkpoint_file):
                shutil.move(checkpoint_file, old_checkpoint)

            out_dict = dict(
                model=model.state_dict(),
                model_config=config["model"],
                train_config=config["training"],
                optimizer=optimizer.state_dict(),
                best=best_training_params,
                ema=ema_model.state_dict(),
            )

            torch.save(out_dict, checkpoint_file)
            progress.set_description(description(epoch_idx, batch_idx, "Plotting"))

            # Plot diagnostics
            fig, ax_grad = plt.subplots(1, 1, constrained_layout=True)

            ax_grad.autoscale(axis="x", tight=True)
            ax_grad.plot(
                num_examples,
                grad_norms,
                label="Gradient norm",
                zorder=4,
                color="tab:red",
            )
            ax_grad.plot(
                num_examples,
                learning_rates,
                label="Learning rate",
                color="tab:orange",
                linestyle="--",
            )
            ax_grad.set_yscale("log")
            ax_grad.tick_params(axis="y", labelcolor="tab:red")
            ax_grad.set_ylabel("Gradient norm", color="tab:red")

            ax_loss = ax_grad.twinx()
            ax_loss.autoscale(axis="x", tight=True)
            ax_loss.set(xlabel="Number of examples", yscale="log")
            ax_loss.set_ylabel("Loss", color="tab:blue")
            ax_loss.tick_params(axis="y", labelcolor="tab:blue")

            ax_loss.scatter(
                outlier_inds, outlier_losses, label="Outliers", color="black", zorder=8
            )

            ax_loss.plot(num_examples, train_losses, label="Train. loss", zorder=5)
            ax_loss.plot(num_examples, val_losses, label="Val. loss", zorder=6)
            ax_loss.plot(num_examples, ema_losses, label="Val. loss (EMA)", zorder=7)
            ax_loss.grid(which="both")
            ax_loss.legend(loc="upper left", ncols=2).set_zorder(10)
            ax_grad.tick_params(axis="y", labelcolor="tab:red")

            fig.savefig(out_dir / "loss_prog.png", dpi=200)
            plt.close(fig)

        if max_epochs > 0 and epoch_idx >= max_epochs:
            break


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
