from model_edm2 import Denoiser2
import shutil
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
import utils
import numpy as np
import math
import copy
import os
from torch.utils.data import DataLoader
from ema import EMA
import matplotlib.pyplot as plt
import json
import tomllib
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(100)
start_state = torch.get_rng_state()

SCRIPT_DIR = utils.get_script_dir()
ROOT_DIR = SCRIPT_DIR / ".."
NOISE_LEVELS = [0.05, 0.1, 0.5, 1.0]


class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, noise_std=None, labels=None, include_logvar=False):
        rnd_normal = torch.randn([images.shape[0], 1, 1], device=images.device)

        if noise_std is None:
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        else:
            sigma = noise_std

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        noisy_im = images + noise
        denoised, logvar = net(noisy_im, sigma, labels, return_logvar=True)

        # Base loss
        base_loss = (denoised - images) ** 2

        # "Step size": scale for diff loss
        h = 0.25

        # Derivative loss
        diff_denoised = torch.diff(denoised)
        diff_images = torch.diff(images)
        diff_loss_1 = (diff_denoised - diff_images) ** 2
        diff_loss_1 = torch.nn.functional.pad(diff_loss_1, (0, 1))

        # Second derivatative loss
        diff2_denoised = torch.diff(diff_denoised)
        diff2_images = torch.diff(diff_images)
        diff_loss_2 = (diff2_denoised - diff2_images) ** 2
        diff_loss_2 = torch.nn.functional.pad(diff_loss_2, (1, 1))

        total_weight = 1 + 1/h + 1/h**2
        loss = weight * (base_loss + diff_loss_1 / h + diff_loss_2 / h**2) / total_weight
        base_loss = loss.mean().item()

        # Weight by homoscedastic uncertainty
        if include_logvar:
            loss = loss / logvar.exp() + logvar

        return loss.mean(), base_loss, noisy_im, denoised


# ----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".


def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_batches=0):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_batches > 0:
        lr *= min(cur_nimg / (rampup_batches * batch_size), 1)
    return lr


def calc_loss(y, model, noise_std=None, condition_vector=None, include_logvar=False):
    loss_fn = EDM2Loss(-1.2, 1.2, 1.0)
    return loss_fn(model, y, noise_std=noise_std, labels=condition_vector, include_logvar=include_logvar)


def validation_loss(model, loader, visualize=False, epoch_idx=0, batch_idx=0, folder=Path(".")):
    model.eval()
    losses = []
    for _, (_, vec, x) in enumerate(loader):
        with torch.no_grad():
            x = x.float().to(device)
            vec = vec.float().to(device)
            _, loss, *_ = calc_loss(x, model, condition_vector=vec)
            losses.append(loss)
    loss = np.mean(losses)

    if visualize:
        # Load first batch with fixed noise to visualize results
        with torch.no_grad():
            _, vec, y = next(iter(loader))
            vec = vec.float().to(device)
            y = y.float().to(device)
            noise_std = torch.rand((y.shape[0], 1, 1), device=device)
            fixed_noise = torch.tensor(NOISE_LEVELS, device=device)
            noise_std[: len(fixed_noise), 0, 0] = fixed_noise

            _, _, noisy_im, denoise_pred = calc_loss(y, model, noise_std, condition_vector=vec)

            suptitle = f"Epoch: {epoch_idx + 1:04d}, Loss: {loss:.4f}"
            fig, _ = visualize_denoising(
                len(fixed_noise),
                noisy_im=noisy_im.cpu(),
                denoise_pred=denoise_pred.cpu(),
                actual=y.cpu(),
                title=suptitle,
            )
            fig.savefig(folder / "denoise_2d.png")
            plt.close(fig)

            visualize_denoising_1d(noisy_im.cpu(), denoise_pred.cpu(), y.cpu(), folder=folder)

    return loss


def visualize_denoising(N, noisy_im, denoise_pred, actual, title=""):
    fig, axes = plt.subplots(3, N, constrained_layout=True, figsize=(7, 5.5))

    data = (noisy_im, denoise_pred, actual)

    if title:
        fig.suptitle(title)

    fig.supxlabel("Noise std. dev")

    titles = [f"$\\sigma = {sigma}$" for sigma in NOISE_LEVELS]
    ylabels = ["Noisy", "Denoised", "Original"]

    for irow, (row, dataset) in enumerate(zip(axes, data)):
        for icol, ax in enumerate(row):
            ax.imshow(dataset[icol, ...], aspect="auto", vmin=-1.5, vmax=1.5, interpolation="none", cmap="gray")
            if icol == 0:
                ax.set_ylabel(ylabels[irow], rotation="horizontal", va="center", ha="right")
            if irow == 2:
                ax.set_xlabel(titles[icol])

            ax.set_xticks([])
            ax.set_yticks([])
            for direction in ["top", "left", "bottom", "right"]:
                ax.spines[direction].set_visible(False)

    return fig, axes


def visualize_denoising_1d(noisy_im, denoise_pred, actual, title="", folder=Path(".")):
    dataset = utils.ThrusterDataset("data/training", None, 1)
    colors = ["tab:blue", "tab:blue", "black"]
    alphas = [0.25, 1.0, 1.0]
    for i, sigma in enumerate(NOISE_LEVELS):
        plotter = utils.ThrusterPlotter1D(
            dataset, [noisy_im[i], denoise_pred[i], actual[i]], colors=colors, alphas=alphas
        )
        fig, axes = plotter.plot(["nu_an", "ui_1", "ni_1", "Tev", "phi", "Id"], denormalize=True, nrows=2)
        fig.savefig(folder / f"denoise_1d_{sigma}.png")
        plt.close(fig)


def train_one_batch(y, model, optimizer, ema_model, ema, scaler, condition_vector=None, include_logvar=False, use_amp=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    y = y.float().to(device)
    if condition_vector is not None:
        condition_vector = condition_vector.to(device)

    if use_amp:
        with torch.amp.autocast_mode.autocast(device.type):
            loss, base_loss, *_ = calc_loss(y, model, condition_vector=condition_vector, include_logvar=include_logvar)

        scaler.scale(loss).backward()
    else:
        loss, base_loss, *_ = calc_loss(y, model, condition_vector=condition_vector, include_logvar=include_logvar)
        loss.backward()


    # Make sure gradients are finite
    for param in model.parameters():
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    ema.step_ema(ema_model, model)

    return loss.item(), base_loss


def train(args):
    config_file = args.config
    with open(config_file, "rb") as fp:
        config = tomllib.load(fp)

    train_args = config["training"]
    folder = Path(train_args["folder"])
    os.makedirs(folder, exist_ok=True)

    ref_lr = train_args["lr"]
    min_lr = train_args["min_lr"]
    decay_epochs = train_args["decay_epochs"]
    max_epochs = train_args["epochs"]
    batch_size = train_args["batch_size"]
    evaluation_iters = train_args["eval_freq"]
    checkpoint_iters = train_args["checkpoint_freq"]
    use_amp = train_args.get("use_amp", True)

    checkpoint_file = folder / "checkpoint.pth.tar"
    old_checkpoint = folder / "checkpoint_prev.pth.tar"

    dim = 1
    train_dataset = utils.ThrusterDataset("data/training", dimension=dim)
    test_dataset = utils.ThrusterDataset("data/val_small", dimension=dim)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
        pin_memory_device=device.type,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        pin_memory_device=device.type,
        num_workers=2,
    )

    decay_batches = decay_epochs * len(train_dataset) // batch_size

    # Load config file
    model = Denoiser2.from_config(config["model"]).to(device)
    print("Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if "adam_beta" in config:
        betas = (config["adam_beta"][0], config["adam_beta"][1])
    else:
        betas = (0.9, 0.995)

    optimizer = optim.Adam(model.parameters(), lr=ref_lr, weight_decay=train_args["weight_decay"], betas=betas)
    scaler = torch.amp.grad_scaler.GradScaler()

    ema = EMA(beta=train_args["ema"], step_start=2000)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # Load checkpoint if found
    if train_args["use_checkpoint"] and os.path.exists(checkpoint_file):
        state = torch.load(checkpoint_file, weights_only=False)
        model.load_state_dict(state["model"])
        print("Model loaded")
        optimizer.load_state_dict(state["optimizer"])

        if "ema" in state:
            ema_model.load_state_dict(state["ema"])
            ema.step_start = 0
            print("EMA loaded")

    model.train()

    # Specify training parameters
    best_training_params = None
    best_training_loss = math.inf

    val_losses = []
    ema_losses = []
    train_losses = []
    grad_norms = []
    learning_rates = []

    def description(epoch_idx, batch_idx, stage):
        return f"Epoch {epoch_idx + 1}, batch {batch_idx + 1} ({stage})"

    batch_idx = -1
    epoch_idx = -1

    outlier_inds = []
    outlier_losses = []
    outlier_batches = []
    outlier_vecs = []
    outliers = {}

    while True:
        epoch_idx += 1
        progress = tqdm(train_loader)
        for filenames, vec, y in progress:
            batch_idx += 1

            progress.set_description(description(epoch_idx, batch_idx, "Training"))

            # Compute batch loss and step optimizer
            _, batch_loss = train_one_batch(
                y,
                model,
                optimizer,
                ema_model,
                ema,
                scaler,
                condition_vector=vec,
                include_logvar=batch_idx < decay_batches,
                use_amp=use_amp,
            )
            train_losses.append(batch_loss)

            # Update learning rate
            lr = learning_rate_schedule(batch_idx * batch_size, batch_size, ref_lr, decay_batches)
            lr = max(lr, min_lr)
            learning_rates.append(lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # Compute gradient norm
            grads = [param.grad.detach().flatten().cpu() for param in model.parameters() if param.grad is not None]
            grad_norm = torch.concat(grads).norm().numpy()
            if not np.isfinite(grad_norm):
                print(f"Non-finite grad norm at {grad_norm}")
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

                sorted_outliers = dict(sorted(outliers.items(), key=lambda item: item[1], reverse=True))

                with open(folder / "outliers.json", "w") as fd:
                    json.dump(sorted_outliers, fd, indent=4)

            # Evaluate on test set, if it's time to do so
            if batch_idx % evaluation_iters == 0:
                progress.set_description(description(epoch_idx, batch_idx, "Validating"))
                ema_loss = validation_loss(
                    ema_model, test_loader, visualize=True, epoch_idx=epoch_idx, batch_idx=batch_idx, folder=folder
                )
                val_loss = validation_loss(model, test_loader)
                val_losses.append(val_loss)
                ema_losses.append(ema_loss)

            progress.set_postfix_str(f"{batch_loss=:.4f}, {val_loss=:.4f}")

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
                optimizer=optimizer.state_dict(),
                best=best_training_params,
                ema=ema_model.state_dict(),
            )

            torch.save(out_dict, checkpoint_file)
            progress.set_description(description(epoch_idx, batch_idx, "Plotting"))

            # Plot diagnostics
            fig, ax_grad = plt.subplots()
            num_examples = batch_size * np.arange(batch_idx + 1)
            num_examples_val = batch_size * np.arange(0, batch_idx + 1, evaluation_iters)

            ax_grad.autoscale(axis="x", tight=True)
            ax_grad.plot(num_examples, grad_norms, label="Gradient norm", zorder=4, color="tab:red")
            ax_grad.plot(num_examples, learning_rates, label="Learning rate", color="tab:orange", linestyle="--")
            ax_grad.set_yscale("log")
            ax_grad.tick_params(axis="y", labelcolor="tab:red")
            ax_grad.set_ylabel("Gradient norm", color="tab:red")

            ax_loss = ax_grad.twinx()
            ax_loss.autoscale(axis="x", tight=True)
            ax_loss.set_xlabel("Number of examples")
            ax_loss.set_ylabel("Loss", color="tab:blue")
            ax_loss.set_yscale("log")
            ax_loss.tick_params(axis="y", labelcolor="tab:blue")
            # ax_loss.set_yscale("symlog")

            ax_loss.scatter(outlier_inds, outlier_losses, label="Outliers", color="black", zorder=8)
            max_epoch_lines = (batch_idx * batch_size) // len(train_dataset)
            for x in np.arange(max_epoch_lines + 1):
                ax_loss.axvline(float(x * len(train_dataset)), color="gray", linestyle="--")

            ax_loss.plot(num_examples, train_losses, label="Training loss", zorder=5)
            ax_loss.plot(num_examples_val, val_losses, label="Validation loss", zorder=6)
            ax_loss.plot(num_examples_val, ema_losses, label="Validation loss (EMA)", zorder=7)
            ax_loss.grid(which="both")
            ax_loss.legend(loc="upper left", ncols=2).set_zorder(10)
            ax_grad.tick_params(axis="y", labelcolor="tab:red")

            fig.tight_layout()
            fig.savefig(folder / "loss_prog.png", dpi=200)
            plt.close(fig)

        if max_epochs > 0 and epoch_idx >= max_epochs - 1:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="config_small.toml")
    args = parser.parse_args()
    train(args)
