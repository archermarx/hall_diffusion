"""Estimate diagonal denoiser-error statistics as a function of noise."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hall_diffusion import models
from hall_diffusion.utils import utils
from hall_diffusion.utils.thruster_data import ThrusterDataset


def estimate_moments(residual_sum, residual_square_sum, count):
    """Return E[e], E[e^2], and Var[e] for e = denoised - clean."""
    mean = residual_sum / count
    mean_square = residual_square_sum / count
    variance = (mean_square - mean.square()).clamp_min(0)
    return mean, mean_square, variance


def plot_process_std(process_variance, noise_levels, channel_names, output_dir):
    """Plot the spatially averaged centered standard deviation per field."""
    field_std = np.sqrt(np.maximum(process_variance.mean(axis=-1), 0))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for channel, name in enumerate(channel_names):
        ax.plot(noise_levels, field_std[:, channel], label=name)
    ax.set(xlabel="Noise std", ylabel="Average process std", xscale="log", yscale="log")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.savefig(output_dir / "process_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_variance(process_variance, noise_levels, grid, channel_names, output_dir):
    """Plot the spatial structure of the centered process standard deviation."""
    process_std = np.sqrt(np.maximum(process_variance, 0))
    positive = process_std[process_std > 0]
    floor = positive.min() if positive.size else np.finfo(float).tiny
    log_std = np.log10(np.maximum(process_std, floor))
    rows = math.ceil(len(channel_names) / min(3, len(channel_names)))
    columns = min(3, len(channel_names))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False)
    image = None
    for channel, ax in enumerate(axes.flat):
        if channel >= len(channel_names):
            ax.axis("off")
            continue
        image = ax.pcolormesh(grid, noise_levels, log_std[:, channel], shading="auto")
        ax.set(title=channel_names[channel], xlabel="Axial position", ylabel="Noise std", yscale="log")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=r"$\log_{10}$ process std")
    fig.savefig(output_dir / "process_std_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bias_subsets(subset_bias, overall_bias, noise_levels, channel_names, output_dir):
    """Show whether each field's signed bias is stable across data subsets."""
    subset_mean = subset_bias.mean(axis=-1)
    overall_mean = overall_bias.mean(axis=-1)
    columns = min(3, len(channel_names))
    rows = math.ceil(len(channel_names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False)
    for channel, ax in enumerate(axes.flat):
        if channel >= len(channel_names):
            ax.axis("off")
            continue
        for subset in range(subset_mean.shape[0]):
            ax.plot(noise_levels, subset_mean[subset, :, channel], alpha=0.65)
        ax.plot(noise_levels, overall_mean[:, channel], color="black", linewidth=2.5, label="Overall")
        ax.set(
            title=channel_names[channel],
            xlabel="Noise std",
            ylabel="Spatially averaged signed bias",
            xscale="log",
        )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.legend()
    fig.savefig(output_dir / "bias_subset_field_mean.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results(
    output_dir,
    residual_sum,
    residual_square_sum,
    subset_residual_sum,
    subset_counts,
    count,
    state_shape,
    noise_levels,
    grid,
    channel_names,
):
    """Finalize, save, and plot the sufficient residual statistics."""
    mean, mean_square, variance = estimate_moments(residual_sum, residual_square_sum, count)
    mean = mean.reshape(len(noise_levels), *state_shape).cpu().numpy()
    mean_square = mean_square.reshape(len(noise_levels), *state_shape).cpu().numpy()
    variance = variance.reshape(len(noise_levels), *state_shape).cpu().numpy()
    subset_bias = (
        (subset_residual_sum / subset_counts[:, None, None])
        .reshape(len(subset_counts), len(noise_levels), *state_shape)
        .cpu()
        .numpy()
    )

    np.savez(
        output_dir / "process_variance.npz",
        process_variance=mean_square,
        centered_variance=variance,
        mean_residual=mean,
        bias_subset_mean=subset_bias,
        bias_subset_count=subset_counts.cpu().numpy(),
        residual_convention=np.array("denoised_minus_clean"),
        variance_convention=np.array("uncentered_mse"),
        noise_levels=noise_levels,
    )
    plot_process_std(variance, noise_levels, channel_names, output_dir)
    plot_spatial_variance(variance, noise_levels, grid, channel_names, output_dir)
    plot_bias_subsets(subset_bias, mean, noise_levels, channel_names, output_dir)


def main(model_dir, data_dir, seed=0, bias_subsets=4):
    device = utils.get_device()
    output_dir = Path(model_dir)
    checkpoint = utils.load_checkpoint(output_dir / "checkpoint.pth.tar", device)
    model_config = checkpoint["model_config"]
    if "label_dim" in model_config:
        model_config["condition_dim"] = model_config.pop("label_dim")
    dataset_settings = models.dataset_settings(model_config)
    model = models.from_config(model_config.copy(), device=device)
    model.load_state_dict(checkpoint["ema"], strict=False)
    model.eval()
    del checkpoint

    dataset = ThrusterDataset(data_dir, **dataset_settings)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
        num_workers=2,
        prefetch_factor=2,
    )
    noise_levels = np.geomspace(1e-3, 100, 26)
    state_shape = dataset[0][2].shape
    state_size = math.prod(state_shape)
    bias_subsets = min(max(int(bias_subsets), 1), len(dataset))
    channel_names = [name for name, _ in sorted(dataset.fields().items(), key=lambda item: item[1])]

    residual_sum = torch.zeros((len(noise_levels), state_size), device=device)
    residual_square_sum = torch.zeros_like(residual_sum)
    subset_residual_sum = torch.zeros((bias_subsets, len(noise_levels), state_size), device=device)
    subset_counts = torch.zeros(bias_subsets, dtype=torch.long, device=device)
    count = 0

    with torch.inference_mode():
        for batch_index, data in enumerate(tqdm(loader)):
            params, clean = data[1].to(device), data[2].to(device)
            subset_indices = (count + torch.arange(clean.shape[0], device=device)) % bias_subsets
            count += clean.shape[0]
            subset_counts += torch.bincount(subset_indices, minlength=bias_subsets)

            for noise_index, sigma in enumerate(noise_levels):
                noisy = clean + sigma * torch.randn_like(clean)
                denoised = model(noisy, torch.as_tensor(sigma, device=device), params)
                residual = (denoised - clean).flatten(start_dim=1)
                residual_sum[noise_index] += residual.sum(dim=0)
                residual_square_sum[noise_index] += residual.square().sum(dim=0)
                for subset in range(bias_subsets):
                    subset_residual_sum[subset, noise_index] += residual[subset_indices == subset].sum(dim=0)

            if (batch_index + 1) % 20 == 0:
                save_results(
                    output_dir,
                    residual_sum,
                    residual_square_sum,
                    subset_residual_sum,
                    subset_counts,
                    count,
                    state_shape,
                    noise_levels,
                    dataset.grid,
                    channel_names,
                )

    save_results(
        output_dir,
        residual_sum,
        residual_square_sum,
        subset_residual_sum,
        subset_counts,
        count,
        state_shape,
        noise_levels,
        dataset.grid,
        channel_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to model directory")
    parser.add_argument("--data-dir", required=True, help="Path to evaluation data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bias-subsets", type=int, default=4)
    args = parser.parse_args()
    main(args.model_dir, args.data_dir, args.seed, args.bias_subsets)
