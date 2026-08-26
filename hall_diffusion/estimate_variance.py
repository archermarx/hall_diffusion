"""
Estimate the per-field diffusion model variance as a function of noise, qc(t)
"""
# Stdlibs
import argparse
import tomllib
from pathlib import Path
import os
import shutil
import uuid
import math
from collections.abc import Callable

# Third-party deps
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Local deps
from hall_diffusion import models
from hall_diffusion.models.controlnet import ControlNet
from hall_diffusion.utils import utils
from hall_diffusion.utils.thruster_data import ThrusterDataset

def main(model_dir, data_dir):
    device = utils.get_device()

    # Load model from file
    model_dir = Path(model_dir)
    model_dict = utils.load_checkpoint(model_dir / "checkpoint.pth.tar", device)
    model_config = model_dict["model_config"]
    if "label_dim" in model_config:
        model_config["condition_dim"] = model_config.pop("label_dim")
    model = models.from_config(model_config, device=device)
    model.load_state_dict(model_dict["ema"], strict=False)
    del model_dict

    # Load training dataset
    dataset = ThrusterDataset(data_dir, scalars_in_tensor=False, fourier_features=False)
    batch_size = 512
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type=='cuda',
        num_workers=2,
        prefetch_factor=2,
    )

    # Set up grid of noise levels, logarithmically-spaced
    num_noise_levels = 26
    noise_levels = np.geomspace(1e-3, 100, num_noise_levels)

    prev_proc_var = noise_levels**2 / (1 + noise_levels**2)

    num_channels, num_cells = dataset[0][2].shape
    channel_names = [name for name, _ in sorted(dataset.fields().items(), key=lambda item: item[1])]
    proc_var = torch.zeros((num_noise_levels, num_channels), device=device)

    model.eval()

    n = 0

    plot_interval = 20
    with torch.inference_mode():
        for i_batch, data in enumerate(tqdm(loader)):
            # add noise to samples
            # data: (b,c,n)
            params, x0 = data[1].to(device), data[2].to(device)
            n += x0.shape[0]
            for i, sigma in enumerate(noise_levels):
                # noised data: (b,c,n)
                # denoised data: (b,c,n)
                x = x0 + sigma * torch.randn_like(x0, device=device)
                sigma = torch.tensor(sigma, device=device)
                denoised = model(x, sigma, params)

                # compute residual: (b,c,n)
                residual = (denoised-x0)**2

                # average over cell dimension: (b,c)
                # sum over batch dimension: (c,)
                residual = residual.mean(axis=2).sum(axis=0)

                # accumulate into proc_var
                proc_var[i, :] += residual[:]

            # plot current results
            if i_batch % plot_interval == 0:
                fig, (ax, legend_ax) = plt.subplots(
                    1,
                    2,
                    gridspec_kw={"width_ratios": [4, 1]},
                )
                ax.set(xlabel = "Noise std", ylabel = "Process std", xscale='log', yscale='log')
                ax.plot(noise_levels, np.sqrt(prev_proc_var), color='black', linewidth=2, label="Previous")
                ax.plot(noise_levels, noise_levels, color='black', linewidth=2, label="q(t) $\\propto$ t", linestyle='--')
                for i in range(num_channels):
                    proc_vars = proc_var[:, i].cpu().numpy() / n
                    proc_stds = np.sqrt(proc_vars)
                    line, = ax.plot(noise_levels, proc_stds, label = channel_names[i])

                # Match the legend's top-to-bottom order to the lines' vertical
                # order at the right edge of the plot.
                handles, labels = ax.get_legend_handles_labels()
                ordered = sorted(
                    zip(handles, labels),
                    key=lambda item: item[0].get_ydata()[-1],
                    reverse=True,
                )
                legend_ax.axis("off")
                legend_ax.legend(
                    [handle for handle, _ in ordered],
                    [label for _, label in ordered],
                    loc="center left",
                )
                fig.savefig(model_dir / "process_variance.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
        
                np.savez(model_dir / "process_variance.npz", proc_var.cpu().numpy() / n, noise_levels)

    np.savez(model_dir / "process_variance.npz", proc_var.cpu().numpy() / n, noise_levels)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to model")
    parser.add_argument("--data-dir", type=str, help="Path to data")
    args = parser.parse_args()

    main(args.model_dir, args.data_dir)

