from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import thruster_data

# Noise levels at which we plot progress during training
NOISE_LEVELS_FOR_PLOTTING = [0.05, 0.1, 0.5, 1.0]

def plot_training_progress(log_file, out_dir, evaluation_iters, outlier_inds):
    plot_df = pd.read_csv(log_file)
    if "event" in plot_df:
        train_df = plot_df[plot_df["event"] == "train"]
        eval_df = plot_df[plot_df["event"] == "validation"]
    else:
        train_df = plot_df
        eval_df = plot_df[plot_df['batch_idx'] % evaluation_iters == 0]

    fig, (ax_loss, ax_grad) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # --- Panel 1: Loss ---
    smoothed = train_df['train_loss'].rolling(evaluation_iters, min_periods=1, center=True).mean()
    ax_loss.plot(
        train_df['example_idx'], train_df['train_loss'],
        color="tab:blue", alpha=0.2, linewidth=0.8, label="Train. loss (raw)",
    )
    ax_loss.plot(train_df['example_idx'], smoothed, color="tab:blue", linewidth=1.5, label="Train. loss (smoothed)")
    ax_loss.plot(eval_df['example_idx'], eval_df['val_loss'], color="black", label="Val. loss")
    ax_loss.plot(eval_df['example_idx'], eval_df['ema_loss'], color="tab:red", linestyle="--", label="Val. loss (EMA)")

    if not eval_df.empty:
        best_val = eval_df['val_loss'].min()
        ax_loss.axhline(best_val, linestyle=":", color="gray", linewidth=1.0)
        ax_loss.annotate(
            f"Best val: {best_val:.4f}",
            xy=(train_df['example_idx'].iloc[-1], best_val),
            xytext=(-6, 4), textcoords="offset points",
            ha="right", va="bottom", fontsize=8, color="gray",
        )

    for x in outlier_inds:
        ax_loss.axvline(x, color="black", alpha=0.3, linewidth=0.8)
    # Dummy handle so outliers appear in the legend
    if outlier_inds:
        ax_loss.axvline(float("nan"), color="orange", alpha=0.3, linewidth=0.8, label="Outliers")

    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Loss")
    x_min = train_df['example_idx'].iloc[0]
    x_max = train_df['example_idx'].iloc[-1]
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    ax_loss.set_xlim(x_min, x_max)
    ax_loss.grid(which="both")
    ax_loss.legend(loc="upper right", ncols=2)
    ax_loss.tick_params(axis='y', which='both', right=True, labelright=True)

    # --- Panel 2: Gradient norm + learning rate ---
    ax_grad.plot(train_df['example_idx'], train_df['grad_norm'], color="tab:red", linewidth=0.8, label="Gradient norm")
    ax_grad.set_yscale("log")
    ax_grad.set_ylabel("Gradient norm", color="tab:red")
    ax_grad.tick_params(axis="y", labelcolor="tab:red")
    ax_grad.set_xlabel("Number of examples")
    ax_grad.grid(which="both")

    ax_lr = ax_grad.twinx()
    ax_lr.plot(
        train_df['example_idx'], train_df['learning_rate'],
        color="black", linestyle="--", linewidth=0.8, label="Learning rate",
    )
    ax_lr.set_yscale("log")
    ax_lr.set_ylabel("Learning rate")
    handles = [*ax_grad.get_legend_handles_labels()[0], *ax_lr.get_legend_handles_labels()[0]]
    labels = [*ax_grad.get_legend_handles_labels()[1], *ax_lr.get_legend_handles_labels()[1]]
    ax_grad.legend(handles, labels, loc="upper right")

    fig.savefig(Path(out_dir) / "loss_prog.png", dpi=200)
    plt.close(fig)


def plot_condition_diagnostic(
    counts,
    time_s,
    discharge_current_a,
    current_range,
    record_ids,
    title="",
    folder=Path("."),
):
    """Plot rows of raw TLPPs, occupancies, and source current traces."""
    counts = np.asarray(counts)
    time_s = np.asarray(time_s)
    discharge_current_a = np.asarray(discharge_current_a)
    if counts.ndim == 2:
        counts = counts[None, ...]
        time_s = time_s[None, ...]
        discharge_current_a = discharge_current_a[None, ...]
        record_ids = [record_ids]
    else:
        record_ids = list(record_ids)
    if counts.ndim != 3:
        raise ValueError(f"TLPP diagnostic expects (examples, height, width), got {counts.shape}")
    if time_s.ndim != 2 or discharge_current_a.shape != time_s.shape:
        raise ValueError("TLPP diagnostic time and current arrays must have shape (examples, samples)")
    if not (len(record_ids) == len(counts) == len(time_s)):
        raise ValueError("TLPP diagnostic inputs must contain the same number of examples")
    current_min, current_max = current_range
    extent = (current_min, current_max, current_min, current_max)

    row_count = len(counts)
    fig, axes = plt.subplots(
        row_count,
        3,
        figsize=(13, 2.0 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    display_max = max(1.0, float(np.log1p(counts).max()))
    for row, (count_image, times, current, record_id) in enumerate(
        zip(counts, time_s, discharge_current_a, record_ids, strict=True)
    ):
        axes[row, 0].imshow(
            np.log1p(count_image),
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="none",
            cmap="magma",
            vmin=0,
            vmax=display_max,
        )
        axes[row, 0].set_ylabel(r"$I(t)$ [A]")
        axes[row, 0].text(
            -0.36,
            0.5,
            record_id,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=7,
        )

        occupancy = count_image > 0
        axes[row, 1].imshow(
            occupancy,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="none",
            cmap="gray_r",
            vmin=0,
            vmax=1,
        )
        axes[row, 1].set_ylabel(f"Occupied: {occupancy.mean():.2%}")

        steady_state = times >= 1e-3
        if not np.any(steady_state):
            raise ValueError("TLPP diagnostic current traces must extend to at least 1000 us")
        axes[row, 2].plot(
            times[steady_state] * 1e6,
            current[steady_state],
            color="tab:blue",
            linewidth=1.0,
        )
        axes[row, 2].set_ylabel("Current [A]")
        axes[row, 2].grid(True, alpha=0.3)

    axes[0, 0].set_title("TLPP counts (log display)")
    axes[0, 1].set_title("Occupancy")
    axes[0, 2].set_title("Discharge current")
    axes[-1, 0].set_xlabel(r"$I(t-\tau)$ [A]")
    axes[-1, 1].set_xlabel(r"$I(t-\tau)$ [A]")
    axes[-1, 2].set_xlabel(r"Time [$\mu$s]")
    if title:
        fig.suptitle(title)
    fig.savefig(Path(folder) / "condition_diagnostic.png", dpi=200)
    plt.close(fig)


def plot_denoising_2d(N, noisy_image, denoised_prediction, ground_truth, title="", folder=Path(".")):
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

    fig.savefig(folder / "denoise_2d.png")
    plt.close(fig)

def plot_denoising_1d(
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
