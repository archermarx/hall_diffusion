from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import thruster_data

# Noise levels at which we plot progress during training
NOISE_LEVELS_FOR_PLOTTING = [0.05, 0.1, 0.5, 0.75]

def plot_training_progress(log_file, out_dir, evaluation_iters, outlier_inds):
    plot_df = pd.read_csv(log_file)
    eval_df = plot_df[plot_df['batch_idx'] % evaluation_iters == 0]

    fig, (ax_loss, ax_grad) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # --- Panel 1: Loss ---
    smoothed = plot_df['train_loss'].rolling(evaluation_iters, min_periods=1, center=True).mean()
    ax_loss.plot(plot_df['example_idx'], plot_df['train_loss'], color="tab:blue", alpha=0.2, linewidth=0.8, label="Train. loss (raw)")
    ax_loss.plot(plot_df['example_idx'], smoothed, color="tab:blue", linewidth=1.5, label="Train. loss (smoothed)")
    ax_loss.plot(eval_df['example_idx'], eval_df['val_loss'], color="black", label="Val. loss")
    ax_loss.plot(eval_df['example_idx'], eval_df['ema_loss'], color="tab:red", linestyle="--", label="Val. loss (EMA)")

    if not eval_df.empty:
        best_val = eval_df['val_loss'].min()
        ax_loss.axhline(best_val, linestyle=":", color="gray", linewidth=1.0)
        ax_loss.annotate(
            f"Best val: {best_val:.4f}",
            xy=(plot_df['example_idx'].iloc[-1], best_val),
            xytext=(-6, 4), textcoords="offset points",
            ha="right", va="bottom", fontsize=8, color="gray",
        )

    for x in outlier_inds:
        ax_loss.axvline(x, color="black", alpha=0.3, linewidth=0.8)
    # Dummy handle so outliers appear in the legend
    if outlier_inds:
        ax_loss.axvline(float("nan"), color="black", alpha=0.3, linewidth=0.8, label="Outliers")

    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlim(plot_df['example_idx'].iloc[0], plot_df['example_idx'].iloc[-1])
    ax_loss.grid(which="both")
    ax_loss.legend(loc="upper right", ncols=2)

    # --- Panel 2: Gradient norm + learning rate ---
    ax_grad.plot(plot_df['example_idx'], plot_df['grad_norm'], color="tab:red", linewidth=0.8, label="Gradient norm")
    ax_grad.set_yscale("log")
    ax_grad.set_ylabel("Gradient norm", color="tab:red")
    ax_grad.tick_params(axis="y", labelcolor="tab:red")
    ax_grad.set_xlabel("Number of examples")
    ax_grad.grid(which="both")

    ax_lr = ax_grad.twinx()
    ax_lr.plot(plot_df['example_idx'], plot_df['learning_rate'], color="black", linestyle="--", linewidth=0.8, label="Learning rate")
    ax_lr.set_yscale("log")
    ax_lr.set_ylabel("Learning rate")
    handles = [*ax_grad.get_legend_handles_labels()[0], *ax_lr.get_legend_handles_labels()[0]]
    labels = [*ax_grad.get_legend_handles_labels()[1], *ax_lr.get_legend_handles_labels()[1]]
    ax_grad.legend(handles, labels, loc="upper right")

    fig.savefig(Path(out_dir) / "loss_prog.png", dpi=200)
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