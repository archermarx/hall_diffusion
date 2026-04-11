from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_progress(log_file, out_dir, evaluation_iters, outlier_inds, outlier_losses):
    plot_df = pd.read_csv(log_file)
    eval_df = plot_df[plot_df['batch_idx'] % evaluation_iters == 0]

    fig, ax_grad = plt.subplots(1, 1, constrained_layout=True)

    ax_grad.autoscale(axis="x", tight=True)
    ax_grad.plot(plot_df['example_idx'], plot_df['grad_norm'], label="Gradient norm", zorder=4, color="tab:red")
    ax_grad.plot(plot_df['example_idx'], plot_df['learning_rate'], label="Learning rate", color="tab:orange", linestyle="--")
    ax_grad.set_yscale("log")
    ax_grad.tick_params(axis="y", labelcolor="tab:red")
    ax_grad.set_ylabel("Gradient norm", color="tab:red")

    ax_loss = ax_grad.twinx()
    ax_loss.autoscale(axis="x", tight=True)
    ax_loss.set(xlabel="Number of examples", yscale="log")
    ax_loss.set_ylabel("Loss", color="tab:blue")
    ax_loss.tick_params(axis="y", labelcolor="tab:blue")

    ax_loss.scatter(outlier_inds, outlier_losses, label="Outliers", color="black", zorder=8)
    ax_loss.plot(plot_df['example_idx'], plot_df['train_loss'], label="Train. loss", zorder=5)
    ax_loss.plot(eval_df['example_idx'], eval_df['val_loss'], label="Val. loss", zorder=6)
    ax_loss.plot(eval_df['example_idx'], eval_df['ema_loss'], label="Val. loss (EMA)", zorder=7)
    ax_loss.grid(which="both")
    ax_loss.legend(loc="upper left", ncols=2).set_zorder(10)
    ax_grad.tick_params(axis="y", labelcolor="tab:red")

    fig.savefig(Path(out_dir) / "loss_prog.png", dpi=200)
    plt.close(fig)
