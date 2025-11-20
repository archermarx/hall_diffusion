import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import argparse
import pandas as pd
import matplotlib

from utils.thruster_data import ThrusterDataset

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=Path)
parser.add_argument("--mcmc", type=Path)
parser.add_argument("--ref", type=Path)
parser.add_argument("--num-mcmc", type=int, default=2**14)
parser.add_argument("-o", "--output", type=str, default="_plt.png")
parser.add_argument("-m", "--mode", choices=["traces", "quantiles"], default="quantiles")
parser.add_argument("-f", "--fields", nargs='+', required=True)

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 15
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams['font.serif'] = "Computer Modern"

QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

field_info = {
    "Tev": dict(
        ylabel=r"Electron temperature (eV)",
        title="Electron temperature",
        letter_pos = "top",
    ),
    "inv_hall": dict(
        ylabel=r"$\nu_{an}/\omega_{ce}$",
        ylog=True,
        title="Anomalous collision freq.",
        letter_pos = "bottom",
    ),
    "ne": dict(
        ylabel=r"Plasma density (m$^{-3}$)",
        ylog=True,
        title="Plasma density",
        letter_pos = "bottom",
    ),
    "ui_1": dict(
        ylabel=r"Ion velocity (km/s)",
        yscalefactor=1 / 1000,
        title="Ion velocity",
        letter_pos = "top",
    ),
    "E": dict(
        ylabel=r"Electric field (kV/m)",
        yscalefactor=1 / 1000,
        title="Electric field",
        letter_pos = "top",
    ),
    "phi": dict(
        ylabel=r"Potential (V)",
        title="Electrostatic potential",
        letter_pos = "bottom",
    ),
    "nn": dict(
        ylabel=r"Neutral density (m$^{-3}$)",
        ylog=True,
        title="Neutral density",
        letter_pos = "bottom",
    ),
}

def letter_args(pos, pad):
    # default: top left
    x = pad
    y = 1 - pad
    ha = "left"
    va = "top"

    if "top" in pos:
        y = 1 - pad
        va = "top"
    elif "bottom" in pos:
        y = pad
        va = "bottom"

    if "left" in pos:
        x = pad
        ha = "left"
    elif "right" in pos:
        x = 1 - pad
        ha = "right"

    return dict(xy=(x,y), ha=ha, va=va, xycoords="axes fraction", fontsize=25)

def load_samples(sample_dir, num_samples=None):
    dataset = ThrusterDataset(sample_dir)
    indices = np.arange(len(dataset))
    if num_samples is not None:
        indices = np.random.choice(indices, size=num_samples)

    samples_norm = np.array([dataset[i][2] for i in indices])
    samples = dataset.denormalize_tensor(samples_norm)
    
    return dataset, samples

def plot_quantiles(ax, x, qs, color=None, zorder=0):
    color = "tab:blue" if color is None else color
    alpha_95 = 0.25
    alpha_50 = 0.5

    line_95_lo = qs[0, :]
    line_50_lo = qs[1, :]
    line_med = qs[2, :]

    line_50_hi = qs[3, :]
    line_95_hi = qs[4, :]

    ax.plot(x, line_med, color=color, zorder=zorder + 2, label="Median")

    ax.fill_between(
        x,
        line_50_lo,
        line_50_hi,
        color=color,
        alpha=alpha_50,
        linewidth=0,
        zorder=zorder + 1,
        label="50\\% CI",
    )
    ax.fill_between(
        x,
        line_95_lo,
        line_50_lo,
        color=color,
        alpha=alpha_95,
        linewidth=0,
        zorder=zorder,
        label="95\\% CI",
    )
    ax.fill_between(
        x,
        line_50_hi,
        line_95_hi,
        color=color,
        alpha=alpha_95,
        linewidth=0,
        zorder=zorder + 1,
    )

def plot_traces(ax, x, data, color, zorder=0):

    num_samples = data.shape[0]
    N = 100
    indices = np.random.choice(np.arange(num_samples), N)

    #factor = 0.5 * (1 + np.sqrt(1 - 4 * 0.5 / N))
    alpha = 0.5 / (N / 10)

    ax.plot(x, data[indices].T, color=color, alpha=alpha, zorder=zorder)


def plot_field(
    ax,
    x,
    data,
    ylabel,
    ylog=False,
    yscalefactor=1.0,
    xlabel=None,
    color="tab:blue",
    zorder=0,
    mode="quantiles",
    **kwargs,
):
    if xlabel is None:
        xlabel = "Axial location (channel lengths)"

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        yscale="log" if ylog else "linear",
        xlim=(x[0], x[-1]),
    )

    if mode == "quantiles":
        qs = np.quantile(data, QUANTILES, axis=0) * yscalefactor
        assert qs.shape == (len(QUANTILES), data.shape[1])
        plot_quantiles(ax, x, qs, color=color, zorder=zorder)
    else:
        plot_traces(ax, x, data * yscalefactor, color=color, zorder=zorder)


def get_field(field, field_names, data):
    index_dict = {n: i for (i, n) in enumerate(field_names)}

    if field == "inv_hall":
        q_e = 1.6e-19
        m_e = 9.1e-31
        nu_an = data[:, index_dict["nu_an"], :]
        w_ce = data[:, index_dict["B"], :] * q_e / m_e
        return nu_an / w_ce
    else:
        return data[:, index_dict[field], :]


def plot_comparison(
    axes,
    x,
    samples,
    field,
    field_names,
    titles=None,
    ref=None,
    show_legend=True,
    show_xlabel=True,
    letters=None,
    mode="quantiles"
):
    fields_loaded = [get_field(field, field_names, s) for s in samples]
    if ref is None:
        field_ref = None
    else:
        field_ref = get_field(field, field_names, ref)[0, :] * field_info[field].get(
            "yscalefactor", 1.0
        )

    for i, (ax, y) in enumerate(zip(axes, fields_loaded)):
        plot_field(ax, x, y, mode=mode, **field_info[field])
        if field_ref is not None:
            ax.plot(
                x, field_ref, color="black", linewidth=2, label="Data", linestyle="-."
            )

        if not show_xlabel:
            ax.set(xlabel="", xticklabels=[])
        if i > 0:
            ax.set(ylabel="", yticklabels=[])

        if titles is not None:
            ax.set(title=titles[i])

    # Equilize ylims
    ymin = min(ax.get_ylim()[0] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for i, ax in enumerate(axes):
        ax.set_ylim(ymin, ymax)

        # Add plot letters
        if letters is not None:
            letter_pos = field_info[field].get("letter_pos", "")
            ax.annotate(f"({letters[i]})", **letter_args(letter_pos, 0.05))

        if show_legend and i == len(axes) - 1:
            ax.legend()


if __name__ == "__main__":
    args = parser.parse_args()
    
    dataset_gen, samples_generated = load_samples(args.samples)
    dataset_mcmc, samples_mcmc = load_samples(args.mcmc, num_samples=args.num_mcmc)

    if args.ref is not None:
        _, samples_ref = load_samples(args.ref)
    else:
        samples_ref = None


    x = dataset_gen.grid / 0.025
    field_names = list(dataset_gen.fields.keys())


    row_height = 2.3
    fields = args.fields
    nfields = len(fields)

    fig, axes = plt.subplots(
        nfields, 2, layout="constrained", figsize=(7, nfields * row_height), dpi=200
    )

    titles = [
        f"MCMC ({args.num_mcmc} samples)",
        f"Generated ({samples_generated.shape[0]} samples)",
    ]

    letters = "abcdefghijklmnopqrstuvwxyz"

    for i, field in enumerate(fields):
        plot_comparison(
            axes[i, :],
            x,
            [samples_mcmc, samples_generated],
            field,
            field_names,
            ref=samples_ref,
            show_xlabel=(False if i < nfields - 1 else True),
            show_legend=(True if i == 0 else False),
            titles=(titles if i == 0 else None),
            letters=[letters[2 * i], letters[2 * i + 1]],
            mode=args.mode,
        )

    fig.savefig(args.output)
