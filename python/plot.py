import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from pathlib import Path
import argparse
import matplotlib
import tomllib
import math

from utils import utils
from utils.thruster_data import ThrusterDataset

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=Path)
parser.add_argument("--mcmc", type=Path)
parser.add_argument("--ref", type=Path)
parser.add_argument("--num-mcmc", type=int, default=2**14)
parser.add_argument("-o", "--output", type=str, default="_plt.png")
parser.add_argument("-m", "--mode", choices=["traces", "quantiles"], default="quantiles")
parser.add_argument("-f", "--fields", nargs='+', required=True)
parser.add_argument("--observation", type=Path)
parser.add_argument("--type", choices=["sidebyside", "comparison"], default = "comparison")
parser.add_argument("--nolegend", action="store_true")

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 15
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams['font.serif'] = "Computer Modern"


# === Plotting style args ===
# linestyle: (offset, (on, off, ...))
DATA_LINESTYLE = (0, (4.0, 4.0))
DATA_LINE_ARGS = dict(linewidth=2.5, linestyle=DATA_LINESTYLE)
XLABEL = "z (channel lengths)"
OBS_COLOR = "orange"
LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Build quantiles/credible intervals we want to plot
CREDIBLE_INTERVALS = [0.5, 0.95]
QUANTILE_ALPHAS = [0.45, 0.15]
QUANTILES = [0.5]
for ci in CREDIBLE_INTERVALS:
    QUANTILES.append(0.5 - ci/2)
    QUANTILES.append(0.5 + ci/2)

# Discharge channel length for the thruster (TODO: embed this in dataset)
CHANNEL_LENGTH = 0.025


def alpha_blend(color, alpha, background="white"):
    r, g, b, _ = colors.to_rgba(color)
    rb, gb, bb, _ = colors.to_rgba(background)
    r = (1 - alpha) * rb + alpha * r
    g = (1 - alpha) * gb + alpha * g
    b = (1 - alpha) * bb + alpha * b
    return (r, g, b, 1.0)


FIELD_INFO = {
     "B": dict(
        ylabel=r"Field strength (G)",
        title="Magnetic field strength",
        letter_pos = "top right",
        yscalefactor = 10_000,
    ),
    "Tev": dict(
        ylabel=r"Electron temperature (eV)",
        title="Electron temperature",
        letter_pos = "top right",
    ),
    "inv_hall": dict(
        ylabel=r"$\nu_{an}/\omega_{ce}$",
        ylog=True,
        title="Anomalous collision freq.",
        letter_pos = "bottom",
    ),
    "nu_an": dict(
        ylabel=r"Anom. coll. freq (Hz)",
        ylog=True,
        title="Anomalous collision freq.",
        letter_pos = "top right",
    ),
    "ne": dict(
        ylabel=r"Plasma density (m$^{-3}$)",
        ylog=True,
        title="Plasma density",
        letter_pos = "top right",
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
        letter_pos = "top right",
    ),
    "phi": dict(
        ylabel=r"Potential (V)",
        title="Electrostatic potential",
        letter_pos = "top right",
    ),
    "nn": dict(
        ylabel=r"Neutral density (m$^{-3}$)",
        ylog=False,
        title="Neutral density",
        letter_pos = "top right",
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
    samples = dataset.norm.denormalize_tensor(samples_norm)

    return dataset, samples


def plot_quantiles(ax, x, qs, color=None, zorder=0):
    color = "tab:blue" if color is None else color

    # Plot median
    ax.plot(x, qs[0, :], color=color, zorder=zorder + len(CREDIBLE_INTERVALS), label="Median", linewidth=2)

    # Plot credible intervals
    for (i, ci) in enumerate(CREDIBLE_INTERVALS):
        i_q1 = 2 * i + 1
        i_q2 = 2 * i + 2

        ax.fill_between(
            x,
            qs[i_q1, :],
            qs[i_q2, :],
            color=alpha_blend(color, QUANTILE_ALPHAS[i], "white"),
            linewidth=0,
            zorder=zorder+len(CREDIBLE_INTERVALS) - i - 1,
            label=f"{round(ci*100):d}\\% CI",
        )


def plot_traces(ax, x, data, color, zorder=0):
    num_samples = data.shape[0]
    N = 250
    indices = np.arange(num_samples)
    indices = np.random.choice(indices, N)
    alpha = 0.3 / (N / 10)

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
        xlabel = XLABEL

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


def plot_multifield(axes, x, samples, fields, field_names, ref=None, mode="quantiles", observation=None):
    assert (len(axes) == len(fields))

    for (ax, field) in zip(axes, fields):
        y = get_field(field, field_names, samples)

        scale_factor = FIELD_INFO[field].get("yscalefactor", 1.0)
        plot_field(ax, x, y, **FIELD_INFO[field], mode=mode)

        if ref is not None:
            y_ref = get_field(field, field_names, ref)[0, :] * scale_factor
            ax.plot(x, y_ref, color="black", label="Data", **DATA_LINE_ARGS)

        if observation is not None and field in observation["fields"]:
            y_ref_obs = get_field(field, field_names, observation["ref"])[0, :] * scale_factor

            _, x_data, y_data = utils.get_observation_locs(
                observation["fields"],
                field, x*CHANNEL_LENGTH,
                normalizer=observation["data"].norm,
                form="denormalized"
            )

            obs_args = dict(color=OBS_COLOR, label="Observed", zorder=100)

            x_data = x_data / CHANNEL_LENGTH

            if len(x_data) != len(y_ref_obs):
                if y_data is None:
                    y_data = np.interp(x_data, x, y_ref_obs)
                else:
                    y_data = y_data * scale_factor
                ax.scatter(x_data, y_data, **obs_args)
            else:
                ax.plot(x_data, y_ref_obs, **DATA_LINE_ARGS, **obs_args)


def add_letter(ax, field, ind):
    letter_pos = FIELD_INFO[field].get("letter_pos", "")
    ax.annotate(f"({LETTERS[ind]})", **letter_args(letter_pos, 0.05))


def plot_multifield_comparison(args, **kwargs):
    dataset_gen, samples_generated = load_samples(args.samples)
    _, samples_mcmc = load_samples(args.mcmc, num_samples=args.num_mcmc)

    x = dataset_gen.grid / CHANNEL_LENGTH
    field_names = list(dataset_gen.fields())

    row_height = 2.3
    fields = args.fields
    nfields = len(fields)

    fig, axes = plt.subplots(nfields, 2, layout="constrained", figsize=(7, nfields * row_height), dpi=200)

    # Plot fields
    field_args = dict(fields=fields, field_names=field_names)
    plot_multifield(axes[:, 0], x, samples_mcmc, **field_args, **kwargs)
    plot_multifield(axes[:, 1], x, samples_generated, **field_args, **kwargs)

    # Plot configuration
    axes[0,0].set_title(f"MCMC ({args.num_mcmc} samples)")
    axes[0,1].set_title(f"Generated ({samples_generated.shape[0]} samples)")

    # Remove shared labels
    [ax.set(xlabel="", xticklabels=[]) for ax in axes[:-1, :].ravel()]
    [ax.set(ylabel="", yticklabels=[]) for ax in axes[:, 1]]

    # Shared y-limits on horizontal
    for (i, row) in enumerate(axes):
        ymin = min(ax.get_ylim()[0] for ax in row)
        ymax = max(ax.get_ylim()[1] for ax in row)
        for (j, ax) in enumerate(row):
            ax.set(ylim=(ymin, ymax))
            add_letter(ax, fields[i], 2*i + j)

    fig.savefig(args.output)


def plot_sidebyside(args, **kwargs):
    col_width = 3.5
    row_height = 3
    num_fields = len(args.fields)

    num_cols = 2
    num_rows = math.ceil(num_fields / num_cols)
    figsize = (col_width * num_cols, row_height * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, layout="constrained", dpi=200)

    dataset, samples = load_samples(args.samples)
    x = dataset.grid / CHANNEL_LENGTH
    field_names = list(dataset.fields())

    axes_linear = axes.ravel()
    plot_multifield(axes_linear[:num_fields], x, samples, args.fields, field_names, **kwargs)
    [ax.set_axis_off() for ax in axes_linear[num_fields:]]

    for (i, (field, ax)) in enumerate(zip(args.fields, axes_linear)):
        add_letter(ax, field, i)
        if field == "ui_1":
            ax.legend(fontsize=12)

    # Remove xlabels and ticks on all but lat plto
    for (i, row) in enumerate(axes):
        if i < num_rows - 1:
            [ax.set(xlabel="", xticklabels=[]) for ax in row]

    fig.savefig(args.output)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.observation is None:
        observation = None
    else:
        with open(args.observation, "rb") as fp:
            obs_dict = utils.read_observation(tomllib.load(fp)["observation"])
            base_data, samples_ref = load_samples(obs_dict["base_sim"])
            observation = dict(fields=obs_dict["fields"], data=base_data, ref=samples_ref)

    if args.ref is not None:
        _, samples_ref = load_samples(args.ref)
    else:
        samples_ref = None

    common_args = dict(ref=samples_ref, observation=observation, mode=args.mode)

    if args.type == "comparison":
        plot_multifield_comparison(args, **common_args)
    elif args.type == "sidebyside":
        plot_sidebyside(args, **common_args)
    else:
        raise NotImplementedError()





