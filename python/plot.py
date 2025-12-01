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
parser.add_argument("--type", choices=["sidebyside", "multifield"], default = "multifield")
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

field_info = {
     "B": dict(
        ylabel=r"Field strength (G)",
        title="Magnetic field strength",
        letter_pos = "top",
        yscalefactor = 10_000,
    ),
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
    "nu_an": dict(
        ylabel=r"Anom. coll. freq (Hz)",
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
    mode="quantiles",
    observation=None,
):
    fields_loaded = [get_field(field, field_names, s) for s in samples]
    scale_factor = field_info[field].get("yscalefactor", 1.0)
    if ref is None:
        field_ref = None
    else:
        field_ref = get_field(field, field_names, ref)[0, :] * scale_factor

    for i, (ax, y) in enumerate(zip(axes, fields_loaded)):
        plot_field(ax, x, y, mode=mode, **field_info[field])
        if field_ref is not None:

            ax.plot(x, field_ref, color="black", label="Data", **DATA_LINE_ARGS)

            if observation is not None and field in observation:

                _, x_data, y_data = utils.get_observation_locs(observation, field, x*CHANNEL_LENGTH)
                obs_args = dict(color=OBS_COLOR, label="Observed", zorder=100)

                x_data = x_data / CHANNEL_LENGTH

                if len(x_data) != len(field_ref):
                    if y_data is None:
                        y_data = np.interp(x_data, x, field_ref)
                    else:
                        y_data = y_data * scale_factor
                    ax.scatter(x_data, y_data, **obs_args)
                else:
                    ax.plot(x_data, field_ref, **DATA_LINE_ARGS, **obs_args)


       
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

        if show_legend and i == 0:
            ax.legend()


def plot_multifield_comparison(args):

    if args.ref is not None:
        _, samples_ref = load_samples(args.ref)
    else:
        samples_ref = None

    dataset_gen, samples_generated = load_samples(args.samples)
    _, samples_mcmc = load_samples(args.mcmc, num_samples=args.num_mcmc)

    x = dataset_gen.grid / CHANNEL_LENGTH
    field_names = list(dataset_gen.fields())

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

    if args.observation is None:
        observation = None
    else:
        with open(args.observation, "rb") as fp:
            observation = tomllib.load(fp)["observation"]["fields"]

    for i, field in enumerate(fields):
        plot_comparison(
            axes[i, :],
            x,
            [samples_mcmc, samples_generated],
            field,
            field_names,
            ref=samples_ref,
            show_xlabel=(False if i < nfields - 1 else True),
            show_legend=(True if i == 0 and not args.nolegend else False),
            titles=(titles if i == 0 else None),
            letters=[LETTERS[2 * i], LETTERS[2 * i + 1]],
            mode=args.mode,
            observation=observation,
        )

    fig.savefig(args.output)

def plot_sidebyside(args):

    col_width = 3.5
    num_fields = len(args.fields)

    fig, axes = plt.subplots(1,num_fields, constrained_layout='true', dpi=200, figsize=(col_width * num_fields, 3))

    dataset, samples = load_samples(args.samples)
    x = dataset.grid / CHANNEL_LENGTH
    field_names = list(dataset.fields())


    if args.observation is None:
        observation = None
    else:
        with open(args.observation, "rb") as fp:
            observation = tomllib.load(fp)["observation"]["fields"]

    if args.ref is not None:
        _, samples_ref = load_samples(args.ref)
    else:
        samples_ref = None
    

    for (i, (ax, field)) in enumerate(zip(axes, args.fields)):
        fields_loaded = get_field(field, field_names, samples)

        plot_field(ax, x, fields_loaded, **field_info[field], color='tab:blue')
        ax.set_xticks(range(math.ceil(max(x))))

        if samples_ref is None:
            field_ref = None
        else:
            field_ref = get_field(field, field_names, samples_ref)[0, :] * field_info[field].get(
                "yscalefactor", 1.0
            )

        if field_ref is not None:

            ax.plot(x, field_ref, color="black", label="Data", **DATA_LINE_ARGS)
            ylim = ax.get_ylim()

            if observation is not None and field in observation:
                locs = observation[field].get("locs", "all")
                obs_args = dict(color=OBS_COLOR, label="Observed", zorder=10)
                if isinstance(locs, list):
                    locs = np.array(locs) / CHANNEL_LENGTH
                    y = np.interp(locs, x, field_ref)
                    ax.scatter(locs, y, **obs_args)
                else:
                    ax.plot(x, field_ref, **DATA_LINE_ARGS, **obs_args)
            else:
                ax.plot([-1], [-1], linewidth=2, **obs_args)

            ax.set_ylim(ylim)


        # Add plot letters
        letter_pos = field_info[field].get("letter_pos", "")
        ax.annotate(f"({LETTERS[i]})", **letter_args("top left", 0.05))

        if field == "ui_1":
            ax.legend(fontsize=12)


    fig.savefig(args.output)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.type == "multifield":
        plot_multifield_comparison(args)
    elif args.type == "sidebyside":
        plot_sidebyside(args)
    else:
        raise NotImplementedError()
    




