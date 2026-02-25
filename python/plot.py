import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from pathlib import Path
import argparse
import matplotlib
import tomllib
import math
import pandas as pd
import os
import run_mcmc

from utils import utils
from utils.thruster_data import ThrusterDataset

parser = argparse.ArgumentParser()
parser.add_argument("--anom", action="store_true", help="Generate plots of anomalous transport profiles + eigenfuncions")
parser.add_argument("--samples", type=Path)
parser.add_argument("--mcmc", type=Path)
parser.add_argument("--ref", type=Path)
parser.add_argument("--ref-label", type=str, default="Reference")
parser.add_argument("--num-mcmc", type=int, default=2**14)
parser.add_argument("-o", "--output", type=str, default="_plt.png")
parser.add_argument(
    "-m", "--mode", choices=["traces", "quantiles", "curves"], default="quantiles"
)
parser.add_argument("-f", "--fields", nargs="+", required=False)
parser.add_argument("--observation", type=Path)
parser.add_argument(
    "--type", choices=["sidebyside", "comparison"], default="comparison"
)
parser.add_argument("--obs-style", choices=["line", "marker"], default="marker")

parser.add_argument("--nolegend", action="store_true")
parser.add_argument("--rows", type=int, default=2)
parser.add_argument("--vline-loc", type=float)
parser.add_argument("--ref2", type=Path)
parser.add_argument("--ref2-label", type=str)

usetex = False
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": "cambria",
    "font.size": 18,
    "mathtext.fontset": "dejavuserif",
    "text.usetex": usetex,
    "axes.formatter.use_mathtext": False,
    "axes.titlesize": "medium",
})

# === Plotting style args ===
# linestyle: (offset, (on, off, ...))
DATA_LINESTYLE = (0, (4.0, 3.0))
DATA_LINE_ARGS = dict(linewidth=2.5, linestyle=DATA_LINESTYLE)
OBS_COLOR = "orange"
XLABEL = "z (channel lengths)"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
COLORS = ["tab:blue", "tab:orange", "tab:green"]
LEGEND_ARGS = dict(fancybox=False, framealpha=1.0, labelspacing=0.4, borderaxespad=0, borderpad=0.25, handlelength=1.5, edgecolor='black')

# Build quantiles/credible intervals we want to plot
CREDIBLE_INTERVALS = [0.5, 0.95]
QUANTILE_ALPHAS = [0.45, 0.15]
QUANTILES = [0.5]
for ci in CREDIBLE_INTERVALS:
    QUANTILES.append(0.5 - ci / 2)
    QUANTILES.append(0.5 + ci / 2)

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
        letter_pos="top right",
    ),
    "inv_hall": dict(
        ylabel=r"Inverse Hall parameter",
        ylog=True,
        title="Inverse Hall parameter",
        letter_pos="top",
    ),
    "nu_an": dict(
        ylabel=r"Anom. coll. freq (Hz)",
        ylog=True,
        title="Anomalous collision freq.",
        letter_pos="top right",
    ),
    "ne": dict(
        ylabel=r"Plasma density (m$^{-3}$)",
        ylog=True,
        title="Plasma density",
        letter_pos="top right",
    ),
    "ni_all": dict(
        ylabel=r"Ion density (m$^{-3}$)",
        ylog=True,
        title="Ion density",
        letter_pos="top right",
    ),
    "ni_1": dict(
        ylabel=r"Ion density (m$^{-3}$)",
        ylog=True,
        title="Ion density",
        letter_pos="top right",
    ),
    "ni_2": dict(
        ylabel=r"Ion density (m$^{-3}$)",
        ylog=True,
        title="Ion density",
        letter_pos="top right",
    ),
    "ni_3": dict(
        ylabel=r"Ion density (m$^{-3}$)",
        ylog=True,
        title="Ion density",
        letter_pos="top right",
    ),
    "ui_1": dict(
        ylabel=r"Ion velocity (km/s)",
        yscalefactor=1 / 1000,
        title="Ion velocity",
        letter_pos="top",
    ),
    "ui_2": dict(
        ylabel=r"Ion velocity (km/s)",
        yscalefactor=1 / 1000,
        title="Ion velocity",
        letter_pos="top",
    ),
    "ui_3": dict(
        ylabel=r"Ion velocity (km/s)",
        yscalefactor=1 / 1000,
        title="Ion velocity",
        letter_pos="top",
    ),
    "ui_1": dict(
        ylabel=r"Ion velocity (km/s)",
        yscalefactor=1 / 1000,
        title="Ion velocity",
        letter_pos="top",
    ),
    "ue": dict(
        ylabel=r"Electron velocity (km/s)",
        yscalefactor=1 / 1000,
        title="Electron velocity",
        letter_pos="top right",
        ylim = (-30, 5),
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
        ylog=True,
        title="Neutral density",
        letter_pos="top right",
    ),
    "nu_iz": dict(
        ylabel=r"Ionization freq. (Hz)",
        ylog=True,
        title="Ionization rate",
        letter_pos="top right",
    )
}

FILETYPES = [".png", ".pdf", ".svg"]

def savefig(fig, output_file: Path, dpi=200, filetypes: list[str] | None=None):
    if filetypes is None:
        filetypes = FILETYPES

    base, _ = os.path.splitext(output_file)
    for filetype in filetypes:
        fig.savefig(base+filetype, dpi=dpi)

def load_samples(sample_dir, num_samples=None, burn_frac=0.0):
    dataset = ThrusterDataset(sample_dir)
    start_ind = round(burn_frac * len(dataset))
    indices = np.arange(start_ind, len(dataset))
    if num_samples is not None:
        indices = np.random.choice(indices, size=num_samples)

    samples_norm = np.array([dataset[i][2] for i in indices])
    samples = dataset.norm.denormalize_tensor(samples_norm)
    return dataset, samples


def plot_quantiles(ax, x, qs, color=None, zorder=0):
    color = "tab:blue" if color is None else color

    # Plot median
    ax.plot(
        x,
        qs[0, :],
        color=color,
        zorder=zorder + len(CREDIBLE_INTERVALS),
        label="Median",
        linewidth=2,
    )

    # Plot credible intervals
    for i, ci in enumerate(CREDIBLE_INTERVALS):
        i_q1 = 2 * i + 1
        i_q2 = 2 * i + 2

        ax.fill_between(
            x,
            qs[i_q1, :],
            qs[i_q2, :],
            color=alpha_blend(color, QUANTILE_ALPHAS[i], "white"),
            linewidth=0,
            zorder=zorder + len(CREDIBLE_INTERVALS) - i - 1,
            label=f"{round(ci * 100):d}{"\\" if usetex else ""}% CI",
        )


def plot_traces(ax, x, data, color, zorder=0):
    num_samples = data.shape[0]

    if color is None:
        cmap = plt.get_cmap("tab10")
        N = 10
        colors = [cmap(i) for i in np.linspace(0, 1, N)]
        for i in range(N):
            ax.plot(x, data[i].T, zorder=zorder, color=colors[i])
    else:
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
    show_ylabel=True,
    show_title=False,
    **kwargs,
):
    if xlabel is None:
        xlabel = XLABEL

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel if show_ylabel else None,
        title=ylabel if show_title else None,
        yscale="log" if ylog else "linear",
        xlim=(0, x[-1]),
        xticks=range(math.floor(x[-1]) + 1)
    )

    if mode == "quantiles":
        qs = np.quantile(data, QUANTILES, axis=0) * yscalefactor
        assert qs.shape == (len(QUANTILES), data.shape[1])
        plot_quantiles(ax, x, qs, color=color, zorder=zorder)
    elif mode == "traces":
        plot_traces(ax, x, data * yscalefactor, color=color, zorder=zorder)
    else:
        plot_traces(ax, x, data * yscalefactor, color=None, zorder=zorder)


def get_field(field, field_names, data):
    index_dict = {n: i for (i, n) in enumerate(field_names)}

    if "," in field:
        return {f: get_field(f, field_names, data)[f] for f in field.split(",")}
    elif field == "inv_hall":
        q_e = 1.6e-19
        m_e = 9.1e-31
        nu_an = data[:, index_dict["nu_e"], :]
        w_ce = data[:, index_dict["B"], :] * q_e / m_e
        return {field: nu_an / w_ce}
    elif field == "nu_iz":
        nn = data[:, index_dict["nn"], :]
        ne = data[:, index_dict["ne"], :]
        ni_1 = data[:, index_dict["ni_1"], :]
        Te = data[:, index_dict["Tev"], :]
        energy = 1.5 * Te
        rate_coeff_1 = pd.read_csv("rate_coeffs/ionization_Xe0_Xe1.dat", delimiter="\t", skiprows=1)
        rate_coeff_2 = pd.read_csv("rate_coeffs/ionization_Xe0_Xe2.dat", delimiter="\t", skiprows=1)
        rate_coeff_3 = pd.read_csv("rate_coeffs/ionization_Xe0_Xe3.dat", delimiter="\t", skiprows=1)
        kiz_1 = np.interp(energy, rate_coeff_1["Energy (eV)"], rate_coeff_1["Rate coefficient (m^3/s)"])
        kiz_2 = np.interp(energy, rate_coeff_2["Energy (eV)"], rate_coeff_2["Rate coefficient (m^3/s)"])
        kiz_3 = np.interp(energy, rate_coeff_3["Energy (eV)"], rate_coeff_3["Rate coefficient (m^3/s)"])
        return {field: (kiz_1 + kiz_2 + kiz_3) * nn * ni_1 / ne}
    else:
        f = data[:, index_dict[field], :]
        return {field: f}


def plot_multifield(axes, x, samples, fields, field_names, vline_loc=None, ref=None, ref_label=None, ref2=None, ref2_label=None, observation=None, obs_style='marker', **FIELD_PLOT_ARGS):
    assert len(axes) == len(fields)
    obs_args = dict(color=OBS_COLOR, label="Observation", zorder=9)

    # Load and process reference simulation data 
    # TODO: load this before this function
    # TODO: support arbitrary reference simulation data
    refs = []
    lw = 2.0 #DATA_LINE_ARGS["linewidth"]
    if ref is not None:
        refs.append(dict(file=ref, style=dict(label=ref_label, linestyle=DATA_LINESTYLE, linewidth=lw)))
    if ref2 is not None:
        refs.append(dict(file=ref2, style=dict(label=ref2_label, linewidth=lw, zorder=1002, linestyle="-.")))

    for ref in refs:
        if os.path.isdir(ref["file"]):
            ref["dataset"], ref["data"] = load_samples(ref["file"])
        elif os.path.isfile(ref["file"]):
            ref["dataset"], ref["data"] = None, pd.read_csv(ref["file"])
        else:
            raise FileNotFoundError(ref["file"])

    if observation is not None and "nu_an" in observation["fields"]:
        observation["fields"]["inv_hall"] = observation["fields"]["nu_an"]

    for ax, field in zip(axes, fields):
        # Load data from tensor and plot
        field_data = get_field(field, field_names, samples)

        for ifield, (subfield, y) in enumerate(field_data.items()):
            scale_factor = FIELD_INFO[subfield].get("yscalefactor", 1.0)
            plot_field(ax, x, y, color=COLORS[ifield], **FIELD_INFO[subfield], **FIELD_PLOT_ARGS)

            # Plot reference simulations
            for (i, ref) in enumerate(refs):

                if i == 0 and len(field_data.keys()) == 1:
                    ref["style"]["color"] = "black"
                elif i == 1 :
                    ref["style"]["color"] = "red"
                else:
                    ref["style"]["color"] = "black"

                ref_dataset = ref["dataset"]
                ref_data = ref["data"]

                if ref_data is not None and ref_dataset is not None:
                    # Ref data is a normalized tensor
                    ref_field = get_field(subfield, field_names, ref_data)
                    y_ref = ref_field[subfield][0, :] * scale_factor
                    x_ref = ref_dataset.grid / CHANNEL_LENGTH
                    ax.plot(x_ref, y_ref, **ref["style"])
                elif ref_data is not None and subfield in ref_data.columns and (subfield + "_lo") in ref_data.columns and (subfield + "_hi") in ref_data.columns:
                    # Ref data is in a pandas dataframe
                    y_ref = ref_data[subfield].to_numpy() * scale_factor
                    y_lo = ref_data[subfield + "_lo"].to_numpy() * scale_factor
                    y_hi = ref_data[subfield + "_hi"].to_numpy() * scale_factor
                    x_ref = ref_data["z"].to_numpy() / CHANNEL_LENGTH
                    if "linestyle" in ref["style"]:
                        ref["style"].pop("linestyle")
                    ax.plot(x_ref, y_ref, "-o", **ref["style"])
                    ax.fill_between(x_ref, y_lo, y_hi, color = 'gray', alpha=0.4, zorder=0, linewidth=0)
                elif ref_data is not None and subfield in ref_data.columns:
                    # Ref data is in a pandas dataframe
                    y_ref = ref_data[subfield].to_numpy() * scale_factor
                    x_ref = ref_data["z"].to_numpy() / CHANNEL_LENGTH
                    ax.plot(x_ref, y_ref, **ref["style"])
                else:
                    continue

            # Plot observations
            # TODO: treat observations as another ref simulaiton
            if observation is not None and subfield in observation["fields"]:
                y_ref_obs = (
                    get_field(subfield, field_names, observation["ref"])[subfield][0, :] * scale_factor
                )

                _, x_data, y_data = utils.get_observation_locs(
                    observation["fields"],
                    subfield,
                    x * CHANNEL_LENGTH,
                    normalizer=observation["data"].norm,
                    form="denormalized",
                )

                x_data = x_data / CHANNEL_LENGTH

                # Indicate observations either with shaded gray region or with orange dashed line
                if vline_loc is None:
                    if len(x_data) != len(y_ref_obs):
                        if y_data is None:
                            y_data = np.interp(x_data, x, y_ref_obs)
                        else:
                            y_data = y_data * scale_factor

                        if obs_style == 'marker':
                            ax.scatter(x_data, y_data, **obs_args)
                        else:
                            ax.plot(x_data, y_data, **obs_args, linewidth=lw, linestyle = DATA_LINESTYLE)
                    else:
                        ax.plot(x_data, y_ref_obs, linewidth=lw, **obs_args, linestyle=DATA_LINESTYLE)
            
                else:
                    ax.axvspan(vline_loc / CHANNEL_LENGTH, 4, color = (0.9, 0.9, 0.9), label = "Observation")


def add_letter_field(ax, field, ind):
    f = field.split(",")[0]
    letter_pos = FIELD_INFO[f].get("letter_pos", "")
    add_letter(ax, ind, pos=letter_pos)

def add_letter(ax, ind, pos="top left", pad=0.05):
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

    LETTERS = "abcdefghijklmnopqrstuvwxyz"
    args = dict(xy=(x, y), ha=ha, va=va, xycoords="axes fraction", fontsize=25)
    ax.annotate(f"({LETTERS[ind]})", **args)

def get_handles_labels(axes):
    # Find all handles and labels
    handles = []
    labels = []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l

    # Reduce to unique label values
    labels, unique_label_inds = np.unique(np.array(labels), return_index=True)
    handles = np.array(handles)[unique_label_inds]

    # Arrange order
    ind_dict = {label: i for i, label in enumerate(labels)}
    ind_dict["Median"] = -1

    # Re-order
    indices = sorted(range(len(labels)), key=lambda i: ind_dict[labels[i]])
    handles = handles[indices]
    labels = labels[indices]

    return handles, labels

def plot_multifield_comparison(args, **kwargs):
    dataset_gen, samples_generated = load_samples(args.samples)
    _, samples_mcmc = load_samples(args.mcmc, num_samples=args.num_mcmc, burn_frac=0.5)

    x = dataset_gen.grid / CHANNEL_LENGTH
    field_names = list(dataset_gen.fields())

    col_width = 3
    row_height = 3
    fields = args.fields

    num_rows = 2
    num_cols = len(fields)

    fig, axs = plt.subplots(
        num_rows, num_cols, layout="constrained", figsize=(col_width*num_cols, row_height*num_rows), dpi=200
    )

    multifield_args = dict(show_ylabel=False, show_title=True)

    # Plot fields
    plot_multifield(axs[0, :], x, samples_mcmc, fields=fields, field_names=field_names, **kwargs, **multifield_args) # type: ignore
    plot_multifield(axs[1, :], x, samples_generated, fields=fields, field_names=field_names, **kwargs, **multifield_args) # type: ignore

    # Remove redundant titles and xlabels
    for (i, row) in enumerate(axs):
        if i < num_rows-1:
            [ax.set(xlabel="", xticklabels="") for ax in row]
        if i > 0:
            [ax.set(title="") for ax in row]

    # Add legend
    handles, labels = get_handles_labels(axs)
    axs[-1,0].legend(handles, labels, fontsize=12, **LEGEND_ARGS)

    # Shared y-limits per field and plot numbering
    for i in range(num_cols):
        ymin = min(ax.get_ylim()[0] for ax in axs[:,i])
        ymax = max(ax.get_ylim()[1] for ax in axs[:,i])
        ylim = FIELD_INFO[fields[i].split(",")[0]].get("ylim",(ymin, ymax))
        for (j, ax) in enumerate(axs[:, i]):
            ax.set(ylim=(ylim))
            add_letter_field(ax, fields[i], i + num_cols * j)

    ylabel_args = dict(fontsize=25)
    axs[0, 0].set_ylabel("MCMC", **ylabel_args)
    axs[1, 0].set_ylabel("Diffusion", **ylabel_args)

    savefig(fig, args.output)


def plot_sidebyside(args, **kwargs):
    col_width = 3.6
    row_height = 3
    num_fields = len(args.fields)

    num_rows = args.rows
    num_cols = math.ceil(num_fields / num_rows)
    figsize = (col_width * num_cols, row_height * num_rows + 0.5)
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=figsize, layout="constrained", dpi=200, squeeze=False
    )
    axes_linear = axs.ravel()

    multifield_args = dict(show_ylabel=True, show_title=False)

    dataset, samples = load_samples(args.samples)
    x = dataset.grid / CHANNEL_LENGTH
    field_names = list(dataset.fields())

    # Plot required fields on multiple axes
    plot_multifield(axes_linear[:num_fields], x, samples, args.fields, field_names, **multifield_args, **kwargs)
    
    # Hide unused axes
    [ax.set_axis_off() for ax in axes_linear[num_fields:]]

    # Add plot letters and collect handles
    for i, (field, ax) in enumerate(zip(args.fields, axes_linear)):
        f = field.split(",")[0]
        add_letter_field(ax, f, i)
        ylim = FIELD_INFO[f].get("ylim", ax.get_ylim())
        ax.set(ylim=ylim)
        
    # Add legend
    handles, labels = get_handles_labels(axs)

    # TODO: make legend pos relocatable
    axes_linear[0].legend(handles, labels, fontsize=13, **LEGEND_ARGS)

    # Remove xlabels and ticks on all but last row of plots
    for (i, row) in enumerate(axs):
        if i < num_rows - 1:
            [ax.set(xlabel="", xticklabels=[]) for ax in row]

    savefig(fig, args.output)

def plot_anom(args):
    col_width = 3.6
    row_height = 3.5

    figsize = (col_width * 2, row_height)
    fig, axs = plt.subplots(1, 2, figsize=figsize, layout="constrained", dpi=200)

    L_ch = CHANNEL_LENGTH
    domain = (0.0, 0.08)
    N = 128
    grid, basis_functions = run_mcmc.setup_grid(domain, N, L_ch)
    grid_norm = grid / L_ch

    eig_ind = 0
    anom_ind = 1
    ax_eig = axs[0]
    ax_anom = axs[1]

    ax_anom.set(yscale="log", xticks=range(math.ceil(grid_norm[-1])), xlabel="z (channel lengths)", ylabel = "Inverse Hall parameter")

    num_samples = 3
    for i in range(num_samples):
        params = run_mcmc.sample_parameters()
        f_base, f_withnoise = run_mcmc.anom_model_with_noise(grid, params, basis_functions, L_ch)
        ax_anom.plot(grid_norm, f_base, color=COLORS[i])
        ax_anom.plot(grid_norm, f_withnoise, color=COLORS[i], linestyle="-.")

    add_letter(ax_anom, anom_ind, "bottom right")
    add_letter(ax_eig, eig_ind, "bottom right")

    ax_eig.set(ylim=(-1,1), xlabel="z (channel lengths)", ylabel="$\\phi(z)$")
    for (i, b) in enumerate(basis_functions.T):
        ax_eig.plot(grid_norm, b, label = f"$\\phi_{i}$")

    legend_args = LEGEND_ARGS.copy()
    legend_args.update(loc="upper center", ncols=len(basis_functions.T), fontsize=12, columnspacing=0.5, handlelength=0.7)
    ax_eig.legend(**legend_args)

    savefig(fig, args.output)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.anom:
        plot_anom(args)
        quit()

    if args.observation is None:
        observation = None
    else:
        with open(args.observation, "rb") as fp:
            obs_dict = utils.read_observation(tomllib.load(fp)["observation"])
            base_data, samples_ref = load_samples(obs_dict["base_sim"])
            observation = dict(
                fields=obs_dict["fields"], data=base_data, ref=samples_ref
            )

    common_args = dict(
        observation=observation,
        mode=args.mode,
        ref=args.ref,
        ref_label = args.ref_label,
        obs_style=args.obs_style,
        vline_loc=args.vline_loc,
        ref2 = args.ref2,
        ref2_label = args.ref2_label,
    )

    if args.type == "comparison":
        plot_multifield_comparison(args, **common_args)
    elif args.type == "sidebyside":
        plot_sidebyside(args, **common_args)
    else:
        raise NotImplementedError()
