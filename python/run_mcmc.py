import sys
import MCMCIterators.samplers
import numpy as np
import json
import math
import copy
from abc import ABC, abstractmethod
import scipy.special
import os
from pathlib import Path
import itertools
import argparse
import matplotlib.pyplot as plt
import random
import time
from typing import TypedDict, NotRequired

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--max-samples", type=int, default=10)
parser.add_argument("-i", "--write-interval", type=int, default=2)
parser.add_argument("-a", "--analysis-interval", type=int, default=100)
parser.add_argument("--calibrate-temperature", action="store_true")
parser.add_argument("--out-dir", type=str)
parser.add_argument("--init-stats", type=str)
parser.add_argument("--data-file", type=str)
parser.add_argument("--julia-env", type=str, default="")
parser.add_argument("--port", type=int, default=3000)

# Parameters and constants
qe = 1.6e-19
me = 9.1e-31
MAX_RANK = 5

# GPR Kernel
def kernel(x1, x2, length):
    return np.exp(-0.5 * (x1 - x2) ** 2 / length**2)


def gp_covariance_matrix(grid, kernel_length):
    cov = np.array([[kernel(x1, x2, kernel_length) for x2 in grid] for x1 in grid])
    return 0.5 * (cov + cov.T)


# Distributions
class Distribution(ABC):
    @abstractmethod
    def sample(self, *shape) -> float | np.ndarray:
        pass

    @abstractmethod
    def logpdf(self, x) -> float:
        pass

    @abstractmethod
    def cdf(self, x) -> float:
        pass


class Uniform(Distribution):
    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b

    def sample(self, *shape):
        return (self.b - self.a) * np.random.rand(*shape) + self.a

    def logpdf(self, x):
        if x < self.a or x > self.b:
            return -np.inf
        else:
            return np.log(1 / (self.b - self.a))

    def cdf(self, x):
        if x < self.a:
            return 0.0
        elif x > self.b:
            return 1
        else:
            return (x - self.a) / (self.b - self.a)


def _normal_log_pdf(x, mean, std):
    return -np.log(std) - 0.5 * np.log(2 * np.pi) - 0.5 * (x - mean) ** 2 / std**2


def _normal_cdf(x, mean, std):
    return 0.5 * (1 + scipy.special.erf((x - mean) / (std * np.sqrt(2))))


class Normal(Distribution):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def sample(self, *shape):
        return self.mean + np.random.randn(*shape) * self.std

    def logpdf(self, x):
        return _normal_log_pdf(x, self.mean, self.std)

    def cdf(self, x):
        return _normal_cdf(x, self.mean, self.std)


class LogNormal(Distribution):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def sample(self, *shape):
        log_x = self.mean + np.random.randn(*shape) * self.std
        return np.exp(log_x)

    def logpdf(self, x):
        if x <= 0.0:
            return -np.inf
        log_x = np.log(x)
        return _normal_log_pdf(log_x, self.mean, self.std) - log_x

    def cdf(self, x):
        if x <= 0:
            return 0.0
        return _normal_cdf(np.log(x), self.mean, self.std)


class Truncated(Distribution):
    def __init__(self, dist: Distribution, lo=-np.inf, hi=np.inf):
        self.dist = dist
        self.lo = lo
        self.hi = hi
        if self.hi <= self.lo:
            raise ArithmeticError("Upper bound must be higher than lower bound")

        self.cdf_lo = self.dist.cdf(self.lo) if np.isfinite(self.lo) else 0.0
        self.cdf_hi = self.dist.cdf(self.hi) if np.isfinite(self.hi) else 1.0

    def sample(self, *shape):
        """
        Sample from truncated distribution using rejection sampling
        """
        num_sample = math.prod(shape)
        inds_sample = np.ones(num_sample, dtype=bool)
        x = np.zeros(num_sample)

        while num_sample > 0:
            samples = self.dist.sample(num_sample)
            x[inds_sample] = samples
            inds_sample = np.logical_or(x < self.lo, x > self.hi)
            num_sample = np.sum(inds_sample)

        x = x.reshape(shape)
        if len(shape) == 0:
            x = float(x)

        return x

    def logpdf(self, x):
        if x < self.lo:
            return -np.inf
        elif x > self.hi:
            return -np.inf
        else:
            return self.dist.logpdf(x) / (self.cdf_hi - self.cdf_lo)

    def cdf(self, x):
        if x < self.lo:
            return 0.0
        elif x > self.hi:
            return 1.0
        else:
            return (self.dist.cdf(x) - self.cdf_lo) / (self.cdf_hi - self.cdf_lo)


# Model parameters
param_distributions = dict(
    anom_minimum=Truncated(Normal(np.log(0.05), np.log(10)), -np.inf, 0.0),
    anom_width=Truncated(Normal(0.3, 0.5), 0, np.inf),
    anom_slope=Uniform(0.0, 1.0),
    anom_step=Uniform(0.0, 1.0),
    anom_scale=Normal(np.log(0.1), np.log(5)),
    anom_center=Truncated(Normal(1.1, 1 / 3), 0, 2),
    neutral_velocity=Truncated(Normal(300, 100), 0, np.inf),
    wall_loss_coeff=Truncated(Normal(1.0, 0.5), 0, np.inf),
)

for i in range(MAX_RANK):
    param_distributions[f"w_{i}"] = Normal(0.0, 1.0)


def sample_parameters():
    params = {k: v.sample() for (k, v) in param_distributions.items()}
    return params


def log_prior(params):
    logp = 0.0
    for k, v in params.items():
        dist = param_distributions[k]
        logp += dist.logpdf(v)

    return logp


# Transport model
def anom_model(z, params):
    a = params["anom_minimum"]
    w = params["anom_width"]
    s = params["anom_slope"]
    m = params["anom_step"]
    S = params["anom_scale"]
    L = params["anom_center"]

    # Exponentiate log variables
    a = np.exp(a)
    S = np.exp(S)

    z0 = z / L - 1
    f = 1 - s + s / (1 + np.exp(-m * z0 / (1 - m)))
    g = 1 - (1 - a) * np.exp(-((z0 / w) ** 2))
    f_anom = S * f * g

    return f_anom

def anom_model_with_noise(grid, params, basis_functions, channel_length):
    f_anom_base = anom_model(grid / channel_length, params)

    gpr_weights = np.array([params[f"w_{i}"] for i in range(basis_functions.shape[1])])
    noise = basis_functions @ gpr_weights
    return f_anom_base, np.exp(np.log(f_anom_base) + noise)

def setup_grid(domain, N, L_ch):
    # Generate grid
    grid = np.linspace(domain[0], domain[1], N + 2)

    # Compute kernel and decompose into eigenfunctions
    noise_length = L_ch
    K = gp_covariance_matrix(grid, noise_length)
    eig_results = np.linalg.eig(K)
    eigenvalues = np.real(eig_results.eigenvalues)
    eigenvectors = np.real(eig_results.eigenvectors)

    # With this scaling, the leading coefficients of the basis functions are normally-distributed
    coeffs = np.sqrt(eigenvalues[:MAX_RANK])
    basis_functions = eigenvectors[:, :MAX_RANK] * coeffs
    return grid, basis_functions

def setup_sim(args):
    # Problem setup
    baseline_sim = args.data_file

    # Load base simulation settings (including data to calibrate against)
    with open(baseline_sim, "r") as fd:
        base_outputs = json.load(fd)

    # Create config from this
    base_inputs = copy.deepcopy(base_outputs["input"])

    # Get geometric parameters from config
    N = base_inputs["simulation"]["grid"]["num_cells"]
    domain = base_inputs["config"]["domain"]
    L_ch = base_inputs["config"]["thruster"]["geometry"]["channel_length"]

    grid, basis_functions = setup_grid(domain, N, L_ch)
    # Find and configure HallThruster.jl
    # julia_env=args.julia_env
    # julia_args = ["julia", f'--project="{julia_env}"', "-e"]
    # cmd_output = subprocess.run(julia_args + ["using HallThruster; print(HallThruster.PYTHON_PATH)"], capture_output=True)
    # het_pythonpath = cmd_output.stdout.decode("utf-8")
    sys.path.append("hallthruster")

    return dict(
        grid=grid,
        basis_functions=basis_functions,
        inputs=base_inputs,
        outputs=base_outputs,
        L_ch=L_ch,
        jl_env=args.julia_env,
        port=args.port,
    )


# Likelihood
def log_likelihood(params, sampler_args, base_args):
    _, f_anom = anom_model_with_noise(
        base_args["grid"], params, base_args["basis_functions"], base_args["L_ch"]
    )
    inputs = copy.deepcopy(base_args["inputs"])

    # Input anom model
    inputs["config"]["anom_model"]["zs"] = list(base_args["grid"])
    inputs["config"]["anom_model"]["cs"] = list(f_anom)

    # Input wall loss model
    inputs["config"]["wall_loss_model"]["loss_scale"] = params["wall_loss_coeff"]

    # Input neutral velocity
    inputs["config"]["propellants"][0]["velocity_m_s"] = params["neutral_velocity"]

    # Update hall thruster arguments
    if "output_file" in sampler_args:
        inputs["postprocess"]["output_file"] = sampler_args["output_file"]

    verbose = sampler_args.get("verbose", False)
    inputs["simulation"]["verbose"] = verbose
    inputs["simulation"]["print_errors"] = verbose

    # Run simulation
    outputs = het.run_simulation(
        inputs, jl_script="run.jl", jl_daemon=f"using DaemonMode; DaemonMode.runargs({base_args["port"]})", jl_env=base_args["jl_env"]
    )

    # Get ion vel from data and from sim
    data_vel = np.array(base_args["outputs"]["output"]["average"]["ions"]["Xe"][0]["u"])
    sim_vel = np.array(outputs["output"]["average"]["ions"]["Xe"][0]["u"])

    # Calculate log likelihood assuming pointwise Gaussian likelihood
    log_L = np.sum(_normal_log_pdf(sim_vel, data_vel, sampler_args["std_vel"]))

    if sampler_args["calibrate_temperature"]:
        data_Te = np.array(base_args["outputs"]["output"]["average"]["Tev"])
        sim_Te = np.array(outputs["output"]["average"]["Tev"])
        log_L += np.sum(_normal_log_pdf(sim_Te, data_Te, sampler_args["std_Tev"]))

    return log_L, data_vel, sim_vel

def log_posterior(param_vec, sampler_args, base_args, payload: dict | None = None):
    # convert param_vec into dict
    param_dict = {k: v for (k, v) in zip(param_distributions, param_vec)}

    log_p = log_prior(param_dict)

    if not np.isfinite(log_p):
        return -np.inf

    start = time.perf_counter()
    log_L, _, _ = log_likelihood(param_dict, sampler_args, base_args)
    elapsed = time.perf_counter() - start

    if payload is not None:
        payload["elapsed"] = elapsed

    if not np.isfinite(log_L):
        return -np.inf

    return log_p + log_L


def load_data(file):
    with open(file, "r") as fp:
        avg_outputs = json.load(fp)["output"]["average"]

    uion = np.array(avg_outputs["ions"]["Xe"][0]["u"])
    Te = np.array(avg_outputs["Tev"])
    ne = np.array(avg_outputs["ne"])
    E = np.array(avg_outputs["E"])
    phi = np.array(avg_outputs["potential"])
    f_anom = np.array(avg_outputs["nu_anom"]) / np.array(avg_outputs["B"]) * me / qe
    z = np.array(avg_outputs["z"])
    return dict(z=z, uion=uion, Te=Te, ne=ne, f_anom=f_anom, E=E, phi=phi)


class FieldArgs(TypedDict):
    ax_kwargs: NotRequired[dict[str, str]]
    scale: NotRequired[float]


def analyze_results(stats_file, sample_dir, results_file, baseline_sim):
    with open(stats_file, "r") as fp:
        mcmc_stats = json.load(fp)

    accepted = np.array(mcmc_stats["accept"])
    p_accept = sum(accepted) / len(accepted)
    print("Analyzing results...")
    print(f"Acceptance percentage: {p_accept}")

    baseline_data = load_data(baseline_sim)
    files_to_load = [
        f"{sample_dir}/{i:06d}.json" for (i, accept) in enumerate(accepted) if accept
    ]
    max_traces = 200
    indices = random.choices(range(len(files_to_load)), k=max_traces)
    sim_results = [load_data(files_to_load[i]) for i in indices]

    fig, axes = plt.subplots(2, 3, layout="constrained", figsize=(10, 5.5))
    axes = axes.flatten()

    field_details: dict[str, FieldArgs] = dict(
        uion = FieldArgs(scale = 1 / 1000, ax_kwargs = dict(ylabel="Ion velocity [km/s]")),
        Te=FieldArgs(scale=1, ax_kwargs=dict(ylabel="Electron temperature [eV]")),
        ne=FieldArgs(ax_kwargs=dict(yscale="log", ylabel="Plasma density [m$^{-3}$]")),
        f_anom=FieldArgs(ax_kwargs=dict(yscale="log", ylabel=r"$\nu_{an} / \omega_{ce}$")),
        phi=FieldArgs(ax_kwargs=dict(ylabel="Potential [V]")),
        E=FieldArgs(scale=1 / 1000, ax_kwargs=dict(ylabel="Electric field [kV/m]")),
    )

    sim_kwargs = dict(color="black", alpha=min(2 / len(sim_results), 1.0), zorder=0)
    data_kwargs = dict(color="red", zorder=1)

    for s in sim_results:
        for ax, (field, args) in zip(axes, field_details.items()):
            ax.plot(s["z"], s[field] * args.get("scale", 1.0), **sim_kwargs)

    for ax, (field, args) in zip(axes, field_details.items()):
        ax.plot(
            baseline_data["z"],
            baseline_data[field] * args.get("scale", 1.0),
            **data_kwargs,
        )
        ax_kwargs = args.get("ax_kwargs", {})
        ax.set(xlabel="z [m]", **ax_kwargs)

    plt.savefig(results_file, dpi=200)
    plt.close(fig)

class SamplerStats(TypedDict):
    samples: list[list[float]]
    logpdfs: list[float]
    accept: list[bool]
    cov: list[list[float]]
    p_accept: float

def main(args, base_args):
    max_samples = args.max_samples
    write_interval = args.write_interval
    analysis_interval = args.analysis_interval
    out_dir = Path(args.out_dir)
    stats_file = out_dir / "stats.json"
    sample_dir = out_dir / "samples"
    results_file = Path(args.out_dir) / "results.png"

    print(f"{max_samples=}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    sampler_args = {
        "std_vel": 1000,
        "std_Tev": 1.0,
        "output_file": str(sample_dir / f"{0:06d}.json"),
        "calibrate_temperature": args.calibrate_temperature,
        "verbose": False,
    }

    # ===============================================
    #   Load initial sample and covariance
    # ===============================================

    if os.path.exists(stats_file) or args.init_stats is not None:
        if os.path.exists(stats_file):
            _stats_file = stats_file
        else:
            _stats_file = args.init_stats

        with open(_stats_file, "r") as fp:
            stats: SamplerStats = json.load(fp)

        initial_cov = np.array(stats["cov"])
        accepted_samples = [
            s for (s, accept) in zip(stats["samples"], stats["accept"]) if accept
        ]
        initial_sample_vec = np.array(accepted_samples[-1])

        if args.init_stats is not None:
            stats = SamplerStats(
                samples=[],
                logpdfs=[],
                accept=[],
                cov=initial_cov.tolist(),
                p_accept=1.0,
            )
    else:
        with open("initial_sample.json", "r") as fd:
            initial_sample = json.load(fd)

        initial_sample_vec = np.array([v for (_, v) in initial_sample.items()])
        initial_cov = 0.003 * np.eye(len(initial_sample))
        stats = SamplerStats(
            samples=[], logpdfs=[], accept=[], cov=initial_cov.tolist(), p_accept=1.0
        )

    # ===============================================
    #   Initialize sampler
    # ===============================================

    index = len(stats["accept"])
    payload = dict(elapsed=".")

    def logpdf(x):
        return log_posterior(x, sampler_args, base_args, payload)

    sampler = MCMCIterators.samplers.DelayedRejectionAdaptiveMetropolis(
        logpdf, initial_sample_vec, initial_cov, adapt_start=100
    )
    sampler_args["output_file"] = str(sample_dir / f"{index:06d}.json")

    # ===============================================
    #   Main sampler loop
    # ===============================================

    for i, (sample, logp, accepted_bool) in itertools.islice(
        enumerate(sampler), max_samples
    ):
        index += 1

        # Update output file name
        sampler_args["output_file"] = str(sample_dir / f"{index:06d}.json")

        stats["samples"].append(sample.tolist())
        if logp == -np.inf:
            logp = -1e99
        p_accept = np.sum(stats["accept"]) / len(stats["accept"])
        stats["logpdfs"].append(logp)
        stats["accept"].append(accepted_bool)
        stats["cov"] = sampler.cov.tolist()
        stats["p_accept"] = p_accept

        print(
            f"{index:06d}, time: {payload["elapsed"]:.3f} s, logp: {float(logp):.3e}, accept={accepted_bool}, p_accept: {100 * stats['p_accept']:.2f}%",
            flush=True,
        )

        # Write stats to file
        if i % write_interval == 0:
            with open(stats_file, "w") as fp:
                json.dump(stats, fp)

        if (i % analysis_interval == 0 or (i == max_samples - 1)) and p_accept > 0.0:
            analyze_results(stats_file, sample_dir, results_file, args.data_file)


if __name__ == "__main__":
    args = parser.parse_args()
    base_args = setup_sim(args)
    import hallthruster as het
    main(args, base_args)
