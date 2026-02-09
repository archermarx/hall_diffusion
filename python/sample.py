# Stdlibs
import argparse
import tomllib
from pathlib import Path
import os
import shutil
import uuid
import math
import random

# Third-party deps
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Local deps
import models.edm
import models.edm2
from utils import utils
from utils.thruster_data import ThrusterDataset
import noise

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, nargs="?")
parser.add_argument("config", type=str, nargs="?")
parser.add_argument("-o", "--out-dir", type=str)
parser.add_argument("-n", "--num-samples", type=int)
parser.add_argument("-s", "--num-steps", type=int)
parser.add_argument("--test-residual", action="store_true")
parser.add_argument("--test-dir", type=Path)

DEVICE = torch.device("cpu")

def build_observation(dataset, observations, default_stddev=1.0):
    _, param_vec, data_tensor = dataset[0]
    data_tensor = torch.tensor(data_tensor, device=DEVICE)
    param_vec = torch.tensor(param_vec, device=DEVICE)
    (num_channels, resolution) = data_tensor.shape

    obs_matrix_loc = torch.zeros(num_channels, resolution)
    obs_matrix_dat = torch.zeros(num_channels, resolution)
    obs_matrix_var = torch.zeros(num_channels, resolution)

    default_stddev = observations.get("stddev", 1.0)

    grid = dataset.grid

    m = num_channels * resolution
    n = 0

    obs_fields = observations["fields"]

    for obs_field in obs_fields:
        # Get tensor row index
        row_index = dataset.fields()[obs_field]

        # Get stddev (TODO: read from file to allow more values, but this needs to wait for improved stddev handling)
        stddev = obs_fields[obs_field].get("stddev", default_stddev)

        # Get observation from file
        x_inds, x_data, y_data = utils.get_observation_locs(
            obs_fields, obs_field, grid, normalizer=dataset.norm, form="normalized"
        )

        if (len(x_data) == resolution) and np.all(x_inds == np.arange(resolution)):
            # If x_data == grid, then we're observing an entire row
            print("Observing row " + obs_field)
            obs_matrix_loc[row_index, :] = 1.0
            obs_matrix_dat[row_index, :] = data_tensor[row_index, :]
            obs_matrix_var[row_index, :] = stddev**2
            n += resolution
        else:
            # Partial/sparse observation of the row
            # If y not provided, we use the underlying data matrix from the dataset
            # Otherwise we use the y found in the file
            # TODO: stddevs that vary point-to-point
            obs_matrix_loc[row_index, x_inds] = 1.0
            obs_matrix_var[row_index, x_inds] = stddev**2
            n += len(x_inds)

            if y_data is None:
                print("Using underlying data")
                obs_matrix_dat[row_index, x_inds] = data_tensor[row_index, x_inds]
            else:
                print("Using data from file")
                obs_matrix_dat[row_index, x_inds] = torch.tensor(
                    y_data, dtype=torch.float32
                )

    # Dimensions
    # m = num_channels * resolution
    # n = num_observations
    # A (linear observation operator) = (n, m)
    # y (observed data) = (n,)
    obs_matrix_loc = obs_matrix_loc.reshape(-1)
    obs_A = torch.zeros(n, m)

    j = 0
    for i in range(m):
        if obs_matrix_loc[i] == 1.0:
            obs_A[j, i] = 1.0
            j += 1

    obs_y = obs_A @ obs_matrix_dat.reshape(-1)
    obs_var = obs_A @ obs_matrix_var.reshape(-1)

    # Read scalar parameters if present
    if (params := observations.get("params", None)) is not None:
        for p, i in dataset.params().items():
            if p in params:
                param_vec[i] = dataset.norm.normalize(params[p], p)

    return obs_A.to(DEVICE), obs_y.to(DEVICE), obs_var.to(DEVICE), param_vec

def edm_sampling_timesteps(num_steps, noise_min, noise_max, exponent):
    inv_rho = 1 / exponent
    i = torch.arange(0, num_steps)
    f1 = noise_max**inv_rho
    f2 = (noise_min**inv_rho - noise_max**inv_rho) / (num_steps - 2)
    timesteps = (f1 + i * f2) ** exponent
    timesteps[-1] = 0
    return timesteps


def print_grad_hook(grad):
    print(f"Gradient received: shape={grad.shape}, norm={grad.norm():.4f}")

#=====================================================
# Physics residuals
#=====================================================

# Constants
q_e = 1.6e-19
m_e = 9.1e-31
qe_me = q_e / m_e

# Normalization
q_0 = q_e
m_0 = m_e
phi_0 = 0.01
n_0 = 1e18
L_0 = 1/math.cbrt(n_0)
u_0 = math.sqrt(q_0 * phi_0 / m_0)
t_0 = L_0 / u_0
f_0 = 1/t_0
E_0 = phi_0 / L_0

def interior_gradient(x, dx):
    return (x[:, 2:] - x[:, :-2]) / (2 * dx)

def residual_result(a, b):
    return (a-b)**2, a, b

def residual_E(x_0, dataset):
    """
    Compute electric field residual: E = -grad(phi)
    """
    # Get phi and E in real units, then normalize consistently
    phi = dataset.get_denorm(x_0, "phi") / phi_0
    E = dataset.get_denorm(x_0, "E") / E_0

    # Compute potential gradient using finite differences (interior only)
    E_calc = -interior_gradient(phi, dataset.dx / L_0)

    # Compute normalized electric field residual
    return residual_result(E_calc, E[:, 1:-1])

def residual_grad_pe(x_0, dataset):
    """
    Compute pressure gradient residual: gradpe = grad(pe)
    """
    # Get pressure and pressure gradient in real units, then normalize consistently
    pe = dataset.get_denorm(x_0, "pe") / (n_0 * phi_0)
    grad_pe = dataset.get_denorm(x_0, "∇pe") / (n_0 * phi_0 / L_0)

    # Compute pressure gradient with finite differences (interior only)
    grad_pe_calc = interior_gradient(pe, dataset.dx / L_0)

    # Compute squared normalized pressure gradient residual
    return residual_result(grad_pe_calc, grad_pe[:, 1:-1])

def residual_ohm(x_0, dataset):
    """
    Compute Ohm's law residual: ue = -mu * (E + grad_pe / ne)
    With mu = q/m * (nu_e / (nu_e**2 + wce**2))
    """
    # Extract needed properties in real units, then normalize consistently
    wce = qe_me * dataset.get_denorm(x_0, "B") / f_0
    ne = dataset.get_denorm(x_0, "ne") / n_0
    nu_e = dataset.get_denorm(x_0, "nu_e") / f_0
    E = dataset.get_denorm(x_0, "E") / E_0
    grad_pe = dataset.get_denorm(x_0, "∇pe") / (n_0 * phi_0 / L_0)
    ue = dataset.get_denorm(x_0, "ue") / u_0

    # Get ion properties for ion drag calculations
    ni_1 = dataset.get_field(x_0, f"ni_1", action="denormalize") / n_0
    ni_2 = dataset.get_field(x_0, f"ni_2", action="denormalize") / n_0
    ni_3 = dataset.get_field(x_0, f"ni_3", action="denormalize") / n_0

    ui_1 = dataset.get_field(x_0, f"ui_1", action="denormalize") / u_0
    ui_2 = dataset.get_field(x_0, f"ui_2", action="denormalize") / u_0
    ui_3 = dataset.get_field(x_0, f"ui_3", action="denormalize") / u_0
    
    niui = ni_1 * ui_1 + ni_2 * ui_2 + ni_3 * ui_3

    # Compute electron current using normalized ohm's law
    ion_drag = niui * nu_e
    mu = nu_e / (nu_e**2 + wce**2)
    neue_calc = -mu * (ne * E + grad_pe - ion_drag)

    return residual_result(neue_calc[:, 1:-1], (ne * ue)[:, 1:-1])

def residual_ne(x_0, dataset):
    """
    Compute ne residual: ne = sum(Z * ni_Z)
    """
    # Get electron density
    ne   = dataset.get_field(x_0, "ne", action="denormalize") / n_0
    ni_1 = dataset.get_field(x_0, f"ni_1", action="denormalize") / n_0
    ni_2 = dataset.get_field(x_0, f"ni_2", action="denormalize") / n_0
    ni_3 = dataset.get_field(x_0, f"ni_3", action="denormalize") / n_0
    ne_calc = 1 * ni_1 + 2 * ni_2 + 3 * ni_3

    return residual_result(ne_calc, ne)

# Register allowed physics residuals and some metadata
PHYSICS_RESIDUALS = dict(
    E = dict(
        func = residual_E,
        unit = "V/m",
        normalizer = E_0,
     ),
    grad_pe = dict(
        func = residual_grad_pe,
        unit = "eV/m^4",
        normalizer = phi_0 * n_0 / L_0,
    ),
    ohm = dict(
        func = residual_ohm,
        unit = "A/m^2",
        normalizer = q_0 * n_0 * u_0,
    ),
    ne = dict(
        func = residual_ne,
        unit = "m^{-3}",
        normalizer = n_0,
    ),
)

# Compute physics residual and some stats
def physics_residual_info(x_0, dataset, res_name, res_info):
    residual, _, _ = res_info["func"](x_0, dataset)

    mean_res = residual.mean().item()
    min_res = residual.min().item()
    max_res = residual.max().item()

    status = f"{res_name}: {mean_res:.2e} ({min_res:.2e} - {max_res:.2e})"

    return {"residual": residual, "mean": mean_res, "min": min_res, "max": max_res, "status": status}

def calc_physics_residuals(x_0, dataset):
    return {res_name: physics_residual_info(x_0, dataset, res_name, res_info) for (res_name, res_info) in PHYSICS_RESIDUALS.items()}


#=====================================================
# Conditioning on observations and PDEs
#=====================================================

def guidance_score(x_t, x_0, observation, proc_var, pde_strength, pde_args, dataset):
    (batch_size, _, _) = x_0.shape

    total_loss = torch.tensor(0.0, device=DEVICE)

    if observation["var"] is not None:
        obs_vec = observation["data"]
        var = observation["var"]
        H = observation["operator"]

        # # Pseudoinverse guidance (with noise)
        # n, m = H.shape
        # assert obs_vec.shape == (n,)

        # mat1 = torch.inverse(proc_var * H @ H.T + var * torch.eye(n, device=DEVICE)) @ H
        # assert mat1.shape == (n, m)

        # x_vec = x_0.reshape(batch_size, -1)
        # vec1 = (obs_vec[None, ...] - torch.matmul(H, x_vec.T).T)
        # assert vec1.shape == (batch_size, n)

        # vec2 = (vec1 @ mat1)
        # assert vec2.shape == (batch_size, m)

        # mat_x = (vec2.detach() * x_vec).sum()
        # score = torch.autograd.grad(mat_x, x_t)[0]

        #=====================================================
        # Diffusion posterior sampling (get observation loss)
        #=====================================================
        x_vec = x_0.reshape(batch_size, -1)
        measurement = torch.matmul(H, x_vec.T).T
        total_var = var + proc_var
        obs_loss = torch.sum((measurement - obs_vec[None, ...])**2 / total_var)

        total_loss += obs_loss

    if pde_strength > 0:
        #=====================================================
        # DiffusionPDE loss (get physics/PDE loss)
        #=====================================================
        print_residuals = pde_args.get("print_residuals", False)
        pde_errs = calc_physics_residuals(x_0, dataset)
        pde_weights = pde_args.get("strengths", dict())

        pde_loss = torch.tensor(0.0, device=DEVICE)
        for (k, v) in pde_errs.items():
            err = v["residual"].mean()
            w = pde_weights.get(k, 0.0)

            if w > 0:
                pde_loss += w * err

        if print_residuals:
            print(" | ".join([x["status"] for x in pde_errs.values()]))

        total_loss += pde_strength * pde_loss

    # Take gradient
    score = -torch.autograd.grad(total_loss, x_t)[0]

    return score

def reverse_step(
    denoiser,
    x_t,
    t_prev,
    t,
    observation,
    dataset,
    pde_args,
    step_scale=1.0,
    method="midpoint",
    model_args=dict(),
):
    assert method in ["heun", "midpoint", "euler"]

    (b, _, _) = x_t.shape

    ones = torch.ones((b, 1, 1), device=x_t.device)

    dt = t - t_prev
    t_mid = 0.5 * (t + t_prev)
    proc_var = t**2 / (t**2 + 1)

    t_max_guidance = 100

    x_t = x_t.detach()
    x_t.requires_grad = True
    denoiser.zero_grad()

    # Compute initial step to get predicted sample location
    x_denoised = denoiser(x_t, t_prev * ones, **model_args)
    deriv_0 = -(x_denoised - x_t) / t_prev
    step_0 = step_scale * dt * deriv_0

    if method == "midpoint":
        step_0 *= 0.5

    x_pred = x_t + step_0

    if method == "midpoint" or (method == "heun" and t > 0):
        # Compute corrector step
        t2 = t_mid if method == "midpoint" else t
        x_denoised = denoiser(x_pred, t2 * ones, **model_args)
        deriv_1 = -(x_denoised - x_t) / t2

        if method == "midpoint":
            step_1 = step_scale * dt * deriv_1
        else:
            step_1 = step_scale * dt * (deriv_0 + deriv_1) / 2

        x_pred = x_t + step_1

    pde_start = pde_args.get("start_time", 1.0)
    pde_stop = pde_args.get("stop_time", 0.0)

    if (pde_stop <= t <= pde_start):
        pde_strength = abs(dt) * pde_args.get("strengths", dict()).get("base", 0.0) / (t**2 + 1)
    else:
        pde_strength = 0.0

    # Guidance loss
    if t_mid < t_max_guidance:
        obs_score = guidance_score(
            x_t, x_denoised, observation, proc_var, pde_strength, pde_args, dataset
        )
        # x_1 += dt * t_mid * obs_score
        x_pred += obs_score

    return x_pred.detach()


def reverse(
    denoiser,
    x,
    timesteps,
    dataset,
    observation,
    showprogress=False,
    pde_args=dict(),
    model_args=dict(),
    **kwargs,
):
    """
    Perform iterative denoising to generate a 1D image from Gaussian noise using a provided denoising model

    Args:
        denoiser: a Denoiser model
        x: a tensor containing standard Gaussian noise. The dimension of this tensor should be (b, c, w)
            where b is the batch size, c is the number of channels, and w is the width
        im_masks: An optimal tuple of (data, mask) to apply during generation. These should have the same shape as x.
            When provided, the model will infill areas where the mask is set to 0.
        showprogress: whether we should print a tqdm progress bar.
    Returns:
    """
    (b, c, w) = x.shape
    num_steps = len(timesteps)

    output = torch.zeros((num_steps, b, c, w))
    output[0, ...] = x

    for step_idx, t in enumerate(tqdm(timesteps, disable=(not showprogress))):
        if step_idx == 0:
            continue

        t_prev = timesteps[step_idx - 1]
        x = reverse_step(
            denoiser,
            x,
            t_prev,
            t,
            observation,
            dataset,
            model_args=model_args,
            pde_args=pde_args,
            **kwargs,
        )

        # Check for NaN or Inf
        if not torch.all(torch.isfinite(x)):
            print("NaN/Inf detected during sampling. Exiting")
            exit(1)

        output[step_idx, ...] = x

    output[-1, ...] = x
    return output


def sample(model, noise_sampler, num_samples, args):
    # Load sampling arguments
    num_steps = args.get("num_steps", 256)
    noise_min = args.get("noise_min", 0.002)
    noise_max = args.get("noise_max", 80)
    exponent = args.get("step_exponent", 7)
    step_scale = args.get("step_scale", 1.0)
    method = args.get("method", "midpoint")

    # Determine if we're doing condional or unconditional sampling
    # If there is an `observation` field, then we're conditioning on a partial observation of that simulation
    # If not, we're sampling unconditionally
    # If we sample unconditonally, we need to get some scalar parameters to condition on
    # These are drawn from the same distributions as the training set
    if "observation" in args:
        obs_args = utils.read_observation(args["observation"])
        obs_file = Path(obs_args["base_sim"])

        # Load data for conditioning
        dataset = ThrusterDataset(obs_file)
        obs_operator, obs_data, obs_var, param_vec = build_observation(
            dataset, obs_args
        )
        obs = dict(operator=obs_operator, data=obs_data, var=obs_var)
    elif "unconditional_data_dir" in args:
        dataset = ThrusterDataset(args["unconditional_data_dir"])
        obs = dict(operator=None, var=None, data=None)
        param_vec_inds = random.choices(range(len(dataset)), k=num_samples)
        param_vec = torch.tensor(
            np.array([dataset[i][1] for i in param_vec_inds]), device=DEVICE
        )
    else:
        raise NotImplementedError(
            "Observation not provided and no data directory for unconditional sample generation."
        )

    # Sample initial noise
    xt = noise_sampler.sample(num_samples) * noise_max

    # Load timesteps
    steps = edm_sampling_timesteps(num_steps, noise_min, noise_max, exponent)

    output = reverse(
        model,
        xt,
        steps,
        dataset,
        showprogress=True,
        observation=obs,
        step_scale=step_scale,
        method=method,
        model_args=dict(condition_vector=param_vec),
        pde_args=args.get("pde_guidance", None)
    )

    final = output[-1, ...]

    # Save generated samples
    out_dir = Path(args["out_dir"])

    if args.get("replace_samples", False) and out_dir.exists():
        shutil.rmtree(out_dir)

    # Make folder and write metadata
    os.makedirs(out_dir, exist_ok=True)
    dataset.write_metadata(out_dir)

    # Write sample data
    data_dir = out_dir / "data"
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_samples):
        file = data_dir / f"{uuid.uuid4()}.npz"
        tens = final[i, :].cpu().numpy()
        np.savez(file, data=tens, params=param_vec.cpu().numpy())

def test_residual(test_dir):
    dataset = ThrusterDataset(test_dir)
    x = dataset.grid
    x_int = x[1:-1]

    fig, axes = plt.subplots(2,2, figsize=(8,8), layout="constrained")
    axes = axes.ravel()

    _, _, tensor = dataset[4]
    tensor = tensor[None, ...]

    print("==========================")
    print("Normalization constants:")
    print("==========================")
    print(f"{phi_0=}")
    print(f"{L_0=:.3e}")
    print(f"{E_0=:.3e}")
    print(f"{n_0=:.3e}")
    print(f"{u_0=:.3e}")
    print(f"{t_0=:.3e}")
    print("==========================")

    for ax in axes:
        ax.axhline([0.0], color = 'black', zorder=0, linestyle='--')

    # E vs grad_phi
    E_res, E_calc, E_base = residual_E(tensor, dataset)
    axes[0].set(xlabel = "z (m)", ylabel = "Electric field (norm)", title = "E = -grad(phi)")
    axes[0].plot(x_int, E_calc[0, :], label = "Extracted")
    axes[0].plot(x_int, E_base[0, :], label = "Calculated", linestyle = "--")
    axes[0].plot(x_int, np.sqrt(E_res[0, :]), label = "Residual")
    axes[0].legend()

    # pe vs grad_pe
    grad_pe_res, grad_pe_calc, grad_pe_base = residual_grad_pe(tensor, dataset)
    axes[1].set(xlabel = "z (m)", ylabel = "Pressure gradient (norm)", title = "Pressure gradient")
    axes[1].plot(x_int, grad_pe_base[0, :], label = "Extracted")
    axes[1].plot(x_int, grad_pe_calc[0, :], label = "Calculated", linestyle = "--")
    axes[1].plot(x_int, np.sqrt(grad_pe_res[0, :]), label = "Residual")
    axes[1].legend()

    # Ohm's law
    je_res, je_calc, je_base = residual_ohm(tensor, dataset)
    axes[2].set(xlabel = "z (m)", ylabel = "Electron current density (norm)", title = "Ohm's law")
    axes[2].plot(x_int, je_base[0, :], label = "Extracted")
    axes[2].plot(x_int, je_calc[0, :], label = "Calculated", linestyle = "--")
    axes[2].plot(x_int, np.sqrt(je_res[0, :]), label = "Residual")
    axes[2].legend()

    # Number densities
    ne_res, ne_calc, ne_base = residual_ne(tensor, dataset)
    axes[3].set(xlabel = "z (m)", ylabel = "Number density (norm)", title = "ne = sum(Z ni)")
    axes[3].plot(x, ne_base[0, :], label = "Extracted")
    axes[3].plot(x, ne_calc[0, :], label = "Calculated", linestyle = "--")
    axes[3].plot(x, np.sqrt(ne_res[0, :]), label = "Residual")
    axes[3].legend()

    plt.savefig("residuals.png", dpi=200)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.test_residual:
        test_residual(args.test_dir)
        exit(0)

    DEVICE = utils.get_device()

    # Load sampling configuration
    with open(args.config, "rb") as fp:
        sampling_config = tomllib.load(fp)

    # Read command line args and replace TOML args if needed
    if args.out_dir is not None:
        sampling_config["out_dir"] = args.out_dir

    if args.num_steps is not None:
        sampling_config["num_steps"] = args.num_steps

    if args.num_samples is not None:
        sampling_config["num_samples"] = args.num_samples

    # Load model and config from checkpoint
    model_dict = torch.load(args.model, weights_only=False)
    model_config = model_dict["model_config"]
    print(f"{model_config=}")
    arch = model_config.get("architecture", "edm2")
    if arch == "edm2":
        model = models.edm2.Denoiser.from_config(model_config).to(DEVICE)
    else:
        model = models.edm.Denoiser.from_config(model_config).to(DEVICE)

    # Determine which weights to load
    model_type = sampling_config.get("model_type", "ema")
    assert model_type in ["ema", "best", "last"]
    model_type = "model" if model_type == "last" else model_type
    model.load_state_dict(model_dict[model_type])

    # Switch model to evalution mode and sample
    model.eval()

    num_samples = sampling_config.get("num_samples", 64)
    batch_size = sampling_config.get("batch_size", num_samples)

    full_batches = math.floor(num_samples / batch_size)
    remainder = num_samples - full_batches * batch_size
    batches = [batch_size for i in range(math.floor(num_samples / batch_size))]
    if remainder > 0:
        batches.append(remainder)

    # Create noise sampler and initial noise samples
    if "train_config" in model_dict and "noise_sampler" in model_dict["train_config"]:
        noise_sampler_args = model_dict["train_config"]["noise_sampler"]
    else:
        noise_sampler_args = dict(type="gaussian", scale=1.0)

    channels = model.img_channels
    resolution = model.img_resolution

    if noise_sampler_args["type"] == "gaussian":
        noise_sampler = noise.RandomNoise(channels, resolution, device=DEVICE)
    elif noise_sampler_args["type"] == "rbf":
        scale = noise_sampler_args["scale"]
        assert isinstance(scale, float | int)
        noise_sampler = noise.RBFKernel(
            channels, resolution, scale=scale, device=DEVICE
        )
    else:
        raise NotImplementedError()

    # Sample in batches
    for i, batch_num_samples in enumerate(batches):
        sample(model, noise_sampler, batch_num_samples, sampling_config)

        # Make sure we don't remove old samples
        sampling_config["replace_samples"] = False
