# Stdlibs
import argparse
import tomllib
from pathlib import Path
import os
import shutil
import uuid
import math
import random
from typing import Literal

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
parser.add_argument("-b", "--batch-size", type=int)
parser.add_argument("-s", "--num-steps", type=int)
parser.add_argument("--test-dir", type=Path)

DEVICE = torch.device("cpu")

ODEMethod = Literal["euler", "heun", "midpoint"]

def build_observation(dataset, observations, param_vec=None, default_stddev=1.0):
    _, data_params, data_tensor = dataset[0]

    data_tensor = torch.tensor(data_tensor, device=DEVICE)
    (num_channels, resolution) = data_tensor.shape

    obs_matrix_loc = torch.zeros(num_channels, resolution, device=DEVICE)
    obs_matrix_dat = torch.zeros(num_channels, resolution, device=DEVICE)
    obs_matrix_var = torch.zeros(num_channels, resolution, device=DEVICE)

    default_stddev = observations.get("stddev", observations.get("std_dev", 1.0))

    grid = dataset.grid

    m = num_channels * resolution
    n = 0

    obs_fields = observations["fields"]

    for obs_field in obs_fields:
        # Get tensor row index
        row_index = dataset.fields()[obs_field]

        # Get stddev (TODO: read from file to allow more values, but this needs to wait for improved stddev handling)
        obs_dict = obs_fields[obs_field]
        stddev = obs_dict.get("stddev", obs_dict.get("std_dev", default_stddev))

        # Get observation from file
        x_inds, x_data, y_data = utils.get_observation_locs(
            obs_fields, obs_field, grid, normalizer=dataset.norm, form="normalized"
        )

        x_inds = np.unique(np.array(x_inds)).tolist()

        if (len(x_data) == resolution) and np.all(x_inds == np.arange(resolution)):
            # If x_data == grid, then we're observing an entire row
            print(obs_field + ":\tobserving entire row.")
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
                print(obs_field + ":\tusing data from ref sim at selected axial locs.")
                obs_matrix_dat[row_index, x_inds] = data_tensor[row_index, x_inds]
            else:
                print(obs_field + ":\tusing data from file.")
                obs_matrix_dat[row_index, x_inds] = torch.tensor(y_data, dtype=torch.float32, device=DEVICE)

    # Dimensions
    # m = num_channels * resolution
    # n = num_observations
    # A (linear observation operator) = (n, m)
    # y (observed data) = (n,)
    obs_matrix_loc = obs_matrix_loc.reshape(-1)
    obs_A = torch.zeros(n, m, device=DEVICE)

    j = 0
    for i in range(m):
        if obs_matrix_loc[i] == 1.0:
            obs_A[j, i] = 1.0
            j += 1

    obs_y = obs_A @ obs_matrix_dat.reshape(-1)
    obs_var = obs_A @ obs_matrix_var.reshape(-1)

    # If no param vec specified here, we use the one from the reference dataset
    if param_vec is None:
        param_vec = torch.tensor(data_params, device=DEVICE)

    # Read scalar parameters if present
    if (params := observations.get("params", None)) is not None:
        for p, i in dataset.params().items():
            if p in params:
                param_vec[:, i] = dataset.norm.normalize(params[p], p)

    return obs_A, obs_y, obs_var, param_vec


def edm_sampling_timesteps(num_steps, noise_min, noise_max, exponent):
    inv_rho = 1 / exponent
    i = torch.arange(0, num_steps)
    f1 = noise_max**inv_rho
    f2 = (noise_min**inv_rho - noise_max**inv_rho) / (num_steps - 2)
    timesteps = (f1 + i * f2) ** exponent
    timesteps[-1] = 0
    return timesteps

# =====================================================
# Conditioning on observations and PDEs
# =====================================================
def guidance_score(x_t, x_0, observation, proc_var):
    (batch_size, _, _) = x_0.shape

    obs_vec = observation["data"]
    var = observation["var"]
    H = observation["operator"]

    # =====================================================
    # Diffusion posterior sampling (get observation loss)
    # =====================================================
    x_vec = x_0.reshape(batch_size, -1)
    measurement = torch.matmul(H, x_vec.T).T
    total_var = var + proc_var
    obs_loss = torch.sum((measurement - obs_vec[None, ...]) ** 2 / total_var)
    score = -torch.autograd.grad(obs_loss, x_t, retain_graph=False)[0]

    return score

def reverse_step(
    denoiser,
    x_t,
    t_prev,
    t,
    observation,
    step_scale=1.0,
    method: ODEMethod="midpoint",
    model_args=dict(),
):
    (b, _, _) = x_t.shape

    ones = torch.ones((b, 1, 1), device=x_t.device)

    dt = t - t_prev
    t_mid = 0.5 * (t + t_prev)

    use_const_guidance=True

    if use_const_guidance:
        proc_var = t**2 / (t**2 + 1)
        t_max_guidance = 100
    else:
        min_var = torch.min(observation["var"])
        proc_var_scale = torch.sqrt(min_var) * 15
        proc_var = proc_var_scale * t**2 / (t**2 + 1)
        t_max_guidance = 10

    x_t = x_t.detach()
    x_t.requires_grad = True
    denoiser.zero_grad()

    # Compute initial step to get predicted sample location
    x_denoised = denoiser(x_t, t_prev * ones, **model_args)
    deriv_1 = -step_scale * (x_denoised - x_t) / t_prev

    if not use_const_guidance and (observation["var"] is not None) and t_mid < t_max_guidance:
        obs_score = guidance_score(x_t, x_denoised, observation, proc_var)
        deriv_1 += -t_prev * obs_score

    if method == "midpoint":
        step_1 = 0.5 * dt * deriv_1
    else:
        step_1 = dt * deriv_1

    x_pred = x_t + step_1

    if method == "midpoint" or (method == "heun" and t > 0):
        # Compute corrector step
        t2 = t_mid if method == "midpoint" else t
        x_denoised = denoiser(x_pred, t2 * ones, **model_args)
        deriv_2 = -step_scale * (x_denoised - x_t) / t2

        # Guidance loss
        if not use_const_guidance and (observation["var"] is not None) and t_mid < t_max_guidance:
            obs_score = guidance_score(x_t, x_denoised, observation, proc_var)
            deriv_2 += -t2 * obs_score
        
        if method == "midpoint":
            step_2 = dt * deriv_2
        else:
            step_2 = dt * 0.5 * (deriv_1 + deriv_2)

        x_pred = x_t + step_2

    # Guidance loss
    if use_const_guidance and observation["var"] is not None and t_mid < t_max_guidance:
        obs_score = guidance_score(x_t, x_denoised, observation, proc_var)
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
            model_args=model_args,
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
    if (uncond_dir := args.get("unconditional_data_dir", None)) is not None:
        unconditional_dataset = ThrusterDataset(uncond_dir)
        param_vec = unconditional_dataset.sample_params(num_samples=num_samples, device=DEVICE)
    else:
        unconditional_dataset = None
        param_vec = None

    print(args)
    if "observation" in args:
        obs_args = utils.read_observation(args["observation"])
        obs_file = Path(obs_args["base_sim"])

        # Load data for conditioning
        dataset = ThrusterDataset(obs_file)

        if (obs_params:= obs_args.get("params", None)) is not None:
            if set(obs_params) != set(dataset.params()) and param_vec is None:
                # We didn't completely specify the parameter vector and have nothing to fall back on
                raise RuntimeError("Incomplete parameter specification without data directory. Exiting.")

        elif "params" not in obs_args:
            # Use the parameter vector from the ref simulation
            param_vec = None

        obs_operator, obs_data, obs_var, param_vec = build_observation(dataset, obs_args, param_vec)
        obs = dict(operator=obs_operator, data=obs_data, var=obs_var)
    else:
        if param_vec is None or unconditional_dataset is None:
            raise RuntimeError("No observation specified and no data directory given. Exiting")

        dataset = unconditional_dataset
        obs = dict(operator=None, var=None, data=None)

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
        pde_args=args.get("pde_guidance", None),
    )

    final = output[-1, ...]

    # Save generated samples
    out_dir = Path(args["out_dir"])
    data_dir = out_dir / "data"

    if args.get("replace_samples", False) and data_dir.exists():
        shutil.rmtree(data_dir)

    # Make folder and write metadata
    os.makedirs(out_dir, exist_ok=True)
    dataset.write_metadata(out_dir)

    # Write sample data
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_samples):
        file = data_dir / f"{uuid.uuid4()}.npz"
        tens = final[i, :].cpu().numpy()
        np.savez(file, data=tens, params=param_vec.cpu().numpy())


if __name__ == "__main__":
    args = parser.parse_args()

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

    if args.batch_size is not None:
        sampling_config["batch_size"] = args.batch_size

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
        noise_sampler = noise.RBFKernel(channels, resolution, scale=scale, device=DEVICE)
    else:
        raise NotImplementedError()

    # Sample in batches
    for i, batch_num_samples in enumerate(batches):
        sample(model, noise_sampler, batch_num_samples, sampling_config)

        # Make sure we don't remove old samples
        sampling_config["replace_samples"] = False
