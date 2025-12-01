# Stdlibs
import argparse
import tomllib
from pathlib import Path
import os
import shutil
import uuid
import math

# Third-party deps
import torch
from tqdm import tqdm
import numpy as np

# Local deps
import models.edm
import models.edm2
from utils import utils
from utils.thruster_data import ThrusterDataset
import noise

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("config", type=str)

DEVICE = torch.device("cpu")

def build_observation(dataset, observations, default_stddev = 1.0, device=DEVICE):
    data_tensor = torch.tensor(dataset[0][2])
    (num_channels, resolution) = data_tensor.shape

    obs_matrix_loc = torch.zeros(num_channels, resolution)
    obs_matrix_dat = torch.zeros(num_channels, resolution)
    obs_matrix_var = torch.zeros(num_channels, resolution)

    default_stddev = observations.get("std_dev", 1.0)

    grid = dataset.grid

    m = num_channels * resolution
    n = 0

    obs_fields = observations["fields"]

    for obs_field in obs_fields:
        # Get tensor row index
        row_index = dataset.fields()[obs_field]

        # Get stddev (TODO: read from file to allow more values, but this needs to wait for improved stddev handling)
        stddev = obs_fields[obs_field].get("std_dev", default_stddev)

        # Get observation from file
        x_inds, x_data, y_data = utils.get_observation_locs(obs_fields, obs_field, grid, normalizer=dataset.norm, form="normalized")

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
            obs_matrix_loc[row_index, x_inds] = 1.0
            obs_matrix_var[row_index, x_inds] = stddev**2
            n += len(x_inds)

            if y_data is None:
                print("Using underlying data")
                obs_matrix_dat[row_index, x_inds] = data_tensor[row_index, x_inds]
            else:
                print("Using data from file")
                obs_matrix_dat[row_index, x_inds] = torch.tensor(y_data, dtype=torch.float32)

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

    return obs_A.to(DEVICE), obs_y.to(DEVICE), obs_var.to(DEVICE)


def edm_sampling_timesteps(num_steps, noise_min, noise_max, exponent):
    inv_rho = 1 / exponent
    i = torch.arange(0, num_steps)
    f1 = noise_max**inv_rho
    f2 = (noise_min**inv_rho - noise_max**inv_rho) / (num_steps - 2)
    timesteps = (f1 + i * f2) ** exponent
    timesteps[-1] = 0
    return timesteps


def guidance_score(x_t, x_0, observation, proc_var, pde_strength, dataset):
    (batch_size, _, _) = x_0.shape

    obs_vec = observation["data"]
    var = observation["var"]
    H = observation["operator"]
    H_pinv = torch.linalg.pinv(H)

    n, m = H.shape
    assert H_pinv.shape == (m, n)
    assert obs_vec.shape == (n,)

    # Precompute (rt^2 HH^T + sigma_y^2 I)^{-1} H
    mat1 = torch.inverse(proc_var * H @ H.T + var * torch.eye(n, device=DEVICE)) @ H
    assert mat1.shape == (n, m)

    # Pseudoinverse guidance (with noise)
    x_vec = x_0.reshape(batch_size, -1)
    vec1 = (obs_vec[None, ...] - torch.matmul(H, x_vec.T).T)
    assert vec1.shape == (batch_size, n)

    vec2 = (vec1 @ mat1)
    assert vec2.shape == (batch_size, m)

    mat_x = (vec2.detach() * x_vec).sum()
    score = torch.autograd.grad(mat_x, x_t)[0]

    return score

def reverse_step(
    denoiser,
    x_t,
    t_prev,
    t,
    observation,
    dataset,
    pde_strength=0.0,
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

    if (method == "midpoint"):
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

    # Guidance loss
    if observation["var"] is not None and t_mid < t_max_guidance:
        obs_score = guidance_score(x_t, x_denoised, observation, proc_var, pde_strength, dataset)
        # x_1 += dt * t_mid * obs_score
        x_pred += obs_score

    return x_pred.detach()

def reverse(denoiser, x, timesteps, dataset, observation, showprogress=False, model_args=dict(), **kwargs):
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
        x = reverse_step(denoiser, x, t_prev, t, observation, dataset, model_args=model_args, **kwargs)
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

    # Load observations
    obs_args = utils.read_observation(args["observation"])
    obs_file = Path(obs_args["base_sim"])

    # Load data for conditioning
    dataset = ThrusterDataset(obs_file)
    _, data_vec, _ = dataset[0]
    condition_vec = torch.tensor(data_vec, device=DEVICE)
    #data = torch.tensor(data, device=DEVICE).unsqueeze(0)
    
    # Observation conditioning
    obs_operator, obs_data, obs_var = build_observation(dataset, obs_args, device=DEVICE)
    obs = dict(operator=obs_operator, data=obs_data, var=obs_var)

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
        pde_strength=0.0,
        step_scale=step_scale,
        method=method,
        model_args=dict(condition_vector=condition_vec)
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
        np.savez(file, data=tens, params=condition_vec.cpu().numpy())


if __name__ == "__main__":
    args = parser.parse_args()
    DEVICE = utils.get_device()

    with open(args.config, "rb") as fp:
        sampling_config = tomllib.load(fp)

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
