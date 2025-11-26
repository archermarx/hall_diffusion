# Stdlibs
import argparse
import tomllib
from pathlib import Path
import os
import shutil
import uuid
import math
import bisect

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

def build_observation_operator(dataset, observations):
    num_channels = len(dataset.fields)
    resolution = len(dataset.grid)

    obs_matrix = torch.zeros(num_channels, resolution)
    grid = dataset.grid

    m = num_channels * resolution
    n = 0

    for (obs_field, obs_data) in observations.items():
        ind = dataset.fields[obs_field]

        # print(f"{ind=}, {obs_field=}")
        loc = obs_data.get("locs", "all")

        if loc == "all":
            obs_matrix[ind, :] = 1.0
            n += resolution
        elif isinstance(loc, list):
            for z in loc:
                j = bisect.bisect_left(grid, z)
                obs_matrix[ind, j] = 1.0
                n += 1

    # m = num_channels * resolution
    # n = num_observations

    A = torch.zeros(n, m)
    obs_matrix = obs_matrix.reshape(-1)

    j = 0
    for i in range(m):
        if obs_matrix[i] == 1.0:
            A[j, i] = 1.0
            j += 1

    return A 


def edm_sampling_timesteps(num_steps, noise_min, noise_max, exponent):
    inv_rho = 1 / exponent
    i = torch.arange(0, num_steps)
    f1 = noise_max**inv_rho
    f2 = (noise_min**inv_rho - noise_max**inv_rho) / (num_steps - 2)
    timesteps = (f1 + i * f2) ** exponent
    timesteps[-1] = 0
    return timesteps


def guidance_score(x_t, x_0, obs_var, proc_var, pde_strength, observations, mask, dataset):
    (batch_size, _, _) = x_0.shape

    obs_vec = (mask @ observations.reshape(-1))
    var = (mask @ obs_var.reshape(-1))

    H = mask
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
    ims,
    masks,
    dataset,
    obs_var=None,
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
    if obs_var is not None and t_mid < t_max_guidance:
        obs_score = guidance_score(x_t, x_denoised, obs_var, proc_var, pde_strength, ims, masks, dataset)
        # x_1 += dt * t_mid * obs_score
        x_pred += obs_score

    return x_pred.detach()

def reverse(denoiser, x, timesteps, dataset, im_masks=None, showprogress=False, model_args=dict(), **kwargs):
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

    # Conditional diffusion/infill
    # Provide images to infill as well as mask indicating the area to be infilled
    if im_masks is not None:
        ims, masks = im_masks
    else:
        # If no mask provided, then we use a mask of all ones and a blank reference image
        masks = torch.zeros_like(x, device=x.device)
        ims = torch.zeros_like(x, device=x.device)

    ims.requires_grad = False
    masks.requires_grad = False

    for step_idx, t in enumerate(tqdm(timesteps, disable=(not showprogress))):
        if step_idx == 0:
            continue

        t_prev = timesteps[step_idx - 1]
        x = reverse_step(denoiser, x, t_prev, t, ims, masks, dataset, model_args=model_args, **kwargs)
        output[step_idx, ...] = x

    # ones = torch.ones((b, 1, 1), device=x.device)
    # for i in range(20):
    #     sigma = 0.002
    #     x += torch.randn_like(x, device=x.device) * sigma
    #     x = reverse_step(denoiser, x, sigma, 0, ims, masks, dataset, model_args=model_args, **kwargs)
    
    # with torch.no_grad():
    #     x = denoiser(x, 0.002 * ones, **model_args)

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
    obs_args = args["observation"]
    obs_fields = obs_args["fields"]
    obs_file = Path(obs_args["file"])
    obs_stddev_default = obs_args.get("std_dev", 1.0)
    obs_stddev = []

    for _, field_args in obs_fields.items():
        obs_stddev.append(field_args.get("std_dev", obs_stddev_default))

    # Load data
    dataset = ThrusterDataset(obs_file)
    _, data_vec, data = dataset[0]
    condition_vec = torch.tensor(data_vec, device=DEVICE)
    data = torch.tensor(data, device=DEVICE).unsqueeze(0)
    (_, channels, resolution) = data.shape

    # Observation conditioning
    mask = build_observation_operator(dataset, obs_fields).to(DEVICE)

    # Assign observation variances
    obs_var = torch.zeros((channels, resolution), device=DEVICE)
    indices_to_keep = [dataset.fields[field] for field in obs_fields]
    for i, index in enumerate(indices_to_keep):
        obs_var[index, :] = obs_stddev[i]**2

    # Sample initial noise
    xt = noise_sampler.sample(num_samples) * noise_max

    # Load timesteps
    steps = edm_sampling_timesteps(num_steps, noise_min, noise_max, exponent)

    output = reverse(
        model,
        xt,
        steps,
        dataset,
        im_masks=(data, mask),
        showprogress=True,
        obs_var=obs_var,
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

    os.makedirs(out_dir, exist_ok=True)

    # Write normalization info
    dataset.metadata_params.to_csv(out_dir / "norm_params.csv")
    dataset.metadata_plasma.to_csv(out_dir / "norm_data.csv")
    dataset.metadata_grid.to_csv(out_dir / "grid.csv")

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
