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

    print(f"{A.shape=}")

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


def guidance_score(x, obs_var, proc_var, pde_strength, observations, mask, dataset):
    (batch_size, _, _) = x.shape

    grad = torch.full_like(x, 0.0, device=x.device)

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

    for b in range(batch_size):
        batch_x0 = x[b, ...]
        batch_x0.requires_grad = True

        x_vec = batch_x0.reshape(-1)
        assert x_vec.shape == (m,)

        # # Noise-free pseudoinverse guidance
        # mat = H_pinv @ obs_vec - (H_pinv @ (H @ x_vec.reshape(-1)))
        # mat_x = (mat.detach() * x_vec).sum()
        # guidance = torch.autograd.grad(mat_x, batch_x0)[0]
        # grad[b, ...] = guidance

        vec1 = (obs_vec - (H @ x_vec))
        assert vec1.shape == (n,)


        vec2 = (vec1 @ mat1)
        assert vec2.shape == (m,)
 
        mat_x = (vec2.detach() * x_vec).sum()
        score = torch.autograd.grad(mat_x, batch_x0)[0]

        grad[b, ...] = score

    return grad


def reverse_step(
    D,
    x_t,
    t_prev,
    t,
    ims,
    masks,
    dataset,
    obs_var=None,
    pde_strength=0.0,
    step_scale=1.0,
):
    (b, _, _) = x_t.shape

    ones = torch.ones((b, 1, 1), device=x_t.device)

    dt = t - t_prev
    t_mid = 0.5 * (t + t_prev)

    t_max_guidance = 100

    # Predictor step (midpoint rule)
    with torch.no_grad():
        x_0 = D(x_t, t_prev * ones)
        d0 = -(x_0 - x_t) / t_prev
        x_1 = x_t + step_scale * 0.5 * dt * d0

    proc_var = t**2 / (t**2 + 1)

    # Guidance loss
    if obs_var is not None and t_mid < t_max_guidance:
        obs_score = guidance_score(x_0, obs_var, proc_var, pde_strength, ims, masks, dataset)
        # x_1 += 0.5 * dt * t_prev * obs_score
        x_1 += 0.5 * obs_score

    # Corrector step (midpoint rule)
    with torch.no_grad():
        x_0 = D(x_1, t_mid * ones)
        d1 = -(x_0 - x_1) / t_mid
        x_1 = x_t + step_scale * dt * d1

    # Guidance loss
    if obs_var is not None and t_mid < t_max_guidance:
        obs_score = guidance_score(x_0, obs_var, proc_var, pde_strength, ims, masks, dataset)
        # x_1 += dt * t_mid * obs_score
        x_1 += obs_score

    return x_1


def reverse(D, x, timesteps, dataset, im_masks=None, showprogress=False, **kwargs):
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
        x = reverse_step(D, x, t_prev, t, ims, masks, dataset, **kwargs)
        output[step_idx, ...] = x

    output[-1, ...] = x
    return output


def sample(model, num_samples, args):
    # Load sampling arguments
    num_steps = args.get("num_steps", 256)
    noise_min = args.get("noise_min", 0.002)
    noise_max = args.get("noise_max", 80)
    exponent = args.get("step_exponent", 7)
    step_scale = args.get("step_scale", 1.0)

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

    # Tile data
    (_, num_channels, resolution) = data.shape
    data_denorm = dataset.denormalize_tensor(data)

    # Observation conditioning
    mask = build_observation_operator(dataset, obs_fields).to(DEVICE)

    # Assign observation variances
    obs_var = torch.zeros((num_channels, resolution), device=DEVICE)
    indices_to_keep = [dataset.fields[field] for field in obs_fields]
    for i, index in enumerate(indices_to_keep):
        # Get field scale
        #data_field = data[0, index, :]
        obs_std = obs_stddev[i]
        #print(obs_std)
        obs_var[index, :] = obs_std**2


    # Initial noise samples
    xt = torch.randn((num_samples, num_channels, resolution), device=DEVICE) * noise_max

    # Load timesteps
    steps = edm_sampling_timesteps(num_steps, noise_min, noise_max, exponent)

    # Bake in conditioning information
    def D(x, sigma):
        return model(x, sigma, condition_vector=condition_vec)

    output = reverse(
        D,
        xt,
        steps,
        dataset,
        im_masks=(data, mask),
        showprogress=True,
        obs_var=obs_var,
        pde_strength=0.0,
        step_scale=step_scale,
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

    # Sample in batches
    for i, batch_num_samples in enumerate(batches):
        sample(model, batch_num_samples, sampling_config)

        # Make sure we don't remove old samples
        sampling_config["replace_samples"] = False
