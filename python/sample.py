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

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("config", type=str)

DEVICE = torch.device("cpu")


def edm_sampling_timesteps(num_steps, noise_min, noise_max, exponent):
    inv_rho = 1 / exponent
    i = torch.arange(0, num_steps)
    f1 = noise_max**inv_rho
    f2 = (noise_min**inv_rho - noise_max**inv_rho) / (num_steps - 2)
    timesteps = (f1 + i * f2) ** exponent
    timesteps[-1] = 0
    return timesteps


def guidance_score(x, obs_variance, pde_strength, observations, masks):
    (batch_size, _, _) = x.shape

    grad = torch.full_like(x, 0.0, device=x.device)

    for b in range(batch_size):
        batch_x0 = x[b, ...]
        batch_x0.requires_grad = True

        obs = observations[b, ...]
        mask = masks[b, ...]

        obs_loss = torch.sum(mask * ((obs - batch_x0) ** 2 / obs_variance)) / 2
        pde_loss = pde_strength * (0.0 * batch_x0).mean()
        total_loss = obs_loss + pde_loss

        total_loss.backward()
        grad[b, ...] = batch_x0.grad

    return grad


def reverse_step(
    D,
    x_t,
    t_prev,
    t,
    ims,
    masks,
    obs_variance=None,
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

    # Guidance loss
    if obs_variance is not None and t_mid < t_max_guidance:
        obs_score = guidance_score(x_0, obs_variance, pde_strength, ims, masks)
        # x_1 += 0.5 * dt * t_prev * obs_score
        x_1 -= 0.5 * obs_score

    # Corrector step (midpoint rule)
    with torch.no_grad():
        x_0 = D(x_1, t_mid * ones)
        d1 = -(x_0 - x_1) / t_mid
        x_1 = x_t + step_scale * dt * d1

    # Guidance loss
    if obs_variance is not None and t_mid < t_max_guidance:
        obs_score = guidance_score(x_0, obs_variance, pde_strength, ims, masks)
        # x_1 += dt * t_mid * obs_score
        x_1 -= obs_score

    return x_1


def reverse(D, x, timesteps, im_masks=None, showprogress=False, **kwargs):
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
        x = reverse_step(D, x, t_prev, t, ims, masks, **kwargs)
        output[step_idx, ...] = x

    output[-1, ...] = x
    return output


def sample(model, num_samples, args):
    # Get fields to keep
    num_steps = args.get("num_steps", 256)
    noise_min = args.get("noise_min", 0.002)
    noise_max = args.get("noise_max", 80)
    exponent = args.get("step_exponent", 7)
    observation_stddev = args.get("observation_stddev", 1.0)
    step_scale = args.get("step_scale", 1.0)

    # TODO: make ThrusterDataset take a list of files
    dataset = ThrusterDataset(args["condition_file"])
    _, data_vec, data = dataset[0]
    condition_vec = torch.tensor(data_vec, device=DEVICE)

    # Load data and tile along batch dimension for
    data = torch.tensor(data, device=DEVICE)
    (num_channels, resolution) = data.shape
    data = data.unsqueeze(0).repeat(num_samples, 1, 1)

    # Initial noise samples
    xt = torch.randn((num_samples, num_channels, resolution), device=DEVICE) * noise_max
    assert data.shape == xt.shape

    # Observation conditioning
    obs_locations = np.arange(0, resolution, 1)
    indices_to_keep = [dataset.fields[field] for field in args["fields_to_keep"]]
    mask = torch.zeros((num_samples, num_channels, resolution), device=DEVICE)
    for i in indices_to_keep:
        if i == 0:
            mask[:, i, :] = 1.0
            continue
        mask[:, i, obs_locations] = 1.0

    # Assign observation variances
    obs_variance = torch.ones((num_channels, resolution), device=DEVICE)
    if isinstance(observation_stddev, list):
        for i, index in enumerate(indices_to_keep):
            obs_variance[index, :] = observation_stddev[i] ** 2
    else:
        obs_variance *= observation_stddev**2

    # Load timesteps
    steps = edm_sampling_timesteps(num_steps, noise_min, noise_max, exponent)

    # Bake in conditioning information
    def D(x, sigma):
        return model(x, sigma, condition_vector=condition_vec)

    output = reverse(
        D,
        xt,
        steps,
        im_masks=(data, mask),
        showprogress=True,
        obs_variance=obs_variance,
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
