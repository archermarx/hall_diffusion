# Python deps
import argparse
import math
import os
from pathlib import Path
import tomllib
import uuid

# External deps
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Our deps
from utils.thruster_data import ThrusterPlotter1D, ThrusterDataset
import models.edm
import models.edm2

CHECKPOINT = Path("checkpoint.pth.tar")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_DATASET = "data/val_large"


def observation_loss(y, x, mask, obs_variance):
    diff = (y - x) ** 2 / obs_variance
    return torch.sum(mask * diff) / 2


def pde_loss(x, strength):
    loss = (0.0 * x).mean()
    # UNIMPLEMENTED
    return strength * loss


def guidance_loss(y, x, mask, obs_variance=0.0, pde_strength=0.0, **kwargs):
    # weighted average of observation and PDE losses
    _obs_loss = observation_loss(x, y, mask, obs_variance)
    _pde_loss = pde_loss(x, pde_strength, **kwargs)

    return _obs_loss + _pde_loss


class EDMDiffusionProcess:
    def __init__(
        self,
        noise_min=0.002,
        noise_max=80,
        exponent=7,
    ):
        self.exponent = exponent
        self.noise_min = noise_min
        self.noise_max = noise_max

    def get_timesteps(self, N):
        rho = self.exponent
        inv_rho = 1 / rho

        i = torch.arange(0, N)
        f1 = self.noise_max**inv_rho
        f2 = (self.noise_min**inv_rho - self.noise_max**inv_rho) / (N - 2)
        timesteps = (f1 + i * f2) ** rho
        timesteps[-1] = 0
        return timesteps

    def forward(self, image, steps, normalize=False):
        (b, c, w) = image.shape

        timesteps = self.get_timesteps(steps)[:, None, None, None].to(image.device)

        steps = (
            torch.randn((steps, b, c, w)).to(image.device) * timesteps
            + image[None, ...]
        )

        if normalize:
            steps /= torch.clamp(timesteps, 1, torch.inf)

        return steps

    @staticmethod
    def guidance_score(x, deriv, t, obs_variance, pde_strength, ims, masks):
        (batch_size, width, height) = x.shape

        grad = torch.full_like(x, 0.0, device=x.device)
        # process_variance = t**2 / (t**2 + 1)
        process_variance = 0.0

        for b in range(batch_size):
            batch_x0 = x[b, ...]
            batch_x0.requires_grad = True

            loss_obs = guidance_loss(
                ims[b, ...],
                batch_x0,
                masks[b, ...],
                obs_variance=obs_variance + process_variance,
                pde_strength=pde_strength,
            )
            loss_obs.backward()
            grad[b, ...] = batch_x0.grad

        return grad

    def reverse_step(
        self,
        denoiser,
        x_t,
        t_prev,
        t,
        ims,
        masks,
        condition_vec=None,
        obs_variance=None,
        pde_strength=0.0,
        step_scale=1.0,
    ):
        (b, c, w) = x_t.shape

        ones = torch.ones((b, 1, 1), device=device)

        dt = t - t_prev
        t_mid = 0.5 * (t + t_prev)

        t_max_guidance = 100

        # Predictor step
        with torch.no_grad():
            x_0 = denoiser(x_t, t_prev * ones, condition_vector=condition_vec)
            d0 = -(x_0 - x_t) / t_prev
            x_1 = x_t + step_scale * 0.5 * dt * d0

        # Guidance loss
        if obs_variance is not None and t_mid < t_max_guidance:
            _obs_score = EDMDiffusionProcess.guidance_score(
                x_0, d0, t_prev, obs_variance, pde_strength, ims, masks
            )
            # x_1 += 0.5 * dt * t_prev * _obs_score
            x_1 -= 0.5 * _obs_score

        # Corrector step
        with torch.no_grad():
            x_0 = denoiser(x_1, t_mid * ones, condition_vector=condition_vec)
            d1 = -(x_0 - x_1) / t_mid
            x_1 = x_t + step_scale * dt * d1  # 0.5 * (d0 + d1)

        # Guidance loss
        if obs_variance is not None and t_mid < t_max_guidance:
            _obs_score = EDMDiffusionProcess.guidance_score(
                x_0, d1, t_mid, obs_variance, pde_strength, ims, masks
            )
            # x_1 += dt * t_mid * _obs_score
            x_1 -= _obs_score

        return x_1

    def reverse(
        self,
        denoiser,
        x,
        steps,
        im_masks=None,
        showprogress=False,
        condition_vec=None,
        obs_variance=None,
        pde_strength=0.0,
        step_scale=1.0,
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

        output = torch.zeros((steps, b, c, w))
        output[0, ...] = x

        timesteps = self.get_timesteps(steps)

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

            x = self.reverse_step(
                denoiser,
                x,
                t_prev,
                t,
                ims,
                masks,
                condition_vec=condition_vec,
                obs_variance=obs_variance,
                pde_strength=pde_strength,
                step_scale=step_scale,
            )

            # Add masked and noised reference image at the same noise level
            output[step_idx, ...] = x

        # for i in range(5):
        #     # A few extra consistency steps
        #     x = self.reverse_step(
        #         denoiser, x, 0.0, 0.0, ims, masks, condition_vec=condition_vec, obs_variance=obs_variance, pde_strength=pde_strength, step_scale=step_scale
        #     )

        output[-1, ...] = x

        return output


def save_samples(
    ims,
    rows,
    data_dir: Path | str,
    path_2d: Path | str,
    path_1d: Path | str = "test.png",
    path_tensors: Path | str = "samples",
    num_1d_samples=4,
    condition_vec=None,
    real: torch.Tensor | None = None,
    obs_locations=None,
    obs_fields=None,
):
    num = ims.shape[0]

    if condition_vec is None:
        condition_vec = np.zeros(1)
    else:
        condition_vec = condition_vec.cpu().numpy()

    # write to file
    # if os.path.exists(path_tensors):
    #    shutil.rmtree(path_tensors)

    os.makedirs(path_tensors, exist_ok=True)
    for i in range(num):
        file = Path(path_tensors) / f"{uuid.uuid4()}.npz"
        tens = ims[i, :].cpu().numpy()
        np.savez(file, data=tens, params=condition_vec)

    cols = math.ceil(num / rows)
    fig = plt.figure(constrained_layout=True)

    for i in range(num):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(
            ims[i, ...],
            aspect="auto",
            vmin=-1.5,
            vmax=1.5,
            interpolation="none",
            cmap="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for direction in ["top", "left", "bottom", "right"]:
            ax.spines[direction].set_visible(False)

    fig.savefig(path_2d)
    plt.close(fig)

    dataset = ThrusterDataset(data_dir, None, 1)

    sims = ims[:num_1d_samples, ...]
    if real is not None:
        sims = torch.concat((sims, real.to("cpu")[None, ...]), dim=0)
    sims = [s for s in sims]

    plotter = ThrusterPlotter1D(dataset, sims, obs_locations=obs_locations)
    plotter.colors = ["black"] * num_1d_samples + ["red"]
    plotter.alphas = [0.25 / np.cbrt(num_1d_samples)] * (num_1d_samples) + [1.0]

    fields_to_plot = [
        "ui_1",
        "Tev",
        "ne",
        "inverse_hall",
        "phi",
        "E",
    ]  # , "nn", "pe", "∇pe"]

    fig, *_ = plotter.plot(
        fields_to_plot,
        denormalize=True,
        nrows=math.ceil(len(fields_to_plot) / 3),
        obs_fields=obs_fields,
        obs_locations=None,
    )
    fig.savefig(path_1d)


def generate_conditionally(
    model,
    model_dir: Path | str,
    data_dir: Path | str,
    out_dir="samples",
    fields_to_keep=None,
    num_samples=32,
    num_steps=128,
    observation_stddev=1.0,
    condition_file=None,
    step_scale=1.0,
    **extras,
):
    # change these to determine what fields we should keep
    if fields_to_keep is None:
        fields_to_keep = ["ui_1", "B"]

    # Observation locations (grid point indices)
    obs_locations = np.arange(0, 128, 10)

    # Uncomment to use random seed
    seed = torch.randint(0, 1_000_000_000, (1,))[0]
    print(f"{seed=}")

    # Fixed seed
    # seed = 123113385
    torch.manual_seed(seed)
    np.random.seed(seed)

    if condition_file is None:
        dataset = ThrusterDataset(data_dir)
        data_idx = torch.randint(0, len(dataset.files), (1,))[0]
        print(f"{data_idx=}")
        _, data_vec, data = dataset[data_idx]
    else:
        # TODO: make ThrusterDataset take a list of files
        dataset = ThrusterDataset(condition_file)
        _, data_vec, data = dataset[0]

    indices_to_keep = [dataset.fields[field] for field in fields_to_keep]
    condition_vec = torch.tensor(data_vec, device=device)

    data = torch.tensor(data, device=device)
    (num_channels, resolution) = data.shape

    process = EDMDiffusionProcess()

    num_channels = len(dataset.fields)
    xt = (
        torch.randn((num_samples, num_channels, resolution), device=device)
        * process.noise_max
    )
    mask = torch.zeros((num_samples, num_channels, resolution), device=device)
    for i in indices_to_keep:
        if i == 0:
            mask[:, i, :] = 1.0
            continue
        mask[:, i, obs_locations] = 1.0

    # Assign observation variances
    obs_variance = torch.ones((num_channels, resolution), device=device)
    if isinstance(observation_stddev, list):
        for i, index in enumerate(indices_to_keep):
            obs_variance[index, :] = observation_stddev[i] ** 2
    else:
        obs_variance *= observation_stddev**2

    data_tiled = data.unsqueeze(0).repeat(num_samples, 1, 1)
    assert data_tiled.shape == xt.shape

    output = process.reverse(
        model,
        xt,
        num_steps,
        im_masks=(data_tiled, mask),
        showprogress=True,
        condition_vec=condition_vec,
        obs_variance=obs_variance,
        pde_strength=0.0,
        step_scale=step_scale,
    )

    final = output[-1, ...]

    sample_dir = Path(model_dir) / out_dir
    os.makedirs(sample_dir, exist_ok=True)

    save_samples(
        final,
        rows=4,
        data_dir=Path(data_dir),
        path_2d=sample_dir / "conditional_2d.png",
        path_1d=sample_dir / "conditional_1d.png",
        path_tensors=sample_dir / "tensors",
        condition_vec=condition_vec,
        num_1d_samples=num_samples,
        real=data,
        obs_locations=obs_locations,
        obs_fields=fields_to_keep,
    )


def infer(args):
    with open(args.config_file, "rb") as fp:
        config = tomllib.load(fp)

    architecture = config["model"].get("architecture")
    if architecture == "edm2":
        model = models.edm2.Denoiser.from_config(config["model"]).to(device)
    elif architecture == "edm":
        model = models.edm.Denoiser.from_config(config["model"]).to(device)

    dir_args = config["training"]["directories"]
    sampling_args = config["sampling"]

    out_dir = Path(dir_args["out_dir"])

    # Load checkpoint if found
    checkpoint = Path(out_dir) / "checkpoint.pth.tar"
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=False)

        # Determine which weights to load
        if sampling_args["use_ema"]:
            model.load_state_dict(state["ema"])
        else:
            model.load_state_dict(state["best"])

    print(f"{sampling_args=}")

    # Switch model to evaluation mode and sample
    if "condition_file" not in sampling_args:
        sampling_args["condition_file"] = None

    model.eval()
    generate_conditionally(
        model, data_dir=dir_args["test_data_dir"], model_dir=out_dir, **sampling_args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()
    infer(args)
