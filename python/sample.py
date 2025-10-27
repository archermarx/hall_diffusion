# Python deps
import argparse
import math
import os
from pathlib import Path
import tomllib

# External deps
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Our deps
from utils.thruster_data import ThrusterPlotter1D, ThrusterDataset
from models.edm2 import Denoiser

CHECKPOINT = Path("checkpoint.pth.tar")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_DATASET = "data/val_large"

def observation_loss(y, x, mask):
    diff = (y - x) ** 2
    return torch.mean(((1 - mask) * diff))


def pde_loss(x):
    loss = (0.0 * x).mean()
    # UNIMPLEMENTED
    return loss


def guidance_loss(y, x, mask, obs_strength=0.0, pde_strength=0.0, **kwargs):
    # weighted average of observation and PDE losses
    return obs_strength * observation_loss(y, x, mask) + pde_strength * pde_loss(
        x, **kwargs
    )


class EDMDiffusionProcess:
    def __init__(
        self,
        noise_min=0.002,
        noise_max=80,
        exponent=7,
        reverse_stochastic=False,
        S_churn=40,
        S_min=0.002,
        S_max=50,
    ):
        self.exponent = exponent
        self.noise_min = noise_min
        self.noise_max = noise_max

        self.reverse_stochastic = reverse_stochastic
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max

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

    def reverse_step(
        self,
        denoiser,
        x_0,
        t_prev,
        t,
        ims,
        masks,
        gamma=0.0,
        condition_vec=None,
        obs_strength=0.0,
        pde_strength=0.0,
        step_scale=1.0,
    ):
        if gamma > 0:
            # If gamma > 0, we inject some stochasticity into the reverse process
            # We step from t_prev back to a previous time t_noise by injecting noise (t_noise**2 - t_prev**2)
            # We then proceed as normal, stepping from t_noise to t
            S_noise = 1.003
            t_noise = (1 + gamma) * t_prev
            noise_std = torch.sqrt(t_noise**2 - t_prev**2)
            gaussian_noise = torch.randn_like(x_0).to(x_0.device)
            x_0 += noise_std * S_noise * gaussian_noise
            t_prev = t_noise

        (b, c, w) = x_0.shape
        ones = torch.ones((b, 1, 1), device=device)

        # Predictor step
        dt = step_scale * (t - t_prev)
        with torch.no_grad():
            D = denoiser(x_0, t_prev * ones, condition_vector=condition_vec)
            d0 = (x_0 - D) / t_prev
            x1 = x_0 + dt * d0

            if t > 0:
                # Corrector step
                D = denoiser(x1, t * ones, condition_vector=condition_vec)
                d1 = (x1 - D) / t
                x1 = x_0 + dt * 0.5 * (d0 + d1)

        # Guidance
        for batch_idx in range(b):
            batch_D = D[batch_idx, ...]
            batch_D.requires_grad = True
            loss_obs = guidance_loss(
                ims[batch_idx, ...],
                batch_D,
                masks[batch_idx, ...],
                obs_strength=obs_strength,
                pde_strength=pde_strength,
            )
            loss_obs.backward()

            x1[batch_idx, ...] -= batch_D.grad

        return x1

    def reverse(
        self,
        denoiser,
        x,
        steps,
        im_masks=None,
        showprogress=False,
        condition_vec=None,
        obs_strength=0.0,
        pde_strength=0.0,
    ):
        """
        Perform iterative denoising to generate a 1D image from Gaussian noise using a provided denoising model

        Args:
            denoiser: a Denoiser model
            x: a tensor containing standard Gaussian noise. The dimension of this tensor should be (b, c, w)
                where b is the batch size, c is the number of channels, and w is the width
            im_masks: An optimal tuple of (data, mask) to apply during generation. These should have the same shape as x.
                When provided, the model will infill areas where the mask is set to 1.
            showprogress: whether we should print a tqdm progress bar.
        Returns:
        """
        (b, c, w) = x.shape

        output = torch.zeros((steps, b, c, w))
        output[0, ...] = x

        timesteps = self.get_timesteps(steps)
        # print(timesteps)
        gamma_0 = min(self.S_churn / steps, math.sqrt(2) - 1)

        # Conditional diffusion/infill
        # Provide images to infill as well as mask indicating the area to be infilled
        if im_masks is not None:
            ims, masks = im_masks
        else:
            # If no mask provided, then we use a mask of all ones and a blank reference image
            masks = torch.ones_like(x, device=x.device)
            ims = torch.zeros_like(x, device=x.device)

        ims.requires_grad = False
        masks.requires_grad = False
        # Step reference image through the forward process
        # forward = self.forward(ims, steps)
        step_scale_max = 1.1

        for step_idx, t in enumerate(tqdm(timesteps, disable=(not showprogress))):
            if step_idx == 0:
                continue

            if not self.reverse_stochastic or t < self.S_min or t > self.S_max:
                gamma = 0.0
            else:
                gamma = gamma_0

            t_prev = timesteps[step_idx - 1]

            step_scale = step_scale_max

            x = self.reverse_step(
                denoiser,
                x,
                t_prev,
                t,
                ims,
                masks,
                gamma,
                condition_vec=condition_vec,
                obs_strength=obs_strength,
                pde_strength=pde_strength,
                step_scale=step_scale,
            )

            # Add masked and noised reference image at the same noise level
            # Commented out in favor of other observationnoise conditioning method
            # x = masks * x + (1 - masks) * forward[step_idx]
            output[step_idx, ...] = x

        output[-1, ...] = x

        return output


def save_ims(
    ims,
    rows,
    data_dir: Path | str,
    path_2d: Path | str,
    path_1d: Path | str = "test.png",
    num_1d_samples=4,
    real: torch.Tensor | None = None,
    obs_locations=None,
    obs_fields=None,
):
    num = ims.shape[0]
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
    plotter.alphas = [0.1] * (num_1d_samples) + [1.0]

    fig, *_ = plotter.plot(
        ["nu_an", "ui_1", "ne", "E", "phi", "nn", "Tev"],
        denormalize=True,
        nrows=3,
        obs_fields=obs_fields,
        obs_locations=None,
    )
    fig.savefig(path_1d)


def sample(
    model,
    channels,
    data_dir: Path | str,
    steps=32,
    im_masks=None,
    path_2d="reverse.png",
    path_1d="test.png",
    showprogress=True,
):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """

    num_samples = 16
    im_size = 128
    seed = torch.randint(0, 1_000_000_000, (1,))[0]
    print(f"{seed=}")
    # seed = 902503797
    # torch.manual_seed(seed)

    process = EDMDiffusionProcess(exponent=7)

    xt = (
        torch.randn((num_samples, channels, im_size), device=device) * process.noise_max
    )

    output = process.reverse(
        model, xt, steps, im_masks=im_masks, showprogress=showprogress
    )

    final = output[-1, ...]

    save_ims(final, data_dir=data_dir, rows=4, path_2d=path_1d, path_1d="test.png")


def generate_conditionally(
    model,
    model_dir: Path | str,
    data_dir: Path | str,
    out_dir="samples",
    fields_to_keep=None,
    num_samples=32,
    num_steps=128,
    observation_guidance=500.0,
    **extras,
):

    # change these to determine what fields we should keep
    if fields_to_keep is None:
        fields_to_keep = ["ui_1", "B"]

    # Observation locations (grid point indices)
    obs_locations = np.arange(0, 128)

    # Scalar params for conditioning
    base_params = dict(
        anode_mass_flow_rate_kg_s=5e-6,
        discharge_voltage_v=300.0,
        magnetic_field_scale=1.0,
        cathode_coupling_voltage_v=25.0,
        wall_loss_scale=1.0,
        anom_shift_scale=0.25,
        neutral_velocity_m_s=300.0,
        neutral_ingestion_scale=4.0,
        background_pressure_torr=1e-5,
    )

    # Uncomment to use random seed
    #seed = torch.randint(0, 1_000_000_000, (1,))[0]
    #print(f"{seed=}")

    # Fixed seed
    seed = 123113385
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = ThrusterDataset(data_dir)
    indices_to_keep = [dataset.fields[field] for field in fields_to_keep]

    # Uncomment to use random data idx
    data_idx = 434
    print(f"{data_idx=}")
    _, data_vec, data = dataset[data_idx]

    data_params = dataset.vec_to_params(data_vec)
    base_params["discharge_voltage_v"] = data_params["discharge_voltage_v"]
    condition_vec = torch.tensor(dataset.params_to_vec(base_params), device=device)

    data = torch.tensor(data, device=device)
    (num_channels, resolution) = data.shape

    process = EDMDiffusionProcess()

    num_channels = len(dataset.fields)
    xt = (
        torch.randn((num_samples, num_channels, resolution), device=device)
        * process.noise_max
    )
    mask = torch.ones((num_samples, num_channels, resolution), device=device)
    for i in indices_to_keep:
        mask[:, i, obs_locations] = 0.0

    data_tiled = data.unsqueeze(0).repeat(num_samples, 1, 1)
    assert data_tiled.shape == xt.shape

    output = process.reverse(
        model,
        xt,
        num_steps,
        im_masks=(data_tiled, mask),
        showprogress=True,
        condition_vec=condition_vec,
        obs_strength=observation_guidance,
        pde_strength=0.0,
    )

    final = output[-1, ...]

    sample_dir = Path(model_dir) / out_dir
    os.makedirs(sample_dir, exist_ok=True)

    save_ims(
        final,
        rows=4,
        data_dir=Path(data_dir),
        path_2d=sample_dir / "conditional_2d.png",
        path_1d=sample_dir / "conditional_1d.png",
        num_1d_samples=num_samples,
        real=data,
        obs_locations=obs_locations,
        obs_fields=fields_to_keep,
    )


def infer(args):
    with open(args.config_file, "rb") as fp:
        config = tomllib.load(fp)

    model = Denoiser.from_config(config["model"]).to(device)

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
    model.eval()
    generate_conditionally(
        model, data_dir=dir_args["test_data_dir"], model_dir=out_dir, **sampling_args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()
    infer(args)
