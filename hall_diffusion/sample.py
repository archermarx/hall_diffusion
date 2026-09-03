# Stdlibs
import argparse
import tomllib
from pathlib import Path
import os
import shutil
import uuid

# Third-party deps
import torch
import numpy as np

# Local deps
from hall_diffusion import models
from hall_diffusion.models.controlnet import ControlNet
from hall_diffusion.guidance import guidance_score, legacy_guidance_score, load_variance_model
from hall_diffusion.utils import utils
from hall_diffusion.utils.thruster_data import ThrusterDataset
from hall_diffusion.samplers.edmsampler import EDMSampler, RK2Integrator, ObservationGuidance

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, nargs="?")
parser.add_argument("config", type=str, nargs="?")
parser.add_argument("-o", "--out-dir", type=str)
parser.add_argument("-n", "--num-samples", type=int)
parser.add_argument("-b", "--batch-size", type=int)
parser.add_argument("-s", "--num-steps", type=int)
parser.add_argument("--test-dir", type=Path)
parser.add_argument("--scalars-in-tensor", action="store_true")
parser.add_argument("--fourier-features", action="store_true")
parser.add_argument(
    "--device",
    choices=("auto", "cpu", "mps", "cuda", "xpu"),
    default="auto",
    help="Compute backend to use (default: auto; priority: cuda, mps, xpu, cpu)",
)

LEGACY_MEASUREMENT_NOISE_SCALE = 40.0
ERROR_TYPES = {"absolute", "relative"}
ERROR_SPACES = {"normalized", "unnormalized"}


def _error_spec(observations, field_observation):
    """Resolve an observation-level error specification and field overrides."""
    # Accept the old flat keys when reading existing observations. New configs
    # should use the nested ``error`` table.
    error = {}
    for source in (observations, field_observation):
        if "stddev" in source:
            error["stddev"] = source["stddev"]
        elif "std_dev" in source:
            error["stddev"] = source["std_dev"]
        if "error_type" in source:
            error["type"] = source["error_type"]
        if "error_space" in source:
            error["space"] = source["error_space"]
        error.update(source.get("error", {}))

    error.setdefault("stddev", 1.0)
    error.setdefault("type", "absolute")
    error.setdefault("space", "normalized")

    if error["type"] not in ERROR_TYPES:
        raise ValueError(f"error.type must be one of {sorted(ERROR_TYPES)}, got {error['type']!r}")
    if error["space"] not in ERROR_SPACES:
        raise ValueError(f"error.space must be one of {sorted(ERROR_SPACES)}, got {error['space']!r}")
    return error


def _normalized_error_stddev(error, normalized_values, field, normalizer):
    """Convert absolute/relative uncertainty in either space to model space."""
    stddev = torch.as_tensor(error["stddev"], dtype=normalized_values.dtype, device=normalized_values.device)
    if stddev.ndim > 1 or (stddev.ndim == 1 and stddev.numel() not in {1, normalized_values.numel()}):
        raise ValueError("error.stddev must be a scalar or have one value per observation")
    if torch.any(stddev < 0):
        raise ValueError("error.stddev cannot be negative")

    if error["space"] == "normalized":
        reference = normalized_values
    else:
        reference = normalizer.denormalize(normalized_values, field)

    if error["type"] == "relative":
        stddev = stddev * reference.abs()

    if error["space"] == "unnormalized":
        stddev = normalizer.normalize_stddev(stddev, field, reference=reference)

    return stddev


def build_observation(
    dataset,
    observations,
    num_samples,
    param_vec=None,
    sampling_mode="dps",
    device="cpu",
    verbose=False,
):
    _, data_params, data_tensor = dataset[0]
    data_tensor = data_tensor.to(device)

    noise_std_scale = LEGACY_MEASUREMENT_NOISE_SCALE if sampling_mode == "constant" else 1.0

    (num_channels, resolution) = data_tensor.shape

    observed = torch.zeros(num_channels, resolution, dtype=torch.bool, device=device)
    values = torch.zeros(num_channels, resolution, device=device)
    variances = torch.zeros(num_channels, resolution, device=device)

    grid = dataset.grid

    obs_fields = observations["fields"]

    for obs_field in obs_fields:
        # Get tensor row index
        row_index = dataset.fields()[obs_field]

        obs_dict = obs_fields[obs_field]
        error = _error_spec(observations, obs_dict)

        # Get observation from file
        x_inds, x_data, y_data = utils.get_observation_locs(
            obs_fields, obs_field, grid, normalizer=dataset.norm, form="normalized"
        )

        x_inds = np.asarray(x_inds)
        unique_inds, first_occurrences = np.unique(x_inds, return_index=True)
        x_inds = unique_inds.tolist()
        if y_data is None:
            normalized_values = data_tensor[row_index, x_inds]
        else:
            normalized_values = torch.as_tensor(y_data, dtype=data_tensor.dtype, device=device).flatten()
            normalized_values = normalized_values[torch.as_tensor(first_occurrences, device=device)]

        stddev = noise_std_scale * _normalized_error_stddev(
            error, normalized_values, obs_field, dataset.norm
        )

        if (len(x_data) == resolution) and np.all(x_inds == np.arange(resolution)):
            # If x_data == grid, then we're observing an entire row
            if verbose:
                print(obs_field + ":\tobserving entire row.")
            observed[row_index, :] = True

            values[row_index, :] = normalized_values
            variances[row_index, :] = stddev**2
        else:
            # Partial/sparse observation of the row
            # If y not provided, we use the underlying data matrix from the dataset
            # Otherwise we use the y found in the file
            observed[row_index, x_inds] = True
            variances[row_index, x_inds] = stddev**2

            if y_data is None:
                if verbose:
                    print(obs_field + ":\tusing data from ref sim at selected axial locs.")
                values[row_index, x_inds] = normalized_values
            else:
                if verbose:
                    print(obs_field + ":\tusing data from file.")
                values[row_index, x_inds] = normalized_values

    flat_indices = observed.flatten().nonzero(as_tuple=True)[0]
    operator = torch.eye(observed.numel(), device=device)[flat_indices]
    obs_y = values.flatten()[flat_indices]
    obs_var = variances.flatten()[flat_indices]

    # If no param vec specified here, we use the one from the reference dataset
    if param_vec is None:
        param_vec = data_params.detach().clone().to(device)
        param_vec = param_vec.unsqueeze(0).repeat(num_samples, 1)

    # Read scalar parameters if present
    if (params := observations.get("params", None)) is not None:
        for p, i in dataset.params().items():
            if p in params:
                param_vec[:, i] = dataset.norm.normalize(params[p], p)

    return operator, obs_y, obs_var, param_vec


def parse_observation(
    shape,
    args,
    scalars_in_tensor,
    fourier_features,
    variance_model=None,
    condition_vec=None,
    device="cpu",
    verbose=False,
):
    num_samples, _, resolution = shape
    # Determine if we're doing condional or unconditional sampling
    # If there is an `observation` field, then we're conditioning on a partial observation of that simulation
    # If not, we're sampling unconditionally
    # If we sample unconditonally, we need to get some scalar parameters to condition on
    # These are drawn from the same distributions as the training set
    if (uncond_dir := args.get("unconditional_data_dir", None)) is not None:
        unconditional_dataset = ThrusterDataset(
            uncond_dir,
            downsample_res=resolution,
            scalars_in_tensor=scalars_in_tensor,
            fourier_features=fourier_features,
        )
        param_vec = unconditional_dataset.sample_params(num_samples=num_samples, device=device)
    else:
        unconditional_dataset = None
        param_vec = None

    if condition_vec is not None:
        if not isinstance(condition_vec, torch.Tensor):
            param_vec = torch.tensor(condition_vec, device=device)
        else:
            param_vec = condition_vec.to(device)

    if verbose:
        print("sampling args: ", args)
    if "observation" in args:
        obs_args = utils.read_observation(args["observation"])
        obs_file = Path(obs_args["base_sim"])

        # Load data for conditioning
        dataset = ThrusterDataset(obs_file, scalars_in_tensor=scalars_in_tensor, fourier_features=fourier_features)

        if (obs_params := obs_args.get("params", None)) is not None:
            if set(obs_params) != set(dataset.params()) and param_vec is None:
                # We didn't completely specify the parameter vector and have nothing to fall back on
                raise RuntimeError("Incomplete parameter specification without data directory. Exiting.")

        obs_operator, obs_data, obs_var, param_vec = build_observation(
            dataset,
            obs_args,
            num_samples,
            param_vec,
            sampling_mode=args.get("sampling_mode", "dps"),
            device=device,
        )
        obs = dict(
            operator=obs_operator,
            data=obs_data,
            var=obs_var,
            covariance_jitter=args.get("covariance_jitter", 1e-6),
            variance_model=variance_model,
        )
    else:
        if param_vec is None or unconditional_dataset is None:
            raise RuntimeError("No observation specified and no data directory given. Exiting")

        dataset = unconditional_dataset
        obs = dict(operator=None, var=None, data=None)

    return obs, dataset, param_vec


def sample(
    model,
    shape,
    scalars_in_tensor,
    fourier_features,
    args,
    variance_model=None,
    condition_vec=None,
    save_to_file=True,
    device="cpu",
    verbose=False,
):
    num_samples, _, _ = shape

    obs, dataset, param_vec = parse_observation(
        shape,
        args,
        scalars_in_tensor,
        fourier_features,
        variance_model,
        condition_vec,
        device,
        verbose=verbose,
    )

    # Timestep args
    num_steps = args.get("num_steps", 256)
    noise_max = args.get("noise_max", 80.0)
    noise_min = args.get("noise_min", 0.002)
    exponent = args.get("step_exponent", 7.0)
    sampling_mode = args.get("sampling_mode", "dps")
    if sampling_mode not in {"dps", "constant"}:
        raise ValueError("sampling_mode must be 'dps' or 'constant'")
    score_function = legacy_guidance_score if sampling_mode == "constant" else guidance_score

    # Set up sampler
    integrator = RK2Integrator(
        model,
        guidance_score_fn=ObservationGuidance(
            type=sampling_mode,
            obs_score=score_function,
            observation=obs,
            guidance_start_time=args.get("guidance_start_time", float("inf")),
        ),
        method=args.get("method", None),
        rk_alpha=args.get("rk_alpha", 0.5),
        S_churn=args.get("S_churn", 0.0) / num_steps,
        S_tmin=args.get("S_tmin", 0.0),
        S_tmax=args.get("S_tmax", float("inf")),
        S_noise=args.get("S_noise", 1.003),
        guidance_second_order_below=args.get("guidance_second_order_below", 0.1),
    )
    sampler = EDMSampler(shape, num_steps, noise_min, noise_max, exponent)

    record_trajectory = args.get("record_trajectory", False)
    output = sampler.sample(
        integrator,
        showprogress=True,
        device=device,
        model_args=dict(condition_vector=param_vec),
        record_trajectory=record_trajectory,
    )

    final = output[-1, ...]

    if save_to_file:
        # Save generated samples
        out_dir = Path(args["out_dir"])
        data_dir = out_dir / "data"

        if args.get("replace_samples", False) and data_dir.exists():
            shutil.rmtree(data_dir)

        # Make folder and write metadata
        os.makedirs(out_dir, exist_ok=True)

        dataset.write_metadata(out_dir)

        # Write final sample data to independent output dirs
        os.makedirs(data_dir, exist_ok=True)
        params_cpu = param_vec.cpu().numpy()
        for i in range(num_samples):
            file = data_dir / f"{uuid.uuid4()}.npz"
            tens = final[i, :].cpu().numpy()
            if len(params_cpu.shape) == 1:
                np.savez(file, data=tens, params=params_cpu)
            else:
                np.savez(file, data=tens, params=params_cpu[i, :])

        if record_trajectory:
            np.savez(
                out_dir / "data_allsteps.npz",
                steps=sampler.noise_steps,
                data=output.cpu().numpy(),
                params=params_cpu,
            )

    return output


def infer(
    model,
    sampling_config,
    scalars_in_tensor,
    fourier_features,
    condition_vec=None,
    save_to_file=True,
    verbose=False,
    device="auto",
):
    device = utils.get_device(device) if isinstance(device, str) else device
    print(f"Selected device: {device}")

    # Load model and config from checkpoint
    checkpoint_path = Path(model)
    model_dict = utils.load_checkpoint(checkpoint_path, device)
    model_config = model_dict["model_config"]
    if "label_dim" in model_config:
        model_config["condition_dim"] = model_config.pop("label_dim")

    if verbose:
        print(f"{model_config=}")

    model = models.from_config(model_config, device=device)

    # Determine which weights to load
    model_type = sampling_config.get("model_type", "ema")
    assert model_type in ["ema", "best", "last"]
    model_type = "model" if model_type == "last" else model_type

    model.load_state_dict(model_dict[model_type], strict=False)
    model.requires_grad_(False)
    if isinstance(model, ControlNet):
        base_model = model.trained_unet
    else:
        base_model = model

    # Switch model to evalution mode and sample
    model.eval()

    num_samples = sampling_config.get("num_samples", 64)
    batch_size = sampling_config.get("batch_size", num_samples)

    full_batches, remainder = divmod(num_samples, batch_size)
    batches = [batch_size] * full_batches
    if remainder > 0:
        batches.append(remainder)

    channels = base_model.img_channels
    resolution = base_model.img_resolution

    variance_model = None
    sampling_mode = sampling_config.get("sampling_mode", "dps")
    if sampling_mode not in {"dps", "constant"}:
        raise ValueError("sampling_mode must be 'dps' or 'constant'")
    if "observation" in sampling_config and sampling_mode == "dps":
        variance_file = Path(
            sampling_config.get("process_variance_file", checkpoint_path.parent / "process_variance.npz")
        )
        variance_model = load_variance_model(
            variance_file, sampling_config, (channels, resolution), device
        )

    samples = []

    # Sample in batches
    for batch_index, batch_num_samples in enumerate(batches):
        size = (batch_num_samples, channels, resolution)
        batch_config = {
            **sampling_config,
            "replace_samples": sampling_config.get("replace_samples", False) and batch_index == 0,
        }
        batch_samples = sample(
            model,
            size,
            scalars_in_tensor,
            fourier_features,
            batch_config,
            variance_model=variance_model,
            condition_vec=condition_vec,
            save_to_file=save_to_file,
            device=device,
            verbose=verbose,
        )
        samples.append(batch_samples)

    # Concatenate along batch dimension
    sample_tensor = torch.concatenate(samples, dim=1)
    return sample_tensor


if __name__ == "__main__":
    args = parser.parse_args()

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

    scalars_in_tensor = args.scalars_in_tensor
    fourier_features = args.fourier_features

    infer(
        args.model,
        sampling_config,
        scalars_in_tensor,
        fourier_features,
        condition_vec=None,
        device=args.device,
    )
