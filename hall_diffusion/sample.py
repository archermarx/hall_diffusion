# Stdlibs
import argparse
import tomllib
from pathlib import Path
import os
import shutil
import uuid
import warnings

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
parser.add_argument(
    "--device",
    choices=("auto", "cpu", "mps", "cuda", "xpu"),
    default="auto",
    help="Compute backend to use (default: auto; priority: cuda, mps, xpu, cpu)",
)

LEGACY_MEASUREMENT_NOISE_SCALE = 40.0
ERROR_TYPES = {"absolute", "relative"}
ERROR_SPACES = {"normalized", "unnormalized"}


LEGACY_OBSERVATION_KEYS = {"fields", "params"}
LEGACY_MEASUREMENT_KEYS = {"x", "y", "locs", "normalized", "error_type", "error_space", "stddev", "std_dev"}


def _validate_observation_interface(observations):
    if legacy := LEGACY_OBSERVATION_KEYS.intersection(observations):
        names = ", ".join(sorted(legacy))
        raise ValueError(
            f"Legacy observation key(s) {names} are no longer supported; "
            "move all fields and parameters under 'measurements'."
        )
    if legacy := LEGACY_MEASUREMENT_KEYS.intersection(observations):
        names = ", ".join(sorted(legacy))
        raise ValueError(f"Observation uses retired flat error key(s) {names}; use the nested 'error' table.")
    if "measurements" not in observations:
        raise ValueError("Observation configuration requires a 'measurements' table.")
    if not isinstance(observations["measurements"], dict):
        raise ValueError("observation.measurements must be a table.")


def _validate_measurement_keys(name, measurement, allowed):
    if legacy := LEGACY_MEASUREMENT_KEYS.intersection(measurement):
        names = ", ".join(sorted(legacy))
        raise ValueError(f"Measurement '{name}' uses retired key(s) {names}; use the unified measurement schema.")
    if unexpected := set(measurement).difference(allowed):
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"Measurement '{name}' has unsupported key(s): {names}.")


def _error_spec(observations, measurement, name, required):
    """Resolve and validate observation-level error defaults and overrides."""
    global_error = observations.get("error")
    local_error = measurement.get("error")
    if global_error is None and local_error is None:
        if required:
            raise ValueError(
                f"Tensor measurement '{name}' requires an error specification; use stddev = 0 for an exact value."
            )
        return None
    if global_error is not None and not isinstance(global_error, dict):
        raise ValueError("observation.error must be a table.")
    if local_error is not None and not isinstance(local_error, dict):
        raise ValueError(f"Measurement '{name}' error must be a table.")

    error = dict(global_error or {})
    error.update(local_error or {})
    if missing := {"type", "space", "stddev"}.difference(error):
        raise ValueError(f"Error for measurement '{name}' is missing {sorted(missing)}.")
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
    if not torch.all(torch.isfinite(stddev)) or torch.any(stddev < 0):
        raise ValueError("error.stddev must be finite and nonnegative")

    if error["space"] == "normalized":
        reference = normalized_values
    else:
        reference = normalizer.denormalize(normalized_values, field)

    if error["type"] == "relative":
        stddev = stddev * reference.abs()

    if error["space"] == "unnormalized":
        stddev = normalizer.normalize_stddev(stddev, field, reference=reference)

    return stddev


def _batch_condition_vector(values, num_samples, device):
    values = torch.as_tensor(values, dtype=torch.float32, device=device)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError("condition_vec must be one- or two-dimensional")
    if values.shape[0] == 1:
        values = values.expand(num_samples, -1).clone()
    elif values.shape[0] != num_samples:
        raise ValueError(f"condition_vec has {values.shape[0]} rows, expected 1 or {num_samples}")
    return values


def _explicit_normalized_value(measurement, name, normalizer, device):
    if "value_space" not in measurement:
        raise ValueError(f"Measurement '{name}' provides a value but does not specify value_space.")
    value_space = measurement["value_space"]
    if value_space not in ERROR_SPACES:
        raise ValueError(f"Measurement '{name}' value_space must be 'normalized' or 'unnormalized'.")
    value = torch.as_tensor(measurement["value"], dtype=torch.float32, device=device)
    if value.ndim != 0:
        raise ValueError(f"Scalar measurement '{name}' requires a scalar value.")
    return normalizer.normalize(value, name) if value_space == "unnormalized" else value


def _validate_scalar_measurement(name, measurement):
    _validate_measurement_keys(name, measurement, {"value", "value_space", "error"})
    if "value_space" in measurement and "value" not in measurement:
        raise ValueError(f"Scalar measurement '{name}' has value_space but no value.")
    if "value" in measurement and isinstance(measurement["value"], (list, tuple)):
        raise ValueError(f"Scalar measurement '{name}' requires one value, not per-cell values.")


def _validate_scalar_error(name, error):
    if error is not None and torch.as_tensor(error["stddev"]).ndim != 0:
        raise ValueError(f"Scalar measurement '{name}' requires one error.stddev, not per-cell uncertainties.")


def build_observation(
    dataset,
    observations,
    num_samples,
    param_vec=None,
    sampling_mode="dps",
    device="cpu",
    verbose=False,
):
    _validate_observation_interface(observations)
    _, data_params, data_tensor = dataset[0]
    data_tensor = data_tensor.to(device)

    noise_std_scale = LEGACY_MEASUREMENT_NOISE_SCALE if sampling_mode == "constant" else 1.0

    (num_channels, resolution) = data_tensor.shape

    spatial_fields = dataset.spatial_fields()
    input_params = dataset.input_params()
    performance_scalars = dataset.performance_scalars()
    tensor_channels = dataset.tensor_channels()

    if param_vec is None:
        param_vec = data_params.detach().clone()
    param_vec = _batch_condition_vector(param_vec, num_samples, device)

    operator_rows = []
    observed_values = []
    observed_variances = []
    ignored_param_errors = []
    measurements = observations["measurements"]

    for name, measurement in measurements.items():
        if not isinstance(measurement, dict):
            raise ValueError(f"Measurement '{name}' must be a table.")

        if name in input_params and not dataset.scalars_in_tensor:
            _validate_scalar_measurement(name, measurement)
            error = _error_spec(observations, measurement, name, required=False)
            _validate_scalar_error(name, error)
            if error is not None:
                ignored_param_errors.append(name)
            if "value" in measurement:
                param_vec[:, input_params[name]] = _explicit_normalized_value(
                    measurement, name, dataset.norm, device
                )
            continue

        if name in performance_scalars and not dataset.scalars_in_tensor:
            raise ValueError(
                f"Performance scalar '{name}' cannot be measured when scalars_in_tensor is false."
            )

        if name in input_params or name in performance_scalars:
            _validate_scalar_measurement(name, measurement)
            error = _error_spec(observations, measurement, name, required=True)
            _validate_scalar_error(name, error)
            row_index = tensor_channels[name]
            if "value" in measurement:
                normalized_value = _explicit_normalized_value(measurement, name, dataset.norm, device)
            else:
                normalized_value = data_tensor[row_index].mean()

            row = torch.zeros(num_channels * resolution, dtype=data_tensor.dtype, device=device)
            row[row_index * resolution : (row_index + 1) * resolution] = 1.0 / resolution
            stddev = noise_std_scale * _normalized_error_stddev(
                error, normalized_value, name, dataset.norm
            )
            operator_rows.append(row)
            observed_values.append(normalized_value.reshape(1))
            observed_variances.append(stddev.square().reshape(1))
            continue

        if name not in spatial_fields:
            raise ValueError(f"Unknown measurement '{name}'.")

        _validate_measurement_keys(name, measurement, {"locations", "values", "value_space", "error"})
        if "value_space" in measurement and "values" not in measurement:
            raise ValueError(f"Spatial measurement '{name}' has value_space but no values.")
        if "values" in measurement and not isinstance(measurement["values"], (list, tuple)):
            raise ValueError(f"Spatial measurement '{name}' requires an array of values.")
        if "values" in measurement and "value_space" not in measurement:
            raise ValueError(f"Measurement '{name}' provides values but does not specify value_space.")
        error = _error_spec(observations, measurement, name, required=True)
        x_inds, _, y_data = utils.get_observation_locs(
            measurements, name, dataset.grid, normalizer=dataset.norm, form="normalized"
        )
        x_inds = np.asarray(x_inds)
        unique_inds, first_occurrences = np.unique(x_inds, return_index=True)
        x_inds = unique_inds.tolist()
        if y_data is None:
            normalized_values = data_tensor[tensor_channels[name], x_inds]
        else:
            normalized_values = torch.as_tensor(y_data, dtype=data_tensor.dtype, device=device).flatten()
            normalized_values = normalized_values[torch.as_tensor(first_occurrences, device=device)]

        stddev = noise_std_scale * _normalized_error_stddev(error, normalized_values, name, dataset.norm)
        if stddev.ndim == 0:
            stddev = stddev.expand(normalized_values.numel())
        for cell_index in x_inds:
            row = torch.zeros(num_channels * resolution, dtype=data_tensor.dtype, device=device)
            row[tensor_channels[name] * resolution + cell_index] = 1.0
            operator_rows.append(row)
        observed_values.append(normalized_values)
        observed_variances.append(stddev.square())
        if verbose:
            print(f"{name}:\tobserving {len(x_inds)} location(s).")

    if ignored_param_errors:
        names = ", ".join(sorted(ignored_param_errors))
        warnings.warn(
            f"Ignoring uncertainty for non-tensorized parameter(s): {names}; their values are exact conditions.",
            UserWarning,
            stacklevel=2,
        )

    if operator_rows:
        operator = torch.stack(operator_rows)
        obs_y = torch.cat(observed_values)
        obs_var = torch.cat(observed_variances)
    else:
        operator = None
        obs_y = None
        obs_var = None

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
        dataset = ThrusterDataset(
            obs_file,
            downsample_res=resolution,
            scalars_in_tensor=scalars_in_tensor,
            fourier_features=fourier_features,
        )

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
    dataset_settings = models.dataset_settings(model_config)
    scalars_in_tensor = dataset_settings["scalars_in_tensor"]
    fourier_features = dataset_settings["fourier_features"]

    if verbose:
        print(f"{model_config=}")

    model = models.from_config(model_config.copy(), device=device)

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

    infer(
        args.model,
        sampling_config,
        condition_vec=None,
        device=args.device,
    )
