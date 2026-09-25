# Stdlibs
import argparse
import tomllib
from pathlib import Path
import os
import shutil
import uuid
import warnings

# Third-party deps
import h5py
import torch
import numpy as np

# Local deps
from hall_diffusion import models
from hall_diffusion.adapter_data import condition_artifact_config, preprocess_condition
from hall_diffusion.models.adapter_io import load_adapter
from hall_diffusion.models.conditioning import ConditionedEDM2
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
            diagonal_covariance=bool(
                obs_operator is not None
                and torch.all(torch.count_nonzero(obs_operator, dim=0) <= 1).item()
            ),
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
    adapter_contexts=None,
    adapter_scales=None,
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

    use_amp = args.get("use_amp", False)
    if not isinstance(use_amp, bool):
        raise TypeError("use_amp must be a boolean")

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
        use_amp=use_amp,
    )
    sampler = EDMSampler(shape, num_steps, noise_min, noise_max, exponent)

    record_trajectory = args.get("record_trajectory", False)
    model_args = dict(condition_vector=param_vec)
    if adapter_contexts is not None:
        model_args["contexts"] = adapter_contexts
        model_args["adapter_scales"] = adapter_scales
    output = sampler.sample(
        integrator,
        showprogress=args.get("show_progress", True),
        device=device,
        model_args=model_args,
        record_trajectory=record_trajectory,
        finite_check_interval=args.get("finite_check_interval", 0),
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


def _artifact_condition_settings(artifact):
    """Read portable condition settings, including from legacy training metadata."""
    settings = artifact.get("condition_config")
    if settings is not None:
        return settings
    return (
        (artifact.get("train_config") or {})
        .get("adapter", {})
        .get("condition", {})
    )


def _resolve_condition_settings(spec, artifact):
    """Prefer artifact input semantics and fill missing values from sampling config."""
    artifact_settings = _artifact_condition_settings(artifact)
    keys = {
        "data_key": ("condition_data_key", "tlpp_counts"),
        "id_key": ("condition_id_key", "trace_uuid"),
        "add_channel_dim": ("condition_add_channel_dim", False),
        "add_occupancy_channel": ("condition_add_occupancy_channel", False),
        "transform": ("condition_transform", "none"),
        "scale": ("condition_scale", 1.0),
    }
    resolved = {
        key: artifact_settings.get(key, spec.get(sampling_key, default))
        for key, (sampling_key, default) in keys.items()
    }
    return condition_artifact_config(resolved)


def _expected_condition_ndim(encoder_type):
    return {"mlp": 1, "cnn1d": 2, "cnn2d": 3, "tlpp_vae": 3}[encoder_type]


def _prepare_adapter_condition(value, settings, encoder_type):
    """Apply training-time preprocessing to one raw condition or a condition batch."""
    value = torch.as_tensor(value)
    expected_ndim = _expected_condition_ndim(encoder_type)
    add_channel_dim = settings["add_channel_dim"]
    raw_sample_ndim = expected_ndim - int(add_channel_dim)

    if not add_channel_dim:
        is_batch = value.ndim == expected_ndim + 1
        valid = value.ndim in {expected_ndim, expected_ndim + 1}
        values = value if is_batch else value.unsqueeze(0)
        insert_channel = False
    elif value.ndim == raw_sample_ndim:
        is_batch = False
        valid = True
        values = value.unsqueeze(0)
        insert_channel = True
    elif value.ndim == expected_ndim:
        # A leading singleton can be an explicitly supplied channel. Otherwise
        # this is a batch of channel-less source values.
        is_batch = value.shape[0] != 1
        valid = True
        values = value if is_batch else value.unsqueeze(0)
        insert_channel = is_batch
    elif value.ndim == expected_ndim + 1:
        is_batch = True
        valid = value.shape[1] == 1
        values = value
        insert_channel = False
    else:
        is_batch = False
        valid = False
        values = value.unsqueeze(0)
        insert_channel = False

    if not valid:
        raise ValueError(
            f"the {encoder_type} encoder expects raw conditions with {raw_sample_ndim} dimensions "
            "per sample, with an optional singleton channel and optional batch dimension; "
            f"got shape {tuple(value.shape)}"
        )

    conditions = []
    for sample_value in values:
        if insert_channel:
            sample_value = sample_value.unsqueeze(0)
        conditions.append(
            preprocess_condition(
                sample_value,
                transform=settings["transform"],
                scale=settings["scale"],
                add_occupancy_channel=settings["add_occupancy_channel"],
            )
        )
    result = torch.stack(conditions)
    return result if is_batch else result[0]


def _load_adapter_condition(spec, encoder_type, condition_settings=None):
    """Load one condition record from an HDF5 product by UUID."""
    condition_settings = condition_settings or _resolve_condition_settings(spec, {})
    path = Path(spec["condition_file"])
    data_key = condition_settings["data_key"]
    id_key = condition_settings["id_key"]
    record_id = spec["condition_uuid"]
    with h5py.File(path, "r") as data:
        missing = [key for key in (data_key, id_key) if key not in data]
        if missing:
            raise KeyError(f"adapter condition file {path} is missing: {', '.join(missing)}")
        identifiers = data[id_key].asstr()[:]
        matches = np.flatnonzero(identifiers == record_id)
        if len(matches) != 1:
            raise ValueError(f"condition UUID {record_id!r} occurs {len(matches)} times in {path}")
        value = np.asarray(data[data_key][int(matches[0])])
    try:
        value = _prepare_adapter_condition(value, condition_settings, encoder_type)
    except ValueError as exc:
        raise ValueError(f"condition {record_id!r} in {path}: {exc}") from exc
    expected_ndim = _expected_condition_ndim(encoder_type)
    if value.ndim != expected_ndim:
        raise ValueError(
            f"condition {record_id!r} in {path} has shape {tuple(value.shape)}; "
            f"the {encoder_type} encoder expects {expected_ndim} dimensions per sample"
        )
    return value


def _batch_adapter_condition(value, batch_size, device):
    return value.unsqueeze(0).expand(batch_size, *value.shape).to(device)


def _adapter_condition_batch(
    value,
    encoder_type,
    start,
    batch_size,
    total_samples,
    device,
    *,
    expand_single=True,
):
    """Repeat one condition or select the matching portion of a supplied batch."""
    expected_ndim = _expected_condition_ndim(encoder_type)
    if value.ndim == expected_ndim:
        if expand_single:
            return _batch_adapter_condition(value, batch_size, device)
        return value.unsqueeze(0).to(device)
    if value.ndim != expected_ndim + 1:
        raise ValueError(
            f"processed {encoder_type} condition has shape {tuple(value.shape)}; "
            f"expected {expected_ndim} or {expected_ndim + 1} dimensions"
        )
    if value.shape[0] == 1:
        if expand_single:
            return _batch_adapter_condition(value[0], batch_size, device)
        return value.to(device)
    if value.shape[0] != total_samples:
        raise ValueError(
            f"adapter condition batch contains {value.shape[0]} samples, expected 1 or {total_samples}"
        )
    return value[start : start + batch_size].to(device)


def _variance_data_dir(sampling_config, train_config):
    """Find the calibration dataset for a missing process-variance cache."""
    explicit = sampling_config.get("process_variance_data_dir")
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"process variance data not found: {path}")
        return path

    checkpoint_test_dir = train_config.get("directories", {}).get("test_data_dir")
    candidates = [checkpoint_test_dir, sampling_config.get("unconditional_data_dir")]
    candidates = [Path(candidate) for candidate in candidates if candidate is not None]
    if path := next((candidate for candidate in candidates if candidate.exists()), None):
        return path

    checked = ", ".join(str(candidate) for candidate in candidates) or "none"
    raise FileNotFoundError(
        "process variance must be generated, but no calibration dataset was found "
        f"(checked: {checked}). Set 'process_variance_data_dir' in the sampling config."
    )


def _generate_process_variance(
    variance_file,
    sampling_config,
    train_config,
    model,
    dataset_settings,
    device,
):
    """Generate a missing process-variance artifact for the loaded model."""
    from hall_diffusion.estimate_variance import estimate_process_variance

    data_dir = _variance_data_dir(sampling_config, train_config)
    dataset = ThrusterDataset(data_dir, **dataset_settings)
    print(f"Estimating process variance from {data_dir}; this is only needed once.")
    variance_file = Path(variance_file)
    temporary_file = variance_file.with_name(f".{variance_file.stem}.{uuid.uuid4().hex}.tmp.npz")
    default_batch_size = sampling_config.get("batch_size", sampling_config.get("num_samples", 64))
    try:
        estimate_process_variance(
            model,
            dataset,
            temporary_file,
            device,
            seed=int(sampling_config.get("process_variance_seed", 0)),
            bias_subsets=int(sampling_config.get("process_variance_bias_subsets", 4)),
            batch_size=int(sampling_config.get("process_variance_batch_size", default_batch_size)),
            num_workers=int(sampling_config.get("process_variance_workers", 0)),
            create_plots=False,
        )
        os.replace(temporary_file, variance_file)
    finally:
        temporary_file.unlink(missing_ok=True)
    print(f"Stored process variance alongside the model at {variance_file}.")


def _ensure_process_variance(
    variance_file,
    sampling_config,
    train_config,
    model,
    dataset_settings,
    device,
):
    """Create the process-variance cache on first use and return its path."""
    variance_file = Path(variance_file)
    if not variance_file.exists():
        _generate_process_variance(
            variance_file,
            sampling_config,
            train_config,
            model,
            dataset_settings,
            device,
        )
    return variance_file


def infer(
    model,
    sampling_config,
    condition_vec=None,
    save_to_file=True,
    verbose=False,
    device="auto",
    adapter_conditions=None,
):
    """Load a checkpoint and sample, optionally using raw tensors keyed by adapter name."""
    device = utils.get_device(device) if isinstance(device, str) else device
    print(f"Selected device: {device}")

    # Load model and config from checkpoint
    checkpoint_path = Path(model)
    model_dict = utils.load_checkpoint(checkpoint_path, device)
    model_config = models.resolve_model_config(model_dict["model_config"])
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
    base_model = model
    train_config = model_dict.get("train_config", {})
    del model_dict

    loaded_adapters = []
    adapter_conditions = dict(adapter_conditions or {})
    supplied_condition_names = list(adapter_conditions)
    if any(not isinstance(name, str) or not name for name in supplied_condition_names):
        raise ValueError(
            "adapter condition names must be nonempty strings; "
            f"supplied names: {supplied_condition_names!r}"
        )
    adapter_specs = sampling_config.get("adapters", [])
    condition_fields = {"condition_file", "condition_uuid"}
    has_file_conditions = any(condition_fields.intersection(spec) for spec in adapter_specs)
    if adapter_specs and (adapter_conditions or has_file_conditions):
        adapters = {}
        loaded_artifacts = []
        configured_names = []
        for spec in adapter_specs:
            artifact_name, adapter, artifact = load_adapter(
                spec["checkpoint"], base_model, model_config, weights=spec.get("weights", "ema"),
            )
            name = spec.get("name", artifact_name)
            configured_names.append(name)
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "adapter names must be nonempty strings; "
                    f"configured names: {configured_names!r}"
                )
            if name in adapters:
                raise ValueError(
                    f"adapter name {name!r} appears more than once; "
                    f"configured names: {configured_names!r}"
                )
            adapters[name] = adapter
            loaded_artifacts.append((name, spec, artifact))

        available_names = list(adapters)
        if unknown := set(adapter_conditions).difference(adapters):
            raise ValueError(
                f"conditions were supplied for unknown adapters: {sorted(unknown)}; "
                f"supplied condition names: {sorted(supplied_condition_names)}; "
                f"available adapter names: {available_names}"
            )

        for name, spec, artifact in loaded_artifacts:
            encoder_type = artifact["adapter_config"]["encoder"]["type"]
            condition_settings = _resolve_condition_settings(spec, artifact)
            if name in adapter_conditions:
                condition = _prepare_adapter_condition(
                    adapter_conditions[name], condition_settings, encoder_type
                )
            else:
                if not condition_fields.intersection(spec):
                    continue
                missing = condition_fields.difference(spec)
                if missing:
                    raise ValueError(
                        f"adapter {name!r} has an incomplete file-based condition; "
                        f"missing sampling config fields {sorted(missing)}"
                    )
                condition = _load_adapter_condition(spec, encoder_type, condition_settings)
            loaded_adapters.append((name, spec, condition, encoder_type))
        if loaded_adapters:
            active_adapters = {name: adapters[name] for name, *_ in loaded_adapters}
            model = ConditionedEDM2(base_model, active_adapters).to(device).requires_grad_(False)
        del adapters, loaded_artifacts
    elif adapter_conditions:
        raise ValueError(
            "conditions were supplied but no adapters were configured; "
            f"supplied condition names: {sorted(supplied_condition_names)}; available adapter names: []"
        )

    # Switch model to evalution mode and sample
    model.eval()
    prepare_for_inference = getattr(base_model, "prepare_for_inference", None)
    if prepare_for_inference is not None:
        prepare_for_inference()

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
        variance_file = _ensure_process_variance(
            variance_file,
            sampling_config,
            train_config,
            base_model,
            dataset_settings,
            device,
        )
        variance_model = load_variance_model(
            variance_file, sampling_config, (channels, resolution), device
        )

    adapter_scales = {
        name: float(spec.get("scale", 1.0))
        for name, spec, _, _ in loaded_adapters
    }
    sampling_adapters = [
        adapter for adapter in loaded_adapters if adapter_scales[adapter[0]] != 0
    ]
    singleton_conditions = {}
    for name, _, condition, encoder_type in sampling_adapters:
        expected_ndim = _expected_condition_ndim(encoder_type)
        if condition.ndim == expected_ndim:
            singleton_conditions[name] = condition.unsqueeze(0).to(device)
        elif condition.shape[0] == 1:
            singleton_conditions[name] = condition.to(device)
    with torch.no_grad():
        cached_adapter_contexts = (
            model.prepare_conditions(singleton_conditions) if singleton_conditions else {}
        )

    samples = []
    batch_start = 0

    # Sample in batches
    for batch_index, batch_num_samples in enumerate(batches):
        size = (batch_num_samples, channels, resolution)
        batch_config = {
            **sampling_config,
            "replace_samples": sampling_config.get("replace_samples", False) and batch_index == 0,
        }
        adapter_contexts = None
        if sampling_adapters:
            raw_conditions = {
                name: _adapter_condition_batch(
                    condition,
                    encoder_type,
                    batch_start,
                    batch_num_samples,
                    num_samples,
                    device,
                    expand_single=False,
                )
                for name, _, condition, encoder_type in sampling_adapters
                if name not in cached_adapter_contexts
            }
            with torch.no_grad():
                prepared_contexts = model.prepare_conditions(raw_conditions) if raw_conditions else {}
                adapter_contexts = {
                    name: context.expand_batch(batch_num_samples)
                    for name, context in {**cached_adapter_contexts, **prepared_contexts}.items()
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
            adapter_contexts=adapter_contexts,
            adapter_scales=adapter_scales,
        )
        samples.append(batch_samples)
        batch_start += batch_num_samples

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
