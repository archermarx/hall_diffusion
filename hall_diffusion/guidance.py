"""Moment-projected likelihood guidance for diffusion sampling.

For denoiser residual e = D(x_t, t) - x_0, the variance estimator provides
its mean b(t) and diagonal second moments q(t).  After optional bias
correction, a first-order approximation to a general measurement A gives

    x_0 | x_t ~= N(D(x_t, t) - b(t), diag(q(t)))
    y   | x_t ~= N(A(x_0), R + J_A diag(q(t)) J_A.T).

The projected covariance is treated as locally constant when differentiating
the likelihood, avoiding second derivatives of arbitrary measurement maps.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from hall_diffusion.observation_operators import IonCurrentObservation


@dataclass
class DPSCovarianceCache:
    """Bounded accelerator cache for sample-independent DPS covariance terms."""

    max_bytes: int = 64 * 1024**2
    values: dict[tuple, torch.Tensor] = field(default_factory=dict)
    used_bytes: int = 0

    def get_or_compute(self, key, compute):
        if key is not None and key in self.values:
            return self.values[key]
        value = compute().detach()
        size = value.numel() * value.element_size()
        if key is not None and self.used_bytes + size <= self.max_bytes:
            self.values[key] = value
            self.used_bytes += size
        return value


def load_variance_model(path, sampling_config, state_shape, device):
    """Load and validate diagonal denoiser-error statistics."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"process variance file not found: {path}. Set 'process_variance_file' in the sampling config."
        )

    bias_correction = sampling_config.get("process_bias_correction", False)
    with np.load(path) as data:
        required = {
            "process_variance",
            "centered_variance",
            "mean_residual",
            "noise_levels",
            "residual_convention",
            "variance_convention",
        }
        if missing := required.difference(data.files):
            raise ValueError(
                f"process variance file is missing {sorted(missing)}; regenerate it with estimate_variance.py"
            )
        variance_key = "centered_variance" if bias_correction else "process_variance"
        process_variance = torch.as_tensor(data[variance_key], device=device)
        mean_residual = torch.as_tensor(data["mean_residual"], device=device)
        noise_levels = torch.as_tensor(data["noise_levels"], device=device)
        residual_convention = str(data["residual_convention"].item())
        variance_convention = str(data["variance_convention"].item())

    expected_shape = (noise_levels.numel(), *state_shape)
    if process_variance.shape != expected_shape:
        raise ValueError(
            f"process variance has shape {tuple(process_variance.shape)}, expected {expected_shape}"
        )
    if mean_residual.shape != expected_shape:
        raise ValueError(f"mean_residual has shape {tuple(mean_residual.shape)}, expected {expected_shape}")
    if not torch.all(noise_levels[1:] > noise_levels[:-1]):
        raise ValueError("process variance noise levels must be strictly increasing")
    if torch.any(process_variance < 0) or not torch.all(torch.isfinite(process_variance)):
        raise ValueError("process variance values must be finite and nonnegative")
    if variance_convention != "uncentered_mse":
        raise ValueError("process variance file has an unsupported variance convention")
    if bias_correction and residual_convention != "denoised_minus_clean":
        raise ValueError("bias correction requires residual_convention='denoised_minus_clean'")

    scale = sampling_config.get("process_variance_scale", 1.0)
    bias_scale = sampling_config.get("process_bias_scale", 1.0)
    if scale < 0 or bias_scale < 0:
        raise ValueError("process variance and bias scales must be nonnegative")

    return {
        "noise_levels": noise_levels,
        "process_variance": process_variance,
        "mean_residual": mean_residual,
        "scale": scale,
        "channel_average": sampling_config.get("process_variance_channel_average", False),
        "bias_correction": bias_correction,
        "bias_scale": bias_scale,
    }


def _apply_operator(operator, x):
    if isinstance(operator, torch.Tensor):
        return x.flatten(start_dim=1) @ operator.T
    if isinstance(operator, IonCurrentObservation):
        return operator.batch_apply(x)
    if isinstance(operator, Callable):
        return torch.stack([operator(sample).reshape(-1) for sample in x])
    raise TypeError("observation operator must be a tensor or callable")


def _operator_jacobians(operator, x):
    if isinstance(operator, torch.Tensor):
        return operator.unsqueeze(0).expand(x.shape[0], -1, -1)
    if isinstance(operator, IonCurrentObservation):
        return operator.jacobians(x)

    def jacobian(sample):
        sample = sample.detach().requires_grad_(True)
        return torch.autograd.functional.jacobian(
            lambda value: operator(value).reshape(-1),
            sample,
            create_graph=False,
            vectorize=True,
        ).reshape(-1, sample.numel()).detach()

    return torch.stack([jacobian(sample) for sample in x])


def _interpolate(t, noise_levels, values):
    """Interpolate tabulated statistics linearly in log noise."""
    t = torch.as_tensor(t, dtype=noise_levels.dtype, device=noise_levels.device)
    t = t.clamp(min=noise_levels[0], max=noise_levels[-1])
    upper = torch.searchsorted(noise_levels, t).clamp(1, noise_levels.numel() - 1)
    lower = upper - 1
    weight = (t.log() - noise_levels[lower].log()) / (
        noise_levels[upper].log() - noise_levels[lower].log()
    )
    return torch.lerp(values[lower], values[upper], weight)


def _corrected_mean(x_0, t, variance_model):
    if not variance_model.get("bias_correction", False):
        return x_0
    noise_levels = variance_model["noise_levels"].to(device=x_0.device, dtype=x_0.dtype)
    bias = _interpolate(
        t,
        noise_levels,
        variance_model["mean_residual"].to(device=x_0.device, dtype=x_0.dtype),
    )
    return x_0 - variance_model.get("bias_scale", 1.0) * bias.unsqueeze(0)


def _projected_covariance(x_0, operator, t, variance_model):
    state_variance = _state_variance(x_0, t, variance_model)
    variance = state_variance.reshape(-1)
    if isinstance(operator, torch.Tensor):
        # A linear operator and q(t) are shared by every sample in the batch.
        return (operator * variance) @ operator.T
    jacobian = _operator_jacobians(operator, x_0)
    return (jacobian * variance) @ jacobian.transpose(-1, -2)


def _state_variance(x_0, t, variance_model):
    """Interpolate q(t), retaining its state shape."""
    noise_levels = variance_model["noise_levels"].to(device=x_0.device, dtype=x_0.dtype)
    variance = _interpolate(
        t,
        noise_levels,
        variance_model["process_variance"].to(device=x_0.device, dtype=x_0.dtype),
    )
    variance = variance * variance_model.get("scale", 1.0)
    if variance_model.get("channel_average", False):
        variance = variance.mean(dim=-1, keepdim=True).expand_as(variance)
    return variance


def _noise_covariance(variance, size, *, dtype, device):
    variance = torch.as_tensor(variance, dtype=dtype, device=device)
    if variance.ndim == 0:
        return variance * torch.eye(size, dtype=dtype, device=device)
    if variance.ndim == 1:
        if variance.numel() != size:
            raise ValueError("observation variance has the wrong length")
        return torch.diag(variance)
    if variance.shape != (size, size):
        raise ValueError("observation covariance has the wrong shape")
    return variance


def _covariance_cache_key(observation, kind, t, reference):
    cache = observation.get("covariance_cache")
    if cache is None:
        return None, None
    if isinstance(t, torch.Tensor):
        if t.numel() != 1 or t.device.type != "cpu":
            return cache, None
        t = t.item()
    return cache, (kind, float(t), reference.dtype, str(reference.device))


def _solve_ion_current_covariance(mean, residual, noise, t, observation):
    """Solve the linear/current covariance using its scalar Schur complement.

    The linear block is shared across samples and timesteps can be cached.
    Only the six endpoint derivatives, their cross-covariances with linear
    rows, and one conditional current variance depend on each sample.
    """
    operator = observation["operator"]
    if noise.ndim == 1 and noise.numel() != residual.shape[1]:
        raise ValueError("observation variance has the wrong length")
    noise = noise.expand(residual.shape[1])
    jitter = observation.get("covariance_jitter", 1e-6)
    state_variance = _state_variance(mean, t, observation["variance_model"]).reshape(-1)
    derivatives = operator.current.derivatives(mean)
    endpoint_variance = state_variance.index_select(0, operator.current.indices)
    weighted = derivatives * endpoint_variance
    current_variance = (derivatives * weighted).sum(dim=-1) + noise[operator.current_position] + jitter
    current_residual = residual[:, operator.current_position]
    if operator.linear.shape[0] == 0:
        return (current_residual / current_variance).unsqueeze(-1)

    linear_noise = noise.index_select(0, operator.linear_positions)
    linear_residual = residual.index_select(1, operator.linear_positions)
    cross = weighted @ operator.linear_endpoint_weights.T
    kind = "ion_linear_diagonal" if operator.linear_is_diagonal else "ion_linear_dense"
    cache, cache_key = _covariance_cache_key(observation, kind, t, residual)

    def linear_covariance():
        if operator.linear_is_diagonal:
            outside_variance = operator.linear_outside_endpoint_square @ state_variance + linear_noise + jitter
            diagonal = operator.linear_endpoint_weights.square() @ endpoint_variance + outside_variance
            return torch.stack((diagonal, outside_variance))
        covariance = (operator.linear * state_variance) @ operator.linear.T
        covariance = covariance + torch.diag(linear_noise + jitter)
        return torch.linalg.cholesky(covariance)

    linear_factor = (
        linear_covariance() if cache is None else cache.get_or_compute(cache_key, linear_covariance)
    )
    if operator.linear_is_diagonal:
        diagonal, outside_variance = linear_factor
        solved_linear = linear_residual / diagonal
        solved_cross = cross / diagonal
    else:
        # Both right-hand sides use the same factor for every batch member.
        solved = torch.cholesky_solve(torch.cat((linear_residual.T, cross.T), dim=1), linear_factor).T
        solved_linear, solved_cross = solved.split(mean.shape[0])

    # Joseph form: Var(j - v L x) + Var(v epsilon_L) + R_j,
    # where v = Cov(j, L x) C_LL^-1. This is the same Schur
    # complement expressed as nonnegative sums, avoiding cancellation when
    # tight linear measurements nearly determine the endpoint current.
    if operator.linear_is_diagonal and operator.endpoint_covariance_is_diagonal:
        # Each linear row involves at most one of the six endpoint values.
        # Condition each value using q * R_eff / (a² q + R_eff), which stays
        # accurate even when R_eff is far below float32 epsilon times q.
        rows = operator.endpoint_linear_rows
        remaining_variance = endpoint_variance * (
            outside_variance.index_select(0, rows) / diagonal.index_select(0, rows)
        )
        remaining_variance = torch.where(operator.endpoint_observed, remaining_variance, endpoint_variance)
        conditional_variance = (derivatives.square() * remaining_variance).sum(dim=-1)
    elif operator.linear_is_diagonal:
        remaining_endpoint = derivatives - solved_cross @ operator.linear_endpoint_weights
        conditional_variance = (remaining_endpoint.square() * endpoint_variance).sum(dim=-1)
        conditional_variance += solved_cross.square() @ outside_variance
    else:
        remaining_state = -(solved_cross @ operator.linear)
        remaining_state[:, operator.current.indices] += derivatives
        conditional_variance = (remaining_state.square() * state_variance).sum(dim=-1)
        conditional_variance += (solved_cross.square() * (linear_noise + jitter)).sum(dim=-1)
    conditional_variance += noise[operator.current_position] + jitter
    solved_current = (current_residual - (cross * solved_linear).sum(dim=-1)) / conditional_variance
    solved_linear = solved_linear - solved_cross * solved_current.unsqueeze(-1)
    return operator.combine(solved_linear, solved_current)


def guidance_score(x_t, x_0, t, observation, retain_graph=False):
    """Differentiate the locally Gaussian measurement log likelihood."""
    if observation["data"] is None:
        return torch.zeros_like(x_t)

    mean = _corrected_mean(x_0, t, observation["variance_model"])
    measurement = _apply_operator(observation["operator"], mean)
    observed = torch.as_tensor(
        observation["data"], dtype=measurement.dtype, device=measurement.device
    ).reshape(-1)
    if measurement.shape[1] != observed.numel():
        raise ValueError("observation operator output and observed data have different sizes")

    size = observed.numel()
    residual = measurement - observed
    operator = observation["operator"]
    noise = torch.as_tensor(observation["var"], dtype=measurement.dtype, device=measurement.device)
    jitter = observation.get("covariance_jitter", 1e-6)
    if isinstance(operator, IonCurrentObservation) and noise.ndim <= 1:
        solved = _solve_ion_current_covariance(mean, residual, noise, t, observation)
    elif (
        observation.get("diagonal_covariance", False)
        and isinstance(operator, torch.Tensor)
        and noise.ndim <= 1
    ):
        if noise.ndim == 1 and noise.numel() != size:
            raise ValueError("observation variance has the wrong length")
        cache, cache_key = _covariance_cache_key(observation, "diagonal", t, measurement)

        def diagonal_covariance():
            state_variance = _state_variance(mean, t, observation["variance_model"]).reshape(-1)
            projected_variance = operator.square() @ state_variance
            return noise + projected_variance + jitter

        denominator = (
            diagonal_covariance()
            if cache is None
            else cache.get_or_compute(cache_key, diagonal_covariance)
        )
        solved = residual / denominator
    else:
        if isinstance(operator, torch.Tensor):
            cache, cache_key = _covariance_cache_key(observation, "dense", t, measurement)

            def covariance_factor():
                covariance = _noise_covariance(
                    noise, size, dtype=measurement.dtype, device=measurement.device
                )
                projected_covariance = _projected_covariance(
                    mean, operator, t, observation["variance_model"]
                )
                identity = torch.eye(size, dtype=measurement.dtype, device=measurement.device)
                return torch.linalg.cholesky(covariance + projected_covariance + jitter * identity)

            factor = (
                covariance_factor()
                if cache is None
                else cache.get_or_compute(cache_key, covariance_factor)
            )
            solved = torch.cholesky_solve(residual.T, factor).T
        else:
            covariance = _noise_covariance(
                noise, size, dtype=measurement.dtype, device=measurement.device
            )
            projected_covariance = _projected_covariance(
                mean, operator, t, observation["variance_model"]
            )
            identity = torch.eye(size, dtype=measurement.dtype, device=measurement.device)
            covariance = covariance.unsqueeze(0) + projected_covariance + jitter * identity
            factor = torch.linalg.cholesky(covariance)
            solved = torch.cholesky_solve(residual.unsqueeze(-1), factor).squeeze(-1)
    loss = 0.5 * torch.sum(residual * solved)
    return -torch.autograd.grad(loss, x_t, retain_graph=retain_graph)[0]


def legacy_guidance_score(x_t, x_0, t, observation, retain_graph=False):
    """Reproduce the constant-guidance likelihood used for the paper."""
    if observation["data"] is None:
        return torch.zeros_like(x_t)

    measurement = _apply_operator(observation["operator"], x_0)
    observed = torch.as_tensor(
        observation["data"], dtype=measurement.dtype, device=measurement.device
    ).reshape(-1)
    measurement_variance = torch.as_tensor(
        observation["var"], dtype=measurement.dtype, device=measurement.device
    )
    process_variance = 0.25 * t**2 / (1 + t**2)
    loss = torch.sum((measurement - observed) ** 2 / (measurement_variance + process_variance))
    return -torch.autograd.grad(loss, x_t, retain_graph=retain_graph)[0]
