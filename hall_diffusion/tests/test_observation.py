import numpy as np
import pytest
import torch

from hall_diffusion import sample
from hall_diffusion.guidance import (
    DPSCovarianceCache,
    _solve_ion_current_covariance,
    guidance_score,
    legacy_guidance_score,
)
from hall_diffusion.observation_operators import IonCurrentDensity, IonCurrentObservation
from hall_diffusion.utils.normalization import Normalizer


class LinearNormalizer:
    means = {"field": 10.0, "voltage": 10.0, "thrust": 100.0}
    scales = {"field": 2.0, "voltage": 2.0, "thrust": 10.0}

    def normalize(self, value, name):
        return (value - self.means[name]) / self.scales[name]

    def denormalize(self, value, name):
        return self.means[name] + self.scales[name] * value

    def normalize_stddev(self, stddev, name, reference=None):
        return stddev / self.scales[name]


class Dataset:
    def __init__(self, scalars_in_tensor=False, resolution=2):
        self.scalars_in_tensor = scalars_in_tensor
        self.grid = torch.linspace(0.0, 1.0, resolution)
        self.norm = LinearNormalizer()
        self._spatial = {"field": 0}
        self._params = {"voltage": 0}
        self._performance = {"thrust": 0}
        self._channels = {"field": 0}
        field = torch.linspace(1.0, 2.0, resolution)
        if scalars_in_tensor:
            self._channels.update({"voltage": 1, "thrust": 2})
            self.tensor = torch.stack((field, torch.full_like(field, 2.0), torch.full_like(field, 0.5)))
            self.params = torch.tensor([])
        else:
            self.tensor = field.unsqueeze(0)
            self.params = torch.tensor([2.0])

    def __getitem__(self, index):
        return None, self.params, self.tensor

    def spatial_fields(self):
        return self._spatial

    def input_params(self):
        return self._params

    def performance_scalars(self):
        return self._performance

    def tensor_channels(self):
        return self._channels


def observation(name, measurement, error=None):
    result = {"measurements": {name: measurement}}
    if error is not None:
        result["error"] = error
    return result


@pytest.mark.parametrize(("sampling_mode", "expected_stddev"), [("dps", 0.025), ("constant", 1.0)])
def test_sampling_mode_applies_legacy_constant_noise_scale(sampling_mode, expected_stddev):
    _, _, variance, _ = sample.build_observation(
        Dataset(),
        observation(
            "field",
            {"locations": [1.0]},
            {"type": "absolute", "space": "normalized", "stddev": 0.025},
        ),
        num_samples=1,
        sampling_mode=sampling_mode,
    )
    torch.testing.assert_close(variance, torch.tensor([expected_stddev**2]))


@pytest.mark.parametrize(
    ("error", "expected_stddev"),
    [
        ({"type": "absolute", "space": "normalized", "stddev": 0.5}, 0.5),
        ({"type": "relative", "space": "normalized", "stddev": 0.1}, 0.2),
        ({"type": "absolute", "space": "unnormalized", "stddev": 3.0}, 1.5),
        ({"type": "relative", "space": "unnormalized", "stddev": 0.1}, 0.7),
    ],
)
def test_spatial_error_type_and_space_are_converted_to_normalized_variance(error, expected_stddev):
    operator, data, variance, _ = sample.build_observation(
        Dataset(),
        observation(
            "field",
            {"locations": [1.0], "values": [14.0], "value_space": "unnormalized"},
            error,
        ),
        num_samples=1,
    )
    torch.testing.assert_close(operator, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(data, torch.tensor([2.0]))
    torch.testing.assert_close(variance, torch.tensor([expected_stddev**2]))


def test_field_error_partially_overrides_observation_default():
    _, _, variance, _ = sample.build_observation(
        Dataset(),
        observation(
            "field",
            {"locations": [1.0], "error": {"stddev": 0.25}},
            {"type": "absolute", "space": "normalized", "stddev": 9.0},
        ),
        num_samples=1,
    )
    torch.testing.assert_close(variance, torch.tensor([0.25**2]))


def test_spatial_pointwise_standard_deviations_are_supported():
    _, _, variance, _ = sample.build_observation(
        Dataset(),
        observation(
            "field",
            {"locations": "all"},
            {"type": "absolute", "space": "normalized", "stddev": [0.25, 0.5]},
        ),
        num_samples=1,
    )
    torch.testing.assert_close(variance, torch.tensor([0.25**2, 0.5**2]))


def test_tensor_measurement_requires_explicit_error():
    with pytest.raises(ValueError, match="requires an error specification"):
        sample.build_observation(Dataset(), observation("field", {"locations": "all"}), num_samples=1)


def test_zero_standard_deviation_explicitly_requests_exact_tensor_measurement():
    _, _, variance, _ = sample.build_observation(
        Dataset(),
        observation(
            "field",
            {"locations": [1.0]},
            {"type": "absolute", "space": "normalized", "stddev": 0.0},
        ),
        num_samples=1,
    )
    torch.testing.assert_close(variance, torch.zeros(1))


@pytest.mark.parametrize("resolution", [2, 4])
def test_tensorized_parameter_uses_one_resolution_independent_mean_operator(resolution):
    dataset = Dataset(scalars_in_tensor=True, resolution=resolution)
    operator, data, variance, params = sample.build_observation(
        dataset,
        observation(
            "voltage",
            {"value": 14.0, "value_space": "unnormalized"},
            {"type": "absolute", "space": "unnormalized", "stddev": 3.0},
        ),
        num_samples=3,
    )
    expected = torch.zeros(1, 3 * resolution)
    expected[0, resolution : 2 * resolution] = 1.0 / resolution
    torch.testing.assert_close(operator, expected)
    torch.testing.assert_close(data, torch.tensor([2.0]))
    torch.testing.assert_close(variance, torch.tensor([1.5**2]))
    assert params.shape == (3, 0)


@pytest.mark.parametrize(
    ("measurement", "message"),
    [
        ({"locations": [0.0]}, "unsupported key"),
        ({"values": [14.0]}, "unsupported key"),
        ({"value": [14.0], "value_space": "unnormalized"}, "one value"),
        ({"error": {"type": "absolute", "space": "normalized", "stddev": [0.1]}}, "one error.stddev"),
    ],
)
def test_tensorized_parameter_rejects_per_cell_configuration(measurement, message):
    with pytest.raises(ValueError, match=message):
        sample.build_observation(
            Dataset(scalars_in_tensor=True),
            observation(
                "voltage",
                measurement,
                {"type": "absolute", "space": "normalized", "stddev": 0.1},
            ),
            num_samples=1,
        )


@pytest.mark.parametrize("sampling_mode", ["dps", "constant"])
def test_non_tensorized_parameter_is_an_exact_condition_and_warns_about_error(sampling_mode):
    with pytest.warns(UserWarning, match="Ignoring uncertainty.*voltage"):
        operator, data, variance, params = sample.build_observation(
            Dataset(),
            observation(
                "voltage",
                {
                    "value": 14.0,
                    "value_space": "unnormalized",
                    "error": {"type": "relative", "space": "unnormalized", "stddev": 0.1},
                },
            ),
            num_samples=2,
            sampling_mode=sampling_mode,
        )
    assert operator is data is variance is None
    torch.testing.assert_close(params, torch.full((2, 1), 2.0))


def test_explicit_parameter_measurement_overrides_condition_vector():
    _, _, _, params = sample.build_observation(
        Dataset(),
        observation("voltage", {"value": 14.0, "value_space": "unnormalized"}),
        num_samples=2,
        param_vec=torch.tensor([[8.0], [9.0]]),
    )
    torch.testing.assert_close(params, torch.full((2, 1), 2.0))


def test_tensorized_performance_scalar_uses_mean_operator():
    operator, data, _, _ = sample.build_observation(
        Dataset(scalars_in_tensor=True),
        observation(
            "thrust",
            {},
            {"type": "absolute", "space": "normalized", "stddev": 0.1},
        ),
        num_samples=1,
    )
    torch.testing.assert_close(operator, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.5, 0.5]]))
    torch.testing.assert_close(data, torch.tensor([0.5]))


def test_non_tensorized_performance_scalar_is_rejected():
    with pytest.raises(ValueError, match="cannot be measured"):
        sample.build_observation(
            Dataset(),
            observation(
                "thrust",
                {},
                {"type": "absolute", "space": "normalized", "stddev": 0.1},
            ),
            num_samples=1,
        )


@pytest.mark.parametrize("legacy_key", ["fields", "params"])
def test_legacy_observation_namespaces_are_rejected(legacy_key):
    with pytest.raises(ValueError, match="Legacy observation key"):
        sample.build_observation(Dataset(), {legacy_key: {}}, num_samples=1)


def test_legacy_measurement_keys_are_rejected_with_migration_message():
    with pytest.raises(ValueError, match="retired key.*x"):
        sample.build_observation(
            Dataset(),
            observation(
                "field",
                {"x": [1.0]},
                {"type": "absolute", "space": "normalized", "stddev": 0.1},
            ),
            num_samples=1,
        )


def test_provided_values_require_value_space():
    with pytest.raises(ValueError, match="value_space"):
        sample.build_observation(
            Dataset(),
            observation(
                "field",
                {"locations": [1.0], "values": [14.0]},
                {"type": "absolute", "space": "normalized", "stddev": 0.1},
            ),
            num_samples=1,
        )


def test_log_normalized_uncertainty_uses_local_reference_value():
    normalizer = Normalizer.__new__(Normalizer)
    normalizer.norm_spatial = {
        "names": {"field": 0},
        "mean": np.array([1.0]),
        "std": np.array([2.0]),
        "log": np.array([True]),
    }
    normalizer.norm_params = {"names": {}, "mean": np.array([]), "std": np.array([]), "log": np.array([])}
    normalizer.norm_perf = {"names": {}, "mean": np.array([]), "std": np.array([]), "log": np.array([])}
    result = normalizer.normalize_stddev(torch.tensor([1.0]), "field", reference=torch.tensor([4.0]))
    torch.testing.assert_close(result, torch.tensor([0.125]))


class CurrentDataset(Dataset):
    def __init__(self, scalars_in_tensor=False):
        super().__init__(scalars_in_tensor=scalars_in_tensor, resolution=3)
        normalizer = Normalizer.__new__(Normalizer)
        normalizer.norm_spatial = {
            "names": {"field": 0},
            "mean": np.array([10.0]),
            "std": np.array([2.0]),
            "log": np.array([False]),
        }
        for category, name, mean, std in (("params", "voltage", 10.0, 2.0), ("perf", "thrust", 100.0, 10.0)):
            setattr(normalizer, f"norm_{category}", {
                "names": {name: 0}, "mean": np.array([mean]),
                "std": np.array([std]), "log": np.array([False]),
            })
        self.norm = normalizer
        # Deliberately put velocities before densities to test metadata lookup.
        for quantity in ("ui", "ni"):
            for charge in range(1, 4):
                name = f"{quantity}_{charge}"
                info = normalizer.norm_spatial
                index = len(info["names"])
                info["names"][name] = index
                self._spatial[name] = index
                self._channels[name] = self.tensor.shape[0]
                is_density = quantity == "ni"
                info["mean"] = np.append(info["mean"], np.log(1e17) if is_density else 2000.0)
                info["std"] = np.append(info["std"], 0.5 if is_density else 5000.0)
                info["log"] = np.append(info["log"], is_density)
                physical = torch.tensor([5e17, 3e17, charge * 1e17] if is_density else [1e5, 5e4, charge * 1e4])
                self.tensor = torch.cat((self.tensor, normalizer.normalize(physical, name).unsqueeze(0)))


@pytest.mark.parametrize("scalars_in_tensor", [False, True])
def test_ion_current_density_uses_all_charge_states_at_right_boundary(scalars_in_tensor):
    dataset = CurrentDataset(scalars_in_tensor)
    operator, data, variance, _ = sample.build_observation(
        dataset,
        observation("ion_current_density", {}, {"type": "relative", "space": "unnormalized", "stddev": 0.05}),
        num_samples=1,
    )
    expected = torch.tensor([1.602176634e-19 * 1e21 * (1 + 8 + 27)])
    torch.testing.assert_close(data, expected)
    torch.testing.assert_close(operator(dataset.tensor), expected)
    torch.testing.assert_close(variance, (0.05 * data).square())
    state = dataset.tensor.clone().requires_grad_()
    gradient = torch.autograd.grad(operator(state).sum(), state)[0]
    torch.testing.assert_close(gradient[:, :-1], torch.zeros_like(gradient[:, :-1]))
    for charge in range(1, 4):
        density_gradient = gradient[dataset.tensor_channels()[f"ni_{charge}"], -1]
        velocity_gradient = gradient[dataset.tensor_channels()[f"ui_{charge}"], -1]
        torch.testing.assert_close(
            density_gradient, torch.tensor(0.5 * 1.602176634e-19 * charge**3 * 1e21), rtol=5e-6, atol=0.0,
        )
        torch.testing.assert_close(
            velocity_gradient, torch.tensor(1.602176634e-19 * charge**2 * 1e17 * 5000.0), rtol=5e-6, atol=0.0,
        )


@pytest.mark.parametrize(("sampling_mode", "scale"), [("dps", 1.0), ("constant", 40.0)])
def test_explicit_ion_current_density_combines_with_linear_measurements_and_guides(sampling_mode, scale):
    dataset = CurrentDataset(scalars_in_tensor=True)
    operator, data, variance, _ = sample.build_observation(
        dataset,
        {"measurements": {
            "field": {"locations": [0.0], "error": {"type": "absolute", "space": "normalized", "stddev": 0.1}},
            "ion_current_density": {
                "value": 1000.0, "value_space": "unnormalized",
                "error": {"type": "absolute", "space": "unnormalized", "stddev": 50.0},
            },
            "thrust": {"error": {"type": "absolute", "space": "normalized", "stddev": 0.2}},
        }},
        num_samples=2,
        sampling_mode=sampling_mode,
    )
    torch.testing.assert_close(data, torch.tensor([1.0, 1000.0, 0.5]))
    torch.testing.assert_close(variance, (scale * torch.tensor([0.1, 50.0, 0.2])).square())
    expected_current = 1.602176634e-19 * 1e21 * 36
    torch.testing.assert_close(operator(dataset.tensor), torch.tensor([1.0, expected_current, 0.5]))
    state = dataset.tensor.unsqueeze(0).expand(2, -1, -1).clone().requires_grad_()
    obs = {
        "operator": operator, "data": data, "var": variance,
        "variance_model": {
            "noise_levels": torch.tensor([0.1, 1.0]),
            "process_variance": torch.full((2, *dataset.tensor.shape), 0.01),
        },
    }
    score_fn = legacy_guidance_score if sampling_mode == "constant" else guidance_score
    score = score_fn(state, state, torch.tensor(0.5), obs)
    assert torch.all(torch.isfinite(score))
    assert torch.linalg.vector_norm(score) > 0
    torch.testing.assert_close(score[:, :, :-1], torch.zeros_like(score[:, :, :-1]))
    for charge in range(1, 4):
        for quantity in ("ni", "ui"):
            assert torch.all(score[:, dataset.tensor_channels()[f"{quantity}_{charge}"], -1] < 0)


@pytest.mark.parametrize(("measurement", "error", "message"), [
    ({"locations": [1.0]}, {"type": "absolute", "space": "unnormalized", "stddev": 1.0}, "unsupported key"),
    ({"values": [1000.0]}, {"type": "absolute", "space": "unnormalized", "stddev": 1.0}, "unsupported key"),
    ({"value": [1000.0], "value_space": "unnormalized"}, None, "one value"),
    ({"value": 1000.0}, {"type": "absolute", "space": "unnormalized", "stddev": 1.0}, "value_space"),
    ({"value": 1000.0, "value_space": "normalized"}, {"type": "absolute", "space": "unnormalized", "stddev": 1.0}, "value_space"),
    ({}, None, "requires an error specification"),
    ({}, {"type": "absolute", "space": "normalized", "stddev": 1.0}, "error.space"),
    ({}, {"type": "absolute", "space": "unnormalized", "stddev": [1.0]}, "one error.stddev"),
    ({}, {"type": "absolute", "space": "unnormalized", "stddev": -1.0}, "finite and nonnegative"),
])
def test_ion_current_density_rejects_invalid_scalar_configuration(measurement, error, message):
    with pytest.raises(ValueError, match=message):
        sample.build_observation(CurrentDataset(), observation("ion_current_density", measurement, error), num_samples=1)


def test_ion_current_density_requires_all_six_spatial_fields():
    dataset = CurrentDataset()
    del dataset._spatial["ni_3"]
    with pytest.raises(ValueError, match="requires spatial fields.*ni_3"):
        sample.build_observation(
            dataset,
            observation("ion_current_density", {}, {"type": "absolute", "space": "unnormalized", "stddev": 0.0}),
            num_samples=1,
        )


def test_parse_ion_current_density_observation_supports_callable_operator(monkeypatch):
    dataset = CurrentDataset()
    monkeypatch.setattr(sample, "ThrusterDataset", lambda *args, **kwargs: dataset)
    obs, _, _ = sample.parse_observation(
        (1, dataset.tensor.shape[0], 3),
        {"observation": {
            "base_sim": "reference",
            **observation("ion_current_density", {}, {"type": "absolute", "space": "unnormalized", "stddev": 0.0}),
        }},
        scalars_in_tensor=False,
    )
    assert callable(obs["operator"])
    assert not obs["diagonal_covariance"]
    torch.testing.assert_close(obs["var"], torch.zeros(1))


def current_operator_pair(layout, dtype=torch.float64, current_position=0):
    dataset = CurrentDataset(scalars_in_tensor=True)
    tensor = dataset.tensor.to(dtype)
    current = IonCurrentDensity(dataset.norm, dataset.tensor_channels(), tensor)
    rows = []
    if layout != "current_only":
        row = torch.zeros(tensor.numel(), dtype=dtype)
        row[0] = 1.0
        rows.append(row)
    if layout.startswith("overlap"):
        row = torch.zeros(tensor.numel(), dtype=dtype)
        row[current.indices[3]] = 1.0
        rows.append(row)
    if layout == "overlap_dense":
        row = torch.zeros(tensor.numel(), dtype=dtype)
        row[current.indices[3]] = 0.5
        row[current.indices[0]] = 0.25
        rows.append(row)
    if layout == "mixed_endpoint_row":
        row = torch.zeros(tensor.numel(), dtype=dtype)
        row[current.indices[3]] = 0.5
        row[current.indices[0]] = 0.25
        rows.append(row)
    rows.insert(current_position, current)
    operator = IonCurrentObservation(rows)

    def reference(state):
        # Reproduce the original unbatched operator independently of the
        # analytic derivatives and structured covariance implementation.
        value = sum(
            charge * 1.602176634e-19
            * dataset.norm.denormalize(state[dataset.tensor_channels()[f"ni_{charge}"], -1], f"ni_{charge}")
            * dataset.norm.denormalize(state[dataset.tensor_channels()[f"ui_{charge}"], -1], f"ui_{charge}")
            for charge in range(1, 4)
        )
        return torch.stack([value if isinstance(row, IonCurrentDensity) else row @ state.flatten() for row in rows])

    return tensor, operator, reference


@pytest.mark.parametrize("layout", ["current_only", "disjoint", "overlap_diagonal", "overlap_dense", "mixed_endpoint_row"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("noise_kind", ["scalar", "vector", "full"])
def test_fast_current_guidance_matches_original_dense_autograd(layout, dtype, noise_kind):
    tensor, operator, reference = current_operator_pair(layout, dtype, current_position=0)
    size = operator.linear.shape[0] + 1
    noise = torch.full((size,), 0.1, dtype=dtype)
    noise[operator.current_position] = 2500.0
    if noise_kind == "scalar":
        noise = torch.tensor(2500.0, dtype=dtype)
    elif noise_kind == "full":
        noise = torch.diag(noise) + 0.01 * torch.ones(size, size, dtype=dtype)
    variance = torch.linspace(0.01, 0.2, tensor.numel(), dtype=dtype).reshape_as(tensor)
    obs = {
        "operator": operator, "data": reference(tensor) * 0.85, "var": noise,
        "variance_model": {
            "noise_levels": torch.tensor([0.1, 1.0], dtype=dtype),
            "process_variance": torch.stack((variance, 1.5 * variance)),
            "bias_correction": True,
            "mean_residual": torch.full((2, *tensor.shape), 0.03, dtype=dtype),
            "channel_average": True,
        },
        "covariance_cache": DPSCovarianceCache(),
    }
    # Reuse the cache with different states and a smaller final batch. The
    # nonlinear and cross-covariance terms must still be recomputed each time.
    for batch_size, offset in ((3, 0.01), (1, -0.02)):
        def score(measurement):
            state = tensor.unsqueeze(0).repeat(batch_size, 1, 1).add(offset).requires_grad_()
            denoised = 0.8 * state + 0.03 * state.square()
            return guidance_score(state, denoised, 0.5, {**obs, "operator": measurement})

        torch.testing.assert_close(
            score(operator), score(reference),
            rtol=1e-4 if dtype == torch.float32 else 1e-10,
            atol=1e-6 if dtype == torch.float32 else 1e-10,
        )


def test_fast_current_derivatives_match_autograd_with_mixed_log_metadata():
    dataset = CurrentDataset()
    # Exercise linear density and log velocity, as well as the usual reverse.
    for name, mean, std, log in (("ni_2", 1e17, 1e17, False), ("ui_2", 9.0, 0.5, True)):
        index = dataset.norm.norm_spatial["names"][name]
        dataset.norm.norm_spatial["mean"][index] = mean
        dataset.norm.norm_spatial["std"][index] = std
        dataset.norm.norm_spatial["log"][index] = log
    tensor = dataset.tensor.double()
    current = IonCurrentDensity(dataset.norm, dataset.tensor_channels(), tensor)
    operator = IonCurrentObservation([current])
    state = tensor.unsqueeze(0).repeat(2, 1, 1)
    state[1] += 0.1
    expected = torch.stack([torch.autograd.functional.jacobian(operator, value).reshape(1, -1) for value in state])
    torch.testing.assert_close(operator.jacobians(state), expected)


@pytest.mark.parametrize("layout", ["current_only", "disjoint", "overlap_diagonal", "overlap_dense"])
def test_fast_current_guidance_avoids_autograd_jacobians_and_caches_linear_factor(layout, monkeypatch):
    tensor, operator, reference = current_operator_pair(layout, current_position=1 if layout != "current_only" else 0)
    obs = {
        "operator": operator, "data": reference(tensor) * 0.9, "var": 0.1,
        "variance_model": {
            "noise_levels": torch.tensor([0.1, 1.0], dtype=tensor.dtype),
            "process_variance": torch.full((2, *tensor.shape), 0.02, dtype=tensor.dtype),
        },
        "covariance_cache": DPSCovarianceCache(),
    }
    original_cholesky = torch.linalg.cholesky
    factorization_shapes = []

    def record_cholesky(value):
        factorization_shapes.append(value.shape)
        return original_cholesky(value)

    def unexpected_jacobian(*args, **kwargs):
        raise AssertionError("endpoint current guidance should use analytic derivatives")

    monkeypatch.setattr(torch.autograd.functional, "jacobian", unexpected_jacobian)
    monkeypatch.setattr(torch.linalg, "cholesky", record_cholesky)
    for batch_size in (3, 1):
        state = tensor.unsqueeze(0).repeat(batch_size, 1, 1).requires_grad_()
        result = guidance_score(state, 0.8 * state, 0.5, obs)
        assert torch.all(torch.isfinite(result))
    expected_shapes = [operator.linear.shape[:1] * 2] if layout == "overlap_dense" else []
    assert factorization_shapes == expected_shapes


@pytest.mark.parametrize("linear_noise", [1e-10, 1e-14])
@pytest.mark.parametrize("jitter", [0.0, 1e-6])
@pytest.mark.parametrize("average", [False, True])
def test_tight_endpoint_current_variance_remains_positive_in_float32(linear_noise, jitter, average):
    dataset = CurrentDataset()
    current = IonCurrentDensity(dataset.norm, dataset.tensor_channels(), dataset.tensor)
    weights = dataset.tensor.new_tensor([1.0, 2.0, -0.5, 1.5, -1.0, 0.25])
    rows = []
    for index, weight in zip(current.indices.tolist(), weights):
        row = torch.zeros(dataset.tensor.numel())
        if average:
            start = index - dataset.tensor.shape[-1] + 1
            row[start:index + 1] = weight / dataset.tensor.shape[-1]
        else:
            row[index] = weight
        rows.append(row)
    rows.insert(3, current)
    operator = IonCurrentObservation(rows)
    noise = torch.full((7,), linear_noise)
    noise[operator.current_position] = 0.0
    obs = {
        "operator": operator, "var": noise, "covariance_jitter": jitter,
        "variance_model": {
            "noise_levels": torch.tensor([0.1, 1.0]),
            "process_variance": torch.full((2, *dataset.tensor.shape), 0.2),
        },
        "covariance_cache": DPSCovarianceCache(),
    }
    for offset in (0.0, 0.1):
        mean = dataset.tensor.unsqueeze(0).repeat(2, 1, 1) + offset
        residual = torch.zeros(2, 7)
        residual[:, operator.current_position] = 1.0
        solved = _solve_ion_current_covariance(mean, residual, noise, 0.5, obs)
        assert torch.all(torch.isfinite(solved))
        actual_variance = 1.0 / solved[:, operator.current_position]
        assert torch.all(actual_variance > 0)

        # Independent scalar Gaussian-conditioning formula for each endpoint.
        # Use double precision for the reference without subtracting similar
        # quantities, including the variance from unobserved grid points.
        q = obs["variance_model"]["process_variance"][0, 0, 0].double()
        effective_noise = noise[0].double() + jitter
        coefficient = operator.linear_endpoint_weights.diagonal().double()
        outside_variance = (operator.linear_outside_endpoint_square.double() * q).sum(dim=1)
        posterior_variance = q * (outside_variance + effective_noise) / (
            q * coefficient.square() + outside_variance + effective_noise
        )
        expected_variance = (current.derivatives(mean).double().square() * posterior_variance).sum(dim=1) + jitter
        torch.testing.assert_close(actual_variance.double(), expected_variance, rtol=2e-6, atol=0.0)
