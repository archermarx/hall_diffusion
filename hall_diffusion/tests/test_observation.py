import numpy as np
import pytest
import torch

from hall_diffusion import sample
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
    normalizer.norm_fourier = {"names": {}, "mean": np.array([]), "std": np.array([]), "log": np.array([])}

    result = normalizer.normalize_stddev(torch.tensor([1.0]), "field", reference=torch.tensor([4.0]))
    torch.testing.assert_close(result, torch.tensor([0.125]))
