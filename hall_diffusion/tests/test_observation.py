import numpy as np
import pytest
import torch

from hall_diffusion import sample
from hall_diffusion.utils.normalization import Normalizer


class Dataset:
    grid = torch.tensor([0.0, 1.0])
    norm = None

    def __getitem__(self, index):
        return None, torch.tensor([3.0]), torch.tensor([[4.0, 5.0]])

    def fields(self):
        return {"field": 0}

    def params(self):
        return {}


class LinearNormalizer:
    def normalize(self, value, field):
        return (value - 10.0) / 2.0

    def denormalize(self, value, field):
        return 10.0 + 2.0 * value

    def normalize_stddev(self, stddev, field, reference=None):
        return stddev / 2.0


class NormalizedDataset(Dataset):
    norm = LinearNormalizer()

    def __getitem__(self, index):
        return None, torch.tensor([3.0]), torch.tensor([[1.0, 2.0]])


def test_constant_mode_applies_legacy_noise_scale_to_standard_deviation(monkeypatch):
    monkeypatch.setattr(
        sample.utils,
        "get_observation_locs",
        lambda *args, **kwargs: ([1], [1.0], None),
    )
    operator, data, variance, _ = sample.build_observation(
        Dataset(),
        {"stddev": 0.025, "fields": {"field": {}}},
        num_samples=1,
        sampling_mode="constant",
    )

    torch.testing.assert_close(operator, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(data, torch.tensor([5.0]))
    torch.testing.assert_close(variance, torch.tensor([1.0]))


def test_dps_mode_uses_supplied_standard_deviation(monkeypatch):
    monkeypatch.setattr(
        sample.utils,
        "get_observation_locs",
        lambda *args, **kwargs: ([1], [1.0], None),
    )
    _, _, variance, _ = sample.build_observation(
        Dataset(),
        {"stddev": 0.025, "fields": {"field": {}}},
        num_samples=1,
        sampling_mode="dps",
    )

    torch.testing.assert_close(variance, torch.tensor([0.025**2]))


@pytest.mark.parametrize(
    ("error", "expected_stddev"),
    [
        ({"type": "absolute", "space": "normalized", "stddev": 0.5}, 0.5),
        ({"type": "relative", "space": "normalized", "stddev": 0.1}, 0.2),
        ({"type": "absolute", "space": "unnormalized", "stddev": 3.0}, 1.5),
        ({"type": "relative", "space": "unnormalized", "stddev": 0.1}, 0.7),
    ],
)
def test_error_type_and_space_are_converted_to_normalized_variance(error, expected_stddev):
    _, data, variance, _ = sample.build_observation(
        NormalizedDataset(),
        {
            "fields": {
                "field": {
                    "locations": [1.0],
                    "values": [14.0],
                    "value_space": "unnormalized",
                }
            },
            "error": error,
        },
        num_samples=1,
    )

    torch.testing.assert_close(data, torch.tensor([2.0]))
    torch.testing.assert_close(variance, torch.tensor([expected_stddev**2]))


def test_field_error_overrides_observation_default():
    _, _, variance, _ = sample.build_observation(
        NormalizedDataset(),
        {
            "error": {"type": "absolute", "space": "normalized", "stddev": 9.0},
            "fields": {
                "field": {
                    "locations": [1.0],
                    "error": {"stddev": 0.25},
                }
            },
        },
        num_samples=1,
    )

    torch.testing.assert_close(variance, torch.tensor([0.25**2]))


def test_pointwise_standard_deviations_are_supported():
    _, _, variance, _ = sample.build_observation(
        NormalizedDataset(),
        {
            "error": {
                "type": "absolute",
                "space": "normalized",
                "stddev": [0.25, 0.5],
            },
            "fields": {"field": {"locations": "all"}},
        },
        num_samples=1,
    )

    torch.testing.assert_close(variance, torch.tensor([0.25**2, 0.5**2]))


def test_invalid_error_configuration_is_rejected():
    with pytest.raises(ValueError, match="error.type"):
        sample.build_observation(
            NormalizedDataset(),
            {
                "error": {"type": "percentage", "space": "normalized", "stddev": 1.0},
                "fields": {"field": {"locations": "all"}},
            },
            num_samples=1,
        )


def test_log_normalized_uncertainty_uses_local_reference_value():
    normalizer = Normalizer.__new__(Normalizer)
    normalizer.norm_tensor = {
        "names": {"field": 0},
        "mean": np.array([1.0]),
        "std": np.array([2.0]),
        "log": np.array([True]),
    }
    normalizer.norm_params = {"names": {}, "mean": np.array([]), "std": np.array([]), "log": np.array([])}

    result = normalizer.normalize_stddev(
        torch.tensor([1.0]), "field", reference=torch.tensor([4.0])
    )

    torch.testing.assert_close(result, torch.tensor([0.125]))
