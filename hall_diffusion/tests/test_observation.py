import torch

from hall_diffusion import sample


class Dataset:
    grid = torch.tensor([0.0, 1.0])
    norm = None

    def __getitem__(self, index):
        return None, torch.tensor([3.0]), torch.tensor([[4.0, 5.0]])

    def fields(self):
        return {"field": 0}

    def params(self):
        return {}


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
