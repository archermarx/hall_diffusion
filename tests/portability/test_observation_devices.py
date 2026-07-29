from __future__ import annotations

import numpy as np
import pytest
import torch

from hall_diffusion.sample import build_observation


class IdentityNormalizer:
    def normalize(self, values, field):
        return values

    def denormalize(self, values, field):
        return values


class TinyDataset:
    grid = np.linspace(0.0, 1.0, 4)
    norm = IdentityNormalizer()

    def __getitem__(self, index):
        params = torch.tensor([1.0, 2.0])
        data = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        return None, params, data

    def fields(self):
        return {"density": 0, "velocity": 1}

    def params(self):
        return {"first": 0, "second": 1}


def _assert_observation_device(device: torch.device) -> None:
    observations = {
        "stddev": 0.5,
        "fields": {"density": {"x": "all"}},
    }
    operator, data, variance, params = build_observation(
        TinyDataset(), observations, device=device
    )
    assert {operator.device.type, data.device.type, variance.device.type} == {
        device.type
    }
    assert params.device.type == device.type
    assert torch.isfinite(data).all()


def test_observation_tensors_are_created_on_cpu() -> None:
    _assert_observation_device(torch.device("cpu"))


@pytest.mark.mps
def test_observation_tensors_are_created_on_real_mps(
    real_mps_device: torch.device,
) -> None:
    _assert_observation_device(real_mps_device)
