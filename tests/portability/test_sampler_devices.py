from __future__ import annotations

import pytest
import torch

from hall_diffusion.samplers.edmsampler import EDMSampler, RK2Integrator


class RecordingDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.seen: list[tuple[str, str | None]] = []

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition_device = (
            None if condition_vector is None else condition_vector.device.type
        )
        self.seen.append((x.device.type, condition_device))
        return 0.75 * x + self.anchor


@pytest.mark.parametrize("method", ["midpoint", "ralston", "heun"])
def test_rk2_methods_keep_noise_model_and_condition_on_cpu(method: str) -> None:
    model = RecordingDenoiser()
    condition = torch.ones(2, 3)
    sampler = EDMSampler((2, 1, 8), 4, 0.01, 1.0, 3.0)
    output = sampler.sample(
        RK2Integrator(model, method=method),
        showprogress=False,
        device=torch.device("cpu"),
        model_args={"condition_vector": condition},
    )
    assert output.device.type == "cpu"
    assert torch.isfinite(output).all()
    assert model.anchor.device.type == "cpu"
    assert set(model.seen) == {("cpu", "cpu")}


@pytest.mark.mps
@pytest.mark.parametrize("method", ["midpoint", "ralston", "heun"])
def test_rk2_methods_use_real_mps_without_cpu_fallback(
    method: str, real_mps_device: torch.device
) -> None:
    model = RecordingDenoiser().to(real_mps_device)
    condition = torch.ones(2, 3, device=real_mps_device)
    sampler = EDMSampler((2, 1, 8), 4, 0.01, 1.0, 3.0)
    output = sampler.sample(
        RK2Integrator(model, method=method),
        showprogress=False,
        device=real_mps_device,
        model_args={"condition_vector": condition},
    )
    # The upstream API intentionally returns CPU trajectory history. Device
    # recordings prove that generation itself never fell back to CPU.
    assert output.device.type == "cpu"
    assert torch.isfinite(output).all()
    assert model.anchor.device.type == "mps"
    assert set(model.seen) == {("mps", "mps")}
