from __future__ import annotations

import pytest
import torch

from hall_diffusion.models.edm2 import EDM2Denoiser


MODEL_ARGS = dict(
    resolution=16,
    in_channels=2,
    condition_dim=3,
    base_channels=8,
    channel_mult=[1, 2],
    num_blocks=1,
    attn_resolutions=[],
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(314159)
    x = torch.randn(2, 2, 16, generator=generator)
    sigma = torch.rand(2, 1, 1, generator=generator) + 0.1
    condition = torch.randn(2, 3, generator=generator)
    return x, sigma, condition


def test_edm2_forward_on_cpu_is_finite() -> None:
    torch.manual_seed(2718)
    model = EDM2Denoiser(**MODEL_ARGS).eval()
    x, sigma, condition = _inputs()
    with torch.no_grad():
        output = model(x, sigma, condition_vector=condition)
    assert output.device.type == "cpu"
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.mps
def test_edm2_forward_on_real_mps_without_fallback(
    real_mps_device: torch.device,
) -> None:
    torch.manual_seed(2718)
    model = EDM2Denoiser(**MODEL_ARGS).eval().to(real_mps_device)
    x, sigma, condition = (value.to(real_mps_device) for value in _inputs())
    with torch.no_grad():
        output = model(x, sigma, condition_vector=condition)
    assert output.device.type == "mps"
    assert torch.isfinite(output).all()


@pytest.mark.mps
def test_cpu_and_mps_edm2_outputs_agree_within_float32_tolerance(
    real_mps_device: torch.device,
) -> None:
    torch.manual_seed(2718)
    cpu_model = EDM2Denoiser(**MODEL_ARGS).eval()
    mps_model = EDM2Denoiser(**MODEL_ARGS).eval().to(real_mps_device)
    mps_model.load_state_dict(cpu_model.state_dict())
    x, sigma, condition = _inputs()

    with torch.no_grad():
        cpu_output = cpu_model(x, sigma, condition_vector=condition)
        mps_output = mps_model(
            x.to(real_mps_device),
            sigma.to(real_mps_device),
            condition_vector=condition.to(real_mps_device),
        ).cpu()

    torch.testing.assert_close(mps_output, cpu_output, rtol=3e-4, atol=3e-4)
