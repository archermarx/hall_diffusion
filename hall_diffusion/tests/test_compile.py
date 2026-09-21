from unittest.mock import patch

import pytest
import torch

from hall_diffusion.models.conditioning import ConditionAdapter, ConditionedEDM2, TLPPVAEConditionEncoder
from hall_diffusion.models.edm2 import EDM2Denoiser
from hall_diffusion.utils.utils import compile_model


def test_compile_model_returns_original_when_disabled():
    model = torch.nn.Linear(2, 1)

    with (
        patch("hall_diffusion.utils.utils.torch.compile") as compile_fn,
        patch("hall_diffusion.utils.utils.torch.set_float32_matmul_precision") as precision_fn,
    ):
        result = compile_model(model, enabled=False)

    assert result is model
    compile_fn.assert_not_called()
    precision_fn.assert_not_called()


def test_compile_model_compiles_when_enabled():
    model = torch.nn.Linear(2, 1)
    compiled = object()

    with (
        patch("hall_diffusion.utils.utils.torch.compile", return_value=compiled) as compile_fn,
        patch("hall_diffusion.utils.utils.torch.set_float32_matmul_precision") as precision_fn,
    ):
        result = compile_model(model, enabled=True)

    assert result is compiled
    precision_fn.assert_called_once_with("high")
    compile_fn.assert_called_once_with(model, fullgraph=True)


def test_edm2_training_forward_is_capturable_as_one_graph():
    model = EDM2Denoiser(
        resolution=8,
        in_channels=2,
        condition_dim=0,
        base_channels=4,
        channel_mult=[1, 2],
        num_blocks=1,
        attn_resolutions=[],
        channels_per_head=4,
    ).train()
    compiled = torch.compile(model, backend="eager", fullgraph=True)

    output = compiled(torch.randn(2, 2, 8), torch.rand(2, 1, 1))
    output.sum().backward()

    assert output.shape == (2, 2, 8)
    assert not any("resample_filter" in name for name in model.state_dict())


@pytest.mark.parametrize(
    "condition",
    [
        pytest.param(torch.randint(0, 5, (2, 1, 128, 128), dtype=torch.int32), id="raw-counts"),
        pytest.param(torch.randn(2, 128), id="cached-means"),
    ],
)
def test_tlpp_vae_conditioned_training_forward_is_capturable_as_one_graph(condition):
    base = EDM2Denoiser(
        resolution=8,
        in_channels=2,
        condition_dim=0,
        base_channels=4,
        channel_mult=[1, 2],
        num_blocks=1,
        attn_resolutions=[],
        channels_per_head=4,
    ).eval()
    encoder = TLPPVAEConditionEncoder(
        token_dim=8,
        num_tokens=2,
        hidden_dim=8,
        depth=1,
        freeze_vae=True,
        initialize_pretrained=False,
    )
    model = ConditionedEDM2(base, {"tlpp": ConditionAdapter(base, encoder, channels_per_head=4)}).train()
    compiled = torch.compile(model, backend="eager", fullgraph=True)

    output = compiled(
        torch.randn(2, 2, 8),
        torch.rand(2, 1, 1),
        conditions={"tlpp": condition},
    )
    output.sum().backward()

    assert output.shape == (2, 2, 8)
    assert any(parameter.grad is not None for parameter in model.adapters["tlpp"].heads.parameters())
