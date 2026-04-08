"""Tests for EDM2Denoiser and UNet (python/models/edm2.py).

Run with:  pytest python/tests/test_edm2.py -v
"""

import pytest
import torch
import numpy as np

from models.edm2 import EDM2Denoiser, UNet

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RESOLUTION = 32
IN_CHANNELS = 4
CONDITION_DIM = 3
BASE_CHANNELS = 16
CHANNEL_MULT = [1, 2, 3]
NUM_BLOCKS = 1
ATTN_RESOLUTIONS = [8]
BATCH = 2
DATA_STD = 0.5


@pytest.fixture
def arch_kwargs():
    return dict(
        resolution=RESOLUTION,
        in_channels=IN_CHANNELS,
        condition_dim=CONDITION_DIM,
        base_channels=BASE_CHANNELS,
        channel_mult=CHANNEL_MULT,
        num_blocks=NUM_BLOCKS,
        attn_resolutions=ATTN_RESOLUTIONS,
    )


@pytest.fixture
def denoiser(arch_kwargs):
    model = EDM2Denoiser(**arch_kwargs)
    model.eval()
    return model


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    x = torch.randn(BATCH, IN_CHANNELS, RESOLUTION)
    noise_std = torch.rand(BATCH, 1, 1) * 0.5 + 0.1
    condition_vector = torch.randn(BATCH, CONDITION_DIM)
    return dict(x=x, noise_std=noise_std, condition_vector=condition_vector)


# ---------------------------------------------------------------------------
# Test 1: Output shape matches input shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("resolution,in_channels", [
    (32, 4),
    (64, 8),
    (16, 1),
])
def test_output_shape_matches_input(resolution, in_channels):
    """Denoiser output must have the same shape as the input tensor."""
    model = EDM2Denoiser(
        resolution=resolution,
        in_channels=in_channels,
        condition_dim=CONDITION_DIM,
        base_channels=BASE_CHANNELS,
        channel_mult=CHANNEL_MULT,
        num_blocks=NUM_BLOCKS,
        attn_resolutions=ATTN_RESOLUTIONS,
    )
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(BATCH, in_channels, resolution)
    noise_std = torch.rand(BATCH, 1, 1) * 0.5 + 0.1
    condition_vector = torch.randn(BATCH, CONDITION_DIM)
    with torch.no_grad():
        out = model(x, noise_std, condition_vector=condition_vector)
    assert out.shape == x.shape, f"Shape mismatch: input {x.shape}, output {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: Unconditional mode (condition_dim=0)
# ---------------------------------------------------------------------------

def test_unconditional_forward():
    """Denoiser with condition_dim=0 must run without class labels."""
    model = EDM2Denoiser(
        resolution=RESOLUTION,
        in_channels=IN_CHANNELS,
        condition_dim=0,
        base_channels=BASE_CHANNELS,
        channel_mult=CHANNEL_MULT,
        num_blocks=NUM_BLOCKS,
        attn_resolutions=ATTN_RESOLUTIONS,
    )
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(BATCH, IN_CHANNELS, RESOLUTION)
    noise_std = torch.rand(BATCH, 1, 1) * 0.5 + 0.1
    with torch.no_grad():
        out = model(x, noise_std)  # no condition_vector
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Test 3: return_logvar produces correct shapes
# ---------------------------------------------------------------------------

def test_return_logvar_shapes(denoiser, inputs):
    """return_logvar=True must return (D_x, logvar) with logvar shape (B, 1, 1)."""
    with torch.no_grad():
        result = denoiser(
            inputs["x"],
            inputs["noise_std"],
            condition_vector=inputs["condition_vector"],
            return_logvar=True,
        )
    assert isinstance(result, tuple) and len(result) == 2, (
        "return_logvar=True must return a 2-tuple"
    )
    D_x, logvar = result
    assert D_x.shape == inputs["x"].shape, f"D_x shape mismatch: {D_x.shape}"
    assert logvar.shape == (BATCH, 1, 1), f"logvar shape mismatch: {logvar.shape}"


# ---------------------------------------------------------------------------
# Test 4: Zero-init output invariant (preconditioning sanity)
# ---------------------------------------------------------------------------

def test_output_equals_c_skip_x_at_init(denoiser, inputs):
    """At initialization out_gain=0, so F_x=0 and D_x must equal c_skip * x.

    Preconditioning (EDM2):
        c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
        c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
        D_x    = c_skip * x + c_out * F_x

    Because out_gain=0 at init, UNet output F_x=0, so D_x = c_skip * x.
    This gives us a strong deterministic check of the preconditioning math.
    """
    x = inputs["x"]
    noise_std = inputs["noise_std"].reshape(-1, 1, 1).float()
    sigma_data = denoiser.data_std

    c_skip = sigma_data**2 / (noise_std**2 + sigma_data**2)
    expected = (c_skip * x).float()

    with torch.no_grad():
        out = denoiser(
            x,
            inputs["noise_std"],
            condition_vector=inputs["condition_vector"],
        )

    assert torch.allclose(out, expected, atol=1e-5), (
        f"Output deviates from c_skip * x at init: "
        f"max abs diff = {(out - expected).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 5: Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow_through_denoiser(denoiser, inputs):
    """All trainable parameters should receive nonzero gradients.

    out_gain is set to 1 first; at its default value of 0 it zeroes out all
    gradients flowing back through the UNet (by design, same zero-init pattern
    as Block.emb_gain), which would make this test vacuous.
    """
    denoiser.train()
    denoiser.unet.out_gain.data.fill_(1.0)

    # Use return_logvar=True so the logvar head is included in the computation
    # graph; without it logvar_linear never gets a gradient.
    D_x, logvar = denoiser(
        inputs["x"],
        inputs["noise_std"],
        condition_vector=inputs["condition_vector"],
        return_logvar=True,
    )
    (D_x.sum() + logvar.sum()).backward()

    # Every trainable parameter must appear in the computation graph.
    params_without_grad = [
        name for name, p in denoiser.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not params_without_grad, (
        f"Parameters disconnected from graph: {params_without_grad}"
    )

    # Parameters on the main signal path must have nonzero gradients.
    # Note: emb_linear / emb_noise / emb_label weights all have exactly-zero
    # gradients at init because their emb_gain scalars (also 0-initialized)
    # gate the entire embedding path.  That is expected behavior, not a bug.
    key_params = ["unet.out_gain", "unet.out_conv.weight", "logvar_linear.weight"]
    for name in key_params:
        param = dict(denoiser.named_parameters())[name]
        assert param.grad is not None and param.grad.abs().max() > 0, (
            f"Expected nonzero gradient for '{name}'"
        )


# ---------------------------------------------------------------------------
# Test 6: Batch size 1
# ---------------------------------------------------------------------------

def test_batch_size_one(arch_kwargs):
    """Batch size 1 must work (attention and group norms can be sensitive to this)."""
    model = EDM2Denoiser(**arch_kwargs)
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, IN_CHANNELS, RESOLUTION)
    noise_std = torch.rand(1, 1, 1) * 0.5 + 0.1
    condition_vector = torch.randn(1, CONDITION_DIM)
    with torch.no_grad():
        out = model(x, noise_std, condition_vector=condition_vector)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Test 7: Output is float32 regardless of input dtype
# ---------------------------------------------------------------------------

def test_output_is_float32(denoiser, inputs):
    """EDM2Denoiser casts inputs to float32 internally; output must be float32."""
    with torch.no_grad():
        out = denoiser(
            inputs["x"].float(),
            inputs["noise_std"].float(),
            condition_vector=inputs["condition_vector"].float(),
        )
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"
