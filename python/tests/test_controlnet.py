"""Tests for the ControlNet adapter (python/models/controlnet.py).

Run with:  pytest python/tests/test_controlnet.py -v
"""

import pytest
import torch

from models.controlnet import ControlNet
from models.edm2 import EDM2Denoiser

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RESOLUTION = 32       # small for speed
IN_CHANNELS = 4
CONDITION_DIM = 3
CONTROL_DIM = 8
BASE_CHANNELS = 16    # small for speed
CHANNEL_MULT = [1, 2, 3]
NUM_BLOCKS = 1
ATTN_RESOLUTIONS = [8]
BATCH = 2


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
def controlnet(arch_kwargs):
    net = ControlNet(control_dim=CONTROL_DIM, **arch_kwargs)
    net.eval()
    return net


@pytest.fixture
def inputs():
    """Return a dict of standard random inputs."""
    torch.manual_seed(0)
    x = torch.randn(BATCH, IN_CHANNELS, RESOLUTION)
    noise_std = torch.rand(BATCH, 1, 1) * 0.5 + 0.1
    condition_vector = torch.randn(BATCH, CONDITION_DIM)
    ctrl_vec = torch.randn(BATCH, CONTROL_DIM)
    # c_noise is the pre-computed noise label expected by ControlNet.forward
    c_noise = (noise_std.flatten().log() / 4)
    return dict(
        x=x,
        noise_std=noise_std,
        condition_vector=condition_vector,
        ctrl_vec=ctrl_vec,
        c_noise=c_noise,
    )


# ---------------------------------------------------------------------------
# Test 1: Zero-init invariant
# ---------------------------------------------------------------------------

def test_controls_are_zero_at_init(controlnet, inputs):
    """All control tensors must be exactly zero before any training step."""
    with torch.no_grad():
        controls = controlnet(
            inputs["ctrl_vec"],
            inputs["c_noise"],
            inputs["condition_vector"],
        )
    for key, tensor in controls.items():
        assert tensor.abs().max().item() == 0.0, (
            f"Control signal for '{key}' is non-zero at init: "
            f"max abs = {tensor.abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Test 2: Shape consistency
# ---------------------------------------------------------------------------

def test_control_shapes_match_encoder_skips(controlnet, denoiser, inputs):
    """controls[key].shape must equal the corresponding encoder skip shape."""
    with torch.no_grad():
        controls = controlnet(
            inputs["ctrl_vec"],
            inputs["c_noise"],
            inputs["condition_vector"],
        )

    # Collect encoder skip shapes from the UNet by running its encoder.
    x = inputs["x"].clone()
    x_in = x  # no preconditioning scaling needed for shape check
    x_in = torch.cat([x_in, torch.ones_like(x_in[:, :1])], dim=1)
    enc_shapes = {}
    for name, block in denoiser.unet.enc.items():
        x_in = block(x_in) if "conv" in name else block(x_in, torch.zeros(BATCH, max(BASE_CHANNELS * m for m in CHANNEL_MULT)))
        enc_shapes[name] = x_in.shape

    assert set(controls.keys()) == set(enc_shapes.keys()), (
        "ControlNet and UNet encoder key sets differ"
    )
    for key in controls:
        assert controls[key].shape == enc_shapes[key], (
            f"Shape mismatch at '{key}': "
            f"control={controls[key].shape}, skip={enc_shapes[key]}"
        )


def test_control_keys_match_unet_encoder(controlnet, denoiser):
    """ControlNet encoder keys must be identical to UNet encoder keys."""
    assert list(controlnet.enc.keys()) == list(denoiser.unet.enc.keys())


# ---------------------------------------------------------------------------
# Test 3: Identity at init
# ---------------------------------------------------------------------------

def test_denoiser_output_identical_with_zero_controls(denoiser, controlnet, inputs):
    """denoiser(x, controls=zero_controls) == denoiser(x, controls=None) at init."""
    with torch.no_grad():
        controls = controlnet(
            inputs["ctrl_vec"],
            inputs["c_noise"],
            inputs["condition_vector"],
        )
        out_with = denoiser(
            inputs["x"],
            inputs["noise_std"],
            condition_vector=inputs["condition_vector"],
            controls=controls,
        )
        out_without = denoiser(
            inputs["x"],
            inputs["noise_std"],
            condition_vector=inputs["condition_vector"],
            controls=None,
        )

    assert torch.allclose(out_with, out_without, atol=1e-6), (
        f"Outputs differ at init: max abs diff = "
        f"{(out_with - out_without).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4: Gradient flow
# ---------------------------------------------------------------------------

def test_ctrl_gain_receives_gradients_when_unet_frozen(denoiser, controlnet, inputs):
    """ctrl_gain scalars must accumulate gradients; UNet params must not.

    Note: UNet.out_gain and UNet.out_conv.out_gain are both initialised to 0,
    which would zero out all gradients flowing back through the UNet at init.
    We set out_gain = 1 here so the gradient path is open, isolating the
    ControlNet gradient behaviour from that unrelated initialisation detail.
    """
    denoiser.unet.out_gain.data.fill_(1.0)
    denoiser.unet.requires_grad_(False)
    controlnet.train()
    denoiser.train()

    controls = controlnet(
        inputs["ctrl_vec"],
        inputs["c_noise"],
        inputs["condition_vector"],
    )
    out = denoiser(
        inputs["x"],
        inputs["noise_std"],
        condition_vector=inputs["condition_vector"],
        controls=controls,
    )
    loss = out.sum()
    loss.backward()

    # Every ctrl_gain scalar should have a gradient.
    for name, param in controlnet.ctrl_gain.items():
        assert param.grad is not None, f"ctrl_gain['{name}'] has no gradient"
        assert param.grad.abs().item() > 0.0, (
            f"ctrl_gain['{name}'] gradient is zero"
        )

    # No UNet parameter should have a gradient (out_gain was set but is frozen).
    for name, param in denoiser.unet.named_parameters():
        assert param.grad is None, (
            f"UNet parameter '{name}' received a gradient while frozen"
        )


# ---------------------------------------------------------------------------
# Test 5: Controls become non-zero after a gradient step
# ---------------------------------------------------------------------------

def test_controls_nonzero_after_training_step(denoiser, controlnet, inputs):
    """After one optimizer step the control signals should no longer be zero.

    Same note as test 4: out_gain is set to 1 so gradients flow through the
    UNet back to the ControlNet's ctrl_gain parameters.
    """
    denoiser.unet.out_gain.data.fill_(1.0)
    denoiser.unet.requires_grad_(False)
    optimizer = torch.optim.SGD(controlnet.parameters(), lr=1.0)

    controlnet.train()
    denoiser.train()
    optimizer.zero_grad()

    controls = controlnet(
        inputs["ctrl_vec"],
        inputs["c_noise"],
        inputs["condition_vector"],
    )
    out = denoiser(
        inputs["x"],
        inputs["noise_std"],
        condition_vector=inputs["condition_vector"],
        controls=controls,
    )
    out.sum().backward()
    optimizer.step()

    # Now recompute controls — they should be nonzero.
    controlnet.eval()
    with torch.no_grad():
        controls_after = controlnet(
            inputs["ctrl_vec"],
            inputs["c_noise"],
            inputs["condition_vector"],
        )

    any_nonzero = any(
        t.abs().max().item() > 0.0 for t in controls_after.values()
    )
    assert any_nonzero, "All control signals are still zero after a training step"


# ---------------------------------------------------------------------------
# Test 6: from_unet constructor
# ---------------------------------------------------------------------------

def test_from_unet_copies_encoder_weights(denoiser, inputs):
    """ControlNet.from_unet must produce identical encoder outputs to the UNet."""
    net = ControlNet.from_unet(denoiser.unet, control_dim=CONTROL_DIM)

    # Architecture must match.
    assert list(net.enc.keys()) == list(denoiser.unet.enc.keys())

    # Encoder weights must be equal (deep copies, not aliases).
    for key in net.enc:
        cn_sd = net.enc[key].state_dict()
        unet_sd = denoiser.unet.enc[key].state_dict()
        for param_name in unet_sd:
            assert torch.equal(cn_sd[param_name], unet_sd[param_name]), (
                f"Weight mismatch in enc['{key}'].{param_name} after from_unet"
            )

    # Encoder weights must be copies, not the same tensors.
    for key in net.enc:
        cn_sd = net.enc[key].state_dict()
        unet_sd = denoiser.unet.enc[key].state_dict()
        for param_name in unet_sd:
            assert cn_sd[param_name].data_ptr() != unet_sd[param_name].data_ptr(), (
                f"enc['{key}'].{param_name} is an alias, not a deep copy"
            )

    # Embedding weights must also be copied correctly.
    assert torch.equal(net.emb_noise.weight, denoiser.unet.emb_noise.weight)

    # ctrl_gain scalars must still be zero (output projections are fresh).
    for name, param in net.ctrl_gain.items():
        assert param.item() == 0.0, f"ctrl_gain['{name}'] is non-zero after from_unet"

    # Zero-init invariant still holds after copying encoder weights.
    net.eval()
    with torch.no_grad():
        controls = net(inputs["ctrl_vec"], inputs["c_noise"], inputs["condition_vector"])
    for key, tensor in controls.items():
        assert tensor.abs().max().item() == 0.0, (
            f"Control '{key}' is non-zero at init even after from_unet"
        )
