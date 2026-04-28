"""ControlNet adapter for the EDM2-style 1D UNet.

The ControlNet mirrors the UNet encoder but replaces the stem conv's input
with a (B, control_channels, resolution) control signal.  The stem conv
projects control_channels → cblock[0] (base_channels at level 0), expanding
rather than compressing, then the remaining mirrored encoder stages run as
normal.  The output is a dict of zero-initialized control signals keyed by
encoder stage name, injected additively into the UNet's skip-connection list
before the decoder consumes them.

At initialisation all control signals are exactly zero (via a learnable gain
scalar starting at 0, matching the Block.emb_gain pattern), so the augmented
model behaves identically to the unmodified UNet.
"""
import torch
import torch.nn as nn
from typing import TypedDict

from .edm2 import (
    get_precondition_factors,
    UNet, MPConv, EDM2Denoiser, mp_silu,
)

class UNetConfig(TypedDict):
    resolution: int
    in_channels: int
    condition_dim: int
    base_channels: int
    channel_mult: list[int]
    channel_mult_noise: int | None
    channel_mult_emb: int | None
    num_blocks: int
    attn_resolutions: list[int]
    label_balance: float
    concat_balance: float

class MPSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mp_silu(x)

def make_zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class ZeroConv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = make_zero_module(nn.Conv1d(num_channels, num_channels, kernel_size=1, padding=0))
    
    def forward(self, x):
        return self.conv(x)

class ControlNet(torch.nn.Module):
    def __init__(
        self,
        model_ckpt,                  # Checkpoint from which to load pretrained model
        control_channels: int,       # Number of channels in the (B, control_channels, resolution) control signal.
        resolution: int,             # Must match UNet resolution.
        in_channels: int,            # Must match UNet in_channels.
        condition_dim: int,          # Must match UNet condition_dim. 0 = unconditional.
        data_std=0.5,
        **unet_kwargs,
    ):
        super().__init__()
        self.condition_dim = condition_dim
        self.data_std = data_std

        # Baseline trained unet
        self.trained_unet = EDM2Denoiser(resolution=resolution, in_channels=in_channels, condition_dim=condition_dim, **unet_kwargs)
        self.trained_unet.requires_grad_(False)

        # Load ControlNet copy of encoder layers
        self.controlnet = EDM2Denoiser(include_decoder=False, resolution=resolution, in_channels=in_channels, condition_dim=condition_dim, **unet_kwargs)

        # Load weights from checkpoints
        if model_ckpt is not None:
            self.trained_unet.load_state_dict(model_ckpt, strict=True)
            self.controlnet.load_state_dict(model_ckpt, strict=False)

        # Control embedding block: stack of convolutions with a zero-conv at the end
        base_channels = self.trained_unet.unet.base_channels
        self.controlnet_ctrl_emb = nn.Sequential(
            MPConv(control_channels, 64, [3]),
            MPSiLU(),
            MPConv(64, 128, [3]),
            MPSiLU(),
            MPConv(128, base_channels, [3]),
            MPSiLU(),
            ZeroConv(base_channels),
        )

        # Zero convolution modules for encoder levels
        self.controlnet_down_zero_convs = nn.ModuleList([])
        for _, block in self.controlnet.unet.enc.items():
            self.controlnet_down_zero_convs.append(ZeroConv(block.out_channels))

    def get_trainable_params(self):
        # Get all trainable controlnet parameters
        # Copy of unet encoder
        params = list(self.controlnet.parameters())

        # Control embedding block and zero convs
        params += list(self.controlnet_ctrl_emb.parameters())
        params += list(self.controlnet_down_zero_convs.parameters())
        return params

    def forward(self, x, control, noise_std, condition_vector=None):
        x = x.to(torch.float32)
        noise_std = noise_std.to(torch.float32).reshape(-1, 1, 1)
        condition_vector = (
            None
            if self.condition_dim == 0
            else torch.zeros((1, self.condition_dim), device=x.device)
            if condition_vector is None
            else condition_vector.to(torch.float32).reshape(-1, self.condition_dim)
        )

        c_in, c_out, c_skip, c_noise = get_precondition_factors(noise_std, self.data_std)

        # Preconditioning
        x_in = c_in * x

        # Add extra ones channel
        x_in = torch.cat([x_in, torch.ones_like(x_in[:, :1])], dim=1)

        # Noise and class embedding of trained unet, along with encoder outputs
        with torch.no_grad():
            trained_unet_emb = self.trained_unet.unet.embed(c_noise, condition_vector)
            trained_unet_out, trained_unet_skips = self.trained_unet.unet.encode(x_in, trained_unet_emb)

        # Noise and class embedding of controlnet
        controlnet_emb = self.controlnet.unet.embed(c_noise, condition_vector)

        # Control embedding block
        controlnet_ctrl = self.controlnet_ctrl_emb(control)

        # Our copy of the unet encoder
        _, controlnet_skips = self.controlnet.unet.encode(x_in, controlnet_emb, controls = controlnet_ctrl)

        # Apply zero-convs to controlnet outputs
        controlnet_skips = [zero_conv(controlnet_out) for zero_conv, controlnet_out in zip(self.controlnet_down_zero_convs, controlnet_skips)]

        # Add controlnet skips to trained unet skips
        skips = [trained_unet_skip + controlnet_skip for trained_unet_skip, controlnet_skip in zip(trained_unet_skips, controlnet_skips)]

        trained_unet_out = self.trained_unet.unet.decode(trained_unet_out, skips, trained_unet_emb)
        ctrlnet_out = c_skip * x + c_out * trained_unet_out
        return ctrlnet_out
