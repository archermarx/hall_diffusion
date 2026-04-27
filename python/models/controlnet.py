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

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from typing import TypedDict

from .edm2 import (
    get_precondition_factors,
    UNet, MPConv, mp_silu,
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
        self.conv = nn.Conv1d(num_channels, num_channels, 1, padding=0)
    
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
        self.trained_unet = UNet(resolution=resolution, in_channels=in_channels, condition_dim=condition_dim, **unet_kwargs)

        # Load ControlNet copy of encoder layers
        self.controlnet = UNet(include_decoder=False, resolution=resolution, in_channels=in_channels, condition_dim=condition_dim, **unet_kwargs)

        # Load weights from checkpoints
        if model_ckpt is not None:
            ckpt = torch.load(model_ckpt)
            self.trained_unet.load_state_dict(ckpt, strict=True)
            self.controlnet.load_state_dict(ckpt, strict=False)

        # Control embedding block: stack of convolutions with a zero-conv at the end
        base_channels = self.trained_unet.base_channels
        self.controlnet_ctrl_emb = nn.Sequential(
            MPConv(control_channels, 64, [3]),
            MPSiLU(),
            MPConv(128, base_channels, [3]),
            MPSiLU(),
            ZeroConv(base_channels),
        )

        # Zero convolution modules for encoder levels
        self.controlnet_down_zero_convs = nn.ModuleList([])
        for _, block in self.controlnet.enc.items():
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

        c_in, c_out, c_skip, c_noise = get_precondition_factors(noise_std, 0.5)

        # Preconditioning
        x = c_in * x

        # Add extra ones channel
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

        # Noise and class embedding of trained unet, along with encoder outputs
        with torch.no_grad():
            trained_unet_emb = self.trained_unet.embed(c_noise, condition_vector)
            trained_unet_out, trained_unet_skips = self.trained_unet.encode(x, trained_unet_emb)

        # Noise and class embedding of controlnet
        controlnet_emb = self.controlnet.embed(c_noise, condition_vector)

        # Control embedding block
        controlnet_ctrl = self.controlnet_ctrl_emb(control)

        # Our copy of the unet encoder
        _, controlnet_skips = self.controlnet.encode(x, controlnet_emb, controls = controlnet_ctrl)

        # Add zero-convolutions to controlnet outputs and add to trained unet skip connectoins
        skips = [
            zero_conv(controlnet_out) + unet_out \
                for (controlnet_out, unet_out, zero_conv) \
                in zip(controlnet_skips, trained_unet_skips, self.controlnet_down_zero_convs)
        ]

        trained_unet_out = self.trained_unet.decode(trained_unet_out, skips, trained_unet_emb)

        return c_skip * x + c_out * trained_unet_out