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

import numpy as np
import torch

from .edm2 import (
    MPConv, mp_silu, mp_sum,
    _encoder_channel_dims, _build_embeddings, _encoder_blocks,
)


class ControlNet(torch.nn.Module):
    def __init__(
        self,
        resolution,             # Must match UNet resolution.
        in_channels,            # Must match UNet in_channels.
        control_channels,       # Number of channels in the (B, control_channels, resolution) control signal.
        condition_dim,          # Must match UNet condition_dim. 0 = unconditional.
        base_channels=192,
        channel_mult=[1, 2, 3, 4, 5],
        channel_mult_noise=None,
        channel_mult_emb=None,
        num_blocks=3,
        attn_resolutions=[16, 8],
        label_balance=0.5,
        **block_kwargs,         # Forwarded to Block (same as UNet).
    ):
        super().__init__()
        cblock, cnoise, cemb = _encoder_channel_dims(
            base_channels, channel_mult, channel_mult_noise, channel_mult_emb
        )
        self.resolution = resolution
        self.in_channels = in_channels
        self.label_balance = label_balance

        # Embedding layers — own copies, same architecture as UNet.
        self.emb_fourier, self.emb_noise, self.emb_label = _build_embeddings(
            cnoise, cemb, condition_dim
        )

        # Mirror UNet encoder.  The stem conv is rebuilt with control_channels
        # as its input (replacing in_channels + 1) so it expands directly into
        # feature space without a separate projection layer.  All other blocks
        # are kept identical to the UNet encoder.
        self.enc = torch.nn.ModuleDict()
        self.ctrl_conv = torch.nn.ModuleDict()
        self.ctrl_gain = torch.nn.ParameterDict()

        first = True
        for name, block, cout in _encoder_blocks(
            resolution, in_channels, cblock, cemb, num_blocks, attn_resolutions, block_kwargs
        ):
            if first:
                # Replace stem conv input channels: control_channels → cblock[0].
                block = MPConv(control_channels, cout, kernel=[3])
                first = False
            self.enc[name] = block
            self.ctrl_conv[name] = MPConv(cout, cout, kernel=[1])
            self.ctrl_gain[name] = torch.nn.Parameter(torch.zeros([]))

    @classmethod
    def from_unet(cls, unet, control_channels, **block_kwargs):
        """Build a ControlNet by copying weights from a trained UNet encoder.

        The ControlNet's embedding layers and encoder blocks are initialised
        with deep copies of the UNet's corresponding weights, so the mirrored
        encoder starts with the same learned representations.  The
        zero-initialized output projections (ctrl_conv / ctrl_gain) are always
        freshly initialised — they must be trained from scratch.

        Args:
            unet:             A UNet instance whose embedding and encoder weights
                              will be copied into the new ControlNet.
            control_channels: Number of channels in the (B, control_channels,
                              resolution) control signal.
            **block_kwargs: Extra keyword arguments forwarded to ControlNet.__init__
                         (e.g. dropout).  Architecture hyperparameters are
                         inferred automatically from the UNet.

        Returns:
            A new ControlNet with encoder weights copied from ``unet``.
        """
        import copy

        condition_dim = unet.emb_label.weight.shape[1] if unet.emb_label is not None else 0

        net = cls(
            resolution=unet.resolution,
            in_channels=unet.in_channels,
            control_channels=control_channels,
            condition_dim=condition_dim,
            base_channels=unet.base_channels,
            channel_mult=unet.channel_mult,
            channel_mult_noise=unet.channel_mult_noise,
            channel_mult_emb=unet.channel_mult_emb,
            num_blocks=unet.num_blocks,
            attn_resolutions=unet.attn_resolutions,
            label_balance=unet.label_balance,
            **block_kwargs,
        )

        # --- Copy encoder and embedding weights ----------------------------
        net.emb_fourier.load_state_dict(copy.deepcopy(unet.emb_fourier.state_dict()))
        net.emb_noise.load_state_dict(copy.deepcopy(unet.emb_noise.state_dict()))
        if net.emb_label is not None and unet.emb_label is not None:
            net.emb_label.load_state_dict(copy.deepcopy(unet.emb_label.state_dict()))

        for key in net.enc:
            if "conv" in key:
                continue  # stem conv has different input channels — cannot copy
            net.enc[key].load_state_dict(copy.deepcopy(unet.enc[key].state_dict()))

        return net

    def forward(self, ctrl, noise_labels, class_labels):
        """Compute control signals from a spatial control input.

        Args:
            ctrl:         (B, control_channels, resolution) spatial control signal.
            noise_labels: (B,) pre-computed noise labels (log(sigma)/4),
                          same values used by the paired EDM2Denoiser call.
            class_labels: (B, condition_dim) class/parameter conditioning,
                          or None if unconditional.

        Returns:
            dict mapping encoder stage names to control tensors. All values
            are zero at initialisation. Pass directly to UNet.forward as the
            `controls` keyword argument.
        """
        # Embedding (same formula as UNet.forward).
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(
                emb,
                self.emb_label(class_labels * np.sqrt(class_labels.shape[1])),
                t=self.label_balance,
            )
        emb = mp_silu(emb)

        # Run mirrored encoder.  The stem conv (first block) accepts ctrl
        # directly; remaining blocks are standard Block(x, emb) calls.
        x = ctrl
        controls = {}
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            controls[name] = self.ctrl_conv[name](x, gain=self.ctrl_gain[name])

        return controls
