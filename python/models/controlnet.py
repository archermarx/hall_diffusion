"""ControlNet adapter for the EDM2-style 1D UNet.

The ControlNet mirrors the UNet encoder, takes a conditioning vector of
dimension `control_dim`, projects it to (in_channels, resolution), runs it
through the mirrored encoder, and returns a dict of zero-initialized control
signals keyed by encoder stage name. These signals are injected additively
into the UNet's skip-connection list before the decoder consumes them.

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
        control_dim,            # Dimensionality d of the (batch, d) control vector.
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

        # Project (B, control_dim) → (B, in_channels * resolution).
        self.ctrl_linear = MPConv(control_dim, in_channels * resolution, kernel=[])

        # Mirror UNet encoder; attach zero-initialized output projections per stage.
        self.enc = torch.nn.ModuleDict()
        self.ctrl_conv = torch.nn.ModuleDict()
        self.ctrl_gain = torch.nn.ParameterDict()

        for name, block, cout in _encoder_blocks(
            resolution, in_channels, cblock, cemb, num_blocks, attn_resolutions, block_kwargs
        ):
            self.enc[name] = block
            self.ctrl_conv[name] = MPConv(cout, cout, kernel=[1])
            self.ctrl_gain[name] = torch.nn.Parameter(torch.zeros([]))

    @classmethod
    def from_unet(cls, unet, control_dim, **block_kwargs):
        """Build a ControlNet by copying weights from a trained UNet encoder.

        The ControlNet's embedding layers and encoder blocks are initialised
        with deep copies of the UNet's corresponding weights, so the mirrored
        encoder starts with the same learned representations.  The
        zero-initialized output projections (ctrl_conv / ctrl_gain) are always
        freshly initialised — they must be trained from scratch.

        Args:
            unet:        A UNet instance whose embedding and encoder weights
                         will be copied into the new ControlNet.
            control_dim: Dimensionality of the (batch, d) control vector.
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
            control_dim=control_dim,
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
            net.enc[key].load_state_dict(copy.deepcopy(unet.enc[key].state_dict()))

        return net

    def forward(self, ctrl_vec, noise_labels, class_labels):
        """Compute control signals from a conditioning vector.

        Args:
            ctrl_vec:     (B, control_dim) conditioning vector.
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

        # Project control vector to spatial feature map.
        B = ctrl_vec.shape[0]
        x = self.ctrl_linear(ctrl_vec).reshape(B, self.in_channels, self.resolution)

        # Add constant channel (same as UNet.forward).
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

        # Run mirrored encoder; apply zero-initialized output projections.
        controls = {}
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            controls[name] = self.ctrl_conv[name](x, gain=self.ctrl_gain[name])

        return controls
