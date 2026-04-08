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

from .edm2 import Block, MPConv, MPFourier, mp_silu, mp_sum


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
        cblock = [base_channels * x for x in channel_mult]
        cnoise = (
            base_channels * channel_mult_noise
            if channel_mult_noise is not None
            else cblock[0]
        )
        cemb = (
            base_channels * channel_mult_emb
            if channel_mult_emb is not None
            else max(cblock)
        )
        self.resolution = resolution
        self.in_channels = in_channels
        self.label_balance = label_balance

        # Embedding layers — own copies, same architecture as UNet.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = (
            MPConv(condition_dim, cemb, kernel=[]) if condition_dim != 0 else None
        )

        # Project (B, control_dim) → (B, in_channels * resolution).
        self.ctrl_linear = MPConv(control_dim, in_channels * resolution, kernel=[])

        # Mirror UNet encoder (identical loop to UNet.__init__ lines 321-343).
        self.enc = torch.nn.ModuleDict()
        self.ctrl_conv = torch.nn.ModuleDict()
        self.ctrl_gain = torch.nn.ParameterDict()

        cout = in_channels + 1
        for level, channels in enumerate(cblock):
            res = resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                name = f"{res}x{res}_conv"
                self.enc[name] = MPConv(cin, cout, kernel=[3])
            else:
                name = f"{res}x{res}_down"
                self.enc[name] = Block(
                    cout, cout, cemb, flavor="enc", resample_mode="down", **block_kwargs
                )
            self.ctrl_conv[name] = MPConv(cout, cout, kernel=[1])
            self.ctrl_gain[name] = torch.nn.Parameter(torch.zeros([]))

            for idx in range(num_blocks):
                cin = cout
                cout = channels
                name = f"{res}x{res}_block{idx}"
                self.enc[name] = Block(
                    cin,
                    cout,
                    cemb,
                    flavor="enc",
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
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

        # --- Infer architecture kwargs from the UNet -----------------------
        # Recover channel counts from the enc ModuleDict key names.
        enc_keys = list(unet.enc.keys())
        # Resolution of the first (largest) level: "RxR_conv" or "RxR_block0"
        first_res = int(enc_keys[0].split("x")[0])
        resolution = first_res

        # in_channels: the stem conv has cin = in_channels + 1.
        # MPConv stores weight as (out, in, *kernel), so in_channels = weight.shape[1].
        stem_conv = unet.enc[enc_keys[0]]
        in_channels = stem_conv.weight.shape[1] - 1

        # Recover channel_mult, num_blocks, attn_resolutions from enc structure.
        channel_mult = []
        attn_resolutions = []
        num_blocks = None

        level_data = {}  # res → list of block names at that level
        for key in enc_keys:
            res_str = key.split("_")[0]  # e.g. "32x32"
            level_data.setdefault(res_str, []).append(key)

        base_channels = None
        for res_str, keys in level_data.items():
            res = int(res_str.split("x")[0])
            # The block at this level that gives us the output channel count.
            last_block = unet.enc[keys[-1]]
            cout = last_block.out_channels
            if base_channels is None:
                # First level: stem conv, cout = base_channels * mult[0].
                # We can infer base_channels from subsequent levels once we
                # have the full list; defer until after the loop.
                pass
            channel_mult.append(cout)

            # Count only block{idx} entries at this level for num_blocks.
            blocks_at_level = [k for k in keys if "block" in k]
            if num_blocks is None:
                num_blocks = len(blocks_at_level)

            # Check attention in the first block at this level (if any).
            for k in blocks_at_level:
                blk = unet.enc[k]
                if hasattr(blk, "attn") and blk.attn is not None:
                    attn_resolutions.append(res)
                    break

        # Infer base_channels: GCD of all channel counts should give it.
        import math
        base_channels = channel_mult[0]
        for c in channel_mult[1:]:
            base_channels = math.gcd(base_channels, c)
        channel_mult = [c // base_channels for c in channel_mult]

        # Embedding dimensions.
        # emb_noise weight shape: (cemb, cnoise) — input to emb_noise comes from emb_fourier.
        cnoise = unet.emb_noise.weight.shape[1]
        cemb = unet.emb_noise.weight.shape[0]
        channel_mult_noise = cnoise // base_channels if cnoise != channel_mult[0] * base_channels else None
        channel_mult_emb = cemb // base_channels if cemb != max(c * base_channels for c in channel_mult) else None

        condition_dim = unet.emb_label.weight.shape[1] if unet.emb_label is not None else 0
        label_balance = unet.label_balance

        # --- Construct the ControlNet with the inferred architecture -------
        net = cls(
            resolution=resolution,
            in_channels=in_channels,
            control_dim=control_dim,
            condition_dim=condition_dim,
            base_channels=base_channels,
            channel_mult=channel_mult,
            channel_mult_noise=channel_mult_noise,
            channel_mult_emb=channel_mult_emb,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            label_balance=label_balance,
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
