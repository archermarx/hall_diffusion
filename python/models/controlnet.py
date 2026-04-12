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

from .edm2 import (
    EDM2Denoiser,
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

        # Freeze embeddings — they are copied so the ControlNet encoder
        # processes noise-level and class conditioning in the same
        # representation space as the UNet.  Allowing them to drift would
        # misalign the injected controls with the UNet decoder's expectations.
        net.emb_fourier.requires_grad_(False)
        net.emb_noise.requires_grad_(False)
        if net.emb_label is not None:
            net.emb_label.requires_grad_(False)

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


class ControlNetDenoiser(torch.nn.Module):
    """Pairs a frozen EDM2Denoiser with a trainable ControlNet adapter.

    During the forward pass the ControlNet converts a spatial control signal
    (e.g. the output of ThrusterDataset.generate_measurements) into a dict of
    encoder-stage controls, which are then injected additively into the frozen
    denoiser's skip connections.  When ``ctrl`` is None the denoiser runs
    identically to the unmodified EDM2Denoiser.
    """

    def __init__(self, denoiser: EDM2Denoiser, controlnet: "ControlNet"):
        super().__init__()
        self.denoiser = denoiser    # frozen — requires_grad_(False) at construction
        self.controlnet = controlnet  # trainable

    def train(self, mode: bool = True):
        """Keep the frozen denoiser in eval mode regardless of the training flag.

        nn.Module.train() propagates to all submodules, which would put the
        denoiser's MPConv layers into training mode and trigger forced weight
        normalization (self.weight.copy_(...)) on every forward pass.  That
        in-place write accumulates floating-point rounding error into the
        frozen weights over thousands of steps.
        """
        super().train(mode)
        self.denoiser.eval()   # always eval — never touch frozen weights
        return self

    def forward(self, x, noise_std, condition_vector=None, ctrl=None, return_logvar=False):
        controls = None
        if ctrl is not None:
            # Compute c_noise the same way EDM2Denoiser does so the embeddings stay aligned.
            noise_std_f = noise_std.to(torch.float32).reshape(-1, 1, 1)
            c_noise = noise_std_f.flatten().log() / 4

            # Mirror EDM2Denoiser's condition_vector handling.
            if self.denoiser.label_dim == 0:
                class_labels = None
            elif condition_vector is None:
                class_labels = torch.zeros([1, self.denoiser.label_dim], device=x.device)
            else:
                class_labels = condition_vector.to(torch.float32).reshape(-1, self.denoiser.label_dim)

            # Force float32 regardless of any outer autocast context.
        # Under AMP, autocast recasts conv/linear ops to float16 even when inputs
        # are float32.  The ControlNet encoder runs over out-of-distribution ctrl
        # inputs (vs the image noise it was pre-trained on), and larger models with
        # more attention blocks are more likely to produce inf/nan in float16 for
        # pathological batches.  float32 here costs ~1.5× memory on this sub-module
        # but eliminates that risk; the resulting controls are float32 and safely
        # upcast the skip additions in UNet.forward.
        with torch.autocast(x.device.type, enabled=False):
            controls = self.controlnet(
                ctrl.to(torch.float32),
                c_noise,
                class_labels.to(torch.float32) if class_labels is not None else None,
            )

        return self.denoiser(
            x, noise_std,
            condition_vector=condition_vector,
            return_logvar=return_logvar,
            controls=controls,
        )

    @classmethod
    def from_config(cls, config: dict, device: torch.device) -> "ControlNetDenoiser":
        """Load base EDM2 checkpoint, freeze it, and attach a fresh ControlNet.

        The base model checkpoint is expected at ``config['base_model'] / checkpoint.pth.tar``.
        Architecture hyper-parameters (in_channels, label_dim, resolution) are
        taken from the current config when present, falling back to the values
        stored in the base checkpoint otherwise.
        """
        base_ckpt_path = Path(config["base_model"]) / "checkpoint.pth.tar"
        base_ckpt = torch.load(base_ckpt_path, weights_only=False, map_location=device)

        # Use the base checkpoint's stored config exactly — architecture must match
        # the saved weights, so current-dataset values (in_channels, label_dim, etc.)
        # must not override them here.
        base_model_cfg = dict(base_ckpt["model_config"])

        denoiser = EDM2Denoiser.from_config(base_model_cfg).to(device)
        denoiser.load_state_dict(base_ckpt["model"])
        denoiser.requires_grad_(False)

        controlnet = ControlNet.from_unet(denoiser.unet, config["control_channels"]).to(device)
        return cls(denoiser, controlnet)
