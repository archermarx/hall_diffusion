# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

# Model modified from above to use 1D convolutions.

import torch
import torch.nn as nn
import numpy as np
import tomllib
import argparse

# ----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (
        value.shape,
        value.dtype,
        value.tobytes(),
        shape,
        dtype,
        device,
        memory_format,
    )
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(
        value, shape=shape, dtype=dtype, device=device, memory_format=memory_format
    )


# ----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


# ----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.


def resample1d(x, f=[1, 1], mode="keep"):
    if mode == "keep":
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    # print(f"{f=}")
    # f = np.outer(f, f)[np.newaxis, :]
    f = const_like(x, f)
    c = x.shape[1]
    if mode == "down":
        return nn.functional.conv1d(
            x, f.tile([c, 1, 1]), groups=c, stride=2, padding=(pad,)
        )
    assert mode == "up"
    return nn.functional.conv_transpose1d(
        x, (f * 4).tile([c, 1, 1]), groups=c, stride=2, padding=(pad,)
    )


# ----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).


def mp_silu(x):
    return nn.functional.silu(x) / 0.596


# ----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t**2)


# ----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


# ----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).


class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


# ----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).


class MPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel)
        )

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 3
        return nn.functional.conv1d(x, w, padding=(w.shape[-1] // 2,))


# ----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).


class Block(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        emb_channels,  # Number of embedding channels.
        flavor="enc",  # Flavor: 'enc' or 'dec'.
        resample_mode="keep",  # Resampling: 'keep', 'up', or 'down'.
        resample_filter=[1, 1],  # Resampling filter.
        attention=False,  # Include self-attention?
        channels_per_head=64,  # Number of channels per attention head.
        res_balance=0.3,  # Balance between main branch (0) and residual branch (1).
        attn_balance=0.3,  # Balance between main branch (0) and self-attention (1).
        clip_act=256,  # Clip output activations. None = do not clip.
        kernel_width=3,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(
            out_channels if flavor == "enc" else in_channels,
            out_channels,
            kernel=[kernel_width],
        )
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[kernel_width])
        self.conv_skip = (
            MPConv(in_channels, out_channels, kernel=[1])
            if in_channels != out_channels
            else None
        )
        self.attn_qkv = (
            MPConv(out_channels, out_channels * 3, kernel=[1])
            if self.num_heads != 0
            else None
        )
        self.attn_proj = (
            MPConv(out_channels, out_channels, kernel=[1])
            if self.num_heads != 0
            else None
        )

    def forward(self, x, emb):
        # Main branch.
        x = resample1d(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).to(y.dtype))
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        if self.num_heads != 0:
            assert self.attn_qkv is not None
            assert self.attn_proj is not None
            y = self.attn_qkv(x)
            # Shape: (B, heads, C_per_head, 3, L) — pixel-normalize across C_per_head, then split
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2])
            q, k, v = normalize(y, dim=2).unbind(3)  # (B, heads, C_per_head, L) each
            # scaled_dot_product_attention expects (B, heads, L, C_per_head)
            q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
            y = nn.functional.scaled_dot_product_attention(q, k, v)
            y = self.attn_proj(y.transpose(2, 3).reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


# ----------------------------------------------------------------------------
# Shared encoder construction helpers.
# Used by both UNet and ControlNet to avoid duplicated architecture code.

def _encoder_channel_dims(base_channels, channel_mult, channel_mult_noise, channel_mult_emb):
    """Compute (cblock, cnoise, cemb) from the standard multiplier args."""
    cblock = [base_channels * x for x in channel_mult]
    cnoise = base_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
    cemb = base_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
    return cblock, cnoise, cemb


def _build_embeddings(cnoise, cemb, condition_dim):
    """Return (emb_fourier, emb_noise, emb_label) embedding modules."""
    emb_fourier = MPFourier(cnoise)
    emb_noise = MPConv(cnoise, cemb, kernel=[])
    emb_label = MPConv(condition_dim, cemb, kernel=[]) if condition_dim != 0 else None
    return emb_fourier, emb_noise, emb_label


def _encoder_blocks(resolution, in_channels, cblock, cemb, num_blocks, attn_resolutions, block_kwargs):
    """
    Build the encoder stage of the EDM2 architecture.
    Yields (name, block, cout) for every encoder level in order.

    The stem conv at level 0 receives ``in_channels + 1`` input channels
    (the +1 is the constant ones channel added in ``forward``).
    ------------------------------------------------------------------------
    Each level after the first (level 0) is structured as

    (prev level) -> [EncD] -> [Enc(A)] -> ... -> [Enc(A)]
       c1 x w      c1 x w/2   c2 x w/2           c2 x w/2

    where [Enc(A)] is either [Enc] or [EncA] depending on whether this resolution has self attention,
    [EncD] is variant of an encoder block that downsamples the image,
    w and c1 are the resolution and number of channels of the previous level, and c2
    is the number of channels after this level.
    At level 0, [EncD] is replaced with [InConv], which increases the channel count
    from in_channels + 1 to input_channels without resampling.
    """

    enc = nn.ModuleDict()
    cout = in_channels + 1
    res = resolution
    for level, channels in enumerate(cblock):
        if level == 0:
            # [InConv]
            # Input (stem) convolutional layer
            # Increases number of channels from in_channels + 1 to channels
            cin, cout = cout, channels
            enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=[3])
            #yield f"{res}x{res}_conv", MPConv(cin, cout, kernel=[3]), cout
        else:
            # [EncD]
            # Encoder block that downsamples
            # Starts all levels except the first
            enc[f"{res}x{res}_down"] = Block(cout, cout, cemb, flavor="enc", resample_mode="down", **block_kwargs)
        for idx in range(num_blocks):
            # [Enc(A)]
            # Encoder block (with or without attention)
            cin, cout = cout, channels
            enc[f"{res}x{res}_block{idx}"] = Block(cin, cout, cemb, flavor="enc", attention=(res in attn_resolutions), **block_kwargs)
            # yield (
            #     f"{res}x{res}_block{idx}",
            #     Block(cin, cout, cemb, flavor="enc", attention=(res in attn_resolutions), **block_kwargs),
            #     cout,
            # )

        if res % 2 != 0:
            raise ValueError("Resolution {res} is not divisible by 2 at UNet level {level}!")
        res //= 2

    return enc, cout


# ----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

class UNet(nn.Module):
    def __init__(
        self,
        resolution,  # Image resolution.
        in_channels,  # Image channels.
        condition_dim,  # Class label dimensionality. 0 = unconditional.
        base_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 3, 4, 5],  # Per-resolution multipliers for the number of channels.
        channel_mult_noise=None,  # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb=None,  # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[16, 8],  # List of resolutions with self-attention.
        label_balance=0.5,  # Balance between noise embedding (0) and class embedding (1).
        concat_balance=0.5,  # Balance between skip connections (0) and main path (1).
        include_decoder=True, # Whether to include the decoder block (useful for ControlNets)
        **block_kwargs,  # Arguments for Block.
    ):
        super().__init__()
        cblock, cnoise, cemb = _encoder_channel_dims(
            base_channels, channel_mult, channel_mult_noise, channel_mult_emb
        )
        self.resolution = resolution
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mult = list(channel_mult)
        self.channel_mult_noise = channel_mult_noise
        self.channel_mult_emb = channel_mult_emb
        self.num_blocks = num_blocks
        self.attn_resolutions = list(attn_resolutions)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = nn.Parameter(torch.zeros([]))
        self.include_decoder = include_decoder

        # Build noise and class embeddings
        self.emb_fourier, self.emb_noise, self.emb_label = _build_embeddings(cnoise, cemb, condition_dim)

        # Construct encoder stage
        self.enc, cout = _encoder_blocks(resolution, in_channels, cblock, cemb, num_blocks, attn_resolutions, block_kwargs)

        if include_decoder:
            # Decoder.
            self.dec = nn.ModuleDict()
            skips = [block.out_channels for block in self.enc.values()]
            for level, channels in reversed(list(enumerate(cblock))):
                res = resolution >> level
                if level == len(cblock) - 1:
                    self.dec[f"{res}x{res}_in0"] = Block(cout, cout, cemb, flavor="dec", attention=True, **block_kwargs)
                    self.dec[f"{res}x{res}_in1"] = Block(cout, cout, cemb, flavor="dec", **block_kwargs)
                else:
                    self.dec[f"{res}x{res}_up"] = Block(cout, cout, cemb, flavor="dec", resample_mode="up", **block_kwargs)
                for idx in range(num_blocks + 1):
                    cin = cout + skips.pop()
                    cout = channels
                    self.dec[f"{res}x{res}_block{idx}"] = Block(cin, cout, cemb, flavor="dec", attention=(res in attn_resolutions), **block_kwargs)

            # Output convolution
            self.out_conv = MPConv(cout, in_channels, kernel=[3])

    def embed(self, noise_labels, class_labels):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance,)
        emb = mp_silu(emb)
        return emb

    def encode(self, x, emb, controls=None):
        skips = []
        for name, block in self.enc.items():
            if "conv" in name:
                x = block(x)
                if controls is not None:
                    x += controls
            else:
                x = block(x, emb)

            skips.append(x)
        return x, skips

    def decode(self, x, skips, emb):
        if self.include_decoder:
            # Decoder.
            for name, block in self.dec.items():
                if "block" in name:
                    x = mp_cat(x, skips.pop(), t=self.concat_balance)
                x = block(x, emb)
            x = self.out_conv(x, gain=self.out_gain)
        return x

    def forward(self, x, noise_labels, class_labels):

        # Extra ones channel replaces biases removed in core blocks of model
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

        # Noise + class embedding
        emb = self.embed(noise_labels, class_labels)

        # Encoder
        x, skips = self.encode(x, emb)

        # Decoder
        x = self.decode(x, skips, emb)
        return x


# ----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.

def get_precondition_factors(noise_std, data_std):
    c_skip = data_std**2 / (noise_std**2 + data_std**2)
    c_out = noise_std * data_std / (noise_std**2 + data_std**2).sqrt()
    c_in = 1 / (data_std**2 + noise_std**2).sqrt()
    c_noise = noise_std.flatten().log() / 4

    return c_in, c_out, c_skip, c_noise


class EDM2Denoiser(nn.Module):
    def __init__(
        self,
        resolution,  # Image resolution.
        in_channels,  # Image channels.
        condition_dim,  # Class label dimensionality. 0 = unconditional.
        data_std=0.5,  # Expected standard deviation of the training data.
        **unet_kwargs,  # Keyword arguments for UNet.
    ):
        super().__init__()
        self.img_resolution = resolution
        self.img_channels = in_channels
        self.condition_dim = condition_dim
        self.data_std = data_std
        self.unet = UNet(
            resolution=resolution,
            in_channels=in_channels,
            condition_dim=condition_dim,
            **unet_kwargs,
        )

    def get_trainable_params(self):
        return self.parameters()

    def forward(
        self, x, noise_std, condition_vector=None, **unet_kwargs
    ):
        x = x.to(torch.float32)
        noise_std = noise_std.to(torch.float32).reshape(-1, 1, 1)
        condition_vector = (
            None
            if self.condition_dim == 0
            else torch.zeros([1, self.condition_dim], device=x.device)
            if condition_vector is None
            else condition_vector.to(torch.float32).reshape(-1, self.condition_dim)
        )

        # Preconditioning weights.
        c_in, c_out, c_skip, c_noise = get_precondition_factors(noise_std, self.data_std)

        # Run the model.
        x_in = c_in * x
        F_x = self.unet(x_in, c_noise, condition_vector, **unet_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        return D_x

    @staticmethod
    @torch.no_grad
    def check_shape(resolution, channels, base_channels, num_blocks):
        batch_size = 1024
        condition_dim = 8
        for dim in (1,):
            model = EDM2Denoiser(
                resolution=resolution,
                in_channels=channels,
                base_channels=base_channels,
                num_blocks=num_blocks,
                condition_dim=condition_dim,
            )
            cond_vec = torch.randn((batch_size, condition_dim))
            x = torch.rand((batch_size, channels) + (resolution,) * dim)
            noise_std = torch.rand((batch_size, 1) + (1,) * dim)
            out = model(x, noise_std, condition_vector=cond_vec)
            print(f"{dim}D: input = {x.shape=}, output = {out.shape=}")
            assert x.shape == out.shape

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--in-channels", type=int, default=17)
    parser.add_argument("--base-channels", type=int, default=64)

    args = parser.parse_args()

    EDM2Denoiser.check_shape(
        resolution=args.resolution,
        channels=args.in_channels,
        base_channels=args.base_channels,
        num_blocks=args.blocks,
    )
