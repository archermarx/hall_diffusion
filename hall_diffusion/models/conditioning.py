"""Modality-independent residual adapters for frozen 1D EDM2 denoisers.

An adapter turns an arbitrary supported condition into tokens and uses those
tokens as key/value context for cross-attention over every frozen encoder skip
and the bottleneck.  Its output projections start at zero, so attaching a new
adapter leaves the base denoiser exactly unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .edm2 import EDM2Denoiser, get_precondition_factors, normalize
from .tlpp_vae import TLPP_LATENT_DIM, TLPPVAEEncoder, preprocess_tlpp_counts


@dataclass
class ConditionTokens:
    """Token representation shared by all condition encoders."""

    tokens: Tensor  # (batch, tokens, channels)


def _position_encoding_1d(length: int, channels: int, *, device, dtype) -> Tensor:
    """Fixed position encoding that works for any input length."""
    if channels == 0:
        return torch.empty((length, 0), device=device, dtype=dtype)
    position = torch.linspace(-1, 1, length, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange((channels + 1) // 2, device=device, dtype=torch.float32)
    freqs = math.pi * (1 + freqs // 2)
    encoding = torch.empty((length, channels), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * freqs[: encoding[:, 0::2].shape[1]])
    encoding[:, 1::2] = torch.cos(position * freqs[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype)


def _position_encoding_2d(height: int, width: int, channels: int, *, device, dtype) -> Tensor:
    y_channels = channels // 2
    x_channels = channels - y_channels
    y = _position_encoding_1d(height, y_channels, device=device, dtype=dtype)
    x = _position_encoding_1d(width, x_channels, device=device, dtype=dtype)
    y = y[:, None, :].expand(height, width, -1)
    x = x[None, :, :].expand(height, width, -1)
    return torch.cat((y, x), dim=-1).reshape(height * width, channels)


class ConditionEncoder(nn.Module):
    """Base class for encoders which return contextual tokens."""

    token_dim: int


class MLPConditionEncoder(ConditionEncoder):
    def __init__(self, input_dim: int, token_dim: int, num_tokens: int = 4, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        if depth < 1:
            raise ValueError("MLP encoder depth must be positive")
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
            width = hidden_dim
        layers.append(nn.Linear(width, num_tokens * token_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, value: Tensor) -> ConditionTokens:
        if value.ndim != 2:
            raise ValueError(f"MLP condition must have shape (B, D), got {tuple(value.shape)}")
        tokens = self.network(value.to(torch.float32)).reshape(value.shape[0], self.num_tokens, self.token_dim)
        return ConditionTokens(tokens=tokens)


class TLPPVAEConditionEncoder(ConditionEncoder):
    """Turn raw 128x128 TLPP counts into learned context tokens via a VAE latent."""

    def __init__(
        self,
        token_dim: int,
        num_tokens: int = 4,
        hidden_dim: int = 256,
        depth: int = 2,
        freeze_vae: bool = True,
        checkpoint: str | None = None,
        initialize_pretrained: bool = True,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.freeze_vae = freeze_vae
        self.vae = TLPPVAEEncoder()
        self.bridge = MLPConditionEncoder(
            input_dim=TLPP_LATENT_DIM,
            token_dim=token_dim,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        if initialize_pretrained:
            if checkpoint is None:
                raise ValueError("TLPP VAE condition encoder requires a checkpoint")
            self.vae.load_pretrained(checkpoint)
        self.vae.requires_grad_(not freeze_vae)
        if freeze_vae:
            self.vae.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vae:
            self.vae.eval()
        return self

    def encode_latent(self, value: Tensor) -> Tensor:
        prepared = preprocess_tlpp_counts(value)
        if self.freeze_vae:
            with torch.no_grad():
                return self.vae(prepared)
        return self.vae(prepared)

    def forward(self, value: Tensor) -> ConditionTokens:
        # Frozen VAE means may be cached by the adapter training data path.
        latent = value if value.ndim == 2 else self.encode_latent(value)
        return self.bridge(latent)


def _group_count(channels: int, maximum: int = 32) -> int:
    """Largest GroupNorm group count that divides the channel width."""
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    raise AssertionError("every positive channel count is divisible by one")


class _ResidualBlock(nn.Module):
    """Pre-activation, GroupNorm residual block for 1D or 2D feature maps."""

    def __init__(self, conv, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.norm0 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv0 = conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv1 = conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            conv(in_channels, out_channels, kernel_size=1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.conv0(F.silu(self.norm0(x)))
        x = self.conv1(F.silu(self.norm1(x)))
        return (x + residual) * (2**-0.5)


class _ConvConditionEncoder(ConditionEncoder):
    def __init__(self, in_channels: int, token_dim: int, channels: list[int], dimensions: int, blocks_per_stage: int = 2):
        super().__init__()
        if not channels:
            raise ValueError("CNN encoder needs at least one channel stage")
        if blocks_per_stage < 1:
            raise ValueError("CNN encoder blocks_per_stage must be positive")
        self.token_dim = token_dim
        conv = nn.Conv1d if dimensions == 1 else nn.Conv2d
        self.stem = conv(in_channels, channels[0], kernel_size=3, padding=1)
        stages: list[nn.Module] = []
        cin = channels[0]
        for index, cout in enumerate(channels):
            # The first stage preserves native detail; later stages reduce token count.
            stride = 1 if index == 0 else 2
            blocks = [_ResidualBlock(conv, cin, cout, stride=stride)]
            blocks.extend(_ResidualBlock(conv, cout, cout) for _ in range(blocks_per_stage - 1))
            stages.append(nn.Sequential(*blocks))
            cin = cout
        self.stages = nn.ModuleList(stages)
        self.project = conv(cin, token_dim, kernel_size=1)

    def features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

class Conv1dConditionEncoder(_ConvConditionEncoder):
    def __init__(self, in_channels: int, token_dim: int, channels: list[int], blocks_per_stage: int = 2):
        super().__init__(in_channels, token_dim, channels, dimensions=1, blocks_per_stage=blocks_per_stage)

    def forward(self, value: Tensor) -> ConditionTokens:
        if value.ndim != 3:
            raise ValueError(f"1D condition must have shape (B, C, L), got {tuple(value.shape)}")
        y = self.project(self.features(value.to(torch.float32)))
        tokens = y.transpose(1, 2)
        tokens = tokens + _position_encoding_1d(tokens.shape[1], self.token_dim, device=tokens.device, dtype=tokens.dtype)
        return ConditionTokens(tokens=tokens)


class Conv2dConditionEncoder(_ConvConditionEncoder):
    def __init__(self, in_channels: int, token_dim: int, channels: list[int], blocks_per_stage: int = 2):
        super().__init__(in_channels, token_dim, channels, dimensions=2, blocks_per_stage=blocks_per_stage)

    def forward(self, value: Tensor) -> ConditionTokens:
        if value.ndim != 4:
            raise ValueError(f"2D condition must have shape (B, C, H, W), got {tuple(value.shape)}")
        y = self.project(self.features(value.to(torch.float32)))
        tokens = y.flatten(2).transpose(1, 2)
        tokens = tokens + _position_encoding_2d(
            y.shape[-2], y.shape[-1], self.token_dim, device=tokens.device, dtype=tokens.dtype
        )
        return ConditionTokens(tokens=tokens)


def build_condition_encoder(
    config: Mapping[str, object], *, initialize_pretrained: bool = True
) -> ConditionEncoder:
    """Build one of the repository-owned condition encoders from config."""
    kind = config["type"]
    token_dim = int(config["token_dim"])
    if kind == "mlp":
        return MLPConditionEncoder(
            input_dim=int(config["input_dim"]), token_dim=token_dim,
            num_tokens=int(config.get("num_tokens", 4)), hidden_dim=int(config.get("hidden_dim", 256)),
            depth=int(config.get("depth", 2)),
        )
    if kind == "cnn1d":
        return Conv1dConditionEncoder(
            int(config["in_channels"]), token_dim, list(config["channels"]), int(config.get("blocks_per_stage", 2))
        )
    if kind == "cnn2d":
        return Conv2dConditionEncoder(
            int(config["in_channels"]), token_dim, list(config["channels"]), int(config.get("blocks_per_stage", 2))
        )
    if kind == "tlpp_vae":
        freeze_vae = config.get("freeze_vae", True)
        if not isinstance(freeze_vae, bool):
            raise TypeError("TLPP VAE freeze_vae must be a boolean")
        checkpoint = config.get("checkpoint")
        if checkpoint is not None and not isinstance(checkpoint, str):
            raise TypeError("TLPP VAE checkpoint must be a path string")
        return TLPPVAEConditionEncoder(
            token_dim=token_dim,
            num_tokens=int(config.get("num_tokens", 4)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            depth=int(config.get("depth", 2)),
            freeze_vae=freeze_vae,
            checkpoint=checkpoint,
            initialize_pretrained=initialize_pretrained,
        )
    raise ValueError(f"unknown condition encoder type {kind!r}; expected mlp, cnn1d, cnn2d, or tlpp_vae")


class CrossAttentionResidual(nn.Module):
    """A zero-gated residual prediction for one frozen feature tensor."""

    def __init__(self, channels: int, emb_channels: int, token_dim: int, channels_per_head: int):
        super().__init__()
        if channels % channels_per_head:
            raise ValueError(f"feature channels ({channels}) must divide channels_per_head ({channels_per_head})")
        self.channels = channels
        self.num_heads = channels // channels_per_head
        self.head_dim = channels_per_head
        self.query = nn.Linear(channels, channels, bias=False)
        self.key_value = nn.Linear(token_dim, channels * 2, bias=False)
        self.time = nn.Linear(emb_channels, channels * 2)
        self.output = nn.Conv1d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature: Tensor, emb: Tensor, context: ConditionTokens) -> Tensor:
        # Frozen feature tensors are constants, but all adapter operations remain differentiable.
        batch, channels, length = feature.shape
        if context.tokens.shape[0] != batch:
            raise ValueError("condition batch size must match denoiser batch size")
        q = normalize(feature, dim=1).transpose(1, 2)
        q = self.query(q).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.key_value(context.tokens.to(q.dtype))
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(batch, length, channels).transpose(1, 2)
        scale, shift = self.time(emb.to(y.dtype)).chunk(2, dim=1)
        y = F.silu(y * (scale.unsqueeze(-1) + 1) + shift.unsqueeze(-1))
        return self.output(y)


class ConditionAdapter(nn.Module):
    """One independently trainable encoder plus residual heads for an EDM2 base."""

    def __init__(self, base: EDM2Denoiser, encoder: ConditionEncoder, channels_per_head: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.site_names = list(base.unet.enc.keys()) + ["bottleneck"]
        site_channels = [block.out_channels for block in base.unet.enc.values()]
        site_channels.append(site_channels[-1])
        emb_channels = base.unet.emb_noise.weight.shape[0]
        common_channels = math.gcd(*site_channels)
        head_channels = channels_per_head or min(32, common_channels)
        self.heads = nn.ModuleDict({
            name: CrossAttentionResidual(channels, emb_channels, encoder.token_dim, head_channels)
            for name, channels in zip(self.site_names, site_channels, strict=True)
        })

    def encode_condition(self, condition: Tensor) -> ConditionTokens:
        return self.encoder(condition)

    def residuals(self, features: Mapping[str, Tensor], emb: Tensor, context: ConditionTokens) -> dict[str, Tensor]:
        return {name: self.heads[name](features[name], emb, context) for name in self.site_names}


class ConditionedEDM2(nn.Module):
    """Frozen EDM2 base with any number of independently trained adapters."""

    def __init__(self, base: EDM2Denoiser, adapters: Mapping[str, ConditionAdapter]):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.adapters = nn.ModuleDict(adapters)
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # MPConv mutates weights when training; frozen must also mean eval mode here.
        self.base.eval()
        return self

    def get_trainable_params(self):
        return (parameter for parameter in self.adapters.parameters() if parameter.requires_grad)

    def prepare_conditions(self, conditions: Mapping[str, Tensor]) -> dict[str, ConditionTokens]:
        unknown = set(conditions).difference(self.adapters)
        if unknown:
            raise ValueError(f"unknown adapter(s): {sorted(unknown)}")
        return {name: self.adapters[name].encode_condition(value) for name, value in conditions.items()}

    def _base_features(self, x: Tensor, noise_std: Tensor, condition_vector: Tensor | None):
        x = x.to(torch.float32)
        noise_std = noise_std.to(torch.float32).reshape(-1, 1, 1)
        if self.base.condition_dim == 0:
            labels = None
        elif condition_vector is None:
            labels = torch.zeros((x.shape[0], self.base.condition_dim), device=x.device)
        else:
            labels = condition_vector.to(torch.float32).reshape(-1, self.base.condition_dim)
        c_in, c_out, c_skip, c_noise = get_precondition_factors(noise_std, self.base.data_std)
        x_in = torch.cat((c_in * x, torch.ones_like(x[:, :1])), dim=1)
        with torch.no_grad():
            emb = self.base.unet.embed(c_noise, labels)
        if x.requires_grad:
            # DPS differentiates its observation likelihood through the
            # denoiser with respect to x. Preserve that input Jacobian while
            # keeping the frozen encoder graph-free during adapter training.
            bottleneck, skips = self.base.unet.encode(x_in, emb)
        else:
            with torch.no_grad():
                bottleneck, skips = self.base.unet.encode(x_in, emb)
        features = dict(zip(self.base.unet.enc.keys(), skips, strict=True))
        features["bottleneck"] = bottleneck
        return x, c_out, c_skip, emb, bottleneck, skips, features

    def forward(
        self,
        x: Tensor,
        noise_std: Tensor,
        condition_vector: Tensor | None = None,
        *,
        conditions: Mapping[str, Tensor] | None = None,
        contexts: Mapping[str, ConditionTokens] | None = None,
        adapter_scales: Mapping[str, float] | None = None,
    ) -> Tensor:
        if conditions is not None and contexts is not None:
            raise ValueError("pass raw conditions or prepared contexts, not both")
        if contexts is None:
            contexts = {} if conditions is None else self.prepare_conditions(conditions)
        unknown = set(contexts).difference(self.adapters)
        if unknown:
            raise ValueError(f"unknown adapter(s): {sorted(unknown)}")
        x, c_out, c_skip, emb, bottleneck, skips, features = self._base_features(x, noise_std, condition_vector)
        residuals = {name: torch.zeros_like(value) for name, value in features.items()}
        for name, context in contexts.items():
            scale = 1.0 if adapter_scales is None else float(adapter_scales.get(name, 1.0))
            if scale == 0:
                continue
            for site, value in self.adapters[name].residuals(features, emb, context).items():
                residuals[site] = residuals[site] + scale * value
        modified_skips = [skip + residuals[name] for name, skip in zip(self.base.unet.enc.keys(), skips, strict=True)]
        # Do not use no_grad: gradients must traverse the frozen decoder to adapter residuals.
        output = self.base.unet.decode(bottleneck + residuals["bottleneck"], modified_skips, emb)
        return c_skip * x + c_out * output.to(torch.float32)
