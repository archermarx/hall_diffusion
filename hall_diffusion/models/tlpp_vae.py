"""Inference-only encoder for the production 128x128 TLPP VAE."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn


TLPP_SAMPLE_SHAPE = (1, 128, 128)
TLPP_LATENT_DIM = 128
TLPP_LOG_EPSILON = 1e-6


def preprocess_tlpp_counts(counts: Tensor) -> Tensor:
    """Convert raw TLPP counts to the probability-normalized log01 VAE input."""
    values = counts.to(torch.float32)
    total = values.sum(dim=(-2, -1), keepdim=True)
    probability = values / total.clamp_min(1.0)
    return torch.log1p(probability / TLPP_LOG_EPSILON) / math.log1p(1.0 / TLPP_LOG_EPSILON)


class TLPPVAEEncoder(nn.Module):
    """The convolutional encoder and latent-mean head from the trained VAE."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
        )
        self.mu = nn.Linear(64 * 16 * 16, TLPP_LATENT_DIM)

    def forward(self, value: Tensor) -> Tensor:
        return self.mu(self.encoder(value).flatten(1))

    def load_pretrained(self, checkpoint: str | Path) -> None:
        """Initialize from a full TLPP VAE training checkpoint."""
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"TLPP VAE checkpoint does not exist: {path}")

        # The supplied file is a trusted training/resume checkpoint and includes
        # NumPy/Python RNG state that is not supported by weights_only loading.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise ValueError(f"TLPP VAE checkpoint must contain a dictionary: {path}")
        if tuple(state.get("sample_shape", ())) != TLPP_SAMPLE_SHAPE:
            raise ValueError(
                f"TLPP VAE checkpoint must use sample shape {TLPP_SAMPLE_SHAPE}, "
                f"got {state.get('sample_shape')!r}"
            )
        vae_config = state.get("vae", {})
        if not isinstance(vae_config, dict):
            raise ValueError("TLPP VAE checkpoint is missing VAE configuration")
        if vae_config.get("latent_dim") != TLPP_LATENT_DIM:
            raise ValueError(
                f"TLPP VAE checkpoint must use latent_dim={TLPP_LATENT_DIM}, "
                f"got {vae_config.get('latent_dim')!r}"
            )
        if vae_config.get("probability_mode") != "log01":
            raise ValueError(
                "TLPP VAE checkpoint must use probability_mode='log01', "
                f"got {vae_config.get('probability_mode')!r}"
            )
        model_state = state.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("TLPP VAE checkpoint is missing model_state")

        prefixes = ("net.encoder.", "net.mu.")
        encoder_state = {
            name.removeprefix("net."): value
            for name, value in model_state.items()
            if name.startswith(prefixes)
        }
        try:
            self.load_state_dict(encoder_state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"TLPP VAE checkpoint has incompatible encoder weights: {path}") from exc
