"""Portable checkpoint helpers for independently trained condition adapters."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from hall_diffusion.configuration import resolve_model_config
from .conditioning import ConditionAdapter, build_condition_encoder


ADAPTER_FORMAT_VERSION = 1


def base_signature(model_config: dict) -> dict:
    """Architecture fields that must agree before an adapter can attach."""
    config = resolve_model_config(model_config)
    ignored = {"architecture", "scalars_in_tensor", "fourier_features", "downsample_res", "base_conditioning"}
    return {key: deepcopy(value) for key, value in config.items() if key not in ignored}


def make_adapter_artifact(name: str, adapter: ConditionAdapter, adapter_config: dict, base_model_config: dict, *,
                          ema_state: dict | None = None, optimizer_state: dict | None = None,
                          train_config: dict | None = None, training_state: dict | None = None):
    return {
        "format_version": ADAPTER_FORMAT_VERSION,
        "artifact_type": "condition_adapter",
        "name": name,
        "adapter_config": deepcopy(adapter_config),
        "base_signature": base_signature(base_model_config),
        "model": adapter.state_dict(),
        "ema": ema_state,
        "optimizer": optimizer_state,
        "train_config": deepcopy(train_config),
        "training_state": deepcopy(training_state),
    }


def save_adapter(path: str | Path, **kwargs) -> None:
    torch.save(make_adapter_artifact(**kwargs), Path(path))


def load_adapter(path: str | Path, base, base_model_config: dict, *, weights: str = "ema"):
    """Load an adapter artifact onto an architecture-compatible EDM2 base."""
    artifact = torch.load(Path(path), weights_only=False, map_location="cpu")
    if artifact.get("artifact_type") != "condition_adapter":
        raise ValueError(f"{path} is not a condition-adapter artifact")
    if artifact.get("format_version") != ADAPTER_FORMAT_VERSION:
        raise ValueError(f"unsupported adapter format {artifact.get('format_version')!r}")
    if artifact["base_signature"] != base_signature(base_model_config):
        raise ValueError("adapter architecture does not match the supplied EDM2 base")
    config = artifact["adapter_config"]
    adapter = ConditionAdapter(base, build_condition_encoder(config["encoder"]), config.get("channels_per_head"))
    state = artifact.get(weights) if weights == "ema" else None
    state = state if state is not None else artifact["model"]
    adapter.load_state_dict(state, strict=True)
    return artifact["name"], adapter, artifact
