"""Configuration defaults and resolution helpers.

Defaults live here so the values used to construct a model are also the values
persisted in its checkpoint.
"""

from copy import deepcopy


EDM2_DEFAULTS = {
    "data_std": 0.5,
    "base_channels": 192,
    "channel_mult": [1, 2, 3, 4, 5],
    "channel_mult_noise": None,
    "channel_mult_emb": None,
    "num_blocks": 3,
    "attn_resolutions": [16, 8],
    "label_balance": 0.5,
    "concat_balance": 0.5,
    "channels_per_head": 32,
    "res_balance": 0.3,
    "attn_balance": 0.3,
    "clip_act": 256,
    "kernel_width": 3,
    "resample_filter": [1, 1],
}

TRAINING_DEFAULTS = {
    "condition_dropout": 0.0,
    "use_amp": True,
    "load_workers": 2,
    "prefetch_factor": 4,
    "ema_epochs": None,
    "ema_start_epochs": 128,
}

OPTIMIZER_DEFAULTS = {
    "adam_betas": [0.9, 0.999],
    "weight_decay_epochs": None,
}

LOSS_DEFAULTS = {
    "P_mean": -0.4,
    "P_std": 1.0,
    "sigma_data": 0.5,
    "deriv_h": 1.0,
}


def _apply_defaults(values: dict, defaults: dict) -> None:
    for key, value in defaults.items():
        values.setdefault(key, deepcopy(value))


def resolve_model_config(config: dict) -> dict:
    """Return a complete, independent model configuration.

    Sparse configurations from old checkpoints are intentionally accepted and
    filled with the same defaults used by current model constructors.
    """
    resolved = deepcopy(config)

    # Older checkpoints used label_dim for the conditioning-vector width.
    if "label_dim" in resolved:
        resolved.setdefault("condition_dim", resolved["label_dim"])
        del resolved["label_dim"]

    resolved.setdefault("architecture", "edm2")
    if resolved["architecture"] == "edm2":
        _apply_defaults(resolved, EDM2_DEFAULTS)
        resolved.setdefault("scalars_in_tensor", resolved.get("condition_dim") == 0)
        resolved.setdefault("fourier_features", False)
        # Older configs did not distinguish an unconditional base from the
        # historical vector-label path.  Preserve that behavior by default.
        resolved.setdefault(
            "base_conditioning",
            "none" if resolved.get("condition_dim") == 0 else "legacy_vector",
        )
        if "resolution" in resolved:
            resolved.setdefault("downsample_res", resolved["resolution"])

    return resolved


def resolve_training_config(config: dict) -> dict:
    """Return training configuration with all supported optional values set."""
    resolved = deepcopy(config)
    _apply_defaults(resolved, TRAINING_DEFAULTS)

    resolved.setdefault("loss", {})
    _apply_defaults(resolved["loss"], LOSS_DEFAULTS)

    # The optimizer table itself and several of its entries are required.
    if "optimizer" in resolved:
        _apply_defaults(resolved["optimizer"], OPTIMIZER_DEFAULTS)

    return resolved


def resolve_config(config: dict) -> dict:
    """Resolve a complete training-file configuration for use and storage."""
    resolved = deepcopy(config)
    resolved["model"] = resolve_model_config(resolved["model"])
    resolved["training"] = resolve_training_config(resolved["training"])
    return resolved
