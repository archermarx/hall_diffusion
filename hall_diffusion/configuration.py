"""Configuration defaults and resolution helpers.

Defaults live here so the values used to construct a model are also the values
persisted in its checkpoint.
"""

from copy import deepcopy
import math
from numbers import Real


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
    "torch_compile": False,
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
        # Dataset-derived Fourier conditioning is no longer supported. Keep
        # accepting the key so old configs/checkpoints remain loadable.
        resolved["fourier_features"] = False
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


def duration_in_epochs(
    config: dict,
    dataset_size: int,
    epochs_key: str,
    examples_key: str,
    *,
    required: bool = True,
    allow_zero: bool = False,
) -> float | None:
    """Resolve an epoch or example duration to one floating-point epoch value."""
    if dataset_size <= 0:
        raise ValueError("training dataset must contain at least one example")

    examples = config.get(examples_key)
    if examples is not None:
        minimum = 0 if allow_zero else 1
        if (
            isinstance(examples, bool)
            or not isinstance(examples, int)
            or examples < minimum
        ):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"training {examples_key} must be a {qualifier} integer")
        return float(examples / dataset_size)

    epochs = config.get(epochs_key)
    if epochs is None:
        if required:
            raise ValueError(
                f"training requires either {epochs_key!r} or {examples_key!r}"
            )
        return None
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, Real)
        or not math.isfinite(epochs)
        or epochs < 0
        or (not allow_zero and epochs <= 0)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"training {epochs_key} must be a finite {qualifier} number")
    return float(epochs)


def training_duration(config: dict, dataset_size: int) -> float:
    """Resolve total training duration to floating-point epochs."""
    return duration_in_epochs(config, dataset_size, "epochs", "max_examples")


def training_example_target(epochs: float, dataset_size: int) -> int:
    """Convert a possibly fractional epoch duration to its nearest whole example."""
    return max(1, round(epochs * dataset_size))


def limited_batch_size(processed_examples: int, target_examples: int, batch_size: int) -> int:
    """Return how many examples from the next batch fit in the training budget."""
    return max(0, min(batch_size, target_examples - processed_examples))


def resolve_config(config: dict) -> dict:
    """Resolve a complete training-file configuration for use and storage."""
    resolved = deepcopy(config)
    resolved["model"] = resolve_model_config(resolved["model"])
    resolved["training"] = resolve_training_config(resolved["training"])
    return resolved
