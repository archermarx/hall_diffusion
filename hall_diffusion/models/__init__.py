from . import edm2
from .conditioning import (
    ConditionAdapter,
    ConditionedEDM2,
    TLPPVAEConditionEncoder,
    build_condition_encoder,
)

try:
    from hall_diffusion.configuration import resolve_model_config
except ModuleNotFoundError:  # Support running hall_diffusion/train.py directly.
    from configuration import resolve_model_config


def dataset_config(config: dict) -> dict:
    """Return the model config that governs dataset construction
    as well as the parameters of that base model.

    The current model family has one base architecture: EDM2.  This helper is
    retained because sampling and datasets share the checkpoint-derived
    settings.
    """
    return resolve_model_config(config)


def dataset_settings(config: dict) -> dict:
    """Derive dataset construction settings from a stored model config."""
    data_config = dataset_config(config)
    scalars_in_tensor = data_config.get("scalars_in_tensor", data_config.get("condition_dim") == 0)
    return {
        "scalars_in_tensor": scalars_in_tensor,
        "fourier_features": data_config["fourier_features"],
        "downsample_res": data_config.get("downsample_res", data_config.get("resolution")),
    }


def from_config(config: dict, device):
    # Resolving here keeps sparse model configs from older checkpoints valid.
    config = resolve_model_config(config)
    arch = config.get("architecture", "edm2")
    assert arch == "edm2"

    config.pop("architecture", None)
    config.pop("scalars_in_tensor", None)
    config.pop("downsample_res", None)
    config.pop("fourier_features", None)
    config.pop("base_conditioning", None)
    model = edm2.EDM2Denoiser(**config).to(device)

    return model


def make_conditioned_model(base, adapter_configs: dict[str, dict], device):
    """Attach new, zero-initialized adapters to an already-loaded EDM2 base."""
    adapters = {
        name: ConditionAdapter(base, build_condition_encoder(config), config.get("channels_per_head"))
        for name, config in adapter_configs.items()
    }
    return ConditionedEDM2(base, adapters).to(device)
