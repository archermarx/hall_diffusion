from pathlib import Path

import torch
from . import edm2
from . import controlnet as controlnet_mod


def dataset_config(config: dict) -> dict:
    """Return the model config that governs dataset construction
    as well as the parameters of that base model.

    For a plain edm2 model this is just ``config`` itself.  For a controlnet
    the dataset must be built to match the *base* model's format (e.g.
    ``scalars_in_tensor``, ``downsample_res``), so we load those values from
    the base checkpoint rather than requiring them to be re-specified in the
    controlnet config.
    """
    if config.get("architecture") == "controlnet":
        base_path = Path(config["base_model"]) / "checkpoint.pth.tar"
        ckpt = torch.load(base_path, weights_only=False, map_location="cpu")
        config = ckpt["model_config"]

    return config.copy()


def dataset_settings(config: dict) -> dict:
    """Derive dataset construction settings from a stored model config."""
    data_config = dataset_config(config)
    scalars_in_tensor = data_config.get("scalars_in_tensor", data_config.get("condition_dim") == 0)
    return {
        "scalars_in_tensor": scalars_in_tensor,
        "fourier_features": data_config.get("fourier_features", False),
        "downsample_res": data_config.get("downsample_res", data_config.get("resolution")),
    }


def from_config(config: dict, device: torch.device):
    config = config.copy()
    arch = config.get("architecture", "edm2")
    assert arch in {"edm2", "controlnet"}

    match arch:
        case "edm2":
            config.pop("architecture", None)
            config.pop("scalars_in_tensor", None)
            config.pop("downsample_res", None)
            config.pop("fourier_features", None)
            model = edm2.EDM2Denoiser(**config).to(device)
        case "controlnet":
            base_path = Path(config["base_model"]) / "checkpoint.pth.tar"
            base_ckpt = torch.load(base_path, weights_only=False, map_location="cpu")
            base_cfg = dataset_config(config)
            for key in ("architecture", "scalars_in_tensor", "downsample_res", "fourier_features"):
                base_cfg.pop(key, None)
            model = controlnet_mod.ControlNet(
                model_ckpt=base_ckpt["model"], control_channels=config["control_channels"], **base_cfg
            ).to(device)
        case _:
            raise NotImplementedError()

    return model
