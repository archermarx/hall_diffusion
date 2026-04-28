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

    return config

def from_config(config: dict, device: torch.device):
    arch = config["architecture"]
    assert arch in {"edm2", "controlnet"}

    match arch:
        case "edm2":
            model = edm2.EDM2Denoiser(**config).to(device)
        case "controlnet":
            base_path = Path(config["base_model"]) / "checkpoint.pth.tar"
            base_ckpt = torch.load(base_path, weights_only=False, map_location="cpu")
            base_cfg = dataset_config(config)
            model = controlnet_mod.ControlNet(model_ckpt = base_ckpt["model"], control_channels = config["control_channels"], **base_cfg).to(device)
        case _:
            raise NotImplementedError()

    return model