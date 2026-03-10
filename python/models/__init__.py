import torch
import edm2
from abc import ABC, abstractmethod

class DenoisingDiffusionModel(ABC, torch.nn.Module):
    pass

def from_config(config, device: torch.Device) -> DenoisingDiffusionModel:
    # Load model and config from checkpoint
    arch = config.get("architecture", "edm2")
    assert arch in set(["edm2"])
    match arch:
        case "edm2":
            model = edm2.EDM2Denoiser.from_config(config).to(device)
        case _:
            raise NotImplementedError()

    return model