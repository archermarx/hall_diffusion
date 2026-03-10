import torch
from . import edm2

def from_config(config, device: torch.device):
    # Load model and config from checkpoint
    arch = config.get("architecture", "edm2")
    assert arch in set(["edm2"])
    print(config)
    match arch:
        case "edm2":
            model = edm2.EDM2Denoiser.from_config(config).to(device)
        case _:
            raise NotImplementedError()

    return model