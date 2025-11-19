import os
from pathlib import Path
import sys
import torch

def get_script_dir():
    return Path(os.path.dirname(os.path.realpath(sys.argv[0])))

def get_device():
    # Device setup
    device = torch.device("cpu")

    if torch.backends.mps.is_available():
        # Apple metal performance shaders
        device = torch.device("mps")
    elif torch.cuda.is_available():
        # NVIDIA CUDA (or AMD ROCM)
        device = torch.device("cuda")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU name: {torch.cuda.get_device_name(0)}")
    elif torch.xpu.is_available():
        # Intel XPU
        device = torch.device("xpu")

    return device