import os
from pathlib import Path
import sys
import torch
import numpy as np
import bisect
import pandas as pd

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

def get_observation_locs(obs, field, grid):
    resolution = len(grid)
    locs = obs[field].get("locs", "all")

    field_data = None
    
    if locs == "all":
        return grid, np.arange(resolution), field_data
    
    if isinstance(locs, str):
        # Try and open as a file
        file = pd.read_csv(locs)
        z_locs = file["z"].to_numpy()

        # Check for field data
        if field in file.columns:
            field_data = file[field].to_numpy()

    elif isinstance(locs, list):
        z_locs = np.array(locs)

    z_new = np.zeros_like(z_locs)
    indices = np.zeros_like(z_locs, dtype=int)
    for (i, x) in enumerate(z_locs):
        j = bisect.bisect_left(grid, x)
        indices[i] = j
        z_new[i] = grid[j]

    return indices, z_new, field_data
