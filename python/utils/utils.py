import os
from pathlib import Path
import sys
import torch
import numpy as np
import bisect
import pandas as pd
import tomllib

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

def get_observation_locs(obs, field, grid, form="normalized", normalizer=None):
    x = obs[field].get("x", "all")
    y = obs[field].get("y", None)

    if x == "all":
        x = grid

    assert (y is None) or (len(x) == len(y))
    x = np.array(x)
    
    if y is not None:
        y = np.array(y)

    x_new = np.zeros_like(x)
    indices = np.zeros_like(x, dtype=int)
    for (i, _x) in enumerate(x):
        j = bisect.bisect_left(grid, _x)
        indices[i] = j
        x_new[i] = grid[j]

    if y is not None:
        normalized = obs[field]["normalized"]

        if form == "normalized" and not normalized:
            if normalizer is None:
                raise RuntimeError("Normalized data requested but no normalizer provided.")

            y = normalizer.normalize(y, field)
        elif form == "denormalized" and normalized:
            if normalizer is None:
                raise RuntimeError("De-normalized data requested but no normalizer provided.")

            y = normalizer.denormalize(y, field)
        elif form != "denormalized" and form != "normalized":
            raise RuntimeError("Data must be requested in either normalized or denormalized form")

    return indices, x_new, y

def read_observation(obs):
    if isinstance(obs, str):
        with open(obs, "rb") as fp:
            obs_dict = tomllib.load(fp)
    elif isinstance(obs, dict):
        obs_dict = obs
    else:
        raise NotImplementedError()

    if "extra" in obs_dict:
        obs_dict.update(read_observation(obs_dict["extra"]))
                        
    return obs_dict
