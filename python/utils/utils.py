import os
from pathlib import Path
import sys
import logging
import torch
import numpy as np
import bisect
import pandas as pd
import tomllib
from contextlib import contextmanager
import pathlib
from datetime import timedelta

def paths_to_strings(d: dict):
    out = {}
    for (k, v) in d.items():
        if isinstance(v, pathlib.Path) or isinstance(v, pathlib.WindowsPath) or isinstance(v, pathlib.PosixPath):
            out[k] = str(v)
            print(f"Converted path {v} to string.")
        elif isinstance(v, dict):
            out[k] = paths_to_strings(v)
        else:
            out[k] = v

    return out

def get_script_dir():
    return Path(os.path.dirname(os.path.realpath(sys.argv[0])))

def get_device():
    # Device setup
    device = torch.device("cpu")
    # print("get_device called!")
    # traceback.print_stack()

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
        x_new = grid
        indices = np.arange(len(x_new))
    else:
        x = np.array(x)
        x_new = np.zeros_like(x)
        indices = np.zeros_like(x, dtype=int)
        for (i, _x) in enumerate(x):
            j = bisect.bisect_left(grid, _x)
            indices[i] = j
            x_new[i] = grid[j]
    
    assert (y is None) or (len(x) == len(y))

    if y is not None:
        y = np.array(y)
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

def get_logger(name: str, filename: str | None = None, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Always add stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Optionally add file handler
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
