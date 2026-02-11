import pandas as pd
import numpy as np
from pathlib import Path
from typing import TypedDict
import torch

class NormInfo(TypedDict):
    names: dict[str, int]
    mean: np.ndarray
    std: np.ndarray
    log: np.ndarray

class Normalizer:
    def __init__(self, dir):
        self.dir = Path(dir)
        self.norm_tensor, self.metadata_tensor = Normalizer.read_normalization_info(self.dir/"norm_data.csv")
        self.norm_params, self.metadata_params = Normalizer.read_normalization_info(self.dir/"norm_params.csv")

    @staticmethod
    def read_normalization_info(path: Path|str) -> tuple[NormInfo, pd.DataFrame]:
        df = pd.read_csv(Path(path))
        mean = df["Mean"].to_numpy()
        std = df["Std"].to_numpy()
        log = df["Log"].to_numpy()
        names = {field: i for (i, field) in enumerate(df["Field"])}
        out: NormInfo = {"names": names, "mean": mean, "std": std, "log": log}
        return out, df
    
    def write_normalization_info(self, path: Path|str):
        path = Path(path)
        self.metadata_params.to_csv(path/ "norm_params.csv", index=False)
        self.metadata_tensor.to_csv(path / "norm_data.csv", index=False)
    
    def find_name(self, name: str):
        if name in self.fields():
            norm = self.norm_tensor
        elif name in self.params():
            norm = self.norm_params
        else:
            raise KeyError(f"{name} is not a valid field or param in the given dataset.")

        index = norm["names"][name]
        return index, norm

    def fields(self) -> dict:
        return self.norm_tensor["names"]

    def params(self) -> dict:
        return self.norm_params["names"]

    def normalize(self, val, name: str):
        index, norm = self.find_name(name)
        mean, std, log = norm["mean"], norm["std"], norm["log"]

        mod = torch if isinstance(val, torch.Tensor) else np

        if log[index]:
            val = mod.log(val)

        return (val - mean[index]) / std[index]
    
    def denormalize(self, val, name: str):
        index, norm = self.find_name(name)
        mean, std, log = norm["mean"], norm["std"], norm["log"]
        val = mean[index] + val * std[index]

        mod = torch if isinstance(val, torch.Tensor) else np

        if log[index]:
            val = mod.exp(val)

        return val

    def normalize_params(self, param_vec):
        normed = np.zeros_like(param_vec)
        for (name, i) in self.params().items():
            normed[i] = self.normalize(param_vec[i], name)
        return normed

    def denormalize_params(self, param_vec):
        denormed = np.zeros_like(param_vec)
        for (name, i) in self.params().items():
            denormed[i] = self.denormalize(param_vec[i], name)
        return denormed

    def normalize_tensor(self, tensor):
        normed = np.zeros_like(tensor)
        for (name, i) in self.fields().items():
            normed[:, i, :] = self.normalize(tensor[:, i, :], name)
        return normed

    def denormalize_tensor(self, tensor):
        denormed = np.zeros_like(tensor)
        for (name, i) in self.fields().items():
            denormed[:, i, :] = self.denormalize(tensor[:, i, :], name)
        return denormed
    
    def __eq__(self, other):
        return all(self.metadata_params.eq(other.metadata_params)) and all(self.metadata_tensor.eq(other.metadata_tensor))
