from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch


class NormInfo(TypedDict):
    names: dict[str, int]
    mean: np.ndarray
    std: np.ndarray
    log: np.ndarray


def empty_norm_info() -> NormInfo:
    return {"names": {}, "mean": np.array([]), "std": np.array([]), "log": np.array([], dtype=bool)}


def concat_norm_info(*infos: NormInfo) -> NormInfo:
    """Combine normalization metadata without mutating any category map."""
    names = {}
    offset = 0
    for info in infos:
        names.update({name: index + offset for name, index in info["names"].items()})
        offset += len(info["mean"])
    return {
        "names": names,
        "mean": np.concatenate([info["mean"] for info in infos]),
        "std": np.concatenate([info["std"] for info in infos]),
        "log": np.concatenate([info["log"] for info in infos]),
    }


class Normalizer:
    def __init__(self, dir, scalars_in_tensor=False, fourier_features=False):
        self.dir = Path(dir)
        self.fourier_features = fourier_features
        self.scalars_in_tensor = scalars_in_tensor
        self.norm_spatial, self.metadata_tensor = Normalizer.read_normalization_info(self.dir / "norm_data.csv")
        self.norm_params, self.metadata_params = Normalizer.read_normalization_info(self.dir / "norm_params.csv")

        perf_path = self.dir / "norm_perf.csv"
        if perf_path.exists():
            self.norm_perf, self.metadata_perf = Normalizer.read_normalization_info(perf_path)
        else:
            self.norm_perf = empty_norm_info()
            self.metadata_perf = pd.DataFrame(columns=self.metadata_tensor.columns)

        if self.scalars_in_tensor:
            if not self.norm_perf["names"]:
                raise FileNotFoundError(f"Tensorized scalars require normalization metadata at {perf_path}")
            self.norm_tensor = concat_norm_info(self.norm_spatial, self.norm_params, self.norm_perf)
        else:
            self.norm_tensor = concat_norm_info(self.norm_spatial)

        if self.fourier_features:
            self.norm_fourier, self.metadata_fourier = Normalizer.read_normalization_info(self.dir / "norm_fourier.csv")
        else:
            self.norm_fourier = empty_norm_info()

    @staticmethod
    def read_normalization_info(path: Path|str) -> tuple[NormInfo, pd.DataFrame]:
        df = pd.read_csv(Path(path))
        mean = df["Mean"].to_numpy()
        std = df["Std"].to_numpy()
        log = df["Log"].to_numpy()
        names = {field: i for (i, field) in enumerate(df["Field"])}
        out: NormInfo = {"names": names, "mean": mean, "std": std, "log": log}
        return out, df
    
    def write_normalization_info(self, path: Path | str):
        path = Path(path)
        self.metadata_params.to_csv(path / "norm_params.csv", index=False)
        self.metadata_tensor.to_csv(path / "norm_data.csv", index=False)
        if self.norm_perf["names"]:
            self.metadata_perf.to_csv(path / "norm_perf.csv", index=False)
        if self.fourier_features:
            self.metadata_fourier.to_csv(path / "norm_fourier.csv", index=False)
    
    def find_name(self, name: str):
        if name in self.spatial_fields():
            norm = self.norm_spatial
        elif name in self.input_params():
            norm = self.norm_params
        elif name in self.performance_scalars():
            norm = self.norm_perf
        elif name in self.norm_fourier["names"]:
            norm = self.norm_fourier
        else:
            raise KeyError(f"{name} is not a valid field, parameter, or performance scalar in the dataset.")

        index = norm["names"][name]
        return index, norm

    def spatial_fields(self) -> dict:
        return self.norm_spatial["names"]

    def input_params(self) -> dict:
        return self.norm_params["names"]

    def performance_scalars(self) -> dict:
        return self.norm_perf["names"]

    def tensor_channels(self) -> dict:
        return self.norm_tensor["names"]

    # Compatibility for code that operates on the active model tensor.
    def fields(self) -> dict:
        return self.tensor_channels()

    def params(self) -> dict:
        return self.input_params()
    
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

    def normalize_stddev(self, stddev, name: str, reference=None):
        """Map a standard deviation in physical units into normalized units.

        For log-normalized quantities this uses first-order uncertainty
        propagation about ``reference``. A reference is therefore required and
        must be strictly positive for those quantities.
        """
        index, norm = self.find_name(name)
        scale = norm["std"][index]

        if not norm["log"][index]:
            return stddev / scale

        if reference is None:
            raise ValueError(f"A reference value is required to scale uncertainty for log-normalized field '{name}'.")

        mod = torch if isinstance(reference, torch.Tensor) else np
        if bool(mod.any(reference <= 0)):
            raise ValueError(f"Reference values must be positive for log-normalized field '{name}'.")
        return stddev / (reference * scale)

    def normalize_params(self, param_vec):
        normed = np.zeros_like(param_vec)
        for (name, i) in self.input_params().items():
            normed[i] = self.normalize(param_vec[i], name)
        return normed

    def denormalize_params(self, param_vec):
        denormed = np.zeros_like(param_vec)
        for (name, i) in self.input_params().items():
            denormed[i] = self.denormalize(param_vec[i], name)
        return denormed

    def normalize_tensor(self, tensor):
        normed = np.zeros_like(tensor)
        for (name, i) in self.tensor_channels().items():
            normed[:, i, :] = self.normalize(tensor[:, i, :], name)
        return normed

    def denormalize_tensor(self, tensor):
        denormed = np.zeros_like(tensor)
        for (name, i) in self.tensor_channels().items():
            denormed[:, i, :] = self.denormalize(tensor[:, i, :], name)
        return denormed

    def __eq__(self, other):
        if not isinstance(other, Normalizer):
            return NotImplemented
        return (
            self.metadata_params.equals(other.metadata_params)
            and self.metadata_tensor.equals(other.metadata_tensor)
            and self.metadata_perf.equals(other.metadata_perf)
        )
