import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import TypedDict

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
        self.metadata_params.to_csv(path/ "norm_params.csv")
        self.metadata_tensor.to_csv(path / "norm_data.csv")
    
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

        if log[index]:
            val = np.log(val)

        return (val - mean[index]) / std[index]
    
    def denormalize(self, val, name: str):
        index, norm = self.find_name(name)
        mean, std, log = norm["mean"], norm["std"], norm["log"]

        val = mean[index] + val * std[index]
        if log[index]:
            val = np.exp(val)

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

class ThrusterDataset(Dataset):
    def __init__(self, dir, subset_size: int | None = None, start_index: int = 0, files=None):
        super().__init__()
        self.dir = Path(dir)
        self.data_dir = self.dir / "data"
        self.files = os.listdir(self.data_dir)

        if files is not None:
            filter_files = set(files)
            self.files = [f for f in self.files if f in filter_files]
        elif subset_size is not None and subset_size > 0:
            self.files = self.files[start_index : (subset_size + start_index)]

        self.metadata_grid = pd.read_csv(self.dir / "grid.csv")
        self.grid = self.metadata_grid["z (m)"].to_numpy()
        self.norm = Normalizer(dir)
        self.num_fields = len(self.norm.norm_tensor["names"])
        self.num_params = len(self.norm.norm_params["names"])

    def write_metadata(self, path: Path | str):
        path = Path(path)
        self.norm.write_normalization_info(path)
        self.metadata_grid.to_csv(path / "grid.csv")

    def fields(self):
        return self.norm.fields()
    
    def params(self):
        return self.norm.params()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.data_dir / self.files[idx])

        tensor = data["data"]
        params = data["params"]

        return self.files[idx], params, tensor


class ThrusterPlotter1D:
    def __init__(
        self,
        dataset: ThrusterDataset,
        sims: list | None = None,
        labels: list | None = None,
        colors: list | None = None,
        alphas: list | None = None,
    ):
        self.norm = dataset.norm
        self.xmax = dataset.grid[-1]

        if sims is None:
            self.sims = []
        else:
            self.sims = sims

        if labels is None:
            self.labels = ["" for _ in self.sims]
        else:
            self.labels = labels

        assert len(self.labels) == len(self.sims)
        self.colors = colors
        self.alphas = alphas

    def add_sims(self, sims, label: str = ""):
        self.sims.append(sims)
        self.labels.append(label)

    def get_field(self, field, denormalize=False):
        if field == "inverse_hall":
            ys = []
            nu_an = self.get_field("nu_an", denormalize=denormalize)
            B = self.get_field("B", denormalize=denormalize)
            # nu_an and B are both stored as logs
            for _nu, _B in zip(nu_an, B):
                if denormalize:
                    wce = np.log(1.6e-19) + _B - np.log(9.1e-31)
                else:
                    wce = _B 
                
                ys.append(_nu - wce)

            return ys

        ys = []
        for sim in self.sims:
            y = sim[self.norm.fields()[field], :].numpy()
            ys.append(self.norm.denormalize(y, field))

        return ys

    def _plot_field(self, ax, field, denormalize=False, obs_locations=None):
        (_, w) = self.sims[0].shape

        if field == "Id" or field == "T":
            x = np.linspace(0.5, 1, w)
            ax.set_xlabel("Time [ms]")
        else:
            x = np.linspace(0, self.xmax, w)
            ax.set_xlabel("Axial location [channel lengths]")


        norm_tensor = self.norm.norm_tensor
        ind = norm_tensor["names"][field]

        if field == "inverse_hall":
            log = True
        else:
            log = norm_tensor["log"][ind]

        ys = self.get_field(field, denormalize=denormalize)

        for i, y in enumerate(ys):
            if denormalize and log:
                ax.set_yscale("log")

            if self.colors is not None and self.alphas is not None:
                ax.plot(x, y, color=self.colors[i], alpha=self.alphas[i])
                if obs_locations is not None and i == len(ys) - 1:
                    ax.scatter(x[obs_locations], y[obs_locations], color=self.colors[i], alpha=self.alphas[i], zorder=5)
            else:
                ax.plot(x, y)
                if obs_locations is not None and i == len(ys) - 1:
                    ax.scatter(x[obs_locations], y[obs_locations], zorder=5)

        ax.set_title(field)

    def plot(self, fields: str | list, denormalize=False, nrows=1, obs_fields=None, obs_locations=None):
        if not isinstance(fields, list):
            fields = [fields]

        ncols = math.ceil(len(fields) / nrows)

        width = 3 * ncols
        height = 2.8 * nrows

        fig = plt.figure(figsize=(width, height), constrained_layout=True)

        axes = []

        for i, field in enumerate(fields):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.margins(x=0)
            if obs_fields is not None and field in obs_fields:
                self._plot_field(ax, field, denormalize=denormalize, obs_locations=obs_locations)
            else:
                self._plot_field(ax, field, denormalize=denormalize)
                
            axes.append(ax)

        return fig, axes


if __name__ == "__main__":
    # Test loading and plotting
    dataset = ThrusterDataset("data/training", None, 1)
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    files, labels, sims = next(iter(loader))
    sims = [sims[i] for i in range(batch_size)]

    plotter = ThrusterPlotter1D(dataset, sims)
    fig, axes = plotter.plot(["nu_an", "ui_1", "ni_1", "E", "Tev", "nn"], denormalize=True, nrows=2)
    fig.savefig("test.png")
