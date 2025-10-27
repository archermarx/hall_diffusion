import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math

class ThrusterDataset(Dataset):
    def __init__(self, dir, subset_size: int | None = None, start_index: int = 0, dimension=1, files=None):
        super().__init__()
        self.dir = Path(dir)
        self.data_dir = self.dir / "data"
        self.files = os.listdir(self.data_dir)

        if files is not None:
            filter_files = set(files)
            self.files = [f for f in self.files if f in filter_files]
        elif subset_size is not None and subset_size > 0:
            self.files = self.files[start_index : (subset_size + start_index)]

        self.metadata_plasma = pd.read_csv(self.dir / "norm_data.csv")
        self.metadata_params = pd.read_csv(self.dir / "norm_params.csv")

        assert dimension == 1 or dimension == 2
        self.dimension = dimension
        self.fields = {field: i for (i, field) in enumerate(self.metadata_plasma["Field"])}
        self.params = {field: i for (i, field) in enumerate(self.metadata_params["Field"])}

    def params_to_vec(self, param_dict):
        df = self.metadata_params.set_index("Field").to_dict(orient="index")
        param_vec = np.zeros(len(df))

        for i, (key, val) in enumerate(df.items()):
            p = param_dict[key]
            if val["Log"]:
                p = np.log(p)

            param_vec[i] = (p - val["Mean"]) / val["Std"]

        return param_vec

    def vec_to_params(self, param_vec):
        df = self.metadata_params
        param_dict = {}
        for index, row in df.iterrows():
            mean = row["Mean"]
            std = row["Std"]
            is_log = row["Log"]
            p = param_vec[index] * std + mean
            assert isinstance(is_log, bool)
            if is_log:
                p = np.exp(p)
            param_dict[row["Field"]] = p

        return param_dict

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
        obs_locations=None,
    ):
        self.metadata = dataset.metadata_plasma
        self.xmax = 3.0

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
        field_map = {field: i for (i, field) in enumerate(self.metadata["Field"])}
        df_field = self.metadata.set_index("Field")

        mean = df_field["Mean"][field]
        std = df_field["Std"][field]

        ys = []

        for sim in self.sims:
            y = sim[field_map[field], :].numpy()
            if denormalize:
                mean = df_field["Mean"][field]
                std = df_field["Std"][field]
                y = y * std + mean

            ys.append(y)

        return ys

    def _plot_field(self, ax, field, denormalize=False, obs_locations=None):
        (_, w) = self.sims[0].shape

        if field == "Id" or field == "T":
            x = np.linspace(0.5, 1, w)
            ax.set_xlabel("Time [ms]")
        else:
            x = np.linspace(0, self.xmax, w)
            ax.set_xlabel("Axial location [channel lengths]")

        df_field = self.metadata.set_index("Field")
        log = df_field["Log"][field]

        ys = self.get_field(field, denormalize=denormalize)

        for i, y in enumerate(ys):
            if denormalize and log:
                y = np.exp(y)
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
        height = 3 * nrows

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
