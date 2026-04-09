from dataclasses import field
from attrs import field
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import scipy.stats

from .normalization import Normalizer

class ThrusterDataset(Dataset):
    def __init__(self, dir, subset_size: int | None = None, start_index: int = 0, scalars_in_tensor=False, files=None, downsample_res=None):
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
        self.downsample_res = downsample_res

        if downsample_res is not None:
            self.grid = np.linspace(self.grid[0], self.grid[-1], downsample_res)

        self.dx = self.grid[2] - self.grid[1]
        self.norm = Normalizer(dir)
        self.num_fields = len(self.norm.norm_tensor["names"])
        self.num_params = len(self.norm.norm_params["names"])
        self.scalars_in_tensor = scalars_in_tensor
        self.resolution = len(self.grid)

        if self.scalars_in_tensor:
            self.num_fields += self.num_params

    def write_metadata(self, path: Path | str):
        path = Path(path)
        self.norm.write_normalization_info(path)
        self.metadata_grid.to_csv(path / "grid.csv", index=False)

    def fields(self):
        return self.norm.fields()
    
    def params(self):
        return self.norm.params()
    
    def get_field(self, tens, name, action=None):
        row = tens[:, self.fields()[name], :]
        if action == "normalize":
            return self.norm.normalize(row, name)
        elif action == "denormalize":
            return self.norm.denormalize(row, name)
        elif action is None:
            return row
        else:
            raise NameError(f"Action '{name}' not allowed. Action must be 'normalize', 'denormalize' or `None`.")
        
    def get_denorm(self, tens, name):
        return self.get_field(tens, name, action="denormalize")
        
    def get_param(self, p, name, action=None):
        param = p[self.params()[name]]
        if action == "normalize":
            return self.norm.normalize(param, name)
        elif action == "denormalize":
            return self.norm.denormalize(param, name)
        elif action is None:
            return param
        else:
            raise NameError(f"Action '{name}' not allowed. Action must be 'normalize', 'denormalize' or `None`.")

    def sample_params(self, num_samples, device):
        param_vec_inds = random.choices(range(len(self)), k=num_samples)
        param_vecs = torch.tensor(np.array([self[i][1] for i in param_vec_inds]), device=device)
        return param_vecs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.data_dir / self.files[idx])

        tensor = torch.tensor(data["data"], dtype=torch.float32)
        params = torch.tensor(data["params"], dtype=torch.float32)

        if self.scalars_in_tensor:
            # Add params to the end of the tensor as constant channels
            p = params.unsqueeze(1).expand(-1, tensor.shape[1])
            assert p.shape == (self.num_params, tensor.shape[1])
            tensor = torch.cat([tensor, p], dim=0)
            params = torch.tensor([])

        if self.downsample_res is not None:
            tensor = tensor.unsqueeze(0) # add batch dimension for interpolation
            tensor = torch.nn.functional.interpolate(tensor, size=self.downsample_res, mode="linear", align_corners=True)
            tensor = tensor.squeeze(0) # remove batch dimension

        return self.files[idx], params, tensor

    @staticmethod
    def generate_measurements(sim: torch.Tensor):
        """Mask whole and partial fields for conditioning, and add noise to unmasked pixels. Returns a precision mask and the resulting data tensor after masking and adding noise."""

        def sample_mask(lo, hi):
            return round(scipy.stats.beta(a=5, b=1).rvs() * (hi - lo) + lo)
        
        def sample_std(size=None):
            return scipy.stats.beta(a=1, b=5).rvs(size) * 0.25

        num_fields_to_mask = sample_mask(1, dataset.num_fields-1)
        field_inds_to_mask = set(np.random.choice(dataset.num_fields, size=num_fields_to_mask, replace=False))
        precision = torch.ones(dataset.num_fields, dataset.resolution, dtype=torch.float32)
        masked_sim = torch.randn(sim.shape) * 0.5
        num_fields = len(dataset.fields().keys())

        for field_index in range(dataset.num_fields):
            if field_index in field_inds_to_mask:
                # Unobserved fields are set to zero precision.
                precision[field_index, :] = 0.0
            elif field_index >= num_fields:
                # Parameters are masked uniformly in space and given a single uniform noise level.
                std = sample_std()
                masked_sim[field_index] = sim[field_index] + np.random.standard_normal() * std
                precision[field_index, :] = 1 / std**2
            else:
                # Remove at least half of the pixels, but potentially up to all but one pixel
                # The beta distribution biases toward removing more pixels.
                num_pixels_to_remove = sample_mask(dataset.resolution // 2, dataset.resolution - 1)
                num_pixels_left = dataset.resolution - num_pixels_to_remove

                # Remove the decided number of pixels at random locations, creating a mask of which pixels were removed
                pixels_to_remove = np.random.choice(dataset.resolution, size=num_pixels_to_remove, replace=False)
                removal_mask = torch.zeros(dataset.resolution, dtype=torch.bool) 
                removal_mask[pixels_to_remove] = True

                # Set precision to 0 for removed pixels.
                # The beta distribution biases toward smaller std (higher precision) but allows for some variability.
                # The maximum allowed standard deviation is 0.25, as higher values would be unrealistic.
                measurement_std = torch.tensor(sample_std(size=num_pixels_left), dtype=torch.float32)
                precision[field_index, removal_mask] = 0.0
                precision[field_index, ~removal_mask] = 1.0 / measurement_std**2
                
                # Add noise to the remaining pixels based on the sampled standard deviation
                masked_sim[field_index, ~removal_mask] = sim[field_index, ~removal_mask] + torch.randn((num_pixels_left,), dtype=torch.float32) * measurement_std

        return torch.log(1+precision), masked_sim



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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default="data/training")
    args = parser.parse_args()

    dataset = ThrusterDataset(args.data_dir, None, 1, scalars_in_tensor=True)
    batch_size = 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    files, labels, sims = next(iter(loader))
    
    names = [k for k in dataset.fields().keys()] + [k for k in dataset.params().keys()]
    param_names = dataset.params().keys()
    print(param_names)

    sim = sims[0]
    precision, measurements = ThrusterDataset.generate_measurements(sim)

    sim_min, sim_max = -1, 1
    sim_normalized = (sim - sim_min) / (sim_max - sim_min)
    sim_rgb = sim_normalized.unsqueeze(2).repeat(1, 1, 3)
    # Set masked pixels to magenta
    sim_rgb[precision==0, :] = torch.tensor([1.0, 0.0, 1.0]) # Magenta color for masked pixels

    fig, axs = plt.subplots(1, 3, figsize=(16,6), layout='constrained')
    axs[0].imshow(sim_rgb, aspect='auto')
    axs[1].imshow(precision, aspect='auto', cmap='gray')

    # Remove green channel for masked pixels
    masked_sim_rgb = measurements.unsqueeze(2).repeat(1, 1, 3)
    masked_sim_rgb[precision==0, 1] = 0.0 # Set green channel to 0 for masked pixels
    axs[2].imshow(masked_sim_rgb, aspect='auto', cmap='gray')

    for (i, ax) in enumerate(axs):
        ax.set(yticks = range(len(names)), yticklabels=names if i == 0 else [], xlabel = "Axial index")
        ax.set_box_aspect(1)

    axs[0].set_title(f"Sim with masked pixels in magenta")
    axs[1].set_title(f"Measurement precision")
    axs[2].set_title(f"Resulting data tensor")

    # Need to add colorbar to right of second plot
    cb = fig.colorbar(axs[1].images[0], ax=axs[1], location='right', shrink=0.5)
    cb.set_label('log(1 / $\\sigma^2$)')

    plt.show()

    # Make 1D plots of each field showing original tensor and simulated data
    for name, index in dataset.fields().items():
        kept_index = precision[index] != 0
        if sum(kept_index) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8,4))
        x = dataset.grid
        ax.plot(x, sim[index].numpy(), label="Original sim")

        precision_kept = precision[index, kept_index]
        x_kept = x[kept_index]
        sim_kept = measurements[index, kept_index].numpy()
        error_bars = 2*torch.sqrt(1 / torch.exp(precision_kept)).numpy()
        ax.scatter(x_kept, sim_kept, label="Simulated data with noise", color="orange", s=20, zorder=5)
        ax.errorbar(x_kept, sim_kept, yerr=error_bars, fmt='none', ecolor='orange', alpha=0.5, zorder=4)
        ax.set_title(name)
        ax.set_xlabel("Axial index")
        ax.legend()
        plt.show()
