import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

if __name__ == "__main__":
    from normalization import Normalizer
else:
    from .normalization import Normalizer


class ThrusterDataset(Dataset):
    def __init__(
        self,
        dir,
        subset_size: int | None = None,
        start_index: int = 0,
        scalars_in_tensor=False,
        fourier_features=False,
        files=None,
        downsample_res=None,
    ):
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
        self.norm = Normalizer(dir, scalars_in_tensor, fourier_features)
        self.num_fields = len(self.norm.norm_tensor["names"])
        self.num_params = len(self.norm.norm_params["names"])
        self.scalars_in_tensor = scalars_in_tensor
        self.fourier_features = fourier_features
        self.resolution = len(self.grid)

    def write_metadata(self, path: Path | str):
        path = Path(path)
        self.norm.write_normalization_info(path)
        df_grid = pd.DataFrame({"z (m)": self.grid})
        df_grid.to_csv(path / "grid.csv", index=False)

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
            raise NameError(f"Action '{action}' not allowed. Action must be 'normalize', 'denormalize' or `None`.")

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
            tensor = tensor.unsqueeze(0)  # add batch dimension for interpolation
            tensor = torch.nn.functional.interpolate(
                tensor, size=self.downsample_res, mode="linear", align_corners=True
            )
            tensor = tensor.squeeze(0)  # remove batch dimension
        else:
            # Should be 128 (TODO: fix this hardcode)
            if tensor.shape[1] == 130:
                tensor = tensor[:, 1:-1]
            
            assert tensor.shape[1] == 128

        if self.fourier_features:
            max_features = 64
            fourier_tensor = torch.tensor(data["fourier"], dtype=torch.float32)
            perf_tensor = torch.tensor(data["perf"], dtype=torch.float32)

            _, num_fourier_channels = fourier_tensor.shape
            fourier_tensor = fourier_tensor[:max_features, :].reshape((max_features * num_fourier_channels,))

            discharge_current, thrust = perf_tensor
            # Append mean discharge current + fourier features (freq, real ampl, imag ampl) to param vec
            params = torch.concat([params, torch.tensor([discharge_current]), fourier_tensor])

        return self.files[idx], params, tensor

    def generate_measurements(self, sim: torch.Tensor, sqrt_weight=True):
        """Mask whole and partial fields for conditioning, and add noise to unmasked pixels.

        Args:
            sim: (B, num_fields, resolution) batch of simulations.

        Returns:
            condition_tensor: (B, 2*num_fields, resolution) log-precision and noisy/masked simulation.
        """
        B = sim.shape[0]
        num_spatial = len(self.fields())
        num_params = self.num_fields - num_spatial
        res = self.resolution
        dev = sim.device

        # Beta(5,1) ~ U^0.2  and  Beta(1,5) ~ 1 - U^0.2  (inverse-CDF trick for Beta(a,1)/Beta(1,b))
        # This keeps all sampling on-device with no CPU transfers.
        def beta_hi(shape):  # Beta(5,1): biased toward 1
            return torch.rand(shape, device=dev) ** 0.2

        def beta_lo(shape):  # Beta(1,5): biased toward 0
            return 1.0 - torch.rand(shape, device=dev) ** 0.2

        # --- Field masking ---
        # Sample a mask probability per sample; Beta(5,1) biases toward masking more fields.
        p_mask = beta_hi((B,))  # (B,)
        field_masked = torch.rand(B, self.num_fields, device=dev) < p_mask[:, None]  # (B, num_fields)

        # --- Initialize outputs (masked/removed entries keep random noise) ---
        precision = torch.zeros(B, self.num_fields, res, device=dev)
        masked_sim = torch.randn(B, self.num_fields, res, device=dev) * 0.5

        # --- Spatial fields ---
        sf_mask = field_masked[:, :num_spatial]  # (B, num_spatial)

        # Sample a removal probability per (b, f); Beta(5,1) biases toward removing more pixels.
        p_remove = beta_hi((B, num_spatial))  # (B, num_spatial)
        pixel_kept = torch.rand(B, num_spatial, res, device=dev) >= p_remove[:, :, None]  # (B, num_spatial, res)

        std_min = 1e-4
        std_max = 1e0
        # Sample std deviations uniformly in log space
        log_stds_s = torch.rand(B, num_spatial, res, device=dev) * (np.log(std_max) - np.log(std_min)) + np.log(std_min)

        stds_s = torch.exp(log_stds_s)  # (B, num_spatial)
        noise_s = torch.randn(B, num_spatial, res, device=dev) * stds_s

        active_s = (~sf_mask[:, :, None]) & pixel_kept  # unmasked field + kept pixel
        masked_sim[:, :num_spatial] = torch.where(active_s, sim[:, :num_spatial] + noise_s, masked_sim[:, :num_spatial])
        precision[:, :num_spatial] = torch.where(active_s, (1.0 / stds_s) ** 2, torch.zeros_like(stds_s))

        # --- Parameter fields (constant across space; single noise draw per field) ---
        if num_params > 0:
            pf_mask = field_masked[:, num_spatial:]  # (B, num_params)

            log_stds_p = torch.rand(B, num_params, device=dev) * (np.log(std_max) - np.log(std_min)) + np.log(std_min)
            stds_p = torch.exp(log_stds_p)  # (B, num_params)
            noise_p = torch.randn(B, num_params, device=dev) * stds_p

            active_p = ~pf_mask  # (B, num_params)
            masked_sim[:, num_spatial:] = torch.where(
                active_p[:, :, None],
                sim[:, num_spatial:] + noise_p[:, :, None],
                masked_sim[:, num_spatial:],
            )
            precision[:, num_spatial:] = torch.where(
                active_p[:, :, None], ((1.0 / stds_p) ** 2)[:, :, None], torch.zeros_like(precision[:, num_spatial:])
            )

        observed = precision > 0
        frac_observed = observed.float().mean(dim=[1, 2])  # (B,)
        log_precision = torch.log(precision + 1)

        if sqrt_weight:
            loss_weight_observed = torch.sqrt(1 + log_precision) / frac_observed[:, None, None]
        else:
            loss_weight_observed = (1 + log_precision) / frac_observed[:, None, None]

        loss_weight_unobserved = 1 / (1 - frac_observed[:, None, None] + 1e-8)
        loss_weight = torch.where(observed, loss_weight_observed, loss_weight_unobserved)

        condition_tensor = torch.cat([log_precision, observed, masked_sim], dim=1)
        return condition_tensor, loss_weight


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
            ax.set_xlabel("Time (ms)")
        else:
            x = np.linspace(0, self.xmax, w)
            ax.set_xlabel("Axial location (m)")

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
    batch_size = 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    files, labels, sims = next(iter(loader))

    names = [k for k in dataset.fields().keys()] + [k for k in dataset.params().keys()]
    param_names = dataset.params().keys()
    print(param_names)

    condition_tensor, _ = dataset.generate_measurements(sims)
    precision = condition_tensor[:, : dataset.num_fields, :]
    measurements = condition_tensor[:, dataset.num_fields :, :]

    sim = sims[0]
    precision = precision[0]
    measurements = measurements[0]

    sim_min, sim_max = -1, 1
    sim_normalized = (sim - sim_min) / (sim_max - sim_min)
    sim_rgb = sim_normalized.unsqueeze(2).repeat(1, 1, 3)
    # Set masked pixels to magenta
    sim_rgb[precision == 0, :] = torch.tensor([1.0, 0.0, 1.0])  # Magenta color for masked pixels

    fig, axs = plt.subplots(1, 3, figsize=(16, 6), layout="constrained")
    axs[0].imshow(sim_rgb, aspect="auto")
    axs[1].imshow(precision, aspect="auto", cmap="gray")

    # Remove green channel for masked pixels
    masked_sim_rgb = measurements.unsqueeze(2).repeat(1, 1, 3)
    masked_sim_rgb[precision == 0, 1] = 0.0  # Set green channel to 0 for masked pixels
    axs[2].imshow(masked_sim_rgb, aspect="auto", cmap="gray")

    for i, ax in enumerate(axs):
        ax.set(yticks=range(len(names)), yticklabels=names if i == 0 else [], xlabel="Axial index")
        ax.set_box_aspect(1)

    axs[0].set_title("Sim with masked pixels in magenta")
    axs[1].set_title("Measurement precision")
    axs[2].set_title("Resulting data tensor")

    # Need to add colorbar to right of second plot
    cb = fig.colorbar(axs[1].images[0], ax=axs[1], location="right", shrink=0.5)
    cb.set_label("log(1 / $\\sigma^2$)")

    plt.show()

    # Make 1D plots of each field showing original tensor and simulated data
    for name, index in dataset.fields().items():
        kept_index = precision[index] != 0
        if sum(kept_index) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        x = dataset.grid
        ax.plot(x, sim[index].numpy(), label="Original sim")

        precision_kept = precision[index, kept_index]
        x_kept = x[kept_index]
        sim_kept = measurements[index, kept_index].numpy()
        error_bars = 2 * torch.sqrt(1 / torch.exp(precision_kept)).numpy()
        ax.scatter(x_kept, sim_kept, label="Simulated data with noise", color="orange", s=20, zorder=5)
        ax.errorbar(x_kept, sim_kept, yerr=error_bars, fmt="none", ecolor="orange", alpha=0.5, zorder=4)
        ax.set_title(name)
        ax.set_xlabel("Axial index")
        ax.legend()
        plt.show()
