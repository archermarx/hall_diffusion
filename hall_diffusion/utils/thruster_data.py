import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

if __name__ == "__main__":
    from normalization import Normalizer
else:
    from .normalization import Normalizer

def binned_psd(t, signal, n_bins=50, fmin=None, fmax=None, pow_min=1e-6):
    fs = 1 / np.mean(np.diff(t))

    freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
    psd = (np.abs(np.fft.rfft(signal)) ** 2) / (len(signal) * fs)

    freqs, psd = freqs[1:], psd[1:]  # drop DC

    fmin = fmin or freqs[0]
    fmax = fmax or freqs[-1]

    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs, psd = freqs[mask], psd[mask]

    bin_edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    bin_power = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i + 1])
        if mask.any():
            bin_power[i] = psd[mask].mean()

    if np.all(np.isnan(bin_power)):
        bin_power[:] = pow_min
    else:
        with np.errstate(divide="ignore"):
            # Interpolate NaNs in log-space
            valid = ~np.isnan(bin_power)
            bin_power[~valid] = np.exp(
                np.interp(np.log(bin_centers[~valid]), np.log(bin_centers[valid]), np.log(bin_power[valid]))
            )

    bin_power = np.maximum(pow_min, bin_power)

    return bin_centers, bin_power

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
        max_freqs=64,
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
        self.resolution = len(self.grid)

        # Frequencies to analyze in fourier spectrum
        self.fourier_features = fourier_features
        self.max_freqs = max_freqs
        self.min_freq = 5e3
        self.max_freq = 5e5
        # Minimum power spectral density
        self.min_pow = 1e-6
        # Factor used to normalize power spectra
        self.power_norm_factor = np.abs(np.log(self.min_pow))

    def write_metadata(self, path: Path | str):
        path = Path(path)
        self.norm.write_normalization_info(path)
        df_grid = pd.DataFrame({"z (m)": self.grid})
        df_grid.to_csv(path / "grid.csv", index=False)

    def fields(self):
        return self.norm.tensor_channels()

    def params(self):
        return self.norm.input_params()

    def spatial_fields(self):
        return self.norm.spatial_fields()

    def input_params(self):
        return self.norm.input_params()

    def performance_scalars(self):
        return self.norm.performance_scalars()

    def tensor_channels(self):
        return self.norm.tensor_channels()

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

    def _signal_to_vec(self, t, signal, truncate=True):
        if truncate:
            num_pts = len(t)
            t = t[num_pts // 2 :]
            signal = signal[num_pts // 2 :]

        mean = signal.mean()
        rms = torch.maximum(signal.std(), torch.tensor([1e-2]))
        signal_norm = (signal - mean) / rms
        rms_norm = rms / mean
        mean_norm = self.norm.normalize(mean, "discharge_current_A")

        _, bin_powers = binned_psd(
            t, signal_norm, n_bins=self.max_freqs, fmin=self.min_freq, fmax=self.max_freq, pow_min=self.min_pow
        )
        bin_powers = torch.tensor(bin_powers).log() / self.power_norm_factor

        return torch.concat([torch.tensor([mean_norm, rms_norm]), bin_powers])

    def __getitem__(self, idx):
        filename = self.data_dir / self.files[idx]
        data = np.load(filename)

        tensor = torch.tensor(data["data"], dtype=torch.float32)
        params = torch.tensor(data["params"], dtype=torch.float32)
        perf = None

        if self.scalars_in_tensor:
            resolution = tensor.shape[1]
            # Add params and performance quantitiesto the end of the tensor as constant channels
            perf = torch.tensor(data["perf"], dtype=torch.float32)
            param_tens = params.unsqueeze(1).expand(-1, resolution)
            perf_tens = perf.unsqueeze(1).expand(-1, resolution)

            assert param_tens.shape == (self.num_params, resolution)
            assert perf_tens.shape == (len(perf), resolution)

            tensor = torch.cat([tensor, param_tens, perf_tens], dim=0)
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
            if "fourier_amplitudes" in data:
                fourier = torch.tensor(data["fourier_amplitudes"], dtype=torch.float32)
            else:
                # Get fourier info and save to file
                time = torch.tensor(data["time"], dtype=torch.float32)
                t_vals, I_vals = time[:, 0], time[:, 2]
                fourier = self._signal_to_vec(t_vals, I_vals)
                np.savez(filename, **data, fourier_amplitudes=fourier)

            params = torch.concat((params, fourier))

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
