import tomllib
import warnings
from pathlib import Path

import pandas as pd
import pytest
import torch

from hall_diffusion.sample import build_observation
from hall_diffusion.utils.normalization import Normalizer


ROOT = Path(__file__).parents[2]
REFERENCE = ROOT / "mcmc_reference/ref_3charge/normalized"


class ReferenceMetadataDataset:
    scalars_in_tensor = False

    def __init__(self):
        self.norm = Normalizer(REFERENCE)
        self.grid = torch.as_tensor(pd.read_csv(REFERENCE / "grid.csv")["z (m)"].to_numpy())
        self.tensor = torch.zeros(len(self.norm.spatial_fields()), len(self.grid))
        self.param_values = torch.zeros(len(self.norm.input_params()))

    def __getitem__(self, index):
        return None, self.param_values, self.tensor

    def spatial_fields(self):
        return self.norm.spatial_fields()

    def input_params(self):
        return self.norm.input_params()

    def performance_scalars(self):
        return self.norm.performance_scalars()

    def tensor_channels(self):
        return self.norm.tensor_channels()


@pytest.mark.parametrize(
    "relative_path",
    [
        "experimental_methods/perez_luna/observation.toml",
        "experimental_methods/perez_luna/observation_withE.toml",
        "experimental_methods/roberts/observation.toml",
        "experimental_methods/roberts/observation_lif_only.toml",
    ],
)
def test_checked_in_observation_files_use_valid_unified_measurements(relative_path):
    with open(ROOT / relative_path, "rb") as file:
        observation = tomllib.load(file)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        build_observation(ReferenceMetadataDataset(), observation, num_samples=1)


@pytest.mark.parametrize("relative_path", ["configs/sample_mcmc.toml", "configs/sample_mcmc_forward.toml"])
def test_checked_in_inline_observations_use_valid_unified_measurements(relative_path):
    with open(ROOT / relative_path, "rb") as file:
        observation = tomllib.load(file)["observation"]

    build_observation(ReferenceMetadataDataset(), observation, num_samples=1)
