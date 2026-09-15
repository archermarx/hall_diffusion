import pickle

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from hall_diffusion.utils.thruster_data import ThrusterDataset


def write_dataset(path):
    records = 3
    resolution = 128
    fields = np.arange(records * 2 * resolution, dtype=np.float32).reshape(records, 2, resolution)
    parameters = np.array([[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]], dtype=np.float32)
    performance = np.array([[3.0], [4.0], [5.0]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=fields)
        handle.create_dataset("parameters", data=parameters)
        handle.create_dataset("performance", data=performance)
        handle.create_dataset("grid", data=np.linspace(0.0, 1.0, resolution))
        handle.create_dataset("source_files", data=np.array(["one.npz", "two.npz", "three.npz"], dtype="S"))
        handle.create_dataset("field_names", data=np.array(["density", "potential"], dtype="S"))
        handle.create_dataset("field_means", data=[10.0, 20.0])
        handle.create_dataset("field_stds", data=[2.0, 4.0])
        handle.create_dataset("field_log_transformed", data=[True, False])
        handle.create_dataset("parameter_names", data=np.array(["voltage", "propellant"], dtype="S"))
        handle.create_dataset("parameter_means", data=[300.0, 0.0])
        handle.create_dataset("parameter_stds", data=[50.0, 1.0])
        handle.create_dataset("parameter_log_transformed", data=[False, False])
        handle.create_dataset("performance_names", data=np.array(["thrust"], dtype="S"))
        handle.create_dataset("performance_means", data=[0.2])
        handle.create_dataset("performance_stds", data=[0.1])
    return fields, parameters, performance


def test_hdf5_dataset_reads_records_and_metadata_lazily(tmp_path):
    path = tmp_path / "training.h5"
    fields, parameters, _ = write_dataset(path)

    dataset = ThrusterDataset(path, subset_size=1, start_index=1)

    assert len(dataset) == 1
    assert dataset._h5 is None
    assert dataset.fields() == {"density": 0, "potential": 1}
    assert dataset.params() == {"voltage": 0, "propellant": 1}
    assert dataset.norm.norm_spatial["log"].tolist() == [True, False]
    filename, params, tensor = dataset[0]
    assert filename == "two.npz"
    torch.testing.assert_close(params, torch.from_numpy(parameters[1]))
    torch.testing.assert_close(tensor, torch.from_numpy(fields[1]))
    assert dataset._h5 is not None


def test_hdf5_dataset_can_append_scalar_channels_and_filter_sources(tmp_path):
    path = tmp_path / "training.hdf5"
    fields, parameters, performance = write_dataset(path)

    dataset = ThrusterDataset(path, files=["three.npz"], scalars_in_tensor=True)
    filename, params, tensor = dataset[0]

    assert filename == "three.npz"
    assert params.numel() == 0
    assert tensor.shape == (5, 128)
    torch.testing.assert_close(tensor[:2], torch.from_numpy(fields[2]))
    torch.testing.assert_close(tensor[2], torch.full((128,), parameters[2, 0]))
    torch.testing.assert_close(tensor[3], torch.full((128,), parameters[2, 1]))
    torch.testing.assert_close(tensor[4], torch.full((128,), performance[2, 0]))
    assert dataset.tensor_channels() == {
        "density": 0,
        "potential": 1,
        "voltage": 2,
        "propellant": 3,
        "thrust": 4,
    }


def test_hdf5_dataset_is_worker_pickle_safe_and_ignores_fourier_features(tmp_path):
    path = tmp_path / "training.h5"
    write_dataset(path)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        dataset = ThrusterDataset(path, fourier_features=True, downsample_res=64)
    restored = pickle.loads(pickle.dumps(dataset))
    filename, params, tensor = restored[0]

    assert filename == "one.npz"
    assert params.shape == (2,)
    assert tensor.shape == (2, 64)
    assert restored.fourier_features is False


def test_hdf5_dataset_loads_with_training_workers(tmp_path):
    path = tmp_path / "training.h5"
    write_dataset(path)
    dataset = ThrusterDataset(path)
    dataset[0]  # Open a handle in the parent before workers are created.

    filenames, parameters, fields = next(iter(DataLoader(dataset, batch_size=2, num_workers=2)))

    assert filenames == ("one.npz", "two.npz")
    assert parameters.shape == (2, 2)
    assert fields.shape == (2, 2, 128)


def test_hdf5_dataset_rejects_an_incomplete_schema(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=np.zeros((1, 2, 128)))

    with pytest.raises(ValueError, match="missing"):
        ThrusterDataset(path)
