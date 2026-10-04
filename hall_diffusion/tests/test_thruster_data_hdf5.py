import multiprocessing
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from hall_diffusion.utils.thruster_data import HDF5ChunkBatchSampler, ThrusterDataset

UUIDS = [
    "00000000-0000-0000-0000-000000000001",
    "00000000-0000-0000-0000-000000000002",
    "00000000-0000-0000-0000-000000000003",
]


def write_dataset(path):
    records = 3
    resolution = 128
    fields = np.arange(records * 2 * resolution, dtype=np.float32).reshape(records, 2, resolution)
    parameters = np.array([[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]], dtype=np.float32)
    performance = np.array([[3.0], [4.0], [5.0]], dtype=np.float32)
    time = np.arange(records * 4 * 3, dtype=np.float32).reshape(records, 4, 3)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=fields, chunks=(2, 2, resolution))
        handle.create_dataset("parameters", data=parameters)
        handle.create_dataset("performance", data=performance)
        handle.create_dataset("time", data=time)
        handle.create_dataset("grid", data=np.linspace(0.0, 1.0, resolution))
        handle.create_dataset("UUID", data=np.asarray(UUIDS, dtype=h5py.string_dtype("utf-8")))
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
        handle.create_dataset(
            "time_names",
            data=np.array(["time_s", "discharge_current_A", "thrust_mN"], dtype="S"),
        )
    return fields, parameters, performance, time


def write_normalization(path, name):
    pd.DataFrame({"Field": [name], "Mean": [0.0], "Std": [1.0], "Log": [False]}).to_csv(
        path, index=False
    )


def test_directory_dataset_returns_saved_time_tensor(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"z (m)": np.linspace(0.0, 1.0, 128)}).to_csv(tmp_path / "grid.csv", index=False)
    write_normalization(tmp_path / "norm_data.csv", "density")
    write_normalization(tmp_path / "norm_params.csv", "voltage")
    write_normalization(tmp_path / "norm_perf.csv", "thrust")

    time = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.savez(
        data_dir / "sample.npz",
        data=np.zeros((1, 128), dtype=np.float32),
        params=np.array([1.0], dtype=np.float32),
        perf=np.array([2.0], dtype=np.float32),
        time=time,
    )

    dataset = ThrusterDataset(tmp_path)
    _, _, _, time_tensor = dataset[0]

    torch.testing.assert_close(time_tensor, torch.from_numpy(time))
    with pytest.raises(ValueError, match="only available for HDF5"):
        dataset.times()


def test_hdf5_dataset_reads_records_and_metadata_lazily(tmp_path):
    path = tmp_path / "training.h5"
    fields, parameters, _, time = write_dataset(path)

    dataset = ThrusterDataset(path, subset_size=1, start_index=1)

    assert len(dataset) == 1
    assert dataset._h5 is None
    assert dataset.record_ids == [UUIDS[1]]
    assert dataset.fields() == {"density": 0, "potential": 1}
    assert dataset.params() == {"voltage": 0, "propellant": 1}
    assert dataset.times() == {"time_s": 0, "discharge_current_A": 1, "thrust_mN": 2}
    assert dataset.time_variables() == dataset.times()
    assert dataset.norm.norm_spatial["log"].tolist() == [True, False]
    record_uuid, params, tensor, time_tensor = dataset[0]
    assert record_uuid == UUIDS[1]
    torch.testing.assert_close(params, torch.from_numpy(parameters[1]))
    torch.testing.assert_close(tensor, torch.from_numpy(fields[1]))
    torch.testing.assert_close(time_tensor, torch.from_numpy(time[1]))
    torch.testing.assert_close(dataset.get_time(time_tensor, "time_s"), torch.from_numpy(time[1, :, 0]))
    batched_time = torch.from_numpy(time[:2])
    torch.testing.assert_close(
        dataset.get_time(batched_time, "discharge_current_A"), batched_time[..., 1]
    )
    assert dataset._h5 is not None


def test_hdf5_dataset_can_append_scalar_channels_and_filter_uuids(tmp_path):
    path = tmp_path / "training.hdf5"
    fields, parameters, performance, time = write_dataset(path)

    dataset = ThrusterDataset(path, uuids=[UUIDS[2]], scalars_in_tensor=True)
    record_uuid, params, tensor, time_tensor = dataset[0]

    assert record_uuid == UUIDS[2]
    assert params.numel() == 0
    assert tensor.shape == (5, 128)
    torch.testing.assert_close(tensor[:2], torch.from_numpy(fields[2]))
    torch.testing.assert_close(tensor[2], torch.full((128,), parameters[2, 0]))
    torch.testing.assert_close(tensor[3], torch.full((128,), parameters[2, 1]))
    torch.testing.assert_close(tensor[4], torch.full((128,), performance[2, 0]))
    torch.testing.assert_close(time_tensor, torch.from_numpy(time[2]))
    assert dataset.tensor_channels() == {
        "density": 0,
        "potential": 1,
        "voltage": 2,
        "propellant": 3,
        "thrust": 4,
    }


def test_hdf5_dataset_is_worker_pickle_safe(tmp_path):
    path = tmp_path / "training.h5"
    write_dataset(path)

    dataset = ThrusterDataset(path, downsample_res=64)
    restored = pickle.loads(pickle.dumps(dataset))
    record_uuid, params, tensor, _ = restored[0]

    assert record_uuid == UUIDS[0]
    assert params.shape == (2,)
    assert tensor.shape == (2, 64)


def test_hdf5_dataset_loads_with_training_workers(tmp_path):
    path = tmp_path / "training.h5"
    write_dataset(path)
    dataset = ThrusterDataset(path)
    dataset[0]  # Open a handle in the parent before workers are created.

    worker_context = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else None
    record_uuids, parameters, fields, times = next(
        iter(
            DataLoader(
                dataset,
                batch_size=2,
                num_workers=2,
                multiprocessing_context=worker_context,
            )
        )
    )

    assert record_uuids == tuple(UUIDS[:2])
    assert parameters.shape == (2, 2)
    assert fields.shape == (2, 2, 128)
    assert times.shape == (2, 4, 3)


def test_hdf5_batch_reads_preserve_requested_order(tmp_path):
    path = tmp_path / "training.h5"
    fields, parameters, _, time = write_dataset(path)
    dataset = ThrusterDataset(path)

    samples = dataset.__getitems__([2, 0, 1])

    assert [sample[0] for sample in samples] == [UUIDS[2], UUIDS[0], UUIDS[1]]
    for sample, expected_params, expected_fields in zip(
        samples, parameters[[2, 0, 1]], fields[[2, 0, 1]], strict=True
    ):
        torch.testing.assert_close(sample[1], torch.from_numpy(expected_params))
        torch.testing.assert_close(sample[2], torch.from_numpy(expected_fields))
    for sample, expected_time in zip(samples, time[[2, 0, 1]], strict=True):
        torch.testing.assert_close(sample[3], torch.from_numpy(expected_time))


def test_hdf5_chunk_sampler_keeps_each_storage_chunk_together(tmp_path):
    path = tmp_path / "training.h5"
    write_dataset(path)
    dataset = ThrusterDataset(path)
    sampler = HDF5ChunkBatchSampler(dataset, batch_size=2, generator=torch.Generator().manual_seed(4))

    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert sorted(flattened) == list(range(len(dataset)))
    assert len(batches) == 2
    assert abs(flattened.index(0) - flattened.index(1)) == 1


def test_hdf5_dataset_rejects_an_incomplete_schema(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=np.zeros((1, 2, 128)))

    with pytest.raises(ValueError, match="missing"):
        ThrusterDataset(path)
