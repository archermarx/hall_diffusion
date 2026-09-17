import pickle

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from hall_diffusion.adapter_data import (
    ConditionDataset,
    HDF5ConditionBatchSampler,
    collate_condition_batch,
    preprocess_condition,
)
from hall_diffusion.sample import _batch_adapter_condition, _load_adapter_condition


UUIDS = [
    "00000000-0000-0000-0000-000000000001",
    "00000000-0000-0000-0000-000000000002",
    "00000000-0000-0000-0000-000000000003",
]


class BaseDataset:
    is_hdf5 = True
    hdf5_chunk_size = 2

    def __init__(self):
        self.record_ids = UUIDS

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, index):
        return self._sample(index)

    def __getitems__(self, indices):
        return [self._sample(index) for index in indices]

    def _sample(self, index):
        return self.record_ids[index], torch.tensor([index], dtype=torch.float32), torch.full((2, 4), index)


def write_conditions(path, identifiers=None):
    # Deliberately use a different physical row order from the base dataset.
    identifiers = identifiers or [UUIDS[2], UUIDS[0], UUIDS[1]]
    values_by_id = {UUIDS[0]: 10, UUIDS[1]: 20, UUIDS[2]: 30}
    values = np.stack([np.full((2, 3), values_by_id[record_id], dtype=np.uint16) for record_id in identifiers])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("trace_uuid", data=np.asarray(identifiers, dtype=h5py.string_dtype("utf-8")))
        handle.create_dataset("tlpp_counts", data=values, chunks=(2, 2, 3))


def test_hdf5_conditions_join_by_uuid_and_apply_explicit_preprocessing(tmp_path):
    path = tmp_path / "conditions.h5"
    write_conditions(path)
    dataset = ConditionDataset(
        BaseDataset(),
        {
            "type": "hdf5",
            "path": path,
            "data_key": "tlpp_counts",
            "id_key": "trace_uuid",
            "add_channel_dim": True,
            "add_occupancy_channel": True,
            "scale": 0.1,
        },
    )

    record_id, params, target, condition = dataset[1]

    assert record_id == UUIDS[1]
    torch.testing.assert_close(params, torch.tensor([1.0]))
    assert target.shape == (2, 4)
    assert condition.shape == (2, 2, 3)
    torch.testing.assert_close(condition[0], torch.full((2, 3), 2.0))
    torch.testing.assert_close(condition[1], torch.ones((2, 3)))
    assert dataset._h5 is not None


def test_hdf5_condition_batch_reads_restore_requested_order_and_work_with_workers(tmp_path):
    path = tmp_path / "conditions.h5"
    write_conditions(path)
    dataset = ConditionDataset(
        BaseDataset(),
        {"type": "hdf5", "path": path, "data_key": "tlpp_counts", "add_channel_dim": True},
    )

    samples = dataset.__getitems__([2, 0, 1])
    assert [sample[0] for sample in samples] == [UUIDS[2], UUIDS[0], UUIDS[1]]
    assert [sample[3][0, 0, 0].item() for sample in samples] == [30, 10, 20]

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._h5 is None
    record_ids, _, _, conditions = next(
        iter(DataLoader(restored, batch_size=2, num_workers=2, collate_fn=collate_condition_batch))
    )
    assert record_ids == UUIDS[:2]
    assert conditions.shape == (2, 1, 2, 3)
    assert conditions[:, 0, 0, 0].tolist() == [10, 20]


def test_hdf5_condition_sampler_follows_condition_storage_order(tmp_path):
    path = tmp_path / "conditions.h5"
    write_conditions(path)
    dataset = ConditionDataset(
        BaseDataset(),
        {"type": "hdf5", "path": path, "data_key": "tlpp_counts"},
    )

    sampler = HDF5ConditionBatchSampler(dataset, batch_size=2, shuffle=False)
    batches = list(sampler)

    assert batches == [[2, 0], [1]]
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2]


def test_hdf5_conditions_require_unique_complete_identifiers(tmp_path):
    duplicate_path = tmp_path / "duplicate.h5"
    write_conditions(duplicate_path, [UUIDS[0], UUIDS[0], UUIDS[2]])
    with pytest.raises(ValueError, match="duplicate identifiers"):
        ConditionDataset(
            BaseDataset(),
            {"type": "hdf5", "path": duplicate_path, "data_key": "tlpp_counts"},
        )

    missing_path = tmp_path / "missing.h5"
    write_conditions(missing_path, [UUIDS[0], UUIDS[2]])
    with pytest.raises(ValueError, match="missing 1 base identifier"):
        ConditionDataset(
            BaseDataset(),
            {"type": "hdf5", "path": missing_path, "data_key": "tlpp_counts"},
        )


def test_hdf5_condition_cache_is_written_in_base_uuid_order_and_reused(tmp_path):
    source_path = tmp_path / "conditions.h5"
    sorted_path = tmp_path / "conditions_sorted.h5"
    write_conditions(source_path)
    source = {
        "type": "hdf5",
        "path": source_path,
        "sorted_file": sorted_path,
        "data_key": "tlpp_counts",
        "sort_batch_size": 2,
    }

    dataset = ConditionDataset(BaseDataset(), source)

    assert dataset.path == sorted_path
    with h5py.File(sorted_path, "r") as handle:
        assert handle["trace_uuid"].asstr()[:].tolist() == UUIDS
        assert handle["tlpp_counts"][:, 0, 0].tolist() == [10, 20, 30]
        assert handle["tlpp_counts"].chunks == (2, 2, 3)
        assert handle.attrs["sorted_by"] == "trace_uuid"

    source_path.unlink()
    reused = ConditionDataset(BaseDataset(), source)
    assert reused.path == sorted_path
    assert reused._condition_rows == range(3)


def test_condition_dataset_only_accepts_hdf5_sources():
    with pytest.raises(ValueError, match="must be 'hdf5'"):
        ConditionDataset(BaseDataset(), {"type": "files"})


def test_sampling_loads_one_hdf5_condition_by_uuid(tmp_path):
    path = tmp_path / "conditions.h5"
    write_conditions(path)

    condition = _load_adapter_condition(
        {
            "condition_file": path,
            "condition_uuid": UUIDS[1],
            "condition_data_key": "tlpp_counts",
            "condition_add_channel_dim": True,
            "condition_add_occupancy_channel": True,
            "condition_scale": 0.1,
        },
        "cnn2d",
    )
    batch = _batch_adapter_condition(condition, 3, torch.device("cpu"))

    assert condition.shape == (2, 2, 3)
    torch.testing.assert_close(condition[0], torch.full((2, 3), 2.0))
    torch.testing.assert_close(condition[1], torch.ones((2, 3)))
    assert batch.shape == (3, 2, 2, 3)


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        ("sqrt", [0.0, 0.5, 0.75]),
        ("log1p", torch.log1p(torch.tensor([0.0, 4.0, 9.0])).mul(0.25).tolist()),
    ],
)
def test_condition_transforms_are_applied_before_scale(transform, expected):
    actual = preprocess_condition([0.0, 4.0, 9.0], transform=transform, scale=0.25)
    torch.testing.assert_close(actual, torch.tensor(expected))


def test_condition_transforms_validate_name_and_domain():
    with pytest.raises(ValueError, match="must be 'none', 'sqrt', or 'log1p'"):
        preprocess_condition([1.0], transform="log")
    with pytest.raises(ValueError, match="nonnegative"):
        preprocess_condition([-0.5], transform="sqrt")
    with pytest.raises(ValueError, match="greater than -1"):
        preprocess_condition([-1.0], transform="log1p")


def test_occupancy_channel_uses_raw_counts_and_is_not_scaled():
    raw = torch.tensor([[[0.0, 1.0], [4.0, 0.0]]])
    actual = preprocess_condition(raw, transform="sqrt", scale=0.25, add_occupancy_channel=True)

    expected_counts = torch.tensor([[0.0, 0.25], [0.5, 0.0]])
    expected_occupancy = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert actual.shape == (2, 2, 2)
    torch.testing.assert_close(actual[0], expected_counts)
    torch.testing.assert_close(actual[1], expected_occupancy)

    with pytest.raises(ValueError, match="exactly one input channel"):
        preprocess_condition(torch.ones(2, 2, 2), add_occupancy_channel=True)
    with pytest.raises(ValueError, match="nonnegative"):
        preprocess_condition(torch.tensor([[[-0.5]]]), add_occupancy_channel=True)
