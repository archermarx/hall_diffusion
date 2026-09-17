"""Pair existing 1D diffusion samples with condition arrays by record ID."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm


def preprocess_condition(
    value,
    transform: str = "none",
    scale: float = 1.0,
    add_occupancy_channel: bool = False,
) -> torch.Tensor:
    """Transform and scale one condition, optionally appending raw occupancy."""
    if transform not in {"none", "sqrt", "log1p"}:
        raise ValueError("condition transform must be 'none', 'sqrt', or 'log1p'")
    if not np.isfinite(scale):
        raise ValueError("condition scale must be finite")

    raw = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    if not torch.isfinite(raw).all():
        raise ValueError("raw condition contains non-finite values")
    condition = raw
    if transform == "sqrt":
        if torch.any(condition < 0):
            raise ValueError("sqrt condition transform requires nonnegative values")
        condition = torch.sqrt(condition)
    elif transform == "log1p":
        if torch.any(condition <= -1):
            raise ValueError("log1p condition transform requires values greater than -1")
        condition = torch.log1p(condition)
    if scale != 1.0:
        condition = condition * scale
    if not torch.isfinite(condition).all():
        raise ValueError("transformed condition contains non-finite values")
    if add_occupancy_channel:
        if raw.ndim < 2 or raw.shape[0] != 1:
            raise ValueError("occupancy channel requires a condition with exactly one input channel")
        if torch.any(raw < 0):
            raise ValueError("occupancy channel requires nonnegative values")
        occupancy = (raw > 0).to(condition.dtype)
        condition = torch.cat((condition, occupancy), dim=0)
    return condition


class HDF5ConditionBatchSampler(Sampler[list[int]]):
    """Batch logical samples according to their physical condition-file rows."""

    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False, generator=None):
        if dataset.kind != "hdf5":
            raise TypeError("HDF5ConditionBatchSampler requires HDF5 conditions")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        locality_size = dataset.hdf5_chunk_size or batch_size
        chunks_by_id = {}
        for logical_index, row in enumerate(dataset._condition_rows):
            chunks_by_id.setdefault(row // locality_size, []).append(logical_index)
        self._chunks = [
            sorted(indices, key=dataset._condition_rows.__getitem__)
            for _, indices in sorted(chunks_by_id.items())
        ]

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(len(self._chunks), generator=self.generator).tolist()
        else:
            order = range(len(self._chunks))
        batch = []
        for chunk_index in order:
            batch.extend(self._chunks[chunk_index])
            while len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class ConditionDataset(Dataset):
    """Wrap a target dataset with conditions joined to its record IDs.

    HDF5 condition files are indexed once by identifier and opened lazily in
    each DataLoader worker.  Their physical row order need not match the base
    dataset.
    """

    def __init__(self, base, source: dict):
        self.base = base
        self.source = source
        self.kind = source.get("type", "hdf5")
        if self.kind != "hdf5":
            raise ValueError("condition source type must be 'hdf5'")
        self.key = source.get("data_key", "condition")
        self.add_channel_dim = bool(source.get("add_channel_dim", False))
        self.add_occupancy_channel = bool(source.get("add_occupancy_channel", False))
        self.transform = source.get("transform", "none")
        self.scale = float(source.get("scale", 1.0))
        # Validate preprocessing eagerly, before a training worker is started.
        preprocess_condition([], self.transform, self.scale)

        self._h5 = None
        self._h5_pid = None
        self.path = Path(source["path"])
        self.id_key = source.get("id_key", "trace_uuid")
        if sorted_file := source.get("sorted_file"):
            self._use_or_create_sorted_file(
                Path(sorted_file),
                batch_size=int(source.get("sort_batch_size", 256)),
            )
        self._init_hdf5()

    def _inspect_hdf5(self, path):
        if not path.is_file():
            raise FileNotFoundError(f"condition HDF5 file does not exist: {path}")
        with h5py.File(path, "r") as handle:
            missing = [key for key in (self.key, self.id_key) if key not in handle]
            if missing:
                raise ValueError(f"condition HDF5 file {path} is missing: {', '.join(missing)}")
            values = handle[self.key]
            identifiers = handle[self.id_key]
            if values.ndim < 2:
                raise ValueError(
                    f"condition HDF5 dataset {self.key!r} must have a record dimension and at least one value dimension"
                )
            if identifiers.ndim != 1:
                raise ValueError(f"condition identifier dataset {self.id_key!r} must be one-dimensional")
            if values.shape[0] != identifiers.shape[0]:
                raise ValueError("condition values and identifiers have different record counts")
            if values.shape[0] == 0:
                raise ValueError("condition HDF5 file contains no records")
            condition_ids = identifiers.asstr()[:].tolist()
            chunk_size = values.chunks[0] if values.chunks is not None else None
        return condition_ids, chunk_size

    def _condition_rows_for(self, condition_ids):
        row_by_id = {}
        duplicates = []
        for row, record_id in enumerate(condition_ids):
            if record_id in row_by_id:
                duplicates.append(record_id)
            else:
                row_by_id[record_id] = row
        if duplicates:
            preview = ", ".join(duplicates[:3])
            raise ValueError(f"condition HDF5 file contains duplicate identifiers, including {preview}")

        missing = [record_id for record_id in self.base.record_ids if record_id not in row_by_id]
        if missing:
            preview = ", ".join(missing[:3])
            raise ValueError(f"condition HDF5 file is missing {len(missing)} base identifier(s), including {preview}")
        return [row_by_id[record_id] for record_id in self.base.record_ids]

    def _init_hdf5(self):
        condition_ids, self.hdf5_chunk_size = self._inspect_hdf5(self.path)
        if condition_ids == self.base.record_ids:
            self._condition_rows = range(len(condition_ids))
        else:
            self._condition_rows = self._condition_rows_for(condition_ids)

    def _use_or_create_sorted_file(self, sorted_path: Path, batch_size: int):
        """Create a condition-only HDF5 cache in base-record order once."""
        if sorted_path == self.path:
            raise ValueError("sorted condition file must differ from its source file")
        if batch_size <= 0:
            raise ValueError("condition sort_batch_size must be positive")
        if not self.base.is_hdf5:
            raise ValueError("sorted condition caches require an HDF5 base dataset")
        chunk_size = self.base.hdf5_chunk_size
        if sorted_path.is_file():
            self.path = sorted_path
            return
        if sorted_path.exists():
            raise ValueError(f"sorted condition path exists but is not a file: {sorted_path}")

        condition_ids, _ = self._inspect_hdf5(self.path)
        source_rows = self._condition_rows_for(condition_ids)
        sorted_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = sorted_path.with_name(f".{sorted_path.name}.{os.getpid()}.tmp")
        if temporary_path.exists():
            temporary_path.unlink()

        print(f"Creating UUID-sorted condition cache at {sorted_path}")
        try:
            with h5py.File(self.path, "r") as source, h5py.File(temporary_path, "w") as destination:
                source_values = source[self.key]
                record_count = len(self.base.record_ids)
                chunk_records = min(chunk_size, record_count)
                value_chunks = (chunk_records, *source_values.shape[1:])
                output_values = destination.create_dataset(
                    self.key,
                    shape=(record_count, *source_values.shape[1:]),
                    dtype=source_values.dtype,
                    chunks=value_chunks,
                    compression=source_values.compression,
                    compression_opts=source_values.compression_opts,
                    shuffle=source_values.shuffle,
                    fletcher32=source_values.fletcher32,
                )
                output_ids = destination.create_dataset(
                    self.id_key,
                    shape=(record_count,),
                    dtype=h5py.string_dtype("utf-8"),
                    chunks=(chunk_records,),
                )
                for name, value in source.attrs.items():
                    destination.attrs[name] = value
                for name, value in source_values.attrs.items():
                    output_values.attrs[name] = value
                destination.attrs["record_count"] = record_count
                destination.attrs["sorted_by"] = self.id_key

                starts = range(0, record_count, batch_size)
                total_batches = (record_count + batch_size - 1) // batch_size
                for start in tqdm(starts, total=total_batches, desc="sorting conditions"):
                    stop = min(start + batch_size, record_count)
                    rows = np.asarray(source_rows[start:stop])
                    order = np.argsort(rows)
                    sorted_rows = rows[order]
                    sorted_values = source_values[sorted_rows]
                    inverse = np.empty_like(order)
                    inverse[order] = np.arange(len(order))
                    output_values[start:stop] = sorted_values[inverse]
                    output_ids[start:stop] = self.base.record_ids[start:stop]
                destination.flush()
            if sorted_path.exists():
                temporary_path.unlink()
            else:
                os.replace(temporary_path, sorted_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        self.path = sorted_path

    def _hdf5_handle(self):
        pid = os.getpid()
        if self._h5 is not None and self._h5_pid != pid:
            self._h5.close()
            self._h5 = None
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
            self._h5_pid = pid
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_h5", None)
        if handle is not None:
            handle.close()

    def __len__(self):
        return len(self.base)

    def raw_condition(self, index):
        """Return one untransformed condition record for diagnostics."""
        row = self._condition_rows[int(index)]
        return np.asarray(self._hdf5_handle()[self.key][row])

    def _format_condition(self, value, record_id):
        array = np.asarray(value)
        if self.add_channel_dim:
            array = np.expand_dims(array, axis=0)
        try:
            return preprocess_condition(
                array,
                self.transform,
                self.scale,
                add_occupancy_channel=self.add_occupancy_channel,
            )
        except ValueError as exc:
            raise ValueError(f"condition for {record_id}: {exc}") from exc

    def _read_hdf5_conditions(self, indices, record_ids):
        """Read arbitrary rows in contiguous runs and restore requested order."""
        rows = [self._condition_rows[int(index)] for index in indices]
        sorted_positions = sorted(range(len(rows)), key=rows.__getitem__)
        conditions = [None] * len(rows)
        values = self._hdf5_handle()[self.key]
        run_start = 0
        while run_start < len(sorted_positions):
            run_end = run_start + 1
            first_row = rows[sorted_positions[run_start]]
            last_row = first_row
            while run_end < len(sorted_positions):
                next_row = rows[sorted_positions[run_end]]
                if next_row != last_row + 1:
                    break
                last_row = next_row
                run_end += 1

            block = values[first_row : last_row + 1]
            for offset, sorted_position in enumerate(sorted_positions[run_start:run_end]):
                conditions[sorted_position] = self._format_condition(
                    block[offset], record_ids[sorted_position]
                )
            run_start = run_end
        return conditions

    def __getitem__(self, index):
        record_id, params, target = self.base[index]
        value = self._hdf5_handle()[self.key][self._condition_rows[index]]
        condition = self._format_condition(value, record_id)
        return record_id, params, target, condition

    def __getitems__(self, indices):
        base_getitems = getattr(self.base, "__getitems__", None)
        samples = base_getitems(indices) if base_getitems is not None else [self.base[index] for index in indices]
        record_ids = [sample[0] for sample in samples]
        conditions = self._read_hdf5_conditions(indices, record_ids)
        return [(*sample, condition) for sample, condition in zip(samples, conditions, strict=True)]


def collate_condition_batch(batch):
    record_ids, params, targets, conditions = zip(*batch, strict=True)
    try:
        stacked_conditions = torch.stack(conditions)
    except RuntimeError as exc:
        raise ValueError("all conditions in a batch must share one shape") from exc
    return list(record_ids), torch.stack(params), torch.stack(targets), stacked_conditions
