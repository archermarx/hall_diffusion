"""Pair existing 1D diffusion samples with condition arrays by record ID."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ConditionDataset(Dataset):
    """Wrap a target dataset with condition files named by its record IDs."""

    def __init__(self, base, source: dict):
        self.base = base
        self.source = source
        self.kind = source.get("type", "files")
        self.key = source.get("key", "condition")
        if self.kind == "files":
            self.directory = Path(source["directory"])
            missing = [
                record_id for record_id in base.record_ids if not (self.directory / record_id).is_file()
            ]
            if missing:
                preview = ", ".join(missing[:3])
                raise FileNotFoundError(f"missing {len(missing)} condition file(s), including {preview}")
        elif self.kind != "params":
            raise ValueError("condition source type must be 'files' or 'params'")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        record_id, params, target = self.base[index]
        if self.kind == "params":
            return record_id, params, target, params
        path = self.directory / record_id
        with np.load(path, allow_pickle=False) as data:
            if self.key not in data:
                raise KeyError(f"condition file {path} does not contain key {self.key!r}")
            condition = torch.as_tensor(np.array(data[self.key]), dtype=torch.float32)
        if not torch.isfinite(condition).all():
            raise ValueError(f"condition file {path} contains non-finite values")
        return record_id, params, target, condition


def collate_condition_batch(batch):
    record_ids, params, targets, conditions = zip(*batch, strict=True)
    try:
        stacked_conditions = torch.stack(conditions)
    except RuntimeError as exc:
        raise ValueError("all conditions in a batch must share one shape") from exc
    return list(record_ids), torch.stack(params), torch.stack(targets), stacked_conditions
