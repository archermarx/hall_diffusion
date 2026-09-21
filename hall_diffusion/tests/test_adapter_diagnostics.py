import numpy as np
import pandas as pd
import pytest
import torch

from hall_diffusion.train_adapter import _amp_enabled, _condition_source, _interval_due, _progress_postfix
from hall_diffusion.utils.visualization import plot_condition_diagnostic, plot_training_progress


def test_batch_intervals_match_training_config_semantics():
    assert not _interval_due(99, 100)
    assert _interval_due(100, 100)
    assert not _interval_due(100, -1)
    with pytest.raises(ValueError, match="positive or -1"):
        _interval_due(1, 0)


def test_adapter_amp_is_cuda_only():
    assert _amp_enabled(torch.device("cuda"), True)
    assert not _amp_enabled(torch.device("cuda"), False)
    assert not _amp_enabled(torch.device("cpu"), True)


def test_adapter_progress_metrics_have_fixed_width():
    rendered = [
        _progress_postfix(1.0, 12345.0, float("nan")),
        _progress_postfix(1e-12, -0.25, 987654321.0),
    ]

    assert len(rendered[0]) == len(rendered[1])
    assert rendered[0] == "loss= 1.000e+00, grad= 1.234e+04, val=       nan"


def test_condition_source_accepts_prebuilt_sorted_file_without_unsorted_source():
    source = {
        "type": "hdf5",
        "train_sorted_file": "train_sorted.h5",
        "test_sorted_file": "test_sorted.h5",
        "data_key": "tlpp_counts",
    }

    resolved = _condition_source(source, "train")

    assert resolved["path"] == "train_sorted.h5"
    assert "sorted_file" not in resolved
    assert resolved["data_key"] == "tlpp_counts"


def test_condition_source_still_builds_or_reuses_sorted_cache_when_both_paths_are_given():
    source = {
        "type": "hdf5",
        "train_file": "train.h5",
        "train_sorted_file": "train_sorted.h5",
    }

    resolved = _condition_source(source, "train")

    assert resolved["path"] == "train.h5"
    assert resolved["sorted_file"] == "train_sorted.h5"


def test_condition_source_requires_at_least_one_split_path():
    with pytest.raises(ValueError, match="train_file.*train_sorted_file"):
        _condition_source({"type": "hdf5"}, "train")


def test_adapter_training_progress_plot_accepts_event_rows(tmp_path):
    log_path = tmp_path / "training.csv"
    rows = [
        {
            "event": "train",
            "example_idx": 4,
            "batch_idx": 1,
            "epoch_idx": 0,
            "train_loss": 2.0,
            "val_loss": np.nan,
            "ema_loss": np.nan,
            "grad_norm": 3.0,
            "learning_rate": 3e-4,
        },
        {
            "event": "train",
            "example_idx": 8,
            "batch_idx": 2,
            "epoch_idx": 0,
            "train_loss": 1.5,
            "val_loss": np.nan,
            "ema_loss": np.nan,
            "grad_norm": 2.5,
            "learning_rate": 3e-4,
        },
        {
            "event": "validation",
            "example_idx": 8,
            "batch_idx": 2,
            "epoch_idx": 0,
            "train_loss": 1.75,
            "val_loss": 1.25,
            "ema_loss": 1.2,
            "grad_norm": 2.75,
            "learning_rate": 3e-4,
        },
    ]
    pd.DataFrame(rows).to_csv(log_path, index=False)

    plot_training_progress(log_path, tmp_path, evaluation_iters=2, outlier_inds=[])

    assert (tmp_path / "loss_prog.png").stat().st_size > 0


def test_tlpp_condition_diagnostic_plot(tmp_path):
    counts = np.zeros((6, 8, 8), dtype=np.uint16)
    for index in range(len(counts)):
        counts[index, index, index + 1] = index + 1
    one_time_trace = np.linspace(0, 2e-3, 101)
    time_s = np.repeat(one_time_trace[None, :], len(counts), axis=0)
    current = np.stack(
        [2.0 + np.sin(2 * np.pi * (10_000 + index * 100) * one_time_trace) for index in range(6)]
    )

    plot_condition_diagnostic(
        counts,
        time_s,
        current,
        current_range=(-5.0, 105.0),
        record_ids=[f"example-uuid-{index}" for index in range(6)],
        title="Epoch: 0001, Loss: 1.0",
        folder=tmp_path,
    )

    assert (tmp_path / "condition_diagnostic.png").stat().st_size > 0
