import numpy as np
import pandas as pd
import pytest

from hall_diffusion.train_adapter import _interval_due
from hall_diffusion.utils.visualization import plot_condition_diagnostic, plot_training_progress


def test_batch_intervals_match_training_config_semantics():
    assert not _interval_due(99, 100)
    assert _interval_due(100, 100)
    assert not _interval_due(100, -1)
    with pytest.raises(ValueError, match="positive or -1"):
        _interval_due(1, 0)


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
    counts = np.zeros((8, 8), dtype=np.uint16)
    counts[2, 3] = 10
    counts[5, 6] = 2
    time_s = np.linspace(0, 2e-3, 101)
    current = 2.0 + np.sin(2 * np.pi * 10_000 * time_s)

    plot_condition_diagnostic(
        counts,
        time_s,
        current,
        current_range=(-5.0, 105.0),
        record_id="example-uuid",
        title="Epoch: 0001, Loss: 1.0",
        folder=tmp_path,
    )

    assert (tmp_path / "condition_diagnostic.png").stat().st_size > 0
