from pathlib import Path

import torch

from hall_diffusion import sample


def test_variance_data_prefers_explicit_path(tmp_path):
    explicit = tmp_path / "explicit"
    checkpoint_data = tmp_path / "checkpoint"
    unconditional = tmp_path / "unconditional"
    for path in (explicit, checkpoint_data, unconditional):
        path.mkdir()

    result = sample._variance_data_dir(
        {
            "process_variance_data_dir": explicit,
            "unconditional_data_dir": unconditional,
        },
        {"directories": {"test_data_dir": checkpoint_data}},
    )

    assert result == explicit


def test_variance_data_falls_back_to_unconditional_data(tmp_path):
    unconditional = tmp_path / "unconditional"
    unconditional.mkdir()

    result = sample._variance_data_dir(
        {"unconditional_data_dir": unconditional},
        {"directories": {"test_data_dir": tmp_path / "missing"}},
    )

    assert result == unconditional


def test_process_variance_is_generated_only_on_the_first_request(tmp_path, monkeypatch):
    variance_file = tmp_path / "model" / "process_variance.npz"
    calls = []

    def generate(path, *args):
        calls.append(Path(path))
        Path(path).parent.mkdir()
        Path(path).touch()

    monkeypatch.setattr(sample, "_generate_process_variance", generate)
    arguments = ({}, {}, torch.nn.Identity(), {}, torch.device("cpu"))

    first = sample._ensure_process_variance(variance_file, *arguments)
    second = sample._ensure_process_variance(variance_file, *arguments)

    assert first == second == variance_file
    assert calls == [variance_file]
