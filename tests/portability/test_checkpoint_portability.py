from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from hall_diffusion.utils import utils


def _tensor_devices(value: Any) -> set[str]:
    if isinstance(value, torch.Tensor):
        return {value.device.type}
    if isinstance(value, dict):
        return set().union(*(_tensor_devices(item) for item in value.values()), set())
    if isinstance(value, (list, tuple)):
        return set().union(*(_tensor_devices(item) for item in value), set())
    return set()


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu"])
def test_non_cuda_checkpoint_load_maps_storage_to_cpu(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    call: dict[str, Any] = {}

    def fake_load(path: Path, **kwargs: Any) -> dict[str, str]:
        call["path"] = path
        call.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(torch, "load", fake_load)
    result = utils.load_checkpoint(Path("model.pth.tar"), torch.device(device))

    assert result == {"status": "ok"}
    assert call["map_location"] == "cpu"
    assert call["weights_only"] is False


def test_cuda_checkpoint_load_preserves_upstream_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call: dict[str, Any] = {}

    def fake_load(path: Path, **kwargs: Any) -> dict[str, str]:
        call.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(torch, "load", fake_load)
    result = utils.load_checkpoint(Path("model.pth.tar"), torch.device("cuda"))

    assert result == {"status": "ok"}
    assert "map_location" not in call
    assert call["weights_only"] is False


def test_cpu_saved_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cpu-checkpoint.pth.tar"
    torch.save({"model": {"weight": torch.ones(2)}}, path)
    loaded = utils.load_checkpoint(path, torch.device("cpu"))
    assert _tensor_devices(loaded) == {"cpu"}


@pytest.mark.mps
def test_real_cuda_saved_checkpoint_loads_for_mps(real_mps_device: torch.device) -> None:
    checkpoint = os.environ.get("HALL_DIFFUSION_TEST_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set HALL_DIFFUSION_TEST_CHECKPOINT to the CUDA-saved checkpoint")

    loaded = utils.load_checkpoint(Path(checkpoint), real_mps_device)
    assert _tensor_devices(loaded) <= {"cpu"}

    # Loading the selected state into an MPS module proves CPU-first storage
    # can subsequently cross the backend boundary.
    state = loaded["ema"]
    first = next(iter(state.values()))
    probe = torch.nn.Parameter(torch.empty_like(first, device=real_mps_device))
    probe.data.copy_(first)
    assert probe.device.type == "mps"
