from __future__ import annotations

import inspect

import pytest
import torch

import train


@pytest.mark.parametrize(
    ("device", "requested", "expected"),
    [
        ("cuda", True, True),
        ("cuda", False, False),
        ("mps", True, False),
        ("cpu", True, False),
        ("xpu", True, False),
    ],
)
def test_amp_is_enabled_only_for_cuda(
    device: str, requested: bool, expected: bool
) -> None:
    assert train.amp_enabled(torch.device(device), requested) is expected


def test_existing_pinned_memory_path_is_guarded_by_cuda() -> None:
    # Upstream already has the desired backend guard. Keep a regression test
    # without forcing a source refactor solely for testability.
    source = inspect.getsource(train.train)
    assert 'pin = DEVICE.type == "cuda"' in source


def test_grad_scaler_is_constructed_for_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    sentinel = object()

    def fake_scaler(device: str, *, enabled: bool):
        calls.append((device, enabled))
        return sentinel

    monkeypatch.setattr(torch.amp, "GradScaler", fake_scaler)
    assert train.create_grad_scaler(torch.device("cuda"), enabled=True) is sentinel
    assert calls == [("cuda", True)]


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu"])
def test_grad_scaler_is_not_constructed_off_cuda(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("GradScaler must not be constructed off CUDA")

    monkeypatch.setattr(torch.amp, "GradScaler", forbidden)
    assert train.create_grad_scaler(torch.device(device), enabled=True) is None
