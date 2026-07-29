from __future__ import annotations

import pytest
import torch

from hall_diffusion.utils import utils


@pytest.fixture(autouse=True)
def mock_cuda_description(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "test CUDA device")


def test_auto_prefers_cuda_before_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    assert utils.get_device("auto").type == "cuda"


@pytest.mark.parametrize("name", ["cpu", "mps", "cuda", "xpu"])
def test_explicit_available_device_is_selected(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: name == "mps")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: name == "cuda")
    monkeypatch.setattr(torch.xpu, "is_available", lambda: name == "xpu")
    assert utils.get_device(name).type == name


@pytest.mark.parametrize("name", ["mps", "cuda", "xpu"])
def test_explicit_unavailable_device_has_clear_error(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match=rf"{name}.*not available"):
        utils.get_device(name)


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="auto.*cpu.*mps.*cuda.*xpu"):
        utils.get_device("tpu")


def test_auto_always_returns_a_supported_device() -> None:
    assert utils.get_device("auto").type in {"cpu", "mps", "cuda", "xpu"}
