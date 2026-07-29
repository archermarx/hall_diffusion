"""Shared setup for the explicitly-invoked portability suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]

# Current upstream mixes package imports (hall_diffusion.*) and script-style
# imports (models, utils, loss). Supporting both paths here lets this suite
# test portability without repairing that backend-independent import defect.
for path in (REPO_ROOT, REPO_ROOT / "hall_diffusion"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "mps: requires a real, available Apple MPS backend; never falls back to CPU",
    )


@pytest.fixture
def real_mps_device() -> torch.device:
    if not torch.backends.mps.is_built():
        pytest.skip("This PyTorch build has no MPS support")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available in this process")
    return torch.device("mps")
