from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script", ["hall_diffusion/sample.py", "hall_diffusion/train.py"]
)
def test_cli_documents_device_selection(script: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--device" in result.stdout
    for choice in ("auto", "cpu", "mps", "cuda", "xpu"):
        assert choice in result.stdout


def _string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_values(item)


def test_shipped_tomls_contain_no_personal_absolute_paths() -> None:
    offenders = []
    for path in REPO_ROOT.rglob("*.toml"):
        with path.open("rb") as stream:
            values = list(_string_values(tomllib.load(stream)))
        for value in values:
            if value.startswith(("/Users/", "/home/")) or re.match(
                r"^[A-Za-z]:[\\/]", value
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {value}")
    assert offenders == []


@pytest.mark.parametrize(
    "generator",
    [
        "experimental_methods/perez_luna/perez_luna.py",
        "experimental_methods/roberts/roberts.py",
    ],
)
def test_observation_generators_do_not_serialize_absolute_paths(
    generator: str,
) -> None:
    source = (REPO_ROOT / generator).read_text()
    assert ".absolute()" not in source
