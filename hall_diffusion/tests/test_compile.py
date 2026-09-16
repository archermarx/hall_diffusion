from unittest.mock import patch

import torch

from hall_diffusion.utils.utils import compile_model


def test_compile_model_returns_original_when_disabled():
    model = torch.nn.Linear(2, 1)

    with patch("hall_diffusion.utils.utils.torch.compile") as compile_fn:
        result = compile_model(model, enabled=False)

    assert result is model
    compile_fn.assert_not_called()


def test_compile_model_compiles_when_enabled():
    model = torch.nn.Linear(2, 1)
    compiled = object()

    with patch("hall_diffusion.utils.utils.torch.compile", return_value=compiled) as compile_fn:
        result = compile_model(model, enabled=True)

    assert result is compiled
    compile_fn.assert_called_once_with(model)
