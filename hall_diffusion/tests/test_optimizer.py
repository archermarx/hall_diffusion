from unittest.mock import patch

import torch

from hall_diffusion.utils.utils import create_adamw


def test_adamw_keeps_pytorch_default_implementation_off_cuda():
    parameter = torch.nn.Parameter(torch.ones(1))

    optimizer = create_adamw([parameter], torch.device("cpu"), lr=1e-3)

    assert optimizer.defaults["fused"] is None
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()


def test_adamw_requests_fused_implementation_on_cuda():
    parameter = torch.nn.Parameter(torch.ones(1))
    sentinel = object()

    with patch("torch.optim.AdamW", return_value=sentinel) as adamw:
        result = create_adamw([parameter], torch.device("cuda"), lr=1e-3)

    assert result is sentinel
    assert adamw.call_args.kwargs["fused"] is True


def test_adamw_state_remains_compatible_with_existing_checkpoints():
    old_parameter = torch.nn.Parameter(torch.ones(1))
    old_optimizer = torch.optim.AdamW([old_parameter], lr=1e-3)
    old_parameter.grad = torch.ones_like(old_parameter)
    old_optimizer.step()

    new_parameter = torch.nn.Parameter(torch.ones(1))
    new_optimizer = create_adamw([new_parameter], torch.device("cpu"), lr=1e-3)
    new_optimizer.load_state_dict(old_optimizer.state_dict())

    assert new_optimizer.state[new_parameter]["step"].item() == 1
