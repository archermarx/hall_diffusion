import torch

from hall_diffusion.utils.utils import snapshot_state_dict


def test_snapshot_state_dict_is_an_independent_cpu_copy():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.BatchNorm1d(3),
    )
    snapshot = snapshot_state_dict(model)
    original = {name: value.clone() for name, value in model.state_dict().items()}

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
        for buffer in model.buffers():
            buffer.add_(1)

    assert snapshot._metadata == model.state_dict()._metadata
    for name, value in snapshot.items():
        assert value.device.type == "cpu"
        assert value.data_ptr() != model.state_dict()[name].data_ptr()
        torch.testing.assert_close(value, original[name])
