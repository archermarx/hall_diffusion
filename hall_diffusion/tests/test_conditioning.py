import torch
from torch import nn

from hall_diffusion.models.adapter_io import load_adapter, save_adapter
from hall_diffusion.models.conditioning import (
    ConditionAdapter,
    ConditionedEDM2,
    Conv1dConditionEncoder,
    Conv2dConditionEncoder,
    MLPConditionEncoder,
)
from hall_diffusion.models.edm2 import EDM2Denoiser


def make_base():
    base = EDM2Denoiser(
        resolution=16, in_channels=2, condition_dim=0, base_channels=8,
        channel_mult=[1, 2], num_blocks=1, attn_resolutions=[], channels_per_head=8,
    ).eval()
    # EDM2 starts with a zero output gain; make this test sensitive to adapter
    # injection instead of obtaining equality trivially from initialization.
    with torch.no_grad():
        base.unet.out_gain.fill_(1)
    return base


def test_new_adapter_is_an_exact_noop_and_base_stays_eval():
    torch.manual_seed(1)
    base = make_base()
    adapter = ConditionAdapter(base, MLPConditionEncoder(3, token_dim=8), channels_per_head=8)
    model = ConditionedEDM2(base, {"scalar": adapter})
    x = torch.randn(2, 2, 16)
    sigma = torch.full((2, 1, 1), 0.5)
    expected = base(x, sigma)
    actual = model(x, sigma, conditions={"scalar": torch.randn(2, 3)})
    assert torch.allclose(actual, expected, atol=1e-6)
    model.train()
    assert not model.base.training


def test_adapter_can_change_output_without_changing_base_weights():
    torch.manual_seed(2)
    base = make_base()
    adapter = ConditionAdapter(base, MLPConditionEncoder(3, token_dim=8), channels_per_head=8)
    model = ConditionedEDM2(base, {"scalar": adapter})
    before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=1e-2)
    x = torch.randn(2, 2, 16)
    sigma = torch.full((2, 1, 1), 0.5)
    output = model(x, sigma, conditions={"scalar": torch.randn(2, 3)})
    output.square().mean().backward()
    optimizer.step()
    assert all(torch.equal(before[name], value) for name, value in base.state_dict().items())
    changed = model(x, sigma, conditions={"scalar": torch.randn(2, 3)})
    assert not torch.allclose(output, changed)


def test_all_builtin_encoders_return_usable_tokens_for_unrelated_shapes():
    one_d = Conv1dConditionEncoder(2, token_dim=12, channels=[8, 12])
    two_d = Conv2dConditionEncoder(1, token_dim=12, channels=[8, 12])
    mlp = MLPConditionEncoder(5, token_dim=12, num_tokens=3)
    assert one_d(torch.randn(2, 2, 31)).tokens.shape == (2, 16, 12)
    assert two_d(torch.randn(2, 1, 37, 53)).tokens.shape == (2, 19 * 27, 12)
    assert mlp(torch.randn(2, 5)).tokens.shape == (2, 3, 12)
    assert any(isinstance(module, nn.GroupNorm) for module in two_d.modules())


def test_cnn_encoder_supports_configurable_residual_depth():
    encoder = Conv1dConditionEncoder(2, token_dim=8, channels=[8, 16], blocks_per_stage=3)
    assert len(encoder.stages) == 2
    assert all(len(stage) == 3 for stage in encoder.stages)
    assert encoder(torch.randn(2, 2, 17)).tokens.shape == (2, 9, 8)


def test_adapter_artifact_transfers_to_a_different_base_with_the_same_architecture(tmp_path):
    base = make_base()
    adapter = ConditionAdapter(base, MLPConditionEncoder(3, token_dim=8), channels_per_head=8)
    base_config = {
        "architecture": "edm2", "resolution": 16, "in_channels": 2, "condition_dim": 0,
        "base_channels": 8, "channel_mult": [1, 2], "num_blocks": 1,
        "attn_resolutions": [], "channels_per_head": 8,
    }
    path = tmp_path / "adapter.pth.tar"
    save_adapter(
        path, name="scalar", adapter=adapter,
        adapter_config={"encoder": {"type": "mlp", "input_dim": 3, "token_dim": 8}, "channels_per_head": 8},
        base_model_config=base_config,
        condition_config={
            "data_key": "features",
            "id_key": "trace_uuid",
            "add_channel_dim": False,
            "add_occupancy_channel": False,
            "transform": "sqrt",
            "scale": 0.5,
        },
        training_state={"batch_idx": 12, "example_idx": 48, "epoch_idx": 2},
    )
    name, loaded, artifact = load_adapter(path, base, base_config, weights="model")
    assert name == "scalar"
    assert isinstance(loaded.encoder, MLPConditionEncoder)
    assert artifact["condition_config"]["data_key"] == "features"
    assert artifact["condition_config"]["transform"] == "sqrt"
    assert artifact["condition_config"]["scale"] == 0.5
    assert artifact["training_state"] == {"batch_idx": 12, "example_idx": 48, "epoch_idx": 2}
    other_base = make_base()
    name, loaded, _ = load_adapter(path, other_base, base_config)
    assert name == "scalar"
    assert isinstance(loaded.encoder, MLPConditionEncoder)

    incompatible_config = {**base_config, "in_channels": 3}
    try:
        load_adapter(path, other_base, incompatible_config)
    except ValueError as exc:
        assert "architecture does not match" in str(exc)
    else:
        raise AssertionError("architecture mismatch should fail")
