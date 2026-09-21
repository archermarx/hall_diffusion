import math

import torch

from hall_diffusion.models.adapter_io import load_adapter, save_adapter
from hall_diffusion.models.conditioning import (
    ConditionAdapter,
    ConditionedEDM2,
    TLPPVAEConditionEncoder,
    build_condition_encoder,
)
from hall_diffusion.models.edm2 import EDM2Denoiser
from hall_diffusion.models.tlpp_vae import TLPPVAEEncoder, preprocess_tlpp_counts


def write_vae_checkpoint(path):
    torch.manual_seed(11)
    source = TLPPVAEEncoder()
    torch.save(
        {
            "sample_shape": (1, 128, 128),
            "vae": {"latent_dim": 128, "probability_mode": "log01"},
            "model_state": {
                **{f"net.{name}": value for name, value in source.state_dict().items()},
                "net.logvar.weight": torch.zeros(128, 64 * 16 * 16),
            },
        },
        path,
    )
    return source


def make_base():
    return EDM2Denoiser(
        resolution=16,
        in_channels=2,
        condition_dim=0,
        base_channels=8,
        channel_mult=[1, 2],
        num_blocks=1,
        attn_resolutions=[],
        channels_per_head=8,
    ).eval()


def base_config():
    return {
        "architecture": "edm2",
        "resolution": 16,
        "in_channels": 2,
        "condition_dim": 0,
        "base_channels": 8,
        "channel_mult": [1, 2],
        "num_blocks": 1,
        "attn_resolutions": [],
        "channels_per_head": 8,
    }


def encoder_config(checkpoint, freeze_vae=True):
    return {
        "type": "tlpp_vae",
        "checkpoint": str(checkpoint),
        "freeze_vae": freeze_vae,
        "token_dim": 16,
        "num_tokens": 4,
        "hidden_dim": 16,
        "depth": 1,
    }


def test_tlpp_preprocessing_matches_probability_log01_reference():
    counts = torch.zeros(2, 1, 128, 128)
    counts[0, 0, 4, 5] = 1
    counts[0, 0, 8, 9] = 3

    actual = preprocess_tlpp_counts(counts)
    expected = torch.zeros_like(counts)
    probability = torch.tensor([0.25, 0.75])
    expected[0, 0, 4, 5], expected[0, 0, 8, 9] = (
        torch.log1p(probability / 1e-6) / math.log1p(1e6)
    )

    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[1]) == 0


def test_tlpp_encoder_loads_checkpoint_and_supports_both_training_modes(tmp_path):
    checkpoint = tmp_path / "last.pt"
    source = write_vae_checkpoint(checkpoint)
    counts = torch.randint(0, 5, (2, 1, 128, 128), dtype=torch.int32)

    frozen = build_condition_encoder(encoder_config(checkpoint, freeze_vae=True))
    trainable = build_condition_encoder(encoder_config(checkpoint, freeze_vae=False))

    assert isinstance(frozen, TLPPVAEConditionEncoder)
    expected_latent = source(preprocess_tlpp_counts(counts))
    torch.testing.assert_close(frozen.encode_latent(counts), expected_latent)
    assert frozen(counts).tokens.shape == (2, 4, 16)

    frozen(counts).tokens.square().mean().backward()
    assert all(parameter.grad is None for parameter in frozen.vae.parameters())
    assert any(parameter.grad is not None for parameter in frozen.bridge.parameters())

    trainable(counts).tokens.square().mean().backward()
    assert any(parameter.grad is not None for parameter in trainable.vae.parameters())
    assert any(parameter.grad is not None for parameter in trainable.bridge.parameters())

    frozen.train()
    assert not frozen.vae.training

    base = make_base()
    conditioned = ConditionedEDM2(base, {"tlpp": ConditionAdapter(base, frozen, 8)})
    trainable_ids = {id(parameter) for parameter in conditioned.get_trainable_params()}
    assert trainable_ids.isdisjoint(id(parameter) for parameter in frozen.vae.parameters())
    assert trainable_ids.issuperset(id(parameter) for parameter in frozen.bridge.parameters())


def test_adapter_artifact_does_not_need_source_vae_checkpoint(tmp_path):
    checkpoint = tmp_path / "last.pt"
    write_vae_checkpoint(checkpoint)
    config = encoder_config(checkpoint)
    base = make_base()
    adapter = ConditionAdapter(base, build_condition_encoder(config), channels_per_head=8)
    counts = torch.randint(0, 5, (1, 1, 128, 128), dtype=torch.int32)
    expected = adapter.encode_condition(counts).tokens

    artifact_path = tmp_path / "adapter.pth.tar"
    save_adapter(
        artifact_path,
        name="tlpp",
        adapter=adapter,
        adapter_config={"encoder": config, "channels_per_head": 8},
        base_model_config=base_config(),
    )
    checkpoint.unlink()

    name, loaded, _ = load_adapter(artifact_path, make_base(), base_config(), weights="model")
    assert name == "tlpp"
    torch.testing.assert_close(loaded.encode_condition(counts).tokens, expected)
