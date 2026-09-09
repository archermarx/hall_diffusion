from hall_diffusion import models
from hall_diffusion.configuration import EDM2_DEFAULTS, LOSS_DEFAULTS, resolve_config, resolve_model_config

import torch


def test_dataset_settings_are_inferred_from_checkpoint_config():
    config = {
        "scalars_in_tensor": True,
        "fourier_features": True,
        "downsample_res": 64,
        "condition_dim": 0,
    }
    settings = models.dataset_settings(config)

    assert settings == {
        "scalars_in_tensor": True,
        "fourier_features": True,
        "downsample_res": 64,
    }
    assert config["fourier_features"] is True


def test_legacy_checkpoint_infers_tensorized_scalars_from_zero_condition_dimension():
    settings = models.dataset_settings({"condition_dim": 0, "resolution": 128})

    assert settings["scalars_in_tensor"] is True
    assert settings["downsample_res"] == 128


def test_model_config_populates_defaults_without_overwriting_explicit_values():
    original = {
        "resolution": 16,
        "in_channels": 2,
        "condition_dim": 3,
        "channels_per_head": 8,
    }

    resolved = resolve_model_config(original)

    assert resolved["channels_per_head"] == 8
    assert resolved["base_channels"] == EDM2_DEFAULTS["base_channels"]
    assert resolved["fourier_features"] is False
    assert resolved["scalars_in_tensor"] is False
    assert resolved["downsample_res"] == 16
    assert "base_channels" not in original


def test_training_config_populates_nested_defaults():
    config = {
        "model": {"resolution": 16, "in_channels": 2, "condition_dim": 0},
        "training": {"optimizer": {"lr": 1e-3}},
    }

    resolved = resolve_config(config)

    assert resolved["training"]["condition_dropout"] == 0.0
    assert resolved["training"]["optimizer"]["adam_betas"] == [0.9, 0.999]
    assert resolved["training"]["loss"] == LOSS_DEFAULTS


def test_old_sparse_checkpoint_model_config_remains_constructible():
    # This is the reconstruction path used by sampling for older checkpoints.
    legacy_config = {
        "resolution": 8,
        "in_channels": 1,
        "label_dim": 0,
        "base_channels": 4,
        "channel_mult": [1],
        "num_blocks": 1,
        "attn_resolutions": [],
    }

    model = models.from_config(legacy_config, device=torch.device("cpu"))

    assert model.condition_dim == 0
    assert model.unet.enc["8x8_block0"].num_heads == 0
    assert "channels_per_head" not in legacy_config


def test_old_controlnet_config_resolves_sparse_base_checkpoint(monkeypatch):
    legacy_base = {"resolution": 8, "in_channels": 1, "condition_dim": 0}
    monkeypatch.setattr(models.torch, "load", lambda *args, **kwargs: {"model_config": legacy_base})

    resolved = models.dataset_config({"architecture": "controlnet", "base_model": "old-model"})

    assert resolved["channels_per_head"] == EDM2_DEFAULTS["channels_per_head"]
    assert resolved["downsample_res"] == 8


def test_new_controlnet_config_uses_embedded_resolved_base_config(monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("an embedded base_model_config should not reload checkpoint metadata")

    monkeypatch.setattr(models.torch, "load", unexpected_load)
    config = {
        "architecture": "controlnet",
        "base_model": "new-model",
        "base_model_config": {"resolution": 8, "in_channels": 1, "condition_dim": 0},
    }

    resolved = models.dataset_config(config)

    assert resolved["channels_per_head"] == EDM2_DEFAULTS["channels_per_head"]
