from hall_diffusion import models
from hall_diffusion.configuration import (
    EDM2_DEFAULTS,
    LOSS_DEFAULTS,
    duration_in_epochs,
    limited_batch_size,
    resolve_config,
    resolve_model_config,
    training_duration,
    training_example_target,
)

import pytest
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
        "fourier_features": False,
        "downsample_res": 64,
    }
    assert config["fourier_features"] is True


def test_explicit_legacy_fourier_setting_is_disabled():
    config = resolve_model_config({"fourier_features": True})

    assert config["fourier_features"] is False


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
    assert resolved["training"]["torch_compile"] is False
    assert resolved["training"]["optimizer"]["adam_betas"] == [0.9, 0.999]
    assert resolved["training"]["loss"] == LOSS_DEFAULTS


def test_max_examples_overrides_epochs_and_sets_schedule_horizon():
    epochs = training_duration({"epochs": 100, "max_examples": 250}, dataset_size=40)

    assert epochs == 6.25
    assert training_example_target(epochs, dataset_size=40) == 250
    assert limited_batch_size(240, 250, batch_size=16) == 10
    assert limited_batch_size(250, 250, batch_size=16) == 0


def test_epoch_duration_remains_backward_compatible():
    assert training_duration({"epochs": 12}, dataset_size=40) == 12.0
    assert training_duration({"epochs": 12.5}, dataset_size=40) == 12.5


def test_ema_example_durations_override_epoch_durations():
    config = {
        "ema_epochs": 8,
        "ema_examples": 250,
        "ema_start_epochs": 4,
        "ema_start_examples": 0,
    }

    assert duration_in_epochs(config, 40, "ema_epochs", "ema_examples") == 6.25
    assert duration_in_epochs(
        config,
        40,
        "ema_start_epochs",
        "ema_start_examples",
        allow_zero=True,
    ) == 0.0


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_max_examples_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        training_duration({"max_examples": value}, dataset_size=40)


def test_training_duration_requires_epochs_or_examples():
    with pytest.raises(ValueError, match="either 'epochs' or 'max_examples'"):
        training_duration({}, dataset_size=40)


@pytest.mark.parametrize("value", [0, -1, float("inf"), True])
def test_training_epochs_must_be_a_finite_positive_number(value):
    with pytest.raises(ValueError, match="finite positive number"):
        training_duration({"epochs": value}, dataset_size=40)


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
