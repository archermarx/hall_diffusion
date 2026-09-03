from hall_diffusion import models


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
