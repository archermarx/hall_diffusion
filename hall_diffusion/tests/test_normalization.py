import pandas as pd

from hall_diffusion.utils.normalization import Normalizer


def write_norm(path, name):
    pd.DataFrame({"Field": [name], "Mean": [1.0], "Std": [2.0], "Log": [False]}).to_csv(path, index=False)


def test_normalizer_keeps_categories_separate_and_builds_active_tensor_channels(tmp_path):
    write_norm(tmp_path / "norm_data.csv", "field")
    write_norm(tmp_path / "norm_params.csv", "parameter")
    write_norm(tmp_path / "norm_perf.csv", "performance")

    normalizer = Normalizer(tmp_path, scalars_in_tensor=True)

    assert normalizer.spatial_fields() == {"field": 0}
    assert normalizer.input_params() == {"parameter": 0}
    assert normalizer.performance_scalars() == {"performance": 0}
    assert normalizer.tensor_channels() == {"field": 0, "parameter": 1, "performance": 2}


def test_writing_tensor_metadata_preserves_performance_normalization_without_fourier_features(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    write_norm(source / "norm_data.csv", "field")
    write_norm(source / "norm_params.csv", "parameter")
    write_norm(source / "norm_perf.csv", "performance")

    Normalizer(source, scalars_in_tensor=True).write_normalization_info(output)

    assert (output / "norm_perf.csv").exists()
