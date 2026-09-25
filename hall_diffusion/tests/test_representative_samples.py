import numpy as np
import pytest
import torch

from hall_diffusion.utils.representative_samples import pca_k_medoids


def test_pca_k_medoids_selects_original_samples_from_separated_groups():
    samples = torch.tensor(
        [
            [[0.0, 0.0]],
            [[0.1, -0.1]],
            [[-0.1, 0.1]],
            [[10.0, 10.0]],
            [[10.1, 9.9]],
            [[9.9, 10.1]],
        ]
    )

    result = pca_k_medoids(samples, 2, n_components=2)

    assert isinstance(result.medoids, torch.Tensor)
    assert result.medoids.device == samples.device
    torch.testing.assert_close(result.medoids, samples[result.medoid_indices])
    assert len(set(result.labels[:3])) == 1
    assert len(set(result.labels[3:])) == 1
    assert result.labels[0] != result.labels[3]
    assert result.embedding.shape == (6, 2)
    assert result.explained_variance_ratio.shape == (2,)
    assert result.inertia >= 0
    assert 1 <= result.n_iter <= 100


def test_weights_change_the_selected_medoid():
    samples = np.array([[0.0], [2.0], [10.0]])

    unweighted = pca_k_medoids(samples, 1, n_components=1)
    weighted = pca_k_medoids(samples, 1, n_components=1, weights=[10.0, 1.0, 1.0])

    assert unweighted.medoid_indices.tolist() == [1]
    assert weighted.medoid_indices.tolist() == [0]
    np.testing.assert_array_equal(weighted.medoids, samples[[0]])


def test_pca_component_count_is_capped_by_sample_rank():
    samples = np.arange(24, dtype=np.float32).reshape(3, 2, 4)

    result = pca_k_medoids(samples, 1, n_components=20)

    assert result.embedding.shape == (3, 2)
    assert result.explained_variance_ratio.shape == (2,)


def test_duplicate_samples_still_produce_distinct_medoid_indices():
    samples = np.array([[0.0, 0.0], [0.0, 3.0], [0.0, 0.0]])

    result = pca_k_medoids(samples, 3, n_components=2)

    assert sorted(result.medoid_indices) == [0, 1, 2]


def test_large_float32_values_do_not_overflow_distance_calculation():
    rng = np.random.default_rng(12)
    samples = (rng.normal(size=(128, 25, 128)) * np.float32(1e20)).astype(np.float32)

    result = pca_k_medoids(samples, 8, n_components=16)

    assert len(np.unique(result.medoid_indices)) == 8
    assert np.isfinite(result.embedding).all()
    assert np.isfinite(result.explained_variance_ratio).all()
    assert np.isfinite(result.inertia)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_medoids": 0}, "n_medoids"),
        ({"n_medoids": 2, "weights": [1.0, 0.0, 0.0]}, "positive weight"),
        ({"n_medoids": 1, "weights": [1.0, -1.0, 1.0]}, "nonnegative"),
        ({"n_medoids": 1, "weights": [1.0, 1.0]}, "shape"),
        ({"n_medoids": 1, "n_components": 0}, "n_components"),
    ],
)
def test_pca_k_medoids_validates_arguments(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        pca_k_medoids(np.arange(3.0)[:, None], **kwargs)


def test_pca_k_medoids_rejects_nonfinite_samples():
    with pytest.raises(ValueError, match="finite"):
        pca_k_medoids(np.array([[0.0], [np.nan]]), 1)
