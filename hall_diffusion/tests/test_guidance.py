import numpy as np
import pytest
import torch

from hall_diffusion.guidance import (
    DPSCovarianceCache,
    _corrected_mean,
    _interpolate,
    guidance_score,
    load_variance_model,
)


def variance_model(num_channels, variance=0.3):
    return {
        "noise_levels": torch.tensor([0.1, 1.0, 10.0]),
        "process_variance": torch.full((3, num_channels), variance),
        "scale": 1.0,
    }


def test_nonlinear_guidance_is_finite():
    x_t = torch.tensor([[[2.0, 1.0]]], requires_grad=True)
    x_0 = x_t.square()

    def measurement(state):
        return torch.stack((state[0, 0] * state[0, 1], state[0].sum()))

    observation = {
        "operator": measurement,
        "data": torch.tensor([1.0, 1.0]),
        "var": torch.tensor([0.1, 0.2]),
        "variance_model": variance_model(1),
    }
    score = guidance_score(x_t, x_0, torch.tensor(1.0), observation)

    assert score.shape == x_t.shape
    assert torch.all(torch.isfinite(score))
    assert torch.linalg.vector_norm(score) > 0


def test_linear_operator_supports_full_measurement_covariance():
    x_t = torch.tensor([[[0.5, -0.25]]], requires_grad=True)
    x_0 = 0.75 * x_t
    observation = {
        "operator": torch.eye(2),
        "data": torch.zeros(2),
        "var": torch.tensor([[0.2, 0.05], [0.05, 0.3]]),
        "variance_model": variance_model(1),
    }

    score = guidance_score(x_t, x_0, torch.tensor(0.5), observation)

    assert score.shape == x_t.shape
    assert torch.all(torch.isfinite(score))


def test_fieldwise_variance_is_propagated_through_linear_operator():
    x_t = torch.tensor([[[0.5], [-0.25]]], requires_grad=True)
    scale = 0.4
    x_0 = scale * x_t
    operator = torch.tensor([[1.0, 0.2], [-0.3, 0.8]])
    q = torch.tensor([0.1, 0.4])
    observation = {
        "operator": operator,
        "data": torch.zeros(2),
        "var": 0.2,
        "variance_model": {
            "noise_levels": torch.tensor([0.1, 1.0]),
            "process_variance": q.repeat(2, 1),
        },
    }

    score = guidance_score(x_t, x_0, torch.tensor(0.5), observation)
    covariance = 0.2 * torch.eye(2) + operator @ torch.diag(q) @ operator.T
    residual = operator @ x_0.reshape(-1)
    expected = -scale * (operator.T @ torch.linalg.solve(covariance, residual)).reshape_as(x_t)

    torch.testing.assert_close(score, expected)


def test_diagonal_linear_guidance_avoids_cholesky(monkeypatch):
    x_t = torch.tensor(
        [[[0.5, -0.25, 0.75]], [[-0.5, 0.25, -0.75]]], requires_grad=True
    )
    scale = 0.4
    operator = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    process_variance = torch.tensor([[0.1, 0.4, 0.2], [0.1, 0.4, 0.2]])
    observation = {
        "operator": operator,
        "data": torch.zeros(2),
        "var": torch.tensor([0.2, 0.3]),
        "variance_model": {
            "noise_levels": torch.tensor([0.1, 1.0]),
            "process_variance": process_variance,
        },
        "diagonal_covariance": True,
        "covariance_jitter": 1e-6,
    }

    def unexpected_cholesky(*args, **kwargs):
        raise AssertionError("diagonal guidance should not use Cholesky")

    monkeypatch.setattr(torch.linalg, "cholesky", unexpected_cholesky)
    score = guidance_score(x_t, scale * x_t, torch.tensor(0.5), observation)

    projected = operator.square() @ process_variance[0]
    residual = scale * x_t.flatten(start_dim=1) @ operator.T
    solved = residual / (observation["var"] + projected + observation["covariance_jitter"])
    expected = -scale * (solved @ operator).reshape_as(x_t)
    torch.testing.assert_close(score, expected)


def test_dense_linear_covariance_factorization_is_shared_across_batch(monkeypatch):
    x_t = torch.tensor(
        [[[0.5, -0.25]], [[-0.5, 0.25]], [[0.75, 0.1]]], requires_grad=True
    )
    operator = torch.tensor([[1.0, 0.2], [-0.3, 0.8]])
    observation = {
        "operator": operator,
        "data": torch.zeros(2),
        "var": torch.tensor([0.2, 0.3]),
        "variance_model": variance_model(1),
    }
    original_cholesky = torch.linalg.cholesky
    covariance_shapes = []

    def record_cholesky(covariance):
        covariance_shapes.append(covariance.shape)
        return original_cholesky(covariance)

    monkeypatch.setattr(torch.linalg, "cholesky", record_cholesky)
    score = guidance_score(x_t, 0.75 * x_t, torch.tensor(0.5), observation)

    assert covariance_shapes == [torch.Size([2, 2])]
    assert score.shape == x_t.shape
    assert torch.all(torch.isfinite(score))


def test_dense_linear_covariance_factorization_is_cached_across_batches(monkeypatch):
    operator = torch.tensor([[1.0, 0.2], [-0.3, 0.8]])
    observation = {
        "operator": operator,
        "data": torch.zeros(2),
        "var": torch.tensor([0.2, 0.3]),
        "variance_model": variance_model(1),
        "covariance_cache": DPSCovarianceCache(),
    }
    original_cholesky = torch.linalg.cholesky
    factorization_count = 0

    def record_cholesky(covariance):
        nonlocal factorization_count
        factorization_count += 1
        return original_cholesky(covariance)

    monkeypatch.setattr(torch.linalg, "cholesky", record_cholesky)
    scores = []
    for batch_size in (3, 1):
        x_t = torch.randn(batch_size, 1, 2, requires_grad=True)
        scores.append(guidance_score(x_t, 0.75 * x_t, 0.5, observation))

    assert factorization_count == 1
    assert all(torch.all(torch.isfinite(score)) for score in scores)


def test_process_variance_is_linear_in_log_noise():
    noise_levels = torch.tensor([0.1, 1.0, 10.0])
    process_variance = torch.tensor([[1.0], [3.0], [7.0]])

    interpolated = _interpolate(torch.tensor(10**0.5), noise_levels, process_variance)

    torch.testing.assert_close(interpolated, torch.tensor([5.0]))


def test_per_cell_variance_is_used_by_default_and_can_be_channel_averaged():
    operator = torch.eye(2)

    def get_score(channel_average):
        x_t = torch.tensor([[[1.0, 1.0]]], requires_grad=True)
        x_0 = x_t
        return guidance_score(
            x_t,
            x_0,
            torch.tensor(0.5),
            {
                "operator": operator,
                "data": torch.zeros(2),
                "var": 0.1,
                "variance_model": {
                    "noise_levels": torch.tensor([0.1, 1.0]),
                    "process_variance": torch.tensor([[[0.1, 0.9]], [[0.1, 0.9]]]),
                    "channel_average": channel_average,
                },
            },
        )

    per_cell = get_score(False).reshape(-1)
    averaged = get_score(True).reshape(-1)

    assert not torch.isclose(per_cell[0], per_cell[1])
    torch.testing.assert_close(averaged[0], averaged[1])


def test_bias_correction_subtracts_interpolated_denoiser_residual():
    x_0 = torch.tensor([[[1.0, 2.0]]])
    variance = {
        "noise_levels": torch.tensor([0.1, 1.0]),
        "mean_residual": torch.tensor([[[0.2, -0.1]], [[0.4, 0.3]]]),
        "bias_correction": True,
        "bias_scale": 0.5,
    }

    corrected = _corrected_mean(x_0, torch.tensor(10**-0.5), variance)
    expected_bias = torch.tensor([[[0.3, 0.1]]])

    torch.testing.assert_close(corrected, x_0 - 0.5 * expected_bias)


def test_process_variance_shape_must_match_checkpoint_derived_state_shape(tmp_path):
    path = tmp_path / "process_variance.npz"
    shape = (2, 3, 4)
    np.savez(
        path,
        process_variance=np.ones(shape),
        centered_variance=np.ones(shape),
        mean_residual=np.zeros(shape),
        noise_levels=np.array([0.1, 1.0]),
        residual_convention=np.array("denoised_minus_clean"),
        variance_convention=np.array("uncentered_mse"),
    )

    with pytest.raises(ValueError, match=r"expected \(2, 4, 4\)"):
        load_variance_model(path, {}, state_shape=(4, 4), device="cpu")
