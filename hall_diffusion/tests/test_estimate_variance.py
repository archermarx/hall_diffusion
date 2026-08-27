import torch

from hall_diffusion.estimate_variance import estimate_moments


def test_residual_moments_are_centered_consistently():
    residuals = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [-1.0, 1.0, 0.0],
            [0.5, -0.5, 1.0],
            [-0.5, -0.5, -1.0],
        ]
    )
    residual_sum = residuals.sum(dim=0, keepdim=True)
    residual_square_sum = residuals.square().sum(dim=0, keepdim=True)

    mean, mean_square, variance = estimate_moments(
        residual_sum, residual_square_sum, residuals.shape[0]
    )

    torch.testing.assert_close(mean[0], residuals.mean(dim=0))
    torch.testing.assert_close(mean_square[0], residuals.square().mean(dim=0))
    torch.testing.assert_close(variance[0], residuals.var(dim=0, correction=0))
