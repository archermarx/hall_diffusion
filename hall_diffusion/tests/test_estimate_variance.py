import numpy as np
import torch

from hall_diffusion.estimate_variance import estimate_moments, estimate_process_variance


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


class TinyDataset(torch.utils.data.Dataset):
    grid = np.array([0.0, 1.0])

    def __len__(self):
        return 4

    def __getitem__(self, index):
        params = torch.tensor([index], dtype=torch.float32)
        fields = torch.full((1, 2), index / 4)
        return str(index), params, fields

    def fields(self):
        return {"field": 0}


class IdentityDenoiser(torch.nn.Module):
    def forward(self, noisy, sigma, params):
        return noisy


def test_process_variance_can_be_generated_for_an_already_loaded_model(tmp_path):
    output_file = tmp_path / "process_variance.npz"

    estimate_process_variance(
        IdentityDenoiser(),
        TinyDataset(),
        output_file,
        torch.device("cpu"),
        batch_size=2,
        num_workers=0,
        noise_levels=[0.1, 1.0],
        show_progress=False,
        create_plots=False,
    )

    with np.load(output_file) as result:
        assert result["process_variance"].shape == (2, 1, 2)
        assert result["centered_variance"].shape == (2, 1, 2)
        np.testing.assert_array_equal(result["noise_levels"], [0.1, 1.0])
