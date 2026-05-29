import torch
import numpy as np
from models.controlnet import ControlNet

# ----------------------------------------------------------------------------
# Loss function for EDM2 model
# Modified to include first- and second-deriviative losses to hopefully reduce noise
class EDM2Loss:
    def __init__(
        self,
        P_mean=-0.4,
        P_std=1.0,
        sigma_data=0.5,
        deriv_h=1.0,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.deriv_h = deriv_h

    def __call__(
        self,
        x,
        model,
        noise_std=None,
        condition_vec=None,
        ctrl=None,
    ):
        batch_size, _, _ = x.shape
        rnd_normal = torch.randn([batch_size, 1, 1], device=x.device)

        if noise_std is None:
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        else:
            sigma = noise_std

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(x) * sigma

        noisy_im = x + noise

        if isinstance(model, ControlNet):
            assert isinstance(ctrl, tuple)
            denoised = model(noisy_im, ctrl[0], sigma, condition_vec)
            ctrl_loss_weight = ctrl[1]
        else:
            denoised = model(noisy_im, sigma, condition_vec)
            ctrl_loss_weight = 1.0

        # Base loss
        base_loss = (denoised - x) ** 2

        # Add control loss weight if using ControlNet
        base_loss = base_loss * ctrl_loss_weight

        # Derivative loss
        diff_denoised = torch.diff(denoised)
        diff_images = torch.diff(x)
        diff_loss_1 = (diff_denoised - diff_images) ** 2
        diff_loss_1 = torch.nn.functional.pad(diff_loss_1, (0, 1))

        # Second derivative loss
        diff2_denoised = torch.diff(diff_denoised)
        diff2_images = torch.diff(diff_images)
        diff_loss_2 = (diff2_denoised - diff2_images) ** 2
        diff_loss_2 = torch.nn.functional.pad(diff_loss_2, (1, 1))

        h = self.deriv_h

        total_weight = 1 + h + h**2
        loss =  weight * (base_loss + diff_loss_1 * h + diff_loss_2 * h**2) / total_weight
        base_loss = loss.mean().item()

        return loss.mean(), base_loss, noisy_im, denoised