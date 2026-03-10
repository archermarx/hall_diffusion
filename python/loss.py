import torch
from abc import ABC, abstractmethod

# ----------------------------------------------------------------------------
class LossFunction(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, x, model, noise_std, condition_vec) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        pass

    @staticmethod
    def from_config(noise_sampler=None, **kwargs):
        match (kwargs.get("type", "edm2")):
            case "edm2":
                loss_fn = EDM2Loss(noise_sampler, **kwargs)
            case "flow matching":
                loss_fn = FlowMatchingLoss(**kwargs)
            case _:
                raise NotImplementedError

# ----------------------------------------------------------------------------
# Loss function for Flow matching
class FlowMatchingLoss:
    def __init__(self, t_loc=0.0, t_scale=1.0, noise_sampler="logit", **kwargs):
        self.t_loc = t_loc
        self.t_scale = t_scale
        self.noise_sampler = noise_sampler

    def __call__(self, x, model, t=None, condition_vec=None):
        batch_size, _, _ = x.shape

        # Sample t from a logit-normal distribution if not provided
        if t is None:
            match self.noise_sampler:
                case "logit":
                    t = torch.sigmoid(self.t_loc + self.t_scale * torch.randn([batch_size, 1, 1], device=x.device))
                case "uniform":
                    t = 0.999 * torch.rand([batch_size, 1, 1], device=x.device)
                case _:
                    raise NotImplementedError()

        # Convert t to a sigma to work with EDM2 preconditioning
        sigma = t / (1 - t + 1e-8)

        # Blend image with noise and then denoise
        xi = torch.randn_like(x, device=x.device)
        noisy_im = (1 - t) * x + t * xi
        denoised = model(noisy_im, sigma, condition_vec)

        # Calculate loss
        # Note: this is the reconstruction loss, which is identical to the flow-matching loss
        # if we have a model that predicts x, i.e.
        # (v - (xi - x))**2 == (D(x; sigma) - x)**2 if v = xi - D(x; sigma)
        loss = ((denoised - x)**2).mean()

        return loss, loss.item(), noisy_im, denoised

# ----------------------------------------------------------------------------
# Loss function for EDM2 model
# Modified to include first- and second-deriviative losses to hopefully reduce noise
class EDM2Loss:
    def __init__(
        self,
        noise_sampler,
        P_mean=-0.4,
        P_std=1.0,
        sigma_data=0.5,
        include_logvar=False,
        deriv_h=1.0,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.noise_sampler = noise_sampler
        self.deriv_h = deriv_h
        self.include_logvar = include_logvar

    def __call__(
        self,
        x,
        model,
        noise_std=None,
        condition_vec=None,
    ):
        batch_size, _, _ = x.shape
        rnd_normal = torch.randn([batch_size, 1, 1], device=x.device)

        if noise_std is None:
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        else:
            sigma = noise_std

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        noise = self.noise_sampler.sample(batch_size) * sigma

        noisy_im = x + noise

        if self.include_logvar:
            denoised, logvar = model(noisy_im, sigma, condition_vec, return_logvar=True)
        else:
            denoised = model(noisy_im, sigma, condition_vec)
            logvar = torch.tensor(0.0)

        # Base loss
        base_loss = (denoised - x) ** 2

        # "Step size": scale for diff loss

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
        loss = (
            weight * (base_loss + diff_loss_1 * h + diff_loss_2 * h**2) / total_weight
        )
        base_loss = loss.mean().item()

        # Weight by homoscedastic uncertainty
        if self.include_logvar:
            loss = loss / logvar.exp() + logvar

        return loss.mean(), base_loss, noisy_im, denoised