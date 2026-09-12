import math
import torch
from typing import Literal
from tqdm import tqdm

ODEMethod = Literal["midpoint", "ralston", "heun"]

RK_METHODS = {
    "midpoint": 0.5,
    "ralston": 2.0 / 3.0,
    "heun": 1.0,
}

class ObservationGuidance:
    def __init__(self, type, obs_score, observation, guidance_start_time=float('inf')):
        self.type = type
        self.observation = observation
        self.obs_score = obs_score
        self.guidance_start_time = guidance_start_time

    def __call__(self, x, x_denoised, t):
        return self.obs_score(x, x_denoised, t, self.observation)

class RK2Integrator:
    def __init__(
            self,
            model,
            guidance_score_fn = None,
            method: ODEMethod | None = None,
            rk_alpha: float = 0.5,
            S_churn: float = 0.0,
            S_tmin: float = 0.0,
            S_tmax: float = float('inf'),
            S_noise: float = 1.003,
            guidance_second_order_below: float = float('inf'),
        ):
        if method is not None:
            rk_alpha = RK_METHODS[method]
        if guidance_second_order_below < 0:
            raise ValueError("guidance_second_order_below must be nonnegative")

        self.model = model
        self.guidance_score_fn = guidance_score_fn
        self.rk_alpha = rk_alpha
        self.guidance_second_order_below = guidance_second_order_below

        # Stochasticity parameters
        self.S_churn = S_churn
        self.gamma = min(S_churn, math.sqrt(2) - 1)
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    def eval_deriv(self, x, t, model_args=None, guidance_override=None, compute_guidance=True):
        model_args = {} if model_args is None else model_args
        ones = torch.ones((x.shape[0], 1, 1), device=x.device)
        x0 = self.model(x, t * ones, **model_args)
        score = (x0 - x) / t**2
        guidance_score = None
        if self.guidance_score_fn is not None and \
            self.guidance_score_fn.type == "dps" and \
            t < self.guidance_score_fn.guidance_start_time:

            if compute_guidance:
                guidance_score = self.guidance_score_fn(x, x0, t)
            else:
                if guidance_override is None:
                    raise ValueError("guidance_override is required when compute_guidance is false")
                guidance_score = guidance_override
            score += guidance_score
        deriv = -score * t
        return x0, deriv, guidance_score

    def step(self, x, t1, t2, model_args=None):
        alpha = self.rk_alpha
        c = 1 / (2 * alpha)

        # Step length
        h = t2 - t1

        # Evaluate denoiser prediction
        x0, d1, guidance1 = self.eval_deriv(x, t1, model_args=model_args)

        # Take first step to midpoint
        x_mid, t_mid = x + alpha * h * d1, t1 + alpha * h

        # Take second step
        if t_mid != 0:
            recompute_guidance = guidance1 is None or t_mid <= self.guidance_second_order_below
            _, d_mid, _ = self.eval_deriv(
                x_mid,
                t_mid,
                model_args=model_args,
                guidance_override=guidance1,
                compute_guidance=recompute_guidance,
            )
            x2 = x + h * ((1 - c) * d1 + c * d_mid)
        else:
            x2 = x + h * d1

        return x2, x0

    def step_with_guidance(self, x, t1, t2, model_args=None):
        x = x.detach()
        x.requires_grad = True

        # Add stochasticity if required
        if self.S_churn > 0 and self.S_tmin <= t1 <= self.S_tmax:
            t1, t1_old = (1 + self.gamma) * t1 , t1
            noise_std = (t1**2 - t1_old**2).sqrt() * self.S_noise
            eps = torch.randn_like(x)
            x = x + noise_std * eps

        x_pred, x_denoised = self.step(x, t1, t2, model_args=model_args)

        if self.guidance_score_fn is not None and \
            self.guidance_score_fn.type == "constant" and \
            t1 < self.guidance_score_fn.guidance_start_time:

            obs_score = self.guidance_score_fn(x, x_denoised, 0.5 * (t1 + t2))
            x_pred += obs_score

        return x_pred.detach()

class EDMSampler():
    def __init__(self, shape, num_steps, noise_min, noise_max, exponent):
        self.shape = shape
        self.num_steps = num_steps
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.exponent = exponent
        self.noise_steps = self.get_noise_steps()

    def get_noise_steps(self):
        inv_rho = 1 / self.exponent
        i = torch.arange(0, self.num_steps)
        f1 = self.noise_max**inv_rho
        f2 = (self.noise_min**inv_rho - self.noise_max**inv_rho) / (self.num_steps - 2)
        timesteps = (f1 + i * f2) ** self.exponent
        timesteps[-1] = 0
        return timesteps

    def sample(self, integrator, showprogress=True, device=None, model_args=None, record_trajectory=False):
        model_args = {} if model_args is None else model_args
        # Generate initial noise and timesteps
        x = self.noise_max * torch.randn(self.shape, device=device)
        # Move the schedule once instead of repeatedly copying CPU scalars into
        # CUDA kernels throughout the denoiser and integration arithmetic.
        timesteps = self.noise_steps.to(device)

        (b, c, w) = x.shape
        num_steps = len(timesteps)

        output = torch.empty((num_steps, b, c, w)) if record_trajectory else None
        if record_trajectory:
            output[0] = x

        for step_idx, t in enumerate(pbar := tqdm(timesteps, disable=(not showprogress))):
            if step_idx == 0:
                continue

            t_prev = timesteps[step_idx - 1]
            if x.device.type == "mps":
                t_prev = t_prev.clone()  # Work around pytorch/pytorch#193057.
            pbar.set_description(f"Noise level: {t_prev:.4f}->{t:.4f}")

            x = integrator.step_with_guidance(x, t_prev, t, model_args=model_args)

            # Check for NaN or Inf
            if not torch.all(torch.isfinite(x)):
                print("NaN/Inf detected during sampling. Exiting")
                exit(1)

            if record_trajectory:
                output[step_idx] = x

        if record_trajectory:
            output[-1] = x
            return output
        return x.detach().cpu().unsqueeze(0)
