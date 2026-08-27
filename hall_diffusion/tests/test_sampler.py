import torch

from hall_diffusion.samplers.edmsampler import EDMSampler, ObservationGuidance, RK2Integrator


class IdentityDenoiser(torch.nn.Module):
    def forward(self, x, noise_std):
        return x


class CountingGuidance:
    def __init__(self):
        self.calls = 0

    def __call__(self, x, x_0, t, observation):
        self.calls += 1
        return torch.ones_like(x)


def make_integrator(threshold):
    score = CountingGuidance()
    guidance = ObservationGuidance("dps", score, observation=None)
    return RK2Integrator(
        IdentityDenoiser(),
        guidance_score_fn=guidance,
        method="midpoint",
        guidance_second_order_below=threshold,
    ), score


def test_guidance_is_reused_above_threshold():
    integrator, score = make_integrator(threshold=0.1)

    integrator.step(torch.ones(1, 1, 2), torch.tensor(1.0), torch.tensor(0.8))

    assert score.calls == 1


def test_guidance_is_recomputed_below_threshold():
    integrator, score = make_integrator(threshold=0.1)

    integrator.step(torch.ones(1, 1, 2), torch.tensor(0.08), torch.tensor(0.06))

    assert score.calls == 2


def test_trajectory_recording_is_disabled_by_default():
    sampler = EDMSampler((1, 1, 2), num_steps=4, noise_min=0.01, noise_max=1.0, exponent=2.0)
    output = sampler.sample(RK2Integrator(IdentityDenoiser()), showprogress=False)

    assert output.shape == (1, 1, 1, 2)
