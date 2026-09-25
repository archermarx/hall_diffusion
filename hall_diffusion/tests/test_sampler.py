import pytest
import torch

from hall_diffusion.samplers.edmsampler import EDMSampler, ObservationGuidance, RK2Integrator


class IdentityDenoiser(torch.nn.Module):
    def forward(self, x, noise_std):
        return x


class GradRecordingDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, noise_std):
        self.calls.append((torch.is_grad_enabled(), x.requires_grad))
        return x


class CountingGuidance:
    def __init__(self):
        self.calls = 0

    def __call__(self, x, x_0, t, observation):
        self.calls += 1
        return torch.ones_like(x)


class TimeRecordingIntegrator:
    def __init__(self):
        self.times = []

    def step_with_guidance(self, x, t1, t2, model_args=None):
        self.times.append((t1, t2))
        return x


def make_integrator(threshold, model=None):
    score = CountingGuidance()
    guidance = ObservationGuidance("dps", score, observation=None)
    return RK2Integrator(
        model or IdentityDenoiser(),
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


def test_reused_midpoint_guidance_does_not_build_an_autograd_graph():
    model = GradRecordingDenoiser()
    integrator, _ = make_integrator(threshold=0.1, model=model)

    integrator.step_with_guidance(torch.ones(1, 1, 2), torch.tensor(1.0), torch.tensor(0.8))

    assert model.calls == [(True, True), (False, False)]


def test_recomputed_midpoint_guidance_retains_its_input_graph():
    model = GradRecordingDenoiser()
    integrator, _ = make_integrator(threshold=0.1, model=model)

    integrator.step_with_guidance(torch.ones(1, 1, 2), torch.tensor(0.08), torch.tensor(0.06))

    assert model.calls == [(True, True), (True, True)]


def test_trajectory_recording_is_disabled_by_default():
    sampler = EDMSampler((1, 1, 2), num_steps=4, noise_min=0.01, noise_max=1.0, exponent=2.0)
    output = sampler.sample(RK2Integrator(IdentityDenoiser()), showprogress=False)

    assert output.shape == (1, 1, 1, 2)


def test_sampler_keeps_timestep_control_flow_on_cpu():
    sampler = EDMSampler((1, 1, 2), num_steps=4, noise_min=0.01, noise_max=1.0, exponent=2.0)
    integrator = TimeRecordingIntegrator()

    sampler.sample(integrator, showprogress=False)

    assert integrator.times
    assert all(isinstance(t, float) for times in integrator.times for t in times)


def test_final_finite_check_still_rejects_invalid_samples():
    class InvalidIntegrator:
        def step_with_guidance(self, x, t1, t2, model_args=None):
            return torch.full_like(x, torch.nan)

    sampler = EDMSampler((1, 1, 2), num_steps=4, noise_min=0.01, noise_max=1.0, exponent=2.0)

    with torch.no_grad(), pytest.raises(FloatingPointError, match="NaN/Inf"):
        sampler.sample(InvalidIntegrator(), showprogress=False)
