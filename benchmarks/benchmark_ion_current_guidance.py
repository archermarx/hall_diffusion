"""Compare endpoint-current guidance with the original dense autograd path.

Run from the repository root:
    python -m benchmarks.benchmark_ion_current_guidance --batch-size 16 --resolution 64

Uses synthetic normalized states and checked-in normalization metadata. The
timings include likelihood guidance and its backward pass, but no neural
denoiser. Cached linear terms are warmed up before timing.
"""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from hall_diffusion.guidance import DPSCovarianceCache, guidance_score
from hall_diffusion.observation_operators import IonCurrentDensity, IonCurrentObservation
from hall_diffusion.utils.normalization import Normalizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--metadata", type=Path,
        default=Path(__file__).resolve().parents[1] / "mcmc_reference/ref_3charge/normalized",
    )
    args = parser.parse_args()
    if min(args.batch_size, args.resolution, args.repeats, args.threads) <= 0:
        parser.error("batch size, resolution, repeats, and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    norm = Normalizer(args.metadata)
    channels = norm.tensor_channels()
    reference = torch.zeros(len(channels), args.resolution, device=device)
    states = 0.1 * torch.randn(args.batch_size, *reference.shape, device=device)
    current = IonCurrentDensity(norm, channels, reference)
    variance = torch.linspace(0.01, 0.1, reference.numel(), device=device).reshape_as(reference)
    variance_model = {
        "noise_levels": reference.new_tensor([0.1, 1.0]),
        "process_variance": torch.stack((variance, 2 * variance)),
    }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    print(f"device={device}, threads={args.threads}, batch={args.batch_size}, resolution={args.resolution}")
    print("case, original_ms, optimized_ms, speedup, relative_score_error")
    for fields in ([], ["B"], ["B", "ui_1"]):
        rows = []
        for name in fields:
            matrix = reference.new_zeros((args.resolution, reference.numel()))
            start = channels[name] * args.resolution
            matrix[:, start:start + args.resolution] = torch.eye(args.resolution, device=device)
            rows.extend(matrix.unbind())
        operator = IonCurrentObservation([*rows, current])

        def original(state):
            value = sum(
                charge * 1.602176634e-19
                * norm.denormalize(state[channels[f"ni_{charge}"], -1], f"ni_{charge}")
                * norm.denormalize(state[channels[f"ui_{charge}"], -1], f"ui_{charge}")
                for charge in range(1, 4)
            )
            return torch.stack([*(row @ state.flatten() for row in rows), value])

        observed = original(reference).detach()
        observed[-1] *= 0.9
        noise = reference.new_full((len(rows) + 1,), 0.01)
        noise[-1] = (0.05 * observed[-1]).square()
        observation = {
            "data": observed, "var": noise, "variance_model": variance_model,
            "covariance_cache": DPSCovarianceCache(),
        }

        def run(measurement):
            state = states.clone().requires_grad_()
            denoised = 0.8 * state + 0.03 * state.square()
            return guidance_score(state, denoised, 0.5, {**observation, "operator": measurement})

        baseline_score = run(original)
        optimized_score = run(operator)
        torch.testing.assert_close(optimized_score, baseline_score, rtol=1e-4, atol=1e-5)
        relative_error = ((optimized_score - baseline_score).norm() / baseline_score.norm()).item()
        timings = []
        for measurement in (original, operator):
            elapsed = []
            for _ in range(args.repeats):
                synchronize()
                start = perf_counter()
                run(measurement)
                synchronize()
                elapsed.append(1000 * (perf_counter() - start))
            timings.append(median(elapsed))
        label = "+".join([*fields, "current"])
        print(f"{label}, {timings[0]:.3f}, {timings[1]:.3f}, {timings[0] / timings[1]:.1f}x, {relative_error:.2e}", flush=True)


if __name__ == "__main__":
    main()
