# Cross-Platform Portability Audit

## Scope

This audit covers inference, training, checkpoint loading, tensor placement,
accelerator-specific operations, configuration paths, environment setup, and
documentation at upstream commit
`b7c2f05157ae9b8dfa07658bdc20b0982583128c`.

The inclusion rule is:

> Without this change, the code either fails to run or behaves differently
> across CUDA, MPS, CPU, or clean installations on different machines.

Scientific settings, sampling methodology, performance redesigns, and
backend-independent defects are excluded.

## Procedure

1. A read-only agent audited the untouched worktree without receiving proposed
   fixes or the historical portability branch.
2. Its findings were frozen.
3. Each finding was independently checked against the source and, where
   possible, reproduced.
4. Only then was the historical `cuda-mps-portability` branch inspected for
   corroborating evidence. No historical commit is being cherry-picked.

## Included findings

### Checkpoint device tags

- **Files:** `hall_diffusion/sample.py`, `hall_diffusion/train.py`
- **Behavior:** Both restored checkpoints without `map_location`.
- **Evidence:** The CUDA-saved `spt100_small.pth.tar` raises
  `RuntimeError: Attempting to deserialize object on a CUDA device` when CUDA
  is unavailable. It deserializes successfully when remapped.
- **Affected systems:** MPS and CPU machines loading CUDA checkpoints.
- **Decision:** Include backend-aware checkpoint mapping. Preserve the original
  `torch.load` behavior when CUDA is selected; map through CPU only for
  non-CUDA backends.
- **Justification:** Without remapping, the checkpoint fails before its model
  state can be transferred to MPS or CPU.

### Explicit backend selection

- **Files:** `hall_diffusion/utils/utils.py`, `hall_diffusion/sample.py`,
  `hall_diffusion/train.py`
- **Behavior:** Backend selection was automatic only, with no way to request
  or validate a specific backend.
- **Evidence:** Both CLIs rejected `--device`; an intended accelerator run
  could not distinguish successful selection from fallback.
- **Affected systems:** All backends.
- **Decision:** Include `auto|cpu|mps|cuda|xpu`, with automatic priority CUDA,
  MPS, XPU, then CPU.
- **Justification:** Without explicit validation, the same command can run on
  different hardware without making that difference clear.

### CUDA-only mixed precision

- **File:** `hall_diffusion/train.py`
- **Behavior:** Training constructed the CUDA-default GradScaler on every
  backend and selected autocast independently from that scaler.
- **Evidence:** On non-CUDA machines PyTorch warns and disables the scaler,
  leaving a different partial mixed-precision path.
- **Affected systems:** MPS, CPU, and XPU training.
- **Decision:** Enable the existing float16 autocast and GradScaler path only
  for CUDA. Preserve that CUDA path unchanged.
- **Justification:** Without the guard, non-CUDA training enters a
  CUDA-specific mechanism that PyTorch cannot use as configured.

### Personal absolute paths

- **Files:** Perez-Luna and Roberts observation TOMLs and their generators.
- **Behavior:** Shipped files referenced
  `/home/archermarks/projects/hall_diffusion/...`, and generators recreated an
  absolute path.
- **Evidence:** The path does not exist on a clean checkout or the target Mac.
- **Affected systems:** Any machine without the original developer directory.
- **Decision:** Use `mcmc_reference/ref_3charge/normalized` and have generators
  emit the same repository-relative path.
- **Justification:** Without this change, shipped and regenerated observations
  fail on other machines.

## Verified as already portable

- EDM2 forward execution succeeds on CPU and real MPS.
- Deterministic CPU and MPS outputs agree within the documented float32
  tolerance.
- Observation tensors and condition vectors remain on the selected backend.
- Gaussian noise is generated on the requested backend.
- Midpoint, Ralston, and Heun smoke sampling succeeds on real MPS without
  `PYTORCH_ENABLE_MPS_FALLBACK`.
- DataLoader pinned memory is already enabled only for CUDA.
- Current PyTorch lock entries support both CPython 3.13 and 3.14 on macOS ARM,
  so no Python pin is required.

## Investigated and excluded

- **CPU trajectory history:** It imposes accelerator-to-CPU copies but is the
  current sampler API. Changing it would be a performance/memory redesign, not
  an MPS runnability fix.
- **Timing and trace profiling:** Backend-aware diagnostic improvements are
  useful but are not required for ordinary inference or training. They are
  excluded from this minimum patch.
- **RBF noise and Euler integration:** Neither implementation exists in current
  upstream. Adding them would introduce scientific functionality rather than
  make existing functionality portable.
- **Python version pin:** The current lock includes CPython 3.14-compatible
  macOS ARM PyTorch wheels, so the earlier PyTorch 2.8 installation failure is
  no longer evidence for a pin.
- **Additional platform architectures:** Intel macOS, Windows ARM, ROCm, and
  real XPU execution require separate machines and scope.

## Backend-independent follow-ups

These findings are documented but deliberately unchanged:

1. The training data configuration unconditionally pops `fourier_features`.
2. Pytest configuration still points to the former `python/tests` layout.
3. Script-style and package-style imports behave differently.
4. Some pre-existing README examples refer to the former source layout.
5. Best-checkpoint state handling retains a shallow state dictionary.
6. Dataset worker paths can mutate source `.npz` files.
7. The sampler applies `S_noise` twice in the churn perturbation.

None of these defects is specific to CUDA, MPS, or CPU. The portability suite
works around the stale test/import layout rather than expanding this patch.
