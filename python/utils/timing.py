import sys
from collections import defaultdict
from contextlib import contextmanager
import time
import torch

class StepTimer:
    """Per-section wall-clock profiler with optional CUDA synchronization.

    Usage:
        timer = StepTimer(enabled=True, print_every=50)
        with timer.section("forward"):
            loss = model(x)
        timer.step()   # call once per training step; prints a table every print_every steps

    Each section time is measured with GPU sync so it reflects real compute time, not just
    kernel-dispatch time.  Disable (enabled=False) for zero-overhead operation.
    """

    def __init__(self, enabled: bool = False, print_every: int = 50, file = None):
        self.enabled = enabled
        self.print_every = print_every
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._step = 0
        self._file = file

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        self._totals[name] += elapsed
        self._counts[name] += 1

    def step(self):
        if not self.enabled:
            return
        self._step += 1
        if self._step % self.print_every == 0:
            self._print_report()

    def _print_report(self):
        if not self._totals:
            return
        accounted = sum(self._totals.values())

        # Write to file specified or stdout if none given
        with open(self._file, "a+") if self._file else sys.stdout as f:
            print(f"\n--- StepTimer report  (step {self._step}) ---", file=f)
            for name, total in sorted(self._totals.items(), key=lambda kv: -kv[1]):
                count = self._counts[name]
                amortized_ms = 1000.0 * total / self._step
                pct = 100.0 * total / accounted if accounted > 0 else 0.0
                if count == self._step:
                    # This happens every step, so per-call time is not interesting; omit it for cleaner report
                    print(f"  {name:<30s}  {amortized_ms:7.2f} ms/step  ({pct:5.1f}%)", file=f)
                else:
                    # This happens only some steps, so per-call time is interesting to know
                    per_call_ms = 1000.0 * total / count
                    print(f"  {name:<30s}  {amortized_ms:7.2f} ms/step  ({pct:5.1f}%)  [{per_call_ms:.1f} ms/call, {count}x/{self._step} steps]", file=f)
            print(f"  {'total accounted':<30s}  {1000.0 * accounted / self._step:7.2f} ms/step\n", file=f)