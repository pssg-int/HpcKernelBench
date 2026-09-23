"""
Check #10 (GATE SEMANTICS, stencil): the correctness gate must be able to FAIL
at the spec's own step counts.

Regression guard for a bug found 2026-09-21: the domain's synthetic weights
used to sum to 0.5, so the true field decayed like 0.5^T (~4e-153 at T=1000,
exactly 0.0 at T=10240) while the gate's scale stayed O(1). An all-zeros
output then passed the 1e-5 tolerance from about T=20 onward, i.e. every
spec-length (T=1000 / 10240) result was effectively ungated.

Drives stencil.py's real StencilWorkload/reference_stencil and harness.py's
real check_correctness -- not a reimplementation. Exits non-zero on failure.

Run:  python bench/audit/scripts/check10_stencil_gate_not_vacuous.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
from kernelbench import harness
from kernelbench.domains import stencil

TOL = 1e-5  # spec variant 1 tolerance
failures = []


def gate(out, ref, scale):
    return harness.check_correctness(out, ref, TOL, "fp64 ref", mode="max_scaled_err", scale=scale)


print("=== stencil gate must reject wrong outputs at spec-scale T ===")
for shape, T in [((128, 128), 5), ((128, 128), 100), ((128, 128), 1000), ((128, 128), 10240)]:
    w = stencil.StencilWorkload(name="probe", kind="star", dims=2, radius=1,
                                grid_shape=shape, timesteps=T)
    ref, scale = stencil.reference_stencil(w, {})
    for label, bad in [("all zeros", np.zeros_like(ref)), ("x1.5 (wrong scale)", ref * 1.5)]:
        c = gate(bad, ref, scale)
        status = "REJECTED" if not c.passed else "PASSED  <-- gate is vacuous"
        print(f"  T={T:6d} {label:20s} err={c.value:.2e}  {status}")
        if c.passed:
            failures.append((T, label))
    ok = gate(ref.copy(), ref, scale)
    if not ok.passed:
        failures.append((T, "correct output rejected"))

print("\n=== weights are a nonnegative, sum-to-1 combination for every shape ===")
for dims in (1, 2, 3):
    for kind in ("star", "box"):
        for r in (1, 2, 3, 4):
            wd = stencil._build_weights(stencil._support_offsets(kind, dims, r), dims)
            if abs(sum(wd.values()) - 1.0) > 1e-12 or min(wd.values()) <= 0:
                failures.append((f"{kind}{dims}d{r}r", "weights not convex"))
                print(f"  BAD {kind}{dims}d{r}r sum={sum(wd.values())}")
print("  done")

print("\n=== flat-field warning fires when misplaced-data bugs become invisible ===")
w = stencil.StencilWorkload(name="probe-flat", kind="star", dims=2, radius=1,
                            grid_shape=(64, 64), timesteps=10240)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    stencil.reference_stencil(w, {})
print(f"  64x64, T=10240: {'warned' if caught else 'NO WARNING'}")
if not caught:
    failures.append(("flat-field", "no warning"))

if failures:
    print(f"\nFAILED: {failures}")
    sys.exit(1)
print("\nOK: gate rejects zeros and wrong-scale outputs at every tested T")
