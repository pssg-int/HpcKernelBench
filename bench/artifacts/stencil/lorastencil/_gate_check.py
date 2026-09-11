"""
Standalone correctness check for lorastencil-star2d3r.

Why this exists (see STATUS.md's "Gate verification" section): star2d3r is
not one of kernelbench.domains.stencil.SMOKE's three shapes
(star2d1r/star3d1r/box3d1r), so `--smoke` cannot exercise this adapter, and
this artifact's own block-alignment constraint (grid rows a multiple of 32,
columns a multiple of 64, no tail guard -- see adapter.py's module
docstring) is not satisfied by either spec.yaml grid size (16384, 10240) at
a clean margin either. Same posture as the flashfftstencil sibling's own
`_gate_check.py`: call `kernelbench.harness.run_variant()` directly (the
same call the runner CLI makes) against a small, 32/64-aligned synthetic
star2d3r workload, using the exact `stencil-cpu-gpu-kernel-fp64` variant
object the CLI would load.

Usage:
    $PY _gate_check.py   # PY from bench/env.sh, run from this directory
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # -> bench/

from kernelbench import spec
from kernelbench import harness
from kernelbench.domains import stencil

adapter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter.py")
_spec_mod = importlib.util.spec_from_file_location("lorastencil_adapter", adapter_path)
adapter = importlib.util.module_from_spec(_spec_mod)
_spec_mod.loader.exec_module(adapter)

ok, reason = adapter.available()
print("available:", ok, reason)
if not ok:
    raise SystemExit(1)

impl = adapter.create("fp64")

sp = spec.load("stencil")
variant = sp.variant("stencil-cpu-gpu-kernel-fp64")
wl = stencil.StencilWorkload(name="star2d3r-gate-128", kind="star", dims=2,
                              radius=3, grid_shape=(128, 128), timesteps=3,
                              precision="fp64")
r = harness.run_variant(impl, wl, variant, {"seed": 42, "precision": "fp64"},
                         reference=stencil.REFERENCES["stencil"],
                         correctness_mode=stencil.CORRECTNESS_MODE["stencil"],
                         reference_name="numpy fp64 periodic-wrap sweep",
                         warmup_override=1, reps_override=3)

print("valid:", r.correctness.passed if r.correctness else None)
print("correctness:", r.correctness)
print("warnings:", r.warnings)
print("workload after prepare (mutation check): timesteps=", wl.timesteps)

raise SystemExit(0 if (r.correctness and r.correctness.passed) else 1)
