"""
Standalone correctness check for flashfftstencil-box2d1r.

Why this exists (see STATUS.md's "Gate verification" section): the mandated
`kernelbench.runner --matrices box2d1r` command is not usable as-is because
`load_workload("box2d1r")` has no CLI way to override `grid_shape`, and its
spec-sized default (16384, or 10240 for variant 2) is not a multiple of 6 --
this artifact's own overlap-add tiling constraint (`INPUT_WIDTH` must be a
multiple of `sub_input_width = unit-(KERNEL_WIDTH-1) = 6`, see adapter.py's
module docstring and STATUS.md's "Shape constraint" section). `--smoke` is
also not usable: this adapter only implements box2d1r (no star2d1r, no 3D),
so none of the 3 `--smoke` shapes are supported.

This script calls `kernelbench.harness.run_variant()` directly (same
underlying call the runner CLI makes) against a small, explicitly
6-divisible synthetic box2d1r workload, using the exact
`stencil-cpu-gpu-kernel-fp64` variant object the CLI would load -- so this
is the real harness correctness gate, just pointed at a workload the CLI
itself cannot construct.

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
_spec_mod = importlib.util.spec_from_file_location("flashfftstencil_adapter", adapter_path)
adapter = importlib.util.module_from_spec(_spec_mod)
_spec_mod.loader.exec_module(adapter)

ok, reason = adapter.available()
print("available:", ok, reason)
if not ok:
    raise SystemExit(1)

impl = adapter.create("fp64")

sp = spec.load("stencil")
variant = sp.variant("stencil-cpu-gpu-kernel-fp64")
wl = stencil.StencilWorkload(name="box2d1r-gate-96", kind="box", dims=2,
                              radius=1, grid_shape=(96, 96), timesteps=1,
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
