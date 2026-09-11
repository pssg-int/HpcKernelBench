"""
Standalone interior-cropped correctness check for an5d-stencil, all 6 shapes.

The mandated harness gate (kernelbench.runner --smoke) compares the FULL
output array against the periodic-wrap reference, which necessarily fails
near the domain edge: AN5D's own common.h::init_grid fills a FIXED halo
ONCE and the generated sweep never refreshes it (see adapter.py's module
docstring, point 2 "BOUNDARY"), unlike the harness's reference, which
re-wraps every sweep. A boundary-driven discrepancy can only propagate
`radius` cells inward per sweep (local stencil support), so after
`timesteps` sweeps, cropping `radius*timesteps` cells off each edge isolates
whether the KERNEL ARITHMETIC itself (not the boundary convention) is
correct. Not a replacement for the mandated gate -- a supplementary
measurement, same pattern already used by the spider/lorastencil siblings'
own STATUS.md. All numbers here are reported in STATUS.md.

Also exercises the 3 shapes NOT in kernelbench.domains.stencil.SMOKE
(box2d1r, star2d3r, box2d3r) directly via run_variant()-equivalent calls on
a small custom workload, matching lorastencil's own precedent for a shape
not covered by --smoke.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # -> bench/
import numpy as np

from kernelbench.domains import stencil as stdom

adapter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter.py")
spec = importlib.util.spec_from_file_location("an5d_adapter", adapter_path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

ok, reason = adapter.available()
assert ok, reason

CASES = [
    # (label, kind, dims, radius, grid_shape, timesteps)
    ("smoke-star2d1r (mandated --smoke shape)", "star", 2, 1, (256, 256), 5),
    ("smoke-star3d1r (mandated --smoke shape)", "star", 3, 1, (32, 32, 32), 5),
    ("smoke-box3d1r  (mandated --smoke shape)", "box", 3, 1, (24, 24, 24), 5),
    ("box2d1r  (not in --smoke; custom grid)", "box", 2, 1, (64, 64), 5),
    ("star2d3r (not in --smoke; custom grid)", "star", 2, 3, (64, 64), 5),
    ("box2d3r  (not in --smoke; custom grid)", "box", 2, 3, (64, 64), 5),
]

all_pass = True
for label, kind, dims, radius, grid_shape, T in CASES:
    wl = stdom.StencilWorkload(name=f"an5d-check-{kind}{dims}d{radius}r", kind=kind,
                                dims=dims, radius=radius, grid_shape=grid_shape, timesteps=T)
    impl = adapter.create("fp64")
    params = {"seed": 42, "precision": "fp64"}

    handle = impl.prepare(wl, params)  # mutates wl.offsets/weights to AN5D's native coefficients
    out = impl.run(handle)
    got = impl.to_host(out)

    ref, scale = stdom.reference_stencil(wl, params)  # uses the SAME mutated wl -> matching ground truth

    full_diff = np.abs(got - ref)
    full_scaled = full_diff / np.maximum(scale, 1e-300)
    full_err = full_scaled.max()

    crop = radius * T
    interior_shape = tuple(s - 2 * crop for s in grid_shape)
    if min(interior_shape) > 0:
        sl = tuple(slice(crop, s - crop) for s in grid_shape)
        int_scaled = full_diff[sl] / np.maximum(scale[sl], 1e-300)
        int_err = int_scaled.max()
    else:
        int_err = None

    tol = 1e-5  # stencil-cpu-gpu-kernel-fp64 variant's correctness tolerance (spec.yaml)
    verdict = "PASS" if (int_err is not None and int_err <= tol) else "FAIL/N-A"
    if int_err is None or int_err > tol:
        all_pass = False
    print(f"{label:42s} full_err={full_err:.3e}  "
          f"interior_err(crop={crop:3d}, shape={interior_shape})={int_err!s:>12s}  "
          f"tol={tol}  {verdict}")

    impl.free(handle)

print()
print("ALL INTERIOR CHECKS PASS" if all_pass else "SOME INTERIOR CHECKS DID NOT PASS")
