"""
Standalone interior-cropped correctness check for spider-box2d7r-sptc.

The standard harness gate (kernelbench.runner --smoke) compares the FULL
output array against the periodic-wrap reference, which necessarily fails
in the outer radius=7 boundary band (SPIDER pads with a fixed zero halo,
never wraps -- see adapter.py's module docstring, point "BOUNDARY"). This
script isolates whether the KERNEL ITSELF is correct in the region where
the two boundary conventions provably agree (any output cell whose full
15x15 footprint stays inside the interior reads identical input values
under either convention).

Not a replacement for the mandated gate -- a supplementary measurement,
same pattern cb-spmv's STATUS.md documents using for an extra independent
check. Both numbers (full-array and interior-cropped) are reported in
STATUS.md.
"""
import os
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # -> bench/
import numpy as np

from kernelbench import domains
from kernelbench.domains import stencil as stdom

adapter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter.py")
import importlib.util
spec = importlib.util.spec_from_file_location("spider_adapter", adapter_path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

ok, reason = adapter.available()
assert ok, reason

# smallest grid that is a multiple of both BLOCK_ROW=64 and BLOCK_COL=128
wl = stdom.StencilWorkload(name="spider-native-check", kind="star", dims=2,
                            radius=1, grid_shape=(64, 128), timesteps=5)

impl = adapter.create("fp16")
params = {"seed": 42, "precision": "fp16"}

handle = impl.prepare(wl, params)   # mutates wl in place to SPIDER's native shape/weights/T=1
out = impl.run(handle)
got = impl.to_host(out)              # (64,128) float64, full array incl. boundary band

ref, scale = stdom.reference_stencil(wl, params)  # uses the SAME mutated wl -> matching ground truth

r = 7  # wl.radius after prepare()'s override
full_diff = np.abs(got - ref)
full_scaled = full_diff / np.maximum(scale, 1e-300)
print(f"FULL array   : max_scaled_err = {full_scaled.max():.6e}  "
      f"(at {np.unravel_index(full_scaled.argmax(), full_scaled.shape)})")

interior = (slice(r, -r), slice(r, -r))
int_diff = full_diff[interior]
int_scale = scale[interior]
int_scaled = int_diff / np.maximum(int_scale, 1e-300)
print(f"INTERIOR crop: max_scaled_err = {int_scaled.max():.6e}  "
      f"(shape {got[interior].shape}, r={r} cropped off each edge)")

print(f"grid_shape={wl.grid_shape}, radius={wl.radius}, kind={wl.kind}, "
      f"timesteps={wl.timesteps}, weight_sum={sum(wl.weights.values()):.1f}")

impl.free(handle)
