"""
Check #6: demonstrate the consequence of check #5's finding concretely.
blas-l1-vector-kernel's ACTUAL parsed tolerance is 1e-4 (fBLAS's fp32
bound), not the fp64 bound (1e-6) the spec text also states and that
dense.py's DEFAULT_PRECISION="fp64" is specifically justified by
(module docstring point 3: "the one precision for which the parsed
number is guaranteed to match the spec's own intent").

Show a deliberately-corrupted fp64 dot-product result whose error sits
in the (1e-6, 1e-4) gap: the spec's own intended fp64 bound would REJECT
it, but the harness's ACTUAL (fp32-value) parsed tolerance ACCEPTS it.
"""
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")
import numpy as np
from kernelbench import spec, harness
from kernelbench.domains import dense

v = spec.load("blas-level1-2").variant("blas-l1-vector-kernel")
print("actual parsed tolerance for blas-l1-vector-kernel:", v.tolerance,
      "provenance:", v.tolerance_provenance)
print("dense.py's DEFAULT_PRECISION['blas-level1-2'] =",
      dense.DEFAULT_PRECISION["blas-level1-2"])

w = dense.DenseShape(name="probe-dot", kernel="blas-level1-2",
                      variant="blas-l1-vector-kernel", shape_class="vector",
                      n=4096, op="dot", precision="fp64", seed=42)
params = {"seed": 42, "op": "dot"}
ref, scale = dense.reference_blas12(w, params)  # CORRECTNESS_MODE=max_abs_err, scale unused

# a "buggy" fp64 dot result: correct value plus an error of 5e-5 -- BELOW the
# harness's actual (wrong, fp32-sourced) 1e-4 tolerance, but ABOVE the
# spec's own explicitly-stated fp64 bound of 1e-6
bad_out = ref + 5e-5

corr_actual = harness.check_correctness(bad_out, ref, v.tolerance, "probe",
                                        mode="max_abs_err")
corr_spec_fp64 = harness.check_correctness(bad_out, ref, 1e-6, "probe",
                                           mode="max_abs_err")
print(f"\nerror injected: 5e-5")
print(f"gate using harness's ACTUAL tolerance ({v.tolerance}): "
      f"passed={corr_actual.passed}  (value={corr_actual.value:.2e})")
print(f"gate using spec's STATED fp64 bound (1e-6):        "
      f"passed={corr_spec_fp64.passed}  (value={corr_spec_fp64.value:.2e})")
print("\n=> a fp64 blas12 result 50x worse than the spec's own fp64 bound "
      "would be silently accepted by the harness as-is, because the parser "
      "latched onto the fp32 'flteps' value from the spec's combined "
      "'1e-4 for fp32, 1e-6 for fp64' sentence." )
