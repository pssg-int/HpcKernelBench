"""
Check #9: complete the reference-independence survey for dense.py
(numpy-gemm / numpy-gemv / numpy-blas12 vs their references, at fp64
default) to confirm the same bit-identical pattern found in sparse.py,
stencil.py and spectral.py (checks #2 and #8) also holds here.
"""
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")
import numpy as np
from kernelbench.domains import dense

w = dense.DenseShape(name="probe-gemm", kernel="gemm", variant="probe",
                      shape_class="square", M=64, N=64, K=64,
                      precision="fp64", seed=42)
params = {"seed": 42}
ref, scale = dense.reference_gemm(w, params)
impl = dense.NumpyGemm(precision="fp64")
h = impl.prepare(w, params)
out = impl.to_host(impl.run(h))
print("numpy-gemm (fp64) vs reference_gemm (fp64):",
      f"max|out-ref|={np.abs(out-ref).max():.3e}",
      f"bit-identical={np.array_equal(out, ref)}")

wv = dense.DenseShape(name="probe-gemv", kernel="gemv", variant="probe",
                       shape_class="square", M=1024, N=1, K=1024,
                       precision="fp64", seed=42)
refv, scalev = dense.reference_gemv(wv, params)
implv = dense.NumpyGemv(precision="fp64")
hv = implv.prepare(wv, params)
outv = implv.to_host(implv.run(hv))
print("numpy-gemv (fp64) vs reference_gemv (fp64):",
      f"max|out-ref|={np.abs(outv-refv).max():.3e}",
      f"bit-identical={np.array_equal(outv, refv)}")

wb = dense.DenseShape(name="probe-blas", kernel="blas-level1-2", variant="probe",
                       shape_class="vector", n=4096, op="dot",
                       precision="fp64", seed=42)
paramsb = {"seed": 42, "op": "dot"}
refb, scaleb = dense.reference_blas12(wb, paramsb)
implb = dense.NumpyBlas12(precision="fp64")
hb = implb.prepare(wb, paramsb)
outb = implb.to_host(implb.run(hb))
print("numpy-blas12/dot (fp64) vs reference_blas12 (fp64):",
      f"max|out-ref|={np.abs(outb-refb).max():.3e}",
      f"bit-identical={np.array_equal(outb, refb)}")
