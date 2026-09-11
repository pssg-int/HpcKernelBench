"""
Check #2 (REFERENCE INDEPENDENCE, sparse.py): at the domain's own
DEFAULT_PRECISION for spmv (fp64), does "scipy-csr-spmv" (ScipySpMV.run =
`A @ x` via scipy's CSR matmul) differ AT ALL from reference_spmv (also
`A @ x` via scipy's CSR matmul, at fp64) -- i.e. is the correctness gate for
that implementation actually checking scipy against itself, not against an
independent computation?

Contrast against NaiveSpMV (a genuinely different, hand-written Python-loop
code path) to show the gate DOES do real work for that implementation.

CPU only.
"""
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")

import numpy as np
from kernelbench.domains import sparse
from kernelbench.impls import cpu_ref
from kernelbench import matrices

m = matrices.synthetic("probe-uniform", rows=2000, cols=2000, nnz_per_row=16,
                        pattern="uniform")
params = {"seed": 42, "precision": "fp64"}

ref, scale = cpu_ref.reference_spmv(m, params)

# ScipySpMV at the domain's DEFAULT_PRECISION for spmv (fp64)
print("DEFAULT_PRECISION[spmv] =", sparse.DEFAULT_PRECISION["spmv"])
impl = cpu_ref.ScipySpMV(precision=sparse.DEFAULT_PRECISION["spmv"])
h = impl.prepare(m, params)
out = impl.to_host(impl.run(h))

diff = np.abs(out.astype(np.float64) - ref)
print(f"scipy-csr-spmv  vs reference_spmv (both fp64, scipy CSR @ x):")
print(f"  max|out-ref|      = {diff.max():.3e}")
print(f"  bit-identical?    = {np.array_equal(out.astype(np.float64), ref)}")
print(f"  identical x used? = {np.array_equal(h['x'].astype(np.float64) if h['x'].dtype!=np.float64 else h['x'], h['x'])}")

# now the naive, hand-written CSR loop -- a genuinely different code path
impl2 = cpu_ref.NaiveSpMV(precision="fp64")
h2 = impl2.prepare(m, params)
out2 = impl2.to_host(impl2.run(h2))
diff2 = np.abs(out2.astype(np.float64) - ref)
print(f"\nnaive-csr-spmv (hand Python loop) vs reference_spmv:")
print(f"  max|out-ref|   = {diff2.max():.3e}")
print(f"  bit-identical? = {np.array_equal(out2.astype(np.float64), ref)}")

# Now show what happens at fp32 (the other precision option) -- still scipy
# vs scipy, just at different width, so it's a *rounding* check only.
impl3 = cpu_ref.ScipySpMV(precision="fp32")
h3 = impl3.prepare(m, {"seed": 42, "precision": "fp32"})
out3 = impl3.to_host(impl3.run(h3))
ref32, scale32 = cpu_ref.reference_spmv(m, {"seed": 42, "precision": "fp32"})
diff3 = np.abs(out3.astype(np.float64) - ref32)
print(f"\nscipy-csr-spmv at fp32 vs fp64 reference_spmv:")
print(f"  max|out-ref| = {diff3.max():.3e}  (nonzero because of rounding only,"
      " not algorithmic independence)")

# Same check for spmm (default precision fp32) and sddmm
print("\n--- spmm ---")
print("DEFAULT_PRECISION[spmm] =", sparse.DEFAULT_PRECISION["spmm"])
implA = cpu_ref.ScipySpMM(precision="fp64")  # force fp64 to test same-precision case
hA = implA.prepare(m, {"seed": 42, "N": 32})
outA = implA.to_host(implA.run(hA))
refA, scaleA = cpu_ref.reference_spmm(m, {"seed": 42, "N": 32, "precision": "fp64"})
diffA = np.abs(outA.astype(np.float64) - refA)
print(f"scipy-csr-spmm at fp64 vs reference_spmm (fp64): max|out-ref|={diffA.max():.3e}, "
      f"bit-identical={np.array_equal(outA.astype(np.float64), refA)}")
