"""
Check #5: verify dense.py's own docstring claims about spec.py's tolerance
parser (points 2 and 3 of dense.py's module docstring) are actually true,
since the module's DEFAULT_PRECISION=fp64-for-everything design decision is
justified by these specific claims. If the parser actually behaves
differently than claimed, the "fp64 default is the only precision the parsed
tolerance number matches" argument would be undermined.
"""
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")
from kernelbench import spec

for kernel, vid in [
    ("gemm", "gemm-square-kernel"),
    ("gemv", "gemv-dense-kernel"),
    ("cholesky", "cholesky-dense-single-node-fp64-kernel"),
    ("blas-level1-2", "blas-l1-vector-kernel"),
    ("blas-level1-2", "blas-l2-gemv-symv-kernel"),
]:
    v = spec.load(kernel).variant(vid)
    print(f"{kernel:16s} {vid:38s} tolerance={v.tolerance!r:>10} "
          f"provenance={v.tolerance_provenance!r}")
    print(f"    correctness text (first 160 chars): {v.correctness_text[:160]!r}")
