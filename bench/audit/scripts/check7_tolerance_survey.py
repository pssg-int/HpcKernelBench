"""
Check #7: survey the parsed tolerance for every variant actually wired to a
CORRECTNESS_MODE/DEFAULT_PRECISION in sparse.py, dense.py, stencil.py,
spectral.py, to see whether the "first number after a comparison operator"
parse picks the precision-appropriate value or a mismatched one, beyond the
blas-l1-vector-kernel case already confirmed buggy in check5/6.
"""
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")
from kernelbench import spec

checks = [
    ("spmv", "spmv-csr-kernel", "fp64"),
    ("spmm", "spmm-gpu-kernel-f32", "fp32"),
    ("sddmm", "sddmm-csr-kernel-f32", "fp32"),
    ("gemm", "gemm-square-kernel", "fp64 (DEFAULT_PRECISION)"),
    ("gemv", "gemv-dense-kernel", "fp64 (DEFAULT_PRECISION)"),
    ("stencil", "stencil-cpu-gpu-kernel-fp64", "fp64"),
    ("fft", "fft-1d-batched-kernel", "fp32 (DEFAULT_PRECISION)"),
    ("ntt", "ntt-kernel-isolated", "int64/exact"),
]
for kernel, vid, default_prec in checks:
    v = spec.load(kernel).variant(vid)
    print(f"{kernel:10s} {vid:32s} default_precision={default_prec:26s} "
          f"tolerance={v.tolerance!r:>10} prov={v.tolerance_provenance!r}")
