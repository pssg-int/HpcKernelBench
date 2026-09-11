"""
Adapter for FT-GEMM ("Anatomy of High-Performance GEMM with Online Fault
Tolerance on GPUs", ICS'23, conf/ics/WuZLHJWC23).
Repo: https://github.com/shixun404/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs.

Wraps `ft_sgemm_large` (source/kernel/ft_sgemm/include_code_gen/
ft_sgemm_large.cuh) through bridge.cu/.so (this directory, NOT part of the
artifact) -- the fused-ABFT (Algorithm-Based Fault Tolerance) "large"-tile
SGEMM kernel, WITH fault tolerance ON. This is the paper's own headline
contribution (online error detection/correction fused directly into the
GEMM main loop via row/column checksums computed on-the-fly in shared
memory -- confirmed self-contained by reading the kernel: it takes the
SAME 8-argument (M,N,K,A,B,C,alpha,beta) signature as the plain kernel, no
extra checksum-vector arguments, unlike the artifact's OTHER "abft_baseline"
config (kernel_number==10 in source/kernel/ft_sgemm/sgemm.cu) which uses
separate non-fused checksum buffers).

The plain fault-tolerance-OFF counterpart (`sgemm_large`, same tile config)
is ALSO wrapped in bridge.cu (`ftgemm_sgemm_large_run`) and was used to
verify the shared operand-layout convention, but is not registered as a
separate IMPL_NAME here -- see STATUS.md for why FT-on (not FT-off) is
this directory's canonical, registered implementation, and for the plain
kernel's own (comfortably passing) measured error.

"large" tile config (ms=ns=64, ks=8) was chosen over small/medium/tall/
wide/huge because its alignment requirement (M,N multiple of 64; K multiple
of 8) is the loosest of the five and is the ONLY one satisfied by all
three of dense.py's --smoke gemm shapes (256x256x256, 384x256x512, and the
64x64x64-per-batch-item smoke-gemm-batched-b4-64 case).

LAYOUT (verified empirically before being wired in here -- see STATUS.md's
probe transcript): the kernel's own stride arithmetic makes A COLUMN-MAJOR
(M,K) and B ROW-MAJOR (K,N); C is written COLUMN-MAJOR (M,N). Since a
column-major (M,K) array is bit-identical to a C-contiguous (row-major)
(K,M) array holding the SAME transpose, prepare() feeds `np.ascontiguousarray(A.T)`
for the "A" argument (no numeric transform, a memory-layout reinterpretation
only) and `B` completely as-is (already row-major (K,N), matching
dense.py's own operand convention exactly -- no transform needed at all)
for the "B" argument. to_host() reads the (N,M) row-major raw output back
and transposes once more to a plain (M,N) row-major result. Verified on a
random 128x192x64 shape against dense.py's own reference_gemm formula:
max_scaled_err 1.70e-7 for the plain kernel, 1.01e-4 for the FT-on kernel
(both fp32-rounding-scale; see STATUS.md for the tolerance-parsing finding
this larger-but-still-fp32-appropriate FT-on error surfaces).

PRECISION: `float` (SGEMM) throughout -- PRECISIONS = ["fp32"].

SHAPE ALIGNMENT CONSTRAINT (inherited, not introduced): no tail/boundary
guard for M/N not a multiple of 64 or K not a multiple of 8 (float4-
vectorized loads would read/write out of bounds) -- prepare() raises
NotImplementedError otherwise, same posture as turbofno-cgemm's own tile
constraint in this same directory.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "gemm"
IMPL_NAME = "ftgemm-sgemm-large"
PAPER_KEY = "conf/ics/WuZLHJWC23"
PRECISIONS = ["fp32"]  # SGEMM: float throughout

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "bridge.so")

_TILE_M = 64
_TILE_N = 64
_TILE_K = 8

# matches kernelbench/domains/dense.py's own _rng_operand/_B_OFFSET exactly
# (private helpers in that module; replicated verbatim here rather than
# importing underscore-prefixed cross-module symbols -- same convention
# turbofno's sibling adapter in this directory already uses).
_B_OFFSET = 1_000_003


def _rng_operand(rows: int, cols: int, seed: int, dtype) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(dtype)


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    FP = ctypes.POINTER(ctypes.c_float)
    lib.ftgemm_ft_sgemm_large_run.argtypes = [
        FP, FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float]
    lib.ftgemm_ft_sgemm_large_run.restype = None
    return lib


class FtGemmSgemmLarge:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME}: SGEMM kernel is `float` throughout; "
                f"requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, workload, params: dict):
        w = workload
        M, N, K, batch = int(w.M), int(w.N), int(w.K), int(w.batch)
        if M % _TILE_M or N % _TILE_N or K % _TILE_K:
            raise NotImplementedError(
                f"{IMPL_NAME}: shape M={M},N={N},K={K} not aligned to the "
                f"kernel's own tile sizes ({_TILE_M}/{_TILE_N}/{_TILE_K}) "
                f"-- ft_sgemm_large.cuh has no boundary/tail handling")

        seed = params.get("seed", w.seed)
        dtype = np.float32

        A_list, B_list, Ccm_list = [], [], []
        for b in range(batch):
            A = _rng_operand(M, K, seed + b, dtype)              # (M,K)
            B = _rng_operand(K, N, seed + _B_OFFSET + b, dtype)  # (K,N)
            # (M,K) col-major == C-contiguous (K,M) of the transpose --
            # memory-layout reinterpretation, not a numeric transform.
            A_cm = np.ascontiguousarray(A.T)   # (K,M)
            B_rm = np.ascontiguousarray(B)      # (K,N), as-is -- see docstring
            C_cm = np.zeros((N, M), dtype=dtype)  # (N,M) row-major == (M,N) col-major
            A_list.append(A_cm); B_list.append(B_rm); Ccm_list.append(C_cm)

        return {
            "A": A_list, "B": B_list, "C": Ccm_list,
            "M": M, "N": N, "K": K, "batch": batch,
        }

    def run(self, h):
        FP = ctypes.POINTER(ctypes.c_float)
        M, N, K, batch = h["M"], h["N"], h["K"], h["batch"]
        outs = []
        for b in range(batch):
            A_cm, B_rm, C_cm = h["A"][b], h["B"][b], h["C"][b]
            C_cm.fill(0.0)  # avoid reading uninitialized memory under beta=0
            self.lib.ftgemm_ft_sgemm_large_run(
                A_cm.ctypes.data_as(FP), B_rm.ctypes.data_as(FP), C_cm.ctypes.data_as(FP),
                M, N, K, ctypes.c_float(1.0), ctypes.c_float(0.0))
            outs.append(C_cm.T.copy())  # (N,M) row-major -> (M,N) row-major
        h["_out"] = np.stack(outs) if batch > 1 else outs[0]
        return h["_out"]

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        h.clear()


def create(precision: str):
    return FtGemmSgemmLarge(precision)
