"""
RoDe adapter for the spmm track.

Paper: "RoDe: A Row Decomposition-based Approach for Sparse Matrix-Matrix
Multiplication on GPUs", PPoPP'24. `PAPER_KEY = conf/ppopp/PangFQZL24` (see
../../../benchspecs/spmm/spec.yaml's own `evidence` entry for this key).
Artifact: https://github.com/CRAFT-THU/RoDe -- the canonical row-decomposition
CSR SpMM/SDDMM baseline this track's whole GPU-SpMM literature cites (DTC-SpMM,
FlashSparse, Voltrix, SSpMM, MP-SpMM all compare against it).

Selected as a core general-GPU-SpMM baseline under the revised
kernel-centrality rule (2026-09-05): row-decomposition CSR SpMM is exactly
`spmm-gpu-kernel-f32`'s claimed common denominator ("The common denominator
most GPU SpMM papers (RoDe, GE-SpMM, SMaT, DTC-SpMM's inner kernel) actually
report" -- benchspecs/spmm/spec.yaml), fp32, general (non-block-structured)
SuiteSparse input -- the spec's own default-calibrated regime, not a tiebreak
pick.

Kernel wrapped (ARTIFACT_GUIDE.md rule 1): `RoDeSpmm_n32` / `RoDeSpmm_n128`
(source/RoDe_SpMM/RoDeSpmm.cu -- RoDe's actual CUDA SpMM kernel), compiled and
linked completely unmodified into `librode_wrapper.so` (see build.sh /
wrapper.cu). RoDe ships exactly two dense-width instantiations (N=32, N=128);
N=256/512 (also in the spec's dense-dims sweep) are not implemented by the
artifact and raise NotImplementedError from prepare() (rule 8).

Preprocessing (rule 2, timed in prepare()): RoDe's own row-decomposition --
`SparseMatrix::RowDivide2Segment` (source/utils/matrix_utils.cu:989-1041) --
partitions each row's nonzeros into <=512-element contiguous segments (the
paper's actual contribution: balanced work per persistent CTA regardless of
per-row nnz skew). `wrapper.cu::row_divide_to_segment` is a byte-for-byte
port of that exact algorithm operating directly on our CSR row_offsets (see
wrapper.cu's header comment for why the class it originally lives in --
SPC::SparseMatrix, which additionally drags in abseil-cpp + Glog + gflags for
unrelated host-side logging/random-fill code the numeric kernel never uses --
is not linked). This is RoDe's preprocessing ALGORITHM reproduced exactly,
not a reimplementation of the kernel; the actual SpMM computation always
calls RoDe's own compiled RoDeSpmm_n32/n128.

Dense operand B: numpy `default_rng` matching `cpu_ref.reference_spmm`'s
`_dense_operand` exactly (see insum/adapter.py's docstring for why this
matters -- gpu_cuda.py's torch-RNG `_dense` helper draws a bit-different B
than the gate's numpy-RNG reference).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "rode-spmm"
PAPER_KEY = "conf/ppopp/PangFQZL24"
PRECISIONS = ["fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "librode_wrapper.so")

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    lib.rode_prepare.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, p, p, p]
    lib.rode_prepare.restype = p
    lib.rode_run.argtypes = [p, ctypes.c_int, p, p, p, p]
    lib.rode_run.restype = ctypes.c_int
    lib.rode_free.argtypes = [p]
    lib.rode_free.restype = None
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "librode_wrapper.so not built -- run build.sh"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        _load_lib()
    except Exception as e:
        return False, f"ctypes load failed: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return RodeSpMM(precision)


class RodeSpMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME} only ships fp32 RoDeSpmm_n32/RoDeSpmm_n128 "
                f"CUDA kernels (RoDe's double-precision overloads exist in "
                f"source but are never instantiated by its own eval driver "
                f"and are not wired here); requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        lib = _load_lib()

        N = int(params["N"])
        if N not in (32, 128):
            raise NotImplementedError(
                f"{IMPL_NAME} only ships RoDeSpmm_n32/RoDeSpmm_n128 (dense "
                f"width N in {{32, 128}} -- the only two instantiations in "
                f"RoDe_SpMM/RoDeSpmm.cu); requested N={N}")

        A = matrix.csr
        M, K = A.shape
        nnz = int(A.nnz)
        row_offsets = np.ascontiguousarray(A.indptr, dtype=np.int32)
        col_indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        values = np.ascontiguousarray(A.data, dtype=np.float32)

        # --- RoDe's own preprocessing (row-decomposition into segments),
        # timed as preprocessing per the harness contract ---
        handle = lib.rode_prepare(
            M, K, nnz,
            row_offsets.ctypes.data_as(ctypes.c_void_p),
            col_indices.ctypes.data_as(ctypes.c_void_p),
            values.ctypes.data_as(ctypes.c_void_p))
        if not handle:
            raise RuntimeError("rode_prepare returned NULL")

        # numpy RNG matching cpu_ref.reference_spmm's _dense_operand exactly.
        rng = np.random.default_rng(params.get("seed", 42))
        B_np = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)
        B = torch.as_tensor(B_np, device="cuda")
        C = torch.zeros((M, N), dtype=torch.float32, device="cuda")

        return {"handle": handle, "B": B, "C": C, "N": N, "M": M}

    def run(self, h):
        lib = _load_lib()
        # RoDe's kernels atomicAdd into C for rows split across segments /
        # a residue tail -- zero before every call (same convention as the
        # insum/inferfast adapters) so repeated timed reps each compute a
        # fresh C = A@B, not a running sum.
        h["C"].zero_()
        # Both of RoDe's kernels (segmented-rows + residue) launched on the
        # DEFAULT stream (stream=0), not the two real CUDA streams RoDe's own
        # eval driver uses for concurrency: torch's CudaEventTimer records
        # its start/stop events on torch's current (default) stream, and
        # torch.cuda.Stream() objects are created CUDA-non-blocking, so
        # kernels issued on them would not be guaranteed ordered against
        # those events -- a correctness/timing hazard, not worth RoDe's own
        # intra-call stream-overlap optimization here. Both kernels still
        # run exactly as compiled, just serialized on stream 0 instead of
        # overlapped across two streams.
        err = lib.rode_run(
            h["handle"], h["N"],
            ctypes.c_void_p(h["B"].data_ptr()),
            ctypes.c_void_p(h["C"].data_ptr()),
            ctypes.c_void_p(0), ctypes.c_void_p(0))
        if err != 0:
            raise RuntimeError(f"RoDe kernel returned cudaError_t={err}")
        return h["C"]

    def to_host(self, out):
        import torch
        return out.detach().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        _load_lib().rode_free(h["handle"])
        h.clear()
        torch.cuda.empty_cache()
