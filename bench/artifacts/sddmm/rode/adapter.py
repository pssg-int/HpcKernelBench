"""
RoDe adapter for the sddmm track.

Paper: "RoDe: A Row Decomposition-based Approach for Sparse Matrix-Matrix
Multiplication on GPUs", PPoPP'24. `PAPER_KEY = conf/ppopp/PangFQZL24` (see
benchspecs/sddmm/spec.yaml's own `evidence` entry for this key: "N in
{32,128}, fp32; ITER=10 ... correctness check present but NOT compiled into
default AE binary" -- the same artifact used for the spmm track's rode
adapter, this time wrapping its RoDe_SDDMM/ kernel).

Same core-GPU-baseline rationale as the spmm/rode adapter under the revised
kernel-centrality rule (2026-09-05): row-decomposition CSR SDDMM is exactly
`sddmm-csr-kernel-f32`'s claimed common denominator ("the common denominator
most GPU papers in the track report" -- spec's own `claim` field), fp32,
general (non-block-structured) SuiteSparse input -- the spec's default-
calibrated regime.

Kernel wrapped (ARTIFACT_GUIDE.md rule 1): `RoDeSDDMM_n32` / `RoDeSDDMM_n128`
(source/RoDe_SDDMM/RoDeSddmm.cu), compiled and linked completely unmodified
into `librode_sddmm_wrapper.so` (see build.sh / wrapper.cu). `source/` here
is a SYMLINK to ../../spmm/rode/source -- read-only reuse of the spmm
track's already-cloned RoDe checkout, not a re-clone (see source.provenance).
RoDe ships exactly two dense-feature-width instantiations (K=32, K=128);
K=256/512 (also in the spec's dense-dims sweep) are not implemented by the
artifact and raise NotImplementedError from prepare() (rule 8) -- identical
constraint to the spmm/rode adapter's N in {32,128}.

Preprocessing (rule 2, timed in prepare()): the SAME row-decomposition
algorithm as the spmm/rode adapter (RoDe's SDDMM eval driver calls the
identical `RowDivide2Segment(512,4,32)` -- see wrapper.cu's header comment).
`wrapper.cu::row_divide_to_segment` is a byte-for-byte port of that
algorithm operating directly on the CSR row_offsets, for the same reason
the spmm adapter didn't link RoDe's SPC::SparseMatrix class (abseil-cpp +
Glog + gflags, unavailable on this machine, pulled in for host-side logging/
random-fill the numeric kernel never touches -- see
bench/artifacts/spmm/rode/STATUS.md for the full justification, not
re-derived here). Separately: RoDeSddmm.cu (unlike RoDeSpmm.cu) has a dead
`#include "matrix_utils.h"` that would otherwise drag in the same abseil
dependency at COMPILE time even though nothing in the file uses it --
resolved via an include-path substitution (shim_include/matrix_utils.h),
not a source patch; see build.sh and that file's header comment.

Preprocessing finding (own bug, fixed, not RoDe's kernel): RoDe's two eval
drivers use DIFFERENT SegmentLength constants for RowDivide2Segment --
eval_sddmm_f32_n32.cu: `#define SEG_LENGTH 512`; eval_sddmm_f32_n128.cu:
`#define SEG_LENGTH 32` -- each matching that variant's own kernel-internal
grid computation (RoDeSDDMMKernel_n32<...,512> vs
RoDeSDDMMKernel_n128<...,32>). An earlier version of this adapter hardcoded
512 for both K, which for K=128 produced grid dimensions covering only the
first 32 of any >32-wide row segment -- the remaining entries were silently
never written by the kernel (left at whatever `out` was pre-zeroed to),
surfacing as ~1e-1..1e0 correctness-gate failures on smoke-powerlaw
(skewed-degree rows exercise the segment path; uniform/banded's near-
constant row length happened not to trigger it at the K=128 dims tested).
Fixed in `sddmm_prepare()` (wrapper.cu): SegmentLength is now selected from
K (512 for K=32, 32 for K=128), matching RoDe's own two eval drivers
exactly. See wrapper.cu's header comment for the full derivation.

Dense operands: A (row-indexed, M x K) seed 42, B (col-indexed, N x K) seed
43 -- numpy `default_rng`, matching `cpu_ref.reference_sddmm`'s
`_dense_operand` exactly (same seed/seed+1 convention as
kernelbench.impls.gpu_cuda.TorchSDDMM/CustomSDDMM and the spmm/rode
adapter's B). RoDe's SDDMM kernel (unlike Fused3S's) already multiplies by
the sparse value S[i,j] INSIDE the kernel (`accumulator_fragment[x] *
values_tile[x]`, confirmed by reading RoDeSddmm.cu) and writes each output
directly at its own CSR position (`out + row_offset + n_idx`) -- so no
post-hoc "* S.data" or TC-block reindex step is needed here, unlike the
fused3s adapter.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "sddmm"
IMPL_NAME = "rode-sddmm"
PAPER_KEY = "conf/ppopp/PangFQZL24"
PRECISIONS = ["fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "librode_sddmm_wrapper.so")

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    lib.sddmm_prepare.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, p, p, p]
    lib.sddmm_prepare.restype = p
    lib.sddmm_run.argtypes = [p, ctypes.c_int, p, p, p, p, p]
    lib.sddmm_run.restype = ctypes.c_int
    lib.sddmm_free.argtypes = [p]
    lib.sddmm_free.restype = None
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "librode_sddmm_wrapper.so not built -- run build.sh"
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
    return RodeSDDMM(precision)


class RodeSDDMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME} only ships fp32 RoDeSDDMM_n32/RoDeSDDMM_n128 "
                f"CUDA kernels; requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        lib = _load_lib()

        K = int(params["K"])
        if K not in (32, 128):
            raise NotImplementedError(
                f"{IMPL_NAME} only ships RoDeSDDMM_n32/RoDeSDDMM_n128 (dense "
                f"feature width K in {{32, 128}} -- the only two "
                f"instantiations in RoDe_SDDMM/RoDeSddmm.cu); requested K={K}")

        S = matrix.csr
        M, N = S.shape
        nnz = int(S.nnz)
        row_offsets = np.ascontiguousarray(S.indptr, dtype=np.int32)
        col_indices = np.ascontiguousarray(S.indices, dtype=np.int32)
        values = np.ascontiguousarray(S.data, dtype=np.float32)

        # --- RoDe's own preprocessing (row-decomposition into segments),
        # timed as preprocessing per the harness contract ---
        handle = lib.sddmm_prepare(
            M, N, nnz, K,
            row_offsets.ctypes.data_as(ctypes.c_void_p),
            col_indices.ctypes.data_as(ctypes.c_void_p),
            values.ctypes.data_as(ctypes.c_void_p))
        if not handle:
            raise RuntimeError("sddmm_prepare returned NULL")

        # numpy RNG matching cpu_ref.reference_sddmm's _dense_operand
        # exactly: A row-indexed (M,K) seed, B col-indexed (N,K) seed+1.
        seed = params.get("seed", 42)
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed + 1)
        A_np = rng_a.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
        B_np = rng_b.uniform(-1.0, 1.0, size=(N, K)).astype(np.float32)
        A = torch.as_tensor(A_np, device="cuda")
        B = torch.as_tensor(B_np, device="cuda")
        out = torch.zeros(nnz, dtype=torch.float32, device="cuda")

        return {"handle": handle, "A": A, "B": B, "out": out, "K": K, "nnz": nnz}

    def run(self, h):
        lib = _load_lib()
        # RoDe's SDDMM kernels Store() directly (no atomicAdd) into `out` at
        # each nonzero's own CSR position, and RowDivide2Segment's block/
        # residue partition covers every nonzero exactly once -- so a fresh
        # zero isn't strictly required for correctness, but is kept (cheap)
        # for the same repeated-timed-reps hygiene as the spmm/rode adapter.
        h["out"].zero_()
        err = lib.sddmm_run(
            h["handle"], h["K"],
            ctypes.c_void_p(h["A"].data_ptr()),
            ctypes.c_void_p(h["B"].data_ptr()),
            ctypes.c_void_p(h["out"].data_ptr()),
            ctypes.c_void_p(0), ctypes.c_void_p(0))
        if err != 0:
            raise RuntimeError(f"RoDe SDDMM kernel returned cudaError_t={err}")
        return h["out"]

    def to_host(self, out):
        import torch
        return out.detach().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        _load_lib().sddmm_free(h["handle"])
        h.clear()
        torch.cuda.empty_cache()
