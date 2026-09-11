"""
Sputnik adapter for the sddmm track.

Paper: "Sparse GPU Kernels for Deep Learning" (Gale, Zaharia, Young,
Elsen), SC'20. `PAPER_KEY = conf/sc/GaleZYE20`. See
`../../spmm/sputnik/adapter.py`'s module docstring for the artifact-URL
ruling (canonical google-research/sputnik cloned instead of the paper
table's listed fork -- stripped Apache-2.0 notices, see
`source.provenance`/STATUS.md) and the general regime note (Sputnik's own
DNN-pruned-weight evaluation regime vs. this track's general SuiteSparse
gate -- core/mismatch-leaning, gated anyway per ARTIFACT_GUIDE.md rule 4).
`source/` here is a SYMLINK to `../../spmm/sputnik/source` (same convention
as `artifacts/sddmm/rode/source -> ../../spmm/rode/source`).

## What was wrapped

`sputnik::CudaSddmm` (`source/sputnik/sddmm/cuda_sddmm.h` /
`cuda_sddmm.cu.cc`, UNMODIFIED) -- fp32, computes
`output_values[nz] = dot(lhs_matrix[row(nz)], rhs_matrix[col(nz)])` at each
stored (row,col) position of the given sparsity pattern. Dispatches
internally on `k % 4` / `k % 2` (`cuda_sddmm.cu.cc:169-195`), so every K in
this track's sweep is supported directly -- no `NotImplementedError` needed.

Sputnik's own signature has NO separate "sparse value" operand at all (it
computes a pure pattern-restricted dot product); this track's
`cpu_ref.reference_sddmm` (and every other sddmm adapter already in this
directory -- `kernelbench.impls.gpu_cuda.TorchSDDMM`/`CustomSDDMM`)
elementwise-multiplies the dot product by the ORIGINAL matrix's stored
value afterward (`out = dot(A[i],B[j]) * S.data[nz]`,
`operation: "P[i,j] = S[i,j] * dot(A[i,:], B[j,:])"` per
`benchspecs/sddmm/spec.yaml`). This adapter follows the SAME established
convention: `run()` calls Sputnik's raw kernel, then multiplies by the
matrix's own values on-device -- matching `TorchSDDMM.run()`'s
`out.values() * h["Sdata"]` exactly (same convention, same reasoning: the
harness's reference bakes this scaling in, so every adapter timed against
it must too).

Preprocessing: Sputnik's `SortedRowSwizzle`, ported byte-for-byte in
`wrapper.cu` (same port as `../../spmm/sputnik/wrapper.cu`, duplicated
rather than shared to keep this directory's build self-contained -- see
that file's header comment for why it is ported rather than compiled from
`matrix_utils.cu.cc`).

Dense operands: `A` (M x K) seed 42, `B` (N x K) seed 43 --
`numpy.random.default_rng`, matching `cpu_ref.reference_sddmm`'s
`_dense_operand` seed/seed+1 convention exactly (same as
`../rode/adapter.py`'s A/B and `gpu_cuda.TorchSDDMM`/`CustomSDDMM`).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "sddmm"
IMPL_NAME = "sputnik-sddmm-f32"
PAPER_KEY = "conf/sc/GaleZYE20"
PRECISIONS = ["fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libsputnik_sddmm_wrapper.so")

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    lib.sputnik_sorted_row_swizzle.argtypes = [ctypes.c_int, p, p]
    lib.sputnik_sorted_row_swizzle.restype = None
    lib.sputnik_sddmm_run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, p, p, p, p, p, p]
    lib.sputnik_sddmm_run.restype = ctypes.c_int
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libsputnik_sddmm_wrapper.so not built -- run build.sh"
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
    return SputnikSDDMM(precision)


class SputnikSDDMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp32 (sputnik::CudaSddmm is "
                f"fp32-only in the wrapped overload); requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        lib = _load_lib()

        K = int(params["K"])
        S = matrix.csr.astype(np.float32)
        S.sort_indices()
        M, N = S.shape
        nnz = int(S.nnz)
        row_offsets = np.ascontiguousarray(S.indptr, dtype=np.int32)
        column_indices = np.ascontiguousarray(S.indices, dtype=np.int32)
        values = np.ascontiguousarray(S.data, dtype=np.float32)

        row_indices = np.empty(M, dtype=np.int32)
        lib.sputnik_sorted_row_swizzle(
            M, row_offsets.ctypes.data_as(ctypes.c_void_p),
            row_indices.ctypes.data_as(ctypes.c_void_p))

        seed = params.get("seed", 42)
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed + 1)
        A_np = rng_a.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
        B_np = rng_b.uniform(-1.0, 1.0, size=(N, K)).astype(np.float32)

        row_offsets_dev = torch.from_numpy(row_offsets).cuda()
        column_indices_dev = torch.from_numpy(column_indices).cuda()
        row_indices_dev = torch.from_numpy(row_indices).cuda()
        values_dev = torch.from_numpy(values).cuda()   # S's own values (post-multiply, see run())
        A_dev = torch.from_numpy(A_np).cuda()
        B_dev = torch.from_numpy(B_np).cuda()
        out_dev = torch.zeros(nnz, dtype=torch.float32, device="cuda")

        return {
            "M": M, "K": K, "N": N, "nnz": nnz,
            "row_indices": row_indices_dev, "row_offsets": row_offsets_dev,
            "column_indices": column_indices_dev,
            "A": A_dev, "B": B_dev, "values": values_dev, "out": out_dev,
        }

    def run(self, h):
        lib = _load_lib()
        h["out"].zero_()
        ret = lib.sputnik_sddmm_run(
            h["M"], h["K"], h["N"], h["nnz"],
            ctypes.c_void_p(h["row_indices"].data_ptr()),
            ctypes.c_void_p(h["row_offsets"].data_ptr()),
            ctypes.c_void_p(h["column_indices"].data_ptr()),
            ctypes.c_void_p(h["A"].data_ptr()),
            ctypes.c_void_p(h["B"].data_ptr()),
            ctypes.c_void_p(h["out"].data_ptr()))
        if ret != 0:
            raise RuntimeError(f"{IMPL_NAME}: sputnik_sddmm_run returned "
                               f"cudaError_t={ret}")
        # Sputnik's own kernel computes only the raw dot product (no
        # "values" operand exists in its signature); this harness's
        # reference multiplies by the ORIGINAL matrix's stored value
        # (see module docstring) -- same convention as
        # gpu_cuda.TorchSDDMM/CustomSDDMM, applied here identically.
        return h["out"] * h["values"]

    def to_host(self, out):
        import torch
        return out.detach().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
