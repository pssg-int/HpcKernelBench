"""
Sputnik adapter for the spmm track.

Paper: "Sparse GPU Kernels for Deep Learning" (Gale, Zaharia, Young,
Elsen), SC'20. `PAPER_KEY = conf/sc/GaleZYE20`. The paper-table's listed
artifact URL is https://github.com/luckylsk34/Sparse-Kernels; the canonical
library is https://github.com/google-research/sputnik. **Ruling (see
source.provenance and STATUS.md for the full diff): the listed repo is a
stripped, renamed, single-commit derivative of sputnik with its Apache-2.0
license notices removed -- this integration clones the canonical
google-research/sputnik repository instead**, at a pinned commit
(`bbf5840`), and wraps its `sputnik::CudaSpmm` (fp32 CSR-times-dense
overload) directly.

Selection rationale: core baseline under the revised kernel-centrality
rule -- Sputnik's SpMM/SDDMM CUDA kernels ARE this paper's headline
contribution, and every later paper this track integrates (RoDe, DTC-SpMM,
FlashSparse, SSpMM, ...) cites Sputnik as a baseline. Regime note (per the
task brief): Sputnik's OWN evaluation targets DNN-pruned weights at
moderate sparsity (ResNet-50/Transformer, V100, 49 real pruned models);
gating it here on general SuiteSparse matrices is core/mismatch-leaning
relative to its own paper's regime, same as every other artifact in this
directory -- gated anyway, a pass is a pass (ARTIFACT_GUIDE.md rule 4: the
gate is not loosened or skipped for a regime mismatch).

## What was wrapped

`sputnik::CudaSpmm` (`source/sputnik/spmm/cuda_spmm.h` /
`cuda_spmm.cu.cc`, UNMODIFIED) -- the fp32 CSR x dense overload, called with
`bias=nullptr` (source's own no-op guard, `if (bias != nullptr)`, skips the
optional bias-add/ReLU epilogue entirely, giving plain SpMM). This function
internally dispatches on `n % 4` / `n % 2` residue handling
(`cuda_spmm.cu.cc:415-497`) to pick a vectorized (float4/float2) or scalar
kernel automatically, so **every N in this track's sweep (32, 128, 256,
512) is supported directly** -- no `NotImplementedError` needed for the
dense-width dimension, unlike RoDe/SSpMM/NM-SpMM's fixed-instantiation
kernels.

Preprocessing (rule 2, timed in `prepare()`): Sputnik's own row-length
`SortedRowSwizzle` (`source/sputnik/matrix_utils.cu.cc:302-317` --
argsort rows by descending nnz-per-row so the persistent-CTA grid gets
balanced work). This adapter calls a byte-for-byte port of that exact
function (`wrapper.cu::sputnik_sorted_row_swizzle` -- see that file's
header comment for why it is ported rather than compiled from
`matrix_utils.cu.cc` directly: that file's OTHER functions need Google Glog
+ Abseil, neither vendored on this machine, for unrelated
test/benchmark-only data generation code the actual kernel and
`SortedRowSwizzle` itself never touch). This is Sputnik's real
preprocessing algorithm, reproduced exactly, not a reimplementation of the
kernel -- same precedent as `../rode/wrapper.cu`'s `row_divide_to_segment`.

Dense operand `B` drawn with `numpy.random.default_rng(seed).uniform(-1,1)`,
matching `cpu_ref.reference_spmm`'s `_dense_operand` exactly (same
RNG-match convention as every other adapter in this directory).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "sputnik-spmm-f32"
PAPER_KEY = "conf/sc/GaleZYE20"
PRECISIONS = ["fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libsputnik_wrapper.so")

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
    lib.sputnik_spmm_run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, p, p, p, p, p, p]
    lib.sputnik_spmm_run.restype = ctypes.c_int
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libsputnik_wrapper.so not built -- run build.sh"
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
    return SputnikSpMM(precision)


class SputnikSpMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp32 (sputnik::CudaSpmm's "
                f"half2 overload exists in source but is not wrapped here); "
                f"requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        lib = _load_lib()

        N = int(params["N"])
        A = matrix.csr.astype(np.float32)
        A.sort_indices()
        M, K = A.shape
        nnz = int(A.nnz)
        row_offsets = np.ascontiguousarray(A.indptr, dtype=np.int32)
        column_indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        values = np.ascontiguousarray(A.data, dtype=np.float32)

        # --- Sputnik's own preprocessing: SortedRowSwizzle, host-side,
        # ported byte-for-byte (see module docstring) ---
        row_indices = np.empty(M, dtype=np.int32)
        lib.sputnik_sorted_row_swizzle(
            M, row_offsets.ctypes.data_as(ctypes.c_void_p),
            row_indices.ctypes.data_as(ctypes.c_void_p))

        rng = np.random.default_rng(params.get("seed", 42))
        B_np = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)

        row_offsets_dev = torch.from_numpy(row_offsets).cuda()
        column_indices_dev = torch.from_numpy(column_indices).cuda()
        row_indices_dev = torch.from_numpy(row_indices).cuda()
        values_dev = torch.from_numpy(values).cuda()
        B_dev = torch.from_numpy(B_np).cuda()
        C_dev = torch.zeros((M, N), dtype=torch.float32, device="cuda")

        return {
            "M": M, "K": K, "N": N, "nnz": nnz,
            "row_indices": row_indices_dev, "values": values_dev,
            "row_offsets": row_offsets_dev, "column_indices": column_indices_dev,
            "B": B_dev, "C": C_dev,
        }

    def run(self, h):
        lib = _load_lib()
        h["C"].zero_()
        ret = lib.sputnik_spmm_run(
            h["M"], h["K"], h["N"], h["nnz"],
            ctypes.c_void_p(h["row_indices"].data_ptr()),
            ctypes.c_void_p(h["values"].data_ptr()),
            ctypes.c_void_p(h["row_offsets"].data_ptr()),
            ctypes.c_void_p(h["column_indices"].data_ptr()),
            ctypes.c_void_p(h["B"].data_ptr()),
            ctypes.c_void_p(h["C"].data_ptr()))
        if ret != 0:
            raise RuntimeError(f"{IMPL_NAME}: sputnik_spmm_run returned "
                               f"cudaError_t={ret}")
        return h["C"]

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
