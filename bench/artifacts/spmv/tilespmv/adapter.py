"""
Adapter for TileSpMV (IPDPS 2021, "TileSpMV: A Tiled Algorithm for Sparse
Matrix-Vector Multiplication on GPUs", conf/ipps/NiuLDJ0T21).

Wraps the artifact's own tiled-format construction (Tile_create,
source/src/csr2tile.h) and its own GPU kernels (stir_spmv_cuda_kernel_v5/v6,
source/src/tilespmv_cuda.h) plus its vendored CSR5 deferred-COO handler
(source/src/external/CSR5_cuda/) through bridge.cu/.so (this directory) --
see that file's docstring for exactly what had to be split out of the
artifact's monolithic main.cu/call_tilespmv_cuda() driver (which loads a
.mtx file itself and times a WARMUP_NUM + BENCH_REPEAT loop internally) and
why, plus a genuine finding: the artifact's own FINAL timed loop silently
drops the CSR5 deferred-COO contribution that its warmup loop includes,
which this bridge does not reproduce (see bridge.cu's "the CSR5 correction"
section).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmv"
IMPL_NAME = "tilespmv"
PAPER_KEY = "conf/ipps/NiuLDJ0T21"
PRECISIONS = ["fp64"]  # artifact's MAT_VAL_TYPE (common.h) is `double`

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "bridge.so")


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    p = ctypes.c_void_p
    lib.tilespmv_prepare.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.tilespmv_prepare.restype = p
    lib.tilespmv_run.argtypes = [p, p]
    lib.tilespmv_run.restype = None
    lib.tilespmv_copy_y.argtypes = [p, ctypes.POINTER(ctypes.c_double)]
    lib.tilespmv_copy_y.restype = None
    lib.tilespmv_free.argtypes = [p]
    lib.tilespmv_free.restype = None
    return lib


class TileSpMV:
    name = "tilespmv"
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"tilespmv's MAT_VAL_TYPE (common.h) is `double`; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, matrix, params: dict):
        # Tile_create (csr2tile.h) reads CSR directly -- no COO conversion
        # needed (unlike cb-spmv). This call performs the artifact's own
        # tiled-format construction + row-balance bookkeeping (tilespmv_cpu)
        # + every device malloc/H2D copy call_tilespmv_cuda() would do + the
        # CSR5 handle build for any deferred-COO remainder + the one-shot
        # v5 schedule-building launch -- all genuinely one-time preprocessing
        # under this variant's protocol, timed once here.
        A = matrix.csr
        A.sort_indices()
        rowptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        colidx = np.ascontiguousarray(A.indices, dtype=np.int32)
        val = np.ascontiguousarray(A.data, dtype=np.float64)
        rows, cols = A.shape

        rng = np.random.default_rng(params.get("seed", 42))
        x = np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=cols).astype(np.float64))

        handle = self.lib.tilespmv_prepare(
            rows, cols, int(A.nnz),
            rowptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            colidx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if not handle:
            raise RuntimeError("tilespmv_prepare returned a null handle")
        y_host = np.zeros(rows, dtype=np.float64)
        # keep rowptr/colidx/val/x alive: ctypes doesn't, and
        # tilespmv_prepare only reads them synchronously (no async H2D), so
        # this is precautionary (same pattern as artifacts/spmv/cb-spmv).
        return {"handle": handle, "rows": rows, "y_host": y_host,
                "_rowptr": rowptr, "_colidx": colidx, "_val": val, "_x": x}

    def run(self, h):
        # exactly one y=Ax call (tile kernel + CSR5 deferred-COO correction
        # if this matrix has any -- see bridge.cu), stream=0 (the default/
        # legacy stream -- same stream torch.cuda.Event() records on by
        # default, so CudaEventTimer correctly brackets this even though the
        # kernels are launched via raw ctypes, not torch).
        self.lib.tilespmv_run(h["handle"], ctypes.c_void_p(0))
        return h

    def to_host(self, out) -> np.ndarray:
        self.lib.tilespmv_copy_y(
            out["handle"], out["y_host"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out["y_host"].copy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        self.lib.tilespmv_free(h["handle"])
        h.clear()


def create(precision: str):
    return TileSpMV(precision)
