"""
Adapter for CB-SpMV (ICS 2025, "CB-SpMV: A Data Aggregating and Balance
Algorithm for Cache-Friendly Block-Based SpMV on GPUs",
conf/ics/CongSC00Q25).

Wraps the artifact's own kernel (cb_spmv_detail::spmv_cuda_kernel_comp /
_gather, source/cb-spmv/src/cb-spmv.cuh) and its own cache-block format
construction (coo2block_gather, source/cb-spmv/src/coo2block.h) through
bridge.cu/.so (this directory) -- see that file's docstring for exactly
what had to be split out of the artifact's monolithic `cb_spmv()` driver
(which bundles format construction + an ITER=10 batched-loop timing
measurement together) and why.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmv"
IMPL_NAME = "cb-spmv"
PAPER_KEY = "conf/ics/CongSC00Q25"
PRECISIONS = ["fp64"]  # artifact's ValType (macros.h) is `double`

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
    lib.cbspmv_prepare.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.cbspmv_prepare.restype = p
    lib.cbspmv_run.argtypes = [p, p]
    lib.cbspmv_run.restype = None
    lib.cbspmv_copy_y.argtypes = [p, ctypes.POINTER(ctypes.c_double)]
    lib.cbspmv_copy_y.restype = None
    lib.cbspmv_free.argtypes = [p]
    lib.cbspmv_free.restype = None
    return lib


class CbSpMV:
    name = "cb-spmv"
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"cb-spmv's ValType (macros.h) is `double`; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, matrix, params: dict):
        # This call performs the artifact's own cache-block format
        # construction (coo2block_gather, CPU/OpenMP) plus every device
        # malloc/H2D copy cb_spmv() would do -- exactly the "H2D + the
        # artifact's format construction" this contract asks prepare() to
        # cover, timed once as preprocessing.
        A = matrix.csr.tocoo()
        row = np.ascontiguousarray(A.row, dtype=np.int32)
        col = np.ascontiguousarray(A.col, dtype=np.int32)
        val = np.ascontiguousarray(A.data, dtype=np.float64)
        rows, cols = A.shape

        rng = np.random.default_rng(params.get("seed", 42))
        x = np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=cols).astype(np.float64))

        handle = self.lib.cbspmv_prepare(
            rows, cols, int(A.nnz),
            row.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            col.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if not handle:
            raise RuntimeError("cbspmv_prepare returned a null handle")
        y_host = np.zeros(rows, dtype=np.float64)
        # keep row/col/val/x alive: ctypes doesn't, and cbspmv_prepare only
        # reads them synchronously (no async H2D), so this is precautionary
        return {"handle": handle, "rows": rows, "y_host": y_host,
                "_row": row, "_col": col, "_val": val, "_x": x}

    def run(self, h):
        # exactly one y=Ax launch (stream=0, the default/legacy stream --
        # same stream torch.cuda.Event() records on by default, so
        # CudaEventTimer times this launch correctly even though the
        # kernel itself is launched outside torch).
        self.lib.cbspmv_run(h["handle"], ctypes.c_void_p(0))
        return h

    def to_host(self, out) -> np.ndarray:
        self.lib.cbspmv_copy_y(
            out["handle"], out["y_host"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out["y_host"].copy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        self.lib.cbspmv_free(h["handle"])
        h.clear()


def create(precision: str):
    return CbSpMV(precision)
