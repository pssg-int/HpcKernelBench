"""
Adapter for spmv-acc (HPDC 2023, "Efficient Algorithm Design of Optimizing
SpMV on GPU", conf/hpdc/ChuHDDC0WH23).

Wraps the artifact's own kernel-library entry point, sparse_csr_spmv()
(source/src/acc/api/spmv.h), compiled with KERNEL_STRATEGY_ADAPTIVE
(compat/building_config.h) -- the paper's headline auto-selecting
strategy, dispatching between its own flat/line/line-enhance/vector-row
kernels based on the matrix's nnz distribution (source/src/acc/
hip-adaptive/adaptive.cpp, unmodified). See bridge.cpp (this directory)
for exactly what is wrapped and STATUS.md for the toolchain that finally
built this (a HIP-on-CUDA module, hip/5.6.1/gcc/11.3.0/nompi/cuda/12.3.0/
zen2 on zaratan) after this artifact was BUILD-FAILED on the reference
machine (Perlmutter, whose only HIP module was version-incompatible with
every CUDA toolkit available there).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmv"
IMPL_NAME = "spmv-acc"
PAPER_KEY = "conf/hpdc/ChuHDDC0WH23"
PRECISIONS = ["fp64"]  # sparse_csr_spmv()'s csr_desc<int, double> -- fp64 only

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "bridge.so")


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, (
                "BUILD-FAILED: bridge.so never produced -- see STATUS.md "
                "for the required HIP-on-CUDA toolchain (needs a HIP module "
                "compiled against a compatible CUDA; none was available on "
                "the reference machine, Perlmutter)"
            )
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    p = ctypes.c_void_p
    lib.spmvacc_prepare.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.spmvacc_prepare.restype = p
    lib.spmvacc_run.argtypes = [p]
    lib.spmvacc_run.restype = None
    lib.spmvacc_copy_y.argtypes = [p, ctypes.POINTER(ctypes.c_double)]
    lib.spmvacc_copy_y.restype = None
    lib.spmvacc_free.argtypes = [p]
    lib.spmvacc_free.restype = None
    return lib


class SpmvAcc:
    name = "spmv-acc"
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"spmv-acc's csr_desc<int, double> instantiation is fp64 only; "
                f"requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, matrix, params: dict):
        # This IS the artifact's own format handling (sparse_csr_spmv takes
        # CSR directly, no conversion needed) + every device malloc/H2D
        # copy the artifact's own CLI driver would do -- one-time
        # preprocessing under this variant's protocol, timed once here.
        A = matrix.csr
        A.sort_indices()
        rowptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        colidx = np.ascontiguousarray(A.indices, dtype=np.int32)
        val = np.ascontiguousarray(A.data, dtype=np.float64)
        rows, cols = A.shape

        rng = np.random.default_rng(params.get("seed", 42))
        x = np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=cols).astype(np.float64))

        handle = self.lib.spmvacc_prepare(
            rows, cols, int(A.nnz),
            rowptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            colidx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if not handle:
            raise RuntimeError("spmvacc_prepare returned a null handle")
        y_host = np.zeros(rows, dtype=np.float64)
        # keep rowptr/colidx/val/x alive: ctypes doesn't, and
        # spmvacc_prepare only reads them synchronously (no async H2D), so
        # this is precautionary (same pattern as artifacts/spmv/tilespmv).
        return {"handle": handle, "rows": rows, "y_host": y_host,
                "_rowptr": rowptr, "_colidx": colidx, "_val": val, "_x": x}

    def run(self, h):
        # one y = alpha*A*x + beta*y call (alpha=1, beta=0), dispatched by
        # the artifact's own adaptive strategy (see bridge.cpp). Default
        # HIP stream (0), matching every other bridge in this repo and
        # CudaEventTimer's recording stream.
        self.lib.spmvacc_run(h["handle"])
        return h

    def to_host(self, out) -> np.ndarray:
        self.lib.spmvacc_copy_y(
            out["handle"], out["y_host"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out["y_host"].copy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        self.lib.spmvacc_free(h["handle"])
        h.clear()


def create(precision: str):
    return SpmvAcc(precision)
