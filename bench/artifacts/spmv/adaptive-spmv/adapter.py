"""
Adapter for "Adaptive SpMV/SpMSpV on GPUs for Input Vectors of Varied
Sparsity" (TPDS 2021, journals/tpds/LiAY21).

Per the integration brief, wraps the DENSE-VECTOR SpMV path only: the
paper's headline contribution is an ML-based selector (hice/ml/,
decision-tree/SVM/GBDT/random-forest) choosing among 8 candidate GPU
SpMV/SpMSpV kernels; HolaSpmv (source/hice-spmspv/hice/la/include/spmspv/
csc-spmspv/detail/device/holaspmv.h: hola_pre<T>/hola_spmv<T>, from the
same research group's own prior GPU-SpMV work) is the dense-input-vector
member of that set. SpMSpV (sparse x, hola_spmspv in the same header) is a
different track and is not wrapped here. Wrapped via bridge.cu/.so (this
directory) since hola_pre/hola_spmv are templates with no extern "C"
boundary and the artifact ships no standalone dense-SpMV driver -- see
bridge.cu's docstring for exactly what had to be supplied to compile
holaspmv.h standalone (two trivial macros/templates the artifact's own
unrelated SpMSpV include chain would otherwise have supplied; no
kernel/algorithm code).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmv"
IMPL_NAME = "adaptive-spmv-hola"
PAPER_KEY = "journals/tpds/LiAY21"
PRECISIONS = ["fp64"]  # bridge.cu explicitly instantiates hola_pre/hola_spmv at <double>

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
    lib.holaspmv_prepare.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.holaspmv_prepare.restype = p
    lib.holaspmv_run.argtypes = [p]
    lib.holaspmv_run.restype = None
    lib.holaspmv_copy_y.argtypes = [p, ctypes.POINTER(ctypes.c_double)]
    lib.holaspmv_copy_y.restype = None
    lib.holaspmv_free.argtypes = [p]
    lib.holaspmv_free.restype = None
    return lib


class AdaptiveSpmvHola:
    name = "adaptive-spmv-hola"
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                "this bridge instantiates hola_pre/hola_spmv at <double> only "
                f"(the artifact's own header also explicitly instantiates <float>, "
                f"not wired up here); requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, matrix, params: dict):
        # HolaSpmv's own scratch-buffer sizing (hola_pre) + H2D copies --
        # genuinely one-time preprocessing under this variant's protocol.
        A = matrix.csr
        A.sort_indices()
        rowptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        colidx = np.ascontiguousarray(A.indices, dtype=np.int32)
        val = np.ascontiguousarray(A.data, dtype=np.float64)
        rows, cols = A.shape

        rng = np.random.default_rng(params.get("seed", 42))
        x = np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=cols).astype(np.float64))

        handle = self.lib.holaspmv_prepare(
            rows, cols, int(A.nnz),
            rowptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            colidx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if not handle:
            raise RuntimeError("holaspmv_prepare returned a null handle")
        y_host = np.zeros(rows, dtype=np.float64)
        return {"handle": handle, "rows": rows, "y_host": y_host,
                "_rowptr": rowptr, "_colidx": colidx, "_val": val, "_x": x}

    def run(self, h):
        # one y=Ax call (hola_spmv<double>); the kernel's own
        # DetermineBlockStarts pass zeroes y as part of this call (see
        # bridge.cu), so nothing else is needed here. Default/legacy
        # stream throughout (the artifact's kernels never take a stream
        # argument), matching CudaEventTimer's recording stream.
        self.lib.holaspmv_run(h["handle"])
        return h

    def to_host(self, out) -> np.ndarray:
        self.lib.holaspmv_copy_y(
            out["handle"], out["y_host"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out["y_host"].copy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        self.lib.holaspmv_free(h["handle"])
        h.clear()


def create(precision: str):
    return AdaptiveSpmvHola(precision)
