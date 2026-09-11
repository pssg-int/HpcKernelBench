"""
Adapter for SSpMV / LeSpMV (DAC 2025, "SSpMV: A Sparsity-aware SpMV
Framework Empowered by Multimodal Machine Learning", conf/dac/LinLDZLY25).

The paper's contribution is an ML-based per-matrix format/algorithm
SELECTOR (SmartAdpter/, a trained model choosing among CSR/BSR/DIA/ELL/
SELL/CSR5/...); the underlying kernels it selects among live in the
LeSpMV/ library. Integrating the full selector would mean running a
Python-side model-inference step before every matrix (out of scope for a
"wrap the kernel" adapter, and the trained model/weights are not checked
into this repo). Per the "wrap the kernel, not the paper's benchmark
script" rule, this adapter wraps LeSpMV's default, always-available CSR
kernel (LeSpMV_csr<int,double>, kernel_flag=1 -> __spmv_csr_omp_simple, an
OpenMP-parallel row-per-thread CSR SpMV -- source/LeSpMV/src/spmv_csr.cpp)
via csr_shim.cpp/.so, a thin extern "C" wrapper (see that file's docstring;
it forwards to the artifact's own kernel unmodified).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmv"
IMPL_NAME = "sspmv-lespmv-csr"
PAPER_KEY = "conf/dac/LinLDZLY25"
PRECISIONS = ["fp64"]  # LeSpMV_csr<int,double> instantiation used; fp32 also exists (<int,float>) but unwrapped here

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHIM_SO = os.path.join(_HERE, "csr_shim.so")


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SHIM_SO):
            return False, f"not built: {_SHIM_SO} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SHIM_SO)
    lib.sspmv_csr_f64.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.sspmv_csr_f64.restype = None
    return lib


class SSpMVCsr:
    name = "sspmv-lespmv-csr"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"sspmv-lespmv-csr wraps the LeSpMV_csr<int,double> instantiation only; "
                f"requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, matrix, params: dict):
        # This IS the artifact's own format construction (CSR -> its
        # CSR_Matrix<int,double> aggregate, built inside csr_shim.cpp) --
        # trivial for CSR-to-CSR, but still timed once as preprocessing per
        # the contract, since it is real host-side array preparation
        # (dtype casts + a fresh contiguous copy of indptr/indices).
        A = matrix.csr
        indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        data = np.ascontiguousarray(A.data, dtype=np.float64)
        rows, cols = A.shape

        rng = np.random.default_rng(params.get("seed", 42))
        x = np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=cols).astype(np.float64))
        y = np.zeros(rows, dtype=np.float64)

        return {
            "rows": rows, "cols": cols, "nnz": int(A.nnz),
            "indptr": indptr, "indices": indices, "data": data,
            "x": x, "y": y,
        }

    def run(self, h):
        self.lib.sspmv_csr_f64(
            h["rows"], h["cols"], h["nnz"],
            h["indptr"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            h["indices"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            h["data"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            h["x"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            h["y"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return h["y"]

    def to_host(self, out) -> np.ndarray:
        return np.array(out, dtype=np.float64, copy=True)

    def timer(self):
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        h.clear()


def create(precision: str):
    return SSpMVCsr(precision)
