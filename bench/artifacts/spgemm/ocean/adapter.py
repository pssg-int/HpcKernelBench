"""
Ocean adapter for the spgemm track.

Paper: "Ocean: Fast Estimation-Based Sparse General Matrix-Matrix
Multiplication on GPU", ICS'26. `PAPER_KEY = conf/ics/LiG26` (matched by
`artifact_url` in `../../output/included.json`).
Artifact: https://github.com/CornellHPC/Ocean-SpGEMM

Kernel entry point wrapped (ARTIFACT_GUIDE.md rule 1): `somespgemm::SpGEMM::
run(A, B)` (`source/kernels/SpGEMM.cuh`) -- Ocean's own library entry point,
the SAME class `source/src/main.cu`'s CLI calls; we call it directly via
`wrapper.cu` (this directory, NOT under source/ -- see its header comment
for exact provenance of every call) instead of going through the CLI's
file-I/O / JSON-config plumbing. `wrapper.cu` touches zero lines of
source/; it only adds `extern "C"` entry points around already-defined,
unmodified classes/functions (`somespgemm::CSR`, `somespgemm::cuCSR`,
`somespgemm::convert`, `somespgemm::SpGEMM`).

Operation: C = A @ A (self-product), per benchspecs/spgemm/spec.yaml's
`operation` field -- two independent device copies of A are built (not one
aliased pointer passed twice) to avoid relying on any A!=B assumption
inside Ocean's own (unaudited in full) kernel code.

prepare() does the artifact's own format conversion (rule 2, timed as
preprocessing): our workload's CSR -> a host `somespgemm::CSR` -> Ocean's
own `convert(cuCSR&, const CSR&)` H2D copy (via `ocean_build_csr`), once per
operand.

Disclosed timing contamination (see STATUS.md for the full writeup):
`SpGEMM::run()` constructs a fresh `Workspace` (20 `cudaStreamCreate` + 200
`cudaEventCreate` calls) on EVERY call, which is intrinsic to how the
artifact is written -- there is no already-separate, already-idempotent
function to hoist this into prepare() the way AmgT's `CSR2BSR_GPU` allows
(see bench/artifacts/spgemm/amgt/STATUS.md). `ocean_spgemm_new()` sets
`warmup_iters=0, bench_iters=1` so Ocean's OWN internal warmup+bench
averaging loop collapses to exactly one prologue -> analysis ->
symbolic/estimation -> numeric pass per call (the harness's own
warmup/measured-reps protocol governs repetition instead), but every
`run()` call here still pays Workspace-construction overhead in addition to
the spec's `timing_scope` (symbolic+numeric+C-allocation) -- this adapter's
measured per-call time is a disclosed UPPER BOUND on the spec's intended
scope, not an exact match.

PRECISION: fp64 only (Ocean's `data_t` is `double`, Common.h has no other
value type) -- `PRECISIONS = ["fp64"]`.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spgemm"
IMPL_NAME = "ocean-spgemm"
PAPER_KEY = "conf/ics/LiG26"
PRECISIONS = ["fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libocean_wrapper.so")

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    i32p = ctypes.POINTER(ctypes.c_int32)
    f64p = ctypes.POINTER(ctypes.c_double)
    i32 = ctypes.c_int32
    i64 = ctypes.c_int64

    lib.ocean_build_csr.argtypes = [i32, i32, i64, i32p, i32p, f64p]
    lib.ocean_build_csr.restype = p
    lib.ocean_free_csr.argtypes = [p]
    lib.ocean_free_csr.restype = None
    lib.ocean_spgemm_new.argtypes = [i32]
    lib.ocean_spgemm_new.restype = p
    lib.ocean_spgemm_free.argtypes = [p]
    lib.ocean_spgemm_free.restype = None
    lib.ocean_spgemm_run.argtypes = [p, p, p]
    lib.ocean_spgemm_run.restype = p
    lib.ocean_csr_dims.argtypes = [p, ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_longlong)]
    lib.ocean_csr_dims.restype = None
    lib.ocean_csr_to_host.argtypes = [p, i32p, i32p, f64p]
    lib.ocean_csr_to_host.restype = None
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libocean_wrapper.so not built -- run build.sh"
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
    return OceanSpGEMM(precision)


def _as_i32p(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _as_f64p(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class OceanSpGEMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp64 (Ocean's data_t is "
                f"'double', no other value type in Common.h); requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        lib = _load_lib()
        self._matrix = matrix

        A = matrix.csr.astype(np.float64)
        A.sort_indices()
        rows, cols = A.shape
        nnz = A.nnz

        row_off, row_off_p = _as_i32p(A.indptr)
        col_ids, col_ids_p = _as_i32p(A.indices)
        data, data_p = _as_f64p(A.data)

        # --- artifact's own format conversion, timed as preprocessing ----
        # two independent device copies (A and B are semantically distinct
        # operands in C = A @ A, see module docstring).
        a_handle = lib.ocean_build_csr(rows, cols, nnz, row_off_p, col_ids_p, data_p)
        b_handle = lib.ocean_build_csr(rows, cols, nnz, row_off_p, col_ids_p, data_p)
        spgemm_handle = lib.ocean_spgemm_new(0)

        return {"a": a_handle, "b": b_handle, "spgemm": spgemm_handle,
                "last_c": None, "rows": rows, "cols": cols}

    def run(self, h):
        lib = _load_lib()
        # free the previous call's result -- run() is invoked once per
        # warmup/measured iteration (only the FIRST call's result is ever
        # read via to_host(), for the correctness gate), so without this the
        # device would leak one fresh cuCSR per rep across ~30 calls.
        if h["last_c"] is not None:
            lib.ocean_free_csr(h["last_c"])
        c_handle = lib.ocean_spgemm_run(h["spgemm"], h["a"], h["b"])
        h["last_c"] = c_handle
        return c_handle

    def to_host(self, out):
        lib = _load_lib()
        import scipy.sparse as sp
        from kernelbench.impls.cpu_ref import _canonical_pattern, _reindex_to_pattern

        rows_c = ctypes.c_int()
        cols_c = ctypes.c_int()
        nnz_c = ctypes.c_longlong()
        lib.ocean_csr_dims(out, ctypes.byref(rows_c), ctypes.byref(cols_c), ctypes.byref(nnz_c))
        rows, cols, nnz = rows_c.value, cols_c.value, nnz_c.value

        row_off = np.empty(rows + 1, dtype=np.int32)
        col_ids = np.empty(nnz, dtype=np.int32)
        data = np.empty(nnz, dtype=np.float64)
        lib.ocean_csr_to_host(
            out,
            row_off.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            col_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        C = sp.csr_matrix((data, col_ids, row_off), shape=(rows, cols))
        A64 = self._matrix.csr.astype(np.float64)
        A64.sort_indices()
        pattern = _canonical_pattern(A64)
        return _reindex_to_pattern(C, pattern)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        lib = _load_lib()
        if h.get("last_c") is not None:
            lib.ocean_free_csr(h["last_c"])
        lib.ocean_spgemm_free(h["spgemm"])
        lib.ocean_free_csr(h["a"])
        lib.ocean_free_csr(h["b"])
        h.clear()
        import torch
        torch.cuda.empty_cache()
