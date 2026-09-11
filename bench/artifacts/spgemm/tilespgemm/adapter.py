"""
TileSpGEMM adapter for the spgemm track.

Paper: "TileSpGEMM: A Tiled Algorithm for Parallel Sparse General
Matrix-Matrix Multiplication on GPUs" (PPoPP'22; `conf/ppopp/NiuLJS0022`
in output/included.json). Selected as a core baseline under the revised
kernel-centrality rule (`output/kernel_centrality.json` key
`spgemm|conf/ppopp/NiuLJS0022`: centrality `core`, regime `matches` --
"SuiteSparse matrices (A^2, general A x B), fp64, GPU"; see
`output/baseline_selection.md`'s spgemm section).
Artifact: https://github.com/SuperScientificSoftwareLaboratory/TileSpGEMM

Kernel entry point wrapped (ARTIFACT_GUIDE.md rule 1): `tilespgemm()`
(`source/src/tilespgemm-cuda.h`) -- the artifact's own SpGEMM computation,
the SAME function `source/src/main.cu`'s CLI calls (inside an `#ifdef
DEBUG` block main.cu's own `common.h` already enables unconditionally,
`#define DEBUG 1`, no `-D` flag needed -- so this IS the artifact's
real/default computational path, not a debug-only stub). `tsg_bridge.cu`
(this directory, NOT part of the artifact) calls it directly via our own
CSR arrays, bypassing main.cu's `.mtx` file parsing (`mmio_allinone`) and
its synthetic `value[i] = i % 10` overwrite (main.cu:100-101) -- our
workload's real fp64 values are used throughout, required for a
meaningful correctness gate. See tsg_bridge.cu's file header for the full
boundary description and the disclosed timing-contamination note (same
posture as `../ocean/adapter.py`'s Workspace-construction disclosure).

Operation: C = A @ A (self-product), per benchspecs/spgemm/spec.yaml's
`operation` field -- matches the artifact's own default `-aat 0` mode
(main.cu:132-140 literally aliases matrixB's rowpointer/columnindex/value
to matrixA's; tsg_bridge.cu's `tsg_prepare()` reproduces this exact
aliasing, verified safe since csr2tile_row_major/csr2tile_col_major only
READ the shared arrays -- see tsg_bridge.cu's comments).

prepare() does the artifact's own CSR->tile format conversion
(`csr2tile_row_major`/`csr2tile_col_major`, source/src/csr2tile.h,
unmodified) -- this IS the preprocessing spgemm-square-kernel-f64's spec
excludes from the timed window (rule 2).

Disclosed timing contamination (full account in tsg_bridge.cu's header):
`tilespgemm()` mallocs+H2D-copies the already-tiled A/B host arrays into
fresh device buffers AND D2H-copies the tiled C result back, all inside
the SAME call it also does the symbolic+numeric computation in, then frees
every device buffer before returning -- there is no separate, already-
idempotent sub-boundary to hoist the H2D setup into prepare() without
restructuring the artifact's own control flow (a kernel-code change,
forbidden by rule 3). This adapter's measured per-call time is therefore a
disclosed UPPER BOUND on spgemm-square-kernel-f64's timing_scope
(symbolic+numeric+C-allocation only; A/B's tile-structure H2D transfer is
supposed to be excluded, "operands resident on device before timing
starts") -- not an exact match, the same posture as Ocean's disclosed
Workspace-construction overhead.

PRECISION: fp64 only (`MAT_VAL_TYPE`/`VALUE_TYPE` are `double` throughout
source/src/common.h, unconditionally) -- `PRECISIONS = ["fp64"]`.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spgemm"
IMPL_NAME = "tilespgemm"
PAPER_KEY = "conf/ppopp/NiuLJS0022"
PRECISIONS = ["fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "tsg_bridge.so")

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

    lib.tsg_prepare.argtypes = [ctypes.c_int32, ctypes.c_longlong, i32p, i32p, f64p]
    lib.tsg_prepare.restype = p
    lib.tsg_run.argtypes = [p]
    lib.tsg_run.restype = None
    lib.tsg_finalize_csr.argtypes = [p]
    lib.tsg_finalize_csr.restype = None
    lib.tsg_dims.argtypes = [p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(ctypes.c_longlong)]
    lib.tsg_dims.restype = None
    lib.tsg_to_host.argtypes = [p, i32p, i32p, f64p]
    lib.tsg_to_host.restype = None
    lib.tsg_free.argtypes = [p]
    lib.tsg_free.restype = None
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "tsg_bridge.so not built -- run build.sh"
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
    return TileSpGEMM(precision)


def _as_i32p(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _as_f64p(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class TileSpGEMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp64 (MAT_VAL_TYPE/VALUE_TYPE are "
                f"'double' unconditionally in source/src/common.h); requested {precision}")
        self.precision = precision
        self._handle = None

    def prepare(self, matrix, params: dict):
        lib = _load_lib()
        self._matrix = matrix

        A = matrix.csr.astype(np.float64)
        A.sort_indices()
        rows, cols = A.shape
        if rows != cols:
            raise NotImplementedError(
                f"{IMPL_NAME}: self-product C=A@A requires a square matrix, "
                f"got {A.shape} (matches the artifact's own main.cu -aat 0 "
                f"guard: 'matrix squaring must have rowA == colA')")
        nnz = A.nnz

        row_off, _ = _as_i32p(A.indptr)
        col_ids, _ = _as_i32p(A.indices)
        data, _ = _as_f64p(A.data)

        # --- artifact's own CSR->tile conversion, timed as preprocessing --
        # (csr2tile_row_major/csr2tile_col_major inside tsg_prepare(), see
        # tsg_bridge.cu). Self-product: same CSR passed once, aliased by
        # both A and B inside the bridge exactly as main.cu's own -aat 0
        # path does.
        handle = lib.tsg_prepare(
            rows, nnz,
            row_off.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            col_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: tsg_prepare failed (see stderr)")
        self._handle = handle
        self._first_to_host_done = False
        return {"handle": handle, "rows": rows, "cols": cols}

    def run(self, h):
        lib = _load_lib()
        # tilespgemm(): one full symbolic+numeric pass (REPEAT_NUM==1); see
        # tsg_bridge.cu / adapter.py module docstring for the disclosed
        # per-call H2D-of-A/B contamination this bundles in.
        lib.tsg_run(h["handle"])
        return h["handle"]

    def to_host(self, out):
        lib = _load_lib()
        import scipy.sparse as sp
        from kernelbench.impls.cpu_ref import _canonical_pattern, _reindex_to_pattern

        # tile2csr() (source/src/tile2csr.h, unmodified): host-only, CPU
        # conversion of the tiled C back to plain CSR. Runs ONCE here, from
        # to_host() -- never inside the timed run() path (harness.py calls
        # to_host() exactly once, on the first run()'s output, for the
        # correctness gate).
        lib.tsg_finalize_csr(out)
        rows_c = ctypes.c_int()
        cols_c = ctypes.c_int()
        nnz_c = ctypes.c_longlong()
        lib.tsg_dims(out, ctypes.byref(rows_c), ctypes.byref(cols_c), ctypes.byref(nnz_c))
        rows, cols, nnz = rows_c.value, cols_c.value, nnz_c.value

        row_off = np.empty(rows + 1, dtype=np.int32)
        col_ids = np.empty(max(nnz, 1), dtype=np.int32)
        data = np.empty(max(nnz, 1), dtype=np.float64)
        lib.tsg_to_host(
            out,
            row_off.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            col_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        C = sp.csr_matrix((data[:nnz], col_ids[:nnz], row_off), shape=(rows, cols))
        A64 = self._matrix.csr.astype(np.float64)
        A64.sort_indices()
        pattern = _canonical_pattern(A64)
        return _reindex_to_pattern(C, pattern)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        lib = _load_lib()
        lib.tsg_free(h["handle"])
        h.clear()
        import torch
        torch.cuda.empty_cache()
