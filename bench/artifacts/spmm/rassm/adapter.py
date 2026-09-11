"""
RASSM adapter for the spmm track.

Paper: "RASSM: Residue-based Acceleration of Single Sparse Matrix
Computation via Adaptive Tiling", ASPLOS'25. `PAPER_KEY = conf/asplos/JainGC25`
(matched by title in `../../output/included.json`).
Artifact: https://github.com/gt-tinker/RASSM (CPU; implements SpMM and SDDMM
with its own residue-based adaptive tiling preprocessing).

Contamination note (ARTIFACT_GUIDE.md rule 1): RASSM ships no library, only
an end-to-end CLI binary (`rassm`) that reads a matrix from an .mtx file it
opens itself, runs its own internal timing loop over its own deterministic
dense operand (`generate_dense()`: `(i % 100) / 100.0`, not something a
correctness gate could be matched against), and prints only aggregate
"Median Time"/"GFLOPS" text -- it never returns the output matrix C, and the
one correctness-check code path (`RUN_CHECK` in experiments.h) only prints a
boolean and is compiled out entirely (`#ifdef RUN_CORRECTNESS_CHECK`, never
defined). So this adapter wraps at the finest available boundary instead:
`wrapper.cpp` (this directory, NOT under source/) calls RASSM's own
header-only preprocessing pipeline (CSR -> CSC -> Residue ->
adaptive_2d_greedy_Ti_greedy_Tj_tile_generator -> ATM) and its k-stream SpMM
kernel (`spmm_atm_kstream_compiler_vectorized`) directly, bypassing
`main.cpp`'s CLI/file-I/O/timing-loop/generate_dense entirely -- every
function called is the artifact's own unmodified code (see wrapper.cpp's
header comment for the exact provenance of each call). This is a real
architectural difference from the other two spmm artifacts in this track
(insum, inferfast), which both wrap a genuine library entry point directly;
RASSM's is a CLI-only artifact, so the "kernel entry point" wrapped here is
one level further in, at the template-function level its own executable is
built from.

prepare() runs RASSM's own tiling preprocessing (Residue matrix + adaptive
greedy panel generation + ATM construction) -- ARTIFACT_GUIDE.md rule 2:
this is the artifact's format conversion, timed as preprocessing, not
folded into run()'s per-call kernel time.

PRECISION: fp32 requested by the harness's default, but RASSM's kernel is
hardcoded to `TYPE = double` in `source/code/include/config.h` (not a
template parameter the CLI or this wrapper can vary) -- there is no fp32
code path. `PRECISIONS = ["fp64"]` (ARTIFACT_GUIDE.md: "what the artifact
actually supports"); gating this adapter needs `--precision fp64`.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "rassm-spmm-kstream"
PAPER_KEY = "conf/asplos/JainGC25"
PRECISIONS = ["fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "librassm_wrapper.so")

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    i = ctypes.c_int
    lib.rassm_prepare.argtypes = [
        i, i, i,                                            # nrows, ncols, nnz
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),                     # rows, cols, vals
        i, i, i, i, i, i, i, i, i,                            # feature..resolution
    ]
    lib.rassm_prepare.restype = p
    lib.rassm_run.argtypes = [p, ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_double), i, i]
    lib.rassm_run.restype = None
    lib.rassm_free.argtypes = [p]
    lib.rassm_free.restype = None
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "librassm_wrapper.so not built -- run build.sh"
    try:
        _load_lib()
    except Exception as e:
        return False, f"ctypes load failed: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return RassmSpMM(precision)


class RassmSpMM:
    name = IMPL_NAME
    platform = "cpu"

    # RASSM's own CLI defaults (source/code/src/main.cpp's
    # boost::program_options defaults / source/scripts/run-rassm.sh) --
    # reused as-is, not tuned, since we're not sweeping parameters here.
    RI = 64
    RJ = 64
    TARGET_CACHE_SIZE = 1024 * 1024  # config.h's DEFAULT_LLC
    CACHE_SPLIT = 4
    RESOLUTION = 1
    OI_AWARE = True
    TEMPORAL_INPUT = False
    TEMPORAL_OUTPUT = False

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp64 -- RASSM's kernel is "
                f"compiled with TYPE=double (config.h), not a template "
                f"parameter the CLI exposes; requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        lib = _load_lib()
        A = matrix.csr.tocoo()
        A.sum_duplicates()
        nrows, ncols = matrix.csr.shape
        nnz = A.nnz
        N = int(params["N"])

        rows = np.ascontiguousarray(A.row, dtype=np.int32)
        cols = np.ascontiguousarray(A.col, dtype=np.int32)
        vals = np.ascontiguousarray(A.data, dtype=np.float64)

        handle = lib.rassm_prepare(
            ctypes.c_int(nrows), ctypes.c_int(ncols), ctypes.c_int(nnz),
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(N), ctypes.c_int(self.RI), ctypes.c_int(self.RJ),
            ctypes.c_int(self.TARGET_CACHE_SIZE), ctypes.c_int(self.CACHE_SPLIT),
            ctypes.c_int(int(self.TEMPORAL_INPUT)), ctypes.c_int(int(self.TEMPORAL_OUTPUT)),
            ctypes.c_int(int(self.OI_AWARE)), ctypes.c_int(self.RESOLUTION))
        if not handle:
            raise RuntimeError("rassm_prepare returned NULL")

        # B: numpy RNG matching kernelbench.impls.cpu_ref.reference_spmm's
        # _dense_operand exactly (see insum/adapter.py's docstring for why
        # this matters). RASSM's own generate_dense() is a fixed
        # deterministic pattern, not usable as a gate-matching operand, so
        # we bypass it entirely and hand rassm_run() our own B buffer --
        # its kernel indexes B/C row-major (see wrapper.cpp), same
        # convention the reference and every other spmm impl in this
        # harness use, so no transpose is needed either way.
        rng = np.random.default_rng(params.get("seed", 42))
        B = np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=(ncols, N)).astype(np.float64))
        C = np.zeros((nrows, N), dtype=np.float64)

        return {"lib": lib, "handle": handle, "B": B, "C": C, "N": N}

    def run(self, h):
        h["C"].fill(0.0)  # kernel accumulates (+=); zero for a clean C=A@B
        h["lib"].rassm_run(
            h["handle"],
            h["B"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            h["C"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(h["N"]), ctypes.c_int(1))
        return h["C"]

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        from kernelbench.harness import Timer
        return Timer()  # CPU wall-clock, perf_counter -- this is a CPU impl

    def free(self, h):
        h["lib"].rassm_free(h["handle"])
