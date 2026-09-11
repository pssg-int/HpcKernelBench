"""
Adapter for RASSM (ASPLOS'25, gt-tinker/RASSM) — CPU SDDMM.

RASSM's own `rassm` CLI binary ships a `--kernel sddmm --type RASSM` path
(code/src/main.cpp) that builds an Adaptive Tile Matrix (ATM) via the paper's
Residue-based adaptive tiling and runs its own `sddmm_kstream` kernel over it
— but that whole path is DEAD CODE in the shipped build: the S_atm
construction and the sddmm_kstream call in
code/include/experiments.h::data_movement_experiment_sddmm() sit behind
`#ifdef RUN_CSR_ATM_KSTREAM`, and that macro is never #defined anywhere in
code/include/config.h (checked by grep across the whole include tree). So
`rassm --kernel sddmm` silently no-ops regardless of `--type`.

Rather than patch the artifact's config.h/experiments.h to flip that macro
(which would mean editing the artifact's own driver files), ../wrapper.cpp
is a small NEW driver — not part of the artifact — that calls the exact same
sequence of UNMODIFIED artifact functions the #ifdef'd block would have:
Residue::adaptive_2d_greedy_Ti_greedy_Tj_tile_generator (the paper's own
residue-based tile generator) to build panels, the ATM constructor to build
the tiled format from those panels, and sddmm_kstream (the paper's own
kernel) to compute the product. See wrapper.cpp's header comment and
STATUS.md for the full account, including the ATM nnz-reordering issue this
adapter has to undo before the correctness gate can compare apples to apples.

RASSM's `TYPE` (code/include/config.h) is compile-time fixed to `double` —
there is no fp32 compute path in the artifact at all, so `sddmm_kstream`
always computes in double regardless of the requested precision. This
adapter still services both "fp32" and "fp64" precision requests by
generating the dense A/B operands with the SAME fp32-rounding-then-widen
recipe kernelbench.impls.cpu_ref.reference_sddmm uses for its fp32 variant
(round to float32, then widen to float64) — matching the reference's actual
operand values bit-for-bit — while the kernel itself always runs in double.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "librassm_sddmm.so")

KERNEL = "sddmm"
IMPL_NAME = "rassm-adaptive-tiled-sddmm"
PAPER_KEY = "conf/asplos/JainGC25"
# The artifact only ever computes in double; both labels are serviced by
# rounding the dense operands to match what the reference used for that
# label (see module docstring) while the kernel itself stays double.
PRECISIONS = ["fp32", "fp64"]

_lib = None


def _load_lib():
    global _lib
    if _lib is None:
        if not os.path.exists(LIB_PATH):
            raise FileNotFoundError(f"{LIB_PATH} not built — run build.sh")
        lib = ctypes.CDLL(LIB_PATH)
        i32p = ctypes.POINTER(ctypes.c_int32)
        f64p = ctypes.POINTER(ctypes.c_double)

        lib.rassm_sddmm_prepare.argtypes = [
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,   # nrows, ncols, nnz
            i32p, i32p, f64p,                                  # row_idx, col_idx, vals
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,   # feature, Ri, Rj
            ctypes.c_int32,                                    # cache_bytes
        ]
        lib.rassm_sddmm_prepare.restype = ctypes.c_void_p

        lib.rassm_sddmm_nnz.argtypes = [ctypes.c_void_p]
        lib.rassm_sddmm_nnz.restype = ctypes.c_int32

        lib.rassm_sddmm_export_layout.argtypes = [ctypes.c_void_p, i32p, i32p]
        lib.rassm_sddmm_export_layout.restype = None

        lib.rassm_sddmm_run.argtypes = [ctypes.c_void_p, f64p, f64p, f64p, ctypes.c_int32]
        lib.rassm_sddmm_run.restype = None

        lib.rassm_sddmm_free.argtypes = [ctypes.c_void_p]
        lib.rassm_sddmm_free.restype = None

        _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(LIB_PATH):
            return False, "librassm_sddmm.so not built — run build.sh"
        _load_lib()
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _dense_operand(rows: int, dim: int, seed: int, dtype) -> np.ndarray:
    # identical to kernelbench.impls.cpu_ref._dense_operand — kept as a
    # separate copy (same formula, not a shared code object) per the
    # reference-independence rule in DOMAIN_GUIDE.md.
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(dtype)


class RassmSDDMM:
    name = "rassm-adaptive-tiled-sddmm"
    platform = "cpu"

    # residue-tiling / adaptive-tile-generator parameters, matching the
    # artifact CLI's own defaults in code/src/main.cpp's boost
    # program_options block (--Ri 64 --Rj 64 --cache-split 4 --oi-aware true
    # --targLLC/--targCache DEFAULT_LLC=1MiB, code/include/config.h).
    RI = 64
    RJ = 64
    CACHE_BYTES = 1024 * 1024

    def __init__(self, precision: str = "fp32"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"rassm-adaptive-tiled-sddmm: requested precision {precision!r} "
                f"not in {PRECISIONS}")
        self.precision = precision
        self._round_dtype = np.float32 if precision == "fp32" else np.float64

    def prepare(self, matrix, params):
        lib = _load_lib()
        S = matrix.csr  # NOT sorted -- keep the exact storage order reference_sddmm uses
        M, N = S.shape
        K = int(params["K"])
        seed = params.get("seed", 42)

        row_idx = np.repeat(np.arange(M, dtype=np.int32), np.diff(S.indptr)).astype(np.int32)
        col_idx = np.ascontiguousarray(S.indices, dtype=np.int32)
        vals = np.ascontiguousarray(S.data, dtype=np.float64)
        nnz = int(S.nnz)

        cptr = lib.rassm_sddmm_prepare(
            M, N, nnz,
            row_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            K, self.RI, self.RJ, self.CACHE_BYTES)
        if not cptr:
            raise RuntimeError("rassm_sddmm_prepare returned null")

        atm_nnz = lib.rassm_sddmm_nnz(cptr)
        if int(atm_nnz) != nnz:
            raise RuntimeError(
                f"ATM nnz ({atm_nnz}) != input nnz ({nnz}); the artifact's own "
                "tiling changed the sparsity pattern, which should be impossible")

        # ---- undo ATM's internal nnz reordering (see module/wrapper.cpp docstring) ----
        atm_row_ptr = np.empty(M + 1, dtype=np.int32)
        atm_cols = np.empty(nnz, dtype=np.int32)
        lib.rassm_sddmm_export_layout(
            cptr,
            atm_row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            atm_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

        atm_rows = np.repeat(np.arange(M, dtype=np.int64), np.diff(atm_row_ptr))
        atm_key = atm_rows * np.int64(N) + atm_cols.astype(np.int64)
        orig_key = row_idx.astype(np.int64) * np.int64(N) + col_idx.astype(np.int64)

        order = np.argsort(atm_key, kind="stable")
        sorted_key = atm_key[order]
        pos = np.searchsorted(sorted_key, orig_key)
        if nnz and (pos >= len(sorted_key)).any():
            raise RuntimeError("ATM layout export: reindex out of range — pattern mismatch")
        perm = order[pos]
        if nnz and not np.array_equal(sorted_key[pos], orig_key):
            raise RuntimeError(
                "ATM layout export: (row,col) pattern mismatch between the "
                "artifact's internal ATM layout and the input CSR — reindexing "
                "would silently misalign the correctness gate")

        # dense operands: rounded exactly like cpu_ref.reference_sddmm's fp32
        # branch (round to the requested dtype, then widen to float64) so
        # this adapter's inputs match the reference's inputs bit-for-bit.
        A = _dense_operand(M, K, seed, self._round_dtype).astype(np.float64)      # row-indexed
        B = _dense_operand(N, K, seed + 1, self._round_dtype).astype(np.float64)  # col-indexed
        D2 = np.ascontiguousarray(A)  # D2[i*K+k] in the kernel — row-indexed
        D1 = np.ascontiguousarray(B)  # D1[col*K+k] in the kernel — col-indexed
        O_buf = np.zeros(nnz, dtype=np.float64)

        # to_host() below only receives the run() return value, not this
        # handle, so stash perm on self here; prepare()/run()/to_host() are
        # always called on the same instance within one run_variant() call.
        self._last_perm = perm

        return {
            "lib": lib, "cptr": cptr, "K": K, "D1": D1, "D2": D2,
            "O_buf": O_buf, "perm": perm, "nnz": nnz,
        }

    def run(self, h):
        h["O_buf"].fill(0.0)  # sddmm_kstream accumulates with +=
        h["lib"].rassm_sddmm_run(
            h["cptr"],
            h["D1"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            h["D2"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            h["O_buf"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            h["K"])
        return h["O_buf"]

    def to_host(self, out):
        # 'out' is h["O_buf"] in the ATM's internal j-order; reindex to the
        # caller's original CSR nnz order (perm computed once in prepare(),
        # since it only depends on the static sparsity pattern).
        return np.asarray(out, dtype=np.float64)[self._last_perm]

    def timer(self):
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        h["lib"].rassm_sddmm_free(h["cptr"])
        h.clear()


def create(precision: str):
    return RassmSDDMM(precision)
