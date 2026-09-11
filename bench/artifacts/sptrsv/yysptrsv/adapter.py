"""
YuenyeungSpTRSV adapter for the sptrsv track.

Paper: "YuenyeungSpTRSV: A Thread-Level and Warp-Level Fusion
Synchronization-Free Sparse Triangular Solve" (TPDS'21,
journals/tpds/ZhangSLHWDW21). Artifact:
https://github.com/JiyaSu/YuenyeungSpTRSV

The paper's own CLI (CUDA/main.cu) reads a .mtx file from disk and drives
`YYSpTRSV_csr(...)`, a C++ function that ALSO owns its own internal
BENCH_REPEAT=100 mean-timing loop (survey.md: no warmup, mean not median --
exactly what this benchmark's protocol replaces). Per ARTIFACT_GUIDE.md rule
1 ("wrap the kernel, not the paper's benchmark script"), we don't call
`YYSpTRSV_csr` at all -- we call its two real ingredients directly:

  * `matrix_warp` (source/CUDA/tranpose.h) -- the host-side (pure CPU, no
    CUDA) thread/warp-level partitioning step that decides, per group of
    WARP_SIZE=32 consecutive rows, whether to solve them at thread-level or
    warp-level based on average nnz/row vs. a `border` threshold (10, the
    artifact's own main.cu default). This IS the paper's "analysis" step
    (its own abstract: "does not need long preprocessing time to calculate
    levels" -- this partitioning is what replaces level-set construction) --
    maps directly to prepare(), timed once as preprocessing.
  * `yySpTRSV_csr_kernel` (source/CUDA/YYSpTRSV.h) -- the actual fused
    thread-level/warp-level synchronization-free CUDA solve kernel.

Both are included VERBATIM in shim.cu (this directory, NOT part of the
artifact -- same role bench/artifacts/spmv/sspmv/csr_shim.cpp plays for
LeSpMV); shim.cu adds extern "C" host-side launchers, no kernel line
edited except one build-system-level compatibility macro (see shim.cu's
own comment and STATUS.md): `__shfl_down` (pre-CUDA-9, unmasked) was
removed by CUDA 12 and is scoped-macro-mapped to `__shfl_down_sync` with a
full-warp mask, undef'd immediately after the vendored header is included
-- the vendored file on disk is never modified.

Format: plain CSR of L, diagonal LAST per row -- the artifact's own native
format ("users do not need to conduct format conversion", per its
abstract) and exactly what `kernelbench.impls.cpu_ref.unit_lower_triangular`
already produces after `.sort_indices()` (L is lower-triangular, so within
any row every off-diagonal entry has col < row and the diagonal has
col == row, the row-maximum, hence sorted last) -- no extra reordering
needed, so this adapter's prepare() does LESS format conversion than
split-sptrsv's (no CSR->CSC step).

Per-call state: `yySpTRSV_csr_kernel` mutates `d_get_value` (a per-row
"ready" flag array the synchronization-free thread-level path spin-waits
on), `d_x`, and `d_id_extractor` (an atomic counter used for dynamic
work-item assignment across warps) in place during a solve, so all three
must be reset before every call -- matching the artifact's OWN correct
per-iteration practice (YYSpTRSV_csr's BENCH_REPEAT loop does exactly this
trio of cudaMemsets before each timed launch) and explicitly sanctioned by
benchspecs/sptrsv/spec.yaml's `timing_scope` field as belonging INSIDE the
timed region ("this matches YuenyeungSpTRSV's correct cudaMemset-before-
timing practice"). shim.cu's `yy_solve` does this reset immediately before
the kernel launch, both inside the one function CudaEventTimer wraps.

RHS pool: identical construction to
kernelbench.impls.cpu_ref.ScipySpTRSV.prepare() (same seed/count read from
the same `params` dict the harness passes to every impl in a given run),
so every sptrsv competitor in a given run solves the same draws -- same
fixed-pool-cycled-by-a-cursor pattern documented there and in
bench/artifacts/sptrsv/split-sptrsv/adapter.py.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "sptrsv"
IMPL_NAME = "yysptrsv-fused"
PAPER_KEY = "journals/tpds/ZhangSLHWDW21"
PRECISIONS = ["fp64"]  # VALUE_TYPE == double, unconditionally (source/CUDA/common.h)

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_HERE, "shim.so")


def available() -> tuple[bool, str]:
    """(True, "") if built and loadable; (False, reason) otherwise. Never raises."""
    try:
        if not os.path.exists(_LIB):
            return False, f"not built: {_LIB} missing (run build.sh)"
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available on this host"
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_LIB)
    p = ctypes.c_void_p
    lib.yy_warp_partition.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, p, p, p, p, p, p, p]
    lib.yy_warp_partition.restype = None
    lib.yy_solve.argtypes = [p, p, p, p, ctypes.c_int, ctypes.c_int, p, p, p,
                              ctypes.c_int, p]
    lib.yy_solve.restype = None
    lib.yy_sync.restype = ctypes.c_int
    return lib


def create(precision: str):
    return YuenyeungSpTRSV(precision)


class YuenyeungSpTRSV:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"{IMPL_NAME} wraps a VALUE_TYPE==double kernel only "
                f"(source/CUDA/common.h); requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, matrix, params: dict):
        import torch
        from kernelbench.impls.cpu_ref import unit_lower_triangular

        # SAME canonical L every sptrsv competitor in this benchmark solves
        # (kernelbench.domains.sparse's shared derivation, reused -- not
        # artifact-specific, see module docstring).
        L64 = unit_lower_triangular(matrix.csr)
        L64.sort_indices()
        m = int(L64.shape[0])
        nnz = int(L64.nnz)

        rowptr_h = L64.indptr.astype(np.int32)
        colidx_h = L64.indices.astype(np.int32)
        val_h = L64.data.astype(np.float64)

        # --- one-shot structural analysis: the artifact's own host-side
        # thread/warp partitioning (matrix_warp), called ONCE here -- timed
        # as preprocessing, per ARTIFACT_GUIDE.md rule 2.
        Len = ctypes.c_int(0)
        warp_occupy = ctypes.c_double(0.0)
        element_occupy = ctypes.c_double(0.0)
        warp_num_h = np.zeros(m + 1, dtype=np.int32)
        self.lib.yy_warp_partition(
            m, m, nnz,
            rowptr_h.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            colidx_h.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val_h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(Len),
            warp_num_h.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(warp_occupy), ctypes.byref(element_occupy))
        Len = Len.value

        rowptr = torch.as_tensor(rowptr_h, device="cuda")
        colidx = torch.as_tensor(colidx_h, device="cuda")
        val = torch.as_tensor(val_h, device="cuda")
        warp_num = torch.as_tensor(warp_num_h[:Len], device="cuda")

        get_value = torch.zeros(m, dtype=torch.int32, device="cuda")
        x = torch.zeros(m, dtype=torch.float64, device="cuda")
        id_extractor = torch.zeros(1, dtype=torch.int32, device="cuda")

        # fixed RHS pool -- identical construction to
        # kernelbench.impls.cpu_ref.ScipySpTRSV.prepare() (see module
        # docstring): U(0,1) draws, b = L @ x_ref via one fp64 scipy sparse
        # matvec, never via the technique under test.
        num_rhs = int(params.get("sptrsv_num_rhs", 64))
        seed = params.get("seed", 42)
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.0, 1.0, size=(num_rhs, m))
        # np.ascontiguousarray: (L64 @ X.T).T is a TRANSPOSED VIEW (strided,
        # not C-contiguous -- confirmed via a standalone repro during this
        # integration, see STATUS.md). torch.as_tensor(..., device="cuda")
        # preserves that non-contiguous layout, but shim.cu's yy_solve
        # treats `d_b` as a flat contiguous pointer (`d_b[row]` inside the
        # vendored kernel), so an uncopied strided view would feed the
        # solver SCRAMBLED right-hand sides -- a bug in this adapter's own
        # operand construction, not in the artifact's kernel. Forcing a
        # real contiguous copy here is the fix.
        B = np.ascontiguousarray((L64 @ X.T).T)  # (num_rhs, m)

        params["sptrsv_L_nnz"] = nnz
        params["sptrsv_num_rhs"] = num_rhs
        params["sptrsv_rhs_seed"] = int(seed)
        params["sptrsv_triangular_factor"] = (
            "unit-lower-triangular-of-A (canonical structural proxy; "
            "benchspecs/sptrsv/spec.yaml triangular_factor_derivation)")
        params["yysptrsv_warp_occupy"] = warp_occupy.value
        params["yysptrsv_element_occupy"] = element_occupy.value

        B_dev = torch.as_tensor(B, dtype=torch.float64, device="cuda")

        return {
            "m": m, "nnz": nnz, "Len": Len,
            "rowptr": rowptr, "colidx": colidx, "val": val,
            "warp_num": warp_num,
            "get_value": get_value, "x": x, "id_extractor": id_extractor,
            "B": B_dev,
            "cursor": 0,
        }

    def run(self, h):
        i = h["cursor"] % h["B"].shape[0]
        h["cursor"] += 1
        self.lib.yy_solve(
            ctypes.c_void_p(h["rowptr"].data_ptr()),
            ctypes.c_void_p(h["colidx"].data_ptr()),
            ctypes.c_void_p(h["val"].data_ptr()),
            ctypes.c_void_p(h["get_value"].data_ptr()),
            h["m"], h["nnz"],
            ctypes.c_void_p(h["B"][i].data_ptr()),
            ctypes.c_void_p(h["x"].data_ptr()),
            ctypes.c_void_p(h["warp_num"].data_ptr()),
            h["Len"],
            ctypes.c_void_p(h["id_extractor"].data_ptr()))
        return h["x"]

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
