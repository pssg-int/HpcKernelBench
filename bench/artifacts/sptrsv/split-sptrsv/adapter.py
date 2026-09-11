"""
Split_SpTRSV adapter for the sptrsv track.

Paper: "A Split Execution Model for SpTRSV" (TPDS'21,
journals/tpds/AhmadYU21). Artifact: https://github.com/ParCoreLab/Split_SpTRSV

The paper's own CLI (src/main.cpp) drives a CPU-GPU split-execution
pipeline gated behind libUFget (SuiteSparse auto-download) and Intel MKL
(the CPU-side solve) -- neither available/needed here, and per
ARTIFACT_GUIDE.md rule 1 ("wrap the kernel, not the paper's benchmark
script") not wanted anyway. What we actually want is the GPU KERNEL
itself: the paper's synchronization-free CUDA SpTRSV
(source/include/sptrsv_syncfree_cuda.cuh --
sptrsv_syncfree_cuda_analyser / sptrsv_syncfree_cuda_executor), the
classic Liu-et-al. warp-per-row, level-free algorithm the paper's own
split-execution model dispatches to for its GPU-only case (mode 2 in the
CLI). This header has NO dependency on MKL/libUFget/icpc -- only
<cuda_runtime.h> and the artifact's own common.h typedefs -- so it
compiles standalone via plain nvcc; see build.sh / STATUS.md.

shim.cu (this directory, NOT part of the artifact -- same role
bench/artifacts/spmv/sspmv/csr_shim.cpp plays for LeSpMV) #includes that
header VERBATIM and adds two extern "C" host-side launchers around its
two unmodified __global__ kernels; no kernel line was edited (see
STATUS.md diff evidence).

Format: CSC of L, with each column's FIRST stored entry the diagonal and
the remaining entries the rows below it -- exactly what
sptrsv_syncfree_cuda_executor's own indexing assumes (`pos =
d_cscColPtr[col]` read as "the diagonal"). Since L (this track's
canonical unit-lower-triangular derivation,
kernelbench.impls.cpu_ref.unit_lower_triangular -- SAME helper the
CPU_IMPLS entry and the correctness reference use, so every sptrsv
competitor in this benchmark solves the identical triangular system) has,
for column j, only rows i>=j, the diagonal (row==j) is *structurally* the
row-minimum entry in that column, so scipy's ordinary ascending-row CSC
sort (`.tocsc().sort_indices()`) lands it first automatically -- no
manual reordering needed, no kernel code touched.

Per-call state: the sync-free algorithm's in-degree synchronization
counter (`d_graphInDegree`) and partial-sum accumulator (`d_left_sum`)
are BOTH mutated in place by the executor kernel during a solve, so they
must be reset before every call, not just once. This is exactly the case
benchspecs/sptrsv/spec.yaml's `timing_scope` field anticipates ("Any
per-call reset of algorithm-internal state (e.g. synchronization-free
'ready' flag arrays, atomic counters) happens INSIDE the timed region
... as part of a genuine solve call", citing YuenyeungSpTRSV's own
correct cudaMemset-before-timing practice as the model). The one-shot
STRUCTURAL in-degree count (a pure function of L's sparsity pattern,
independent of solve state) is computed ONCE in prepare() via the
artifact's own `sptrsv_syncfree_cuda_analyser` kernel and cached; run()
does a cheap D2D copy of that cached template into a scratch buffer plus
a d_left_sum zero, both inside `split_sptrsv_solve` (shim.cu), before
launching the executor -- so the full "genuine solve call" (reset +
compute) is what CudaEventTimer times, and the one-shot structural
analysis is excluded, per the spec's analysis/solve split.

RHS pool: identical construction to
kernelbench.impls.cpu_ref.ScipySpTRSV.prepare() (same seed/count read
from the same `params` dict the harness passes to every impl in a given
run), so this artifact and the CPU floor solve the exact same draws in a
given run -- same fixed-pool-cycled-by-a-cursor pattern documented there
(and in kernelbench/domains/graph.py's ScipyBFS) for the identical
harness-level "only the first call is gated" caveat.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "sptrsv"
IMPL_NAME = "split-sptrsv-syncfree"
PAPER_KEY = "journals/tpds/AhmadYU21"
PRECISIONS = ["fp64"]  # val_type == double, unconditionally (source/include/common.h)

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
    lib.split_sptrsv_indegree.argtypes = [p, ctypes.c_int, ctypes.c_int, p]
    lib.split_sptrsv_indegree.restype = None
    lib.split_sptrsv_solve.argtypes = [p, p, p, p, p, p, ctypes.c_int, p, p]
    lib.split_sptrsv_solve.restype = None
    lib.split_sptrsv_sync.restype = ctypes.c_int
    return lib


def create(precision: str):
    return SplitSpTRSV(precision)


class SplitSpTRSV:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"{IMPL_NAME} wraps a val_type==double kernel only "
                f"(source/include/common.h); requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, matrix, params: dict):
        import torch
        from kernelbench.impls.cpu_ref import unit_lower_triangular

        # SAME canonical L every sptrsv competitor in this benchmark solves
        # (kernelbench.domains.sparse's shared derivation, reused -- not
        # artifact-specific, see module docstring).
        L64 = unit_lower_triangular(matrix.csr)

        # --- artifact's own format requirement: CSR -> CSC, diagonal-first
        # per column by L's own structure (see module docstring) -- timed
        # as preprocessing, per ARTIFACT_GUIDE.md rule 2.
        Lcsc = L64.tocsc()
        Lcsc.sort_indices()
        m = int(Lcsc.shape[0])
        nnz = int(Lcsc.nnz)

        colptr = torch.as_tensor(Lcsc.indptr.astype(np.int32), device="cuda")
        rowidx = torch.as_tensor(Lcsc.indices.astype(np.int32), device="cuda")
        val = torch.as_tensor(Lcsc.data.astype(np.float64), device="cuda")

        # one-shot structural analysis (see module docstring)
        indeg_template = torch.zeros(m, dtype=torch.int32, device="cuda")
        self.lib.split_sptrsv_indegree(
            ctypes.c_void_p(rowidx.data_ptr()), m, nnz,
            ctypes.c_void_p(indeg_template.data_ptr()))
        err = self.lib.split_sptrsv_sync()
        if err != 0:
            raise RuntimeError(f"split_sptrsv_indegree: CUDA error {err}")

        # fixed RHS pool -- identical construction to ScipySpTRSV.prepare()
        # (see module docstring): U(0,1) draws, b = L @ x_ref via one fp64
        # scipy sparse matvec, never via the technique under test.
        num_rhs = int(params.get("sptrsv_num_rhs", 64))
        seed = params.get("seed", 42)
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.0, 1.0, size=(num_rhs, m))
        # np.ascontiguousarray: (L64 @ X.T).T is a TRANSPOSED VIEW (strided,
        # not C-contiguous -- confirmed via a standalone repro during this
        # integration, see STATUS.md). torch.as_tensor(..., device="cuda")
        # preserves that non-contiguous layout, but shim.cu's split_sptrsv_
        # solve treats `d_b` as a flat contiguous pointer (`d_b[global_x_id]`
        # inside the vendored kernel), so an uncopied strided view would feed
        # the solver SCRAMBLED right-hand sides -- a bug in this adapter's
        # own operand construction, not in the artifact's kernel. Forcing a
        # real contiguous copy here is the fix.
        B = np.ascontiguousarray((L64 @ X.T).T)  # (num_rhs, m)

        params["sptrsv_L_nnz"] = int(L64.nnz)
        params["sptrsv_num_rhs"] = num_rhs
        params["sptrsv_rhs_seed"] = int(seed)
        params["sptrsv_triangular_factor"] = (
            "unit-lower-triangular-of-A (canonical structural proxy; "
            "benchspecs/sptrsv/spec.yaml triangular_factor_derivation)")

        B_dev = torch.as_tensor(B, dtype=torch.float64, device="cuda")

        return {
            "m": m,
            "colptr": colptr, "rowidx": rowidx, "val": val,
            "indeg_template": indeg_template,
            "indeg_work": torch.empty_like(indeg_template),
            "left_sum": torch.zeros(m, dtype=torch.float64, device="cuda"),
            "B": B_dev,
            "x": torch.empty(m, dtype=torch.float64, device="cuda"),
            "cursor": 0,
        }

    def run(self, h):
        i = h["cursor"] % h["B"].shape[0]
        h["cursor"] += 1
        self.lib.split_sptrsv_solve(
            ctypes.c_void_p(h["colptr"].data_ptr()),
            ctypes.c_void_p(h["rowidx"].data_ptr()),
            ctypes.c_void_p(h["val"].data_ptr()),
            ctypes.c_void_p(h["indeg_template"].data_ptr()),
            ctypes.c_void_p(h["indeg_work"].data_ptr()),
            ctypes.c_void_p(h["left_sum"].data_ptr()),
            h["m"],
            ctypes.c_void_p(h["B"][i].data_ptr()),
            ctypes.c_void_p(h["x"].data_ptr()))
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
