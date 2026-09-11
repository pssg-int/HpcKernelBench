"""
Adapter for TLPGNN (TLPGNN: A Lightweight Two-Level Parallelism Paradigm for
Graph Neural Network Computation on GPU, HPDC'22, charlifu/TLPGNN) -- GPU
GCN-aggregation kernel.

## What boundary was wrapped

`source/gcn/naive_kernel.cu`'s `gcn_conv_cuda_forward` is the kernel the
artifact's OWN documented entry point (`README.md`: "cd gcn/ && python
test_kernel.py --dataset citeseer --size 32") JIT-compiles and calls
directly (`source/gcn/test_kernel.py` -- `torch.utils.cpp_extension.
load_inline(cuda_sources=[open("naive_kernel.cu").read()])`, then
`gcn_module.forward(features, col_starts, rows)`). This adapter calls the
SAME `forward` entry point against the SAME kernel source, AOT-compiled
instead of JIT'd (see build.sh), with our workload's adjacency converted to
the CSC layout the kernel expects.

`source/gcn/atomic_kernel.cu` (a SEPARATE kernel using a global atomic work
queue for dynamic vertex scheduling, exercised only by the undocumented
`test_atomic_kernel.py`, which hardcodes a personal absolute dataset path
`/mnt/raid0_ssd_8tb/qiang/...`) computes the IDENTICAL per-vertex formula as
`naive_kernel.cu` -- only the work-distribution strategy differs (static
`blockIdx`-based vs. dynamic atomic-counter-based). Both are exactly this
domain's kernel-only aggregation call; this adapter wraps `naive_kernel.cu`
specifically because it is the one the artifact's own README instructs a
user to run.

## CRITICAL FINDING: the kernel computes ROW-MEAN aggregation, not this
## domain's GCN-symmetric-normalized reference -- and ignores edge WEIGHTS entirely

Read `gcn_conv_cuda_forward_kernel` in full: for each destination vertex
`des_v`, it sums `features[rows[i]]` over `i` in `[col_starts[des_v],
col_starts[des_v+1])` and multiplies by `deg = 1.0/(e_pos-s_pos)` -- this is
**D^-1 A @ X** (unweighted row-mean neighbor aggregation, GraphSAGE-mean
style), computed from `col_starts`/`rows` alone: **there is no per-edge
value parameter anywhere in `forward`'s signature** (`col_starts`, `rows`
only -- no `values` tensor). Two consequences, both load-bearing for the
correctness gate:

1. **Normalization mismatch.** This domain's ONE reference
   (`kernelbench.impls.cpu_ref.reference_gnn_aggregation`, per
   DOMAIN_GUIDE.md's one-CPU_IMPLS-entry-per-kernel contract) is
   `D^-1/2(A+I)D^-1/2 @ X` (GCN symmetric normalization). `D^-1 A @ X`
   (mean, no self-loop) and `D^-1/2(A+I)D^-1/2 @ X` (symmetric, +I) are
   NOT related by any diagonal pre/post-scaling of the dense operand `X`
   in general -- unlike TC-GNN's `forward_AGNN` (see
   `artifacts/gnn-aggregation/tc-gnn/adapter.py`), TLPGNN's kernel exposes
   no value channel to inject the correct per-edge weights through, so
   there is no way to make this kernel compute the domain's reference
   without editing kernel code (out of scope, rule 3). This adapter
   therefore feeds the kernel EXACTLY what its own `test_kernel.py` would
   (the raw structural adjacency, no self-loop added, matching the
   artifact's own convention) and reports the resulting correctness-gate
   comparison against this domain's symmetric-norm reference AS A GENUINE
   RESULT (ARTIFACT_GUIDE.md rule 4): a real, expected, unforced gate
   failure caused by a structural normalization mismatch, not a numerical
   precision issue -- this is BUILT + wrapped + gated, not a SKIP, because
   the kernel does implement a real, well-known sparse-adjacency
   aggregation (same nnz*N flop shape this track's `_cost_gnn_agg` charges
   for), just not the one specific normalization this domain references.
2. **Edge weights are structurally ignored.** Even where `A`'s nonzero
   VALUES are meaningful (e.g. this domain's synthetic smoke matrices carry
   real, nontrivial weights), the kernel only ever reads the CSC
   structure (`col_starts`/`rows`) -- every structural neighbor contributes
   `features[neighbor] * (1/degree)` regardless of what `A[i,j]` actually
   was. Not a bug: this is a documented design point of mean/GraphSAGE-style
   aggregation kernels (most GNN benchmark graphs are unweighted anyway),
   but it means the gate failure below is not merely "wrong normalization
   constant" -- it is "a different aggregation altogether" for any weighted
   input.

## Preprocessing (prepare(), timed once)

`test_kernel.py` loads its graph via `sp.load_npz(...).tocsc()`, i.e. the
kernel consumes the ADJACENCY IN CSC FORM (`col_starts` = CSC `indptr`,
`rows` = CSC `indices`) so that iterating column `des_v`'s slice yields
`des_v`'s IN-neighbors under the "destination-major" convention the kernel
uses. `prepare()` reproduces this exactly: `matrix.csr.tocsc()`, then
`sort_indices()` (matching `test_kernel.py`'s own
`ret['graph'].sort_indices()` call) -- this genuinely is the artifact's own
format conversion (ARTIFACT_GUIDE.md rule 2), timed as preprocessing.

## Compute precision

`AT_DISPATCH_FLOATING_TYPES` dispatches on `features.scalar_type()` and
covers BOTH `float` and `double` -- unlike TC-GNN, TLPGNN's kernel genuinely
supports fp64 as well as fp32 (no tensor-core reduced-precision path at
all; it's a plain warp/thread SpMM). `PRECISIONS = ["fp32"]` here only
because this domain's ONE implemented precision is fp32
(`DEFAULT_PRECISION["gnn-aggregation"] = "fp32"`); the kernel itself would
accept `precision="fp64"` unmodified if this domain ever added an fp64
variant.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, "build")

KERNEL = "gnn-aggregation"
IMPL_NAME = "tlpgnn-naive-mean-agg"
PAPER_KEY = "conf/hpdc/FuJH22"
# The kernel dispatches on the input tensor's own dtype (AT_DISPATCH_FLOATING_TYPES
# covers float and double); fp32 here only because this domain's one implemented
# CPU_IMPLS/reference precision is fp32. See module docstring's "Compute precision".
PRECISIONS = ["fp32"]

_MOD = None
_IMPORT_ERROR = ""


def _load():
    global _MOD, _IMPORT_ERROR
    if _MOD is not None:
        return _MOD
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR)
    try:
        import torch  # noqa: F401
        if BUILD_DIR not in sys.path:
            sys.path.insert(0, BUILD_DIR)
        import tlpgnn_gcn
        _MOD = tlpgnn_gcn
        return _MOD
    except Exception as e:
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        raise


def available() -> tuple[bool, str]:
    so_present = any(f.startswith("tlpgnn_gcn") and f.endswith(".so")
                      for f in os.listdir(BUILD_DIR)) if os.path.isdir(BUILD_DIR) else False
    if not so_present:
        return False, "tlpgnn_gcn*.so not built -- run build.sh"
    try:
        _load()
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        return True, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg:
            msg = ("libstdc++ ABI mismatch -- run with "
                   "LD_PRELOAD=/usr/lib64/libstdc++.so.6 (see STATUS.md)")
        return False, msg


def _dense_operand(rows: int, dim: int, seed: int, dtype) -> np.ndarray:
    # identical formula to kernelbench.impls.cpu_ref._dense_operand -- kept
    # as a separate copy (not a shared code object) per the
    # reference-independence rule in DOMAIN_GUIDE.md.
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(dtype)


class TLPGNNAggregation:
    name = IMPL_NAME
    platform = "cuda"
    normalization = "row-mean D^-1 A (NO self-loop, edge weights ignored) -- see module docstring"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp32 here (the kernel itself "
                f"supports fp64 too via AT_DISPATCH_FLOATING_TYPES, but this "
                f"domain has no fp64 variant to gate against); requested "
                f"{precision!r}")
        self.precision = precision

    def prepare(self, matrix, params):
        import torch
        mod = _load()
        A = matrix.csr
        M, N = A.shape
        if M != N:
            raise NotImplementedError(
                f"TLPGNN's kernel treats a single node count for both the "
                f"CSC column range and the row-index space; got a {M}x{N} "
                f"non-square matrix, not serviced")
        if A.nnz == 0:
            raise NotImplementedError(
                "empty adjacency: every column would divide by "
                "deg=1/(e_pos-s_pos)=1/0, not serviced")
        # any isolated vertex (zero in-degree in the CSC sense) divides by
        # zero inside the kernel itself (1.0/(e_pos-s_pos) with e_pos==s_pos)
        # -- a real, unguarded constraint of the shipped kernel, not
        # something this adapter works around.
        Acsc_probe = A.tocsc()
        indptr = Acsc_probe.indptr
        if np.any(np.diff(indptr) == 0):
            raise NotImplementedError(
                "gcn_conv_cuda_forward_kernel computes deg = 1.0/(e_pos-s_pos) "
                "unconditionally; a vertex with zero in-neighbors (isolated "
                "column in CSC) divides by zero. This matrix has >=1 such "
                "vertex, not serviced by the kernel as shipped.")
        F = int(params["N"])
        seed = params.get("seed", 42)

        # Artifact's own format conversion (see module docstring): CSR -> CSC,
        # matching test_kernel.py's read_data()/sort_indices() exactly. No
        # self-loop is added -- the artifact's own script does not add one
        # either (see module docstring's normalization-mismatch finding).
        Acsc = A.tocsc()
        Acsc.sort_indices()
        params["gnn_agg_nnz"] = int(Acsc.nnz)
        params["gnn_agg_normalization"] = self.normalization

        col_starts = torch.tensor(Acsc.indptr, dtype=torch.int32, device="cuda")
        rows_t = torch.tensor(Acsc.indices, dtype=torch.int32, device="cuda")
        X = _dense_operand(N, F, seed, np.float32)
        features = torch.as_tensor(X, device="cuda")

        return {"mod": mod, "features": features, "col_starts": col_starts, "rows": rows_t}

    def run(self, h):
        return h["mod"].forward(h["features"], h["col_starts"], h["rows"])[0]

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


def create(precision: str):
    return TLPGNNAggregation(precision)
