"""
Adapter for StraGCN (StraGCN: GPU-Accelerated Strassen's Sparse-Dense Matrix
Multiplication for GNNs, SC'25, CGCL-codes/StraGCN) -- GPU SpMM stage of a
Strassen-decomposed GCN aggregation.

StraGCN's `GCN_ST` torch CUDA extension (source/StraGCN/GNN_strassen.cpp +
strassen.cu, built UNMODIFIED by build.sh) exposes three entry points:
`forward(A, X, W)` = spmm_strassen(A, X@W) (a full GCN layer), `backward`,
and `spmmstra(A_rowPtr, A_colIdx, A_values, offset, X)` = spmm_strassen(A, X)
alone -- exactly this track's kernel boundary, Y = A_hat @ X, so this adapter
calls `spmmstra` directly rather than `forward` (which bundles in the dense
weight GEMM X@W, a second kernel outside this track's scope).

## Preprocessing: reuses the artifact's OWN block-decomposition, unmodified

`spmmstra` does not take a plain CSR triple. It expects A pre-split, via
Strassen's classic 2x2-block 7-multiply decomposition, into 7 sparse
matrices (M1..M7, each `half x half` where half=ceil(V/2)), concatenated
into flat (rowPtr, colIdx, values) arrays plus an `offset` array marking
each block's start within the concatenated colIdx/values. This adapter
imports `split_CSR`/`preAdd` directly from source/StraGCN/dataset.py
(UNCHANGED) and does the flat-array packing GCN.py itself does inline --
this genuinely IS "the artifact's own format conversion" (ARTIFACT_GUIDE.md
rule 2), so it lives entirely in prepare(), timed once as preprocessing.
`split_CSR` is a pure-Python O(nnz) loop (not vectorized) -- a real,
un-worked-around characteristic of this artifact's own preprocessing cost,
not something this adapter hides or speeds up.

Normalization: A_hat = D^-1/2 (A+I) D^-1/2 is computed independently here
(own code, not calling kernelbench.impls.cpu_ref's helpers) -- the SAME
formula this track's domain module (kernelbench/domains/sparse.py) uses,
confirmed to be StraGCN's own convention too: source/StraGCN/dataset.py's
`__main__` block computes `deg_inv = degree**-0.5; values =
deg_inv[row]*deg_inv[col]` on the raw adjacency, the identical GCN symmetric
normalization (StraGCN's own script does not add self-loops explicitly, but
Kipf & Welling's literal A+I is used here to match this domain's reference
exactly, since this adapter's output must clear that reference's gate).

## A real, confirmed artifact bug: silently wrong output for ODD feature width

`Add_total_s`/`Add_finish_s` (strassen.cu) hardcode X's row stride as
`NN = 2*halfn = 2*ceil(F/2)`, but a PyTorch tensor of shape (V, F) has row
stride F. These are equal only when F is EVEN; for odd F, NN != F, and every
row past the first is read/written at a systematically wrong offset --
confirmed empirically (small hand-built cases): F=4 (even) gives max abs
err ~5e-8, F=5 (odd) gives max abs err ~1.75 on values in the O(1) range --
silently wrong, no crash, no exception. (A SEPARATE, more benign effect: odd
V, even F reads/writes a few floats past the end of the X/output buffers'
last row-chunk; empirically harmless on this allocator since the stray
access lands in unused allocation padding and never touches a valid output
row -- observed correct results in that case, unlike odd F.) This track's
spec sweeps F (called N there) over {32, 128, 256, 512}, all even, so gate
verification below is unaffected -- but `prepare()` still asserts F is even
and raises with this explanation for odd F rather than silently returning a
wrong answer, per ARTIFACT_GUIDE.md's correctness-gate-is-non-negotiable
rule (an assertion here is honest; letting an odd-F call through to the gate
would just be a slower way of finding the same bug).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "source", "StraGCN")
BUILD_DIR = os.path.join(HERE, "build")

KERNEL = "gnn-aggregation"
IMPL_NAME = "stragcn-spmmstra-gcnnorm"
PAPER_KEY = "conf/sc/HeLDL0M025"
# GCN_ST's kernels operate entirely in torch::kFloat (fp32); no fp64/fp16
# path exists in strassen.cu.
PRECISIONS = ["fp32"]

_GCN_ST = None
_IMPORT_ERROR = ""


def _load():
    global _GCN_ST, _IMPORT_ERROR
    if _GCN_ST is not None:
        return _GCN_ST
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR)
    try:
        import torch  # noqa: F401 -- must be imported before the extension so its shared libs are resolvable
        if BUILD_DIR not in sys.path:
            sys.path.insert(0, BUILD_DIR)
        import GCN_ST  # noqa: N814 -- matches the extension's actual module name
        _GCN_ST = GCN_ST
        return _GCN_ST
    except Exception as e:
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        raise


def available() -> tuple[bool, str]:
    so_present = any(f.startswith("GCN_ST") and f.endswith(".so")
                      for f in os.listdir(BUILD_DIR)) if os.path.isdir(BUILD_DIR) else False
    if not so_present:
        return False, "GCN_ST*.so not built -- run build.sh"
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


def _gcn_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    """A_hat = D^-1/2 (A+I) D^-1/2, computed independently of
    kernelbench.impls.cpu_ref -- see module docstring."""
    n = A.shape[0]
    A64 = A.astype(np.float64)
    A_tilde = (A64 + sp.identity(n, format="csr", dtype=np.float64)).tocsr()
    A_tilde.sort_indices()
    deg_row = np.asarray(A_tilde.sum(axis=1)).ravel()
    deg_col = np.asarray(A_tilde.sum(axis=0)).ravel()
    dinv_row = np.zeros_like(deg_row)
    dinv_col = np.zeros_like(deg_col)
    nzr, nzc = deg_row > 0, deg_col > 0
    dinv_row[nzr] = deg_row[nzr] ** -0.5
    dinv_col[nzc] = deg_col[nzc] ** -0.5
    A_hat = (sp.diags(dinv_row) @ A_tilde @ sp.diags(dinv_col)).tocsr()
    A_hat.sort_indices()
    return A_hat


def _build_strassen_blocks(A_hat: sp.csr_matrix):
    """
    Split A_hat into StraGCN's 7-block Strassen decomposition and pack it
    into the flat (rowPtr, colIdx, values, offset) layout GCN_ST.spmmstra
    expects -- calling source/StraGCN/dataset.py's OWN split_CSR/preAdd
    (unmodified) for the block split itself, per ARTIFACT_GUIDE.md rule 2
    ("the artifact's own format conversion goes in prepare()"). The flat
    packing loop below mirrors GCN.py's own inline packing (a trivial
    concatenation, not part of "the kernel").
    """
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    from dataset import preAdd, split_CSR  # artifact's own preprocessing, unmodified

    m = A_hat.shape[0]
    half = (m + 1) // 2
    (A11_rp, A11_ci, A11_v, A12_rp, A12_ci, A12_v,
     A21_rp, A21_ci, A21_v, A22_rp, A22_ci, A22_v) = split_CSR(
        A_hat.indptr.tolist(), A_hat.indices.tolist(), A_hat.data.tolist())
    M1, M2, M5, M6, M7 = preAdd(A11_rp, A11_ci, A11_v, A12_rp, A12_ci, A12_v,
                                A21_rp, A21_ci, A21_v, A22_rp, A22_ci, A22_v, half)
    M3 = sp.csr_matrix((A11_v, A11_ci, A11_rp), shape=(half, half))
    M4 = sp.csr_matrix((A22_v, A22_ci, A22_rp), shape=(half, half))

    AR, AC, AVa, offset = [], [], [], []
    offtmp = 0
    for B in (M1, M2, M3, M4, M5, M6, M7):
        offset.append(offtmp)
        offtmp += int(B.nnz)
        AR.extend(int(x) for x in B.indptr)
        AC.extend(int(x) for x in B.indices)
        AVa.extend(float(x) for x in B.data)
    return AR, AC, AVa, offset


class StraGCNAggregation:
    name = "stragcn-spmmstra-gcnnorm"
    platform = "cuda"
    normalization = "gcn-sym: D^-1/2 (A+I) D^-1/2 (independent copy, see module docstring)"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"GCN_ST's kernels are hardcoded torch::kFloat (fp32) in "
                f"strassen.cu; requested precision {precision!r} is not "
                "serviced")
        self.precision = precision

    def prepare(self, matrix, params):
        import torch
        GCN_ST = _load()
        A = matrix.csr
        M, N = A.shape
        if M != N:
            raise NotImplementedError(
                f"StraGCN's Strassen block decomposition assumes a square "
                f"adjacency (one `half` split reused for both dimensions); "
                f"got {M}x{N}")
        F = int(params["N"])
        if F % 2 != 0:
            raise NotImplementedError(
                "GCN_ST's strassen.cu hardcodes the dense operand's row "
                f"stride as 2*ceil(F/2), which only equals the true stride "
                f"F when F is even -- F={F} (odd) triggers a confirmed "
                "silent-wrong-output bug (see module docstring); not "
                "serviced. This track's spec only sweeps even F "
                "(32/128/256/512), so this never blocks a spec-conforming run.")
        seed = params.get("seed", 42)

        A_hat = _gcn_normalize(A)
        params["gnn_agg_nnz"] = int(A_hat.nnz)
        params["gnn_agg_normalization"] = self.normalization

        AR, AC, AVa, offset = _build_strassen_blocks(A_hat)
        X = _dense_operand(M, F, seed, np.float32)

        h = {
            "GCN_ST": GCN_ST,
            "AR": torch.tensor(AR, dtype=torch.int32, device="cuda"),
            "AC": torch.tensor(AC, dtype=torch.int32, device="cuda"),
            "AVa": torch.tensor(AVa, dtype=torch.float32, device="cuda"),
            "offset": torch.tensor(offset, dtype=torch.int32, device="cuda"),
            "X": torch.as_tensor(X, device="cuda"),
        }
        return h

    def run(self, h):
        return h["GCN_ST"].spmmstra(h["AR"], h["AC"], h["AVa"], h["offset"], h["X"])

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
    return StraGCNAggregation(precision)
