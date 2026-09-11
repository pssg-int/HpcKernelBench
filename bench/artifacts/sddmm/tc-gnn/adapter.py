"""
Adapter for TC-GNN (TC-GNN: Bridging Sparse GNN Computation and Dense Tensor
Cores on GPUs, USENIX ATC'23, YukeWang96/TC-GNN_ATC23) -- sddmm track.

REUSE, NOT A REBUILD (task brief): this artifact is already built for the
gnn-aggregation track at `../../gnn-aggregation/tc-gnn/` (`build.sh`
compiled `source/TCGNN_conv/{TCGNN.cpp,TCGNN_kernel.cu}`, unmodified, into
that directory's `build/TCGNN*.so`). This adapter wraps a DIFFERENT entry
point from the same compiled extension: `TCGNN.forward_ef`
(`TCGNN.cpp`'s pybind name for `sddmm_forward` -> `sddmm_forward_cuda` ->
`sddmm_forward_cuda_kernel` in `TCGNN_kernel.cu`) -- TC-GNN's own SDDMM
kernel, read directly from source (task brief instruction), not assumed
from the paper.

## What `forward_ef` actually computes (read directly from TCGNN_kernel.cu)

`sddmm_forward_cuda_kernel` takes ONE dense input matrix `in_mat` (shape
`[num_nodes, embedding_dim]`) plus the SAME structural
nodePointer/edgeList/blockPartition/edgeToColumn/edgeToRow TC-GNN's
`preprocess()` builds for `forward`/`forward_AGNN`, and computes, for every
edge `(row, col)` in the sparsity pattern:
`edgeFeature[eIdx] = dot(in_mat[row], in_mat[col])` -- a WMMA (tf32)
tensor-core `X @ X^T`, SAMPLED at the sparsity pattern, using the SAME
input matrix for both the "row" and "column" operand (confirmed by
reading the kernel: `dense_X`/`dense_Y` are both loaded from the SAME
`in_mat` pointer, one row-major and one column-major, no second tensor
argument anywhere in `sddmm_forward`'s signature). Output is written at
`edgeFeature[eIdx]` where `eIdx` is the ORIGINAL edge-list index (no
TC-block-layout reindex on the way out, unlike Fused3S's
`saveSddmmResult` -- confirmed by reading the store loop:
`edgeFeature[eIdx] = sparse_A_val[t]` with `eIdx` taken from
`sparse_A[...]`, which was populated from `eIdx = tid + eIdx_start` over
the SAME `eIdx_start..eIdx_end` range `nodePointer` defines -- i.e. the
output lands in the SAME CSR nnz order as the edge list we feed in).
Unlike RoDe's SDDMM (which multiplies by the sparse value `S[i,j]`
INSIDE the kernel), `forward_ef` is an UNWEIGHTED dot product -- like
Fused3S (`../../sddmm/fused3s/adapter.py`), `S[i,j]` is applied
POST-HOC in `to_host()` here.

## The "self-SDDMM only" constraint, and how it is bridged (adapter-side, no kernel touch)

`forward_ef` structurally CANNOT take two independent dense operands: its
signature has only one `input` tensor. This track's own reference
(`kernelbench.impls.cpu_ref.reference_sddmm`) is the GENERAL two-operand
form, `out[i,j] = S[i,j] * dot(A[i,:], B[j,:])` with A (M x K, row-indexed,
seed) and B (N x K, col-indexed, seed+1) drawn INDEPENDENTLY -- i.e. A != B
in general, even when M==N. Every other adapter in this track that hits
this same gap (Fused3S) has a genuine two-operand Q/K kernel argument to
route A and B through; TC-GNN's `forward_ef` does not.

**Bridge used here: a bipartite double-cover / node-doubling embedding.**
Build a combined feature matrix `Z` of shape `(M+N) x K` with `Z[0:M] = A`,
`Z[M:M+N] = B`, and re-index the sparsity pattern's own edges from
`(i, j)` (`i` in `[0,M)`, `j` in `[0,N)`) to `(i, M+j)` -- i.e. treat A's M
rows and B's N rows as `M+N` DISTINCT nodes in one combined node space,
with every original edge now going from an "A-node" to a "B-node" (a
shifted column id, `nodePointer` extended with `N` more, permanently
zero-out-degree, rows for the B-side). `forward_ef(Z, ...)` then computes
`dot(Z[i], Z[M+j]) = dot(A[i], B[j])` for exactly this domain's own
reference formula, at exactly the ORIGINAL nnz positions (edge order is
preserved: the edge list is built directly from `S.indptr`/`S.indices`,
just with column ids shifted by `M`, so `forward_ef`'s own eIdx-preserving
output land order matches `S`'s own CSR order 1:1, no permutation step
needed). This is pure ADAPTER-SIDE preprocessing (an index remap + a
concatenation of the two dense operands) -- no kernel code is touched, and
it computes the exact SAME mathematical quantity the general two-operand
SDDMM does, not an approximation. Verified numerically against a
brute-force numpy computation before wiring into this adapter (non-square
M!=N and cora-sized square cases, both passed at TF32-level scaled error,
~2e-4 -- see the accompanying validation notes in STATUS.md).

Cost: TC-GNN's own `preprocess()`/kernel now iterate over `M+N` nodes
(instead of just `M` or `N`) and `(M+N+15)//16` row windows, roughly
doubling its own structural bookkeeping relative to a hypothetical
native two-operand kernel -- irrelevant to correctness, and this
integration performs no timing measurement (ARTIFACT_GUIDE.md rule 5), but
noted for anyone reading these numbers as throughput later.

## Two real, confirmed findings from reading + testing `sddmm_forward_cuda_kernel` directly

**1. Confirmed one-element out-of-bounds WRITE (heap corruption) whenever
`num_nodes` (here, `M+N`, since BOTH sides are folded into one combined
node space) is an exact multiple of `BLK_H=16`.** `TCGNN.cpp`'s
`preprocess()` is the SAME shared structural routine `forward`/
`forward_AGNN` use (see `../../gnn-aggregation/tc-gnn/STATUS.md` finding
#2 for the full canary-buffer reproduction, not re-run here since it is
the same compiled code) -- its row-window loop is off-by-one,
`for (iter=0; iter<num_nodes+1; iter+=16)`, writing one element past
`blockPartition`'s valid range when `num_nodes % 16 == 0`. Guarded:
`prepare()` raises `NotImplementedError` for `(M+N) % 16 == 0`.

**2. NEW finding, specific to `sddmm_forward_cuda_kernel` (not shared with
`forward`/`forward_AGNN`): confirmed silent data corruption for
`embedding_dim` (K) not a multiple of `BLK_W=8`.** The kernel's dense-tile
staging loop (`TCGNN_kernel.cu`, inside `sddmm_forward_cuda_kernel`'s
`warp_iter` loop) bounds-checks each fetch only against the TOTAL matrix
size, `source_idx >= numNodes * embedding_dim`, NOT against the CURRENT
row's own valid range (`row*embedding_dim + embedding_dim`). For the last
`warp_iter` chunk when `embedding_dim % BLK_W != 0`, `dense_dimIdx` values
past `embedding_dim % BLK_W` push `source_idx` into the START of the NEXT
node's row (row-major, stride `embedding_dim`) instead of being
zero-padded -- silently mixing in a DIFFERENT node's feature values rather
than reading garbage or crashing. **Confirmed empirically** (standalone
script, not the domain's own dims): K=20 and K=33 (`M=50,N=70`, random
sparsity) both show `max_scaled_err` of `0.25`-`0.55` (catastrophic, two to
three orders of magnitude above TF32 noise); K=24 (a multiple of 8) on the
SAME shapes/seeds shows ordinary `~2.3e-4` TF32-level error. Guarded:
`prepare()` raises `NotImplementedError` for `K % BLK_W (8) != 0`. This
track's own dims sweep for `sddmm-tensorcore-blocked-fp16` is `{32, 128}`
-- both multiples of 8 -- so this guard is never exercised by a
spec-conforming run of THIS variant, but is reported since it was found
while reading/testing the kernel this adapter wraps (ARTIFACT_GUIDE.md
rule 8's spirit: name the constraint even when the track's own sweep
happens not to hit it).

## Fairness (S[i,j] applied post-hoc, same pattern as Fused3S)

`forward_ef` never reads `S.data` -- `to_host()` multiplies the kernel's
raw dot-product output by `S[i,j]` (stashed in `prepare()`, same CSR nnz
order the kernel already outputs in, no reindex needed) exactly the way
`../../sddmm/fused3s/adapter.py` and
`kernelbench.impls.gpu_cuda.TorchSDDMM` do for their own unweighted
kernels.

## Compute precision

Same TF32 WMMA tensor-core family as `forward`/`forward_AGNN`
(`wmma::precision::tf32`, fp32 accumulate). `sddmm`'s dedicated
tensor-core-class variant, `sddmm-tensorcore-blocked-fp16` (tolerance
`1e-2`), is exactly the right home (same convention as
`../../sddmm/fused3s`/`../../sddmm/flashsparse`). `PRECISIONS = ["fp16"]`.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_GNNAGG_TCGNN = os.path.join(HERE, "..", "..", "gnn-aggregation", "tc-gnn")
_BUILD_DIR = os.path.join(_GNNAGG_TCGNN, "build")   # reused, not local

KERNEL = "sddmm"
IMPL_NAME = "tcgnn-ef-sddmm-f16"
PAPER_KEY = "conf/usenix/WangFWHD23"
PRECISIONS = ["fp16"]

BLK_H = 16
BLK_W = 8

_TCGNN = None
_IMPORT_ERROR = ""


def _load():
    global _TCGNN, _IMPORT_ERROR
    if _TCGNN is not None:
        return _TCGNN
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR)
    try:
        import torch  # noqa: F401 -- must be imported before the extension so its shared libs are resolvable
        if _BUILD_DIR not in sys.path:
            sys.path.insert(0, _BUILD_DIR)
        import TCGNN  # noqa: N814 -- matches the extension's actual module name
        _TCGNN = TCGNN
        return _TCGNN
    except Exception as e:
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        raise


def available() -> tuple[bool, str]:
    so_present = (os.path.isdir(_BUILD_DIR)
                  and any(f.startswith("TCGNN") and f.endswith(".so") for f in os.listdir(_BUILD_DIR)))
    if not so_present:
        return False, (f"reused build/ not found at {_BUILD_DIR} -- run "
                        f"../../gnn-aggregation/tc-gnn/build.sh (this "
                        f"track's own build.sh only symlinks source/, it "
                        f"does not compile) then this artifact's build.sh")
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
    # as a separate copy per the reference-independence rule.
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(dtype)


class TCGNNSDDMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME}: TCGNN_kernel.cu's WMMA fragments are "
                f"hardcoded torch::kFloat (fp32) tensors (internally TF32 "
                f"tensor-core compute); requested precision {precision!r} "
                "is not serviced (run with --precision fp16 against "
                "sddmm-tensorcore-blocked-fp16, see STATUS.md)")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        TCGNN = _load()

        S = matrix.csr
        M, N = S.shape
        K = int(params["K"])
        if K % BLK_W != 0:
            raise NotImplementedError(
                f"{IMPL_NAME}: sddmm_forward_cuda_kernel's dense-tile "
                f"staging loop bounds-checks each fetch only against the "
                f"TOTAL matrix size (numNodes*embedding_dim), not the "
                f"current row's own range -- for embedding_dim (K) not a "
                f"multiple of BLK_W={BLK_W}, the tail of the last "
                f"warp_iter chunk silently reads the NEXT node's feature "
                f"row instead of being zero-padded (confirmed empirically: "
                f"K=20/33 on a random M=50,N=70 pattern gate at "
                f"max_scaled_err 0.25-0.55, vs ~2e-4 at K=24; see module "
                f"docstring finding #2). K={K} is not a multiple of "
                f"{BLK_W}, not serviced.")
        num_nodes = M + N   # doubled-node-space embedding, see module docstring
        if num_nodes % BLK_H == 0:
            raise NotImplementedError(
                f"{IMPL_NAME}: TCGNN.cpp's preprocess() writes "
                f"blockPartition[num_nodes/{BLK_H}] one element past the "
                f"end of the num_row_windows-sized tensor whenever "
                f"num_nodes is an exact multiple of BLK_H={BLK_H} "
                f"(confirmed for the same shared preprocess() by "
                f"../../gnn-aggregation/tc-gnn/STATUS.md finding #2). "
                f"This adapter's combined node space is M+N={num_nodes} "
                f"(see module docstring's bipartite-embedding bridge), "
                f"which is a multiple of {BLK_H}, not serviced.")

        seed = params.get("seed", 42)
        A = _dense_operand(M, K, seed, np.float32)
        B = _dense_operand(N, K, seed + 1, np.float32)

        nnz = int(S.nnz)
        indptr = np.ascontiguousarray(S.indptr, dtype=np.int64)
        indices = np.ascontiguousarray(S.indices, dtype=np.int64)

        # --- bipartite double-cover embedding (module docstring) ---
        Z = np.zeros((num_nodes, K), dtype=np.float32)
        Z[:M] = A
        Z[M:] = B
        node_pointer = np.empty(num_nodes + 1, dtype=np.int32)
        node_pointer[: M + 1] = indptr.astype(np.int32)
        node_pointer[M + 1:] = nnz  # B-side nodes: flat, zero out-degree
        edge_list = (indices + M).astype(np.int32)  # shift into B's node-id space

        num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
        row_pointers = torch.tensor(node_pointer)
        column_index = torch.tensor(edge_list)
        blockPartition = torch.zeros(num_row_windows, dtype=torch.int32)
        edgeToColumn = torch.zeros(nnz, dtype=torch.int32)
        edgeToRow = torch.zeros(nnz, dtype=torch.int32)

        TCGNN.preprocess(column_index, row_pointers, num_nodes,
                          BLK_H, BLK_W, blockPartition, edgeToColumn, edgeToRow)

        Z_t = torch.as_tensor(Z, device="cuda")

        # S[i,j] applied post-hoc in to_host() (module docstring "Fairness").
        self._last_s_values = np.ascontiguousarray(S.data, dtype=np.float64)

        h = {
            "TCGNN": TCGNN,
            "Z": Z_t,
            "row_pointers": row_pointers.cuda(),
            "column_index": column_index.cuda(),
            "blockPartition": blockPartition.cuda(),
            "edgeToColumn": edgeToColumn.cuda(),
            "edgeToRow": edgeToRow.cuda(),
        }
        return h

    def run(self, h):
        return h["TCGNN"].forward_ef(
            h["Z"], h["row_pointers"], h["column_index"],
            h["blockPartition"], h["edgeToColumn"], h["edgeToRow"])[0]

    def to_host(self, out):
        import torch
        raw = out.detach().to("cpu", dtype=torch.float64).numpy()
        return raw * self._last_s_values

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return TCGNNSDDMM(precision)
