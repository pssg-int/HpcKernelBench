"""
Adapter for Fused3S (Fused3S: Fast Sparse Attention on Tensor Cores, ICS'25,
HPCForge/Fused3S) — the FULL fused SDDMM+softmax+SpMM sparse-attention
pipeline, `PAPER_KEY = conf/ics/LiC25`.

This is a SIBLING of `bench/artifacts/sddmm/fused3s/` (already BUILT+GATED),
not a second build: `source` here is a **symlink** to
`../../sddmm/fused3s/source` (`ln -s`), so `build.sh` only verifies the
already-compiled `F3S.cpython-*.so` is present (see its own comments) — no
new C++/CUDA is compiled by this integration; the two adapters wrap
DIFFERENT boundaries of the exact same kernel:

  - sddmm/fused3s/adapter.py: `F3S.f3s_1tb1tcb(..., applySoftmax=False,
    saveSddmmResult=True)` — isolates the raw SDDMM stage only.
  - THIS adapter: `F3S.f3s_1tb1tcb(..., applySoftmax=True,
    saveSddmmResult=False)` — runs the SAME kernel launch through its own
    online-softmax + SpMM stages too, producing the actual attention
    output `O = softmax_row(A ⊙ (QK^T)) @ V`, i.e. this track's own
    `SparseAttentionWorkload` operation.

## Why `coo_arbitrary`, and why not the other 6 structured-mask patterns

Per this module's own instructions and `kernelbench.domains.ml`'s module
docstring, Fused3S's own claim is specifically the GRAPH-ADJACENCY variant
of the sparse-attention-kernel spec (`sparse-attn-graph-adjacency-kernel-
fp16`, real SuiteSparse-style datasets, sparsity a FIXED property of the
data, not a swept ratio) — a variant `kernelbench.domains.ml` does not
implement as a `SparseAttentionWorkload` (it owns the structured/swept-ratio
variant only; see ml.py's module docstring). `SparseAttentionWorkload`'s
`coo_arbitrary` pattern (`_coo_arbitrary_mask`: i.i.d. `Bernoulli(density)`,
"the one pattern in this track with no structural constraint at all", per
ml.py's own comment) is the closest available proxy for an arbitrary graph
adjacency WITHIN this track's existing workload machinery — the same
"closest synthetic proxy for a real structural pattern" logic
`_SPARSE_ATTN_SHAPES`'s own `block-local-s1024-blk64` entry uses for
vit-sparse's RegionViT gate. This adapter therefore raises
`NotImplementedError` for every OTHER named pattern (causal, sliding_window,
longformer_local_global, bigbird_local_global_random, dilated_1d,
block_local) — those are this track's OWN structured-mask regime, already
covered by gpa/sparse-transformer/vit-sparse; wrapping Fused3S against them
would misrepresent a fixed-graph-density kernel as a swept-structured-ratio
one.

## Fused3S consumes a CSR adjacency as the attention pattern

Exactly like the gpa (`spfa_csr`) and sddmm/fused3s adapters: the
workload's shared `(S,S)` bool mask (`w.mask`, built once, identical across
every batch/head per this track's convention — see `reference_sparse_
attention`'s docstring) is converted to CSR ONCE in `prepare()` (this
adapter's own preprocessing, ARTIFACT_GUIDE rule 2) and fed through
`F3S.preprocess_gpu` (the artifact's own CSR -> tensor-core-block layout
conversion) to get `(row_window_offset, sparse_a_to_x_index, tcb_bitmap)` —
reused, UNCHANGED, across every one of the workload's `B*H` independent
attention problems in `run()` (the sparsity pattern is identical for every
batch/head, only Q/K/V values differ — see `_qkv_sparse`'s "same S for
every batch/head" convention). `S` (the workload's sequence length) plays
the role of Fused3S's own `numNodes` (a graph with S nodes, adjacency = the
mask).

## Scale folding: Fused3S's kernel applies NO 1/sqrt(d) internally

Read `f3sKernel1tb1tcb`'s SDDMM section in full (`F3S_kernel.cu`): the raw
`HMMA16816` dot product goes STRAIGHT into the online-softmax max/exp/sum
machinery with no scaling step anywhere in the kernel body or its host
wrapper `f3sCuda1tb1tcb` (confirmed: neither takes a `scalingFactor`
argument, unlike `f3sCuda1tb1rwScheduled`'s sibling in the same file, which
DOES). To match this track's `O = softmax(QK^T/sqrt(d)) @ V` convention,
this adapter folds the `1/sqrt(d)` scale into **Q** before casting to fp16
(`Q' = Q/sqrt(d)`, so `Q'K^T = QK^T/sqrt(d)` exactly) — the standard
"pre-scaled query" technique several production attention kernels use
(mathematically identical to scaling the dot product afterward; the
gpa/spfa_csr adapter needed no such folding because ITS kernel already
applies `dot(Q_i,K_j)/sqrt(d)` internally — see that adapter's own
docstring point 1).

## Masking: the TC-block bitmap IS the exclusion mechanism

Non-edge (query,key) pairs are excluded from the softmax entirely via the
kernel's own TC-block bitmap (`addPartialSums`'s `sum[ind]=0.0f` for
non-edge slots, then `if(D_frag[i]!=0.0f) D_frag[i]=exp(...)` else left at
literal `0` — i.e. masked positions contribute exactly `0` to both the
softmax numerator and denominator, the same effect as an additive `-inf`
mask). `SparseAttentionWorkload`'s mask always keeps the diagonal (every
row has >=1 valid edge, forced in `_build_sparse_mask`), so no row's
softmax denominator is identically zero and no all-masked-row edge case
arises. One caveat inherited from the kernel's own design (documented, not
worked around): a genuine (unmasked) raw score that happens to equal
EXACTLY `0.0` in fp32 would be misclassified as "masked" by this bitwise
check — vanishingly unlikely for continuous N(0,1)-derived operands, and a
property of the artifact, not this adapter.

Batching: Fused3S's kernel signature has no batch/head dimension (`Q`/`K`/
`V` are 2D `(numNodes, embeddingDim)`) — `run()` loops over the workload's
`B*H` independent attention problems, calling the SAME preprocessed layout
once per (batch,head) slice, exactly the gpa/spfa_csr adapter's precedent.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "source", "src")

KERNEL = "sparse-attention-kernel"
IMPL_NAME = "fused3s-1tb1tcb-attn"
PAPER_KEY = "conf/ics/LiC25"
# Tensor-core fp16-in/fp32-accumulate arithmetic; no separate fp32 compute
# path exists in the artifact (see sddmm/fused3s/adapter.py's identical note).
PRECISIONS = ["fp16"]

BLK_H = 16   # config.h
BLK_N = 8    # config.h BLK_W / F3S_kernel.cu BLK_N

_F3S = None
_IMPORT_ERROR = ""


def _load():
    global _F3S, _IMPORT_ERROR
    if _F3S is not None:
        return _F3S
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR)
    try:
        if SRC_DIR not in sys.path:
            sys.path.insert(0, SRC_DIR)
        import F3S  # noqa: N814 -- matches the extension's actual module name
        _F3S = F3S
        return _F3S
    except Exception as e:
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        raise


def available() -> tuple[bool, str]:
    if not os.path.islink(os.path.join(HERE, "source")):
        return False, "source/ is not a symlink to ../../sddmm/fused3s/source"
    so_present = os.path.isdir(SRC_DIR) and any(
        f.startswith("F3S.cpython") and f.endswith(".so") for f in os.listdir(SRC_DIR))
    if not so_present:
        return False, "F3S*.so not found under source/src -- build ../../sddmm/fused3s/ first"
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


def _qkv_like_ml(B, H, S, d, seed, np_dtype):
    """Bit-identical to kernelbench.domains.ml._qkv_sparse -- see this
    track's other adapters (gpa, vit-sparse, sparse-transformer) for the
    same, independently-typed pattern."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    K = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    V = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    return Q, K, V


class Fused3SSparseAttention:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} only computes in fp16/fp32-accumulate tensor-core "
                f"arithmetic; requested precision {precision!r} is not serviced")
        self.precision = precision

    def prepare(self, workload, params: dict):
        import torch
        F3S = _load()

        w = workload
        if w.kernel != "sparse-attention-kernel":
            raise NotImplementedError(f"{IMPL_NAME} only services sparse-attention-kernel")
        if w.pattern != "coo_arbitrary":
            raise NotImplementedError(
                f"{IMPL_NAME} only wraps the coo_arbitrary (arbitrary-sparsity, "
                f"closest available proxy for a real graph adjacency) pattern -- "
                f"Fused3S's own claim is the graph-adjacency regime, not this "
                f"track's structured/swept-ratio mask families; got pattern={w.pattern!r}")

        B, H, d, S = w.B, w.H, w.d, w.S
        seed = params.get("seed", w.seed)
        np_dtype = np.float16
        Q, K, V = _qkv_like_ml(B, H, S, d, seed, np_dtype)

        # ---- artifact's own CSR -> TC-block preprocessing, done ONCE (the
        # pattern is identical for every batch/head -- see module docstring).
        mask_np = w.mask  # (S,S) bool, shared workload data
        row_idx, col_idx = np.nonzero(mask_np)
        order = np.argsort(row_idx, kind="stable")
        row_idx, col_idx = row_idx[order], col_idx[order]
        indptr = np.zeros(S + 1, dtype=np.int32)
        np.add.at(indptr, row_idx + 1, 1)
        indptr = np.cumsum(indptr).astype(np.int32)
        indices = col_idx.astype(np.int32)

        indices_t = torch.as_tensor(indices, dtype=torch.int32, device="cuda").contiguous()
        indptr_t = torch.as_tensor(indptr, dtype=torch.int32, device="cuda").contiguous()
        num_row_windows = (S + BLK_H - 1) // BLK_H
        block_partition = torch.zeros(num_row_windows, dtype=torch.int32, device="cuda")
        nnz = int(indices.shape[0])
        edge_to_column = torch.zeros(nnz, dtype=torch.int32, device="cuda")
        edge_to_row = torch.zeros(nnz, dtype=torch.int32, device="cuda")

        (row_window_offset, _sorted_row_windows, _tcblock_rowid, _tcblocktile_id,
         _tcblock_offset, sparse_a_to_x_index, tcb_bitmap, _n_tcb) = F3S.preprocess_gpu(
            indices_t, indptr_t, S, BLK_H, BLK_N, block_partition, edge_to_column, edge_to_row)

        # ---- scale folded into Q (see module docstring "Scale folding").
        scale = 1.0 / math.sqrt(d)
        Qs = (Q.astype(np.float32) * scale).astype(np.float16)

        # (B,H,S,d) -> B*H independent (S,d) fp16 CUDA tensors, this
        # kernel's own single-graph 2D interface (see module docstring
        # "Batching").
        q_slices = [torch.from_numpy(np.ascontiguousarray(Qs[b, h])).to("cuda")
                    for b in range(B) for h in range(H)]
        k_slices = [torch.from_numpy(np.ascontiguousarray(K[b, h])).to("cuda")
                    for b in range(B) for h in range(H)]
        v_slices = [torch.from_numpy(np.ascontiguousarray(V[b, h])).to("cuda")
                    for b in range(B) for h in range(H)]

        self._last_bhsd = (B, H, S, d)

        return {
            "F3S": F3S, "row_window_offset": row_window_offset,
            "sparse_a_to_x_index": sparse_a_to_x_index, "tcb_bitmap": tcb_bitmap,
            "S": S, "q_slices": q_slices, "k_slices": k_slices, "v_slices": v_slices,
        }

    def run(self, h):
        F3S = h["F3S"]
        outs = []
        for q, k, v in zip(h["q_slices"], h["k_slices"], h["v_slices"]):
            _time, output, _sddmm = F3S.f3s_1tb1tcb(
                h["row_window_offset"], h["sparse_a_to_x_index"], h["tcb_bitmap"],
                h["S"], q, k, v,
                True,    # applySoftmax -- the FULL fused pipeline this adapter wraps
                False)   # saveSddmmResult -- not needed here
            outs.append(output)
        return outs

    def to_host(self, outs):
        import torch
        B, H, S, d = self._last_bhsd
        stacked = torch.stack(outs, dim=0).reshape(B, H, S, d)
        return stacked.detach().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return Fused3SSparseAttention(precision)
