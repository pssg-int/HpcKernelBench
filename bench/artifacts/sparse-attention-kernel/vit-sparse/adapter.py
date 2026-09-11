"""
Adapter for vit-sparse (IPDPS'26 "Achieving Low Latency Inference on High
Resolution Images by Exploiting Sparsity in Vision Transformers",
KLab-AI3/ipdps-26_vit). `PAPER_KEY = conf/ipps/LiK26`.

The artifact's full pipeline is T1 adjacency -> T2 reorder+block-extraction
-> T3 tile profiling -> T4 ILP (Gurobi) scheduling -> T5 benchmarking. This
adapter wraps T2's block extraction (`src/extract_blocks.py::extract_blocks`,
unmodified) as PREPROCESSING (ARTIFACT_GUIDE rule 2 -- structural block
partitioning from the mask, timed once) and T5's actual per-block kernel call
(`src/run_bench.py::attn_block`, unmodified -- an SDPA call with an additive
mask bias built from the block's own true adjacency submask via
`build_block_mask_from_adj`, also unmodified) as the timed kernel, run once
per extracted block per batch element.

T4 (Gurobi ILP scheduling) is deliberately NOT exercised: reading
`src/run_bench.py`'s actual argparse (the ONLY script T5 in the README's
stage-by-stage doc claims to consume `--schedule`, but the released script's
CLI has no such argument -- it takes `--adj`/`--blocks` directly) confirms
the ILP schedule is not actually wired into T5's execution path at all in
this release; T5 runs directly off `--blocks` (T2's own output), independent
of T3/T4. This is a real README-vs-code mismatch (same species of finding as
the survey's STOF "100 iterations" note) -- not a shortcut this adapter
invented. No Gurobi dependency is therefore needed to exercise the artifact's
actual per-block attention kernel.

FINDING, not an artifact bug per se but a real integration boundary (see
STATUS.md "Finding" section for the full derivation): `attn_block()` computes
an INDEPENDENT, block-local softmax with no cross-block online-softmax
accumulator (confirmed by reading the whole file -- no running max/sum state
is threaded between the per-block calls in `run_bench.py`'s main loop; each
block's CSV row is an isolated latency measurement). This is mathematically
correct as a wrapper of the FULL SxS masked-attention kernel (comparable to
this track's fp64 reference) ONLY when every query row's entire mask support
falls inside exactly ONE extracted block -- otherwise two blocks would each
normalize their own partial softmax independently, which does not equal a
joint per-row softmax over the full row support. `prepare()` therefore
extracts blocks from the workload's own fixed mask and VALIDATES this
single-block-containment property before proceeding, raising
NotImplementedError (the sanctioned "shape-constrained artifact" behavior,
same as PAT's/Fused3S's adapters) if it does not hold for the given
workload/tile-size combination, rather than adding the missing cross-block
merge logic into the artifact's own code (ARTIFACT_GUIDE rule 3: patch
minimally, never touch kernel/algorithm logic).

Gate workload: `block_local` pattern with `n_global_blocks=0` (pure
block-diagonal, no CLS-style global block) and a column/row tile size
(`max_h=max_w`) matching the pattern's own `block` parameter -- this is the
ONE structured-mask family in this module for which single-block containment
holds EXACTLY (verified empirically: 16/16 blocks, 0 rows split, for
S=1024/block=64 -- see STATUS.md), giving a genuinely non-degenerate
block-sparse execution (16 separate 64x64-tile SDPA calls instead of one
1024x1024 dense call) through the artifact's own, unmodified code -- not a
degenerate single-dense-block wrap. sliding_window and the other patterns in
this module were empirically checked too and DO split rows across blocks for
essentially every non-trivial tile size (see STATUS.md); this adapter raises
cleanly for those rather than silently producing a wrong number.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "source")

KERNEL = "sparse-attention-kernel"
IMPL_NAME = "vit-sparse-block-tiled-attn"
PAPER_KEY = "conf/ipps/LiK26"
# run_bench.py's get_dtype() advertises fp16/bf16/fp32, but attn_block()
# (src/run_bench.py L101-136, unmodified) hardcodes its additive mask bias
# as `torch.zeros(..., dtype=torch.float32)` regardless of q/k/v's own
# dtype -- an ARTIFACT BUG found during integration: calling attn_block()
# with fp16 (or bf16) q/k/v raises
# `RuntimeError: invalid dtype for bias - should match query's dtype` from
# torch's own scaled_dot_product_attention, confirmed reproducible (see
# STATUS.md's "Real artifact bug found" section). Patching attn_block()
# itself would be touching kernel code (ARTIFACT_GUIDE rule 3), so this
# adapter services fp32 ONLY and does not work around the bug.
PRECISIONS = ["fp32"]

_NP_DTYPE = {"fp16": np.float16, "fp32": np.float32}


def _ensure_path():
    if SOURCE not in sys.path:
        sys.path.insert(0, SOURCE)


def available() -> tuple[bool, str]:
    if not os.path.isdir(SOURCE):
        return False, "source/ not cloned -- run build.sh"
    try:
        _ensure_path()
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        from src.extract_blocks import extract_blocks  # noqa: F401
        from src.run_bench import build_block_mask_from_adj, attn_block  # noqa: F401
        return True, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg:
            msg += (" -- re-run with LD_PRELOAD=/usr/lib64/libstdc++.so.6 "
                    "set BEFORE the python process starts; see STATUS.md")
        return False, msg


def _qkv_like_ml(B, H, S, d, seed, np_dtype):
    """
    Bit-identical to kernelbench.domains.ml._qkv_sparse: same default_rng
    seed, same shapes (B,H,S,d), same call order (Q, then K, then V) -- kept
    as its own, separately-typed copy (not imported from ml.py) matching the
    established pattern in bench/artifacts/attention-kernel/pat/adapter.py's
    `_qkv_like_ml` / bench/artifacts/sddmm/fused3s/adapter.py's
    `_dense_operand`.
    """
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    K = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    V = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    return Q, K, V


def _check_single_block_containment(A_csr, blocks) -> bool:
    """
    True iff every row with at least one nnz has its ENTIRE nnz column set
    contained within exactly one extracted block's column range -- the
    property `attn_block()`'s independent per-block softmax needs to be a
    correct wrapper of full masked attention (see module docstring's
    "FINDING" section). `blocks` is `extract_blocks.extract_blocks()`'s own
    (unmodified) output.
    """
    row_to_blocks: dict[int, list[int]] = {}
    for bi, b in enumerate(blocks):
        for r in range(b.r0, b.r1):
            row_to_blocks.setdefault(r, []).append(bi)
    indptr, indices = A_csr.indptr, A_csr.indices
    for r in range(A_csr.shape[0]):
        cols = indices[indptr[r]:indptr[r + 1]]
        if cols.size == 0:
            continue
        contributing = 0
        for bi in row_to_blocks.get(r, []):
            b = blocks[bi]
            if ((cols >= b.c0) & (cols < b.c1)).any():
                contributing += 1
                if contributing > 1:
                    return False
        if contributing != 1:
            return False
    return True


class VitSparseBlockTiledAttn:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"{IMPL_NAME} only services {PRECISIONS} -- attn_block()'s "
                f"hardcoded fp32 mask-bias bug rules out fp16/bf16 (see "
                f"this module's PRECISIONS comment / STATUS.md); "
                f"requested {precision!r}")
        self.precision = precision
        self.np_dtype = _NP_DTYPE[precision]

    def prepare(self, workload, params: dict):
        _ensure_path()
        import torch
        import scipy.sparse as sp
        from src.extract_blocks import extract_blocks
        from src.run_bench import build_block_mask_from_adj

        w = workload
        if w.kernel != "sparse-attention-kernel":
            raise NotImplementedError(f"{IMPL_NAME} only services sparse-attention-kernel")

        mask = w.mask  # workload's own fixed, checksummed (S,S) bool mask
        A = sp.csr_matrix(mask.astype(np.uint8))

        # Tile size: default to the workload's own block_local `block`
        # pattern_param when present (the one family this adapter can prove
        # single-block containment for -- see module docstring); otherwise
        # fall back to a caller-supplied `block_tile` param, honoring
        # ARTIFACT_GUIDE's "artifact's own format conversion goes in
        # prepare()" rather than silently guessing something that will fail
        # the containment check below anyway.
        tile = int(params.get("block_tile", w.pattern_params.get("block", 0)) or 0)
        if tile <= 0:
            raise NotImplementedError(
                f"{IMPL_NAME}: no usable tile size for pattern {w.pattern!r} "
                "(pass params['block_tile'] or use a block_local workload) "
                "-- see module docstring's single-block-containment finding")
        rho_min = float(params.get("rho_min", 0.99))

        blocks = extract_blocks(A, rho_min=rho_min, max_h=tile, max_w=tile)
        if not _check_single_block_containment(A, blocks):
            raise NotImplementedError(
                f"{IMPL_NAME}: workload {w.name!r} (pattern={w.pattern!r}, "
                f"tile={tile}) fails single-block containment -- some query "
                "row's mask support spans >1 extracted block, which "
                "attn_block()'s independent per-block softmax cannot combine "
                "correctly (no cross-block online-softmax state exists in "
                "the artifact's own code). Use a block_local(n_global_blocks="
                "0) workload with block_tile==pattern_params['block'], the "
                "one family this adapter proved exact containment for -- "
                "see STATUS.md.")

        B, H, S, d = w.B, w.H, w.S, w.d
        seed = params.get("seed", w.seed)
        Q, K, V = _qkv_like_ml(B, H, S, d, seed, self.np_dtype)
        torch_dtype = {"fp16": torch.float16, "fp32": torch.float32}[self.precision]

        # run_bench.py's own [N,H,D] layout (no batch axis) -- loop over our
        # workload's B dimension in run(), calling the SAME per-batch kernel
        # B times (a real multi-item batch would do exactly this; nothing is
        # hidden from the timed region).
        q_list, k_list, v_list = [], [], []
        for b in range(B):
            q_list.append(torch.from_numpy(np.ascontiguousarray(Q[b].transpose(1, 0, 2))).to("cuda", torch_dtype))
            k_list.append(torch.from_numpy(np.ascontiguousarray(K[b].transpose(1, 0, 2))).to("cuda", torch_dtype))
            v_list.append(torch.from_numpy(np.ascontiguousarray(V[b].transpose(1, 0, 2))).to("cuda", torch_dtype))

        # Precompute each block's boolean mask tensor once (ARTIFACT_GUIDE
        # rule 2: format conversion is preprocessing, not per-call cost) via
        # the artifact's own, unmodified build_block_mask_from_adj().
        block_masks = [build_block_mask_from_adj(A, blk, device="cuda") for blk in blocks]

        out_list = [torch.empty((S, H, d), dtype=torch_dtype, device="cuda") for _ in range(B)]

        return {
            "q_list": q_list, "k_list": k_list, "v_list": v_list,
            "blocks": blocks, "block_masks": block_masks, "out_list": out_list,
            "B": B, "H": H, "S": S, "d": d,
        }

    def run(self, h):
        from src.run_bench import attn_block
        blocks, block_masks = h["blocks"], h["block_masks"]
        for b in range(h["B"]):
            q, k, v, out = h["q_list"][b], h["k_list"][b], h["v_list"][b], h["out_list"][b]
            for blk, m in zip(blocks, block_masks):
                qb = q[blk.r0:blk.r1]
                kb = k[blk.c0:blk.c1]
                vb = v[blk.c0:blk.c1]
                out[blk.r0:blk.r1] = attn_block(qb, kb, vb, mask=m)
        return h["out_list"]

    def to_host(self, out_list):
        import torch
        # (S,H,d) per batch -> stack -> (B,S,H,d) -> permute to this track's
        # (B,H,S,d) convention.
        stacked = torch.stack(out_list, dim=0)  # (B,S,H,d)
        return stacked.detach().permute(0, 2, 1, 3).contiguous().to(
            "cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return VitSparseBlockTiledAttn(precision)
