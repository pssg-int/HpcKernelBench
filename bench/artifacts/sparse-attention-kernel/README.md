# sparse-attention-kernel track — artifact adapters

Attention restricted to a 0/1 sparsity mask: a structured/swept-ratio
variant (sliding-window/local, Longformer, BigBird, dilated,
block-local/regional — `sparse-attn-structured-mask-kernel-fp16`) and a
real-graph-adjacency variant (Fused3S's own claim: fused SDDMM+softmax+SpMM
over a real graph's sparse adjacency, sparsity a fixed dataset property —
`sparse-attn-graph-adjacency-kernel-fp16`). Spec:
`benchspecs/sparse-attention-kernel/spec.yaml`. Domain module:
`kernelbench/domains/ml.py` (kernel `"sparse-attention-kernel"`) implements
the structured-mask variant's workloads only — see that module's own
docstring for why the graph-adjacency variant is not a
`SparseAttentionWorkload` at all.

## Baseline selection rule (user decision, 2026-09-05)

Baselines are chosen from `kernel-papers/output/baseline_selection.md`
(produced by `select_baselines.py` from the per-(paper, track) ratings in
`output/kernel_centrality.json`), not "newest N artifacts":

1. **kernel centrality** `core` (the kernel IS the paper's headline
   contribution, evaluated at kernel level) > `component` (part of a
   bigger system/pipeline/codegen output) — `tangential` is never a
   baseline;
2. **regime match** with the spec's inputs: `matches` > `partial` >
   `mismatch`;
3. **single-NVIDIA-GPU path** required (current scope);
4. **recency** only as the tiebreak; up to 5 per track.

All 4 papers this track's own `output/kernel_centrality.json` rates are
`core`/`matches`: gpa, sparse-transformer (STOF), vit-sparse (already
integrated before this task) and Fused3S (integrated by this task, reusing
its existing `sddmm/fused3s` build for a different kernel-call boundary —
see below).

## Artifacts

| dir | paper (PAPER_KEY) | centrality / regime | outcome |
|---|---|---|---|
| `gpa` | TomczakK25, IPDPS'25 (`conf/ipps/TomczakK25`) | core / matches | **BUILT+GATED** — CSR explicit-mask sparse FlashAttention (`spfa_csr`), the most general of the repo's 6 kernel variants. |
| `sparse-transformer` | STOF, PPoPP'26 (`conf/ppopp/DaiDRYLL0CS26`) | core / matches | **BUILT+GATED (with a confirmed artifact bug)** — `binding_attn` (CUTLASS/FlashAttention-family fused kernel). |
| `vit-sparse` | ipdps-26_vit, IPDPS'26 (`conf/ipps/LiK26`) | core / matches | **BUILT+GATED** — ILP-scheduled tile-aware sparse ViT attention, pure-Python composition around `torch.nn.functional.scaled_dot_product_attention`. |
| `fused3s` | Fused3S, ICS'25 (`conf/ics/LiC25`) | core / matches | **BUILT+GATED** — the FULL fused SDDMM+softmax+SpMM pipeline (`F3S.f3s_1tb1tcb(..., applySoftmax=True)`), wrapping the `coo_arbitrary` mask family as the closest available proxy for a real graph adjacency (see below). Reuses `../sddmm/fused3s`'s already-compiled extension (`source` is a symlink, no rebuild). Smoke (`S=32,d=16`): `max_scaled_err=3.35e-04`; a larger real-registry shape (`S=8192,d=64,density=0.001`): `max_scaled_err=4.15e-04` (tolerance `1e-2` both). |

## `fused3s`: two adapters, one compiled kernel, two boundaries

Fused3S is ALSO integrated under `../sddmm/fused3s/` (the isolated SDDMM
stage of the same fused kernel, `applySoftmax=False`). This directory adds
the FULL pipeline boundary (`applySoftmax=True`) under this track, reusing
the exact same compiled `F3S.cpython-*.so` via a symlinked `source/` — no
second build, per this task's instructions. See `fused3s/STATUS.md`'s
comparison table for exactly which flags differ between the two adapters.

`kernelbench.domains.ml.SparseAttentionWorkload` has no graph-adjacency
workload type (see that module's own docstring) — `fused3s/adapter.py`
wraps `coo_arbitrary` (i.i.d. Bernoulli-sparsity, "the one pattern in this
track with no structural constraint at all") as the closest available
proxy for Fused3S's own real-graph-adjacency regime, the same "closest
synthetic proxy" logic `vit-sparse`'s own gate uses `block_local` for. It
refuses (`NotImplementedError`) every other named structured-mask pattern,
which remain gpa/sparse-transformer/vit-sparse's territory.

## Gate summary (login node, reduced protocol — not spec-conforming)

```
$PY -m kernelbench.runner --kernel sparse-attention-kernel --list
implementations: cpu=['numpy-masked-attention']
                 cuda=['torch-sdpa-masked (dense fallback, torch bool attn_mask)']
paper artifacts:
  ok fused3s          fused3s-1tb1tcb-attn
  ok gpa              gpa-spfa-csr
  ok sparse-transformer stof-binding-attn
  ok vit-sparse       vit-sparse-block-tiled-attn
```

Real functional/gate checks performed this pass (all reduced-protocol,
non-conforming per ARTIFACT_GUIDE rule 5 — login node, no timing sweeps):

- `fused3s-1tb1tcb-attn`: `sparse-attn-structured-mask-kernel-fp16`,
  smoke `smoke-sparse-attn-coo` 1/1 valid (`err=3.35e-04`); other 6 smoke
  workloads correctly `UNSUPPORTED` (non-`coo_arbitrary`); real-registry
  `sf-sweep-microbench-sf001` (S=8192) 1/1 valid (`err=4.15e-04`).
- `gpa`/`sparse-transformer`/`vit-sparse`: previously gated (see their own
  STATUS.md), unchanged by this task.
