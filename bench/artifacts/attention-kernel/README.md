# attention-kernel track — artifact adapters

Fused scaled-dot-product attention (prefill/bidirectional self-attention and
KV-cache decode). Spec: `benchspecs/attention-kernel/spec.yaml`. Domain
module: `kernelbench/domains/ml.py` (kernel `"attention-kernel"`).

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

`output/baseline_selection.md`'s attention-kernel section names 5 core,
regime-matching papers: PAT, FlashAttention-T, MetaAttention (already
integrated before this task), plus ByteTransformer and E.T. (integrated by
this task). Two papers this pass surfaced but did NOT pick (kept out of
this track's integration set, listed here for completeness):
- `conf/asplos/TangLWSLXZL26` (RedFuser) — `component`/matches: a general
  cascaded-reduction fusion compiler; attention is one of several fused
  patterns, not the paper's headline contribution.
- `conf/cgo/ZhangDSHSP26` (Hexcute) — `component`/matches: a general
  GPU-layout-synthesis compiler; attention is one of four benchmarked
  kernel classes.
- `conf/sc/WangWXHQDWL22` (LightSeq2) — `component`/partial: optimizes a
  whole Transformer TRAINING pipeline; attention is one of many fused
  kernels, evaluated end-to-end, not in isolation.
- `conf/ics/FuYDS24` (MEATTEN) — `core`/matches but CPU-only (ARM
  multicore); this track's own CPU variant (`attn-cpu-kernel-multicore`)
  exists in the spec but has no adapter integrated here.

## Artifacts

| dir | paper (PAPER_KEY) | centrality / regime | outcome |
|---|---|---|---|
| `pat` | PAT, ASPLOS'26 (`conf/asplos/YiZHYSWZZLL26`) | core / matches | **BUILT+GATED** — prefix-aware paged KV-cache decode attention (fp16/bf16), CUTLASS-based. |
| `flashattention-t` | FlashAttention-T, PPoPP'26 (`conf/ppopp/Xu0BXX00000C26`) | core / matches | **BUILT+GATED** — fully-tensorized fused attention (Ampere kernel). |
| `metaattention` | MetaAttention, PPoPP'26 (`conf/ppopp/ChenC0XMM0X00W026`) | core / matches | **BUILT+GATED** — TileLang-generated attention kernel, unified across FlashAttention-style variants. |
| `bytetransformer` | ByteTransformer, IPDPS'23 (`conf/ipps/ZhaiJWJZCLZ23`) | core / matches | **BUILT+GATED** — fused padding-free (remove-padding) BERT-style bidirectional MHA (`Attention<HALF>::fused_rm_infer`/`fused_long_rm_infer`, WMMA fp16). `bert-base-short`: `max_scaled_err=2.61e-03` (tol `1e-2`); an ad hoc short-bucket shape independently confirms the OTHER kernel dispatch path too (`err=1.49e-03`). d=64 only, S in [1,352], bidirectional only, no GQA — real artifact constraints, not integration shortcuts. |
| `et` | E.T., SC'21 (`conf/sc/ChenHPLG0D021`) | core / matches | **BUILT+GATED — correctness gate FAILS.** Wraps `OTF_attention_kernelLauncher` (dense self-attention only, per this task's instructions; the structured-pruned and shared-QK variants are not wrapped). **Confirmed artifact bug**: `kernels/attention.cu`'s `softmax_blk()` helper calls `h2exp2()` (`2^x`) where the softmax definition needs `h2exp()` (`e^x`), with no compensating scale — isolated via an exact hand-computed-logit test (kernel output for logits `[0,1,2,3]` is bit-for-bit `2^[0,1,2,3]` normalized, not `e^[0,1,2,3]`). Gate fails at `max_scaled_err` 0.47–1.7 (tol `1e-2`) on 3 independent real shapes. See `et/STATUS.md` for the full isolation. |

## Decode vs. prefill vs. CPU coverage

- Decode (`attn-decode-kv-cache-kernel`): `pat` (its own headline claim).
- Prefill (`attn-prefill-kernel-fp16bf16`): `flashattention-t`,
  `metaattention`, `bytetransformer` (bidirectional only, S<=352, d=64),
  `et` (both causal and bidirectional, but its gate fails — see above).
- CPU (`attn-cpu-kernel-multicore`): no adapter integrated (MEATTEN was
  surveyed but not selected for integration this pass — see "Baseline
  selection rule" above).

## Gate summary (login node, reduced protocol — not spec-conforming)

```
$PY -m kernelbench.runner --kernel attention-kernel --list
implementations: cpu=['numpy-attention', 'numpy-flash-attention']
                 cuda=['torch-sdpa (flash/mem-efficient backend)']
paper artifacts:
  ok bytetransformer  bytetransformer-fused-rm-attn
  ok et               et-otf-attn
  ok flashattention-t flashattention-t-fwd
  ok metaattention    metaattention-softmax
  ok pat              pat-decode-attn
```

Real functional/gate checks performed this pass (all reduced-protocol,
non-conforming per ARTIFACT_GUIDE rule 5 — login node, no timing sweeps):

- `bytetransformer-fused-rm-attn`: `attn-prefill-kernel-fp16bf16`,
  `bert-base-short` 1/1 valid (`err=2.61e-03`); ad hoc short-bucket shape
  (S=64) also valid (`err=1.49e-03`).
- `et-otf-attn`: `attn-prefill-kernel-fp16bf16`,
  `bert-base-short`/`bert-large`/`gpt2-small` 0/3 valid — confirmed
  artifact bug, see `et/STATUS.md`.
- `pat`/`flashattention-t`/`metaattention`: previously gated (see their own
  STATUS.md), unchanged by this task.
