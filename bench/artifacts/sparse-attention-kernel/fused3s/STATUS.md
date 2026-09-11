# fused3s (Fused3S) — sparse-attention-kernel — STATUS: BUILT+GATED

- Paper: "Fused3S: Fast Sparse Attention on Tensor Cores", ICS'25.
  `PAPER_KEY = conf/ics/LiC25` (same paper as `bench/artifacts/sddmm/fused3s/`).
- Selection rationale: core baseline under the revised kernel-centrality
  rule — centrality `core`, regime `matches` ("real graph adjacency (single
  + batched graphs), tensor cores, GPU" — Fused3S's own claim is explicitly
  the sole basis for the spec's `sparse-attn-graph-adjacency-kernel-fp16`
  variant), single-NVIDIA-GPU path. Already integrated under `sddmm/fused3s`
  for the SDDMM-stage-only boundary; this directory adds the FULL fused
  SDDMM+softmax+SpMM boundary under the sparse-attention-kernel track, per
  this task's instructions.

## Not a second build — reuses `sddmm/fused3s`'s compiled extension

`source` in this directory is a **symlink** to
`../../sddmm/fused3s/source` (`ln -s`) — the exact same cloned repo
(`HPCForge/Fused3S`, commit `65d43a9cf212d6f18d8931de8e1c66a728756d7d`) and
the exact same compiled `F3S.cpython-311-x86_64-linux-gnu.so` under
`source/src/`. `build.sh` here performs NO compilation — it only verifies
the symlink and the already-built extension are present and importable
(errors out pointing at `../../sddmm/fused3s/build.sh` if not). Provenance,
toolchain, and build-system patches (none needed — `F3S`'s build is
unmodified from the sddmm integration) are recorded in
`../../sddmm/fused3s/STATUS.md`; `source.provenance` here just points at
that file rather than duplicating it.

## What boundary this adapter wraps (vs. the sddmm/fused3s adapter)

Both adapters call the SAME compiled kernel, `F3S.f3s_1tb1tcb`, with
different flags:

| | `sddmm/fused3s` | `sparse-attention-kernel/fused3s` (this dir) |
|---|---|---|
| `applySoftmax` | `False` | `True` |
| `saveSddmmResult` | `True` | `False` |
| returns | raw `sddmmResult` (masked dot products) | `output` — the full attention result `O = softmax_row(A ⊙ QK^T) @ V` |
| track | `sddmm` | `sparse-attention-kernel` |

This is the SAME single fused CUDA kernel launch either way (there is no
separable sub-kernel boundary below the CUDA-call level, per the sddmm
adapter's own STATUS.md) — only which of its outputs are read and which
optional stages actually execute differ, both via flags the kernel itself
exposes.

## Why `coo_arbitrary`, not the other 6 structured-mask patterns

`kernelbench.domains.ml`'s module docstring is explicit: the spec's
graph-adjacency variant (Fused3S's own claim: fused SDDMM+softmax+SpMM
over a REAL graph's adjacency, sparsity a FIXED dataset property, not a
swept ratio) is NOT implemented as a `SparseAttentionWorkload` — that
dataclass owns the structured/swept-ratio variant only (causal,
sliding_window, longformer, bigbird, dilated, block_local — already
covered by gpa/sparse-transformer/vit-sparse). `coo_arbitrary`
(`_coo_arbitrary_mask`: i.i.d. `Bernoulli(density)`, ml.py's own comment:
"the one pattern in this track with no structural constraint at all") is
the closest available proxy for an arbitrary graph adjacency within this
existing workload machinery — the same "closest synthetic proxy" logic
`vit-sparse`'s own gate uses `block_local` for (per `_SPARSE_ATTN_SHAPES`'s
comment on `block-local-s1024-blk64`). `adapter.py::prepare()` raises
`NotImplementedError` for all 6 other named patterns, naming this
rationale explicitly.

## Preprocessing done ONCE in `prepare()` (ARTIFACT_GUIDE rule 2)

1. **Mask -> CSR -> Fused3S's own TC-block layout.** The workload's shared
   `(S,S)` bool mask is converted to CSR (`np.nonzero` + stable sort by
   row, matching scipy's own CSR construction) then run through
   `F3S.preprocess_gpu` — the artifact's own CSR->tensor-core-block
   conversion, IDENTICAL to what the sddmm/fused3s adapter does with a
   real SuiteSparse matrix's CSR. Since the mask (and therefore this
   layout) is the SAME for every batch/head in this track's convention,
   it is computed ONCE and reused across all `B*H` kernel launches in
   `run()`.
2. **Scale folded into Q.** Read `f3sKernel1tb1tcb`'s SDDMM section in
   full (`F3S_kernel.cu`): the raw `HMMA16816` dot product feeds directly
   into the online-softmax machinery with **no `1/sqrt(d)` scaling
   anywhere** in the kernel body or its host wrapper `f3sCuda1tb1tcb`
   (confirmed: neither takes a `scalingFactor` argument, unlike a sibling
   function `f3sCuda1tb1rwScheduled` in the same file, which does). To
   match this track's `softmax(QK^T/sqrt(d))@V` convention, `Q' =
   Q/sqrt(d)` is computed in fp32 THEN cast to fp16 — the standard
   "pre-scaled query" technique (mathematically identical to scaling the
   dot product afterward). The gpa/spfa_csr adapter needed no such
   folding because ITS kernel already scales internally — see that
   adapter's own docstring for the contrast.

## Masking mechanism (kernel's own, not this adapter's)

Non-edge (query,key) pairs are excluded from the softmax by the kernel's
own TC-block bitmap (`addPartialSums` zeroes non-edge slots; the softmax
loop then special-cases an exact `0.0` raw score to skip exponentiation
entirely, contributing `0` to both numerator and denominator — the same
net effect as an additive `-inf` mask). `SparseAttentionWorkload`'s mask
always keeps the diagonal (`_build_sparse_mask` forces this), so no row's
softmax denominator is ever identically zero.

## Batching

`F3S.f3s_1tb1tcb`'s `Q`/`K`/`V` are 2D `(numNodes, embeddingDim)` — no
batch/head dimension, matching a single graph. `run()` loops over the
workload's `B*H` independent attention problems, calling the kernel once
per (batch,head) slice with the SAME preprocessed TC-block layout —
identical precedent to the gpa/spfa_csr adapter's own `B*H` loop.

## Verification (login node, correctness gate only, reduced protocol)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
$PY -m kernelbench.runner --kernel sparse-attention-kernel \
    --variant sparse-attn-structured-mask-kernel-fp16 --smoke \
    --impl fused3s-1tb1tcb-attn --precision fp16
```

Result: **1/1 valid** (the `smoke-sparse-attn-coo` workload;
B=2,H=2,d=16,S=32,density=0.15), `max_scaled_err = 3.35e-04` (tolerance
`1e-2`) — the other 6 smoke workloads correctly report `UNSUPPORTED`
(non-`coo_arbitrary` patterns, per this adapter's own scope).

A second, larger real-registry check confirms the finding is not a
small-shape coincidence:

```
$PY -m kernelbench.runner --kernel sparse-attention-kernel \
    --variant sparse-attn-structured-mask-kernel-fp16 \
    --matrices sf-sweep-microbench-sf001 \
    --impl fused3s-1tb1tcb-attn --precision fp16 --warmup 2 --reps 3
```

Result: **1/1 valid** (B=1,H=1,d=64,S=8192, density target 0.001, 9567
TC-blocks), `max_scaled_err = 4.15e-04` (tolerance `1e-2`) — a clean pass
with margin consistent with genuine fp16 quantization/MMA rounding noise,
at a scale two orders of magnitude larger than the smoke shape, and with a
different (S,d) combination than the smoke test, giving independent
confidence in the CSR-preprocessing/scale-folding/batching-loop
correctness rather than a shape-specific coincidence.

Not spec-conforming (login node, smoke/reduced protocol, shared GPU) —
timing numbers from these runs must not be published, only the correctness
result.

## Not done

- The other 6 named structured-mask patterns (causal, sliding_window,
  longformer_local_global, bigbird_local_global_random, dilated_1d,
  block_local): out of scope by design (see "Why coo_arbitrary" above),
  not a coverage gap.
- Rectangular (non-square) adjacency: not applicable here —
  `SparseAttentionWorkload`'s mask is always `(S,S)` square by
  construction, unlike the sddmm track's general `matrix.csr` input.
- Timing sweeps and compute-node runs: none — login node is build (well,
  verification) + correctness gate only.

## Provenance

- Artifact commit: `65d43a9cf212d6f18d8931de8e1c66a728756d7d` (same clone as
  `sddmm/fused3s`, symlinked, not re-cloned)
- nvcc/host-compiler/GPU-arch: identical to `sddmm/fused3s/STATUS.md`
  (same compiled `.so`)
- Python/torch: `/pscratch/sd/c/cunyang/gnn/plexus_env` — torch 2.8.0+cu128

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05; job
  ran on gpu-b11-6.zaratan.umd.edu, physical card NVIDIA A100-SXM4-40GB),
  login-node "build" (verification only, no compilation — see below).
- Toolchain: identical to `../../sddmm/fused3s/STATUS.md`'s zaratan
  reproduction entry — nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0,
  torch 2.8.0+cu128, Python 3.12.14; same compiled `F3S.cpython-312-*.so`
  reused via the `source` symlink.
- Build: OK — `build.sh` confirmed the symlink and the shared
  `F3S.cpython-312-x86_64-linux-gnu.so` (built under `sddmm/fused3s`) are
  present, then imported `F3S` successfully (`import OK`). No compilation
  performed, per design. Build-system changes: none.
- Gate:
  - `sparse-attn-structured-mask-kernel-fp16` `--smoke` (fp16): PASS on the
    `coo` smoke workload, `max_scaled_err = 3.35e-04 <= tol 1e-2`, 1/1 runs
    valid (6 unsupported — the other 6 named structured-mask patterns,
    correctly rejected by `prepare()`'s `coo_arbitrary`-only scope).
  - `sparse-attn-structured-mask-kernel-fp16` on `sf-sweep-microbench-sf001`
    (fp16, `--warmup 2 --reps 3`): PASS, `max_scaled_err = 4.15e-04 <= tol
    1e-2`, 1/1 runs valid.
- Deviation from the recorded ruling: none — both errors match the recorded
  Perlmutter values to 3 significant figures (3.35e-04 and 4.15e-04) and the
  same 6 patterns report UNSUPPORTED.
- Verdict here: BUILT+GATED — same as the recorded ruling.
