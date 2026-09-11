# Sparse-attention-kernel track — evaluation methodology survey

Track input: `data/track_inputs/sparse-attention-kernel.json`, 4 papers (all
surveyed — fewer than the 5-paper minimum's "or all, if fewer" applies).
2 surveyed via arXiv fulltext + artifact-repo benchmark scripts read with
`gh api`; the other 2 have no arXiv id in the track input (searched and not
found — ipdps-26_vit's own README says "BibTeX to be added after
publication"; Fused3S is ACM-only), so their methodology comes entirely from
the artifact repo's benchmark/test scripts.

Operation surveyed throughout: attention restricted to a 0/1 sparsity mask,
`O = softmax( (Q K^T / sqrt(d)) ⊙_mask + (-inf outside mask) ) V`, where the
mask's *origin* differs sharply across the 4 papers (see Divergences) — from
a swept synthetic sparsity ratio to a real dataset's own graph adjacency.

---

## 1. LiK26 — "Achieving Low Latency Inference on High Resolution Images by Exploiting Sparsity in Vision Transformers" (IPDPS'26)

- key: `conf/ipps/LiK26`, no arXiv id (repo README: "please cite our paper
  (BibTeX to be added after publication)"; a targeted web search for the
  exact title + venue also returned no preprint).
- **workloads/inputs**: sparse-ViT attention masks from 3 named models —
  DynamicViT (data-dependent dynamic token pruning), RegionViT (fixed
  regional block-sparse), VisionLongformer (sliding-window + global tokens).
  Masks are represented as binary adjacency matrices, reordered via one of 3
  structure-aware clustering methods (`hc`/hierarchical-clustering,
  `ic`/iterative-clustering, or SciPy's `rcm`/Reverse-Cuthill-McKee baseline,
  or `none`) to expose nearly-dense computation blocks, then an ILP (Gurobi)
  solves a hardware-aware tile-size/schedule assignment per block
  (`src/solve_ilp_gurobi.py`) before the schedule is executed and timed
  (`src/run_bench.py`).
- **timing protocol**: `run_bench.py`'s `time_cuda()` — CUDA-event pair
  (`torch.cuda.Event(enable_timing=True)`) per iteration, explicit `warmup`
  and `iters` CLI args (README example: `--iters 100 --warmup 10`), computes
  BOTH `mean_ms` and `p50_ms` (median) over the timed iterations — one of the
  two most rigorous timing setups in this track.
- **timing scope**: `run_bench.py` times only the block-granularity attention
  execution (stage T5); adjacency construction (T1), reordering+block
  extraction (T2), tile profiling (T3), and ILP scheduling (T4) are separate,
  un-timed-together pipeline stages — i.e. preprocessing is naturally
  reported separately by the pipeline's own structure, though the README
  does not state whether T1-T4's wall time is captured/reported anywhere.
- **precision & correctness**: `--dtype` CLI arg supports fp16/bf16/fp32.
  **No correctness check is called from `run_bench.py`** — the only
  accuracy-relevant switch is `--assume_dense_blocks` (treat an extracted
  block as fully dense for speed vs. apply the true adjacency submask "for
  correctness"), which is a speed/accuracy *trade-off* the user selects, not
  a verification gate. No reference-output comparison exists in the timed
  path.
- **metric**: per-block `mean_ms`/`p50_ms` at (BQ, BK) tile granularity,
  written to CSV; no FLOP or throughput metric is computed anywhere in
  `run_bench.py` (sparsity's benefit is implicit in which/how-many blocks the
  ILP schedule executes, not measured as an explicit effective-vs-dense
  ratio). The paper's headline claim (one-liner: "up to 2.1x over
  FlashAttention fixed tiling") is presumably an end-to-end aggregate over
  all scheduled blocks for a full model forward pass, but that aggregation
  step is not visible in this script.
- **baselines**: FlashAttention (explicit install dependency and the
  reproducibility-notes' "FlashAttention kernels may introduce minor
  non-determinism" caveat); implicit fixed-tile-size baseline is the "no
  reordering" (`none`) / naive block partition.
- source: repo `KLab-AI3/ipdps-26_vit` — `README.md`, `src/run_bench.py`
  (full file read).

## 2. DaiDRYLL0CS26 — STOF, "Accelerating Sparse Transformer Inference on GPU" (PPoPP'26)

- key: `conf/ppopp/DaiDRYLL0CS26`, arXiv:2506.06095 (earlier arXiv version
  titled "Flexible Operator Fusion for Fast Sparse Transformer with Diverse
  Masking on GPU" — same paper, retitled for camera-ready).
- **workloads/inputs**: 5 named mask types selectable via `--mask_id`:
  0=Causal, 1=Sliding(window), 2=Longformer (local+global), 3=BigBird
  (local+global+random, `fill_rate=0.1` for the random-block component),
  4=Dilated. Band width and global width fixed at `sqrt(seq_len)` (32 for
  seq_len=1024). MHA sweep: `seq_len` 128→4096 (stride 2x), batch 1→16,
  `head_num=12, head_size=64` (BERT-base-shaped) by default; an extended
  sweep pushes seq_len to 4k-16k at batch=1.
- **timing protocol**: `src/benchmk_attn_unified.py` — **`warmup_iters = 10`,
  `running_iters = 20`** (both hard-coded constants, confirmed directly in
  source), `std::` — i.e. Python `time.time()`-style host timers around
  `warmup_iters + running_iters` total loop passes, reduced to
  `total_time * 1000 / running_iters` (a **mean**, not median). This
  **contradicts** the paper's own arXiv text, which states "Warm-ups for all
  experiments and run 100 times to record the average performance" — the
  actually-released MHA-comparison script (used for the paper's Figure 9-10)
  runs only 20 timed iterations, not 100 (a paper-text-vs-code discrepancy;
  a different script, e.g. `tuning_STOF_cost.py`, may be where "100" applies
  but was not independently checked).
- **timing scope**: kernel-only per-config latency; separately, `Table 4`
  reports **tuning cost in seconds** (STOF's operator-fusion-template search
  is itself autotuning, so its search time is a first-class reported number,
  not silently folded into per-iteration latency).
- **precision & correctness**: FP16 throughout (`data_type = torch.float16`).
  `check_tensor()` in `src/correct_verify_attn.py` computes `max_diff` and
  `mean_diff` (elementwise abs difference vs. a `torch_attn_std()` reference
  — manual `baddbmm`+causal-mask+softmax+dropout(0)+einsum FP16 computation,
  not FP32/FP64) and **prints** them; no assert/threshold gate exists in the
  file, and `correct_verify_attn.py` is a separate script from
  `benchmk_attn_unified.py` — correctness and performance are never
  verified in the same run, matching the same disconnected-check pattern
  this project's attention-kernel spec already flagged for MEATTEN.
- **metric**: speedup ratio normalized to PyTorch-Native latency; no
  GFLOP/s or TFLOP/s is computed anywhere in the released benchmark driver —
  the paper's "skipping useless computations" claim is never converted into
  an effective-vs-dense FLOP number.
- **baselines**: 8 total — PyTorch Native, PyTorch Compile, FlashAttention2,
  FlexAttention, ByteTransformer, Bolt, MCFuser, SPLAT.
- **quality gate**: none — no perplexity or downstream-task accuracy is
  evaluated at any tested sparsity/mask configuration.
- source: arXiv fulltext (`arxiv.org/html/2506.06095v3`) + repo
  `HeyDavid633/PPoPP26-pap161-AE` — `README.md`, `script/fig9-10.sh`,
  `src/benchmk_attn_unified.py`, `src/correct_verify_attn.py`.

## 3. LiC25 — Fused3S, "Fast Sparse Attention on Tensor Cores" (ICS'25)

- key: `conf/ics/LiC25`, no arXiv id (ACM-OA PDF only); DOI
  10.1145/3721145.3730430.
- **workloads/inputs**: this is a **graph-attention** kernel, not a
  language/vision-sequence kernel — the "sparsity mask" is a real graph's own
  adjacency matrix (irregular, dataset-defined, not a tunable ratio). 15
  named single-graph datasets tested (`citeseer, cora, pubmed, Ell, github,
  Artist, com-amazon.ungraph, Blog, amazon0505, igb_small, yelp, reddit,
  igb_medium, ogbn-products, amazonProducts`), 2 large-scale
  (`igb_large, ogbn-papers100M`), and 9 batched-graph datasets for the
  DF-GNN comparison (`ZINC, PascalVOC-SP, COCO-SP, Peptides-func/struct,
  ogbg-molhiv/ppa/molpcba/code2`). Q,K,V are `torch.rand` fp32, cast to fp16;
  `--embedding_dim` default 128.
- **timing protocol**: `event_timing_decorator()` in
  `scripts/baseline_comp/baseline_comp_kernel_only.py` — CUDA events,
  **warmup = 3 iterations, timed = `niter = 10` iterations** (both confirmed
  directly in source), reports **median, mean, AND std** of both time and
  throughput (`np.median`/`np.mean`/`np.std`) — the most statistically
  complete reporting of any paper in this track.
- **timing scope**: kernel-only (`f3s_1tb1rw_scheduled_permuteV` etc. return
  a device-measured `time` directly from the CUDA kernel); CSR→tensor-core-
  block preprocessing (`F3S.preprocess_gpu`) is a separate, un-timed-in-the-
  same-loop call.
- **precision & correctness**: fp16 fused kernel vs. fp32 PyTorch dense
  masked-softmax reference in `scripts/tests/test_f3s_accuracy.py`. Metric is
  **relative Frobenius-norm error** `||fused - true_fp32|| / ||true_fp32||`
  (not elementwise max-abs), computed separately for the SDDMM-only stage
  (`--check_sddmm`) and the final fused output; values are printed
  (`n_test=1, n_runs=1` by default), **not gated by any assert/threshold** —
  same disconnected-correctness pattern as STOF and (this project's earlier
  survey of) MEATTEN.
- **metric**: **explicit effective (nnz-based) FLOP formulas** are computed
  per algorithm — `f3s_flop`: SDDMM = `nnz*d*2`, softmax = `nnz*5`, SpMM =
  `nnz*d*2` (same formula pattern for `dfgnn_flop`/`flashSparse_flop`) — then
  `throughput = flop*1000/time_ms` → FLOPS. This is the **only paper in the
  track with an explicit, code-verified FLOP formula**, and it is
  unambiguously the *effective* (sparse/nnz-based) convention, not a
  dense-equivalent one; no dense-equivalent FLOP number is computed anywhere
  for comparison.
- **baselines**: DF-GNN and FlashSparse (both vendored as git submodules
  under `baselines/`), plus native PyTorch-Geometric `pyg_gtconv`.
- **hardware**: auto-detected among A100/H100/A30/V100/T4
  (`get_gpu_model()`); README's stated primary targets are A30/H100/GH200
  ("kernels optimized for Ampere... ongoing work... Hopper").
- **quality gate**: none — only numerical closeness to the dense reference is
  checked; no downstream GNN task accuracy (e.g. node-classification F1) is
  measured at the kernel level.
- source: repo `HPCForge/Fused3S` — `README.md`,
  `scripts/baseline_comp/baseline_comp_kernel_only.py` (full file read),
  `scripts/tests/test_f3s_accuracy.py` (full file read),
  `scripts/downloadDataset.py`.

## 4. TomczakK25 — "Longer Attention Span: Increasing Transformer Context Length With Sparse Graph Processing Techniques" (IPDPS'25)

- key: `conf/ipps/TomczakK25`, arXiv:2502.01659.
- **workloads/inputs**: 6 CUDA/PyTorch-extension kernel variants shipped as
  separate installable packages — explicit masks in **COO** and **CSR**
  format (arbitrary user-supplied sparsity, closest to a "dynamic/general"
  pattern), and implicit (parameter-defined, no materialized mask) **Local**
  (1D sliding-window), **Local-1D-Dilated**, **Local-2D-Dilated**
  (block-diagonal with dilation), and **Global (No Local)** (fixed global
  tokens + local window) kernels. The paper additionally reconstructs
  Longformer (local+global, dilated local+global) and BigBird
  (local+global+random) as compositions of these primitives. **Sparsity
  factor Sf explicitly swept from 0.0001 to 1.0** — the only paper in this
  track with a genuine, named sparsity-ratio sweep axis. Context length L =
  8192/16384/24576 for controlled microbenchmarks; extended test reaches
  **160 million tokens on a single A100**; a theoretical 1-billion-token
  projection is discussed but not measured.
- **hardware**: 3 GPU generations — A100 SXM4 80GB, L40 48GB, V100 SXM2
  32GB — each job Slurm-allocated 1 GPU + 8 CPU cores + 32GB host memory.
- **timing protocol**: **10 warmup iterations + 15 timed iterations**,
  **average** reported (not median) — a weaker statistic than Fused3S's or
  ipdps-26_vit's, and fewer timed samples than STOF's 20 or this project's
  general >=20-sample convention.
- **precision & correctness**: `torch.allclose` vs. PyTorch SDPA / dense
  FlashAttention reference, **`atol=1e-8, rtol=1e-5`, NaN-values-equal** —
  the tightest, most rigorous tolerance in this track (all 4 papers combined),
  though the precision the comparison runs at (fp16 vs fp32) was not
  independently confirmed from the fetched excerpt.
- **metric**: runtime in seconds (log-scale), speedup vs. FlashAttention/SDP,
  and **achievable maximum context length under a fixed memory budget** — a
  capability metric distinct from raw throughput, unique to this paper.
  **Explicit complexity/FLOP-accounting statement**: dense SOTA baselines are
  characterized as `O(L²d + SfL²d)` (full dense pass PLUS the sparse-mask
  work — a "compute dense then mask" pattern), while the proposed kernels are
  claimed **work-optimal at `O(SfL²d)`** — i.e. **effective/sparse-only**
  work. This is the clearest, most explicit effective-vs-dense-equivalent
  FLOP framing of any paper surveyed across both tracks in this task.
- **baselines**: PyTorch `scaled_dot_product_attention`, dense FlashAttention.
- **quality gate**: none — no perplexity/downstream accuracy at any Sf.
- source: arXiv fulltext (`arxiv.org/html/2502.01659v2`) + repo
  `KLab-AI3/Graph-Processing-Attention-IPDPS-2025` — `README.md` (full file
  read; `verification/verify.py` exists but was not read in depth).

---

## Divergences

- **FLOP accounting is the single biggest gap this track shares, and the
  task's own key axis is confirmed by the survey.** Only TomczakK25 states an
  explicit effective-vs-dense-equivalent formula (`O(SfL²d)` vs.
  `O(L²d+SfL²d)`); Fused3S computes an explicit but *effective-only*
  (nnz-based) FLOPS number with no dense-equivalent given for comparison;
  STOF and ipdps-26_vit report **no FLOP/throughput metric at all** — only
  wall-clock latency or speedup ratio. A reader cannot compare a "2.1x"
  (ipdps-26_vit), "1.6x MHA" (STOF), or "1.6-16.3x" (Fused3S) claim against
  each other or against a roofline without a common, explicitly-labeled FLOP
  convention — this spec fixes one (see Step 2).
- **Sparsity-pattern taxonomy is not uniform.** STOF and TomczakK25 test
  *named, parametric* patterns (causal/sliding/Longformer/BigBird/dilated)
  controlled by a swept ratio or bandwidth; ipdps-26_vit tests *model-defined*
  patterns (DynamicViT's data-dependent dynamic top-k-style token pruning,
  RegionViT's fixed regional blocks, VisionLongformer's sliding+global) that
  are not swept by the benchmark itself but inherited from the source model;
  Fused3S tests *real dataset-defined* patterns (a graph's own adjacency,
  irregular, no ratio knob at all). A single sparsity-ratio-sweep variant
  cannot cover all three without either (a) forcing Fused3S's fixed-density
  graphs into an artificial "ratio" framing, or (b) pretending ipdps-26_vit's
  model masks are freely tunable. This spec keeps a parametric-sweep variant
  (STOF/TomczakK25/ipdps-26_vit-as-candidate) separate from a fixed
  real-graph variant (Fused3S).
- **No paper evaluates a quality gate.** The task's own key axis (perplexity/
  accuracy must not degrade) is not implemented by ANY of the 4 papers — all
  four only check numerical closeness of the *attention output tensor* to a
  dense reference (STOF/Fused3S: unenforced print-only diffs; TomczakK25:
  `allclose` at a genuinely tight tolerance; ipdps-26_vit: no check at all in
  the timed path), never that the *sparsity pattern itself* preserves
  downstream model quality (ViT top-1 for ipdps-26_vit's 3 models, LM
  perplexity for STOF/TomczakK25's causal-LM-shaped configs, node-
  classification accuracy for Fused3S's GNN datasets). This spec adds a
  quality-gate requirement that no surveyed artifact currently satisfies —
  an explicit fix, not a reproduction of the field's current (missing)
  practice, per this project's engineering-not-diplomacy principle.
- **Correctness tolerance and enforcement vary by ~6 orders of magnitude and
  from "hard gate" to "never called."** TomczakK25: `atol=1e-8, rtol=1e-5`
  (tightest, and actually enforced via `allclose` inside its own tests, per
  the README's Reproducibility Notes referencing benchmark correctness).
  STOF: `max_diff`/`mean_diff` printed, no threshold, no assert, check
  function lives in a script separate from the perf benchmark. Fused3S:
  relative Frobenius-norm error printed, no threshold, no assert, separate
  script from the perf benchmark. ipdps-26_vit: **no correctness check
  anywhere in the timed path** (`--assume_dense_blocks` is a user-selected
  speed/accuracy trade-off, not a verification gate). This spec fixes ONE
  tolerance policy and makes the gate mandatory in the SAME run as the timed
  measurement for every candidate, matching this project's established
  pattern (see `benchspecs/attention-kernel/spec.yaml`'s treatment of the
  same issue for PAT/MEATTEN/ByteTransformer/E.T.).
- **Timing rigor**: Fused3S (median+mean+std, CUDA events, warmup=3/reps=10)
  and ipdps-26_vit (mean+median/p50, CUDA events, CLI-configurable
  warmup/iters, README default 10/100) are the most careful. STOF's released
  script (warmup=10, running=20, mean-only) directly **contradicts its own
  paper text** ("run 100 times") — a concrete instance of the
  `benchspec_instructions.md` warning that "timing loops in code reveal
  warmup/repetition/timer placement more reliably than paper text."
  TomczakK25 (warmup=10, reps=15, average-only) is the loosest of the 4.
- **Fused3S's "sequence length" is a graph's node count, not a text/image
  sequence length**, and its "sparsity ratio" is the graph's own edge density
  (typically <1% for the citation/social-network graphs, higher for small
  molecular graphs) — not a value any candidate kernel can choose. This is
  why Fused3S is kept in a separate variant rather than folded into the
  Sf-swept structured-mask variant.
