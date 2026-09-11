# Survey: dnn-operator-fusion

Track input: `data/track_inputs/dnn-operator-fusion.json` (5 papers, all 5
surveyed, at varying depth). Three papers (RedFuser, STOF, LightSeq2) yielded
rich fulltext via arXiv HTML/ar5iv rendering plus (for STOF) direct
inspection of the artifact's benchmark code. Two papers (SumMerge, the CGO'25
RISC-V-backend paper) could only be surveyed at abstract depth — see notes
under each and in Divergences.

---

## 1. RedFuser (ASPLOS 2026) — `alibaba/redfuser`, arXiv 2603.10026

- **fusion patterns benchmarked** ("cascaded reductions": loops with
  inter-loop data dependency feeding a reduction, e.g. softmax→GEMM), four
  representative subgraphs:
  1. **Multi-Head Attention (MHA)**: softmax + GEMM, shapes from BERT, ViT,
     LLaMA-65B.
  2. **Multi-Latent Attention (MLA)**: DeepSeek-R1 shapes.
  3. **MoE routing**: "GEMM for calculating scores for each expert, followed
     by a softmax + topk to select experts", configs from Switch
     Transformer, ERNIE, DeepSeek-V2-Lite, Qwen3.
  4. **FP8 per-token-quantization + GEMM**: configs from ERNIE, DeepSeek-R1,
     Qwen3.
- **timing granularity**: **subgraph-level only** — "We evaluate RedFuser on
  four representative subgraphs", normalized-latency results; **no full
  end-to-end (whole-model) numbers are reported at all** — an explicit scope
  limitation of the paper.
- **compilation/search time**: the paper describes an "Auto-Tuning
  mechanism that optimizes key parameters, including block tile size,
  threads per block, software pipeline depth" and mentions "runtime
  configuration selection", but reports **no quantitative search-time
  numbers anywhere** in the evaluation section obtained.
- **baselines**: PyTorch Eager (v2.7); PyTorch Dynamo / `torch.compile`
  (Inductor→Triton backend); TVM v0.21 (Relax frontend, default pipeline —
  explicitly **excluding** TVM's CUTLASS/FlashInfer backends "in order to
  compare TVM's own operator fusion capabilities"); hand-optimized
  FlashAttention2 and FlashMLA.
- **hardware/precision**: NVIDIA A10 (24GB) and H800 (80GB), CUDA 12.8,
  Ubuntu 22.04; precision not stated explicitly in the evaluation summary
  obtained (FP8 is itself one of the four benchmarked workloads).
- **metric**: normalized latency vs. PyTorch Eager; example results — MHA:
  1.09x of FlashAttention2, 2.8x/2.6x over Dynamo/TVM on LLaMA-65B; MLA:
  102% of FlashMLA, 2.4x/8.7x over Dynamo/TVM; MoE routing: 1.7x/6.6x over
  Dynamo/TVM; FP8 Quant+GEMM: 3.4x/12.1x over Dynamo/TVM.
- **repo confirms artifact match**: `alibaba/redfuser` README states "Built
  on top of Apache TVM" and lists an ASPLOS 2026 roadmap (flash-attention,
  flash-decoding, MoE routing, quant GEMM examples) — consistent with the
  paper.
- **source**: arXiv 2603.10026 fulltext via
  `WebFetch(arxiv.org/html/2603.10026)`; repo README via `gh api`.

---

## 2. STOF (PPoPP 2026) — `HeyDavid633/PPoPP26-pap161-AE`, arXiv 2506.06095
   (arXiv id not present in the track's input JSON; recovered via Semantic
   Scholar `DOI:10.1145/3774934.3786434` lookup)

- **fusion patterns benchmarked**: three canonical elementwise/reduction/GEMM
  categories used for the kernel-level study — **MI+MI** (Bias+LayerNorm),
  **CI+MI** (GEMM+LayerNorm), **CI+CI** (GEMM+GEMM) [MI = memory-intensive,
  CI = compute-intensive] — plus row-wise/block-wise **sparse-MHA fusion**
  (attention + flexible mask), plus **downstream-operator fusion** of the
  Transformer-block operators beyond MHA (paper refers to them as ops
  "#7-#9, #10-#12, #13-#14").
- **timing granularity — explicitly separated at THREE levels, code-verified**:
  1. *Single/subgraph-kernel level* — MHA performance normalized to PyTorch
     Native, `src/benchmk_attn_unified.py`: **`warmup_iters = 10`,
     `running_iters = 20`** (ground truth from code; the paper-text summary
     obtained via HTML render had said "100", which the code contradicts —
     kept the code value per the instructions' guidance that timing loops in
     code are more reliable than paper text), CUDA-event-style loop
     (`for i in range(warmup_iters + running_iters): if i == warmup_iters:
     <start timer>`), swept over `mask_id in {0,1,2,3}` (four sparsity mask
     patterns) and sequence lengths 128-4096, batch 1-16.
  2. *End-to-end model level* — `src/benchmk_end2end.py`:
     **`warmup_iters = 10`, `running_iters = 10`** (different rep count than
     the kernel-level script). Models: BERT-Base, BERT-Large, GPT2, LLaMA,
     T5, ViT, at (batch, seqlen) settings **(1,128), (8,512), (16,2048)**;
     also a long-context sweep 4k-16k at batch=1.
  3. *Compile/tuning-cost level* — `src/tuning_STOF_cost.py`, reported
     separately in **Table 4**: at (16, 2048), STOF's own tuning is "on
     average 6.7x and 6.9x faster than MCFuser and Bolt"; STOF's tuning
     overhead is "less than 3% of the total tuning time" — this is the only
     paper in the track that quantifies and directly compares search/compile
     time across systems.
  Correctness is verified by dedicated scripts (`correct_verify_attn.py`,
  `correct_verify_end2end.py`) computing `max_diff`/`mean_diff` against a
  PyTorch reference (not a fixed printed tolerance in the slice read).
- **baselines**: PyTorch Native, PyTorch Compile, FlashAttention2,
  FlexAttention, ByteTransformer, Bolt, MCFuser, SPLAT ("FlexAttention, FA2,
  and SPLAT are optimized only for MHA" — noted by the authors as a scope
  caveat on those three).
- **hardware/precision**: NVIDIA RTX 4090 (Ada) and A100 (Ampere); **FP16
  throughout**, "for unified comparison across all methods."
- **metric**: max speedups reported — 1.6x MHA computation, 1.4x end-to-end
  inference, vs. best baseline.
- **source**: arXiv 2506.06095 fulltext via `WebFetch(arxiv.org/html/...)`;
  repo `HeyDavid633/PPoPP26-pap161-AE` `README.md`,
  `src/benchmk_attn_unified.py`, `src/benchmk_end2end.py`, `script/fig9-10.sh`
  (via `gh api`).

---

## 3. LightSeq2 (SC 2022) — `bytedance/lightseq`, arXiv 2110.05722

- **fusion patterns benchmarked** (hand-written CUDA kernels, no codegen/
  autotuning): (a) elementwise chains — "bias adding, dropout, and residual
  kernels" fused into a single kernel launch; (b) LayerNorm — two-pass
  mean/variance reduction fused; (c) Softmax — multi-step numerically-stable
  fusion with shape-tunable templates; (d) cross-attention output kernel —
  merges context computation across decoder layers "to reduce the launch of
  kernels and improve concurrency."
- **timing granularity — explicitly separated at TWO levels**:
  1. *Single-kernel/operator level*: "run each operator/layer for **10**
     times and take the average time" (e.g. LayerNorm ~4x speedup,
     Adam/SGD trainer 2.3x/2.4x speedup over Apex, reported per-operator).
  2. *Full end-to-end model level* (forward+backward **training**, not
     inference-only): "All experiments ran for **40 minutes** to measure
     under a stable state" — no isolated warm-up count given for this level;
     the 40-minute steady-state window is itself the de facto warm-up +
     averaging window.
  No compile/search-time level exists for this paper (kernels are hand
  written, not autotuned) — an explicit, correctly-scoped absence rather
  than an omission.
- **baselines — differ per model family** (a fairness-relevant detail in
  itself): Fairseq+Apex (Transformer MT), HuggingFace (ViT/BERT/GPT-2),
  DeepSpeed (BERT), plus native PyTorch/TensorFlow.
- **models/shapes**: WMT14 En-De Transformer 6e6d/12e12d/24e24d/48e48d,
  hidden 512 (Base) / 1024 (Big); ViT-B-32/ViT-L-32 on CIFAR-10, 224x224,
  seqlen 50; BERT-Base/Large on GLUE/MRPC; GPT-2 Base (117M) / Large (762M)
  on WikiText.
- **hardware/precision**: NVIDIA V100 (32GB) and A100; distributed up to 5
  nodes x 8 GPUs = 40 GPUs; **FP16 mixed precision** ("FP16 for parameters,
  cast to FP32 during computation").
- **metric**: overall speedup 1.4-2.8x (V100) / 1.5-3.5x (A100); WMT14
  training speedup +308% vs. existing systems; BERT 1.44x/1.28x
  (Base/Large) vs. DeepSpeed; GPT-2 1.7-1.9x; ViT 1.2-1.7x; also reports
  GPU-memory footprint (65% of Fairseq's) and GPU utilization (99% vs.
  80-95%).
- **source**: arXiv 2110.05722 fulltext via
  `WebFetch(ar5iv.labs.arxiv.org/html/2110.05722)` (plain `arxiv.org/html/`
  and the raw `/pdf/` endpoint both failed to render usable text for this
  id; ar5iv succeeded).

---

## 4. SumMerge (ICS 2021) — `rohan-bp/summerge`, no arXiv id

- **Track-boundary note**: SumMerge is not really an "operator-chain fusion"
  paper in the same sense as the other four (it does not fuse a *sequence*
  of distinct operators). It exploits **weight-repetition inside a single
  quantized conv/GEMM operator**: "the same weight is bound to repeat many
  times within and across filters... a weight-repetition aware inference
  kernel to factorize and memoize out common sub-computations, reducing
  arithmetic per inference." It was classified into this track presumably
  because its output is likewise a single specialized fused kernel replacing
  a naive multi-step (dequant→conv) sequence; treated here as a
  single-fused-kernel data point, not a subgraph/model-level one.
- **artifact repo `rohan-bp/summerge` is a stub**: contents are only
  `.gitignore` and a two-line `README.md` ("SumMerge Implementation") — no
  benchmark/run scripts are present, consistent with the track input's own
  `artifact_status: "likely"` flag (i.e., unverified/likely-real but not
  confirmed functional). The Semantic-Scholar-reported open-access PDF
  (`dl.acm.org/doi/pdf/10.1145/3447818.3460375`) returned **HTTP 403** to
  WebFetch (paywalled from this environment).
- **what is knowable (abstract only)**: platform = CPU (per track input
  `platform: ["cpu"]`); approach = algorithmic (factorization/memoization,
  not codegen/autotuning); targets quantized DNN inference; claims to
  "maintain the compression benefits of quantization" while reducing
  arithmetic. No shapes, baselines, timing protocol, or metric could be
  confirmed from sources reachable in this environment.
- **source**: track input abstract only; repo `rohan-bp/summerge` (via
  `gh api`, confirmed empty of code); ACM PDF blocked (403).

---

## 5. CGO 2025 RISC-V micro-kernel compiler backend — arXiv 2502.04063
   ("A Multi-level Compiler Backend for Accelerated Micro-kernels Targeting
   RISC-V ISA Extensions")

- **Artifact-link mismatch (data-quality issue, flagged)**: the track
  input's `artifact_url`
  (`github.com/ww2766-www/MLIR-Based-Optimization-for-Protocol-Buffer`) is
  **not** this paper's artifact. That repo's own README describes an
  unrelated course project — "an MLIR-based approach to accelerating
  Protocol Buffer (protobuf) serialization/deserialization" with a
  `protoacc.decode_varint` op, benchmarked on CloudLab against
  HyperProtoBench, achieving "geometric mean decode speedup of about 4.8x."
  This has nothing to do with DNN micro-kernels or RISC-V streaming-register
  accelerators. Verified by reading the repo's README directly via `gh api`.
  **No benchmark/run-script evidence could be gathered for this paper**;
  everything below is abstract-only.
- **what is knowable (abstract only, from arXiv 2502.04063)**: targets a
  "custom RISC-V-based accelerator with hardware loops and streaming
  registers"; approach is a multi-level MLIR-based compiler backend using
  "incremental register allocation over structured IRs" instead of
  classical spilling heuristics, positioned against CPU-focused LLVM/GCC
  backends and against higher-level DNN compilers (MLIR, TVM, OpenXLA) that
  handle fusion but leave "performance-critical micro-kernels... to
  specialized code generators or handwritten assembly." Headline metric:
  "up to 90% FPU utilization across key DNN kernels." No specific kernel
  shapes, baseline speedup numbers, warmup/repetition protocol, or precision
  were recoverable — arXiv `/html/`, `ar5iv`, and `/pdf/` fetches all either
  404'd or returned undecodable binary content to the fetch tool.
- **source**: arXiv 2502.04063 abstract (via `WebFetch(arxiv.org/abs/...)`);
  mismatched-repo confirmation via `gh api
  repos/ww2766-www/MLIR-Based-Optimization-for-Protocol-Buffer`.

---

## Divergences

- **Timing-granularity separation is the single biggest methodological
  split in this track**, and only STOF gets it fully right (kernel-level /
  end-to-end-model-level / compile-time-level, each with its own script and
  its own warmup/rep count verified in code: 10/20, 10/10, and a dedicated
  tuning-cost script respectively). RedFuser reports **subgraph-level only**
  and never gives a whole-model number. LightSeq2 reports **operator-level
  (10 reps) and full-training-model-level (40-minute steady state)** but has
  no compile-time axis (correctly, since it's hand-written). SumMerge and
  the RISC-V-backend paper sit at a **single-fused-kernel** granularity
  (one op replacing a naive sequence, or one micro-kernel's FPU utilization)
  that is neither "subgraph" nor "whole model." A spec that wants to compare
  across this track **must** keep these three-plus granularities in
  separate variants, exactly as the STOF paper itself does internally.
- **Compile/search-time reporting is the second biggest gap**: STOF is the
  *only* paper that quantifies and cross-compares tuning cost (Table 4,
  vs. MCFuser/Bolt). RedFuser has an autotuning mechanism but reports zero
  quantitative search-time numbers — a real fairness gap, since a codegen
  system's net win depends on how many inference/training steps amortize
  that search cost. LightSeq2 and SumMerge are hand-written/algorithmic
  (no search cost applies, and they correctly don't report one). The
  RISC-V-backend paper's compiler-backend register allocation is itself a
  compile-time cost with no numbers surfaced in what was accessible.
- **Fusion-pattern taxonomies differ across papers even though the
  underlying building blocks overlap heavily**: RedFuser's "cascaded
  reduction" (inter-loop-dependent reductions, e.g. softmax→GEMM) vs. STOF's
  MI/CI (memory-intensive/compute-intensive) pairing taxonomy vs.
  LightSeq2's plain elementwise-chain-fusion language are three different
  vocabularies describing overlapping patterns (attention, LayerNorm,
  softmax, back-to-back GEMMs). The spec below defines one canonical
  pattern taxonomy so results are comparable across papers using any of
  these three vocabularies.
- **Baselines are inconsistent both across and within papers**: only
  RedFuser and STOF include `torch.compile`/Dynamo-Inductor; **cuDNN and
  TensorRT never appear as directly named baselines** in any of the three
  well-documented papers (cuDNN is presumably embedded inside the PyTorch
  Eager baseline everywhere; none of these papers targets a TensorRT
  deployment scenario — they're all training- or PyTorch-inference-focused).
  LightSeq2 additionally uses a **different baseline per model family**
  (Fairseq+Apex for MT, HuggingFace for ViT/BERT/GPT-2, DeepSpeed for BERT)
  rather than one fixed baseline set, which the spec resolves by mandating a
  common minimum baseline subset while still allowing extra per-pattern
  hand-tuned references (FlashAttention2-style) as bonus comparisons.
- **Precision is inconsistent**: STOF fixes FP16 throughout for "unified
  comparison"; LightSeq2 uses FP16 mixed precision; RedFuser's fourth
  workload is specifically FP8; SumMerge targets low-bit quantized weights
  on CPU (precision unspecified beyond "quantized"). No paper in the track
  reports FP32 results as a baseline precision point.
- **Two of five papers (SumMerge, RISC-V-backend) could not be verified
  beyond abstract level** given this environment's access (ACM paywall
  403'd; arXiv full-text render failed for the RISC-V paper across three
  mirrors; the RISC-V paper's listed artifact repo is unrelated to the
  paper). Their inclusion in the spec below is therefore qualitative
  (taxonomy placement) rather than a source of concrete numeric defaults.
