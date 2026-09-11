# Attention-kernel track — evaluation methodology survey

Track input: `data/track_inputs/attention-kernel.json`, 9 papers (all surveyed
— fewer than the 5-paper minimum's alternative "or all, if fewer" applies
trivially since 9 > 5). 6 surveyed via arXiv fulltext and/or artifact-repo
benchmark scripts read with `gh api`; the remaining 3 (Hexcute, MetaAttention,
FlashAttention-T) have no arXiv listing in the track input, so their
methodology comes from the artifact repo's benchmark scripts and
artifact-evaluation README instead — which for all three is at least as
concrete as fulltext (they are camera-ready-adjacent AE packages written
specifically to let a stranger reproduce exact figures/tables).

Operation surveyed throughout: fused scaled-dot-product attention,
`O = softmax(Q K^T / sqrt(d) + mask) V`, either as (a) a full-sequence
forward pass (prefill/training, all of Q attends over all of K,V) or (b) a
single-query decode step against a KV cache (LLM autoregressive generation).

---

## 1. PAT — "Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel" (ASPLOS'26)

- key: `conf/asplos/YiZHYSWZZLL26`, arXiv:2511.22333
- **workloads/inputs**: decode-attention only (KV cache, 1 query token/step).
  GQA head configs (num_q_heads, num_kv_heads) ∈ {(64,8), (32,8), (16,8),
  (32,32)}, head_dim=128 fixed. Kernel-only sweep: batch=1134 on A100 (chosen
  to saturate the GPU and avoid tail-effect bubbles), KV lengths swept
  32-4096. Prefix-sharing workloads: (a) synthetic tree-structured batches
  parameterized by branch factor list B=[…] and shared-prefix length list
  L=[…], e.g. B=[1,4,16], L=[128,256,1024]; (b) real traces "ToolAgent" and
  "Conversation" with measured 51.9-75.0% prefix ratio per request and
  2.8-82.6% intra-batch shared-prefix coverage (avg 2.72 distinct shared
  prefixes/batch). End-to-end: continuous-batching serving via vLLM at
  QPS=7/9 (quick) up to the full sweep, Qwen and other vLLM-supported models.
- **timing protocol**: `benchmark/benchmark_kernel.py`'s `Timer()` — 10-iter
  warmup (fixed, not configurable), then `n_repeats` (CLI arg; README/paper
  states 20) timed iterations using paired `torch.cuda.Event` start/end with
  `torch.cuda.synchronize()` around each iteration; reduces to **median**
  latency in milliseconds (`np.median(latencies)`), not mean.
- **timing scope**: kernel-only for the kernel-bench script (KV-cache/block
  tables are pre-built, not counted); end-to-end serving latency (TTFT, TPOT,
  P99 TPOT) separately for the full-system claim.
- **precision & correctness**: KV cache stored FP16, softmax/accumulation in
  FP32. Correctness gate baked into the benchmark script itself:
  `_compare_and_report()` computes max-abs-diff and mean-abs-diff vs. a
  reference output (vLLM's own `flash_attn_with_kvcache`) and asserts
  `max_diff <= max_error` with **max_error = 1e-2**, PASS/FAIL reported per
  baseline.
- **metric**: primary = average latency (ms) / latency reduction (%, "53.5%
  average" vs. baselines); secondary = TPOT and P99 TPOT for end-to-end.
- **baselines**: FlashAttention v2.5.9 (via vLLM's bundled flash-attn),
  FlashInfer v0.2.5, FastTree, RelayAttention, RelayAttention++ (authors'
  own extension), DeFT, Cascade Inference — 7 total, all present as separate
  scripts in `benchmark/`.
- source: arXiv fulltext (`arxiv.org/html/2511.22333`) + repo
  `flashserve/PAT` — `README.md`, `benchmark/benchmark_kernel.py`,
  `benchmark/{DeFT,FastTree,FlashInferAttn,RelayAttention}.py`.

## 2. RedFuser — "An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators" (ASPLOS'26)

- key: `conf/asplos/TangLWSLXZL26`, arXiv:2603.10026
- **workloads/inputs**: not attention-only — RedFuser targets the general
  "cascaded reduction" pattern (a reduction feeding a GEMM with an inter-loop
  dependency), of which softmax-then-PV in attention is one instance. Tested
  attention patterns: MHA (BERT, ViT, LLaMA-65B shapes) and MLA (DeepSeek-R1);
  also MoE routing (softmax+top-k: Switch Transformer, ERNIE, DeepSeek-V2-Lite,
  Qwen3 configs) and FP8 per-token-quantization+GEMM (ERNIE, DeepSeek-R1,
  Qwen3) as separate, non-attention cascaded-reduction instances. Detailed
  shapes are in the paper's Appendix A.5 (not machine-extractable from the
  fulltext render); sequence lengths 1K-8K used in ablations.
- **hardware**: NVIDIA A10 (24GB) and H800 (80GB), Ubuntu 22.04, CUDA 12.8.
- **timing protocol / correctness**: not stated in the extractable fulltext
  (repo is a fork of Apache TVM/Relax with generic `apps/`, `tests/`
  boilerplate; no attention-specific benchmark driver was found distinct from
  the TVM test harness in the top-level listing).
- **metric**: normalized speedup relative to PyTorch v2.7 eager (e.g. "2.8x
  and 2.6x" for MHA); absolute TFLOP/s or GB/s not reported in the excerpt.
- **baselines**: PyTorch v2.7 eager, PyTorch Dynamo/Inductor (Triton
  codegen), TVM v0.21 Relax (default pipeline, CUTLASS/FlashInfer excluded),
  and hand-optimized FlashAttention-2 / FlashMLA.
- source: arXiv fulltext (`arxiv.org/html/2603.10026`) + repo
  `alibaba/redfuser` root listing (TVM-derived tree; no attention-specific
  bench script located).

## 3. Hexcute — "A Compiler Framework for Automating Layout Synthesis in GPU Programs" (CGO'26)

- key: `conf/cgo/ZhangDSHSP26`, no arXiv id; artifact repo is a dedicated
  `hexcute-bench` companion repo (not the compiler repo itself).
- **workloads/inputs**: attention is one of four kernel classes benchmarked
  (GEMM, Attention, FP8 GEMM, warp-specialized kernels), plus mixed-type MoE
  and Mamba selective-scan as separate evaluations. Concrete attention shape
  visible in `scripts/run_a100.sh`:
  `benchmark_attention.py --batch-size 1 --num-heads 32 --num-heads-k 32
  --head-size 64 --seqlen-q 2048 --seqlen-k 2048` (MHA, non-GQA, prefill-style
  since seqlen-q == seqlen-k), plus a separate flash-**decoding** benchmark
  `--batch-size 1 --num-heads 16 --num-heads-k 16 --head-size 128
  --seqlen-k 1024`.
- **hardware**: NVIDIA A100 PCIe 80GB and H100 PCIe/SXM 80GB (H100 PCIe is
  the reference; H100 SXM results "may differ slightly"). GPU clocks are
  explicitly **locked** before every run (`nvidia-smi -lgc 1410`, requires
  `--privileged` in Docker) and restored after (`-rgc`) — a reproducibility
  control none of the other 8 papers document doing.
- **timing protocol / correctness**: not visible in the top-level scripts
  (delegated to `hidet/examples/cute/benchmark_attention.py`, not fetched);
  output is parsed post-hoc by `scripts/parse_results.sh` into a LaTeX table
  (paper's Table II) and PDF plots (Figures 24-29).
- **baselines**: matches cuBLAS/CUTLASS and FlashAttention for GEMM/attention
  respectively per the paper's one_liner; FlashAttention 3 (v2.8.2, built
  with FP8/SM80/backward/paged-KV/append-KV/softcap all disabled to isolate
  the forward dense kernel) is explicitly built as a comparison target in the
  troubleshooting section.
- source: repo `hexcute/hexcute-bench` — `README.md`, `scripts/run_a100.sh`,
  `scripts/{run_h100,run_moe,run_scan,parse_results}.sh`.

## 4. FlashAttention-T — "Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism" (PPoPP'26)

- key: `conf/ppopp/Xu0BXX00000C26`, no arXiv id; artifact-evaluation package
  README is unusually explicit about scope and claims.
- **workloads/inputs**: three distinct evaluations, each isolated in its own
  subdirectory: (1) main throughput comparison — FP16-FP32 fused-attention
  forward pass across "diverse attention configurations" on A100 (Ampere,
  ILP scheduling) and H100 (Hopper, TLP/warp-specialized scheduling)
  (Figure 8); (2) synthetic numerical-precision comparison using random
  attention inputs "sampled from specific distributions" on A100 (Figure 11);
  (3) end-to-end MMLU accuracy with 3 real LLMs — Llama2-13B, Mistral-NeMo,
  Qwen3-14B (Table 2).
- **hardware**: A100 80GB SXM4 and H100 80GB PCIe/SXM5, CUDA 12.8/12.9;
  authors flag that cloud-rented GPUs (runpod.io) may show lower absolute
  throughput than their bare-metal numbers due to stricter power limits
  (H100 PCIe: 310W on runpod vs. 350W in-house) — relative speedups should
  still hold, an explicit caveat about power-limit confounds most attention
  papers omit.
- **timing protocol**: standalone CUDA/C++ binaries (`fp16-fwd-bench`,
  `fp16-fwd-bench-orig` for the FA-2 baseline) built per-config and run via
  CLI, writing a CSV report directly — bypassing Python-launcher overhead
  entirely, unlike FA-2's own upstream benchmark harness. Exact warmup/repeat
  counts inside the compiled binary were not visible from the README alone.
  Correctness claim: RMSE vs. an FP64 reference attention output **< 1e-3**,
  "same order of magnitude as FlashAttention-2, with no observed numerical
  failure" — this is the only paper in the track that benchmarks precision
  as a first-class, separate experiment (not just a pass/fail gate).
- **metric**: primary = attention throughput speedup, reported as
  **1.05-1.17x** average over FlashAttention-2/3 across configs (a notably
  modest claim compared to the track's other papers, consistent with this
  being a tensor/vector-load-balance optimization on top of an already
  near-roofline kernel, not a new algorithm).
- **baselines**: FlashAttention-2, FlashAttention-3, Triton, FlashInfer.
- source: repo `guoqiao7/FlashAttention-T` — `README.md`,
  `1-figure8-main-results/README.md`.

## 5. MetaAttention — "A Unified and Performant Attention Framework across Hardware Backends" (PPoPP'26)

- key: `conf/ppopp/ChenC0XMM0X00W026`, no arXiv id.
- **workloads/inputs**: a code-generation framework, so "the kernel" is
  whatever attention variant the user's Python spec describes — examples
  shipped include softmax attention, RetNet, Mamba2, and MLA. Two headline
  performance figures: Figure 11 (**62 configurations** on NVIDIA H100,
  cross-checked against baselines per-config, output validated as 24 CSV
  files with `MetaAttention` rows) and Figure 14 (**25 configurations** on
  AMD MI250X, 10 CSV files) — i.e. this is the only track paper that
  cross-validates AMD ROCm and NVIDIA CUDA backends for the same generated
  kernel family.
- **timing protocol**: pytest-based (`pytest --run-benchmarks -m 'benchmark
  and h100'`), ~90 min for the H100 sweep and ~20 min for the MI250 sweep;
  results rendered directly to PDF plots, with an explicit fairness note in
  the README: "missing optional baselines are reported and omitted rather
  than replaced with fabricated values."
- **correctness**: separate pytest suite (`tests/functional`, run with
  `--run-gpu -m 'functional and gpu'`) checks "forward results, backward
  gradients where supported, reference implementations, dtypes, and
  operator-specific tolerances" — a 10-case legacy parity matrix plus direct-
  engine and factory coverage; exact numeric tolerance values are not stated
  in the README (only "operator-specific").
- **baselines**: hand-written libraries (FlashAttention, FlashLinearAttention
  explicitly named) and native PyTorch; paper's stated claim is "performance
  comparable to hand-written libraries and better than native PyTorch."
- **hardware**: 1x NVIDIA Hopper GPU (H100 for Fig. 11) or 1x AMD MI200-series
  (MI250X for Fig. 14).
- source: repo `SJTU-IPADS/MetaAttention` — `README.md` (main branch; a
  separate `PPoPP_AE` branch holds the frozen AE snapshot, not fetched here).

## 6. MEATTEN — "Optimizing Attention by Exploiting Data Reuse on ARM Multi-core CPUs" (ICS'24)

- key: `conf/ics/FuYDS24`, no arXiv id; OA PDF at ACM DL.
- **workloads/inputs**: `benchmark/sdpa/bench_meformer_sdpa.cpp` sweeps
  `n_batch, n_head, seq_len` from the command line with `head_dim` **hardcoded
  to 64**; the source comment shows a swept range `seq_len=80..1600, step 80`
  used for the paper's line plots. Inputs are uniform-random FP32 in
  U(-1,1) for Q/K/V, with a random scale and mask tensor also FP32.
- **timing protocol**: host-side wall-clock (`dClock()`, a wrapped
  `gettimeofday`), **`REPEAT=3`** (a compile-time constant, i.e. only 3
  repetitions per configuration) — no explicit warmup phase in the timed
  function itself (the operator-descriptor `reshape`/`setup` calls for the
  XNNPACK baseline happen outside the timed region, which the MEATTEN kernel
  path does not need since it has no separate "plan" step).
  Note: this is the loosest timing protocol among all 9 papers surveyed
  (n=3, no median/min/max, host clock at microsecond-ish resolution).
- **precision & correctness**: FP32 throughout (CPU, no tensor-core /FP16
  path). `compareTensor()` checks **element-wise diff outside ±1e-3** (an
  absolute tolerance, not relative) against a reference tensor, printing the
  first offending (batch, i, j) and a pass/fail-style "no diff" message — not
  wired into the benchmark driver as a hard gate (the perf loop does not call
  `compareTensor()` itself in the visible code path).
- **metric**: throughput derived from wall-clock time (paper reports GFLOP/s
  in the text); ARM-specific claim is multi-dimensional loop
  permutation/tiling guided by an analytic cost model over the on-chip cache
  hierarchy.
- **baselines**: XNNPACK's `xnn_*_scaled_dot_product_attention_nhtc_f32`
  operator (Google's SIMD-optimized CPU kernel library) is the only baseline
  visible in the benchmark source; the paper text additionally claims
  comparison against unspecified "state-of-the-art libraries and compilers."
- **hardware**: 3 unspecified ARM multi-core platforms (figure only, not
  machine-readable from the README).
- source: repo `HPC4AI/MeAtten` — `README.md`,
  `benchmark/sdpa/bench_meformer_sdpa.cpp`.

## 7. ByteTransformer — "A High-Performance Transformer Boosted for Variable-Length Inputs" (IPDPS'23)

- key: `conf/ipps/ZhaiJWJZCLZ23`, arXiv:2210.03052 (cited by the repo, not
  listed in the track-input JSON's `arxiv` field)
- **workloads/inputs**: BERT-encoder self-attention (bidirectional, no causal
  mask), variable-length via a padding-free algorithm. `benchmark/bert_bench.sh`
  fixes `head_num=12, head_size=64` (BERT-base) and sweeps
  `batch_sizes=(1 2 4 8 16)` x `seqlens=(32 64 128 192 256 320 384 448 512
  576 640 704 768 832 896 960 1024)`, with `avg_seqlen_percent=60` — i.e. the
  padding-free benefit is measured specifically at 60% average fill (40%
  padding waste in the naive baseline), a realistic-but-favorable-to-the-
  paper operating point since the padding-free speedup grows with more
  padding.
- **precision**: FP16 only in the released benchmark script
  (`precisions=("fp16")`), though the C++ build supports both FP16/FP32
  (`-DDataType=FP16` CMake flag).
- **timing protocol**: not visible as warmup/reps in the shell driver itself
  — `bert_transformer_test.py` (not fetched) presumably contains the timed
  loop; the log parsing (`tail -n 1 ... | awk '{print $3}'`) extracts a
  single latency number per (batch, seqlen) cell, i.e. **the released script
  reports one number per configuration**, with no visible min/max/median
  spread reported to the user.
- **correctness**: unit tests (`unit_test/`, `bert_transformer_test.py
  --export_data`) exist separately from the perf benchmark and generate
  `.in`/`.out` reference tensors for a C++ correctness binary
  (`bert_transformer_test`), but no tolerance value is visible in the README.
- **metric**: latency in ms per (batch, seqlen, precision) cell (README's own
  results table for A100+CUDA11.6+PyTorch1.13); paper claims Fused-MHA
  6.13x over PyTorch, end-to-end 55-138% faster than FasterTransformer/
  TurboTransformer/DeepSpeed/TF-XLA/PyTorch.
- **baselines**: PyTorch, PyTorch JIT, TensorFlow XLA, Tencent
  TurboTransformer, Microsoft DeepSpeed-Inference, NVIDIA FasterTransformer
  (with and without its own remove-padding optimization).
- source: arXiv fulltext (abstract-level; full evaluation section not
  extractable from the abs-page render) + repo `bytedance/ByteTransformer` —
  `README.md`, `benchmark/bert_bench.sh`.

## 8. LightSeq2 — "Accelerated Training for Transformer-Based Models on GPUs" (SC'22)

- key: `conf/sc/WangWXHQDWL22`, arXiv:2110.05722 (cited by the monorepo
  README, which conflates LightSeq v1 (NAACL'21, inference), LightSeq2
  (SC'22, training — this paper) and v3 (int8) under one umbrella repo).
- **workloads/inputs**: BERT, GPT2, Transformer (encoder-decoder, WMT14
  En-De headline benchmark), and ViT — training and inference. Fine-grained
  per-configuration shape sweeps (batch/seqlen/hidden) are not visible in the
  top-level monorepo README; `docs/performance.md` (not fetched) is where the
  README points for "more performance results."
  Repo-level table (v1-v3 combined, not necessarily this SC'22 paper's exact
  numbers): Transformer training speedup 1.44-1.99x fp16 / 1.44-1.86x int8
  across batch-token sizes 512-15000; BERT training 1.59-2.12x fp16.
- **hardware**: training experiments on 1x A100; inference on 8x A100.
- **precision**: FP32 baseline vs. LightSeq FP16 and (v3) INT8 mixed
  precision — this paper (v2, training) specifically claims the FP16 numbers
  above; int8 training support was added in v3.
- **timing protocol / correctness**: not documented in the fetched README;
  the monorepo's `examples/training/*` scripts (not fetched) are the likely
  location.
- **metric**: speedup vs. "PyTorch fp16 training" / "PyTorch QAT" baseline
  (ratio, not absolute GFLOP/s or ms).
- **baselines**: PyTorch (fp16 mixed-precision autocast) for the fp16 claim,
  PyTorch quantization-aware-training (QAT) for the int8 claim.
- source: repo `bytedance/lightseq` `README.md` (monorepo covering v1-v3,
  methodology attribution to the specific SC'22 LightSeq2 paper is
  approximate — flagged in Divergences below) + arXiv abstract page
  (2110.05722, abstract only, no fulltext evaluation section retrieved).

## 9. E.T. — "Re-Thinking Self-Attention for Transformer Models on GPUs" (SC'21)

- key: `conf/sc/ChenHPLG0D021`, no arXiv id.
- **workloads/inputs**: BERT-style self-attention (bidirectional encoder).
  Repo ships 3 example encoders under `test/`: on-the-fly attention with
  tensor-tile-pruned linear transforms (`encoder_tile_test`), attention-aware
  pruning with pruned self-attention (`encoder_prune_test`), and a
  sequence-length-aware optimized encoder (`encoder_length_test`) — all
  driven with random data (no fixed benchmark-shape list visible in the
  README; shapes are compile-time/CLI args to the test binaries, not
  extracted here). Kernel source (`kernels/attention.cu`) confirms FP16
  (`half`/`half2`) arithmetic with warp-level softmax (max-reduce then
  shifted-exp then sum-reduce, matching the numerically-stable softmax
  pattern) and tensor-core MMA (`wmma`, 16x16x16 fragments) for the QK^T/PV
  GEMMs.
- **hardware**: NVIDIA V100S, CUDA 11.4 (the paper predates A100-generation
  attention kernels entirely — 2021, oldest paper in this track).
- **timing protocol / correctness**: not visible in the top-level README;
  presumably inside the `test/` binaries (not fetched in depth — out of
  time budget for this survey, noted as an open question).
- **metric / baselines**: paper's stated claim ("beats TensorRT/
  FasterTransformer") from the track-input one_liner; not independently
  re-derived from the fetched README, which only documents build/run
  instructions, not a results table.
- source: repo `cctry/E.T.` — `README.md`, `kernels/attention.cu`
  (partial read).

---

## Divergences

- **Prefill vs. decode is the single biggest scope split.** PAT and (one of
  two benchmarks in) Hexcute measure single-query KV-cache **decode**
  attention (memory-bound, causal-by-construction, GQA-heavy). ByteTransformer,
  E.T., MEATTEN, and (the other Hexcute benchmark) measure full-sequence
  **prefill/encoder** attention (compute-bound-ish, Q length == K/V length).
  RedFuser, MetaAttention, and FlashAttention-T are agnostic/both. These are
  not interchangeable claims — a kernel tuned for one is not a fair
  competitor on the other's numbers — hence this spec's two separate GPU
  variants (Section "Step 2" below).
- **Causal vs. bidirectional mask.** Decode attention is always causal by
  construction (queries only see past). Prefill attention splits: BERT/ViT-
  style (ByteTransformer, E.T., MEATTEN) is bidirectional/no mask; LLM
  prefill (implicit in RedFuser's LLaMA-65B shapes, Hexcute's
  `benchmark_attention.py`, FlashAttention-T's Llama/Mistral/Qwen accuracy
  eval) is causal. No paper in this track benchmarks the *same* kernel under
  both masks to isolate the mask's cost — this spec adds that as an explicit
  sub-case.
- **Timing rigor varies by roughly 2 orders of magnitude in care.** PAT uses
  paired CUDA events with 10-iter warmup and reports median (best practice in
  this track). MEATTEN's `REPEAT=3` host-clock loop is the loosest — no
  warmup, n=3, no distributional statistic. ByteTransformer's released driver
  reports a single number per cell with no visible repetition at all. This
  spec's protocol (below) does not adopt any single paper's practice
  wholesale; it fixes explicit warmup/reps/statistic and flags MEATTEN's n=3
  as a violation to correct, matching the project's other tracks (see
  `benchspecs/spmm/spec.yaml`'s analogous treatment of RoDe/GE-SpMM/SMaT).
- **Correctness tolerance is inconsistent in both value and enforcement.**
  PAT: max-abs-diff <= 1e-2, hard-asserted in the benchmark script itself
  (PASS/FAIL recorded). FlashAttention-T: RMSE < 1e-3 vs. FP64 reference,
  treated as a first-class *result* (Figure 11), not just a gate. MEATTEN:
  elementwise abs diff outside ±1e-3 flagged, but the check function is not
  called from the perf-timing driver (i.e. correctness and performance are
  never verified in the same run in the released code). MetaAttention:
  "operator-specific tolerances," value unspecified in the README.
  ByteTransformer/E.T.: correctness testing exists as a separate unit-test
  target from performance benchmarking, no tolerance published. This spec
  fixes one tolerance policy per precision and makes the gate mandatory
  before any timing counts — following the project's established pattern of
  not trusting a track's own unverified-by-default correctness checks (see
  `benchspecs/spmm/spec.yaml` notes on GE-SpMM/SMaT's commented-out checks).
- **GPU vs. CPU is a hardware-family split, not just a variant knob.**
  MEATTEN is the only CPU (ARM multi-core) paper in the track; all 8 others
  target NVIDIA GPUs (MetaAttention additionally targets AMD MI200). CPU
  attention has no tensor cores, no CUDA events, different baselines
  (XNNPACK, not cuBLAS/cuDNN/FlashAttention), and an apples-to-oranges
  GFLOP/s scale vs. GPU — kept as a fully separate variant.
  RedFuser's TVM-derived repo also nominally supports non-GPU backends
  (`hexagon`, `android_rpc` targets visible in `apps/`) but none of its
  attention evaluation in the fulltext excerpt used them; treated as GPU-only
  for this track.
  MetaAttention is the one paper that treats AMD ROCm as a first-class,
  separately-validated target (Figure 14, MI250X) alongside NVIDIA (Figure
  11, H100) — this spec's GPU variants are written vendor-neutral
  (GB/s / TFLOP/s, not CUDA-specific) so an AMD run is a valid submission
  against the same variant, not a separate one.
- **Compiler/codegen papers (RedFuser, Hexcute, MetaAttention) evaluate a
  moving target, not a fixed kernel.** Their "kernel under test" is whatever
  their compiler emits for a given attention spec, compared against
  hand-written libraries (FlashAttention/FlashMLA/FlashMLA/FlashInfer) and
  other compilers (Triton/TVM). This spec treats their generated attention
  kernel as just another candidate submission to the prefill or decode
  variant (whichever shape it targets), not a separate variant — the fusion
  boundary (softmax+GEMM fused vs. not) is an implementation detail of the
  candidate, not part of the benchmark's definition of the operation.
- **FP8 attention is not actually benchmarked by any of the 9 papers.**
  RedFuser lists "FP8 GEMM" as a benchmarked kernel class, but its FP8
  workloads (ERNIE/DeepSeek-R1/Qwen3) are per-token-quantized MoE/GEMM, not
  full QK^TV attention; Hexcute's repo README separately buckets "Attention"
  and "FP8 GEMM" as distinct benchmark categories. No paper reports an FP8
  attention correctness tolerance or throughput number. This spec therefore
  does NOT add a dedicated fp8-attention variant (see `open_questions`).
