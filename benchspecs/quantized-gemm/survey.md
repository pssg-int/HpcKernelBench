# Survey: quantized-gemm track evaluation methodology

Track input: `data/track_inputs/quantized-gemm.json` (16 papers, low/mixed-precision
GEMM kernels — INT1..8, FP4/FP5/FP6/FP8, MX-microscaled, and lossless-compressed
weight formats for LLM inference, plus a few non-LLM arbitrary-precision/training
microbenchmark papers). **13 of 16 papers surveyed below**: 6 via arXiv fulltext
(html/abs, 3 with usable prose after PDF-extraction failures), all 13 cross-checked
against artifact repo README + benchmark/test source via `gh api`. Not surveyed:
APE (`conf/ics/MaWFZXH0Z22`, no OA/arXiv text extracted, out of scope — CPU/DSA
bitwidth emulation, not GEMM-specific), TCBNN (`journals/tpds/LiS21`, appears only
as APNN-TC's BNN baseline), Cowan CGO'20 (`conf/cgo/CowanMCBC20`, ARM/CPU codegen,
redundant with QIGen's CPU coverage below).

---

## 1. Tilus (Ding et al., ASPLOS'26) — arXiv 2504.12984, fulltext

- **workloads/inputs**: matmuls extracted from 3 real LLMs — Gemma-2-9B,
  Qwen2.5-32B, Llama-3.3-70B-Instruct — characterized as `(BS, N, K)` tuples,
  e.g. `BS=16, K=8192, N=57344` (a Llama-3.3-70B MLP shape). Not a fixed
  matrix suite; shapes are pulled live from named model configs.
- **timing protocol**: CUDA events; 50 executions per kernel, **median**
  reported; **L2 cache explicitly cleared before each execution**; end-to-end
  model runs use 10 repetitions.
- **timing scope**: kernel-only (per-op latency); no preprocessing timing
  discussion (weight packing assumed ahead-of-time / not timed).
- **precision & correctness**: uint1-uint8, int2-int8, float3-float8
  (configurable exponent/mantissa, e.g. e3m2) — arbitrary 1-8 bit weights,
  fp16/bf16 activation. Correctness validation method not stated in the
  fulltext extraction (paper focuses on latency/speedup).
- **metric**: latency (ms), speedup vs. baselines.
- **baselines**: cuBLAS, Triton v3.1.0 (autotuned), Ladder/BitBLAS v0.0.1
  (autotuned), Quant-LLM (heuristic policy), Marlin v0.1.1.
- **hardware**: NVIDIA L40S (48GB) primary; A100/H100 for cross-arch checks;
  CUDA 12.6.3.
- **source**: arXiv fulltext (arxiv.org/html/2504.12984).

## 2. ZipServ (Fan et al., ASPLOS'26) — repo README + `kernel_benchmark/test_mm.cu`

- **workloads/inputs**: not a quantization scheme but **lossless** weight
  compression (Tensor-Core-Aware Triple Bitmap Encoding) with a fused
  decompress+GEMM kernel (ZipGEMM) — included here because its evaluation
  pattern (fused dequant/decompress inside the GEMM) is structurally
  identical to weight-only quantized GEMM. End-to-end tests target
  Llama-3-8B/70B-Instruct and Mistral-24B via vLLM (`-b/-i/-o` = batch/input-
  len/output-len flags, e.g. `-b 8 -i 64 -o 256`).
- **timing protocol**: `cudaEvent` start/stop around a loop of
  `BENCHMARK_ITERATION` calls preceded by a `WARM_UP_ITERATION` loop; time
  divided by iteration count (**mean**, not median). A reference value is
  computed on the host (nested loop over `K_GLOBAL × N_GLOBAL`) for
  correctness comparison against the compressed kernel's output, run
  alongside (not instead of) the timing loop.
- **timing scope**: kernel-only microbenchmark (`test_mm`, `test_decompress`)
  vs. end-to-end vLLM inference (`run_linfer.py`) reported as two distinct
  scripts/numbers, never merged.
- **precision & correctness**: bit-exact (lossless) — decompressed weights
  must reconstruct the original bf16/fp16 values exactly; this is a stronger
  guarantee than the tolerance-based checks used by lossy-quantization
  papers below.
- **metric**: up to 30% model-size reduction; up to 2.21x kernel speedup over
  cuBLAS; 1.22x average end-to-end speedup over vLLM.
- **baselines**: cuBLAS (kernel-level), vLLM dense (end-to-end).
- **hardware**: Ampere+ (compute capability ≥ 8.0: A100, A6000, RTX 4090).
- **source**: `github.com/xxyux/ZipServ` README.md + `kernel_benchmark/test_mm.cu`.

## 3. Quantix (Chen et al., PPoPP'26) — repo (thin) + abstract/web

- **workloads/inputs**: non-uniform (clustering-based) 3-bit LLM weight
  quantization; artifact repo (`quantix/` dir, CMake build) contains no
  README beyond a one-line description — no benchmark script content
  retrievable via API listing alone (only top-level dirs seen: `quantix/`,
  `CMakeLists.txt`).
- **precision**: W3 (non-uniform clustering-based 3-bit weights),
  hardware-aligned bit-shuffling for access, fused dequant+matmul pipeline
  targeting both CUDA cores and Tensor Cores.
- **metric** (from abstract/press coverage): average kernel-level speedup
  4.82x over FP16 cuBLAS; end-to-end speedup up to 11.46x over SOTA
  quantization methods; reported on NVIDIA L40 GPUs.
- **baselines**: FP16 cuBLAS (kernel), unspecified SOTA quantization systems
  (end-to-end).
- **source**: abstract (`data/track_inputs`) + WebSearch (dl.acm.org listing)
  + `github.com/yuang-chen/Quantix-PPoPP26` repo listing. Flagged as the
  thinnest-evidence paper in this survey — protocol details (warmup/reps/
  timer/tolerance) could not be recovered without the paywalled PDF.

## 4. QFactory (Zhang et al., ATC'25) — repo README (`QFactory-AE`)

- **workloads/inputs**: Qtile-graph compiler generating quantized kernels;
  AE reproduction targets Figures 6-13 and Table 3 across **three GPU
  generations** — V100+A100 (Server A) and H100 (Server B) — explicitly
  because kernel behavior differs by architecture.
- **timing protocol**: not exposed in README (delegated to `reproduce.sh
  fig{N}` scripts not fetched), but the figure/table split reveals the
  intended structure precisely:
  - Fig 6/7/8: kernel performance (by GPU: H100 / A100 / V100)
  - Fig 9: **scaling matrix sizes** (an explicit shape-sweep ablation)
  - Fig 10: **scaling bit-widths** (an explicit precision-sweep ablation)
  - Table 3: **varying batch size** (an explicit batch-sweep ablation)
  - Fig 11/12: end-to-end (H100/A100)
  - Fig 13: performance breakdown (preprocessing vs. kernel vs. other)
- **timing scope**: kernel and end-to-end kept in separate, clearly-labeled
  figures — good practice worth preserving in the spec.
- **baselines**: integrates and patches Marlin (`marlin-hopper.patch` for
  H100), BitBLAS (`vllm-bitblas`), llama.cpp, vLLM — i.e. treats Marlin and
  BitBLAS as the SOTA kernel baselines to beat, and vLLM/llama.cpp as
  end-to-end serving baselines.
- **source**: `github.com/zqh-wz/QFactory-AE` README.md.

## 5. MARLIN (Frantar et al., PPoPP'25) — arXiv 2408.11743 (abstract only
   usable) + repo `bench.py` (full protocol, read directly)

- **workloads/inputs**: exact per-layer `(K, N)` shapes hard-coded for 5
  model families: LLaMA-7B `(4096,3*4096),(4096,4096),(4096,2*10752),
  (10752,4096)`; LLaMA-13B/33B/65B analogous (fused QKV = 3K columns, fused
  gate+up = 2\*intermediate); Falcon-180B `(14848, 14848*5+1024),
  (14848*5, 14848)` (parallel-attention+FC fusion). Batch sizes swept
  `{1,2,4,8,16,32,64,128}` by default, `{...,256,512,1024,2048}` in "ALL"
  mode. Weight group size `-1` (per-column) and `128` (GPTQ-standard).
- **timing protocol** (from `bench.py`): `warmup=1, iter=10`; **host-side**
  `time.time()`, with `torch.cuda.synchronize()` called only once **before**
  the timed loop starts (not between each of the 10 iterations) — the
  comment explicitly justifies this as mimicking "realistic model inference
  as many launches are submitted to the kernel queue" i.e. hiding launch
  overhead is treated as *representative*, not a bug. A `time.sleep(1.)`
  is inserted between benchmarked configs "to cool down the GPU ... to
  avoid throttling."  Statistic: **mean** (single total / iter).
- **timing scope**: kernel-only (`marlin.mul` vs `torch.matmul` dense fp16),
  no preprocessing (weight packing) in the timed path.
- **precision & correctness**: INT4 weight, fp16 activation. Correctness
  test (`test.py`, separate from `bench.py`): dequantized-kernel output vs.
  a **round-trip reference** (the same INT4-quantized weight dequantized in
  fp16, not the original unquantized weight) — `mean(|C - C_ref|) /
  mean(|C_ref|) < 0.001`. This isolates *kernel* correctness from
  *quantization-algorithm* accuracy loss.
- **metric**: TFLOP/s (`2*M*N*K/time`), GB/s (`(2*A+4*B+2*C+2*s numel)/time`
  — note the read side counts INT4-packed B, not dequantized), and
  `speedup = dense_time / quant_time` per layer, aggregated by
  time-weighted average across a model's 4 layer types.
- **baselines**: dense fp16 `torch.matmul` (own kernel's only baseline in
  the repo; paper additionally reports vLLM integration speedup up to 2.8x
  end-to-end).
- **source**: `github.com/IST-DASLab/marlin` `bench.py`, `test.py`.

## 6. MXBLAS (Wang et al., SC'25) — repo README + `bench/shapes_def.py` +
   `mxblas/utils.py` (timing/correctness helpers)

- **workloads/inputs**: GEMM shapes from **7 non-Llama model families' linear
  layers** — GPT-Neo-1.3B/2.7B, GPT-J-6B, GPT-NeoX-20B, PaLM-8B/62B, T5
  (small/base/large/3B/11B), FLAN-T5 — each as 4 `(in,out)` tuples (attn
  proj, MLP up, MLP down, output/vocab proj), e.g. GPT-NeoX-20B:
  `(6144,6144),(6144,24576),(24576,6144),(6144,50432)`. `bench_all.py`
  accepts `--models=...,--Ms=1024,2048,4096,8192,--scaling_pattern=TT,BB,GB,CC`
  where the scaling pattern is the MX block-scaling granularity
  (Tensor-wise/Block-wise/Group-wise/Channel-wise combinations).
- **timing protocol** (`mxblas.utils.GPU_bench`, read directly):
  `warmup=30, iters=100` default; **mean** time via a `DurationTimer`
  context manager; optional Kineto-based per-kernel isolation
  (`bench_kineto(..., flush_l2=False)`) — **L2 cache is explicitly NOT
  flushed** between iterations in the default path.
- **precision & correctness**: FP8 (E4M3) micro-scaled (MX) format, both
  operands quantized; scale = `x.abs().max()/448` per block, clamped to
  `[-448,448]` (the E4M3 dynamic range). Correctness helpers:
  `relative_error` (mean elementwise |Δ|/|ref|, zeros excluded) and
  `calc_diff` (cosine-similarity-based `1 - 2(x·y)/(|x|²+|y|²)`, more
  robust to FP8's coarse mantissa than plain relative error). Example test
  output: `difference rate: 0.0715%` (~7e-4) at `M=N=K=8192`.
- **metric**: TFLOPS from `tests/test_mxgemm.py` output; box-plotted speedup
  distribution across scaling patterns/models (Figure 9 of the paper).
- **baselines**: CUTLASS, DeepGEMM, COAT, SGLang's `sgl_kernel`, FBGEMM,
  Transformer Engine — i.e. every major FP8/MX GEMM library, not just
  vendor cuBLAS.
- **hardware**: NVIDIA Hopper only (compute capability 9.0, needs TMA/WGMMA/
  MBarrier); CUDA ≥ 12.1.
- **source**: `github.com/yatorho/MXBLAS` README.md, `bench/shapes_def.py`,
  `bench/kernels.py`, `mxblas/utils.py`.

## 7. Quant-LLM / FP6-LLM / TC-FPx (Xia et al., ATC'24) — arXiv 2401.14112
   fulltext + repo README

- **workloads/inputs**: linear layers from LLaMA (7B/13B/33B/65B) and OPT
  (30B/65B/175B); end-to-end tests use LLaMA-13B/70B, OPT-30B. Batch sizes
  {8, 16, 32} for kernel-level; end-to-end uses 0.5K prompt / 1.5K generated
  tokens per request, batch up to memory limit (32 for LLaMA-70B, 16 for
  OPT-30B on 80GB A100).
- **timing protocol**: NVIDIA Nsight Compute (kernel profiling), Nsight
  Systems (end-to-end latency breakdown); explicit warmup/rep counts not
  disclosed in the README/fulltext extraction.
- **timing scope**: kernel-level (A100-40GB) and end-to-end (A100-SXM4-80GB
  DGX) reported as **separate hardware configurations**, not just separate
  numbers — a scale-dependent methodological choice (single-GPU capacity
  differs by SKU).
- **precision & correctness**: FP6 (`e3m2`) and FP5 (`e2m2`) weights,
  fp16 activation (W6A16/W5A16), per-output-channel scale. Kernel
  correctness tests exist (`tests/python`, `tests/cpp`) comparing FP6-LLM
  output against the FP16 PyTorch/cuBLAS reference, but no explicit
  numeric tolerance is stated in the README (paper likely states it in the
  eval section, not extracted here).
- **model accuracy**: reported **separately from kernel speed** — Table 1/2
  give zero-shot task accuracy and perplexity for FP6 across LLaMA sizes
  1B/13B/65B, explicitly contrasted with 4-bit's larger degradation. This
  is the clearest example in the track of a paper treating "does it still
  work" and "is it faster" as two independent, both-mandatory results.
- **metric**: speedup vs. baseline (up to 8.9x/2.6x/1.9x over
  bitsandbytes/cuBLAS/TensorRT-LLM respectively; on average 7.2x/2.1x/1.3x);
  end-to-end tokens/GPU-second.
- **baselines**: cuBLAS W16A16, TensorRT-LLM W8A16 (coarse+fine-grained) and
  W4A16, bitsandbytes W4A16 (kernel); DeepSpeed FP16 (end-to-end).
- **source**: arXiv fulltext (arxiv.org/html/2401.14112v2) +
  `github.com/usyd-fsalab/fp6_llm` README.md.

## 8. MixQ (SC'24) — repo README + `benchflops.py` (read directly)

- **workloads/inputs**: Llama-2-7B/13B/70B, Llama-3-8B, Falcon-7B/40B,
  ChatGLM-7B, Qwen2-7B; text-generation throughput measured via
  `benchflops.py --batch_size {32,512,...} --bit {4,8}` on WikiText2 data
  (333,088 validation lines).
- **timing protocol** (`benchflops.py`, read directly): NOT an isolated
  kernel microbenchmark — it drives the **full HF-style model forward
  pass** through `generate()`: a 10-step unsynchronized warmup loop
  (`model(inputs, use_cache=True)`, no explicit sync/timing), then a real
  loop of `n_generate` decode steps each wrapped in
  `torch.cuda.synchronize()` before/after `time.time()`; reported statistic
  is `np.median(generate_time)` → `tokens/s = 1/median(latency) * batch`.
  **This measures end-to-end per-token decode latency of the whole model,
  not the quantized GEMM kernel in isolation** — a significant scope
  divergence from MARLIN/Tilus/MXBLAS above.
- **precision & correctness**: W8A8O16 (INT8 weight+activation, fp16
  output) primary; also integrates QUIK's 4-bit path. Locality-based
  outlier prediction (predicts 95.8% of per-token outliers) plus a
  "quantization-ahead-of-detection" verification step — outlier handling is
  itself a correctness mechanism, not just an accuracy knob.
- **metric**: decode tokens/second, peak VRAM (GB and % of device memory).
  Example repo-reported number: MixQ W8A8O16 35.02 it/s vs AWQ 16.71 it/s
  at batch=32 on WikiText2 PPL task (~2.1x).
- **model accuracy**: `evalppl.py`/`runppl.sh` (perplexity) and `mmlu.sh`
  (MMLU downstream accuracy) exist as separate scripts from the throughput
  scripts — again splitting speed and accuracy into distinct artifacts.
- **baselines**: AWQ, bitsandbytes-int8, QUIK, FP16 HF baseline.
- **hardware**: NVIDIA A100-PCIE-40GB (numbers in README).
- **source**: `github.com/Qcompiler/MIXQ` README.md, `benchflops.py`.

## 9. Magicube (Li, Osawa, Hoefler; SC'22, Best Paper Finalist) — arXiv
   2209.06979 (PDF extraction failed, binary) + repo README (full protocol)

- **workloads/inputs**: **DLMC** (Diffusion Language Model Compression
  benchmark's underlying sparse-matrix collection — `dlmc.tar.gz` from
  `storage.googleapis.com/sgk-sc2020/`), the standard sparse-DNN matrix
  suite (also used by Sputnik/vectorSparse); mixed low-precision int4/int8
  combinations for both SpMM and SDDMM. Not GEMM in the dense sense — a
  **quantized-sparse** operator, included in this track for its Tensor-Core
  low-precision-int techniques, but structurally a different workload class
  (sparse operand) from the other 12 papers here.
- **timing protocol**: not visible without deeper source read; README shows
  wall-clock reproduction *budgets* only (e.g. "SpMM main comparison: ~8
  hours (Magicube) + ~13 hours (baselines)"), implying a full sweep over
  the DLMC suite's many sparsity levels/matrices rather than a handful of
  synthetic shapes.
- **precision & correctness**: mixed int4/int8 on Tensor Cores; accuracy
  validated end-to-end via sparse Transformer inference quality ("comparable
  accuracy" claim), not via a stated numeric kernel tolerance.
- **metric**: speedup vs. vendor-optimized library (avg 1.44x, up to 2.37x);
  1.43x vs. SOTA on end-to-end sparse Transformer inference.
- **baselines**: vendor-optimized sparse library (cuSPARSE-class),
  vectorSparse, cuDNN-fp16 dense (end-to-end sparse Transformer comparison).
- **hardware**: NVIDIA A100-SXM4-40GB; GCC 8.4.1, CUDA 11.4.0, PyTorch 1.9.0.
- **source**: `github.com/ParCIS/Magicube` README.md (PDF fulltext blocked
  by binary/FlateDecode extraction failure via WebFetch).

## 10. QGTC (Wang, Feng, Ding; PPoPP'22) — repo README + bench script
    `2_7c_QGTC_GEMM_INT8.py` (read directly)

- **workloads/inputs**: not LLM-shaped — square/rectangular synthetic GEMMs
  `M=K ∈ {1024,2048,4096}`, `N ∈ {16,32,64}` (small N mimics GNN feature
  dimensions, its actual target domain: quantized GNN layers), swept across
  **bitwidth ∈ {1,...,8}** for both operands via bit-decomposed GEMM
  (`QGTC.bitMM2Bit_profile`). Also evaluates full GCN/GIN models
  (Cluster-GCN, batched GIN) on 4 graph datasets (artist, soc-BlogCatalog,
  ppi, ogbn-arxiv).
- **timing protocol**: profiling delegated to an internal CUDA-side
  `bitMM2Bit_profile` call (source not fetched); reported metric is TFLOPs
  per (bitwidth, shape) combination, one point per run — no explicit
  warmup/rep count visible from the Python driver script itself.
- **precision & correctness**: arbitrary bit-serial int1-int8 via
  bit-decomposition (each bit-plane as a separate 1-bit GEMM, then
  weighted-summed) — a fundamentally different implementation strategy
  from the group-scaled dequantization used by MARLIN/Tilus/FP6-LLM/MXBLAS.
  No numeric correctness gate visible in the driver scripts.
- **metric**: TFLOPs (bit-serial GEMM), epoch time (ms) for GNN models.
- **baselines**: cuBLAS `cublasGemmEx` INT8 (kernel-level, own
  `cuBLASGemmEX/` subdir), DGL (GNN end-to-end).
- **source**: `github.com/YukeWang96/QGTC_PPoPP22` README.md,
  `2_7c_QGTC_GEMM_INT8.py`.

## 11. NITI (Wang, Rasmussen, Louri, Song; TPDS'22) — arXiv 2009.13108
    (abstract only)

- **workloads/inputs**: MNIST, CIFAR10 (8-bit throughout), ImageNet
  (16-bit accumulation, 8-bit datapath) — small vision CNNs, **not LLM
  GEMM shapes**, and **training** (forward+backward+weight-update), not
  inference. Included in the track input list but scoped out of this
  track's main variants below for that reason.
- **precision**: pure integer arithmetic throughout training; per-layer
  block-scaling exponent for dynamic range instead of a float scale factor.
  Gradients need ≤5 bits.
- **baselines**: equivalent-architecture FP training (accuracy comparison
  only, per the abstract).
- **source**: abstract via `arxiv.org/abs/2009.13108` (fulltext extraction
  returned abstract-level content only; not re-attempted given the paper's
  out-of-scope status for GEMM-kernel benchmarking).

## 12. APNN-TC (Feng, Wang, Geng, Li, Ding; SC'21) — arXiv 2106.12169 (PDF
    extraction failed) + repo README (full quantitative detail)

- **workloads/inputs**: **fixed-shape microbenchmark**, not LLM-derived:
  GEMM `[M,N,K] = [64, N, K]` with `N=K ∈ {128,256,...,1024}` (step 128);
  CONV `[H,W]=[16,16]`, kernel `3×3`, `O=C ∈ {128,...,1024}`. Bit-width
  combos named `w{1,2}a{2,3,4}` (e.g. `w1a2` = 1-bit weight, 2-bit
  activation). Also full-network AlexNet/VGG-variant/ResNet18 at `w1a2`.
- **timing protocol**: reported as `Time (ms)` per shape (both APNN-TC and
  CUTLASS baseline binaries print one number per run); exact
  warmup/repeat/statistic not stated in the README's example output
  (implementation detail, not extracted).
- **precision & correctness**: **arbitrary-bit emulation via INT1 Tensor
  Core primitives** (XOR/AND Boolean ops), the same family of technique as
  QGTC — both operands (weight AND activation) can be sub-8-bit. No
  numeric correctness gate visible in the README; the paper's contribution
  is framed purely as an emulation-algorithm + performance result.
- **metric**: TOPS (`w1a2` GEMM at `N=K=1024`: **18.66 TOPS** vs. CUTLASS
  INT4 at the same shape: **9.18 TOPS** — a concrete literal artifact
  output pair usable as a regression baseline); ms latency for full
  networks (AlexNet: APNN-TC 0.36ms vs. CUTLASS-fp32 4.26ms, 11.71x).
- **baselines**: CUTLASS INT4/INT1 (GEMM/CONV kernels), CUTLASS FP32/FP16/
  INT8 (full networks), a from-scratch BNN baseline citing TCBNN
  (`journals/tpds/LiS21`, also in this track) for state-of-the-art
  1-bit comparison.
- **source**: `github.com/BoyuanFeng/APNN-TC` README.md (arXiv PDF
  extraction returned binary/FlateDecode streams, not usable text).

## 13. QIGen (Pegolotti et al., CGO'26) — repo README (CPU, not GPU)

- **workloads/inputs**: LLaMA and OPT, weights quantized via GPTQ
  (`IST-DASLab/gptq`); the only **CPU** kernel-generation paper surveyed —
  x86 with AVX2, generic quantization group size, 2/3/4-bit weights.
- **precision & correctness**: nonuniform quantization (matrix split into
  regions quantized at different bit-widths). README lists no benchmark
  numbers or timing-script contents directly (usage documented via a
  `demo.ipynb` notebook and a `swap_modules_llama(...)` API, not a
  standalone `bench.py`); correctness/accuracy handling deferred entirely
  to the upstream GPTQ quantizer.
- **hardware**: x86 CPU only, no GPU — a distinct hardware axis from every
  other paper in this survey (all NVIDIA GPU except this one).
- **source**: `github.com/IST-DASLab/QIGen` README.md.

---

## Divergences

- **Scope: kernel-only vs. whole-model decode.** MARLIN, Tilus, MXBLAS,
  APNN-TC, QGTC, ZipServ (`test_mm`) all isolate the GEMM/GEMM-like kernel.
  **MixQ's headline throughput number (`benchflops.py`) is a full
  autoregressive decode loop through the entire model** (tokenizer →
  transformer forward → sampling), not a GEMM microbenchmark — its
  "speedup" conflates kernel gains with attention/KV-cache/framework
  overhead. The spec must keep these as separate variants and never let a
  paper's "kernel speedup" number be compared against another paper's
  "decode tokens/s" number.
- **Timer and statistic disagree paper-to-paper.** MARLIN: host `time.time()`
  with sync only once before the whole timed loop (mean); ZipServ:
  `cudaEvent` around a summed loop, divided by count (mean); Tilus:
  `cudaEvent`, 50 reps, **median**, **L2 explicitly cleared** each rep;
  MXBLAS: `DurationTimer`, 30 warmup / 100 iters, mean, **L2 explicitly NOT
  flushed** (`flush_l2=False`). The L2-flush divergence is a real fairness
  gap: MXBLAS's own weight tile can stay resident in L2 across 100
  back-to-back identical-shape calls, overstating throughput relative to a
  real server interleaving many different requests. The spec adopts
  Tilus's flush-before-each-rep practice.
- **No shared GEMM shape suite exists.** Every paper derives shapes from a
  *different* model roster: MARLIN → LLaMA-1/Falcon-180B; MXBLAS →
  GPT-Neo/GPT-J/GPT-NeoX/PaLM/T5 family (deliberately *not* Llama, to avoid
  Llama-bias); Tilus → Gemma-2/Qwen2.5/Llama-3.3; FP6-LLM → LLaMA/OPT; MixQ
  → Llama-2/3, Falcon, ChatGLM, Qwen2; QGTC/APNN-TC → synthetic square/
  small-N shapes unrelated to any named model. No two papers benchmark the
  identical `(M,N,K)` list. The spec's shape suite below is a deliberate
  union across ≥4 model families specifically to close this gap.
- **Batch-size sweeps differ in range and hide the memory→compute-bound
  crossover.** MARLIN's own README-reproduction default is a single
  point (batch=16); its "ALL" sweep (1-2048) is off by default. FP6-LLM
  reports only {8,16,32}. FP6-LLM's own numbers show why the full range
  matters: at batch 32 FP6-LLM is *slower* than TensorRT's coarse-grained
  W4A16 (0.94x) — a result invisible if only batch=8/16 were quoted. MXBLAS
  and QFactory (Fig. 9, Table 3) are the only two that treat batch/shape
  scaling as first-class ablations rather than a single headline number.
  The spec makes both a small-batch and a large-batch variant mandatory
  specifically to prevent single-batch cherry-picking.
- **Correctness tolerance is stated by exactly one paper's artifact
  (MARLIN: `mean|Δ|/mean|ref| < 1e-3`, verified in `test.py`), and only
  loosely corroborated by another (MXBLAS's *example* output "difference
  rate: 0.0715%", not asserted as a threshold anywhere in code read).
  Magicube, APNN-TC, QGTC, Quantix, and ZipServ's kernel-level test either
  don't state a numeric tolerance in the README/driver script, or (ZipServ)
  compute a reference but don't show the assertion threshold. The spec
  adopts MARLIN's 1e-3 relative-L1 bound as the default gate for weight-only
  formats and MXBLAS's cosine-similarity `calc_diff` for FP8/MX formats
  where absolute/relative error is known to behave poorly near zero.
- **Model-level (perplexity/task) accuracy is reported by only 2 of 13
  surveyed papers as a first-class, separately-scripted artifact** (FP6-LLM:
  Table 1/2 zero-shot + perplexity; MixQ: `evalppl.py`/`mmlu.sh`). MARLIN,
  Tilus, MXBLAS, QFactory, ZipServ, Magicube, QGTC, APNN-TC report **only**
  kernel/end-to-end **speed**, treating "does the quantization scheme still
  work" as someone else's (the underlying GPTQ/AWQ/etc. quantizer's)
  problem. Because the task instructions explicitly require accuracy
  alongside speed, and because a fused-dequant kernel that computes the
  *wrong* dequantization is fast but useless, the spec makes a model-level
  accuracy report a **mandatory companion measurement** for every speed
  variant below, not an optional extra.
- **Two structurally different quantized-GEMM families are conflated in
  this track's paper list.** (a) Weight-only, group-scaled dequantization
  fused into a GEMM (MARLIN, Tilus, FP6-LLM, QFactory, Quantix, QIGen,
  ZipServ) — the dominant, LLM-serving-relevant pattern. (b) Bit-serial
  arbitrary-precision emulation via INT1 primitives and bit-decomposition
  (APNN-TC, QGTC) — an older, more academic pattern using **synthetic
  fixed shapes unrelated to any real LLM**, and MixQ/MXBLAS's **both-operand**
  quantization (W8A8 / MX-FP8), a third, distinct arithmetic regime. The
  spec below focuses its mandatory variants on (a) (matches the track's
  stated "LLM-shaped GEMM" framing) and (c) (W8A8/MX), and keeps (b) as an
  explicitly out-of-scope note rather than forcing it into an LLM-shaped
  variant it was never designed for.
- **NITI is a training paper, not inference**, with 8-bit-throughout
  forward+backward+update GEMMs on small vision CNNs. Its problem shape
  (needs a full training loop, gradient GEMMs, per-layer block-scaling
  distinct from weight quantization) doesn't fit this track's inference
  GEMM variants; it is excluded from the variant design and left as an
  open question (a possible future "quantized-training-GEMM" track).
