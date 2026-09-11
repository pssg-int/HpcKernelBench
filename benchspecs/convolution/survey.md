# Convolution track — evaluation-methodology survey

Track input: `data/track_inputs/convolution.json`, 9 papers. All 9 are CNN-inference
forward-convolution work (direct/im2col-GEMM, sparse-pruned, quantized/bit-precision,
FPGA-systolic, and vectorization-generator papers that use conv as a target kernel).
**None of the 9 target 3D/stencil-style scientific convolution** (no PDE/FFT-conv/turbulence
paper is in this input list, despite the track's general framing mentioning "scientific
conv" — see Divergences). 7 of 9 were surveyed via artifact repo code (build/run/timing
scripts read directly); 2 (Hidet, VeGen) only via abstract because their current repo HEAD
is a post-paper product release with the original evaluation scripts no longer at the root.

---

## 1. Ansor-AF-DS (ICS'24) — conf/ics/LiXSS24

Source: `README.md` and `benchmarks/run_tests_times_conv.sh` in
github.com/HPCRL/Ansor-AF-DS (fetched via `gh api`).

- **Workloads**: conv2d (2D CNN conv, TVM/Ansor task form) + matmul. Networks: YOLO and
  ResNet (TVM's built-in network task sets), swept over a `pz_num` problem-size index
  (`-1`/empty = all problem sizes in the network).
- **Hardware**: NVIDIA RTX 3090 and RTX 4090 (two hard-coded SM-count/shared-mem configs:
  `128 SM, 48KB` used for both in the shown commands — script also takes SM count and
  shared-mem size as explicit parameters, i.e. hardware is parameterized, not hidden).
  Baseline TVM v0.9 (Ansor).
- **What's actually measured — two distinct axes**:
  1. **Kernel throughput** of the found schedule (their contribution matches, not beats,
     plain Ansor's found-code quality — see `cudnn-ansor3090.py`/`cudnn-ansor4090.py`
     scripts that plot cuDNN vs. Ansor GFLOP/s).
  2. **Autotuning search cost**: their whole contribution is reaching the *same* code
     quality Ansor reaches after 1000 trials, but with 5 "start points" × 64 "initial
     states" and a 0.6 threshold — an order-of-magnitude fewer compile-execute trials.
     `cal_var/` explicitly computes tuning-time **variability** ("in 2 minutes" vs.
     "after 1000 trials").
- **Timing protocol**: not fully recoverable from the README alone (the actual measured
  micro-timing is inside the forked TVM `time_evaluator` C++/Python runtime, not read in
  full — flagged as an evidence gap). TVM's standard practice (used by this and the CGO
  quantized-kernel paper below) is device-side timed repeats via `time_evaluator(number=,
  repeat=)`, i.e., an outer repeat of an inner batched-number loop, statistic = min or
  mean of per-repeat times.
- **Baseline**: cuDNN (named scripts `cudnn-ansor3090.py`, `cudnn-ansor4090.py`), plain
  Ansor, ablations Ansor-AF (analytical-flops-only) and Ansor-DS (design-space-only).
- **Precision**: not stated in README; TVM/Ansor conv2d tasks default to fp32 unless a
  dtype is specified — not confirmed from the repo excerpt read.
- **Divergence signal**: this paper's primary "benchmark" axis is **time-to-reach a target
  kernel quality**, not the kernel's own steady-state GFLOP/s (which is designed to match
  vanilla Ansor). This directly motivates a separate "tuning-cost" variant in the spec.

## 2. Tetris (PPoPP'24) — conf/ppopp/LiuZYLQ24

Source: `Readme.md`, `evalution/conv2d_timing.py`, `evalution/fig6.sh`,
`Tetris/cudnn_test.cc`, `Tetris/cuda_utils.cc`, `Tetris/tensor_utils.h` in
github.com/XG-zheng/Tetris-artifact-evalution (fetched via `gh api`, including the
inlined `Tetris/` kernel-source submodule).

- **Workloads**: pruned/sparse 2D conv layers extracted from **VGG19, ResNet50, YOLOv3,
  MobileNet** checkpoints at multiple sparsity levels (weight-file names encode e.g.
  `vgg19-92-acc-71.7` → 92% pruned, 71.7% retained accuracy). Batch size ∈ {1, 4, 8, 16}
  (`conv2d_timing.py: batch_list = [1, 4, 8, 16]`).
- **Hardware**: single V100-PCIE-32GB.
- **Precision**: fp32 (`CUDNN_DATA_FLOAT` throughout `cuda_utils.cc`).
- **Timing protocol (exact, from `tensor_utils.h`)**: `#define WARM 30`, `#define REPEAT
  100`. Pattern in `cudnn_test.cc`/`cuda_utils.cc`: 30 untimed warmup calls +
  `cudaDeviceSynchronize()`, then a single `cudaEventRecord`/loop-of-100/`cudaEventRecord`
  bracket, `elapsedTime / REPEAT` reported as **the mean per-call time** (not median, not
  min — a single aggregate CUDA-event bracket around all 100 reps, so no per-rep
  distribution is even recorded). Same WARM/REPEAT applied identically to all baselines
  (cuDNN, sputnik, cuSPARSE, Tetris) for a fair walltime comparison.
- **Baselines, all timed with the identical harness**: cuDNN (`CUDNN_CONVOLUTION_FWD_ALGO_
  IMPLICIT_GEMM`, auto-selected via `cudnnFindConvolutionForwardAlgorithm`), sputnik
  sparse-GEMM **with** and **without** im2col, cuSPARSE, and TACO-UCF (Unified Convolution
  Framework, a separate published sparse-conv baseline, driven by `run_ucf.sh`).
- **Metric**: primary = ms per call (`elapsedTime/REPEAT`), secondary = "FLOPS" printed as
  `...T` (TFLOP/s), formula `2 × OH × OW × N × OC × IC × K²`. **Important accounting
  detail**: `out_channel *= (1 - sparsity)` is applied to *every* baseline including dense
  cuDNN before this formula is evaluated — i.e. the FLOP count (and hence GFLOP/s) is
  computed on the *post-pruning* channel count for all methods equally, so the comparison
  is fair among these baselines, but it is a **structured-pruning FLOP definition**, not an
  unstructured-nonzero-count definition. A method exploiting unstructured sparsity within
  the surviving channels would report the *same* nominal GFLOP/s as one that doesn't,
  understating its algorithmic advantage — the spec must define both a
  structured/dense-equivalent FLOP count and an nnz-based effective-FLOP count.
- **Correctness**: `cudnn_test.cc` computes a CPU reference `HostConv2d` and calls
  `TensorEqual(h_output, h_check, ...)`, printing `"Cudnn Pass."`/`"Cudnn Error."` — a
  binary pass/fail gate (exact tolerance value not shown in the excerpt read, but the
  pattern — CPU double/float reference, boolean gate before timing counts — is exactly the
  fairness principle the instructions ask for).

## 3. Analytical CNN loop-tiling DSE (ASPLOS'21) — conf/asplos/0033XSRS21 (arXiv 2101.09808)

Source: `README.md` and `TileLoopGenerator/testbed.cpp`, `resnet.txt`, `yolo.txt`,
`deepwise.txt` in github.com/HPCRL/ASPLOS_artifact (fetched via `gh api`; arXiv
abs/pdf fetch via WebFetch returned only metadata/binary, no usable fulltext — repo code
was the effective source).

- **Workloads — exact layer-shape lists, all batch=1** (format `N Cout Hout Wout Cin Kh Kw
  Sh Sw`):
  - `resnet.txt`: 11 distinct ResNet-18-style conv shapes (56×56 through 7×7 spatial,
    64→512 channels, 1×1 and 3×3 kernels, stride 1/2; the initial 7×7 stem layer is not
    included in this file's list).
  - `yolo.txt`: 11 Darknet/YOLO-style shapes (544×544 down to 17×17, 32→1024 channels,
    plus one 1×1 "detection head" shape with 28272 output channels).
  - `deepwise.txt`: 9 MobileNet-style depthwise-separable 3×3 shapes (112×112 down to
    7×7, 32→1024 channels, all groups=channels i.e. depthwise).
- **Hardware**: CPU only. Primary results on 1-socket 8-core Intel i7-9700K (Coffee Lake,
  AVX2); AVX512 numbers separately on 1-socket 18-core i9-10980XE (Cascade Lake); an
  additional 2-socket 2×14-core Xeon E5-2680v2 (Broadwell) mentioned as tested. README
  explicitly warns the bandwidth/capacity model constants are **hard-coded per CPU** and
  must be edited to reproduce on other hardware.
- **Timing protocol (from `testbed.cpp`)**: **50 iterations** (`totiter = 50`); before
  *every single iteration* a `flushsz = 100,000,000`-float dirty-buffer is written and
  reduced (explicit cache flush — deliberately prevents cross-iteration cache warmth);
  timer = `omp_get_wtime()` (host wall clock) around one call of the generated kernel;
  statistic reported = **arithmetic mean of the 50 GFLOPS samples** (`avg flops` in the
  script's stdout, which the README instructs users to `grep -irn "avg"` for). No
  min/max/median reported. GFLOPS formula: `2 × Cout × Cin × Kh × Kw × Hout × Wout / time`.
- **Baselines**: oneDNN (pinned commit `ad368f484`), AutoTVM-tuned TVM v0.6 (both 1-thread
  and 18-thread/AVX512 variants, separate tuning + timing scripts), and their own generated
  micro-kernels (avx2 6×16, avx512 7×32/10×32).
- **Precision**: fp32 (no quantization; separate from the CGO/TCBNN low-precision papers).
- **Correctness**: not shown in the timed `testbed.cpp` path itself; a separate
  `validation_all/` directory (hardware-counter and GFLOP validation via LIKWID) exists but
  its numerical-correctness tolerance wasn't confirmed from the files read — flagged open.
- **Notable fairness practice**: the mandatory per-iteration cache flush is *more*
  rigorous than most papers in this survey (it prevents an inflated "everything fits in
  L2/L3 already" number) — the spec adopts this as the CPU-variant default.

## 4. DSXplore (IPDPS'21) — conf/ipps/WangFD21 (arXiv 2101.00745)

Source: `README.md`, `main.py`, `run-bench.py` in github.com/YukeWang96/DSXplore_IPDPS21
(fetched via `gh api`; arXiv abs/html fetch returned only metadata, no fulltext body).

- **Workloads**: sliding-channel convolution (SCC), a custom grouped-conv variant, as a
  drop-in CUDA op (`SCC_conv/`) inside standard CIFAR-10 models: VGG11/13/16/19,
  ResNet18/34/50, MobileNet. Sweep parameters: `groups ∈ {1,2,4,8}`, `overlap_ratio ∈
  {0.25, 0.33, 0.50, 0.75}`.
- **Hardware**: NVIDIA GPU (`nvcc 11.1` required); specific card not stated in the README
  excerpt read.
- **Critical harness finding**: `main.py`, the repo's actual benchmark entry point, is a
  **full 350-epoch PyTorch training loop** (data loading, forward+backward, SGD step,
  checkpointing) over CIFAR-10 — it prints per-batch loss/accuracy via a progress bar, not
  a kernel-time measurement. `run-bench.py` just sweeps `{model, groups, overlap}` configs
  through this same training script. **There is no isolated forward-conv-only timing
  harness in the repo** — the paper's claimed kernel-level speedups are not reproducible
  from this artifact without extracting the `SCC_conv` op and writing a separate
  microbenchmark. This is recorded as an open gap / fairness risk, not assumed resolved.
- **Precision/metric/baselines/timing-protocol**: not recoverable from the artifact's own
  benchmark script (see above); the arXiv abstract page did not surface the fulltext
  Evaluation section text through WebFetch (metadata-only response both for `/abs/` and
  `/html/`).

## 5. TC-BNN / Bit-Tensor-Core BNN (TPDS'21) — journals/tpds/LiS21 (arXiv 2006.16578)

Source: `README.md`, `Makefile`, file listing in github.com/pnnl/TCBNN (fetched via `gh
api`; arXiv abs/html fetch returned only metadata).

- **Workloads**: 1-bit (XNOR + popcount) bit-tensor-core kernels for both **bit-GEMM**
  (`bmm/`) and **bit-convolution** (`bconv/`), exercised through full network `.cu`
  drivers: `mnist_mlp.cu`, `cifar10_vgg.cu`, `cifar10_resnet.cu`, `imagenet_vgg.cu`,
  `imagenet_resnet.cu`, `alexnet.cu` (ImageNet). Also multi-GPU scale-up/out variants
  (`benn_scaleup.cu`/`benn_scaleout.cu`, NCCL/MPI) — out of scope for a single-device
  kernel benchmark.
- **Hardware**: NVIDIA Turing GPU, compute capability 7.5 required (`-arch=sm_75` in
  Makefile) — e.g. RTX 2080Ti/Titan RTX class; exact card in the paper's evaluation not
  confirmed from the repo files read.
- **Metric**: headline number from the abstract/README is **end-to-end throughput,
  "5.6K images/second" for ResNet-18 on ImageNet**, claimed 77% faster than prior
  state-of-the-art bit-serial kernels (their own earlier SBTC/SBNN, SC'19). This is a
  whole-network images/sec number, not an isolated conv-kernel GFLOP/s — consistent with
  how this paper should be treated as an **end-to-end** rather than **kernel-only** data
  point in the spec.
- **Precision**: 1-bit weights/activations (XNOR-popcount), contrasted against fp32/other
  software baselines (not itemized beyond "existing software solutions without
  tensorcores" in the abstract).
- **Timing protocol / correctness**: not recoverable from the README/Makefile alone; the
  actual timing loop lives in `kernel.cuh` (36KB) and the per-network `.cu` drivers, which
  were not fully read — flagged open.

## 6. Quantized ML kernel codegen (CGO'20) — conf/cgo/CowanMCBC20

Source: `README.md`, `convolutions.sh`, `end_to_end.sh` in
github.com/cowanmeg/cgo-artifact-2020 (fetched via `gh api`).

- **Workloads**: **layer-by-layer ResNet-18** convolutions (ImageNet-shaped), at
  weight-bits × activation-bits precisions **1×1, 1×2, 2×2** (plus ablation-only points
  2×1, 3×1, 1×3, restricted to "first conv layer only" for the ablations). Also a
  **single-thread** variant of the 1×2 (A2W1) config. Also fp32 floating-point baseline
  layers (`convolutions_fp.py`).
- **Hardware**: ARM CPU — Raspberry Pi 3B+ (Cortex-A53), driven from an x86 host via TVM's
  RPC server (`rpc.sh`) for cross-compiled micro-kernel synthesis.
- **Timing protocol**: `repeats=10` (`convolutions.sh: repeats=10`); statistic = **mean +
  standard deviation**, both explicitly written out — `end_to_end.sh` writes a CSV with
  header `arch,type,kernel,avg-ms,std-dev-ms`. This is the **only paper in this track's
  survey that discloses a variance/spread statistic alongside the mean** in its own
  artifact, a good precedent the spec adopts (spec requires median+min/max, a stronger
  version of the same instinct).
- **Two measurement scopes, both provided as separate scripts** — this paper's own
  artifact already implements the kernel-only vs. end-to-end split the instructions ask
  for: `convolutions.sh` = **per-layer kernel-only** timing; `end_to_end.sh` = **whole
  ResNet-18 graph-runtime** timing, further split into "with microkernel" vs. `--nokernel`
  ablation (i.e., generated-quantized-microkernel vs. TVM's generic fallback), which is
  effectively their own preprocessing/codegen-benefit ablation.
- **Baselines**: hand-written PyTorch quantized conv on-device (`convolutions_pytorch.sh`)
  and TVM fp32 (`convolutions_fp.py`), compared against their auto-synthesized ultra-low-
  bit micro-kernels (`synthesize.sh`, a one-shot ARM micro-kernel synthesis step that is
  explicitly separated from the timed benchmark scripts — i.e. this paper already treats
  synthesis/codegen time as a reported-but-not-amortized-into-kernel-time preprocessing
  cost, matching the instructions' fairness principle).
- **Correctness**: not itemized in the scripts read (README doesn't mention a numerical
  tolerance or reference check) — flagged open; quantized-kernel correctness is inherently
  about matching a *reference quantized* computation, not an fp32 reference (see spec).

## 7. Odyssey / systolic-array DSE (DAC'23) — conf/dac/BasalamaWC23

Source: `README.md`, `tests/conv.json`, `tests/conv_all.json`, `tests/mm.json` in
github.com/UCLA-VAST/Odyssey (fetched via `gh api`).

- **Workloads**: representative conv layers sampled from **VGG16, MobileNetV2, ResNet50,
  ResNet152** (4 layers each in `conv.json`/`conv_all.json`, format `{i: Cin, o: Cout, r:
  Hout, c: Wout, p: Kh, q: Kw}`), plus a synthetic 128×128×128 matmul test case.
- **Hardware / measurement paradigm — categorically different from every other paper in
  this survey**: Odyssey is a **design-space-exploration tool for HLS systolic-array
  designs** (built on AutoSA), driven by a hybrid genetic/mathematical-programming search
  (`ampl`/`IPOPT` solver) plus a padding-based search algorithm. What is "measured" is an
  **HLS-estimated cycle count / latency / DSP-BRAM resource model** from AutoSA's cost
  model during design-space search, not a directly profiled wall-clock kernel run — the
  abstract does claim validation "with FPGA implementations" but the artifact's own
  `run_tests.py` entry point is the DSE search, not a bitstream-execution harness.
  **This paper cannot be folded into the same GFLOP/s-based kernel-only variant as the
  GPU/CPU papers** without conflating a synthesis-model estimate with a measured number —
  recorded as an open question / scope note rather than resolved.
- **Precision/timing-protocol/correctness**: not applicable in the same sense (HLS design
  search, not repeated kernel timing); out of scope for the spec's GFLOP/s variants,
  flagged explicitly in Divergences below.

## 8. Hidet (ASPLOS'23) — conf/asplos/DingYZLWP23 — abstract-level only

Source: abstract + current `README.md` at github.com/hidet-org/hidet. The current repo
HEAD is the post-paper production compiler (`pip install hidet`, PyTorch-2.x `torch.compile`
backend); the original ASPLOS'23 artifact-evaluation benchmark scripts are not present at
the repo root (likely in an older tag/branch not checked, or a separate eval-only repo) —
not pursued further given time budget and because 7 other papers already gave deep
code-level evidence. Known from the abstract only: task-mapping programming paradigm,
targets NVIDIA GPU + Google TPU tensor programs including conv, claims rivaling
hand-optimized libraries (cuDNN/cuBLAS-class) on inference latency. No timing-protocol,
precision, or exact workload detail confirmed independently for this survey.

## 9. VeGen (ASPLOS'21) — conf/asplos/ChenMCA21 — abstract-level only

Source: abstract only. VeGen is a vectorizer *generator* for non-SIMD ISAs (e.g. VNNI-style
multiply-accumulate instructions), evaluated on "DSP/image/ML kernels" including
convolution as one of several target kernels — convolution is not its primary or sole
subject the way it is for papers 1–7. Treated as **peripheral evidence** for this track
(cited for completeness since it's in the input list) rather than a primary source for the
spec's protocol design; not surveyed further.

---

## Divergences

- **Scope mismatch with the general track description**: the track brief mentions
  "scientific and sparse conv," but the actual 9-paper input is 100% CNN-inference
  convolution (image/vision workloads) — 0 papers target 3D/stencil/PDE-style scientific
  convolution. The spec below is grounded in what these 9 papers actually do (CNN conv:
  direct/im2col, sparse-pruned, quantized, bit-precision, FPGA-systolic, vectorized), and
  explicitly flags the absence of a scientific-conv variant as an open question rather than
  inventing one unsupported by any surveyed paper.
- **FLOP accounting under sparsity/pruning** (Tetris): all baselines are compared using a
  FLOP count computed on the *post-pruning* (structured) channel count, not an
  unstructured-nonzero count — fair among the baselines actually compared, but not the
  only legitimate FLOP definition. The spec requires **both** a dense-equivalent
  (algorithm-invariant) count and an nnz/effective count, reported separately with the
  algorithm class disclosed (direct/im2col-GEMM/sparse-pruned/quantized/1-bit), addressing
  the instructions' Winograd-style concern even though no literal Winograd paper is in this
  set.
- **What counts as "the benchmark" differs by paper**: Ansor-AF-DS's real contribution is
  autotuning **search time** to reach a fixed kernel quality (not steady-state GFLOP/s,
  which is designed to match plain Ansor); the ASPLOS analytical-model paper's contribution
  is *avoiding* search entirely (closed-form tiling); Tetris, cgo, and TCBNN all report
  steady-state per-call throughput with no search-time component disclosed as part of the
  benchmark. The spec splits these into a separate **tuning-cost** variant (V4) rather than
  blending search time into per-call GFLOP/s for any paper.
- **Kernel-only vs. end-to-end is inconsistently separated across papers**: cgo (CGO'20)
  is the only paper whose own artifact cleanly splits kernel-only (`convolutions.sh`) from
  end-to-end (`end_to_end.sh`, with a `--nokernel` ablation); Tetris and ASPLOS are
  kernel-only by construction (single-layer timing harnesses); TCBNN's headline metric
  ("images/sec") is inherently whole-network; **DSXplore's own artifact has no kernel-only
  timing path at all** — its `main.py` is a full training loop, so the paper's claimed
  kernel speedup is not independently reproducible from the repo as shipped. The spec's
  kernel-only variant (V1/V2/V3) is the common denominator; the end-to-end note documents
  where each paper actually sits.
- **Statistic reported**: every paper that discloses a statistic reports a **mean**
  (Tetris: `elapsedTime/REPEAT` over one aggregate CUDA-event bracket; ASPLOS: mean of 50
  wall-clock samples; cgo: mean + std-dev over 10 reps) — **none report median or
  min/max**. Per the instructions' fairness principle ("prefer median + report
  min/max"), the spec upgrades the protocol to require per-repetition timestamps so
  median/min/max can be computed, rather than reproducing the papers' single-aggregate-mean
  practice.
- **Cache state before timing (CPU)**: ASPLOS explicitly flushes a 100M-float buffer
  before *every* timed iteration (deliberately cold-cache-ish); no other CPU-relevant
  paper in this set states its cache-state policy. The spec adopts explicit cache flush
  as the CPU-variant default and flags it as a fairness upgrade over papers that stay
  silent on this.
- **Correctness tolerance**: only Tetris (binary Pass/Error vs. a CPU reference conv,
  exact epsilon not confirmed from the code excerpt read) and, implicitly, cgo (which
  presumably validates its quantized kernels against a reference quantized computation,
  not stated explicitly) address correctness in the surveyed material; ASPLOS, DSXplore,
  TCBNN, Odyssey, Ansor-AF-DS, Hidet, and VeGen do not state a numerical tolerance in the
  files/sections read. The spec fixes an explicit tolerance per precision class (open
  question: exact epsilon Tetris uses in `TensorEqual` was not confirmed).
