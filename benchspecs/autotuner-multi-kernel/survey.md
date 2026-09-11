# autotuner-multi-kernel track — evaluation methodology survey

Track input: `data/track_inputs/autotuner-multi-kernel.json`, 11 papers. All 11
are addressed below; 9 via arXiv fulltext or artifact-repo code read with
`gh api` (well above the 5-paper minimum), 2 (Ivanov/Polygeist, Exo 2) only at
abstract/secondary-summary depth because their fulltext could not be fetched
within this survey's tool budget (ACM paywall on the DOI page, and the Exo 2
arXiv HTML mirror 404'd on every version tried) — flagged individually and in
Open Questions.

No paper in the input JSON had an `arxiv` field populated; every arXiv id
below was found by WebSearch and cross-checked against the DOI/venue.

---

## 1. López, Karlsson, Bientinesi — "Compilation of Generalized Matrix Chains
   with Symbolic Sizes" (CGO 2026)

- key: `conf/cgo/LopezKB26`, arXiv:2511.20198 (fulltext via arxiv.org/html)
- **workloads**: synthetic — 10³ randomly sampled matrix-chain "shapes" per
  experiment, each a chain of matrices carrying features (general, symmetric,
  positive-definite, triangular) optionally transposed/inverted. Kernels
  invoked by the generated dispatch code are drawn from BLAS (gemm, trsm,
  trmm, symm) and LAPACK (getrf-based solves, gesv). Not a named/fixed input
  set — the "benchmark" is the distribution of symbolic-size expressions
  itself, not a curated matrix corpus.
- **timing protocol**: "repeated ten times and summarized by the median."
  Warmup not mentioned. Timer not named (implied host-side wall clock).
- **timing scope**: kernel-execution time of the generated call sequence only;
  code-generation/dispatch-selection time is NOT reported as a separate
  benchmarked quantity (the multi-versioning dispatch overhead is folded into
  the measured call, not isolated).
- **precision**: not stated explicitly; BLAS/LAPACK convention implies fp64.
- **correctness**: not discussed in the accessible evaluation section.
- **metric**: execution-time ratio vs. the FLOP-optimal sequence, FLOP-count
  ratios, empirical CDFs, speedup vs. Armadillo (avg. 2.30-2.34x).
- **baselines**: Armadillo 14.6.1 (backed by OpenBLAS 0.3.27), plus an
  in-house naive left-to-right evaluation order.
- **hardware**: Intel Xeon Gold 6132 @ 2.60GHz, 192GB RAM, 14 physical cores
  pinned; multithreaded via OpenBLAS.
- source: arXiv fulltext (arxiv.org/html/2511.20198).

## 2. Ikarashi, Qian, Droubi, Reinking, Bernstein, Ragan-Kelley — "Exo 2:
   Growing a Scheduling Language" (ASPLOS 2025)

- key: `conf/asplos/IkarashiQDRBR25`, arXiv:2411.07211 — **abstract/secondary-
  summary depth only**; the arXiv HTML mirror returned 404 on both the base
  and `v2` URLs, and the `/abs` fetch only surfaced the abstract.
- **workloads (from abstract + secondary sources)**: 80+ hand-scheduled
  kernels built as user-defined scheduling libraries on top of Exo 2's
  extensible primitives; appendix reportedly contains full BLAS-level-1/2 and
  GEMM library code. Domain breadth (DSP-style vs. pure linear algebra) not
  independently confirmed.
- **hardware**: claimed "3 different platforms" — likely x86 (AVX-512-class),
  ARM, and the Gemmini systolic-array accelerator via Chipyard/FireSim FPGA
  simulation (per one secondary source), not independently confirmed from the
  paper itself.
- **timing protocol / correctness / baselines / metric**: NOT recoverable from
  the sources this survey could access — open question.
- source: arXiv abstract page + WebSearch-aggregated secondary summaries only.
  Lowest-confidence entry in this survey; a follow-up pass should re-fetch the
  arXiv PDF directly (`/pdf/2411.07211`) rather than the HTML mirror.

## 3. Qiao et al. — "Pruner: A Draft-then-Verify Exploration Mechanism to
   Accelerate Tensor Program Tuning" (ASPLOS 2025)

- key: `conf/asplos/Qiao0HFZZZCATLY25`, arXiv:2402.02361 (abstract) + GitHub
  README (`qiaolian9/Pruner`, code-confirmed)
- **workloads**: 14 DNN models — CNNs (ResNet50, WideResNet, Inception-V3,
  DenseNet-121, MobileNetV2, DCGAN, DeepLabV3) and Transformers (BERT-Base/
  Tiny, GPT-2, Llama, OPT, Mistral, ViT, DeTR), each in both full- and
  half-precision; plus single-operator microbenchmarks (matmul, conv, random
  shapes) and the TenSet dataset (2,308 subgraphs x ~4,000 schedules each,
  ~16M tensor programs total) for offline cost-model evaluation.
- **hardware**: 3 NVIDIA GPU platforms — A100, Titan V, Jetson Orin-AGX
  (edge), plus Tensor Core half-precision runs.
- **timing protocol**: PyTorch-level operator comparison repeats each
  computation 500x and reports the average (this is dispatch-inclusive,
  Python-interface timing, not a pure-kernel device timer). Search budget is
  explicit: 800 tuning trials for single-operator evaluation, 2,000 trials
  (200 rounds x 10 programs/round) for end-to-end network tuning.
- **tuning time vs. kernel time — BOTH reported, in the same table**: the
  README's results table reports, at matched n-trials=2000 on ResNet-50/A100:
  Ansor 6,691s search / 1.592ms latency; Pruner 5,563s / 1.476ms; MoA-Pruner
  4,978s / 1.457ms; TenSetMLP 5,469s / 1.621ms; Pruner w/ finetuned model
  4,212s / 1.469ms. This is the single clearest example in the track of a
  paper treating search cost and achieved performance as jointly first-class.
- **baselines**: Ansor, TenSetMLP, TLP, MetaSchedule, Felix, Adatune, Roller,
  TLM (search-based compilers); PyTorch 2.2, Triton, Torch-TensorRT, cuBLAS
  (inference frameworks/libraries).
- **correctness**: not explicitly described beyond comparing achieved
  latencies against reference implementations — open question.
- **metric**: latency (ms/us), speedup (e.g. "2.6x vs. Ansor"), search/
  compile time (s or min), top-k/best-k dataset metrics, GPU memory (MB).
- source: arXiv abstract + GitHub README (`raw.githubusercontent.com/
  qiaolian9/Pruner/main/README.md`).

## 4. He, Yoneki — "CuAsmRL: Optimizing GPU SASS Schedules via Deep
   Reinforcement Learning" (CGO 2025)

- key: `conf/cgo/HeY25`, arXiv:2501.08071 (fulltext via arxiv.org/html)
- **workloads**: 6 representative LLM kernels — compute-bound {fused
  feed-forward, matmul+LeakyReLU, batched matmul, flash-attention},
  memory-bound {softmax, RMSNorm}; fp16, "common kernel sizes and
  configurations."
- **hardware**: single NVIDIA A100 80GB PCIe (Ampere).
- **timing protocol**: average of 5 runs; each run uses CUDA events, with
  100 warmup iterations and 100 measured iterations; L2 cache explicitly
  cleared between iterations. The most rigorous, fully-specified protocol
  found anywhere in this track.
- **tuning time vs. kernel time**: reported SEPARATELY, not amortized into
  one number — RL training/search time is framed as "typically less than 5
  hours, a one-time cost," while kernel throughput improvement (up to 26%,
  average 9%) is the headline performance metric.
- **baselines**: Triton-generated kernels at -O3 (ptxas), PyTorch + cuBLAS
  v12.1, CUTLASS v3.5 (GEMM), FlashAttention-2 v2.3.3.
- **correctness**: probabilistic testing — randomized inputs + reference
  outputs generated and compared against the RL-mutated program's output;
  manual verification of individual optimization steps also performed.
- **metric**: normalized kernel throughput (Fig. 6), Nsight Compute hardware
  counters (memory throughput, SM utilization, IPC), RL training statistics
  (episodic reward, policy entropy, KL divergence).
- source: arXiv fulltext (arxiv.org/html/2501.08071v1).

## 5. Liu, Diao, Yang, Chen, Peng, Xu — "Gensor: A Graph-Based Construction
   Tensor Compilation Method for Deep Learning" (IPDPS 2025)

- key: `conf/ipps/LiuDYC0025`, arXiv:2502.11407 (fulltext via arxiv.org/html)
- **workloads**: 32 operator configurations (Conv2d x3, GEMM x3, GEMV x3,
  AvgPool x3, plus additional configs to reach 32 total); DNN models
  Bert-small/ResNet-50/MobileNetV2/GPT-2 on cloud, Bert-small/ResNet-50/
  MobileNetV2 on edge; a dynamic-shape Bert-small evaluation.
- **hardware**: cloud = NVIDIA RTX 4090 + Intel i7-13700K, 24GB, 450W; edge =
  NVIDIA Orin Nano + ARM v8l, 8GB, 15W.
- **timing protocol**: not explicitly specified in the accessible text (no
  warmup/repetition/statistic recoverable) — open question. Repeated
  qualitative claim: "Gensor can generate operator kernels in seconds."
- **tuning time vs. kernel time**: both reported but in SEPARATE figures, not
  fused — Figure 8 compares compilation/optimization duration across
  systems; Figures 6-7 report achieved FLOPS independently. No single joint
  table like Pruner's.
- **baselines**: cuBLAS (handwritten library), Ansor, Roller (tree-based
  construction), PyTorch official implementations, DietCode.
- **correctness**: stated only implicitly ("while ensuring the correctness of
  calculation, outperforms Roller") — no described validation methodology.
- **metric**: FLOPS, compilation time (s), inference speedup, SM occupancy,
  memory bandwidth utilization, L2 cache hit rate.
- source: arXiv fulltext (arxiv.org/html/2502.11407).

## 6. Ivanov, Zinenko, Domke, Endo, Moses — "Retargeting and Respecializing
   GPU Workloads for Performance Portability" (CGO 2024)

- key: `conf/cgo/IvanovZDEM24` — **abstract/secondary-summary depth only**;
  the DOI page requires ACM access and no arXiv preprint was found for this
  paper.
- **workloads (from secondary sources)**: Rodinia benchmark suite, 24 CUDA
  benchmarks total, 9 excluded as targeting outdated CUDA architectures the
  authors judged not worth retuning, leaving 15 benchmarks retargeted+tuned.
- **approach**: extends Polygeist (MLIR-based) into an end-to-end GPU
  compiler that accepts CUDA source, performs optimization, and searches over
  GPU-kernel-granularity choices (thread-block/warp-level tiling) as its
  autotuning axis, targeting both NVIDIA and AMD GPUs.
- **hardware**: cross-vendor — NVIDIA and AMD GPUs (specific models not
  confirmed from accessible sources).
- **timing protocol / correctness / baselines / metric**: NOT recoverable
  from the sources this survey could access within budget; the track input's
  own one-liner claims "27% geomean on Rodinia" as the headline result, but
  the underlying warmup/repetition/statistic/timer choices are unverified —
  open question. A follow-up pass should fetch `c.wsmoses.com/papers/
  polygeist24.pdf` directly (found via WebSearch but not fetched here).

## 7. Georganas, Kalamkar, Voronin, Kundu, Noack, Pabst, Breuer, Heinecke —
   "Harnessing Deep Learning and HPC Kernels via High-Level Loop and Tensor
   Abstractions on CPU Architectures" (IPDPS 2024) [TPP / PARLOOPER]

- key: `conf/ipps/GeorganasKVKNPB24`, arXiv:2304.12576 (fulltext via
  arxiv.org/html, v2)
- **workloads**: standalone kernels — GEMM (multiple shapes), MLP (bias-add +
  ReLU), ResNet-50 convolution layers, block-sparse x dense matmul;
  end-to-end — BERT training/inference, GPT-J, Llama-2, block-sparse BERT
  inference, ResNet-50 training.
- **hardware**: 4 CPU architectures — SPR (Intel Xeon 8480+, 2-socket, 56
  cores/socket, AVX-512+AMX, DDR5-4800), GVT3 (AWS Graviton3, 64 Neoverse-V1
  cores, SVE/MMLA, DDR5-4800), Zen4 (AMD Ryzen 9 7950X, 16 cores, AVX-512,
  DDR5-6000), ADL (Intel i9-12900K, 8P+8E cores, DDR5-5600).
- **timing protocol**: not standardized/stated explicitly for the raw kernel
  measurements — open question. However the paper DOES give an explicit,
  quantified search-time comparison: TVM-Ansor autotuning took 17/18/24/50
  minutes across 4 problem shapes, versus PARLOOPER's declarative-knob search
  taking 2 seconds/9 seconds/2 minutes/22 minutes for the same shapes — the
  most direct "tuning time" numbers found anywhere in this track outside
  Pruner's table.
- **baselines**: oneDNN, ARM Compute Library (AArch64), TVM/Ansor, Mojo
  (GEMM), AMD AOCL/AOCC (Zen4).
- **correctness**: not discussed explicitly in the accessible text — open
  question.
- **metric**: GFLOPS, % of peak, geometric-mean speedup (e.g. 1.98x BF16 on
  SPR vs. oneDNN).
- source: arXiv fulltext (arxiv.org/html/2304.12576v2).

## 8. Xu, Song, Zhou, Li, Hao, Zhao — "A Holistic Approach to Automatic
   Mixed-Precision Code Generation and Tuning for Affine Programs" (PPoPP
   2024) [PrecTuner / LnLAMP]

- key: `conf/ppopp/XuSZLH024` — no arXiv preprint found; ACM DOI PDF returned
  HTTP 403. Methodology reconstructed from the GitHub repo
  (`sheenisme/lnlamp`, code-confirmed) + WebSearch-aggregated result claims.
- **workloads — CONFIRMED PolyBench usage**: the repo embeds
  `polybench_benchmark` as a **git submodule** pointing at
  `github.com/sheenisme/polybench_benchmark` (a PolyBench fork), and the
  README states the tool "embedded the experimental test sub-module
  (polybench_benchmark) in this tool" and that "all the experimental data in
  this article were obtained using this tool." Named kernels appearing in
  the secondary result summary: nussinov, fdtd-2d, heat-3d, seidel-2d
  (dynamic-programming / PDE-stencil kernels), implying the full standard
  PolyBench 4.2.1 ~30-kernel set is the target suite. **This is the only
  paper in the entire track that is directly and unambiguously anchored to
  PolyBench** — the framing implied by this track's naming ("kernel suites
  like Polybench") describes 1 of 11 papers, not the norm.
- **hardware**: Intel i5 CPU + NVIDIA RTX 2070 GPU, Ubuntu 20.04.5 LTS (from
  README); single-core CPU, multi-core CPU, and GPU variants all tested.
- **timing protocol**: not recoverable from accessible excerpts (no warmup/
  repetition counts found) — open question.
- **tuning time vs. kernel time**: PrecTuner's mechanism is architecturally
  different from the search-based tuners above — it initializes one
  parameter `r` via automatic sampling, then SOLVES a performance model
  (closed-form/fitted, via the repo's
  `lnlamp_internal_curve_fitting.py`/`lnlamp_internal_calculate_errors.py`)
  for the best `r`, rather than enumerating/searching a large space. Tuning
  cost is therefore architecturally minimized rather than measured and
  reported as a separate large search-time number; all headline results are
  post-tuning kernel speedups.
- **baselines**: Pluto (polyhedral CPU baseline, single- and multi-core),
  PPCG (GPU baseline — PrecTuner's generated mixed-precision C is literally
  fed as PPCG's input), LuIs (a prior fixed-point mixed-precision competitor;
  excluded from some comparisons because it doesn't support parallel
  execution).
- **correctness**: precision is tuned under an explicit "quality degradation
  budget" — the paper reports that PrecTuner achieves smaller errors than
  LuIs at comparable speedup, implying a quantified per-kernel error metric,
  but the exact metric/tolerance was not confirmed from accessible sources —
  open question.
- **metric**: speedup — 3.28x vs. LuIs; 1.81x (single-core) / 1.52-1.73x
  (multi-core) vs. Pluto; 1.71x vs. PPCG on GPU.
- source: `gh api repos/sheenisme/lnlamp/contents` (confirms
  `polybench_benchmark` submodule) + README raw fetch + WebSearch-aggregated
  abstract/result summary.

## 9. Ye, Lai, Sivasubramaniam, Ceze, Chen, Yu — "SparseTIR: Composable
   Abstractions for Sparse Compilation in Deep Learning" (ASPLOS 2023)

- key: `conf/asplos/YeLSCC23`, arXiv:2207.04606 (fulltext via
  ar5iv.labs.arxiv.org/html)
- **workloads**: sparse operators — SpMM, SDDMM, sparse attention (Longformer
  and Pixelated-Butterfly-Transformer sparsity patterns), sparse convolution,
  RGMS (relational gather-matmul-scatter). Datasets: 7 GNN graphs (Cora,
  CiteSeer, PubMed, PPI, OGB-ArXiv, OGB-Proteins, Reddit), 5 heterogeneous
  graphs for RGCN (AIFB, MUTAG, BGS, OGB-BioKG, AM), SemanticKITTI for sparse
  convolution, PrunedBERT (HuggingFace) for sparse attention. End-to-end:
  GraphSAGE training, RGCN inference, MinkowskiNet.
- **hardware**: NVIDIA RTX 3070 and Tesla V100 (any GPU with >=16GB used for
  the larger workloads).
- **timing protocol**: "discard the samples for the first 10 runs as
  warm-up... and repeat for 100 cycles," measured via CUDAEvent APIs, with
  L2 cache flushing enabled (`FLUSH_L2=ON`) between iterations — one of the
  two most rigorous, fully-specified protocols in this track (with CuAsmRL).
- **tuning time vs. kernel time**: SparseTIR builds a search space over
  composable sparse formats/transformations for autotuning, but the
  accessible excerpt reports only POST-tuning kernel execution time;
  compile/search-cost amortization is argued qualitatively ("the compiled
  operator will be re-used many times") rather than quantified — open
  question.
- **baselines**: cuSPARSE, dgSPARSE (GE-SpMM, DA-SpMM, PRedS), TACO, DGL,
  PyG, Sputnik, Triton, TorchSparse, cuBLAS, Graphiler.
- **correctness**: "computation results... compared with existing
  frameworks/libraries to confirm numerical accuracy" — no explicit
  tolerance given in the accessible text — open question.
- **metric**: speedup (1.20-2.34x GNN ops, 1.05-2.98x sparse attention,
  0.56-7.45x sparse convolution; end-to-end 1.08-1.52x GraphSAGE training,
  4.20-40.18x RGCN inference), GPU memory footprint, cache hit rates, FLOPs.
- source: arXiv fulltext (ar5iv.labs.arxiv.org/html/2207.04606).

## 10. Won, Mendis, Emer, Amarasinghe — "WACO: Learning Workload-Aware
    Co-optimization of the Format and Schedule of a Sparse Tensor Program"
    (ASPLOS 2023)

- key: `conf/asplos/WonMEA23` — PDF fetch returned corrupted/binary content
  from both `charithmendis.com` and `commit.csail.mit.edu` mirrors; ACM DOI
  page did not yield fulltext either. Methodology reconstructed from the
  GitHub repo (`nullplay/Workload-Aware-Co-Optimization`, code-confirmed) +
  WebSearch-aggregated abstract/result claims — **lower confidence on
  protocol details than papers read via fulltext**.
- **workloads**: 4 sparse tensor algorithms — SpMV, SpMM, SDDMM, MTTKRP —
  evaluated, per secondary sources, over 726 distinct sparsity patterns. The
  repo itself ships ~20 example `.csr` matrices directly under `dataset/`
  (named e.g. `Chevron3_4x16_1.csr`, `NACA0015_16x8_9.csr`, `SiO.csr`,
  `bcsstk38.csr`, `cavity24.csr` — SuiteSparse-style provenance) plus a
  `dataset/src/coo_to_csr.hpp` converter, and separate `WACO/{SpMV,SpMM,
  SDDMM,training_data_generator}` code directories — implying the full
  726-pattern training/eval set is generated or downloaded separately, not
  checked into the repo verbatim.
- **hardware**: CPU (specific model not confirmed from accessible sources —
  open question); MKL and ASpT used as CPU baselines.
- **timing protocol / correctness**: not confirmed from primary sources
  within this survey's budget — open question.
- **tuning time vs. kernel time**: WACO's whole contribution is a learned
  cost model + approximate-nearest-neighbor (ANN) search that jointly
  retrieves format+schedule, explicitly positioned as a fast alternative to
  expensive enumerative autotuning — i.e. search/tuning cost IS a central
  design axis of the paper — but exact search-time numbers were not
  extracted from the accessible sources — open question.
- **baselines**: MKL, ASpT (TACO is also implicit, given the
  `code_generator` directory's TACO-lineage naming pattern).
- **metric**: speedup, 1.14-1.43x over MKL/ASpT per the abstract.
- source: `gh api repos/nullplay/Workload-Aware-Co-Optimization/contents`
  (dataset + code layout) + WebSearch-aggregated abstract.

## 11. Zheng, Liang, Wang, Chen, Sun — "FlexTensor: An Automatic Schedule
    Exploration and Optimization Framework for Tensor Computation on
    Heterogeneous System" (ASPLOS 2020)

- key: `conf/asplos/Zheng0WCS20` — no arXiv preprint found; ACM DOI PDF not
  fetchable. Methodology reconstructed from the GitHub repo
  (`pku-liang/FlexTensor`, code-confirmed) + WebSearch-aggregated abstract.
- **workloads (confirmed from repo `flextensor/baselines/` file list)**: a
  broad synthetic DL-operator microbenchmark suite — conv1d/2d/3d, transposed
  conv1d/2d/3d, depthwise conv, dilated conv, grouped conv, shift-conv2d,
  gemm, gemv, block-matrix-circulant, bilinear, unpooling1d/2d, gated-
  PixelCNN, PixelCNN — rather than named real DNN models. A `sparse/` and
  `taco/` baseline subdirectory also exists, implying a (secondary) sparse-op
  comparison arm not detailed in the abstract.
- **hardware**: heterogeneous — GPU (NVIDIA, CUDA), CPU (repo's
  `examples/opt_*_cpu.py` vs. `*_gpu.py` split), and FPGA (Xilinx VU9P, per
  the track's own one-liner). Genuinely the widest hardware-portability claim
  in the track alongside TPP-mlir/Gensor.
- **timing protocol / correctness**: not confirmed from primary sources
  within budget — open question.
- **tuning time vs. kernel time**: FlexTensor combines heuristic and ML
  search methods to explore its schedule space; all abstract-level claims are
  post-tuning speedups (1.83x GPU vs. cuDNN, 1.72x CPU vs. MKL-DNN, 1.5x FPGA
  vs. OpenCL baselines) with no quantified search-time number found in
  accessible sources — open question.
- **baselines (confirmed from repo)**: AutoTVM, cuDNN, MKL, cuBLAS, TACO,
  hand-written CUDA/OpenCL.
- **metric**: speedup vs. the respective per-platform vendor-library
  baseline.
- source: `gh api repos/pku-liang/FlexTensor/contents/flextensor/{baselines,
  examples}` + WebSearch-aggregated abstract.

---

## Divergences

- **Kernel suite composition — PolyBench is the exception, not the rule.**
  Only PrecTuner/LnLAMP (paper 8) is directly and confirmably anchored to the
  PolyBench affine-loop suite. The other 10 papers target DNN-operator-style
  kernels (GEMM/conv/attention: FlexTensor, Pruner, Gensor, TPP-mlir),
  sparse linear algebra (SpMM/SDDMM/SpMV/MTTKRP: SparseTIR, WACO), assembly-
  level scheduling of already-fused LLM kernels (CuAsmRL), matrix chains of
  BLAS/LAPACK calls (GMC), a systems benchmark suite (Rodinia: Ivanov), or a
  broad hand-scheduled kernel library (Exo 2). This track's framing as
  "compilers/autotuners over kernel suites like Polybench" describes 1 of
  11 papers' actual input suite.
- **Search/tuning cost is reported as first-class by only a minority, and in
  incommensurable units.** Pruner reports matched-n-trials wall-clock search
  time in the same table as achieved latency. TPP-mlir/PARLOOPER gives an
  explicit wall-clock comparison against TVM-Ansor's autotuning time.
  CuAsmRL frames RL training as a one-time wall-clock cost, reported
  separately from throughput. Gensor reports compile time and FLOPS in
  separate figures, not jointly. SparseTIR argues amortization qualitatively
  without a number. GMC, PrecTuner, WACO, and FlexTensor do not report a
  separate tuning-cost number at all — for GMC and PrecTuner this is
  architecturally defensible (closed-form/model-solve, not enumerative
  search) but for WACO (whose whole contribution is a fast ANN search) and
  FlexTensor (which explicitly does heuristic+ML search) the omission hides
  a cost their own design claims to have minimized.
- **Timing rigor spans a wide range.** CuAsmRL and SparseTIR converge
  independently on a similar rigorous protocol (10-100 warmup, 100
  repetitions, per-iteration device-side (CUDA-event) timers, L2 cache
  flush between iterations). GMC states only "median of 10" with no warmup.
  Pruner's operator-level numbers come from a 500-iteration PyTorch-level
  mean, which includes Python/framework dispatch overhead, not a pure device
  kernel timer. Gensor, WACO, Ivanov, Exo 2, PrecTuner, and FlexTensor have
  no warmup/repetition/statistic recoverable from the sources this survey
  could access.
- **Correctness gates are inconsistently documented.** CuAsmRL and SparseTIR
  both describe an explicit numeric validation step (probabilistic testing;
  comparison against library/framework results). PrecTuner's whole mechanism
  is built around a quality-degradation error budget. GMC, Gensor, WACO, and
  FlexTensor have no recoverable correctness-check description in the
  material this survey could access.
- **Precision handling differs in kind, not just value.** Pruner, CuAsmRL,
  and TPP-mlir test a FIXED alternate dtype (fp16/BF16) alongside fp32 —
  precision is a discrete axis. PrecTuner instead treats precision as a
  CONTINUOUSLY TUNED parameter under an explicit error budget — precision
  and speed trade off along a frontier, not a fixed-dtype switch. GMC/WACO/
  FlexTensor default to BLAS/LAPACK-convention fp64/fp32 without an explicit
  mixed-precision axis.
- **Hardware scope ranges from single-GPU to explicit portability studies.**
  CuAsmRL evaluates one A100. TPP-mlir evaluates 4 distinct CPU
  microarchitectures explicitly to demonstrate portability. Gensor spans
  cloud (RTX 4090) and edge (Orin Nano). Ivanov and FlexTensor target
  cross-vendor GPU (NVIDIA+AMD) and cross-device (GPU+CPU+FPGA) respectively.
  A benchmark spec that mandated matching every paper's hardware breadth
  would be unreproducible on a single lab machine; see `open_questions`.
