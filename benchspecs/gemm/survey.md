# GEMM track — evaluation-methodology survey

Track input: `data/track_inputs/gemm.json`, 35 papers. This document surveys the
subset whose primary contribution is a dense-GEMM (or GEMM-chain / batched-GEMM)
kernel implementation, since several of the 35 use GEMM only as a building block
for an unrelated application kernel (cellular automata, stencils, EVD, quantum
circuits, SVM, cryo-EM, FFT-fusion — see "Out of scope" below). 9 papers were
surveyed via arXiv fulltext and/or the open-source artifact repo, exceeding the
5-paper minimum, chosen for platform diversity: NVIDIA GPU, AMD GPU, ARM CPU
(x2), FPGA (x2, one via OpenCL), distributed CPU/GPU cluster, and a GPU
GEMM-chain (Kronecker) op.

## 1. `conf/ics/WuZLHJWC23` — Anatomy of High-Performance GEMM with Online Fault Tolerance on GPUs (ICS'23)

Source: arXiv fulltext (`ar5iv.labs.arxiv.org/html/2305.01024`) + repo
`shixun404/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs` (`README.md`).

- **workloads/inputs**: square SGEMM, M=N=K ∈ {1024, 1536, ..., 6144} (step
  512, 11 sizes); irregular sweep with K fixed at 256, M and N independently
  swept 64→490 (step 32), and a second irregular sweep with K fixed at 1024.
  Platforms: NVIDIA Tesla T4 and A100.
- **timing protocol**: "reported performance data are averaged over tens of
  runs to minimize fluctuations" — mean statistic, repetition count not
  pinned to an exact number ("tens"). No warmup explicitly mentioned. Timer
  mechanism not stated in the text (repo README table is reproducible via
  `./ft_sgemm START END STEP START_KERNEL END_KERNEL`).
- **timing scope**: kernel execution only; fault injection happens "at the
  source code level" (software-injected, not a hardware/timing artifact).
- **precision & correctness**: FP32 (SGEMM) only; "we validate the
  correctness of our final computation results by comparing them with
  cuBLAS" — no numeric tolerance given.
- **metric**: GFLOPS.
- **baselines**: cuBLAS (CUDA 11.6); prior non-fused ABFT baseline (Ding et
  al. 2011). Repo README reproduces the full comparison table (cublas vs. 6
  in-house kernel variants vs. 6 ABFT variants) across all 11 square sizes.

## 2. `conf/sc/YangFDSW21` — LibShalom: Optimizing Small and Irregular-shaped Matrix Multiplications on ARMv8 Multi-Cores (SC'21)

Source: paper PDF (`eprints.whiterose.ac.uk/177559/6/sc21.pdf`, extracted via
pypdf — Sections 7–8, "Experimental Setup" / "Evaluation Methodology").

- **workloads/inputs**: small GEMM, M=N=K ∈ [8, 120] (typical of SeisSol,
  Nekbox mini-kernels); irregular GEMM, M or N ∈ [32, 256] (CNN-derived
  shapes); NN and NT layouts primary (TN/TT confirmed similar); matrices
  filled with random floats in [0,1). Platforms: Phytium 2000+, Kunpeng 920
  (KP920), ThunderX2 — all ARMv8, FP32 peak 1126–2662 GFLOPS.
- **timing protocol**: "we run each GEMM kernel 10 times and report the
  geometric mean of the runtime. We show the variations across different
  runs as a min-max bar." LIBXSMM specifically is warmed up first so its JIT
  compile cost is excluded from its own measured time (baseline-specific
  fairness patch, not applied to LibShalom itself since it has no JIT step).
- **timing scope**: two explicit cache states are benchmarked separately —
  **hot cache** (data pre-loaded before the timed loop) and **cold cache**
  (data not resident); for irregular GEMM, K is deliberately set to 5000 "to
  drive the last run data out of the last level data cache" and avoid
  artificially hot-cache numbers. This hot/cold split changes results
  materially (paper shows different rankings between Figs. 7 and 8).
- **precision & correctness**: FP32 primary; FP64 also tested ("throughput is
  roughly half of FP32... across all test methods"). No numeric correctness
  tolerance stated.
- **metric**: GFLOPS.
- **baselines**: OpenBLAS, BLIS, ARMPL, LIBXSMM, BLASFEO (5 ARM-tuned
  libraries). BLASFEO excluded from irregular/multi-threaded experiments
  because it has no multi-threaded backend — an explicit fairness exclusion.

## 3. `conf/sc/WuMZDW0WW24` — autoGEMM: Pushing the Limits of Irregular Matrix Multiplication on Arm Architectures (SC'24)

Source: paper PDF (`osti.gov/servlets/purl/2480030`, extracted via pypdf —
Section "Evaluation Environment and Data" through "Scaling Evaluation").

- **workloads/inputs**: small matrices from (1,1,1) to (128,128,128); irregular
  matrices = ResNet-50 layer-by-layer GEMM shapes (tall-skinny, long-rectangle,
  small). Platforms: KP920, Graviton2, Altra, Apple M2, A64FX (5 Arm chips).
- **timing protocol**: not explicitly stated as warmup/rep counts in the
  extracted text (open question); step-wise ablation methodology (each
  optimization added incrementally and re-measured) is described in detail.
- **timing scope**: **explicitly excludes JIT/code-generation time from the
  measured runtime** — "For fairness, we exclude the time to generate the
  code JIT from the runtime and only the actual computation time was
  considered for LIBXSMM since it uses the JIT method." This is a clean
  precedent for a preprocessing/codegen-excluded steady-state variant.
- **precision & correctness**: precision not explicit in extracted sections
  (inferred FP32; open question). No numeric correctness tolerance found.
- **metric**: GFLOPS, % of hardware peak, and pairwise speedup ratios.
- **baselines**: OpenBLAS, LIBXSMM, LibShalom, Eigen, Fujitsu SSL2. Notably,
  **LibShalom itself could not run on 2 of the 5 test chips** (M2, A64FX)
  because it doesn't compile with clang and lacks SVE support — a real
  cross-platform baseline-availability gap that a fair spec must handle
  (report "N/A", don't silently drop the platform).

## 4. `conf/ipps/LiYTLZ22` — A Fine-grained Prefetching Scheme for DGEMM Kernels on GPU with Auto-tuning Compatibility (IPDPS'22)

Source: repo `FFFLJL/Tensile-FGPS` (`README.md`, embedded Tensile benchmark
YAML).

- **workloads/inputs**: DGEMM (double precision), `ProblemType` declares
  `Batched: True`, `TransposeA/B: True`; sample benchmark config sweeps
  `ThreadTile ∈ {[4,4],[4,6],[6,4]}`, `WorkGroup`, `StaggerU`, etc. with a
  fixed problem `Exact: [16384, 128, 1, 128]` (M, N, batch=1, K in Tensile's
  ordering). Platform: AMD GPU via ROCm/Tensile.
- **timing protocol**: Tensile's own benchmark harness — `KernelTime: True`
  (device-side kernel timer, not host wall-clock), `EnqueuesPerSync: 1`,
  `SyncsPerBenchmark: 2` (i.e., 2 sync points per benchmarked config,
  1 kernel enqueue between syncs in this config). Exact statistic
  (mean/median/min) used by Tensile's library-client reporting is not shown
  in the repo README (open question — Tensile's own docs would clarify).
- **timing scope**: kernel-only (Tensile always measures device kernel time,
  no host-device transfer in the timed region).
- **precision & correctness**: FP64. The sample YAML sets
  `NumElementsToValidate: 0` — **correctness checking is disabled** in this
  particular auto-tuning-search config (validation is presumably re-enabled
  for the final reported numbers, but that isn't confirmed from available
  source — open question).
- **metric**: Tensile reports GFLOPS via its library-client benchmark output
  (not directly visible in README, inferred from Tensile conventions).
- **baselines**: ablation of `FGPS: [True, False]` — the paper's own
  prefetching scheme on/off, integrated into Tensile's auto-tuning search
  (i.e., baseline = Tensile/rocBLAS's existing auto-tuned kernels without
  FGPS).

## 5. `conf/sc/HuangC22` — CA3DMM: A New Algorithm Based on a Unified View of Parallel Matrix Multiplication (SC'22)

Source: repo `scalable-matrix/CA3DMM` (`README.md`, worked example with real
program output).

- **workloads/inputs**: distributed dense GEMM, arbitrary M,N,K (example:
  M=N=K=8000), transA/transB flags, CPU or NVIDIA-GPU device backend, MPI+
  OpenMP hybrid (example: 24 MPI ranks × 24 OpenMP threads on one node,
  process grid 4×2×3).
- **timing protocol**: **`ntest` explicit repetition count is a CLI argument**
  (example uses `ntest=10`); statistic reported is the **average** execution
  time over `ntest` runs (not median/min); per-phase breakdown (A/B/C
  redistribution, allgather, 2D-Cannon local compute, reduce-scatter) is
  printed both per-run and as an average.
- **timing scope**: the tool's own output **already separates "matmul only"
  (local DGEMM compute, 706 ms in the example) from "total execution"
  (802–812 ms, includes redistribution/comm)** — a built-in kernel-vs-e2e
  split that is a direct precedent for this spec's kernel/e2e variant
  design.
- **precision & correctness**: DGEMM (FP64, via MKL). Explicit
  `validation` CLI flag (0/1) triggers a correctness check against a
  reference; example output reports "0 error(s)" (binary pass/fail, no
  numeric tolerance surfaced to the user).
- **metric**: per-rank GFlops (64.40 GFlops/rank in the example) + wall time
  in ms for each phase.
- **baselines**: compared against "state-of-the-art PGEMM implementations"
  (2D/3D ScaLAPACK-family ancestors) per the paper abstract; exact library
  names not in the README (open question — check paper body).

## 6. `conf/sc/MatteisLH20` — fBLAS: streaming linear algebra on FPGA (SC'20)

Source: arXiv fulltext (`ar5iv.labs.arxiv.org/html/1907.07929`).

- **workloads/inputs**: GEMM (and other BLAS routines) on Intel FPGAs (Arria
  10 GX1150, Stratix 10 GX2800); square matrices sized as **5× the design's
  on-chip memory tile size** (an FPGA-specific sizing convention, not a fixed
  numeric list — the actual numbers depend on the synthesized design's tile
  size). FP32 and FP64 both evaluated.
- **timing protocol**: "Performance is reported in floating point operations
  per second (Ops/s) based on the averaged execution time. In all cases the
  99% confidence interval is within 5% of the measured mean." — mean
  statistic with a stated confidence bound, but the exact repetition count
  producing that CI is not given (open question).
- **timing scope**: both individual-module and composed-streaming-kernel
  configurations tested; for memory-bound routines, input is generated
  directly on-chip to isolate compute from host-transfer bandwidth (H2D/D2H
  explicitly excluded in those cases).
- **precision & correctness**: FP32/FP64; no numeric correctness tolerance
  found in the extracted text (open question).
- **metric**: Ops/s, plus a theoretical "expected performance" computed from
  `#DSPs × frequency` at full utilization (used as an efficiency reference,
  not a competing baseline).
- **baselines**: Intel MKL 2019 on a Xeon E5-2630 v4 CPU (including MKL's
  batched routines). No cross-FPGA-vendor comparison (Xilinx explicitly
  deferred to future work).

## 7. `conf/ppopp/JangdaY24` — Fast Kronecker Matrix-Matrix Multiplication on GPUs (PPoPP'24)

Source: repo `abhijangda/fastkron` (`tests/benchmarks/run_benchmarks.py`) +
arXiv abstract page (`arxiv.org/html/2401.10187`, partial — baselines list
only, full methodology came from the repo).

- **workloads/inputs**: Kronecker matmul (`X × (P⊗P⊗...⊗P)` or its transpose
  "kmm"/"mkm" forms), parameterized by `Shape(m, n, p, q, mmtype)`; both
  synthetic microbenchmarks and a "real world dataset" (per arXiv abstract
  page section headers, exact shapes not captured). Platforms: NVIDIA V100 /
  A100, single- and multi-GPU (up to 16 GPUs), plus an x86 CPU backend.
- **timing protocol** (from `GPyTorchEval`/`FastKronEval` classes in
  `run_benchmarks.py`): **1 warmup call of 10 iterations (discarded)**, then
  **4 trials, each averaging 5 iterations**
  (`min(run_case(5), run_case(5), run_case(5), run_case(5))`); the reported
  latency is the **min of the 4 trial-means** (min-of-means statistic).
  Timer is host-side `time.time()`, with `torch.cuda.synchronize()` called
  before stopping the clock for the CUDA backend — i.e., synchronized
  wall-clock, not CUDA events.
- **timing scope**: kernel-only (post-sync host timer around the compute
  call only). Autotuning ("Autotuning Time", Section 6.1 of the paper) is
  reported and evaluated **separately** from steady-state execution
  performance.
- **precision & correctness**: float/double/half all supported
  (`elemtype` parameter); correctness validation code exists in the repo
  test suite (`tests/`, GoogleTest-based) but is decoupled from the
  benchmark harness (i.e., correctness and performance are measured by
  different binaries — a clean separation worth preserving in the spec).
- **metric**: GFLOPS = `shape.flops() / time`, where `flops()` computes the
  exact Kronecker-chain FLOP count (not a plain M·N·K formula, since the
  chain restructures the computation — a genuinely different FLOP-counting
  rule from plain dense GEMM that the spec must call out separately if a
  Kronecker/chained variant were ever added).
- **baselines**: GPyTorch (`operators.KroneckerProductLinearOperator`),
  COGENT, cuTensor, Cyclops Tensor Framework (CTF, at 16-GPU scale), Distal.

## 8. `conf/ipps/DengYFD26` — Demystifying ARM SME to Optimize General Matrix Multiplications / MPGEMM (IPDPS'26)

Source: repo `mpgemm/MPGEMM` (`README.md` + `benchmark/singlePerformance.c`
source).

- **workloads/inputs**: GEMM shapes drawn from **DeepSeek and LLaMA
  linear-layer projections** — concrete shapes found in the benchmark driver:
  `(M,N,K)` ∈ {(64,2112,7168), (64,24576,1536), (64,32768,512),
  (64,7168,16384), (64,4096,7168), (64,7168,2048)} at `nreps=500`, a second
  set at `(128,128,128)`-class with `nreps=100`, and a near-square set at
  `(4096,4096,4096)`-class with `nreps=20` (repetition count is manually
  scaled inversely with matrix size across the commented-out size groups in
  the source). Platform: Apple M4 Pro (ARM SME).
- **timing protocol**: for-loop of `nreps` direct kernel calls; **statistic =
  min over `nreps`** (`best = min(best, end)` every iteration, no separate
  discarded warmup — the first iteration is folded into the min-tracking
  loop, not excluded).
- **timing scope**: kernel-only; `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)`
  is set at the top of `main()` — this particular driver (`singlePerformance.c`)
  measures **single-threaded** performance only; a separate
  `multiPerformance.c` driver exists for multi-core numbers, and a
  dedicated `correct.c` binary performs correctness checking **decoupled
  from the performance binaries** (clean precedent, same pattern as
  FastKron).
- **precision & correctness**: **all of FP64, FP32, FP16, and INT8 are swept
  in the same benchmark run** (`test_fp64_gemm`, `test_fp32_gemm`,
  `test_fp16_gemm`, `test_int8_gemm` all called per shape) — this is a
  genuine multi-precision GEMM study, not a purely-quantized one, so it
  stays in scope for this track (only the INT8 leg would belong to the
  quantized-GEMM track). Correctness is checked by the separate `correct.x`
  binary; tolerance not visible from the README (open question).
- **metric**: `GFLOPS = 2*M*N*K / (1e9 * best_time_seconds)`.
- **baselines**: Apple Accelerate (vendor BLAS), LIBXSMM, KleidiAI, OpenBLAS
  — all SME-enabled implementations. Paper claims 1.23× average speedup over
  Accelerate.

## 9. `journals/tpds/LinXWYLL25` — High Performance OpenCL-Based GEMM Kernel Auto-Tuned by Bayesian Optimization / CL-DB-GEMM (TPDS'25)

Source: repo `lsl036/CL-DB-GEMM` (`README.md` + `scripts/benchmark/settings.py`,
inherited from the upstream CLBlast benchmark harness this artifact is built
on).

- **workloads/inputs**: GEMM configs in `settings.py`: "multiples of 128" and
  "multiples of 129" (the +1 offset specifically stresses tile-boundary /
  padding handling), "around 2048", "small matrices in steps of 16" and "in
  steps of 1" (fine-grained small-GEMM sweep down to M=N=K=1). A
  `GEMMBATCHED` config also exists: `batch_num ∈ {8, 64}` at M=N=K swept
  32→640 (step 32×20), and `batch_num` at powers of 2 from 1 to 4096 with
  M=N=K=128 fixed — i.e., this artifact's own benchmark suite already covers
  **batched GEMM** explicitly. A `GEMMSTRIDEDBATCHED` config exists too.
  Precision: half/single/double, each with its own tuner entry point
  (`RunHKernel.py` / `RunSKernel.py` / `RunDKernel.py`). Target: OpenCL
  devices — GPU and FPGA (paper's stated target hardware class).
- **timing protocol**: `num_runs` per benchmark config, ranging **10–20**
  depending on config (smaller/faster configs get more reps: `num_runs=20`
  for "multiples of 128"/"129"/GEMMBATCHED-8; `num_runs=10` for "around
  2048" and larger GEMMBATCHED sweeps) — repetition count is manually tuned
  per config rather than a single fixed number.
  This entire file is the **upstream CLBlast harness**, not code written
  specifically for this paper's contribution — the paper's own contribution
  (double-buffer kernel + Bayesian-Optimization tuner) plugs into this
  existing benchmark scaffold, so the timer/statistic conventions below are
  CLBlast's, inherited rather than paper-specific (flagged as inferred).
- **timing scope**: OpenCL device-queue kernel time (CLBlast's client
  benchmarking convention uses OpenCL profiling events around the kernel
  enqueue) — inferred from CLBlast norms, not independently confirmed by
  reading the C++ client source in this survey pass (open question).
- **precision & correctness**: half/single/double as three separate sweeps;
  no numeric correctness tolerance found in the surveyed files (open
  question — CLBlast's own test suite, separate from `scripts/benchmark/`,
  likely defines one).
- **metric**: GFLOPS (`y_key: "GFLOPS"` throughout `settings.py`).
- **baselines**: upstream CLBlast (single-buffer, non-tuned) as the direct
  ablation baseline; paper abstract additionally claims tuning-search-time
  reduction from the Bayesian-Optimization tuner vs. CLBlast's original
  exhaustive/heuristic tuner (a preprocessing-cost claim, not a steady-state
  kernel-time claim).

## Out of scope (in `gemm.json` but not surveyed as core GEMM benchmarks)

These use GEMM/Tensor-Cores as a means to accelerate a different primary
kernel; their evaluation methodology is about that application, not about
dense-GEMM shapes/protocol, so they were excluded from the survey proper
(not from the 917-paper corpus — just from this track's benchmark-design
grounding):
`conf/ppopp/ChenLWBWMYZCY24` (ConvStencil, stencil→matmul), `conf/sc/ZhangLYCZCY24`
(LoRAStencil, stencil→matmul), `journals/tpds/NavarroQMFH25` (CAT, cellular
automata→tensor cores), `conf/sc/WangHZSJWDZ25` (2-stage EVD back-transform,
uses BLAS2/3 but the contribution is algorithmic reordering of an eigensolver),
`conf/sc/LiuLLFYSZWPCGHW21` (Sunway quantum-circuit tensor contraction),
`conf/ics/ZhangSW20` (TensorSVM, Gram-matrix approximation), `conf/sc/FuXGMSWSD0Y25`
(T2-RELION, cryo-EM difference kernel), `conf/sc/WuZDZHC25` (TurboFNO,
fused FFT-GEMM-iFFT — GEMM is one fused stage among several), `conf/ipps/YaoZLX26`
(Ozaki-scheme FP16 stencil dynamical core), `conf/sc/LtaiefACRSKDBAK24` (GWAS
kernel ridge regression, mixed-precision Cholesky/GEMM inside a larger
pipeline), `conf/ics/DezfuliC25` (tile-fusion — fuses **sparse×dense**
CSR-GEMM chains (spmm-spmm / gemm-spmm); its repo confirms MKL comparison but
the operator is sparse-dependent, closer to the SpMM track than dense GEMM).
Also noted but not deep-surveyed: `conf/cgo/LopezKB26` (BLAS/LAPACK call-sequence
selection for symbolic-size chains — evaluates existing BLAS kernels'
composition, not a new GEMM kernel) and `conf/cgo/ZhangDSHSP26` (Hexcute,
layout-synthesis compiler benchmarked partly via GEMM but spanning many
kernel types).

## Divergences

- **Statistic**: no common convention. Best-of-N / min (WuZLHJWC23's implicit
  averaging is closer to mean, MPGEMM's explicit `min`, FastKron's
  min-of-4-means) vs. plain mean (fBLAS, CA3DMM) vs. geometric mean of 10
  (LibShalom). Best-of-N is the most common *effective* practice and is
  optimistic (reports the luckiest run, not typical behavior) — **the spec
  fixes this to median with min/max also reported**, per the design
  instructions, diverging from most of the literature's practice.
- **Warmup**: only FastKron states an explicit warmup count (1×10-iteration
  discarded call). MPGEMM's min-tracking loop has no discarded warmup at
  all. Most papers don't mention warmup. **The spec fixes an explicit
  warmup count.**
- **Cache state for small GEMM**: LibShalom is the only paper that treats
  this as a first-class, explicitly-reported axis (hot vs. cold cache,
  ~2× difference observed); everyone else times whichever state their loop
  structure happens to produce, unstated. Given the magnitude of the effect
  on exactly the shape class (small/irregular) several other papers in this
  track target, **the spec makes cache state an explicit, both-reported
  axis for the small/irregular variant** rather than an unstated one.
- **Preprocessing / codegen / autotuning-search cost**: consistently
  reported *separately* wherever it's addressed at all — autoGEMM excludes
  LIBXSMM's JIT time from the timed region "for fairness"; FastKron reports
  "Autotuning Time" as a distinct evaluation, separate from steady-state
  GFLOPS; Tensile-FGPS's own tuning-search config disables correctness
  validation (`NumElementsToValidate: 0`) during the search, implying
  validation is a separate, final-config-only step; CL-DB-GEMM's headline
  claim is about *reducing tuning time*, reported apart from per-kernel
  GFLOPS. This is one point of real convergence across the track, and the
  spec's steady-state variants adopt it directly: one-shot codegen/JIT/
  autotuning-search cost is reported once, never amortized silently into
  per-call time.
- **Correctness tolerance**: not one of the 9 surveyed papers states a
  numeric relative-error tolerance in the text/README that was extracted
  (several state *that* they validate against a vendor BLAS reference, none
  give the number). **The spec fixes explicit tolerances per precision**
  (this is a fairness gap the literature leaves open, not a resolvable
  divergence — flagged here and fixed rather than merely noted).
- **Timer type is platform-bound, not a real disagreement**: GPU work uses
  device-side kernel time (Tensile's `KernelTime: True`) or a
  synchronize-then-host-clock pattern (FastKron); CPU/ARM work uses a
  wall-clock function (MPGEMM's `dClock()`); FPGA work reports "averaged
  execution time" with a stated CI. These aren't in tension with each other
  — they're the correct timer for each platform — so the spec specifies one
  timer rule *per platform class* rather than forcing a single mechanism
  track-wide.
- **Shape sourcing splits into two real usage patterns**: (a) large
  near-square "textbook" HPC sweeps (WuZLHJWC23: 1024–6144 step 512; CA3DMM:
  8000³ example; CL-DB-GEMM: "around 2048"), and (b) small/irregular shapes
  sourced from real applications — scientific mini-kernels (LibShalom:
  SeisSol/Nekbox, 8–120³), DNN inference (autoGEMM: ResNet-50 layers;
  WuZLHJWC23's own irregular sweep, K fixed 256/1024, M,N 64–490), and LLM
  linear layers (MPGEMM: DeepSeek/LLaMA-derived M,N,K). **The spec keeps
  these as two separate variants** rather than merging them, since a single
  size list would either be dominated by one regime or dilute both.
- **Precision scope boundary with the quantized-GEMM track**: MPGEMM sweeps
  FP64/FP32/FP16/INT8 together as one multi-precision study; the paper is
  legitimately in-scope for this track (it is not "purely low-precision"),
  but its INT8 leg specifically belongs conceptually to the quantized-GEMM
  track. **The spec includes fp64/fp32/fp16 in the precision axis and
  explicitly excludes int8/sub-byte formats**, deferring to the sibling
  track for those.
- **Batched GEMM is present but thin**: only 2 of 9 surveyed papers exercise
  batched GEMM in their own benchmark harness (Tensile-FGPS's
  `ProblemType.Batched: True`, though its sample config actually uses
  batch=1; CL-DB-GEMM's `GEMMBATCHED`/`GEMMSTRIDEDBATCHED` configs with real
  batch-count sweeps). Given the track's explicit "incl. batched" scope,
  the spec still gives it a dedicated variant, built primarily from
  CL-DB-GEMM's concrete batch-count/size sweep since it's the only source
  with real numbers.
