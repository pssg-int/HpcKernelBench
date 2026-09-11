# Stencil track — evaluation-methodology survey

12 of the 21 papers in `data/track_inputs/stencil.json` were surveyed with concrete,
sourced facts (well above the ≥5/≥6 bar): 5 via arXiv fulltext (ar5iv HTML — plain
`arxiv.org/abs` and `arxiv.org/html` both failed to serve fulltext for pre-2023
papers; `arxiv.org/html` only worked for the 2026 SPIDER preprint), 7 via the
GitHub artifact repo's README + actual benchmark/driver source (`gh api`). Two
papers have no arXiv id and no OA link (ConvStencil, LoRAStencil), so the repo was
the only available primary source for them, per the instructions' source-preference
order.

---

## AN5D — automated stencil framework for high-degree temporal blocking on GPUs
CGO 2020, `conf/cgo/MatsumuraZWEM20`, arXiv 2001.01473 (ar5iv fulltext)

- **Workloads**: synthetic `star2d{1..4}r` / `box2d{1..4}r` (2D) and
  `star3d{1..4}r` / `box3d{1..4}r` (3D), plus named kernels `j2d5pt`, `j2d9pt`,
  `j2d9pt-gol`, `gradient2d` (2D) and `j3d27pt` (3D). Compile-time-constant
  coefficients except the `j2d*` division-based kernels.
- **Grid sizes**: 16384² for all 2D stencils, 512³ for all 3D stencils.
- **Timing protocol**: 1000 iterations for the primary numbers (120 iterations
  during the parameter-search/autotuning phase only). Each benchmark point is
  **repeated 5× on top of one discarded warmup run**, and the **average** is
  reported. A minimum-runtime constraint of 400 ms is enforced (implies
  iteration count/size choices are calibrated so a trial isn't overhead-bound).
- **Timing scope**: kernel only, PCI-E transfer excluded.
- **Temporal blocking degree** `bT` is itself a tuned parameter: `Sconf` fixes
  `bT=4`; the "Tuned" configuration sweeps `bT ∈ [1,16]` (2D) / `[1,8]` (3D) and
  reports the best.
- **Hardware**: Tesla P100 SXM2 and Tesla V100 SXM2, CUDA 10.0.130.
- **Baselines**: STENCILGEN, hybrid tiling, plain loop tiling; YASK/ARTEMIS cited
  as prior art.
- **Metric**: GFLOP/s.
- **Source**: ar5iv.labs.arxiv.org/html/2001.01473.

This paper is the origin of the `star/box{2d,3d}{radius}r` naming convention that
the entire 2024–2026 Tensor-Core-stencil lineage below reuses almost verbatim.

---

## K-Athena — performance-portable structured-grid finite-volume MHD
TPDS 2021, `journals/tpds/GreteGO21`, arXiv 1905.04341 (ar5iv fulltext)

- **Workload**: linear fast magnetosonic wave test problem on a static,
  structured 3D grid (a real finite-volume MHD solver, not a synthetic stencil
  microbenchmark) — 2nd-order accurate, box-shaped stencil support from the
  Riemann-solver + reconstruction pipeline.
- **Grid sizes**: 256³ for single-GPU/roofline numbers; 64³–256³ per GPU for
  weak scaling; up to 3072³ (also 1408³/1536³/2944³ reported) for strong scaling
  on Summit.
- **Timing protocol**: performance is the **80th percentile of per-cycle
  performance across several runs** (chosen specifically to damp inter-run and
  network variability on a shared HPC system) — not a single measurement, not a
  fixed iteration count timed once.
- **Hardware**: Intel Skylake Xeon Gold 6148 (20-core), Xeon Phi 7230 (KNL), AMD
  Opteron 6274, NVIDIA V100/P100/K20X, IBM POWER9 (Summit).
- **Precision**: fp64 throughout ("second-order double precision MHD").
- **Baselines**: original (CPU) Athena++, and the GAMER code.
- **Metric**: cell-updates/s. >10⁸ cell-updates/s on a single V100; 1.94×10¹²
  aggregate cell-updates/s at 76% parallel efficiency on 24,576 Summit GPUs
  (vs. 172,032 CPU cores).
- **Source**: ar5iv.labs.arxiv.org/html/1905.04341.

---

## Bisbas et al. — temporal blocking of FD stencils with sparse off-the-grid sources
IPDPS 2021, `conf/ipps/BisbasLLNGK21`, arXiv 2010.10248 (ar5iv fulltext), Devito artifact

- **Workloads**: three 3D finite-difference wave-propagation operators generated
  by the Devito DSL compiler — isotropic acoustic (2nd-order-in-time, space
  order 4/8/12), anisotropic acoustic/TTI (rotated-Laplacian pseudo-acoustic,
  space order 4/8/12), isotropic elastic (1st-order-in-time coupled system,
  space order 4/8/12).
- **Grid size**: 512³ velocity model, grid spacing 10 m (isotropic/elastic) or
  20 m (TTI).
- **Time steps**: CFL-determined, not an arbitrary round number —
  228 (isotropic acoustic), 436 (isotropic elastic), 587 (anisotropic acoustic),
  all covering 512 ms of simulated time.
- **Hardware**: single-socket 8-core Broadwell E5-2673 v4 (AVX2) and
  single-socket 16-core Skylake Platinum 8171M (AVX512); GCC 7.5.0 / ICC 2021.1.
- **Precision**: fp32.
- **Timing/tuning**: autotuner sweeps the whole blocking-parameter space for the
  global performance maximum; explicit repetition/statistic not stated in the
  extracted text.
- **Baseline**: Devito's own highly-optimized **spatially-blocked-only** code
  (i.e., the ablation is temporal-blocking on vs. off inside the same compiler).
- **Metric**: GPoints/s, plus roofline analysis (paper explicitly frames results
  as breaking through the L3-cache-bound ceiling).
- **Results**: space-order-4 speedups of ~1.6× (acoustic), ~1.44× (TTI), ~1.3×
  (elastic, Broadwell) from temporal blocking alone.
- **Source**: ar5iv.labs.arxiv.org/html/2010.10248.

---

## StencilsOnFPGA — HLS accelerator design for structured-mesh explicit solvers
IPDPS 2021, `conf/ipps/KamalavasanMRF21`, arXiv 2101.01177 (ar5iv fulltext)

- **Workloads**: Poisson-5pt-2D (2nd-order 5-point), Jacobi-7pt-3D (7-point),
  and a Reverse-Time-Migration (RTM) forward pass (25-point, 8th-order 3D).
- **Grid sizes**: much smaller than the GPU literature because of on-chip
  capacity — Poisson 200×100 up to 400×400 (also 15000² / 20000² with spatial
  blocking enabled); Jacobi 50³–300³; RTM 32³–50³.
- **Iteration counts**: Poisson 60,000 (baseline) / 6,000 (spatially blocked);
  Jacobi 29,000 (baseline) / 120 (spatially blocked); RTM 1,800 (baseline).
- **Hardware**: Xilinx Alveo U280 (8490 DSP blocks, 6.6 MB BRAM + 34.5 MB URAM,
  8 GB HBM, 32 GB DDR4), single precision (fp32).
- **Timing**: no explicit repetition count or statistic in the extracted text;
  bandwidth computed as bytes-moved / measured time.
- **Baseline**: NVIDIA Tesla V100 GPU (runtime and energy comparison, no CPU
  baseline given).
- **Metric**: GB/s bandwidth and raw runtime; result headline is
  "runtime equivalent to a V100 GPU with >2× energy savings" for the largest
  non-trivial application.
- **Source**: ar5iv.labs.arxiv.org/html/2101.01177.

---

## SPIDER — sparse Tensor Cores for stencil computation via strided swapping
PPoPP 2026, `conf/ppopp/GuW0Y26`, arXiv 2506.22035 (arxiv.org/html fulltext + `gh api` repo)

- **Workloads/shapes**: `1d1r`/`1d2r` (1D); `box2d1r`/`box2d3r`/`box2d2r`
  (2D — `box2d2r` used only in the Fig. 12 TCStencil-ablation comparison);
  `cudnn_box2d9p`/`cudnn_box2d25p` used as the cuDNN im2col baseline. SPIDER's
  own kernel sources (`src/1d_half_sparse`, `src/2d_half_sparse{,_for_ablation}`,
  `src/2d_half_dense`) implement **1D and 2D only** — no 3D kernel, despite the
  paper comparing against ConvStencil/LoRAStencil baselines that do support 3D.
- **Grid sizes** (from `scripts/Figure10_run.sh`/`Figure11_run.sh`/`Figure12_run.sh`):
  1D up to `(1, 10,240,000)` at `T=100,000`; 2D primary point `10240×10240` at
  `T=10,240`; Fig. 11 sweeps 2D from 512×512 up to 10240×10240 (step 512) at the
  same `T`; Fig. 12 sweeps 1280×1280 up to 20480×20480 at `T=10,000`.
  (Metric header literally: `method, shape, dim_3, dim_2, dim_1, iters, time,
  GStencil/s`.)
- **Hardware**: NVIDIA A100-80GB PCIe (repo mentions up to 4× A100 available).
- **Precision**: fp16 via `mma.sp.ordered_metadata` sparse Tensor Core MMA;
  cuDNN/ConvStencil baselines run fp64, TCStencil baseline runs fp16 (normalized
  in the paper's comparison).
- **Timing (from `gpu_2d_7r_half.cu`)**: `cudaEvent` pair wraps a host loop of
  `times` (= T) back-to-back kernel launches with ping-pong buffers; **each
  launch is individually followed by `cudaDeviceSynchronize()` inside the timed
  loop**; **single trial, no discarded warmup run**.
- **Preprocessing**: explicitly **excluded** from the timed region; paper text
  states "our evaluation does not reflect the offline transformation overhead,"
  and separately notes DRStencil's ~1-hour tuning cost as a caveat rather than a
  quantified, reported number.
- **Baselines**: cuDNN, DRStencil, TCStencil, ConvStencil, LoRAStencil,
  FlashFFTStencil. Headline: 6.20× over cuDNN, 2.00× average over the best prior
  Tensor-Core stencil system.
- **Sources**: arxiv.org/html/2506.22035 (Evaluation section);
  `github.com/KevinWu2017/SPIDER` — `README.md`, `scripts/Figure10_run.sh`,
  `scripts/Figure11_run.sh`, `scripts/Figure12_run.sh`,
  `src/2d_half_sparse/gpu_2d_7r_half.cu` (timing code).

---

## ConvStencil — transform stencil computation to matmul on Tensor Cores
PPoPP 2024, `conf/ppopp/ChenLWBWMYZCY24`, no arXiv id — repo is the primary source

- **Shapes/CLI**: `convstencil_{1d,2d,3d} shape input_size... time_iteration_size
  [options]`; shapes `1d1r`/`1d2r` (1D), `star2d1r`/`box2d1r`/`star2d3r`/`box2d3r`
  (2D), `star3d1r`/`box3d1r` (3D) — this is the exact catalog AN5D introduced.
- **Timing (from `src/2d/gpu.cu`)**: host-side `std::chrono::steady_clock` wraps
  a loop of `times` (T) back-to-back kernel launches on ping-pong device
  buffers, one `cudaDeviceSynchronize()` after the whole loop (not per launch),
  **single trial, no discarded warmup**. `GStencil/s = input_m*input_n*times*K /
  secs / 1e9` where `K` is a per-shape multiplier (1 for some shapes, 3 for
  others in the source read) whose exact meaning could not be fully resolved
  from source alone — flagged in Open Questions.
- **Correctness**: an internal CPU naive reference with `tolerance = 1e-7` exists
  in `main.cu` but is gated behind a `#define CHECK_ERROR` that is **commented
  out by default** — the public benchmark path does not check correctness.
- **Hardware/software**: single NVIDIA A100, CUDA ≥11.0 (12.2 tested), cuDNN
  ≥8.0, GCC ≥9.4.
- **Used as baseline by**: SPIDER's harness runs it at grid `10240×10240`
  (2D)/`10,240,000` (1D) with `T=10240`/`100000` respectively — i.e. its de
  facto standard evaluation point comes from a *later* paper's reproduction, not
  its own README (which gives only the CLI, no headline numbers).
- **Source**: `github.com/microsoft/ConvStencil` — `README.md`,
  `src/2d/gpu.cu`, `src/2d/main.cu`.

---

## LoRAStencil — low-rank adaptation of stencil computation on Tensor Cores
SC 2024, `conf/sc/ZhangLYCZCY24`, no arXiv id — repo is the primary source

- **Shapes/CLI**: `lorastencil_{1d,2d,3d} shape input_size time_size` — identical
  shape catalog to ConvStencil (`1d1r`, `1d2r`, `star2d1r`, `box2d1r`,
  `star2d3r`, `box2d3r`, `star3d1r`, `box3d1r`).
- **Hardware/software**: NVIDIA A100, CUDA ≥11.0 (12.2 tested), GCC ≥9.4.
- **Contribution focus**: decomposes stencil weight matrices into rank-1 factors
  to remove TCU "dimension residue" redundancy; no additional timing detail
  beyond the shared CLI convention is exposed in the README.
- **Used as baseline by SPIDER** at the same grid/`T` points as ConvStencil
  above (`10240×10240`@`T=10240` 2D, `10.24M`@`T=100000` 1D).
- **Source**: `github.com/HPHEX/LoRAStencil` — `README.md`.

---

## FlashFFTStencil — bridging FFT to memory-efficient stencil computation on TCUs
PPoPP 2025, `conf/ppopp/HanLCBZYCZCY25`, no arXiv id — repo is the primary source

- **Shapes/CLI**: `{1d,2d,3d} shape input_size time_iteration_size [options]`;
  shapes named by point-count rather than radius — `Heat-1D`/`1D5P`/`1D7P` (1D),
  `Heat-2D`/`Box2D9P`/`Star2D9P`/`Box2D25P` (2D), `Heat-3D`/`Box3d27P` (3D).
  `Box2D9P`/`Box2D25P` are the natural point-count aliases of ConvStencil's
  `box2d1r`/`box2d3r`.
- **Hardware/software**: tested on A100 and H100 (any Ampere/Hopper GPU claimed
  to work), CUDA 12.1, PyTorch 2.0 required for the benchmarking harness only
  (not the compute path).
- **Headline result**: 2.57× average speedup over prior SOTA; 103.0× over a
  cuFFT-based 1D stencil implementation.
- **Timing convention**: same `time_iteration_size` CLI argument as
  ConvStencil/LoRAStencil — confirms the whole lineage shares one timing idiom.
- **Source**: `github.com/KevinWu2017/FlashFFTStencil` — `README.md`.

---

## PERKS — locality-optimized execution model for iterative memory-bound GPU apps
ICS 2023, `conf/ics/ZhangWCMWEM23`, OA PDF (OSTI) + repo (`gh api`, primary for methodology)

- **Workloads**: 2D 5/9/13/17/21/25-point Jacobi-style stencils plus sparse
  variants (`2ds9pt`, `2ds25pt`), 3D stencils, and a conjugate-gradient solver —
  the persistent-kernel execution model is explicitly meant to generalize across
  any iterative memory-bound solver, not just stencils.
- **CLI flags** (`jacobi.driver.cpp`): `--fp32`, `--check`, `--usesm`, `--bdim`,
  `--blkpsm`, `--iter=T`, `--warmup`, `--warmiter`, `--doubletile`. Default
  precision is fp64 unless `--fp32` is passed.
- **Timing (from `jacobi-2d.cu`)**: when `--warmup` is set, the code runs an
  adaptive calibration phase — it measures the elapsed time of
  `l_warmupiteration` (default 1000) iterations via CUDA events, then computes
  `nowiter = ceil(350 / measured_ms_per_iter_batch)` **additional** iterations
  needed so the *real* timed run's wall time is ≥350 ms (same idea as AN5D's
  400 ms floor, different constant). The real measurement itself is a
  **single trial** (`RUNS = 1`, no repetition/statistic), CUDA-event-timed,
  wrapping `iteration` (T) back-to-back kernel launches
  (persistent-kernel or traditional-launch mode), followed by
  `cudaDeviceSynchronize()` + error check.
- **Metric**: the same run simultaneously prints Computation Time (ms),
  **GCells/s**, **GFLOPS/s** (using a stencil-specific FLOPs-per-cell constant —
  17 FLOP/cell hard-coded for their 9-point case), and **bandwidth (GB/s)** —
  i.e. this paper already reports the geometry-only rate *and* two derived
  metrics side by side, which is the pattern this track's spec should copy.
- **Correctness**: `TOLERANCE = 1e-5` against a CPU reference in
  `jacobi_reference.hpp`; a `REFCHECK` build mode exists and forces
  `iteration=4` for a fast correctness-only run.
- **Source**: `github.com/neozhang307/PERKS` — `README.md`,
  `stencil/2dstencil/jacobi.driver.cpp`, `stencil/2dstencil/jacobi-2d.cu`.

---

## AOStencil — locality-aware instruction-stream scheduling for stencils on ARM
ICS 2025, `conf/ics/LiuYYLLQ25`, no arXiv fulltext — repo is the primary source

- **Workloads**: DSL-defined arbitrary-coefficient stencils; example benchmarks
  cover 2D 9-point box, 2D star, 3D 7-point star, 3D 27-point box
  (`benchmark/{phytium,kunpeng}/{2d9pt_box,2d9pt_star,3d7pt_star,3d27pt_box}.py`),
  plus separate fp64 (`*_f64`) benchmark directories alongside the fp32 default.
  Auto-tuned via a genetic algorithm over blocking/register/thread parameters.
- **Grid sizes** (from `2d9pt_box.py`): square 2D grids at 8192, 16384, 24576
  per side; 3D sizes not directly confirmed from the files read (flagged as an
  open question).
- **Iteration count**: `times = 100` fixed, hard-coded in the generated C main
  (`str_main.py`'s `rand_main_2d`/`rand_main_3d` templates).
- **Timing**: CPU wall clock via `omp_get_wtime()`, **single trial**, no
  repetition/statistic, no explicit warmup in the final-measurement path (the
  GA autotuner uses a separate, smaller `test_time_per_iter=50` internally
  during its own search phase, not as a warmup for the reported number).
- **Correctness**: a separate `check_main_2d`/`check_main_3d` path recomputes
  the same `times` iterations with a scalar CPU reference kernel and reports
  max relative error against a `1e-5` threshold.
- **Hardware**: Phytium FT-2000+ and Kunpeng 920 ARM many-core CPUs, NUMA-aware
  (16 nodes × 8 cores in the example configuration).
- **Source**: `github.com/buaa-hipo/AOStencil` — `README.md`,
  `benchmark/phytium/{run.sh,2d9pt_box.py,phytium_test.py,str_main.py}`.

---

## WindStencil — high-order stencil computation for inviscid CFD on GPU
ICS 2026, `conf/ics/ZhangZLLJZYLL26`, no arXiv/OA — repo is the primary source

- **Workload**: a production compressible Navier–Stokes solver (OpenCFD-SC
  lineage), 7th-order 3D WENO scheme for the split-form inviscid flux — a
  **high-order, multi-kernel, application-level** stencil (fuses what is
  normally 9 separate kernels into one, "BF3W").
- **Grid sizes**: non-cubic, CFD-realistic domains — 480×64×32 up to
  480×512×64 (MI60 sweep), 600×64×32 up to 600×256×128 (MI200 sweep), and a
  fixed 600M-cell (9600×256×256) case for strong scaling up to 1024 GPUs.
- **Timing protocol**: headline number is explicitly labeled
  **"End-to-end speedup (1000 iterations, full application)"** — i.e. it bulk
  times 1000 whole-program steps, not per-sweep — and the README **separately**
  reports a "kernel-focused gain" (isolated-kernel time ratio) alongside the
  "full-application gain" (whole-program time ratio) for every grid, which is
  exactly the kernel-only vs. end-to-end split this track's spec needs.
- **Hardware**: AMD GPUs via HIP/ROCm (MI60, MI200-series), FP64 compute, MPI
  multi-GPU with device-aware halo exchange.
- **Correctness**: not a pointwise tolerance check — physically **conserved
  quantities** (Total Energy, Total Temperature) are compared between optimized
  and unoptimized runs at every step over a 10,000-step run; max relative
  differences reported as 1.12×10⁻⁵ (energy) and 4.16×10⁻⁵ (temperature).
- **Roofline reporting**: FP64 %-of-peak given alongside speedup (37.2% MI60,
  ~30.8% MI200; attainable-roof utilization ~51% on MI200), and weak/strong
  scaling efficiency curves.
- **Source**: `github.com/BabyXPrince/WindStencil` — `README.md`.

---

## ozHOPE — Ozaki-scheme FP16 Tensor Core acceleration of a shallow-water dynamical core
IPDPS 2026, `conf/ipps/YaoZLX26`, no arXiv/OA — repo README + abstract

- **Workload**: HOPE, a high-order finite-volume shallow-water dynamical core;
  convolution-reformulated stencil computation accelerated via the Ozaki scheme
  on FP16 Tensor Cores while emulating working precision tunable up to FP64.
- **Benchmark cases**: the **Williamson standard test suite** for atmospheric
  dynamical cores — steady-state geostrophic flow, Rossby–Haurwitz wave,
  perturbed jet flow — i.e. domain-standard validation problems, not a
  synthetic grid-size sweep.
- **Hardware/software**: NVIDIA A40 / RTX A6000, CUDA 11.8, cuDNN 9.1.0 (via a
  cuDNN-frontend Python interface), PyTorch 2.6.
- **Correctness**: diagnostic variables compared against FP64 reference
  results per benchmark case — accuracy/convergence-order preservation is
  itself a first-class reported result, not just a pass/fail gate.
- **Headline result** (abstract): average end-to-end speedup 1.82×, up to
  4.34×, while maintaining the numerical method's expected order of
  convergence.
- **Source**: `github.com/jnyao/ozHOPE` — `README.md`; abstract from
  `data/track_inputs/stencil.json`.

---

## Divergences

1. **Bulk-timing convention is universal — but nothing else about iteration
   count is.** Every single surveyed paper, without exception, times **T
   back-to-back sweeps as one elapsed-time interval** and divides by T for a
   per-step number; *none* times individual sweeps separately. But T itself is
   chosen by four incompatible conventions: (a) a fixed microbenchmark constant
   shared by convention across a paper family (ConvStencil-lineage: T=10,240 2D
   / T=100,000 1D; AOStencil: T=100 fixed); (b) adaptively calibrated to a
   minimum wall-clock floor (AN5D: ≥400 ms; PERKS: ≥350 ms); (c) physically
   determined by a CFL condition or convergence target (Devito: 228–587 steps;
   WindStencil correctness check: 10,000 steps); (d) an application's own
   "several runs" of production-length cycles with a percentile statistic
   (K-Athena: 80th percentile). The spec fixes T explicitly per variant rather
   than picking one convention as universally "correct."

2. **Repetition/statistic rigor varies from strong to nonexistent.** AN5D is
   the most rigorous (5 repeats + 1 discarded warmup, reports the mean).
   K-Athena reports an 80th percentile over several runs. Every GPU
   Tensor-Core-stencil paper surveyed (ConvStencil, LoRAStencil,
   FlashFFTStencil-by-inheritance, SPIDER) and both CPU papers (PERKS,
   AOStencil) publish a **single, unrepeated trial with no discarded warmup** in
   their public benchmark harness. This is the single biggest fairness gap in
   the track's own literature; the spec mandates a discarded warmup + median
   over ≥3 repeats for every variant, explicitly overriding the papers' own
   practice.

3. **Precision splits the track into two hardware classes.** Classic
   memory-bound papers default to fp64 (K-Athena explicitly, PERKS by default
   unless `--fp32`) or use fp32 by choice (Devito, StencilsOnFPGA, AOStencil's
   default example). The entire 2024–2026 Tensor-Core-matmul lineage
   (ConvStencil, LoRAStencil, FlashFFTStencil, SPIDER) is fp16-only because
   Tensor/Sparse-Tensor-Core MMA throughput is the whole point; ozHOPE goes
   further and treats "restoring fp64-equivalent accuracy from fp16 hardware"
   (via the Ozaki scheme) as its actual contribution. This precision split maps
   directly onto the spec's two kernel-only variants below.

4. **Metric definitions are not mutually comparable, even when they share a
   name.** GFLOP/s (AN5D, PERKS) is only meaningful once a FLOPs-per-point-update
   convention is fixed, and that convention is **not standardized**: PERKS
   hard-codes 17 FLOP/cell for one specific 9-point kernel; ConvStencil's own
   printed `GStencil/s` formula multiplies by an unexplained per-shape factor
   (1× vs. 3× seen in source) that could not be fully resolved from code alone.
   GCell-updates/s (a.k.a. GStencil/s — K-Athena, PERKS, and the whole
   Tensor-Core lineage all report *some* form of this) depends only on grid
   size × iteration count, making it the only truly apples-to-apples number
   across the whole track; the spec makes it primary and demotes GFLOP/s to a
   secondary, convention-must-be-stated figure. StencilsOnFPGA additionally
   reports GB/s, appropriate for its bandwidth-bound FPGA target but not
   directly convertible to the other two without knowing bytes-moved-per-update.

5. **Preprocessing/format-conversion scope is asymmetric by construction.**
   Classic memory-bound papers have essentially no preprocessing (arrays are
   allocated and initialized once; there's no format to convert to). The
   Tensor-Core-matmul lineage has *real*, nontrivial one-shot preprocessing
   (parameter/lookup-table packing, sparse-metadata compression, FFT-plan setup
   for FlashFFTStencil) that SPIDER's paper explicitly excludes from its
   reported throughput and does not otherwise quantify as its own number
   (beyond noting a competitor's "~1 hour" tuning cost as a caveat). The spec's
   Tensor-Core variant requires this preprocessing time be *measured and
   reported once*, not merely excluded and left unquantified as the surveyed
   papers do.

6. **Grid-size convention converges tightly for synthetic microbenchmarks, but
   application papers ignore it entirely (correctly).** 2D synthetic grids
   cluster in the 8K–16K-per-side range and 3D synthetic grids cluster at 512³
   across both the classic lineage (AN5D 16384²/512³, Devito 512³) and the
   Tensor-Core lineage (ConvStencil/LoRAStencil/SPIDER 10240×10240), with
   AOStencil's ARM sweep (8192/16384/24576) landing in the same band. FPGA
   (StencilsOnFPGA, 50³–400² due to on-chip capacity) and full physics
   applications (K-Athena up to 3072³ for scaling studies but 256³ for
   single-node perf; WindStencil's non-cubic CFD domains) deliberately don't
   follow this convention because their sizes are dictated by device capacity
   or physical realism, not by a shared microbenchmark tradition. The spec
   keeps a separate, paper-native "end-to-end application" variant rather than
   forcing these into the synthetic-grid convention.

7. **Correctness rigor ranges from disabled to physics-grade.** PERKS
   (`TOLERANCE=1e-5`) and AOStencil (`1e-5` relative error) both gate on an
   explicit pointwise tolerance against a CPU reference. ConvStencil has the
   same kind of check available (`tolerance=1e-7`) but it is **commented out
   by default** in the public artifact — the published benchmark path runs with
   no correctness check enabled at all. Application papers (WindStencil,
   ozHOPE) validate physically conserved quantities or convergence order rather
   than pointwise numerical agreement, which is a materially weaker but
   domain-appropriate guarantee. The spec requires an explicit tolerance gate
   for both kernel-only variants and flags the application variant's weaker
   guarantee as an open question rather than silently equating it.

8. **A real shared shape taxonomy exists and should be used as the spec's
   canonical suite.** AN5D's `{star,box}{2d,3d}{radius}r` naming is reused
   verbatim by ConvStencil, LoRAStencil, and SPIDER's CLI; FlashFFTStencil and
   AOStencil use point-count names (`Box2D9P`, `2d9pt_box`, ...) that name the
   same underlying shapes. This is a genuine, evidence-backed "standard suite"
   for this track, not an invented one.
