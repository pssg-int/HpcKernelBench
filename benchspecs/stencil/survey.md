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
   FlashFFTStencil-by-inheritance, SPIDER), the GPU paper PERKS, and the CPU
   paper AOStencil publish a **single, unrepeated trial with no discarded
   warmup** in their public benchmark harness (PERKS's paper text states ten
   runs per evaluation; only its repo driver is single-trial). This is the single biggest fairness gap in
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

---

## Metrics addendum (2026-09-21) — every paper's own measures

Every one of the 21 papers in `data/track_inputs/stencil.json` now has its own
measures recorded. Only the 5 papers with a working adapter (AN5D, ConvStencil, LoRAStencil,
FlashFFTStencil, SPIDER) are in `spec.yaml` `metrics.per_paper_native`; the
other 16 (4 skipped in integration, 1 never integrated, 11 with no path to
this harness) are listed at the end of this addendum as reference material
only. Depth differs and is recorded per entry as `source_quality`:

- **Fulltext read (12)**: AN5D (ar5iv), K-Athena (ar5iv), Bisbas/Devito (ar5iv),
  StencilsOnFPGA (ar5iv), CLAIRE (ar5iv 2008.12820) — via fetched text;
  PERKS (arXiv 2204.02064), SARIS (arXiv 2404.05303), StencilFlow (arXiv
  2010.15218), YaskSite (FAU CGO'21 preprint), Futhark memory optimizations
  (futhark-lang.org/publications/sc22-mem.pdf), HLS Transformations (arXiv
  1805.08288) — PDFs downloaded and text-extracted with pypdf; SPIDER was
  already fulltext above.
- **Repo only (6)**: AOStencil, WindStencil, ozHOPE, ConvStencil, LoRAStencil,
  FlashFFTStencil (no fulltext obtained; unchanged from the sections above).
- **Abstract only (3)**: Dendro-GR (SC'22), Rise/Harris (CGO'21), PrecTuner
  (PPoPP'24). Their evaluation text was NOT read; the ACM/IEEE versions were not
  reachable and the repo docs list scripts, not results. Extend before relying
  on them.

New facts worth knowing that change earlier statements:

1. **PERKS ran each evaluation ten times** (paper, Sec. 6.1), in fp64 on V100
   and A100. Its repo driver is a single trial (`RUNS = 1`). Both are true; the
   spec records both. PERKS is also a GPU paper, not a CPU one.
2. **SARIS independently reuses AN5D's named kernels** (`j2d5pt`, `j2d9pt`,
   `j2d9pt_gol`, `j3d27pt`) and AN5D's 16384^2 / 512^3 grids for its scale-out
   estimate, which supports treating them as a shared suite.
3. **Metrics that are not throughput at all**: YaskSite's ranking "performance
   loss" ((best - selected) / selected x 100), K-Athena's performance-portability
   score (harmonic mean of architectural efficiencies), StencilFlow's percent of
   roofline, and CLAIRE's time-to-solution. These have no counterpart in the
   common GCell-updates/s number and are why the spec reports paper-native
   metrics alongside it instead of collapsing them.
4. **Modeled vs measured is often mixed inside one paper** (AN5D's model
   performance, YaskSite's ECM prediction, SARIS's 256-core scale-out
   estimate, StencilsOnFPGA's cycle model, PERKS's projected peak). The spec
   tags each metric `model` so it is never merged with a measured value.
5. **Not every stencil-track paper is a stencil-kernel paper.** Futhark memory
   optimizations, HLS Transformations, CLAIRE, Dendro-GR, Rise/Harris and
   PrecTuner measure something adjacent (compiler passes, whole solvers,
   affine programs); the spec marks them `peripheral`.

### Reference-only metric lists (not in the spec)

### K-Athena (TPDS 2021) — skipped in integration
- **Source quality**: fulltext (ar5iv 1905.04341)
- **Native primary**: [timing] cell-updates/s (e.g. >1e8 on one V100; 1.94e12 aggregate on 24,576 Summit GPUs), taken as the 80th percentile of per-cycle performance across several runs
- **Also reports**:
  - [hw] parallel efficiency (%) versus ideal scaling (76% on Summit at 24,576 GPUs)
  - [model] architectural efficiency e(a,p,i) = achieved / min(T_peak, B x I): fraction of the bandwidth-limited roofline peak
  - [profiler] arithmetic intensity in FLOP/byte (nvprof, Intel Advisor, or by hand)
  - [hw] DRAM and L1 bandwidth in GB/s (Empirical Roofline Tool, GPUMembench, Intel Advisor)
  - [model] performance-portability metric P = |H| / sum(1/e_i), harmonic mean of architectural efficiencies (paper reports 62.8%)
  - [timing] GPU-vs-CPU speedup (30x on 24,576 GPUs against 172,032 CPU cores)

### Bisbas / Devito (IPDPS 2021) — never integrated; named by variant 3
- **Source quality**: fulltext (ar5iv 2010.10248)
- **Native primary**: [timing] throughput in GPoints/s; speedup over Devito's own highly optimized spatially-blocked-only code
- **Also reports**:
  - [profiler] cumulative (L1+L2+LLC+DRAM) traffic-based arithmetic intensity (Intel Advisor)
  - [profiler] cache-aware roofline placement of each operator (Intel Advisor), used to show the L3-bound ceiling being broken
  - [timing] speedup per operator and space order (paper: acoustic 1.6x at order 4 and 1.13x at order 8; TTI 1.44x at order 4; elastic 1.3x on Broadwell and 1.22x on Skylake at order 4)
  - NOT reported by the paper (do not invent): compile time, cache-miss counters, bandwidth measurements, tile-parameter sensitivity

### PERKS (ICS 2023) — skipped in integration
- **Source quality**: fulltext (arXiv 2204.02064) plus repo driver source
- **Native primary**: [timing] GCells/s (giga-cells updated per second); geometric-mean speedup of PERKS over each baseline (PPCG, Bricks, SSAM, STENCILGEN, SHM): 2.12x for 2D and 1.24x for 3D stencils
- **Also reports**:
  - [timing] the repo driver prints, in one run: computation time (ms), GCells/s, GFLOP/s (stencil-specific FLOP-per-cell constant, e.g. 17 for the 9-point case), and bandwidth (GB/s)
  - [model] projected peak performance P from a roofline-inspired model, and the concurrency analysis (thread blocks per SM, occupancy) used to choose how many resources to free for caching
  - [timing] conjugate-gradient companion results: sustained memory bandwidth (TB/s) and per-time-step speedup versus Ginkgo (geomean 4.86x on smaller SpMV inputs)
  - [accuracy] TOLERANCE = 1e-5 against a CPU reference (repo)
  - PROTOCOL NOTE: the paper states each evaluation was run ten times, in fp64, on V100 and A100; the repo driver's own final measurement is a single trial (RUNS = 1)

### WindStencil (ICS 2026) — skipped in integration
- **Source quality**: repo (README; paper text not read)
- **Native primary**: [timing] end-to-end speedup over 1000 whole-application iterations, reported separately as kernel-focused gain (isolated-kernel time ratio) and full-application gain (whole-program time ratio) for every grid
- **Also reports**:
  - [model] FP64 percent of peak (37.2% on MI60, ~30.8% on MI200) and attainable-roofline utilization (~51% on MI200)
  - [hw] weak- and strong-scaling efficiency curves (up to 1024 GPUs, 600M-cell strong-scaling case)
  - [accuracy] conserved quantities (total energy and total temperature) compared between optimized and unoptimized runs at every step of a 10,000-step run: max relative difference 1.12e-5 (energy), 4.16e-5 (temperature)

### ozHOPE (IPDPS 2026) — skipped in integration
- **Source quality**: repo (README) plus abstract; evaluation text not read
- **Native primary**: [timing] end-to-end speedup: 1.82x average, up to 4.34x, while preserving the numerical method's convergence order
- **Also reports**:
  - [accuracy] diagnostic variables compared against FP64 reference results for each Williamson case (steady geostrophic flow, Rossby-Haurwitz wave, perturbed jet); accuracy and convergence-order preservation are first-class reported results, not just a pass/fail gate

### StencilsOnFPGA (IPDPS 2021)
- **Source quality**: fulltext (ar5iv 2101.01177)
- **Native primary**: [hw] runtime and effective bandwidth in GB/s = total bytes transferred by the stencil loop / total time (Poisson 867-922, Jacobi 202-438, RTM 77-293 GB/s)
- **Also reports**:
  - [model] predicted clock cycles per mesh point per iteration and total cycles; paper states the model is accurate to within +/-15% of achieved runtime
  - [hw] energy in kJ (measured with xbutil) and average power in W (FPGA 70-90 W, GPU 40-240 W); energy ratio versus V100 (>2x savings for the largest RTM case)
  - [hw] DSP-block and BRAM/URAM utilization (target 80-90% internal memory), unroll factor p, vectorization factor V
  - [hw] achieved clock frequency in MHz (250 Poisson, 246 Jacobi, 261 RTM); model-predicted vs synthesized DSP usage; valid-ratio of tiles (98.4-98.5%)
  - [timing] runtime comparison against an NVIDIA V100 on the same problems

### AOStencil (ICS 2025)
- **Source quality**: repo (README + benchmark scripts; paper text not read)
- **Native primary**: [timing] CPU wall time via omp_get_wtime over a fixed times = 100 sweeps; the throughput unit and the paper's headline speedup figures were NOT confirmed from the sources read
- **Also reports**:
  - [accuracy] maximum relative error against a scalar CPU reference over the same times = 100 iterations, 1e-5 threshold (check_main_2d / check_main_3d)
  - [hw] genetic-algorithm autotuning cost (search phase uses test_time_per_iter = 50; not a warmup for the reported number)

### SARIS (DAC 2024)
- **Source quality**: fulltext (arXiv 2404.05303)
- **Native primary**: [hw] speedup over an RV32G baseline (geomean 2.72x; range 2.36x jacobi_2d to 3.87x j3d27pt), from runtimes extracted from cycle-accurate RTL simulation traces of an eight-core Snitch cluster
- **Also reports**:
  - [hw] FPU utilization (geomean 35% -> 81%) and per-core IPC (0.89 -> 1.11)
  - [hw] cluster power in W from post-layout gate-level simulation at 1 GHz, 25 C, 0.8 V (PrimeTime): geomean 227 mW base, 390 mW SARIS; energy-efficiency gain 1.27x-2.17x, geomean 1.58x
  - [model] 256-core Manticore-256s scale-out ESTIMATE (not measured): FPU utilization 35% -> 64%, geomean speedup 2.14x, peak 406 GFLOP/s, compute-to-memory time ratio (CMTR) for memory-bound codes
  - [model] fraction of peak compute, compared against other published systems (paper's Table 2: SARIS 79% vs AN5D 69% on V100 SXM2)
  - workload note: the paper uses AN5D's named kernels (j2d5pt, j2d9pt, j2d9pt_gol, j3d27pt) plus box2d1r, star2d3r, star3d2r, jacobi_2d, ac_iso_cd, and AN5D's 16384^2 / 512^3 grids for the scale-out estimate

### StencilFlow (CGO 2021)
- **Source quality**: fulltext (arXiv 2010.15218)
- **Native primary**: [hw] GOp/s / TOp/s of floating-point operations (square root counted as one op): 1.31 TOp/s single-device and 4.18 TOp/s multi-device Stratix 10; fp32
- **Also reports**:
  - [model] expected runtime from C = L + I * N cycles (pipeline latency L, initiation interval I = 1, N iterations over vector width), converted to time as C / f; measured runtime is reported against it
  - [hw] runtime in microseconds and achieved bandwidth versus peak (36.4 GB/s = 47% and 58.3 GB/s = 76% of the 76.8 GB/s peak in the paper's examples)
  - [model] percent of roofline (%Roof.) at the program's arithmetic intensity, and the bandwidth needed to saturate compute (paper: 254.0 GB/s)
  - [hw] ALM, FF, M20K and DSP utilization; achieved clock 292-317 MHz; multi-device scaling from 1 to 8 FPGAs (scalar 264 GOp/s -> 1.5 TOp/s; 4-way vectorized 568.2 GOp/s -> 4.2 TOp/s)
  - [timing] runtime and GOp/s against Tesla P100, V100 and a 12-core Xeon on the horizontal-diffusion program (128x128x80)

### YaskSite (CGO 2021)
- **Source quality**: fulltext (CGO'21 preprint)
- **Native primary**: [timing] GLUP/s and MLUP/s (giga/mega lattice updates per second), fp64, measured and set beside the ECM-model prediction
- **Also reports**:
  - [model] ECM-predicted vs measured performance across problem sizes, and its scaling with core count up to memory saturation within one socket
  - [profiler] memory data volume measured with LIKWID versus the model's predicted volume
  - [timing] tuning time in seconds: YaskSite's analytical tuner versus YASK's gradient-descent tuning
  - [model] mean and maximum prediction deviation (%) per problem, and ranking quality as performance loss (%) = (best - selected) / selected x 100, where best is the measured-best variant and selected is the one the model chose (paper Table III: mean 1.0-1.2% on Cascade Lake, 2.2-4.4% on Rome)
  - [timing] speedup from vector folding (2.5x on Rome); run-to-run variation stated as under 5%

### Memory Optimizations in an Array Language / Futhark (SC 2022)
- **Source quality**: fulltext (futhark-lang.org/publications/sc22-mem.pdf)
- **Scope**: peripheral — a compiler memory optimization; only Rodinia Hotspot (repeated stencil) and Parboil LBM are stencil-like
- **Native primary**: [timing] wall time (ms) of the hand-written reference, with Futhark's unoptimized and optimized versions given as speedup ratios against it, plus the optimization impact (optimized / unoptimized); each benchmark's first run discarded and the mean of the rest reported
- **Also reports**:
  - [timing] run counts stated per table (Hotspot 10 runs, LBM 100 runs), on NVIDIA A100 and AMD MI100
  - [timing] Hotspot at 8192 / 16384 / 32768 and LBM at short / long datasets; optimization impact up to ~2.05x on Hotspot (A100) and ~1.6x on LBM (MI100)

### Transformations of HLS Codes for HPC (TPDS 2021)
- **Source quality**: fulltext (arXiv 1805.08288), evaluation figures read as extracted text and NOT re-checked against the rendered figures
- **Scope**: peripheral — a catalog of HLS source transformations; the 4-point 2D stencil (8192x8192, fp32) is one of its three worked examples
- **Native primary**: [hw] GOp/s at each optimization stage (naive, pipelined, vectorized, systolic), with per-step and cumulative speedup over the naive version
- **Also reports**:
  - [hw] LUT, DSP and BRAM utilization as a fraction of the device (maxima taken as 1728K LUTs, 12,288 DSPs, 2688 BRAM)
  - [model] throughput of about one cell per cycle once pipelined; pipeline-model quantities initiation interval I and latency L

### CLAIRE multi-GPU image registration (SC 2020)
- **Source quality**: fulltext (ar5iv 2008.12820)
- **Scope**: peripheral — image registration; its finite-difference, interpolation and FFT kernels are stencil-like, but the reported numbers are for the whole solver
- **Native primary**: [timing] time to solution in seconds (about 5 s for a 256^3 registration on one V100; 3.7 s with gradient storage), and speedup over prior implementations (34x over the CPU version)
- **Also reports**:
  - [timing] per-kernel runtime and the share spent in communication for the finite-difference and interpolation kernels (256^3 to 1024^3), and FFT runtime in ms
  - [hw] sustained bidirectional CUDA-aware MPI bandwidth in GB/s (all-to-all vs peer-to-peer) for the FFT
  - [hw] strong and weak scaling from 1 to 64 GPUs, and per-GPU memory in GB (model: mu_total = (74 + N_t) N mu_0 / p + mu_IP + mu_API)
  - [accuracy] Gauss-Newton and PCG iteration counts, relative mismatch, relative gradient norm; single precision on V100

### Dendro-GR GPU AMR solver (SC 2022)
- **Source quality**: abstract-only (abstract plus the repo's public docs; evaluation text not read)
- **Scope**: peripheral — numerical relativity; the BSSN right-hand side is a high-order stencil inside an AMR solver
- **Native primary**: [hw] GFlops/s (800 GFlops/s on one A100)
- **Also reports**:
  - [timing] speedup of 2.5x over an equivalent two-socket 128-core AMD EPYC 7763 CPU node, and 6x over existing state-of-the-art numerical-relativity codes (the paper itself notes such comparisons are difficult)
  - [hw] strong scaling to 8 A100s and weak scaling to 229,376 x86 cores on Frontera (from the repo docs)
  - [accuracy] gravitational-waveform accuracy assessments (abstract; the mass-ratio range was not read); padding-zone (octant-to-patch and patch-to-octant) timings on CPU and GPU (repo scripts)

### Rise / Harris on mobile CPUs (CGO 2021)
- **Source quality**: abstract-only (abstract plus artifact README; evaluation text not read)
- **Scope**: peripheral — Harris corner detector image pipeline on ARM mobile CPUs
- **Native primary**: [timing] runtime speedup over OpenCV (up to 16x) and over Halide (close to, and up to 1.4x better than); run count and statistic are NOT stated in the README
- **Also reports**:
  - [accuracy] output correctness is checked for both test images (small rgb.png and large venice_wikimedia.jpg)
  - platform: Cortex A7, A15, A53 and A73 on Odroid XU4 and N2 boards

### PrecTuner / lnlamp (PPoPP 2024)
- **Source quality**: abstract-only (abstract plus artifact README; evaluation text not read)
- **Scope**: peripheral — mixed-precision code generation for PolyBench-style affine programs, not a stencil-specific paper
- **Native primary**: [timing] speedup over LuIs (3.28x), over Pluto (1.81x single-core, 1.52-1.73x multi-core), and over PPCG on GPU (1.71x)
- **Also reports**:
  - [accuracy] output-quality degradation against a user-set error budget/threshold; the README does not define the error metric
  - [hw] tuning cost of the search (the paper's central claim is predicting the best parameter without evaluating every variant)


---

## Paper-native metrics addendum (2026-09-23): fulltext for all 5 integrated papers

The three papers listed as "repo only" above were read in full from
author-hosted PDFs, since the ACM links refuse non-browser clients:
ConvStencil (microsoft.com `ppopp24_ConvStencil.pdf`), LoRAStencil
(likun.tech `sc24_lorastencil.pdf`) and FlashFFTStencil (likun.tech
`ppopp25_FlashFFTStencil.pdf`). AN5D (arXiv 2001.01473) and SPIDER (arXiv
2506.22035) were re-read for their metric definitions. Each artifact's print
statement was checked at the commit pinned in `source.provenance`.

| paper | its own performance metric | precision / platform in the paper |
|---|---|---|
| AN5D (CGO'20) | **GFLOP/s** from Table 3's FLOP/Cell (star2d 8x+1, box2d 2(2x+1)^2-1, star3d 12x+1, box3d 2(2x+1)^3-1, named kernels 10-54); GCell/s on the second axis; mean of 5 after 1 warm-up | fp32 + fp64, P100/V100 |
| ConvStencil (PPoPP'24) | **GStencils/s**, Eq. 16 = T·ΠN / (t·1e9) | fp64, A100 |
| LoRAStencil (SC'24) | **GStencil/s**, Eq. 18 = T·ΠN / (t·1e9) | fp64, A100 |
| FlashFFTStencil (PPoPP'25) | **total execution time t and GStencil/s** together; headline = speedup and time (Fig. 6) | fp64, A100 + H100 |
| SPIDER (PPoPP'26) | **GStencils/s** (points updated per second); its figures divide the fp16 result by 4 and scale the radius-7 run by 7/r | fp16 (baselines fp64, normalized /4), A100/H100 |

This corrects two statements made earlier in this file:

1. **Divergence 4's "unexplained per-shape factor" is temporal fusion.** In
   ConvStencil and LoRAStencil, `times` counts kernel launches. For radius-1
   shapes each launch is a 3-step fused kernel (halo 3; `main.cu` builds the
   7x7 weights as the 3-fold self-convolution of the 3x3), so `times*3` is
   the real number of steps. The paper's README example reproduces this:
   box2d1r at 10240^2 with 10240 launches, 17109 ms, prints 188.27, and
   Eq. 16 with T = 30720 real steps gives 188.28. GStencil/s and
   GCell-updates/s are therefore the same quantity for four of the five
   papers. Only AN5D's primary metric (GFLOP/s) is a different unit.
2. **The Tensor-Core lineage's cross-precision comparisons use the papers'
   own normalizations**: FP16 throughput divided by 4, and FlashFFTStencil
   doubles LoRAStencil's time. The harness shows these as `derived` values
   next to the measured ones and never uses them to rank.

Harness: `kernelbench/domains/stencil.py::_native_stencil` recomputes each
paper's metric from the timed region into `metrics.paper_native`. The metrics
it cannot collect (ncu counters, models, baseline speedups) are listed as
`not_collected` with a reason.

---

## Baselines each paper ran (re-read 2026-09-25)

Each integrated paper's evaluation and artifact scripts were re-read to find
which baselines it compared against and HOW it ran them, so the harness can
run each paper against its own baselines (spec.yaml `paper_baselines`,
`kernelbench/domains/stencil.py` `PAPER_BASELINES`). Repos were read at the
commits pinned in each `source.provenance`, plus the baseline repos they
point to.

| paper | baselines it names | how the artifact runs them | integrated here |
|---|---|---|---|
| AN5D (CGO'20) | Loop tiling (PPCG), Hybrid tiling (PPCG, brute-force tuned over 10/5K configs), STENCILGEN (fixed parameters); Sconf vs Tuned is its own ablation | README points to khaki3/StencilBench@const (`make_ppcg.sh`, `build_hybrid.sh`: PPCG generates the CUDA at build time, none checked in) and khaki3/IEEE2017 (STENCILGEN CUDA for j2d5pt, j2d9pt, j2d9pt-gol, j3d27pt, grad-2d, j3d7pt, ...) | none: PPCG needs the same clang <= 3.8 as AN5D's generator; STENCILGEN covers only the named kernels, which the an5d adapter does not bridge |
| ConvStencil (PPoPP'24) | cuDNN, AMOS, Brick, DRStencil, TCStencil | `src/cudnn/conv_{1d3p,1d5p,box2d9p,box2d49p,box3d27p}.cu`: one `cudnnConvolutionForward` per step, ping-pong buffers, IMPLICIT_PRECOMP_GEMM hard-coded, zero padding, fp64, `std::chrono` around the whole loop | cuDNN (`cudnn-stencil`) |
| LoRAStencil (SC'24) | cuDNN, AMOS, Brick, DRStencil, TCStencil, ConvStencil | no baseline code in its repo; SPIDER's scripts run it next to ConvStencil's cuDNN programs | cuDNN, ConvStencil |
| FlashFFTStencil (PPoPP'25) | cuFFT, cuDNN, Brick, DRStencil, TCStencil, ConvStencil, LoRAStencil (time x2) | `benchmarks/cufft-by-pytorch/rfft{1,2,3}D.py`: `irfft2(rfft2(x) * rfft2(w, s=x.size()))`; `benchmarks/cudnn/cudnn-test.cpp`: times EVERY forward algorithm (valid padding, 2 warm-up + 100 runs, CUDA events) and reports each | cuFFT (`torch-cufft-stencil`), cuDNN fastest-algorithm (`cudnn-stencil-fastest`), ConvStencil |
| SPIDER (PPoPP'26) | cuDNN, DRStencil, TCStencil, ConvStencil, LoRAStencil, FlashFFTStencil | `scripts/Figure10_run.sh`/`Figure11_run.sh` build and run ConvStencil's cuDNN programs (`cudnn_box2d9p/25p/49p`, fp64) and ConvStencil/LoRAStencil from its own forks (submodules KevinWu2017/ConvStencil @31ef23e, KevinWu2017/LoRAStencil @dfff2c5) at 10240^2 x T=10240; TCStencil is NOT run: `outputs/TCStencil_best_A100.csv` holds pre-recorded A100 numbers | cuDNN, ConvStencil, FlashFFTStencil (via the derived comparison) |

Findings that change earlier statements or matter for fairness:

1. **Two different cuDNN baselines.** ConvStencil's programs (and so
   LoRAStencil's and SPIDER's cuDNN numbers) fix IMPLICIT_PRECOMP_GEMM;
   FlashFFTStencil takes the fastest of all algorithms. They can differ, so
   both are integrated (`cudnn-stencil`, `cudnn-stencil-fastest`).
2. **torch-conv-stencil was never any paper's baseline.** It pads circularly
   into a new tensor every step and lets torch pick the algorithm; it stays
   as a fallback yardstick only.
3. **SPIDER's TCStencil comparison is not a measurement on the reader's
   machine**: its script reads a CSV of A100 results recorded elsewhere.
4. **SPIDER's LoRAStencil fork adds `gpu_box_2d1r`, but with the same
   factorization** (divides by the corner weight `params[0]`), so LoRAStencil
   box kernels still only work for LoRAStencil's own ring-pattern weights;
   SPIDER's LoRAStencil box numbers were measured with those weights.
5. **FlashFFTStencil's repo does have `src/1D` and `src/3D`** at the pinned
   commit 4579ea1 (`1d_main.cu`, `3d_main.cu`); the flashfftstencil adapter's
   docstring says no 3D directory exists. The adapter still wires 2D only.
6. **FlashFFTStencil's 2D driver re-applies one sweep T times**
   (`src/2D/2d_main.cu:150`, input never becomes the next input), so its
   Table 3 "1000 time steps" is 1000 repetitions of one step, not a 1000-step
   recursion. The harness runs it at T=1.

"Same configuration" runs (spec.yaml `shared_configurations`): each paper's
own evaluation point is run by every implementation that supports it, so the
papers are compared with each other and with their baselines on identical
inputs; where an implementation cannot run a point (grid alignment, shape,
dimensionality), it is recorded as unsupported with the reason.
