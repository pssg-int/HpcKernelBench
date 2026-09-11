# Cholesky track — evaluation-methodology survey

Track input: `data/track_inputs/cholesky.json`, 8 papers. All 8 were surveyed via
arXiv/full-text-PDF (exceeding the 5-paper minimum) plus the two open-source
artifact repos the 8 papers collectively point to: `ecrc/hicma-x` (5 papers:
LtaiefACRSKDBAK24, CaoAPBLKD22, CaoAAPNBDGKLS22, CaoPABLKD21, and indirectly
AbdulahCPBDGKLS22 which also builds on it) and `ecrc/exageostat` (4 papers:
MondalAL0GK22, CaoAAPNBDGKLS22, AbdulahCPBDGKLS22, SalvanaAHLSGK21). Both repos
are the same KAUST/ECRC + UTK-ICL lineage (PaRSEC/StarPU runtime, HiCMA
low-rank library, STARS-H covariance-matrix generator, Chameleon dense tile
library), so this track's literature is unusually homogeneous in tooling but
spans four genuinely different Cholesky *variants*: plain dense tile Cholesky
(distributed, communication-optimal), tile low-rank (TLR/HODLR) Cholesky,
mixed-precision (FP64/FP32/FP16/FP8/INT8) tile Cholesky, and a GPU
tensor-core mixed-precision Cholesky+GEMM pipeline (GWAS/KRR). One paper
(KwasniewskiKBZS21) is a different lineage entirely (ETH Zurich COnfLUX/
COnfCHOX, ScaLAPACK-compatible, ScaLAPACK/SLATE/CANDMC/CAPITAL baselines) and
anchors the plain-dense-fp64 end of the spectrum.

## 1. `conf/sc/KwasniewskiKBZS21` — On the Parallel I/O Optimality of Linear Algebra Kernels: Near-Optimal Matrix Factorizations (SC'21)

Source: arXiv fulltext (`arxiv.org/pdf/2108.09337`, 15 pages extracted via
pypdf) + artifact repo `anonymousSC21/SC21` README (the paper's own public,
BSD-3-Clause-licensed successor is `eth-cscs/conflux`, referenced in-text as
`github.com/eth-cscs/conflux`).

- **workloads/inputs**: randomly generated double-precision matrices; for
  Cholesky, SPD (symmetric positive definite). Matrix sizes N = 2,048 up to
  N = 2^19 = 524,288 (paper abstract) / the artifact README's own sweep
  starts at N=4,096. Node counts P = 2 up to 512 CPU nodes (4 to 1,024 MPI
  ranks, 2 ranks/node on the dual-socket machine). Runs where none of the
  four libraries reach >3% of hardware peak are discarded (adding nodes only
  hurts at that point).
- **timing protocol**: `std::chrono` local timers; the **maximum** execution
  time across all MPI ranks is reported per run. The artifact README states
  explicitly: for each (N, P) combination, **four block sizes** are swept per
  library and **five runs** are gathered for statistics (wall-clock time).
  The paper text itself only says results are stable/reproducible, not the
  exact statistic (mean/median/min) computed from the five runs — flagged as
  an open question.
- **timing scope**: kernel-only factorization time; separately, **aggregate
  communication volume** (I/O cost) is measured via the Score-P profiler and
  reported alongside a theoretical I/O-cost model, validated to within ±3%
  of measured values for MKL/SLATE/COnfLUX/COnfCHOX (CANDMC/CAPITAL's
  author-supplied models overapproximate by 30-40%).
- **precision & correctness**: FP64 only. No numeric residual/backward-error
  tolerance stated for the factorization result itself — correctness here is
  validated *for the I/O-cost model*, not the numerical solution (Cholesky
  correctness is implicitly assumed via using a standard, previously-verified
  BLAS/LAPACK-based kernel).
- **metric**: both time-to-solution (wall clock, "up to 3x" reduction vs.
  best baseline) and communication volume (elements moved per processor,
  "up to 1.6x" reduction vs. second-best).
- **baselines**: Intel MKL v19.1.1.217 (ScaLAPACK-style 2D decomposition),
  SLATE (2D, user-specified block size, default 16), CANDMC and CAPITAL
  (asymptotically-optimal 2.5D decomposition, but block size still
  user-specified/hand-tuned by the paper's authors — "performance was
  significantly improved when we tuned the parameters"). COnfLUX/COnfCHOX
  ship out-of-the-box parameters that beat all four hand-tuned baselines.
  This is the clearest evidence in the whole track that **block/tile size is
  a first-class tuning knob that must be disclosed per implementation**, not
  a fixed constant across the benchmark.
- **machine**: CSCS Piz Daint, XC40 partition, 1,813 CPU nodes, dual-socket
  Intel Xeon E5-2695 v4 (2x18 cores), 64 GiB DDR3/node, Cray Aries Dragonfly.

## 2. `conf/ipps/CaoAPBLKD22` — A Framework to Exploit Data Sparsity in Tile Low-Rank Cholesky Factorization (IPDPS'22)

Source: fulltext PDF (`netlib.org/.../framework-low-rank-ipdps-2022.pdf`, 11
pages extracted via pypdf, "VIII. Performance Results and Analysis" section)
+ artifact repo `ecrc/hicma-x` (`TESTS.md`, `scripts/test_run_check_correctness.sh`).

- **workloads/inputs**: 3D unstructured-mesh-deformation RBF interpolation
  problem built from the SARS-CoV-2 virion geometry (PDB 6VXX); number of
  virus copies in a 1.7μm cube swept from 30 (N=1.49M mesh points/matrix
  dim) to 1,200 (N=52.57M). A free RBF "shape parameter" (O(1e-4) to O(1e-2))
  independently controls how sparse/dense the compressed operator is.
- **tile size / accuracy threshold as tuning knobs**: tile size b is
  explicitly **not** fixed — the paper follows the heuristic b = O(sqrt(N))
  found empirically per matrix (Section VIII-C, "Understanding the Impact of
  Tile Size"; concrete pairs used: N=4.49M→tile 2390, N=2.99M→tile 2440,
  N=1.49M→tile 4880), noting "auto-tuning the tile size with a model is
  beyond the scope of the paper." The TLR compression accuracy threshold
  defaults to 1e-4 but is explicitly swept at 1e-5/1e-7/1e-9 in a dedicated
  ablation (Fig. 12).
- **timing protocol**: "we run our experiments **at least three times** and
  since no noticeable performance variability has been identified, the
  **minimum** time to solution is reported." Repository's `TESTS.md`
  exposes `--nruns` as a first-class CLI flag on the `testing_potrf_tlr`
  binary.
- **timing scope**: pure factorization time-to-solution; the matrix
  compression step (dense→TLR) is measured and reported *separately* as a
  time breakdown (Fig. 11) — it is explicitly noted this compression step
  can dominate once the factorization itself is fast enough.
- **precision & correctness**: double precision (fp64) throughout this
  paper. Correctness is enforced structurally via the accuracy threshold on
  compressed tiles; the artifact's `--check` CLI flag (used in every example
  command in `TESTS.md`/`test_run_check_correctness.sh`) triggers a
  numerical backward-error validation against the threshold at runtime.
- **metric**: time-to-solution and speedup vs. Lorapo (state-of-the-art TLR
  Cholesky baseline): up to 6.8x on Shaheen II, 9.1x on Fugaku. Also a
  roofline/critical-path efficiency metric (>70% of the theoretical
  critical-path bound up to N=11.95M on Shaheen II).
- **baselines**: Lorapo. Solves a formally-dense 3D problem with 52M mesh
  points on 65K cores in ~30 minutes.
- **machines**: Shaheen II (6,174 nodes, dual 16-core Intel Haswell, 128GB
  DDR4/node) and Fugaku (158,976 nodes, 48-core A64FX/node).

## 3. `journals/tpds/AbdulahCPBDGKLS22` — Accelerating Geostatistical Modeling and Prediction With Mixed-Precision Computations: A High-Productivity Approach With PaRSEC (TPDS'21/22)

Source: fulltext PDF (`netlib.org/.../geostat-parsec-tpds-2021.pdf`, 12 pages
extracted via pypdf, "7 Performance Results and Analysis" section) + artifact
repo `ecrc/exageostat`.

- **workloads/inputs**: synthetic Monte-Carlo 2D irregular-location datasets
  (100 realizations × 40K locations each, powered-exponential covariance,
  varying correlation strength and field smoothness); real datasets: US Soil
  Moisture (Mississippi River Basin, random 1M-location subset of a 2.43M
  grid) and Middle-East Wind Speed (Arabian Sea subset, 116,100 locations,
  derived from a WRF simulation).
- **precision as a named, discrete axis**: precision is not "fp64 vs fp32"
  binary but a **named band-percentage vector** "`a`D:`b`S:`c`H" (percent of
  matrix bands computed in Double/Single/Half), e.g. `10D:90S`, `1D:99H`,
  swept explicitly as an independent variable across every performance
  figure. BAND_SIZE per precision level is itself tuned per-precision
  ("we tried several band sizes for each precision and kept only the ones
  showing some difference").
- **timing protocol**: not stated with an explicit warmup/repetition count
  for the Tflop/s scaling figures (large-node-count runs where each point is
  a full at-scale execution) — flagged as an open question; the qualitative
  Monte-Carlo accuracy study *does* have an explicit repetition count (100
  realizations, boxplot statistics), which is a different notion of
  "repetition" (statistical, not timing-noise averaging).
- **timing scope**: full factorization wall time at scale; no separate
  preprocessing/matrix-generation timing breakout found in the extracted
  text.
- **precision & correctness**: mixed FP64/FP32/FP16 (BLAS/LAPACK backends:
  MKL on Shaheen II, AOCL on HAWK, ESSL+CUDA on Summit/GPU). **Correctness is
  validated by downstream statistical quality, not raw numerical residual**:
  parameter-estimation boxplots vs. ground-truth θ, log-likelihood value,
  Mean-Square-Prediction-Error (MSPE), prediction uncertainty, and
  iteration-count-to-converge — all compared across precision configurations
  against the pure-FP64 (`100D`) reference. A precision configuration is
  accepted if these statistical quantities stay close to `100D`, not if a
  per-element error falls below a fixed tolerance.
- **metric**: performance in Tflop/s and PFlop/s; speedup vs. pure `100D`:
  1.56-1.59x (mixed DP/SP on HAWK/Shaheen II), 2.05-2.07x (SP-only vs. DP),
  up to 2.64x (mixed DP/SP/HP on Summit, 9.1 PFlop/s peak, ≈2.06x of the
  machine's DP-Linpack performance).
- **baselines**: pure-DP (`100D`) same-code-path Cholesky; also compared
  against two StarPU-based codes (MOAO_StarPU, ExaGeoStat_StarPU) to isolate
  the PaRSEC-vs-StarPU runtime contribution (1.46x-1.53x speedup from the
  runtime alone, same precision).
- **machines**: Shaheen II, HAWK (5,632 nodes, dual 64-core AMD EPYC 7742,
  256GB/node), Summit (4,356 nodes, dual 22-core Power9 + 3x V100 GPU/node).

## 4. `conf/sc/CaoAAPNBDGKLS22` — Reshaping Geostatistical Modeling and Prediction for Extreme-Scale Environmental Applications (SC'22, Gordon Bell finalist)

Source: fulltext PDF (`marcgenton.github.io/2022.CAAPNBDGKLS.GB.pdf`, 12
pages extracted via pypdf) + artifact repo `ecrc/exageostat`.

- **workloads/inputs**: same MLE geostatistics pipeline as #3, pushed to
  extreme scale: weak-scaling matrix sizes up to ~9M (the largest matrix that
  fits in pure FP64 given per-node HBM2), strong-correlation real-dataset
  runs at 4,096 and 48,384 Fugaku nodes (2,322,432 cores — the largest scale
  in this entire track).
- **tile size and accuracy threshold as disclosed knobs**: tile size 800
  used for the headline 1,024-node mixed-precision comparison (Fig. 7); TLR
  compression accuracy tolerance fixed at 1e-8 for the combined
  mixed-precision+TLR variant.
- **precision**: combines *both* axes other papers treat separately — dense
  mixed FP64/FP32/FP16 **and** TLR compression, chosen per-tile at runtime by
  a "tile-centric, precision-aware, adaptive" decision rule that thresholds
  the ratio of the local machine epsilon to the tile's actual numerical
  content (Section IV-C). FP16 GEMM on Fugaku's A64FX only supports pure
  FP16 accumulation, not FP32-accumulated mixed HGEMM (a hardware
  constraint, not a design choice) — an important platform caveat for any
  A64FX reproduction.
- **timing protocol**: not explicitly stated with a warmup/rep count for the
  at-scale weak/strong-scaling headline numbers — flagged open question
  (consistent with #3: Gordon-Bell-style extreme-scale allocations typically
  report single/few large runs, not a repeated-timing-loop protocol).
- **correctness**: same two-track validation as #3 — real-dataset parameter
  estimation / MSPE quality vs. dense-FP64 baseline (Section on "Real
  Dataset Accuracy Analysis"), plus a separate microbenchmark-level
  correctness check of the FP64-TLR-vs-dense-FP64 GEMM kernel itself
  (Section on "GEMM Performance Evaluation on A64FX", Fig. 5-6).
- **metric**: PFlop/s at scale; speedup vs. dense-FP64 baseline — up to 12x
  (weak scaling, 16K Fugaku nodes), roughly an order of magnitude at
  4,096/48,384-node strong-correlation runs. Memory footprint reduction
  reported separately (up to 79% reduction for the combined MP+TLR variant
  vs. dense FP64).
- **baselines**: dense FP64 Cholesky (reference), and the un-optimized
  precision/structure-agnostic runtime decision from prior work.
- **machines**: Fugaku only (single-node GEMM microbenchmarks up to
  48,384-node full-application runs); one accuracy-only cross-check on
  Shaheen II.

## 5. `conf/ipps/CaoPABLKD21` — Leveraging PaRSEC Runtime Support to Tackle Challenging 3D Data-Sparse Matrix Problems (IPDPS'21)

Source: fulltext PDF (`netlib.org/.../leverage-parsec-ipdps-2021.pdf`, 11
pages extracted via pypdf, "VIII. Performance Results and Analysis" section)
+ artifact repo `ecrc/hicma-x`.

- **workloads/inputs**: synthetic 2D/3D statistics covariance problems
  generated via STARS-H; primary sweep at N=1.08M and N=2.16M, weak-scaling
  up to N=11.88M×11.88M on 2,048 nodes (the largest CPU-only weak-scaling
  point in this track outside the SC22 Fugaku paper).
- **tile size and BAND_SIZE as explicit, ablated tuning knobs**: this is the
  paper that *introduces* BAND_SIZE auto-tuning (transparent to the user,
  Fig. 6 shows the full auto-tune-vs-manual ablation) and dedicates a whole
  subsection (VIII-C, "Suitable Tile Size Selection") to sweeping tile size
  explicitly at N=1.08M/64 nodes and N=2.16M/256 nodes, following the same
  b=O(sqrt(N)) heuristic as #2/#4.
- **accuracy threshold**: default 1e-8 (except an explicit ablation over
  1e-3/1e-5/1e-7/1e-9 in Section VIII-G, "Evaluation of Different Accuracy
  Thresholds").
- **timing protocol**: identical language to #2 — "we run our experiments
  **at least three times**; and since no major performance variability has
  been noticed, the **minimum** time to solution is reported."
- **precision & correctness**: double-precision only. **Explicit numerical
  backward-error statement**: "Numerical backward errors have been
  consistently validated against the application accuracy threshold to
  ensure correctness... an accuracy threshold of 1e-8 ... ultimately yields
  an absolute numerical error of order 1e-9 in the solution of the linear
  system." This is the most concrete residual/backward-error statement found
  in the whole track.
- **metric**: time-to-solution, memory footprint per node (9.31 GB for an
  8.64M matrix on 512 nodes), fraction of sustained Linpack peak.
- **baselines**: prior PaRSEC-HiCMA implementation (ablation, not a
  different library); abstract claims up to 7x speedup from the new
  runtime-level optimizations.
- **machine**: Shaheen II only, MKL 2019.5, process grid P×Q (P≤Q, square as
  possible), 2DBCDD.

## 6. `conf/sc/LtaiefACRSKDBAK24` — Toward Capturing Genetic Epistasis From Multivariate GWAS Using Mixed-Precision Kernel Ridge Regression (SC'24, Gordon Bell finalist)

Source: arXiv fulltext (`arxiv.org/html/2409.01712v1`) + artifact repo
`ecrc/hicma-x` (same repo as #2/#5, extended with a `testing_KRR` /
"Genomics" build target — see `TESTS.md`).

- **workloads/inputs**: real UK BioBank GWAS, 305,880 patients × 43,333 SNPs
  (the KRR kernel-ridge-regression covariance matrix is N=305,880); synthetic
  scaling datasets (msprime-generated + randomly generated) up to 13M
  patients × 20M SNPs.
- **machines**: this is the only paper in the track that is GPU-first and
  spans four different GPU generations/vendors at massive scale — Summit
  (18,432 V100 GPUs, 2/3 of the machine), Leonardo (4,096 A100 GPUs, 1/3),
  Frontier (36,100 MI250X GPUs, nearly the full machine), Alps (8,100 GH200
  Superchips, up to 1,936 nodes — the peak-throughput configuration).
- **precision**: "tile-centric adaptive precision" Cholesky/GEMM spanning
  FP64/FP32/FP16/FP8, *plus* a distinct use of INT8 tensor cores specifically
  for the KRR kernel-matrix-generation step (Euclidean-distance computation
  from integer-valued SNP genotypes, kept segregated from the floating-point
  confounder GEMM operands) — a genuinely different mixed-precision pattern
  from #3/#4's band-percentage scheme, worth keeping as its own evidence
  point rather than merging.
- **timing protocol**: not recoverable from the extracted text with an
  explicit warmup/rep/statistic — flagged open question, same caveat as
  #3/#4 (Gordon-Bell-scale full-machine allocations).
- **metric**: mixed-precision **ExaOp/s** (not Flop/s) — 1.805 ExaOp/s peak
  sustained (Alps, near-full-machine), 2.109 ExaOp/s for the KRR "build"
  (matrix-generation) phase specifically; per-GPU sustained rates given too
  (~57 TFlop/s Leonardo, ~159 TFlop/s Alps).
- **correctness**: validated indirectly via downstream statistical quality —
  Mean-Square-Prediction-Error and Pearson correlation of KRR-predicted
  phenotypes vs. ground truth across five diseases, compared against a plain
  Ridge-Regression baseline (not a numerical-residual check on the Cholesky
  factor itself).
- **baselines**: plain (non-kernel) Ridge Regression for statistical-quality
  comparison; implicitly, pure-FP64 Cholesky for the performance baseline
  (ExaOp/s vs. FP64-only is the framing of the "mixed-precision" headline
  number, though the exact FP64-only comparison figure wasn't captured in
  this extraction pass).

## 7. `journals/tpds/SalvanaAHLSGK21` — High Performance Multivariate Geospatial Statistics on Manycore Systems (TPDS'21)

Source: arXiv fulltext (`ar5iv.arxiv.org/html/2008.07437`) + artifact repo
`ecrc/exageostat`.

- **workloads/inputs**: bivariate (p=2) and trivariate (p=3) synthetic
  Matérn-covariance datasets, n=22,464-24,964 locations (npred=2,500 held out
  for prediction); real Middle-East wind-speed dataset, n=116,100
  (npred=11,610, Arabian Sea subset, WRF-derived — same real dataset as #3).
- **tile size**: nb=720 used for the headline 7,200×7,200 bivariate-matrix
  example, explicitly framed as "a trade-off between the arithmetic
  intensity of the kernel and the degree of parallelism" (same framing as
  the b=O(sqrt(N)) heuristic elsewhere in the track, though not spelled out
  as a formula here).
- **timing protocol**: **explicit and unambiguous** — "we run each
  simulation **three times** on every single hardware with the same
  configuration and **report the average**." Measured variability: 0.1-0.5%
  on shared-memory systems, 1-3% on distributed-memory. This is the one
  paper in the track that states both an exact repetition count *and* the
  statistic (mean, not min/median) in one sentence.
- **timing scope**: full MLE computation (Cholesky-based); no separate
  preprocessing/generation breakout stated.
- **precision & correctness**: double precision (fp64) throughout — no
  mixed-precision variant in this paper (predates #3/#4's mixed-precision
  work by the same group). Correctness validated via: (a) parameter
  estimation boxplots vs. ground truth, (b) MSPE, (c) a novel **multivariate
  MLOE/MMOM criterion** introduced by this paper specifically for assessing
  multivariate prediction quality, and (d) approximation-level comparison
  across Exact / TLR (TLR5=1e-5, TLR7=1e-7, TLR9=1e-9) / Diagonal Super-Tile
  (DST 40/60, DST 70/30) configurations.
- **metric**: wall time (seconds) and speedup vs. exact dense MLE — 4x
  (Skylake), 4.3x (Cascade Lake), 6x (AMD EPYC Rome) on shared memory; 66.7x
  reported at distributed scale. Memory-footprint reduction for TLR variants:
  3.86x-6.68x vs. exact.
- **baselines**: exact (dense) computation; DST approximation family (a
  cheaper, cruder alternative to TLR).
- **machines**: five different shared-memory CPU/GPU node types (Skylake
  28-core, Cascade Lake 20-core, AMD EPYC Rome 64-core, Skylake+4xV100,
  ARM ThunderX2 32-core) plus Shaheen II distributed (64, 128 nodes tested).

## 8. `conf/ipps/MondalAL0GK22` — Parallel Approximations of the Tukey g-and-h Likelihoods and Predictions for Non-Gaussian Geostatistics (IPDPS'22)

Source: fulltext PDF (`marcgenton.github.io/2022.MALSGK.IPDPS.pdf`, 11 pages
extracted via pypdf) + artifact repo `ecrc/exageostat`.

- **workloads/inputs**: synthetic non-Gaussian (Tukey g-and-h) random fields
  at N=8,100 (tile size ts=810, HODLR leaf=100) and N=32,400 (ts=3,240,
  leaf=400); a real dataset with n=358,303 locations; distributed
  weak-scaling from N=562,500 up to N=800K on up to 512 Shaheen-II nodes.
- **approximation-method axis, not just a single "TLR vs exact" split**:
  three families are compared head-to-head — **Exact** (dense), **TLR**
  (TLR-5/7/9 = accuracy 1e-5/1e-7/1e-9), and **HODLR** (HODLR-5/7/9, a
  different hierarchical compression with its own leaf-size parameter
  instead of a flat tile size). TLR beats HODLR by 6.46x-6.82x depending on
  accuracy level and CPU, and the ranking is sensitive to core count — an
  important cross-method comparison other papers in the track don't make.
- **timing protocol**: not recoverable from the extracted text with an
  explicit warmup/rep/statistic (open question) — but concrete wall-clock
  numbers are reported (e.g., exact MLE on 562,500 locations: 255.97 seconds
  on 128/256/512 Shaheen-II nodes, implying weak scaling rather than a
  single-node number).
- **precision & correctness**: double precision. Correctness validated via
  boxplots of the six estimated TGH parameters (ξ,ω,g,h,φ,ν) for
  Exact/TLR/HODLR across accuracy levels vs. the exact MLE reference — a
  parameter-recovery check, not a raw factorization residual.
  compute-intensive.
- **metric**: execution time (seconds); speedup up to 7.29x (shared-memory)
  and 2.96x (distributed) for TLR vs. exact.
- **baselines**: Exact (dense) MLE; HODLR (cross-method baseline, not a
  competing paper/library).
- **machines**: shared-memory (28-core Intel IceLake Gold 6330, 64-core AMD
  EPYC "Milan"), distributed Shaheen-II (64/128/256/512 nodes).

## Artifact repos (referenced by 7 of 8 papers)

- **`ecrc/hicma-x`** (HiCMA-PaRSEC): C, PaRSEC-based, submodules
  `dplasma`/`hcore`/`stars-h`. 10 stars, custom/"Other" license (not a
  standard OSI license — flagged). Testing binary `tests/testing_potrf_tlr`
  exposes every tuning knob discussed above directly as CLI flags: `--N`,
  `--NB` (tile size), `--fixedacc` (TLR accuracy threshold), `--maxrank`,
  `--check` (correctness gate), `--nruns`, `--band_dense`/`--auto_band`
  (BAND_SIZE), `--gpus`, `--adaptive_decision`/`--adaptive_memory` (the
  mixed-precision runtime decision from #4/#6). `scripts/` contains
  ready-to-run examples (`test_run_check_correctness.sh`,
  `test_cpu_4node.sh`, `test_gpu_1node.sh`, `test_tile_size_dense.sh`,
  install scripts for Shaheen/Fugaku/Summit/Frontier/Leonardo/Polaris).
- **`ecrc/exageostat`**: C, BSD-3-Clause, 40 stars, **archived** — the
  repo's own README states active development moved to `ecrc/ExaGeoStatCPP`
  (a C++ rewrite). Depends on StarPU, HiCMA, Stars-H, Chameleon as
  submodules. Test drivers under `examples/` are named by workload/precision
  (`synthetic_dmle_test.c`, `synthetic_smle_test.c` [single-precision],
  `real_csv_dmle_test.c`, etc.) — precision is baked into the binary name
  rather than a runtime flag, unlike hicma-x's `--adaptive_decision`.
- **`anonymousSC21/SC21`** (Kwasniewski et al.'s artifact, listed as
  `verified` in the track input): an anonymized-submission snapshot repo
  (created 2021-04, last pushed 2021-06, **no license file**, 0 stars) — not
  actively maintained. The paper's own text points to the real, maintained,
  BSD-3-Clause-licensed successor `eth-cscs/conflux` (12 stars, last pushed
  2021-08); **the spec/harness should target `eth-cscs/conflux`**, using the
  anonymous repo only as the paper's documented reproduction-package
  reference for the exact experimental protocol (block-size sweep count,
  five-runs-per-config).

## Divergences

- **Correctness notion splits into two genuinely different practices**: the
  plain-dense/TLR CPU papers from the HiCMA-PaRSEC lineage (#1, #2, #5)
  validate a **numerical backward error** against the TLR accuracy threshold
  (#5 gives a concrete number: 1e-8 threshold → ~1e-9 absolute solution
  error) — this is the residual/backward-error notion the design
  instructions ask for. The geostatistics/GWAS application papers (#3, #4,
  #6, #7, #8) instead validate **downstream statistical quality** (parameter
  recovery, log-likelihood, MSPE, MLOE/MMOM, prediction Pearson correlation)
  against a pure-FP64 or exact-dense reference, because for their
  application the quantity that matters is the *inference result*, not the
  Cholesky factor's per-element error. **The spec keeps a numerical
  residual/backward-error gate as the primary, required correctness check
  for every variant** (closer to #1/#2/#5's practice and to this project's
  general fairness principles), and treats the statistical-quality checks as
  a track-specific *secondary* metric to report for the
  mixed-precision/TLR variants, since silently dropping them would hide a
  real methodological signal this track's own papers care about.
- **Timing statistic has no single convention**: minimum-of->=3-runs (#2,
  #5, and the Kwasniewski I/O-cost-model validation implicitly), mean-of-3
  (#7, explicit), five-runs-unspecified-statistic (#1's own artifact
  README). No paper reports median. Best-of-N (minimum) is the most common
  practice in this track and is optimistic (reports the luckiest run). The
  spec fixes this to **median, with min/max also reported**, consistent with
  this project's general fairness principles and diverging from the track's
  own modal practice (minimum).
- **Tile/block size is explicitly a tuning knob, disclosed inconsistently**:
  #1 shows block size is user-specified for 3 of 4 baselines (MKL, SLATE,
  CANDMC/CAPITAL) and only COnfLUX/COnfCHOX ship good out-of-the-box
  defaults; #2, #5 dedicate whole subsections to sweeping tile size and
  settle on an empirical b=O(sqrt(N)) heuristic re-derived per matrix; #4
  uses a fixed 800 for one headline comparison without re-deriving it; #7
  states the trade-off qualitatively without a formula. No paper gives a
  single formula that transfers across matrix sizes/machines with a fixed
  constant. **The spec treats tile size as a required, per-run disclosed
  parameter** (never silently fixed), following #1/#2/#5's practice of
  explicit tile-size ablation.
- **Accuracy threshold (TLR/HODLR) values cluster but aren't identical
  across papers**: 1e-4 (#2 default), 1e-8 (#2's `--check` examples, #4's
  headline, #5's default), swept over {1e-3, 1e-5, 1e-7, 1e-9} (#5, #8). The
  spec's TLR/low-rank variant makes accuracy threshold an explicit,
  multi-point-swept axis rather than a single fixed value, mirroring #5 and
  #8's own practice (the two papers that treat it most rigorously).
- **Precision-scheme representation differs structurally between the
  dense-mixed-precision papers**: #3/#4 use a *band-percentage* vector
  ("`a`D:`b`S:`c`H") applied uniformly to sub/super-diagonal bands; #6 uses a
  genuinely different *tile-centric adaptive* decision (per-tile, based on
  local numerical content vs. machine epsilon) plus a separate INT8 axis for
  a non-Cholesky sub-kernel (distance/GEMM). The spec's mixed-precision
  variant requires the precision-assignment *strategy* itself (not just the
  resulting FP64/FP32/FP16/FP8 fractions) to be disclosed per run, since #1's
  block-size lesson generalizes: an undisclosed tuning/assignment strategy
  makes cross-paper comparison meaningless.
- **Single-node/shared-memory vs. distributed scope**: only #6 (GPU,
  multi-node from the start) and part of #7 (explicit shared-memory sweep
  across 5 CPU types before a distributed Shaheen-II sweep) report
  single-node numbers at all; #1, #2, #3, #4, #5, #8 are distributed-only in
  their reported results (though #2/#5's underlying binary can run on 1
  node). Given the track's explicit request for single-node and distributed
  as separate variants, and that most of this track's own papers only
  report the distributed regime, **the spec's single-node variant is
  smaller in scope than the distributed variant** (matches what the
  literature actually measures) but is kept as a first-class variant since
  it is the more reproducible entry point for a Phase-2 harness without
  requiring a multi-node allocation.
- **Metric units diverge with scale, not with disagreement**: Gflop/s (#1
  implicitly via time+known FLOP count), Tflop/s/PFlop/s (#3, #4), ExaOp/s
  (#6) — this is a scale artifact (small vs. Gordon-Bell-scale problems), not
  a real methodological disagreement, so the spec reports Gflop/s uniformly
  and lets the number's magnitude speak for the scale.

## Open questions

- Exact statistic (mean/median/min) computed over #1's artifact-documented
  "five runs" is not stated in either the paper text or the artifact README
  — would need `eth-cscs/conflux`'s actual benchmark-driver source (not
  fetched in this pass) to confirm.
- Warmup/repetition/statistic protocol for the at-scale Tflop/PFlop/ExaOp
  headline numbers in #3, #4, #6 was not recoverable from the extracted PDF
  text (these are Gordon-Bell-class full-machine-allocation runs, where a
  repeated-timing-loop protocol may not even be practical/affordable — an
  open question whether these papers report a single run per configuration
  for that reason).
- #8's exact timing protocol (warmup/rep/statistic) likewise not
  recoverable from the extracted text, despite concrete wall-clock numbers
  being reported.
- Whether hicma-x's `--check` flag (backward-error validation) is timed
  inside or excluded from the reported time-to-solution was not confirmed
  from the source excerpts read (the CLI examples always pass `--check`
  together with performance-relevant flags, suggesting it may run as part
  of the same invocation) — needs the `testing_potrf_tlr.c` driver source
  to confirm before the harness can safely separate it from the timed
  region.
- #6's exact FP64-only baseline ExaOp/s figure (for the "up to 12x
  mixed-precision speedup" style comparison other papers in this family
  report) was not captured in the html extraction pass — the abstract states
  "1.8 mixed-precision ExaOp/s" but not the matching pure-FP64 number.
- `ecrc/hicma-x`'s license is a custom "Other"/NOASSERTION license (not a
  standard OSI license) — worth flagging to whoever runs artifact-status
  verification for this track, though the repo is otherwise clearly
  intended as open/public (public GitHub org, cited across 5 of this
  track's own papers as "available as open-source").
