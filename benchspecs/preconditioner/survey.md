# Preconditioner track — evaluation-methodology survey

Track input: `data/track_inputs/preconditioner.json`, 5 papers. All 5 surveyed;
none have an arXiv id in the track metadata, so evidence is primarily from
artifact-repo code (build scripts, driver `main()`s, timing macros/classes) —
which is usually more reliable than paper prose anyway, since it shows the
actual warmup/repetition/timer logic. One paper (BootCMatchGX) turned out to
have an arXiv preprint under a different id (found via web search) and its
fulltext was read directly.

---

## 1. DBMPK — "A Diagonal Block Memory-Aware Polynomial Preconditioner for
Linear and Eigenvalue Solvers" (PPoPP 2026)

Source: `github.com/YXJ-123/DBMPK`, files `DBMPK.cpp` (driver/timing loop),
`DBMPK.hpp` (timer macros), `Makefile`, `README.md`. No arXiv/OA fulltext
available (ACM DL paywalled, PPoPP 2026 not yet indexed elsewhere); all facts
below are from the code, cross-checked against the abstract's headline numbers.

- **What's measured**: the matrix-power kernel (MPK) `A^d x` at the core of a
  polynomial preconditioner — NOT a full solver run. Three variants compared
  in the same binary: MKL `mkl_sparse_d_mv` applied `d` times (baseline),
  the authors' plain (non-block) MPK, and DBMPK (diagonal-block MPK).
- **Workload**: single input matrix per run, given via `-in <path>` (Matrix
  Market, general or symmetric — code auto-detects and expands symmetric
  storage). Default polynomial degree `d=20` (`-d` flag); README example uses
  `-d 20`. No fixed matrix list ships in the repo — user supplies matrices.
  This is an **open question**: the paper's exact matrix suite is not
  recoverable from the artifact.
- **Precision**: fp64 throughout (`double* values`, MKL `mkl_sparse_d_mv` is
  the double-precision entry point).
- **Timing protocol**: `gettimeofday`-based macros (`INIT_TIMER`/`START_TIMER`
  /`STOP_TIMER`/`GET_TIMER`, wall-clock, microsecond resolution). Fixed
  `iterations = 50` (a commented-out adaptive-iteration-count line is dead
  code). For each of MKL/MPK/DBMPK: loop 50 times, accumulate elapsed time,
  divide by 50 → **mean**, not median. One un-timed warm-up-like call
  ("firstPower") exists but its result is unused (its only purpose was
  computing the now-dead adaptive iteration count) — so effectively **no
  warmup**.
- **Timing scope**: kernel-only per-iteration application time. DBMPK's
  one-shot setup (`kernel.Setup(A, poly_d)` — partitioning into diagonal
  blocks) is timed separately (`setup_time`) and reported, not amortized into
  the per-iteration numbers. This matches the fairness principle already.
- **Correctness**: **not checked in the code** — `bb_plain` (baseline MPK
  output) and `bb_dbmpk` (DBMPK output) are computed but never compared. This
  is a gap relative to good practice; the spec should add a correctness gate.
- **Metric**: GFLOP/s = `2*d*iterations*nnz*1e-9 / solve_time`, plus raw
  time/iteration in seconds. Effectively 2 flops per nonzero per polynomial
  step (one SpMV-equivalent per step).
- **Baseline**: Intel MKL sparse BLAS (`mkl_sparse_optimize` + repeated
  `mkl_sparse_d_mv`), i.e. the vendor SpMV library applied `d` times per
  polynomial-preconditioner application — not a specialized "matrix-power
  kernel" library.
- **Iteration-count effect**: not measured in this artifact at all — this is
  a raw-kernel microbenchmark; the paper's claimed 18.6–34.0% "end-to-end"
  speedup (per the abstract) refers to full CG/eigenvalue solves elsewhere in
  the paper, not exercised by this repo.

## 2. Graphene — "Accelerating Sparse Linear Solvers on Intelligence
Processing Units" (IPDPS 2025)

Source: `github.com/esa-tu-darmstadt/graphene`, files `README.md`,
`applications/benchmark/benchmark.cpp` (driver), `data/benchmarks/benchmark.py`
(sweep harness), `data/benchmarks/ir_pbicgstab_ilu0.jsonc` (example solver
config). No arXiv id; IEEE Xplore paywalled; evidence from repo only.

- **Platform**: Graphcore IPU (not CPU/GPU) — out of scope for direct
  head-to-head timing against the other 4 CPU/GPU papers in this track, but
  its solver-config protocol is still informative for the spec's
  "iterations reported" axis.
- **Workload**: driver accepts either `--matrix <file>` (any Matrix Market
  file) or `--poisson <nx>,<ny>,<nz>` (synthetic 3D 7-point Poisson,
  generated in-code). The sweep harness (`benchmark.py`) names concrete
  SuiteSparse instances actually used: `Hook_1498`, `G3_circuit`,
  `Geo_1438`, `af_shell7`, plus synthetic Poisson grids; IPU tile counts
  swept (e.g. 5888, up to `1472*n` per IPU).
- **Precision**: primary fp32 (`Type::FLOAT32`); the framework additionally
  supports extended precision (`twofloat32` double-word arithmetic, or a
  software fp64 emulation `float64`) used specifically inside the
  Mixed-Precision Iterative-Refinement (MPIR) solver — i.e. this paper's
  precision axis is itself part of its contribution (native IPU has no fp64).
- **Solver/preconditioner config** (`ir_pbicgstab_ilu0.jsonc`): outer
  Iterative Refinement, `relResidual: 1e-9`, `maxIterations: 3`, extended
  precision correction in `twofloat32`; inner solver PBiCGStab with ILU
  preconditioner, `relResidual: 1e-3`, `maxIterations: 100`. Both inner and
  outer iteration counts and residuals are printed
  (`printPerformanceAfterSolve: true`) — i.e. iteration-count-to-convergence
  is a first-class, always-reported output, not an afterthought.
  Preconditioner setup (ILU factorization) is a config-selectable component,
  separately identifiable in the profiling output directory structure
  (`profiling/<config>_<matrix>_<tiles>tiles/`).
- **Timing protocol**: delegated to GraphCore's Poplar profiler (external
  tool, invoked via `-d <profile_dir>`); the repo's own driver does not
  contain a manual timing loop — no visible warmup/repeat/statistic policy
  in-repo (open question: Poplar profiler's own averaging policy is opaque
  from this artifact alone).
- **Baselines**: none built into this repo's benchmark harness (no
  CPU/cuSPARSE/AmgX comparison code found); cross-hardware comparison
  numbers ("up to 150x vs CPU/GPU", per the track's one-liner) live in the
  paper text, not the artifact.

## 3. SPCG — "Sparsified Preconditioned Conjugate Gradient Solver on GPUs"
(SC 2025)

Source: `github.com/SwiftWare-Lab/SPCG`, files `README.md`,
`gpu_src/ilu0_gpu/nonsp/main.cpp` (full driver read), `script_src/python_scripts/prep/matrix_download.py`.
No arXiv id; ACM DL/ResearchGate abstract only beyond the code.

- **Workload**: fixed, named list of **113 SuiteSparse matrices** hard-coded
  in `matrix_download.py` (e.g. `bcsstk08`...`bcsstk38`, `crystm01-03`,
  `nasa1824`...`nasa4704`, `s1rmq4m1` family, `thermal1`, `wathen100/120`,
  etc. — predominantly SPD structural/thermal/FEM matrices, consistent with
  CG requiring SPD systems). Sparsified matrix variants at multiple
  sparsification ratios (e.g. 0.01, 0.05, 0.1) are generated per matrix via
  MATLAB scripts for the "SP" variant.
- **Precision**: fp32 throughout the GPU driver (`float`, `CUDA_R_32F`,
  `cublasS*`/`cusparseS*` calls) — notable since CG/ILU convergence is often
  precision-sensitive; the paper's own tolerance (below) is set assuming fp32.
- **Preconditioner & solver**: ILU(0) (this file) or ILU(k) (separate
  `iluk_gpu` target) as CG preconditioner; ILU factorization done via
  `cusparseScsrilu02` (+ analysis phase for both L and U triangular solves).
- **Convergence criterion**: `max_iter = 1000`, `tol = 1e-12` (compared
  against squared residual `r1 = dot(r,r)`, i.e. stops once
  `||r||_2 <= 1e-12`, an **absolute**, not relative, residual test).
- **Timing protocol**: CUDA events (`cudaEventRecord`/`cudaEventElapsedTime`).
  Preconditioner setup (`csrilu02` factorization + `SpSV` analysis) is timed
  separately (`precond_time`) from the CG loop; **per-iteration time is
  logged for every iteration** to a CSV (`Iteration,Residual,Time`), and a
  summary row appends `(Overall Time, Preconditioning Time, PCG Time,
  Iterations Spent)` per matrix. **No warmup and no repeated runs** — this
  is a single solve-to-convergence per matrix per invocation; no
  mean/median/min statistic is computed across repetitions (a fairness gap
  worth fixing in the spec).
- **Correctness**: gate after solve — recomputes `A*x` and checks
  `max|Ax - b| < 1e-5` (`qaerr2`), plus a convergence-flag check
  (`k <= max_iter`); process exits non-zero on failure.
  **Iteration-count-to-convergence (`k`) is a first-class reported output**,
  directly in the summary CSV alongside time — exactly the kind of
  "cheap-but-more-iterations" bookkeeping this track's spec must preserve.
- **Metric/headline numbers** (from the track abstract, corroborated by code
  structure): geometric-mean speedup 1.23x / 1.65x over non-sparsified PCG
  for ILU(0)/ILU(k) in the *iterative phase* alone, and 1.68x / 3.73x
  *end-to-end* (i.e. the paper itself separates "iterative-phase-only" from
  "end-to-end incl. setup" speedups — direct evidence for this track's
  setup-vs-apply-vs-iteration-count split).

## 4. BootCMatchGX — "A Multi-GPU Aggregation-Based AMG Preconditioner for
Iterative Linear Solvers" (TPDS 2023)

Source: arXiv **2303.02352** (fulltext read via ar5iv, ID differs from what's
in the track's metadata — it was empty there), fulltext confirmed against
`github.com/bootcmatch/BootCMatchGX` (`README.md`, `BCMGX/src/example/driverSolve.cu`
CLI, `BCMGX/src/test/data/cfg/*.cfg` grid configs, `BCMGX/src/test/data/settings/*.properties`
solver+preconditioner combos).

- **Workload**: synthetic 3D Poisson problems, 7-point finite-difference
  stencil, generated in-code (`--laplacian-3d`, `-g 7p|27p`). Repo ships
  concrete configs from `10^3` up to `160^3` unknowns-per-axis (file names
  `lap3d_{10,50,100,160}x...x..._<PxQxR>.cfg`), matching the paper's
  strong-scaling problem (`300^3 = 27e6` dofs, up to 16 nodes) and
  weak-scaling problem (fixed `130^3 ≈ 2.2e6` dofs/node, 1 to 100 nodes,
  reaching ≈2.2e8 dofs total). No SuiteSparse matrices used in the paper's
  main scaling experiments (the repo's `--matrix` mode exists and ships
  sample `.mtx` files under `src/test/data/mtx/`, but that's for
  unit/regression tests, not the scaling study).
- **Precision**: not explicitly stated in either the paper text or the
  headers checked; the codebase's value type (`vtype`) is used consistently
  across CSR/vector ops but its concrete typedef wasn't located in the
  files read — **open question**, likely fp64 (AMG/CG classical practice)
  but unconfirmed from the sources consulted.
- **Convergence criterion**: PCG (their FCG/CG-based Krylov solvers)
  iterated until relative residual (l2 norm) `< 1e-6` **or** iteration count
  reaches max **1000**.
- **Metrics reported — explicitly separated** (this is the track's best
  example of the desired discipline): operator complexity (OPC), setup time
  (`t_setup`, AMG hierarchy construction), solve time (`t_solve`), time per
  iteration (`t_iter`), and iteration count (`it`) are all reported as
  distinct columns, not folded into one number.
- **Timing protocol**: the driver has rich profiling flags
  (`--summary-prof`, `--detailed-prof`, `--detailed-metrics`, `--trace`,
  per-MPI-rank log files) but the paper's own repeated-run/statistic policy
  (warmup, mean/median, timer) is not stated in the fulltext extracted.
- **Hardware**: Piz Daint (Cray XC50, Nvidia Tesla P100, up to 100 nodes) for
  the main scaling study; Marconi100 (up to 32 GPUs) in an appendix.
- **Baselines**: NVIDIA AmgX (release 2.1) — primary; hypre (release 2.26.0)
  — secondary/appendix. Reported speedup **1.6x–2.0x over AmgX in the solve
  phase**.

## 5. HDagg — "Hybrid Aggregation of Loop-carried Dependence Iterations in
Sparse Matrix Computations" (IPDPS 2022)

Source: `github.com/BehroozZare/HDagg-benchmark` (driver
`demo/SpTRSV_runtime.cpp`/`.h`, sbatch scripts `scripts/Sp{TrSv,ILU0,ICh0}_Final*.sh`)
**plus its actual timing engine**, which lives in a separate dependency repo,
`github.com/sympiler/aggregation` (`utils/FusionDemo.cpp`,
`include/aggregation/FusionDemo.h`) — the benchmark repo subclasses
`FusionDemo` for every scheduling strategy (serial, wavefront/levelset, tree,
HDagg, etc.) and that base class is where warmup/repetition/statistic policy
actually lives, so it had to be pulled in to get the real protocol.

- **Workload**: three kernels — sparse triangular solve (SpTRSV), incomplete
  Cholesky (IC(0)), incomplete LU (ILU(0)). Driver takes `<matrix> <threads>`;
  sbatch scripts glob `matrix/*.mtx` — the **exact matrix list is not
  enumerated in the repo** (must be supplied locally), an open question for
  reproducing the paper's precise suite. Thread counts are swept explicitly
  (`Cores` vector in the driver, `omp_set_num_threads` per data point).
- **Precision**: fp64 (`double* values` throughout `CSRMatrix`/`Vector`-like
  structures; MKL calls used are the double-precision entry points).
- **Timing protocol** (from `FusionDemo::evaluate()`, the shared harness
  every scheduling variant in the paper calls): one-shot `build_set()`
  (scheduling/DAG-aggregation cost) timed as `analysis_time_`
  **outside and separate from** a loop of `num_test_` timed executions of
  `fused_code()` (the actual kernel), default **`num_test_ = 5`**; the
  returned result is the **median** (`time_median(time_array)`) of those 5
  runs — no separate discard-first-run warmup, but median-of-5 already
  damps outliers. Timer: `std::chrono::system_clock` (host wall-clock).
- **Correctness**: `testing()` compares the kernel's output vector against a
  supplied reference solution `correct_x_` with `is_equal(..., 1e-6)`
  (absolute tolerance), logged as a warning (not a hard fail) if it
  mismatches — a soft-gate, weaker than SPCG's hard-fail residual check.
  Scheduling/preprocessing cost (`getSchedulingTime()`) is exposed per
  variant and written to the results CSV alongside kernel runtime — i.e.
  setup-vs-apply is explicitly split in the output schema
  (`Scheduling_Time` and `Executor_Runtime` columns), plus a derived
  "Profitable" ratio = scheduling_time / (serial_time − parallel_time),
  directly quantifying when preprocessing pays for itself.
- **Baseline**: Intel MKL (sparse triangular solve/ILU/IC routines) — the
  abstract's headline "3.56x over MKL" geomean is this comparison.
- **Iteration-count effect**: not applicable — these are direct/one-shot
  sparse kernels (no outer Krylov iteration loop in this artifact); "loop
  iterations" here refers to DAG wavefronts/levels, not solver iterations.

---

## Divergences

The five papers disagree sharply on what "the preconditioner benchmark"
even measures, which is exactly what this track's variant split must
reconcile:

- **Scope of what's timed**: DBMPK and HDagg time a *kernel in isolation*
  (matrix-power / triangular-solve-and-ILU-factorization) with no outer
  Krylov loop at all. SPCG, BootCMatchGX, and Graphene time a *full
  preconditioned solve to convergence*, where the preconditioner's
  setup+apply cost is one piece of a larger iteration count. A benchmark
  that only measured "apply time" would misrepresent DBMPK/HDagg-style work
  (which has no iteration count to report) as directly comparable to
  SPCG/BootCMatchGX/Graphene-style work (where iteration count is the
  entire point of a "sparsified" or "AMG" preconditioner). → the spec
  splits into a **kernel-only micro-benchmark variant** and a **full-solve
  variant that must report iteration count**.
- **Repetition/statistic discipline**: HDagg's shared harness (median of 5)
  is the most rigorous; DBMPK uses mean-of-50 with an unused warmup; SPCG
  and BootCMatchGX's driver code shows **no repeated runs at all** — a
  single solve-to-convergence, so any single unlucky measurement (GPU clock
  throttling, page faults on first touch) directly pollutes the reported
  number. The spec fixes this with an explicit warmup + repetition-count +
  median policy for both variants.
- **Correctness gating**: SPCG hard-fails on residual/iteration-count
  violation; HDagg only warns (soft-gate); DBMPK has **no correctness check
  at all** in the artifact. The spec makes a residual/error-tolerance gate
  mandatory before a timing counts.
- **Precision**: fp64 is the norm (DBMPK, HDagg, likely BootCMatchGX) except
  SPCG (fp32 throughout, including its `1e-12` "convergence" tolerance,
  which is unusually tight for fp32 arithmetic — the achievable residual
  floor in fp32 is closer to 1e-6/1e-7, so in practice iteration counts are
  probably capped by `max_iter=1000` on harder matrices rather than by
  `tol`) and Graphene (fp32 base + software/double-word extended precision
  as its actual research contribution). A precision-specific variant is
  needed, and SPCG's tolerance choice should be flagged as questionable in
  the spec.
- **Preprocessing/setup accounting**: DBMPK, HDagg, SPCG, and BootCMatchGX
  all separate setup time from apply/solve time in their own reporting
  (a rare point of agreement) — reinforcing that the spec's "preprocessing
  reported separately, never silently amortized" principle is not just an
  external fairness rule but already the community's own practice here,
  just inconsistently enforced (SPCG/BootCMatchGX still skip repetition).
- **Iteration-count as a first-class metric**: explicit in SPCG (per-matrix
  CSV column), BootCMatchGX (`it` column, convergence-criterion-driven),
  and Graphene (config-level `maxIterations`/`relResidual`, printed by
  default) — but structurally absent from DBMPK and HDagg, which don't run
  an outer solver at all. This is exactly why the task's "all three axes
  must be reported together" rule needs a variant boundary: it applies
  within the full-solve variant, not the kernel-only one.
- **Hardware target**: HDagg and DBMPK are CPU-only (x86/Arm, OpenMP+MKL);
  SPCG and BootCMatchGX are NVIDIA-GPU (single-GPU vs multi-GPU/MPI+CUDA);
  Graphene is IPU-only, a fundamentally different architecture (no native
  fp64, explicit tile-memory management) that cannot share a timing harness
  with the other four at all — it's evidence for the *iteration-count and
  precision reporting protocol* but excluded from the CPU/GPU kernel-timing
  variants.
