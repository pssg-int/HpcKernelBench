# mixed-precision-solver — evaluation-methodology survey

All 4 papers in the track input were surveyed via their open-source artifact
code (none has an arXiv id or a reachable OA fulltext; all DOIs are
ACM-DL/OSTI-paywalled with no `oa_url` fulltext to fetch). Facts below are
read directly from driver/benchmark source files, not from abstracts.

## 1. F3R — "A Nested Krylov Method Using Half-Precision Arithmetic" (SC25, `conf/sc/SuzukiI25`)

Repo: `github.com/suzuki-hpc/F3R`. Source read: `sc25/README.md`,
`sc25/suite-cpu.py`.

- **Algorithm**: nested FGMRES/Richardson Krylov cascade, precision
  progressively reduced from fp64 (outer) toward fp16 (innermost); low-precision
  inner solvers run only a few iterations per invocation, invoked frequently
  by the nesting structure (not a one-shot inner solve).
- **Reproduction structure**: `T1 -> T2 -> T3_C + T3_G -> T4` (download/
  generate matrices -> compile -> run CPU/GPU suites -> plot). The paper's
  own reproduction artifact IS the benchmark harness.
- **Matrix suite (confirmed by code, `sc25/suite-cpu.py`)**: symmetric list
  (14 total) `audikw_1, Bump_2911, ecology2, Emilia_923, G3_circuit,
  hpcg_7_7_7, hpcg_8_7_7, hpcg_8_8_7, hpcg_8_8_8, ldoor, Queen_4147, Serena,
  thermal2, tmt_sym` — 10 real SuiteSparse SPD matrices + 4 synthetic
  HPCG-generator grids (`make` target inside `sc25/matrix`, self-described as
  "Generate HPCG and HPGMP matrices"). General/nonsymmetric list (14 total,
  out of scope for this SPD-adjacent track, but confirms the paper's own
  BiCGSTAB-family comparison exists): `atmosmodd, atmosmodj, atmosmodl,
  hpgmp_7_7_7, hpgmp_8_7_7, hpgmp_8_8_7, hpgmp_8_8_8, ss, stokes, t2em,
  tmt_unsym, Transport, vas_stokes_1M, vas_stokes_2M`.
- **Default reproduction shortcut**: `suite-cpu.py` runs only every 3rd
  matrix (`step=3`, "one-third of the test matrices to save time") unless the
  script is invoked with a literal `full` argument — the paper's own quick-
  repro path is a cherry-picked subset by construction.
- **Paired baseline, same script, same matrix**: for each matrix,
  `T3C_SYM`/`T3C_GEN` run 9 binaries back-to-back on the SAME input:
  `cg64/cg32/cg16` (or `bicg*` for the general list), `gm64/gm32/gm16`
  (plain GMRES at 3 base precisions), and `f3r64/f3r32/f3r16` (F3R itself at
  3 base precisions) — plus one extra invocation of `f3r16` with a
  `_Float16`-vs-"Best" precision-selection flag. This is the most direct
  same-binary-family, same-matrix, same-script fp64-vs-reduced-precision
  pairing found in this survey.
- **Repetition**: `<average>` CLI integer parameter controls repeated full
  solves; the paper's own quick-repro instructions say `1` is "sufficient to
  reproduce the general trend."
- **Output/accuracy columns (confirmed, CSV header in `suite-cpu.py`)**:
  `Problem,Method,Prec,M2,M3,M4,W,Precond,ACC,Time,Iter,ImplRes,ExplRes` —
  i.e. the harness logs BOTH an implicit (recurrence-tracked) residual and an
  explicit (independently recomputed) residual per run, plus `Iter`
  (iteration count) and an `ACC` field, as first-class CSV columns. Exact
  semantics of `M2/M3/M4/W/Precond/ACC` were not decoded from the senk
  library source within this survey's budget (see open_questions).
- **Precision**: 3 discrete base-precision configurations (64/32/16) run and
  logged as SEPARATE rows per method family, never averaged together.

## 2. OpenMxP — "Climbing the Summit and Pushing the Frontier of Mixed
Precision Benchmarks at Extreme Scale" (SC22, `conf/sc/LuMOEJW22`)

Repo: `github.com/at-aaims/OpenMxP` (HPL-AI benchmark implementation).
Source read: `README.md`, `src/main.cpp`, `src/iterative_refinement.hpp`.

- **Algorithm**: dense LU factorization with `FHIGH = float` for panel
  decomposition and `FLOW = __half` for the GEMM (tensor-core fp16), on a
  synthetically generated (not downloaded) dense matrix (`matgen.hpp`/
  `highammgen.hpp`, "Higham" test-matrix generator), followed by fp64
  iterative refinement (`mgir` uses `Matgen<double>`) to recover accuracy.
- **Iterative refinement loop** (`iterative_refinement2`, in
  `src/iterative_refinement.hpp`): each step recomputes the true residual
  `r = b - Ax` (via `panel_gemv`, NOT reused from the factorization's own
  bookkeeping), computes the official HPL accuracy criterion
  `hplerror = ||r||_inf / (||A||_inf*||x||_inf + ||b||_inf) / (n *
  DBL_EPSILON/2)`, and returns as soon as `hplerror < 16.` — this constant
  16 is HPL/HPL-AI's own standard pass/fail threshold, not a value invented
  by this paper.
- **Iteration cap**: `main.cpp` calls `iterative_refinement2(..., warmup ? 1
  : 50, g)` — max 50 IR iterations for a real (non-warmup) run, 1 for a
  warmup run.
- **Non-convergence handling — BUG-ADJACENT PRACTICE**: if the loop exhausts
  50 iterations without `hplerror < 16`, `iterative_refinement2` returns the
  sentinel `{-1., -1.}` (comment: `// OMG!`). `main.cpp`'s printing code
  (`printf("... resid = %e hpl-harness = %f ...", er.residual,
  er.hpl_harness, ...)`) prints this sentinel unconditionally alongside
  legitimate results — there is no check for `er.residual == -1` before the
  headline `TFlop/s` / `GFlop/s` line is printed. A non-converged run is not
  distinguished from a converged one in the console output.
- **Timing**: `MPI_Wtime()` bracket from `start_time` (right after matrix
  generation, which therefore is NOT counted) through `stop_time` (after IR
  completes); a SEPARATE `start_ir`/`stop_time` bracket isolates IR-only
  time (`"\t IR Time %.2f sec\n"`). Single run per invocation — no
  warmup-vs-timed repetition loop beyond the single `-w`/`--warmup` flag,
  which itself runs one reduced-flop-count pass with `maxit=1` for calibration.
- **Metric**: `hplflops = 2/3*n^3 + 3/2*n^2`; reported as GFlop/s and
  `TFlop/s per GPU` (dividing by `gsize`, the GPU/rank count) — an HPL-AI
  submission-style report block (`"NCCS/OLCF Mixed Precision Benchmark"`,
  system/date/N/NB/P/Q/COMM fields) is printed verbatim.
- **Problem size**: README states tuning guidance (`B` tile size best around
  2560, local-per-rank `LN` best around 125440, global `N = P * LN`) but no
  fixed benchmark size list; sizes are scale-point-specific (Summit/Frontier
  full-system runs in the paper itself).

## 3. hicma-x (HiCMA-PaRSEC) — "Boosting Earth System Model Outputs ... Using
Exascale Climate Emulators" (SC24, `conf/sc/AbdulahBBCCGKKL24`)

Repo: `github.com/ecrc/hicma-x`. Source read: `README.md`, `TESTS.md`,
`tests/testing_climate_emulator.c`, `src/climate_emulator/
climate_emulator_mse.jdf`.

- **Algorithm/pipeline** (`testing_climate_emulator.c`, function-by-function):
  (1) forward spherical-harmonic transform (SHT) on climate reanalysis data,
  (2) SYRK (`C = A*A^T`) to build a dense covariance-like matrix, plus a
  matrix-norm computation, (3) Cholesky factorization
  (`hicma_parsec_potrf`, repeated `params->nruns` times — the ONLY phase
  wrapped in a repeat loop), (4) OPTIONALLY (if
  `CLIMATE_EMULATOR_ENABLE_INVERSE`): diff against a MATLAB reference
  (`climate_emulator_diff_double`), inverse SHT, and MSE computation
  (`climate_emulator_mse`).
  There is NO iterative refinement and NO outer optimization loop — a
  single one-shot factorization pipeline per run (aside from the `nruns`
  repeat of the factorization phase specifically).
- **Precision**: mixed double/single/half/fp8 (`README.md`, "Mixed Precision
  Support"), selected per-tile via `--adaptive_decision` (adaptive tile-
  format/precision selection) combined with `--band_dense_dp` /
  `--band_dense_sp` / `--band_dense_fp8` band-width parameters (`TESTS.md`).
  The exact decision algorithm that assigns a given tile's precision was not
  located in the source read (see open_questions).
- **Accuracy metric — NOT a linear-algebra residual**:
  `climate_emulator_mse.jdf`'s `norm_core()` computes a normalized RMSE
  between the reconstructed climate field (`f_data`) and a reference field
  (`f_spatial`): `sqrt(sum((Re(A_ij) - B_ij)^2)) / (2*L*(L+1))`, printed per
  time-slot ONLY if `verbose > 1`. No comparison to any threshold appears
  anywhere in the read source — the value is diagnostic/logged, not a
  pass/fail gate.
- **Timing**: `SYNC_TIME_START()`/`SYNC_TIME_PRINT()` macro pairs bracket
  EACH phase separately (`geqsht_forward_pre_computed_version`,
  `geqsht_forward_reshape`, SYRK, matrix-norm, factorization [via `nruns`],
  `geqsht_inverse_pre_computed_version`, MSE) — the finest-grained per-phase
  timing split of any paper surveyed in this track (or in the sibling
  cg-krylov/multigrid/preconditioner tracks). No warmup or timed-repetition
  discipline beyond the factorization-only `nruns` loop.
- **TESTS.md example**: `--latitude 120 --NB 1440 --N 14400` (`N = latitude^2`).
- **Correctness check available but optional**: `--check` flag exists
  (listed in `TESTS.md`'s "HiCMA Options") but its implementation was not
  read; `climate_emulator_diff_double` (MATLAB-reference diff) only runs
  under a compile-time `#if CLIMATE_EMULATOR_ENABLE_INVERSE` guard.

## 4. ExaGeoStat — "Reshaping Geostatistical Modeling and Prediction for
Extreme-Scale Environmental Applications" (SC22, `conf/sc/CaoAAPNBDGKLS22`)

Repo: `github.com/ecrc/exageostat` (archived; superseded by
`ecrc/ExaGeoStatCPP`, code itself still readable). Source read:
`README.md`, `examples/real_csv_sdmle_test.c`, `examples/
synthetic_sdmle_test.c`, directory listing of `exageostat_exact/src/compute`
(`MLE_sdexact.c`, `psdportf.c`, `sdpotrf.c` confirm single/double
mixed-precision Cholesky kernels exist as first-class compute routines,
paired with pure-double `dmle`-family binaries built from the SAME codebase).

- **Algorithm**: maximum-likelihood estimation (MLE) for spatial-statistics
  covariance-parameter fitting. The linear-algebra core is a (mixed-
  precision, optionally tile-low-rank) tile Cholesky factorization of the
  dense spatial covariance matrix, called ONCE PER likelihood evaluation
  inside an outer derivative-free optimizer
  (`nlopt_create(NLOPT_LN_BOBYQA, ...)`, `nlopt_set_maxeval(opt,
  data.opt_max_iters)`). There is no per-solve iterative refinement; the
  "iteration count" that exists is the OUTER optimizer's function-evaluation
  count, not a Krylov/refinement iteration count.
- **Precision axis**: `data.precision = 2` selects the mixed single/double
  path (`sdmle`, calling `EXAGEOSTAT_sdmle_Call` /
  `EXAGEOSTAT_MLE_sdregister_Tile`); a parallel pure-double driver
  (`real_csv_dmle_test.c`, not fully read but confirmed present by naming
  convention and by `exageostat_exact/src/compute`'s separate `MLE_sdexact.c`
  vs implied `dexact` routines) provides the fp64 baseline from the SAME
  codebase — the comparison the paper's headline 12x-speedup number is drawn
  from (per this track's input JSON one-liner).
- **Accuracy metric — prediction quality, not a residual**: after MLE fits
  `starting_theta`, a held-out prediction test computes `prediction_error`
  (`EXAGEOSTAT_sdmle_Predict_Tile` / `EXAGEOSTAT_dmle_Predict_Tile`) against
  `Zactual` (true held-out observations) plus `data.mserror1/2/3` and
  optional MLOE/MMOM criteria (`EXAGEOSTAT_dmle_mloe_mmom_Tile`). For
  SYNTHETIC data, ground-truth `initial_theta` is known by construction
  (`MLE_zvg` generator), giving a second, parameter-recovery-based accuracy
  channel independent of prediction error.
- **No pass/fail gate in the driver**: `prediction_error` and the MLOE/MMOM
  values are written to a results file (`write_to_estimatedtheta`) and
  printed (`fprintf(stderr, "Prediction Error: %e \n", ...)`) but never
  compared against a threshold or against the `dmle` baseline's own value
  anywhere in the driver source read.
- **Timing**: `START_TIMING`/`STOP_TIMING` macros wrap (a) the full MLE
  optimization (`data.total_exec_time`, spans the entire
  `nlopt_optimize` call) and (b) the held-out prediction phase (`pred_time`)
  as two SEPARATE brackets — the same setup/apply-style split this track's
  sibling specs (cg-krylov, multigrid, preconditioner) already establish as
  the norm, here mapped onto "fit" vs "predict" instead of "setup" vs
  "solve." No warmup or repeated-run discipline is visible in the driver
  (single `nlopt_optimize` call per invocation).
- **Problem size**: not recoverable from the driver source (matrix size `N`
  is a runtime CLI argument tied to an input locations file; no fixed
  benchmark size ships in the repo).

## Divergences

Four structurally different "how do you recover/measure accuracy after
using reduced precision" strategies, one per paper — none of the four is
reducible to any other, which is why the spec below keeps 4 separate
variants rather than one shared protocol:

1. **OpenMxP**: a fixed, externally-standardized numerical criterion (the
   HPL-AI `hplerror < 16` harness) applied via fp64 iterative refinement
   with a hard iteration cap (50); accuracy is a floating-point residual
   quantity.
2. **F3R**: TWO residual channels (implicit recurrence-tracked, explicit
   recomputed) logged per run at 3 discrete base precisions, with a
   same-script, same-matrix, same-binary-family paired fp64 baseline
   (cg64/gm64) — the strongest baseline-pairing evidence in this track.
3. **ExaGeoStat**: accuracy is STATISTICAL prediction quality (MSPE against
   held-out real observations, or parameter-recovery error against known
   synthetic ground truth), computed only after an outer, non-numerical
   (derivative-free) optimization loop converges; no residual-based gate
   exists at all.
4. **hicma-x**: accuracy is an APPLICATION-DOMAIN RMSE against a reference
   climate field, computed once after a single one-shot (no refinement, no
   outer optimization) factorization pipeline, with per-tile precision fixed
   in advance by an (unreviewed) adaptive-decision heuristic rather than
   corrected after the fact.

Additionally: none of the 4 papers' own drivers enforces a hard pass/fail
gate that BLOCKS a non-converged/inaccurate run's timing number from being
reported: OpenMxP prints its `{-1,-1}` non-convergence sentinel
unconditionally; ExaGeoStat and hicma-x compute accuracy numbers but never
compare them to a threshold in the code paths read. F3R is the only one
whose CSV schema even carries two independent residual channels side by
side, though whether its own senk library enforces agreement between them
internally was not confirmed (library source not read within budget).
