# TRSM track — survey (2 papers)

Track is small (2 papers) and neither paper is *primarily* about TRSM: fBLAS
is a streaming FPGA BLAS library that implements TRSM as one of its Level-3
routines and benchmarks it against MKL; AbdulahCPBDGKLS22 uses TRSM only as
an embedded step inside a distributed mixed-precision tile-Cholesky MLE
solver and never benchmarks it in isolation. Both papers were already
surveyed in more depth for `benchspecs/cholesky/` (AbdulahCPBDGKLS22) and
`benchspecs/gemm/` (fBLAS/MatteisLH20); this file adds the TRSM-specific
detail pulled directly from fBLAS's artifact repo (`spcl/FBLAS`), which is
the only source in this track with concrete TRSM protocol facts.

## 1. `conf/sc/MatteisLH20` — fBLAS: streaming linear algebra on FPGA (SC'20)

Source: arXiv fulltext (`ar5iv.labs.arxiv.org/html/1907.07929`, TRSM-specific
passages) + artifact repo `spcl/FBLAS` (`evaluation/cpu_programs/batched_trsm.cpp`,
`tests/host_api/trsm.json`, `templates/3/trsm_v1..v8.cl`).

- **workloads/inputs**: TRSM with multiple right-hand sides, `op(A)[N,N] * X
  = alpha*B` (A lower-triangular, non-transposed, left side, non-unit diag is
  the one config with disclosed *performance* numbers). Two problem sizes
  with concrete numbers in the paper: **N=8192 and N=32768** (Table V,
  "Comparison to batched CPU routines"), single and double precision.
  Correctness/functional testing (not performance) additionally covers all
  **16 side x uplo x trans x {float,double}** combinations (`Left/Right x
  Lower/Upper x NoTrans/Trans`, diag always `NonUnit`), each compiled as a
  separate FPGA HLS template variant (`trsm_v1.cl` .. `trsm_v8.cl`, one
  per side/uplo/trans combination, reused for both precisions) — i.e.
  *functional coverage is broader than performance coverage*: the paper
  reports execution time only for the one Left/Lower/NoTrans/NonUnit
  configuration, not for all 16 tested combinations.
- **hardware**: Intel FPGAs (Arria 10 GX1150, Stratix 10 GX2800) vs. a CPU
  batched-TRSM baseline.
- **CPU/MKL baseline protocol** (`evaluation/cpu_programs/batched_trsm.cpp`,
  same harness family as fBLAS's GEMM CPU baseline surveyed in the gemm
  track): sweeps thread count from `MIN_THREADS=1` to `MAX_THREADS=10` via
  `mkl_set_num_threads_local`; calls `cblas_dtrsm_batch`/`cblas_strsm_batch`
  with a single group (`GRP_COUNT=1`) of `ntrsm` identical-shape (N x N
  triangular, N x M RHS) problems; **the first of `runs+1` iterations is
  discarded** ("remove the first run" — a 1-iteration warmup), remaining
  `runs` iterations timed individually via `current_time_usecs()`
  (host wall clock); the driver keeps `min_times`/`min_mean` across the
  thread-count sweep and the README states host programs "run the routine(s)
  multiple time with different number of thread and report **the best
  execution time**" — i.e. **min-of-runs is the reported statistic**, not
  mean or median, despite the code also computing `stddevs`.
  side/uplo/trans/diag fixed to `Left/Lower/NoTrans/NonUnit` in this specific
  CPU harness (matches the one config the paper's Table V reports).
- **timing scope**: kernel-only; A and B are generated on-host and resident
  before timing; no host<->device transfer breakdown is given for the FPGA
  side in the extracted text.
- **precision & correctness**: FP32/FP64. No numeric correctness tolerance
  is stated anywhere for TRSM specifically (confirmed by two independent
  fulltext searches — general fBLAS paper text and a TRSM-focused reread);
  the unit-test JSON (`tests/host_api/trsm.json`) exists to check
  correctness across all 16 side/uplo/trans/precision combos but the
  file itself carries no tolerance value, only routine parameters (`width:
  16`, `tile M size: 32` for every entry).
- **metric**: execution time (µs) at the two named sizes; general fBLAS
  metric elsewhere is Ops/s with a "99% CI within 5% of mean" statistic, but
  this specific statistic/CI claim is not repeated for the TRSM table.
- **baselines**: Intel MKL (batched TRSM) on Xeon E5-2630 v4.

## 2. `journals/tpds/AbdulahCPBDGKLS22` — Accelerating Geostatistical
Modeling and Prediction With Mixed-Precision Computations: A
High-Productivity Approach With PaRSEC (TPDS'21/22)

Source: fulltext PDF (already surveyed in `benchspecs/cholesky/survey.md`
section 3) + artifact repo `ecrc/exageostat`.

- TRSM is **not** an independently benchmarked kernel in this paper. The
  paper's abstract states the two "essential operations in MLE" are "the
  application of the inverse and evaluation of the determinant of the
  covariance matrix... rendered through the Cholesky decomposition and
  triangular solution" — i.e. a triangular solve (TRSM/TRSV-family op)
  follows every tile-Cholesky factorization inside the MLE log-likelihood
  evaluation loop, but the paper's performance numbers (Tflop/s, PFlop/s,
  1.56x-2.64x speedup) are reported for the *fused* factor+solve pipeline
  at a **named, discrete band-percentage precision** (e.g. `10D:90S`,
  `1D:99H`), never for TRSM alone.
- **Correctness for the embedded solve is validated by downstream
  statistical quality** (parameter-recovery boxplots, log-likelihood value,
  MSPE vs. the pure-FP64 `100D` reference), not a numeric residual bound on
  the triangular solve step itself — a fundamentally different notion of
  "correct" than a per-element/relative-error tolerance.
- Relevant for this track only as evidence that TRSM's real HPC usage
  pattern is "one solve immediately following one factorization, precision
  possibly mixed/banded across the matrix," a usage shape this track's other
  paper (fBLAS) does not test at all (fBLAS's TRSM tests are standalone,
  single-precision-per-call, single-shape, not chained after a factorization
  and not mixed-precision).

## Divergences

- **Scope**: fBLAS benchmarks TRSM as an isolated, standalone kernel (the
  common "microbenchmark" claim this spec's `trsm-single-kernel` and
  `trsm-batched-kernel` variants target); AbdulahCPBDGKLS22 never isolates
  TRSM at all — it is fused into a larger factor+solve+likelihood pipeline.
  This spec cannot construct a "TRSM as embedded in Cholesky-MLE" variant
  without inventing protocol details AbdulahCPBDGKLS22 does not disclose
  (see open_questions in spec.yaml); it is used only as supporting evidence
  for the application-context note above.
- **Correctness definition**: fBLAS states none (open question, numeric
  tolerance invented by this spec); AbdulahCPBDGKLS22 uses a downstream
  statistical-quality gate incompatible with a single-kernel numeric-error
  spec.
- **Precision model**: fBLAS is single-precision-per-call (fp32 XOR fp64);
  AbdulahCPBDGKLS22 is multi-precision-per-matrix (band-percentage mixture).
  These are different enough that this spec (single-precision-per-call,
  matching fBLAS, the only paper with an actual TRSM microbenchmark) does
  not attempt to cover AbdulahCPBDGKLS22's mixed/banded precision model —
  flagged as an open question rather than fabricated into a variant.
- **Statistic**: fBLAS's own reported statistic is min-of-runs (optimistic,
  same pattern seen across other tracks in this project's survey work) even
  though its own code also computes std-dev; this spec fixes median with
  min/max reported alongside, per this project's general fairness
  principle.
- **Functional vs. performance coverage**: fBLAS validates correctness on
  16 side/uplo/trans/precision combinations but reports *performance* for
  only 1 of them. This spec requires performance be reported for the full
  matrix of layout variants, not just the one combination the paper
  publishes.
