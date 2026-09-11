# QR track — evaluation-methodology survey

Track input: `data/track_inputs/qr.json`, 4 papers (all 4 surveyed, exceeding
the 3-paper minimum). None of the 4 papers has an arXiv id or a fetchable OA
link in the track input, and none could be found by other means during this
pass — every methodology fact below is recovered from each paper's artifact
repository (build scripts, test drivers, and, for one paper, the driver
source that talks directly to the underlying library's own testing
convention) rather than paper fulltext. This is a genuinely heterogeneous
track: one paper is dense Householder QR with an alternative to column
pivoting for rank-deficient least squares (PAQR); one is sparse multifrontal
QR with a learned symbolic-analysis reordering choice; one is a GPU tensor
(t-QR, transform-domain batched QR) primitive library; one is a ScaLAPACK
PDGEQRF-tuning autotuner whose underlying correctness/timing machinery turns
out to be ScaLAPACK's own standard testing driver. None of the 4 targets
TSQR/CAQR specifically (the "tall-skinny" side of the task's key axis is
therefore only indirectly represented, via PAQR's rank-deficient
least-squares matrices and cuTensor-tubal's per-slice square/near-square
tensor frontal faces — flagged in Divergences).

## 1. `conf/ipps/SidLakhdarCBALGTJWDDA23` — PAQR: Pivoting Avoiding QR
factorization (IPDPS'23, best-paper nominee)

Source: artifact repo `luszczek/paqr` (root `README.md`,
`test_paqr_lapack.sh`, `lapack/test_dgepoqrf.c`) — no arXiv/OA text
available; the repo itself states it originated as an SC'22 artifact
description before becoming this IPDPS'23 paper.

- **the paper's core axis**: three QR variants are compared head-to-head on
  the SAME input matrix construction: unpivoted Householder QR (LAPACK
  `dgeqrf`), column-pivoted QR / RRQR (LAPACK `dgeqp3`, "QRCP"), and the
  paper's own PAQR (`dgepoqrf`) — an algorithm designed to match QR's cost
  while approaching QRCP's forward-error accuracy on rank-deficient
  matrices, by avoiding the O(N) latency cost of column-pivoting's
  step-by-step column search rather than avoiding pivoting's stability
  benefit outright (a different kind of "pivoting avoidance" than this
  project's `lu` track's no-pivot papers, which avoid pivoting's
  communication cost, not its search-latency cost).
- **rank-deficiency as a disclosed, swept input construction**: the LAPACK
  test driver (`test_dgepoqrf.c`) builds matrices via a `type` parameter (0-6)
  that controls WHERE the rank-deficient (all-zero) columns are placed: type
  1 = full rank; type 2 = deficient columns in the first half; type 4 =
  deficient columns split at both ends (complement of 2); type 5 = deficient
  columns in the second half; type 6 = random 50/50 per-column deficiency.
  `test_paqr_lapack.sh` runs all of {full-rank, deficient-at-beginning,
  deficient-at-middle, deficient-at-end} as four separate cases per
  invocation — deficiency LOCATION, not just deficiency COUNT, is treated as
  a first-class swept parameter, since column-pivoting-avoidance algorithms
  are specifically sensitive to where the deficient columns land relative to
  the elimination order.
- **workloads/inputs**: four reproduction tiers by component — MATLAB
  (`test(100)`/`test(1000)` for the paper's Table 1), LAPACK CPU (`sh
  test_paqr_lapack.sh 100 100` / `10000 10000` for Table 2), MAGMA batched
  GPU (`test_paqr_batch.sh`, Table 3), and ScaLAPACK distributed (`test.sh`).
  Each ships as a **separate pinned-version Docker image** (LAPACK 3.10.0,
  MAGMA 2.5.0, ScaLAPACK 2.2, MATLAB R2020a) rather than a single unified
  harness — reproducibility is per-component, not end-to-end.
- **timing protocol**: LAPACK arm uses C `clock()` (CPU time, not wall
  clock) around each of the four factorizations in sequence, with **exactly
  one measured call per factorization per invocation** — no internal
  repetition loop in `test_dgepoqrf.c` itself (any repetition would have to
  come from re-invoking the script externally, not observed in the shipped
  harness). ScaLAPACK arm's README states its five reported result lines
  correspond to "warmup QR 1, warmup QR 2, QR, PAQR, QRCP" — **two full
  factorizations used purely as warmup**, then one measured run each of QR,
  PAQR, and QRCP (also single-shot, no repetition visible).
- **precision & correctness**: double precision (`dgepoqrf`/`dgeqrf`/
  `dgeqp3`) throughout the LAPACK/ScaLAPACK arms. **No orthogonality or
  residual check is computed anywhere in the LAPACK test driver shown** —
  `test_dgepoqrf.c` only times the three factorizations and prints elapsed
  seconds; the paper's own forward-error claims (its central contribution:
  PAQR approaches QRCP's forward-error accuracy) are presumably computed in
  the separate MATLAB harness (`matlab/src/test.m`, not fetched during this
  pass — flagged as an open question) rather than in the LAPACK/ScaLAPACK
  performance-measurement path. This is a real methodological split: the
  ACCURACY claim and the PERFORMANCE claim are measured by two entirely
  different, non-overlapping test harnesses in this artifact.
- **baselines**: `dgeqrf` (unpivoted Householder QR) and `dgeqp3` (QRCP,
  LAPACK's column-pivoted QR) at every scale tested; MAGMA batched-QR variant
  for GPU/small-matrix-batch use cases.

## 2. `conf/sc/LinYWT021` — STM-multifrontal QR: streaming task mapping
multifrontal QR factorization empowered by GCN (SC'21)

Source: artifact repo
`lsl036/STM-Multifrontal-QR-Factorization-Empowered-by-GCN` (root
`README.md`, `STMMQR/README.md`, `STMMQR/test/qrtest.c` full source read,
`STMMQR/test.sh`, `STMMQR/test.txt`) — no arXiv/OA text available.

- **two-part contribution, two-part evaluation**: (a) symbolic analysis —
  a GCN classifier adaptively picks among 4 fill-reducing reordering
  algorithms (AMD, COLAMD, METIS, NESDIS) instead of using one fixed
  default; (b) numerical factorization — a NUMA-aware task-stream mapping
  strategy ("STM-Multifrontal QR"), evaluated independently of which
  reordering was used to feed it. The repo README explicitly notes "GCN and
  STM-MQR are **not really combined**" in the shipped artifact due to
  classifier-accuracy/attribute-generation cost tradeoffs — the two halves
  of the paper's own contribution are not jointly reproducible from this
  artifact as a single pipeline.
- **workloads/inputs**: University of Florida (SuiteSparse) Matrix
  Collection, per the paper's own abstract, "nearly 80% matrices" of the
  collection tested for the head-to-head performance claim. The shipped
  artifact itself only ships a 15-matrix demo list (`test.txt`:
  `dwt_992`, `lns_3937`, `bcsstk14`, `epb1`, ... `mycielskian13`) plus a
  separate 408-matrix list (`GCNdata_408.txt`) used specifically for the
  GCN classifier's own train/test split (not the head-to-head performance
  claim) — the full ~80%-of-SuiteSparse evaluation set is not reproducible
  from what ships in this artifact.
- **correctness check — a manufactured-solution forward-error test, not an
  orthogonality check**: `qrtest.c`'s `check_error()` function builds a
  KNOWN solution x = [0, 1, ..., n-1], forms b = A*x, solves the
  least-squares system via the computed QR factors (Q'b, then triangular
  solve against R), and reports `||x_computed - x_true||_2 / n` — this is a
  manufactured-solution accuracy check on the SOLVE, not a direct
  `||Q^T*Q - I||` orthogonality measurement or an `||A - QR||` backward-error
  residual. A separate internal tolerance `tol = 20*(m+n)*eps*max2Norm(A)`
  (the standard SuiteSparseQR default-tolerance formula) is computed and
  passed into the factorization call itself (controls the RANK-DETECTION
  threshold inside `SparseQR`), but the driver does not appear to gate the
  reported `res` value against this `tol` — both are logged to
  `Results/QR_time.txt` per matrix (`matrix-ID, analysis time, numerical
  factorization time, error verification`) without an explicit pass/fail
  decision in the code shown.
- **timing protocol**: `gettimeofday()` wall-clock wrapped around a loop of
  `cycleNum` repeated `SparseQR()` calls, divided by `cycleNum` to report a
  per-call average — **but `cycleNum` is hardcoded to 1** in the shipped
  driver (`int cycleNum = 1;`), so despite the averaging-loop structure
  being present in the code, the actual measured protocol is a single
  factorization per matrix, not a repeated-timing-loop average.
- **baselines**: MKL sparse QR and the original (un-modified)
  SuiteSparseQR, per the paper's own abstract claim (average 20.78%
  numerical-factorization-time reduction from the GCN reordering choice
  alone, ~1.22x overall vs. original SuiteSparseQR on the TaiShan/Kunpeng
  ARM server).
- **machine**: TaiShan Server, 120 processors across 4 NUMA nodes (ARM),
  compared against MKL on an Intel Xeon 6248.

## 3. `journals/tpds/ZhangLWW20` — cuTensor-Tubal: Efficient Primitives for
Tubal-Rank Tensor Learning Operations on GPUs (TPDS'20)

Source: artifact repo
`YangletLiu/cuTensor_CUDA_Library_for_Transform_based_Tensors` (root
`README.md`, `GPU/test/tqr/testtqr.cpp` full source read, `GPU/test/test.sh`,
`GPU/test/testtprod.sh` for the sweep-script pattern) — no arXiv/OA text
available. t-QR (transform-domain, i.e. tubal-rank / FFT-domain-batched QR)
is one of 7 primitives this library implements for 3rd-order tensors (the
others: t-FFT, t-product, t-SVD, t-inverse, t-normalization); each ships in
three implementation variants — **based**, **batched**, and **streamed**.

- **algorithmic shape — the QR-track-relevant fact**: t-QR treats a 3rd-order
  tensor's third dimension ("tube" length, parameter `tupe` in the test
  driver) as a batch axis in the FFT/transform domain: an m x n x tupe
  tensor's t-QR reduces to `tupe` independent 2D QR factorizations of the
  m x n frontal slices in transform space, run either as one-at-a-time
  (`based`), as a single batched cuBLAS/cuSOLVER-style batched call
  (`batched`), or overlapped across CUDA streams (`streamed`) — this is a
  genuinely different notion of "batched QR" than PAQR's or STM-QR's single
  large sparse/dense matrix, and is the one paper in this track whose QR
  shape is neither classically "tall-skinny" nor "square" in isolation but a
  BATCH of many small-to-medium 2D QR problems.
  is closer to PAQR's/STM-QR's single-large-matrix framing.
- **workloads/inputs**: `testtqr.cpp` takes `m n tupe {based|batched|streamed}`
  directly as CLI arguments — no fixed matrix suite or selection criterion in
  the artifact; the general library sweep pattern used elsewhere in the same
  `GPU/test/` directory (e.g. `testtprod.sh`: square m=n swept 200 to 2000 in
  steps of 100 at a fixed third dimension) was NOT replicated as a
  ready-to-run script specifically for `tqr` (no `testtqr.sh`/sweep script
  ships in the repo, unlike `tprod`/`tfft`) — reproducing the paper's actual
  t-QR performance figures requires re-deriving the sweep the paper used, not
  just running a provided script.
- **precision**: single precision (`float`/`cuComplex` — real input cast to
  complex for the FFT-domain computation, matching the transform-based
  tensor model's requirement of a complex frequency-domain representation).
- **timing protocol**: host-side C `clock()` around the (asynchronous) GPU
  call, **single measured call per invocation, no warmup, no repetition**
  in the shipped test driver.
- **precision & correctness**: a correctness-comparison code path EXISTS in
  `testtqr.cpp` (the `streamed` branch computes `batchedtqr` on a saved copy
  of the input and diffs the `streamed` result's matrix entries and `tau`
  Householder-scalar array against it) but is **compiled out by default**
  (`#if 0` guarding the entire diff block) — i.e. the artifact ships a
  cross-implementation consistency check that is disabled in the
  performance-measurement path, the same "correctness check present but not
  default-enabled" pattern already found in this project's `lu` track
  (OpenMxP's `--checksum` flag) and `cholesky` track (hicma-x's `--check`
  flag). No true orthogonality (`||Q^T*Q - I||`) or backward-error
  (`||A - QR||`) check is present anywhere in this driver, even
  disabled — the diff is purely an implementation-vs-implementation
  cross-check, not a check against a ground-truth QR property.

## 4. `conf/ipps/ChoDKLLL23` — Harnessing the Crowd for Autotuning
High-Performance Computing Applications (IPDPS'23)

Source: artifact repo `gptune/GPTune`
(`examples/Scalapack-PDGEQRF/scalapack_MLA.py` full source read,
`examples/Scalapack-PDGEQRF/scalapack-driver/spt/pdqrdriver.py` full source
read) — no arXiv/OA text available. This paper (GPTuneCrowd) is not itself a
QR-factorization paper; it is a crowd-sourced transfer-learning autotuner
whose headline QR-relevant result is a 1.57x tuned-performance improvement on
ScaLAPACK's `PDGEQRF`. It is included in this track because its evaluation
methodology for that result turns out to directly inherit ScaLAPACK's own
QR-factorization testing driver's timing AND correctness conventions,
independent of the autotuning contribution itself.

- **the tuned parameters are the same tile-size/process-grid axes as the
  `lu` and `cholesky` tracks**: block size (`mb`, `nb`, both in units of a
  base `bunit`), process grid (`p`, `q` with `p*q = nproc`), and
  `nodes`/`cores`/`npernode` — GPTuneCrowd's search space for `PDGEQRF` is
  structurally identical to what this project's `cholesky` spec already
  identifies as the single most consequential undisclosed-tuning-knob axis
  for dense factorizations (KwasniewskiKBZS21's finding, reused verbatim
  across the `lu`/`cholesky`/`qr` tracks in this project).
  provides both.
- **correctness IS gated, via ScaLAPACK's own testing-driver convention, not
  GPTuneCrowd's own code**: `write_input()` writes ScaLAPACK's standard
  `PDQRDRIVER` test-input format (`FACTOR MVAL NVAL MBVAL NBVAL PVAL QVAL
  THRESH`, matching ScaLAPACK's own `TESTING/LIN` driver input convention
  exactly), and `read_output()` parses lines of the form `WALL M N MB NB
  NPROW NPCOL WTIME TMFLOPS PASSED FRESID` — **only lines where the 10th
  field is the literal string `PASSED`** (i.e. ScaLAPACK's own internal
  factorization-residual check, `FRESID`, against the disclosed `THRESH`
  parameter) are counted into the returned timing array at all; a
  configuration whose factorization fails ScaLAPACK's own residual check is
  silently excluded from the autotuner's observed timings, not merely
  flagged. `THRESH` itself is one of the constant fields threaded through
  every call (not exposed as a tuned parameter in the example script shown).
- **timing protocol — an explicit best-of-N, not an average**:
  `read_output()`'s accumulation rule is `if (mytime < times[idxparam]):
  times[idxparam] = mytime` over `niter` repeated ScaLAPACK-driver
  invocations per configuration — a **minimum-of-niter** statistic. The
  `pdqrdriver()` function's own default is `niter=10`, but the example
  driver script (`scalapack_MLA.py`'s `objectives()`) explicitly calls it
  with **`niter=2`** for the autotuning objective function itself (the
  paper's own README-documented example invocation additionally sets
  `-nrun 10` — the number of DIFFERENT configurations the tuner tries, a
  search-budget parameter, not a per-configuration timing-repetition count,
  and should not be confused with `niter`).
- **workloads/inputs**: matrix size swept up to a user-set `mmax`/`nmax`
  ceiling (5,000 x 5,000 in the README's own example invocation),
  `ntask`-many distinct matrix sizes tuned per run (5 in the same example).
- **baseline**: default/untuned ScaLAPACK `PDGEQRF` parameters, and a
  non-transfer-learning autotuner (GPTune's own OpenTuner/HpBandSter-backed
  single-machine tuning mode) as the comparison point for the paper's
  headline 1.57x claim.

## Divergences

- **Correctness notion differs in kind, not just in strictness, across all
  four papers**: PAQR's LAPACK/ScaLAPACK performance-measurement path has
  **no correctness check at all** (accuracy is measured only in a separate,
  non-overlapping MATLAB harness); STM-QR uses a **manufactured-solution
  forward-error check** (`||x_computed - x_true||/n` on a synthetic
  known-solution linear system) logged but not gated against its own
  computed tolerance; cuTensor-tubal ships a **cross-implementation
  consistency diff** (streamed vs. batched output) that is compiled out by
  default and checks nothing against a ground-truth QR property regardless;
  GPTuneCrowd is the ONLY one of the four whose measured numbers are
  actually gated by a real QR-factorization correctness check — but that
  check is ScaLAPACK's own internal `FRESID`/`PASSED` mechanism, not
  anything GPTuneCrowd's own paper describes or claims credit for. **None of
  the four papers computes or reports `||Q^T*Q - I||` (orthogonality)
  directly** — the property the task's own design brief for this track
  singles out as the QR-specific mandatory correctness check. This is a
  bigger gap than either the `lu` or `cholesky` tracks show for their own
  headline correctness notions (residual, growth factor) and is treated as
  a required spec addition below, not an optional refinement.
- **Rank-deficiency / tall-skinny-vs-square shape is unevenly represented**:
  PAQR is the only paper whose input construction directly targets
  rank-deficient least-squares matrices (with deficiency LOCATION as a swept
  parameter); STM-QR's sparse SuiteSparse matrices are overwhelmingly square
  (structural/FEM-style matrices) rather than tall-skinny; cuTensor-tubal's
  "batched many small 2D QRs" shape has no tall-skinny/square distinction in
  the traditional sense at all (each frontal-slice QR can be any m x n
  shape, but the artifact's own examples default to square-ish m=n); and
  GPTuneCrowd's ScaLAPACK `PDGEQRF` tuning defaults to square `mmax=nmax`
  in its own example. **No paper in this track's input set targets TSQR or
  CAQR** (Communication-Avoiding QR) specifically, despite this being one of
  the task's named key axes — this is a genuine gap in what this small
  track's own literature covers, not a methodological disagreement to
  reconcile; flagged as an open question below rather than papered over.
- **Timing statistic**: single-shot (no repetition at all) for PAQR's
  LAPACK arm and STM-QR (`cycleNum` hardcoded to 1 despite the averaging-loop
  code structure existing); two-warmup-then-single-shot for PAQR's
  ScaLAPACK arm; single-shot for cuTensor-tubal; explicit **minimum-of-2**
  (best-of-N) for GPTuneCrowd's ScaLAPACK-driver calls. No paper in this
  track reports a repeated-timing-loop median or mean at all — this track's
  own literature has the *least* timing-repetition discipline of the three
  tracks surveyed together (`lu`, `qr`, `eigensolver`; contrast `lu`'s
  MunksgaardHSO22 explicit mean-of-10 or `cholesky`'s stated min-of->=3/
  mean-of-3 conventions). The spec below fixes this to a real repeated
  measurement with median statistic, a bigger departure from this track's
  own literature than either sibling track required.
- **Correctness-check-present-but-disabled-by-default is a recurring pattern
  across THREE separate artifacts in this small track**: cuTensor-tubal's
  `#if 0`-guarded diff block (this survey); and, by direct analogy already
  documented in this project's `lu` and `cholesky` surveys, OpenMxP's
  `--checksum` flag and hicma-x's `--check` flag. This project's fairness
  principle (correctness gate before timing counts) is violated by DEFAULT
  in more than half of the artifacts surveyed across all three of this
  project's dense-factorization tracks so far — worth flagging as a
  cross-track pattern, not just a per-paper footnote.

## Open questions

- PAQR's actual forward-error/accuracy measurement methodology (presumably
  in `matlab/src/test.m`, reproducing the paper's Table 1) was not fetched
  during this pass — needed to confirm whether an orthogonality or
  backward-error check exists ANYWHERE in this paper's own artifact, even
  outside the performance-measurement path.
- STM-QR's `check_error()` forward-error value (`res`) is computed and
  logged per matrix but the driver source shown does not appear to gate
  correctness against the `tol` value it also computes — whether a
  pass/fail decision happens elsewhere (e.g. in `result-table`-style
  post-processing not included in the `test/` directory) was not confirmed.
- The full ~80%-of-SuiteSparse matrix set STM-QR's own abstract claims to
  have evaluated is not reproducible from the 15-matrix `test.txt` shipped
  in the artifact; whether the full set is available anywhere (e.g. the
  `Data/` directory's actual contents beyond the demo list, or an external
  supplementary archive) was not confirmed.
- cuTensor-tubal's t-QR has no ready-to-run sweep script (`testtqr.sh`)
  unlike `tprod`/`tfft` in the same repository — the exact matrix-size
  sweep and `tupe` (tensor tube-length) values used to produce the paper's
  own reported t-QR figures were not recoverable from the artifact alone.
- No paper in this track's 4-paper input set targets TSQR/CAQR
  (Communication-Avoiding QR) or an explicitly tall-skinny (m >> n)
  benchmark shape — whether this is a genuine gap in the track's assigned
  paper set (as opposed to a search-recall gap in the corpus that produced
  `qr.json`) was not investigated during this pass.
- GPTuneCrowd's `THRESH` (ScaLAPACK residual-check tolerance) value used in
  its own reported PDGEQRF-tuning results was not confirmed from the
  extracted source (the field is threaded through every driver call but its
  concrete numeric value in the paper's own experiments was not located) —
  needed before this track's spec can state whether it matches or differs
  from the field-standard ScaLAPACK testing-suite default.
