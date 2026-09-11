# Eigensolver track — evaluation-methodology survey

Track input: `data/track_inputs/eigensolver.json`, 5 papers (all 5 surveyed,
exceeding the 3-paper minimum). One paper (LiaoLLR21) has an arXiv id and was
surveyed via fulltext, downloaded and extracted locally with `pypdf` (same
WebFetch-PDF-rendering workaround used for the `lu`/`cholesky` tracks); the
other four have no arXiv id or fetchable OA link and are surveyed entirely
from their artifact repositories (README + source). This track splits
cleanly along two axes the task's own design brief names directly: **full
spectrum vs. partial (k eigenpairs)** — LiaoLLR21 and the
WangDZWZLJZ25/WangHZSJWDZ25 pair compute the FULL spectrum with eigenvectors;
ShiLXSH21 computes only the eigenpairs inside a chosen frequency interval,
and YangNYLDXJ026's matrix-power kernel is the computational building block
underneath exactly that kind of polynomial-filtered partial-spectrum solve.
Notably, **every paper in this 5-paper set is symmetric (or symmetric-
definite generalized)** — no genuinely nonsymmetric eigensolver appears in
this track's input, a gap flagged explicitly in Divergences/open_questions
rather than papered over with an invented nonsymmetric variant.

## 1. `journals/tpds/LiaoLLR21` — A Parallel Structured Divide-and-Conquer
Algorithm for Symmetric Tridiagonal Eigenvalue Problems (TPDS'21)

Source: arXiv fulltext, downloaded and extracted locally
(`arxiv.org/pdf/2008.01990`, 17 pages via pypdf 6.14.2 — WebFetch's built-in
PDF rendering again returned only binary stream data) + artifact repo
`shengguolsg/PSMMA` (the paper's own stated code-release location, not
independently re-fetched beyond confirming the URL in-text). This is the
track's one **full-spectrum, distributed** paper: PSDC (Parallel Structured
Divide-and-Conquer), built on a rank-structured matrix-multiplication kernel
(PSMMA) that exploits the Cauchy-like structure of the divide-and-conquer
update matrix.

- **problem class**: symmetric TRIDIAGONAL eigenvalue problem, FULL
  spectrum (all N eigenpairs, including eigenvectors) — the paper explicitly
  notes real symmetric dense problems reduce to this via a prior
  tridiagonalization step ("we first reduce each matrix into its tridiagonal
  form by calling ScaLAPACK routines and then call PSDC").
- **correctness — both formulas given explicitly and concretely, the most
  complete correctness disclosure found across all three of this project's
  dense-factorization/decomposition tracks so far**: (1) orthogonality,
  defined in-text as `||I - Q*Q^T||_max` (max absolute entry norm), reported
  in a table (Table 3) at 64/256/1024/4096 processes for three synthetic
  matrix families (Clement, Hermite, Toeplitz — classic structured
  tridiagonal test matrices from the numerical-linear-algebra literature),
  values all in the 2.5e-14 to 3.8e-14 range, stated to be "in the same
  order as those by ScaLAPACK"; (2) backward-error residual, given as an
  explicit formula: `Residual = ||A - Q*Lambda*Q^*||_c / ||A||_2`, where
  `||X||_c` denotes the MAXIMUM Frobenius norm of X's individual COLUMNS
  (not the whole-matrix Frobenius norm) and `||X||_2` is X's spectral norm
  (largest singular value) — a column-wise-max normalization this project's
  other two dense-factorization tracks have not encountered.
- **workloads/inputs**: three synthetic structured matrix families (Clement,
  Hermite, Toeplitz, all standard closed-form tridiagonal test matrices, N
  up to 30,000 for the strong-scaling experiments) plus three REAL
  application matrices: a spherical-harmonic-transform (SHT) tridiagonal
  operator (N=30,000, closed-form construction), and two real symmetric
  sparse matrices from the SuiteSparse collection, SiO (N=33,401) and
  Si5H12 (N=19,896), reduced to tridiagonal form via ScaLAPACK before PSDC
  is applied — the SAME real-matrix-then-tridiagonalize pattern this
  track's dense-EVD papers (#3/#4 below) use for the OTHER end of the
  full-spectrum pipeline (dense-to-tridiagonal, vs. PSDC's
  tridiagonal-to-eigenpairs).
- **deflation threshold K — a required, disclosed, problem-specific
  parameter**: PSDC's rank-structured technique (PSMMA) is only applied when
  the "secular equation" size exceeds a threshold K; the paper uses K=20,000
  for its strong-scaling example (N=30,000) and K=15,000 for its real-matrix
  example, explicitly noting "the largest K for matrix Si5H12 is 15,489 when
  NB=64" — K is tuned per-matrix, not a single fixed constant.
- **tile/block size**: NB=64 chosen after an explicit ablation (NB in {64,
  128, 256} tested, found "very small" difference, NB=64 selected as better
  than NB<=32) — matches this project's general tile-size-disclosure
  convention, independently re-derived by this paper's own methodology.
- **timing protocol — explicit, unambiguous, and a clear best-of-N**: "To
  avoid performance variance during multiple executions, we evaluated the
  performance of [the kernel] twice in the same program and called that
  program three times, and chose the BEST results among these six
  executions" — a 2x3=6-way minimum, the most explicit best-of-N statistic
  found anywhere across this project's three dense-factorization-track
  surveys (more explicit than either `cholesky`'s or `qr`'s best-of-N
  evidence points, which state the count but not the exact 2-inner-x-3-outer
  structure).
- **metric**: speedup vs. PDSTEDC (ScaLAPACK's own divide-and-conquer
  tridiagonal eigensolver), 1.4x-1.6x across all tested cases; also compared
  against PHDC (a competing D&C variant built on STRUMPACK/HSS
  compression) — PSDC wins at high process counts, PHDC wins at LOW process
  counts ("It is better to use STRUMPACK when using few processes since
  HSS-based multiplications can save more floating point operations than
  BLR-based multiplications") — a genuine crossover, not a uniform win.
- **machine**: Tianhe-2, Guangzhou, 24 MPI ranks/node (one rank/core), dual
  12-core Intel Xeon E5-2692 v2/node; scales to 4,096 processes (171 nodes).

## 2. `journals/tpds/ShiLXSH21` — Planetary Normal Mode Computation:
Parallel Algorithms, Performance, and Reproducibility (TPDS'21)

Source: artifact repo `js1019/NormalModes` (root `README.md`,
`demos/README.md`, `demos/global_conf`) — no arXiv id in the track input; the
OA link (`osti.gov/biblio/1836943`) is a landing page, not fetched as
fulltext during this pass (flagged below). This is the track's one
**partial-spectrum, generalized-eigenvalue-problem** paper: it solves for
ONLY the eigenpairs inside a chosen frequency interval, not the full
spectrum, via a polynomial-filtered Lanczos method (built on the `pEVSL`
library).

- **problem class**: GENERALIZED symmetric eigenvalue problem, `A*x =
  lambda*B*x`, arising from a Continuous-Galerkin mixed-finite-element
  discretization of the elastic-gravitational system governing planetary
  normal modes — NOT a standard `A*x = lambda*x` problem, the one paper in
  this track whose eigenproblem has a nontrivial right-hand-side operator B.
  A "rotation" branch further extends this to include self-gravitation and
  rotation effects (not surveyed in detail here).
- **partial spectrum via polynomial-filtered Lanczos — the FREQUENCY
  INTERVAL is the tolerance-like, required, disclosed parameter this
  track's design brief calls for**: `demos/global_conf`'s
  `lowfreq`/`upfreq` fields (in mHz) directly bound the eigenvalue search
  window; the README states the actual eigenvalue bounds used internally are
  `(2*pi*freq*1e-3)^2` (a unit conversion from frequency to the operator's
  native eigenvalue units) — "you will then obtain ALL the eigenpairs in
  [lowfreq, upfreq] mHz," an interior-eigenvalue-slicing formulation, not a
  top-k/bottom-k extremal-eigenvalue formulation.
- **discretization order as an additional, disclosed accuracy parameter**:
  `pOrder` (1 or 2, the FEM polynomial basis order) trades matrix size
  against solution accuracy — a genuinely different kind of "tolerance" than
  a numerical convergence threshold, specific to this paper's PDE-discretization
  origin, that the other 4 papers in this track have no analogue of.
- **workloads/inputs**: four provided real/synthetic 3D planetary models of
  increasing size, shipped directly in the artifact (`models/input/`):
  a constant-property solid elastic ball (CONST, ~3k), a standard Earth
  model (PREM, ~3k), a Moon model (Mtopo, ~100k, pOrder=1 only — pOrder=2
  requires external model generation via the companion `PlanetaryModels`
  repo, not shipped here), and a Mars model (RTMDWAK, ~8k) — model names
  encode approximate mesh/DOF scale in thousands.
- **JOB parameter — problem-physics variant, disclosed**: `JOB=1` (no
  reference gravity) vs. `JOB=2` (with reference gravity) selects a
  genuinely different operator, not just a performance knob.
- **precision & correctness**: the README states the metric directly:
  relative residual `||A*x - lambda*B*x|| / ||lambda||` — this is the exact
  generalized-eigenproblem analogue of the `||Ax - lambda*x||` residual this
  track's design brief names as the mandatory correctness check — "typically
  around 1e-13" achieved in practice, though no numeric GATING threshold
  (as opposed to a typically-observed value) is stated in the README itself.
- **timing/scale**: not recoverable from the README/demos-config artifact
  alone; the paper's own abstract (from the track input JSON) states the
  underlying SC'18 polynomial-filtering eigensolver "scales to 20k cores" —
  no warmup/repetition/statistic protocol was found in the artifact
  materials read during this pass (flagged as an open question, same
  pattern as several `cholesky`/`lu`-track papers at comparable scale).
- **reproducibility framing**: this paper's own title and abstract are
  explicitly about reproducibility (it was selected as the SC'19 Student
  Cluster Competition's reproducibility-challenge benchmark, per the
  README's "News" section) — a track-appropriate signal that this paper's
  own evaluation methodology was designed for external reproduction from
  the start, unlike most papers surveyed across this project so far.

## 3. `conf/sc/WangHZSJWDZ25` — Rethinking Back Transformation in 2-stage
Eigenvalue Decomposition on Heterogeneous Architectures (SC'25)

Source: artifact repo `hansheng1001/EVD4SC2025` (root `README.md`,
`src/EVD/myEVD.cu` full source read, including its `main()` argument
parsing and its compile-time-gated correctness-check block) — no arXiv/OA
source available. This is the track's **full-spectrum, single-GPU, dense**
paper focused specifically on the BACK-TRANSFORMATION stage (recovering
eigenVECTORS of the original dense matrix from the tridiagonal
eigenvectors) of the standard "2-stage" symmetric EVD pipeline
(dense -> banded -> tridiagonal -> tridiagonal-eigensolve -> back-transform
through banded -> back-transform through dense).

- **problem class**: standard symmetric dense eigenvalue problem, `A*x =
  lambda*x`, FULL spectrum WITH eigenvectors (this paper's whole point is
  making eigenvector recovery, not just eigenvalues, fast).
- **required, disclosed tuning parameters**: matrix size `n`, block size `b`
  (default 32), and back-transformation panel width `nb` (default `4*b`) —
  all three are CLI-required positional arguments (`myEVD <n> <b> <nb>`, the
  program refuses to run without all three) — the cleanest example of
  tile-size-as-mandatory-CLI-argument (rather than an optional flag with a
  silent default) found across this project's three dense-factorization-
  track surveys.
- **precision-adjacent technique**: the source references an "ozimmu"
  (Ozaki-scheme, mixed-precision emulated-high-precision GEMM via low-
  precision tensor cores) syr2k kernel (`g_tc_ozimmu_syr2k_ZY`), suggesting
  at least part of the back-transformation pipeline can use tensor-core-
  emulated FP64 GEMM rather than native FP64 — not confirmed as the
  DEFAULT code path from the excerpt read (flagged as an open question).
- **correctness — computed but DISABLED BY DEFAULT, the third occurrence of
  this exact pattern found across this project's three dense-factorization
  tracks**: gated behind a `#if CHECH_EVD_RESULT_ENABLE` compile-time flag
  (default state not confirmed, but the flag's existence at all implies it
  is not unconditionally on), the code computes: (1) `checkOrthogonality()`
  (an orthogonality check on the final assembled eigenvector matrix, exact
  formula not extracted, only its call site); (2) an explicit backward-error
  residual, computed as `||A - Q*D*Q^T||_2 / ||A||_2 / m` ("Backforward
  err" in the source's own print statement) — a spectral-norm ratio
  ADDITIONALLY normalized by the matrix dimension `m`, a THIRD distinct
  residual normalization convention found in this track alone (compare
  LiaoLLR21's column-wise-Frobenius-over-spectral-norm above, and this
  track's dense-EVD-sibling ASE/cuEV's plain max-entrywise-difference
  convention below).
- **metric**: TFLOPS computed specifically for the final GEMM-heavy
  back-transformation step, via `2*n*n*(m - n/3) / (time*1e9)` — a
  GEMM-shaped flop count (this paper's own contribution replaces BLAS3
  bulge-chasing-back-transform operations with BLAS2 ones for part of the
  pipeline, so this metric targets the specific sub-kernel the paper
  optimizes, not the whole EVD's flop count).
- **timing protocol**: no warmup or repeated-measurement loop found in
  `main()` — a single run per invocation, `std::chrono::high_resolution_
  clock` used throughout for each pipeline stage.

## 4. `conf/ppopp/WangDZWZLJZ25` — Improving Tridiagonalization Performance
on GPU Architectures (PPoPP'25)

Source: artifact repo `ynkrue/cuEV` — this repository, as it exists today,
has been renamed/rewritten to "ASE" (Accelerated Symmetric Eigensolver), an
apparent successor/consolidation artifact that explicitly cites BOTH this
paper AND `WangHZSJWDZ25` (paper #3 above) as its algorithmic references, and
implements the full pipeline both papers together describe (tridiagonalize +
back-transform). Sources read: root `README.md`, `bench/bench.cpp` (full
source), `test/solve_test.cu` (full source) — no arXiv/OA source available
for the original PPoPP'25 paper text itself; this survey's evidence is
therefore for the SUCCESSOR artifact's implementation of the paper's
algorithm, flagged explicitly as such.

- **the paper's own headline finding, stated directly in its abstract (track
  input JSON), is a striking performance-attribution fact worth preserving
  verbatim**: "In Nvidia's cuSOLVER library, the FP64 precision
  tridiagonalization process only reaches 2.1 TFLOPs out of 67 TFLOPs on
  H100 GPU, and it consumes ... over 97% of the elapsed time in the entire
  EVD process" — this is the paper's own justification for benchmarking
  tridiagonalization AS ITS OWN KERNEL, separately from the full EVD
  pipeline, rather than only reporting an end-to-end EVD number (a
  timing-scope decision directly analogous to this project's general
  preprocessing-separately-reported principle, here applied to an
  algorithmic STAGE rather than to setup/preprocessing).
- **algorithm** (from the successor artifact's own README, citing both
  source papers): Double Blocking Band Reduction (DBBR, dense -> banded),
  data repacking, wavefront bulge chasing (banded -> tridiagonal, persistent
  kernel with point-to-point atomics), divide-and-conquer tridiagonal
  eigensolve (delegated to LAPACK `*stedc`, NOT reimplemented — this
  artifact's D&C stage is exactly the same primitive PSDC (#1 above)
  proposes replacing, giving this track an internal comparison point: PSDC
  claims 1.4-1.6x over PDSTEDC specifically because artifacts like this one
  still call a stock D&C routine), back-transformation via a
  register-sliding-window kernel + WY-block GEMMs.
- **workloads/inputs — the successor artifact's `bench.cpp`**: a
  DETERMINISTIC (fixed-seed LCG) symmetric matrix generator, entries in
  (-1,1), default size sweep {512, 1024, 2048, 4096, 8192, 16384, 32768} —
  covers this track's largest single-GPU dense sizes found in any artifact
  surveyed.
- **timing protocol**: CUDA events, explicit **best-of-N** (`iters=3`
  default, `best = std::min(best, ms)` over the loop) — matches this
  track's LiaoLLR21 finding (best-of-N is the modal practice, not an
  exception) rather than diverging from it.
- **metric — the field-standard full-symmetric-EVD-with-eigenvectors FLOP
  count, worth adopting as this track's uniform convention**:
  `GFLOPS = (22/3)*n^3 / time` — matches the LAPACK convention for
  `JobZ='V'` (eigenvectors requested), i.e. this is not an
  implementation-specific formula but the field's own standard count,
  distinct from either the tridiagonal-D&C-only or GEMM-only formulas
  found in papers #1 and #3 above.
- **per-stage timing breakdown available as an optional flag**
  (`--timing`): DBBR / bulge-chasing / D&C / bulge-chasing-back / SBR-back,
  reported as average-ms-per-stage — directly operationalizes this paper's
  own "tridiagonalization dominates" finding as a required-disclosure
  breakdown, matching this track's general principle (borrowed from
  `cholesky`) that a dominant sub-cost should be separately reported, not
  folded into one number.
- **precision & correctness — SEPARATE test binary, unlike EVD4SC2025's
  disabled-by-default compile flag**: `test/solve_test.cu` is a full
  GoogleTest suite, run independently of `bench.cpp`. It validates against
  LAPACK's `dsyevd` (its own stated reference) on: random matrices (5
  sizes, 32 to 1024), the IDENTITY matrix, a matrix with REPEATED
  eigenvalues (explicitly engineered "maximal deflation" — a stress test
  for the D&C algorithm's deflation logic), and a matrix with CLUSTERED
  (near-equal) eigenvalues ("forces the secular [equation] path" — a stress
  test for the numerically-hardest case a D&C tridiagonal eigensolver
  faces). Checks: eigenvalue SORTEDNESS, eigenvalue VALUES within a
  size-dependent tolerance vs. `dsyevd`, reconstruction residual
  `max|A - Q*Lambda*Q^T| / scale` (max-entrywise, not Frobenius or
  spectral-norm — a FOURTH distinct residual normalization convention in
  this one track), and orthogonality `max|Q^T*Q - I|`. This is the single
  most thorough correctness-test design found across this project's three
  dense-factorization-track surveys, and notably fixes the
  "disabled-by-default" problem its own sibling paper's artifact
  (EVD4SC2025) has — worth flagging as a positive counter-example, not just
  citing the negative pattern again.

## 5. `conf/ppopp/YangNYLDXJ026` — A Diagonal Block Memory-Aware Polynomial
Preconditioner for Linear and Eigenvalue Solvers (PPoPP'26)

Source: artifact repo `YXJ-123/DBMPK` (root `README.md`, `DBMPK.cpp` full
source read) — no arXiv/OA source available; this is the most recent paper
in the whole corpus this project has surveyed (PPoPP 2026). This paper is
**not itself an eigensolver** — it optimizes the matrix-power kernel (MPK,
`A^d * x`) that polynomial preconditioners for BOTH linear-system solvers
AND eigenvalue solvers depend on. It is included in this track because
polynomial-FILTERED eigensolvers (e.g. Chebyshev-filtered subspace
iteration, or the polynomial-filtered Lanczos method ShiLXSH21/`pEVSL` use,
paper #2 above) are built directly on top of exactly this kernel — this
survey treats DBMPK as the KERNEL-LEVEL half of the same polynomial-
filtering technique ShiLXSH21 exercises at the APPLICATION level, not as an
unrelated inclusion.

- **problem/kernel shape**: sparse matrix-power kernel `A^d * x`, `A` read
  from a Matrix Market (`.mtx`) file (SuiteSparse-collection-compatible
  format, general or symmetric storage handled explicitly in the CSR-
  conversion code); polynomial DEGREE `d` (CLI flag `-d`, default 20 in the
  README's own example invocation) is the direct kernel-level analogue of
  ShiLXSH21's filter-polynomial-degree parameter (not directly exposed in
  the artifact materials read for #2, but algorithmically the same knob).
- **workloads/inputs**: any SuiteSparse-format sparse matrix supplied via
  `-in`; no fixed suite/selection criterion is enforced by the artifact
  itself (a user-supplied path), unlike this project's `spmv`/`spmm` tracks'
  curated SuiteSparse subsets.
- **timing protocol**: a FIXED, hardcoded repetition count of **50
  iterations** (`int iterations = 50;`, with a commented-out
  adaptive-repetition-count alternative left in the source but not active)
  — the one paper in this whole track (and one of very few across all three
  of this project's dense-factorization-track surveys) with a clean, fixed,
  non-best-of-N repetition count; no explicit statistic beyond total-
  time-over-iterations (i.e., MEAN), matching MunksgaardHSO22's convention
  in the `lu` track.
- **metric**: `GFlop/s = 2*d*iterations*nnz*1e-9 / total_time` — the
  standard SpMV-chain flop count (2 flops/nonzero/multiply, `d` multiplies
  per application of `A^d`), compared directly against an MKL sparse-BLAS
  baseline (`mkl_sparse_d_mv`, called in the same `d`-deep chain) using the
  SAME flop formula for both, a clean apples-to-apples metric definition.
- **precision & correctness**: **no correctness/residual check found** in
  the portion of `DBMPK.cpp` read (the driver reads the matrix, runs the
  power kernel, and reports performance only) — consistent with this being
  a pure micro-kernel-performance paper rather than a solver-correctness
  paper; any correctness claim for the polynomial PRECONDITIONER built on
  top of this kernel (as opposed to the kernel itself) would necessarily
  live in a downstream solver's own convergence behavior, not in this
  kernel-level artifact.

## Divergences

- **Full spectrum vs. partial (k eigenpairs) is a real, not superficial,
  split in this track**: #1, #3, #4 all compute the FULL spectrum (with
  eigenvectors); #2 computes ONLY the eigenpairs inside a disclosed
  frequency interval (an interior-eigenvalue-slicing formulation, not a
  top-k-extremal one); #5 is the kernel underneath exactly the polynomial-
  filtering technique that makes #2's partial-spectrum approach efficient.
  A spec that only covered full-spectrum dense/tridiagonal solvers would
  miss the one paper in this track (#2) that is explicitly about NOT
  computing the full spectrum, and the design brief for this track calls
  this split out directly.
- **No paper in this 5-paper track solves a genuinely NONSYMMETRIC
  eigenproblem.** #1, #3, #4 are symmetric (tridiagonal or dense); #2 is a
  symmetric-definite GENERALIZED problem (A*x = lambda*B*x, not the plainer
  A*x = lambda*x, but still built from a symmetric FEM discretization); #5's
  underlying kernel is agnostic to symmetry but is applied, per its own
  paper's framing, to Krylov solvers that in this track's own literature are
  all symmetric-eigenproblem-oriented. This is analogous to the `qr` track's
  TSQR/CAQR gap: a genuine absence in this specific 5-paper set, not a
  methodological disagreement to reconcile. Flagged in open_questions rather
  than resolved by fabricating a nonsymmetric variant this track's own
  papers do not evidence.
- **Residual/backward-error NORMALIZATION differs in FOUR distinct ways
  across just 3 of the 5 papers that report one at all** (a notably higher
  divergence rate than either the `lu` or `qr` tracks, where at most 2-3
  distinct residual conventions were found): LiaoLLR21's `||A - Q*Lambda*
  Q^*||_c / ||A||_2` (max-column-Frobenius-norm over spectral-norm);
  WangHZSJWDZ25's `||A - Q*D*Q^T||_2 / ||A||_2 / m` (spectral-norm ratio,
  ADDITIONALLY divided by matrix dimension); the ASE/cuEV successor
  artifact's `max|A - Q*Lambda*Q^T| / scale` (plain max-entrywise
  difference, normalized by the input's own max-abs-entry scale); and
  ShiLXSH21's `||A*x - lambda*B*x|| / ||lambda||` (a per-eigenpair relative
  residual, generalized-eigenproblem-specific, no matrix norm involved at
  all since it operates on a single eigenvector at a time, not the full
  factor). This project's own fairness principle (report both time and a
  numerical correctness signal) is well-served by ANY of these, but a
  cross-paper GFLOP/s comparison without also disclosing which residual
  convention gated it would be comparing apples whose skins were peeled
  four different ways.
- **Correctness-check-present-but-disabled/separated-from-timing is again a
  recurring pattern**, the fourth distinct occurrence across this project's
  three dense-factorization-track surveys (after OpenMxP's `--checksum` in
  `lu`, hicma-x's `--check` in `cholesky`, cuTensor-tubal's `#if 0` diff in
  `qr`): WangHZSJWDZ25's EVD4SC2025 artifact gates its own correctness
  check behind `#if CHECH_EVD_RESULT_ENABLE`. Its own successor artifact
  (the ASE/cuEV repo covering paper #4 too) FIXES this by moving correctness
  entirely into a separate, always-run GoogleTest suite (`solve_test.cu`) —
  a genuine methodological improvement worth citing as a positive
  counter-example alongside the now-familiar negative pattern.
- **Timing statistic**: explicit best-of-6 (#1, the most granular best-of-N
  structure found in this whole project so far), best-of-3 (#4's successor
  artifact), no repeated-measurement loop at all (#3), fixed-50-iterations-
  mean (#5), not recoverable (#2). No paper in this track reports median.
  Consistent with this project's established fairness principle, the spec
  below fixes the statistic to median with min/max reported.
- **Tile/block-size disclosure is unusually strong in this track compared
  to `lu`/`qr`**: #1 explicitly ablates NB in {64,128,256}; #3 REQUIRES
  block size `b` and back-transform panel width `nb` as positional CLI
  arguments (refuses to run without them, the strictest disclosure
  enforcement found in any artifact surveyed across this project); #4's
  DBBR stage has its own internal blocking not separately exposed as a CLI
  parameter in the excerpt read (flagged as an open question). This is one
  respect in which this track's own literature is ALREADY closer to this
  project's fairness principle than `lu`/`qr` were, and the spec below
  mostly just formalizes what these two papers already do.

## Open questions

- ShiLXSH21's own paper text (OA link is an OSTI landing page, not fetched
  as fulltext) was not read during this pass — the artifact's README/
  demos-config gives a strong methodology picture (frequency-interval
  slicing, pOrder, JOB) but no timing/repetition/statistic protocol was
  found; the paper's own abstract states the underlying SC'18 method
  "scales to 20k cores" but no per-run timing convention was recovered.
- WangHZSJWDZ25's `ozimmu`-referenced (Ozaki-scheme, tensor-core-emulated
  FP64 GEMM) code path's actual role in the DEFAULT `myEVD` execution path
  was not confirmed from the source excerpt read — whether the paper's own
  reported numbers use native FP64 GEMM or the Ozaki-emulated path (or
  both, as a comparison) needs the full `myBase.h`/`kernelOther.h` sources,
  not fetched during this pass.
- Whether EVD4SC2025's `CHECH_EVD_RESULT_ENABLE` flag defaults to on or off
  in the shipped `CMakeLists.txt`/build configuration was not confirmed
  (only the `#if` guard's existence in `myEVD.cu` itself was read) — needed
  to know whether the paper's own reported performance numbers were
  measured with or without the correctness check active in the same run.
- DBMPK's polynomial degree `d=20` default and `iterations=50` fixed
  repetition count are both taken from the README's own example invocation
  and the source's own hardcoded constant respectively; whether the paper's
  own reported 26.6%-38.4% MPK-performance-improvement headline figures used
  these exact values, or a sweep the paper reports but the shipped artifact
  does not default to, was not confirmed (the abstract in the track input
  JSON states a range, implying a sweep exists somewhere).
- Whether DBMPK's own paper (not read as fulltext, no arXiv/OA source
  available) reports ANY correctness/accuracy check for the polynomial
  preconditioner's effect on downstream solver convergence (as opposed to
  the raw kernel's own performance) was not investigated during this pass —
  the artifact alone gives no signal either way.
- Whether this track's assigned 5-paper input set omitting a genuinely
  nonsymmetric eigensolver reflects a real gap in the underlying corpus, or
  a narrower selection within a broader available set, was not investigated
  during this pass (same open question already flagged for the `qr` track's
  TSQR/CAQR gap).

