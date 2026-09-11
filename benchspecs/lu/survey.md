# LU track — evaluation-methodology survey

Track input: `data/track_inputs/lu.json`, 3 papers (all 3 surveyed, exceeding the
5-paper/all-papers minimum). Two sources are fulltext (arXiv PDF + OSTI PDF,
both extracted locally with `pypdf` since WebFetch could not render either PDF's
embedded font streams as text), the third via its artifact repo's build
scripts and kernel source (no arXiv/OA text available). This is a small,
methodologically split track: one paper is a *distributed, communication-
optimal, tournament-pivoted* dense LU (the LU half of the same COnfLUX/
COnfCHOX paper that anchors the `cholesky` track's plain-dense-fp64 variant);
one is an extreme-scale *no-pivoting, mixed-precision-with-iterative-
refinement* LU (the HPL-AI benchmark family); one is a single-GPU *no-
pivoting, single-precision* blocked LU decomposition used as one of seven
case-study kernels in a memory-optimizing array-language compiler paper. All
three therefore land on genuinely different points of the "pivoting strategy"
axis the task instructions flag as central to this track, which turns out to
be the single most consequential divergence found.

## 1. `conf/sc/KwasniewskiKBZS21` — On the Parallel I/O Optimality of Linear
Algebra Kernels: Near-Optimal Matrix Factorizations (SC'21)

Source: arXiv fulltext, downloaded and extracted locally
(`arxiv.org/pdf/2108.09337`, 15 pages via pypdf 6.14.2 — WebFetch's built-in
PDF rendering returned only binary stream data for this file) + artifact repo
`eth-cscs/conflux` README (the paper's maintained successor to the anonymized
`anonymousSC21/SC21` snapshot the track input points at; same target-repo
substitution already made in the `cholesky` spec for this paper's Cholesky
half). This is the same paper that anchors `cholesky`'s
`cholesky-dense-distributed-fp64` variant; here the focus is COnfLUX, its LU
counterpart.

- **pivoting strategy — the key LU-specific fact**: COnfLUX uses **row-masking
  tournament pivoting** (Section 7.3), citing Grigori/Demmel/Xiang's CALU
  tournament-pivoting result as its stability justification: "It is shown to
  be as stable as partial pivoting [29], which might be an issue for, e.g.,
  incremental pivoting." Two deliberate departures from textbook tournament
  pivoting are stated: (a) rows are never physically swapped — only pivot row
  *indices* are propagated via mask, to avoid the O(N³/P) row-swap I/O cost
  that would otherwise match the leading-order factorization cost; (b) the
  2.5D-decomposition-derived block size doubles as the tournament's reduction
  factor. Table 1 in the paper gives the concrete extra communication/
  computation cost of pivoting itself: `TournPivot` costs
  `v²⌈log₂(√P+1)⌉` communication and `v³/3⌈log₂(√P+1)⌉` computation per step
  (Cholesky's `potrf`-based diagonal step costs 0 communication by
  comparison — pivoting is LU's genuinely extra cost over Cholesky, not just
  a labeling difference).
- **baselines' pivoting differs from COnfLUX's own**: "Both MKL and SLATE base
  on the standard **partial pivoting** algorithm using the 2D decomposition."
  CANDMC/CAPITAL's pivoting strategy is not stated in the extracted text
  (flagged below as an open question) — their comparison in the paper is via
  author-supplied communication-cost *models*, not a stated pivoting scheme.
- **workloads/inputs**: randomly generated FP64 matrices, N = 2,048 up to
  N = 524,288; P = 2 up to 512 CPU nodes (identical sweep to the Cholesky
  half — same generator, same paper, same benchmark harness runs both
  factorizations back-to-back per the artifact README).
- **timing protocol**: `std::chrono`, max wall-clock across MPI ranks; 4 block
  sizes × 5 runs per (N,P) per the `eth-cscs/conflux` README ("`--run=5`...the
  number of repetitions (excluding a mandatory warm up round)"); exact
  statistic (mean/median/min) over those 5 runs not stated in either the
  paper text or the README — identical open question to the Cholesky half.
- **precision & correctness**: FP64 only. **No numerical residual or growth-
  factor check found** anywhere in the extracted paper text (searched for
  "residual", "correct", "verify", "backward error", "accuracy" — the only
  correctness-adjacent hits are about the *I/O-cost model's* validation
  against measured communication volume, not the factorization's numerical
  result). This matches the Cholesky half's finding exactly: I/O-cost-model
  validation stands in for numerical correctness in this paper.
- **metric**: wall-clock time-to-solution ("up to 3x" vs. best baseline) and
  aggregate communication volume (elements/processor, "up to 1.6x" reduction).
- **machine**: CSCS Piz Daint XC40, 1,813 CPU nodes, dual-socket Xeon E5-2695
  v4, 2 MPI ranks/node.

## 2. `conf/sc/LuMOEJW22` — Climbing the Summit and Pushing the Frontier of
Mixed Precision Benchmarks at Extreme Scale (SC'22)

Source: OSTI-hosted PDF, downloaded and extracted locally
(`osti.gov/servlets/purl/1997799`, 12 pages via pypdf — same WebFetch
PDF-rendering failure as paper 1) + artifact repo `at-aaims/OpenMxP` (README +
`src/` file listing: `getrf_nopiv.hpp`/`sgetrf_nopiv.cpp`,
`iterative_refinement.hpp`, `higham_mat_impl.cpp`, `panel_check.hpp`,
`main.cpp`). This is the HPL-AI benchmark family (the mixed-precision
companion to the double-precision HPL/Linpack benchmark used to rank
supercomputers), scaled to Summit and Frontier.

- **pivoting strategy**: **none.** Stated explicitly in the paper text: the
  whole point of the HPL-AI algorithmic design is "to omit the pivoting step
  during the LU factorization [5]," which is what "allows the use of a mixed
  precision solution process to obtain lower precision L̃ and Ũ factors." The
  repo's source file is even named `getrf_nopiv.hpp`/`sgetrf_nopiv.cpp`
  (LAPACK-style `getrf` naming with an explicit `_nopiv` suffix) — pivoting
  removal is a first-class, named design choice, not an oversight.
- **how no-pivot LU stays numerically safe**: two independent mechanisms
  found in the artifact (neither present for its own sake — both are
  necessary because no-pivot LU has an unbounded growth factor on an
  arbitrary matrix): (a) the input matrix is generated with a **Higham
  matrix generator** (`higham_mat_impl.cpp`) — a well-conditioned,
  by-construction test matrix, matching the standard HPL-AI-family practice
  of avoiding pathological growth rather than detecting it after the fact;
  (b) **FP64 iterative refinement (IR)** corrects the mixed-precision result
  back to double-precision accuracy regardless of the (small, by
  construction) error accumulated by the unpivoted factorization. The paper
  spells out the 3-step IR loop (residual → discrepancy solve → update) and
  states IR is stopped "once the discrepancy converges below a threshold
  providing a solution converged to double precision accuracy" — no numeric
  threshold value is given in the extracted text (flagged below).
- **precision recipe** (all four levels named explicitly, unlike a bare
  fraction): diagonal-block `getrf`/panel `TRSM` in **FP32**; the
  performance-critical trailing-matrix `GEMM` in **FP16×FP16→FP32**
  accumulate; residual computation `r = b − Ax̃` **regenerated on-the-fly in
  FP64** on the CPU via a parallel GEMV with a single `MPI_Allreduce`;
  solution-discrepancy solve in mixed FP32/FP64, stored in double.
- **workloads/inputs**: dense random matrix, generated + IR-corrected;
  headline scale N=1,368,570 (1.411 EFLOPS on a Summit partition), further
  scaled on a fraction of Frontier to 2.387 EFLOPS. Local per-GCD matrix size
  N_L tuned to fit AMD MI250X GCD memory (16GB/GCD on Summit-class V100
  nodes' comparison hardware vs. 96GB on Frontier's Instinct GPUs, per the
  paper's own hardware table).
- **timing protocol**: an explicit, dedicated **Warm up** subsection —
  documents a measured 20% Summit / negligible-but-nonzero Frontier
  performance gap between first and subsequent runs at full-application
  scale, from six consecutive full HPL-AI runs launched in one batch job; the
  repo exposes a `-w`/`--warmup` CLI flag. No fixed measured-repetition count
  or statistic is stated for the reported EFLOPS headline numbers themselves
  (these are single full-machine-scale runs, not a repeated-timing-loop
  protocol) — same "Gordon-Bell-scale runs don't state a rep/statistic"
  pattern already documented for several `cholesky`-track papers.
- **metric**: `GFLOPS/GCD = ((2/3)N³ + (3/2)N²)/time` — the standard
  LU-factorization-plus-triangular-solve FLOP count (HPL/HPL-AI convention,
  includes the RHS solve, not just the factorization) — aggregated to
  EFLOPS at full-system scale.
- **correctness gate as implemented in the artifact**: `panel_check.hpp`
  provides two checksum-style routines (`panel_check` templates) — one
  computes raw checksums of the diagonal/upper/lower panel blocks (a
  MPI-reduced debug checksum, not a numerical tolerance test), the other
  (`HMGen`-parameterized overload) computes a **max-relative-error against
  the known Higham-generator closed form** for diagonal/upper/lower entries
  and MPI-`MAX`-reduces it — this is a construction-specific correctness
  check (only valid because the test matrix is the Higham generator, not a
  general-purpose residual test) gated behind a `--checksum` CLI flag, i.e.
  **not run by default** in a performance measurement.
- **baselines**: no third-party library baseline in this paper (it targets
  full-machine HPC-AI ranking, not a head-to-head library comparison); the
  implicit baseline is the double-precision HPL result at the same scale.

## 3. `conf/sc/MunksgaardHSO22` — Memory Optimizations in an Array Language
(SC'22)

Source: artifact repo `diku-dk/futhark-mem-sc22` (root `README.md` +
`benchmarks/Makefile` + `benchmarks/lud/futhark/lud.fut`,
`lud-input.fut`, `Makefile` + `benchmarks/lud/reference/lud.cpp`, `Makefile`
+ `benchmarks/result-table.py`) — no arXiv id or OA link is present in the
track input, and none could be found; this paper's evaluation methodology is
recovered entirely from its artifact's build/run scripts. This paper is
**not primarily an LU paper** — it is a Futhark (functional GPU array
language) memory-optimization compiler paper whose evaluation reproduces
seven benchmark kernels from prior GPU-benchmark literature (NW, **LUD**,
Hotspot, LBM, OptionPricing, LocVolCalib, NN), each compared against a
reference (typically Rodinia) OpenCL/CUDA implementation. LUD ("Table II") is
its one LU-factorization-shaped kernel and is what places this paper in the
`lu` track's input set.

- **pivoting strategy**: **none** — this is the Rodinia LUD benchmark
  (blocked LU decomposition), whose input matrix is deliberately generated to
  make pivoting unnecessary: `lud-input.fut`'s doc-comment states outright
  "Generate well-conditioned matrix to avoid floating point errors.
  Algorithm from Rodinia," using a fixed exponential-decay-from-the-diagonal
  formula (`10·exp(λ·i)`, λ=−0.001) rather than Higham's generator (paper
  2's mechanism) or a randomized construction — a third, independent way
  this small track's papers arrive at the same "no-pivot LU needs an
  engineered input matrix" pattern.
- **precision**: **FP32 (single precision) only** — notably lower precision
  than either of the other two LU papers' baseline precision (COnfLUX: FP64;
  OpenMxP: FP64-corrected via IR). This is a genuine precision-scope
  divergence, not just a scale artifact.
  block-size mismatch already flagged in `cholesky`'s Divergences section.
- **tile/block size — disclosed but mismatched between implementations under
  comparison**: the Futhark version fixes `block_size = 32` in `lud.fut`; the
  reference OpenCL/Rodinia implementation (`lud.cpp`) defaults to
  `BLOCK_SIZE 16` (compile-time macro, independently overridable via
  `RD_WG_SIZE_0`/`RD_WG_SIZE_0_0`). The artifact runs each implementation at
  its own default block size rather than sweeping/matching them — a
  divergence from this project's fairness principle (tile size should be a
  disclosed, matched-or-swept parameter, not left to differ silently between
  the two things being compared).
- **workloads/inputs**: three problem sizes only, N ∈ {8,192, 16,384, 32,768}
  (from `lud.fut`'s `-- ==` benchmark-input annotations), single square
  dense matrix per size, no matrix suite/selection criterion (this is a
  microbenchmark case study, not a systematic sweep).
- **timing protocol**: `RUNS` Makefile variable, **default 10** repetitions
  (`benchmarks/lud/futhark/Makefile`: `RUNS ?= 10`; `benchmarks/lud/reference
  /Makefile`: same default, passed to the Rodinia binary's own `-r $(RUNS)`
  flag). **Statistic: explicit arithmetic mean** — `result-table.py`'s
  `mean(x) = sum(x)/len(x)`, applied uniformly to both the Futhark and
  reference runtimes before computing the reported speedup ratio. **Timer**:
  the Futhark side uses `futhark bench --backend=opencl`'s own device-side
  OpenCL event timer (`bin/futhark`, precompiled in the artifact); the
  reference side uses the Rodinia binary's own host-side `gettimeofday()`
  wrapper (`common.c`'s `gettime()`), invoked once per internal repetition
  inside the same process and piped through `jq` into the same JSON schema —
  **two different timer mechanisms (device event vs. host wall-clock)** feed
  the same mean-of-10 statistic, a timer-provenance mismatch this track's
  spec should not silently inherit.
- **precision & correctness**: **no residual, orthogonality, or growth-factor
  check found** in any of the benchmark/build scripts inspected — `futhark
  bench` here runs the bare `generate`-only entry point with no paired
  expected-output block, and the reference Rodinia binary's own `lud.cpp`
  likewise contains no validation call in the timed path. This is a
  deliberate omission specific to *this* artifact's harness, not a Futhark
  language limitation — Futhark's own idiomatic `-- ==` test syntax does
  support `output { ... }` / `auto output` correctness blocks, they are just
  not used for this particular benchmark.

## Divergences

- **Pivoting strategy is the single defining axis of this track, and every
  paper lands on a different point of it**: partial pivoting (#1's MKL/SLATE
  baselines, textbook LAPACK-style, O(N) latency but well-understood
  worst-case 2^(N−1) growth bound, essentially never observed in practice);
  **tournament pivoting** (#1's own COnfLUX, "as stable as partial pivoting"
  per its own cited source, but with an explicitly flagged caveat about
  incremental-pivoting use cases, and CANDMC/CAPITAL's pivoting scheme is
  simply not stated in the available text); and **no pivoting at all** (#2
  and #3, both compensating by *engineering the input matrix* rather than by
  bounding growth for an arbitrary matrix — #2 additionally compensates with
  FP64 iterative refinement, #3 has no compensating correctness mechanism at
  all beyond the input-matrix choice). A benchmark spec for this track that
  reports only time/GFLOP/s without disclosing which of these three
  strategies produced the number — and, for the no-pivot cases, whether the
  input matrix was specially constructed to make that safe — would make
  cross-paper comparison actively misleading, not just incomplete.
- **No paper in this track reports a growth factor**, despite pivoting
  strategy being exactly the axis that growth factor is designed to
  diagnose. #2 and #3 both sidestep the question by construction (Higham
  generator; Rodinia's diagonally-dominant-by-formula generator) rather than
  measuring it; #1 doesn't report any numerical correctness signal for LU at
  all (I/O-cost-model validation only, matching its Cholesky half). This
  project's own fairness principle (report a numerical correctness signal,
  don't just assume a "standard" kernel is correct) argues for adding a
  growth-factor measurement as a first-class, spec-level addition — flagged
  in `notes_on_fairness` below, since it is not evidenced by any of the
  three papers themselves.
- **Precision scope differs**: FP64-only (#1), mixed FP16/FP32-with-FP64-IR
  targeting FP64-equivalent accuracy (#2), and FP32-only (#3) — a genuine
  three-way split, not a scale artifact (contrast with `cholesky`'s
  Gflop/Tflop/PFlop/ExaOp-unit divergence, which *is* purely a scale
  artifact).
- **Single-node vs. distributed scope**: #1 is distributed-only in its
  reported results (though the underlying binary can run on 1 node, same
  caveat as several `cholesky`-track papers); #2 is inherently a
  multi-node/multi-GPU extreme-scale benchmark by design (the IR step's cost
  is only amortized away "at larger scales," per the paper's own "Comments"
  section in the README); #3 is single-GPU only. Between the three papers,
  this track has real coverage of both single-node and distributed regimes,
  unlike some `cholesky` papers where the "single-node" variant had to be
  inferred from an underlying binary's capability rather than an actually
  reported number.
- **Timing statistic**: mean (#3, explicit), unstated-over-5-runs (#1,
  matches its Cholesky half exactly), and "no fixed rep/statistic, single
  full-scale run analyzed for warm-up variability instead" (#2, Gordon-Bell-
  scale). No paper in this track reports median. Consistent with this
  project's fairness principle (established in the `cholesky` spec), the LU
  spec below fixes the statistic to median with min/max reported.
- **Timer provenance mismatch inside a single paper's own harness** (#3
  only): device-side OpenCL event timer for the Futhark implementation vs.
  host-side wall-clock for the Rodinia reference, feeding the same reported
  speedup ratio — worth a specific fairness note since it is not a
  cross-paper divergence but an intra-paper one.

## Open questions

- CANDMC/CAPITAL's (paper #1's baseline libraries) pivoting strategy for LU
  is not stated in the extracted arXiv text — their comparison in the paper
  is via author-supplied communication-cost models rather than a described
  pivoting algorithm. Needed to complete the pivoting-strategy comparison
  table below.
- The exact statistic (mean/median/min) `eth-cscs/conflux`'s artifact
  computes over its "5 runs" is not stated in the paper text or README
  (same open question already flagged in the `cholesky` survey for this
  paper's Cholesky half) — would need the benchmark-driver source itself.
- OpenMxP/HPL-AI's iterative-refinement convergence threshold (the numeric
  value the solution discrepancy must fall below) is not stated in the
  extracted OSTI PDF text — likely a fixed constant in `iterative_refinement
  .hpp`, not fetched during this survey pass. The standard HPL/HPL-AI
  convention uses a residual-based stopping test with an established
  constant (commonly quoted as 16 in the HPL rules document); this spec
  provisionally adopts that convention below but flags it as **not
  confirmed** from this paper's own text or source.
- Whether OpenMxP's `panel_check.hpp` `--checksum` correctness path is ever
  enabled during the paper's own reported EFLOPS runs (as opposed to being a
  debug-only path) was not confirmed — the CLI flag exists but the paper
  text doesn't state whether the headline numbers were measured with or
  without it active.
- Futhark-mem-sc22's LUD case study has no correctness check in its shipped
  harness at all; whether the underlying `futhark bench` / Rodinia binaries
  were separately validated (e.g., via Futhark's own `futhark test`) before
  the performance numbers were collected, outside of what `make table2`
  itself runs, was not confirmed.
