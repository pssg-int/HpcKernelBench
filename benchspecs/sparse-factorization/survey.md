# Survey: sparse-factorization

Scope note: this track is specifically about **sparse** direct factorization
(LU, Cholesky, QR on matrices stored in sparse formats, with a symbolic
analysis / fill-reducing-ordering step ahead of numeric factorization). This
is distinct from this repo's separate `lu`, `qr`, and `cholesky` tracks, which
cover dense/tile factorization (HPL-style mixed precision, I/O-optimal dense
kernels, TLR/mixed-precision dense Cholesky). All 5 input papers were
surveyed; 2 via arXiv/PDF fulltext (gSoFa, GPTuneCrowd), 3 via artifact-repo
README/build-and-run scripts (PanguLU, STM-multifrontal QR, swCholesky) since
none of the 5 have a fulltext-fetchable arXiv/OA copy of the conference/journal
version except gSoFa.

## conf/ipps/ChoDKLLL23 — GPTuneCrowd: "Harnessing the Crowd for Autotuning HPC Applications" (IPDPS 2023)

Source: arXiv-hosted author PDF, `https://liuyangzhuan.github.io/docs/GPTuneCrowd_2023.pdf`
(fulltext read in full).

- **What it actually is**: not a new factorization kernel — a crowd-sourced,
  transfer-learning Bayesian-optimization autotuner (GPTuneCrowd, built on
  GPTune). It is in-scope for this track because two of its four case studies
  autotune parameters of sparse direct/iterative solvers (SuperLU_DIST's 2D
  LU, Hypre's BoomerAMG), and a third (NIMROD) internally calls SuperLU_DIST's
  3D sparse LU factorization once per Newton/GMRES step.
- **Workloads/inputs** (sparse-factorization-relevant case studies only):
  - SuperLU_DIST 2D sensitivity-analysis case study: tuning parameters
    `[COLPERM, LOOKAHEAD, nprows, NSUP, NREL]`; sensitivity computed on matrix
    **Si5H12** (PARSEC group, SuiteSparse Matrix Collection), 500
    pre-collected samples on 4 Cori Haswell nodes; validated on a *different*
    matrix, **H2O** (same PARSEC group, "similar sparsity pattern"), also 4
    Haswell nodes.
  - Hypre case study: structured 3D Poisson grid `nx=ny=nz=100` (sensitivity
    analysis, 1000 samples, 1 Cori Haswell node) then `nx=ny=nz=120`
    (validation run); this is a solver, not a sparse direct factorization, but
    shares the "reduce tuning search space" methodology.
  - (Not sparse-factorization: PDGEQRF is *dense* ScaLAPACK QR; NIMROD is an
    MHD PDE code whose *inner* linear solve calls SuperLU_DIST 3D.)
- **Timing protocol**: each tuning "run" = a Bayesian-optimization search of
  `NS` function evaluations (10 or 20 shown); each function evaluation runs
  the target application **once** at a proposed parameter configuration and
  records wall-clock runtime as the BO objective. Each experiment condition
  repeated **3-5 times with different random seeds**; plots show mean of
  best-so-far result over evaluations, shaded band = std dev across seeds.
  No GPU/CPU-side event timer specified beyond "measured runtime" of the
  whole application invocation (implies host-side wall-clock around the
  whole SuperLU_DIST factor+solve call).
- **Timing scope**: end-to-end — reordering method (`COLPERM`) is itself a
  *tuned parameter*, not fixed/excluded, so its cost and effect are folded
  into the single measured runtime along with symbolic + numeric
  factorization + triangular solve. Symbolic and numeric phases are **not**
  reported separately.
- **Precision & correctness**: fp64 implied (SuperLU_DIST/Hypre defaults);
  the paper does not check or report factorization correctness/residuals —
  runs that error out (e.g., OOM at a bad `NSUP`/`nprows` combination) are
  simply **discarded** when fitting the tuner's surrogate model, which is a
  soft form of correctness gating but not a quantified tolerance.
- **Metric**: wall-clock runtime (seconds) of the tuned application per
  function evaluation; "speedup" = ratio of the *tuned* runtime under a
  reduced/transfer-learned search vs. the tuned runtime of a non-transfer
  Bayesian-optimization baseline (`NoTLA`), reported at a fixed evaluation
  budget (e.g., "10th evaluation"). This is a *tuning-quality* speedup, not a
  kernel-throughput speedup vs. a vendor library.
- **Baselines**: `NoTLA` (GPTune's plain single-task BO, no transfer
  learning) is the only baseline; there is no comparison to a fixed/default
  parameter configuration of SuperLU_DIST/Hypre as an absolute reference
  point in the sparse-factorization case studies (Fig. 6/7 do show "original
  tuning problem" vs. "reduced tuning problem" curves, both still BO-tuned).
  No comparison to cuSOLVER/MKL/PARDISO etc.

## conf/sc/FuZWLLYZGLZ0023 — PanguLU (SC 2023, Best Paper)

Source: GitHub README + `examples/run.sh` + `examples/Makefile` at
`SuperScientificSoftwareLaboratory/PanguLU` (fulltext of the SC23 paper
itself was not fetchable — ACM DL/IEEE Xplore paywalled, SC23 proceedings
page returned HTTP 403 to WebFetch; evaluation numbers below are corroborated
via web-search snippets of the abstract/press coverage, not primary text —
flagged as lower-confidence and listed in Open Questions).

- **What it is**: a 2D regular block-cyclic **sparse LU** direct solver for
  `Ax=b` on distributed heterogeneous (multi-node, multi-GPU) systems,
  written in C+MPI+OpenMP+CUDA. Deliberately avoids the
  multifrontal/supernodal dense-BLAS approach (argues it wastes FLOPs on
  fill-in zeros); instead keeps blocks sparse and dispatches per-block to an
  adaptively-selected sparse BLAS kernel.
- **Pipeline phases, each separately timed** (per the repo's own release
  changelog, v3.0.0-v3.5.0): **reorder** phase (optional MC64 row-scaling +
  optional METIS fill-reducing ordering) → **symbolic factorization** phase
  → **preprocessing** (block partitioning into the 2D grid, `-nb` block
  size) → **numeric factorization** phase (reports GFLOP/s since v3.5.0) →
  **triangular solve** (`pangulu_gstrs`). This is the clearest example in
  this track of a paper/artifact that treats symbolic-vs-numeric-vs-ordering
  as separately-reportable phases rather than one bundled number.
- **Workloads/inputs**: per web-search summary of the SC23 results, "sparse
  matrix benchmarks are all downloaded from the SuiteSparse Matrix
  Collection"; the geometric-mean comparison point is "1.50x and 3.32x
  [speedup], compared with PanguLU and SuperLU_DIST respectively" (this
  phrasing from search snippets is ambiguous/possibly a paraphrase artifact —
  see Open Questions) and headline numbers are speedups of **11.70x (A100
  cluster) / 17.97x (MI50 cluster)** over SuperLU_DIST on 128 GPUs, scaling
  up to **47.51x / 74.84x** over a single GPU of the same type. Exact matrix
  list/count not recovered from either the repo or search snippets.
- **Run/build interface** (from `examples/run.sh`, confirmed by direct repo
  read): `mpirun -np <process_count> ./pangulu_example.elf -nb <block_size>
  -f <path.mtx> [-r <rhs>]`. Matrix Market (`.mtx`) input; `-nb` (block/tile
  rank size) is a required, user-supplied parameter — i.e., block size is
  explicitly NOT a silently-fixed default, matching this track's general
  "disclose tile/block parameters" concern seen in the sibling `cholesky`
  track.
- **Precision**: supports fp32/fp64 real and complex via compile-time flags
  (`-DCALCULATE_TYPE_R64/CR64/R32/CR32`); no single precision is a hidden
  default — must be chosen at build time.
- **Baselines**: SuperLU_DIST (the dominant sparse-LU baseline across this
  entire track — also the library GPTuneCrowd and NIMROD tune/call).
- **Correctness**: not confirmed from available sources (the `pangulu_gssv`
  convenience function factors+solves, implying a residual check would be
  natural, but no explicit tolerance was found in README or run scripts) —
  flagged as an open question.

## journals/tpds/GaihreLL22 — gSoFa: Scalable Sparse Symbolic LU Factorization on GPUs (TPDS 2022)

Source: arXiv fulltext, `arxiv.org/abs/2007.00840` (v2, TPDS-extended version
of the ICS'21/DAC preliminary work), read via ar5iv HTML mirror. Corroborated
by the repo README (`Anil-Gaihre/gSoFa`) which documents the exact input
pipeline.

- **What it is**: the first GPU-native **symbolic**-factorization design for
  nonsymmetric sparse LU (fill-in pattern + supernode identification only —
  explicitly does **not** do numeric factorization or triangular solve; the
  paper's title and repo directory names, `with_space_optimization` /
  `without_space_optimization`, both refer to symbolic-phase memory
  optimizations only).
- **Workloads/inputs**: 13 named SuiteSparse Matrix Collection matrices
  spanning circuit simulation, structural, CFD, thermal, economic, chemical
  engineering, electromagnetics, bioengineering domains; sizes from
  **BCSSTK18** (order 11,948, ~149K nnz) up to **AUDIKW_1** (order 943,695,
  ~77.6M nnz); also BBMAT, EPB2, G7JAC200SC, LHR71C, MARK3JAC, RMA10 (smaller
  tier) and DIELFILTER, HAMRLE3, PRE2, STOMACH, TWOTONE (larger tier).
- **Critical scoping detail (repo-confirmed)**: the input to gSoFa is **not**
  the raw matrix — the README states "We use SuperLU_DIST... to get the exact
  reordered matrix that it uses for the Symbolic Factorization for fair
  comparison. The reordered matrix ... is given as an input to gSoFa." I.e.,
  fill-reducing ordering is run once by the baseline, its output is dumped to
  CSR/ASCII files, and *that fixed, pre-ordered matrix* is fed to both gSoFa
  and the baselines. **Ordering is explicitly out of the timed critical path**
  for this paper — the cleanest example in this track of that design choice.
- **Hardware**: OLCF Summit nodes, 6x V100 (16-32GB) + dual-socket POWER9
  (42 cores) per node; scaled 1-44 nodes (6-264 GPUs).
- **Timing protocol**: "average of three runs" (per-paper text); no explicit
  warmup mentioned; timer is wall-clock (`wtime.h` in the repo) around the
  GPU symbolic-factorization kernel launch region; profiled in more detail
  with `nvprof` for supplementary figures.
- **Metrics**: primary is wall-clock time / speedup; also reports
  Traversed-Edges-Per-Second (TEPS, a graph-traversal-style throughput unit
  appropriate since symbolic factorization is fundamentally a
  graph-reachability computation) and **memory throughput** in GB/s
  normalized to the V100's 900GB/s peak (avg 385.2GB/s = 43% of peak, max
  47%) — this is the only paper in the track reporting a memory-bandwidth
  utilization metric rather than a footprint (bytes used).
- **Baselines**: GLU3.0 (sequential Gilbert-Peierls, CPU, 1 thread), a
  custom "parallel fill2" implemented on both CPU (42 cores) and GPU for an
  apples-to-apples algorithm comparison, and SuperLU_DIST's own (distributed
  CPU) symbolic factorization phase at matching node counts.
- **Reported speedups**: avg 33.06x (range 11.32-97.11x) vs. parallel fill2
  (CPU); avg 50.1x (1.01-171.1x) vs. GLU3.0; avg 5x vs. SuperLU_DIST at 44
  nodes (gSoFa is *worse* than SuperLU_DIST at 1 node — i.e., the crossover
  point/single-node numbers are part of the paper's own honest reporting).
- **Correctness**: not explicitly quantified as a tolerance in the excerpt
  fetched; symbolic factorization is a discrete/combinatorial computation
  (fill-in pattern is either exactly right or wrong, no floating-point
  tolerance applies) so "correctness" here is presumably exact nonzero-count
  or nonzero-position agreement with GLU3.0/SuperLU_DIST's fill pattern.

## conf/sc/LinYWT021 — STM-multifrontal QR (SC 2021)

Source: GitHub README (`lsl036/STM-Multifrontal-QR-Factorization-Empowered-by-GCN`)
+ `STMMQR/README.md` + `STMMQR/test.sh` + `STMMQR/test.txt` (all read
directly; ACM DL fulltext paywalled, not fetched).

- **What it is**: multifrontal sparse **QR**, split into (1) symbolic
  analysis, where a GCN classifier adaptively picks the fill-reducing
  ordering algorithm (AMD / COLAMD / METIS / NESDIS) per-matrix instead of
  always using one default, and (2) numerical factorization, optimized via
  a "streaming task mapping" (STM) parallel scheduler with NUMA-affinity
  data placement, built on top of SuiteSparseQR's codebase (CAMD/CCOLAMD/
  metis-5.1.0 vendored in the repo).
- **Ordering is explicitly IN SCOPE and is the paper's main object of
  study** — the opposite scoping choice from gSoFa: this paper's contribution
  *is* picking the ordering method, so it cannot be excluded from the
  benchmark; instead it must be a first-class, disclosed/reported axis.
- **Workloads/inputs**: the public repo's runnable smoke-test set is 16 named
  SuiteSparse matrices listed in `STMMQR/test.txt` (dwt_992, lns_3937,
  bcsstk14, epb1, reorientation_8, sme3Dc, 3D_51448_3D, cvxqp3, t2d_q9,
  xenon1, bayer10, Zd_Jac2_db, ex18, c-67, language, mycielskian13 — sizes
  from ~70KB to ~96MB .mtx files, i.e., roughly 1K to several-100K rows). The
  paper's actual reported evaluation is much larger: README claims results
  "for nearly 80% matrices of the University of Florida Sparse Matrix
  Collection" and a separate file `GCNdata_408.txt` lists 408 matrices used
  to train/validate the GCN ordering-classifier — the full matrix list for
  the paper's own numbers was not recovered (open question).
- **Timing protocol / harness output** (from `STMMQR/README.md`, run via
  `./qrtest <matrix.mtx> <matrix-id> [ordering-id]`): writes
  `Results/QR_time.txt` recording, **per matrix, per run**: matrix ID,
  **analysis (symbolic) factorization time**, **numerical factorization
  time**, and an **error-verification** field — i.e., symbolic and numeric
  are reported as two separate numbers per run, and a correctness/error
  check is a first-class output of the harness (exact tolerance/formula not
  specified in the README text itself). `test.sh` runs each of the 16
  matrices **once** (no explicit repetition count visible in the script).
- **Hardware**: "Taishan Server", 120 processors / 4 NUMA nodes (ARM,
  per paper abstract); baseline comparison run on Intel Xeon 6248 (MKL
  Sparse QR).
- **Metrics/results**: GCN-adaptive ordering selection reduces average
  numerical factorization time by 20.78% vs. the (single) default reordering
  approach, at ~4% extra memory overhead (this is the only *explicit*
  memory-footprint delta reported anywhere in this track's survey, though
  as a relative percentage rather than an absolute byte count); overall the
  STM-Multifrontal-QR system is reported as "equal to or higher than" MKL
  on ~80% of matrices tested, and 1.22x average speedup over the original
  (non-GCN, non-STM) SuiteSparseQR baseline.
- **Baselines**: MKL sparse QR, original (vanilla) SuiteSparseQR.
- **Caveat directly stated by the authors**: the README explicitly says "GCN
  and STM-MQR are **not really combined**" in the released artifact "due to
  lack of accuracy of GCN automatic classifier and extra cost of attributes
  generation" — i.e., the shipped `qrtest` binary is the STM numeric-phase
  optimization alone (with a fixed default ordering choice unless overridden
  via the `[ordering-id]` CLI arg), not the full GCN+STM system the paper's
  headline numbers describe. This is a real artifact/paper mismatch worth
  flagging for any reproduction attempt.

## journals/tpds/LiLYLGYQ20 — swCholesky (TPDS 2020)

Source: GitHub README (`buaa-hipo/swCholesky`) + `software/README.md` +
`software/swCholesky/test.py` (all read directly; TPDS fulltext is IEEE
Xplore-paywalled and was not fetched — evaluation protocol below is
reconstructed from the harness script, not paper text, and should be treated
as lower-confidence than the other 4 entries).

- **What it is**: sparse **Cholesky** factorization (SPD matrices) optimized
  for the Sunway SW26010 manycore architecture (the same processor family as
  TaihuLight), using per-supernode task queues dispatched to a custom dense
  kernel library (`swGEMM`, `BLAS-sw`) plus an autotuning mechanism to match
  supernode computation to the architecture's core-group/scratchpad-memory
  structure.
- **Ordering is precomputed/given, not timed**: `test.py`'s job-launch
  template reads matrices from a `../FloridaSparse0/` directory (a named but
  unpublished subset of the SuiteSparse "Florida Sparse Matrix" collection —
  exact matrix list/count is **not** in the repo, open question) plus a
  matching `perm/` and `perm_x86/` sibling directory of **precomputed
  permutation files**, one per matrix — i.e., fill-reducing ordering is
  computed once, offline, and only the *numeric Cholesky factorization* of
  the already-permuted matrix is what the timed binary (`sw_cholesky`) does.
  This matches gSoFa's choice (ordering out-of-scope) and is the opposite of
  STM-QR/PanguLU/GPTuneCrowd.
- **Autotuned/exposed parameters** (positional CLI args to `sw_cholesky`,
  from `test.py`'s `gen_exec_cmd`): `chunk`, `costParam`, `levelParam`,
  `finalSeqNode` — supernode-merging / task-granularity knobs; the paper's
  "auto-tuning mechanism to select... kernel task queues" (per abstract)
  presumably sweeps these, though the checked-in `test.py` only exercises
  one fixed configuration per matrix (`chunk=1, costParam=64, levelParam=-1,
  finalSeqNode=4`) — the actual autotuning search loop was not found in the
  public repo.
- **Timing protocol**: **not recoverable from the repo.** Jobs are submitted
  via Sunway's `bsub`/`sw3run` batch launcher to per-matrix log files
  (`log/log_exec/<matrix>.mtx`); no repetition count, warmup, or timer
  mechanism is visible in `test.py` itself (it calls `os.system(exec_cmd)`
  once per matrix, no loop). The repo does ship `data/Figure_7..12_*.csv`
  files that appear to be the exact numbers plotted in the paper's figures,
  suggesting the actual protocol lives in un-published post-processing
  scripts, not in what's public.
- **Hardware**: Sunway manycore (SW26010-class; job launcher flags
  `-cgsp 64`, `-cross_size 28000` are Sunway-specific core-group/scratchpad
  settings) — this is a fairly exotic, hard-to-reproduce-off-China target
  for a general benchmark harness (flagged in Open Questions / fairness
  notes).
- **Baselines / correctness / memory**: none of these three are visible from
  the public repo; the paper abstract only says the goal is efficient
  mapping to Sunway vs. unspecified prior approaches. Flagged as open
  questions — would need the TPDS PDF (not accessible in this survey pass)
  to fill in.

## Divergences

1. **Symbolic vs. numeric factorization timing.** gSoFa measures **only**
   symbolic (numeric explicitly out of scope). swCholesky measures **only**
   numeric (ordering precomputed, and Cholesky's "symbolic" step —
   elimination-tree/supernode-partition construction — is not separately
   reported in what's public). PanguLU and STM-multifrontal-QR report
   **both, separately**, per phase. GPTuneCrowd reports **neither
   separately** — its case studies measure end-to-end application wall time
   with ordering-choice folded in as a tuned parameter. This track's spec
   must offer both a symbolic-only and a numeric-only kernel variant, and
   keep an explicit end-to-end variant for papers (and real HPC use, e.g.
   NIMROD/PDGEQRF-style outer loops) that legitimately care about the
   bundled number.

2. **Fill-reducing ordering: in or out of the timed region.** gSoFa and
   swCholesky exclude it (given as precomputed input, "for fair comparison"
   per gSoFa's own README). PanguLU and STM-multifrontal-QR make it a
   first-class, separately-timed phase — and for STM-QR, *choosing* the
   ordering algorithm (AMD/COLAMD/METIS/NESDIS) is literally the paper's
   contribution, so it cannot be hidden. GPTuneCrowd treats the ordering
   choice (`COLPERM` for SuperLU_DIST) as one of several autotuned knobs
   whose cost is folded into a single end-to-end number. The spec resolves
   this with a dedicated ordering-sensitivity variant (report factorization
   time under each of a fixed ordering-method list, on top of the
   ordering-excluded kernel variants) rather than silently picking one
   convention.

3. **Matrix suite size and overlap are both small.** No two papers in this
   track's survey share a named matrix except that STM-QR's `bcsstk14` (SPD,
   suitable as a Cholesky test case) coincidentally overlaps with the kind of
   matrix swCholesky's undocumented `FloridaSparse0` set would need. gSoFa
   (13 matrices), STM-QR's public smoke set (16) and its paper-scale claim
   ("~80% of the UF collection", i.e., likely thousands), GPTuneCrowd (2
   named PARSEC matrices), PanguLU (count unrecovered), swCholesky (count/
   names unrecovered) are five essentially disjoint or unknown sets. There is
   no shared "standard suite" to inherit directly — the spec below curates a
   fixed, named, union-of-what-was-actually-used-plus-standard-substitutes
   list per factorization family, rather than deferring to any one paper's
   ad hoc set.

4. **Correctness checking is inconsistent and mostly unquantified.** Only
   STM-multifrontal-QR's harness explicitly writes an "error verification"
   field per run (formula not confirmed from available text). gSoFa's
   "correctness" is really exact fill-pattern agreement (no floating-point
   tolerance applies to symbolic factorization). GPTuneCrowd discards
   failed/erroring configurations from its BO surrogate but does not check
   numeric accuracy of successful ones. PanguLU and swCholesky: no
   correctness methodology was recoverable at all from public sources. The
   spec mandates a residual-based gate (`||Ax-b||/||b||` after solve, or
   exact-fill-pattern match for symbolic-only variants) uniformly, which is
   stricter than at least 3 of these 5 papers' own public practice.

5. **Memory footprint is reported inconsistently and never as an absolute
   number.** gSoFa reports memory-bandwidth *utilization* (GB/s vs. peak),
   not footprint. STM-multifrontal-QR reports a *relative* footprint delta
   (+4% vs. baseline) attributable to its GCN classifier, not an absolute
   byte count. PanguLU's changelog claims a v5.0.0 GPU-memory-layout
   optimization but gives no number. gSoFa/PanguLU/STM-QR/swCholesky are all
   plausibly memory-capacity-bound at scale (fill-in growth is the classic
   sparse-direct-solver failure mode) yet none of the 5 papers reports peak
   RSS/peak-GPU-memory in absolute terms. The spec adds this as a required
   *reported* (not necessarily gating) quantity per the track instructions.

6. **Hardware heterogeneity is extreme and largely non-reproducible as-is.**
   Summit V100 nodes (gSoFa), unspecified A100/MI50 clusters (PanguLU),
   ARM "Taishan Server" (STM-QR), Sunway SW26010 (swCholesky), and NERSC
   Cori Haswell/KNL (GPTuneCrowd) are five different, largely
   inaccessible-today platforms. Per this project's Phase-2 ground rule
   (Perlmutter/any-machine reproducibility, not tied to the original paper's
   machine), the spec below is written platform-agnostically (CPU-multicore
   + optional single-GPU), which is achievable for gSoFa (GPU) and
   PanguLU/GPTuneCrowd (multi-core CPU or GPU) but requires a genuine
   reimplementation/recompilation for STM-QR (ARM NUMA-specific) and is
   essentially unreproducible off Sunway hardware for swCholesky — flagged
   explicitly rather than glossed over.
