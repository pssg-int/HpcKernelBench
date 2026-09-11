# SpGEMM track — evaluation-methodology survey

Track input: `data/track_inputs/spgemm.json` (13 papers). Surveyed 6 papers in
depth (5 via arXiv fulltext and/or artifact repo inspection, 1 — TileSpGEMM —
via its author-hosted PDF since it has no arXiv id), chosen to cover the
track's real spread of evaluation regimes: single-GPU shared-memory (Ocean,
TileSpGEMM), multi-GPU distributed (Trident, Hussain/CombBLAS), CPU
shared-memory with row reordering (clusterwise-spgemm), and tensor-core
mixed-precision SpGEMM embedded inside a solver (AmgT).

---

## Ocean (ICS 2026) — `conf/ics/LiG26`

Fast estimation-based GPU SpGEMM; replaces the exact symbolic pass with a
HyperLogLog (HLL) cardinality estimate.

- **Workloads**: two SuiteSparse-derived sets, each with an explicit FLOP
  (not nnz) selection threshold:
  - *square* set: 337 matrices, `flops(A×A) ≥ 100M`. Confirmed by reading the
    repo's own reproducibility list `utils/square.csv` (338 lines incl.
    header ⇒ 337 matrices; columns `id,Group,Name,rows,cols,entries`).
  - *rectangular* set: 64 matrices, `nnz(C) ≤ 10B` and `flops(A×Aᵀ) ≥ 100M`.
    Confirmed via `utils/rectangular.csv` (65 lines incl. header ⇒ 64).
- **Operation**: `C = A×A` (square set) and `C = A×Aᵀ` (rectangular set) —
  i.e. self-product, not general `A×B`.
- **Timing protocol** (from the repo's own `config/bench.json`, the config
  file the README says "should be used for performance benchmarking"):
  `warmup_iters: 10`, `bench_iters: 10`, statistic = arithmetic mean of the
  measured runs (`config/analysis.json`, used only for detailed
  stage-by-stage breakdown, drops to 1/1 — i.e. the detailed
  symbolic/numeric split is measured *without* repetition, only the
  headline number is averaged). A 30 s per-run timeout is enforced; timed-out
  or failed baselines are assigned the slowest *valid* runtime observed
  across all tools on that matrix (a "penalty fill-in", not a DNF/exclusion).
- **Timing scope / symbolic phase**: this is the paper's central
  contribution — it explicitly breaks out symbolic vs. numeric cost
  (Table 3 / Fig. 9): exact symbolic (spECK) ≈28% of total runtime on
  average; Ocean's estimation step ≈4% + a post-processing step ≈8%.
  Symbolic/numeric are reported *separately*, not folded silently into one
  number — the paper's whole argument depends on that breakdown being
  visible.
- **Precision**: fp64 values; uint32 indices.
- **Correctness**: `check_correctness` flag with
  `compare_relative_error: 1e-6` (target) and
  `compare_autofail_relative_error: 1e-4` (hard fail), verified against a
  ground-truth file (recommended: spECK's output, per README).
- **Metric**: GFLOP/s, geometric mean across the matrix set; FLOPs defined as
  "twice the number of intermediate products" (i.e. 2× the naive/symbolic
  multiply count, **not** `2×nnz(C)`).
- **Baselines**: spECK, opSparse, TileSpGEMM, HSMU-SpGEMM, cuSPARSE (limited,
  since cuSPARSE fails/OOMs on many large matrices).
- Source: `arxiv.org/html/2604.19004` (fulltext) +
  `github.com/CornellHPC/Ocean-SpGEMM` (`readme.md`, `config/bench.json`,
  `config/analysis.json`, `utils/square.csv`, `utils/rectangular.csv`).

---

## Trident (ICS 2026) — `conf/ics/BellavitaPPVG26`

Communication-avoiding, hierarchy-aware (intra-node NVLink vs inter-node
Slingshot) distributed SpGEMM via a hybrid 2D/1D "Trident" partitioning.

- **Workloads**: 10 named unstructured real-world matrices from SuiteSparse
  and the HipMCL repository (Table 2): HV15R (2.0M×2.0M, 283M nnz),
  mouse_gene (45K×45K, 29M nnz), archaea (1.6M, 204M nnz), eukarya (3.2M,
  359M nnz), isolates_subgraph4, isolates_subgraph5, cage15 (5.2M, 99M nnz),
  uniparc, reddit, dielFilterV3real. No explicit inclusion-threshold stated
  beyond "large, unstructured, sparse matrices from real-world sources" —
  effectively a curated list, not a filtered suite.
- **Operation**: primarily `C = A×A` (self-product, "matrix squaring");
  secondary experiment multiplies by a restriction operator, `C = A×R`, for
  the AMG-coarsening use case.
- **Timing protocol**: "the reported numbers represent the average of six
  trials, excluding the first trial" — i.e. warmup = 1, reps = 5 averaged
  (mean, not median). Timer mechanism not stated explicitly in the fetched
  text.
- **Timing scope**: symbolic-step compute/communication time is broken out
  separately in the paper's component figures, but the headline
  runtime/speedup numbers are full end-to-end SpGEMM (symbolic+numeric+comm).
- **Precision**: not explicitly stated in the fetched sections (distributed
  SpGEMM literature in this space is fp64 by default; unconfirmed here).
- **Baselines**: Trilinos (1D, sparsity-aware), CombBLAS Sparse SUMMA (2D,
  hierarchy-oblivious), an "Improved Sparse SUMMA" device-resident variant.
- **Metric**: wall-clock runtime (s) and speedup ratio; up to 2.38× vs. best
  baseline, geomean 1.54×.
- **Scale**: NERSC Perlmutter, 4 A100 GPUs/node, Slingshot-11; swept 4, 16,
  64, 256, 400 GPUs depending on matrix.
- Source: `arxiv.org/html/2603.21444` (fulltext) +
  `github.com/HicrestLaboratory/Trident` (`README.md`, `scripts/`) — no
  standalone per-run bench script was inspected line-by-line (repo ships
  Slurm/Kokkos build scripts and a `scripts/parse_results.py` rather than a
  single canonical `run.sh`).

---

## TileSpGEMM (PPoPP 2022) — `conf/ppopp/NiuLJS0022`

Tiled CSR SpGEMM for a single GPU; addresses load imbalance, locality, and
accumulator selection. No arXiv id — retrieved the author-hosted PDF
(`ssslab.cn/assets/papers/2022-niu-tilespgemm.pdf`) and extracted text with
`pypdf` since the ACM DOI page is paywalled.

- **Workloads**: 142 square matrices from SuiteSparse Matrix Collection,
  selection criterion = "no less than one billion floating point operations
  when computing **both** `C=A²` and `C=AAᵀ`" (i.e. an intersection filter on
  two different products, not a single-product threshold). Two named
  sub-suites for deep-dive figures:
  - **18 representative matrices** (Table 2, full names/sizes/flops/
    compression-rate extracted): the first 12 are the classic "Williams et
    al." SpMV benchmark set (pdb1HYS, consph, cant, pwtk, rma10,
    conf5_4-8x8-05, shipsec1, mac_econ_fwd500, mc2depi, cop20k_A, scircuit,
    webbase-1M) reused across nearly all subsequent sparse-kernel papers,
    plus 6 large/hard cases (af_shell10, pkustk12, SiO2, case39,
    TSOPF_FS_b300_c2, gupta3) with compression rates (intermediate
    products / `nnz(C)`) from 1.1× up to 136×.
  - a 16-matrix subset matching tSparse's own paper, used only for the
    fp16-tensor-core comparison (chosen because it "may best utilize the
    half precision tensor cores").
- **Operation**: `C = A²` (default) or `C = AAᵀ` (`-aat 1` CLI flag) — CLI
  signature is literally `./test -d <gpu> -aat <0|1> <path/to/mtx>`, i.e. the
  tool only ever multiplies a matrix by itself or its transpose; general
  `A×B` with two independent operands is not exercised.
- **Timing protocol**: **this is the flaw the spec must fix.** The shipped
  source (`src/common.h`) defines `#define REPEAT_NUM 1`, and
  `tilespgemm-cuda.h` times a loop `for (ri = 0; ri < REPEAT_NUM; ri++)` then
  takes `time_min` over that (single-element) array. I.e. the public
  artifact runs the timed kernel **exactly once per invocation**, with no
  warmup and no repetition, despite the code being structured as if a
  multi-rep min were intended. No `synchronize`-then-average methodology is
  actually exercised in the release; timer is host-side `gettimeofday`
  wrapped around `cudaDeviceSynchronize()` calls (not CUDA events).
- **Timing scope / symbolic & memory-allocation phase**: explicitly
  *separated* from the reported kernel number. Per the README's own
  numbered description of program output: line 6 = CSR→tile conversion time
  (Fig. 12, reported separately); lines 8–14 = "execution time of the three
  algorithm steps **and all memory allocation** on CPU and GPU" (Fig. 10,
  reported as its own stacked-bar breakdown — device malloc for `C` is
  visibly a large fraction of total time for some matrices, e.g. 'gupta3');
  line 17 = the headline "TileSpGEMM runtime (ms) and performance (GFLOP/s)"
  used in Fig. 6/7, which is the tiled-symbolic + numeric compute only, with
  format conversion and allocation excluded and reported separately.
- **Precision**: fp64 (all baselines) is primary; a secondary fp16
  tensor-core comparison against tSparse uses the 16-matrix subset.
- **Correctness**: "checking result after comparing our output with the one
  generated by cuSPARSE" — no explicit tolerance value found in the paper
  text or README (comparison is presumably close-to-exact since values are
  doubles, but the threshold is not published).
- **Metric**: GFLOP/s = `2×flops(intermediate products) / time`; the paper
  is explicit that "the compression rate is the ratio of the number of
  intermediate products (i.e. **half** the number of floating point
  operations) of `C=A²` to `nnz(C)`" — i.e. FLOPs is defined via the
  symbolic/naive intermediate-product count, matching Ocean's definition,
  not `2×nnz(C)`.
- **Baselines**: cuSPARSE v11.4, bhSPARSE, NSPARSE, spECK (all fp64), plus
  tSparse (fp16, tensor cores) on its own 16-matrix subset. On the 142-matrix
  set, cuSPARSE/bhSPARSE/NSPARSE frequently fail/OOM (cuSPARSE completes 0
  matrices on the smaller GPU); only spECK and TileSpGEMM finish all 142 —
  a data point the spec should account for (a "completes all matrices"
  requirement, or explicit DNF handling, is necessary for fairness).
- **Hardware**: RTX 3060 and RTX 3090 (Ampere), CUDA/driver 11.4/470.57.02.
- Source: author-hosted PDF fulltext (`ssslab.cn/assets/papers/
  2022-niu-tilespgemm.pdf`, extracted via `pypdf`) +
  `github.com/SuperScientificSoftwareLaboratory/TileSpGEMM`
  (`README.md`, `src/common.h`, `src/tilespgemm-cuda.h`).

---

## Communication-avoiding, memory-constrained distributed SpGEMM (IPDPS 2021) — `conf/ipps/HussainSBA21`

CombBLAS-based distributed SpGEMM for extreme-scale outputs (output larger
than aggregate memory); batches the computation and adds a distributed
symbolic step to size the batches ahead of time.

- **Workloads**: 7 named matrices (Table V) spanning three applications:
  Markov clustering / protein-similarity networks (Eukarya 3M×3M/360M nnz,
  Metaclust20m, Metaclust50 282M×282M/37B nnz, Isolates, Isolates-small, all
  from the IMG database or metagenome predictions), triangle
  counting/social-network (Friendster, from SuiteSparse), and k-mer overlap
  detection for genome assembly (Rice-kmers, from PacBio rice sequencing via
  the BELLA tool). Output nnz ranges 2B–1T. No SuiteSparse-uniform selection
  criterion — matrices are chosen to be representative of the paper's three
  target applications, explicitly favoring `nnz(C) > nnz(A)+nnz(B))` cases.
- **Operation**: primarily `C = A×A` ("we use the computation of A² to show
  various features of our algorithm"); `C = A×Aᵀ` also evaluated separately
  for Metaclust20m/Rice-kmers.
- **Timing protocol**: no explicit warmup/repetition/statistic described in
  the fetched sections — runtimes appear to be single-run bar-chart values.
  Timer mechanism unspecified.
- **Timing scope / symbolic phase**: explicitly separated and a first-class
  citizen of the contribution — "Symbolic3D will estimate more batches for
  load-imbalanced cases"; Fig. 8 compares symbolic-step computation vs.
  communication time in isolation, and component breakdowns (A-broadcast,
  B-broadcast, local-multiply, ...) are reported per stage rather than one
  aggregate number.
- **Precision**: not specified in the fetched text.
- **Baselines**: prior 3D Sparse SUMMA from the same group (CombBLAS), and
  1-layer vs. 16-layer (`l=1` vs `l=16`) variants of their own method.
- **Metric**: wall-clock runtime, strong-scaling speedup (e.g. 14× from
  16,384→262,144 cores), parallel efficiency, per-stage time breakdown.
- **Scale**: Cray XC40 (KNL and Haswell partitions), 64 nodes (4,096 cores)
  up to 4,096 nodes (262,144 cores).
- Source: `arxiv.org/abs/2010.08526` abstract + `ar5iv.labs.arxiv.org/html/
  2010.08526` fulltext. Artifact repo listed is the general-purpose
  `github.com/PASSIONLab/CombBLAS` library (this paper's specific
  batching/symbolic code is integrated into it rather than living in a
  separate paper-specific repo), so no paper-specific bench script was
  extracted.

---

## Cluster-wise SpGEMM via matrix reordering (SC 2025 / arXiv preprint) — `conf/sc/Islam00B25`

CPU shared-memory SpGEMM; combines row reordering with a row-clustered CSR
format so that rows with similar sparsity patterns share a cluster and reuse
loaded rows of `B`.

- **Workloads**: 110 SuiteSparse matrices with explicit multi-condition
  selection: "square matrices with more than 8M nonzeros" (so the working
  set exceeds L2 cache — a locality-motivated threshold, not a FLOP
  threshold) **and** "less than 10B nonzeros" (memory-fit upper bound); one
  matrix kept per publisher group to reduce redundancy, except SNAP and
  DIMACS10 groups (kept in full since they're the graph-heavy groups this
  paper targets); 26 matrices carried over from prior reordering-SpGEMM work
  and 32 from another prior study, rest newly added.
- **Operation**: general `C = A×B`, including a "tall-skinny" `B` variant
  (motivated by BFS-frontier matrices from betweenness-centrality
  computations, generated by instrumenting CombBLAS's own BFS/BC code to
  dump the first 10 forward frontier matrices) — this is the one paper in
  the surveyed set that is NOT primarily `A×A`.
- **Timing protocol**: "report the average of 10 runs" (mean, not
  median/min). Warmup and timer mechanism not explicitly documented in the
  fetched text or README.
- **Timing scope / reordering-preprocessing**: reordering cost is explicitly
  *not* folded into the reported per-call SpGEMM time. Instead the paper
  reports (Fig. 10) the number of repeated SpGEMM calls needed to amortize
  the one-time reordering cost — i.e. it treats reordering exactly like the
  spec's "e2e, amortized over k runs" pattern, and reports the break-even
  `k` per matrix as a first-class result rather than picking one fixed `k`
  and hiding the rest.
- **Precision**: not specified in the fetched text; assumed fp64 (standard
  for this line of work; hashtable accumulator implementation in the repo
  uses no explicit precision macro visible from the README).
- **Baselines**: row-wise (unreordered) SpGEMM, plus 10 named reordering
  algorithms (Random, RCM, AMD, ND, GP, HP, Gray-code, Rabbit, Degree,
  SlashBurn) crossed with 3 clustering variants (fixed-length,
  variable-length, hierarchical) — i.e. the "baselines" here are largely
  ablations of the paper's own method rather than external tools.
- **Metric**: speedup ratio relative to unreordered row-wise SpGEMM (e.g.
  1.39× average); no absolute GFLOP/s reported in the fetched summary.
- **Scale**: single Perlmutter CPU node, 64 threads (AMD EPYC 7763 Milan,
  2×64 cores, 512GB DDR4).
- Source: `arxiv.org/html/2507.21253` (fulltext) +
  `github.com/PASSIONLab/clusterwise-spgemm` (`README.md`, `scripts/*` — the
  repo ships ~40 numbered shell scripts, one per reordering×clustering
  combination, rather than a single parameterized bench script).

---

## AmgT (SC 2024) — `conf/sc/LuZWFLCY0C024`

Tensor-core / mixed-precision AMG solver; SpGEMM (setup phase, coarsening
`R·A·P`) and SpMV (solve phase) both get a new unified low-precision-capable
sparse format (mBSR) integrated directly into HYPRE.

- **Workloads**: not a standalone SpGEMM microbenchmark — SpGEMM is measured
  as it naturally occurs inside AMG setup, i.e. once per multigrid level,
  on whatever coarsened operators the solver produces for a given input
  linear system (matrix names from the paper's numerical-PDE/SuiteSparse
  test set were not enumerated in the sources checked here — see open
  questions).
- **Operation**: `C = Rᵢ·A·Pᵢ` per AMG level `i` — general `A×B` where both
  operands come from the solver's own coarsening hierarchy, not a
  benchmark-chosen pair.
- **Timing protocol**: confirmed directly from `AmgT_test/test_new.c` —
  `HYPRE_BoomerAMGSetup()` is called exactly **once** per test run (no
  warmup, no repetition loop around setup/solve); `gettimeofday` wraps
  `setup_time`/`solve_time`. Internally, `time_spgemm`, `spgemm_times`
  (a *count*, confirming SpGEMM is invoked multiple times — once per level —
  within one setup call) and `time_spgemm_preprocess` are accumulated and
  printed separately, alongside `cusparse_spgemm_time` for a like-for-like
  comparison against the cuSPARSE-backed HYPRE path in the same run.
- **Timing scope / symbolic & preprocessing**: `time_spgemm_preprocess`
  (mBSR format conversion) is tracked as a distinct counter from
  `time_spgemm` (the actual multiply), and both are distinct from
  `time_spgemm_all` (their sum) — i.e. the artifact itself keeps these
  numbers separable, it just aggregates them across all AMG levels of one
  solve rather than isolating a single SpGEMM call.
  Symbolic-vs-numeric split within `time_spgemm` itself was not visible in
  the code excerpt inspected.
  This means the "kernel-only" number this paper would supply is an AMG-
  workload-weighted mixture over many differently-shaped `R·A·P` products,
  not one clean measurement.
- **Precision**: fp64 baseline; mixed-precision variant uses tensor cores
  via the mBSR format (executable name `AmgT_Mixed` vs `AmgT_FP64` vs a
  `cuSPARSE`-calling HYPRE baseline — precision is a build-time/binary
  choice, not a runtime flag).
- **Correctness**: implicit — validated through AMG solve convergence
  (`HYPRE_PCGSetTol(solver, 1e-5)`, max 300 PCG iterations) rather than a
  direct elementwise SpGEMM-output comparison; this is an
  application-embedded correctness gate, not a kernel-level tolerance.
- **Metric**: setup_time / solve_time (ms), plus the internal SpGEMM-only
  and SpMV-only sub-times; headline claim is "up to 2.2× over HYPRE"
  end-to-end AMG setup+solve.
- **Baselines**: HYPRE's default cuSPARSE-backed SpGEMM/SpMV path (`./
  cuSPARSE` executable, same driver code).
- **Hardware**: NVIDIA GPUs, compute capability ≥8.0 (A100/H100-class),
  CUDA ≥12.2, OpenMPI ≥4.0.
- Source: `github.com/SuperScientificSoftwareLaboratory/AmgT`
  (`README.md`, `AmgT_test/test_new.c`,
  `AmgT_HYPRE/src/seq_mv/csr_spgemm_device.c`) — no arXiv id / OA link
  available for this paper (IEEE-only), so no fulltext narrative (workload
  list, exact matrix names/sizes) was retrievable; all facts above come from
  the artifact code itself.

---

## Divergences

1. **A×A dominates as the reference operation, but it is a convenience
   convention, not a semantic requirement.** Trident, Ocean (square set),
   TileSpGEMM (default), and Hussain's primary experiments all benchmark
   `C = A×A` (matrix squared) purely because it lets a benchmark harness take
   a single matrix file and needs no second operand. `A×Aᵀ` is the common
   second convention (Ocean rectangular set, TileSpGEMM `-aat 1`, Hussain
   Metaclust20m/Rice-kmers) — again single-operand-derived. Only two papers
   in the survey exercise genuine two-independent-operand `A×B`:
   clusterwise-spgemm (by explicit design — its whole contribution is about
   `B`-row reuse, which is trivial/degenerate when `A=B`) and AmgT (`R·A·P`,
   forced by the AMG algorithm, not chosen for benchmarking convenience).
   **Consequence for the spec**: a benchmark built only on `A×A` would be
   measuring a real but narrower thing than "SpGEMM" in general — it never
   stresses mismatched sparsity structure between operands. The spec should
   make `A×A` the primary/default variant (since 4/6 surveyed papers,
   including the most methodologically rigorous one, use it, and it is the
   only operation every open-source artifact here supports out of the box),
   but keep an explicit secondary `A×Aᵀ` mode and flag general `A×B` as an
   acknowledged gap (see open_questions).

2. **Symbolic/memory-allocation-phase timing is the single most contested
   axis, exactly as flagged in the instructions.** Every paper that reports
   a two-phase (symbolic+numeric) algorithm treats the *symbolic* step as
   an inherent, non-optional part of the per-call SpGEMM cost (Ocean,
   TileSpGEMM, Hussain) — but they radically disagree on what to do with
   *format-conversion* and *memory-allocation* costs:
   - TileSpGEMM reports CSR→tile conversion time and CPU/GPU memory
     allocation time as *separate, explicit output lines*, excluded from the
     headline GFLOP/s number.
   - Ocean's whole contribution is to shrink the symbolic step itself (not
     move it out of scope) — it still counts fully inside the timed call,
     just cheaper (an HLL estimate replaces exact counting).
   - clusterwise-spgemm treats its preprocessing (row reordering) as
     amortizable over repeated calls and reports the break-even iteration
     count explicitly rather than either hiding it or always including it.
   - AmgT can't cleanly separate "preprocessing" from "the algorithm" at
     all, because in AMG the format conversion (mBSR) and the SpGEMM calls
     are both intrinsic, repeated steps of one setup phase, and it reports
     an aggregate `time_spgemm_preprocess` and `time_spgemm` summed across
     all multigrid levels of one solve rather than per-call.
   **Consequence**: the spec must (a) keep symbolic-phase cost inside the
   "kernel" timing for the primary variant (it's not preprocessing, it's
   part of the algorithm every one of these methods needs to produce a
   correctly-sized output), (b) still *require* symbolic/numeric to be
   reported as a visible breakdown alongside the total (not just the total),
   since that breakdown is the actual point of comparison between e.g.
   Ocean's estimation approach and TileSpGEMM's exact two-pass approach, and
   (c) keep one-shot format-conversion/reordering as a separate,
   amortized-over-`k` variant, following clusterwise-spgemm's own
   methodology rather than any of the GPU papers' (which mostly report it
   once, in a figure, without formal amortization accounting).

3. **Timing rigor varies from "10 warmup + 10 measured, config-file-pinned"
   (Ocean) down to "exactly one run, no warmup" (TileSpGEMM's public
   artifact, AmgT).** TileSpGEMM's own source computes a `time_min` variable
   over a `REPEAT_NUM`-sized loop, strongly suggesting the paper's actual
   numbers (Fig. 6/7) came from a build with `REPEAT_NUM` > 1, but the
   *released, citable* artifact ships with `REPEAT_NUM=1` — i.e. the
   as-run/reproducible protocol is single-shot, not the min-of-N the code
   structure implies. Trident averages 6 trials excluding the first
   (arguably reasonable, but uses a mean, not a median, so is sensitive to
   tail latency e.g. from network jitter at scale). No surveyed paper uses
   device-side event timers (CUDA events) exclusively — TileSpGEMM and AmgT
   both wrap host-side `gettimeofday` around explicit
   `cudaDeviceSynchronize()`/HYPRE calls, which is defensible but
   1) includes CPU-side dispatch overhead. and 2) is easy to get wrong if the
   synchronize is missing on some code path (not verifiable from source
   review alone). **The spec fixes this** by mandating a fixed
   warmup/rep/statistic protocol and device-side event timing regardless of
   what any individual paper did, per the fairness principles.

4. **Correctness tolerance is either well-specified and strict (Ocean:
   1e-6 target / 1e-4 autofail, machine-checkable) or effectively
   undocumented (TileSpGEMM: "checking result", no number published;
   Trident/Hussain: not mentioned in the fetched text; AmgT: validated only
   indirectly via downstream PCG convergence, not elementwise).** The spec
   adopts Ocean's numeric convention as the default fp64 gate since it is
   the only one that is both concrete and machine-checkable from public
   source, and defines a separate, looser tolerance for the tensor-core/
   mixed-precision variant (no surveyed paper published one — see
   open_questions).

5. **Matrix-suite selection criteria differ in kind, not just threshold.**
   Ocean and TileSpGEMM both use a **FLOP threshold on the target product**
   (`flops(A×A) ≥ 1e8` / `≥ 1e9` respectively) — appropriate for SpGEMM
   specifically, since `flops(A×A)` can vary by orders of magnitude between
   matrices with similar `nnz(A)` depending on row-density structure (see
   TileSpGEMM's own compression-rate column: 1.13× to 136×). This is *not*
   the same kind of criterion as a typical SpMV/SpMM suite filter (nnz-only,
   since SpMV/SpMM flops scale linearly with nnz regardless of structure).
   clusterwise-spgemm instead uses a cache-capacity-motivated nnz threshold
   (>8M nnz to exceed L2) plus a memory-fit upper bound (<10B nnz), because
   its contribution (locality/reordering) is only interesting once the
   working set doesn't fit on-chip — a threshold tied to its own mechanism,
   not to SpGEMM cost in general. The spec follows the FLOP-threshold
   convention (Ocean/TileSpGEMM) for the primary variant since it is the
   SpGEMM-specific, mechanism-agnostic choice, and keeps nnz bounds only as
   a secondary sanity filter (to bound memory footprint / runtime).
