# Multigrid track — evaluation methodology survey

Track input: `data/track_inputs/multigrid.json`, 5 papers (all 5 surveyed — the
track's full population). All 5 fulltexts were located and read: 4 via arXiv
(BootCMatchGX: 2303.02352; MGARD: 2007.04457) or an author-hosted mirror
(AmgT: ssslab.cn PDF) or an institutional repository (MGopt-APP/ICS23:
White Rose eprints PDF); GPTuneCrowd's fulltext PDF via
liuyangzhuan.github.io. None had usable `pdftotext`/`pypdf` on this machine
(system Python 3.6, no poppler-utils), so a custom stream-decompress +
`Tj`/`TJ` text-extraction script was used, filtered to ASCII-dense lines —
noisy (broken ligatures: "T ensor"→"Tensor", "\002"→"fi" ligature marker,
"\050"/"\051"→"(" / ")") but every evaluation section was fully recovered.
For AmgT, in addition to the paper text, the actual HYPRE driver source
(`AmgT_test/test_new.c`) was read via `gh api` — this is the more reliable
source and is cited explicitly below where it disagrees with or sharpens the
paper's prose.

Operation surveyed: three of the five papers (AmgT, BootCMatchGX, MGopt-APP)
are genuine (algebraic or geometric) multigrid **solvers/preconditioners**
for `Ax=b`. The other two use multigrid **mathematics for a different end**:
MGARD applies multigrid-style hierarchical decomposition to lossy **data
compression/refactoring** (no `Ax=b` is ever solved), and GPTuneCrowd is a
**crowd-based autotuner** that treats Hypre/BoomerAMG as one of several
black-box applications to tune (it does not itself implement AMG). This
split matters throughout — see Divergences.

---

## 1. Lu, Zeng, Wang, Fu, Li, Cheng, Yang, Jin, Casas, Liu — AmgT (SC'24)

- key: `conf/sc/LuZWFLCY0C024`, artifact:
  github.com/SuperScientificSoftwareLaboratory/AmgT
- **workloads**: 16 hand-picked "representative" real-world matrices from the
  SuiteSparse Matrix Collection (Table II), spanning thermal, CFD,
  structural, and power-network domains, orders 14,822–952,203, nnz
  229,947–46,522,475, generating 2–7 AMG grid levels: `spmsrtls`,
  `thermal1`, `Pres_Poisson`, `Chevron2`, `venkat25`, `bcsstk39`, `mc2depi`,
  `stomach`, `parabolic_fem`, `cant`, `TSOPF_RS_b300_c3`, `af_shell4`,
  `msdoor`, `CoupCons3D`, `nd24k`, `ldoor`. RHS vector `b` is all-ones
  (`test_new.c`); no synthetic/generated Poisson problem is used anywhere —
  every input is a real algebraic matrix.
- **timing protocol** (code-verified, `AmgT_test/test_new.c` lines ~380-435):
  `HYPRE_BoomerAMGSetMaxIter(precond, 50)` and
  `HYPRE_BoomerAMGSetTol(precond, 1e-20)` — i.e. the solver is forced to run
  the **full fixed 50 iterations regardless of actual convergence**
  (tolerance is set so low it can never trigger early exit). Cycle type is
  explicitly `HYPRE_BoomerAMGSetCycleType(precond, 1)` = **V-cycle**, with
  `CycleNumSweeps(3,3)` (3 pre-/3 post-smoothing sweeps), relax type 18
  (`l1`-Jacobi-family), coarsening PMIS (type 8), interpolation "extended+i"
  (type 6), max 7 levels, strong threshold 0.25, truncation factor 0.1.
  **Setup and solve are timed as two separate `gettimeofday` brackets**
  (`t_start/t_stop` around `HYPRE_BoomerAMGSetup`, `t1/t2` around
  `HYPRE_BoomerAMGSolve`) — the code enforces the setup/solve split the
  paper text also describes (Algorithms 1 and 2). No warmup pass or repeated
  trials are visible in the driver — this reads as a **single measurement
  per matrix**, not a warmup+N-repetition loop; the paper text does not
  mention a statistic (mean/median) either.
- **timing scope**: setup phase (dominated by SpGEMM: interpolation +
  Galerkin triple product) vs. solve phase (dominated by SpMV: one
  V-cycle = 31 SpMV calls — 30 across the pre-/post-smoothing+restriction
  steps over up to 6 levels, plus 1 residual check — so 50 iterations means
  up to 1,551 SpMV calls) are kept and reported as **fully separate**
  numbers, matching this track's required axis exactly. Format conversion
  (CSR↔mBSR, invoked `2×#levels−1` times) is measured and reported
  separately (paper Fig. 10): "generally...longer than 5% of overall
  execution time" — disclosed, not silently folded in.
- **precision & correctness**: fp64 baseline vs. a **mixed-precision
  cascade** — fp64 at the finest grid level, fp32 at the second level, fp16
  (unavailable on their instruction set, so fp64 was substituted at the
  finest level regardless) at all remaining coarser levels, following a
  configuration from Tsai et al. No residual/convergence tolerance is ever
  actually enforced (see above: tol=1e-20 means "never converge early");
  the paper reports `HYPRE_BoomerAMGGetFinalRelativeResidualNorm` for
  logging but does not gate correctness on it in the benchmark loop itself.
- **metric**: execution time (ms) and derived speedup vs. baseline; no
  GFLOP/s formula for the full solver (a GFLOP/s formula IS used for the
  standalone SpGEMM/SpMV kernel comparisons in the same paper, surveyed
  separately in the spmv/spgemm tracks).
- **baselines**: HYPRE v2.31.0-GPU calling cuSPARSE v12.2 (NVIDIA
  A100/H100) or rocSPARSE v6.1.2 (AMD MI210); also an 8-GPU (A100)
  multi-node comparison against the same HYPRE baseline.
- source: `AmgT_test/test_new.c` (code-verified ground truth for
  iteration/tolerance/cycle/timer), `README.md`; author-hosted paper PDF
  (ssslab.cn — no arXiv preprint exists) text-extracted after direct
  WebFetch failed on the binary PDF.

## 2. Bernaschi, Celestini, Vella, D'Ambra — BootCMatchGX (TPDS'23)

- key: `journals/tpds/BernaschiCVD23`, arXiv:2303.02352 (fulltext), artifact:
  github.com/bootcmatch/BootCMatchGX
- **workloads**: **synthetic** 3D Poisson problem, `-∇·(K∇u)=f` on `[0,1]³`
  with homogeneous Dirichlet BC, `K=1`, discretized by the classic 7-point
  finite-difference stencil — this is the "AMG2023-style Poisson generator"
  this track's key axes call out directly. Strong scaling: fixed
  `300³ ≈ 27×10⁶` dofs, 1→16 GPU-nodes. Weak scaling: fixed `130³ ≈ 2.2×10⁶`
  dofs/node, 1→100 nodes (up to `≈2.2×10⁸` total dofs). No SuiteSparse or
  other real-matrix input is used anywhere in this paper.
- **timing protocol** (paper text, §5): AMG preconditioner setup fixes max
  coarsest-matrix size to `40·nd` and max 40 levels; pairwise aggregation
  merged into aggregates of size ≤8. **Convergence criterion is explicit
  and real**: PCG iterations stop when the relative residual in the ℓ₂ norm
  is `< 1e-6`, or after a max of 1000 iterations — unlike AmgT's fixed
  iteration count, this is genuine convergence-gated timing. Applied
  preconditioner = 1 V-cycle with 4 pre-/post-smoothing ℓ₁-Jacobi sweeps at
  intermediate levels, 20 ℓ₁-Jacobi sweeps at the coarsest level (a
  distributed iterative bottom solve, not a direct solver, chosen
  specifically to avoid GPU-unfriendly sparse-triangular-solve
  dependencies). No explicit warmup/repetition-count or per-run
  statistic (mean/median) language was found — single-run scalability
  numbers appear to be reported directly per data point.
- **timing scope**: setup time (`tsetup`) and solve time (`tsolve`,
  including per-iteration `titer`) are reported as **fully separate**
  quantities (Figs. 3/4/6/7 plot them independently, plus a "total time" =
  their sum) — again matching this track's required setup/solve split.
  Operator complexity (`OPC = Σnnz(A_k)/nnz(A_0)`, k=0..nl−1) is reported
  alongside timing as a solver-quality metric, since a cheaper-but-worse
  hierarchy is not a fair win on time alone.
- **precision & correctness**: fp64 throughout (s.p.d. matrices via CSR,
  row-block MPI distribution). Correctness gate = the ℓ₂ relative-residual
  tolerance above; number of PCG iterations to reach it is reported
  alongside wall time specifically so a solver cannot "win" by running
  fewer, cheaper-but-lower-quality iterations without disclosing it.
- **metric**: wall-clock time (seconds) for setup/solve/total, plus
  iteration count and operator complexity; strong/weak-scaling speedup and
  parallel efficiency derived from these.
- **baselines**: Nvidia AmgX (hybrid MPI-CUDA build), configured via a
  modified `PCG_AGGREGATION_JACOBI.json` to use decoupled aggregation with
  size-8 aggregates — chosen specifically to make AmgX's operator
  complexity comparable to BCMG's rather than comparing against an
  arbitrary AmgX default. Same PCG loop / V-cycle / smoother counts applied
  to both solvers for the comparison. Platform: Piz Daint (CSCS), Cray
  XC40/XC50, Nvidia Tesla P100 per hybrid node, 1 MPI rank/node/GPU.
- source: arXiv 2303.02352 fulltext (custom stream-extraction); repo
  `README.md` (build/CLI options, regression-test targets).

## 3. Yang, Li, Yuan, Dong, Huang, Wang — "MGopt-APP" (ICS'23)

- key: `conf/ics/YangLYDHW23`, artifact: github.com/YXJ-123/MGopt-APP
- **workloads**: two problem classes. (a) **HPCG v3.1 benchmark**'s own
  synthetic problem generator: a 4-level geometric V-cycle on a `192×192×192`
  local grid per MPI rank (grid scale shrinks by 8× per level) — this is a
  structured, geometrically-generated multigrid problem, not an
  algebraically-coarsened one (no AMG "setup phase" in the SpGEMM sense).
  (b) two real applications ported to use the paper's optimized SYMGS:
  YHAMG (AMG library) solving a Poisson problem discretized by 7-point and
  27-point stencils, and CitcomCU (finite-element earth-simulation solver,
  F-cycle) on a `256×256×128`-per-MPI-process local grid.
- **timing protocol** (paper text, §5.4 "Evaluation Methodology" — the most
  explicit, rigorous protocol of all 5 surveyed papers): *"We run each test
  case 10 times on unloaded machines and report the geometric mean of the
  runtime. The variance across different runs is small, less than 5%."*
  8 MPI processes × 7 OpenMP threads/rank on ARM platforms, 4 MPI × 7
  OpenMP (28 cores) on the x86 Xeon platform.
- **timing scope**: end-to-end benchmark performance (GFLOPS) for the whole
  MG/HPCG run — because this is *geometric* MG with a synthetic grid
  generator (not algebraic coarsening from a matrix), there is no separate
  "setup phase" comparable to AmgT/BootCMatchGX's SpGEMM-heavy hierarchy
  construction; the grid hierarchy is defined directly by geometry, and all
  reported time is what an AMG paper would call "solve phase." A separate
  ablation (Table 2) breaks down speedup contributions from each
  optimization (SYMGS-opt / Fusion / BMC / BMC+Fusion) individually.
- **cycle type**: V-cycle (Algorithm 1, the paper's primary vehicle; the
  authors state their technique "can be equally applied to other mainstream
  MG implementations that use SYMGS, including F-cycle and W-cycle" — and
  CitcomCU is explicitly run with F-cycle in the real-application case
  study).
- **precision & correctness**: not explicitly stated as fp32/fp64 (HPCG's
  reference implementation is fp64 by convention; the paper does not
  discuss precision as a variable at all — no mixed-precision claim).
  Correctness is implicit in reproducing HPCG's reference numerical
  results (unchanged solver semantics; only computation/memory-access
  pattern is altered, per §3.1's explicit "preserves SYMGS semantics
  without changing the computation outcome").
- **metric**: GFLOPS (`performance`), plus parallel efficiency
  `T₁/(N·Tₙ)` for the 256-node scaling experiment; speedup vs. baselines.
- **baselines**: `HPCG_for_ARM` (ARM's own vendor-tuned HPCG using ARM
  Performance Library + level-scheduling+BMC), `HPCG_for_MKL` (Intel MKL's
  closed-source HPCG), synchronous BMC (the paper's own parallelization
  baseline before adding the asynchronous scheme), and the un-optimized
  reference HPCG 3.1 release. Platforms: Phytium 2000+, Kunpeng 920,
  ThunderX2 (3× ARMv8), Intel Xeon Gold-5117 (x86); a 256-node Phytium
  2000+ cluster (16,384 cores) for multi-node scaling.
- source: White Rose eprints PDF (`eprints.whiterose.ac.uk/198955/`, CC-BY
  4.0, custom stream-extraction — the extraction was unusually clean for
  this paper, recovering full paragraph text through §7); repo `README.md`
  (confirms HPCG 3.1 parameter compatibility, `C=`/`BX=`/`BY=`/`BZ=` CLI
  flags for manual color/block-count override, but ships only prebuilt
  `.exe` binaries, no source — "source available on request").

## 4. Chen, Wan, Liang, Whitney, Liu, Pugmire, Thompson, Choi, Wolf, Munson, Foster, Klasky — MGARD data refactoring (IPDPS'21)

- key: `conf/ipps/ChenW0W0PTCWMFK21`, arXiv:2007.04457 (fulltext), artifact:
  github.com/CODARcode/MGARD
- **scope note**: this paper does **not** solve `Ax=b`. It applies
  Ainsworth et al.'s multigrid-based hierarchical decomposition
  (fine→coarse projection + orthogonal-projection correction, requiring a
  small per-node linear solve via the Thomas algorithm — "iterative
  processing kernel," IPK) to **lossy scientific-data compression /
  progressive-fidelity data refactoring**, not to iterative linear-system
  solving. It is included in this track (the phase-1 classifier tagged it
  `multigrid`) because the *mathematical machinery* — fine/coarse grid
  hierarchy, node-wise coefficient/correction computation, per-level
  compute kernels — is the multigrid algorithm; the *application* is
  unrelated to AMG solvers. See Divergences.
- **workloads**: **synthetic** grid data from a Gray–Scott
  reaction–diffusion simulation, single- or double-precision floating
  point, cubic (equal-size-per-dimension) domains for simplicity. The
  paper explicitly notes decomposition/recomposition has **deterministic**
  computation cost independent of the actual data values, so timing is
  data-value-independent — only grid dimensions/size matter.
  Weak-scaling test: up to 1024 nodes on Summit, 6 GPUs or 42 CPU cores per
  node, each GPU/CPU-core handling 1 GB in double precision.
- **timing protocol** (paper text, §IV-A "Evaluation methodology"): four
  design variants compared — `SOTA-GPU` (prior MGARD GPU baseline,
  hand-tuned thread-block-size + CUDA-stream-count parameters, "best
  performance achieved by hand tuning"), `SOTA-CPU` (prior MGARD CPU,
  MPI-parallelized, "for a fair comparison" against the multi-core
  baseline), `OPT` (this paper's new GPK/LPK/IPK kernels, no autotuning),
  `OPT+AT` (OPT plus the paper's own heuristic autotuner). No explicit
  warmup/repetition-count/statistic (mean/median) language was found for
  the microbenchmark numbers; the weak-scaling numbers are per-node-count
  single measurements.
- **timing scope**: per-kernel speedups (coefficient computation / mass-
  transfer matrix multiply / correction solve) are measured individually,
  then whole-decomposition throughput is measured against theoretical peak
  memory bandwidth (Summit: 49.8 GB/s per GPU; Turing desktop RTX 2080 Ti:
  32.0 GB/s) — the baseline SOTA-GPU achieves <10.4% of peak; this paper's
  design achieves up to 4 computing-nodes-for-1TB/s and 250 TB/s aggregate
  at 1024 nodes (83% of theoretical peak). There is no "setup vs. solve"
  split at all — this axis does not apply to a one-shot data-transform
  kernel the way it applies to an iterative AMG solver.
- **cycle type / convergence criterion**: N/A — decomposition is a
  single-pass, non-iterative fine-to-coarse sweep (not a V/W/F-cycle, no
  residual-based stopping criterion); "convergence" is replaced by a
  computable, deterministic error bound derived from the number of
  discarded coefficient classes.
- **precision & correctness**: single or double precision floating point
  (chosen per dataset, not compared as a mixed-precision axis within one
  run). Correctness = deterministic reconstruction error bound (not a
  Krylov-residual tolerance).
- **metric**: throughput, GB/s (single node) and aggregated TB/s
  (multi-node), against a hardware peak-bandwidth denominator; also
  end-to-end speedup (145× vs. CPU baseline, 14× vs. prior GPU baseline).
- **baselines**: the prior MGARD CPU and GPU implementations themselves
  (SOTA-CPU/SOTA-GPU, both from the same MGARD codebase, not a third-party
  library) — this paper's baseline is "our own prior published design,"
  not an external competitor like HYPRE/AmgX.
- source: arXiv 2007.04457 fulltext (custom stream-extraction, recovered
  full §I-VII including the "Evaluation methodology" subsection heading
  verbatim); repo `README.md` (module map: MGARD-CPU/CUDA/X/DR/ROI/QOI, no
  runnable benchmark script surfaced in the root listing).

## 5. Cho, Demmel, King, Li, Liu, Luo — GPTuneCrowd (IPDPS'23)

- key: `conf/ipps/ChoDKLLL23`, artifact: github.com/gptune/GPTune
- **scope note**: this paper is an **autotuning framework**, not a
  multigrid solver implementation. It is in this track only because one of
  its five case studies tunes Hypre's BoomerAMG (an AMG preconditioner) as
  a black-box application. GPTuneCrowd never implements AMG kernels itself
  — it treats "run Hypre with parameter set X, get back a runtime" as an
  opaque function evaluation. See Divergences.
- **workloads (Hypre case study specifically)**: Hypre's GMRES with the
  BoomerAMG preconditioner, solving the **Poisson equation on a synthetic
  structured 3D grid** `[nx,ny,nz]` — for the sensitivity analysis,
  `nx=ny=nz=100`; for the reduced-search-space tuning demo,
  `nx=ny=nz=120`. 12 BoomerAMG tuning parameters exposed: `Px, Py, Nproc,
  strong_threshold, trunc_factor, P_max_elmts, coarsen_type, relax_type,
  smooth_type, smooth_num_levels, interp_type, agg_num_levels`.
- **timing protocol**: 1,000 pre-collected performance samples on a single
  Cori Haswell node used to compute Sobol global-sensitivity indices (S1
  main-effect, ST total-effect) per parameter via the SALib
  implementation. The actual tuning demo uses a 20-function-evaluation
  budget; multiple independent tuning runs (three, in the closely-related
  PDGEQRF case study; five, in the NIMROD case study — exact count for the
  Hypre demo specifically not stated) are run with different random seeds
  and the mean ± stdev of best-so-far result is plotted.
- **timing scope / metric**: wall-clock runtime (seconds) of one Hypre
  solve, as reported by the tuned application itself back to the tuner —
  GPTuneCrowd does not decompose this into setup/solve/kernel phases at
  all; runtime is treated as a single opaque objective value to minimize.
- **cycle type / convergence criterion / precision**: not discussed —
  these are internal to Hypre's own defaults and are not exposed as tuning
  parameters or evaluation axes in this paper (the tuning parameters
  listed above are AMG *configuration* choices, e.g. `coarsen_type`,
  `relax_type`, not cycle-type or tolerance selection).
- **sensitivity result** (Table V, `nx=ny=nz=100`): `smooth_type` and
  `agg_num_levels` have the highest Sobol S1/ST scores (S1>0.1, ST>0.5),
  followed by `smooth_num_levels` (S1=0.05, ST=0.35); `Px, strong_threshold,
  trunc_factor, P_max_elmts, coarsen_type, relax_type, interp_type` are all
  near-zero influence. Tuning only the 3 most-sensitive parameters
  (20-evaluation budget) achieves **1.35× better tuned runtime (25.8%
  improvement)** at the 10th evaluation vs. tuning the full 12-parameter
  space.
- **baselines**: `NoTLA` (non-transfer-learning Bayesian optimization from
  the base GPTune package) as the tuning-quality baseline; the "reduced
  search space" vs. "original search space" comparison is the paper's own
  ablation, not a comparison against a different AMG solver.
- source: paper PDF (liuyangzhuan.github.io/docs/GPTuneCrowd_2023.pdf,
  custom stream-extraction, recovered full text through the reference
  list); repo `README.md` (GPTune package structure: `GPTune/`,
  `example_scripts/`, `examples/`, no Hypre-specific driver script
  surfaced in the root listing — the Hypre case study's driver is not in
  the root-level directories checked).

---

## Divergences

- **"Multigrid" spans three unrelated kernel shapes in this track, not
  one.** Three papers (AmgT, BootCMatchGX, MGopt-APP) are genuine AMG/MG
  *solvers* for `Ax=b` with the setup/solve split this track's key axes
  ask about. MGARD is not a solver at all — it repurposes multigrid
  hierarchical-decomposition math for one-shot, non-iterative lossy data
  compression (no setup/solve split applies; "convergence" is replaced by
  a deterministic error bound). GPTuneCrowd is not a multigrid
  implementation at all — it is a generic autotuner that happens to include
  Hypre/BoomerAMG as one black-box tuning target among several (PDGEQRF,
  NIMROD, SuperLU_DIST). A benchmark spec built only from the union of "all
  5 papers' practices" would conflate three incompatible things; the spec
  restricts its core solver-kernel variants to the 3 genuine-solver papers
  and treats MGARD/GPTuneCrowd as out-of-scope-but-documented (see
  `spec.yaml`'s `open_questions`).
- **Setup-phase existence itself depends on whether coarsening is
  algebraic or geometric.** AmgT and BootCMatchGX both construct their grid
  hierarchy *algebraically* from the input matrix (SpGEMM-heavy: matching/
  aggregation + interpolation + Galerkin triple product) — a genuine,
  separately-timeable "setup phase." MGopt-APP's primary vehicle, HPCG,
  defines its grid hierarchy *geometrically* (factor-of-8 refinement of a
  structured grid, no matrix-coarsening SpGEMM at all) — there is no
  analogous "setup phase" to time separately; all reported time is
  solve-phase smoothing+SpMV. A spec that demanded every variant report a
  nonzero, comparable "setup time" would misrepresent MGopt-APP's
  geometric-MG design; the spec keeps a GPU/algebraic-setup variant and a
  separate CPU/geometric-MG variant rather than forcing one setup/solve
  split onto both problem classes.
- **Problem generator: real algebraic matrices vs. synthetic 3D Poisson vs.
  application-embedded grids — no shared choice.** AmgT uses exclusively
  16 hand-picked real SuiteSparse matrices (irregular sparsity, no
  synthetic generator at all). BootCMatchGX uses exclusively a synthetic
  3D 7-point-stencil Poisson generator with fixed dof counts for strong/
  weak scaling (matching this track's named "AMG2023-style Poisson" axis).
  MGopt-APP uses HPCG's own synthetic 27-point-stencil-flavored generator
  as primary, plus two real-application-embedded problems (YHAMG's 7-/27-
  point-stencil Poisson, CitcomCU's finite-element earth-simulation grid)
  as secondary validation. No paper uses both a real-matrix suite and a
  synthetic-Poisson suite side by side. The spec resolves this by keeping
  BOTH input classes as first-class, explicitly-labeled input suites within
  the GPU variants (not silently merged), since results on one are not
  comparable to results on the other (irregular real matrices stress format
  conversion and irregular-hierarchy-shape overhead in ways a regular
  synthetic stencil never does).
- **Convergence criterion: real residual tolerance vs. deliberately-disabled
  early exit.** BootCMatchGX gates PCG iterations on a real, standard
  relative-residual tolerance (`< 1e-6` in the ℓ₂ norm, capped at 1000
  iterations) — genuine convergence-based timing. AmgT, by contrast, sets
  `HYPRE_BoomerAMGSetTol(precond, 1e-20)` — a tolerance so tight it can
  never trigger — specifically so every timed run always executes exactly
  50 iterations regardless of whether the solve has actually converged.
  This is a legitimate choice for *reproducible, comparable per-iteration
  timing* (the same fixed amount of work every run) but it means AmgT's
  reported "solve time" says nothing about how many REAL iterations a
  practitioner would need to reach a working tolerance, and never verifies
  the final answer is numerically acceptable — a flaw the spec fixes by
  requiring BOTH a fixed-iteration-count throughput number (comparable
  across implementations doing the same fixed work) AND a real-tolerance,
  convergence-gated number with the resulting iteration count reported
  (so a solver cannot look fast by simply doing less real work per
  "iteration" it's credited for).
- **Cycle type: V-cycle is the default in every solver paper, but W/F-cycle
  support is asserted, not benchmarked.** All 3 solver papers *use* V-cycle
  as their primary/only benchmarked configuration. MGopt-APP additionally
  states its SYMGS optimization "can be equally applied to...F-cycle and
  W-cycle" and does run CitcomCU with F-cycle in one real-application case
  study, but never benchmarks the SAME technique across V/W/F cycles on the
  SAME problem to isolate cycle-type's effect. No paper reports a W-cycle
  number at all. The spec fixes V-cycle as the required primary
  configuration (matching universal practice) and flags F/W-cycle
  comparison as unresolved (open_questions), since no surveyed paper
  provides the data to specify it responsibly.
- **Precision: fp64-only, real mixed-precision cascade, and "not discussed"
  all appear.** BootCMatchGX and MGopt-APP are fp64-only throughout (no
  precision axis explored). AmgT's mixed-precision variant is a
  **level-dependent cascade** (fp64 finest / fp32 second level / fp16
  remaining coarse levels) inside a single solve — not a single
  global-precision switch — which doesn't fit a simple "fp32 vs fp64"
  binary; the spec keeps AmgT's exact cascade definition as its own
  mixed-precision variant rather than approximating it as generic fp32.
  MGARD supports single OR double precision per dataset but never compares
  them as an axis within one experiment.
- **Timing rigor is bimodal: one paper is exemplary, the rest report almost
  nothing about repetition/statistic.** MGopt-APP states an explicit,
  strong protocol: 10 repeated runs per case on unloaded machines, geometric
  mean reported, cross-run variance <5% (stated in prose). AmgT's own
  driver code shows a **single** `gettimeofday` bracket per phase with no
  visible repetition loop — likely a single run per data point, though the
  paper text doesn't say so explicitly either way. BootCMatchGX and MGARD's
  papers give no warmup/repetition/statistic language at all for their
  headline numbers (scalability plots appear to be single measurements per
  configuration). GPTuneCrowd repeats full tuning RUNS (not individual
  Hypre solves) 3-5× with different random seeds and reports mean±stdev —
  a different kind of repetition (over stochastic tuning trajectories, not
  over a fixed deterministic kernel). The spec fixes MGopt-APP's protocol
  (repeated runs + a stated statistic + a stated variance bound) as the
  baseline requirement for every GPU/CPU solver variant, since it is the
  only one of the 5 papers to actually specify one.
- **Baselines are each paper's own predecessor/competitor library, never a
  shared reference solver.** AmgT vs. HYPRE-GPU/cuSPARSE/rocSPARSE.
  BootCMatchGX vs. Nvidia AmgX (config-matched for comparable operator
  complexity — a genuinely fair-comparison effort). MGopt-APP vs.
  vendor-tuned HPCG (ARM Performance Library / Intel MKL) plus its own
  synchronous-BMC ablation. MGARD vs. its own prior CPU/GPU
  implementations (not a third-party competitor at all). No two solver
  papers share a baseline, so no single "reference implementation" can
  anchor a correctness or speedup number across this track the way, e.g.,
  cuSPARSE anchors the SpMV/SpGEMM tracks; the spec instead fixes
  quantitative correctness gates (residual tolerance, reference-precision
  comparison) independent of any one paper's baseline choice.
