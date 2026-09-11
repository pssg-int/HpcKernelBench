# CG/Krylov track — evaluation methodology survey

Track input: `data/track_inputs/cg-krylov.json`, 7 papers (all 7 surveyed — below
the 5-paper minimum threshold does not apply since the track only has 7).
6 of 7 surveyed via `gh api` reads of the actual artifact source (timing loops,
build/run scripts, default CLI parameters); the 7th (CLAIRE) has an arXiv id
but WebFetch of both `arxiv.org/pdf` and `ar5iv.labs.arxiv.org/html` truncated
before reaching its numerical-experiments section, so it is documented from
the abstract plus the parts of the PDF that did render (problem sizes,
preconditioner tolerances) — flagged as weaker evidence below.

Operation surveyed throughout: solve `A x = b` (A sparse, symmetric
positive-definite) via (preconditioned) Conjugate Gradient. One PCG iteration
= 1 SpMV (`q = A*p`) + optional preconditioner apply (`z = M^-1*r`) + 2 dot
products (reductions) + 3 AXPY-family vector updates.

---

## 1. Ma, Ahmad, Cheshmi, Sukumaran-Rajam, Hassanpour — SPCG (SC'25)

- key: `conf/sc/MaACSH25`, artifact: github.com/SwiftWare-Lab/SPCG
- **workloads**: 114 named real SuiteSparse matrices, code-verified via
  `script_src/python_scripts/prep/matrix_download.py` (explicit `ssgetpy`
  name list — bcsstk\*/bcsstm\*/thermal/structural SPD problems, sizes from
  ~1K rows (`bcsstk08`) up to ~230K rows (`msc23052`), all real
  symmetric-positive-definite; downloaded once and cached under `matrices/`).
  Sparsified variants of each matrix are additionally generated offline via a
  MATLAB script (`matrix_sparsification.m`) at ratios like 0.01/0.05/0.1 —
  this sparsification is the paper's own technique, not part of the input
  suite itself.
- **timing protocol** (code-verified, `gpu_src/ilu0_gpu/nonsp/main.cpp`):
  CUDA events. `precondStart`/`precondStop` bracket ILU(0) analysis+
  factorization+triangular-solve-analysis (measured and printed as
  "Preconditioning Time" — **reported separately**, not folded into the PCG
  loop number). `start` (recorded before precond) / `stop` (after the loop)
  bracket the whole run; the artifact derives `PCG Time = Overall Time -
  Preconditioning Time` and writes both to
  `results_summary_float.csv`. Additionally, **every single PCG iteration**
  gets its own `iterStart`/`iterStop` CUDA-event pair, written per-iteration
  to a per-matrix CSV (`float_residuals_<matrix>.csv`) with `Iteration,
  Residual, Time` — real per-iteration granularity, not a single batched
  bracket.
- **timing scope**: to-convergence — `while (r1 > tol*tol && k < max_iter)`,
  `tol = 1e-12f`, `max_iter = 1000` (hardcoded in `main.cpp`). No H2D/D2H
  timing shown around the loop itself (matrix already resident on device by
  then); the one-time matrix load/H2D copy happens before `start` is
  recorded and is excluded.
- **precision & correctness**: **fp32** (`float` throughout — `d_val`,
  `d_x`, `tol`, cuBLAS `S*` calls, cuSPARSE `CUDA_R_32F`). Correctness:
  final residual `sqrt(r1)` reported; no independent reference-solution
  check found in this file (relies on residual convergence itself as the
  correctness signal).
- **metric**: total time (ms), preconditioning time (ms), PCG-only time (ms),
  iteration count, final residual — all written to
  `results_summary_float.csv`. No GFLOP/s computed in the artifact itself.
- **baselines**: paper abstract states geomean speedups of the sparsified
  variant (SPCG) vs. non-sparsified ILU(0)/ILU(K)-PCG: 1.23x/1.65x
  (iterative-phase only) and 1.68x/3.73x (end-to-end) on an A100.
- source: repo `README.md`, `gpu_src/ilu0_gpu/nonsp/main.cpp` (full ILU0-PCG
  driver read), `script_src/python_scripts/prep/matrix_download.py`.

## 2. Yang, Zhao, Niu, Jia, Shao, Liu, Tan, Jin — Mille-feuille (SC'24)

- key: `conf/sc/YangZNJS0T024`, artifact:
  github.com/SuperScientificSoftwareLaboratory/Mille-feuille
- **workloads**: Matrix Market files from the SuiteSparse Matrix Collection
  (per README); the shipped smoke-test example is a single tiny matrix
  (`test/add20.mtx`, `run_cg.sh` just does `./main-cg test/add20.mtx`) — no
  evaluation-scale matrix list is present in the public artifact itself
  (weaker evidence than SPCG for the actual paper-reported suite).
- **timing protocol** (code-verified, `src/main-cg.cu`): `gettimeofday`
  brackets (`t1`/`t2`) around the CG loop only; format construction
  (CSR→tile, `time_format`) is timed **separately** with its own
  `gettimeofday` pair and reported as a distinct column. **Important
  artifact bug/mismatch**: `main()` calls
  `cg_solve_reduce(..., maxiter=10, threshold=1e-5, ...)`, but inside
  `cg_solve_reduce` the loop is `while (iterations < 10000)` — a **hardcoded
  10000-iteration cap that ignores the passed-in `maxiter=10` entirely**,
  and the residual-based stopping line is present in the source but
  **commented out**: `//while (iterations < 1000 && sqrt(snew) > epsilon)`.
  The loop is fixed-iteration in practice (always runs 10000 iterations,
  never checks whether it has actually converged mid-loop), even though the
  function signature and README suggest a convergence-driven solver.
  Additionally, the reported time is `time_cg/100` — a **hardcoded divisor
  of literal 100**, not the actual iteration count (10000) or a variable —
  so the printed "per-iteration" time is off by 100x from the true
  per-iteration cost of a 10000-iteration run.
- **timing scope**: kernel-only for the CG loop (H2D done before `t1`);
  tile-format construction timed once, separately, and reported (good
  practice) but the loop's own iteration accounting is broken as above.
- **precision & correctness**: tile-grained mixed precision — `MAT_VAL_TYPE`
  (full, likely fp64) plus a companion `MAT_VAL_LOW_TYPE` "Val_Low" array
  used for on-chip dynamic tile-wise precision downcasting (the paper's core
  technique). Correctness: post-loop, the artifact recomputes `b_new = A*x`
  on the CPU and reports `l2_norm = ||b_new - b|| / ||b||` — an independent
  reference check, but AFTER the (mis-terminated) loop, not gating the
  reported time.
- **metric**: `time_cg/100` (ms, per the bug above), nnzR, `l2_norm`,
  final `sqrt(snew)` — written to `data/cg_performance.csv`. No GFLOP/s.
- **baselines**: paper abstract claims "3-5x over cuSPARSE" on A100/MI210.
- source: repo `README.md`, `src/main-cg.cu` (full file read, both the
  `cg_solve_reduce` iteration loop and `main()` driver), `test/run_cg.sh`.

## 3. Zhang, Wahib, Chen, Meng, Wang, Endo, Matsuoka — PERKS (ICS'23)

- key: `conf/ics/ZhangWCMWEM23`, artifact: github.com/neozhang307/PERKS
- **workloads**: user-supplied Matrix Market file (`--mtx=`) OR a synthetic
  `virtualdata` tridiagonal matrix generated in-code (`genTridiag`,
  N=1,048,576) when no `--mtx` is given — no fixed evaluation suite is
  pinned in the public artifact's CG driver itself (the paper's actual
  matrix list is not visible from this file).
- **timing protocol** (code-verified, `conjugateGradient/cg_driver.cu`):
  a single CUDA-event pair (`start`/`stop`) brackets the **entire run of
  `max_iter` iterations** (not per-iteration events); the artifact then
  divides by `total_iter` to report `time/total_iter` ms/iteration and a
  derived GB/s. An optional `--warmup` pass **calibrates `max_iter`
  adaptively**: it times one short run and sets
  `max_iter = max(1, (int)(2*350/time))` so the real timed run targets
  ~350ms total wall time — the same "adapt rep-count to hit a wall-clock
  budget" pattern seen in AlphaSparse (SpMV track), here applied to CG
  iteration count rather than SpMV call count.
- **timing scope**: kernel-only; `--iters=N` sets a hard fixed iteration
  cap (`isStaticIter` flag threads through `ProcessDispatchCG`); this is a
  steady-state per-iteration-kernel-throughput benchmark by design (PERKS's
  whole contribution is a persistent-kernel execution model for iterative
  memory-bound kernels), not a to-convergence solve. `tol = 1e-10f` is
  declared but is not used as a loop-termination condition in this driver.
- **precision & correctness**: templated `ValueT`; default appears to be
  fp64 with an explicit `--fp32` CLI flag for single precision (per
  `conjugateGradient/README.md`'s documented run parameters). Correctness
  is opt-in via `--check`: compares `A*x` against the RHS with a max
  absolute-error scan, printed but not gating.
- **metric**: `time/total_iter` (ms/iteration), a derived GB/s using a
  hand-rolled memory-traffic estimate
  (`spmvaccess = nnz*sizeof(ValueT) + (nnz+N+1)*sizeof(OffsetT)`,
  `totalaccess = max_iter*(spmvaccess + 6*N*sizeof(ValueT))`).
- **baselines**: not visible from this file; PERKS's broader paper targets
  general locality-optimized iterative memory-bound kernels (stencils +
  CG), comparing against non-persistent-kernel baselines.
- source: repo `conjugateGradient/README.md`, `conjugateGradient/cg_driver.cu`
  (full argument-parsing + timing section read).

## 4. Suzuki, Iwashita — F3R (SC'25)

- key: `conf/sc/SuzukiI25`, artifact: github.com/suzuki-hpc/F3R
- **workloads** (code-verified, `sc25/suite-gpu.py`): two named lists of
  real SuiteSparse matrices — `symmetric` (15 matrices: apache2, audikw_1,
  Bump_2911, ecology2, Emilia_923, G3_circuit, ldoor, Queen_4147, Serena,
  thermal2, tmt_sym, plus 4 synthetic `hpcg_<nx>_<ny>_<nz>.mtx` grids) and
  `general` (16 nonsymmetric matrices solved with BiCGSTAB, out of scope for
  this CG-only track, includes 4 synthetic `hpgmp_<nx>_<ny>_<nz>.mtx`
  grids). The `hpcg_*`/`hpgmp_*` matrices are generated locally by
  `matrix/Makefile` rather than downloaded — i.e. this paper **directly
  uses an HPCG-benchmark-style synthetic 3D-stencil generator alongside the
  real SuiteSparse matrices**, confirming this as a real axis in the
  track's own literature, not a spec-author invention.
- **timing protocol** (code-verified, `suite-gpu.py`'s `T3G_SYM`/`T3G_GEN`):
  each solver binary (`cg64/32/16-gpu.exe`, `gm64/32/16-gpu.exe` [GMRES],
  `f3r64/32/16-gpu.exe`) is invoked once per matrix with an explicit
  `<average>` CLI argument — the binary internally repeats the **entire
  to-convergence solve** `<average>` times and reports averaged
  `Time,Iter,ImplRes,ExplRes`. This is "average of N full to-convergence
  solves," not "average of N fixed-size kernel calls" — a materially
  different repetition unit than SPCG's per-iteration-event or PERKS's
  fixed-iteration-count approach, and appropriate here because F3R's own
  inner fp16 solver adaptively varies its iteration count run to run.
- **timing scope**: full to-convergence solve, average of `<average>` runs
  (README recommends `<average>=1` for a quick reproduction, more for
  stable timing). Matrix load / H2D not explicitly separated in the
  surveyed script (opaque inside each `.exe`).
  Per README: **CPU tests use Intel oneAPI icpx with `-mavx512fp16`** (real
  hardware fp16, not emulated) and **GPU tests use nvcc on an A100**.
- **precision & correctness**: explicit 3-way precision sweep per solver
  family — fp64/fp32/fp16 (`64`/`32`/`16` suffix in each binary name) — the
  paper's central axis. `ImplRes`/`ExplRes` columns (implicit residual
  from the recurrence vs. explicit recomputed `b-Ax`) suggest the artifact
  deliberately checks for the residual-recurrence-vs-true-residual drift
  that low precision can introduce, a correctness dimension specific to
  mixed/reduced-precision Krylov methods.
- **metric**: wall time (averaged), iteration count, implicit + explicit
  final residual — written to CSV, later turned into paper tables/figures
  via `plot.py`.
- **baselines**: plain double/single/half-precision CG, GMRES; paper
  abstract claims up to 1.65x/2.42x speedup over double precision.
- source: repo `README.md`, `sc25/README.md` (full reproduction-workflow
  read), `sc25/suite-gpu.py` (full file read).

## 5. Bernaschi, Celestini, Vella, D'Ambra — BootCMatchGX (TPDS'23)

- key: `journals/tpds/BernaschiCVD23`, artifact:
  github.com/bootcmatch/BootCMatchGX
- **workloads**: dual mode, code-verified. (a) User-supplied Matrix Market
  file (`--matrix <FILE>`), regression-test matrices under
  `BCMGX/src/test/data/mtx`. (b) **Synthetic 3D Laplacian generator**
  (`--laplacian-3d <SIZE>` / `--laplacian <SIZE>` with
  `--laplacian-3d-generator [7p|27p]`, generating a `SIZE^3`-unknown 7- or
  27-point-stencil SPD system) — an explicit, code-level HPCG-style
  generator axis, used directly in the shipped regression-test configs
  (e.g. `lap3d_27p_50x50x50_1x1x1`, `lap3d_27p_50x50x50_2x1x1`, scaled
  across process grids for weak-scaling tests).
  Solver+preconditioner combinations are selected via `.properties` files
  under `BCMGX/src/test/data/settings/` (24 files: CGHS/FCG/CGS-variants
  x {AFSAI, BCMG (AMG), l1Jacobi, none} preconditioner).
- **timing protocol** (code-verified, `BCMGX/src/solver/cghs/CG_HS.cu` +
  `BCMGX/src/utility/profiling.h`): the classic to-convergence CG-HS loop —
  `for (iter=0; iter<ip.itnlim; iter++) { ... if (l2_norm < ip.rtol*delta0)
  break; }`. `itnlim` defaults to 2000, `rtol` defaults to 1e-6 (both from
  `sample.properties` / `Params.h`), stopping criterion is the **relative**
  residual (`stop_criterion=1` scales by the initial residual norm
  `delta0`). Fine-grained region timing uses `BEGIN_PROF`/`END_PROF` macros
  (gated by a `detailed_prof` flag, e.g. wrapping `MPI_Allreduce` calls
  inside the CG loop) implemented with `std::chrono` or `gettimeofday`
  depending on build flags; the profiler aggregates **min/max/avg across
  MPI ranks** and writes to a file (`-p`/`-P`/`-M` CLI flags select
  summary vs. detailed vs. per-region profiling output). AMG-preconditioner
  setup (bootstrap hierarchy construction) is a distinct phase from the
  CG-HS iteration loop and is reportable separately via these same flags,
  though the exact top-level "setup time" vs. "solve time" split was not
  directly confirmed in the files read (inferred from the CLI's separate
  `-i`/info and `-M`/detailed-metrics outputs).
- **timing scope**: to-convergence, MPI-CUDA multi-GPU; per-iteration
  residual history is stored (`out->resHist`) for the full run.
- **precision & correctness**: fp64 (`vtype`/`rtol`/`conv_ratio` all
  `double` in `Params.h`); correctness = the same relative-residual
  criterion that gates loop termination (`ASSERT(std::isfinite(l2_norm))`
  guards against NaN/divergence).
- **metric**: iteration count, final residual, wall time (via the
  profiling infrastructure above); paper abstract states up to 2x speedup
  vs. Nvidia AmgX.
- **baselines**: Nvidia AmgX (per abstract); internally also compares
  solver variants (CGHS/FCG/s-step CGS/pipelined CGS) and preconditioners
  (BCMG/AFSAI/l1Jacobi/none) against each other via `make regressionTests`.
- source: repo `README.md`, `BCMGX/src/solver/cghs/CG_HS.cu` (full loop
  read), `BCMGX/src/config/Params.h` (defaults), `BCMGX/src/utility/
  profiling.h`, `BCMGX/src/test/data/settings/*.properties`.

## 6. Noack, Krüger, Koch — Graphene / IPU sparse solvers (IPDPS'25)

- key: `conf/ipps/NoackKK25`, artifact: github.com/esa-tu-darmstadt/graphene
- **workloads** (code-verified, `applications/benchmark/benchmark.cpp`):
  user-supplied matrix file (`-m/--matrix`) OR a synthetic 3D Poisson-matrix
  generator (`-p/--poisson nx,ny,nz`, `host::generate3DPoissonMatrix`) — a
  second, independent confirmation of the synthetic-3D-stencil-generator
  axis in this track's literature (matches BootCMatchGX's and F3R's
  hpcg-generator usage, on a completely different hardware platform). Two
  benchmark modes are selectable via the JSON config's `"benchmark"` field:
  `"solve"` (runs a full configured solver, e.g. PBiCGStab, to whatever
  stopping criterion the JSON's `solver` block specifies) or `"spmv"`
  (isolated SpMV only, no solve loop) — this paper's own benchmark harness
  already separates a kernel-only variant from an end-to-end solve variant,
  independently motivating this spec's fixed-iteration/to-convergence
  split.
- **timing protocol**: NOT measured by application-level wall-clock code in
  this file — profiling is delegated to Poplar's own IPU profiler
  (`Runtime::enableProfiling(profileDirectory)`, opt-in via
  `-d/--profile <DIR>`), a platform-specific instrumentation mechanism
  fundamentally different from the CUDA-event/`gettimeofday`/`std::chrono`
  timers every other surveyed paper uses. No warmup/repetition count is
  visible in this driver file.
  README notes the IPU's cacheless, all-to-all-fabric design permits
  matrix reordering "without performance penalty," and that the framework
  offers software-emulated fp64 and a `twofloat` double-word-arithmetic
  extended-precision type (since the IPU lacks native fp64) — an
  **extended-precision** axis (opposite direction from F3R/Mille-feuille's
  reduced-precision fp16 focus), noted as a divergence below.
- **timing scope**: unresolved from this file — the JSON-configured solver
  block's max-iteration/tolerance defaults were not visible in the files
  read (would require the `Configuration::fromJSON` parser and a sample
  JSON config, not fetched here for time budget).
- **precision & correctness**: `Type::FLOAT32` used throughout the
  benchmark driver's tensor declarations (README's suite of solvers:
  PBiCGStab, Mixed-Precision Iterative Refinement, ILU0/DILU
  preconditioners, Gauss-Seidel); no explicit correctness-tolerance value
  found in this file.
- **metric**: not visible from `benchmark.cpp` itself (delegated to the
  external Poplar profiler's own reporting).
- **baselines**: abstract claims up to 150x vs. CPU/GPU.
- source: repo `README.md` (full read), `applications/benchmark/
  benchmark.cpp` (full read).

## 7. Brunn, Himthani, Biros, Mang, Mehl — CLAIRE (SC'20)

- key: `conf/sc/BrunnHBMM20`, arXiv:2008.12820 — **weaker evidence**: full
  arXiv-HTML/PDF fetches via WebFetch both truncated mid-document (Section
  3, before the Numerical Experiments section); only the abstract and
  fragments of the preconditioner-tolerance discussion rendered.
- **workloads** (from rendered fragments + abstract): 3D diffeomorphic image
  registration on structured grids from 256^3 ("~50M unknowns," single
  V100, "5 seconds") up to 2048^3 ("~25B unknowns," 64 nodes x 4 GPUs = 256
  GPUs on TACC Longhorn — the paper's largest weak-scaling point, claimed
  "152x larger than the largest problem solved in state-of-the-art GPU
  implementations"). Real datasets referenced: NIREP, CLARITY (up to
  20K x 20K x 1K voxels).
- **timing protocol**: not confirmed (the rendering did not reach Section 4).
  Abstract states weak/strong scaling results are reported; no
  warmup/repetition/timer detail could be extracted.
- **timing scope / preconditioner**: the paper introduces two new
  preconditioners (Inv-H0 and a two-level 2LInv-H0) for the reduced-space
  Gauss-Newton Hessian system; the **inner PCG solve for the preconditioner
  itself uses a looser, dataset-dependent tolerance** than the outer solve
  (epsilon_H0 = 1e-3 for the NIREP dataset, 1e-2 for CLARITY) — i.e. this
  paper nests a nominally "kernel-level" tolerance choice inside a much
  larger nonlinear optimization loop, structurally different from the
  other 6 papers' standalone-CG-solve framing (see Divergences).
- **precision & correctness**: not confirmed from rendered fragments
  (medical-imaging registration codes conventionally use fp64; not
  independently verified here).
- **metric**: time-to-solution, weak/strong scaling efficiency (per
  abstract); exact table contents not confirmed.
- **baselines**: "comparison with state-of-the-art CPU and GPU
  implementations" (per abstract); claims 34x vs. CPU version and 50x vs.
  other GPU implementations somewhere in the rendered text, not tied to a
  specific named baseline in the fragments retrieved.
- source: arXiv abstract page + `ar5iv.labs.arxiv.org/html/2008.12820`
  (partial render only — Section 4 not reached); abstract from
  `data/track_inputs/cg-krylov.json`.

---

## Divergences

- **Fixed-iteration vs. to-convergence is a real, not invented, split in
  this track's own artifacts.** PERKS times a fixed `--iters` count
  (steady-state kernel throughput is its explicit research question).
  SPCG, BootCMatchGX, and F3R all solve to a residual-based stopping
  criterion. Graphene's own benchmark harness even exposes both modes
  (`"benchmark": "solve"` vs. `"spmv"`) in one config schema. This spec's
  variant split (kernel-fixed-iter vs. e2e-to-convergence) mirrors a
  distinction the papers themselves already draw, rather than imposing an
  external one.
- **Mille-feuille's public artifact does not actually implement
  to-convergence timing despite its function name and README implying
  it.** `cg_solve_reduce`'s loop condition is hardcoded to
  `iterations < 10000` regardless of the `maxiter=10` argument passed at
  the call site, and the residual-check line that would make it
  convergence-driven is commented out in the shipped code. The reported
  "per-iteration" time is further divided by a literal `100`, not by the
  true iteration count (10000) — a second, independent bug. This spec's
  fixed-iteration variant requires per-iteration event timestamps and an
  iteration count that is verifiably read back from the actual loop
  counter, precisely to catch this class of mismatch.
- **Preconditioner-setup-time reporting ranges from exemplary (SPCG) to
  entirely absent from the surveyed file (Mille-feuille has none since it
  is unpreconditioned; PERKS is unpreconditioned; Graphene's split is
  unconfirmed).** SPCG's `main.cpp` is the strongest model found: distinct
  CUDA-event brackets for ILU(0) analysis+factorization vs. the PCG loop,
  both written to the same output row. This spec's e2e-to-convergence
  variant adopts SPCG's discipline as the mandatory pattern.
  BootCMatchGX's AMG bootstrap-hierarchy setup is a comparably substantial
  one-time cost for its preconditioner, but the exact setup-vs-solve output
  split could not be fully confirmed from the files read (see
  open_questions).
- **The synthetic 3D-stencil ("HPCG-style") generator is used
  independently by three of the seven papers on three different
  platforms**: F3R (`hpcg_<nx>_<ny>_<nz>.mtx`/`hpgmp_*` generated by a
  local Makefile, alongside real SuiteSparse matrices, same solver
  binaries), BootCMatchGX (`--laplacian-3d SIZE --laplacian-3d-generator
  7p|27p`, multi-GPU/MPI), and Graphene (`-p/--poisson nx,ny,nz`, IPU).
  This independently confirms the matrix-suite axis named in this track's
  design brief (SuiteSparse SPD subset vs. HPCG-style generator) as a real
  methodological split, not a spec-author invention, and justifies giving
  it a dedicated variant rather than folding it into the general SuiteSparse
  variant's "recommended_subset."
- **Precision goes in two opposite directions across this track.** F3R and
  Mille-feuille both push toward *reduced* precision (fp16 inner solves /
  tile-grained on-chip fp32 downcasting) to save bandwidth. Graphene pushes
  toward *extended* precision (software-emulated fp64 and a
  `twofloat` double-word type) because the IPU lacks native fp64. SPCG
  defaults to plain fp32. BootCMatchGX is fp64-only. PERKS defaults to
  fp64 with an fp32 flag. No single "precision axis" direction fits all
  seven papers; this spec's precision-cascade variant is scoped to the
  reduced-precision direction (F3R/Mille-feuille), since that is where two
  independent papers actually overlap with concrete, comparable numbers —
  extended-precision (Graphene) is left as an open question.
- **CLAIRE is qualitatively different from the other six papers**: its PCG
  solve is one component nested inside a much larger nonlinear Gauss-Newton
  image-registration optimization, with an inner-solve tolerance
  (1e-3/1e-2) that is itself dataset-dependent and chosen for the outer
  optimization's convergence behavior, not for the CG kernel's own
  standalone accuracy. It cannot be cleanly reduced to any of this spec's
  four variants without stripping away the surrounding registration
  pipeline; it is treated as a related-but-out-of-scope application
  benchmark (see open_questions) rather than forced into the general CG
  variants.
- **Matrix suite selection is inconsistent across artifacts in size, not
  just count.** SPCG's 114-matrix suite tops out around 230K rows
  (`msc23052`); F3R's `symmetric` list goes up to `Queen_4147`
  (~316M nnz) and `audikw_1` (~9.4M nnz) — roughly three orders of
  magnitude larger at the top end. PERKS and Mille-feuille's public
  artifacts ship only a single tiny smoke-test matrix each and give no
  fixed evaluation list at all. This spec's recommended_subset draws from
  the union of SPCG's and F3R's actually-confirmed lists to span this
  range, rather than adopting either paper's list alone.

## evidence

- conf/sc/MaACSH25: "114 named real SuiteSparse SPD matrices (ssgetpy list); fp32; to-convergence tol=1e-12,max_iter=1000; ILU(0) setup timed separately via CUDA events (Preconditioning Time vs PCG Time); per-iteration CUDA-event CSV; 1.23x/1.65x iterative-phase, 1.68x/3.73x e2e vs non-sparsified."
- conf/sc/YangZNJS0T024: "SuiteSparse mtx (no fixed list in artifact); tile-grained mixed precision (fp64 base + fp32 Val_Low); loop hardcoded to 10000 iters ignoring passed maxiter=10, residual-stop line commented out; reported time divided by literal 100, not real iter count; format-conversion timed separately; claims 3-5x vs cuSPARSE."
- conf/ics/ZhangWCMWEM23: "user mtx or synthetic tridiag; fixed --iters (isStaticIter), single CUDA-event bracket over all iters / total_iter; adaptive warmup calibrates max_iter to hit ~350ms; default fp64, --fp32 flag; no residual-based stopping in driver; hand-rolled GB/s traffic model."
- conf/sc/SuzukiI25: "15 symmetric SuiteSparse + 4 synthetic hpcg_*.mtx grids (+16 nonsymmetric/BiCGSTAB, out of scope); fp64/fp32/fp16 explicit 3-way sweep per solver family; <average> repeated full to-convergence solves, not fixed-iter calls; implicit+explicit residual both checked; icpx -mavx512fp16 on CPU, A100 on GPU; up to 1.65x/2.42x vs fp64."
- journals/tpds/BernaschiCVD23: "user mtx or synthetic 3D Laplacian generator (7p/27p, weak-scaling regression configs); fp64; to-convergence itnlim=2000,rtol=1e-6 relative-residual stop; std::chrono/gettimeofday profiling with min/max/avg across MPI ranks; 24 solver x preconditioner (AFSAI/BCMG-AMG/l1Jacobi/none) config combinations; up to 2x vs AmgX."
- conf/ipps/NoackKK25: "user matrix or synthetic 3D Poisson generator; JSON-configured solve-vs-spmv benchmark modes (paper's own harness already splits kernel-only from to-convergence); fp32 default + software-emulated fp64/twofloat extended precision (IPU has no native fp64); timing delegated to Poplar's own IPU profiler, not app-level wall clock; up to 150x vs CPU/GPU."
- conf/sc/BrunnHBMM20: "weak/strong scaling 256^3 to 2048^3 (up to 256 GPUs); inner preconditioner PCG tolerance dataset-dependent (1e-3 NIREP, 1e-2 CLARITY); PCG solve nested inside nonlinear Gauss-Newton registration loop, not a standalone kernel; full timing-protocol/precision details unconfirmed (arXiv fulltext render truncated before Section 4)."
