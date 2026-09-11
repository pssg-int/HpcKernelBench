# AMR-kernel track — evaluation-methodology survey

## Scope note (read this first)

`output/benchmark_groups.json["amr-kernel"]` has exactly 1 paper. This is a
genuinely thin track and this survey covers that 1 paper (below the
instructions' "≥5" bar, but the instructions explicitly permit "all, if
fewer"). Two source-access notes up front:

1. **The paper's fulltext could not be reached.** No arXiv id is recorded
   for this SC 2022 paper (`data/artifacts_found.json` has no `arxiv` key
   for `conf/sc/FernandoNHZSGB22`, only GitHub-README-based artifact
   candidates), and it was not found as an open preprint via web search
   (only the ACM DL / IEEE Xplore paywalled pages and the SC22 conference
   program page were found — `arxiv:2501.14030`, "Accelerating Numerical
   Relativity with Code Generation," is a DIFFERENT, later paper by an
   overlapping author set and was NOT used as a substitute). Per the
   instructions' source order, this survey falls back to source #3
   (abstract) for the numbers the paper itself reports, but is unusually
   well-grounded anyway because of point 2:
2. **The artifact repo is exceptionally detailed and directly readable**
   (`artifact_status: verified`). Both `github.com/paralab/Dendro-GR`
   (the actively maintained repo) and `github.com/paralab/sc22-dgr` (an
   SC22-paper-specific snapshot the maintained repo's own README links to
   for this exact paper's code-generation file) were read via `gh api`:
   `ReadMe.md` (both repos, nearly identical usage instructions), the
   `BSSN_GR/pars/q1.par.json` default parameter file (full content read),
   and the strong-/weak-scaling parameter files and SLURM launch scripts
   under `BSSN_GR/experiment_scripts/ls6/{q1-ss,q1-ws}/`. This gives
   concrete, source-verified refinement/grid/timing-breakdown parameters
   well beyond what the abstract alone provides.

---

## Fernando et al. — A GPU-Accelerated AMR Solver for Gravitational Wave Propagation
SC 2022, `conf/sc/FernandoNHZSGB22`, DOI `10.1109/SC41404.2022.00080`, no
arXiv/OA fulltext link found; artifact `github.com/paralab/Dendro-GR`
(`artifact_status: verified`); companion snapshot repo
`github.com/paralab/sc22-dgr` also read.

- **What it does**: Dendro-GR — a distributed-memory, octree-based
  wavelet-adaptive-mesh-refinement (AMR) framework (built on the Dendro
  library) coupled to a BSSN-formulation numerical-relativity physics
  module (via SymPy-based code generation, "SympyGR"), solving the Einstein
  field equations for binary-black-hole mergers to produce gravitational
  waveforms for LIGO-style data analysis. This SC22 paper specifically adds
  a **multi-GPU (CUDA) extension** of the previously CPU-only solver and
  benchmarks it against the CPU version and prior state-of-the-art NR
  codes.
- **Refinement pattern**: wavelet-error-driven AMR (WAMR) — NOT a fixed
  block-structured pattern; refinement level per octant is set by
  comparing a computed wavelet coefficient of the solution against
  `BSSN_WAVELET_TOL` (default `1e-5` in the shipped `q1.par.json`), with an
  optional spatially-varying tolerance function
  (`BSSN_USE_WAVELET_TOL_FUNCTION: 3`, inner radius `R0=20`, outer
  `R1=220` — tighter tolerance near the two black holes, looser far away).
  A `BSSN_DENDRO_AMR_FAC=0.1` coarsening-hysteresis factor prevents
  refine/coarsen thrashing. Confirmed directly from `BSSN_GR/pars/q1.par.json`.
- **Refinement levels**: `BSSN_MAXDEPTH=15`, `BSSN_MINDEPTH=3` in the
  paper's own default q=1 production config; the strong-scaling config
  (`q1_r2.2.par.json`) uses `BSSN_MAXDEPTH=14`, and the weak-scaling config
  (`q1_ws.par.json`) uses `BSSN_MAXDEPTH=12` — i.e. the scaling-study
  configs deliberately use SHALLOWER max refinement than the physically
  accurate production run, trading solution fidelity for a controllable,
  reproducible per-rank workload (confirmed from both parameter files
  directly).
- **Discretization**: 6th-order (`BSSN_ELE_ORDER=6`) finite-difference
  stencils per octant, RK4 time integration (`BSSN_RK_TYPE=1`), CFL factor
  0.25 (`dt = 0.25 * dx`), Kreiss-Oliger dissipation
  (`KO_DISS_SIGMA=0.4`) for numerical stability — all confirmed from
  `q1.par.json`.
- **Workload/domain**: binary-black-hole merger, mass ratios q ∈ {1, 2, 4}
  (paper's abstract, matching the three shipped default parameter files
  `q1.par.json`/`q2.par.json`/`q4.par.json`); default q=1 domain is a cube
  [-400, 400]^3 (code units), two punctures at x=±4 with mass 0.4824 each
  — confirmed from `q1.par.json`.
- **Timing/profiling breakdown** (confirmed from `ReadMe.md`'s
  "Running experiments" section, both repos): the solver's own profiling
  output reports **min/mean/max across MPI ranks** for each of:
  `step_ets` (# timesteps profiled), `numOcts`/`dof_cg`/`dof_uz` (mesh
  size), `remsh_igt` (**re-grid/regridding runtime** — directly matches the
  parent task's "regridding cost" axis), `evolve` (total RK runtime for
  `step_ets` steps — paper plots `evolve_max`), `unzip_async` (octant-to-
  patch data-layout conversion WITH overlapped communication) vs. `unzip`
  (same, WITHOUT communication — so **communication cost is inferred as
  `comm = unzip_async_max - unzip_max`**, an explicit decomposition), `rhs`
  (the actual BSSN right-hand-side/PDE-solve computation — paper plots
  `rhs_max`), and `zip_async` (patch-to-octant, reverse conversion). This
  is a genuine **regrid-cost vs. solve-cost separation** (`remsh_igt` vs.
  `rhs`/`evolve`), directly matching the parent task's requested axis, and
  it is source-verified, not inferred from the abstract.
- **Separate microbenchmark**: a standalone "Octant-to-Patch"/
  "Patch-to-Octant" padding-zone-computation benchmark
  (`make run_meshgpu_tests`), parameterized by max octree depth, wavelet
  refinement tolerance, partition-load-imbalance tolerance, interpolation
  order (paper used order 6 for all experiments, per `ReadMe.md`), and a
  CPU/GPU flag — this isolates the unzip/zip data-layout-conversion kernel
  from the full BSSN solve, giving a controlled micro-benchmark distinct
  from the end-to-end binary-merger run.
- **Scaling harness** (confirmed from `ReadMe.md` + SLURM scripts under
  `BSSN_GR/experiment_scripts/ls6/`): run on TACC Lonestar6's `gpu-a100`
  queue; strong scaling uses executables `bssnSolverCtx` (CPU) /
  `bssnSolverCUDA` (GPU) with the reduced-refinement `q1_r2.2.par.json`
  (grain size 4 octants/core, deliberately tiny to stress strong scaling);
  weak scaling uses `bssnScalingTest` (CPU) / `bssnWSTestCUDA` (GPU) with
  `q1_ws.par.json` (grain size 16,000 octants/GPU or 1,000 octants/CPU
  core, per `ReadMe.md`). Build requires
  `-DBSSN_DISABLE_INITIAL_GRID_REFINEMENT=ON -DBSSN_PROFILE_SCALING_RUN=ON`
  for scaling runs specifically, and GPU builds additionally need
  `-DOCT2BLK_COARSEST_LEV=31`.
- **Metric (from abstract, since fulltext unreachable)**: 800 GFlops/s on
  a single NVIDIA A100 GPU; 6x speedup over "existing state-of-the-art
  codes" (unnamed in the abstract); 2.5x speedup over a two-socket,
  128-core AMD EPYC 7763 CPU node running an "equivalent CPU
  implementation"; strong scaling to 8 A100s; weak scaling to 229,376 x86
  cores on TACC Frontera (this specific weak-scaling core count is CPU-side,
  per the abstract and the `ReadMe.md`'s cross-reference to the same
  Frontera Texascale-Days result — the GPU-side weak-scaling endpoint in
  the abstract is 8 A100s under strong scaling, not an extended weak-scaling
  GPU core count).
- **Correctness/validation**: NOT resolved from either the abstract or the
  README/scripts read. The repo supports gravitational-waveform (psi4 mode)
  extraction at 6 named observer radii (`BSSN_GW_RADAII: [50,60,70,80,90,100]`)
  and 3 angular L-modes (`BSSN_GW_L_MODES: [2,3,4]`) — this is the
  DOMAIN-LEVEL accuracy artifact (the actual physics output the paper's
  broader research programme validates against other NR codes/LIGO
  templates in companion papers) — but no automated pass/fail numerical-
  tolerance correctness GATE script (e.g. comparing GPU vs. CPU solver
  output, or comparing to a uniform-grid reference) was found in the
  repo's `scripts/` directory (which contains only visualization/data-
  loading utilities: `bssnVis.py`, `constVis.py`, `iovtk.py`, `vtk2vtu.py`,
  etc. — post-processing, not correctness checking). Flagged as an open
  question below rather than a fabricated gate.
- **Source**: `github.com/paralab/Dendro-GR` and
  `github.com/paralab/sc22-dgr` — `ReadMe.md` (both, full read),
  `BSSN_GR/pars/q1.par.json` (full read), `BSSN_GR/experiment_scripts/
  ls6/q1-ss/q1_r2.2.par.json` (partial read, first ~60 lines confirming
  `BSSN_MAXDEPTH`/grain size), `BSSN_GR/experiment_scripts/ls6/q1-ws/
  q1_ws.par.json` (same), `BSSN_GR/experiment_scripts/ls6/q1-ss/
  run_q1_ss_16.sh` (SLURM launch script, full read); abstract text (for
  the paper's own headline numbers, since fulltext was unreachable).

---

## Divergences

Only 1 paper is available, so there is no cross-paper divergence in the
usual sense of this survey template. The substantive divergences are
between this track's one paper and the parent task's assumed axes, and
between the paper's ABSTRACT-level claims and what the REPO could actually
verify:

1. **"Cell-updates/s" is not the paper's own reported metric.** The paper
   reports wall-clock phase times (`evolve`/`unzip`/`rhs`/`remsh_igt`,
   min/mean/max across ranks) and a single headline GFlops/s figure (800
   GFlops/s on one A100), not a normalized cell-updates/s throughput. This
   spec derives a cell-updates/s-equivalent metric from `dof_uz` (grid
   points after padding) and `rhs`/`evolve` time, flagged as a DERIVED
   metric rather than one the paper reports directly.
2. **"Regridding cost vs. solve cost separation" maps unusually cleanly**
   onto this paper's own `remsh_igt` (regrid) vs. `rhs`/`evolve` (solve)
   profiling columns — this is directly source-verified from the README's
   profiling-output-field documentation, not inferred or invented.
3. **"Correctness vs. uniform-grid reference" does NOT have a matching
   artifact in the repo as read.** The repo's own correctness/validation
   story is GW-waveform-domain (psi4 extraction at multiple radii,
   compared in COMPANION papers to other NR codes and LIGO data — not a
   simple uniform-grid numerical-agreement script bundled in this repo).
   This spec's correctness gate is therefore this survey's own
   recommendation (a reduced-resolution reference run), explicitly flagged
   as not sourced from the paper, per Divergences framing used consistently
   elsewhere in this project (e.g. the `nbody` track's equivalent flag).
4. **The paper's fulltext (Methodology/Evaluation prose) was unreachable**
   (no arXiv id, no open preprint found). All headline SPEED numbers above
   (800 GFlops/s, 6x, 2.5x, 8 A100s, 229,376 cores) come from the abstract
   text recorded in `output/included.json`, not from a read of the paper's
   actual evaluation tables/figures — the repo's profiling-field-level
   detail (regrid/solve/unzip/zip breakdown, exact parameter files) is
   comparatively much better source-grounded than the paper's own claimed
   numbers, and this spec is explicit about which parts come from which
   source.

