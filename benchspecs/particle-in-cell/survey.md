# Particle-in-cell track — evaluation-methodology survey

## Scope note (read this first)

`output/benchmark_groups.json["particle-in-cell"]` has exactly 1 paper. This
is a genuinely thin track and this survey covers that 1 paper (below the
instructions' "≥5" bar, but the instructions explicitly permit "all, if
fewer"). Unusually for this project, the paper's **artifact repository could
not be accessed** even though `benchmark_groups.json`/`artifacts_found.json`
list one (`github.com/sherry-roar/polarpic-ad`, `artifact_status: "likely"`,
not `"verified"`) — `gh api repos/sherry-roar/polarpic-ad` returns
`404 Not Found`, and no renamed/relisted variant was found under the
`sherry-roar` account (`gh api users/sherry-roar/repos` lists 18 repos, none
matching; it does host a fork-shaped `WarpX` repo, consistent with the paper
using WarpX as its baseline, but not the PIC-AD artifact itself). This is
reported honestly rather than silently substituted: the artifact is very
likely just not yet public (this is a 2026 HPDC paper), which is exactly
what `artifact_status: "likely"` (vs. `"verified"`) already signals upstream.
In its place, **the arXiv fulltext (id `2604.19337`) was fully reachable**
via `arxiv.org/html/2604.19337` and is unusually detailed for a systems
paper — it is used as the primary and sole source below, per the
instructions' source-preference order (arXiv fulltext is preference #1,
ahead of the repo).

---

## Rao et al. — POLAR-PIC: A Holistic Framework for Matrixized PIC with Co-Designed Compute, Layout, and Communication
HPDC 2026, `conf/hpdc/RaoCPXFZWGWCL26`, arXiv `2604.19337`, DOI
`10.1145/3806645.3807574`; artifact URL recorded but inaccessible (see scope
note); source used: arXiv fulltext (`arxiv.org/html/2604.19337`, fetched
twice — once via `/abs/` which only yielded the abstract, once via `/html/`
which yielded full Evaluation-section detail).

- **What it does**: a co-designed PIC framework with three contributions:
  (i) reformulates Field Interpolation into an MPU (Matrix Processing
  Unit)-friendly outer-product form instead of the traditional
  scatter/gather stencil loop, (ii) maintains a physically-ordered particle
  layout (particles kept contiguous by spatial locality as they move) to
  preserve memory locality across timesteps, (iii) overlaps particle
  redistribution (inter-rank communication after particles cross cell/rank
  boundaries) with the Deposition phase to hide migration cost. All
  simulations use 3rd-order B-spline particle shape factors, a Yee
  finite-difference field solver, direct-current Deposition, and the Boris
  particle pusher — i.e. this is a standard explicit electromagnetic PIC
  cycle (push → deposit → field-solve → interpolate), with the paper's
  contribution concentrated in the particle-processing phase (interpolation
  + deposition + redistribution), not the field solve.
- **Hardware**: primary platform is "LX2", a CPU-based pilot-exascale-system
  node: "two LX2 high-performance CPUs" per node, "over 256 cores
  distributed across two compute dies", each die with dedicated vector
  (VPU) and matrix (MPU) processing units, "128GB of off-die DDR memory
  organized across 4 NUMA domains" per die, "LXLink network...providing up
  to 48 GB/s bidirectional bandwidth per NIC" with RDMA. Cross-platform
  comparison ("LS system") uses "two 28-core CPUs and 8 NVIDIA A800 GPUs
  ...80 GB of HBM2e memory" for the WarpX/GPU side of the comparison.
- **Workloads/inputs** (two named scenarios, not a broad benchmark suite —
  this is the track's single biggest fairness caveat, see Divergences):
  - **Uniform Plasma**: 3D Cartesian grid, 256×128×128 resolution.
  - **Laser-Ion Acceleration**: 192×192×256 grid over an approximately
    7.5μm × 7.5μm × 15μm physical domain (a real-world application
    scenario, not a synthetic stress test).
  - **Particle density sweep**: particles-per-cell (PPC) swept from 1 to
    512 for both scenarios, with thermal-velocity adjustments used to
    create varying particle-migration intensities (i.e. PPC and thermal
    velocity are the two knobs that stress the redistribution/locality
    contributions specifically).
- **Timing protocol**: "100 timesteps for measurement after an initial
  warm-up phase" (warm-up phase LENGTH not stated in the extracted text —
  recorded as an open question below) — "All experiments are repeated three
  times excluding warm-up overhead; we report the average execution time
  across these trials to filter out transient system jitter." This is
  **mean over 3 trials**, not median/min — a deviation this spec's protocol
  corrects (see notes_on_fairness).
- **Timing scope / phase decomposition**: the paper's own metric is
  `TParticle = TInterpolation + Tdeposit + Tredistribute`, where
  `TInterpolation` itself "includes Field Interpolation, Push, SoW [scatter
  or weight], and fused re-packing overheads" — i.e. the interpolation
  phase as reported is already a fused/bundled measurement of several
  sub-steps, not a single kernel in isolation. The field-solve (Yee update)
  is implicitly excluded from `TParticle` by construction (it is a
  grid-only operation, not part of the "particle phase").
- **Metric**: Table 2 reports, at representative PPC points: `TParticle (s)`,
  `PPS (Gparticles/s)` [particles-processed-per-second — directly matches
  the parent task's "particles-pushed/s" axis], `CPP (cycles/particle)`,
  and `Speedup (×)` vs. WarpX. Table 3 gives the same four columns across an
  **ablation sweep of interpolation-kernel variants labeled G0–G7** (exact
  meaning of G0–G7 not resolved from the extracted text — likely a
  grid-tiling/blocking-factor sweep for the MPU outer-product kernel, but
  unconfirmed; recorded as an open question).
- **Correctness/validation**: for the laser-ion-acceleration scenario,
  "Mean Squared Error (MSE) of 3.198 and Mean Absolute Error (MAE) of
  0.8401...corresponding to a relative error below 0.84%" against what the
  text implies is the WarpX reference trajectory (the exact reference
  quantity — field values? particle positions? — is not stated precisely
  in the extracted text). Separately and more directly matching the parent
  task's suggested correctness axis: **"Per-step error curves show that
  total charge, total energy, and longitudinal momentum errors remain
  stably bounded throughout"** the simulation — this IS a genuine
  charge-conservation / energy-drift diagnostic, run per-timestep, exactly
  the kind of gate the parent task's axis list asks for.
- **Baseline**: "the native WarpX reference pipeline on LX2" for the
  same-hardware comparison, and WarpX on 8×A800 GPUs for the
  cross-platform/peak-efficiency comparison (13.2% of theoretical peak on
  LX2-CPU for POLAR-PIC vs. 9.6% of theoretical peak on A800-GPU for
  WarpX). WarpX is a real, widely-used open-source PIC code (not a strawman
  baseline).
- **Scalability**: weak scaling from 1 to 4,096 nodes ("over 2 million
  physical cores") under a "high-turbulence" / "high-migration dynamic"
  stress workload: POLAR-PIC sustains 67.5% weak-scaling efficiency vs.
  WarpX's 42.5%, attributed to the asynchronous
  redistribution/deposition-overlap design achieving a "99.1% overlap
  ratio" for communication.
- **Source**: arXiv fulltext, `arxiv.org/html/2604.19337` (fetched
  directly; the `/abs/` endpoint alone returned only the abstract and was
  insufficient — flagging this so future surveys in this project know to
  prefer `/html/` over `/abs/` for this journal's arXiv rendering).

---

## Divergences

Only 1 paper is available, so there is no cross-paper divergence in the
usual sense of this survey template. The substantive divergences are
between this track's one available paper and (a) the parent task's assumed
axes, and (b) this project's usual artifact-grounding practice:

1. **No public artifact could be read** (scope note above) — this survey,
   unusually among this project's tracks, is grounded entirely in arXiv
   fulltext with zero repo-code verification of the timing loop structure
   (warmup length, exact timer call, exact definition of PPS). This is a
   real gap: the paper's own prose describes the protocol at the level of
   "100 timesteps, 3 trials, average", but the exact timer (wall-clock vs.
   device counter?) and warmup length are not stated in the extracted text
   and could not be cross-checked against source, unlike e.g. the
   `nbody`/`amr-kernel` tracks' `hostCode.cu`/README-level detail.
2. **"Grid resolution" in the parent task's suggested axes maps cleanly**
   to this paper's two named scenarios (256×128×128 and 192×192×256) — no
   mismatch here, unlike some other singleton tracks in this project.
3. **The correctness axis maps unusually well**: the paper's own per-step
   charge/energy/longitudinal-momentum error-bound diagnostic is exactly a
   charge-conservation/energy-drift gate, not something this spec had to
   invent or substitute (contrast with the `nbody` track, where no such
   check existed in the source and the spec had to add one).
4. **Deposition/push phase split partially, not fully, matches the parent
   task's suggested axis**: the paper's `TInterpolation` bucket bundles
   Field Interpolation + Push + SoW + re-packing into one number rather
   than reporting Push as a fully separate phase from Interpolation. This
   spec inherits the paper's own 3-way split (`TInterpolation` /
   `Tdeposit` / `Tredistribute`) rather than inventing a finer split the
   source does not support.
5. **Statistic choice**: the paper reports **mean of 3 trials** (not
   median/min as this project's spec template prefers). This is corrected
   in the spec's protocol section, flagged explicitly per
   `benchspec_instructions.md`'s fairness principle ("fixed timing
   protocol...statistic (prefer median + report min/max)").

