# Lattice-Boltzmann track — evaluation-methodology survey

## Scope note (read this first)

No paper in the corpus carries the classifier slug `lattice-boltzmann` literally
— `output/benchmark_groups.json` has no such key. All 5 papers below were
instead tagged `kernels: "lattice Boltzmann..."` / `categories: stencil_pde`
and folded into the generic `stencil` group by the Phase-1 classifier (see
`benchmark_groups.py`'s own `DOMAIN` dict, which *does* list `lattice-boltzmann`
as an intended sibling slug of `stencil` — the classifier just never emitted it
per-paper). This track's paper list was therefore built by filtering
`output/included.json` for `kernels` containing "lattice Boltzmann" /
"Lattice Boltzmann", not by reading a `benchmark_groups.json["lattice-boltzmann"]`
entry.

**More importantly: as scored by the Phase-1 artifact-finding pipeline, all 5
of these papers have `artifact_status: "none"`.** One artifact was found
anyway by this survey's own manual due-diligence pass (`gh api` + web search)
that the automated pipeline missed — see MahmoudSM24 below, whose grid-refinement
contribution lives on in the actively-maintained `Autodesk/XLB` library, cited
by name in that repo's README. The other 4 papers (Herschlag, Gounley, Liu/
SunwayLB, Qin) were searched for by title/author/lab on GitHub and turned up
nothing public; SunwayLB in particular is Sunway-TaihuLight-specific
architecture code, which is characteristically not released. All 4 are
surveyed from their published abstract only (no OA fulltext PDF could be
extracted — OSTI/escholarship serve scanned/compressed PDF streams this
pipeline's tools could not turn into text), per the instructions' documented
abstract-only fallback. This is flagged prominently, not silently, because it
materially weakens this track's spec relative to `stencil`'s: most protocol
choices below are inferred from the one code-grounded paper (XLB) and from
standard LBM benchmarking convention, not cross-checked against 4 of the 5
papers' actual run scripts.

---

## MahmoudSM24 — Optimized GPU Implementation of Grid Refinement in LBM
IPDPS 2024, `conf/ipps/MahmoudSM24`, no arXiv id; OA PDF (escholarship,
`qt0x86w4w1`) could not be text-extracted; **artifact found via manual search:
`github.com/Autodesk/XLB`** (`gh api`, primary source for this entry — Phase-1
pipeline recorded `artifact_status: none` for this paper, which this survey
corrects)

- **Relationship to the artifact**: XLB is Autodesk's actively-maintained
  differentiable LBM library (JAX/NVIDIA-Warp/Neon backends). Its README
  explicitly cites Mahmoud, Salehipour & Meneghin's IPDPS'24 paper as the
  origin of the Neon-backend grid-refinement capability ("If you use the grid
  refinement capabilities in your work, please also cite: @inproceedings
  {mahmoud2024optimized...}") — i.e. the paper's contribution is live,
  first-party code in this repo, not a third-party reimplementation.
- **Lattice models**: D3Q19 (default) and D3Q27, selectable via `--velocity_set`;
  collision model BGK (default) or KBC, selectable via `--collision_model`
  (KBC requires D3Q27).
- **Precision**: 4 policies via `--precision`: `fp32/fp32`, `fp64/fp64`,
  `fp64/fp32` (mixed store/compute), `fp32/fp16` — the paper's own grid-refinement
  work is precision-orthogonal to its actual contribution (kernel fusion +
  memory-access optimization at refinement interfaces).
- **Grid refinement**: `--num_levels` sets the number of nested-cuboid
  refinement levels; each finer level halves the local grid spacing and steps
  2× more often per coarsest-level step (standard LBM temporal sub-cycling).
  `--mres_perf_opt` selects between `NAIVE_COLLIDE_STREAM` and
  `FUSION_AT_FINEST` kernel-fusion strategies — this axis IS the paper's
  contribution.
- **Domain/workload**: lid-driven cavity (LDC), Reynolds number fixed at 5000,
  refinement peels inward from the boundary (finer resolution concentrated
  near walls) — `examples/performance/mlups_3d_multires.py`. The paper's own
  headline result (from web search, not fulltext-verified) is a domain of
  1596×840×840 on a single A100-40GB, enabled specifically by grid refinement's
  memory savings vs. a uniform fine grid.
- **Timing protocol** (read directly from `mlups_3d_multires.py`): single
  trial, `wp.synchronize()` (device-wide barrier) immediately before
  `start_time = time.time()` and immediately before the stop timestamp, wrapping
  the whole `num_steps` loop as one elapsed interval. **No discarded warmup
  run** in this particular script (the single-resolution sibling script does
  have one — see next bullet).
- **Sibling single-resolution script is materially more rigorous**
  (`examples/performance/mlups_3d.py`): `warmup_iterations = 10` executed and
  device-synchronized *before* timing starts; `--repetitions N` (default 1,
  but explicitly designed for N>1) times N independent trials and reports
  **mean ± std** of both elapsed time and MLUPs, plus per-step ms mean±std.
  This is the most rigorous public timing harness found across this survey's
  entire cross-track effort (stencil, fdtd-seismic, and this track combined) —
  see Divergence 1.
- **Metric**: (E)MLUPS = Million Lattice Updates Per Second, computed as
  `cube_edge^3 * effective_steps / elapsed_s / 1e6`; for the multi-res variant,
  `effective_steps = num_steps * 2^(num_levels-1)` (finer-level sub-stepping is
  counted, not hidden).
- **Correctness**: not present in the two performance-benchmark scripts read;
  the repo does have a `tests/` tree (`kernels/`, `boundary_conditions/`,
  `grids/`) implying unit-level correctness coverage exists elsewhere in the
  codebase, but no pointwise tolerance value specific to the LDC Re=5000
  performance case was confirmed — flagged as an open question.
- **Source**: `github.com/Autodesk/XLB` — `README.md`,
  `examples/performance/mlups_3d.py`, `examples/performance/mlups_3d_multires.py`.

---

## Herschlag, Lee, Vetter & Randles — GPU data access patterns for D3Q19 LBM
TPDS 2021, `journals/tpds/HerschlagLVR21`, OA link is an OSTI PDF
(`osti.gov/servlets/purl/1807241`) that could not be text-extracted;
**abstract only**

- **Lattice model**: D3Q19, explicitly.
- **Workload class**: "complex domains" / "complex geometries" specifically —
  this paper's whole point is that geometric (solid/fluid boundary) data must
  be accessed indirectly on such domains while lattice (distribution-function)
  data is conventionally accessed lexicographically, and it argues this
  default pairing is often suboptimal.
- **Method**: empirical testing **and** performance modeling of "a suite of
  memory access schemes" (plural addressing strategies compared head-to-head,
  not just one optimized kernel vs. a naive baseline) — headline finding is
  that "semi-direct" addressing is often better than the common
  fully-indirect scheme.
- **What could not be confirmed from the abstract alone**: exact grid sizes,
  precision, GPU hardware, MLUPS numbers, timing protocol, correctness
  method. All flagged as open questions.
- **Source**: abstract from `output/included.json`; OA PDF fetch attempted and
  failed (binary PDF stream, not text-extractable by this survey's tools).

---

## Gounley, Vardhan, Draeger, Valero-Lara, Moore & Randles — moment-representation propagation pattern
TPDS 2022, `journals/tpds/GounleyVDVMR22`, OA link is an OSTI PDF
(`osti.gov/servlets/purl/1814345`) that could not be text-extracted;
**abstract only**. (Same research group/lab as Herschlag above — both from
the Randles group's LBM-on-GPU line of work, evidenced by shared last author.)

- **Method**: stores simulation state as **moments** of the LBM distribution
  function rather than the distribution function itself ("effectively
  lossless compression"), for the **regularized** LBM collision operator, in
  **3D**. A companion propagation pattern is engineered specifically to be
  cache-aware for this moment representation.
- **Claimed benefit**: "substantially reduc[es] both the storage and memory
  bandwidth required" — i.e. this paper's contribution is explicitly a
  memory-footprint/bandwidth optimization, directly relevant to whatever this
  track picks as its GB/s secondary metric.
- **What could not be confirmed from the abstract alone**: lattice model
  (D3Q19/D3Q27), grid sizes, precision, hardware, MLUPS numbers, timing
  protocol, correctness method. Search results (title/venue lookup only, not
  fulltext) surfaced a same-lab SC'23-workshop follow-up ("Moment
  Representation of Regularized LBM on NVIDIA and AMD GPUs") that is NOT in
  this corpus and was not surveyed — noted for context only.
- **Source**: abstract from `output/included.json`; OA PDF fetch attempted and
  failed.

---

## Liu, Chen, Liu, Ma, Liu, Zhang, Fu, Yang — SunwayLB
TPDS 2024, `journals/tpds/LiuCLMLZFY24`, no OA link recorded; **abstract
only**; no public artifact found (Sunway-TaihuLight-specific architecture
code — this class of paper is characteristically closed-source, consistent
with every other Sunway-architecture paper surveyed across this whole
kernel-papers corpus)

- **Target hardware**: "advanced heterogeneous systems such as the Sunway
  supercomputers" — SW26010-class or successor many-core CPU architecture,
  not GPU.
- **Scope**: explicitly "industrial applications" CFD at extreme scale;
  contributions include a "customized multi-level domain decomposition and
  data sharing scheme" — i.e. this paper's angle is distributed-memory
  scaling, not single-node kernel throughput, more analogous to `stencil`
  track's variant-3 (end-to-end application) framing than variant-1
  (kernel-only).
- **What could not be confirmed from the abstract alone**: lattice model,
  grid sizes, node counts, precision, GLUPS/MLUPS numbers, timing protocol,
  correctness method.
- **Source**: abstract from `output/included.json` only.

---

## Qin, Li, Li, Tao, Wu — GPU LBM on sparse complex geometries
TPDS 2025, `journals/tpds/QinLLTW25`, no OA link recorded; **abstract only**;
no public artifact found by search

- **Workload class**: "sparse complex geometries" (worked examples cited:
  porous media, arterial blood flow, tissue fluid) — same problem class as
  Herschlag above (complex-geometry LBM), independently confirming this is a
  real, recurring sub-problem in the track's literature, not a one-off.
- **Techniques**: "compact memory layout, sophisticated memory access" plus
  "an address index array and a node classification coding scheme" to
  "improve the GPU thread utilization ratio and reduce the GPU global memory
  access" — i.e. this paper is squarely in the same memory-layout-for-sparse-
  domains sub-problem as Herschlag/Gounley, giving this track's "complex
  geometry" variant 3 independent corroboration from 3 separate papers.
- **What could not be confirmed from the abstract alone**: lattice model,
  grid sizes, precision, GPU hardware, MLUPS numbers, timing protocol,
  correctness method.
- **Source**: abstract from `output/included.json` only.

---

## Divergences

1. **Timing rigor is bimodal in a way this survey can only see one side of.**
   The one paper this survey could ground in real code (XLB/MahmoudSM24) has
   *two* benchmark scripts of visibly different rigor: the multi-resolution
   script (the paper's own direct contribution) times a single trial with no
   discarded warmup, while the sibling single-resolution script has a
   10-iteration discarded warmup AND supports N-repetition mean±std reporting
   as a first-class CLI feature. This spec adopts the *better* of the two as
   its mandatory protocol (matching this whole survey effort's general
   practice, and `stencil` track's own precedent, of fixing the track's worst
   practice rather than copying it) — but this survey could not determine
   whether the other 4 papers' own (unavailable) code has anything like
   `mlups_3d.py`'s repetition support, since none of their artifacts could be
   found.

2. **This track splits cleanly into two workload classes that no single paper
   bridges.** MahmoudSM24/XLB benchmarks a regular, non-refined-vs-refined
   cuboid domain (lid-driven cavity). Herschlag, Gounley, and Qin all
   independently target **complex/sparse geometries** (porous media,
   vasculature, arbitrary solid boundaries) where the dominant cost is
   irregular/indirect memory access rather than raw compute — a qualitatively
   different bottleneck than the regular-grid case. SunwayLB is different
   again (distributed-memory extreme-scale industrial CFD). The spec's 3
   variants map onto these 3 classes rather than forcing one grid/metric
   convention across all of them.

3. **MLUPS/GLUPS as a metric is comparatively well standardized in this
   track** (unlike GFLOP/s in the `stencil` track's Divergence 4) — every
   paper surveyed either reports it directly (XLB) or operates in a problem
   space (bandwidth-bound complex-geometry LBM) where it's the natural
   choice. The open question is whether "lattice update" is counted per
   distribution-function collision-stream (the literal per-direction op) or
   per lattice site (site-level, direction-count-agnostic) — XLB's own
   formula (`cube_edge^3 * steps`) is site-level, not direction-level; this
   spec adopts site-level MLUPS as primary and states the convention
   explicitly, since a D3Q27 kernel doing "the same" MLUPS as a D3Q19 kernel
   is doing more total work per site and this must not be silently conflated.

4. **No paper in this track's own literature states an explicit pointwise
   correctness tolerance** (unlike PERKS/AOStencil in the `stencil` track).
   XLB's repo has a test suite but no benchmark-script-level tolerance value
   was confirmed; the 4 abstract-only papers give no information at all on
   this axis. The spec's correctness gate is this survey's own
   recommendation (analogous to `stencil`'s AN5D-derived 1e-5 relative-error
   convention), explicitly flagged as not sourced from this track's own
   papers.
