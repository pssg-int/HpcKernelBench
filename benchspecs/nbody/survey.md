# N-body track — evaluation-methodology survey

## Scope note (read this first)

`output/benchmark_groups.json["nbody"]` has exactly 1 paper. This is a
genuinely thin track and this survey covers all 1 paper available (below the
instructions' "≥5" bar, but the instructions explicitly permit "all, if
fewer"). Additionally: this single paper is NOT a classical long-range
N-body force-summation code (no gravitational/Coulomb potential, no
multipole/tree approximation, no energy-conservation diagnostic) — it is a
GPU ray-tracing-hardware-accelerated **discrete collision detection (DCD)**
system for dense spherical-particle systems. Its relevance to "N-body" is
that fixed-radius neighbor search / pairwise proximity query is the
foundational primitive underneath short-range N-body methods (cell-list-based
MD, SPH, granular/DEM simulation) — this paper accelerates exactly that
primitive, not the force-accumulation step downstream of it. This mismatch
with the parent task's suggested axes (accuracy parameter as multipole
order/mesh resolution, energy-conservation correctness) is real and is
flagged explicitly throughout rather than papered over.

A companion literature search for the "genuinely classical" N-body papers in
this corpus was done for context (see Related-but-excluded below): the one
paper that WOULD match the parent task's axes cleanly (RT-BarnesHut, a
Barnes-Hut tree-code N-body solver) has no public artifact, so it could not
be used as this track's anchor paper — this is reported as an honest gap,
not glossed over by substituting Mochi's numbers for what a Barnes-Hut paper
would report.

---

## Mandarapu et al. — Rethinking Collision Detection on GPU Ray Tracing Architecture (Mochi)
ICS 2026, `conf/ics/MandarapuFPBK26`, no arXiv/OA link recorded; artifact
`github.com/MDurgaKeerthi/Mochi-DCD-on-RT` (`artifact_status: verified`;
`gh api` used directly to read source)

- **What it does**: reformulates discrete collision detection (DCD) for
  spherical particle systems (including non-uniform radii) to run
  efficiently on GPU ray-tracing (RT-core) hardware via NVIDIA OptiX
  (through the OWL wrapper library). Per-substep pipeline: (1) rebuild the
  RT bounding-volume-hierarchy (BVH) acceleration structure since particles
  have moved, (2) launch a ray-generation kernel that queries the BVH for
  collision candidates (broad phase) and resolves actual sphere-sphere
  intersections (narrow phase) in one RT-hardware pass, (3) update particle
  positions/velocities from resolved collisions.
- **Workload/dataset** (confirmed from `src/s23-3Dparticlesim/script.py`):
  n=100,000 particles, dataset file `datasets/100k_particles.txt` (shipped
  in the repo), domain a unit cube (`BOX_WIDTH=1.0`), `frames=200`
  (timesteps of interest) with `iterations_per_frame=400` collision-response
  substeps EACH (confirmed from `hostCode.cu`) — i.e. 80,000 total
  substeps for the default run, not 200.
- **Baseline comparison**: a Taichi-based CPU/GPU implementation is included
  in the repo (`taichi/` directory) specifically as "a baseline comparison
  for Mochi" per the README.
- **Timing protocol** (confirmed from `hostCode.cu`): `std::chrono::
  steady_clock`, THREE separately-accumulated phase timers per substep —
  `build_time` (BVH rebuild), `cd_time` (collision-detection RT launch +
  response), `update_time` (position/velocity integration) — summed at the
  very end and printed as build/detect/update/total (ms) after all
  `num_frames * iterations_per_frame` substeps complete. **Single trial, no
  discarded warmup, no repetition** in the public artifact as read.
- **Correctness/validation mechanism**: a compile-time `COUNT_COLLISIONS`
  flag (`-DCOUNT_COLLISIONS=1`, off by default in `script.py`'s cmake
  invocation) enables printing the number of resolved collisions per frame
  — this is a collision-COUNT sanity check, not a numerical-tolerance
  correctness gate against a reference trajectory. No physical
  conservation-law check (energy, momentum) was found in the source read.
- **Rendering path**: an optional image-output path (`render=1` by default
  in `script.py`) writes PNG frames via a separate, explicitly-excluded-
  from-phase-timing code block (image writing happens after the 3 timed
  phases each substep, not inside the accumulated timers) — i.e. the
  harness's own structure already separates visualization I/O from the
  timed compute region.
- **Source**: `github.com/MDurgaKeerthi/Mochi-DCD-on-RT` — `README.md`,
  `src/s23-3Dparticlesim/script.py`, `src/s23-3Dparticlesim/hostCode.cu`.

---

## Related-but-excluded (context only, not surveyed as this track's papers)

These were found via keyword search of the whole corpus for
N-body/collision/neighbor-search-adjacent work, and are noted here because
they bear directly on why this track's spec has the shape it does — but none
qualifies as one of this track's papers (either no artifact, or classified
under a different slug by the Phase-1 pipeline):

- **RT-BarnesHut** (`conf/ppopp/NagarajanGSP025`, PPoPP 2025) — "Accelerating
  Barnes-Hut Using Ray-Tracing Hardware." This is the genuinely classical
  N-body paper in this corpus: Barnes-Hut is the textbook tree-code
  multipole-approximation long-range force method the parent task's
  "accuracy parameter (multipole order)" axis directly describes.
  `artifact_status: none` — no public artifact found, so it could not
  anchor this track. Its existence in the corpus (found, but unusable)
  is the single strongest piece of evidence that this track's true
  "classical N-body" literature is present but not reproducible from this
  corpus's artifacts.
- **RTNN** (`conf/ppopp/000122`, PPoPP 2022) — "accelerating neighbor
  search using hardware ray tracing." Has a verified artifact, but was
  classified by the Phase-1 pipeline under `other:ann-search` (aliased to
  `other:ann-search`), not `nbody` — it targets approximate-nearest-neighbor
  workloads (general ML/data-mining kNN), not a physics particle system,
  and is out of scope for this track's own paper-selection boundary (this
  survey does not second-guess that classification).
- **RT-DBSCAN** (`conf/ipps/NagarajanK23`) and **OCTANE**
  (`conf/ics/ToutouniCTH26`) — RT-hardware-accelerated DBSCAN clustering and
  MD neighbor-list construction respectively; both `artifact_status: none`.
  OCTANE in particular ("Breaking the Neighbor-List Bottleneck in GPU
  Molecular Dynamics") is the closest analogue to Mochi's own problem
  (RT-accelerated fixed-radius neighbor search) but for MD instead of rigid-
  body collision — no artifact, so also unusable as an anchor.

---

## Divergences

Only 1 paper is available, so there is no cross-paper divergence in the
usual sense of this survey template. The one substantive divergence is
between this track's ACTUAL available paper and the parent task's ASSUMED
axes:

1. **"Accuracy parameter (multipole order / mesh resolution)" does not apply
   to Mochi.** Mochi's collision detection is exact/exhaustive within the
   RT-hardware's own BVH traversal (every actual sphere-sphere intersection
   within the query radius is found, not an approximation controlled by a
   tunable order parameter) — there is no multipole expansion or
   particle-mesh grid in this paper at all. The spec below therefore has
   no "accuracy parameter" axis for its primary variant; this is flagged as
   a structural mismatch with the parent task's suggested axis list, not
   silently worked around by inventing one.

2. **"Energy-conservation correctness" does not apply either** — no
   conservation-law check was found in the read source (only a raw
   collision-count print, gated behind a disabled-by-default compile flag).
   The spec's correctness gate is therefore this survey's own
   recommendation (trajectory agreement vs. a reference/exhaustive O(n^2)
   collision check on a small particle count), explicitly flagged as not
   sourced from the paper.

3. **"Interactions/s" DOES apply, but is not the paper's own reported
   metric** — the paper's own artifact reports 3 separate phase times
   (build/detect/update) and a raw collision count, not a normalized
   interactions/s or collisions/s throughput figure. This spec derives an
   interactions/s-equivalent metric from the confirmed per-phase timings
   rather than assuming the paper reports one directly.
