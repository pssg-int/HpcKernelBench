# astar-search — survey

Singleton track: 1 paper.

## Al-Khansa et al., "Parallel Bidirectional A* Search for GPU-Accelerated
Pathfinding", ICS 2026 (`conf/ics/Al-KhansaGMH26`)

- **Source**: GitHub README (`HadiKhansaa/BBAStar`, fetched via `gh api`),
  `demo.sh`, `scripts/run_maps_benchmark.py` listing, abstract. No arXiv
  preprint found (checked `data/artifacts_found.json` and WebSearch); paper
  is ICS 2026 (very recent), ACM DOI only.
- **Workloads**:
  - MovingAI benchmark format: `.map` grid files paired with `.map.scen`
    scenario files (start/goal pairs) — the de-facto standard pathfinding
    benchmark corpus (game maps, city maps, road networks). Repo ships one
    sample (`data/maps/arena.map`); `scripts/run_maps_benchmark.py` runs
    additional user-supplied `.map`/`.map.scen` sets.
  - Synthetic procedural grids: 5 obstacle families — `random`, `maze`,
    `blockCenter`, `zigzag`, `rectangle` — generated at configurable size
    and obstacle rate (default 20%). `demo.sh`'s default suite spans sizes
    256, 512, 1024 across 4 of the 5 families. README states "the paper
    evaluates several synthetic grid families... beyond the bundled sample"
    at larger scale (grid growth referenced explicitly in the speedup
    discussion) — exact largest size not confirmed from available sources.
  - 8-connected movement, integer-scaled edge costs (`SCALE_FACTOR`,
    `DIAGONAL_COST` in `include/`).
- **Timing protocol**: `demo.sh --repeats N` runs each case N times
  (default 1) and reports a per-run table plus grouped summary statistics
  (by grid type, by grid size) — exact statistic (mean/median) not fully
  visible from truncated script output but a "repeats" flag exists,
  confirming the harness supports averaging over multiple trials.
- **Timing scope**: kernel run reports "runtime, expanded nodes, path
  cost" per case; compressed-grid loading (`data/generated/*.bin`) is a
  separate one-shot preprocessing artifact distinct from the search call.
- **Precision & correctness**: integer/fixed-point costs (no float
  tolerance issue). Optimality is the natural correctness notion: A* with
  an admissible heuristic must return the minimum-cost path. Repo ships
  two CPU reference baselines (`CPU/astarCPU.cpp`, `CPU/astarHeapCPU.cpp`)
  usable as an independent path-cost oracle.
  found: geomean 8.56x (peak 15.05x) over CPU baselines, geomean 11.91x
  (peak 43.33x) over prior GPU A* work (per WebSearch summary of the ACM
  abstract page — not independently confirmed from primary text).
- **Metric**: runtime (ms/s), nodes expanded, path cost; speedup vs CPU and
  vs prior GPU baselines.
- **Baselines**: single-thread/heap-based CPU A* (bundled), and an
  unconfirmed prior GPU A* implementation (paper text not accessible;
  massively-parallel-A*-on-GPU-style work is the likely reference class,
  e.g. Zhou & Zeng AAAI'20, but this specific comparison was not verified
  from primary source).

## Divergences

N/A — only one paper in this track, so there is no cross-paper
divergence. The two axes worth keeping distinct as separate variants are
(a) the standard external MovingAI corpus vs (b) the repo's own
parameterized procedural-grid generator, since they have different
selection/reproducibility properties (fixed named maps vs. a size/type/
obstacle-rate knob space).
