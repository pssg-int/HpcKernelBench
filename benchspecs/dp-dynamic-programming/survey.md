# Survey — dp-dynamic-programming (singleton track: 1 paper)

## conf/ics/YangLWHT23 — Fast All-Pairs Shortest Paths Algorithm in Large Sparse Graph (ICS 2023)

- **What it does**: solves the All-Pairs Shortest Paths (APSP) problem — the canonical
  dynamic-programming kernel `d(k)[i][j] = min(d(k-1)[i][j], d(k-1)[i][k] + d(k-1)[k][j])`
  (Floyd-Warshall recurrence) — for large *sparse* graphs by combining a divide-and-conquer
  graph partition (METIS) with a hybrid recurrence: dense local Floyd-Warshall DP tiles
  *within* each METIS-partitioned subgraph, combined with global Dijkstra (SSSP) sweeps
  *across* subgraph boundaries, avoiding the expensive global all-to-all communication a
  naive blocked-Floyd-Warshall APSP would require on a distributed multi-GPU cluster. This
  is a distributed-memory, multi-GPU (MPI + CUDA) DP algorithm, not a single-node/single-GPU
  tiled-DP kernel.
- **workloads/inputs** (confirmed from README + `app/graph/` + `app/scripts/run.sh`):
  the README states matrices were drawn from **"the Suite Sparse matrix collection and
  network repository"** (an explicit, if loosely quantified, dataset-suite claim — "a lot
  of sparse graph data", no exact count given in the fetched README). Concretely present in
  the repo: `luxembourg_osm.mtx` (a real road-network graph, unweighted undirected, shipped
  as the README's own worked usage example, `-k 128`) and `delaunay_n16` (a synthetic
  Delaunay-triangulation mesh graph, referenced by name in the SLURM launch script
  `app/scripts/run.sh`, `-k 8`, run on 1 node / 4 DCU accelerators). The paper's own
  headline large-scale result (from the abstract) is a graph with **11,548,845 vertices**
  solved in ~12.45 minutes on 2048 GPUs — the specific named dataset for this run was not
  recoverable from the fetched sources (not in the repo's shipped `graph/` examples).
  Both directed/undirected and weighted/unweighted graph variants are supported CLI flags
  (`-direct`, `-weight`), and the codebase's own directory naming
  (`weight-directed`/`weight-undirected`/`unweight-directed`/`unweight-undirected`) makes
  this a first-class 2x2 axis, not an afterthought.
- **timing protocol**: NOT recoverable from the sources fetched (`fap/fap.cpp`, the core
  solve-path implementation, contains no `time`/`clock`/`MPI_Wtime` calls in the ~280 lines
  read — timing is evidently done by an external wrapper/script not present in the surveyed
  portion of the repo, or the SLURM job's own stdout redirection (`#SBATCH -o
  out_test_20210514`) capturing whatever the binary itself prints, which was not inspected
  further). This is a genuine data gap, not a disagreement to reconcile.
- **precision & correctness** (confirmed from `fap/utils/checkResult.h`): a **sampled
  spot-check**, not full-APSP verification — `check_ans()` randomly selects 20 source
  vertices from the batch just solved and, for each, 10 random destination vertices (200
  `(source, dest)` pairs total), and recomputes the true shortest distance via a reference
  Dijkstra call (`dijkstra()` from `fap/kernel/batch_sssp.h`) for comparison. This is an
  explicit, code-confirmed correctness mechanism (unusually well-documented among this
  project's singleton tracks) but it is sampling-based, not exhaustive — it would not catch
  an error confined to a small fraction of the vertex pairs outside the 200 sampled.
- **metric**: from the abstract — average speedup 16.97x vs CPU Dijkstra, 7.09x vs GPU
  Dijkstra, 7.09x vs the "Part APSP" algorithm (the paper's own named prior
  state-of-the-art distributed APSP baseline), 4.6x vs a "decentralized Part APSP"
  variant; and a scaling result (11.5M-vertex graph, 2048 GPUs, ~12.45 min wall time).
  No GFLOP/s or edges/s throughput metric is used — the paper reports wall-clock time and
  speedup-vs-baseline directly, consistent with APSP's irregular (graph-structure-dependent)
  work per vertex making a normalized flop-rate metric less standard than for dense DP.
- **baselines**: CPU Dijkstra (run per-source, serial reference), GPU Dijkstra
  (per-source, parallel), "Part APSP" (the paper's own named prior distributed-GPU APSP
  algorithm — a centralized graph-partitioning approach this paper explicitly improves on
  by eliminating its global communication step), and a decentralized variant of Part APSP.
- **source**: `github.com/Liu-xiandong/FastAPSP` — `README.md` (algorithm description,
  dependencies: METIS 5.1.0, CUDA 10.1, OpenMPI 4.0.2, g++), `app/singleNodeExample.cpp`
  (CLI usage, `check_result` call site), `app/scripts/run.sh` (SLURM launch, dataset name,
  GPU count), `app/graph/unweight-undirected/` (directory listing — one shipped example
  matrix), `fap/utils/checkResult.h` (the sampled Dijkstra spot-check implementation),
  `fap/fap.cpp` (grepped for timing calls — none found).

## Divergences

- Single-paper track: no cross-paper divergence in evaluation methodology. The main gap is
  a **data gap**: the artifact's actual timing protocol (what is timed, warmup/repeat
  count, statistic, timer) is not recoverable from the sources fetched — `fap.cpp` itself
  contains no timing instrumentation, meaning timing must live in an external
  driver/analysis script not present in the surveyed portion of the repo. This spec's
  protocol section is therefore specified from this project's general convention rather
  than lifted from the artifact, flagged in `open_questions`.
- The paper spans a 2x2 graph-property axis (directed/undirected x weighted/unweighted)
  that is structural to the codebase (separate example directories per combination) rather
  than a tunable parameter of one algorithm; this spec treats it as a required sub-case
  matrix rather than folding it into a single default configuration.
- The correctness mechanism found (`check_ans`, a 200-pair random spot-check against
  reference Dijkstra) is real and code-confirmed, but is weaker than exhaustive
  verification; this spec keeps the spot-check as the practical default (following the
  artifact's own convention, and because it is quite well-documented compared to most
  singleton-track artifacts) but flags full verification on small instances as a stronger
  alternative in `notes_on_fairness`.
