# Survey: bipartite-matching (singleton track)

Track has exactly 1 paper: `conf/ipps/AkathoottB25`, "A Bidirectional GPU
Algorithm for Computing Maximum Matchings in Bipartite Graphs" (Akathoott,
Burtscher, IPDPS 2025). Artifact: `github.com/burtscher/ECL-MM` (official,
verified, "ECL-MM v1.0").

## conf/ipps/AkathoottB25 — ECL-MM

- **Algorithm/scope**: exact maximum bipartite matching via augmenting
  paths. Phase 1: fast deterministic initial-maximal-matching construction
  (no path search). Phase 2: parallel level-synchronous BIDIRECTIONAL BFS
  from unmatched vertices on BOTH partitions simultaneously (the paper's
  central contribution — halves search-tree depth vs. single-directional
  augmenting-path search, reduces synchronization and path overlap).
  Thread/warp granularity adapts to vertex degree for load balancing.
- **Hardware**: RTX-4090 GPU (from abstract: "Our results from an
  RTX-4090 GPU show...").
- **Workloads/inputs** (README lists the EXACT input set used in the
  paper, 21 named instances, format = Matrix Market from
  sparse.tamu.edu / SuiteSparse, converted via bundled `mm2eclbp.cpp` to
  binary CSR `.egr`):
  `amazon0312, r4-2e23, uk-2002, as-Skitter, hugebubbles-00000,
  rgg_n_2_24_s0, uk-2005, cit-Patents, hugetrace-00020, rmat22, wb-edu,
  coPapersDBLP, in-2004, roadNet-CA, web-Google, delaunay_n24, kkt_power,
  road_usa, wikipedia-20070206, europe_osm, kron_g500-logn21,
  soc-LiveJournal1`. This spans web graphs, road networks, citation/social
  networks, synthetic RMAT/random-geometric graphs, and delaunay
  triangulations — a broad, non-cherry-picked mix by construction (matches
  this project's fairness principle of a union/standard suite, here
  already satisfied by the paper's own choice).
- **Timing protocol**: not spelled out step-by-step in the README (no
  worked timing-output example, unlike graphB+ in the same lab). The
  source `ECL-MM_10.cu` (grepped) shows a `CPUTimer`/`GPUTimer`-style
  struct and two explicitly separated, printed phases:
  `initRuntime` (phase-1 initial matching) and `apSearchTime`
  (phase-2 augmenting-path/bidirectional-BFS search), summed into
  `totalRunTime`, plus `overall_throughput` and
  `apSearchPhase_throughput` (both in edges/s) — i.e. the code itself
  ALREADY separates the two algorithmic phases' timing and reports a
  phase-2-only throughput distinct from the overall throughput, giving
  this spec a natural two-phase variant split without having to invent
  one.
- **Precision & correctness**: matching output is a vertex-to-vertex
  pairing (int array); maximum bipartite matching is an EXACT problem —
  correctness = matching is valid (each matched vertex has exactly one
  partner, edges exist in the original graph) AND has maximum cardinality
  (verifiable via Berge's lemma: no augmenting path remains, or by
  comparing to a reference max-flow/Hopcroft-Karp computation on
  small/medium instances where an independent solver is feasible).
- **Metric**: `overall_throughput`/`apSearchPhase_throughput` in edges/s,
  plus wall time per phase; headline abstract numbers are SPEEDUP vs.
  baselines: 4.5x vs. "fastest prior multicore CPU code", and vs.
  "fastest prior GPU code" (the exact factor was truncated by an inline
  MathML tag in the abstract text extraction — visible only as "a factor
  of" with the number lost to markup; treated as an open question below).
- **Baselines**: explicitly named class of comparisons — fastest prior
  multicore CPU maximum-bipartite-matching code, and fastest prior GPU
  code (name not resolved from the abstract excerpt available; likely a
  predecessor from the same Burtscher lab given the ECL- naming
  convention, e.g. an earlier ECL-series matching kernel, or a
  third-party GPU matching implementation — flagged as open question).

Source: `burtscher/ECL-MM` README.md, `ECL-MM_10.cu` (grep for
Timer/runtime/throughput patterns), abstract (`output/included.json`).

## Divergences

Single-paper track: no cross-paper divergence. The paper's own code already
does the phase-separated timing this project's fairness principles ask
for (initRuntime vs. apSearchTime, never silently blended) — this spec
follows that split directly rather than imposing an external one.
