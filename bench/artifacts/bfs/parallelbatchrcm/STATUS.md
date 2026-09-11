# parallelbatchrcm (Speculative Parallel RCM) — bfs

**Status: SKIPPED — a graph REORDERING algorithm, not a BFS kernel; relevant only to the not-yet-implemented `bfs-reordering-cost` variant**

- Paper: "Speculative Parallel Reverse Cuthill-McKee Reordering on Multi- and
  Many-core Architectures" (IPDPS'21). `PAPER_KEY = conf/ipps/MlakarWPS21`.
  Artifact: https://github.com/GPUPeople/ParallelBatchRCM.
- Rated core for bfs (`output/kernel_centrality.json`) because
  `benchspecs/bfs/spec.yaml`'s `bfs-reordering-cost` variant names it (with
  Corder) as the anchor for "cost of a bandwidth-reducing reordering that
  speeds up downstream BFS/SpMV". The kernel it ships computes an RCM
  permutation (BFS is an internal step of RCM), so it cannot be gated by
  `kernelbench/domains/graph.py`'s BFS level-array gate.
- Integrate when/if the reordering-cost variant is implemented (gate: the
  output is a valid permutation and the resulting bandwidth is ≤ that of a
  reference RCM — a structural gate to be defined in the domain module).
