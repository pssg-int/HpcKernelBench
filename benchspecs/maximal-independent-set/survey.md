# Survey: maximal-independent-set (singleton track)

Track has exactly 1 paper: `conf/ics/AkathoottXB25`, "A Multi-GPU Algorithm
for Computing Maximal Independent Sets in Large Graphs" (Akathoott, Xavier,
Burtscher, ICS 2025). Artifact: `github.com/burtscher/MG-MIS` (official,
verified — matches the "same-group, own-repo" pattern seen across several
Burtscher-lab entries in this batch).

## conf/ics/AkathoottXB25 — MG-MIS

- **Algorithm/scope**: single-node, multi-GPU (CUDA + OpenMP host
  orchestration) MIS computation for graphs too large for one GPU's global
  memory but small enough for the combined memory of all GPUs in a node.
  Splits computation into "local" and "remote" phases per GPU, uses
  bulk-only inter-GPU transfers, and avoids communicating priority values
  (per README summary).
- **Hardware** (from abstract): two systems — (a) 4x V100 GPUs, 32GB each
  (128GB combined), (b) 2x RTX 3080 GPUs, 12GB each (24GB combined). The
  repo's compile line targets `sm_70` (V100) by default.
- **Workloads/inputs** (from README "Input generation" section): graphs in
  binary CSR format (`ECLgraph`/`.egr`), generated two ways: (1) Graph500
  reference generator (`graph500/graph500`, R-MAT/Kronecker, parameterized
  by `<scale> <edgefactor>`) followed by a bundled `clean_graph500.cpp`
  dedup/self-loop-removal step, or (2) via the `burtscher/Indigo3Suite`
  companion benchmark suite's graph inputs/generators. No exact named
  real-world graph list is given in the README (contrast with ECL-MM in
  the same lab, whose README lists 21 named SuiteSparse/SNAP instances
  explicitly) — the paper's own size classes are only described relative
  to memory footprint: "more than 32 GB" (tested on the 4xV100 system) and
  "more than 12 GB" (tested on the 2xRTX3080 system), i.e. size classes are
  defined by exceeding a *single* GPU's memory, which is exactly the
  problem MG-MIS targets.
- **Timing protocol**: not directly visible from the README (no worked
  timing example given, unlike ECL-MM/graphB+ in the same lab). The CUDA
  source (`MG-MIS_10.cu`) uses a `CPUTimer` wrapping the main multi-GPU
  parallel region (`#pragma omp parallel num_threads(devices)`), printing
  `runtime`, `iterations`, and two throughput lines
  (`throughput: %.6f Mnodes/s`, `throughput: %.6f Medges/s`); a separate
  `verifyTimer` measures MIS-validity verification standalone, printed
  apart from `runtime` — i.e., the code itself keeps verification OUT of
  the timed region, matching this benchmark's fairness principle by
  construction. `prepTimer` similarly isolates graph-load + device-buffer
  setup from the timed region. No explicit multi-trial/warmup loop is
  visible in the source excerpt inspected (single measured run per
  invocation) — flagged as an open question below since the paper's
  reported numbers may still average external repeated invocations not
  visible in this single-run binary.
- **Precision & correctness**: MIS is a 0/1 membership vector; correctness
  = a valid MIS (no two adjacent vertices both selected, every excluded
  vertex has a selected neighbor) — verified in-code via the separate,
  untimed `verifyTimer` region. Quality (since MIS is not unique) is
  additionally reported: abstract states "the size of the MIS computed by
  MG-MIS is 2.6% smaller [on average]" than the single-GPU UVM baseline it
  compares against — i.e. **MIS cardinality is a joint quality metric**
  alongside speed, exactly the kind of approximate/heuristic-quality
  co-reporting this project's instructions require for non-exact problems
  (MIS computation is combinatorially non-unique; a trivial-but-useless
  "MIS" of size 1 would be fast and wrong-shaped without a size check).
- **Metric**: throughput (Mnodes/s, Medges/s) and wall time, PLUS geometric
  mean speedup over baseline (headline abstract numbers: 17.73x on the
  4xV100/>32GB set, 22.88x on the 2xRTX3080/>12GB set).
- **Baselines**: "the state-of-the-art single-GPU code with UVM" — i.e. the
  same lab's prior single-GPU MIS kernel (likely ECL-MIS or an Indigo3-style
  MIS implementation) run with CUDA Unified Virtual Memory to handle
  graphs that overflow one GPU's physical memory, allowing an
  apples-to-apples comparison against MG-MIS's explicit multi-GPU
  partitioning at the SAME graph sizes.

Source: `burtscher/MG-MIS` README.md, `MG-MIS_10.cu` (grep for
Timer/runtime/throughput patterns), abstract (`output/included.json`).

## Divergences

Single-paper track: no cross-paper divergence. The one methodological gap
worth flagging is internal to this paper's own artifact: the README does
not give a concrete named-graph list or an explicit repeated-trial/warmup
convention the way sibling Burtscher-lab repos (ECL-MM, graphB+) do in this
same survey batch — this spec compensates by defining size classes via the
memory-overflow threshold the paper itself uses as its organizing variable
(graphs that exceed one GPU's memory) rather than inventing named
instances the paper didn't disclose.
