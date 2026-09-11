# Survey: lca-bridge-finding (singleton track)

Track has exactly 1 paper: `conf/ipps/0001SS21`, "Euler Meets GPU: Practical
Graph Algorithms with Theoretical Guarantees" (IPDPS 2021). Artifact:
`github.com/stobis/euler-meets-cuda` (official, verified). arXiv:2103.15217
confirmed (PDF/HTML fetch attempts returned only the abstract page — the
survey below is grounded in the repo's own extensive test-harness scripts,
which is MORE precise than paper prose per this project's source-priority
rule ("timing loops in code reveal warmup/repetition/timer placement more
reliably than paper text")).

## conf/ipps/0001SS21 — Euler Meets GPU

- **Problem scope**: GPU adaptation of the classical Euler-tour technique
  (PRAM heritage) for TWO applications, benchmarked separately:
  (1) **LCA** — lowest common ancestor queries on trees, and
  (2) **bridges** — bridge-finding in undirected graphs. The paper's own
  framing (from abstract) explicitly compares "theoretically optimal"
  Euler-tour-based algorithms against "simpler heuristics" on both HARD
  (adversarial/deep) and EASY (typical) instances — i.e. the benchmark
  design itself is built around a hard/easy instance-difficulty axis, not
  just size.
- **Algorithms compared** (from `test/lca/testVariables.sh`
  `validitySolutionsToTest` / `E1SolutionsToTest` arrays):
  `cuda-inlabel` (the paper's Euler-tour/in-label GPU method),
  `cuda-naive` (GPU heuristic baseline), `cpu-inlabel` (single-thread CPU
  reference of the same algorithm), `multicore-cpu-inlabel` (parallel CPU
  baseline). A `cpu-rmq` variant exists too (sparse-table RMQ, the
  textbook-optimal CPU LCA algorithm) but is used ONLY as the
  ground-truth-answer GENERATOR for correctness checking
  (`validityOutGeneratorAlgorithm="cpu-rmq"`), not as a timed competitor.
- **Hardware**: not stated in the scripts inspected (GPU model/CUDA
  compute-capability macro `NVCCSM` is a user-set Makefile variable, not
  hard-coded) — open question below.
- **Workloads/inputs** (from `test/lca/testVariables.sh`, exact and
  extensive): synthetic generated trees/graphs via `generateSimple.cpp`
  and `generateScaleFree.cpp`. Validity-test sizes sweep from n=5 up to
  n=30,000,000 (`validityTestsSizes` array: 5, 10, 1000, 2000, ..., 9000,
  100000, 500000, ..., 30000000). The main timed experiments (E1-E5) use
  sizes 1M-32M nodes with a "grasp" parameter (`E1GraspSizes = (-1, 1000)`
  — "how far up a father can be", i.e. controls tree depth/shape: -1 =
  shallow/wide, 1000 = deeper/bushier), 5 different random seeds per
  configuration (`E1DifferentSeeds=5`), and a dedicated batch-size sweep
  (E2: query-batch sizes 1 through 10,000,000) and grasp-size sweep (E3:
  1 through 10,000,000) isolating individual algorithmic parameters.
- **Timing protocol**:
  - LCA (`test/lca/testTime.sh` + `testVariables.sh`): for each
    experiment E1-E5, tests are generated once per (size, grasp, seed)
    combination, THEN answers are pre-generated once via the `cpu-rmq`
    reference (`CheckAnswers=true` for all 5 experiments) BEFORE any timed
    run — i.e. correctness-reference computation is explicitly excluded
    from the timing loop by construction of the script's control flow.
    `singleRunTimeout=120` (seconds) caps any single run.
  - Bridges (`bridges_test.sh`): explicit CLI flags `-r <times>` ("Repeat
    tests `<times>` times and get average results. Default value is 1.")
    and `-t <outfile>` for timed runs; `run_experiments.sh all $repeats
    $outFile` is the actual driver. **Statistic is explicitly documented
    as the AVERAGE (mean) of `repeats` runs**, default repeats=1 (i.e. a
    single run unless the user overrides `-r`) — a materially different
    convention from LCA's approach and from most of this survey batch's
    median-based convention.
- **Precision & correctness**: BOTH LCA and bridges are EXACT graph
  problems (no approximation ratio). LCA correctness is validated by
  comparing every implementation's answers against the `cpu-rmq`
  reference-generated ground truth (`CheckAnswers=true`); bridges
  presumably has an analogous exact check though `bridges_test.sh`'s help
  text notably says "TODO do we want to have some tests to check
  correctness?" — i.e. **the paper's own bridges harness does NOT
  guarantee an automated correctness check is wired in by default**
  (flagged as a real gap, not assumed away).
- **Metric**: wall time per run; `stats2csv.py`/`combine_csv.py`/`plot.py`
  post-process raw timing CSVs into the paper's figures. No
  throughput/edges-per-second framing is visible — pure wall-clock time
  as a function of the swept parameter (size, batch size, or grasp size).
- **Baselines**: `cuda-naive` (GPU heuristic), `cpu-inlabel` (serial CPU),
  `multicore-cpu-inlabel` (parallel CPU) — all timed competitors; `cpu-rmq`
  is the ground-truth generator, explicitly NOT a timed competitor for LCA
  (its own comment marks it as excluded: `# "cpu-rmq"` is commented out of
  the E1 solutions-to-test list).

Source: `stobis/euler-meets-cuda` `test/lca/testVariables.sh`,
`test/lca/testTime.sh`, `bridges_test.sh`, abstract
(`output/included.json`), arXiv:2103.15217 (abstract page only, fulltext
not machine-readable via available tools).

## Divergences

Single-paper track: no cross-paper divergence, but a significant INTERNAL
divergence between the paper's own two sub-benchmarks: LCA uses a rich,
scripted multi-seed sweep with an explicit pre-generated ground-truth
correctness check wired into every experiment; bridges uses a much
simpler `-r`-repeats-mean-of-N harness with correctness checking
explicitly marked as a TODO in the script's own help text. This spec keeps
the two as separate variants rather than forcing one convention onto both,
and flags the bridges correctness gap rather than silently assuming
parity with LCA's rigor.
