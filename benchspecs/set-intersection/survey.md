# Survey — set-intersection (singleton track: 1 paper)

## conf/ipps/AljundiAK22 — Degree-Aware Kernels for Computing Jaccard Weights on GPUs (IPDPS 2022)

- **What is measured**: per-edge Jaccard-weight computation on a graph — for every edge (u,v),
  |N(u) ∩ N(v)| / |N(u) ∪ N(v)|, i.e. a sorted-adjacency-list SET INTERSECTION (plus union-size
  bookkeeping) repeated once per edge. The paper's core contribution is a **degree-aware binning**
  scheme: edges are bucketed by the (min or max) degree of their endpoints into ranges, and each
  bin is processed by a kernel configuration tuned to that degree range — search **group size g**
  (threads cooperating on one intersection) and **assembly size a** (number of groups working on one
  vertex's neighbor list, i.e. number of concurrent (u,v) pairs per vertex) — with an ML (random
  forest) model predicting good (g,a) per bin from the bin's degree statistics, replacing manual
  tuning.
- **workloads/inputs**: real-world graph edge lists (format: whitespace-separated vertex-id pairs
  per line, `-i` flag); the repo does not bundle a named graph suite in the top-level listing
  fetched — `-e` takes a JSON experiment file (`parameters/experiment.json` sample) that defines the
  `ranges` (degree bin boundaries) and the (g,a) sweep per bin. The paper's claimed workload
  characteristic driving the contribution is **degree skew / structural variance** (sparsity
  pattern + degree imbalance) within a graph — i.e. the input suite must include graphs with a
  genuinely skewed degree distribution (power-law-like, as in most real social/web graphs), not
  synthetic uniform-degree graphs, since uniform degree is exactly the case where naive
  fixed-(g,a) kernels do not lose to the degree-aware approach.
  Both **directed and undirected** graph handling are supported (`_DIRECTED` CMake option, default
  OFF/undirected).
- **timing protocol**: the executable's `-a` flag sets "number of averages" — each configured
  experiment (bin, g, a) is run `-a` times and the reported timing is the **average** (mean, not
  median) over those runs; no warmup iterations are mentioned separately from the averaged runs
  (implying either no warmup, or warmup is folded into the first of the `-a` runs and biases the
  mean).
- **timing scope**: kernel execution time for the FULL edge set's Jaccard computation, per (degree-
  bin, g, a) configuration — the paper explicitly separates the algorithm into "binning" (assigning
  edges to degree bins, a preprocessing/sort-like step) and the per-bin kernel launches; whether
  binning time is included in the reported kernel time is not stated in the fetched README.
- **precision & correctness**: Jaccard weights are stored in `{INPUT_GRAPH}.corr.bin`, a binary
  array where the i-th element is the Jaccard weight of the i-th edge in CSR order — this appears
  to be the artifact's own output-dump mechanism rather than an automated correctness check; no
  stated tolerance for comparing against a reference implementation was found.
- **metric**: kernel wall time (from which speedup is derived); the paper reports up to 35x, average
  12x speedup vs. "the state of the art" (an existing GPU Jaccard-weight kernel, presumably
  cuGraph's or a fixed-(g,a) predecessor kernel — both are compile options in this same repo,
  `_CUGRAPH` and `_SIMPLE_GPU`/`_SIMPLE_GPU_EDGE`), and states that manually tweaking load
  distribution can make "a state-of-the-art implementation... 5x faster", establishing that a large
  share of the speedup ceiling is about work-distribution tuning rather than a fundamentally
  different intersection algorithm.
- **baselines**: the repo itself compiles multiple baseline kernel variants alongside the
  degree-aware one: `_CUGRAPH` (NVIDIA cuGraph's Jaccard), `_INHOUSE_CUGRAPH`, `_SIMPLE_GPU` (naive
  one-thread(-team)-per-edge GPU kernel), `_SIMPLE_GPU_EDGE`, `_CPU` (a "traditional but effective"
  multi-core shared-memory algorithm, also proposed as a contribution and claimed "comparable" to
  the GPU algorithms), `_DONGARRA` (unclear from name alone — likely another named prior kernel).
- **source**: abstract (`output/included.json`); repo README
  (`gh api repos/SU-HPC/Jaccard-ML/contents/README.md`).

## Divergences

- Single-paper track: no cross-paper divergence. The main internal tension is that the paper
  benchmarks (a) a GPU degree-aware kernel with ML-predicted (g,a), (b) several GPU baseline
  kernels compiled from the SAME repo via CMake flags (apples-to-apples, same codebase), and (c) a
  separate CPU multi-core algorithm claimed "comparable" to the GPU kernels — these are
  structurally different enough (GPU tuned-parameter search vs. CPU fixed algorithm) that this
  spec keeps them as separate variants rather than one pooled number, per this project's general
  fairness principle of not conflating heterogeneous implementation strategies into a single ranked
  list.
