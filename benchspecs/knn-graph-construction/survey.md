# knn-graph-construction — survey

Singleton track: 1 paper.

## Ranawaka, Rahman & Azad, "Distributed Sparse Random Projection Trees for
Constructing K-Nearest Neighbor Graphs", IPDPS 2023
(`conf/ipps/RanawakaRA23`)

- **Source**: GitHub README (`HipGraph/DRPT`, fetched via `gh api`),
  repo directory listing (`cpp/`, `tests/datasets/`, `tests/scripts/`),
  abstract. No arXiv preprint found (checked `data/artifacts_found.json`
  and WebSearch — only IEEE Xplore / computer.org links surfaced); IPDPS
  is generally not open-access, so fulltext was not available.
- **Workloads**: distributed-memory (MPI+OpenMP) construction of an
  approximate k-nearest-neighbor graph (KNNG) from high-dimensional
  points, via a forest of sparse random-projection trees. The only
  dataset confirmed directly from the repo is **MNIST**
  (`tests/datasets/train-images-idx3-ubyte.gz`, 60,000 points, dimension
  784 — the exact usage example in the README uses `-data-set-size 60000
  -dimension 784 -ntrees 8 -nn 10`). The abstract states DRPT's accuracy
  is "comparable to state-of-the-art methods for approximate nearest
  neighbor search" and that it "runs two orders of magnitude faster than
  its peers" — implying comparison against standard ANN baselines
  (methodology consistent with the ANN-benchmarks tradition: FLANN,
  Annoy, HNSW, or similar tree/graph-based ANN methods), but the paper's
  own full dataset list (SIFT1M/GIST1M/GloVe/Fashion-MNIST etc., commonly
  used alongside MNIST in this literature) could not be confirmed from
  available sources.
- **Scale**: abstract states DRPT "scale[s] KNN graph constructions to
  thousands of processes on a supercomputer"; the bundled sample SLURM
  script runs 4 nodes x 4 processes/node = 16 MPI ranks x 8 OpenMP
  threads/rank on a Cori-Haswell-class partition — i.e. this is
  fundamentally a distributed-memory HPC kernel, not a single-node/GPU
  kernel (matches the paper's `platform: distributed` tag).
- **Algorithm parameters**: number of trees (`-ntrees`, e.g. 8), number
  of neighbors (`-nn`, e.g. 10), a locality-based data-gathering toggle
  (`-locality`) that is itself a communication-reduction optimization
  (the paper's stated contribution), input format (ubyte or fbin).
- **Timing/accuracy protocol**: not directly visible in the README beyond
  the CLI parameters; no benchmark/timing script beyond the CMake build
  and the single MNIST example was found in the repo.
- **Metric**: implied from the abstract — construction throughput/time
  (headline claim: "two orders of magnitude faster than its peers") and
  KNNG accuracy vs. state-of-the-art ANN methods (implies a recall-style
  quality metric, standard for approximate KNN, though the paper's exact
  definition — e.g. recall@k against exact brute-force KNNG — was not
  confirmed from available sources).
- **Baselines**: unconfirmed by name from available sources; likely
  state-of-the-art distributed or parallel ANN/KNNG construction methods
  given the "two orders of magnitude faster than its peers" framing.

## Divergences

N/A — only one paper in this track. The main gap is that the paper's own
full evaluation dataset list and exact accuracy metric definition could
not be confirmed (IPDPS is paywalled, no arXiv mirror found); the spec
below uses the one dataset directly confirmed from the repo (MNIST) plus
the standard ANN-benchmarks corpus as a reasonable, disclosed stand-in,
flagged explicitly as an open question rather than presented as verified.
