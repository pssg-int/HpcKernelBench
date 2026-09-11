# SpMSpV track — evaluation survey

Track input: `data/track_inputs/spmspv.json` (2 papers, both surveyed via
fulltext).

## conf/ics/ElbekK26 — BLEST: Blazingly Efficient BFS using Tensor Cores (ICS 2026)

Source: arXiv 2512.21967 fulltext (`arxiv.org/html/2512.21967`) +
GitHub `delbek/blest` (`Benchmark.cuh`, `SuiteSparseMatrixDownloader.hpp`,
`OutputVerifier.cuh`, `main.cu`).

- **Operation modeled**: pull-based BFS reformulated as a sequence of
  bitmap-oriented SpMSpV calls — the frontier IS the sparse input/output
  vector, one SpMSpV call per BFS level, over Binarised Virtual Slice Sets
  (BVSS), a bitmap-packed adjacency layout (not general-value CSR).
- **Workloads/inputs**: two suites. (1) GAP Benchmark Suite: GAP-road (23M v /
  57M e), GAP-twitter (61M v / 1.4B e), GAP-web (50M v / 1.9B e), GAP-kron
  (134M v / 4.2B e), GAP-urand (134M v / 4.2B e). (2) a custom SuiteSparse
  selection with explicit inclusion criterion `|V| >= 23M AND |E| <= 2^32-1`:
  nlpkkt240, uk-2005, it-2004, europe_osm, com-Friendster, Spielman_k600,
  webbase-2001, kmer_V1r, mawi. Real-world graphs only (social, road, web,
  geometric); the repo's `SuiteSparseMatrixDownloader.hpp` implements the
  filter and download programmatically.
- **Timing protocol**: 64 runs per graph ("averaged over 64 runs"), random
  source roots. The repo's `Benchmark::constructGreatSourceVertices` does NOT
  pick uniformly-random roots for its reported numbers — it iteratively
  resamples roots (5 rounds, rejecting roots whose single-run time deviates
  >20% from the running average) to converge on a set of roots with
  *homogeneous* per-root runtime. This is a real fairness concern: it
  actively excludes atypically-fast or atypically-slow roots from the
  reported average, which is a stronger and less disclosed selection effect
  than any convention flagged in this repo's `bfs/spec.yaml` survey
  (Corder's single max-degree root is disclosed as biased; BLEST's root
  *filtering* is not disclosed as such in the abstract/eval text we could
  retrieve).
- **Timing scope**: kernel execution (frontier propagation) only. BVSS
  construction time and graph-reordering preprocessing are measured and
  reported *separately* (Table 1(a) per the fetched summary), not folded
  into the per-search number.
- **Precision & correctness**: no floating point — Boolean/bitwise semiring
  (bit-packed adjacency, frontier membership is a single bit). The repo
  ships a dedicated `OutputVerifier.cuh`, implying an automated
  parent/level correctness check exists, but the exact validation level
  (BFS-tree edge validity vs. simple reached-set match) could not be
  confirmed from the fetched text alone.
- **Metric**: milliseconds only (no GTEPS reported) plus geometric-mean
  speedup vs. baselines. This matches the pattern already recorded for
  BLEST in this repo's `bfs/spec.yaml` evidence block ("BLEST reports zero
  GTEPS numbers (ms only)").
- **Baselines**: BerryBees (prior Tensor-Core BFS, primary comparison,
  3.58x-3.63x speedup reported), Gunrock, GSWITCH, GAP (CPU
  direction-optimizing reference).
- **SpMSpV-specific fact (frontier/vector sparsity)**: the paper explicitly
  reports frontier size varying enormously within a single graph's BFS —
  e.g. GAP-twitter (61.6M vertices) has as few as ~2.33M frontier vertices
  active on its densest level across a 15-level BFS-tree depth, meaning the
  *input vector* to each per-level SpMSpV call ranges from extremely sparse
  (single-digit-percent density near the first/last levels) to a
  double-digit-percent-density "bulge" mid-traversal — the textbook
  small-world frontier shape. This is real, workload-driven, non-uniform
  vector-sparsity variation, distinct from a synthetic sweep.

## journals/tpds/LiAY21 — Adaptive SpMV/SpMSpV on GPUs for Input Vectors of Varied Sparsity (TPDS 2021)

Source: arXiv 2006.16767 fulltext (via ar5iv.labs.arxiv.org/html/2006.16767)
+ GitHub `hpc-research-2020/spmspv-adaptive` (README only; the `hice-spmspv`
kernel/model source tree is present but its bench scripts were not
individually inspected beyond the top-level README's run instructions).

- **Operation modeled**: general SpMV/SpMSpV with an 8-kernel candidate pool
  and a machine-learned runtime kernel *selector* keyed on matrix and
  **input-vector sparsity**. This is the track's most direct match to
  "vector sparsity sweep is THE parameter."
- **Workloads/inputs**: 2,010 SuiteSparse matrices used purely as ML-model
  *training* data (not part of the reported benchmark numbers), plus an
  18-matrix held-out test suite spanning two regimes the paper explicitly
  separates — 9 "high-diameter" (belgium_osm, G3_circuit, roadNet-CA,
  hugetrace-00020, delaunay_n24, dielFilterV3real, rgg_n_2_24_s0, road_usa,
  and a duplicate hugetrace-00020 entry in the fetched summary that is
  likely a fetch artifact) and 9 "low-diameter" (soc-LiveJournal1,
  ljournal-2008, kron_g500-logn21, wikipedia-20070206, hollywood-2009,
  flickr, soc-orkut, indochina-2004, wb-edu). Sizes ~0.8M-23.9M rows,
  ~9.8M-212M nonzeros.
- **Vector sparsity sweep (the key axis)**: for the synthetic/microbenchmark
  results, input-vector density is swept directly — "the number of nonzeros
  of the input vector varying from 1 to n with a uniform interval," i.e. a
  full density ramp from near-empty to fully dense on each fixed matrix.
  For the two real-application results (BFS and PageRank), sparsity is
  NOT synthetically controlled but emerges from the algorithm itself: BFS
  iteration counts range 11-2,803 depending on graph diameter (frontier
  density varies naturally per level, same phenomenon BLEST exploits
  directly), PageRank iteration counts range 27-288 (the PageRank vector is
  effectively dense from iteration 1 onward, so this exercises the
  framework's SpMV-heavy regime rather than SpMSpV proper).
- **Timing protocol**: 10 repeated runs, averaged (not median). GPUs: NVIDIA
  Tesla K40m, P100, V100.
- **Timing scope / overhead accounting**: the framework's own runtime kernel
  selection overhead (feature computation ~1.2-1.7%, vector format
  conversion ~0.7-1.1%, ML model evaluation ~3.3-4.4%, totaling <14%, avg.
  5.2-7.2%) is explicitly measured PER-CALL and INCLUDED in every reported
  number — the paper does not amortize this cost away or report a
  selector-overhead-excluded number anywhere. This is a stronger, more
  transparent practice than most SpMV papers surveyed for
  `spmv/spec.yaml` (several of which never surface conversion/search cost
  at all).
- **Precision**: single precision (fp32) throughout.
- **8 candidate kernels**, organized along three orthogonal axes the paper
  defines: (1) computing pattern — column-major SpMSpV (vector-driven,
  column access), row-major SpMSpV (matrix-driven, row access, with
  validation), or traditional row-major SpMV (matrix-driven, no
  validation); (2) workload distribution — direct one-row/column-per-thread
  vs. load-balanced (preprocessing redistributes nonzeros evenly); (3)
  write-back strategy — atomic accumulation vs. sort-based (three-kernel
  global-memory staging pipeline).
- **Baselines**: cuSPARSE, Merge-based SpMV, Hola SpMV (fixed single-kernel
  SpMV baselines); a fixed column-major SpMSpV kernel from prior work; a
  push-pull adaptive library that only switches between SpMV and SpMSpV
  (coarser-grained than this paper's 8-way selection).
- **Metric**: speedup factor, ML-selector prediction accuracy (85-91%),
  performance loss under misprediction (1.3-8.8%) — this last number is a
  useful, rarely-reported robustness metric: how much throughput is lost
  when the adaptive framework picks the *wrong* kernel, not just how well
  it does on average.
- **Correctness**: not confirmed from the fetched text — no explicit
  tolerance or reference-comparison method could be extracted; flagged as
  an open question below.

## Divergences

- **What "the vector" is** differs fundamentally between the two papers.
  BLEST's vector is always the BFS frontier — boolean, level-synchronous,
  and its sparsity trajectory is dictated by graph topology, not chosen by
  the experimenter. LiAY21's vector is a free experimental parameter,
  explicitly swept from empty to full on the SAME fixed matrix to isolate
  the sparsity axis in a controlled way. Neither is more "correct"; they
  answer different questions (synthetic sensitivity curve vs. realistic
  workload trace), which is why the spec below keeps them as separate
  variants rather than picking one sweep protocol.
- **Timer/repetition count**: BLEST uses 64 runs (matching the Graph500/BFS
  convention this repo's `bfs/spec.yaml` also standardizes on); LiAY21 uses
  10 runs averaged (no median/min/max reported, same "mean-only" pattern
  flagged as a weakness across most of the SpMV track's papers in
  `spmv/spec.yaml`).
- **Metric**: BLEST reports only ms + speedup (no GTEPS, no GFLOP/s — its
  operation is boolean, not arithmetic); LiAY21 reports speedup + a
  selector-accuracy metric that has no equivalent in BLEST's evaluation.
  There is no shared throughput unit between the two papers' own reported
  numbers, which is why the spec below defines its own metric (GNZ/s —
  processed nonzeros per second, precision-and-semiring-agnostic) as the
  common denominator.
- **Kernel-selection overhead disclosure**: LiAY21 explicitly measures and
  includes its own selection overhead per call; BLEST's BVSS construction
  is one-shot per graph and reported separately, never per-call. These are
  different kinds of "extra cost" (per-call adaptive overhead vs. one-shot
  structural preprocessing) and the spec keeps both disclosure
  requirements rather than conflating them.
- **Root/vector-instance selection bias**: BLEST's `constructGreatSourceVertices`
  filters candidate BFS roots to a homogeneous-runtime subset before
  reporting — an undisclosed selection effect not present in LiAY21's
  synthetic density sweep (which has no such per-instance selection since
  density itself is the swept variable, not an instance-selection choice).

## Open questions carried into spec.yaml

- BLEST's exact correctness/validation criterion (what `OutputVerifier.cuh`
  actually checks) could not be confirmed beyond "a verifier exists."
- LiAY21's correctness/validation method could not be confirmed from the
  fetched fulltext at all.
- LiAY21's real-application (BFS/PageRank) numbers were not independently
  re-checked against the `bfs`/`pagerank` tracks' own surveyed protocols in
  this repo; if those specs are revisited, cross-checking the iteration
  counts (11-2,803 for BFS; 27-288 for PageRank) against the graphs used
  here would be worthwhile.
