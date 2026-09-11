# ANN-search track — evaluation methodology survey

Track input: `data/track_inputs/other_ann-search.json`, 7 papers (all 7
surveyed — comfortably above the 4-paper minimum). 4 surveyed via arXiv
fulltext + artifact-repo scripts read with `gh api`; the other 3 (X-HD,
CLOVER, RTNN's repo-level confirmation) have no arXiv id in the track input
or a thin README, so their methodology comes primarily from the artifact
repo's benchmark/run scripts (RTNN additionally has an arXiv id found by
search and its fulltext was fetched).

**The track's own name ("ann-search") undersells what these 7 papers
actually are.** Only 4 (PathWeaver, DRIM-ANN, UpANNS, FANNS) are
high-dimensional *approximate* nearest-neighbor search over the classic
SIFT/DEEP/SPACEV benchmark family with a recall@k accuracy notion. The other
3 (CLOVER, RTNN, X-HD) are **exact** (or tightly error-bounded) low-
dimensional (2D/3D) spatial neighbor-search *primitives* used inside a larger
geometric computation (Hausdorff distance for X-HD), evaluated on completely
different data (point clouds, meshes, GIS polygons, MRI volumes) with no
recall@k concept at all, since exactness makes recall trivially 1.0. This
split is the single biggest methodological divergence in the track and drives
this spec's two-variant design (see Divergences and Step 2).

---

## 1. GengYLWZ26 — X-HD, "Fast Hausdorff Distance Computation with Ray Tracing" (ICS'26)

- key: `conf/ics/GengYLWZ26`, no arXiv id in track input (DOI-only OA link;
  not independently searched beyond the track input given the repo's scripts
  already provide concrete methodology).
- **workloads/inputs**: Hausdorff distance (a max-min point-set-distance
  metric) computed between pairs of real-world point sets from **3 domains**:
  (a) **MRI images** — BraTS2020 validation dataset (`.nii` volumes), 500
  randomly-seeded pairs sampled per run (`run_mri_datasets`), 3D; (b)
  **geospatial** — WKT polygon datasets, 3 paired comparisons (county
  boundaries vs. zip codes, water bodies vs. census block groups, lakes vs.
  parks — real U.S. GIS boundary data), 2D; (c) **graphics** — PLY mesh
  point clouds (`dragon`, `thai_statuette`, `asian_dragon`,
  `happy_buddha` — Stanford-3D-Scanning-Repository-style models), 3D, paired
  4 ways.
- **variants compared**: `eb` (early-break/brute-force), `nn` (a
  nearest-neighbor-library baseline), `itk` (ITK, a real biomedical-imaging
  toolkit — the "5.3x over ITK" one-liner's target), `rt` (X-HD's own
  RT-core algorithm); for the MRI and graphics domains a prior RT-core
  Hausdorff-distance competitor (`rt-hdist`) and **`clover`** (this same
  track's exact-kNN GPU library, `KamelYC25` below) are *also* run as
  baselines — the two papers in this track cross-reference each other.
- **timing protocol**: `-repeat 1` in `run_xhd()` (single measured run per
  config in the released reproducibility script; whether an internal
  multi-iteration loop exists inside `hd_exec` itself was not confirmed from
  the shell driver alone).
- **precision & correctness**: `hd_exec` exposes a `-check` flag, but the
  reproducibility script explicitly passes **`-check=false`** for every run —
  correctness verification exists in the binary but is disabled during the
  timed reproducibility sweep, an even more direct instance of "correctness
  exists but isn't run with the timed benchmark" than seen elsewhere in this
  project's surveys.
- **metric**: per-(dataset-pair, variant, execution) JSON log, presumably
  wall time (log schema not inspected in depth); Hausdorff distance is a
  single deterministic scalar, so there is **no recall@k or QPS concept
  here at all** — "accuracy" would mean "is the computed HD value correct,"
  which is exactly what the disabled `-check` flag would verify.
- **baselines**: ITK, CLOVER, `rt-hdist` (unnamed prior work), `eb`/`nn`
  internal variants.
- **hardware**: not confirmed from the fetched scripts (paths reference a
  private lab workstation); GPU is NVIDIA (OptiX/RT-core dependency implied
  by the CMake/vcpkg structure, not read in depth).
- source: repo `pwrliang/X-HD` — `README.md`, `expr/run_fig5.sh`,
  `expr/common.sh` (full files read).

## 2. KimPNHKLL25 — PathWeaver, "A High-Throughput Multi-GPU System for Graph-Based Approximate Nearest Neighbor Search" (ATC'25)

- key: `conf/usenix/KimPNHKLL25`, no arXiv id.
- **workloads/inputs**: built on **CAGRA** (via RAPIDS cuVS) graph-based
  ANNS. Dataset in the released single-GPU script: `sift-128-euclidean`
  (the standard `ann-benchmarks.com`-family SIFT dataset name, 128-dim,
  L2). Index built once with `build_graph_degree=64`,
  `build_intermediate_graph_degree=128`; PathWeaver's own optimization is a
  **two-phase "ghost" search**: a small subsampled "ghost" graph
  (`ghost_scale_factor=100`, i.e. 1% of the full dataset) is searched first
  (`ghost_topk=1`) to get a good entry point, then the full graph is
  searched from that entry point (`base_topk=10`) — this is
  PathWeaver's own "pipelining-based path extension" mechanism from the
  abstract, made concrete.
- **timing protocol**: `test_iteration = 4` (repeats per config, from
  `single_pathweaver_all.sh`'s `TEST_ITERATION` env var), reduces to **"Mean
  Recall" and "Median Execution Time"** (both terms literally printed by
  `single_pathweaver_one.py`) — i.e. recall is averaged but latency uses the
  median, a mixed-statistic convention.
- **precision & correctness / quality**: recall computed via
  `calculate_recall(results, ground_truth, base_topk)` against the
  dataset's own precomputed ground truth — **`base_topk = 10`, i.e.
  recall@10** is the fixed accuracy metric, reported alongside latency for
  every parameter combination in the swept grid (`ghost_max_iter`,
  `base_max_iter`, `base_neighbor_prune_ratio`, `base_iteration_prune_ratio`)
  — this IS a recall-vs-latency sweep in substance, though the released
  script does not itself plot it as a Pareto curve.
- **metric**: recall@10 (fraction, mean over `test_iteration` runs) +
  median execution time (ms) per query batch, for the single-GPU harness;
  the paper's own multi-GPU headline claim ("3.24x geomean over SOTA") is a
  throughput (QPS) claim not visible in the single-GPU script read here.
- **baselines**: not enumerated in the fetched files (built as a CAGRA/cuVS
  extension; the paper's comparison set per the one-liner is unspecified
  "SOTA" — not independently confirmed).
- **hardware**: 4 GPUs, >=48GB device memory each, NVLink bridge required,
  CUDA 12.1 — this is the only paper in the track whose baseline
  reproducibility requirement is inherently multi-GPU.
- source: repo `AIS-SNU/PathWeaver` — `README.md`,
  `pathweaver/single_pathweaver_all.sh` (full file read),
  `pathweaver/single_pathweaver_one.py` (grepped for dataset/recall/timing
  sections, not fully read line-by-line).

## 3. KamelYC25 — CLOVER, "A GPU-native, Spatio-graph-based Approach to Exact kNN" (ICS'25)

- key: `conf/ics/KamelYC25`, no arXiv id (ACM-OA PDF only).
- **workloads/inputs**: **exact** kNN over **low-dimensional (3D) synthetic
  spatial data** — `fill_with_random_data()` draws uniform points in
  `[-5, 5]^3` (`std::uniform_real_distribution`), N swept from 1,000 to
  1,000,000 (17 fixed sizes: 1K, 2K, 5K, 10K, 20K, 50K, 80K, 100K, 150K,
  200K, 350K, 500K, 600K, 700K, 800K, 900K, 1M), **k=30** fixed for the
  synthetic sweep; a separate mesh-based sweep loads real point-cloud files
  from a `../meshes` directory at **k=128**. Because the search is exact,
  there is no accuracy/recall dimension — correctness is binary
  (exact-match or not), not a continuous recall value.
- **algorithms compared** (in-process, same binary): `bitonic`
  (data-parallel batch insertion via bitonic merge), `warpwise`
  (warp-ballot insertion sort), `hubs`/`hubs_ws` (CLOVER's own
  spatio-graph-with-hubs-and-lower-bounds method, with/without WarpSelect),
  `faiss`/`faiss_ws`/`faiss_bs` (Facebook FAISS linear scan, plain /
  WarpSelect / BlockSelect), `treelogy_kdtree` (a GPU kd-tree library);
  external baseline directories `Arkade` (a modified third-party spatial
  index, run via a different CLI convention per the README's explicit note)
  and `rsll18`/`knn_sweet` (present but not read).
- **timing protocol**: `std::chrono::high_resolution_clock`, **`reps = 2`**
  for the synthetic sweep and **`reps = 1`** for the mesh sweep — the
  **loosest repetition count of any paper surveyed in either track of this
  task** (n=1-2, no median/mean/std computed anywhere in the driver; raw
  per-rep nanosecond values are simply printed). The README's own "Output
  Format Explanation" documents this directly: a "warm-up unit test" time is
  printed first, followed by the array of per-rep times.
- **precision & correctness**: `run_test()` runs a hand-computed 4-point,
  k=3 sanity check before the timed sweep — but its assertions are **dead
  code**: an unconditional `return true;` sits BEFORE the actual comparison
  statements, so the function always reports success regardless of whether
  the algorithm is correct. This is the weakest correctness enforcement
  found in either track surveyed for this task (worse than "disconnected" —
  the check itself cannot fail).
- **metric**: raw wall time (nanoseconds) per (N, k, algorithm) cell.
- **hardware**: NVIDIA V100 (per the abstract's own reported numbers: "On an
  NVIDIA V100... 2.71s... 4x faster than an optimised grid-based method...
  10x faster than a GPU tree... 230x faster than FAISS"); build
  instructions also reference an unspecified "CLOVER Workstation" GPU.
- source: repo `ampslab/clover-knn` — `README.md`, `src/linear-scans.cu`
  (full file read).

## 4. ChenH0LYDY000025 — DRIM-ANN, "An Approximate Nearest Neighbor Search Engine based on Commercial DRAM-PIMs" (SC'25)

- key: `conf/sc/ChenH0LYDY000025`, arXiv:2410.15621.
- **workloads/inputs**: 6 standard billion/100M-scale ANN datasets —
  **SIFT1B, DEEP1B, SPACEV1B** (10^9 vectors each) and **SIFT100M, DEEP100M,
  SPACEV100M** (10^8 vectors, used specifically for the GPU comparison due
  to GPU memory limits). Dimensionality: SIFT=128, DEEP=96, SPACEV=100.
  Query sets: 10,000 queries each, drawn from the corresponding *1B query
  files and applied to the *100M bases for cross-scale comparability.
- **quality gate**: **recall@10 >= 0.8**, an explicit accuracy CONSTRAINT
  (adopted from prior work, per the fetched summary) enforced as a floor
  the system must meet — the only paper in this track with a genuinely
  named, single recall threshold as a first-class evaluation gate (closest
  to, but not identical to, a full recall/QPS Pareto sweep — it is one
  fixed recall target, not several).
- **distance metric**: L2 (Euclidean) throughout, per Figure 1.
- **index**: IVF-PQ, `nlist=2^14`, `nprobe=96`, 256 codebook entries, 16
  subvectors (`m=16`) — the same IVF-PQ family and an m=16 PQ-code
  convention shared with FANNS and UpANNS below, making cross-paper
  comparison at least algorithmically plausible if a common dataset+recall
  target were fixed.
- **hardware**: CPU Intel Xeon Gold 5218 @2.3GHz (AVX/AVX2, OpenMP,
  512GB DDR4, "32-thread CPU" per the paper's own speedup claim), GPU
  NVIDIA A100 PCIe 80GB HBM2e, PIM UPMEM up to 2,560 DPUs / 159GB.
- **metric**: QPS (queries per second) as the primary throughput metric.
  Index-build time and memory are **not** reported separately from search
  time in the fetched excerpt — a gap against this track's own key fairness
  axis.
- **baselines**: 32-thread CPU IVF-PQ and A100-GPU IVF-PQ (2.46x and 2.67x
  speedups respectively per the abstract).
- source: arXiv fulltext (`arxiv.org/html/2410.15621`, abstract page only
  gave partial detail; full HTML fetch supplied dataset/recall/hardware
  detail) + repo `KMC20/DRIM-ANN` `README.md` (citation-only at top level;
  the actual `DRIM-ANN/` subdirectory's own README with build/run
  instructions was not fetched in depth).

## 5. ChenZSLY25 — UpANNS, "Enhancing Billion-Scale ANNS Efficiency with Real-World PIM Architecture" (SC'25)

- key: `conf/sc/ChenZSLY25`, no arXiv id.
- **workloads/inputs**: 3 standard billion-scale datasets — **DEEP**
  (Yandex benchmark), **SIFT** (texmex/BIGANN corpus, i.e. SIFT1B),
  **SPACEV** (Microsoft SPTAG SPACEV1B) — the identical dataset family
  DRIM-ANN uses (both are UPMEM-PIM papers targeting the same standard
  suite, giving reasonable cross-paper comparability within this track).
- **index parameters** (from `run.sh`, all confirmed directly in source):
  `MS=16` (encoded vector length post-PQ), `KSUB=256` (codebook rows,
  identical to DRIM-ANN's 256), `DSUB=8` (dims per encoded subvector; with
  `DIMM=128` this gives 128/8=16 subvectors — the same `m=16` PQ convention
  as DRIM-ANN and FANNS), `CODE_SIZE=16`, **`TOPK=10`** (recall@10,
  matching DRIM-ANN/PathWeaver's convention), `NPROBS=64` (nprobe, same
  ballpark as DRIM-ANN's 96), `BS=10` (**batch size = 10 queries**, a
  small/interactive-latency regime, unlike FANNS's 10K-query
  throughput-oriented batches — see Divergences).
- **hardware**: 7x UPMEM PIM modules, 128GB memory, Faiss 1.8.0 (for the
  CPU/GPU baseline comparison), UPMEM SDK 2023.2.0.
- **metric**: the paper's own one-liner claims "4.3x QPS vs Faiss" — QPS is
  the headline metric, but `run.sh`'s own output is a raw log file
  (`res_sift1b4096top10nprobe64_bs${BS}_ours.txt`), not a formatted
  recall/QPS table; no explicit recall FLOOR (unlike DRIM-ANN's >=0.8) is
  visible in the reproducibility script itself — recall is presumably
  reported post-hoc from the log, not gated during the run.
- **baselines**: Faiss (CPU/GPU, exact version/config not confirmed from
  `run.sh`).
- source: repo `steam009/UpANNS` — `README.md`, `run.sh` (full file read).

## 6. JiangLZLHSRZRHA23 — FANNS, "Co-design Hardware and Algorithm for Vector Search" (SC'23)

- key: `conf/sc/JiangLZLHSRZRHA23`, arXiv:2306.11182.
- **workloads/inputs**: **SIFT100M** (128-dim) and **Deep100M** (96-dim),
  10,000 queries each — the same 100M-scale convention DRIM-ANN's
  GPU-comparison subset uses.
- **quality target**: FANNS's entire premise is "user provides a recall
  requirement, the framework auto-designs hardware+algorithm to meet it,"
  so **three explicit recall targets per dataset** are evaluated by
  construction: **R@1=30%, R@10=80%, R@100=95%** (SIFT) and **R@1=30%,
  R@10=70%, R@100=95%** (Deep) — the closest any paper in this track comes
  to a genuine recall/QPS Pareto framing, though it is 3 discrete recall
  points rather than a continuously swept curve.
- **index**: IVF-PQ, **`nlist` swept from 2^10 to 2^18`** (an actual
  parameter SWEEP, unlike DRIM-ANN/UpANNS's single fixed `nlist`), 16-byte
  PQ codes (`m=16`, same family as DRIM-ANN/UpANNS), OPQ tested with and
  without.
- **hardware**: CPU AWS `m5.4xlarge` (Intel Xeon Platinum 8259CL, 64GB
  DDR4), GPU NVIDIA V100 (5,120 CUDA cores, 32GB HBM), FPGA Xilinx Alveo
  U55c (1.3M LUTs, 9K DSPs, 40MB on-chip memory, 16GB HBM). Software:
  Faiss 1.7.0 (CPU/GPU baselines), Vitis HLS 2022.1 (FPGA).
  distance metric: L2.
- **metric / protocol**: QPS is the primary throughput metric; **query
  batching of size 10,000 is explicitly used for the highest-QPS
  throughput experiments, while latency distributions (implying
  percentile/tail reporting) are captured separately across 100,000
  individual queries** — the clearest, most explicit **batch-vs-
  single-query-latency split** of any paper in this track, directly
  matching the task's own key axis.
- **baselines**: Faiss CPU and Faiss GPU.
- source: arXiv fulltext (`arxiv.org/html/2306.11182`) — the repo's own
  `README.md` is citation-only with no benchmark detail; all evaluation
  detail here is from the arXiv fulltext fetch.

## 7. 000122 — RTNN, "Accelerating Neighbor Search Using Hardware Ray Tracing" (PPoPP'22)

- key: `conf/ppopp/000122`, arXiv:2201.01366.
- **workloads/inputs**: **exact (or controllably approximate) low-
  dimensional (3D) spatial neighbor search** — NOT the high-dimensional
  SIFT/DEEP/etc. ANN family at all. Three real domains: **KITTI LiDAR**
  point clouds (self-driving; 3 frames combined into 1M/12M/25M points,
  mostly planar xy-distribution with a narrow z-range), **Stanford 3D
  Scanning Repository** models (Bunny 360K, Asian Dragon 3.6M, Buddha 4.6M
  points, near-uniform 3D distribution — graphics/vision use case), and
  **Millennium Simulation N-body** cosmological data (2 traces, 9M/10M
  particles, highly non-uniform/clustered — galaxy simulation).
- **search modes**: fixed-radius (range) search is **always exact**
  (`-sm radius`, default); KNN search (`-sm knn`) is exact when query
  partitioning is disabled, or **approximate** via a `-a {0,1,2}` switch
  that relaxes each partition's search radius below what correctness
  strictly requires (default `-a 2`, "most aggressive," falling back to
  exact when the point distribution is detected as uniform). The paper's
  own reported error rate at the default setting: **incorrect results make
  up less than 0.001% of queries** — an explicit, MEASURED approximation
  error rate rather than a recall@k value, a fundamentally different
  accuracy framing than the high-dimensional ANN family's recall@k.
- **hardware**: NVIDIA RTX 2080 (46 RT cores, 2944 CUDA cores, 8GB) and
  RTX 2080 Ti (68 RT cores, 4352 CUDA cores, 11GB).
- **baselines**: cuNSearch (optimized CUDA grid-based, range-search-only),
  FRNN (PyTorch KNN replacement, grid-based), PCLOctree (Point Cloud
  Library octree), FastRNN (a prior RT-core competitor lacking RTNN's
  query-scheduling/partitioning optimizations).
- **metric**: speedup ratio (2.2-65.0x overall per the abstract; range
  search 2.2-44.0x, KNN 3.5-65.0x specifically) — no absolute QPS or
  latency table, and no warmup/repetition/statistic protocol, were visible
  in the fetched fulltext excerpt (open question).
- **approximate-search mechanism note**: query partitioning itself (RTNN's
  core performance technique — building a per-partition BVH sized to each
  partition's queries) is enabled by default and has its own overhead/gain
  tradeoff, separate from the `-a` approximation level; an analytical
  performance model (empirically fit on an RTX 2080) auto-selects the
  number of partitions/batches unless manually overridden.
- source: arXiv fulltext (`ar5iv.labs.arxiv.org/html/2201.01366`) + repo
  `horizon-research/rtnn` `README.md` (full file read).

---

## Divergences

- **The track conflates two structurally different problems, and this is
  the dominant divergence.** PathWeaver, DRIM-ANN, UpANNS, and FANNS are
  genuine high-dimensional (96-128 dim) *approximate* nearest-neighbor
  search over the standard SIFT/DEEP/SPACEV billion-scale benchmark family,
  with recall@k as the accuracy currency and QPS as the throughput currency
  — this is what the task's own key axes (SIFT1M/DEEP/BIGANN/GIST,
  recall@k-vs-QPS Pareto) describe. CLOVER, RTNN, and X-HD are low-
  dimensional (2D/3D) *exact* (or near-exact, bounded-error) spatial
  neighbor-search kernels over completely different data (uniform-random
  synthetic points, LiDAR/graphics/cosmology point clouds, GIS polygons,
  MRI volumes) where recall@k has no meaning (search is exact by
  construction, so recall==1.0 trivially) and the standard ANN datasets
  never appear. This spec keeps these as two separate variants rather than
  forcing recall@k onto the exact-search family or forcing SIFT/DEEP onto
  the spatial family (see Step 2).
- **Recall/quality-gate rigor varies enormously within the high-dim family
  alone.** FANNS evaluates 3 explicit recall targets per dataset by
  construction (its whole framework is "meet a user-given recall"). DRIM-ANN
  enforces one fixed recall floor (>=0.8) as a hard constraint. PathWeaver
  reports mean recall@10 across its full swept-parameter grid (closest to a
  real Pareto sweep in substance, though not plotted as one in the released
  script). UpANNS's `run.sh` doesn't expose any recall target or gate at
  all — recall is presumably read from the raw log post-hoc, with no
  enforcement visible in the reproducibility script. This spec requires ALL
  candidates to report a recall-vs-QPS curve across a FIXED set of recall
  targets (not each candidate's own choice), closing UpANNS's gap and
  standardizing PathWeaver's/DRIM-ANN's/FANNS's differing conventions.
- **Correctness enforcement in the exact-search family ranges from
  "disabled by default" (X-HD's `-check=false`) to "structurally
  unfalsifiable" (CLOVER's `run_test()` has an unconditional `return true;`
  before its actual assertions, so its correctness check can never fail
  regardless of whether the algorithm is correct — the weakest correctness
  enforcement found in either track surveyed for this task).** RTNN is the
  best-behaved of the three: it reports an actual MEASURED error rate
  (<0.001%) for its approximate mode as a first-class number, not a
  disabled/dead-code check. This spec makes exact-match verification (or,
  for RTNN-style bounded approximation, an explicit measured error-rate
  report) mandatory and connected to the timed run for every candidate.
- **Index-build time and memory are not separated from search time/QPS in
  ANY of the 4 high-dimensional papers' released reproducibility scripts**
  (DRIM-ANN, UpANNS: not visible at all; PathWeaver: graph build happens
  once outside the swept-parameter search loop, so it's implicitly
  excluded from the per-config numbers, but never reported as its own
  number either; FANNS: the paper's whole framework performs a
  design-space search over build-time parameters like `nlist`, so build
  cost is presumably significant but not separately quantified in the
  fetched excerpt). This directly violates the task's own key axis
  ("index-build time and memory separately") and is fixed by this spec's
  mandatory separate-reporting requirement.
- **Batch vs. single-query latency is explicit in only one paper.** FANNS
  explicitly separates a 10,000-query-batch throughput measurement from a
  100,000-individual-query latency-distribution measurement — the clearest
  batch-vs-single split surveyed. UpANNS uses a small fixed batch (BS=10)
  throughout with no separate single-query mode visible. PathWeaver,
  DRIM-ANN report only aggregate QPS/latency without stating batch size
  explicitly in the fetched excerpts. This spec adds an explicit
  batch-size axis (1, and a large saturating batch) to both variants.
- **Timing statistics range from "median, properly repeated" (PathWeaver's
  4-iteration mean-recall/median-latency) down to CLOVER's 1-2 raw,
  unaggregated samples per cell** — the loosest repetition count found
  anywhere across both tracks surveyed in this task. X-HD's `-repeat 1`
  is similarly thin. This spec fixes a common warmup/repetition/statistic
  floor for both variants.
- **PQ/IVF parameter conventions are unexpectedly well-aligned across the 3
  PIM/FPGA papers** (DRIM-ANN, UpANNS, FANNS all use `m=16` PQ subvectors,
  and DRIM-ANN/UpANNS share `KSUB=256` codebook entries and similar
  nprobe values) — this is a genuine point of comparability this spec
  preserves rather than needing to reconcile, unlike the FLOP-accounting or
  correctness-enforcement axes above.
- **X-HD is a special case even within the exact/low-dim family**: its
  deliverable is a Hausdorff-DISTANCE VALUE, not a set of nearest
  neighbors — k-NN search is only an internal building block. It has no
  natural "return top-k neighbors" interface to benchmark directly, unlike
  CLOVER and RTNN. This spec treats X-HD's underlying RT-accelerated k-NN
  routine (not the end-to-end Hausdorff-distance computation) as the
  candidate submission to the exact-search variant, and flags the
  end-to-end HD-specific claim as out of this benchmark's scope (a
  different operation, not a k-NN search implementation choice).
