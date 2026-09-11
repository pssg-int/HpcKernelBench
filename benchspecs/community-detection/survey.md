# Community detection — evaluation survey

Track input: `data/track_inputs/community-detection.json` (4 papers). All 4
surveyed: 2 via full arXiv/PDF fulltext (ProbGraph, and G-kway-equivalent
depth for the others via artifact code), 2 via artifact README + source code
(NiuOSC26, HSSInfo have no arXiv preprint and are IEEE/ACM-paywalled).

## 1. Niu, Ozkaya, Sarıyüce, Çatalyürek — "Parallel Louvain Algorithms with
Convergence Guarantee" (IPDPS 2026) — `conf/ipps/NiuOSC26`

- **Algorithm family**: strict modularity-maximizing Louvain. Ships three
  variants in one repo: `louvain_d2cc.cpp` (distance-2 community coloring,
  described in the abstract as an "expository" baseline that prevents
  concurrent access to the same community at high synchronization cost),
  `louvain_just_2_improv.cpp` (Grappolo + two incremental improvements,
  "Grappolo++"), and `louvain_versioning.cpp` (community **versioning**,
  their main contribution: per-community version numbers enforce
  read-consistency of modularity-gain data under concurrent local moves).
- **Convergence criterion**: this IS the paper's contribution. Existing
  parallel Louvain implementations (incl. the widely-used Grappolo) let
  threads move vertices based on **stale** modularity/community-weight data
  read concurrently by other threads, so they lack a convergence proof; the
  paper's two schemes are designed to restore a provable convergence
  guarantee while staying parallel. The code (derived from Grappolo, PNNL)
  implements the classic two-phase Louvain loop (local-move phase to a
  per-phase modularity-gain threshold, then community aggregation, iterated
  until no further gain), instrumented with `numItr`, `currMod` ("Current
  modularity"), and a final "Final modularity" / "Total number of
  iterations" printout — i.e., **modularity** is the objective and quality
  metric, consistent with vanilla Louvain.
- **Timing protocol**: `parallelLouvainMethod()` uses `omp_get_wtime()`
  around phases internally and prints per-phase and total time; driver
  takes `argv[1]`=input edge list, `argv[2]`=thread count. No bundled
  dataset, no warmup/repetition loop, no fixed seed for tie-breaking visible
  in the driver — repetition/statistic protocol is left to the caller.
- **Datasets / hardware**: not confirmed — no paper fulltext available (no
  arXiv; IPDPS 2026 not yet indexed with OA text at survey time) and the
  repo ships no dataset or run script beyond "compile with g++ -O3 -fopenmp".
- **Nondeterminism**: explicitly the paper's subject matter — parallel
  Louvain's local-move phase is inherently racy under naive concurrent
  updates (this is exactly what the two proposed schemes fix), so
  run-to-run variance in the resulting partition/modularity for the
  **baseline** (naive concurrent Louvain) is expected to be nonzero, and the
  paper's claim is that its two schemes reduce/eliminate that variance while
  remaining parallel.
- Source: `gh api repos/erdemUB/IPDPS26/contents` (README.md,
  `louvain_versioning.cpp`, `louvain_d2cc.cpp`, `louvain_just_2_improv.cpp`);
  abstract from `data/track_inputs/community-detection.json`.

## 2. Wang et al. — "Swift Unfolding of Communities: GPU-Accelerated Louvain
Algorithm" (GALA, PPoPP 2025) — `conf/ppopp/WangLLWM00Z25`

- **Algorithm family**: strict modularity-maximizing Louvain, GPU-resident.
  Two innovations per the abstract: (1) modularity-gain-based pruning
  (theoretically-grounded, reduces up to 76% of vertex work), (2)
  workload-aware kernels — a shuffle-based warp-level-primitive kernel and a
  shared-memory hash-table kernel for the two different phases' access
  patterns. Multi-GPU scaling via a "dense-sparse synchronization strategy."
- **Convergence criterion**: standard Louvain modularity-gain threshold,
  exposed as CLI flag `-t` (default `threshold = 0.000001` in
  `gala_main.cu`), i.e. the local-move phase halts when the round's total
  modularity gain drops below 1e-6. Pruning strategy is selectable
  (`-p 0..3`: MG / RM / Vite / MG+RM — "Vite" is a reference to the
  well-known Grappolo-lineage HPC Louvain tool, confirming GALA's pruning
  baselines are the same family as NiuOSC26's).
- **Quality metric**: modularity **Q**, printed explicitly per local-move
  iteration and per aggregation round in the sample output embedded in the
  README (`Iteration:3 Q:0.169214`, ..., `final modularity:0.408695`) — the
  repo's own usage example is a direct, code-level instance of "quality
  reported jointly with time" (the same output line also prints elapsed
  ms), which the spec below generalizes into a hard requirement.
- **Timing protocol**: `main.cu`-equivalent driver (`src/gala_main.cu`)
  calls `start = get_time()` **before** `g.load_bin_graph(...)` and prints
  `elapsed time = ...ms` after community detection finishes — i.e., **graph
  loading/deserialization is included in the reported number**, not
  separated from the community-detection kernel. This is a preprocessing/
  kernel-time conflation the spec below explicitly forbids.
- **Datasets**: README documents `com-lj.ungraph.txt` (SNAP com-LiveJournal,
  ~4M vertices / 34.7M edges) as the worked example, fetched directly from
  `snap.stanford.edu`; `data/prepare_graph.sh` in the repo instead fetches
  `com-amazon.ungraph.txt.gz` (SNAP com-Amazon) — the two documentation
  sources disagree on which single graph is "the" example, but both are
  standard SNAP ground-truth-community graphs.
- **Nondeterminism**: no seed or repetition-count convention documented in
  the README; warp-level shuffle-based tie-breaking during pruning/local
  move is a plausible source of run-to-run variance, unaddressed in the
  artifact.
- **Hardware/build**: CUDA 11.6, GCC/G++ 10.4, GNU Make; single- and
  multi-GPU (dense-sparse sync for the latter), no CPU baseline shipped in
  the repo (comparisons are in the paywalled ACM DL paper only).
- Source: `gh api repos/LinXi-lx/GALA/contents` (README.md,
  `src/gala_main.cu`, `src/louvain_gpu/*`); PPoPP'25 abstract (ACM DL
  paywalled, no arXiv — WebFetch of the DOI page returns only metadata).

## 3. Zhu, Sun, Fang, Xu — "A Low-Memory Community Detection Algorithm With
Hybrid Sparse Structure and Structural Information for Large-Scale
Networks" (HSSInfo, TPDS 2023) — `journals/tpds/ZhuSFX23`

- **Algorithm family**: **not Louvain**. Optimizes **structural information
  theory (SInfo, Li et al.)**, an entropy-based objective, not modularity.
  This is a materially different objective function from papers 1-2: the
  code's core primitive (`entropy_delta_functor` in `hssinfo.cc`) computes
  a two-dimensional-structural-entropy delta for merging two communities
  (`loop1*log2(degree1) + loop2*log2(degree2) - loop_module*log2(degree_module)
  + connect*log2(degree_sum), all /degree_sum`), and the greedy merge loop
  uses `isFinished`/`isUnfinished` thrust predicates to track which
  candidate merges have converged (entropy can no longer be reduced) —
  structurally analogous to Louvain's "no further modularity gain" stopping
  rule, but on a different objective, so **HSSInfo's quality numbers are
  not directly comparable to Q from papers 1-2 or the algorithm from
  paper 4** even on the same graph.
- **Convergence criterion**: greedy pairwise community merge until no
  candidate merge reduces structural entropy further (entropy-delta
  analogue of modularity-gain-based stopping), implemented via device-side
  `thrust` reductions/compactions rather than an explicit iteration cap.
- **Quality/headline metric**: the paper's stated contribution is **memory**
  (abstract: "up to 19x memory shrink" via the hybrid sparse data
  structure), not raw speed — memory footprint is a first-class metric for
  this paper, not a footnote.
- **Timing protocol**: `main.cu` uses `gettimeofday` and explicitly reports
  **two separate numbers**: `init time` (graph load into the `HSSInfo`
  object) and `community detection time` (`info.CommunityDetection()`
  alone) — i.e. HSSInfo's own artifact already separates preprocessing from
  kernel time, in contrast to GALA's conflated number above.
- **Datasets**: README ships one worked CLI example against
  `facebook_combined.once.txt` (SNAP ego-Facebook, 4,039 vertices) and links
  out to a Google-Drive/Baidu-Drive-hosted dataset bundle for the paper's
  full evaluation graphs — the exact large-scale graph list is not
  confirmed from the artifact (links are third-party cloud storage, not
  enumerated in the README, and the paper itself is IEEE-paywalled with no
  arXiv mirror).
- **Hardware/build**: CUDA Toolkit >= 11.2, GCC/G++ 9.4.0, CMake > 3.21,
  Ubuntu 20.04; both a CPU path (`hssinfo.cc`) and a GPU/thrust path
  (`hssinfo.cu`, 31KB, the one wired into the CLI via `main.cu`) exist in
  the repo, platform recorded as `["cpu", "nvidia-gpu"]`.
- **Nondeterminism**: not discussed in the README; the greedy merge order
  over `thrust::device_vector` reductions is plausibly deterministic given
  a fixed input and thread-block configuration, but this is not verified
  from the artifact alone.
- Source: `gh api repos/DrownFish19/HSSInfo/contents` (README.md,
  `hssinfo.cc`, `main.cu`); TPDS abstract from
  `data/track_inputs/community-detection.json`; IEEE Xplore PDF is
  paywalled (403 on WebFetch).

## 4. Besta et al. — "ProbGraph: High-Performance and High-Accuracy Graph
Mining with Probabilistic Set Representations" (SC22) — `conf/sc/BestaMLTIKAJPKGVMH22`

Full text obtained via `arxiv.org/pdf/2208.11469` (arXiv:2208.11469v3).
ProbGraph is a general graph-mining acceleration technique (Bloom
filter / MinHash approximate set intersection) applied to several problems;
**Clustering is one of four applications** (alongside Triangle Counting,
4-Clique Counting, Vertex Similarity), so only the Clustering slice is
in-scope for this track.

- **Algorithm family**: **not Louvain, not entropy-based** — a third,
  distinct family again. ProbGraph's "Clustering" is **Jarvis-Patrick
  clustering** (Listing 4 in the paper): for every edge `(v,u)`, compute a
  vertex-similarity score `S_C(v,u) = |N_v ∩ N_u|` (common-neighbor count;
  Jaccard/Overlap variants also evaluated) and place the edge in cluster
  set `C` iff the score exceeds a user threshold `τ`. This is a **single-
  pass, non-iterative, threshold-based** clustering — there is no
  local-move/aggregation loop and thus no "convergence" concept comparable
  to papers 1-3 at all.
  ProbGraph's contribution is accelerating the `|N_v ∩ N_u|` computation
  itself (via Bloom filters / MinHash sketches) rather than changing the
  clustering algorithm's structure.
- **Quality metric**: `accuracy = |cnt_PG - cnt_EX| / cnt_EX` — the
  relative difference between the **approximate cluster/count output** and
  the exact (CSR, no sketching) baseline's output, for whatever count the
  algorithm produces (cluster count for Clustering, triangle/4-clique count
  for the other problems). This is **not modularity** and not directly
  reducible to it — it measures how well the probabilistic set
  representation approximates the *exact same algorithm's* output, not how
  good the resulting partition is by any global community-quality
  criterion. Reported per-scheme in Fig. 4 (Real-world and Kronecker
  graphs) as relative-count vs. speedup vs. relative-memory scatter plots.
- **Timing protocol** (§VIII-A, "Datasets, Methodology, Architectures" —
  the paper explicitly follows Hoefler & Belli's "twelve ways" benchmarking
  recommendations): **omit the first 1% of performance data as warmup**;
  derive enough repetitions for the mean and a 95% non-parametric
  confidence interval; run on the max cores available unless stated
  otherwise. This is the most rigorous, explicitly-justified timing
  protocol of the four surveyed papers.
- **Datasets** (Table VIII): a large, explicitly-named real-graph suite
  spanning SNAP / KONECT / DIMACS / Network Repository / WebGraph sources
  and 8 domains (biological, interaction, brain, economic, social,
  scientific-computing, discrete-math, chemistry), sizes from 134 vertices
  / 5K edges up to 3.1M vertices / 117M edges (Orkut), plus synthetic
  Kronecker (power-law) graphs of systematically varied `n`, `m`, `m/n`.
- **Hardware**: Dell PowerEdge R910 (Intel Xeon X7550 @ 2.00GHz, 18MB L3,
  1TiB RAM, 32 cores/CPU x4 sockets = up to 128 cores, though speedup plots
  are captioned "All 32 cores"); also Piz Daint XC50 nodes (12-core Xeon
  E5-2690, 64GiB) for distributed-memory experiments.
- **Nondeterminism/seed handling**: "We use the current time in
  milliseconds as a random seed" (§VI-C) for MurmurHash3 — i.e. **runs are
  explicitly NOT reproducible bit-for-bit**; the paper relies on its
  statistical accuracy bounds (concentration inequalities, Ch. VII) rather
  than fixed-seed determinism to make accuracy claims trustworthy.
- **Storage budget**: generic parameter `s ∈ [0,1]` caps additional memory
  vs. plain CSR; evaluation never exceeds 33% additional storage — memory
  is a first-class, explicitly bounded axis, similar in spirit to HSSInfo's
  memory framing.
- **Baselines**: GAP + GMS (GraphMineSuite) tuned exact parallel
  implementations; heuristics without accuracy guarantees (Reduced
  Execution, Partial Graph Processing, two AutoApproximate variants),
  explicitly shown to have *worse* accuracy than ProbGraph at *higher* cost
  — i.e. the paper itself makes the "faster-but-worse-quality is not a win"
  argument against its own heuristic baselines.
- Source: full fulltext PDF via `arxiv.org/pdf/2208.11469` (arXiv:2208.11469v3,
  21 Nov 2022 revision), §II (fundamentals), §III (Listing 4, Jarvis-Patrick
  clustering), §VIII (Evaluation, Datasets/Methodology/Architectures,
  Table VIII).

## Divergences

- **Algorithm family — 3-way split, not the expected Louvain/Leiden/
  label-prop spread.** Papers 1-2 (NiuOSC26, GALA) are strict
  modularity-maximizing Louvain. Paper 3 (HSSInfo) optimizes a *different*
  objective entirely (structural/entropy information, not modularity).
  Paper 4 (ProbGraph) is not an iterative local-move algorithm at all —
  it's single-pass threshold (Jarvis-Patrick) clustering, accelerated via
  approximate set intersection. **No Leiden or label-propagation paper
  appears in this 4-paper track** (label-prop appears only as a citation:
  GALA's "-p 2" pruning option is literally named "Vite," an established
  Grappolo-lineage tool, but Vite itself is not one of the 4 surveyed
  papers). The spec below is scoped to the Louvain-family papers as the
  primary, comparable variant, with an explicit secondary variant for the
  other two objective families so their (real, useful) contributions are
  still benchmarkable without falsely implying cross-family comparability.
- **Quality metric — three non-interchangeable definitions.** Modularity Q
  (papers 1-2) vs. structural-entropy reduction (paper 3) vs.
  relative-count accuracy against the *same algorithm's* exact baseline
  (paper 4). A HSSInfo run and a GALA run on the same input graph produce
  numbers that answer different questions ("how much entropy reduced" vs.
  "how much modularity gained") and are not on the same scale or even the
  same sign convention. This is the single largest fairness risk in the
  track and is why the task's framing note — "a faster but lower-modularity
  result is not a win" — needs to be operationalized as a **hard gate**,
  not a footnote (see `spec.yaml`).
- **Preprocessing/timing-scope handling diverges sharply.** GALA's own
  driver **includes graph loading inside the reported elapsed time**
  (`start = get_time()` before `load_bin_graph`), while HSSInfo's driver
  **explicitly separates** `init time` from `community detection time`.
  NiuOSC26's driver reports internal phase timings via `omp_get_wtime()`
  but leaves total-run accounting to the caller. ProbGraph is the most
  rigorous (Hoefler & Belli protocol, 1%-warmup discard, 95% CI) but that
  protocol is stated for its 4 target problems in aggregate, not
  Clustering-specifically broken out from load time in the fulltext excerpt
  obtained.
- **Convergence criterion**: NiuOSC26 treats *correct, provable convergence
  under parallelism* as its entire contribution (i.e., convergence isn't
  assumed, it's proven); GALA uses a fixed numeric modularity-gain
  threshold (`1e-6`) as a practical stand-in without a convergence proof
  for the parallel case; HSSInfo uses an analogous entropy-delta stopping
  rule; ProbGraph has no convergence concept (single-pass).
- **Nondeterminism/seed handling**: explicitly ProbGraph's admitted weak
  point (time-based hash seed, not reproducible); explicitly NiuOSC26's
  subject matter (fixing races in existing parallel Louvain); silent/
  unaddressed in GALA and HSSInfo's artifacts.
- **Graph suite**: zero graphs are shared by name across all four papers.
  GALA's own two documentation sources (README vs. `prepare_graph.sh`)
  disagree on the single worked example (`com-lj` vs. `com-amazon`).
  HSSInfo's small worked example (`facebook_combined`, SNAP ego-Facebook,
  4,039 vertices) is a plausible common small/control instance since it's
  a standard, tiny, universally-available SNAP graph, but HSSInfo's own
  full evaluation suite (cloud-drive-hosted) is not enumerable from the
  artifact. ProbGraph's suite is the broadest and most rigorously
  documented (Table VIII) but skews toward small-to-medium graphs (max
  3.1M vertices) rather than the billion-edge scale GALA/HSSInfo aim at.
- **No `k` (target community count) parameter exists in this track** —
  unlike graph-partitioning, community detection discovers the number of
  communities rather than taking it as input. This is a structural
  difference from the graph-partitioning track's "k values" axis; the
  closest analogue here is the **modularity-gain / entropy-delta
  threshold** (`-t` in GALA, implicit in HSSInfo/NiuOSC26's stopping
  predicates), which the spec below requires to be disclosed instead.
