# kcore-kclique track — evaluation methodology survey

Track input: `data/track_inputs/kcore-kclique.json`, 5 papers, all surveyed. None
have an arXiv id recorded in the track input, but 2 of the 5 (KCGPU, ProbGraph)
turned out to have public preprints findable by title search
(`arxiv.org/abs/2104.13209`, `arxiv.org/abs/2208.11469`) even though the input
JSON's `arxiv` field is empty for them — both were read via arXiv fulltext
(PDF, extracted with `pypdf` since `ar5iv`/`arxiv.org/html` failed on these
older submissions). Shun20's ACM-hosted PDF (`dl.acm.org`) 403'd on fetch, but
an NSF-PAGES-hosted copy of the same paper (`par.nsf.gov/servlets/purl/10137062`)
was fetchable and read the same way. RDMCE and LiuSZ24 have no findable
preprint; both were surveyed from their artifact repos (README + shell
scripts) via `gh api`.

This track is unusually heterogeneous: the 5 papers cover **4 genuinely
different kernel problems** that only share the "k-core / k-clique" graph
family — static k-core decomposition, batch-dynamic k-core maintenance, exact
k-clique counting, approximate (probabilistic) clique/triangle counting, and
maximal clique enumeration (a 5th, distinct from counting). See Divergences.

---

## 1. RDMCE — "Root-Down Exposure for Maximal Clique Enumeration on GPUs"
(PPoPP'26)

- key: `conf/ppopp/PanQZ26`, artifact: github.com/Pz1996/RDMCE (GPU)
- **workloads**: `scripts/download_dataset.sh` pulls 7 named real-world graphs
  — `com-friendster` (SNAP, ungraph, 31GB), `twitter-higgs`
  (`higgs-social_network`, SNAP), `facebook` (`facebook_combined`, SNAP ego
  network), `soc-orkut-dir`, `soc-sinaweibo`, `web-ClueWeb09-50m`, `wiki-link`
  (all 4 via Network Repository / nrvis.com) — plus a separate
  "MCE Konect datasets" bundle (`dogster`, etc.) mirrored on Zenodo
  (`zenodo.org/records/17648545`) because the original `konect.cc` links used
  by earlier MCE papers (G2-AIMD, MCE-GPU) are dead. Each baseline needs its
  own preprocessed binary format (`datasets/RDMCE/*.bin`,
  `datasets/G2-AIMD/*.bin`, `datasets/MCE-GPU/*.bel`), built once via
  `scripts/preprocessing.sh` (~6h, explicitly separate from the timed runs).
- **timing protocol**: `scripts/overall_eval.sh` wraps each
  (method, dataset) invocation with GNU `time` under a hard **3600s
  timeout**; a timeout is logged as its own outcome, not silently recorded as
  0 or infinity. Single execution per cell — no warmup/repeat loop in the
  script itself.
- **timing scope**: wall-clock around the *whole executable* (binary-graph
  load through enumeration completion), not a device-side kernel-only timer;
  text-to-binary preprocessing is excluded (a separate one-shot step).
- **precision & correctness**: MCE is exact/deterministic (no
  approximation); no automated cross-checker script was found in `scripts/`
  or `exp/` (the analysis/plotting scripts were not read in full — see
  survey Divergences / spec open_questions).
- **metric**: wall-clock time (seconds), aggregated post-hoc by
  `scripts/analyze.sh` into `logs/analysis_results.csv`; no repetition means
  this is effectively a single-sample number, not mean/median.
- **baselines**: G2-AIMD (`bin/bk`, prior GPU MCE system) and MCE-GPU (two
  variants, `-i p` and `-i px`) — both vendored as git submodules under
  `baselines/`.
- source: `gh api repos/Pz1996/RDMCE/contents/{README.md,
  scripts/overall_eval.sh,scripts/download_dataset.sh,scripts/compile.sh}`.

## 2. LiuSZ24 — "Parallel k-Core Decomposition with Batched Updates and
Asynchronous Reads" (PPoPP'24)

- key: `conf/ppopp/LiuSZ24`, artifact:
  github.com/qqliu/batch-dynamic-kcore-decomposition (CPU, built on GBBS)
- **workloads**: 11 named graphs hosted at
  `storage.googleapis.com/graph-files/` — `brain`, `ctr`, `dblp`,
  `friendster`, `livejournal`, `orkut`, `stackoverflow`, `twitter`, `usa`
  (road network), `wiki`, `youtube` — each shipped in 4 forms:
  `_insertion_edges`, `_deletion_edges`, `_initial`, `_batch` (mixed
  50/50 insert-delete), purpose-built for dynamic-update experiments.
- **timing protocol**: `gbbs/scripts/test_approx_kcore.py` drives every run
  with an explicit `-rounds 3` (3 repetitions), the GBBS-framework
  convention; parameters swept per `approx_kcore_setup.txt`: benchmark in
  {PLDS, LDS, EKCore, AKCore}, deltas (eps), lambdas, batch sizes (e.g. 1e6,
  1e7), worker/hyperthread counts, a "levels divisor" (default 50), and an
  `-ins-opt` heuristic flag.
- **timing scope**: `read_approx_kcore_results.py` reports Average runtime,
  Max runtime, Total runtime, Space (bytes), and (when `Output stats: True`)
  Average/Max error ratio vs. the exact baseline — all per (program, graph,
  delta, lambda, batch-size, workers, divisor) cell; whether the initial
  graph load is excluded from the per-batch numbers was not independently
  confirmed from the README text (would require reading `LDS.C`).
- **precision & correctness**: PLDS is a **(2+eps)-approximate** k-core
  algorithm; the harness has a built-in accuracy check against exact core
  numbers (Output stats -> average/max error ratio) — the strongest,
  most explicit correctness path of the 5 papers surveyed.
- **metric**: runtime (seconds, avg/max/total across the 3 rounds); space in
  bytes; error ratio for approximate variants.
- **baselines**: LDS (sequential level-structure baseline, adapted from
  Henzinger-Neumann-Wiese) and EKCore/AKCore (their own exact/approximate
  *static* references, used to bound the batch-dynamic algorithms' latency
  and error).
- source: `gh api repos/qqliu/batch-dynamic-kcore-decomposition/contents/
  {README.md,gbbs/benchmarks/EdgeOrientation/README.md,
  gbbs/scripts/test_approx_kcore.py}`.

## 3. AlmasriHNXH22 — "Parallel K-Clique Counting on GPUs" (ICS'22)

- key: `conf/ics/AlmasriHNXH22`, artifact: github.com/ssmoha7/KCGPU (GPU).
  Track input's `arxiv` field is empty but a public preprint exists:
  `arxiv.org/abs/2104.13209` (read via `arxiv.org/pdf/2104.13209` through
  `ar5iv.labs.arxiv.org`).
- **workloads**: 5 SNAP graphs — `as-skitter` (1.7M v / 11.1M e), `com-dblp`
  (317K v / 1.0M e), `com-orkut` (3.1M v / 117.2M e), `com-friendster`
  (65.6M v / 1.8B e), `com-lj` (4.0M v / 34.7M e).
- **k values**: k = 4 through k = 11 (paper's Table 1).
- **timing protocol** (paper): single run per (graph, k) cell, no
  warmup/repeat stated; a hard **5-hour cutoff** — times exceeding it are
  not reported (DNF), not recorded as 0. Matches the repo: `src/main.cu`'s
  KCLIQUE branch wraps the counting call in `for (int i = 0; i < 1; i++)`
  — literally one iteration, with an earlier `while (k < 11)` sweep loop
  commented out (manual re-invocation per k instead).
- **timing scope**: the paper's reported execution time **includes**
  preprocessing (graph orientation via degree/degeneracy ordering) **and**
  counting, but **excludes** disk read time (confirmed both in the paper
  text and in the repo, which logs `"Preprocess time"` and `"count time"`
  as separate `Log()` lines but the *paper's* headline numbers sum both —
  see Divergences).
- **precision & correctness**: exact (integer) counts; no automated
  reference-count check visible in `src/main.cu` — it only `Log()`s the
  count and an ad hoc `m/time` "teps" value.
- **metric**: primarily wall-clock seconds in the paper (no
  GTEPS/throughput unit used in the paper's own tables per the extracted
  text); the repo code computes and prints `m/time` ("teps",
  edges-per-second) but this isn't the paper's reported metric.
- **baselines**: ARB-COUNT (CPU, "state-of-the-art parallel graph
  orientation" implementation, 60-thread Xeon, 240GB RAM) and Pivoter (CPU,
  "state-of-the-art parallel pivoting" implementation); also a rough
  comparison to a 96-thread system from Lonkar & Beamer. All CPU — this is
  presented as the first GPU-specialized k-clique counter, so there was no
  GPU peer to compare against at publication time.
- **algorithm-variant richness** (from `src/main.cu`, not necessarily all
  exercised in the paper's headline table): `KcliqueConfig` sweeps
  `Algo` in {GraphOrient, Pivoting}, `BinaryEncode` in {true, false},
  `PartSize` (warp-partition width) in {1, 2, 4, 8, 16, 32}, and
  `processBy` in {ByNode, ByEdge} — a rich orthogonal config space. A
  separate `KCLIQUE_LOCAL` mode (`kclique_local.cuh`) computes **per-vertex
  local** k-clique counts rather than one global aggregate — a distinct
  "local" algorithm class alongside the counting kernel.
- **k-core in the same repo** (not itself in AlmasriHNXH22's paper scope,
  but shipped in the same codebase, `kcore/kcore.cuh`): a bucket-based
  parallel **peeling** implementation (`findKcoreIncremental_async`) whose
  public `count()` returns `k - 1`, i.e. the graph's **degeneracy** (a
  single scalar — the largest non-empty core number), not a full per-vertex
  coreness array.
- source: `arxiv.org/abs/2104.13209` (fulltext via ar5iv) +
  `gh api repos/ssmoha7/KCGPU/contents/{README.md,Makefile,src/main.cu,
  kcore/kcore.cuh,kclique}`.

## 4. BestaMLTIKAJPKGVMH22 — "ProbGraph: High-Performance and High-Accuracy
Graph Mining with Probabilistic Set Representations" (SC'22)

- key: `conf/sc/BestaMLTIKAJPKGVMH22`, artifact: github.com/spcl/probgraph
  (CPU, OpenMP). Track input's `arxiv`/`oa_url` don't list one, but a public
  preprint exists: `arxiv.org/abs/2208.11469` (read via
  `arxiv.org/pdf/2208.11469`, extracted page-by-page with `pypdf` — both
  `ar5iv` and `arxiv.org/html` failed to convert this document).
- **workloads**: ~30 named real graphs from SNAP, KONECT, DIMACS, Network
  Repository, and WebGraph, spanning biological / interaction / brain /
  economic / social / scientific-computing / discrete-math / chemistry
  origins (paper's Table VIII), 134 vertices up to 45K v / 14.5M e (plus
  Orkut at 3.1M v / 117M e separately) — **plus synthetic Kronecker
  power-law graphs** used to systematically vary a single graph property
  (n, m, or m/n) that can't be isolated with real-world data.
- **problems evaluated**: Triangle Counting (TC), Clustering (Jaccard /
  Overlap / Common-Neighbors similarity), 4-Clique Counting (4c), Vertex
  Similarity. **Important scope gap**: the repo's own automated
  reproduction script (`launch_experiments.sh`) only runs
  `PROBLEMS=(tc jp-jc jp-cn jp-ov)` — 4-clique counting (`4c_base`,
  `4c_bf`, `4c_1h` executables exist and are documented in the README, and
  Figure 5 of the paper reports 4-clique results) is **not** included in
  the automated repro loop.
- **timing protocol**: the paper explicitly cites "a recent set of
  recommendations on benchmarking parallel applications" (i.e. a formal
  scientific-benchmarking methodology reference) and follows it: **omit the
  first 1% of performance data as warmup**, and **derive enough samples for
  the mean and a 95% non-parametric confidence interval**. This is the most
  rigorous, most explicitly-stated timing protocol of the 5 papers
  surveyed.
- **timing scope**: representation-construction time and estimation
  (query) time are both measured and reported (paper's Table V/VII split
  construction time from estimation time analytically; the empirical
  evaluation reports end-to-end kernel time per problem). CSR loading is
  done via the GAP Benchmark Suite loader, standard practice for that
  harness family (typically excluded from the timed region, not
  independently confirmed here).
- **precision & correctness**: probabilistic/approximate by design —
  accuracy is `|cntPG − cntEX| / cntEX` against the exact CSR-based
  baseline, reported as a first-class axis (not a pass/fail gate). Headline
  results: >90% accuracy at up to ~50x speedup (4-clique counting on
  Kronecker graphs), >98% accuracy at up to ~30x speedup (Clustering via
  Common Neighbors) using Bloom-filter (BF) estimators; MinHash (k-Hash /
  1-Hash) estimators give higher speedups but lower/less consistent
  accuracy.
- **metric**: NOT a single number — each result is a 3-axis data point:
  speedup vs. exact baseline (X), relative accuracy (Y), relative memory
  overhead vs. default CSR (color/shade, mostly <25%). Architecture: Dell
  PowerEdge R910, Intel Xeon X7550 @2.0GHz, 4 sockets x 32 cores, 1TiB RAM
  (main results use all 32 cores of one socket-group); also Piz Daint XC50
  nodes (12-core Xeon E5-2690, 64GiB) for some experiments.
- **baselines**: exact CSR-based implementations from the GAP and GMS graph
  benchmarking suites; established TC estimators Doulion (edge sampling)
  and Colorful TC (combinatorial pruning); heuristics without theoretical
  guarantees — Reduced Execution, Partial Graph Processing, two
  Auto-Approximate variants (ProbGraph beats these in accuracy by 25-75
  percentage points at comparable speed).
- source: `arxiv.org/abs/2208.11469` (fulltext PDF via `pypdf`) +
  `gh api repos/spcl/probgraph/contents/{README.md,test.sh,
  launch_experiments.sh,src/scripts/run_generic_problem_real_graphs.sh}`.

## 5. Shun20 — "Practical Parallel Hypergraph Algorithms" (PPoPP'20)

- key: `conf/ppopp/Shun20`. **The `artifact_url` on file
  (github.com/DevSheth/prallel-hypergraph-algorithms) is WRONG — see
  Divergences.** The paper's actual artifact is github.com/jshun/ligra,
  self-identified as such inside the course-project README itself
  (reference [2]: "the official github repository for the paper"). Facts
  below (algorithm names, WE/WI split) use `jshun/ligra`'s
  `apps/hyper/HyperKCore.C` / `HyperKCore-Efficient.C`; the evaluation
  numbers come from the paper's own PDF, not from either repo.
- **workloads**: 8 hypergraphs (paper's Table 2) — `com-Orkut` (2.32M v /
  15.3M e, constructed from SNAP community-membership data, one hyperedge
  per community — a **different construction** from the plain
  `com-orkut.ungraph` simple graph used by RDMCE/KCGPU/ProbGraph),
  `Friendster` (7.94M v / 1.62M e, same community-data construction),
  `Orkut-group` (2.78M v / 8.73M e, via KONECT bipartite graphs), `Web`
  (27.7M v / 12.8M e, KONECT), `LiveJournal` (3.2M v / 7.49M e, KONECT) —
  plus 3 synthetic uniform-random hypergraphs: `Rand1` (10^8 vertices /
  hyperedges, cardinality 10), `Rand2` (10^9 v/e, cardinality 10), `Rand3`
  (10^7 v/e, cardinality 100).
- **k values**: N/A for k-core here — this is **full decomposition**
  (per-vertex coreness for all cores simultaneously via peeling), not a
  fixed-k query. The paper instead sweeps an **algorithm-variant axis**:
  work-efficient (WE) vs. work-inefficient (WI) k-core peeling — WE reuses
  an amortized bucket structure across peeling rounds, WI recomputes
  degrees from scratch each round; WE is up to **733x faster in parallel**
  on inputs with many peeling rounds (e.g. Web, which has 3.18x10^5
  rounds).
- **timing protocol**: 72-core Dell PowerEdge R930 (4x 18-core Xeon E7-8867
  v4 @2.4GHz, hyper-threading = 144 threads, 1TB RAM), Cilk Plus + g++
  5.5.0 `-O3`, `numactl -i all` for parallel runs. Reports T1
  (single-thread) and T72h (72-core+HT) wall-clock times and derived
  speedup SU = T1/T72h. **The extracted evaluation text does not state an
  explicit repetition count, warmup, or statistic** (mean/median/min) for
  these numbers — unlike LiuSZ24 (same code lineage, GBBS/Ligra family)
  which does state `-rounds 3` elsewhere.
- **timing scope**: not explicitly confirmed from the extracted text
  whether hypergraph loading is inside or outside T1/T72h (standard Ligra
  convention separates I/O, but this wasn't independently verified here —
  see open_questions).
- **precision & correctness**: exact, deterministic computation; no
  automated cross-check described in the evaluation section (would be
  WE-vs-WI output-count agreement, not stated as automated).
- **metric**: wall-clock seconds (T1, T72h) + parallel speedup ratio;
  **uniquely among the 5 papers**, also reports a microarchitectural table
  (fraction of cycles stalled on memory, LLC local miss rate, memory
  bandwidth in GB/s) for select algorithms including WE k-core, on
  com-Orkut / Rand1 / LiveJournal.
- **baselines**: distributed-memory systems HyperX and MESH, compared via
  numbers *reported in their own papers* (explicitly caveated by the author
  as a rough, hardware-mismatched comparison) — plus one in-house re-run of
  MESH on the same 72-core machine for a single config (PageRank,
  com-Orkut) as a fairer point of reference. For algorithms that don't
  treat hyperedges specially (hypertree/CC/SSSP), also compares to running
  plain Ligra graph algorithms on the "clique-expanded" graph
  representation of the same input (up to 235x more edges, 2.8-30.6x
  slower depending on algorithm).
- source: `par.nsf.gov/servlets/purl/10137062` (fulltext PDF via `pypdf`,
  since `dl.acm.org/doi/pdf/10.1145/3332466.3374527` returned HTTP 403) +
  `gh api repos/jshun/ligra/contents/apps/hyper` (file listing only, to
  confirm `HyperKCore.C`/`HyperKCore-Efficient.C` exist) +
  `gh api repos/DevSheth/prallel-hypergraph-algorithms/contents/README.md`
  (to establish the artifact-attribution problem itself).

---

## Divergences

- **Wrong artifact for Shun20 — a live instance of the "third-party
  reimplementation" trap.** The `artifact_url` on file for `conf/ppopp/Shun20`
  points to a Chennai Institute of Technology CS6023 (GPU Programming)
  **course project** by a named student ("Sheth Dev Yashpal, Roll No:
  CS17B106"), which reimplements only 4 of the paper's ~9 algorithms
  (HyperBFS, HyperBPath, HyperSSSP, HyperPageRank) in CUDA as a class
  assignment, and explicitly does **not** include k-core decomposition —
  the exact algorithm this track cares about. The repo's own README names
  `jshun/ligra` as "the official github repository for the paper." This is
  the same failure mode flagged in this project's `CLAUDE.md` ("third-party
  reimplementations and reproducibility-report forks are not artifacts");
  it slipped through Phase 1's artifact search presumably because the
  course-project repo's README cites the paper's title/DOI verbatim,
  triggering a naive keyword/README match. Recommend correcting this
  paper's `artifact_url` to `github.com/jshun/ligra` in
  `output/included.json` / `papers.json` (out of scope for this survey
  task itself, flagged for follow-up).
- **This track spans 4 distinct kernel problems, not one.** (1) static
  k-core decomposition (full per-vertex coreness via peeling — LiuSZ24's
  EKCore/AKCore, KCGPU's degeneracy-only peeling, Shun20's WE/WI hypergraph
  peeling), (2) batch-dynamic k-core maintenance under a stream of edge
  updates (LiuSZ24's PLDS/LDS/CLDS — a genuinely different cost model,
  amortized per-batch latency rather than one-shot wall time), (3) exact
  k-clique counting via GPU-specialized search/orientation (KCGPU), (4)
  approximate probabilistic clique/triangle counting via Bloom-filter/
  MinHash sketches (ProbGraph), and (5) maximal clique enumeration via
  GPU search-tree exploration with dynamic load balancing (RDMCE) — an
  unbounded-output enumeration problem, architecturally distinct from (3)'s
  bounded combinatorial count. spec.yaml keeps these as 5 separate
  variants rather than forcing a merge that would compare incommensurable
  things (see spec.yaml's own justification for exceeding the usual 1-4
  variant guidance).
- **KCGPU's own paper times preprocessing + kernel together**, unlike this
  project's usual "preprocessing excluded, reported separately" default:
  the paper's headline k-clique-counting numbers **include** DAG
  orientation/degeneracy-ordering time. spec.yaml's kernel-only variant
  splits these out; KCGPU's own reported numbers should be understood as
  measuring a superset of what the spec calls "kernel."
- **Single-run timing dominates**, similar to other tracks surveyed in this
  project (e.g. graph-pattern-mining): RDMCE (single run, 3600s timeout),
  KCGPU (literally `for (i=0;i<1;i++)`, 5h DNF cutoff), and Shun20 (no
  stated repetition) all report what appears to be one measured sample per
  configuration. LiuSZ24 is the partial exception (`-rounds 3`). ProbGraph
  is the clear outlier in the other direction — the only paper of the 5
  that cites a formal benchmarking-methodology reference and follows an
  explicit warmup + mean + 95%-CI protocol.
- **Correctness/accuracy checking is inconsistent across the 5 papers.**
  LiuSZ24 has the strongest built-in path (automated average/max error
  ratio vs. an exact baseline, for its approximate algorithms).
  ProbGraph reports accuracy as a first-class result axis (not a gate) by
  design, since it's inherently approximate. KCGPU and Shun20 describe no
  automated reference-count checking in their evaluation text or repo code
  (correctness is implicit / exact-by-construction, unverified by any
  script found). RDMCE's scripts (as surveyed) show no automated
  cross-checker between RDMCE/G2-AIMD/MCE-GPU maximal-clique counts,
  despite MCE being exactly the kind of exact/deterministic problem where a
  count mismatch would be a correctness bug, not just noise.
- **Graph-suite overlap is real but comes from different provenance
  chains.** SNAP (`com-*`, `as-skitter`, `facebook_combined`) is the one
  source common to all 5 papers, but LiuSZ24's dynamic-graph files, RDMCE's
  MCE-preprocessed binaries, and Shun20's hypergraph constructions
  (community-as-hyperedge, KONECT-bipartite) are each **format-specific,
  non-interchangeable derivatives** of similarly-named source graphs — e.g.
  Shun20's "com-Orkut" hypergraph and KCGPU's/RDMCE's "com-orkut" simple
  graph share a name and an ultimate SNAP source but are structurally
  different objects (one is bipartite vertex-hyperedge incidence, the other
  a plain edge list) and cannot be substituted for each other.
- **Metric units diverge.** RDMCE/KCGPU/Shun20 report raw wall-clock time
  (seconds); LiuSZ24 adds space (bytes) and error-ratio columns; ProbGraph
  reports a 3-axis (speedup, accuracy, memory) tuple per data point rather
  than a single throughput number. There is no shared throughput unit
  across all 5 the way GFLOP/s serves dense linear algebra — spec.yaml
  adopts per-variant throughput conventions (clique-count/s,
  cliques-enumerated/s, per-batch latency) rather than forcing one unit
  track-wide.
