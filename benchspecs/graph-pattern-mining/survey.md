# Graph-pattern-mining track — evaluation methodology survey

Track input: `data/track_inputs/graph-pattern-mining.json`, 10 papers, all
surveyed. None have an arXiv id or a fetchable OA fulltext (7 dl.acm.org PDF
links all returned HTTP 403 — ACM DL paywalls WebFetch); all facts below come
from reading the artifact repos directly with `gh api` (README, run/eval
shell scripts, `.cu`/`.cpp` timing code, pattern/dataset files) — i.e. from
the actual code that produced the papers' numbers, not from prose claims.

Two of the ten (DTMiner, VertexSurge) target a genuinely different problem
than the other eight and are flagged as **out of core scope** in Divergences.

---

## 1. GraphSet — "High Performance Graph Mining through Equivalent Set
Transformations" (SC'23)

- key: `conf/sc/ShiZWCZHYC23`, artifact: github.com/sth1997/GraphSet (CPU+GPU)
- **workloads**: pattern matching + motif counting on `mico`, `patents`,
  `orkut`, `livejournal` (`reproduce/pattern_matching.py`,
  `reproduce/motif_counting.py`); clique counting adds `friendster`
  (`reproduce/clique_counting.py`); FSM on labeled `mico`/`patents`/`youtube`;
  a separate GSI/cuTS comparison (`reproduce/gsi_cuts.py`) uses `Enron`,
  `gowalla`, `roadNetCa/Pa/Tx`, `wikiTalk`. Zenodo artifact
  (zenodo.org/record/8200447) gives file sizes (orkut.g=962MB,
  friendster.g=15GB) but not V/E counts.
- **patterns**: adjacency-matrix bit-strings, no shape names in repo.
  `pattern_matching.py` p1-p6 (5-7 vertices); `motif_counting.py` all size-4
  motifs; `clique_counting.py` k=4,5; `gsi_cuts.py` 33 patterns (5-7 v);
  `scalability.py` p1-p8 incl. a triangle.
- **timing protocol**: CPU (`src/pm_test.cpp`) and GPU (`gpu/gpu_graph.cu`)
  both use `gettimeofday`-based timers (`get_wall_time`/`TimeInterval`);
  single run (`times=1`), no warmup, no averaging, in every script.
- **timing scope**: "counting time" excludes graph loading and
  schedule/IEP-generation (both printed/timed separately) — but the **GPU
  timed region includes managed-memory allocation and H2D copy**
  (`context->init`) inside the same clock as the kernel — a scope leak the
  spec must not repeat.
- **correctness**: the strongest of the ten — gtest hardcoded reference
  counts (`ASSERT_EQ(ans, 19186236035LL)` for Patents+Cycle_6_Tri) plus a
  standalone multithreaded brute-force triangle counter
  (`src/run_brute_force_tc.cpp`) for cross-checking.
- **metric**: seconds; `scalability.py` derives speedup = t(1 GPU)/t(N GPUs).
- **baselines**: GSI and cuTS only (`reproduce/gsi_cuts.py`).
- source: `gh api repos/sth1997/GraphSet/contents/{README.md,reproduce/*,
  src/pm_test.cpp,gpu/gpu_graph.cu,test/pattern_matching_test.cpp,
  src/run_brute_force_tc.cpp}`.

## 2. STMatch — "Accelerating Graph Pattern Matching on GPU with Stack-Based
Loop Optimizations" (SC'22)

- key: `conf/sc/WeiJ22`, artifact: github.com/HPC-Research-Lab/STMatch (GPU)
- **workloads**: `graph_converter/prepare_data.sh` downloads SNAP `wiki-Vote`
  (7,115v/103,689e), `email-Enron` (36,692v/183,831e),
  `com-youtube.ungraph` (1,134,890v/2,987,624e), `soc-LiveJournal1`
  (4,847,571v/68,993,773e), plus a local `mico.txt`; `com-orkut.ungraph`
  (3,072,441v/117,185,083e) and `com-friendster.ungraph` (65.6M v/1.81B e)
  are present but **commented out by default** (1-2h load, 50GB, and
  friendster needs `graph_edge_t` widened to `unsigned long` in
  `src/config.h`) — a concrete example of "billion-edge" being aspirational
  rather than in the default run path.
- **patterns**: 24 query graphs `data/pattern/{1..24}.g`, 5-7 vertices
  (`PAT_SIZE=7` hard cap), used for Table III(a) edge-unlabeled, III(b)
  vertex-unlabeled, IV edge-labeled.
- **timing protocol**: `cudaEvent` (`cudaEventRecord`/`cudaEventElapsedTime`)
  around a single `_parallel_match<<<>>>` call; single run, no
  warmup/repetition anywhere.
- **timing scope**: kernel-only — event recording starts after host CSR
  build (`GraphPreprocessor`/`PatternPreprocessor`) and after H2D transfer
  (`to_gpu()`); loading/CSR/pattern-preprocessing/transfer are all excluded.
- **correctness**: none automated; count = per-warp sums x an automorphism
  multiplicity computed from the pattern's partial order
  (`src/pattern.h::get_partial_order`) — no reference/brute-force check.
- **metric**: milliseconds via `cudaEventElapsedTime`.
- **baselines**: cuTS, GSI, and closed-source Dryadic (Docker image), each
  vendored under `ComparedSystems/`.
- source: `gh api repos/HPC-Research-Lab/STMatch/contents/{README.md,
  graph_converter/prepare_data.sh,src/config.h,src/cu_test.cu,
  data/pattern/*.g}`.

## 3. cuTS — "scaling subgraph isomorphism on distributed multi-GPU systems
using trie based data structure" (SC'21)

- key: `conf/sc/XiangKSHS21`, artifact: github.com/appl-lab/CuTS (GPU,
  distributed)
- **workloads**: 6 SNAP graphs via `download.sh` — `email-Enron` (36,692v/
  183,831e), `loc-gowalla_edges` (196,591v/950,327e), `wiki-Talk`
  (2,394,385v/5,021,410e), `roadNet-CA` (1,965,206v/2,766,607e), `roadNet-PA`
  (1,088,092v/1,541,898e), `roadNet-TX` (1,379,917v/1,921,660e). 33 pattern
  files `1.g`-`33.g` (`data_set/query/ours_format/`); `cuts.py` runs the full
  33 x 6 = 198 cross product.
- **timing protocol**: `cudaEvent_t` (ms precision) in
  `search()`/`search_dfs_bfs_strategy()`; `cuts.py` runs each pair exactly
  once, no warmup/repeat.
- **timing scope**: kernel-only steady-state — the event starts only after
  the graph is loaded, DAG-oriented, and degree-sorted
  (`Graph::sort_search_order`, `src/graph.cpp`) and copied to the GPU;
  orientation/ordering preprocessing and PCIe transfer are excluded from the
  timed region.
- **correctness**: cross-checked against GSI — repo ships
  `compare_with_gsi.xlsx` and a `convert_ours_to_gsi.py` converter so match
  counts on identical (graph, pattern) pairs can be diffed against GSI's
  output.
- **metric**: per-run CSV `data_graph,query_graph,time_ms,match_count`;
  multi-node runs additionally report a speedup metric
  (`multi_nodes_speedup.xlsx`).
- **baselines**: GSI (pkumod/GSI) is the sole named external baseline.
- source: `gh api repos/appl-lab/CuTS/contents/{README.md,download.sh,
  cuts.py,src/host_funcs.cu,src/graph.cpp,data_set/query/ours_format}`.

## 4. GraphFold — "Exploiting Fine-Grained Redundancy in Set-Centric Graph
Pattern Mining" (PPoPP'24)

- key: `conf/ppopp/LinMSZXT24`, artifact: github.com/GPM-lib/GraphFold (GPU)
- **workloads**: 11 real graphs (`.mtx`, fetched via Google-Drive scripts):
  `cit-Patents`, `livej` (LiveJournal), `youtube`, `mico`, `soc-LiveMocha`,
  `soc-gowalla`, `enron`, `soc-delicious`, `soc-pokec`, `orkut`,
  `soc-lastfm`. `run_multi_tables_*.sh` (Tables 3-5) uses a 6-graph subset
  plus **friendster**; `run_fig7_*.sh` (8-GPU scaling) uses only
  livej/orkut/soc-pokec.
- **patterns**: TC (triangle count), CF4-CF7 (4-7-clique finding), MC4 (all
  6 size-4 motifs, via `num_possible_patterns`), and P1-P8 pattern-
  enumeration queries — kernels are literally named `P1_G2Miner`...
  `P8_G2Miner`, confirming these reuse G2Miner's own canonical pattern
  gallery.
- **timing protocol**: single run per config in every `run_*.sh` (no
  loop/repeat/warmup); wall clock via `wtime()` (`gettimeofday`).
- **timing scope**: kernel-only — `wtime()` brackets are placed immediately
  around kernel launch + `cudaDeviceSynchronize()` in `TCSolver.cuh`/
  `PatternSolver.cuh`/`CFSolver.cuh`; graph loading/CSR build and vertex
  orientation + H2D copy happen earlier, in `Engine::RunTC/RunCF/...`, and
  are excluded.
- **correctness**: none automated — GraphFold and its re-implemented G2Miner
  baseline are run on identical inputs and the printed `Result: <count>`
  lines are manually compared.
- **metric**: wall-clock seconds ("Triangle counting time: X seconds"),
  collected into CSVs via grep+tee.
- **baselines**: G2Miner only, re-implemented in-repo — no
  Peregrine/AutoMine/GraphPi.
- source: `gh api repos/GPM-lib/GraphFold/contents/{README.md,run_table2_*.sh,
  run_multi_tables_*.sh,run_fig7_*.sh,src/TCSolver.cuh,src/CFSolver.cuh,
  src/PatternSolver.cuh,src/utils/sm_pattern.h}`.

## 5. GLumin — "Fast Connectivity Check Based on LUTs For Efficient Graph
Pattern Mining" (PPoPP'25)

- key: `conf/ppopp/CaoMLT25`, artifact: github.com/AnySparse/GLUMIN (GPU)
- **workloads**: only `datasets/mico` ships in-repo; README references
  `cit-Patents`, `youtube`, `soc-pokec` (external, via Google Drive) and a
  135-graph set (`dataset_135`) for the Fig. 8 sensitivity sweep — no V/E
  counts recorded in-repo.
- **patterns**: P1-P24 (`src/G2Miner/main.cc`); P4/P5/P23/P24 route to a
  `CliqueSolver` for k=4,5,6,7-clique; P1-P22 are non-clique patterns from
  G2Miner's own pattern gallery (Fig. 7). Default script subsets differ per
  baseline it wraps: G2Miner {P1,P2,P5,P6}, GraphFold {P1,P5,P10,P13},
  AutoMine {P1,P7,P10,P13,P15,P20}.
- **timing protocol**: a `Timer` (`gettimeofday`) with one Start/Stop per
  run — no warmup, no repeats, in the executed path.
- **timing scope**: kernel-only — `Graph g(argv[1])` (CPU load) and
  `GraphGPU gg(g); gg.init_edgelist(g)` (CSR build + H2D transfer) happen
  before `Timer.Start()`.
- **correctness**: none automated (grep for verify/ground_truth: 0 hits);
  only an implicit expectation that the LUT-accelerated variant reproduces
  the same count as its non-LUT baseline, read manually from logs.
- **metric**: wall-clock seconds, `"runtime [G2Miner] = X sec"`.
- **baselines**: G2Miner, GraphFold, AutoMine — each compared against its
  own +LUT variant (GLumin's contribution is a drop-in connectivity-check
  accelerator, not a standalone system).
- source: `gh api repos/AnySparse/GLUMIN/contents/{README.md,run_G2Miner.sh,
  run_GraphFold.sh,run_AutoMine.sh,src/G2Miner/main.cc,include/timer.h}`.

## 6. INFINEL — "An efficient GPU-based processing method for unpredictable
large output graph queries" (PPoPP'24)

- key: `conf/ppopp/ParkOK24`, artifact: github.com/hellogaon/INFINEL (GPU)
- **workloads**: a single query type — triangle **listing** (full
  enumeration, not just counting). Synthetic RMAT24-27 plus real
  LiveJournal/Orkut/Friendster/Twitter. README does not give |V|/|E|, only
  the *output* sizes that motivate the paper: RMAT24/25/26/27 -> 142.5GB/
  275.5GB/776.2GB/2.0TB of triangle-listing output; LiveJournal/Orkut/
  Friendster/Twitter -> 3.4/7.5/50.1/417.9GB.
- **timing protocol**: single execution per config, `ash::stop_watch`
  (wall clock), no repeat/average.
- **timing scope**: has an explicit **dual mode** via a CLI flag
  (`ONLY_KERNEL_MODE_FLAG`) — `n` = end-to-end (CSR/grid-stream load +
  H2D + chunked output writeback, used for the paper's headline figures),
  `y` = kernel-only (used only for one ablation table). This is the one
  artifact in the set that treats "kernel-only vs. end-to-end" as a
  first-class reported axis rather than an implicit default.
- **correctness**: a `VERIFICATION_MODE_FLAG` enables full validation +
  output-list dump against a bundled sample graph's expected output.
- **metric**: wall-clock seconds, printed inline.
- **baselines**: paper claims vs. "conventional two-phase methods" and its
  own INFINEL-SD variant; no baseline system binary ships in the repo (only
  `tl-infinel` is built).
- source: `gh api repos/hellogaon/INFINEL/contents/{README.md,CMakeLists.txt,
  source/query/tl-infinel/tl_main.cpp,tl_procedure.cu,tl_defines.h}`.

## 7. VSGM — "View-Based GPU-Accelerated Subgraph Matching on Large Graphs"
(SC'22)

- key: `conf/sc/JiangZJLZLC22`, artifact: github.com/kygx-legend/vsgm (GPU)
- **workloads**: a single dataset end-to-end: SNAP `com-friendster.ungraph`
  (65.6M v / 1.81B e), converted to a custom CSR binary offline.
- **patterns**: 19 hand-coded pattern graphs `P0-P18`
  (`src/query/pattern.h`) — triangle, square, chordal-square,
  2-tail-triangle, **house**, chordal-house, chordal-roof, three-triangles,
  solar-square, near-5-clique, four-triangles, one-in-three-triangles,
  near-6-clique, square-on-top, near-7-clique, 5-clique-on-top, 5-circles,
  6-circles, hourglass — plus parametrized k-clique (`-kc 3..7`) and
  k-motif/graphlet counting (`-km`).
- **timing protocol**: `TimeMeasurer` (`StartTimer`/`EndTimer`,
  microsecond); one measured run per invocation, no explicit warmup/repeat
  loop in the pipeline or scripts.
- **timing scope**: the timer starts after graph load and view/plan setup,
  and covers on-the-fly view-bin materialization + GPU transfer + matching;
  offline clustering/view-bin-packing preprocessing
  (`features`/`kmeans`/`view_packing` in `prepare_data.sh`) is excluded
  (amortized outside the measured region, not reported as a separate cost).
- **correctness**: none automated — only raw match counts printed per
  query.
- **metric**: wall-clock time (us) + raw match counts.
- **baselines**: none scripted in-repo; README credits "GPU-Accelerated
  Subgraph Enumeration on Partitioned Graphs" (SIGMOD'20) as the pipeline
  VSGM extends, implying it as the paper's baseline, but no executable
  comparison exists in the artifact.
- source: `gh api repos/kygx-legend/vsgm/contents/{README.md,
  prepare_data.sh,run_query.sh,src/app/pattern_matching_pipeline.cu,
  src/query/pattern.h}`.

## 8. Fringe-SGC — "Counting Subgraphs with Fringe Vertices" (SC'25)

- key: `conf/sc/BradleyAB25`, artifact: github.com/burtscher/Fringe-SGC (GPU)
- **workloads**: 8 graphs (`reproduce.py`): USA-road-d.NY, amazon0601,
  rmat16.sym, in-2004, coPapersDBLP, soc-LiveJournal1, internet,
  delaunay_n22 — `.egr` binary format from a Zenodo archive (record
  16733396) sourced from DIMACS9/Galois/SNAP/SuiteSparse (self-loops/
  multi-edges removed, backedges added). Exact V/E counts are computed at
  runtime, not hardcoded in the repo.
- **patterns**: core motifs {vertex, edge, triangle, wedge} each with fringe
  vertices attached, e.g. `patterns/triangle_100.txt` (4 nodes, 12 edges).
  Additional scalability series under `patterns/{tails,wedge,tri_fringe}/`.
- **timing protocol**: one subprocess run per (graph, pattern) in
  `reproduce.py`, no repeat/average/warmup; `CPUTimer` (gettimeofday).
- **timing scope**: kernel-only — excludes graph read, H2D copy, motif-order
  file read, and the separate `fringePreprocess` step (run beforehand as a
  distinct process); only the `occurrences()` GPU search is timed.
- **correctness**: no reference-tool comparison; correctness is algorithmic
  — the raw core count is divided by an automorphism count computed via a
  second recursive call with `doauto=false`.
- **metric**: **throughput** (edges/runtime, edges/s), unlike every other
  surveyed paper's raw seconds — `geometric_fringe.py` further takes the
  **geometric mean across the 8 graphs** per pattern before plotting.
- **baselines**: none — every figure is a Fringe-SGC self-comparison across
  pattern/fringe-count variants (the paper's point is that fringe
  decomposition lets it count patterns too large for any competing SGC
  framework to run at all, so a head-to-head number doesn't exist for those
  cases).
- source: `gh api repos/burtscher/Fringe-SGC/contents/{README.md,
  reproduce.py,geometric_fringe.py,setup.sh,src/fringeCount.cu,
  src/fringePreprocess.cpp,src/info.cpp,patterns/*}`.

## 9. DTMiner — "A Data-Centric System for Efficient Temporal Motif Mining"
(PPoPP'26) — **out of core scope, see Divergences**

- key: `conf/ppopp/Hou0H00YL00L026`, artifact: github.com/yinling123/DTMiner
  (CPU). Repo is extremely sparse (3 commits, first 2026-03-02) — this is a
  brand-new PPoPP'26 paper with a thin artifact.
- **workloads**: 4 **temporal** graphs referenced by absolute path (not
  shipped, no sizes in-repo): `wiki-talk-temporal`, `sx-stackoverflow`,
  `temporal-reddit-reply`, `ethereum`, each with a time-window parameter
  Delta (24h/24h/10h/1h). Pattern set = 13 temporal motif query graphs
  `M1.txt`-`M13.txt` (unnamed beyond numeric id).
- **timing protocol**: single `std::chrono` measurement around the parallel
  match phase only; no warmup/repeat.
- **timing scope**: excludes graph loading and block-map construction
  (logged separately as `preprocess_ms`).
- **correctness**: none automated.
- **metric**: `time_ms match_count preprocess_ms peak_rss_kb` per line —
  wall time + raw counts + peak RSS, no throughput/speedup.
- **baselines**: none — only a thread-count scalability sweep.
- source: `gh api repos/yinling123/DTMiner/contents/{README.TXT,main.cpp,
  CmdArgs.cpp,run_all_motifs.sh,scalable.sh,motifs/*}`.

## 10. VertexSurge — "Variable Length Graph Pattern Match on Billion-edge
Graphs" (ASPLOS'24) — **out of core scope, see Divergences**

- key: `conf/asplos/XieZLCJ024`, artifact: github.com/madsys-dev/VertexSurge
  (CPU)
- **workloads**: 9 graphs (`scripts/graph_info.json`): Twitter2010,
  LDBC_SF100, LDBC_SF1000, Epinions1, LiveJournal1, LastFM, Rabobank
  (directed multigraph), LDBC-FIN_SF10(-L). No V/E counts given in-repo.
- **"patterns"**: variable-length `knows`-style path expansions with a
  runtime `k_max` hop-count CLI arg (e.g. reverse-reachability count,
  3-way community-triangle intersection over variable-length expansions,
  follower-count) — a graph-database query workload, not fixed-shape
  subgraph counting/matching.
- **timing protocol**: single run per binary, no repeat/warmup; internal
  operator profiler reports `QueryUsedTime` in ms.
- **timing scope**: excludes graph load and the offline VLGPM-index
  import step (`bin/importer`); includes only the physical operator DAG
  (scan + VExpand + intersect/count).
- **correctness**: only prints result-tuple counts and a "Selectivity"
  ratio for manual inspection; GTest is linked but no cross-tool reference
  comparison ships.
- **metric**: elapsed ms per query + a selectivity ratio.
- **baselines**: none executed in-repo; a "kuzu-csv" export format implies
  Kuzu was an external baseline in the paper, but no invocation code exists
  in the artifact.
- source: `gh api repos/madsys-dev/VertexSurge/contents/{README.md,
  scripts/graph_info.json,scripts/gen_vlgpm_graph.py,
  src/objects/testify/*.cpp}`.

---

## Divergences

- **DTMiner and VertexSurge target a different problem than the other
  eight and are excluded from the core benchmark variants.** DTMiner mines
  motifs in *temporal* graphs (time-windowed edge streams, Delta-bounded
  matches) — the input isn't a static graph and the "pattern" carries a
  time-ordering constraint the other systems don't have. VertexSurge answers
  *variable-length* path/reachability queries in a graph-database engine
  (Cypher-like `knows*..k` expansions with runtime-chosen k) rather than
  counting/enumerating instances of a **fixed** pattern shape. Folding
  either into the static-pattern-counting variants below would compare
  incommensurable things; they are flagged as a possible future
  `graph-pattern-mining-temporal` / `graph-pattern-mining-vlpath` track
  extension instead (see open_questions).
- **Single-run, no-warmup, no-repetition timing is the norm across ALL ten
  artifacts**, including every GPU paper (GraphSet, STMatch, cuTS,
  GraphFold, GLumin, VSGM, Fringe-SGC all wrap exactly one measured call in
  one `cudaEvent`/`gettimeofday` pair, no loop). This is a bigger gap than
  in most other tracks surveyed in this project (e.g. spmm) where at least
  batched-mean timing is common — here there isn't even that. The spec
  fixes this with an explicit warmup+repeat protocol and reports
  median/min/max, with a size-aware exception (see spec.yaml) since a
  single (graph, pattern) cell can already take minutes at scale.
- **Correctness checking is almost entirely absent.** Only GraphSet (gtest
  hardcoded reference counts + a brute-force triangle counter), cuTS
  (cross-diffed against GSI via a format converter), and INFINEL (an
  opt-in `VERIFICATION_MODE_FLAG` against one bundled sample graph) have
  *any* automated correctness path. The other seven (STMatch, GraphFold,
  GLumin, VSGM, Fringe-SGC, DTMiner, VertexSurge) rely on eyeballing
  printed counts across runs, or nothing at all. The spec makes a
  correctness gate mandatory (see spec.yaml), reusing GraphSet's hardcoded
  reference counts where the graph/pattern overlaps and requiring
  cross-validation between >=2 independently implemented systems (or a
  brute-force/NetworkX oracle on small graphs) elsewhere.
- **Timing-scope leaks in the "kernel-only" claim**: GraphSet's GPU timer
  includes CUDA managed-memory allocation and the H2D copy inside the same
  clock as the search kernel (`context->init` is inside the timed region in
  `gpu/gpu_graph.cu`) — every other kernel-only paper (STMatch, cuTS,
  GraphFold, GLumin) explicitly starts the clock after H2D transfer. The
  spec's kernel-only variant follows the majority (transfer excluded) and
  flags GraphSet's own GPU numbers as measuring a superset of "kernel".
- **COUNTING vs. LISTING is a real scope split, not just a metric
  difference.** Eight of ten papers report a **count** (or a count plus
  automorphism-corrected multiplicity) and never materialize the actual
  match instances. INFINEL is architecturally different: its whole
  contribution is efficiently **listing/writing out** every match, and its
  own numbers show output sizes reaching 2TB (RMAT27 triangle listing) —
  fundamentally different memory/IO profile than a counting-only kernel.
  The core spec below covers COUNTING only (the 8-paper majority); INFINEL
  is scoped in as extra evidence for pattern/graph selection but its
  end-to-end listing variant is out of scope for the kernel-only/e2e
  variants (open_questions).
- **Metric units diverge**: wall-clock seconds/ms dominate (8 of 10); only
  Fringe-SGC reports a throughput unit (edges/s, geometric-mean-aggregated
  across its 8-graph suite); GraphSet/cuTS/VSGM occasionally derive a
  speedup ratio in specific tables (multi-GPU scaling) but default to raw
  time. There is no established "GFLOP/s"-equivalent throughput metric for
  this domain the way dense/sparse linear algebra has one — the spec
  adopts edges-processed-per-second (following Fringe-SGC, and echoing the
  Graph500 GTEPS convention) as the secondary throughput metric, with time
  as primary.
- **Pattern-set naming is inconsistent and mostly unlabeled.** Only VSGM
  gives its 19 patterns human names (house, chordal-square, near-k-clique,
  etc.); GraphSet/GraphFold/GLumin/STMatch/cuTS all define patterns as
  numbered files (`P1.g`...`P33.g`) or adjacency-matrix bit-strings with no
  shape documentation in-repo, so cross-paper pattern-set overlap can only
  be confirmed where the same clique-counting ladder (k=3..7) is used
  explicitly by name: GLumin (`CliqueSolver` k=4,5,6,7 via P4/P5/P23/P24),
  GraphFold (`CF4`-`CF7`), and VSGM (`-kc 3..7`). The spec's pattern set
  below uses this k-clique ladder as the one cross-verified common
  denominator, supplemented by explicitly-defined named patterns (the
  6-graphlet 4-motif census, plus VSGM's house/near-k-clique) rather than
  guessing at what an unlabeled `P7.g` actually looks like.
- **Graph-suite overlap is real but partial.** `mico`, `cit-Patents`,
  `com-youtube`, `soc-LiveJournal1`, and `com-orkut` each recur in >=3 of
  the eight in-scope artifacts' scripts; `com-friendster` recurs in 4
  (GraphSet, STMatch [commented out], GraphFold, VSGM [its only dataset])
  and is explicitly flagged by STMatch's own build system as needing a
  wider integer type — a genuine scale-vs-buildability tradeoff, not a
  cherry-picked outlier. `email-Enron`, `wiki-Vote`/`wiki-Talk`, and
  `loc-gowalla` recur across GraphSet/STMatch/cuTS as a smaller/sparser
  cluster (road-network- and social-graph-adjacent, not GPM-benchmark
  graphs specifically) used mainly for isomorphism baselines (GSI/cuTS).
