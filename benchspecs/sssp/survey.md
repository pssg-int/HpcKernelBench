# SSSP track — evaluation-methodology survey

Surveyed 4/4 papers from `data/track_inputs/sssp.json`. Fulltext was reachable
for 3/4 (ADDS/Wang-PPoPP21 via the authors' own UT Austin mirror, GraphIt/CGO20
via arXiv 1911.07260 + its own artifact-evaluation-guide README, and efg/IPDPS23
via its artifact repo — same paper already surveyed in depth for the `bfs`
track). For the fourth (Khanda-TPDS22), the exact journal paper's fulltext was
unreachable (no arXiv, no OA link, IEEE Xplore paywalled) and its
`artifact_url` turned out to be a **misattributed third-party class-project
repository** (see below); a directly-related earlier conference paper by an
overlapping author subset, with its own public GitHub artifact, was used as
the best available evidence instead and is flagged as such throughout.

---

## 1. Traversing Large Compressed Graphs on GPUs ("efg", IPDPS 2023) — SCOPE MISMATCH, no SSSP

Source: artifact repo `pgera/efg` (`README.md`, `src/` file listing) — the
same paper already surveyed in full for the `bfs` track's `survey.md`.

- **Finding**: the repository implements **only BFS** over Elias-Fano
  compressed graphs (`src/bfs.cuh`, `src/bfs_kernels.cuh`; the only binary
  built is `ef_bfs`). There is no SSSP driver, no weighted-edge handling in
  `ef_layout.h`/`ef_graph.h` (the compression scheme is for unweighted
  adjacency), and the README's own "Performance" table reports **only BFS
  GTEPS** numbers. The paper's classifier-generated `one_liner` ("enables
  BFS/SSSP/PageRank on larger graphs") appears to over-generalize from the
  paper's framing of Elias-Fano compression as a generic graph-representation
  technique applicable *in principle* to other traversal algorithms — the
  actual artifact and, as far as reachable text shows, the actual paper
  evaluate BFS exclusively.
- **Consequence for this track**: this paper does **not** define an SSSP
  benchmark variant on its own (there is nothing SSSP-specific to survey: no
  weight distribution, no delta-stepping parameter, no SSSP correctness
  check). It is retained here as evidence of the general
  compressed-representation technique that a future SSSP-on-compressed-graphs
  variant could reuse, and as a concrete instance of the track-input
  false-positive pattern the project's `CLAUDE.md` already warns about
  (topic-adjacency over-inclusion by the classification pipeline).

---

## 2. A Fast Work-Efficient SSSP Algorithm for GPUs ("ADDS", PPoPP 2021)

Source: fulltext PDF from the authors' own mirror,
`https://www.cs.utexas.edu/~lin/papers/ppopp21.pdf` (Wang, Fussell, Lin, UT
Austin), including its Appendix A "Artifact Description."

- **workloads/inputs**: **226 graphs** selected from the Lonestar 4.0
  benchmark suite and the SuiteSparse Matrix Collection, with an explicit,
  reproducible **selection criterion**: (1) ≥100K vertices and ≥1M edges, (2)
  "suitable for SSSP traversal" — at least 75% of vertices reachable from one
  vertex; a few very large graphs the baselines could not fit in memory were
  excluded; any negative edge weights were converted to positive. Table 2
  breaks the 226 graphs down by average degree (6 buckets, <4 to ≥64) and
  diameter (6 buckets, <40 to ≥640) to demonstrate structural diversity —
  the most rigorous, disclosed graph-selection methodology in this track.
  Both **int and float** edge-weight graph variants are shipped separately
  (`sssp-int.zip`, `sssp-float.zip`), i.e. precision is an explicit,
  first-class axis, not an afterthought.
- **edge-weight distribution**: inherited from the source graphs (SuiteSparse
  matrix values / Lonestar's native weights); no synthetic re-weighting
  scheme is imposed. Not a uniform-random-in-[1,W] convention like GraphIt's.
- **source-vertex selection**: **a single source vertex per graph** ("at
  least 75% of vertices reachable from *one of the vertices*"); the exact
  selection rule for which single vertex is not stated in the reachable text.
  This is a materially different (and narrower) convention than the 64-random
  -root Graph500 convention this track's `bfs` spec standardizes on.
- **timing protocol**: from the artifact's own patch notes (`nf_int.patch`
  item 3d): **"we run each graph 8 times, and we use average timing"** — 8
  repetitions, arithmetic mean, no explicit warmup phase described. The same
  convention is applied to ADDS itself and to all baselines for a like-for-like
  comparison.
- **timing scope**: SSSP computation time only, explicitly excludes graph
  loading and result I/O per the artifact's stated instrumentation
  methodology (measured "the SCC[sic]/SSSP computation time, excluding
  everything else, such as reading in the graph, outputting the result, and
  verifying the result" — phrasing consistent across the group's later
  ECL-SCC paper too, see entry 3 below in the connected-components survey).
- **delta-stepping parameter disclosure**: this is the paper's **central
  contribution**. ADDS **dynamically selects Δ at runtime** based on live GPU
  utilization (Section 5.5), replacing the prior state-of-the-art Near-Far
  algorithm's **static formula** Δ = C×(W/D) (W = average edge weight, D =
  average degree, C a hand-picked constant). Crucially, the paper shows via
  an **ablation study** (Table 5) that disabling dynamic-Δ selection alone
  drops ADDS's speedup over Near-Far from 3.5× to 2.4× on an RTX 3090 — i.e.
  Δ-selection strategy accounts for roughly a third of the paper's reported
  gain, and the paper is unusually transparent about this by design (that IS
  the contribution). To keep the *baseline* comparison fair, the authors
  further **standardized all parallel baselines to derive Δ from the
  published Near-Far formula rather than accepting hand-tuned/default values**
  — a deliberate anti-cherry-picking discipline (contrast with GraphIt, entry
  4, which hand-tunes Δ per graph *per framework*).
- **precision & correctness**: **exact final-distance-array comparison**
  between implementations via a `verify.py` script (`verify_against_*`
  targets) — reports a "mismatch" for any differing distance value; this is a
  stronger, more literal correctness check than any other paper in this
  track (not a hash, not a component-count, not a formal proof — a direct
  per-vertex value diff against Dijkstra/other implementations). One
  documented caveat: nvGRAPH's internal float conversion sometimes produces
  off-by-1 distances vs. the int implementations, so int-graph verification
  against NV is disabled.
- **metric**: primarily **wall-clock time (median... actually mean of 8
  runs)**; secondarily **work efficiency** = 1/(vertex processing count),
  tracked via an explicit low-overhead instrumentation counter added to every
  implementation. **No GTEPS/TEPS metric is used anywhere in this paper** —
  a real divergence from the BFS track's GTEPS convention. The paper reports
  results as **distributions of speedup across all 226 graphs** (e.g. "for 8
  of the 226 graphs, speedup < 0.9×") rather than a single aggregate number —
  the most statistically thorough presentation in this track.
- **baselines**: NF (Near-Far, LonestarGPU 6.0, GPU), Gun-NF / Gun-BF
  (Gunrock Near-Far / Bellman-Ford, GPU), NV (NVIDIA nvGRAPH, GPU), CPU-DS
  (Galois delta-stepping, shared-memory CPU), Dijkstra (serial CPU, Galois).
  Average speedups: 2.9× (NF), 5.8× (Gun-NF), 9.6× (Gun-BF), 13.4× (NV),
  14.2× (CPU-DS), 34.4× (serial Dijkstra), all on an RTX 2080 Ti; 3.5× over
  NF on a newer RTX 3090.
- **hardware**: NVIDIA RTX 2080 Ti and RTX 3090 GPUs (CUDA 10.0); Intel Core
  i9-7900X (10c/20t) for CPU baselines.
- **artifact note**: the paper's real artifact is archived on **Zenodo**
  (`https://zenodo.org/record/4365954`), containing exact patches applied to
  the LonestarGPU/Galois baseline repos — good reproducibility practice
  worth adopting as a model.

---

## 3. A Shared-Memory Algorithm for Updating SSSP in Large Weighted Dynamic Networks — proxy evidence for Khanda et al., TPDS 2022

**IMPORTANT ARTIFACT-ATTRIBUTION FINDING**: the `artifact_url` recorded for
this paper in `data/track_inputs/sssp.json`
(`https://github.com/mmasabalvi/parallel-sssp-update-mpi-openmp-metis`) is
**not the paper's own artifact**. Repo metadata shows owner "Muhammad Masab
Hammad" (FAST-NU, Pakistan), created 2025-04-20, described as
"A Parallel Algorithm Template for Updating Single-Source Shortest Paths in
Large-Scale Dynamic Networks" — i.e. this is a **third-party student
reproduction/class-project implementation** of the paper's ideas (contains a
`Project Report.pdf` and `PPT.pptx`, the hallmarks of a course assignment),
not code by Khanda/Srinivasan/Bhowmick/Norris/Das themselves. This is exactly
the "reproducibility-report fork" trap flagged in this project's `CLAUDE.md`.
It was NOT used as an evidence source.

Instead, the actual TPDS 2022 paper (Khanda, Srinivasan, Bhowmick, Norris,
Das — "A Parallel Algorithm Template for Updating Single-Source Shortest
Paths in Large-Scale Dynamic Networks," extends a shared-memory algorithm to
GPU per its abstract) could not be fetched directly (IEEE-paywalled, no
arXiv). A directly-related, earlier paper by an overlapping author subset
(Sriram Srinivasan, Sara Riazi, Boyana Norris, Sajal K. Das, Sanjukta
Bhowmick — "A Shared-Memory Algorithm for Updating Single-Source Shortest
Paths in Large Weighted Dynamic Networks," publicly deposited at
`par.nsf.gov/servlets/purl/10107624`) presents the **same core two-step
update algorithm** for the same problem and links a real public artifact at
`https://github.com/DynamicSSSP/HIPC18` ("HIPC18 Parallel Dynamic SSSP") —
this is used below as the best-available proxy for the TPDS22 paper's
methodology, flagged everywhere as **proxy evidence, not the TPDS22 paper
itself**, since the TPDS22 abstract explicitly claims a GPU implementation
this precursor does not have (CPU shared-memory only).

- **workloads/inputs**: 2 synthetic RMAT-model graph types — "G" (scale-free,
  a=.45,b=.15,c=.15,d=.25) and "ER" (uniform-random, a=b=c=d=.25) — at scale
  24 (RMAT24: ~16M v / 268M e) and scale 25 (RMAT25: ~33M v / 536M e); plus 3
  real-world SNAP graphs with explicit sizes: com-LiveJournal (3,997,962 v /
  34,681,189 e), com-Youtube (1,134,890 v / 2,987,624 e), soc-Pokec
  (1,632,803 v / 30,622,564 e).
- **edge-weight distribution**: **not disclosed** in the reachable text —
  the paper focuses on graph-structure changes (edge insertion/deletion), not
  on weight generation; SSSP weights are presumably inherited from whatever
  default the RMAT generator / SNAP dataset provides, unconfirmed.
- **source-vertex selection**: **vertex 0, fixed, for all main results**
  ("For consistency, we use vertex 0 as the source vertex in all cases").
  A dedicated ablation (Fig. 4) tests 2 additional randomly-chosen source
  vertices and finds negligible effect on timing/scalability — a useful,
  disclosed sensitivity check, but the headline numbers still use a single
  fixed source, a materially different (and much weaker) convention than
  this track's other papers or the `bfs` track's 64-random-root standard.
- **timing protocol**: experiments on 50-million-edge and smaller batched
  updates (edge insertions/deletions, e.g. Fig. 6: "each experiment was
  repeated four times"), reporting update time vs. thread count (1-72
  threads); no separate warmup phase is described.
- **timing scope / preprocessing**: the *entire point* of the paper is
  updating an existing SSSP tree rather than recomputing from scratch — so
  "preprocessing" (the initial from-scratch SSSP computation) is explicitly
  **out of scope by design**: the algorithm takes an already-computed SSSP
  tree as input and measures only the update cost for a batch of Δ-graph
  changes (x% insertions / (100-x)% deletions).
- **precision & correctness**: formal, **proof-based correctness**
  (Lemmas IV.1–IV.2: the algorithm's parent-child assignment produces valid
  trees whose distances match a known-correct SSSP tree), not an empirical
  distance-array diff. No numerical-tolerance discussion (weights/distances
  are not shown to be floating point in the reachable formulation).
- **metric**: **wall-clock time in seconds**, explicitly and deliberately
  **not TEPS/GTEPS** — the paper states outright: *"We compare time instead
  of TEPS (traversed edges per second) because (i) time is a more
  fundamental metric and (ii) TEPS is not a suitable metric for dynamic
  networks since our goal is to process as few edges as possible."* This is
  a direct, reasoned rejection of the throughput-metric convention used
  elsewhere in this track (and in the `bfs` track) — the most explicit
  metric-choice justification found in either track's survey.
- **baselines**: Galois 2.2.1's static Δ-stepping SSSP (default Δ), applied
  to the *entire updated graph from scratch* — i.e. the baseline is
  "recompute," not another update algorithm. Reported speedup over Galois:
  up to 4.0× (scale-free RMAT24, 1 thread), converging to ~1.1-1.2× at high
  thread counts as Galois itself parallelizes well.
- **hardware**: 36-core dual-socket Intel Xeon E5-2699 v3 (Haswell), 256GB
  DDR4 RAM — CPU-only, no GPU numbers in this precursor (the TPDS22 journal
  version's GPU extension is unconfirmed from available sources).
- **secondary axis studied**: **level of asynchrony** (how many
  synchronization-free hops are processed before a barrier) is treated as a
  first-class tunable parameter with its own dedicated experiments (Fig.
  7-9) — asynchrony level 0 (fully synchronous) through 5000 — showing up to
  ~2× further speedup from relaxing synchronization frequency, at the cost
  of more redundant vertex updates. This is conceptually adjacent to but
  distinct from delta-stepping's Δ parameter (it governs synchronization
  granularity in the update-propagation phase, not bucket coarsening).

---

## 4. Optimizing Ordered Graph Algorithms with GraphIt (CGO 2020)

Source: arXiv fulltext (`arxiv.org/abs/1911.07260`, full PDF read directly)
+ the artifact's own reproducibility guide
(`graphit_eval/priority_graph_cgo2020_eval/readme.md`) + its driver script
`graphit_eval/priority_graph_cgo2020_eval/perf_eval/benchmark.py`. This is
the same paper already covered for BFS-adjacency in the track's classifier
output, but here surveyed specifically for its **Δ-stepping SSSP**
evaluation (Table 4, "SSSP" columns), which is one of six ordered algorithms
the paper benchmarks (SSSP, PPSP, wBFS, A*, k-core, SetCover).

- **workloads/inputs**: named, sized real graphs (Table 3): Orkut (OK, 3M v
  / 234M e), LiveJournal (LJ, 5M v / 69M e, 85M symmetrized), Twitter (TW,
  41M v / 1469M e, 2405M symmetrized), Friendster (FT, 125M v / 3612M e),
  WebGraph (WB, 101M v / 2043M e, 3880M symmetrized), plus 3 road networks
  built from OpenStreetMap data: Massachusetts (MA, 0.45M v / 1.2M e),
  Germany (GE, 12M v / 32M e), RoadUSA (RD, 24M v / 58M e).
- **edge-weight distribution — the richest disclosure in this track**:
  **three explicit, separately-tested weight regimes**, each shipped as a
  differently-named graph variant: (1) `_logn` = weights drawn from
  `[1, log(num_vertices))`, used specifically for wBFS "following the
  convention in previous work [Julienne]"; (2) `_rand1000` = uniform-random
  integer weights in `[1, 1000)`, used for SSSP/PPSP/k-core/SetCover; (3)
  `origweights` = the road networks' **real** travel-distance/time weights
  (no synthetic substitute). Table 4's caption states this explicitly:
  "Graphs marked with † have weight distribution of [1, log n). Road
  networks come with original weights. Other graphs have weight distribution
  between [1, 1000)." No other paper in this track ties weight distribution
  this explicitly to the metric table.
- **delta-stepping parameter disclosure — the other central finding**: Δ is
  **hard-coded per graph AND per framework** in the benchmark driver
  (`framework_app_graph_runtime_param_map`), e.g. for `sssp_delta_stepping`:
  socLive_rand1000→100, road-usad_rand1000→8000, road-usad_origweights→40000,
  twitter_rand1000→4, twitter_logn→1, germany→400000. The paper's own text
  confirms this is not arbitrary: *"The best Δ values for social networks
  (ranging from 1 to 100) are much smaller than deltas for road networks
  with large diameters (ranging from 2^13 to 2^17)... We also tuned the Δ
  values for the comparison frameworks to provide the best performance."*
  — i.e. **every framework in the comparison (GraphIt, GAPBS, Julienne,
  Galois) gets its own hand-tuned, framework-specific, graph-specific Δ**,
  confirmed by the artifact's reproducibility README, which lists distinct
  tuned Δ values per framework for the same graph (e.g. socLive_rand1000: Δ=100
  for GraphIt/GAPBS, Δ=100 for Julienne, Δ=2 [as a power-of-2 exponent] for
  Galois). This is the single clearest illustration in either track's survey
  of the "delta-stepping parameter disclosure" fairness risk the track's
  key-axes brief calls out: a hand-tuned-per-framework Δ can silently
  inflate or deflate a comparison depending on how much tuning effort went
  into each competitor.
- **source-vertex selection**: **10 fixed, hand-picked source vertices per
  graph** (`get_starting_points()`: `["14","38","47","52","53","58","59",
  "69","94","96"]` for most graphs, fewer — 6 — for Friendster "because
  friendster takes a long time"), explicitly filtered to **"points with
  non-zero out degree and don't hang during execution"** — i.e. a curated,
  not randomly-sampled, source set, filtered specifically to avoid runs that
  would fail. This is a materially weaker (more favorable-to-the-tool)
  convention than either the Graph500-style 64-random-root standard or even
  ADDS's single-but-broadly-reachable source.
- **timing protocol**: "The times for SSSP and wBFS are averaged over 10
  starting vertices" — arithmetic mean over the 10 fixed sources (no warmup
  described, no repetition count beyond the 10 distinct sources; PPSP/A*
  are averaged over 10 fixed source-destination pairs "chosen to have a
  balanced selection of different distances" — also curated, not sampled).
- **timing scope**: search/computation phase only; graph loading uses
  pre-converted `.wsg`/`.wadj`/`.gr` formats specific to each framework,
  conversion time not counted (and not directly comparable across
  frameworks, since each framework's preferred format differs).
- **precision & correctness**: correctness of the GraphIt-generated code is
  validated once via the compiler's own test suite (not per-benchmark-run);
  no explicit distance-value cross-check against an independent reference is
  described for the performance numbers in Table 4.
- **metric**: **wall-clock time in seconds** (Table 4), not GTEPS. The
  paper's headline claims are speedup ratios: up to 16.8× faster than
  Julienne, 7.8× faster than Galois, 3.5× faster than hand-optimized GAPBS
  (all specifically on SSSP, per the abstract's breakdown by algorithm).
- **baselines**: GAPBS (hand-optimized C++, eager bucket update, "hidden"
  reference implementation their own Ordered-Processing runtime is modeled
  on), Julienne (lazy bucket update, "used from early 2019," note: the
  Julienne developers later incorporated this very paper's bucketing
  interface into their own subsequent releases — a citation-loop worth
  flagging), Galois v4 (approximate-priority-ordering worklist, not a strict
  bucketing structure), GraphIt-unordered and Ligra (unordered Bellman-Ford,
  included to show the ordered-vs-unordered algorithmic-efficiency gap:
  1.67×-600×+ speedup from using Δ-stepping/priority ordering at all, a
  separate axis from the ordered-framework-vs-framework comparison).
- **hardware**: dual-socket Intel Xeon E5-2695 v3 (12 cores/socket, 24 cores
  / 48 hyperthreads total), 128GB DDR3-1600, 30MB LLC/socket, Ubuntu 18.04 —
  **CPU-only**, no GPU numbers for SSSP anywhere in this paper (unlike ADDS
  and efg, both GPU-only) — a real platform divergence across this track.

---

## Divergences

1. **Track-input scope mismatch.** One of the 4 papers (efg) does not
   actually implement or evaluate SSSP at all — its artifact and (as far as
   reachable) its paper text cover BFS exclusively over Elias-Fano compressed
   graphs. **Resolution**: efg does not define a spec variant; it is
   retained purely as background evidence for a possible future
   compressed-graph variant, following the same precedent the `bfs` spec set
   for RCM's tangential BFS-as-subroutine role.

2. **Artifact misattribution.** The `artifact_url` recorded for paper #2
   (Khanda et al., TPDS22) is a third-party class-project reproduction, not
   the authors' own code. **Resolution**: the spec's evidence for this
   paper is drawn instead from a closely-related, publicly-artifacted
   precursor paper by an overlapping author subset, flagged throughout as
   proxy evidence rather than a direct read of the TPDS22 paper itself
   (which per its own abstract additionally covers a GPU implementation this
   precursor lacks).

3. **Metric: none of the 4 papers use GTEPS/TEPS**, the convention this
   track's `bfs` sibling spec standardized on. ADDS uses wall-clock time +
   an explicit work-efficiency (vertex-processing-count) metric; Srinivasan
   et al. use wall-clock time and **explicitly argue against TEPS** for
   dynamic/update algorithms ("our goal is to process as few edges as
   possible," not maximize throughput); GraphIt uses wall-clock time and
   speedup ratios. **Resolution**: unlike the `bfs` spec, this spec does
   NOT mandate GTEPS as the primary metric for SSSP — it adopts time as
   primary (median of ≥8 runs) with a secondary
   edges-relaxed-per-second-style throughput number reported for
   cross-track comparability, since weighted SSSP's variable per-vertex
   relaxation count makes GTEPS-style edge-counting less standardized than
   for unweighted BFS in the first place (this is itself flagged as an open
   question below).

4. **Source-vertex selection ranges from rigorous to convenience-driven.**
   ADDS uses a single vertex per graph (unspecified selection rule, but
   reachability-filtered by the graph-selection criterion itself);
   Srinivasan et al. use a single **fixed vertex 0** for headline numbers
   (with a disclosed 2-extra-vertex sensitivity check showing low impact);
   GraphIt uses 10 **hand-picked, hang-avoiding** fixed sources per graph.
   None of the 4 papers use a Graph500-style random-root convention.
   **Resolution**: per the task brief's explicit instruction, this spec
   aligns SSSP source selection with the `bfs` spec's 64-fixed-seed-random
   -roots-from-the-giant-component convention, applied to weighted SSSP —
   this is a deliberate strengthening relative to every surveyed paper's own
   practice, justified for the same reason `bfs`'s spec gives (a single or
   hand-picked source cannot be assumed representative, and non-random
   curated sources risk implicitly favoring cases the implementation is known
   to handle well).

5. **Delta-stepping parameter disclosure is the sharpest divergence in this
   track.** ADDS **dynamically adapts** Δ at runtime and forces baselines
   onto a **shared, published formula** rather than allowing per-baseline
   hand-tuning — the most disciplined practice found. GraphIt **hand-tunes**
   Δ per graph *and per competing framework* via what the paper describes as
   manual tuning ("we tuned the Δ values for the comparison frameworks to
   provide the best performance") — every framework in the comparison,
   including the competitors, gets its own best-case Δ, which is fairer than
   NOT tuning competitors but still opens room for tuning-effort asymmetry
   between the authors' own well-understood tool and less-familiar
   baselines. Srinivasan et al.'s update algorithm does not use a
   bucket-Δ at all (it's a different algorithmic family — direct
   priority-queue-free iterative relaxation) but has an analogous
   "asynchrony level" parameter with its own reported sensitivity curve.
   **Resolution**: the spec makes Δ-selection-method disclosure a REQUIRED
   protocol field for every SSSP kernel variant (mirroring `bfs`'s mandatory
   direction-optimization disclosure), and adds a dedicated
   `sssp-delta-sensitivity` variant that makes the Δ-vs-performance
   relationship itself a first-class, reportable result rather than a
   hidden, potentially cherry-picked hyperparameter — directly modeled on
   ADDS's own Figure 7 (execution time and work performed vs. Δ) and
   GraphIt's per-graph Δ table, both of which already do this analysis
   internally but do not standardize how it should be reported for
   cross-paper comparison.

6. **Weight-distribution disclosure ranges from exemplary to absent.**
   GraphIt explicitly ties three distinct, named weight regimes (`_logn`,
   `_rand1000`, real road-network weights) to every reported number in its
   results table. ADDS uses whatever weights the source SuiteSparse/Lonestar
   graphs natively carry, with an explicit int/float split but no synthetic
   re-weighting scheme. Srinivasan et al.'s weight-generation method for
   SSSP is **not disclosed** in the reachable precursor paper at all.
   **Resolution**: the spec makes edge-weight distribution (real vs.
   synthetic; if synthetic, the exact range/scale) a REQUIRED disclosure
   field, modeled directly on GraphIt's own practice, and the graph suite's
   `recommended_subset` fixes an explicit weight-generation protocol per
   graph so that "SSSP on graph X" is not silently ambiguous between a
   near-trivial small-integer-weight instance and a wide-dynamic-range one
   (which can change which algorithmic technique — e.g. bucket count vs. Δ
   granularity — is favored).

7. **Preprocessing/scope boundary is fundamentally different for the
   dynamic-update paper.** For ADDS and GraphIt, "preprocessing" means CSR
   construction / format conversion before a from-scratch SSSP run. For
   Srinivasan et al., the entire premise is that a **prior, already-computed
   SSSP tree is the input**, and only the incremental update cost for a
   batch of graph changes is measured — recomputation-from-scratch is
   the *baseline*, not the thing being timed. **Resolution**: rather than
   forcing this into the kernel-only/e2e-preprocessing variant split that
   works for BFS and static SSSP, the spec adds a dedicated
   `sssp-dynamic-update` variant with its own timing scope definition
   (update cost only, given a valid pre-existing SSSP tree and a batch of Δ
   edge changes) and its own baseline (from-scratch recomputation by a
   standard static SSSP implementation, e.g. Δ-stepping), following the
   paper's own well-reasoned rejection of throughput-style metrics for this
   problem shape.

## Open questions

- Whether GTEPS-style throughput (edges relaxed / second, or similar) has
  any standardized definition for *weighted* SSSP across the broader
  literature beyond this 4-paper sample was not resolved — none of the 4
  papers report it, so the spec's secondary throughput metric is a
  first-principles construction (total edge relaxations, including
  re-relaxations of the same edge under Δ-stepping, divided by time) rather
  than one inherited from a paper's own convention; this should be
  cross-checked against a broader SSSP literature sample before being
  treated as a settled convention.
- The TPDS22 journal version of Khanda et al.'s algorithm (the paper
  actually in this track's input list) claims a **GPU implementation** in
  its abstract that the CPU-only precursor paper used as this survey's proxy
  does not cover; whether the GPU version's protocol (source selection,
  timing, correctness check) matches the CPU precursor's practice is
  unconfirmed and should be revisited if IEEE Xplore access or an author
  preprint becomes available.
- efg's Elias-Fano compression technique is BFS-only in the reachable
  sources; whether the same graph-representation approach could plug into
  a Δ-stepping SSSP kernel (the natural extension the track-input classifier
  apparently assumed) is an open algorithmic question, not a benchmarking
  one — flagged here only because it explains why this paper contributes no
  SSSP-specific spec content despite its inclusion in the track input.
- Whether Srinivasan et al.'s "asynchrony level" parameter and this spec's
  required Δ-selection-method disclosure should be unified into a single
  cross-cutting "synchronization/coarsening parameter disclosure" field
  (since both control a work-efficiency/parallelism tradeoff via a tunable
  granularity knob, just for different algorithmic families — bucket-based
  Δ-stepping vs. iterative relaxation) was considered but not resolved; the
  spec currently keeps them separate since the dynamic-update variant and
  the static-kernel variants are already scoped differently.

## Evidence

```yaml
efg_ipdps23: "SCOPE MISMATCH — repo/paper implement BFS only, no SSSP driver, no weighted-edge support; not used to define any SSSP variant"
adds_ppopp21: "226 graphs (Lonestar+SuiteSparse, explicit size/reachability selection criterion), single source per graph, 8-run mean, dynamic runtime Δ-selection (paper's central contribution, ablated), exact per-vertex distance-array correctness check, metric=time+work-efficiency (no GTEPS), baselines forced onto a shared Δ formula for fairness, real artifact on Zenodo"
khanda_tpds22_proxy: "PROXY EVIDENCE via Srinivasan et al. precursor (github.com/DynamicSSSP/HIPC18, not the track_inputs.json artifact_url which is a misattributed 3rd-party class project): RMAT+3 SNAP graphs, fixed source vertex 0 (sensitivity-checked), 4-repetition timing, proof-based correctness (no distance diff), metric=wall time explicitly NOT TEPS (reasoned rejection), baseline=Galois static Δ-stepping recompute, CPU-only 36-core Xeon, dedicated asynchrony-level parameter study"
graphit_cgo20: "named real graphs w/ 3 explicit weight regimes (logn/rand1000/origweights) tied to every result, Δ hard-coded per-graph-per-framework (all frameworks hand-tuned incl. baselines), 10 curated hang-avoiding source vertices averaged, metric=wall time+speedup (no GTEPS), baselines=GAPBS/Julienne/Galois/Ligra, CPU-only 24-core dual-socket"
```
