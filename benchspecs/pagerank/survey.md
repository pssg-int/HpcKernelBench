# PageRank track — evaluation-methodology survey

Surveyed 4/4 papers from `data/track_inputs/pagerank.json`. All 4 have no
arXiv id in the input file, but fulltext was still reached for all 4:
- GeraK23 (efg): the paper PDF ships **inside its own artifact repo**
  (`pgera/efg/gera-ipdps23.pdf`), downloaded via WebFetch and text-extracted
  with `pypdf` (WebFetch's own markdown conversion choked on the PDF stream).
- ChenC22 (Corder): no arXiv/OA link in the input; found an author-hosted
  mirror via web search, `https://www.cs.nthu.edu.tw/~ychung/Journal/2022-IEEE-TPDS.pdf`,
  same PDF->pypdf extraction.
- CZLH21 (MeLoPPR): the repo README links `arxiv.org/abs/2104.09616`
  (not present in the input JSON's `arxiv` field); fetched via
  `ar5iv.labs.arxiv.org/html/2104.09616` (clean HTML, no PDF extraction
  needed).
- Shun20 (hypergraph algorithms): the input's `oa_url`
  (`dl.acm.org/doi/pdf/...`) returned HTTP 403; found an NSF-PAGES
  open-access mirror via web search, `https://par.nsf.gov/servlets/purl/10137062`,
  PDF->pypdf extraction.

All 4 artifact repos were also inspected via `gh api repos/OWNER/REPO/contents`.

**Scope flag, read this before the variant design below**: only 2 of the 4
papers (GeraK23, ChenC22) measure standard, global, whole-graph PageRank the
way a benchmark reader would expect. Shun20 measures PageRank generalized to
**hypergraphs** (bipartite vertex<->hyperedge propagation, a different
computation). CZLH21/MeLoPPR measures **Personalized PageRank** (single
source, depth-truncated, approximate) — an algorithmically different problem
that happens to share the name "PageRank." Both are kept in the survey
(the task's own input file assigned them to this track, and both are useful
comparison points / cautionary examples) but neither drives the spec's core
variants; see Divergences and `open_questions`.

---

## 1. Traversing Large Compressed Graphs on GPUs (IPDPS 2023, "efg" / GeraK23)

Source: paper PDF `pgera/efg/gera-ipdps23.pdf` (fulltext extracted); repo
`pgera/efg` (`README.md`, `src/main.cu`, `src/inst_trav.h`).

- **workloads/inputs**: same real directed/undirected web & social graph
  corpus used for the paper's BFS evaluation (Table II) — up to 20 named
  instances (soc-lj 4.85M v/68.9M e ... moliere-16 30.22M v/6.68B e), grouped
  social/web/other. **However**, the PageRank results (Fig. 11) are only
  shown as "(a) Small Graphs (b) Larger Graphs" bar charts with **no
  accompanying table naming which specific graphs appear** — unlike the BFS
  Table II. Treated as "a subset of the same corpus, exact members
  unconfirmed" (see `open_questions`).
- **timing protocol**: "PageRank runs are capped at **50 iterations**." No
  epsilon/convergence-tolerance is mentioned anywhere near this sentence —
  this reads as a pure fixed-iteration-count run, not an early-exit-on-
  tolerance-with-a-safety-cap. No warmup/repetition count specific to
  PageRank is stated (the paper's "100 random starts, averaged" convention
  is explicitly for BFS/SSSP, which have a root vertex; PageRank has none).
- **timing scope**: EFG compression (encoding CSR->Elias-Fano) happens once
  and its ratio/size is reported separately (Table II, Fig. 8); the 50
  capped iterations are the only thing timed for the PageRank number itself.
- **precision & correctness**: no explicit correctness check is described
  for PageRank output (contrast: BFS distances get a SHA hash for
  self-consistency in the code, `get_sha_sum` in `src/main.cu` — but that
  code path is BFS-only, not reused for PageRank).
- **metric**: GTEPS (same billion-traversed-edges/second unit used for
  BFS/SSSP in this paper).
- **baselines**: NVIDIA's cuGraph library provides the CSR-based PageRank
  ("we use cugraph ... [and] modified cugraph to support out-of-core
  processing in BFS and SSSP"); EFG is their own representation. Ligra+(TD)
  and CGR are explicitly **not** evaluated for PageRank ("CGR is not
  evaluated here as an implementation was not available") even though both
  are evaluated for BFS — an inconsistent baseline set across kernels within
  the same paper.
- **push/pull**: "In PageRank, all the nodes are active in each iteration...
  The PageRank value of a destination is updated atomically once the edge is
  decoded" — plain edge-centric **push** (source scatters to destination via
  atomics), no direction optimization, no frontier concept (unlike BFS in
  the same paper).
- **hardware**: Titan Xp (12 GiB) primary; V100 (32 GiB) for some scaling.
- **artifact/paper mismatch (important)**: the public repo `pgera/efg`
  contains **zero PageRank code**. `src/main.cu` only wires up a `BFS`
  object; a grep of every header (including the 196KB `inst_trav.h`) for
  "pagerank" returns nothing. The paper's PageRank numbers come from
  NVIDIA's cuGraph library plus an apparently-unreleased EFG-PageRank
  kernel — **the artifact cannot reproduce this paper's PageRank results at
  all**, only its BFS ones. `artifact_status: verified` in the input JSON
  is accurate for the repo's existence/BFS content, but misleading if read
  as "this artifact benchmarks PageRank."

---

## 2. Workload Balancing via Graph Reordering on Multicore Systems ("Corder", TPDS 2022, ChenC22)

Source: fulltext PDF (author mirror, see above); repo
`yuang-chen/Corder-TPDS-21` (`README.md`, `main.cpp`).

- **workloads/inputs**: 7 graphs (Table 4): urand (67.1M v/2.1B e, uniform
  random, Graph500 generator), kron (67.1M v/2.1B e, Graph500 Kronecker
  scale 23), pld (42.9M v/0.6B e, web hyperlinks), live (LiveJournal,
  4.8M v/68.5M e), wiki (18.3M v/0.2B e, hyperlinks), twitter (41.7M v/1.5B
  e), mpi (52.6M v/2.0B e, highly-skewed social/influence graph). `urand` is
  explicitly **excluded from the paper's own reported averages** ("its
  performance ... is not taken into account ... as our work focuses on the
  optimization of skewed graphs") — a self-disclosed cherry-pick.
- **applications benchmarked**: 6 total, including **Pagerank (PR)** and
  **Pagerank Delta (PR-d)** — a variant "where vertices will be deactivated
  when their rank updates are smaller than a threshold d" (numeric value of
  `d` is not stated in the accessible text). Two backends: GPOP
  (partition-centric CSR) and Ligra (vertex-centric).
- **timing protocol — critical detail**: Table 5's column header is
  literally **"Pagerank (1 iteration)"** — the reported seconds are the cost
  of a **single PageRank iteration**, not a full run to convergence (PR-d's
  column in the same table reports much larger numbers, e.g. 7.24s vs PR's
  1.23s on `urand`, consistent with PR-d running to its own delta-based
  stopping criterion rather than 1 iteration). Elsewhere the paper also
  profiles "L2 Cache Misses (in Billions) During **20 Iterations** of
  Pagerank" (Fig. 8) — i.e. the paper uses **both** a 1-iteration
  microbenchmark (Table 5) and a 20-iteration profiling run (Fig. 8) for
  different purposes, no single canonical iteration count.
  Text elsewhere states timing is "averaged over 5 rounds of execution."
- **timing scope**: reordering cost (the paper's actual contribution) is
  reported **separately** in Table 8 (mapping time + composing time, per
  method per graph), with an explicit amortization argument: "the mapping
  time of Corder can be amortized by 0.284 iteration of PR"; total
  reordering overhead "can be roughly amortized by 2 or 3 iterations of PR."
  This 1-iteration-of-PR-as-a-unit-of-cost convention is used repeatedly as
  the paper's own yardstick for amortization.
- **precision & correctness**: no rank-vector correctness check described
  anywhere (deterministic algorithm, no tolerance/reference discussed).
- **metric**: wall-clock seconds (Table 5); `speedup = orig_time /
  reordered_time` for the reordering comparison, not GTEPS/TEPS.
- **baselines (reordering methods)**: original order, Random (RND), HC,
  FBC, DBG, Sort, Gorder, and Corder/fastCorder — the comparison axis is
  reordering method, with PageRank as one of 6 downstream kernels used to
  *measure the reordering's effect*, not a PageRank-algorithm contribution
  in its own right.
- **push/pull**: GPOP is "partition-centric," described as CSR-partitioned
  processing; the paper does not state whether it is push (scatter) or pull
  (gather) at the partition level. For the Ligra backend: "A key feature of
  Ligra-based PageRank is the **atomic update** of data" — atomics on
  destination updates is characteristic of a push/scatter kernel; no
  sparse/dense direction-optimization switch is mentioned for PageRank
  specifically in this paper (unlike Shun20, which discusses it explicitly).
- **hardware**: Intel Xeon Silver 4210 (Skylake), 20 threads (CLI default),
  1MB (1024 KB) partition size (CLI default) — README states GPOP's optimal
  partition size is tuned per micro-architecture/dataset but the paper fixes
  it to "L2 cache size" as a heuristic borrowed from prior work.

---

## 3. MeLoPPR: Software/Hardware Co-design for Memory-efficient Low-latency Personalized PageRank (DAC 2021, CZLH21)

Source: arXiv 2104.09616 fulltext via `ar5iv.labs.arxiv.org/html/2104.09616`;
repo `sharc-lab/MeLoPPR` (`README.md`, `PPR.py`, `toy_example.py`).

- **problem statement — divergence flag**: this is **Personalized PageRank**
  (single source vertex, locally truncated via bounded-depth BFS
  subgraphs), not global whole-graph PageRank. It is included in this
  track's input file but is a materially different computation from
  GeraK23/ChenC22's global PageRank — see Divergences.
- **workloads/inputs**: 6 real graphs (paper's Table II, via ar5iv
  extraction): citeseer (3,327 v/4,676 e), cora (2,708 v/5,278 e), pubmed
  (19,717 v/44,327 e), com-amazon (334,863 v/925,872 e), com-dblp (317,080
  v/1,049,866 e), com-youtube (1,134,890 v/2,987,624 e). The repo ships only
  the first 3 in `datasets/`; the other 3 must be downloaded separately from
  SNAP. All 6 are 2-4 orders of magnitude smaller than GeraK23/ChenC22's
  graphs (largest here, com-youtube, is smaller than the *smallest* graph
  either of those two papers uses).
- **algorithm parameters**: damping **alpha = 0.95** (paper text, via
  `PPR.py`'s hardcoded `alpha = 0.95` — this is the ONLY paper of the 4 that
  states a concrete numeric damping value, and it is notably higher than
  the textbook 0.85, motivated by PPR's locality use case). Stage
  decomposition: paper states "we let **k=200, L=6, and l1=l2=3**"; the
  repo's `PPR.py` instead hardcodes `k = 100`, `max_L = 6`, `l1 = l2 = 3` —
  a paper-vs-repo mismatch on `k` (top-k size), flagged as an open
  question.
- **timing protocol**: no explicit warmup/repetition/statistic language for
  *timing* was found; precision-vs-latency tradeoff numbers are "averaged
  from 1000 random runs" (sparsity analysis) and "averaged from 500 seed
  nodes for each graph" (precision-latency tradeoff) — i.e. repeated across
  many distinct source vertices/queries, not repeated timing trials of one
  fixed query. `PPR.py` ships commented-out `time.time()` calls around the
  BFS-subgraph-extraction and diffusion phases separately, suggesting the
  authors did split "BFS subgraph time" from "diffusion time" during
  development, but this instrumentation is disabled in the released code.
- **timing scope**: not clearly disentangled from the released code (timing
  instrumentation is commented out); the paper's headline latency numbers
  presumably include both the BFS-subgraph extraction and the diffusion
  computation.
- **precision/correctness — NOT L1 distance**: `top_k_precision(list_a,
  list_b, k)` = `|top_k(a) ∩ top_k(b)| / max(|top_k(a)|, |top_k(b)|)` — a
  **top-k SET-overlap ratio**, not an L1/L2 distance between rank vectors.
  Moreover the "ground truth" (`scores_org`/"Global PPR" in the code) is
  itself only computed on a depth-`max_L`-truncated BFS subgraph, not a
  fully converged infinite-iteration PPR vector — so even the reference is
  approximate.
- **metric**: speedup vs `LocalPPR-CPU` (single-stage baseline, same
  software stack); memory savings 1.5x-13.4x (CPU) / 73x-8699x (FPGA);
  speedup up to 15x (CPU) / 707x (FPGA) at ~80% precision, up to 70x (FPGA)
  at ~90% precision.
- **hardware**: CPU = "desktop with Intel i7 core, 2.8GHz, 16GB memory" (no
  further detail); FPGA = Xilinx Kintex-7 KC705 @ 100MHz. Software: Python +
  NetworkX (explicitly a research prototype, not a production-grade kernel);
  memory measured via Python's `tracemalloc`.
- **push/pull**: not the classic graph-traversal push/pull dichotomy. Each
  stage materializes a bounded BFS subgraph, builds a dense row-normalized
  propagation matrix `W` via `networkx.to_scipy_sparse_matrix` /
  `scipy.sparse`, then does dense-ish matrix-vector diffusion (`S_c * W`)
  for `L` steps within that subgraph — closer to a "gather"/pull style
  computation once the local subgraph is fixed.

---

## 4. Practical Parallel Hypergraph Algorithms (PPoPP 2020, Shun20)

Source: fulltext PDF via NSF-PAGES mirror (see above). **Note on the
artifact URL**: the input JSON's `artifact_url`
(`DevSheth/prallel-hypergraph-algorithms`) is a **student CUDA
reimplementation for a GPU-programming course project**, explicitly citing
Julian Shun's paper as `[1]` and the paper's own official repo
(`jshun/ligra`) as `[2]`. It is NOT the paper author's own artifact — a
third-party-reimplementation pattern the project's `CLAUDE.md` traps list
already warns about for artifact-search generally.

- **workloads/inputs (paper's Table 2, own hypergraph datasets, not the
  student repo's)**: com-Orkut, Friendster, Orkut-group, Web, LiveJournal
  (real, from SNAP community data / KONECT bipartite graphs) plus 3
  synthetic random hypergraphs Rand1 (10^8 vertices/hyperedges, cardinality
  10), Rand2 (10^9, cardinality 10), Rand3 (10^7, cardinality 100). These
  are hypergraph inputs (vertex-hyperedge incidence), not directly reusable
  for plain-graph PageRank without degenerating every hyperedge to a
  2-vertex edge.
- **algorithm**: PageRank generalized to hypergraphs via 2-step bipartite
  propagation (paper Eqs. 3-4: vertices -> hyperedges via `HyperedgeProp`,
  hyperedges -> vertices via `VertexProp`), damping factor stated only
  symbolically ("a damping factor 0 <= alpha <= 1"), no concrete numeric
  value given in the accessible text (Ligra's own convention, alpha=0.85,
  is the likely default but unconfirmed for this specific implementation).
- **timing protocol**: "PageRank times are for **1 iteration**" (same
  per-1-iteration convention as ChenC22/Corder's Table 5). Sequential (T1,
  single-thread) vs 72-core-with-hyperthreading (T72h) times reported, plus
  parallel speedup SU = T1/T72h (Table 3), e.g. com-Orkut: T1=3.31s,
  T72h=0.083s, SU=39.9x.
- **timing scope**: preprocessing (hypergraph construction/loading) is not
  discussed as part of the PageRank number specifically; the paper's own
  input generation for synthetic hypergraphs is one-time setup.
- **push/pull — the most explicit of the 4 papers**: direction optimization
  is a first-class feature — "we plot the running time using all sparse
  traversals, all dense traversals, and hybrid traversals with the default
  threshold of **1/20 fraction** of the sum of in-degrees ... for
  `VertexProp` and sum of in-degrees ... for `HyperedgeProp`." Figures 4-5
  show hybrid is fastest-or-tied across all 8 algorithms (including
  PageRank) on both tested graphs (com-Orkut, LiveJournal).
- **precision/correctness**: not discussed for PageRank's numeric output;
  the paper's validation is about performance/speedup, not correctness
  tolerance.
- **metric**: wall-clock seconds + parallel speedup (T1/T72h).
- **hardware**: 72-core Dell PowerEdge R930, 4x 18-core Xeon E7-8867 v4 @
  2.4GHz (45MB cache each), 1TB RAM, Cilk Plus + g++ 5.5.0 `-O3`,
  `numactl -i all` for NUMA balancing across sockets.

---

## Divergences

1. **Fixed-iteration vs to-convergence timing.** All 4 papers use a
   fixed-iteration (or fixed-depth) protocol, and **none** run PageRank to
   an epsilon-based convergence with a stated tolerance: GeraK23 caps at 50
   iterations (no tolerance mentioned); ChenC22/Corder's main PageRank
   number is for exactly **1** iteration (with a separate 20-iteration
   profiling figure elsewhere in the same paper — 2 different fixed counts
   used for 2 different purposes within one paper); Shun20's PageRank
   number is also for exactly **1** iteration; MeLoPPR (a different
   problem, PPR) fixes a BFS-subgraph *depth* (L=6) rather than an
   iteration count. ChenC22's "Pagerank Delta" variant IS threshold-driven
   (vertices deactivate below a delta), but the paper never discloses the
   numeric threshold, so it cannot be treated as a disclosed
   to-convergence baseline either. **Resolution**: the spec defines BOTH a
   fixed-iteration variant (k=20, splitting the difference between the
   1-iteration microbenchmark convention and Corder's own 20-iteration
   profiling convention) AND a to-convergence variant with an explicit
   epsilon=1e-6 — the latter has **no literature precedent in this track**
   and is a spec-level fix to a real gap in the surveyed practice, not a
   codification of existing practice.

2. **Per-iteration vs total-time reporting.** ChenC22 and Shun20 both
   report **per-1-iteration** time as their headline number (useful for
   amortization arguments, but not directly a "how long until I have a
   usable rank vector" number). GeraK23 reports **total** time over its
   50-iteration cap (no per-iteration breakdown given). MeLoPPR reports
   **total** latency for its whole depth-L pipeline. **Resolution**: the
   spec's fixed-iteration variant reports BOTH total time for k=20
   iterations AND the derived per-iteration time (total/k), so results are
   comparable to both reporting conventions without re-running anything.

3. **Push/pull/direction disclosure.** Only Shun20 explicitly discloses and
   quantifies direction optimization (sparse/dense/hybrid, 1/20 threshold).
   GeraK23 is explicitly push-only (no frontier, atomic destination
   update). ChenC22/Corder's Ligra backend uses atomics (push-like) but
   never states whether direction optimization is used for PageRank
   specifically; its GPOP backend's direction is undisclosed entirely.
   MeLoPPR's local dense diffusion doesn't map onto push/pull at all.
   **Resolution**: the spec makes direction disclosure a REQUIRED protocol
   field for every submission (push-CSR-atomic / pull-CSC-gather / hybrid
   with stated switch threshold), following the same principle the
   sibling `bfs` benchspec already adopted for this exact reason (direction
   optimization alone can be a 5-15x effect, larger than many papers'
   actual algorithmic contribution).

4. **Graph suite: no fully shared instance, and 3 different scales.**
   GeraK23 and ChenC22 partially overlap (both have a "twitter" and a
   LiveJournal-family graph, both use Graph500 kron/urand-style synthetic
   generators) but at different exact vertex/edge counts under the same
   name — the identical trap already documented in the `bfs` benchspec's
   survey for the same two papers (twitter: 41.6M v/1.47B e in GeraK23 vs
   41.7M v/1.5B e in ChenC22 — almost certainly the same underlying crawl
   with different preprocessing, NOT proven identical). MeLoPPR's graphs
   are 2-4 orders of magnitude smaller (largest is 1.1M vertices vs the
   other papers' 3-105M). Shun20's inputs are hypergraphs, not directly
   comparable at all. **Resolution**: the spec fixes a disambiguated
   named-instance list (vertex/edge counts pinned) for the global-PageRank
   variants, reuses the `bfs` benchspec's disambiguation of the
   twitter/LiveJournal family where the same underlying graphs are
   involved, and explicitly carves out a separate small-graph tier
   (MeLoPPR's citeseer/cora/pubmed + com-amazon/com-dblp/com-youtube) for
   cheap correctness-reference validation rather than throughput
   measurement.

5. **Damping factor.** Only MeLoPPR states a concrete value (alpha=0.95,
   for the different PPR problem). The 2 global-PageRank papers
   (GeraK23, ChenC22) never state a numeric alpha; Shun20 states only the
   symbolic constraint 0<=alpha<=1. **Resolution**: the spec fixes
   damping=0.85 (the original Page & Brin constant, and the de facto
   default in every major graph library — cuGraph, Ligra, NetworkX) for
   all global-PageRank variants, explicitly noting that this value is
   *assumed*, not drawn from any of the 4 papers' stated numbers, since
   none of the 2 relevant papers discloses one.

6. **Correctness validation.** None of the 4 papers validates PageRank's
   numeric output against an independent reference implementation with a
   stated tolerance. GeraK23 validates BFS via a self-consistency hash but
   this machinery isn't reused for PageRank. ChenC22 and Shun20 don't
   discuss correctness for PageRank at all. MeLoPPR's "precision" metric
   (top-k set overlap against its own depth-truncated approximate
   reference) is a genuinely different notion from a rank-vector distance
   and, being self-referential (ground truth is also approximate), doesn't
   establish absolute correctness either. **Resolution**: the spec adopts
   the L1-distance-vs-fp64-converged-reference gate the task instructions
   specify, computed once per graph via an independent trusted
   implementation (e.g. `scipy`/`networkx` power iteration run to a tight
   epsilon=1e-12) — this is a genuine gap-fix, not a codification of
   existing practice, and should be flagged as such wherever the spec is
   cited against these 4 papers' own reported numbers.

## Open questions

- GeraK23's PageRank figure (Fig. 11, "(a) Small Graphs (b) Larger Graphs")
  has no accompanying table naming the exact graph instances used, unlike
  its BFS Table II. Which of the ~20 Table II graphs are actually included
  in the PageRank evaluation is unconfirmed from the accessible text.
- GeraK23's damping factor and any convergence tolerance for its
  cuGraph-based PageRank calls are never stated; cuGraph's own default
  (alpha=0.85) is assumed, and it is unclear whether the "50-iteration cap"
  is a pure fixed count or a max-iteration safety net layered on a
  tolerance check that just never triggered in their experiments.
- ChenC22/Corder's "Pagerank Delta" numeric threshold `d` was not found in
  the accessible fulltext; would need the actual GPOP/Ligra source
  (`corder` binary or `ligra-patch/`) to pin the exact value used.
- MeLoPPR's paper states k=200 (Sec. on parameters) but the released
  `PPR.py` hardcodes k=100 — unclear which is the paper's actual reported
  configuration; would need the paper's full results tables (not fully
  extracted via ar5iv in this pass) cross-checked against script defaults.
- Shun20's concrete damping-factor value for its own PageRank experiments
  (Table 3) is not stated in the accessible text (only the generic
  algorithm-definition formula is given); Ligra's own convention
  (alpha=0.85) is assumed but unconfirmed for this specific
  hypergraph-PageRank implementation.
- Whether GeraK23's and ChenC22's same-named "twitter" and
  LiveJournal-family graphs are bit-identical snapshots or independently
  collected/preprocessed graphs sharing a name was not resolved (same open
  question already flagged in the sibling `bfs` benchspec survey for these
  same two papers) — the recommended_subset below treats them as the same
  disambiguated instance where sizes are close enough to plausibly be the
  same underlying crawl, but this should be verified before treating
  cross-paper PageRank numbers on "twitter" as bit-for-bit comparable.

## evidence

- `gerak23_ipdps23`: "PageRank capped at 50 iterations (fixed, no epsilon
  stated), GTEPS metric, push-only (atomic destination update, no
  frontier), baseline = NVIDIA cuGraph (modified for out-of-core), no
  correctness check described, no damping factor stated; artifact repo
  ships BFS only — zero PageRank source in `pgera/efg`."
- `chenc22_corder_tpds22`: "headline PageRank number is for exactly 1
  iteration (Table 5), separately profiles 20 iterations for cache misses
  (Fig. 8); PR-Delta variant is threshold-driven but threshold value
  undisclosed; wall-clock seconds + unitless speedup vs original vertex
  order, not GTEPS; reordering cost reported separately with an explicit
  'N iterations of PR to amortize' argument; Ligra backend uses atomics
  (push-like), GPOP backend's direction undisclosed; no damping factor or
  correctness check stated."
- `czlh21_meloppr_dac21`: "Personalized (not global) PageRank; damping
  alpha=0.95 (the only paper of the 4 with a stated numeric damping value);
  correctness = top-k SET-overlap precision against a depth-truncated
  approximate reference, NOT an L1/L2 rank-vector distance; graphs 2-4
  orders of magnitude smaller than the other 3 papers (max 1.1M vertices);
  paper states k=200 but released code hardcodes k=100."
- `shun20_hypergraph_ppopp20`: "PageRank generalized to hypergraphs
  (bipartite propagation, not plain-graph PageRank); PageRank time is for
  1 iteration (same per-1-iteration convention as Corder); ONLY paper of
  the 4 to explicitly disclose and quantify direction optimization
  (sparse/dense/hybrid, 1/20 default switch threshold); reports sequential
  vs 72-core-hyperthreaded time + speedup, not GTEPS; damping factor given
  only symbolically, no numeric value stated; input artifact_url in this
  track's data is a third-party student CUDA reimplementation, not the
  paper author's own repo (`jshun/ligra`)."
