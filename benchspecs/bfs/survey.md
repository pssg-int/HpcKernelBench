# BFS track — evaluation-methodology survey

Surveyed 7/7 papers from `data/track_inputs/bfs.json`. Fulltext or artifact
source code was reachable for 5 papers (BLEST, Graph500-BFS/Fugaku, efg,
Corder, Speculative-RCM); FuseIM and Bit-GraphBLAS were only reachable at
abstract + repo-README depth (ACM DL fulltext returned HTTP 403, no arXiv
preprint, IEEE Xplore paywalled).

---

## 1. BLEST — Blazingly Efficient BFS using Tensor Cores (ICS 2026)

Source: arXiv fulltext, `https://arxiv.org/html/2512.21967` (arXiv 2512.21967).

- **workloads/inputs**:
  - GAP Benchmark Suite (Beamer et al.): GAP-road (23M v / 57M e), GAP-twitter
    (61M v / 1.4B e), GAP-web (50M v / 1.9B e), GAP-kron (134M v / 4.2B e),
    GAP-urand (134M v / 4.2B e).
  - Custom large-graph suite, selection criterion explicit: SuiteSparse
    matrices with `|V| >= 23M and |E| <= 2^32-1` — nlpkkt240, uk-2005,
    it-2004, europe_osm, com-Friendster, Spielman_k600, webbase-2001,
    kmer_V1r, mawi.
- **source-vertex selection**: "All timings are averaged over 64 BFSs,
  executed from the same set of randomly selected roots." — the canonical
  Graph500 convention (64 random roots), but note: the SAME 64 roots are
  reused across all competing implementations on a given graph (fair), and
  the statistic taken is the **arithmetic mean**, not the Graph500-standard
  harmonic mean.
- **timing protocol**: 64 repetitions, arithmetic mean, no warmup iteration
  described. Timer not specified precisely (host-side wall clock around
  device kernels, inferred).
- **timing scope**: search (traversal) time only; graph-format/ordering
  preprocessing ("BVSS" bitmap construction) is measured and reported
  **separately** — Table 1(a) gives per-graph ordering/preprocessing time in
  seconds (e.g., 15.49 s for vsp_msc-type inputs), never folded into the
  per-search number.
- **precision & correctness**: binary/bitmap adjacency (no floating point);
  paper does not discuss an explicit correctness-validation procedure
  (no parent-tree or level check described in the reachable text).
- **metric**: wall-clock **time in milliseconds**. No GTEPS/TEPS number is
  reported anywhere in the paper — a real divergence from the rest of the
  track.
- **baselines**: GAP (CPU direction-optimizing BFS), Gunrock (GPU), GSWITCH
  (GPU BFS autotuner), BerryBees (bit-tensor-core BFS). Speedups: 3.58-4.9x
  over these baselines depending on suite.
- **hardware**: 2x Intel Xeon Gold 6152 + NVIDIA H200 (141 GB).

---

## 2. Doubling Graph Traversal Efficiency to 198 TeraTEPS on Fugaku (SC 2024)

Source: artifact repo `RIKEN-RCCS/Graph500-BFS` — `README.md`,
`mpi/parameters.h`, `run-benchmark.py` (no arXiv/OA fulltext; paper is behind
IEEE Xplore, but the repo IS the reference Graph500 implementation extended
with the paper's 4 techniques, so the code is authoritative for protocol).

- **workloads/inputs**: standard Graph500 R-MAT/Kronecker generator, governed
  by a single `SCALE` parameter; `DEFAULT_EDGE_FACTOR = 16` (`mpi/parameters.h`).
  Graph is regenerated from the seeded generator at each SCALE, never a fixed
  downloaded file — this is the "synthetic Graph500 RMAT scale" axis.
- **source-vertex selection**: `mpi/parameters.h` defines `REAL_BFS_ROOTS 64`
  and `TEST_BFS_ROOTS 16`; the README states explicitly: *"specifying `-n 64`
  is necessary to comply with the Graph500 specification, and all the
  results in the paper were obtained using this setting."* The 16-root
  constant is only a reduced quick-test mode, never used for reported
  numbers.
- **timing protocol**: no separate warmup phase; graph construction happens
  once per SCALE, then 64 BFS searches are timed individually; the reported
  statistic is the **harmonic mean TEPS** across all 64 searches (printed as
  `harmonic_mean_TEPS`) — this is the official Graph500 statistic, distinct
  from BLEST's arithmetic mean.
- **timing scope**: search-phase only for the TEPS number. Preprocessing
  (forest pruning `-C`, group reordering / `VERTEX_REORDERING` modes 0-2,
  multilevel bitmap compression) happens once before the 64 searches and is
  reported as separate one-time cost, but its purpose is explicitly to
  *reduce memory footprint per node* so more of the graph fits — i.e. this
  preprocessing changes what's even measurable, not just speed.
- **direction optimization**: YES — hybrid top-down/bottom-up with tunable
  switch thresholds `DEFAULT_ALPHA = 10.0` (top-down→bottom-up) and
  `DEFAULT_BETA = 14.0` (bottom-up→top-down); the paper's 4th technique,
  "adaptive parameter tuning" (`-A`), auto-tunes these thresholds online.
- **precision & correctness**: `DEFAULT_VALIDATION_LEVEL = 2` — full
  Graph500-standard validation (BFS-tree is a valid spanning tree of the
  reached component; parent-child edges exist in the original graph; levels
  are consistent) is performed for each of the 64 searches before its TEPS
  counts.
- **metric**: **GTEPS via harmonic mean over 64 searches**; the paper's
  headline "198 TeraTEPS" is this number aggregated across 152,064 Fugaku
  nodes.
- **baselines**: prior Fugaku/K-computer Graph500 submissions (their own
  earlier work), i.e. this is the reference/gold-standard implementation of
  the Graph500 benchmark itself, not a paper benchmarking against a 3rd-party
  library.

This paper is effectively **the canonical definition of the Graph500 BFS
protocol** for this track — the other papers should be read as variations
against this baseline.

---

## 3. Traversing Large Compressed Graphs on GPUs (IPDPS 2023, "efg")

Source: artifact repo `pgera/efg` — `README.md`, `src/main.cu` (fulltext PDF
also ships in-repo as `gera-ipdps23.pdf` but text extraction was garbled;
relied on README + driver code, which is precise about the protocol anyway).

- **workloads/inputs**: real directed/undirected web & social graphs — twitter
  (41.6M v / 1.47B e), gsh-h-15 (68.66M v / 1.8B e), frndster (65.61M v /
  3.61B e), uk-07-05 (105.22M v / 3.74B e), kron27_s (63.07M v / 4.22B e);
  larger UVM-only graphs molr16 (30.22M v / 6.68B e) and uk-75_s (105.22M v /
  6.62B e). Larger set hosted externally (p.gera.io); small `sample_graphs/`
  in-repo explicitly flagged "meaningless for performance measurements."
- **source-vertex selection**: `-n 100` (default **100 traversals**), roots
  drawn uniformly at random via a **fixed-seed** `std::mt19937_64 eng(0)` —
  deterministic across runs — or a single fixed `-r <root>`.
- **timing protocol**: exactly **1 warmup traversal**
  (`bfs.traverse(roots_vec[0])` called once before the timed loop), then all
  100 root entries (including index 0, re-executed) are timed individually
  and accumulated. No explicit statistic name is printed beyond per-traversal
  numbers and a **"Total GTEPS"** aggregate.
- **timing scope**: CSR→Elias-Fano compression (`EFLayout`) happens once at
  startup and is reported separately (`CSR size`, `EF size`, `Compression
  ratio` printed once, not part of per-traversal timing). Only the traversal
  itself is timed via `bfs.get_last_elapsed_time()`.
- **precision & correctness**: no floating point; a SHA hash of the per-run
  distance array is printed for cross-run/cross-ordering consistency
  checking, but there is no comparison against an independent CPU reference
  — correctness is "the same traversal hashes the same," not "matches ground
  truth."
- **metric**: `GTEPS = visited_edges * 1e-6 / elapsed_time_ms` per traversal,
  plus an aggregate `Total GTEPS = total_visited_edges * 1e-6 / total_time_ms`
  — this aggregate is an **edge-count-weighted** average across the 100
  traversals, not a simple mean of the 100 per-traversal GTEPS values (and
  not harmonic mean either — a third distinct statistic definition in this
  track).
- **baselines**: comparison is primarily against uncompressed CSR baseline
  (same BFS kernel, no compression) to isolate the effect of Elias-Fano
  compression + UVM.
- **hardware**: NVIDIA Titan Xp (12 GiB) and V100 (32 GiB).

---

## 4. Bit-GraphBLAS: Bit-Level Optimizations of Matrix-Centric Graph Processing on GPU (IPDPS 2022)

Source: artifact repo `hsung2/Bit-GraphBLAS`, `bmv_eval/testbmv.sh` +
directory listing (fulltext behind IEEE paywall, no arXiv; abstract and a
DeepAI/OSTI summary consulted via web search — **tier-3 source**, flagged as
such).

- **workloads/inputs**: the public artifact ships only the BMV (Boolean
  Matrix-Vector multiply) microbenchmark — the primitive underlying
  GraphBLAS-based BFS/SSSP/PR/CC/TC, not a full BFS driver. Inputs: **~700
  named SuiteSparse Matrix Collection matrices**, downloaded on the fly from
  `suitesparse-collection-website.herokuapp.com`, spanning many SuiteSparse
  groups (HB, Pajek, DIMACS10, SNAP, Gset, Rajat, LAW, Chen, Andrianov,
  Boeing, ...) — essentially "most of SuiteSparse," no explicit nnz/size
  cutoff visible in the driver script.
- **timing protocol / preprocessing scope / correctness / metric for BFS
  specifically**: **could not be confirmed** from the reachable sources —
  the eval script sweeps a bit-block tile dimension (`TILEDIM = (4 8 16
  32)`) per matrix and per format variant (`bin_bin_full`, `bin_full_full`,
  `bin_bin_bin`, `baseline`), but repetition count / warmup / statistic are
  inside the compiled `bmv` binaries, not visible in the shell driver.
- **metric (paper-level, from abstract)**: reports speedups, not absolute
  throughput: "up to 40x and 6555x for ... SpMV and SpGEMM respectively,
  making GraphBLAS-based BFS accelerate up to 433x, SSSP, PR, and CC up to
  35x, and TC up to 52x" — all figures are **speedup-vs-baseline** framing,
  no GTEPS given for BFS in accessible sources.
- **baselines**: implied to be a standard GraphBLAS/cuSPARSE-based BFS
  (SuiteSparse:GraphBLAS-style), not independently confirmed.

This paper is the weakest-evidenced in the track; its inclusion in the spec's
`recommended_subset` selection criterion (real-world SuiteSparse matrices) is
useful, but its BFS-specific protocol facts are open questions (see below).

---

## 5. Workload Balancing via Graph Reordering on Multicore Systems ("Corder", TPDS 2022)

Source: fulltext PDF, `https://www.cs.nthu.edu.tw/~ychung/Journal/2022-IEEE-TPDS.pdf`
(author's own mirror; extracted via `pypdf`).

- **workloads/inputs**: 7 large graphs, explicit sizes (Table 4): urand
  (67.1M v / 2.1B e, Graph500-uniform-random generator), kron (67.1M v /
  2.1B e, **Graph500 Kronecker generator, scale 23**), pld (42.9M v / 0.6B e,
  web hyperlinks), live (LiveJournal, SNAP, 4.8M v / 68.5M e), wiki (18.3M v
  / 0.2B e, hyperlinks), twitter (Twitter follower graph, 41.7M v / 1.5B e),
  mpi ("Twitter influence" graph, 52.6M v / 2.0B e). `urand` is explicitly
  **excluded from the reported averages** because it is "not skewed," which
  the paper states outright — a self-disclosed cherry-pick worth flagging.
- **source-vertex selection**: **a single, fixed root** — "the hottest
  vertex" (globally highest-degree vertex in the graph) is used as the BFS
  (and SSSP) starting point for **every** graph, chosen explicitly "for the
  maximum activation." This is a clear divergence from the Graph500-style
  64-random-source convention used elsewhere in the track: a single
  maximum-degree root systematically over-represents best-case reachability
  and is not resistant to root-selection variance.
- **timing protocol**: "averaged over 5 rounds of execution" (Sec. 4.2); no
  warmup mentioned explicitly. Cache/memory activity measured with `perf`
  and `likwid`.
- **timing scope / preprocessing amortization** (Sec. 4.4, Table 8): the
  reordering cost itself is reported **separately**, in seconds, per
  reordering method (RND/sort/HC/DBG/Gorder/Corder) per graph — e.g. Gorder
  costs 66.8 s on one graph. The paper explicitly computes an **amortization
  argument**: "the average number of iterations of PR on original graphs
  needed to amortize [reordering cost]... 1.84 iterations on average" and
  separately notes lightweight per-run apps like BFS ("finishes within 3
  seconds") "can be roughly amortized by 2 or 3 iterations of PR" — i.e. the
  paper itself argues reordering cost should be amortized over a handful of
  runs, not hidden inside a single run's time.
- **6 graph applications benchmarked**: PageRank, PageRank-Delta, Connected
  Component, SSSP (Bellman-Ford), **BFS** (level-wise traversal from the root),
  Parallel Nibble.
- **metric**: `speedup = original_processing_time / reordered_processing_time`
  (wall-clock ratio), **not GTEPS/TEPS**.
- **precision & correctness**: not discussed (deterministic BFS/PR output;
  no explicit validation step described).
- **baselines (reordering methods)**: Original order, Random (RND),
  sort-by-degree, HC, DBG (Degree-Based Grouping), Gorder, and their own
  Corder / fastCorder.
- **hardware**: Intel Xeon Skylake, G++ 9.3.0 -O3, OpenMP.

---

## 6. FuseIM: Fusing Probabilistic Traversals for Influence Maximization on Exascale Systems (ICS 2024)

Source: abstract + `pnnl/ripples` artifact repo, `tools/batched_bfs.cc`,
`README.rst` publication list (**ACM DL fulltext returned HTTP 403**, no
arXiv preprint — tier-2/3 mixed source).

- **workloads/kernel**: not a plain-BFS paper — it fuses many concurrent
  **probabilistic** BFS instances ("Backward/Batched Probabilistic
  Traversals," BPTs) used as the Monte-Carlo RRR-set-sampling step of
  Independent-Cascade influence maximization. Each traversal follows an
  edge only with probability = edge weight (`WeightedDestination<uint32_t,
  float>`), not deterministically — a materially different kernel from
  plain reachability BFS, though built on the same frontier/CSR machinery.
- **source-vertex selection**: `tools/batched_bfs.cc` (a repo dev-tool, not
  necessarily identical to the paper's exact harness) draws a **batch of
  root vertices** (8 in this tool; the paper's whole contribution is fusing
  a much larger number of concurrent traversals — "varying numbers of colors
  and traversal[s]" per a figure caption found via search) uniformly at
  random via a **fixed-seed** `trng::lcg64` generator (`seed(0UL)`).
  Deterministic, reproducible root selection. The GPU-BFS kernel itself lives
  behind a `#if RIPPLES_ENABLE_CUDA || RIPPLES_ENABLE_HIP` compile-time
  switch, i.e. GPU BFS is an optional backend and a CPU path also exists.
- **hardware/scale**: evaluated up to **4096 nodes of OLCF Frontier**
  (32,768 GPUs, 196K CPU cores), per available secondary sources.
- **metric**: speedup of fused-vs-unfused traversal: "up to 182.13x (avg.
  75.15x)" for gIM and "up to 359.86x (avg. 135.17x)" for Ripples — again a
  **speedup framing**, not an absolute GTEPS number.
- **timing protocol, exact graph datasets/sizes, warmup/repetition count,
  preprocessing scope**: **could not be confirmed** — ACM DL fulltext
  blocked (403), no arXiv mirror found. Flagged as open question.
- **baselines**: unfused Ripples and unfused gIM (i.e., the paper's baseline
  is "the same framework without traversal fusion," an internal ablation,
  not a 3rd-party BFS library).

---

## 7. Speculative Parallel Reverse Cuthill-McKee Reordering (IPDPS 2021) — tangential

Source: artifact repo `GPUPeople/ParallelBatchRCM`, `README.md` only
(light-depth survey; this paper is **not primarily a BFS-throughput paper**
and is included in the track because RCM's core subroutine is a BFS).

- **role of BFS**: RCM reordering computes a bandwidth-reducing permutation
  by (1) selecting a pseudo-peripheral start vertex and (2) performing a
  level-structure BFS from it, then sorting each level by degree. The paper
  benchmarks the *reordering* (RCM) itself, not raw BFS traversal throughput.
- **workloads/inputs**: SuiteSparse Matrix Collection (CSR quadratic/square
  matrices); repo ships a `get_sample_matrices.py` helper for a small demo
  set — full paper-scale matrix list not confirmed from the README alone.
- **implementations compared**: `cuSolverRCM` (vendor baseline), `CPU`,
  `CPU_BATCH` (1-24 threads), `GPU`, `GPU_BATCH` — i.e. baselines are a
  vendor library (cuSolver) plus their own sequential/batched variants,
  run on a single quadratic matrix at a time. Their own claim (per the JSON
  `one_liner`): "orders of magnitude faster than cuSolver."
- **metric**: reordering wall time, plus optional resulting bandwidth
  (`-w` flag reports post-reordering matrix bandwidth) — not TEPS.

Given the tangential role, this paper does NOT define a spec variant on its
own; it is evidence for the "preprocessing sometimes silently contains a
full BFS pass" pattern the spec's preprocessing-scope discipline must guard
against (see spec.yaml `notes_on_fairness`).

---

## Divergences

The 7 papers disagree on essentially every axis the instructions ask to
resolve:

1. **Source-vertex selection.** Graph500-BFS/Fugaku and BLEST both use **64
   random roots** (the official Graph500 convention) — but BLEST reduces
   with arithmetic mean while Graph500 uses harmonic mean. efg uses **100**
   random roots (fixed-seed, deterministic) with an edge-weighted aggregate
   GTEPS. Corder uses a **single, fixed, maximum-degree** root — a
   materially different and more favorable selection than random sampling.
   FuseIM (probabilistic BFS) batches a variable, paper-tunable number of
   random roots. **Resolution**: the spec standardizes on 64 random roots,
   fixed-seed, reporting median (not either mean) plus min/max, since a
   single "hottest vertex" (Corder-style) is not a fair generalizable
   sample and a fixed seed is needed for reproducibility across
   implementations.

2. **GTEPS / statistic definition.** Three distinct definitions appear in
   the track: (a) Graph500/Fugaku's harmonic mean of per-search TEPS over 64
   searches (the official spec); (b) efg's edge-count-weighted aggregate
   `sum(edges)/sum(time)`; (c) BLEST doesn't report GTEPS at all, only ms.
   Corder reports neither — only a unitless speedup ratio. **Resolution**:
   the spec fixes GTEPS = (edges connecting BFS-tree vertices) / time,
   reported as median-of-per-search-GTEPS (not harmonic or weighted mean, to
   avoid a few fast/slow outlier searches dominating), with min/max also
   reported; raw time in ms is always reported alongside so BLEST-style
   papers remain comparable.

3. **Preprocessing / reordering scope.** BLEST and efg both report
   preprocessing (bitmap layout construction / Elias-Fano compression) as a
   one-time, separately-reported cost, never amortized into the per-search
   number — good practice, adopted by the spec. Graph500-BFS bundles
   "forest pruning" + "group reordering" as one-time graph-construction-time
   steps, also separate. Corder is the odd one out: it explicitly computes
   an **amortization argument** (reordering cost ÷ cost-per-run, "2-3
   iterations" for BFS-like workloads) — i.e. it treats reordering as
   something that pays for itself after k runs, and reports k directly.
   RCM's BFS-as-subroutine is never separated from total reordering time at
   all (not measured as a distinct BFS number). **Resolution**: the spec
   keeps preprocessing/reordering strictly separate from kernel-only timing,
   but also defines an explicit end-to-end variant with the amortization
   count made a required, reported parameter (following Corder's own
   discipline) rather than folding it in silently.

4. **Direction optimization disclosure.** Graph500-BFS/Fugaku is explicitly
   a hybrid top-down/bottom-up (direction-optimizing) implementation with
   tunable alpha/beta thresholds, and its own baseline in the GAP suite (used
   by BLEST) is also direction-optimizing (the original Beamer GAP BFS).
   BLEST's own method is described as "pull-based ... bitmap-oriented" — it
   is not clear from the reachable text whether it also does a push
   (top-down) phase for early/sparse frontiers, which is exactly the
   optimization that gives direction-optimizing BFS its 10x+ headline
   speedups over naive top-down BFS. efg and Corder do not discuss direction
   optimization at all (their BFS is a means to an end — compression
   testing / reordering evaluation — not the object of comparison).
   **Resolution**: the spec requires every submission to disclose whether
   direction optimization is used and, if so, the switch thresholds/logic,
   since comparing a hybrid implementation's GTEPS against a naive top-down
   baseline (or vice versa) without disclosure is an apples-to-oranges
   comparison that has caused real inflation of reported speedups in this
   literature.

5. **Graph suite: synthetic vs. real, and no shared real-graph list.**
   Every paper that uses real graphs picks a different subset with no fully
   shared instance across all of them; only the *generator family* (Graph500
   R-MAT/Kronecker, edge factor 16, scale N) is shared infrastructure across
   BLEST's GAP-kron/GAP-urand, Corder's kron/urand, and Graph500-BFS's core
   generator. Real-world graph overlap is partial: twitter-family graphs
   appear in BLEST, efg, and Corder, but under different snapshots/sizes (61M
   v in BLEST's GAP-twitter vs. 41.6-41.7M v in efg's/Corder's twitter — these
   are NOT the same dataset despite the shared name, a trap worth flagging
   for anyone assembling the union suite). **Resolution**: the spec's
   `recommended_subset` names exact graphs with vertex/edge counts to
   disambiguate same-named-different-graph collisions, and separates the
   synthetic-generator variant (parameterized, exactly reproducible) from
   the real-graph variant (fixed named instances).

6. **Correctness validation.** Only Graph500-BFS/Fugaku specifies a formal
   validation level (full parent-tree + level-consistency check, official
   Graph500 `validation_level=2`). efg validates only *self-consistency* (a
   hash of the distance array, useful for comparing reorderings of the same
   graph, not for catching a wrong BFS implementation). BLEST, Corder, Bit-
   GraphBLAS, and FuseIM do not describe an explicit correctness check in
   the reachable sources. **Resolution**: the spec adopts Graph500's
   validation-level-2 check (or an equivalent CPU-reference-BFS parent-array
   comparison) as a correctness gate that must pass before any run counts,
   since BFS has no natural continuous-error-tolerance concept — a wrong
   parent/level is simply wrong.

## Open questions

- Bit-GraphBLAS's BFS-specific timing protocol (warmup/repetition/statistic,
  source-vertex convention, whether the 433x BFS speedup figure includes
  format-conversion time) could not be confirmed — fulltext is IEEE-paywalled
  and no arXiv mirror exists. The public artifact only ships the underlying
  BMV microbenchmark, not a full BFS driver script.
- FuseIM's exact evaluation graph list/sizes, warmup/repetition count, and
  whether its GPU-BFS timing includes host-device transfer could not be
  confirmed — ACM DL returned HTTP 403 and no arXiv preprint exists. The
  `tools/batched_bfs.cc` dev-tool in the artifact repo shows the API shape
  (batch of probabilistic BFS from random roots) but is not guaranteed to be
  literally the harness used for the paper's reported numbers.
- Graph500-BFS/Fugaku's exact R-MAT A/B/C/D parameters were not directly
  read from `generator/graph_generator.hpp` (only `user_settings.h` and
  `parameters.h` were inspected); the spec assumes the well-known official
  Graph500 defaults (A=0.57, B=0.19, C=0.19, D=0.05) rather than the
  paper-specific values, which should be verified against
  `mpi/graph_constructor.hpp` or the generator source before treating the
  synthetic-suite variant as bit-for-bit official-Graph500-compliant.
- None of the 7 papers report fp-precision or numerical-tolerance
  correctness criteria, which is expected for BFS (an integer/boolean
  kernel) but means the spec's correctness gate (exact parent/level match)
  has no literature precedent to cross-check tolerance choices against — it
  is set from first principles (BFS is a discrete algorithm; "close" is not
  meaningful), not from a paper's stated tolerance.
