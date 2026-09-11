# dynamic-graph-kernel track — evaluation survey

Track input: `data/track_inputs/dynamic-graph-kernel.json` (2 papers, both
surveyed via fulltext). **Note up front**: the two papers target genuinely
different problems that both fall under "dynamic graph kernel" in the
DBLP-derived classification, and the spec below has to reconcile that
rather than paper over it (see Divergences).

## conf/ppopp/Hou0H00YL00L026 — DTMiner: A Data-Centric System for Efficient Temporal Motif Mining (PPoPP 2026)

Source: full PDF fulltext, obtained from the University of Warwick
institutional repository open-access mirror
(`wrap.warwick.ac.uk/195922/7/3774934.3786416.pdf` — the record's
"Published Version," distinct from the `/1/` accepted-manuscript file
which requires institutional sign-on) + GitHub `yinling123/DTMiner`
(README.TXT, `run_all_motifs.sh`, `perf_all_motifs.sh`, `scalable.sh`).

- **What "dynamic" means here**: DTMiner mines a FIXED, already-complete
  historical log of timestamped edges (a temporal graph is loaded once,
  never mutated during a run). There is no insert/delete of edges at
  benchmark time. "Batch" in this paper means a fixed-size, LLC-sized
  CHUNK of the (already fully known) temporal edge list streamed through
  the Load-Explore-Synchronize (LES) execution model in chronological
  order — a cache-locality mechanism, not a live-update mechanism.
- **Workloads/inputs**: 4 real-world temporal graphs (Table 1), all with
  named public sources:
  - wikitalk (wi): 1,140,149 vertices, 7,833,140 temporal edges, avg degree
    6.87, 2,787,968 static undirected edges, 6.24-year span [SNAP].
  - stackoverflow (so): 2,601,977 v, 63,497,050 edges, avg degree 24.40,
    34,875,685 static edges, 7.6-year span [SNAP].
  - temporal-reddit-reply (re): 8,901,033 v, 646,044,687 edges, avg degree
    72.58, 435,290,421 static edges, 10.1-year span.
  - ethereum (eth): 66,323,478 v, 628,810,973 edges, avg degree 9.48,
    186,064,655 static edges, 3.58-year span.
  Query workload: 13 structurally diverse temporal motifs (M1-M13, 3-5
  edges each, Figure 11), each counted (not enumerated/listed — this is
  a COUNTING benchmark) under a per-dataset time-window delta: 1 day (24h)
  for wikitalk and stackoverflow, 10 hours for temporal-reddit-reply, 1
  hour for ethereum. The repo's own `scalable.sh`/`perf_all_motifs.sh`
  scripts confirm these exact delta values and motif file names (M1-M13
  in a `motifs/` directory) independently of the paper text.
  Note: the repository's dataset name for temporal-reddit-reply is not the
  synthetic-sounding label but a real Reddit-reply crawl.
- **Timing protocol**: hardware fixed to a single dual-socket server (two
  32-core Intel Xeon Platinum 8357B, 48MB LLC, 2.6GHz, 503GB DDR4, 16
  memory channels, Ubuntu 20.04, gcc 11.4.0 -O3, cmake 4.0.1). No explicit
  warmup/repetition-count language was found for the main speedup numbers
  (Figures 12-18 appear to be single-run per configuration, consistent
  with the very long absolute runtimes involved — e.g. up to 181.75s just
  for preprocessing on the largest graph, Table 2 — which makes many-rep
  averaging costly). The random-seed sensitivity of the filtering/hashing
  components is NOT separately studied (contrast with ECL-MST in the mst
  track, which explicitly runs 99 random seeds).
- **Timing scope**: preprocessing (chunking + hierarchical index
  construction) and mining time are measured and reported BOTH
  separately (Table 2: preprocessing time in ms, memory usage in MB, for
  DTMiner vs. baseline Mackey-o) AND combined end-to-end (Figure 18:
  normalized execution time split into preprocessing-time and
  mining-time stacked bars, yielding an explicit 2.13x average
  end-to-end speedup despite DTMiner's preprocessing costing 4.8%-90.7%
  MORE than the baseline's). This preprocessing-reported-both-ways
  practice is exactly the discipline this repo's other specs (e.g.
  `spmv/spec.yaml`) require and is a genuinely strong example of it.
- **Correctness**: motif COUNTING is exact/deterministic (no numeric
  tolerance applies) — the paper does not discuss an explicit
  cross-validation step against a reference count in the fetched text,
  but since the reported metric is a discrete count and speedups are
  computed relative to Mackey-o (itself an extension of a peer-reviewed
  exact algorithm), exact-count agreement is the implicit correctness
  bar.
- **Baselines**: Mackey-o — the paper's own in-house-hardened extension of
  Mackey et al.'s exact temporal-subgraph-isomorphism algorithm,
  incorporating optimizations from Everest (GPU) and Mint (accelerator)
  adapted back to CPU, built specifically to be the strongest available
  CPU baseline since Everest/Mint themselves target different hardware.
  Also compares an intermediate Mackey-M (indexing-only fix) and internal
  ablations (DTMiner-NH: no hash index; DTMiner-NS/-S1/-S2: work-stealing
  levels removed; DTMiner-without: memory-aware sync disabled).
- **Metric**: speedup vs. Mackey-o (1.14x-11.98x, 4.25x average across all
  13 motifs x 4 graphs), LLC misses (absolute and ratio vs. baseline,
  3.3%-60.5% of baseline's cache references), thread scalability (speedup
  vs. own single-thread baseline at 2/4/8/16/32/64 threads).
- **Hardware/software specificity**: NUMA-aware (three-level work stealing:
  intra-NUMA, inter-NUMA, cross-chunk), which is itself an experimental
  axis (Figure 15 isolates each level's contribution).

## journals/tpds/KhandaSBND22 — A Parallel Algorithm Template for Updating SSSP in Large-Scale Dynamic Networks (TPDS 2022)

Source: abstract + metadata via Semantic Scholar
(`DOI:10.1109/TPDS.2021.3084096`); the paper itself is IEEE-paywalled
(Bronze OA status pointing back to the paywalled DOI, not a free-standing
PDF) and no arXiv preprint exists. Supplementary evidence gathered from
two secondary sources: (1) a public GitHub notes gist summarizing the
paper (`gist.github.com/wolfram77/...`), and (2) the GitHub repo listed as
this paper's artifact, `mmasabalvi/parallel-sssp-update-mpi-openmp-metis`
— **which on inspection is an unofficial, independently-authored student
course-project reimplementation** (a "Project Report.pdf" inside the repo
identifies three named students with roll numbers submitting it as
coursework, not Khanda/Srinivasan/Bhowmick/Norris/Das themselves). This is
flagged explicitly because the track input JSON marks this artifact
`"artifact_status": "verified"`, but it should not be treated as
reproducing the paper's own reported numbers — only its general algorithm
family.

- **What "dynamic" means here**: genuinely live graph mutation — edges are
  inserted and deleted over time, and the SSSP tree/distance array must be
  kept correct incrementally rather than recomputed from scratch after
  every change. This is the "real" dynamic-graph-maintenance problem the
  track's stated key axes (batch sizes, insert/delete mix,
  update-throughput vs. query-latency, snapshot consistency) are written
  for.
- **Core algorithmic idea** (from the abstract, corroborated independently
  by the student repo's structurally identical description): identify the
  PORTION of the network affected by a batch of edge changes, and update
  only a "rooted tree data structure that stores the edges of the network
  that are most relevant to the analysis" (i.e. the SSSP tree itself plus
  enough auxiliary edges to detect when a shorter alternative path has
  appeared) rather than re-deriving the whole tree. Implemented on both
  shared-memory (OpenMP) and GPU platforms.
- **Workloads/inputs**: could not be confirmed from available sources
  ("real-world and synthetic networks" per the abstract, no further
  detail retrievable without the paywalled fulltext). The secondary gist
  notes an unresolved ambiguity in the original paper figures around
  "graphs for 50M, 100M changes" and separately "batch size 15, 30, 50" —
  the note author explicitly flags not being certain whether these are
  the same axis (total changes vs. per-batch size) or two different
  reported quantities. This is carried into open_questions below rather
  than guessed at.
- **Update generation / batch axis**: the paper implements and evaluates
  BOTH edge insertion and edge deletion, with the secondary source noting
  "edge deletions are slower (needs tree repair)" than insertions — this
  matches the well-established asymmetry in the incremental-SSSP
  literature generally (an inserted edge can only ever improve or leave
  unchanged a shortest-path distance, a local relaxation; a deleted edge
  that was on the current shortest-path tree can force a search for a
  replacement path through a subtree, a strictly more expensive
  operation) and is exactly the "insert/delete mix" axis this track's
  brief asks for.
- **Baselines** (per the secondary gist, not independently verified):
  Galois and Gunrock — i.e. the paper compares its incremental-update
  approach against recomputing-from-scratch using established
  general-purpose parallel graph frameworks, consistent with the
  abstract's claim of being faster than "state-of-the-art
  recomputing-from-scratch algorithms."
- **Corroborating detail from the (unofficial) student repo**: the repo
  independently reimplements a 4-stage escalation matching the paper's
  own framing exactly — (1) serial Dijkstra baseline with `addEdge`/
  `removeEdge` and full recompute, (2) OpenMP-parallel version adding
  `processIncomingUpdates`/`updateAffectedVertices` (only touching
  affected vertices) plus a tunable `batchSize` parameter controlling how
  many updates are grouped before resynchronizing, plus a
  `validateSSSP` post-update consistency check (parent/distance
  consistency, not a from-scratch-recompute comparison), (3) MPI+OpenMP
  hybrid with `MPI_Allreduce`/`MPI_Bcast`/`MPI_Barrier` for distributed
  global-minimum-vertex determination, (4) MPI+OpenMP+METIS with
  METIS-based graph partitioning added on top and a delta-stepping-style
  update mechanism. That a completely independent, unaffiliated
  implementation converged on the same "batchSize + affected-vertex
  propagation + delta-stepping-flavored update + post-hoc consistency
  check (not a full-recompute comparison)" design, right down to matching
  the paper's described "rooted tree data structure" approach, is decent
  (if indirect) corroboration of the paper's actual algorithmic template,
  even though the repo's own timing numbers (profiler screenshots showing
  sub-second runs on what appear to be small toy graphs) cannot be
  taken as reproducing the paper's reported results.
- **Correctness (inferred)**: the student repo's `validateSSSP` checks
  internal consistency (distance/parent-array agreement) after each
  update rather than comparing against an independent from-scratch
  Dijkstra run on the final graph state — if the original paper follows
  the same pattern, that would be a WEAKER correctness gate than
  comparing against ground truth, and is flagged as an open question
  below rather than assumed.

## Divergences

- **Fundamental operation mismatch**: DTMiner counts temporal motifs over
  a static, complete historical log; Khanda et al. maintain a live SSSP
  tree under a genuine stream of insertions/deletions. There is no shared
  "the algorithm under test" between the two papers — only a shared
  DBLP-level classification as "dynamic graph." The spec below keeps them
  as clearly separate variant FAMILIES rather than forcing one metric to
  cover both.
- **"Batch" means two different things**: DTMiner's chunk size `S_c` is an
  internal performance-tuning knob (sized to fit the LLC, computed from a
  formula involving index-structure size and reserved runtime space —
  section 4.2.1) with NO semantic effect on the final answer (the motif
  count is identical for any valid chunk size; only speed changes).
  Khanda et al.'s batch size is a SEMANTIC choice — how many edge changes
  are grouped before the SSSP tree is guaranteed consistent again — and
  affects both update-latency and the interval during which queries may
  see a stale tree. The spec's variants keep "batch size" as a tuning
  parameter in one family and a consistency-relevant parameter in the
  other, explicitly labeled as such.
- **Update-throughput vs. query-latency, as posed by the track brief,
  applies cleanly only to Khanda et al.** DTMiner has no notion of a
  "query" separate from "run the whole mining job once"; its closest
  analogue is end-to-end job completion time, which the spec below
  captures as a streaming-analytics throughput number rather than
  forcing a query-latency framing onto it.
- **Correctness bar**: DTMiner's is an exact deterministic count (no
  tolerance). Khanda et al.'s (as best can be inferred) may be an
  internal-consistency check rather than a ground-truth comparison — a
  weaker bar this spec explicitly strengthens (see notes_on_fairness).
- **Evidence quality is asymmetric**: DTMiner was read from a
  peer-reviewed, fully-accessible open PDF and both its own repo scripts
  and its formal text agree in every checkable detail (dataset names,
  delta values, motif count). Khanda et al. could only be characterized
  from an abstract, a secondary notes gist, and an unofficial
  third-party reimplementation — materially lower confidence, flagged
  throughout with "per the secondary source" / "could not be confirmed."

## Open questions carried into spec.yaml

- Khanda et al.'s actual graph suite (names, sizes, real vs. synthetic
  split) could not be confirmed; the spec below substitutes DTMiner's
  own real temporal-graph suite (replayed as an insertion stream in
  chronological order) as a principled, verifiable stand-in, explicitly
  flagged as NOT the paper's own suite.
- Whether "50M/100M changes" and "batch size 15/30/50" (per the secondary
  gist) refer to the same or different experimental axes in the original
  paper is unresolved.
- Whether Khanda et al.'s own correctness check is a from-scratch-Dijkstra
  ground-truth comparison or an internal-consistency check (as the
  unofficial reimplementation does) is unresolved; this spec mandates the
  stronger (ground-truth) check regardless, per this repo's general
  fairness principle of gating on correctness before timing counts.
- DTMiner's random-seed / hash-collision sensitivity (its hierarchical
  index uses a hash table for 30% of vertices by default, section 4.2.2)
  is not separately studied in the paper the way ECL-MST's filter-seed
  sensitivity is in the mst track; whether chunk-index hash choice
  materially affects the reported speedups is untested by the paper
  itself.
