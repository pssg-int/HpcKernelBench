# Graph partitioning — evaluation survey

Track input: this track has no `data/track_inputs/graph-partitioning.json`
file in the repo (confirmed absent — every other track's input file exists
under `data/track_inputs/`). The 2 papers were instead recovered from
`output/benchmark_groups.json["graph-partitioning"]`, which groups the same
underlying `output/included.json` corpus by normalized kernel slug and
lists exactly these 2 papers under `"graph-partitioning"` — matching the
task's "2 papers, small track" description exactly. Both surveyed via full
fulltext (one via arXiv HTML, one via the authors' own hosted PDF).

## 1. Salwasser, Seemaier, Gottesbüren, Sanders — "Tera-Scale Multilevel
Graph Partitioning" (TeraPart, IPDPS 2025) — `conf/ipps/SalwasserSG025`

Full text obtained via `arxiv.org/html/2410.19119` (arXiv:2410.19119,
accepted IPDPS 2025). Artifact: `github.com/KaHIP/KaMinPar` (TeraPart ships
as part of the KaMinPar family of partitioners).

- **Problem regime**: extreme-scale, memory-bound. The paper's whole
  contribution is reducing the multilevel framework's *memory* footprint
  (parallel label-propagation clustering and graph contraction reduced from
  O(np) to O(n) auxiliary space, p = #processors) so multilevel
  partitioning — previously infeasible at this scale due to memory, not
  time — becomes possible on trillion-edge graphs. Headline result: "up to
  a 16-fold reduction in peak memory... can partition a graph with one
  trillion edges in under 8 minutes on a single machine using around 900
  GiB of RAM," and up to 16 trillion edges on 128 machines in under 10
  minutes.
- **k values**: `k ∈ {8, 37, 64, 91, 128, 1000, 30000}` — a deliberately
  wide, partly non-power-of-2 sweep (37, 91 are prime-ish/irregular values,
  explicitly chosen to stress the partitioner beyond convenient
  power-of-2 cases).
- **Imbalance parameter**: `ε = 3%` (balance constraint `|V_i| <=
  (1+ε)|V|/k`).
- **Timing/repetition protocol**: 5 repetitions per instance with different
  seeds for shared-memory experiments, 3 repetitions for distributed
  experiments; arithmetic mean reported for running time, peak memory, and
  edge cuts across seeds; harmonic mean for relative speedups; geometric
  mean for aggregate cross-instance metrics. This 3-different-means
  convention (arithmetic/harmonic/geometric, each for a different quantity)
  is deliberate and should not be collapsed into a single "the mean" in
  any downstream reporting.
- **Quality metric**: edge cut, both as raw count and as a percentage of
  total edges; also uses **performance profiles** — for each algorithm A,
  plot the fraction of instances where `cut_A(I) <= tau * min_A'(cut_A'(I))`
  — the standard convention in the modern (Sanders/Schulz-school) graph
  partitioning literature for comparing many algorithms across many
  instances without picking one arbitrary "ground truth" baseline.
- **Datasets**: two explicitly named tiers.
  - **Medium** ("Benchmark Set A"): 72 graphs from SuiteSparse Matrix
    Collection, Network Repository, Pizza&Chili-corpus text-compression
    graphs, and synthetic graphs; 5.4M to 1.8B undirected edges, mostly
    unweighted (6 text-compression graphs weighted).
  - **Huge** ("Benchmark Set B"): 5 named LAW (Laboratory for Web
    Algorithmics) web graphs — gsh-2015 (25.7B edges), clueweb12 (37.4B),
    uk-2014 (42.5B), eu-2015 (80.5B), hyperlink (112.2B) — plus KaGen-
    generated synthetic graphs: random-geometric (rgg2D, 8.59B vertices /
    1.10T edges) and random-hyperbolic (rhg, 8.59B vertices / 1.01T edges),
    both generated with average degree 256 and power-law exponent 3.0.
- **Baselines**: Mt-Metis (shared-memory reference), ParMetis (distributed),
  XtraPuLP (distributed, trillion-edge-capable), HeiStream (streaming,
  quality comparison).
- **Hardware**: shared-memory experiments on an AMD EPYC 9684X (96 cores
  @ 2.55GHz, 1,536 GiB RAM, Ubuntu 24.04); distributed experiments on the
  HoreKa cluster (dual 39-core Intel Xeon Platinum 8368 per node, 256 GiB/
  node, 64 cores used per node due to a power-of-two requirement,
  InfiniBand 4X HDR 200 Gbit/s, up to 128 nodes). Compiler: g++ 13.2.0,
  Intel TBB + Open MPI 4.0.
- **Platform**: CPU / distributed-CPU only — no GPU path.
- Source: full fulltext via `arxiv.org/html/2410.19119` (WebFetch extracted
  the complete Evaluation/Experimental Setup section, including exact k
  values, epsilon, repetition counts, statistics used, dataset breakdown,
  and hardware); artifact README via
  `gh api repos/KaHIP/KaMinPar/contents/README.MD`.

## 2. Lee, Lin, Huang, Jiang, Ho, Lin, Yu — "G-kway: Multilevel
GPU-Accelerated k-way Graph Partitioner" (DAC 2024) — `conf/dac/LeeLHJHL024`

Full text obtained via `tsung-wei-huang.github.io/papers/2024-DAC-gkway.pdf`
(author-hosted copy of the DAC'24 paper; also published as
ACM TODAES 10.1145/3734522). Artifact:
`github.com/wanluanlee/DAC_g-kway`.

- **Problem regime**: EDA/CAD circuit-timing-graph scale, GPU-resident.
  Contribution is two new GPU-parallel multilevel-partitioning stages:
  (1) **union-find-based coarsening with scoring** — merges many vertices
  per level in one pass (vs. prior GPU work GKSG's iterative
  Heavy-Edge-Matching, which needs many sequential matching rounds and can
  only coarsen 2 vertices per matching pair per round), and (2)
  **independent-set-based refinement** — moves many boundary vertices per
  iteration by finding an independent set of legal moves, sorting by gain,
  and selecting the longest balance-respecting prefix via a parallel-scan-
  based delta-partition-weight check (vs. GKSG's exponential-enumeration
  refinement, limited to ~8 vertices/iteration by GPU memory).
- **k values**: `k = {2, 4, 8, 16, 32}` — small, exclusively power-of-2,
  the opposite regime from TeraPart's wide/irregular sweep.
- **Imbalance parameter**: `ε = 3%` ("we set the imbalance ratio to 3%...
  the same as the default values of mt-metis and GKSG") — **identical
  numeric convention to TeraPart**, the one confirmed point of agreement
  between the two papers in this track (see Divergences).
  Coarsening threshold: `|V| / (20 * log2(k))` (also stated as matching
  mt-metis's and GKSG's own defaults).
- **Timing/repetition protocol**: "All data is an average of ten runs" —
  no seed disclosed, no variance/min/max/stddev reported anywhere in the
  paper's tables or text. This is a materially thinner repeatability
  disclosure than TeraPart's (5 seeded reps + 3 different explicit mean
  conventions).
- **Quality metric**: cut size as a raw edge-weight-sum count (not
  percentage); reported per-graph in Table 1 alongside runtime, plus a
  "cut size improvement ratio" (G-kway's cut / baseline's cut) swept across
  k in Figure 5.
- **Datasets**: two tiers, one of which is **not confirmed publicly
  downloadable**.
  - 6 **industrial circuit timing graphs** (pci_bridge, vga_lcd, wb_dma,
    usb, tv80, mem_ctrl; 6.4M-25.2M vertices, 8.5M-31.6M edges), generated
    via the authors' own cited tools (Taskflow [3], OpenTimer [4]) — no
    public download link or generation script is given in the paper or the
    artifact repo; **reproducibility of this half of the benchmark is not
    confirmed**.
  - 4 **DIMACS10 Graph Partitioning Challenge graphs**: ldoor (952K v /
    22.8M e, structural mesh), NLR (4.16M v / 12.5M e, road network),
    delaunay (16.8M v / 50.3M e, Delaunay triangulation — matches
    "delaunay_nXX" naming in the public DIMACS10 collection), asia.osm
    (11.95M v / 12.7M e, road network) — **all four are standard, publicly
    downloadable DIMACS10 instances**, and are the only graphs in this
    2-paper track's combined dataset list confirmed to be both public and
    plausibly overlapping with TeraPart's own "(D)"-tagged DIMACS
    inclusion in its 72-graph medium set (exact overlap by name not
    confirmed — TeraPart's fulltext extraction did not enumerate all 72
    names).
- **Baselines**: mt-metis v0.7.2 (CPU, 32 threads — same tool family as
  TeraPart's "Mt-Metis" baseline, though TeraPart does not state a fixed
  thread count for it in the extracted text) and GKSG (prior GPU
  partitioner). **GKSG is not open-source**; the authors re-implemented
  GKSG's algorithm themselves (stated explicitly: "Since GKSG is not
  open-source, we implemented its algorithm on our GPU except for the
  initial partitioning") — the GKSG baseline numbers in this paper are
  therefore a reimplementation, not a run of the original authors' code, a
  reproducibility caveat for anyone trying to reproduce the 3.8x GKSG
  speedup claim specifically.
- **Hardware**: single machine, 40-core Intel Xeon Gold 6138 @ 2.00GHz
  host CPU, 256 GB RAM, **single NVIDIA RTX A6000 GPU (48 GB)**. mt-metis
  run with 32 of the 40 available host threads.
- **Platform**: single-GPU only — no distributed/multi-GPU path (contrast
  TeraPart's up-to-128-node distributed path).
- Source: full fulltext via
  `tsung-wei-huang.github.io/papers/2024-DAC-gkway.pdf` (all sections:
  Introduction, Problem Definition, Algorithm, §4 Experimental Evaluation
  incl. Table 1 full per-graph runtime/cut-size numbers, §4.1 Baselines,
  §4.3-4.4 Runtime/Cut-size analysis across k); artifact README via
  `gh api repos/wanluanlee/DAC_g-kway/contents/README.md`.

## Divergences

- **Scale regime is almost a different problem, not just a different
  implementation.** TeraPart targets `k` up to 30,000 on graphs up to 16
  trillion edges across 128 machines; G-kway targets `k` up to 32 on
  graphs up to ~31M edges on a single GPU. There is **no overlapping (k,
  |E|) operating point tested by either paper** — TeraPart never reports
  k<=32, and G-kway never reports k>32 or graphs anywhere near TeraPart's
  smallest ("huge" tier starts at 25.7B edges, more than 800x G-kway's
  largest graph). A benchmark spec that pretended these were
  directly comparable on a single combined leaderboard would be
  misleading; the spec below keeps them as two explicitly separate scale-
  regime variants.
- **Imbalance convention agrees exactly (ε=3%)** — this is the one
  substantive point of methodological agreement between the two papers,
  and is adopted below as the spec's fixed default rather than being
  independently re-derived.
- **Quality metric reporting differs in units but not definition.**
  Both define cut size identically (sum of edge weights crossing block
  boundaries) but TeraPart reports it as a percentage of total edges (plus
  performance profiles across many instances), while G-kway reports raw
  edge-weight-sum counts (plus a simple ratio-to-baseline). Both are
  losslessly convertible to each other given total edge count, so this is
  a reporting-format divergence, not a definitional one — the spec below
  requires both forms.
- **Repeatability disclosure is asymmetric.** TeraPart: 5 (shared-memory)
  or 3 (distributed) repetitions with explicitly different seeds per
  repetition, arithmetic/harmonic/geometric means used deliberately for
  different quantities. G-kway: "average of ten runs," no seed, no
  variance of any kind reported for any table or figure. G-kway's own
  published numbers cannot currently be assessed for run-to-run
  reliability from the paper text alone.
- **Dataset public-reproducibility differs by tier.** TeraPart's entire
  dataset (SuiteSparse, Network Repository, Pizza&Chili, LAW web graphs,
  KaGen-generated synthetic) is public and regeneratable from stated
  parameters. G-kway's 4 DIMACS10 graphs are public; its 6 industrial
  circuit-timing graphs are not confirmed publicly downloadable or
  regeneratable from the paper/artifact (the generation tools are cited
  but no dataset release or generation script is provided) — roughly half
  of G-kway's own benchmark suite cannot currently be reproduced by a
  third party from the sources surveyed.
- **Hardware/platform is disjoint.** TeraPart: CPU-only, single 96-core
  node up to 128-node distributed cluster. G-kway: single NVIDIA GPU only.
  Direct runtime comparison across the two papers' own reported numbers is
  not meaningful without a shared graph+k operating point run on both
  platforms, which the spec's small-k variant below is designed to enable
  (by picking public DIMACS10 graphs both papers' dataset lists plausibly
  touch, at k values small enough for a single GPU to handle).
- **Baseline overlap**: both compare against a member of the Metis family
  (Mt-Metis / mt-metis) — the one shared baseline tool across the track,
  though TeraPart doesn't state a fixed thread count for it in the
  extracted evaluation text while G-kway pins it at 32 threads explicitly.
  TeraPart additionally benchmarks against ParMetis, XtraPuLP, and
  HeiStream (all distributed/streaming peers G-kway does not test against,
  consistent with G-kway's single-machine-only scope); G-kway additionally
  benchmarks against GKSG, a GPU peer TeraPart does not test against
  (consistent with TeraPart's CPU-only scope) and which is explicitly a
  reimplementation rather than the original authors' code.
