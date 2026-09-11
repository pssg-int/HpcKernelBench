# spectral-sparsification — survey

Singleton track: 1 paper.

## Aghdaei & Feng, "inGRASS: Incremental Graph Spectral Sparsification via
Low-Resistance-Diameter Decomposition", DAC 2024 (`conf/dac/AghdaeiF24`)

- **Source**: arXiv fulltext, `arxiv.org/html/2402.16990` (WebFetch),
  cross-checked against `arxiv.org/abs/2402.16990` and the GitHub README
  (`Feng-Research/inGRASS`, fetched via `gh api`).
- **Workloads**: 13 test graphs from the **SuiteSparse Matrix
  Collection** (sparse.tamu.edu), spanning three domains:
  - Circuit simulation: `G3_circuit` (1.5M nodes, 3.0M edges), `G2_circuit`
    (1.5x10^5 nodes, 2.9x10^5 edges)
  - Finite element analysis: `fe_4elt2` (1.1x10^4 nodes, 3.3x10^4 edges),
    `fe_ocean` (1.4x10^5 nodes, 4.1x10^5 edges), `fe_sphere` (1.6x10^4
    nodes, 4.9x10^4 edges)
  - Other (Delaunay triangulations / large sparse graphs): `delaunay_n18`
    through `delaunay_n22` (2.6x10^5 - 4.2x10^6 nodes), `M6`, `333SP`,
    `AS365` (each ~3.7x10^6 nodes, ~1.1x10^7 edges), `NACA15` (1.0x10^6
    nodes, 3.1x10^6 edges).
- **Timing protocol**: setup-phase time and update-phase (incremental)
  time reported SEPARATELY, with the update phase measured across 10
  iterative update cycles per graph (not a single incremental edit).
  Hardware: Linux/Ubuntu, 1TB RAM, 3.6GHz 64-core CPU (this is a
  CPU/Julia algorithm, `platform: cpu` — no GPU involved despite the
  track's general HPC-accelerator focus).
- **Timing scope**: setup (initial nearly-linear-time sparsifier
  construction via the low-resistance-diameter decomposition, LRD) is
  reported distinctly from each incremental update (O(log N)-time,
  amortized per edge-batch insertion).
- **Precision & correctness / quality**: spectral sparsification is
  inherently approximate — there is no single correct sparsifier, only a
  quality/size tradeoff. Primary quality metric: **relative condition
  number kappa(L_G, L_H)** between the original graph Laplacian L_G and
  the sparsifier Laplacian L_H (spectral similarity; lower = better
  fidelity). Secondary: sparsifier density |E_H|/|V| (how aggressively
  edges were removed — the entire point of sparsification is trading
  edge count for approximation quality, so density and kappa must be
  reported together, never density alone).
- **Metric**: primary = time (setup time and per-update time,
  separately); quality = kappa(L_G, L_H), reported jointly with time per
  the general fairness principle for approximate/heuristic algorithms.
- **Baselines**: **GRASS** (Graph Spectral Sparsification, prior
  state-of-the-art spectral sparsification tool — the direct algorithmic
  predecessor/competitor) as primary baseline; **random edge selection**
  as a secondary/ablation lower-bound baseline (demonstrates that
  spectral-criticality-guided selection, not just density reduction,
  drives the quality).

## Divergences

N/A — only one paper in this track. The main methodological point worth
encoding explicitly as a spec requirement (not a cross-paper
disagreement) is that inGRASS's core contribution is the INCREMENTAL
update path (O(log N) per update after nearly-linear setup) — a
benchmark that only measures one-shot setup time would miss the paper's
actual claim, so the spec below keeps setup and incremental-update
throughput as two separate, equally-weighted variants rather than
blending them into a single "sparsification time" number.
