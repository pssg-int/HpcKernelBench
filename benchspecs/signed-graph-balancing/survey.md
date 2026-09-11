# signed-graph-balancing — survey

Singleton track: 1 paper.

## Alabandi, Tešić, Rusnak & Burtscher, "Discovering and Balancing
Fundamental Cycles in Large Signed Graphs", SC 2021
(`conf/sc/AlabandiTRB21`)

- **Source**: GitHub repo `burtscher/graphBplus` (README + source, fetched
  via `gh api`), specifically `graphBplus_02.cu` (CUDA), `input_script.sh`
  (dataset acquisition), and the abstract. A free PDF exists at
  `userweb.cs.txstate.edu/~burtscher/papers/sc21.pdf` but could not be
  parsed as text with the tools available in this environment (no
  `pdftotext`/PyPDF/pymupdf installed, WebFetch returned only PDF
  metadata) — repo source + abstract were used instead.
- **Workloads**: signed social-network graphs, CSV edge-list format
  (`user1,user2,sign`). Confirmed real datasets acquired by the repo's own
  `input_script.sh`:
  - SNAP `soc-sign-epinions` (trust/distrust network)
  - SNAP `soc-sign-Slashdot090221` (friend/foe network)
  - `wikiElec.ElecBs3` (Wikipedia admin-election votes, SNAP)
  - Amazon product-rating graphs, one per category (Books, Apps, Electronics,
    Movies&TV, Baby, Toys&Games, Patio/Lawn/Garden, Musical Instruments,
    Clothing/Shoes/Jewelry, Digital Music, Sports&Outdoors, Video Games,
    CDs&Vinyl, Automotive, Amazon Instant Video), ratings binarized to
    positive/negative edges from the SNAP Amazon product-graph corpus.
  - Headline scale case (from the abstract): an Amazon graph with 10M
    vertices, 22M edges, 14M+ fundamental cycles.
- **Timing protocol**: `GPUTimer` struct wraps each phase (spanning-tree
  construction, path-compressed-union-find labeling, cycle detection/
  traversal, balancing, connected-components check) with CUDA events
  (`cudaEventRecord`/`cudaEventElapsedTime`); phase times are accumulated
  into `graphBtime` across the algorithm's own internal `iterations` loop
  (an algorithm-quality knob, not a benchmark-repetition knob — CLI usage
  `./graphBplus graph.csv 100 out.csv` runs 100 internal balancing rounds
  in a single invocation, matching the SNAP-dataset-scale README example).
- **Timing scope**: input CSV parsing/graph read is a separate,
  explicitly-labeled phase ("input time") excluded from `graphBtime`. An
  "overall runtime with I/O" number is also printed, keeping preprocessing
  visible but distinct.
- **Precision & correctness**: this is a combinatorial, not numerical,
  problem — correctness is a globally sign-balanced graph (every cycle
  has an even count of negative edges after the flip choices made).
  graphB+ is a linear-time HEURISTIC for minimum edge flips (not proven
  minimum), so the number of flipped edges is itself a solution-QUALITY
  metric, not just a correctness byproduct.
- **Metric**: headline metric in the abstract is "fundamental cycles
  identified, traversed, and balanced per second" (14M+/s on the 10M-
  vertex/22M-edge Amazon graph on a Titan V GPU, 0.85s total) — a
  cycles/sec throughput metric, alongside wall time.
- **Baselines**: repo ships both a CUDA (GPU) and an OpenMP (CPU, with
  g++ intrinsics) implementation of the same algorithm, enabling a direct
  GPU-vs-CPU comparison of the identical method (not a comparison against
  a different competing algorithm from available sources).

## Divergences

N/A — only one paper in this track. The one design choice worth making
explicit as a fairness fix (not a literature disagreement, since there's
one source) is that the algorithm's own `iterations` CLI parameter trades
solution quality (more edges reconsidered => potentially fewer total
flips) for time, similarly to how other tracks treat autotuning-search
budgets — this spec requires it to be fixed and disclosed rather than
silently varied to produce a favorable time or flip-count number.
