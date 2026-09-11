# Triangle-counting track — evaluation methodology survey

Track input: `data/track_inputs/triangle-counting.json`, 5 papers (all 5
surveyed, exceeding the 5-paper minimum). None of the 5 papers list an arXiv
id in the track-input JSON; 3 turned out to have a freely-postable arXiv
preprint discoverable via web search and were read via arXiv/ar5iv fulltext
(katric, lcc, TriPoll). The remaining 2 GPU papers (ToT, TC-Compare/GroupTC)
have no arXiv preprint — ACM DL returned HTTP 403 and IEEE Xplore returned
HTTP 418 for both, so those two were surveyed entirely from their artifact
repos' driver code and README via `gh api` (which turned out to be the more
reliable source anyway: it shows the actual timing-loop/warmup/preprocessing
boundaries rather than prose that may omit them).

Operation surveyed throughout: exact global triangle count of a static,
undirected, simple graph `G=(V,E)`: `T = |{(u,v,w) : uv,vw,wu ∈ E}| / 6`, all
5 papers compute this via an orientation (each edge directed low-degree →
high-degree or by vertex order) + local/neighbor-intersection reformulation.
**No paper in this track uses approximate/sampling-based counting** — all 5
report exact integer counts. This track's benchmark spec is therefore
exact-counting only; no approximate-counting variant is offered (see
Divergences).

---

## 1. Chen, Yu — "ToT: Triangle Counting on Tensor Cores" (TPDS'25, extends PPoPP'25 "Triangle Counting on Tensor Cores")

- key: `journals/tpds/ChenY25`, artifact `yuang-chen/ToT-TPDS25`
- **workloads**: driver (`apps/tot.cu`) reads a single graph via
  `read_matrix_file` in Matrix-Market (`.mtx`) format — i.e. the SuiteSparse
  Matrix Collection convention (symmetric sparse adjacency matrix), NOT a
  SNAP-style raw edge list. Repo ships only a toy example
  (`toydata/pli.mtx`); the paper's actual dataset table was not accessible
  (ACM 403 / IEEE 418 blocked, no arXiv preprint found).
- **timing protocol**: no iteration loop in the released driver — each phase
  gets exactly one `tot::CUDATimer` start/stop pair, single run, no
  warmup/repeat/statistic of any kind visible in the code.
- **timing scope**: THREE separately timed phases, cleanly split: (1)
  optional `extract_upper_triangular` (an orientation step: undirected →
  directed-by-triangular-extraction, only timed if `-e 1` requested), (2)
  `convert_coo2bmp` — COO → bitmap-format conversion, i.e. preprocessing for
  the tensor-core kernel, timed separately as "[Converting to Bitmap] time",
  (3) `count_triangles_on_tensors` — the actual kernel, timed separately as
  "[Counting Triangles] time". Preprocessing is never silently folded into
  the counting-kernel time.
- **precision & correctness**: kernel uses `half` (fp16) Tensor Core MMA. The
  README explicitly **warns**: "this implementation ... may introduce
  precision errors during count summation, particularly with large graphs,"
  and only offers `-e 1` as a mitigation that works "in some cases" — an
  unresolved, self-documented correctness gap. Correctness is checked via a
  `-v` flag that recomputes the exact count on the CPU (`bfs_tc`, host-side
  reference) and prints both GPU and CPU counts side by side for manual
  comparison; there is no automated tolerance check or assert in the
  released code.
- **metric**: per-phase wall time in ms (printed via `printf`); triangle
  count (integer). No throughput unit (GTEPS or similar) is computed by the
  driver itself.
- **baselines**: not visible in the released code; the paper's abstract
  (from the track-input JSON) claims "3.81x speedup, 17x memory savings"
  over unspecified state-of-the-art tensor-core baselines.
- source: repo `apps/tot.cu`, `README.md`, `tot/tot.h` (`gh api
  repos/yuang-chen/ToT-TPDS25/contents/...`). ACM DL (403) and IEEE Xplore
  (418) blocked fulltext; no arXiv preprint found via web search.

## 2. Li, Xu, Pham, Tu, Zhou — "A Comparative Study of Intersection-Based Triangle Counting Algorithms on GPUs" / GroupTC (IPDPS'24)

- key: `conf/ipps/Li0PTZ24`, artifact `Jangbao/TC-Compare`
- **workloads**: 9 GPU algorithms run through one harness — 8 re-collected
  prior implementations (Bisson, Fox, Green, H-INDEX, Hu, TRUST, polak,
  tricore) plus the paper's own **GroupTC**. Two dataset families: (a)
  **SNAP** — 19 named real graphs fetched by
  `data_getter/snap/WgetSNAPData.sh`: as-caida, p2p-Gnutella31, email-EuAll,
  soc-Slashdot0902, web-NotreDame, com-dblp, amazon0601, roadNet-CA,
  wiki-Talk, web-BerkStan, as-skitter, cit-Patents, soc-pokec,
  sx-stackoverflow, com-lj, soc-LiveJournal1, com-orkut, twitter7,
  com-friendster — spanning ~88K edges (p2p-Gnutella31) to >1B edges
  (twitter7, com-friendster); (b) **Graph500** — the reference Kronecker
  generator (`graph500-2.1.4`'s `seq-csr` binary), example invocation
  `-R -s 17 -e 2` (RMAT, scale 17, edge-factor 2) in
  `data_getter/g500/run.sh` — the exact scale sweep used in the paper itself
  is not visible in the released script (only this one example is checked
  in).
- **timing protocol**: `approach/GroupTC/tc.cu`'s `gpu_run()` hard-codes
  `int iterator_count = 100;`, loops `cudaMemset` → kernel launch →
  `cudaDeviceSynchronize()` → `thrust::reduce`, accumulates wall time via a
  `wtime()` host timer each iteration, and reports
  `total_kernel_use / iterator_count` — a plain **mean over all 100
  iterations, no warmup discard, no median/min/max**. The README's example
  commands show the identical trailing `100` argument for all 8 other
  compared algorithms (`./bisson -f ... 1 1 100`, `./green ... 1 100 512
  32`, etc.), confirming the same 100-iteration mean-only protocol is used
  uniformly across all 9 algorithms.
- **timing scope**: kernel-only. The one-time H2D copy of the CSR arrays
  happens in `TC_gpu()` **outside** the 100-iteration loop. SNAP→CSR
  conversion and any algorithm-specific reordering (e.g. GroupTC's own
  `rid_dcsr` — reordered-ID DCSR — format) are done entirely offline by
  separate binaries in `preprocessing/` (`SNAP2CSR.cpp` etc.), never inside
  the timed region.
- **precision & correctness**: exact integer count via binary-search set
  intersection (int32 throughout). No automated ground-truth check is
  present in `tc.cu`; correctness is implicitly argued by requiring all 9
  algorithms' printed counts to agree on the same dataset (README: "If you
  get results similar to the ones below, congratulations!"), not by an
  automated assert against a published/independent value.
- **metric**: "avg kernel use" in seconds (README worked example: `iter 100,
  avg kernel use 0.000148 s`), plus the raw triangle count. No GTEPS or
  other normalized throughput is computed.
- **baselines**: the 8 re-collected prior GPU implementations, run through
  the identical driver/timing harness as GroupTC itself — an unusually
  well-controlled same-methodology comparison.
- source: repo README, `approach/GroupTC/tc.cu`,
  `data_getter/snap/WgetSNAPData.sh`, `data_getter/g500/run.sh` (`gh api`).
  IEEE Xplore PDF returned HTTP 418; no arXiv preprint found.

## 3. Sanders, Uhl — KaTric, "Engineering a Distributed-Memory Triangle Counting Algorithm" (IPDPS'23)

- key: `conf/ipps/SandersU23`, artifact `niklas-uhl/katric`, arXiv:2302.11443
- **workloads**: 8 real graphs (Table I in the paper) — LiveJournal (4.8M
  v/42.8M e), Orkut (3.1M v/117.2M e), Twitter (41.7M v/1.2B e), Friendster
  (68.3M v/1.8B e), uk-2007-05 (105.9M v/3.3B e, 50GB on disk), webbase-2001
  (118.1M v/854.8M e), plus road networks Europe (18M v/22.2M e) and USA
  (23.9M v/28.9M e); synthetic graphs generated via **KaGen**: 2D random
  geometric (RGG2D), random hyperbolic (RHG, γ=2.8), Erdős–Rényi G(n,m), and
  R-MAT, all at ~16 expected edges/vertex, used for weak/strong-scaling
  sweeps (`experiments/paper-suites/tric.suite.yaml` shows generator params
  `n=16..18, m=20..22, scale_weak: True`).
- **timing protocol**: the paper states running time is what's reported; the
  released experiment-runner config fixes `iterations: 2` per
  (algorithm, core-count, graph) combination in both
  `real-world.suite.yaml` and `tric.suite.yaml` — i.e. only **2
  repetitions**, no explicit warmup step, and no stated aggregation
  statistic (mean vs. median) in either the paper text or the runner
  config.
- **timing scope**: the paper explicitly states it reports "the total
  running time of each algorithm **excluding** the time for reading the
  input graph and building the graph data structure" — I/O + distribution
  is reported separately — **but including edge orientation and
  neighborhood (adjacency-list) sorting inside the measured algorithm
  time**. This is the one paper of the 5 that does NOT cleanly separate
  orientation/degree-ordering preprocessing from the counted kernel time.
- **precision & correctness**: exact integer triangle count; no automated
  correctness check described in the fetched text. Two algorithm presets
  are provided (DiTriC — async buffered messaging; CeTriC — communication
  reduction) that should agree on the same input, an implicit cross-check
  though not automated in the harness.
- **metric**: wall-clock time (s), strong- and weak-scaling curves, speedup
  vs. baselines, max outgoing messages/PE, bottleneck communication volume.
- **baselines**: **HavoqGT** (vertex-centric, topology-dependent routing,
  neighborhood partitioning + message aggregation — limited to 2^14 cores in
  the paper's tests because its preprocessing alone exceeds 900s) and
  **TriC** (the "2020 graph champion," static message aggregation without
  edge orientation — fails beyond 2^14 cores or OOMs on some instances).
- **hardware**: SuperMUC-NG (LRZ) thin nodes, Intel Skylake Xeon Platinum
  8174 (48 cores/96GB/node), Omni-Path 100Gbit/s; g++-11.2.0 + Intel MPI
  2021, `-O3`; scale 2,048–32,768 cores (2^11–2^15).
- source: arXiv fulltext (`ar5iv.labs.arxiv.org/html/2302.11443`) +
  `experiments/paper-suites/{real-world,tric}.suite.yaml`,
  `experiments/README.md` (`gh api`).

## 4. Strausz, ... — "Asynchronous Distributed-Memory Triangle Counting and LCC with RMA Caching" (IPDPS'22)

- key: `conf/ipps/StrauszVGBH22`, artifact `fvella/lcc`, arXiv:2202.13976
- **workloads**: synthetic R-MAT (Graph500-style parameters a=0.57, b=c=0.19,
  d=0.05) at three sizes: S21/EF16 (2.1M v/33.6M e), S23/EF16 (8.4M
  v/134.2M e), S30/EF16 (1073.7M v/17,179.9M e); real graphs from
  SNAP/KONECT/web-crawl sources — SNAP-Orkut undirected (3M v/117.2M e),
  SNAP-LiveJournal undirected (4M v/34.7M e) and directed soc-LiveJournal1
  (4.8M v/69M e), SNAP-Skitter (1.7M v/11.1M e), uk-2005 directed (39.5M
  v/936.4M e), wiki-en directed (13.6M v/437.2M e).
- **timing protocol**: two DIFFERENT rigor levels by scale. Shared-memory
  experiments: "median and repeated every experiment until the 5% of the
  median was within the 95% CI" — an adaptive-repetition, CI-gated protocol.
  Distributed-memory experiments: "two different job allocations with three
  executions per allocation" (6 runs total), reporting "the median of the
  longest-running node among all runs with the corresponding 95% CI" — i.e.
  a straggler-aware median. This is the single most statistically rigorous
  protocol found among the 5 surveyed papers.
- **timing scope**: "the read-in of the graph and the relative distribution
  phase" is explicitly excluded from every reported time; only LCC/triangle
  computation is timed. Whether edge orientation is bundled into that
  computation time (like katric) or reported separately (like TriPoll) is
  not explicitly stated in the fetched text.
- **precision & correctness**: exact counting; no automated correctness
  check documented in the fetched evaluation text.
- **metric**: edges processed per microsecond, speedup relative to the
  smallest tested configuration, running time with vs. without RMA caching,
  % reduction in communication time, cache hit/miss rate.
- **baselines**: **TriC** (2020 "graph champion" for distributed triangle
  counting), a binary-search intersection method, Sorted-Set-Intersection
  (SSI), and a non-cached ablation of the authors' own implementation.
- **hardware**: shared-memory node — Intel Xeon Gold 6154 (16 cores,
  3.00GHz), ICC 2021.1 `-O3`. Distributed — Piz Daint XC50, 12-core Intel
  Xeon E5-2690 v3 (2.60GHz)/64GB RAM/node, Cray Aries network, ICC 19.1 +
  Cray MPICH 7.7.16.
- source: arXiv fulltext (`ar5iv.labs.arxiv.org/html/2202.13976`).

## 5. Steil et al. — TriPoll, "Computing Surveys of Triangles in Massive-Scale Temporal Graphs with Metadata" (SC'21)

- key: `conf/sc/SteilRIPSP21`, artifact `llnl/tripoll`, arXiv:2107.12330
- **workloads**: real web/social graphs — LiveJournal (4.85M v/69.0M
  e/286M triangles), Friendster (66M v/3.6B e/4.2B triangles), Twitter
  (42M v/2.4B e/34.8B triangles), uk-2007-05 (106M v/6.6B e/286.7B
  triangles), web-cc12-hostgraph (101M v/3.8B e/415B triangles), Web Data
  Commons 2012 (3.56B v/224.5B e/9.65T triangles), Reddit (835M v/9.4B
  e/88.1B triangles); synthetic R-MAT up to scale 32 for weak-scaling
  experiments.
- **timing protocol**: `src/tsv_tc.cpp` (the artifact's main triangle-count
  driver) has **no iteration loop at all** — a single run per invocation,
  using two `ygm::timer` instances: `preprocess_timer` wraps graph
  construction + DODGR conversion, `step_timer` is reset and reused around
  each of the two counting algorithms. No warmup, no repetitions, no
  statistic aggregation — a single-shot wall-clock number per phase,
  printed directly.
- **timing scope**: the cleanest 3-way separation among all 5 papers.
  `preprocess_timer` wraps EXACTLY graph construction
  (`async_add_edge`/`uniquify_edges`) plus `make_dodgr` — the degree-ordered
  directed-graph conversion, i.e. the orientation/degree-ordering step —
  and is printed as "Total preprocessing time" **before** any triangle
  counting starts. The counting kernel itself gets its own freshly-reset
  timer around `tc_push_pull`, and a second one around `tc_push_only`, each
  printed as its own "Found N triangles in T seconds" line, fully
  decoupled from preprocessing.
- **precision & correctness**: exact counting via `uint64_t` accumulation +
  `world.all_reduce_sum`. The driver runs BOTH `tc_push_pull` and
  `tc_push_only` on the same graph and prints both triangle counts — an
  implicit cross-check (a mismatch would signal a bug) though not enforced
  by an automated assert.
- **metric**: wall-clock time (s), speedup vs. baseline node counts,
  communication volume (GB, Table 4 in the paper), work rate
  (wedges/node/time, Figure 5) — the closest thing to a normalized
  throughput metric found among the 5 papers.
- **baselines**: Pearce et al. 2017 (async MPI, degree-1-vertex-pruning
  preprocessing), Tom et al. 2019 (2D cyclic decomposition + hashing), TriC
  (Ghosh 2020).
- **hardware**: Catalyst cluster (LLNL), dual Intel Xeon E5-2695v2 (24
  cores/128GB RAM/node), InfiniBand QDR; scale 2–256 nodes (up to 6,144 MPI
  ranks).
- source: arXiv fulltext (`ar5iv.labs.arxiv.org/html/2107.12330`) + repo
  `src/tsv_tc.cpp` (`gh api`).

---

## Divergences

- **Exact vs. approximate counting**: no divergence — all 5 papers count
  exactly. This track's spec is exact-only; no approximate/sampling variant
  is defined (see `open_questions`).

- **Orientation/degree-ordering preprocessing — in or out of the timed
  region**: genuinely split three ways.
  - **Cleanly separated, reported before counting starts**: TriPoll
    (`preprocess_timer` wraps DODGR construction/degree-ordering; counting
    gets its own fresh timer) and ToT (COO→bitmap conversion and the
    optional upper-triangular extraction each get their own `CUDATimer`,
    distinct from the counting-kernel timer).
  - **Never in any timed region at all (fully offline)**: TC-Compare/
    GroupTC — all format conversion and reordering (including GroupTC's own
    `rid_dcsr` vertex-reordering) happens via standalone preprocessing
    binaries, never inside the 100-iteration `gpu_run()` loop.
  - **Folded into the measured "algorithm" time**: katric explicitly states
    it excludes only I/O + graph-structure-building, but edge orientation
    and neighborhood sorting run INSIDE the timed region. lcc's stated
    exclusion ("read-in ... and relative distribution phase") is worded
    similarly to katric's and likely bundles orientation the same way,
    though this is not stated explicitly for lcc.

  This is exactly the disagreement the benchspec instructions anticipate:
  the spec resolves it with a kernel-only variant (orientation/ordering
  excluded, reported separately, following TriPoll's/ToT's/TC-Compare's
  practice) and an end-to-end variant (orientation included, following
  katric's practice) rather than picking a side.

- **Graph suite — no shared suite across the track**: the 2 single-GPU
  papers (ToT, TC-Compare) and the 3 distributed-CPU papers (katric, lcc,
  TriPoll) use almost disjoint graph sets and even different INPUT FORMAT
  conventions (ToT: SuiteSparse-style Matrix-Market; TC-Compare/katric/lcc/
  TriPoll: SNAP-style raw edge lists). Among the 3 distributed papers, the
  graphs that recur in ≥2 of them are: LiveJournal, Orkut/Friendster,
  Twitter, and the uk-2007-05/webbase-2001 web-graph family — but no two
  papers use byte-identical instances (different crawl years / dedup /
  symmetrization choices). TC-Compare additionally uses Graph500's reference
  Kronecker generator; katric and lcc both use RMAT/KaGen-family synthetic
  generators with materially different scale ladders (katric: n=16-18;
  lcc: S21/S23/S30; TriPoll: up to scale 32).

- **Timing rigor spans the full range from "none" to "adaptive
  CI-gated"**: TriPoll and ToT run a single, un-repeated measurement per
  graph; katric repeats twice with no stated statistic; TC-Compare repeats
  100x but reports a plain mean (no warmup discard, no median/min/max); lcc
  is the outlier in rigor — adaptive repetition until 5% of the median falls
  inside a 95% CI, with a straggler-aware median for the distributed case.
  The spec adopts lcc's protocol as the fairness fix for the distributed
  variants, and mandates warmup+median+min/max for the GPU variants (fixing
  TC-Compare's/ToT's weaker practice), per the "fixed timing protocol"
  fairness principle in the instructions.

- **Correctness verification — the weakest axis across the whole track**:
  only ToT has an automated-looking (though not asserted) reference-count
  check (`-v` flag runs a CPU BFS-based exact count). TC-Compare and TriPoll
  rely on *implicit* cross-checks (multiple algorithms/variants must agree
  on the same graph) rather than an independent ground truth. Katric and lcc
  document no correctness check at all in the fetched text. **None of the 5
  papers compare against an independently published triangle count** (e.g.
  SNAP's own per-dataset "Number of triangles" statistic, published on each
  SNAP dataset's `snap.stanford.edu/data` page) — this is a real gap the
  spec fixes by mandating exactly that as the correctness gate for
  SNAP-sourced graphs.

- **Metric — no shared throughput unit**: GPU papers report raw kernel time
  (ms/s) and the triangle count, nothing normalized. Distributed papers
  report time + scaling curves + communication volume; TriPoll's "work rate
  (wedges/node/time)" and lcc's "edges/µs" are the only attempts at a
  normalized rate, and they're not the same unit. The spec introduces a
  single common secondary throughput metric (GTEPS, `|E| / time`, following
  Graph500's own naming convention for exactly this kind of graph
  algorithm) so results are comparable across graph sizes within a variant,
  while keeping raw wall-clock time as the primary, since that's what every
  one of the 5 papers actually reports.

- **Platform split with zero overlap**: no paper in this track straddles
  both single-GPU and distributed-CPU-cluster execution — this is a
  harder split than most other kernel tracks (e.g. SpMM) where some papers
  at least share a "single accelerator" regime. The spec therefore splits
  variants by platform outright (2 GPU variants, 2 distributed-CPU
  variants) rather than trying to force a single cross-platform comparison.
