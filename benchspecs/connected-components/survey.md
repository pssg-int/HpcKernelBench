# Connected-components track — evaluation-methodology survey

Surveyed 4/4 papers from `data/track_inputs/connected-components.json`.
Fulltext was reachable for all 4 (2 via GitHub-hosted PDFs shipped directly
in the papers' own artifact repos, 1 via the author's own web mirror, 1 via
arXiv fulltext + the paper's own artifact-reproducibility README).

**Framing note up front**: unlike a track literally titled "weakly-connected
components" would suggest, the 4 papers this track's classifier actually
selected span three genuinely different sub-problems that share only the
word "connectivity": **biconnected components (BCC)** (2 papers — cut
vertices/edges on undirected graphs), **strongly connected components
(SCC)** (1 paper — inherently directed), and **2D/3D binary-image
connected-component labeling (CCL)** (1 paper — pixel/voxel grid
connectivity in computer vision, not general sparse graphs at all). None of
the 4 papers is a "plain" weakly-connected-components (WCC) paper in the
Shiloach-Vishkin/Afforest label-propagation sense the track brief's key axes
anticipated — though that exact algorithmic family shows up as an internal
**subroutine** inside the BCC papers (see below), which is where the
task's "algorithm family disclosure" axis actually bites. This survey
reports what the 4 papers actually do, and the accompanying spec is
organized around the three real sub-problems rather than forcing them into
a single WCC-shaped variant.

---

## 1. GPU Algorithms for Biconnected Components on Large Graphs (IPDPS 2026)

Source: fulltext PDF shipped directly in the artifact repo
(`Abhijeetkumar96/FAST-BCC-on-GPUs/Fast-BCC-Filter-IPDPS-2026.pdf`, read in
full) + `README.md` + `run_all.sh`.

- **algorithm family**: builds an O(n)-size "skeleton" subgraph from an
  **arbitrary spanning tree** (not a BFS tree, the paper's core theoretical
  contribution — Theorem 3.3 proves BCCs of G equal BCCs of a
  plain-edges(T) ∪ cross-edge-forest skeleton for ANY rooted spanning tree),
  then runs a **connected-components (CC) subroutine** on that skeleton to
  assign BCC labels. Crucially, the CC subroutine used for BOTH the initial
  spanning-tree construction AND the final CC-labeling step is **GConn**
  (Hong, Dhulipala, Shun — PACT 2020), and the paper explicitly states it
  chooses **"k-out sampling"** from GConn's technique menu for the spanning
  tree step — k-out sampling is the same algorithmic family as **Afforest**
  (sample a small number of edges per vertex before falling back to full
  connectivity), a direct, disclosed hit on this track's "algorithm family"
  axis even though the paper's headline contribution is BCC, not CC itself.
- **directed vs. undirected**: BCC is only defined for undirected graphs;
  the paper's preprocessing explicitly **symmetrizes any directed input**
  and further **adds edges to force single-connectivity** where the raw
  graph is disconnected ("Where necessary, we... added minimal edges to
  ensure a single connected component") — i.e. the benchmark's own input
  graphs are edited to guarantee a precondition the algorithm needs, a
  disclosed but real representativeness caveat (a graph's natural
  component structure is altered before it is ever benchmarked).
- **workloads/inputs**: >30 real-world graphs spanning Web (indochina-2004,
  arabic-2005, uk-2005, webbase-2001, it-2004, GAP-web, sk-2005,
  uk-union-2006-07), Social (wiki-topcats, sx-stackoverflow,
  ljournal-2008, hollywood-2009, com-Orkut, GAP-twitter, com-Friendster),
  Road (germany_osm, road_usa, europe_osm), Citation/Biological (as-Skitter,
  coPapersDBLP, coPapersCiteseer, MOLIERE-2016, AGATHA-2015), and
  Synthetic (kron_g500_logn18/19/20/21, 2 RGG instances, GAP-kron) — sizes
  from 1.14M v/11.1M e up to 184M v/11.6B e, with exact per-instance
  vertex/edge/diameter counts given in Table II.
- **component-count verification**: not explicitly described as a
  per-run correctness check in the reachable evaluation section (Section 6
  focuses entirely on performance); the artifact's `run_all.sh` output does
  not appear to include a correctness-gate step distinct from timing, an
  **open question** flagged below.
- **timing protocol**: **"All numbers reported in this section are the
  median of five runs"** for GPU results; the artifact separately exposes a
  `CPU_ROUNDS` environment variable (default suggested value 5 in the
  artifact-evaluation appendix) described as controlling "Number of
  repetitions for each CPU baseline execution (used to compute AVERAGE
  runtime)" — i.e. **GPU results use median-of-5, CPU baseline results use
  mean-of-5**, a real, disclosed but unremarked-upon statistic mismatch
  within the same paper.
- **timing scope**: kernel time only; the artifact's build/run separation
  and the paper's own kernel-level breakdown figures (Fig. 3, 4, 6) show
  spanning-tree construction, Euler-tour, tag-computation, and final-CC
  phases are each individually timed and summed, with format/loading
  excluded.
- **preprocessing / batching disclosure**: for the out-of-memory (OOM)
  variant (PEA-BCC), the algorithm partitions edges into a hyperparameter
  **BATCH_SIZE** (default 500,000 edges in the artifact) with an explicit
  I/O-complexity analysis (O(m/B) passes) and a dedicated scalability study
  (Fig. 7) showing performance is NOT monotonic in batch size — smaller
  batches sometimes outperform the maximum-possible batch size by enabling
  better CPU-GPU transfer/compute overlap (9% improvement on one graph).
  This is a rich, disclosed hyperparameter-sensitivity study, unusual for
  this track.
- **metric**: wall-clock time in milliseconds (Table II), plus speedup vs.
  baseline; the paper explicitly **excludes general-purpose graph
  frameworks (Gunrock, cuGraph, GBBS) from its baseline set** with a stated
  reason ("lack native support for BCC primitives or are previously
  outperformed by the specialized baselines"), a disclosed but consequential
  baseline-selection scoping decision.
- **baselines**: WK-BCC (2017, GPU, BFS-tree-based), WK-BCC (2018, GPU,
  sampling+WK-17), FAST-BCC (Dong et al. 2023 — paper #2 below — official
  CPU repo used directly; the GPU port of FAST-BCC itself was
  re-implemented by this paper's authors since Dong et al.'s own code is
  CPU-only). WK-BCC's original source "was not publicly accessible," so
  this paper **re-implemented WK-BCC from the published algorithm
  description** — a disclosed re-implementation-not-original-code caveat
  worth flagging for anyone treating the WK-BCC numbers as authoritative.
- **hardware**: NVIDIA L40S (48GB, Ada) + dual Intel Xeon Platinum 8480+
  host, and NVIDIA L4 (24GB, Ada) + dual AMD EPYC 7742 host — explicitly
  **two GPUs chosen for complementary memory sizes** to separately stress
  the in-memory and out-of-memory algorithm variants.
- **artifact quality**: genuine own-authors artifact (unlike the SSSP
  track's Khanda misattribution), archived on Zenodo
  (`10.5281/zenodo.18669968`) with a full SC-style Artifact
  Description/Evaluation appendix — the most complete artifact-provenance
  documentation found in either track survey.

---

## 2. Provably Fast and Space-Efficient Parallel Biconnectivity ("FAST-BCC", PPoPP 2023)

Source: arXiv HTML fulltext (`arxiv.org/html/2301.01356`) + the artifact
repo `ucrparlay/FAST-BCC`'s `README.md` and `scripts/run_fastbcc.sh`.

- **algorithm family**: multicore CPU, Tarjan-Vishkin-descended but with a
  novel space-efficient "skeleton" construction (fence edges vs. plain
  edges w.r.t. an arbitrary spanning tree) that reduces auxiliary space
  from Θ(m) to O(n) relative to the classic TV algorithm — this is the
  paper directly extended by paper #1 above (which further reduces the
  GPU-side skeleton from Θ(m) to Θ(n) via cross-edge filtering). The
  underlying connectivity subroutine is not GConn/k-out-sampling here
  (that is paper #1's GPU-specific choice) but a general parallel CC
  primitive over ParlayLib.
- **directed vs. undirected**: undirected only; README states "Make sure
  the input graph is a symmetric graph" — directed inputs must be
  pre-symmetrized by the user, same requirement as paper #1.
  **Both BCC papers in this track REQUIRE undirected input and force
  symmetrization**, a track-wide, not paper-specific, constraint.
  workloads/inputs: **27 graphs** spanning 5 families with exact sizes —
  Social (YouTube 1.13M v/5.98M e, Orkut 3.07M v/234M e, LiveJournal 4.85M
  v/85.7M e, Twitter 41.7M v/2.41B e, Friendster 65.6M v/3.61B e), Web
  (Google 876K v/8.64M e, sd_arc 89.2M v/3.88B e, ClueWeb 978M v/74.7B e,
  Hyperlink14 1.72B v/124B e, Hyperlink12 3.56B v/226B e — the largest
  graphs in either track's survey by a wide margin), Road (California
  1.97M v/5.53M e, USA 23.9M v/57.7M e, Germany 12.3M v/32.3M e), k-NN
  (Household5, CHEM5, 5 GeoLife variants at different neighbor counts,
  Cosmo50 321M v/1.96B e), and Synthetic (grid SQR/REC at 100M v, sampled
  variants, chain Chn7/Chn8 at 10M-100M v) — the **only paper in either
  track survey that includes k-NN spatial graphs and grid/chain synthetic
  topologies**, a workload-family axis absent from every other paper here.
- **component-count verification**: explicit — **"compare the number of
  BCCs reported by each algorithm with SEQ [sequential Hopcroft-Tarjan] to
  verify correctness"**, the clearest component-count correctness gate
  found in this track.
- **timing protocol**: **"each test 10 times and report the median"** —
  the strictest repetition count in this track (5 for both BCC-track
  papers above/below it, 9 for the SCC paper).
- **timing scope**: not explicitly confirmed whether graph loading is
  included in the reported per-test time (open question, flagged below);
  the artifact's benchmark scripts write results to CSV alongside the
  reported number of CCs/BCCs and the largest-BCC size, suggesting timing
  wraps the algorithm call specifically.
- **metric**: running time in seconds; self-relative speedup (parallel vs.
  single-thread of the SAME implementation) AND geometric-mean speedup vs.
  the best competing baseline PER GRAPH — reports both a within-algorithm
  scalability number and a cross-algorithm comparison number, a useful
  two-axis presentation.
- **baselines**: GBBS (BFS-tree-based, "processes all graphs" —
  general-purpose), Slota-Madduri SM'14 (reports best of two variants,
  "limited to connected graphs" — a disclosed applicability caveat), TV
  (Tarjan-Vishkin, "faithful implementation for memory evaluation only,"
  i.e. included to demonstrate its space blowup rather than to be timed
  as a competitive baseline), sequential Hopcroft-Tarjan (SEQ, the
  correctness reference AND a performance floor). Reported: 3.1× faster
  than best baseline on average; on low-diameter graphs 36× self-relative
  speedup and 1.6× vs. GBBS; on high-diameter graphs 47× self-relative
  speedup and 10× vs. GBBS (i.e. the paper's advantage is much larger on
  road-network-shaped high-diameter graphs than on social-network-shaped
  low-diameter ones — a structural-dependency finding directly relevant to
  graph-suite design).
- **hardware**: 96-core (192 hyperthread) 4-socket Intel Xeon Gold 6252,
  1.5TB RAM — **CPU-only**, `numactl -i all` round-robin memory
  interleaving; no GPU numbers (contrast with paper #1's GPU-only focus —
  the two BCC papers are on opposite ends of the CPU/GPU platform axis for
  the *same* underlying algorithmic idea, since paper #1 is explicitly a
  GPU port/extension of this paper's technique).

---

## 3. A GPU Algorithm for Detecting Strongly Connected Components ("ECL-SCC", SC 2023)

Source: fulltext PDF from the author's own mirror,
`https://userweb.cs.txstate.edu/~burtscher/papers/sc23c.pdf` (Alabandi,
Sands, Biros, Burtscher, read in full) + artifact repo
`burtscher/ECL-SCC`'s `README.md` and shipped sample inputs.

- **directed vs. undirected**: **SCC is inherently a directed-graph
  problem** (a maximal subset where every pair is mutually reachable) —
  the only paper in either connectivity-adjacent track surveyed so far
  that is directed by definition rather than by input-graph happenstance.
- **algorithm family**: a genuinely new family relative to the classic
  Forward-Backward (FB)/BFS-pivot approach (Fleischer 2000) that every
  prior parallel/GPU SCC paper in the related-work section uses (including
  its own baseline, SCC-GPU/Li et al. 2017, and the CPU baseline
  iSpan/Ji et al. 2018, which uses a spanning-tree+relaxed-synchronization
  approach). ECL-SCC instead has **every vertex simultaneously act as a
  pivot**: each vertex maintains a max-incoming-ID and max-outgoing-ID
  "signature," signatures are propagated to a fixed point, and edges whose
  endpoints have differing signatures are removed — a "maximum-ID
  propagation" family distinct from both FB/BFS-pivot AND from the
  BCC papers' Tarjan-Vishkin/spanning-tree family. This is a genuinely
  novel algorithm-family axis this track's 4 papers collectively span.
- **workloads/inputs**: TWO structurally distinct graph classes, each with
  its own dedicated table of properties (avg degree, max in/out-degree,
  #SCCs, #size-1 SCCs, #size-2 SCCs, largest-SCC size, DAG depth):
  (a) **mesh graphs from radiative-transfer (RTE) simulations**, built with
  the MFEM finite-element library at multiple discretization orders and
  "ordinates" (light-propagation directions), split into "small" (6 mesh
  families, 8-32 ordinates each, 196K-515K vertices) and "large" (7 mesh
  families, 8-61 ordinates, 1.57M-8.39M vertices) sets — nearly-constant
  low vertex degree (~2-3), deep DAGs, many small SCCs; (b) 10 named
  **power-law graphs from SuiteSparse** (cage14, circuit5M, com-Youtube,
  flickr, Freescale1/2, soc-LiveJournal1, web-Google, wiki-Talk,
  wikipedia), selected specifically because they were **"also used in
  prior work"** (SCC-GPU and iSpan's own papers) — an explicit
  baseline-comparability selection criterion. The mesh-graph family is
  unique to this paper across BOTH tracks surveyed so far — a
  domain-specific workload (scientific-computing transport-sweep graphs)
  with no analogue in any BFS/SSSP/other-CC paper.
- **component-count verification / correctness**: explicit — **"We
  verified the solutions of all ECL-SCC runs by comparing them to the
  results obtained by Tarjan's algorithm"** — a direct reference-algorithm
  comparison (not just a component count, an actual per-vertex SCC
  membership check, per the paper's own signature-based representation).
- **timing protocol**: **"ran each experiment nine times and report the
  median runtime"**, with an explicit, disclosed exception: **"Due to very
  long runtimes, we were only able to run iSpan once on some of the
  inputs"** (some large-mesh iSpan runs exceeded 24 hours and were
  reported as ">1 hour" or excluded). For the MESH graphs specifically, the
  reported throughput is NOT simply "median of N repeated runs on one
  graph" — it is the **average runtime across all ordinates within a mesh
  family** (e.g. the beam-hex family has 30 distinct graphs/ordinates;
  their individual median-of-9 runtimes are themselves averaged, then
  throughput is computed from that doubly-aggregated number). This
  two-level aggregation (median-across-repetitions, then
  mean-across-distinct-graph-instances-in-a-family) is a genuinely
  different statistical convention from anything else surveyed in either
  track and should not be conflated with a simple repetition-count median.
- **timing scope**: explicitly instrumented to measure **only the SCC
  computation, excluding graph reading, output, and verification** — and
  further, explicitly **excludes host-device data transfer**, with a
  stated rationale: *"we are not advocating using GPU code for finding
  SCCs of graphs stored on the CPU. Rather, we are targeting environments
  where the graph is already on the GPU from a prior processing step"* —
  the most explicit host-device-transfer-exclusion justification found in
  either track's survey (contrast with the bfs track's spec, which
  requires OUTPUT-transfer exclusion but treats it as a default rather
  than something needing paper-level justification).
- **metric**: **throughput = vertices / runtime, in millions of vertices
  per second** — explicitly chosen over raw time specifically because it
  is "higher-is-better" and "normalized by the graph size, [making] the
  results less input-size dependent" — a third distinct metric convention
  in the connectivity-adjacent literature (neither GTEPS-style edge
  throughput nor raw wall-clock time), with its own stated justification.
- **baselines**: SCC-GPU (Li et al. 2017, "the fastest GPU code" the
  authors could find), iSpan (Ji et al. 2018, "the fastest parallel CPU
  code"). Results are graph-class-dependent in a way no other paper in
  either track surveys as explicitly: ECL-SCC is 6.2-8.4× faster than
  SCC-GPU on mesh graphs but only ~1-2× (sometimes SLOWER, e.g. 1.12×
  slower on beam-hex/Titan-V, 1.02× slower on twist-hex) on power-law
  graphs where SCC-GPU is specifically optimized — an honest, disclosed
  regime-dependent result rather than a single blanket speedup claim.
- **hardware**: TWO GPU generations (Volta Titan V, 12GB; Ampere A100,
  40GB) and TWO CPU vendors (AMD Ryzen Threadripper 2950X 16c/32t; Intel
  Xeon Gold 6226R dual-socket 32c/64t) — the most cross-platform-diverse
  single evaluation in either track's survey, explicitly to demonstrate
  robustness of the speedup trend across hardware generations/vendors
  rather than reporting one machine's numbers.

---

## 4. Optimized Block-Based Algorithms to Label Connected Components on GPUs ("BUF"/"BKE", TPDS 2020)

Source: artifact repo `prittt/YACCLAB` (the paper's benchmark IS the
YACCLAB open-source project itself) — top-level `README.md`,
`doc/config_cuda_2d.yaml` configuration file (read in full).

- **DOMAIN MISMATCH FROM THE OTHER 3 PAPERS**: this paper's "connected
  components" are **2D/3D binary-IMAGE pixel/voxel connectivity labeling**
  (standard 8-connectivity for 2D, and both 3D variants for the extension),
  a computer-vision/image-processing primitive on a **regular grid**, not
  general sparse graphs. The underlying computational structure (each
  pixel has a small, fixed, geometrically-determined neighbor set; the
  algorithm assigns each connected blob a unique integer label) is a
  fundamentally different workload shape from the other 3 papers' arbitrary
  sparse-graph inputs, even though "connected components" is the exact
  problem name in both cases. This paper's contribution — **BUF
  (Block-based Union-Find)** and **BKE (Block-based Komura Equivalence)**
  — optimizes the classic Union-Find CC algorithm family (the same family
  Shiloach-Vishkin and Afforest descend from) specifically for the
  block-structured memory-access pattern of image grids, a genuinely
  different optimization target from the sparse-adjacency-list algorithms
  the other 3 papers optimize.
- **algorithm family**: Union-Find (BUF) and Komura-Equivalence-based
  (BKE) — both are **block-based** extensions of classic sequential CCL
  algorithms, explicitly positioned in the README as optimizing "existing
  GPU solutions introducing a block-based approach" — i.e. a
  data-layout/memory-access optimization applied to a well-known algorithm
  family, not a new algorithmic family per se, distinct from the
  Tarjan-Vishkin-descended (BCC papers) and max-ID-propagation (SCC paper)
  families found elsewhere in this track.
  workloads/inputs: **named, real image datasets** (not synthetic
  grids/matrices): `3dpes`, `fingerprints`, `hamlet`, `medical`,
  `mirflickr`, `tobacco800`, `xdocs` — spanning surveillance video frames
  (3dpes), biometric fingerprint scans, historical document scans
  (hamlet, tobacco800, xdocs), medical imaging (medical), and general
  photography (mirflickr) — a workload-diversity axis (image domain, not
  graph topology) that has no analogue in the other 3 papers.
- **test taxonomy — the richest multi-axis test suite found in either
  track**: YACCLAB's `config_cuda_2d.yaml` defines SEVEN distinct test
  categories that can each be independently enabled: `correctness`,
  `average` (runtime), `average_with_steps` (per-phase runtime
  breakdown), `density` (performance vs. foreground-pixel density),
  `granularity` (performance vs. connected-component size distribution),
  `memory` (memory-access-count instrumentation), and `blocksize`
  (performance vs. CUDA thread-block dimensions, swept over a
  `2..64 x 2..64` grid). No other paper surveyed in either track defines
  this many independently-reportable test axes.
- **component-count / correctness verification**: a **dedicated
  `correctness` test category**, run separately from all timing tests,
  against `check`, `3dpes`, `fingerprints`, `hamlet`, `medical`,
  `mirflickr`, `tobacco800`, `xdocs` — explicitly checks label
  EQUIVALENCE (which pixels share a label, not just how many labels exist)
  against a reference labeling, a stronger check than a bare component
  count and closer in spirit to ADDS's exact-distance-diff convention in
  the sssp track than to the two BCC papers' count-only checks.
- **timing protocol**: `tests_number.average: 10` — **10 repetitions**
  for the average-runtime, density, granularity, and blocksize test
  categories (the config file's exact, disclosed default); the timing
  statistic computed from those 10 repetitions (mean vs. median) is not
  visible in the config file itself and would require reading
  `performance_evaluator.h`/`yacclab_test.h` in the C++ source, not
  reached in this survey (open question, flagged below).
- **timing scope**: the paper/benchmark is unusually explicit about
  DECOMPOSING timing scope into named phases via the `average_with_steps`
  category — this is the only paper in either track survey with a
  standardized, benchmark-framework-level (not just paper-specific
  ad-hoc) mechanism for reporting per-phase timing breakdowns as a
  first-class, independently-toggleable test category rather than a
  one-off figure.
- **metric**: average run-time (the benchmark's own term), presumably in
  a wall-clock unit, PLUS a distinct memory-access-count metric (the
  `memory` test category) as an independent, hardware-portable secondary
  metric — the ONLY paper in either track that treats memory-traffic
  instrumentation as a formal, independently-runnable test category rather
  than an incidental profiling aside.
  8-connectivity is stated as an invariant ("Notice that 8-connectivity is
  always used in the project") — i.e. the connectivity definition itself
  (an axis that would be meaningless for general sparse graphs, since
  "8-connectivity" only makes sense on a regular pixel grid) is fixed
  project-wide rather than being a per-paper choice, underscoring how
  different this sub-problem's parameter space is from the other 3
  papers'.
- **baselines**: an unusually large baseline set — the `algorithms:` list
  in the 2D GPU config alone names 18 competing algorithms (UF, OLE,
  STAVA, RASMUSSON, DLS, M8DLS, LBUF, BE, DLP, KE, HA8, C_SAUF, C_BBDT,
  C_DRAG, BUF, BKE, plus 2 commented-out entries BRB/ACCL) — the richest
  baseline set in either track's survey by a wide margin, reflecting
  YACCLAB's role as a standing community benchmark rather than a
  single-paper artifact.
- **hardware**: per the repo's own CI badge table, GPU testing has been
  run on an NVIDIA 2080Ti across two CUDA generations (9.2 and 11.4) as
  part of continuous integration, not a one-time paper-specific run — a
  distinct "benchmark maintained as living infrastructure" pattern absent
  from the other 3 (each a one-time paper artifact).

---

## Divergences

1. **Sub-problem identity.** The 4 papers span 3 genuinely different
   connectivity problems (BCC ×2, SCC ×1, image-pixel CCL ×1) that share
   only surface vocabulary, not a comparable operation, input format, or
   output shape. **Resolution**: rather than force a single WCC-shaped
   spec (which none of the 4 papers would actually populate), this spec
   defines separate variant families for BCC, SCC, and image-CCL,
   following the same "define what the papers actually measure" discipline
   the `bfs` spec's own survey applied when a paper's role was tangential
   (RCM) rather than distorting the shared variants to fit.

2. **Directed vs. undirected is not a within-paper choice here — it's a
   between-sub-problem constraint.** Both BCC papers REQUIRE undirected
   input and symmetrize directed graphs before benchmarking (paper #1
   additionally force-connects disconnected components by adding edges).
   The SCC paper is directed by definition — undirected input would make
   the problem meaningless (every SCC would trivially be a WCC). The
   image-CCL paper's inputs are neither — connectivity is defined
   geometrically on a pixel grid, not via an edge list at all.
   **Resolution**: the spec's BCC variant mandates symmetrization as part
   of its protocol (with the caveat, per paper #1, that "artificially
   force-connecting a disconnected graph before benchmarking BCC on it"
   is flagged as a fairness concern rather than silently adopted); the SCC
   variant mandates directed input and explicitly forbids symmetrization;
   the image-CCL variant does not apply the directed/undirected axis at
   all.

3. **Algorithm-family disclosure is present but at a different layer than
   the track brief anticipated.** The brief names "label-propagation /
   Shiloach-Vishkin / Afforest" as the axis to check — none of the 4
   papers' HEADLINE algorithm is literally one of these (the BCC papers'
   headline contribution is the spanning-tree/skeleton construction, not
   the CC-labeling subroutine it calls; the SCC paper's headline is
   max-ID propagation; the image-CCL paper's headline is Union-Find/
   Komura-block-optimization). However, the Afforest-family axis DOES
   appear, disclosed, one layer down: paper #1 explicitly names "k-out
   sampling" (an Afforest-descended technique) as its choice for the
   internal CC subroutine its BCC algorithm depends on. **Resolution**:
   the spec's BCC variant makes the CHOICE OF INTERNAL CC SUBROUTINE
   (which family — Shiloach-Vishkin log(n)-round pointer-jumping,
   Afforest/k-out sampling, or a from-scratch alternative) a REQUIRED
   disclosure field, since paper #1's own ablation-adjacent framing treats
   this choice as consequential ("we acknowledge that alternative sampling
   techniques could yield further improvements, depending on the graph
   type") without a dedicated ablation actually being run.

4. **Component-count/membership verification ranges from rigorous to
   unconfirmed.** FAST-BCC (paper #2) explicitly compares BCC COUNT
   against a sequential Hopcroft-Tarjan reference. ECL-SCC (paper #3)
   compares full SCC MEMBERSHIP (not just count) against Tarjan's
   algorithm. YACCLAB (paper #4) has a dedicated, separately-runnable
   `correctness` test category checking label EQUIVALENCE. Paper #1
   (FAST-BCC-Filter/PEA-BCC, IPDPS26) does NOT describe an explicit
   per-run correctness check in its reachable evaluation section — its
   artifact-evaluation appendix states correctness is judged by
   "comparing the generated result figures with the corresponding figures
   presented in the paper," which is a REPRODUCIBILITY check (do repeated
   runs match the paper's OWN numbers), not a CORRECTNESS check (are the
   BCC labels actually right) in the sense the other 3 papers apply.
   **Resolution**: the spec requires a component-count-or-membership
   check against an independent sequential reference (Hopcroft-Tarjan for
   BCC, Tarjan for SCC, a known-correct raster-scan labeler for image
   CCL) as a hard gate before any timing counts, following the strongest
   practice found (ECL-SCC's membership-level check) rather than the
   weakest (paper #1's implicit gap).

5. **Timing-statistic conventions differ within and across papers.**
   FAST-BCC (paper #2) uses median-of-10. FAST-BCC-Filter/PEA-BCC (paper
   #1) uses median-of-5 for GPU results but mean-of-5 (via `CPU_ROUNDS`)
   for its own CPU baseline numbers WITHIN THE SAME PAPER. ECL-SCC uses
   median-of-9, with an additional and genuinely different
   mean-across-distinct-graph-instances aggregation layered on top for its
   mesh-graph family specifically (not a simple repeated-run median at
   all). YACCLAB uses a disclosed repetition count (10) but an unconfirmed
   statistic (mean or median not visible in the reachable config file).
   **Resolution**: the spec fixes median-of-≥5 as the primary statistic
   for all connectivity variants (matching the majority convention and the
   `bfs`/`sssp` sibling specs' own choice), REQUIRES the statistic to be
   explicitly named per submission (since "5 runs, report a number" is
   ambiguous between mean and median as this survey's own findings show),
   and treats ECL-SCC's graph-family-averaging convention as inapplicable
   outside a literal multi-ordinate mesh-graph workload rather than
   generalizing it.

6. **Baseline-set scoping is disclosed but consequential in different
   ways.** Paper #1 explicitly EXCLUDES general-purpose frameworks
   (Gunrock, cuGraph, GBBS) from its BCC comparison with a stated
   rationale (no native BCC support / previously outperformed). Paper #2
   (FAST-BCC) INCLUDES GBBS as a general-purpose baseline specifically
   because it does have native BCC support. Both choices are individually
   reasonable and disclosed, but mean the two BCC papers' baseline sets
   are not directly comparable to each other. **Resolution**: the spec's
   `notes_on_fairness` documents this rather than forcing a single
   canonical baseline list, since "GBBS has BCC support" is a fact about
   GBBS's feature completeness at a point in time, not a methodological
   choice either paper controls.

## Open questions

- Paper #1 (FAST-BCC-Filter/PEA-BCC, IPDPS26)'s reachable evaluation
  section does not describe an explicit per-run BCC-correctness check
  (component count or label match against a reference); its own
  artifact-evaluation appendix substitutes a REPRODUCIBILITY check
  (matching the paper's own reported figures) for a CORRECTNESS check.
  Whether the artifact's `run_all.sh`/build system performs an
  undisclosed correctness gate not visible in the README/paper text could
  not be resolved without executing the code, which was out of scope for
  this survey.
- Whether FAST-BCC (paper #2)'s reported per-test timing includes graph
  loading/format-parsing time was not confirmed from the reachable
  README/script text; the benchmark script's CSV output format (writing
  `#CC,#BCC,Largest_BCC,time` per graph) is consistent with either
  interpretation.
- YACCLAB's exact timing statistic (mean vs. median across the
  disclosed 10 repetitions) was not confirmed — the `config_*.yaml` files
  disclose the repetition COUNT but not the aggregation function; this
  would require reading `performance_evaluator.h`/`yacclab_test.h` in the
  C++ source, which was not reached given this survey's time budget.
  Similarly, YACCLAB's exact correctness-reference implementation (which
  algorithm is treated as ground truth for the dedicated `correctness`
  test category) was not identified from the reachable README.
- Whether the two BCC papers' shared "force the input graph to be a
  single connected component by adding edges" preprocessing step
  (disclosed by paper #1, not explicitly confirmed for paper #2 though
  its README's "Make sure the input graph is a symmetric graph"
  requirement is silent on the single-component question) measurably
  changes the resulting BCC structure in a way that affects reported
  speedups was not investigated — this is flagged as a genuine
  representativeness concern in notes_on_fairness but not resolved
  quantitatively.

## Evidence

```yaml
fastbcc_filter_ipdps26: "arbitrary-spanning-tree skeleton (Theorem 3.3, not BFS-tree), internal CC subroutine = GConn k-out-sampling (Afforest-family, disclosed choice), forces symmetrization+single-connectivity on inputs, 30+ real graphs across 5 families w/ exact Table-II sizes, median-of-5 (GPU) vs mean-of-5 (CPU baseline, CPU_ROUNDS env var) statistic mismatch within one paper, no explicit per-run correctness check found (AE appendix substitutes reproducibility-vs-paper-figures), excludes general-purpose frameworks from baselines w/ stated reason, real Zenodo artifact w/ full AD/AE appendix"
fastbcc_ppopp23: "Tarjan-Vishkin-descended space-efficient skeleton (fence vs plain edges), requires symmetric/undirected input, 27 graphs across 5 families incl. UNIQUE k-NN and grid/chain synthetic topologies, explicit BCC-count-vs-sequential-Hopcroft-Tarjan correctness check, median-of-10 (strictest repetition count in track), metric=time+self-relative-speedup+vs-baseline-geomean (two distinct speedup axes), baselines=GBBS+SM14+TV(memory-only)+SEQ, CPU-only 96-core 4-socket, largest graphs in either track survey (Hyperlink12: 3.56B v/226B e)"
eclscc_sc23: "max-ID-propagation family (novel, distinct from FB/BFS-pivot every other SCC paper cited uses), directed-by-definition, two structurally distinct graph classes (MFEM RTE mesh graphs [unique to this paper] + 10 named SuiteSparse power-law graphs selected for baseline-comparability), explicit SCC-membership-vs-Tarjan correctness check, median-of-9 w/ disclosed single-run exceptions for very-long-running iSpan cases, UNIQUE two-level aggregation for mesh families (median-of-9-per-ordinate then mean-across-ordinates), metric=throughput(vertices/sec, explicitly justified as size-normalized), explicitly excludes host-device transfer w/ stated rationale (GPU-resident-graph assumption), baselines=SCC-GPU+iSpan, regime-dependent results honestly disclosed (loses to SCC-GPU on some power-law graphs), 2 GPU generations x 2 CPU vendors evaluated"
yacclab_tpds20: "DOMAIN MISMATCH: 2D/3D binary-image pixel/voxel CCL, not general sparse graphs; Union-Find(BUF)/Komura-Equivalence(BKE) block-based algorithms; named real image datasets (3dpes/fingerprints/hamlet/medical/mirflickr/tobacco800/xdocs); 7 independently-toggleable test categories (correctness/average/average_with_steps/density/granularity/memory/blocksize) - richest test taxonomy in either track; dedicated correctness category checking label equivalence; 10 repetitions disclosed (statistic mean-vs-median NOT confirmed); 18-algorithm baseline set (richest in either track); 8-connectivity fixed project-wide; living CI-tested benchmark infrastructure, not a one-time paper artifact"
```
