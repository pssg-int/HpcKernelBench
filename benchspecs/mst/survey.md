# mst track — evaluation survey

Track input: `data/track_inputs/other_mst.json` (2 papers, both surveyed
via fulltext).

## conf/ipps/SandersS23 — Engineering Massively Parallel MST Algorithms (IPDPS 2023)

Source: full fulltext via ar5iv.labs.arxiv.org/html/2302.12199 (arXiv
mirror of the IPDPS paper) + GitHub `mschimek/kamsta`
(`evaluation/graphs.yaml`, `benchmarks/mst_benchmarks.hpp`,
`benchmarks/mst.cpp`, `evaluation/README.md`).

- **Algorithm family**: a scalable, distributed-memory variant of
  Boruvka's algorithm ("hybridBoruvka"), plus a "Filter-Boruvka" variant
  that adapts the filtering concept from the best practical sequential
  MST algorithm (Osipov et al.'s Filter-Kruskal — also cited by ECL-MST
  below, an interesting cross-paper connection) to the distributed
  setting for graphs with poor locality / high average degree. A local
  contraction ("local kernelization") preprocessing step is applied for
  partitioned graphs with many local edges. The repo's
  `benchmarks/mst_benchmarks.hpp` confirms exactly these two algorithm
  names (`hybridBoruvka`, `filter_hybridBoruvka`) as CLI-selectable
  options, plus `local_kernelization_level` and `filter_threshold`
  tuning parameters matching the paper's description.
- **Workloads/inputs**: two suites.
  - Synthetic (KaGen-generated, confirmed by the repo's
    `evaluation/graphs.yaml` and `mst_benchmarks.hpp` `GraphType` enum):
    2D-GRID, 2D-RGG, 3D-RGG, RHG (random hyperbolic), GNM (random,
    Erdos-Renyi-style), and RMAT. Weak scaling fixes 2^17 vertices and
    2^21 edges PER CORE, scaled up with core count. The repo's own
    example config uses `log_n=17, log_m=20` with a core sweep [8, 16,
    32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] and thread
    counts [1, 8] — i.e. up to 16384*8 = 131,072 logical execution
    contexts in the shipped example, consistent with the paper's stated
    "up to 65,536 cores" (cores vs. hardware-threads distinction not
    fully disambiguated in the fetched text).
  - Real-world (strong scaling): 6 named graphs — friendster, twitter,
    uk-2007, it-2004, wdc-14, US-road — spanning ~24M to ~1.7B vertices
    and 57M to 123B edges.
- **Weight distribution**: edge weights drawn UNIFORMLY AT RANDOM from
  [1, 255) (an 8-bit-ish integer range), following prior methodology per
  the fetched summary. The repo's `graphs.yaml` independently confirms
  `max_edge_weight: 254` as the default — i.e. weights in [1,254], the
  same effective range. This uniform, wide-range, low-tie-probability
  weight distribution is a DELIBERATE choice that keeps Boruvka's
  lightest-edge-per-component step nearly always unambiguous (few ties),
  which matters for this track's dedicated weight-distribution variant
  below.
- **Timing protocol**: each configuration run at least 3 times with 1
  warmup round discarded, and the paper reports ONLY THE MEAN (stating
  variance is "usually very low"). This is a materially weaker
  statistical-reporting practice than this repo's established convention
  (median + min/max/stdev, e.g. `spmv/spec.yaml`, `bfs/spec.yaml`) and is
  flagged in notes_on_fairness. The repo's `mst_benchmarks.hpp`
  `CmdParameters::iterations` (default via `-i`) is a direct rep-count
  knob confirming the paper's stated "at least three times."
  Graph-generation time and encoding are handled with a specific,
  disclosed exclusion: "the time for encoding is not included... however,
  we account for decoding the compressed edge list twice" — i.e. one of
  the surveyed papers' more precise (if unusually specific) timing-scope
  statements in this whole repo's corpus so far.
- **Correctness**: an explicit `do_check` boolean flag exists in the CLI
  (`cp.add_bool('c', "checks", ...)`), and the repo's `mst_benchmarks.hpp`
  `check()` function computes the actual MST's total edge-weight sum via
  `sum_edge_weights` + a distributed `mpi::allreduce_sum`, and compares it
  against an expected MST gathered via `gather_mst(input)` (a reference
  computation, presumably sequential, run on the same input) — this is
  EXACTLY the "total-weight verification as correctness gate" this
  track's brief calls for, confirmed directly from source code rather
  than inferred from paper text. The check is opt-in (off by default in
  the CLI), which is itself a fairness note (see below).
- **Baselines**: two prior distributed MST algorithms — "sparseMatrix" (Baer
  et al.) and "MND-MST" (Panja et al.).
- **Metric**: running time (s), throughput in edges/second, speedup (up to
  800x over prior distributed algorithms), weak- and strong-scaling plots
  across core counts.

## conf/sc/FallinGSB23 — A High-Performance MST Implementation for GPUs / ECL-MST (SC 2023)

Source: full fulltext PDF via the authors' self-hosted mirror
(`cs.txstate.edu/~mb92/papers/sc23b.pdf`, since the ACM DL PDF returned
403 to automated fetch) + GitHub `burtscher/ECL-MST`
(`source/ECL-MST_10.cu`, README).

- **Algorithm family**: a single-GPU implementation that starts from
  fully parallelizing Kruskal's algorithm (chunk-based, deterministic
  reservations, no sorting once two specific optimizations are combined)
  and demonstrates it CONVERGES to a parallelization of Boruvka's
  algorithm — i.e. the paper's central technical claim is that, once
  fully parallelized on a GPU, "Kruskal's" and "Boruvka's" cease to be
  distinguishable implementations. This is directly relevant to this
  track's "algorithm family" axis: ECL-MST is evidence that the
  family label itself becomes ambiguous under aggressive parallelization,
  which the spec below treats as a disclosure point rather than a strict
  categorical filter.
- **Workloads/inputs**: 17 named real+synthetic graphs (Table 2 in the
  paper, reproduced with exact edge/vertex/type/connected-component/avg-degree/max-degree
  stats): 2d-2e20.sym (grid, 4.19M e / 1.05M v), amazon0601 (co-purchases,
  4.89M e / 403K v), as-skitter (Internet topology, 22.2M e / 1.70M v),
  citationCiteseer (publication cit., 2.31M e / 268K v), cit-Patents
  (patent cit., 33.0M e / 3.77M v), coPapersDBLP (publication cit., 30.5M
  e / 540K v), delaunay_n24 (triangulation, 100.7M e / 16.8M v),
  europe_osm (road map, 108.1M e / 50.9M v), in-2004 (web links, 27.2M e
  / 1.38M v), internet (Internet topology, 387K e / 125K v),
  kron_g500-logn21 (Kronecker synthetic, 182.1M e / 2.10M v), r4-2e23.sym
  (random synthetic, 67.1M e / 8.39M v), rmat16.sym / rmat22.sym (RMAT
  synthetic, 968K e/65.5K v and 65.7M e/4.19M v), soc-LiveJournal1
  (community, 85.7M e / 4.85M v), USA-road-d.NY (road map, 730K e / 264K
  v), USA-road-d.USA (road map, 57.7M e / 23.9M v). Sourced from DIMACS,
  Galois, SNAP, and the SuiteSparse Matrix Collection — explicit
  inclusion criterion stated as "cover a wide range of types and sizes."
  Self-loops and multi-edges removed; missing back-edges added to
  guarantee undirectedness; unweighted graphs get RANDOMLY INSERTED
  weights so an MST is computable at all (weight-generation method for
  the paper's own reported numbers not further specified beyond "random,"
  distinct from the demo weight-assignment formula visible in the repo's
  shipped `main.cu`, which uses a specific deterministic hash
  `1 + ((i*k) % g.nodes)` — flagged as a discrepancy between the
  paper's stated randomness and the repo's actual demo code).
- **Weight distribution / random-seed sensitivity**: ECL-MST's own filter
  threshold is picked from 20 randomly-sampled edge weights (targeting
  ~3x the eventual tree size); the paper runs a DEDICATED 99-random-seed
  sensitivity study (Section 5.4) showing that for most graphs the choice
  of seed barely matters, but for high-average-degree scale-free graphs
  (coPapersDBLP specifically) throughput varies by nearly 4x across seeds
  — i.e. the paper itself demonstrates that weight/threshold-sampling
  randomness is NOT a negligible axis for this algorithm family, directly
  supporting this track's brief to make weight distribution a first-class
  benchmark axis rather than a fixed, unreported choice.
  Filtering itself is only applied when average degree >= 4 (a
  disclosed, hard-coded threshold `c=4`), confirmed both in paper text
  and in `ECL-MST_10.cu`'s `avg_deg >= 4` check.
  edge weight ranges are graph-native (int, unbounded beyond what the
  synthetic-insertion scheme produces) rather than a fixed [1,255)-style
  range — a real divergence from KaMSTa's own weight-generation choice
  (see Divergences).
- **Timing protocol**: 9 repetitions per input per code, MEDIAN reported
  (this is the strongest statistical-reporting practice found across all
  6 papers surveyed for this batch of 3 tracks, and matches this repo's
  own established convention exactly). Variance is reported as
  extremely low: 0.0005% average / 0.0028% max on System 1, 0.0014%
  average / 0.0107% max on System 2. Graph-read time is EXCLUDED from
  all reported numbers; for GPU codes, host-device memory-transfer time
  is ALSO excluded by default but a SECOND, separately-reported "memcpy"
  variant of ECL-MST's own numbers is provided that includes it (Tables
  3-4: `ECL-MST` vs. `ECL-MST memcpy` rows) — an unusually complete example
  of the "preprocessing/transfer reported separately, never silently
  mixed in" fairness principle this whole repo's specs are built around.
- **Correctness**: the GPU result is compared against a SERIAL CPU
  reference Kruskal implementation at the end of every run; the paper's
  own `verify()` function (confirmed in `ECL-MST_10.cu`) checks BOTH the
  total MST weight AND per-edge set agreement (`onlyCpuMST`/`onlyGpuMST`
  mismatch counts) — stricter than pure total-weight verification, which
  is itself a fairness concern flagged below (MST is not unique under
  tied weights, so a valid-but-differently-tie-broken MST could
  incorrectly fail an edge-set-identity check even though its total
  weight matches). Verification time is explicitly excluded from the
  measured runtime.
- **Baselines**: 8 total — 4 GPU-parallel (Gunrock, Jucele/Vasconcellos et
  al., RAPIDS cuGraph [float and double variants], UMinho/Sousa et al.),
  3 CPU-parallel (Lonestar, PBBS, UMinho CPU), 1 CPU-serial (PBBS
  Serial) — the most thoroughly baselined paper of all 6 surveyed across
  the 3 tracks in this batch.
- **Metric**: absolute runtime (s), throughput (mega-edges/second =
  edges/runtime), geometric-mean speedup, with results split into "MSF
  GeoMean" (all 17 inputs, multi-component-capable codes only) vs. "MST
  GeoMean" (only single-connected-component inputs, so Jucele/Gunrock —
  which can only produce an MST, not a general MSF — can be fairly
  included). This MSF-vs-MST-capability-aware geomean split is a genuine
  fairness mechanism worth preserving in the spec below.
- **Hardware**: two full systems reported (Titan V + AMD Threadripper
  2950X/16-core; RTX 3080 Ti + dual Intel Xeon Gold 6226R/16-core each),
  both GPU generations show consistent 4.5x-4.6x average speedup over the
  next-fastest code, supporting a "not GPU-generation-specific" claim.

## Divergences

- **Scale/platform**: KaMSTa targets distributed-memory clusters up to
  65,536 cores on graphs up to 123 billion edges; ECL-MST targets a
  SINGLE GPU on graphs up to ~180M edges (the paper states all tested GPU
  codes, including its own, cap out around 2 billion vertices+edges due
  to GPU memory). These are not directly comparable at the same
  hardware scale, which is why the spec below keeps them in separate
  variants rather than trying to force one throughput number to span
  both regimes.
- **Weight distribution**: KaMSTa uses a fixed, narrow, uniform-random
  integer range [1,254]; ECL-MST uses graph-native/synthetically-inserted
  weights with an UNSTATED distribution for its own headline numbers
  (only the demo code's specific hash-based assignment is visible), but
  DOES explicitly study weight/threshold-sampling-seed sensitivity as a
  dedicated experiment. Neither paper sweeps the weight distribution
  itself as a first-class reported axis (both treat it as a fixed
  choice, with ECL-MST at least probing seed sensitivity around its
  filter threshold) — this is exactly the gap this track's brief asks
  the spec to fill directly.
- **Correctness check strength**: KaMSTa checks total-weight-sum equality
  only (the numerically correct invariant given possible weight ties);
  ECL-MST checks BOTH total weight AND per-edge-set identity (stricter,
  and potentially over-strict under weight ties). The spec below adopts
  KaMSTa's weaker-but-correct total-weight check as the MANDATORY gate
  and treats ECL-MST-style edge-set agreement as an optional stricter
  diagnostic, not a required pass/fail criterion.
- **Statistical reporting**: KaMSTa reports mean-of->=3-runs;
  ECL-MST reports median-of-9-runs plus explicit variance figures. The
  spec adopts ECL-MST's stronger convention uniformly, consistent with
  every other spec.yaml already in this repo.
- **Algorithm-family labeling**: KaMSTa cleanly separates "hybridBoruvka"
  vs. "filter_hybridBoruvka" as two distinct, CLI-selectable algorithms.
  ECL-MST's central finding is that a fully-parallelized Kruskal IS a
  Boruvka parallelization — the family label becomes an implementation
  detail, not a meaningfully distinguishing choice, at GPU scale. The
  spec's algorithm_family disclosure field is worded to accommodate both
  framings.
- **Correctness-check opt-in-ness**: KaMSTa's `do_check` is off by
  default in its own CLI (an opt-in flag); ECL-MST's verification runs
  unconditionally as part of every reported run. The spec makes
  correctness-gating mandatory for ANY number to count, regardless of
  either paper's own default.

## Open questions carried into spec.yaml

- The exact core-count vs. hardware-thread-count semantics of KaMSTa's
  "up to 65,536 cores" claim (physical cores vs. logical/hyperthreaded
  contexts) could not be fully disambiguated between the paper text and
  the repo's example config (which sweeps up to 16,384 MPI ranks x 8
  threads = 131,072 logical contexts).
- ECL-MST's paper-reported weight-generation method for unweighted graphs
  is stated only as "random," while the repo's own demo code
  (`main.cu`) uses a specific deterministic hash formula — whether this
  hash-based scheme is what actually produced the paper's Table 2-5
  numbers, or whether a different (perhaps truly-random) scheme was used
  for the paper itself and the hash-based version is only a
  convenience default for demo/reproducibility purposes, is unresolved.
- Whether KaMSTa's real-world-graph weight assignment (the 6 strong-scaling
  graphs: friendster, twitter, uk-2007, it-2004, wdc-14, US-road) uses the
  same [1,254] synthetic range as its synthetic-graph generator, or
  whether any of those 6 graphs carry native edge weights that were used
  instead, could not be confirmed from the fetched text.
