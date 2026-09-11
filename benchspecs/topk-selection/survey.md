# Top-K selection track — evaluation methodology survey

Track input: `data/track_inputs/topk-selection.json`, 3 papers, all 3
surveyed (2 via code-verified artifact + arXiv fulltext, 1 via code-verified
artifact only), above the >=2-paper minimum.

Operation surveyed: given an array (or per-query score set) of n elements,
return the k elements with the largest value, either as a value-only set or
as (value, index) pairs, computed exactly or approximately.

---

## 1. Zhang, Naruse, Li, Wang — AIR Top-K / GridSelect (SC'23)

- key: `conf/sc/ZhangNLW23`, artifact: github.com/ZhangJingrong/gpu_topK_benchmark
  (18 stars; author list confirmed via the repo's own README bibtex entry)
- **workloads/inputs** (code-verified, `benchmark/benchmark.cu`,
  `script/run-k.sh`, `script/run-n.sh`): synthetic arrays only. `n` (array
  length) swept as powers of two: `run-k.sh` fixes n_power in {15,20,25,30}
  (N = 32,768 .. 1,073,741,824) while sweeping k_power in {3..20} (k = 8 ..
  1,048,576, capped at k<N); `run-n.sh` fixes k_power in {5,8,15}
  (k=32,256,32768) while sweeping n_power in {11..30} (N = 2,048 ..
  ~1.07B). batch_size in {1} (run-k) or {1, 100} (run-n). Three data
  distributions, each with its own CLI flag in `test_util.h`/`benchmark.cu`:
  **Uniform** (`curandGenerateUniform`, (0,1]), **Normal**
  (`curandGenerateNormal`, mean=0, stddev=1, via `-g`), and an explicitly
  **adversarial "Unfriendly"** distribution (`-r 12`): `fill_random_bits()`
  masks all but the low 12 bits of each 32-bit float's bit pattern to a
  fixed high-bit prefix, specifically designed to defeat radix-select's
  early-termination pruning (low entropy in the bits a radix pass would
  normally discriminate on) — a genuine correctness/robustness stress test,
  not just another speed data point.
- **timing protocol** (code-verified, `benchmark.cu` ~L70-140,
  `script/run-k.sh`/`run-n.sh`): CLI flags `-w <warmup_niter>` /
  `-n <niter>`; the sweep scripts fix **warmup=10** and **niter=100**
  (dropped to niter=20 for `faiss_block`/`faiss_warp` when N>524,288, to
  keep total sweep time tractable). Each of the `niter` timed iterations
  re-generates fresh random test data (`TestData` re-drawn per iteration)
  rather than reusing one buffer — deliberately excludes any
  data-dependent caching effect across iterations. Timer: `nv::Timer`
  (CUDA-event-based, `elapsed_ms()`). Reported as **mean +/- stddev** over
  the `niter` timed calls (`calc_mean_and_stddev`); no median/min/max
  captured.
- **timing scope**: kernel-only — CUDA malloc/data-generation is excluded
  from the timed region (timer starts after `TestData` construction
  completes and a stream sync); H2D of the input array is implicit in
  `TestData`'s constructor (device-resident curand fill, no host buffer
  round-trip), so there is effectively no separate H2D cost to exclude. No
  preprocessing/format-conversion step exists for this kernel class — the
  fastest algorithms compared (bitonic, radix, sample-select, bucket-select)
  all operate directly on the raw unsorted array.
- **precision & correctness**: `data_t = float` (fp32), `idx_t = int`.
  Correctness (`-c` flag, `check_result()` in `test_util.h`) is checked
  **only on the last of the `niter` iterations** (to save time): validates
  (a) every returned index is `< len`, (b) every returned index maps back
  to its claimed value in the original input (`in[idx[i]] == v[i]`), (c) no
  duplicate indices in the returned set, and (d) the **sorted top-k value
  multiset** matches the CPU ground truth's sorted top-k value multiset
  exactly (bit-exact `!=` on floats, no tolerance) — ties may resolve to
  different indices as long as the value sets match.
- **metric**: mean wall time (ms) per call; no GFLOP/s or GB/s formula
  (top-k is a data-dependent selection primitive, not a fixed-flop kernel)
  — the paper's own plots (`k-as-x.py`, `n-as-x.py`) are log-scale
  running-time-vs-k and running-time-vs-N curves per
  algorithm/distribution/batch combination, plus a `speedup.py`-computed
  speedup table.
- **baselines**: 10 algorithms compared head-to-head in one harness: `cub`
  (NVIDIA CUB sort-based), `sampleselect` + `sampleselect-bucket` +
  `sampleselect-quick` (3 SampleSelect variants), `drtopk_radix` +
  `drtopk_bitonic` (prior SC-published DrTopK), `faiss_warp` +
  `faiss_block` (FAISS WarpSelect/BlockSelect), `raft_radix_
  11bits_extra_pass` (labeled "AIR Top-K" in the paper's own plotting
  code), `grid_select` (their second method). Claims up to 574x over prior
  radix top-K, 1.98-21.48x depending on batch size (per track-input
  abstract).
- source: repo `README.md`, `benchmark/benchmark.cu`,
  `benchmark/test_util.h`, `script/run-k.sh`, `script/run-n.sh`,
  `script/k-as-x.py`, `script/n-as-x.py`.

## 2. Parravicini, Cattaneo, Sironi, Santambrogio, et al. — Top-K SpMV FPGA (DAC'21)

- key: `conf/dac/ParraviciniCSS21`, arXiv:2103.04808, artifact:
  github.com/AlbertoParravicini/approximate-spmv-topk. This paper's compute
  kernel is top-k-fused-SpMV — a sparse-embedding-similarity row score
  followed by an on-chip top-k selection of the highest-scoring columns per
  row/query. It is also surveyed from the SpMV angle in
  `benchspecs/spmv/survey.md` entry #8, which explicitly carves it out as
  "a related-but-distinct benchmark (top-k sparse similarity search), out
  of scope for the general SpMV spec" — this is that follow-up survey.
  Facts below are cross-verified against that entry's code-verified
  findings (both drawn from the same `test_spmv_topk.py` harness and arXiv
  fulltext) rather than re-derived from scratch.
- **workloads/inputs**: mostly **synthetic** (`test_spmv_topk.py`): rows in
  {5M, 10M, 15M}, embedding cols (= the top-k candidate-set width per row)
  in {512, 1024}, avg nnz/row in {20, 40}, two distributions — **uniform**
  and **left-skewed Gamma** — plus one **real** dataset: sparsified GloVe
  embeddings (2.196M rows x 300 cols, 54.9M nnz). K fixed at 100.
- **timing protocol**: `NITER=30` repetitions per config; no explicit
  warmup flag found in the harness (a divergence from AIR Top-K's explicit
  warmup=10).
- **timing scope**: reduced-precision packet-wise CSR compression is a
  build-time/host preprocessing step; whether it (or host<->FPGA DMA) sits
  inside the timed window is not clearly stated in the reviewed material —
  unlike Serpens's explicit "Kernel time" label in the SpMV track, no
  equivalent label was found here.
- **precision & correctness — the paper's headline technique**: reduced
  fixed-point precision tested at 20-bit (Q1.19), 25-bit (Q1.24), 32-bit
  fixed (Q1.31), and 32-bit float. Correctness is evaluated with
  **ranking-quality metrics** — Precision, Kendall's tau, NDCG — against an
  exact CPU baseline, NOT elementwise numeric tolerance: the output is an
  approximate top-K *index set*, and reduced precision can change which
  elements are even members of the top-K set, so an elementwise score-error
  bound doesn't capture what "correct" means here.
- **metric**: execution time / speedup; a roofline-style plot shows
  attained GB/s vs peak HBM bandwidth.
- **baselines**: CPU = 2x Intel Xeon Gold 6248 running `sparse_dot_topn`
  (multithreaded); GPU = Tesla P100 (549 GB/s), cuSPARSE + Thrust radix
  sort. Claims 100x vs CPU, 2x vs GPU with 20% higher bandwidth, 14.2x
  power-efficiency.
- source: `benchspecs/spmv/survey.md` entry #8 (code-verified via
  `test_spmv_topk.py`, arXiv fulltext via ar5iv — both re-confirmed for
  this track's purposes).

## 3. Chen et al. — UpANNS (SC'25)

- key: `conf/sc/ChenZSLY25`, artifact: github.com/steam009/UpANNS (branch
  `UpANNS-SIFT1B`, the repo's default branch). No arXiv preprint exists;
  the DOI/ACM link is paywalled, so this entry is code-verified only.
- **workloads/inputs** (code-verified, `README.md`, `run_*.sh`,
  `search_host.cpp`): three real billion-scale embedding datasets —
  **SIFT** (128-dim, up to 1B vectors, corpus-texmex.irisa.fr), **DEEP**
  (96-dim, 1B vectors, Yandex), **SPACEV** (Microsoft SPTAG, 1B vectors) —
  a genuinely different regime from the other two papers' flat arrays: N
  up to 1e9 points, but each query only asks for a small k against an
  IVF-PQ index (not a flat-array top-k). `TOPK` swept in `run_topk.sh` over
  {100, 10, 1}; `NPROBS`/`nprobe` (search-width knob, trades recall for
  latency) swept in `run_nprobs.sh`/`run_nprobe.sh` over {32, 64, 128,
  256}; `NR_DPUS` (PIM device count, the parallel-hardware-scale knob)
  swept 500-900 in `run_scale.sh`; batch size (`BS`) swept in
  `run_cputopk.sh` over {10, 100, 1000}.
- **timing protocol** (code-verified, `search_host.cpp` L170-1490):
  wall-clock `elapsed()` (a `gettimeofday`-style timer) bracketing **named
  phases**, not a warmup/repeat loop — this is a single large batch of
  `nq_test` queries (e.g. 9,000+ in the reviewed config) run **once**, with
  per-phase timestamps printed: "Pre process the query", "send the query
  to DPU and begin search" (`t2-t1`), "finish search" (`t3-t2`, the actual
  DPU/PIM `system.exec()` call), "DPU-CPU" (`t4-t3`, result transfer),
  "Post-process" (`t5-t4`, host-side merge+sort). No iteration count, no
  warmup, no repeated-trial statistic (mean/median/stdev) anywhere in the
  reviewed code — a single run's wall time is what gets printed.
- **timing scope**: the measured region excludes offline index
  construction (`demo_*.cpp` "train"/"add" steps, run separately as
  `train_sift1B.cpp` etc., never timed in the same run as search) but
  *includes* the coarse-quantizer (IVF) search step, PIM data transfer, and
  host-side result merge/sort — the full online query pipeline end to end,
  not an isolated "kernel-only" top-k call.
- **precision & correctness**: PQ-encoded (product-quantized) vectors,
  `KSUB=256` codebook rows, `MS=16`/`DSUB=8` sub-quantizer config (i.e.
  approximate distances, not exact). Correctness is **Recall@1, @10, @100**
  against an exact ground-truth nearest-neighbor file (`gt`) shipped with
  each dataset — the standard ANN-benchmark accuracy metric, fundamentally
  different from both other surveyed papers' correctness models (AIR
  Top-K's exact-set equality; Top-K SpMV's ranking-quality metrics against
  its own exact computation) since here the ground truth is the true
  nearest neighbor in a real, continuous embedding space, not a synthetic
  array whose exact top-k is trivially computable.
- **metric**: no GFLOP/s or GB/s; reported as wall time (s) per named
  phase, plus Recall@{1,10,100} — a latency-at-a-recall-operating-point
  report, not a throughput number. Abstract claims 4.3x QPS vs Faiss.
- **baselines**: CPU Faiss (`demo_sift1B.cpp`/`demo_deep1B.cpp`/
  `demo_SPACE1B.cpp`, same NPROBE/NLIST/TOPK/BS params via `-D` macros) and
  a "PIM-naive" ablation branch (per README: "PIM-naive-{dataset}").
- source: repo `README.md`, `search_host.cpp`, `run.sh`, `run_topk.sh`,
  `run_nprobs.sh`, `run_nprobe.sh`, `run_scale.sh`, `run_cputopk.sh`,
  `run_cpu.sh`, `run_etd.sh`.

---

## Divergences

- **Exact vs. approximate top-k are not one axis with two settings; this
  track's three papers occupy three genuinely different points.** AIR
  Top-K/GridSelect computes the mathematically exact top-k of a given array
  (correctness = exact value-multiset equality). Top-K SpMV computes an
  approximate top-k under a REDUCED-PRECISION score computation (the scores
  themselves are approximate, so "correct" is redefined as ranking-quality
  vs. an exact reference). UpANNS computes an approximate top-k over an
  APPROXIMATE INDEX (IVF-PQ — even at full precision the search only visits
  `nprobe` of `nlist` partitions, never guaranteeing the true nearest
  neighbors), so correctness is Recall@k against ground truth. Three
  different failure modes force three different correctness models;
  collapsing them into one variant would hide which failure mode a
  technique addresses.
- **The "n" axis means three different things.** AIR Top-K: n is the exact
  size of the array top-k'd in a single kernel call (up to ~1.07B elements,
  entirely GPU-resident). Top-K SpMV: n is the per-row candidate count
  (512-1024 embedding columns), with millions of ROWS (independent top-k
  calls) batched together — a batched-many-small-top-k regime very
  different from AIR Top-K's few-huge-top-k regime. UpANNS: n is the total
  DATASET size (up to 1e9 vectors), but no single top-k call ever touches
  all of n — an IVF quantizer prunes to `nprobe` partitions first, so the
  effective per-call candidate count is a small, index-dependent fraction
  of n. The spec's exact-kernel variant's n-sweep is therefore NOT
  comparable to UpANNS's or Top-K SpMV's "dataset size" numbers, and the
  spec keeps them as separate variants rather than one shared n axis.
- **Only one paper (AIR Top-K) tests non-uniform/adversarial input
  distributions as a first-class axis**, and it is also the only paper
  whose top-k algorithm family (radix-select) has a well-documented
  distribution-dependent worst case (bucket/radix passes prune based on
  bit-value spread; an adversarial low-entropy distribution defeats that
  pruning). Top-K SpMV's uniform-vs-left-skewed-Gamma split targets a
  different concern (realistic embedding-similarity score distributions,
  not adversarial algorithm-breaking inputs). UpANNS uses only real
  embedding data with no synthetic distribution sweep at all. The spec's
  exact-kernel variant is the only one that inherits a genuine
  adversarial-input requirement, since it is the only variant whose target
  algorithm family is documented to be input-distribution-sensitive.
- **None of the three surveyed papers benchmarks a value-only (no-index)
  output mode.** All three return (value, index) pairs — AIR Top-K's
  `d_out`/`d_out_idx`, Top-K SpMV's ranked column-index list, UpANNS's
  `faiss::idx_t* I` — because in every real use case surveyed here (array
  top-k with provenance, sparse similarity search, ANN search) the INDEX
  is the actually-wanted output and the value is incidental (used only for
  the correctness check). The spec's key/value axis is therefore fixed at
  value+index for all three variants (the only setting grounded in this
  track's literature); a value-only mode is left as an untested extension
  in `open_questions`.
- **Timing rigor spans "commodity GPU-benchmarking practice" to "no
  repeated-trial statistic at all."** AIR Top-K: warmup=10, niter=100,
  mean+-stddev over CUDA-event-timed calls, fresh random data per call.
  Top-K SpMV: NITER=30, no warmup found. UpANNS: a SINGLE wall-clock-timed
  run of one large query batch, no warmup, no repetition, no statistic
  beyond that one number, per named phase — the weakest timing protocol of
  any paper surveyed across topk-selection AND (per
  `benchspecs/spmv/survey.md`) the SpMV track's 9 SpMV-implementing papers.
  The spec fixes warmup=20/reps=100/median+range for the exact-kernel
  variant and requires the ANN variant to run the SAME query batch >=20
  times and report a real distribution instead of one number.

## Recommended n/k provenance

`topk-array-kernel-exact`'s n range (2^15 to 2^30) and k-sweep pattern (a
small-k regime matching realistic ML/DB top-k usage, plus a k-as-
fraction-of-n power sweep) are taken directly from AIR Top-K's own
`run-k.sh`/`run-n.sh` — the only surveyed paper with a systematic n/k
sweep; the three data distributions (Uniform, Normal, adversarial
"Unfriendly") are AIR Top-K's own three CLI-selectable distributions,
adopted verbatim rather than invented.
