# Survey — content-defined-chunking (singleton track: 1 paper)

## journals/tpds/UdayashankarMA26 — Vectorized Sequence-Based Chunking for Data Deduplication (TPDS 2026)

- **What is measured**: SeqCDC, a content-defined chunking (CDC) algorithm for data deduplication —
  lightweight boundary detection + content-defined SKIPPING (avoiding re-examining bytes unlikely
  to be a boundary) + SSE/AVX/NEON/AltiVec vectorization, benchmarked via the authors' own
  DedupBench harness (a separate, actively maintained multi-paper artifact: DedupBench 2023,
  VectorCDC/FAST'25, SeqCDC/Middleware'24 + this TPDS'26 journal extension, and a low-entropy-
  analysis CLOUD'24 paper all share this one repo).
- **workloads/inputs**: DedupBench supports **11 chunking algorithms** (AE-Max, AE-Min, CRC-32,
  FastCDC, Fixed-size, Gear, MAXP, Rabin, RAM, SeqCDC, TTTD) and **6 fingerprinting/hashing
  algorithms** (MD5, SHA1, SHA256, SHA512, MurmurHash3-128, xxHash3-128), with SIMD acceleration
  (unaccelerated, SSE-128, AVX-256, AVX-512, ARM NEON-128, IBM VSX-128/AltiVec) available for a
  subset (AE, MAXP, RAM, and — this paper's own contribution — SeqCDC). The canonical evaluation
  dataset is the **DEB (Deduplication Evaluation Benchmark) dataset**: VM disk images from Bitnami
  stacks (tomcatstack, mysql, rubystack, jenkins, ejbca-singlevm, kafka, elasticsearch,
  airflow-singlevm, opencart, grafana, redis — 11 named images, some retired since Sept 2024), used
  across the FAST'25/Middleware'24/TPDS'26 papers; a synthetic `random_dataset` (zero real
  deduplication, throughput-only) is also bundled for quick smoke tests. Chunk-size distribution is
  configured per-algorithm (`chunking_algo`-specific parameter section in `config.txt`), with 8KB
  average chunk size as the repo's own convention for its preconfigured runs.
- **timing protocol**: not explicitly itemized (no visible warmup/repeat/statistic stated in the
  README); the harness prints "Throughput and avg chunk size... to stdout" from a single
  `dedup.exe <dataset_dir> config.txt` invocation — appears to be a single-pass measurement over
  the whole dataset directory, not a repeated-and-averaged microbenchmark loop.
- **timing scope**: end-to-end chunking + fingerprinting/hashing over a full dataset directory
  (`dedup.exe` runs both stages together per the config); the paper's own abstract explicitly
  frames the throughput number as **chunking algorithm cost** ("modern data chunking algorithms are
  slow and compute-intensive because they scan large amounts of data while simultaneously making
  data-driven boundary decisions") — whether the released harness's single throughput number
  isolates chunking-only cost from the hashing-stage cost is not stated in the fetched README (the
  two are likely fused in `dedup.exe`'s single pass over the data).
- **precision & correctness**: CDC boundary-finding is deterministic given the algorithm + content
  (same input bytes always produce the same chunk boundaries) — correctness here means dedup SPACE
  SAVINGS achieved on real content matches a reference/exact computation, not a numerical tolerance;
  `measure-dedup.exe hash.out` is the artifact's own dedicated space-savings-measurement tool, run
  as a SEPARATE step from the throughput-producing `dedup.exe` run.
- **metric**: throughput (the paper's abstract states SeqCDC achieves "10x higher throughput than
  unaccelerated and 1.2x-1.35x higher throughput than vector-accelerated data chunking algorithms");
  space savings % (secondary, explicitly noted in the abstract as "minimally affecting deduplication
  space savings" — i.e. the paper's own framing already treats throughput and dedup-ratio as a
  joint claim, matching this track's brief).
- **baselines**: the other 10 chunking algorithms in the same harness (both unaccelerated and, where
  supported, SIMD-accelerated versions) — an apples-to-apples same-codebase comparison, similar in
  spirit to set-intersection/Jaccard-ML's same-repo baseline structure.
- **source**: abstract (`output/included.json`); repo README
  (`gh api repos/UWASL/dedup-bench/contents/README.md`).

## Divergences

- Single-paper track: no cross-paper divergence in the formal sense, but the repo itself spans
  multiple co-published papers (DedupBench/CCECE'23, VectorCDC/FAST'25, SeqCDC/Middleware'24, this
  TPDS'26 extension, a CLOUD'24 low-entropy study) sharing one harness — this spec is scoped to
  SeqCDC (the TPDS'26 paper actually in this track) but necessarily benchmarks it against the
  other 10 algorithms bundled in the same tool, since that IS the paper's own comparison set.
- The paper's throughput claim ("10x over unaccelerated, 1.2-1.35x over vector-accelerated
  alternatives") is stated as a range without per-chunk-size or per-dataset breakdown in the
  fetched abstract — this spec's variant requires reporting throughput jointly with achieved
  average chunk size and dedup ratio specifically to make that range's provenance checkable (this
  is exactly the "chunk-size distribution jointly with throughput/dedup-ratio" axis in this
  track's brief).
