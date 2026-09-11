# Survey — erasure-coding (singleton track: 1 paper)

## conf/sc/Uezato21 — Accelerating XOR-based erasure coding using program optimization techniques (SC 2021)

- **What is measured**: XOR-based Reed-Solomon-style erasure coding (encode: compute parity blocks
  from data blocks via XOR-only operations; decode: reconstruct missing blocks from surviving
  data+parity), accelerated by treating the XOR-schedule as a straight-line program (SLP) and
  applying compiler-construction optimizations: (1) grammar-compression to reduce XOR-instruction
  count, (2) deforestation (functional-program optimization) to reduce memory accesses, (3) the
  (red-blue) pebble game to reduce cache misses via better instruction scheduling.
- **workloads/inputs**: configurable Reed-Solomon (k data blocks, m parity blocks) via CLI
  `--data-block <k>` / `--parity-block <m>`, default RS(10,4); the repo's own README examples cover
  RS(10,4), RS(10,3), RS(9,3), RS(8,2). Decode is tested via `--enc-dec [erased_indices...]`, e.g.
  `--enc-dec` with default erasure pattern `[2,4,5,6]` (4 blocks erased, matching m=4's max
  recoverable loss for RS(10,4)) — i.e. **degraded-read/decode is exercised at maximum tolerable
  erasure count** by default, not partial/single-block loss. `--stat-dec 2 4 5 6` computes decode
  SLP statistics (XOR count, memory-access count, transfer count) for a specific named erasure
  pattern without running the timed benchmark. Data size in the shown examples is ~10MB
  (`data size = 10158080` bytes) — block size is derived from data_block count and total size
  (`BLOCK_SIZE_PER_ITER` constant governs the SIMD-lane granularity per XOR operation).
- **timing protocol**: `--loop-iter <N>` (default **1000**, from `main.rs:131`,
  `opt.loop_iter.unwrap_or(1000)`); the loop wraps `Instant::now()` immediately before each
  `run::run_program` call and (implicitly, from surrounding code) takes the elapsed time after —
  i.e. **1000 repetitions, per-iteration wall-clock timing via Rust's `std::time::Instant`**, with
  the reported result being **avg + standard deviation** (`Encode: avg = 7188.13 MB/s, sd =
  405.67`) — NOT median/min/max, and no separate warmup iterations are visible before the 1000-loop
  starts (the first of the 1000 timed iterations is NOT discarded).
- **timing scope**: kernel-only (the XOR-SLP execution itself, `run::run_program`); SLP
  compilation/optimization (grammar compression, deforestation, pebble-game scheduling) happens
  ONCE before the loop starts (visible from the code structure: `estimate_compile` calls appear
  outside the `for _ in 0..loop_iter` loop) — i.e. the paper's own harness already separates
  one-shot program-optimization cost from the per-run XOR-execution cost, matching this project's
  general preprocessing-separation principle without needing a spec-level correction.
- **precision & correctness**: XOR-based erasure coding over GF(2) is **exact/lossless by
  construction** — no floating-point tolerance applies; correct decode means bit-exact
  reconstruction of the erased blocks. No automated correctness check is visible in the fetched
  README's benchmark-invocation examples (only `--stat-enc`/`--stat-dec`/`--enc-dec`/`--all-stat`
  flags shown, none of which is described as a pass/fail correctness gate).
- **metric**: throughput, MB/s, reported separately for Encode and Decode, with standard deviation.
  The `--optimize-level` flag exposes THREE optimization tiers (`Nooptim`, `Fusion`,
  `FusionSchedule` — the paper's own three-stage ablation: no optimization / grammar-compression-
  and-deforestation-only / plus pebble-game scheduling), letting each optimization's individual
  contribution be isolated rather than only the fully-optimized number reported.
- **baselines**: Intel ISA-L (Intel's high-performance erasure-coding library, ~6.7 GB/s per the
  abstract, benchmarked via a SEPARATE documented procedure, `HOWTO_BENCHMARK_ISAL.md`, not fetched
  in this survey); the paper's own prior/naive XOR-based approach (~4.9 GB/s, presumably
  `--optimize-level Nooptim` in this same tool). The paper's claimed result: ~8.92 GB/s, beating
  both.
- **source**: abstract (`output/included.json`, arXiv id `2108.02692` present in
  `data/artifacts_found.json`); repo README (`gh api repos/yuezato/xorslp_ec/contents/README.md`)
  and `src/main.rs` (`gh api .../contents/src/main.rs`).

## Divergences

- Single-paper track: no cross-paper divergence. The main internal design choice this spec must
  resolve is the paper's own OPTIMIZATION-LEVEL ablation (Nooptim/Fusion/FusionSchedule) — this is
  a legitimate, paper-native axis (not a fairness gap to fix) and is kept as an explicit part of the
  variant rather than only reporting the fully-optimized headline number, so the individual
  contribution of each program-optimization technique remains visible.
- The paper's own timing statistic is avg+sd over 1000 undiscarded iterations (no warmup); this
  spec adopts median+min/max instead per this project's general fairness principle (a raw,
  un-warmed-up mean is sensitive to the first iteration's cold-cache/branch-predictor state), a
  DEVIATION from the artifact's own literal convention, noted explicitly below.
