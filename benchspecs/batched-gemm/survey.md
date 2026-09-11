# batched-gemm — survey

Singleton track: 1 paper.

## conf/ics/0001SLFY024 — FASTEN (ICS 2024)

"FASTEN: Fast GPU-accelerated Segmented Matrix Multiplication for Heterogeneous
Graph Neural Networks." Segmented/grouped GEMM: `T` segments (types), each
segment `i` is an `[M_i, K] x [K, N]` GEMM against its own per-type weight
matrix, all `M_i` rows packed into one `[sum(M_i), K]` tensor — the canonical
"variable-batch" / grouped-GEMM shape (as opposed to uniform-shape batched
GEMM where every batch element has the same M).

- **workloads/inputs**: two families in the released `test/test_ops.py`:
  (1) two synthetic fixed-slice sets (`slices0`: 3 segments, sizes
  63/27/38; `slices1`: 4 segments, sizes 127/129/1/255) and (2) 6 real HGNN
  datasets loaded from `datasets_csv/{AIFB,AM,BGS,MUTAG,ACM,DBLP,IMDB,
  Freebase}.csv` (per-relation-type segment-size distributions from
  real heterogeneous graphs — this is FASTEN's actual "batch/segment size
  suite"). Feature width `K` swept in `{16, 32, 64, 80}` (correctness test)
  and `{32, 64, 128}` (perf test, `test_perf`). `T` (segment/type count) is
  also swept synthetically 100→1900 (step 200) at fixed `M=1,000,000` in
  `test_perf_random`, the closest thing to a controlled "batch-count" sweep
  in the repo.
- **timing protocol**: perf test (`test_perf`) uses Triton's `proton`
  profiler: one warmup call to trigger kernel compilation/output
  allocation, a second warmup call ("to trigger backward kernels"), then
  **exactly one** profiled call per `proton.scope(...)` — no explicit
  repetition loop, no statistic (mean/median) computed by the harness
  itself; `proton`'s own instrumentation determines what's recorded per
  call. No warmup/rep count is stated for `test_perf_random` either (same
  single-call-per-scope pattern).
  README's headline numbers (13.65x vs CUTLASS, 4.72x vs cuBLAS) are from
  "operator-wise benchmarks" — the exact aggregation method (mean over
  configs? over runs?) is not stated in the README.
- **timing scope**: kernel only (segment matmul call); `Engine.AUTO`
  picks between a Triton kernel and a `torch`/`cutlass` (via `pyg_lib`)
  path — autotuning/engine-selection cost is NOT surfaced as a separate
  number anywhere in the benchmark script.
  Both **forward** and **backward** (gradient) phases are benchmarked
  separately, a genuinely distinct axis this track's single paper
  introduces.
- **precision & correctness**: `float32` only in both the correctness test
  (`test_segment_matmul`) and the perf test; TF32 matmul is explicitly
  enabled (`torch.backends.cuda.matmul.allow_tf32 = True`) in `test_perf`,
  which silently lowers effective mantissa precision relative to a "fp32"
  label. Correctness tolerance in `test_segment_matmul`:
  `atol=1e-1, rtol=1e-2` for forward/backward when `M/T < 2048`, relaxed
  further to `atol=1.0, rtol=1e-2` for the `other.grad` check when
  `M/T >= 2048` ("gradient accumulation starts to be significantly
  different with large samples") — a notably loose, size-dependent
  tolerance with no principled derivation given.
- **metric**: FLOPs reported via `get_matmul_flops` (backward counted as
  `2x` forward flops) and bytes via `get_matmul_bytes`, logged as `proton`
  metadata alongside the timed scope — not printed as a GFLOP/s number by
  the script itself (that conversion would happen downstream, e.g. in
  `benchmark_results` post-processing not included in the repo excerpt
  read).
- **baselines**: `pyg_lib`'s CUTLASS-backed `segment_matmul`/
  `grouped_matmul` and a plain PyTorch loop-based path (`Engine.TORCH`).

Source: repo `README.md`, `test/test_ops.py` (read in full).

## Divergences / open items

- This is a 1-paper track, so there is no cross-paper divergence to
  resolve; the "divergence" worth flagging is internal: FASTEN's own
  artifact reports a single profiled call per configuration with no
  repetition/statistic, which this spec fixes (see `notes_on_fairness`).
- FASTEN's TF32-enabled "fp32" perf numbers are not apples-to-apples with
  a true fp32 (TF32-disabled) baseline; the spec makes TF32 on/off an
  explicit, reported axis rather than a silent default.
