# GPULZ — STATUS

**Outcome: BUILT+GATED — bit-exact round-trip PASSES on all 3 smoke workloads**

Paper: "GPULZ: Optimizing LZSS Lossless Compression for Multi-byte Data on
Modern GPUs", ICS 2023. PAPER_KEY = `conf/ics/ZhangTDYSTC23` (matches
`output/included.json`'s entry for `https://github.com/hpdps-group/ICS23-GPULZ`
and `benchspecs/lossless-compression/spec.yaml`'s own evidence key for this
artifact). Repo: `https://github.com/hpdps-group/ICS23-GPULZ`
(commit `314d6cfcfc6c8dbd5ee173859b672528f29d31ab`, 2025-04-18;
`git clone --depth 1` into `./source/`).

## Why a bridge was needed (no kernel code touched)

GPULZ ships no library entry point, only an end-to-end CLI binary built from
`source/gpulz.cu`. Its own README says explicitly: "At present, GPULZ
performs compression and decompression simultaneously; however, we plan to
offer options for performing these tasks separately in future iterations" —
its single driver, `main()`, reads a file, then always runs compress AND
decompress back-to-back with no way to invoke either half alone.

`bridge_gpulz.cu` (this directory) splits `main()`'s body at its own
internal boundary (its own `compStart`/`compStop`/`decompStart`/`decompStop`
`cudaEvent` markers already mark exactly where compression ends and
decompression begins, `source/gpulz.cu` lines 450-485) into
`gpulzc_prepare`/`gpulzc_run`/`gpulzc_compressed_bytes`/`gpulzc_free`
(compress path) and `gpulzd_run_and_copy` (decompress path, gate-only).
Every kernel call is lifted verbatim from `main()`:
- compress: `compressKernelI` + `cub::DeviceScan::ExclusiveSum` (x2, flag
  array + compressed-data array offsets) + `compressKernelIII`, all
  unmodified `__global__`/library calls in `gpulz.cu`;
- decompress: `decompressKernel`.

Same `#define main gpulz_unused_main` + `#include "source/gpulz.cu"` +
`#undef main` trick as this benchmark's `fzgpu-compress`/`pfpl-compress`
bridges (lossy-compression track) — compiles GPULZ's own file verbatim into
this translation unit (no edits to `source/` at all — `git diff` inside
`source/` is empty) while renaming away its `main()` so it can coexist in a
shared library alongside this integration's own `extern "C"` API.

**One necessary addition**, same category as this benchmark's other GPU
compression artifacts: the cub exclusive-sum scratch-buffer sizing+alloc is
hoisted from `run()` into `prepare()` (main()'s own two-call
determine-then-allocate pattern is otherwise re-run every single compress
call) — the required scratch size depends only on `numOfBlocks+1` (a
data-independent shape constant fixed once per workload), so allocating it
once in `prepare()` and reusing it across every `run()` call is a pure
preprocessing hoist, not a behavior change; `gpulzc_run()` still repeats the
two `cub::DeviceScan::ExclusiveSum` calls themselves every time (matching
`main()`'s own per-invocation redo of the scan), only the scratch
allocation moved.

No re-zeroing of any output buffer is needed between `run()` calls (unlike
`fzgpu-compress`, which resets `atomicAdd`-accumulated counters): every
GPULZ compress-kernel output here is written via a direct per-block/
per-thread INDEX assignment (`compressedDataSizeGlobal[blockIdx.x] = ...`,
never `+=`), so repeated calls against the same (unchanged) input
deterministically overwrite identical values — confirmed by the exact
round-trip PASS below, which used no per-call reset.

## Configuration used (GPULZ's own compile-time defaults, unmodified)

`BLOCK_SIZE=2048` bytes, `THREAD_SIZE=128`, `WINDOW_SIZE=32`,
`INPUT_TYPE=uint32_t` (`source/gpulz.cu` lines 12-15, GPULZ's own shipped
defaults — the README says these are edited directly in the source for
different configurations; not changed here). The spec's block/window/
symbol-length sweep (`benchspecs/lossless-compression/spec.yaml`'s
`lossless-comp-gpu-multibyte-dual-scope` variant, grid
`{2048,4096,8192,16384}` x `{32,64,128,255}` x `{1,2,4}`) is **NOT**
implemented by this first integration pass — GPULZ only exposes these as
compile-time `#define`s, so a full sweep would need one rebuilt `bridge.so`
per grid point; deferred as follow-up work, flagged here rather than
silently fixed to one point and left unmentioned.

## Byte semantics

GPULZ's kernels operate on a flat, dtype-agnostic multi-byte symbol stream
— they only ever compare/copy raw `INPUT_TYPE` bit patterns
(`buffer[bufferPointer] == buffer[windowPointer]`), never interpret them as
IEEE floats. Re-purposing the compression domain's fp32 `Field` workload's
raw in-memory bytes as the input stream is therefore exactly the same
"multi-byte scientific/sensor integer data" input class GPULZ's own paper
targets (spec variant 2), not a semantic mismatch — no NaN/Inf special
casing needed since no arithmetic is performed on the symbol values.

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 -Xcompiler -fPIC --shared bridge_gpulz.cu -o bridge.so`.
nvcc 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`). `cub/cub.cuh` resolved from
nvcc's own bundled Thrust/CUB include path — no vendoring needed. Built
clean except one pre-existing, harmless unused-variable warning inside
GPULZ's own `main()` (`gpulz.cu:335`, `decomp` declared but never used, a
CLI flag GPULZ's own code never reads) — not touched. Idempotent.

## Adapter

`IMPL_NAME = "gpulz-compress"`, `PRECISIONS = ["fp32"]` (matches the
compression domain's `DEFAULT_PRECISION`; GPULZ itself is byte/symbol-
agnostic). `direction = "compress"` per this domain's contract — mirrors
the `fzgpu-compress`/`cuszp-compress` adapters already integrated for the
lossy-compression track (same domain module, same split-timing discipline).

- `prepare()`: H2D copy of the workload's raw bytes into GPULZ's own
  `BLOCK_SIZE`-aligned zero-padded buffer (matching `main()`'s own
  `cudaMemset`-then-partial-`cudaMemcpy` pattern) + allocation of every
  device buffer `gpulzc_prepare` needs, incl. the cub scratch (see above) —
  timed once as preprocessing.
- `run()`: exactly one `gpulzc_run()` call = one compress pass (4 steps
  above).
- `to_host()`: decompress (`gpulzd_run_and_copy`, GPULZ's own kernel,
  unmodified) + D2H copy of just the real (unpadded) `fileSize` bytes —
  **deliberately outside the timed region**: `run()` above is the only call
  the harness's `CudaEventTimer` ever brackets.
- `free()`: records `achieved_compressed_bytes`/`compression_ratio` (GPULZ's
  own formula from `main()`, D2H of 2 cumulative-offset scalars).

## Gate verification (mandated command, run as specified)

```
$PY -m kernelbench.runner --kernel lossless-compression \
    --variant lossless-comp-gpu-multibyte-dual-scope --impl gpulz-compress \
    --smoke --warmup 1 --reps 3
```

```
  running gpulz-compress smoke-smooth-3d      ... 0.124 ms  0.45 GB/s  (err 0.00e+00 <= None)
  running gpulz-compress smoke-turbulent-3d   ... 0.121 ms  0.46 GB/s  (err 0.00e+00 <= None)
  running gpulz-compress smoke-multiscale-3d  ... 0.113 ms  0.49 GB/s  (err 0.00e+00 <= None)

3/3 runs valid
conforming: False (smoke: synthetic matrices, reduced protocol, overridden warmup/reps, login-node shared GPU -- expected/flagged, not a defect)
```

`correctness.metric == "exact"`, `value == 0.0`, `passed == True` on all 3
workloads — **bit-exact round-trip PASSES**. Achieved compression ratios:
0.966 (smooth), 0.966 (turbulent), 0.966 (multiscale) — **all BELOW 1.0**,
i.e. GPULZ's LZSS match-finding EXPANDS this smoke data rather than
shrinking it. This is an expected characteristic, not a defect: the smoke
fields are synthetic float32 data with high-entropy mantissa bits (Gaussian-
bump/noise-derived, `kernelbench/domains/compression.py`), so
`WINDOW_SIZE=32`-deep dictionary matching rarely finds a repeated 4-byte
`uint32_t` symbol run long enough to beat the 1-flag-bit + 4-raw-bytes cost
of an unmatched literal — LZSS-family compressors are fundamentally
data-content-dependent and this track's own literature (GPULZ's own paper)
reports large ratio swings (1.9x-8.7x) only on real SDRBench-scale
quantization-code streams, never on tiny synthetic random fields like this
integration's login-node smoke gate. Full JSON at
`bench/results/lossless-compression_lossless-comp-gpu-multibyte-dual-scope_1786202475.json`.

## Verdict

`gpulz-compress: BUILT+GATED bit-exact roundtrip PASS, ratio=0.966 (smoke, expansion on high-entropy synthetic data -- expected LZSS behavior, not a bug)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge), torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80` unchanged from the recorded build.
- Build: OK. Build-system changes: none (build.sh already machine-neutral; idempotent rebuild, no
  diff to `source/`).
- Gate: `lossless-comp-gpu-multibyte-dual-scope` gpulz-compress: PASS on all 3 smoke workloads
  (smooth/turbulent/multiscale), `err 0.00e+00 <= None` (exact metric), 3/3 runs valid. Ratios
  0.60/0.65/0.64 GB/s throughput; compression ratio still < 1.0 (expansion on high-entropy synthetic
  data, same as recorded).
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
