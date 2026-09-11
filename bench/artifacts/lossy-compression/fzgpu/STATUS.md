# FZ-GPU — STATUS

**Outcome: BUILT+GATED — gate PASSES on all 3 smoke workloads**

Paper: "FZ-GPU: A Fast and High-Ratio Lossy Compressor for Scientific
Computing Applications on GPUs", HPDC 2023. PAPER_KEY =
`conf/hpdc/ZhangTD0F0TC23` (this is also the lossy-compression spec's own
`lossy-comp-gpu-dual-scope` evidence entry, `benchspecs/lossy-compression/
spec.yaml`: "arXiv 2304.12557 fulltext ... kernel-time throughput reported
separately from PCIe-inclusive 'overall throughput'"). Repo:
`https://github.com/szcompressor/FZ-GPU`
(commit `c7e83f7614c8505e9b539190606b11589273696b`, 2026-06-21;
`git clone --depth 50` into `./source/`). Sourced from this integration's
"further candidates" pool (`output/benchmark_groups.json`'s
`lossy-compression` group, year-desc, nvidia-gpu, non-distributed) after
the newer FFCz/PackKV/lsCOMP candidates were SKIPPED (see their sibling
STATUS.md files) and PFPL/cuSZp were already BUILT+GATED.

## Why a bridge was needed (no kernel code touched)

FZ-GPU ships no library entry point, only an end-to-end CLI binary built
from `source/src/fz.cu`. Its own README says explicitly: "Currently,
FZ-GPU performs compression and decompression together, but we plan to
provide options for performing compression and decompression separately in
the future" — its single driver function, `runFzgpu()`, reads a file, then
always runs compress AND decompress back-to-back with no way to invoke
either half alone.

`bridge_fzgpu.cu` (this directory) splits `runFzgpu()`'s body at its own
internal boundary — the function's own `compressionEnd`/
`decompressionStart` timestamps already mark exactly where compression
ends and decompression begins (`source/src/fz.cu` lines 461-481) — into
`fzgpuc_prepare`/`fzgpuc_run`/`fzgpuc_compressed_bytes`/`fzgpuc_free`
(compress path) and `fzgpud_run_and_copy` (decompress path, gate-only).
Every kernel call is lifted verbatim from `runFzgpu()`:
- compress: `cusz::experimental::launch_construct_LorenzoI_var` (dimension-
  aware Lorenzo predictor/quantizer) + `compressionFusedKernel`
  (bitshuffle + lossless encode, both unmodified `__global__` definitions
  in `fz.cu`);
- decompress: `decompressionFusedKernel` + `launch_reconstruct_LorenzoI_var`.

Same `#define main fzgpu_unused_main` + `#include "source/src/fz.cu"` +
`#undef main` trick as `pfpl`'s bridge (see that adapter's STATUS.md for
the full rationale) — this compiles FZ-GPU's own file verbatim into our
translation unit (no edits to `source/` at all — `git diff` inside
`source/` is empty) while renaming away its `main()` so it can coexist in
a shared library alongside our own `extern "C"` API.

**One deliberate, clearly-flagged deviation, NOT a kernel change**:
`runFzgpu()` computes its absolute error bound as `eb * range`, where
`range = max-min` over ITS OWN file-read, zero-padded host buffer. That
computation is entangled with FZ-GPU's own file-loading/padding path,
which this bridge does not replicate (there is no file — the harness
hands over an in-memory array). Instead, `bridge_fzgpu.cu` passes the
harness's own already-resolved ABSOLUTE bound
(`w.correctness_tolerance`, computed in Python from the REAL,
unpadded data — see `adapter.py::prepare()`) directly into
`launch_construct_LorenzoI_var`/`launch_reconstruct_LorenzoI_var`'s `eb`
parameter, which both expect a plain absolute value (exactly what
`eb * range` produces in the original). This changes HOW the absolute
bound is computed (in Python, from the real data — the same convention
every other implementation in this benchmark uses), not what the kernels
do with it once they have it, and it sidesteps a real correctness risk
(FZ-GPU's own zero-padding would otherwise pollute its internal `range`
computation for buffer lengths that aren't already chunk-aligned, like
this benchmark's 24x24x24 smoke fields).

**One necessary addition**, the same category as this benchmark's other
GPU artifacts: `compressionFusedKernel` uses `atomicAdd` on
`deviceOffsetCounter` and writes `deviceBitFlagArr`/`deviceStartPosition`/
`deviceCompressedSize` per chunk — all require a zeroed starting state.
`runFzgpu()` zeroes them once (before its single compress-then-decompress
pass); `fzgpuc_run()` re-zeroes them before every call instead, since our
harness calls `run()` independently many times (an isolated correctness
check, then warmup, then measured reps).

## Build-system fixes

1. **Arch flag**: FZ-GPU's own `Makefile` has no explicit `-arch` flag
   (relies on nvcc's build-time default); this integration builds directly
   with `nvcc -arch=sm_80` for this machine's A100.
2. `--extended-lambda` carried over from FZ-GPU's own `Makefile` (needed
   by `include/kernel/lorenzo.cuh`'s extended `__device__` lambdas).
3. No `claunch_cuda.cu`/`claunch_cuda.o` link step (present in FZ-GPU's
   own `Makefile` for its `main()`/CLI plumbing) — the two template
   functions this integration calls
   (`launch_construct_LorenzoI_var`/`launch_reconstruct_LorenzoI_var`,
   `include/kernel/lorenzo_var.cuh`) are header-only; the build links
   clean without it.
4. No other patches. Built clean except one pre-existing, harmless
   unused-variable warning inside the artifact's own header
   (`lorenzo_var.cuh:639`, `SEQ_1D` declared but unused) — not touched.

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 --extended-lambda -Xcompiler -fPIC --shared
-I source/include bridge_fzgpu.cu -o bridge.so`. nvcc 12.9
(`/opt/nvidia/hpc_sdk/.../cuda/12.9`). Idempotent.

## Adapter

`IMPL_NAME = "fzgpu-compress"`, `PRECISIONS = ["fp32"]` (FZ-GPU's kernels
are float32-only — `fz.cu` never instantiates an fp64 path).
`direction = "compress"` per this domain's contract.

- `prepare()`: H2D copy of the workload's fp32 data into FZ-GPU's own
  chunk-aligned padded buffer (real data in front, zero-padded tail,
  matching FZ-GPU's own `read_binary_to_new_array`'s value-initialized
  `new T[]()`) + allocation of every device buffer `fzgpuc_prepare` needs
  — timed once as preprocessing. Dimensions are read off the workload's
  own shape (row-major C order, last axis fastest, matching CUDA's `dim3`
  x-fastest convention and FZ-GPU's own `x y z` CLI argument order).
- `run()`: exactly one `fzgpuc_run()` call = one compress pass (the two
  kernels above, plus the required per-call reset).
- `to_host()`: decompress (`fzgpud_run_and_copy`, FZ-GPU's own kernels,
  unmodified) + D2H copy of just the real (unpadded) `dataTypeLen`
  values — **deliberately outside the timed region**: `run()` above is
  the only call the harness's `CudaEventTimer` ever brackets.
- `free()`: records `achieved_compressed_bytes`/`compression_ratio`
  (FZ-GPU's own formula, D2H of the 4-byte atomic offset counter) and, via
  one more untimed decode of whatever the LAST timed `run()` produced,
  `achieved_max_abs_error` and `psnr_db` — same discipline as this
  benchmark's other lossy-compression GPU adapters.

## Gate verification (mandated command, run as specified)

```
$PY -m kernelbench.runner --kernel lossy-compression \
    --variant lossy-comp-kernel-cpu-ebound --impl fzgpu-compress \
    --smoke --warmup 1 --reps 3
```

```
  running fzgpu-compress smoke-smooth-3d      ... 0.116 ms  0.48 GB/s  (err 3.42e-03 <= 3.4191e-03)
  running fzgpu-compress smoke-turbulent-3d   ... 0.102 ms  0.54 GB/s  (err 1.29e-03 <= 1.2903e-03)
  running fzgpu-compress smoke-multiscale-3d  ... 0.098 ms  0.57 GB/s  (err 3.81e-03 <= 3.8086e-03)

3/3 runs valid
conforming: False (smoke: synthetic matrices, reduced protocol, overridden warmup/reps, login-node shared GPU -- expected/flagged, not a defect)
```

`max_abs_err == eb` to displayed precision on all 3 workloads — **PASS**
on all 3. Achieved compression ratios: 6.32 (smooth), 3.78 (turbulent),
6.32 (multiscale) — varied and data-content-dependent, unlike this
benchmark's tiny-smoke-field cuSZp measurement (see cuSZp's STATUS.md),
consistent with FZ-GPU's dimension-aware Lorenzo predictor actually
exploiting local smoothness differently per field kind even at this small
size. PSNR ~64.7-64.9 dB. Full JSON at
`bench/results/lossy-compression_lossy-comp-kernel-cpu-ebound_1786161987.json`.

## Verdict

`fzgpu-compress: BUILT+GATED err<=eb ratio=6.32/3.78/6.32 (smooth/turbulent/multiscale, smoke)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge), torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80 --extended-lambda` unchanged from the recorded build.
- Build: OK. Build-system changes: none (idempotent rebuild via `../../toolchain.sh`, no diff to
  `source/`). Same pre-existing harmless unused-variable warning as recorded (`lorenzo_var.cuh:639`,
  `SEQ_1D`).
- Gate: `lossy-comp-kernel-cpu-ebound` fzgpu-compress: PASS on all 3 smoke workloads
  (smooth/turbulent/multiscale), `max_abs_err <= eb` on all 3 (e.g. 3.42e-03 <= 3.4191e-03), 3/3 runs
  valid.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
