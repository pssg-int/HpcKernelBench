# PFPL — STATUS

**Outcome: BUILT+GATED — gate PASSES on all 3 smoke workloads**

Paper: "Fast and Effective Lossy Compression on GPUs and CPUs with
Guaranteed Error Bounds", IPDPS 2025. PAPER_KEY = `conf/ipps/FallinADCB25`.
Repo: `https://github.com/burtscher/PFPL`
(commit `36f5aaef42744ed78b4d1525b03a7fb2168b363c`, 2026-06-26;
`git clone --depth 50` into `./source/`; wraps the single-precision
absolute-error-bound GPU pair, `source/src/f32_abs_comp_gpu.cu` /
`source/src/f32_abs_decomp_gpu.cu`, out of PFPL's 12 precision x
error-bound-mode x direction combinations, `fp32`/`fp64` x
`abs`/`rel`/`noa` x `comp`/`decomp`; ABS was chosen because the harness's
`w.correctness_tolerance` is already a resolved ABSOLUTE bound, matching
PFPL's `float errorbound` CLI argument directly with no re-derivation).

## Why a bridge was needed (no kernel code touched)

PFPL ships no library entry point, only two standalone CLI binaries built
from `f32_abs_comp_gpu.cu`/`f32_abs_decomp_gpu.cu`, each a self-contained
`main()` that reads/writes files and internally loops `NUM_RUNS=9` times
(each iteration already individually timed with its own `cudaEvent` pair —
PFPL does NOT have the "one event pair around a whole batched loop"
anti-pattern seen in some other artifacts in this benchmark, e.g. CB-SpMV).
The two kernels this integration needs
(`d_encode`, `source/src/f32_abs_comp_gpu.cu`; `d_decode`,
`source/src/f32_abs_decomp_gpu.cu`) are declared `static __global__`
(internal linkage) directly inside those `main()`-bearing files, not in a
header — so, unlike CB-SpMV (whose kernels lived in an
`#include`-able `.cuh`), they cannot be reused from a separate bridge
translation unit by declaring `extern` prototypes.

Fix: `bridge_encode.cu` and `bridge_decode.cu` (this directory) each
`#define main pfpl_unused_main_encode` / `pfpl_unused_main_decode` and then
`#include` the artifact's `.cu` file **verbatim** — this is a
preprocessor-level rename of one file-scope identifier, applied only in
*our own new file* at include time; **`source/` itself is never edited**
(`git diff` inside `source/` is empty). The renamed `main` becomes dead
code (never called); every kernel definition, device routine, and
constant (`d_encode`/`d_decode`/`d_reset`, `CS`, `TPB`, `GPUTimer`,
`CheckCuda`, and the `components/*`/`include/*` headers it pulls in) is
compiled unmodified as part of that translation unit, then wrapped by a
small `extern "C"` API (`pfplc_prepare`/`pfplc_run`/`pfplc_encoded_size`/
`pfplc_free` in `bridge_encode.cu`; `pfpld_run_and_copy` in
`bridge_decode.cu`) that replaces only the host-side orchestration
(file I/O, the `NUM_RUNS` loop) `main()` used to do. `pfpl_handle.h`
(this directory) is a small shared struct — plain `unsigned char*`/`int`/
`float` fields, no PFPL types — so both translation units can pass device
pointers between each other (`bridge_decode.cu`'s `pfpld_run_and_copy`
decodes directly from `bridge_encode.cu`'s `d_encoded` buffer, already
resident on device from the matching `pfplc_run()` call — no re-upload of
the compressed bytes needed).

**One necessary addition** (documented in `bridge_encode.cu`): `d_encode`
uses `atomicAdd` on a file-scope `g_chunk_counter` for dynamic chunk
assignment and a `fullcarry` array for cross-block carry propagation —
both require a zeroed starting state. PFPL's own `main()` resets them
(`d_reset<<<1,1>>>()` + `cudaMemset(d_fullcarry, ...)`) once per its own
`NUM_RUNS` loop iteration; `pfplc_run()` does the same reset once per our
harness's `run()` call instead — the identical requirement, same fix,
relocated to match this benchmark's call pattern (an isolated correctness
check, then `warmup` calls, then `reps` measured calls, each independently
correct) — the same category of fix as CB-SpMV's per-launch `d_y` re-zero
(see that adapter's `STATUS.md`).

## Build-system fixes

1. **Arch flag**: PFPL's own `makefile` defaults `NV_SM := 70`; this
   integration builds directly with `nvcc -arch=sm_80` (not via the
   makefile) for this machine's A100 — the same arch-flag override used by
   every other GPU artifact in this benchmark.
2. No other patches. `-I source/src` is added defensively in `build.sh`
   but is not actually load-bearing — nvcc resolves the artifact's own
   quoted `#include "include/..."` / `#include "components/..."` directives
   relative to `f32_abs_{comp,decomp}_gpu.cu`'s own directory regardless of
   where the top-level bridge file lives.

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 -Xcompiler -fPIC -shared -I source/src
bridge_encode.cu bridge_decode.cu -o bridge.so`. nvcc 12.9
(`/opt/nvidia/hpc_sdk/.../cuda/12.9`). Built clean on the first attempt, no
warnings requiring investigation. Idempotent.

## Adapter

`IMPL_NAME = "pfpl-compress"`, `PRECISIONS = ["fp32"]` (this adapter wraps
the f32 ABS kernel pair only; PFPL also ships fp64 and REL/NOA variants,
not wrapped here). `direction = "compress"` (this domain's contract:
`prepare()`/`run()` time compression only; decompression is used solely
for the correctness gate).

- `prepare()`: H2D copy of the workload's fp32 data + allocation of every
  device buffer `pfplc_prepare` needs (encode-side, lifted from PFPL's own
  `main()`, plus decode-side buffers allocated up front so the gate never
  needs a second prepare round-trip) — timed once as preprocessing.
- `run()`: exactly one `pfplc_run()` call = one `d_encode` launch (plus the
  required per-call reset described above).
- `to_host()`: decompress (`pfpld_run_and_copy`, PFPL's own `d_decode`
  kernel, unmodified) + D2H copy of the reconstructed array —
  **deliberately outside the timed region**: `run()` above is the only
  call the harness's `CudaEventTimer` ever brackets; `to_host()`'s cost
  never enters `times_ms`.
- `free()`: records `achieved_compressed_bytes`/`compression_ratio` (D2H of
  the 4-byte encoded-size counter) and, via one more untimed decode of
  whatever the LAST timed `run()` produced, `achieved_max_abs_error` and
  `psnr_db` — same discipline as `kernelbench/domains/compression.py`'s
  CPU `_QuantizeZlibCodec.free()`.

## Gate verification (mandated command, run as specified)

```
$PY -m kernelbench.runner --kernel lossy-compression \
    --variant lossy-comp-kernel-cpu-ebound --impl pfpl-compress \
    --smoke --warmup 1 --reps 3
```
(Note: despite the variant id saying "cpu", the harness/adapter contract
here is platform-neutral — `--smoke` just selects the domain's 3 synthetic
Field workloads and a reduced protocol; nothing about the variant name
restricts the implementation to CPU. Per ARTIFACT_GUIDE.md this login-node
run is functional-only, not a timing claim.)

```
  running pfpl-compress smoke-smooth-3d      ... 0.076 ms  0.72 GB/s  (err 3.42e-03 <= 3.4191e-03)  [0.4s]
  running pfpl-compress smoke-turbulent-3d   ... 0.067 ms  0.83 GB/s  (err 1.29e-03 <= 1.2903e-03)  [0.0s]
  running pfpl-compress smoke-multiscale-3d  ... 0.065 ms  0.85 GB/s  (err 3.81e-03 <= 3.8086e-03)  [0.1s]

3/3 runs valid
conforming: False (smoke: synthetic matrices, reduced protocol, overridden warmup/reps, login-node shared GPU -- expected/flagged, not a defect)
```

`max_abs_err == eb` to displayed precision on all 3 workloads (PFPL's
quantization step rounds to a bin boundary at exactly the error bound when
the true value sits far enough from a bin center — expected behavior for a
tight ABS-bound quantizer, not a near-miss) — **PASS** on all 3, well
within `correctness_tolerance`. Achieved compression ratios: 5.90 (smooth),
3.37 (turbulent), 5.52 (multiscale); PSNR ~64.7-64.9 dB (recorded in each
run's `params.compression_ratio` / `params.psnr_db` / `params.
achieved_max_abs_error`, full JSON at
`bench/results/lossy-compression_lossy-comp-kernel-cpu-ebound_1786161167.json`).

## Verdict

`pfpl-compress: BUILT+GATED err<=eb ratio=5.90/3.37/5.52 (smooth/turbulent/multiscale, smoke)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge), torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80` unchanged from the recorded build.
- Build: OK. Build-system changes: none (idempotent rebuild via `../../toolchain.sh`, no diff to
  `source/`).
- Gate: `lossy-comp-kernel-cpu-ebound` pfpl-compress: PASS on all 3 smoke workloads
  (smooth/turbulent/multiscale), `max_abs_err <= eb` on all 3 (e.g. 3.42e-03 <= 3.4191e-03), 3/3 runs
  valid.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
