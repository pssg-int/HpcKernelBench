# adaptive-spmv (HolaSpmv dense-vector path) — STATUS

**Outcome: BUILT+GATED — clean pass, fp64.**

Paper: "Adaptive SpMV/SpMSpV on GPUs for Input Vectors of Varied
Sparsity", TPDS 2021. PAPER_KEY = `journals/tpds/LiAY21`. Selected as a
**core baseline (partial regime match) under the revised kernel-centrality
rule** (`output/kernel_centrality.json` key `spmv|journals/tpds/LiAY21`:
centrality `core`, regime `partial`, `gpu_single_card: true` — "adaptive
ML-based kernel-selection framework specifically for SpMV/SpMSpV evaluated
at kernel level on GPUs; emphasis on varied input-vector sparsity is
narrower than the spec's plain dense-vector SpMV variant").
Repo: `https://github.com/hpc-research-2020/spmspv-adaptive`
(commit `b71de128ef68b56353e19a87104827e13c7ab895`, 2020-06-29;
`git clone --depth 50` into `./source/`).

## Scope decision (per the integration brief)

The paper's actual headline is an ML-based selector (`hice-spmspv/hice/ml/`
— decision-tree/SVM/GBDT/random-forest training pipeline in the repo's own
README) that picks among **8** candidate GPU SpMV/SpMSpV kernels at
runtime based on matrix/vector sparsity features; training the selector
itself is out of scope (needs a labeled training-matrix corpus and offline
model training the repo's own scripts assume, not a kernel to benchmark).
Per the brief, this adapter wraps the **dense-vector SpMV path only**:
`hola_spmv<T>` / `hola_pre<T>`
(`source/hice-spmspv/hice/la/include/spmspv/csc-spmspv/detail/device/
holaspmv.h`) — HolaSpmv, from the same research group's own prior GPU-SpMV
work, the dense-input-vector member of the 8-kernel set. `hola_spmspv`
(same header, sparse x) is a different track (SpMSpV) and is untouched.

## What was wrapped

`hola_spmv`/`hola_pre` are clean, already-templated, standalone functions
(explicitly instantiated at both `float` and `double` at the bottom of
`holaspmv.h` by the artifact itself — this is the artifact's own public
surface, not an instantiation this integration invented) with NO
extern "C" boundary and no standalone driver in this repo (the repo's own
`hice/la/script/` only exercises the full adaptive framework through its
CMake/CLI harness). `bridge.cu` (this directory) exposes them:

- **`holaspmv_prepare()`** (timed once, as preprocessing): H2D-copies the
  CSR matrix and x, then calls the artifact's own `hola_pre<double>()` to
  size HolaSpmv's own scratch buffer (`(nnz/blockSize + 2) * 4` bytes —
  O(number of blocks), NOT O(nnz): a small per-block row-start bookkeeping
  array, genuinely structural) and allocates it.
- **`holaspmv_run()`** (timed per iteration): ONE call to
  `hola_spmv<double>()`. Its first internal action, the artifact's own
  `DetermineBlockStarts` kernel, both computes this call's block-start
  bookkeeping into the scratch buffer AND zeroes the output vector
  (`holaspmv.h`: `outvec[id] = 0;`) — i.e. the artifact's own design
  recomputes this bookkeeping and zeroes y on every call rather than
  hoisting it into a one-time setup step, so no separate `cudaMemset` of y
  is added here (that would only duplicate work the kernel already does
  as part of what gets timed, and change what "the kernel" means for this
  artifact).

No kernel code is modified: `hola_pre`/`hola_spmv` and everything they call
are included verbatim.

## Build-system fix (no tracked file touched)

`holaspmv.h` is written to be included transitively, after the artifact's
own `spmspv/config.h` and `spmspv/csc-spmspv/detail/util.h` have already
run (normally via `spmspv/csc-spmspv/spmspv.h`, the SpMSpV entry point this
integration deliberately does not pull in, per the "dense-vector path
only" scope above). Included standalone, two names that chain would have
supplied are genuinely undefined at compile time: `LM_WARP_SIZE` (a plain
warp-size constant, `#define LM_WARP_SIZE 32` in `spmspv/config.h`) and
`divup<T>` (a ceiling-division template — the artifact's own
`hola_common.h` ships an identical one under the name `hola_divup`, so
this is not a new algorithm, just the same one-line arithmetic under the
name `holaspmv.h` itself calls). `bridge.cu` supplies both directly
(`#define LM_WARP_SIZE 32`; a 3-line `divup` template) immediately before
`#include "holaspmv.h"`, with the reasoning documented in `bridge.cu`'s own
comment. No file under `source/` is modified — `source.provenance`'s
`patch=none` is accurate.

CUB (`<cub/cub.cuh>`, used by `hola_common.cuh`) is bundled with the CUDA
12.9 toolkit already on `CPATH` via `bench/artifacts/toolchain.sh` — the
artifact's own `cmake/external/ExternalCub.cmake` (which fetches a
standalone CUB checkout for older toolkits that don't bundle it) was not
needed. ModernGPU (the repo's other listed dependency) is used only by the
SpMSpV-specific files (`csc_sort_*.cuh`, `csc_bucket_*.cuh`) that this
integration never includes — confirmed by grep before starting the bridge,
so it was never fetched.

## Build

```
./build.sh
```
`nvcc -O3 -w -arch=sm_80 -Xcompiler -fPIC -shared bridge.cu -o bridge.so`.
nvcc 12.9. Idempotent (single invocation).

## Adapter

`IMPL_NAME = "adaptive-spmv-hola"`, `PRECISIONS = ["fp64"]` (this bridge
instantiates `hola_pre`/`hola_spmv` at `<double>`; the artifact's header
also explicitly instantiates `<float>`, not wired up here for time-budget
reasons — a straightforward follow-on if fp32 is wanted later). `prepare()`
sorts the harness's CSR and generates `x` via
`np.random.default_rng(seed).uniform(-1,1,size=cols)`, exactly matching
`cpu_ref.reference_spmv`'s RNG (the documented RNG-mismatch trap). `run()`
is one `holaspmv_run()` call (default/legacy stream throughout — the
artifact's kernels never take a stream argument). `to_host()` D2H-copies
`y`.

## Gate verification

### Harness smoke (`--smoke`, synthetic 4000x4000, reduced protocol)
```
$PY -m kernelbench.runner --kernel spmv --variant spmv-csr-kernel --impl adaptive-spmv-hola --smoke
```
```
  running adaptive-spmv-hola smoke-uniform   ... 0.013 ms  15.27 GFLOP/s  (err 3.27e-16 <= 1e-09)
  running adaptive-spmv-hola smoke-banded    ... 0.011 ms  17.86 GFLOP/s  (err 2.70e-16 <= 1e-09)
  running adaptive-spmv-hola smoke-powerlaw  ... 0.014 ms  18.39 GFLOP/s  (err 3.38e-16 <= 1e-09)
3/3 runs valid
```

### Real SuiteSparse matrices (mandated command)
```
$PY -m kernelbench.runner --kernel spmv --variant spmv-csr-kernel --impl adaptive-spmv-hola --matrices webbase-1M --warmup 1 --reps 3
$PY -m kernelbench.runner --kernel spmv --variant spmv-csr-kernel --impl adaptive-spmv-hola --matrices cant --warmup 1 --reps 3
```
```
  webbase-1M (1,000,005 x 1,000,005, nnz 3,105,536): 0.065 ms  95.71 GFLOP/s  (err 7.71e-16 <= 1e-09)
  cant       (62,451 x 62,451, nnz ~4.0M):           0.053 ms  150.16 GFLOP/s (err 5.50e-16 <= 1e-09)
```
Both **PASS** at fp64 unit-roundoff on real, structurally distinct
matrices (webbase-1M: sparse directed web graph, avg ~3 nnz/row; cant:
dense structural-mechanics matrix, avg ~64 nnz/row) — no correctness
findings for this artifact, unlike the other two integrated in this
session.

## Verdict

`adaptive-spmv-hola: BUILT+GATED err=5.5e-16-7.7e-16 (tol 1e-9), fp64,
clean pass on smoke + 2 real SuiteSparse matrices of very different
density.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80` (as shipped by build.sh, unchanged).
- Build: OK. Build-system changes: none.
- Gate: spmv-csr-kernel smoke: 3/3 valid, PASS all three (err 2.70e-16 -
  3.38e-16 <= 1e-9). spmv-csr-kernel --matrices webbase-1M --warmup 1 --reps 3:
  PASS, err 7.71e-16 <= 1e-9. spmv-csr-kernel --matrices cant --warmup 1
  --reps 3: PASS, err 5.54e-16 <= 1e-9. (webbase-1M.mtx was not yet cached;
  fetched once on the login node via
  `python -c "from kernelbench import matrices; matrices.fetch_suitesparse('webbase-1M')"`
  since the GPU compute node has no outbound internet — no adapter/kernel
  change involved.)
- Deviation from the recorded ruling: none — same clean pass, same order of
  magnitude of error on both real matrices (5.50e-16 -> 5.54e-16 on cant,
  7.71e-16 -> 7.71e-16 on webbase-1M; trivial float summation-order noise).
- Verdict here: BUILT+GATED — same as the recorded ruling.
