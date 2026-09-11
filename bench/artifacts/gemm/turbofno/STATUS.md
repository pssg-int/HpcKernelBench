# turbofno (TurboFNO) — gemm

**Status: BUILT+GATED err=1.86e-07..2.37e-07 (all <= tol 1e-6)**

- Paper: "TurboFNO: High-Performance Fourier Neural Operator with Fused
  FFT-GEMM-iFFT on GPU", SC'25. `PAPER_KEY = conf/sc/WuZDZHC25`.
- Artifact: https://github.com/shixun404/TurboFNO
- Commit cloned: `215d916419b9a7745b803e20d20a45afd335ce14`, `git clone --depth 50`.
  (`git log -1` reports the commit date as `2026-07-01`, ahead of today —
  this is the upstream repo's own commit timestamp as returned by GitHub,
  not a clock issue on this machine; recorded as observed.)
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`), host compiler
  `g++-12` (SUSE 12.3.0), `-gencode arch=compute_80,code=sm_80`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.
  No cuFFT/cuBLAS link needed (see below) — no TurboFFT submodule init was
  required.

## What the artifact actually is

TurboFNO's README states its own contribution includes "Custom
high-performance FFT and GEMM kernels, matching or exceeding cuFFT and
cuBLAS performance" — a genuine paper-authored kernel, not a cuBLAS
passthrough. The standalone complex GEMM is `cgemm`
(`source/fusion_variants/1D_E_baseline/cgemm.cuh`): a hand-written,
shared-memory-tiled `__global__ void cgemm(int M, int N, int K, float2 *A,
float2 *B, float2 *C, float2 alpha, float2 beta)`. It has no host-side
launcher of its own — every caller in the repo (`fused.cu`'s `main()`, the
other 1D/2D fusion variants) computes grid/block dims and calls it inline —
so wrapping it means reproducing that same launch geometry, not
reimplementing anything inside the kernel.

Kernel entry point wrapped (rule 1: finest available boundary, not the
fused FFT+GEMM+iFFT benchmark unit in the same directory's `fused.cu`):
`cgemm` itself, called once per (batch element).

## Macro/tile-config discovery (rule 3: build-system glue only, kernel untouched)

`cgemm.cuh` uses macros (`THREADBLOCK_M/N/K`, `WARP_M/N`, `THREAD_M/N`,
`WARP_NUM_ROW`, `THREAD_NUM_ROW`, `LOAD_PER_THREAD_A/B`, `TID`, `WID`,
`BID_X`, `BID_Y`) that are **not** defined in the file itself. Traced them
to `source/utils/TurboFNO.h`, the artifact's own header:

```
THREADBLOCK_M=64  THREADBLOCK_N=64  THREADBLOCK_K=8
WARP_M=32  WARP_N=16  THREAD_M=4  THREAD_N=4
WARP_NUM_ROW = THREADBLOCK_M/WARP_M     THREAD_NUM_ROW = WARP_M/THREAD_M
LOAD_PER_THREAD_A = THREADBLOCK_M*THREADBLOCK_K/THREAD_NUM
LOAD_PER_THREAD_B = THREADBLOCK_N*THREADBLOCK_K/THREAD_NUM
TID = threadIdx.x   WID = threadIdx.x/32
BID_X = blockIdx.x  BID_Y = blockIdx.y
```

This is the artifact's own single canonical tile config — `cmake/
turbofno_targets.cmake` + every `fusion_variants/*/CMakeLists.txt` feed
every 1D_* variant `include_directories(${PROJECT_ROOT}/utils)`, so
`TurboFNO.h` is the one header the repo's own build already uses for this
kernel. **Used verbatim, not invented for this integration.**

## Build (rule 3/6: minimal patches, all recorded; kernel code untouched)

1. **`wrapper.cu`** (new file, `bench/artifacts/gemm/turbofno/wrapper.cu`,
   outside `source/`): `#include`s `source/utils/TurboFNO.h` and
   `source/fusion_variants/1D_E_baseline/cgemm.cuh` **unmodified**, and adds
   one `extern "C" cudaError_t turbofno_cgemm_launch(...)` that computes
   grid/block dims and dynamic shared-memory size exactly as `fused.cu`'s
   own `main()` does for the same kernel, then calls
   `cgemm<<<gridDim,blockDim,shmem_size>>>(...)`. No prebuilt TurboFNO
   library exists for `cgemm` alone to link against (it's a header-only
   kernel definition, not a compiled `.so` with a C entry point) — this
   TU compiles the kernel directly, which is why there is no
   `source/build/*.so` step here (unlike inferfast's two-stage build).
2. **No submodule / no cuFFT / no cuBLAS**: confirmed `cgemm.cuh` needs only
   `<mma.h>`/`<stdio.h>` from the CUDA toolkit (`mma.h` is `#include`d but
   unused — 1D_E_baseline's `cgemm` is a plain register/shared-memory-tiled
   kernel, no `wmma::` calls) and the macros above. `git submodule update
   --init` for `TurboFFT` was never run — genuinely unnecessary for this
   target, confirming the task brief's guess.
3. **Host compiler**: same `g++-12` fix as `inferfast/build.sh` (system
   default `g++-14` fails nvcc 12.9's `<bits/alloc_traits.h>` with
   `__has_construct is undefined`) — passed via `-ccbin`, no source edits.

`build.sh` is idempotent (single `nvcc -shared` invocation, safe to re-run).

## adapter.py

- `KERNEL = "gemm"`, `IMPL_NAME = "turbofno-cgemm"`, `PRECISIONS = ["fp32"]`
  (`float2`'s components are fp32; there is no other precision path).
- **Real-vs-complex decision (task's option (a), since (b) does not apply):**
  checked `TurboFFT/` (FFT-only, no GEMM) and `utils/` (macros/config only,
  `TurboFNO.h`, `utils.cu(h)`) — no real-valued GEMM microkernel exists
  anywhere in this repo. `cgemm` is inherently `float2`-typed throughout, so
  option (a) is the only route: `prepare()` draws the real part from the
  *same* seeded-RNG recipe `dense.py`'s `_rng_operand`/`reference_gemm` use
  (`np.random.default_rng(seed).uniform(-1,1,size=...).astype(fp32)`,
  including the same `seed + 1_000_003 (+ b)` offset for B) and sets the
  imaginary part to exactly `0.0`; `alpha={1,0}`, `beta={0,0}`. Complex
  multiply-add then reduces algebraically to the real product: every
  partial product's imaginary contribution is `a*0` or `0*c`, so the
  accumulated imaginary part of `C` stays exactly `0` in IEEE arithmetic,
  and the real part is exactly the real GEMM result. `to_host()` returns
  only the `.real` component (`.imag` is not inspected — verified `0`, but
  discarding it rather than asserting on it, since asserting inside
  `to_host()` would risk turning a benign fp rounding artifact into a false
  BUILD-FAILED). This is a single, unmodified invocation of the artifact's
  own compiled kernel (rule 1), not a kernel-code patch.
- **Structural caveat, recorded honestly, not fixed (out of scope per the
  task)**: `cgemm` performs full complex arithmetic (4 real multiplies + 2
  adds per MAC — see `cgemm.cuh`'s `c[i][j].x += a.x*b.x - a.y*b.y;
  c[i][j].y += a.x*b.y + a.y*b.x`) versus 1 multiply + 1 add for a real
  GEMM, i.e. it does strictly more raw arithmetic per element than a native
  real-GEMM kernel would. `dense.py`'s `_cost_gemm` counts `2*M*N*K*batch`
  — the REAL-GEMM flop count — so **`turbofno-cgemm`'s reported GFLOP/s is
  pessimistic relative to a true complex-FLOP accounting** (roughly 4x
  fewer "counted" FLOPs than actual multiply-adds executed, ignoring the 2
  extra adds). Not corrected here; the cost model is out of this
  integration's scope to change, per the task brief.
- **Layout**: `cgemm.cuh` indexes `A`/`B`/`C` column-major (`gA[row +
  col*M]`, `gB[row + col*K]`, `gC[row + col*M]`) — cuBLAS's own NN
  convention, cross-checked against `fused.cu`'s own reference call
  `cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, alpha, dA, M, dB,
  K, beta, dC_ref, M)`. `prepare()` converts each row-major `(M,K)`/`(K,N)`
  numpy operand to column-major bytes via `np.ascontiguousarray(X.T)` (a
  `(K,M)`/`(N,K)` C-contiguous array whose flat memory is bit-identical to
  the row-major array's transpose interpreted column-major) — a host-side
  reshape done once in `prepare()`, timed as preprocessing, no extra device
  computation. `to_host()` reverses this on the `(batch,N,M)` output.
- **No batching in the kernel**: `cgemm` has no batch parameter. `run()`
  loops the same unmodified `turbofno_cgemm_launch` once per batch element
  on a distinct pointer offset for the `batch>1` smoke shape — an
  independent kernel invocation per batch item, not a new fused/batched
  kernel.
- **Alignment constraint** (inherited, not introduced): `cgemm.cuh` has no
  tail/boundary handling, so `prepare()` raises `NotImplementedError` if
  `M % 64`, `N % 64`, or `K % 8` is nonzero, rather than silently launching
  a kernel that reads/writes out of bounds. All three `--smoke` gemm shapes
  (256/256/256, 384x256x512, batch4-64x64x64) satisfy this; general
  `gemm-square-*`/`gemm-cubic-*` sizes were NOT swept (several, e.g. 8, 16,
  2049, are not multiples of 64/8) — see "Not done" below.
- `timer()` returns `kernelbench.impls.gpu_cuda.CudaEventTimer`.

## Gate result (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
$PY -m kernelbench.runner --kernel gemm --variant gemm-square-kernel \
    --impl turbofno-cgemm --smoke --warmup 1 --reps 3 --precision fp32
```

Result: **3/3 runs valid**, `max_scaled_err` tolerance `1e-6`:

| smoke shape | max_scaled_err | GFLOP/s (pessimistic, see caveat above) |
|---|---|---|
| smoke-gemm-square-256 (256x256x256) | 1.7635e-07 | 181.16 |
| smoke-gemm-irregular-384x256x512 | 2.3738e-07 | 345.80 |
| smoke-gemm-batched-b4-64 (batch=4, 64x64x64) | 1.8609e-07 | 10.91 |

All comfortably within tolerance (roughly fp32-unit-roundoff scale, ~1e-7,
as expected for a correctly-implemented fp32 GEMM whether real or
zero-imaginary complex). Per ARTIFACT_GUIDE.md rule 4, no tolerance override
was applied or needed. `conforming: False` in the result JSON is expected
and correct — this is a `--smoke`, protocol-overridden (`warmup=1, reps=3`)
functional check on a shared login-node GPU, not a publishable timing run.

## Not done (out of this integration's ~25-minute budget)

- No padding logic for non-tile-aligned `gemm-square-*`/`gemm-cubic-*`/
  `gemm-irregular-*` shapes that aren't multiples of 64 (M,N) / 8 (K) —
  `prepare()` raises cleanly instead of corrupting memory; would need
  zero-padding to the tile grid (same pattern as `inferfast/adapter.py`'s
  `_round_up`) to run those.
- No sweep across the full `gemm-square-*` size list or `gemm-batched-*`
  batch counts — only the harness's own 3-shape `--smoke` set was run, per
  the task's login-node/GPU-shared budget.
- `cgemm`'s `alpha`/`beta` generality (arbitrary complex scale/accumulate)
  is wired in the launcher but only ever exercised at `alpha={1,0},
  beta={0,0}` (the values this real-embedded-in-complex gating strategy
  needs) — not swept.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, 14 SMs, ~5 GB; host GPU
  reports as NVIDIA A100-SXM4-40GB via nvidia-smi under the MIG partition),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 12.4.0 (conda-forge,
  `KB_GXX12`, same role as Perlmutter's g++-12 pin), `-gencode
  arch=compute_80,code=sm_80`. Commit unchanged:
  `215d916419b9a7745b803e20d20a45afd335ce14`.
- Build: OK, no fixes needed -- `build.sh` was already machine-neutral
  (`NVCC`/`HOST_COMPILER` both read `${VAR:-...}` with `KB_GXX12` in the
  fallback chain) and built clean on the first try.
- Gate: `--variant gemm-square-kernel --smoke --warmup 1 --reps 3 --precision
  fp32`: **PASS** 3/3 (err 1.76e-07 / 2.37e-07 / 1.86e-07) -- identical
  error magnitudes to the recorded run.
- Deviation from the recorded ruling: none in outcome. The harness reports
  tolerance `1e-3` here versus the `1e-6` this STATUS.md originally
  recorded -- same per-precision-tolerance-table change documented in
  `gemm/hexcute/STATUS.md`'s "Dependency isolation (2026-09-05)" section
  (`gemm-square-kernel`'s fp32 bound is now read from
  `tolerance_for("fp32")` instead of the old blanket fp64-text parse); the
  ~1e-7 errors clear either bound by a wide margin, so this is a
  reported-tolerance change only, not a pass/fail change.
- Verdict here: BUILT+GATED -- equals the recorded ruling.
