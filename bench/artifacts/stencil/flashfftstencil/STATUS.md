# FlashFFTStencil (2D box2d1r) — STATUS

**Outcome: BUILT+GATED — gate PASSES**

Paper: "FlashFFTStencil: Bridging Fast Fourier Transform and Fast Stencil
Computation on Tensor Core Units", PPoPP 2025.
PAPER_KEY = `conf/ppopp/HanLCBZYCZCY25`.
Repo: `https://github.com/KevinWu2017/FlashFFTStencil`
(commit `4579ea11ccb490aaa5d6d6246e1345c0e9b48c1f`, 2025-06-05;
`git clone --depth 1` into `./source/`, untouched — read-only).

## Toolchain / provenance

- nvcc 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc`,
  `Cuda compilation tools, release 12.9, V12.9.41`).
- Arch flag: `-arch=sm_80` (A100).
- cuFFT: NOT under `cuda/12.9/lib64` on this HPC SDK install (verified
  absent) — found instead at
  `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/{include,lib}`
  (`libcufft.so` present there); `build.sh` points `-I`/`-L` there and links
  `-lcufft`. This is the one build-system fix needed (an include/lib-path
  correction, not a code change).
- Python: `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.

## What the artifact actually implements (read, not assumed)

`source/src/2D/2d_main.cu`'s `main()` hardcodes `KERNEL_WIDTH = 3` — the
only shape this artifact's 2D path implements is a 3x3 box kernel, i.e.
`box2d1r` in this domain's AN5D naming (radius 1, 2D, Chebyshev-ball
support). The CLI's `kernel_shape` argv (`"Heat-2D"` vs `"Box2D9P"`) only
changes a printed label (`kernel_shape_output`) in the metrics-row print —
it does **not** branch the actual computation, confirmed by reading the
full `main()` body; there is no star2d1r (5-point) code path. There is also
no `source/src/3D` directory in this clone at all — 3D stencils are simply
not implemented by this artifact. Both are documented gaps, not oversights.

Data type is `double` throughout: `h_input_gpu`/`h_kernel`/`h_output`
(`2d_main.cu`) and every `wmma::fragment<..., double, ...>` in the actual
compute kernel `rfft_2d_8_nwarp<>`
(`source/src/2D/rfft_2d/2d_rfft_1_async.cu`) — a genuine fp64 Tensor-Core
(DMMA) matmul path, confirmed by reading the fragment declarations, not
fp16. Despite FlashFFTStencil being listed in spec.yaml's
`stencil-tcu-matmul-kernel-fp16` variant's paper-family `suite` field
(alongside ConvStencil/LoRAStencil/SPIDER), **this artifact's own numbers
are computed and reported in double precision**, so `PRECISIONS = ["fp64"]`
and this adapter is gated against **variant 1
(`stencil-cpu-gpu-kernel-fp64`)**, whose `dense_operand: fp64` and
`box2d1r` recommended-subset entry genuinely match what is being measured
— not variant 2, whose fp16 tolerance/precision framing doesn't fit a
kernel that never touches fp16 data.

## Bridge code added (why, from reading `2d_main.cu` end to end)

`main()` bundles three things that need separating per the integration
contract: (a) `CreatePlan()` — one-shot cuFFT of a 3x3 kernel into
constant-memory DFT tables; (b) host-side overlap-add packing of a flat
`INPUT_WIDTH x INPUT_WIDTH` field into a tiled device buffer (`unit=8`
zero-padded tiles, `sub_input_width = unit-(KERNEL_WIDTH-1) = 6` valid
samples per tile — chosen so `6+3-1=8`, exactly the minimal circular size
that avoids wraparound aliasing, i.e. true overlap-add, not overlap-save);
(c) `T` back-to-back kernel launches wrapped in one `cudaEvent` pair.

`bridge.cu` (this directory, new file, NOT part of the artifact) `#include`s
`source/src/2D/rfft_2d/2d_rfft_1_async.cu` verbatim (which itself pulls in
`create_fft_pfa_plan.cu` and `elementwise_mul.cuh` via its own relative
`#include`s, unmodified) and splits the driver into:
- `fftstencil2d_prepare()` — (a)+(b), called once: builds the DFT tables
  from a **caller-supplied** kernel (not `rand()`), packs the field, H2D
  copies. This is genuinely the artifact's own preprocessing, timed once by
  `prepare()`.
- `fftstencil2d_run()` — exactly one `rfft_2d_8_nwarp<1>` launch, re-zeroing
  `d_output` first (see below).
- `fftstencil2d_copy_output()` / `fftstencil2d_free()` — D2H copy / cleanup.

No kernel-logic changes: `rfft_2d_8_nwarp<>` and `CreatePlan` are compiled
from the artifact's own `.cu` files, byte-for-byte, via `#include`.

**Necessary addition (same class of fix as the cb-spmv sibling adapter)**:
the kernel accumulates into `output` via `atomicAdd` (its own overlap-add
scheme, `2d_rfft_1_async.cu`); the artifact's own driver only runs its
T-loop once per process and never re-zeros/re-validates between launches.
Our harness calls `run()` independently many times (isolated correctness
check, warmup, each measured rep), so `fftstencil2d_run()` re-zeros
`d_output` (`cudaMemset`, O(grid) — cheap relative to the kernel) before
every launch.

## Weight injection + the one non-obvious correctness fix

`CreatePlan(double *k, int KERNEL_WIDTH, bool)` takes the 3x3 kernel as an
explicit argument, so `adapter.py::prepare()` feeds it the workload's own
`box2d1r` weights (`workload_.dense_kernel(dtype=np.float64, flip=True)`)
instead of the CLI's `rand()`-filled values — no weight *mutation* needed.

However, `CreatePlan`/the tiling geometry anchor the `KERNEL_WIDTH x
KERNEL_WIDTH` kernel at the **top-left** corner of each `unit x unit=8x8`
zero-padded FFT tile (`create_fft_pfa_plan.cu`:
`h_kernel[i*unit+j] = k[i*KERNEL_WIDTH+j]` for `i,j` in `0..2`), not
centered at `(1,1)`. This is a fixed coordinate/anchor convention in the
algorithm's tiling geometry, not something any choice of kernel *values*
can cancel. Derived analytically (feeding `dense_kernel(flip=True)`, the
overlap-add output at position `p` works out to
`sum_o weights[o]*u[(p-(1,1))+o]`, i.e. the reference's answer shifted by
`(1,1)`) and **confirmed empirically** on a 12x12 synthetic grid before
writing the final adapter: of all `(kernel-flip, output-roll)` combinations
tried, only `kernel=dense_kernel(flip=True)` + `roll=(-1,-1)` on both axes
matched the reference, at `max_rel_err = 1.39e-15` (every other combination
tried gave errors in the `1e-2`–`1` range). `adapter.py::to_host()` applies
this roll (`np.roll(raw, shift=(-1,-1), axis=(0,1))`) — a coordinate
bookkeeping step on the artifact's own already-computed output, not a
change to what either side numerically computes; `run()` remains exactly
one kernel launch and `prepare()` never touches the field's own values.
Documented in full in `adapter.py`'s module docstring.

No boundary crop was needed: the kernel's `atomicAdd` output indexing
already wraps at the domain edges
(`idx_x = (row >= sub_input_width) ? (row - sub_input_width) : ...`,
`2d_rfft_1_async.cu`), i.e. the overlap-add scheme implements a genuinely
**global periodic-wrap** convolution over the whole `INPUT_WIDTH x
INPUT_WIDTH` domain — matching `stencil.py`'s periodic-wrap reference
exactly (after the anchor-shift roll above), not a tile-local
approximation.

## Timesteps

`main()`'s `T`-loop relaunches `rfft_2d_8_nwarp<>` reading the **same**
`d_input` and overwriting the **same** `d_output` every iteration —
`d_output` never becomes the next `d_input`. This is repeat-for-timing-
stability, not the domain's ping-pong T-sweep recursion (same situation as
the SPIDER sibling artifact, flagged in advance). `adapter.py::prepare()`
therefore mutates `workload.timesteps = 1` in place before returning, so
the harness's own `reference_stencil()` (called after `prepare()`, on the
same mutated workload object) computes a matching single-sweep answer.
`run()` performs exactly one kernel launch per call.

## Shape constraint (documented gap)

`INPUT_WIDTH` must be a multiple of `sub_input_width = unit-(KERNEL_WIDTH-1)
= 6` — the artifact's own tiling constraint, not introduced by this
adapter. **Neither spec.yaml grid size satisfies this**: variant 1's
16384 (16384/6 = 2730.67) and variant 2's 10240 (10240/6 = 1706.67) are both
non-divisible by 6. `adapter.py::prepare()` raises a clear
`NotImplementedError` for any grid width that doesn't satisfy this (rather
than silently rounding); the gate below therefore uses a small, explicitly
6-divisible synthetic grid instead of the spec-sized default. Only
`box2d1r` is implemented (see "What the artifact actually implements"
above for why `star2d1r` and all 3D shapes are out of scope) — so
`--smoke` (which requires `smoke-star3d1r`/`smoke-box3d1r`) is **not**
supported by this adapter and was not attempted.

## Build

```
./build.sh
```
Idempotent (single `nvcc -shared` invocation): `nvcc -O3 -arch=sm_80
-Xcompiler -fPIC -shared -I <cufft-include> -L <cufft-lib> bridge.cu
-lcufft -o bridge.so`. Verified from a clean state (`rm -f bridge.so &&
./build.sh`) — builds clean, no warnings suppressed beyond `-w` (artifact's
own headers are noisy; no bridge.cu-specific warnings observed with `-w`
removed during development).

## Gate verification

The mandated command (`--matrices box2d1r`) is **not usable as-is** because
`domain.load_workload("box2d1r")` has no CLI way to override
`grid_shape`, and its default (spec.yaml's 16384) fails the /6 constraint
above (this is exactly the "shape/boundary issue blocks it within budget"
case the integration task anticipated as a possible outcome). Per the
task's own permitted fallback, a standalone script called
`kernelbench.harness.run_variant()` directly, using the **same**
`stencil-cpu-gpu-kernel-fp64` variant/spec object the runner CLI would load,
against a small synthetic `box2d1r` workload with a 6-divisible grid
(`96x96`, `timesteps=1`, `precision="fp64"`, `warmup=1, reps=3` — matching
the worked example's own non-conforming-but-documented login-node pattern):

```python
sp = spec.load("stencil")
variant = sp.variant("stencil-cpu-gpu-kernel-fp64")
wl = stencil.StencilWorkload(name="box2d1r-gate-96", kind="box", dims=2,
                              radius=1, grid_shape=(96, 96), timesteps=1,
                              precision="fp64")
r = harness.run_variant(impl, wl, variant, {"seed": 42, "precision": "fp64"},
                         reference=stencil.REFERENCES["stencil"],
                         correctness_mode=stencil.CORRECTNESS_MODE["stencil"],
                         reference_name="numpy fp64 periodic-wrap sweep",
                         warmup_override=1, reps_override=3)
```

Output:
```
valid: True
correctness: CorrectnessResult(passed=True, metric='max_scaled_err',
  value=4.400138080714661e-15, tolerance=1e-05,
  reference='numpy fp64 periodic-wrap sweep', note='',
  max_pointwise_rel_err=6.430035071919887e-12,
  max_abs_err=1.1657341758564144e-15, l2_rel_err=9.376653547358306e-16)
warnings: ['protocol overridden for this run (warmup=1, reps=3); result is
  NOT spec-conforming and must not be published as such', 'high variance:
  stdev 0.1553 ms is >25% of median 0.0416 ms — check for a
  shared/contended device']
workload after prepare (mutation check): timesteps= 1
```

`max_scaled_err = 4.40e-15`, tolerance `1e-5` (variant 1's stated bound) —
**PASS**, by ten orders of magnitude of margin (expected: both sides are
fp64, and a single sweep leaves no room for compounding error). The
high-variance warning and the `warmup=1/reps=3` override are expected —
this is a login-node correctness check on a shared GPU, not a timing claim
(per this integration's own rules: "Login node: build + a minimal
functional/gate check only... timing measurements are NOT [reliable]").

## Verdict

`flashfftstencil: BUILT+GATED err=4.40e-15 (tol 1e-5, variant
stencil-cpu-gpu-kernel-fp64, workload box2d1r 96x96 T=1 — spec-sized grid
16384 incompatible with the artifact's own /6 tiling constraint, see
above)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, parent card reported by
  `nvidia-smi` as "NVIDIA A100-SXM4-40GB, 8.0" on `gpu-b11-6`), login-node
  build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8, `$KB_CUDA_HOME`), host
  compiler: g++ 12.4.0 (conda-forge `kb-gcc12` env, via `-ccbin`, added —
  see below), Python 3.12.14 (`kb-env`). cuFFT resolved via `CUFFT_PREFIX`,
  which `bench/artifacts/toolchain.sh` already exports pointing at
  `$KB_CUDA_MATHLIBS/targets/x86_64-linux` (conda-forge CUDA's own cuFFT,
  `libcufft.so`/`cufft.h` present there) — `build.sh`'s existing
  `CUFFT_PREFIX="${CUFFT_PREFIX:-<Perlmutter HPC SDK path>}"` fallback
  pattern picked this up with no change needed. Same source commit
  (`4579ea11ccb490aaa5d6d6246e1345c0e9b48c1f`) per `source.provenance`.
- Build: OK. Build-system changes: `build.sh` added
  `HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"` and
  `-ccbin "$HOST_COMPILER"` to the `nvcc -shared` compile of `bridge.cu`
  — this artifact's `nvcc` call previously had no `-ccbin`, so `nvcc` did
  its own host-compiler search rather than following `$PATH`, and on this
  machine that search finds `$KB_CUDA_HOME`'s own bundled g++ 13.4.0 first
  (nvcc 12.8 cannot parse it — same failure class documented in
  `bench/artifacts/convolution/hidet/STATUS.md`'s Reproduction section, and
  the same `-ccbin "$HOST_COMPILER"`/`KB_GXX12` fix already used by most
  other artifacts in this repo). No change to `bridge.cu`, `adapter.py`, or
  anything under `source/`.
- Gate: the mandated `--matrices box2d1r` runner command is still not usable
  here for the same reason recorded originally (grid-size /6 constraint);
  wrote `_gate_check.py` (this directory, new file) reproducing the exact
  standalone `harness.run_variant()` call STATUS.md's "Gate verification"
  section shows inline (same variant object, same synthetic `box2d1r`
  96x96, T=1, seed=42, `warmup_override=1, reps_override=3`). Result:
  `max_scaled_err = 4.400138080714661e-15` vs tolerance `1e-05` —
  **PASS**, bit-for-bit identical to the recorded value.
- Deviation from the recorded ruling: none — identical correctness number.
- Verdict here: BUILT+GATED (err 4.40e-15, tol 1e-5, workload box2d1r 96x96
  T=1) — equals the recorded ruling.
