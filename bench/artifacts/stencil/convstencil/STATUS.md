# ConvStencil — STATUS

**Outcome: BUILT+GATED (2D only) — gate PASSES on the one shape wired up; 3D/1D not integrated (time budget)**

Paper: "ConvStencil: Transform Stencil Computation to Matrix Multiplication
on Tensor Cores", PPoPP 2024. PAPER_KEY = `conf/ppopp/ChenLWBWMYZCY24`.
Repo: `https://github.com/microsoft/ConvStencil` (commit
`89688a1b51ec41b4a81028b0661363ba3afd6050`, 2024-04-07;
`git clone --depth 1` into `./source/`).

This is a BACKUP substitution for the batch's original 3rd-newest stencil
candidate, WindStencil (SKIPPED: AMD ROCm/HIP-only, no CUDA build path on
this machine; see `../windstencil/STATUS.md`). ConvStencil was chosen
because its shape catalog (star2d1r, box2d1r, star2d3r, box2d3r, star3d1r,
box3d1r, 1d1r, 1d2r) covers all 3 of the harness's `--smoke` shapes on
paper, and it advertises user-supplied custom weights.

## Why a bridge was needed, and why it only covers 2D

The artifact's CLI (`source/src/2d/main.cu`) advertises `--custom` for
supplying weights, but reading the whole file shows it is broken for the
star-shaped 2D cases: the shape switch selects `param_star_2d1r` for
`star_2d1r`/`star_2d3r`, but that 49-double array is declared
zero-initialized and is **never written to anywhere in the file** — only
`param_box_2d1r` is populated (via a box-cubed-to-7x7 polynomial expansion
driven by `--custom`'s 9 inputs). So the CLI's `star2d1r` shape (needed for
`smoke-star2d1r`) silently computes with all-zero weights in this exact
commit; that's a bug/dead code path in the artifact itself, not something
this integration introduced or needs to route around by touching kernel
code — it's simply never reached, because `bridge.cu` (this directory, NOT
part of the artifact) calls `gpu_box_2d1r()` (`source/src/2d/gpu.cu`)
directly with weights supplied from the harness's own
`StencilWorkload.weights`, bypassing `main.cu` entirely.

`gpu_box_2d1r()` is otherwise used **verbatim, unmodified** — no kernel
math was touched. `bridge.cu`'s docstring has the full reasoning; summary:

1. **Weight layout**: ConvStencil's kernel always operates on a fixed 7x7
   (radius-3) box support (`HALO=3` hardcoded in `gpu.cu`), regardless of
   the requested shape's true radius. Radius-1 shapes (star2d1r, box2d1r)
   are expressed by zero-padding the unused outer ring of the 7x7 weight
   array (`adapter.py::_build_params`). Verified the index convention
   against `main.cu`'s own `naive_box2d1r` reference loop:
   `param[i*7+j]` weights `in[row+(i-3)][col+(j-3)]` — matches
   `StencilWorkload.dense_kernel(flip=False)`'s cross-correlation
   convention exactly, radius fixed at 3.

2. **T-sweep semantics** (checked directly in `gpu.cu`): `gpu_box_2d1r`'s
   internal `times`-argument loop ping-pongs device buffers **without
   ever refreshing the halo band between sweeps** — sweep 0 uses the
   halo as H2D-copied in; sweeps 1..times-1 reuse that same, now-stale
   halo. This does NOT match `kernelbench/domains/stencil.py`'s
   periodic-wrap T-sweep recursion (every sweep re-wraps from the
   *current* field). So the artifact's internal multi-`times` path is
   **not used**; `bridge.cu` exposes a single-sweep entry point
   (`times=1` always) and `adapter.py::run()` performs its own T-sweep
   loop, re-building the periodic-wrap padded host buffer
   (`np.pad(..., mode="wrap")`) from the current field before each
   single-sweep call. This is the "genuinely different T-sweep
   semantics" case flagged in the integration brief — `workload.timesteps`
   maps to this adapter's OWN outer loop, not to the CLI's `times` arg.

3. **Boundary/padding offsets**: `gpu_box_2d1r` expects the host input as
   an already-padded `(m+2*HALO) x (n+2*HALO+2)` buffer (HALO=3), matching
   how `main.cu` sizes its own host buffer. Reading `gpu.cu`'s `kernel2d`
   output-store index arithmetic (`out + begin + IDX(HALO+col/7, HALO,
   ldm)` with `begin`'s column component `blockIdx.y*BLOCK_SIZE_COL+1`)
   showed valid output starts at buffer row `HALO`(=3) and buffer column
   `HALO+1`(=4); since `cols_total - n = 2*HALO+2 = 8 = 4+4`, this is
   consistent with a **symmetric 4-wide column margin** (3-wide row
   margin). `bridge.cu`/`adapter.py` pad periodic-wrap with these margins
   (row 3/3, col 4/4) and extract the result at the same offsets.

   This offset hypothesis was **verified empirically before being wired
   into adapter.py**, not just derived on paper: a standalone ctypes test
   (`convstencil2d_sweep` called directly) against a hand-written
   periodic-wrap numpy reference (`np.roll`-based), on a random 16x20
   field, gave:
   - identity weights (center=1 only): max err **0.0**
   - pure column shift (offset (0,1)): max err **0.0**
   - pure row shift (offset (1,0)): max err **0.0**
   - a genuine 5-point star (center 0.5, +-0.125 axis neighbors,
     sign-alternating): max err **2.22e-16** (fp64 rounding only)

## Coverage: 2D only (1D, 3D not wired up — time budget)

`source/src/3d/{gpu_box.cu,gpu_star.cu}` use a differently-shaped
tensor-core tiling (separate box/star kernels, different shared-memory
layout) and `source/src/1d/{gpu_1r.cu,gpu_2r.cu}` likewise were not
reverse-engineered in the time available for this integration pass.
Rather than guess at their padding/layout and risk silently-wrong output,
`adapter.py::prepare()` raises a clear `NotImplementedError` for
`workload.dims != 2`. This is intentional per the integration brief's
explicit fallback: "gate at least the 2D shape cleanly and document 1D/3D
as attempted-but-incomplete rather than silently dropped." Confirmed this
fails loudly (not silently) in the actual `--smoke` run below.

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 -Xcompiler -fPIC -shared -I source/src/2d bridge.cu
source/src/2d/gpu.cu -o bridge.so`. nvcc 12.9
(`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc`), target A100 /
sm_80 (matches the known-good arch flag the sibling SPIDER artifact already
uses to build its vendored ConvStencil submodule via
`cmake -DCMAKE_CUDA_ARCHITECTURES=80` on this exact machine). Idempotent
(single `nvcc` invocation compiling `bridge.cu` + `source/src/2d/gpu.cu`
directly, no intermediate build dir, no patches to `source/`). Build ran
clean, no warnings suppressed beyond `-w` (matches the artifact's own
CMakeLists.txt convention of a bare `-O3` build for this target).

## Adapter

`IMPL_NAME = "convstencil-tcu"`, `PRECISIONS = ["fp64"]` — ConvStencil's
`DATA_TYPE` (`2d_utils.h`) and every `wmma` fragment in `gpu.cu` are
`double`-templated; there is no fp16/fp32 path in this artifact despite the
spec variant's own name (`stencil-tcu-matmul-kernel-fp16` targets fp16 as
its *default* protocol precision, but this specific artifact computes in
fp64 — confirmed by reading `2d_utils.h`'s `#define DATA_TYPE double` and
every fragment declaration in `gpu.cu`). `--precision fp64` passed
explicitly on every gate command per the brief's precision-autodetect
footgun note.

`prepare()`: builds the 49-double weight array from
`workload.weights` (raises `NotImplementedError` for `dims != 2` or
non-periodic boundary before doing any work). `run()`: the adapter's own
T-sweep loop (see point 2 above), one `convstencil2d_sweep` ctypes call per
timestep. `to_host()`: the field is already host-resident (`gpu_box_2d1r`
does its own D2H copy internally), so this is a dtype-normalizing copy,
not a device transfer.

## Gate verification (mandated command, run as specified)

```
$PY -m kernelbench.runner --kernel stencil \
    --variant stencil-tcu-matmul-kernel-fp16 --smoke \
    --impl convstencil-tcu --precision fp64
```

```
  [env] running on a LOGIN node: CPU and GPU are shared with other users — timings are indicative only, never publishable
  running convstencil-tcu smoke-star2d1r  ... 2.940 ms  0.11 GCUP/s  (err 1.14e-16 <= 0.01)  [0.4s]
  running convstencil-tcu smoke-star3d1r  ...
Traceback (most recent call last):
  ...
  File ".../artifacts/stencil/convstencil/adapter.py", line 106, in prepare
    raise NotImplementedError(
NotImplementedError: convstencil-tcu: only the 2D kernel is wired up in this integration pass; 1d and 3d were not bridged. dims=3
```

`smoke-star2d1r` (the one 2D smoke shape): `max_scaled_err = 1.14e-16`
against the variant's own tolerance `0.01` — **PASS**, and in fact well
within the stricter fp64-variant tolerance (`1e-5`, variant 1) too. This
matches the independent hand-verification above (2.2e-16 for a comparable
weight pattern) — consistent, not a fluke.

`smoke-star3d1r` then raises the documented `NotImplementedError` (3D not
wired up), which aborts the rest of the `--smoke` run — expected, per the
brief's own note that `--smoke` has no `try/except` around `prepare()`.
`smoke-box3d1r` was consequently never reached in this run; both 3D shapes
are equally uncovered (`prepare()` rejects any `dims != 2`), so this is not
an additional gap beyond what's already documented above.

Full raw log: `gate_full.log` output is not committed here (contains the
artifact's own noisy per-call stdout, `ConvStencil(2D): ...GStencil/s=...`,
from `gpu_box_2d1r`'s own `printf`s inside the T-sweep loop — cosmetic
only, one line per adapter-level `run()` timestep call, not a correctness
signal).

## Verdict

`convstencil: BUILT+GATED err=1.14e-16 | BUILD (2D-only; 1D/3D gate blocked: not wired up, time budget)`

2D (`smoke-star2d1r`) is a genuine, verified pass: independently
cross-checked (identity/shift/5-point-star numpy comparisons, 0 to 2.2e-16
error) before ever touching the harness, then reproduced through the actual
mandated `--smoke` gate command with a consistent, comparably-tiny error
(1.14e-16). 1D and 3D are honestly incomplete, not silently dropped:
`prepare()` raises immediately and specifically for those shapes rather
than risk producing wrong numbers.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, parent card reported by
  `nvidia-smi` as "NVIDIA A100-SXM4-40GB, 8.0" on `gpu-b11-6`), login-node
  build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8, `$KB_CUDA_HOME`), host
  compiler: g++ 12.4.0 (conda-forge `kb-gcc12` env, via `-ccbin`, added —
  see below), Python 3.12.14 (`kb-env`). Same source commit
  (`89688a1b51ec41b4a81028b0661363ba3afd6050`) per `source.provenance`.
- Build: OK. Build-system changes: `build.sh` added
  `HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"` and
  `-ccbin "$HOST_COMPILER"` to the `nvcc -shared` compile of
  `bridge.cu`+`source/src/2d/gpu.cu` — this artifact's `nvcc` call
  previously had no `-ccbin`, so `nvcc` did its own host-compiler search
  rather than following `$PATH`, and on this machine that search finds
  `$KB_CUDA_HOME`'s own bundled g++ 13.4.0 first (nvcc 12.8 cannot parse
  it — same failure class documented in
  `bench/artifacts/convolution/hidet/STATUS.md`'s Reproduction section, and
  the same `-ccbin "$HOST_COMPILER"`/`KB_GXX12` fix already used by most
  other artifacts in this repo). No change to `bridge.cu`, `adapter.py`, or
  anything under `source/`.
- Gate (mandated command, unchanged): smoke-star2d1r
  `err 1.14e-16 <= 0.01` **PASS** — bit-for-bit identical error to the
  recorded number. smoke-star3d1r / smoke-box3d1r: **UNSUPPORTED**
  (`1/1 runs valid (2 unsupported)`, exit 0) — see "Deviation" below.
- Deviation from the recorded ruling: the correctness number is unchanged,
  but the `--smoke` command's behavior on the two unwired 3D shapes differs
  from what the original STATUS.md recorded: originally, hitting
  `smoke-star3d1r` raised an uncaught `NotImplementedError` that aborted
  the rest of the `--smoke` run right after printing the (passing)
  `smoke-star2d1r` line, so `smoke-box3d1r` was "never reached." Here, the
  harness (`bench/kernelbench/runner.py`, not touched by this reproduction
  pass) instead reports both unwired shapes as `UNSUPPORTED` and exits 0
  cleanly, reaching and correctly rejecting `smoke-box3d1r` too — the same
  harness-side behavior change already observed and documented in
  `bench/artifacts/convolution/tetris/STATUS.md`'s Reproduction section
  (a harness improvement, not an artifact-outcome change; `bench/kernelbench/`
  is off-limits per rule 1 and was not touched here).
- Verdict here: BUILT+GATED (2D only; err 1.14e-16, tol 0.01; 1D/3D still
  not wired up, same as recorded) — equals the recorded ruling.
