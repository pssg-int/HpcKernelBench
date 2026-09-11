# LoRAStencil — STATUS

**Outcome: BUILT+GATED (star2d3r only) — gate PASSES; box2d1r/box2d3r/star2d1r
genuinely unwrappable, evidenced below (not a coverage shortcut)**

Paper: "LoRAStencil: Low-Rank Adaptation of Stencil Computation on Tensor
Cores", SC'24. `PAPER_KEY = conf/sc/ZhangLYCZCY24`.
Repo: `https://github.com/HPHEX/LoRAStencil` (commit
`1cdb7e25e09d48c0e0911513bff9165a6f8b8ef8`, 2024-12-26; `git clone --depth 1`
into `./source/`, untouched — read-only, no patch).

## Toolchain / provenance

- nvcc 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`), via
  `bench/artifacts/toolchain.sh`.
- Arch flag: `-arch=sm_80` (A100).
- Python: `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.
- Build: single `nvcc -O3 -std=c++17 -arch=sm_80 -Xcompiler -fPIC -shared
  bridge.cu -o bridge.so` (no CUTLASS/cuFFT/extra libs needed). Verified
  clean from a fresh clone.

## What the artifact actually implements (read, not assumed)

`source/src/2d/gpu.cu` (+`2d_utils.h`) implements FOUR host entry points
for 2D: `gpu_box_2d3r` (dispatched by `main.cu` for BOTH `box2d1r` and
`box2d3r` CLI names — there is no separate `gpu_box_2d1r`), `gpu_star_2d3r`,
and `gpu_star_2d1r`. All four are genuine fp64 (`DATA_TYPE double`
throughout, every `wmma::fragment<..., double, ...>` in `gpu.cu`) Tensor-
Core (DMMA) matmul reformulations — same situation as the convstencil/
flashfftstencil siblings already in this directory (also fp64 despite
appearing in spec.yaml's fp16-labeled `stencil-tcu-matmul-kernel-fp16`
variant's `suite` field). `PRECISIONS = ["fp64"]`, gated against **variant
1** (`stencil-cpu-gpu-kernel-fp64`), matching that established precedent.

`bridge.cu` (this directory, NOT part of the artifact) `#include`s
`source/src/2d/gpu.cu` verbatim and adds two NEW extern-"C" entry points
(`lorastencil2d_box2d3r_sweep`, `lorastencil2d_star2d3r_sweep`) that call
the artifact's own unmodified `gpu_box_2d3r`/`gpu_star_2d3r` with
`times=1` and crop the padded output — no kernel arithmetic touched.

## Why only star2d3r is wrapped (verified empirically, not assumed)

A standalone ctypes probe (built before writing `adapter.py`, see
transcript below) tested all three "shape families" against this domain's
own weight construction (`stencil.py::_build_weights`) on a 64x64 random
field:

1. **box2d1r / box2d3r — genuinely broken for arbitrary weights, not
   merely approximate.** `gpu_box_2d3r`'s own host-side "Factorize
   parameter matrix" step (`gpu.cu:276-352`) is NOT a general low-rank
   decomposition — it is a hardcoded 4-level "onion-ring" peel matching
   LoRAStencil's own synthetic benchmark pattern (`main.cu`'s
   `param_box_2d1r`: each concentric square ring assigned ONE constant
   scalar value). It performs unguarded divisions by specific matrix
   entries at each level:
   - Level 0 (`gpu.cu:290`): `prop = params[(i+3)*7] / params[0]` divides
     by the weight matrix's own CORNER entry (offset `(-3,-3)`). Fed an
     IDENTITY kernel (single nonzero at center, corner exactly 0):
     **all-NaN output** (`0.0/0.0`) — confirmed directly, this is exactly
     what a true zero-padded radius-1 box2d1r would look like once
     embedded in the fixed 7x7 layout.
   - Level 1 (`gpu.cu:307`): `prop = temp[...][1*7+1] / temp[...][7+1]`.
     Fed this domain's OWN box2d3r weights (corner nonzero, so level 0
     survives), the level-0 residual concentrates entirely in the CENTER
     row (this domain's sign-by-first-nonzero-coordinate construction
     makes every off-center row internally constant, so it is peeled
     perfectly at level 0) — but level 1's reference entry sits at a
     DIFFERENT row this domain's residual is exactly zero at, so
     `0.0/0.0` again: **all-NaN output**, confirmed directly (see
     transcript).
   Neither of these is a build-system issue or something a caller-side
   weight-injection convention can route around: they are unconditional
   divisions inside the artifact's own unmodified factorization code that
   assume a specific benchmark-pattern structure this domain's (or any
   generic) weight matrix does not have. Per ARTIFACT_GUIDE.md rule 3
   (kernel/solver code may not be patched), this is left as a documented
   SKIP, not routed around.
2. **star2d1r — kernel ignores its own `params` argument.**
   `gpu_star_2d1r` (`gpu.cu:484-487`) hardcodes
   `param_u = param_v = {0, 1, 2, 4, 2, 1, 0}` as LOCAL C arrays, never
   reading `params` at all. There is no way to inject this domain's
   star2d1r weights into this kernel — confirmed by reading the full
   function body (the `params` pointer is accepted but never dereferenced).
3. **star2d3r — exact, verified.** `gpu_star_2d3r` (`gpu.cu:426-481`)
   builds its two factors by PLAIN ASSIGNMENT from `params`
   (`param_matrix_U[...] = params[col*7+3]`, the vertical center line;
   `param_matrix_V[...] = params[3*7+row]` for `row != 3`, the horizontal
   center line excluding the already-counted center point) — no division
   anywhere. This is mathematically exact for a star-shaped support: a
   star sum decomposes losslessly into an independent vertical-line
   convolution plus an independent horizontal-line convolution (no cross
   terms to approximate away). Confirmed empirically, not just derived:
   fed this domain's own star3r weight construction on a 64x64 random
   field, **max abs error 3.33e-16** against `stencil.py`'s own
   periodic-wrap offset-loop reference (fp64 rounding only).

## Verification transcript (standalone ctypes probe, before wiring into adapter.py)

```
identity max err: nan                                    # box2d3r fn, identity kernel
shift(0,1) vs cross-corr (roll -1 ax1): nan               # box2d3r fn, pure shift kernel
box2d3r domain-weights: max abs err nan max rel err nan   # box2d3r fn, this domain's box3r weights

star2d3r domain-weights: max abs err 3.3306690738754696e-16 max rel err 4.91068163850247e-16
any nan: False
star2d3r identity max err: 0.0
```

This also fixed the weight-layout convention (`params[i*7+j]` weights
offset `(i-3,j-3)`, cross-correlation / `dense_kernel(flip=False)` —
matching convstencil's own convention for this paper family) and the
output valid-region offset (`(row=4, col=4)` within an `(m+8)x(n+8)`
periodic-wrap-padded buffer — a SYMMETRIC `HALO=4` margin on all four
sides, derived by reading `kernel2d_star2d3r`'s/`kernel2d_box2d3r`'s
`store_matrix_sync` index arithmetic and then confirmed by the shift-probe
matching cross-correlation exactly).

## Timesteps and the device-memory leak (real artifact bugs, found while wrapping)

- `gpu_star_2d3r`'s own internal `times`-argument loop ping-pongs device
  buffers WITHOUT ever refreshing the periodic-wrap halo between sweeps
  (identical class of finding as convstencil's `gpu_box_2d1r` and
  flashfftstencil's `rfft_2d_8_nwarp`) — confirmed by reading the loop
  directly (no host round-trip between launches). `bridge.cu` therefore
  always calls with `times=1`; `adapter.py::run()` performs its own
  T-sweep loop, re-padding the periodic-wrap halo from the CURRENT field
  before each single-sweep call (same discipline as convstencil's
  adapter).
- `gpu_box_2d3r`/`gpu_star_2d3r`/`gpu_star_2d1r` never `cudaFree()` their
  `array_d[0]`/`array_d[1]` device buffers (confirmed: zero `cudaFree`
  calls anywhere in `gpu.cu`). Each `run()` timestep therefore leaks two
  small device buffers — acceptable for this integration's bounded,
  small-grid login-node gate (a handful of calls total) but not patched
  (artifact driver code, not a build-system fix, per rule 3).

## Shape alignment constraint (inherited, not introduced)

The kernel grid is `ceil(m/32) x ceil(n/64)` blocks
(`BLOCK_SIZE_ROW=32, BLOCK_SIZE_COL=64`) with no tail/boundary guard in the
shared-memory load loop, so `m` must be a multiple of 32 and `n` a multiple
of 64 or a block reads/writes past the padded allocation. `prepare()`
raises `NotImplementedError` otherwise (neither spec.yaml grid size, 16384
or 10240, satisfies this cleanly at every candidate margin either, so the
gate below uses a small aligned synthetic grid, same posture as the
flashfftstencil sibling's own `/6`-divisibility constraint).

## Gate verification (login node, functional check only)

`star2d3r` is not one of `stencil.py`'s `--smoke` shapes (all three smoke
entries are `star2d1r`/`star3d1r`/`box3d1r`), so — per the flashfftstencil
sibling's own precedent for a shape not covered by `--smoke` — this gate
calls `kernelbench.harness.run_variant()` directly against the same
`stencil-cpu-gpu-kernel-fp64` variant object the runner CLI would load, on
a small 32/64-aligned synthetic `star2d3r` workload (`128x128`, `T=3`,
`precision="fp64"`, `warmup=1, reps=3`):

```
valid: True
correctness: CorrectnessResult(passed=True, metric='max_scaled_err',
  value=2.919701162416632e-16, tolerance=1e-05,
  reference='numpy fp64 periodic-wrap sweep', note='',
  max_pointwise_rel_err=7.530340100412872e-11,
  max_abs_err=1.3877787807814457e-16, l2_rel_err=3.0959415427262287e-16)
warnings: ['protocol overridden for this run (warmup=1, reps=3); result is
  NOT spec-conforming and must not be published as such']
workload after prepare (mutation check): timesteps= 3   # T=3 preserved,
  no timesteps=1 mutation needed (unlike flashfftstencil/spider) since
  this adapter implements its own genuine T-sweep loop
```

`$PY -m kernelbench.runner --kernel stencil --list` confirms:
`ok  lorastencil      lorastencil-star2d3r` under "paper artifacts"
(`available()==True`).

`max_scaled_err = 2.92e-16` against tolerance `1e-5` — **PASS**, by nine
orders of magnitude of margin, consistent with the independent probe's
3.33e-16 result above (same order of magnitude, different random field and
T=3 vs. T=1 — confirms the T-sweep loop itself, not just a single sweep, is
correct). The `warmup=1/reps=3` override is expected and documented — a
login-node correctness check on a shared GPU, not a timing claim.

## Not done

- 1D (`source/src/1d`) and 3D (`source/src/3d`) were not reverse-engineered
  in this integration pass's time budget — same posture as the convstencil
  sibling's own 1D/3D gap.
- box2d1r/box2d3r/star2d1r are not wrapped — see above; this is a verified
  artifact limitation, not a time-budget shortcut.
- No sweep across grid sizes/T beyond the one gate point above — out of the
  login-node budget (ARTIFACT_GUIDE.md rule 5, no timing sweeps).
- The device-memory leak (see above) was not patched; a long-running/
  large-T use of this adapter would need it fixed first.

## Verdict

`lorastencil: BUILT+GATED err=2.92e-16 (tol 1e-5, variant
stencil-cpu-gpu-kernel-fp64, workload star2d3r 128x128 T=3) | SKIP
box2d1r/box2d3r (artifact's own factorization divides 0/0 for any
non-onion-ring weight pattern, verified) | SKIP star2d1r (kernel ignores
its params argument, verified) | BUILD 1d/3d not wired up (time budget)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, parent card reported by
  `nvidia-smi` as "NVIDIA A100-SXM4-40GB, 8.0" on `gpu-b11-6`), login-node
  build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8, `$KB_CUDA_HOME`), host
  compiler: g++ 12.4.0 (conda-forge `kb-gcc12` env, via `-ccbin`, added —
  see below), Python 3.12.14 (`kb-env`). Same source commit
  (`1cdb7e25e09d48c0e0911513bff9165a6f8b8ef8`) per `source.provenance`.
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
- Gate: `--smoke`/`--matrices` still not usable here for the same reasons
  recorded originally (star2d3r not a `--smoke` shape; neither spec.yaml
  grid size satisfies the artifact's 32/64 block-alignment constraint at a
  clean margin); wrote `_gate_check.py` (this directory, new file)
  reproducing the exact standalone `harness.run_variant()` call STATUS.md's
  "Gate verification" section shows inline (same variant object, same
  synthetic star2d3r 128x128, T=3, seed=42,
  `warmup_override=1, reps_override=3`). Result:
  `max_scaled_err = 2.919701162416632e-16` vs tolerance `1e-05` —
  **PASS**, bit-for-bit identical to the recorded value.
  `$PY -m kernelbench.runner --kernel stencil --list` confirms `ok
  lorastencil lorastencil-star2d3r`.
- Deviation from the recorded ruling: none — identical correctness number.
- Verdict here: BUILT+GATED (star2d3r only; err 2.92e-16, tol 1e-5, workload
  star2d3r 128x128 T=3) — equals the recorded ruling. box2d1r/box2d3r/
  star2d1r SKIP and 1d/3d not-wired-up status unchanged (not re-verified in
  this pass, since the recorded evidence is about the artifact's own kernel
  code, not the build environment).
