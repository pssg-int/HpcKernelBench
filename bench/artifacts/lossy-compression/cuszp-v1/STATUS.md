# cuSZp v1 (SC'23) — STATUS

**Outcome: BUILT+GATED — gate PASSES on all 3 smoke workloads**

Paper: "cuSZp: An Ultra-fast GPU Error-bounded Lossy Compression Framework
with Optimized End-to-End Data Movement", SC 2023.
PAPER_KEY = `conf/sc/HuangD0LC23`.
Repo: `https://github.com/szcompressor/cuSZp`, tag `cuSZp-V1.1`
(commit `f4df2f1c5e9e529b05d344f2491a3fa7a5c2c0ed`, 2024-10-24).

## Why this directory exists (relationship to ../cuszp/)

`../cuszp/` wraps the SAME repository's HEAD (no tag pinned at clone time),
which its own STATUS.md's "PAPER_KEY disambiguation" section carefully
worked out is actually the **SC'25** ("VGC"/cuSZp3) code — the dim x mode
versatility matrix (`cuszp_dim_t` 1D/2D/3D x `cuszp_mode_t`
fixed/plain/outlier) already present at that commit. That leaves the SC'23
kernel this benchmark's spec survey originally cited (`benchspecs/
lossy-compression/spec.yaml`'s evidence entry for `conf/sc/HuangD0LC23`)
unwrapped. Checked directly: `cuSZp-V1.1` is a **genuinely distinct code
path**, not just an earlier version of the same API —

- different source files: `src/cuSZp_f32.cu` (a single fused
  compress/decompress kernel pair), no `cuSZp_kernels_*D_f32.cu` split at
  all;
- a narrower entry signature with NO `dim`/`dims`/`mode` parameters:
  `void SZp_compress_deviceptr_f32(float* d_oriData, unsigned char*
  d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound,
  cudaStream_t stream=0)` (`include/cuSZp_entry_f32.h` at this tag) versus
  SC'25's `cuSZp_compress(..., cuszp_dim_t dim, uint3 dims, cuszp_type_t
  type, cuszp_mode_t mode, ...)`.

This confirms the case ARTIFACT_GUIDE's task description anticipated ("the
SC'23 kernel ... is a distinct code path in the same repo") — so this is a
SEPARATE artifact directory with its own build/adapter/gate, not a fix to
`../cuszp/`'s attribution. (`../cuszp/STATUS.md` has been given a one-line
cross-reference to here so both keys are discoverable from either
directory.)

## `source/`: a git worktree, not a second clone or a symlink

`../cuszp/source/` was cloned with `git clone --depth 50` (see its
`source.provenance`), and `cuSZp-V1.1`'s commit (`f4df2f1c...`) turned out
to already be inside that depth-50 history — confirmed with `git rev-parse
cuSZp-V1.1` / `git log -1 cuSZp-V1.1` inside `../cuszp/source`, both
resolving cleanly with no network access needed.

The task description's suggested mechanism was "symlink `source` to the
existing clone". A **plain symlink** was deliberately NOT used: `source/`
would then point at `../cuszp/source`'s own working directory, so checking
out `cuSZp-V1.1` there (`git checkout cuSZp-V1.1`) would move **that**
adapter's checked-out files too — silently invalidating `../cuszp/`'s
already-BUILT+GATED state (its `libcuszp.so` was compiled against the
SC'25 tree; the checked-out source going stale under it is exactly the
kind of shared-mutable-state bug ARTIFACT_GUIDE's patch-discipline rules
exist to prevent).

Instead: `git worktree add ../../cuszp-v1/source cuSZp-V1.1`, run from
inside `../cuszp/source`. This satisfies the actual intent (reuse the
already-fetched repository, no second network clone of a 5+ GB history) via
git's own linked-worktree mechanism: `cuszp-v1/source` is a real,
independent directory with its own HEAD/index, sharing the same `.git`
object database as `../cuszp/source` (`git worktree list` shows both).
Checking out `cuSZp-V1.1` here has zero effect on `../cuszp/source`'s HEAD.
See `source.provenance` for the exact recipe and a plain-clone fallback.

## Why a bridge file was needed (no kernel code touched)

`nm -D` on the first build of `libcuszp_v1.so` showed the exported compress/
decompress symbols Itanium-mangled
(`_Z26SZp_compress_deviceptr_f32PfPhmPmfP11CUstream_st` etc.) — unlike
SC'25's `include/cuSZp.h`, v1's `include/cuSZp_entry_f32.h` has no `extern
"C"` guard, so nvcc compiles its declarations as ordinary C++ symbols.
ctypes cannot look these up by their plain names.

Fixed WITHOUT touching `source/`: `bridge_cuszp_v1.cu` (this directory)
`#include`s the unmodified header and re-exports two `extern "C"`
trampolines (`cuszpv1_compress_f32`/`cuszpv1_decompress_f32`) that do
nothing but forward the call. Same category of fix as GPULZ's
`bridge_gpulz.cu` (`../../lossless-compression/gpulz/`) — a linkage-only
shim, zero kernel-code changes, zero diff inside `source/` (`git -C source
diff` is empty). `build.sh` compiles this bridge file alongside v1's own
unmodified `.cu` sources.

## Adapter

`IMPL_NAME = "cuszp-v1-compress"`, `PRECISIONS = ["fp32"]` (v1 also ships
fp64 kernels, `SZp_*_f64`, not wrapped here — same posture as `../cuszp/`).
`direction = "compress"` per this domain's contract. Device buffers
allocated via torch (`ctypes.c_void_p(tensor.data_ptr())`), same pattern as
`../cuszp/adapter.py`.

- `prepare()`: H2D copy + allocation of the compressed-output buffer
  (`sizeof(float)*nbEle` bytes, the same worst-case bound v1's own
  `src/cuSZp_entry_f32.cu` hostptr wrapper allocates) and the
  decompressed-output buffer — timed once as preprocessing. Note: unlike
  the SC'25 successor, v1's `SZp_compress_deviceptr_f32` still
  `cudaMallocManaged`s/`cudaFree`s its own `cmpOffset`/`flag` scratch
  buffers **inside every call** (`src/cuSZp_entry_f32.cu`) — a real, if
  small, per-call allocation cost this is v1's own design (the SC'25
  paper's whole "optimized end-to-end data movement" pitch is partly about
  removing exactly this), left completely untouched per rule 3.
- `run()`: exactly one `cuszpv1_compress_f32()` call (v1's own single fused
  kernel, `SZp_compress_kernel_f32`, unmodified).
- `to_host()`/`free()`: decompress (v1's own kernel, unmodified) + D2H copy,
  deliberately outside the timed region — same discipline as `../cuszp/`.

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 -std=c++17 -Xcompiler -fPIC --shared -I source/include
-I source/src source/src/{cuSZp_f32,cuSZp_f64,cuSZp_utility,cuSZp_timer,
cuSZp_entry_f32,cuSZp_entry_f64}.cu bridge_cuszp_v1.cu -o libcuszp_v1.so`.
nvcc 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`, via `../../toolchain.sh`).
Built clean, no warnings requiring investigation. Idempotent.

## Gate verification (mandated command, run as specified)

```
$PY -m kernelbench.runner --kernel lossy-compression \
    --variant lossy-comp-kernel-cpu-ebound --impl cuszp-v1-compress \
    --smoke --warmup 1 --reps 3
```

```
  running cuszp-v1-compress smoke-smooth-3d     ... 2.873 ms  0.02 GB/s  (err 3.42e-03 <= 3.4191e-03)
  running cuszp-v1-compress smoke-turbulent-3d  ... 2.886 ms  0.02 GB/s  (err 1.29e-03 <= 1.2903e-03)
  running cuszp-v1-compress smoke-multiscale-3d ... 2.429 ms  0.02 GB/s  (err 3.81e-03 <= 3.8086e-03)

3/3 runs valid
conforming: False (smoke: synthetic matrices, reduced protocol, overridden warmup/reps, login-node shared GPU -- expected/flagged, not a defect)
```

`max_abs_err <= eb` on all 3. Achieved compression ratio ~4.08x on the
smooth-3d smoke field (`achieved_compressed_bytes=13544` for a 55296-element
fp32 field), PSNR ~64.7 dB — markedly lower ratio than SC'25's
outlier-mode 54.00x on the same smoke workload (`../cuszp/STATUS.md`), which
is exactly the paper-to-paper delta this benchmark exists to surface: v1's
"plain" single-mode encoding has no outlier-block optimization, so this
number is real evidence of the SC'23-vs-SC'25 improvement, not a bug in
either adapter. Full JSON at
`bench/results/lossy-compression_lossy-comp-kernel-cpu-ebound_1788656007.json`.

## Not done

- fp64 kernels (`SZp_*_f64`) not wrapped — same scope decision as
  `../cuszp/`.
- Real SDRBench-scale ratio not measured (smoke-only, login-node build+gate
  per ARTIFACT_GUIDE rule 5).

## Verdict

`cuszp-v1-compress: BUILT+GATED err<=eb ratio=4.08 (smoke; real SDRBench-scale ratio not measured here)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge), torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80` unchanged from the recorded build.
- Build: OK. Build-system changes: none (idempotent rebuild via `../../toolchain.sh`, no diff to
  `source/`; the git worktree at `source/` — sharing `../cuszp/source`'s object database, checked out
  at tag `cuSZp-V1.1` — was already present from an earlier pass and required no recreation).
- Gate: `lossy-comp-kernel-cpu-ebound` cuszp-v1-compress: PASS on all 3 smoke workloads
  (smooth/turbulent/multiscale), `max_abs_err <= eb` on all 3 (e.g. 3.42e-03 <= 3.4191e-03), 3/3 runs
  valid.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
