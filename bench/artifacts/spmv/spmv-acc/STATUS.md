# spmv-acc — STATUS

**Outcome: BUILT+GATED — a compatible HIP-on-CUDA toolchain was found on
zaratan; the artifact's own KERNEL_STRATEGY_ADAPTIVE kernel library builds
and gates clean (smoke 3/3, both real SuiteSparse matrices tried PASS at
fp64 unit-roundoff). See "Reproduction on zaratan" below.**

Earlier outcome, kept for the record: **BUILD-FAILED — genuine CUDA path
exists in the repo, but no CUDA toolkit available on this machine is
compatible with this machine's only HIP toolchain.** (true on Perlmutter,
the original reference machine; not true on zaratan, see below.)

Paper: "Efficient Algorithm Design of Optimizing SpMV on GPU", HPDC 2023.
PAPER_KEY = `conf/hpdc/ChuHDDC0WH23`. Selected as a **core baseline under
the revised kernel-centrality rule**
(`output/kernel_centrality.json` key `spmv|conf/hpdc/ChuHDDC0WH23`:
centrality `core`, regime `matches`, `gpu_single_card: true` — "two new
general-purpose GPU SpMV algorithms (flat/line-enhance) with adaptive
selection, evaluated directly against existing GPU SpMV approaches").
Repo: `https://github.com/hpcde/spmv-acc`
(commit `b4e622a047a09db30bc4a8c1db8e0afb612c3bcf`, 2025-05-18;
`git clone --depth 50` into `./source/`).

## Platform check (per the integration brief: "this group often targets
HIP/ROCm; check CUDA vs HIP-only first")

The repo's README title is "HIP acceleration for SpMV solver" and its
build docs lead with ROCm (`module load compiler/rocm/3.9.1`,
`rocsparse`), but this is **not** a HIP/ROCm-only artifact:
`CMakePresets.json` ships a genuine `cuda-hipcc`/`cuda-hipcc-benchmark`
configure preset (`use-nv-hipcc-wrapper` + `use-platform-cuda`, setting
`HIP_NVCC_FLAGS: -arch=sm_89 ...`), which compiles the SAME kernel sources
via `hipcc` with `HIP_PLATFORM=nvidia` (i.e. HIP-on-CUDA, hipcc dispatching
to `nvcc` under the hood). Grepping the entire kernel-library tree
(`src/acc/`) for AMD-only dependencies (`rocblas`/`rocsparse`/`rocthrust`/
`rocprim`/`hipcub`): **zero hits** — the kernel strategies are
self-contained HIP C++ (`hipMalloc`/`hipMemset`/`<<<>>>` launches), and the
one place genuinely AMD-specific code exists
(`src/acc/common/platforms/rocm/{dpp_reduce.h,rocm_global_mem_ops.hpp}`,
AMD DPP-instruction wavefront reduction) is correctly guarded behind
`#ifdef __HIP_PLATFORM_HCC__` everywhere it's referenced
(`common/global_mem_ops.h`, `common/cross_lane_ops.h`) — HIP's own
NVIDIA-backend headers never define that macro, so these files compile out
cleanly on CUDA. **Verdict: a real, general single-NVIDIA-GPU path exists
in this artifact** — this is NOT the "no CUDA path" SKIP case; per the
integration brief the right action is to integrate.

## What was going to be wrapped

The artifact's library entry point is clean and library-shaped (unlike
TileSpMV/CB-SpMV's monolithic drivers): `sparse_csr_spmv(trans, alpha,
beta, h_csr_desc, d_csr_desc, dx, dy)` (`src/acc/api/spmv.h`), dispatching
at compile time (via `strategy_picker.cpp`'s `#ifdef KERNEL_STRATEGY_*`) to
one of 10 kernel strategies. The plan was `KERNEL_STRATEGY_ADAPTIVE`
(`src/acc/hip-adaptive/adaptive.cpp`) — the paper's headline
auto-selecting strategy, choosing between its own `flat`/`line`/
`line-enhance`/`thread-row`/`vector-row` kernels based on the matrix's
average nnz/row and a 4-way nnz-imbalance check across quartile row
boundaries (all computable from the CSR host descriptor, i.e. genuinely
one-time preprocessing → `prepare()`). This would have been a much
smaller bridge than TileSpMV/CB-SpMV's: compile `src/acc/{api,hip-adaptive,
hip-flat,hip-line,hip-line-enhance,hip-thread-row,hip-vector-row}/*.cpp`
plus `strategy_picker.cpp` directly with `hipcc` (bypassing the project's
full CMake+`find_package(HIP)`+`clipp` build, none of which the library
target itself needs), with a hand-written `compat/building_config.h`
standing in for the CMake-`configure_file`-generated header (this
directory's `compat/building_config.h` — `KERNEL_STRATEGY_ADAPTIVE`,
`__WF_SIZE__=32` for the actual NVIDIA warp size, `AVAILABLE_CU=108` for
this A100's actual SM count vs. the project's AMD-oriented default of 60).

## Why it's BUILD-FAILED: a HIP-toolchain/CUDA-header version wall, not an artifact bug

This machine's only HIP module (`module spider hip` → `hip/5.5.1` only;
confirmed via a bounded `module -t avail hip`/`rocm` query, no filesystem
search) sets `HIP_PLATFORM=nvidia`, `HIP_COMPILER=nvcc`, and dispatches
`hipcc` to whichever `nvcc` is on `CUDA_PATH`. Compiling
`src/acc/hip-adaptive/adaptive.cpp` (and every other strategy source
tried: `flat.cpp`, `line_strategy.cpp`, `line_enhance_spmv.cpp`,
`thread_row.cpp`, `native_thread_row.cpp`, `vector_row.cpp`,
`strategy_picker.cpp`) against **every CUDA toolkit available as a module
on this machine** fails, for two independent reasons depending on which
CUDA:

1. **CUDA 12.9** (this repo's usual pin, `bench/artifacts/toolchain.sh`):
   HIP 5.5.1's own vendored shim header,
   `$HIP_PATH/include/hip/nvidia_detail/nvidia_hip_runtime_api.h`, calls
   CUDA Graph APIs with a signature CUDA 12.x changed
   (`cudaGraphNodeGetDependentNodes` — "too few arguments in function
   call"; `cuStreamGetCaptureInfo_v2` — "identifier is undefined") and
   reads `cudaDeviceProp` fields CUDA 12.x removed (`clockRate`,
   `computeMode`, `memoryClockRate`, `kernelExecTimeoutEnabled`,
   `cooperativeMultiDeviceLaunch`). ~28-30 errors per file, ALL inside
   this one HIP-vendored header, none inside any spmv-acc source file
   (confirmed: the error transcript for every strategy file tried is
   dominated by the identical `nvidia_hip_runtime_api.h` line numbers).
2. **CUDA 11.7** (`cudatoolkit/11.7` — the only other CUDA module on this
   machine, and the version HIP 5.5.1 was actually built against, avoiding
   error class #1 above): its `nvcc`'s bundled/expected host-compiler
   support does not recognize this machine's system glibc/GCC 14
   (`identifier "_Float128" is undefined`, in `/usr/include/bits/
   mathcalls.h` and friends, pulled in transitively by `<cuda_runtime.h>`)
   — fails even with an older module compiler explicitly forced via
   `nvcc -ccbin /opt/cray/pe/gcc/10.3.0/bin/g++` (tried; still hits BOTH
   the glibc mismatch in some translation units and, once past that,
   the SAME `cudaDeviceProp`/CUDA-Graph-API mismatches as #1, since the
   forced `-ccbin` did not change which CUDA headers were resolved).

Both failure modes are inside **HIP's own vendored NVIDIA-backend shim
header**, not in any file under `source/`. There is no build-system knob
left to try (arch flags, include paths, and an explicit alternate
`-ccbin` were all exhausted) that doesn't amount to patching a
system-installed HIP module's headers, which is out of scope (not the
artifact's code, and not this integration's file to patch).

This is **not** the new GPU-architecture DEFERRED-HARDWARE case
(ARTIFACT_GUIDE.md rule 9): the blocker here is a HIP-toolchain/CUDA-header
version skew on this specific machine, not a missing Hopper/sm_90a
feature — this A100 (sm_80) is architecturally exactly what the artifact
targets (`HIP_NVCC_FLAGS: -arch=sm_89` in `cuda-hipcc-benchmark`, easily
lowered to sm_80, was never even reached). A machine with a HIP install
built against (or compatible with) CUDA 12.x would very likely build this
cleanly — `build.sh` is left in place, reproducing the exact failure, for
exactly that retry.

## Files in this directory

- `source/` — the clone (git-ignored repo-wide), `source.provenance`.
- `compat/building_config.h` — the hand-written stand-in for the
  CMake-generated config header (see above); never successfully used since
  the underlying HIP-header compile fails first, but left in place as it's
  still correct for a future retry.
- `build.sh` — reproduces the exact failure (module-equivalent env vars,
  both CUDA toolkits tried in sequence); exits 1 with the evidence above.
- `adapter.py` — `available()` always returns `(False, "BUILD-FAILED: ...")`
  since `bridge.so` is never produced; shows up correctly in
  `runner.py --list` as `-- spmv-acc ... BUILD-FAILED: ...` rather than a
  silent absence.

## Verdict

`spmv-acc: BUILD-FAILED (this machine) — genuine single-NVIDIA-GPU path
confirmed in the repo (cuda-hipcc preset, zero AMD-only deps in the kernel
library); blocked by this machine's only HIP module (5.5.1) being
compile-incompatible with every CUDA toolkit available here. No kernel
code reached; no timing or correctness number obtained.`

(Superseded on zaratan -- see "Reproduction on zaratan" below: the same
blocker class does not exist there, a compatible module was found, and
the artifact now BUILDS and GATES clean.)

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- **The toolchain-version wall above is machine-specific, not universal.**
  `module spider hip` on zaratan lists `hip/5.6.1/gcc/11.3.0/nompi/cuda/
  12.3.0/zen2` -- a HIP install built together with, and paired to, its own
  CUDA 12.3.0 (not this repo's usual CUDA 12.8/12.9 pin). Compiling a
  representative source file (`hip-adaptive/adaptive.cpp`) against that
  paired CUDA 12.3 succeeded outright (`hipcc -std=c++14 ... -c
  adaptive.cpp`, exit 0, real `.o` produced) -- none of the CUDA-Graph-API/
  cudaDeviceProp header mismatches hit against CUDA 12.9 on Perlmutter with
  the older HIP 5.5.1 apply here, because this HIP/CUDA pairing was built
  and tested together upstream. `module load` does not work inside a
  plain bash script (same reason `bench/artifacts/toolchain.sh` exists), so
  `build.sh` now probes for this module's install-prefix paths directly
  (`KB_SPMVACC_HIP_PATH`/`KB_SPMVACC_CUDA_PATH`/`KB_SPMVACC_GXX` override;
  the old Perlmutter path is kept as a fallback candidate, so a re-run
  there still reproduces the original BUILD-FAILED unchanged -- rule 8).
- Toolchain used: HIP 5.6.31062 (`hip/5.6.1/gcc/11.3.0/nompi/cuda/12.3.0/
  zen2`), its paired nvcc 12.3.52, its paired g++ 11.3.0 (via `-ccbin`,
  the host compiler this HIP module expects, not this repo's usual
  conda-forge gcc 13 or gcc 12); `-std=c++14 -O3 -fPIC -shared` (no
  explicit `-arch=` flag -- hipcc/nvcc's own default for this pairing
  targeted this A100 correctly, confirmed by the gate below actually
  running kernels on it).
- What was built, per the "What was going to be wrapped" plan above
  (now realized): `bridge.cpp` (this directory, NOT part of the artifact)
  -- a ctypes-callable `spmvacc_prepare/run/copy_y/free` wrapper around the
  artifact's own `sparse_csr_spmv()` (`api/spmv.h`), compiled together with
  the artifact's own unmodified `api/spmv_imp.cpp`, `strategy_picker.cpp`
  (dispatches to `adaptive_sparse_spmv` under `KERNEL_STRATEGY_ADAPTIVE`,
  `compat/building_config.h`), `hip-adaptive/adaptive.cpp`, and every
  strategy it calls into (confirmed by reading `adaptive.cpp`'s call
  sites): `hip-flat/flat.cpp`, `hip-line/line_strategy.cpp`,
  `hip-line-enhance/line_enhance_spmv.cpp`, `hip-vector-row/vector_row.cpp`
  (`hip-thread-row/{thread_row,native_thread_row}.cpp` also compiled in per
  the original plan, though `adaptive.cpp`'s own call into thread-row is
  commented out). No kernel/strategy source under `source/` is modified.
  `bridge.so` is linked with `-rpath $CUDA_PATH/lib64` so
  `libcudart.so.12` (this HIP module's CUDA 12.3 build) resolves regardless
  of the harness Python process's own `LD_LIBRARY_PATH` (which points at
  this repo's usual CUDA 12.8 conda toolkit); `libcuda.so.1` (the driver's
  own runtime lib) resolves from the GPU node's default system path, same
  as every other CUDA bridge in this repo -- confirmed working, not just
  assumed (the gate below ran real kernels on the A100).
- Build: OK (`bridge.so` produced, ~440 KB). Build-system changes:
  `build.sh` rewritten (was a reproduce-the-failure script; now builds a
  real `bridge.so` when a compatible HIP module is found, unchanged in
  spirit -- still no `module load` in the script itself, rule 8's
  probe-a-candidate-list idiom instead); `adapter.py` rewritten from the
  always-`available()=False` stub to a real ctypes-backed adapter (same
  prepare/run/to_host/free shape as every other bridge in this repo).
  New file: `bridge.cpp` (the wrapper described above).
- Gate: `--smoke`: 3/3 valid -- smoke-uniform (err 0.00e+00), smoke-banded
  (err 0.00e+00), smoke-powerlaw (err 4.23e-16), all <= 1e-9. Mandated
  command, both matrices: `--matrices webbase-1M --warmup 1 --reps 3`:
  PASS, err 7.29e-16 <= 1e-9. `--matrices cant --warmup 1 --reps 3`: PASS,
  err 6.07e-16 <= 1e-9. Clean sweep, fp64 unit-roundoff on every workload
  tried -- no correctness findings for this artifact.
- Deviation from the recorded ruling: **the ruling changes** (BUILD-FAILED
  -> BUILT+GATED), but not because anything about the artifact or the
  earlier evidence was wrong -- the earlier BUILD-FAILED evidence (HIP
  5.5.1 vs CUDA 12.9/11.7 on Perlmutter) is still accurate for that
  specific toolchain pairing; zaratan simply has a different, compatible
  HIP module available.
- Verdict here: BUILT+GATED -- err 0.00e+00-7.29e-16 (tol 1e-9), fp64,
  clean pass on smoke + both real SuiteSparse matrices tried.
