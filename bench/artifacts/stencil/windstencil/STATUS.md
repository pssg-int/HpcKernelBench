# WindStencil — STATUS

**Outcome: SKIPPED — AMD ROCm/HIP only, no NVIDIA build path on this machine**

Paper: "WindStencil: Unleashing GPU Potential for High-Order Stencil
Computation in High-Performance Inviscid CFD Simulations", ICS 2026.
PAPER_KEY = `conf/ics/ZhangZLLJZYLL26`.
Repo: `https://github.com/BabyXPrince/WindStencil`
(commit `c994bdb13cb7d9cf09cc7bd117d00cac97b6b163`, 2026-04-04;
`git clone --depth 1` into `./source/`).

This was flagged as "GPU SINGLE-CARD STENCIL" and `benchmark_groups.json`
tags its platform as `nvidia-gpu`, but that tag is wrong at the source
level — verified directly rather than trusted:

## Evidence

```
$ head -1 source/src/kernels/OCFD_split.cpp
#include "hip/hip_runtime.h"

$ grep -rl "hip/hip_runtime.h" source/src/kernels/*.cpp source/src/kernels/*.c | wc -l
6   # every kernel file in src/kernels/ includes the HIP runtime header
```

The README states the target explicitly: "The implementation targets
**AMD GPUs via HIP/ROCm**... Requirements: AMD ROCm with `hipcc`... GPU
architecture flag compatible with your hardware (e.g. `gfx906` for MI60
class, `gfx90a` for MI200 class)." `src/app/opencfd.c` is documented in
the repo's own table as "CUDA-style source for hipify" — i.e. an input
artifact for the hipify *source-generation* tool, not a buildable CUDA
target itself; the actually-built entry point is `opencfd_hip.c`, and
every kernel file under `src/kernels/` (`OCFD_split.cpp`,
`OCFD_Schemes.cpp`, `commen_kernel.cpp`, `cuda_commen.c`,
`cuda_utility.c`) includes `hip/hip_runtime.h` and uses HIP launch syntax,
not the CUDA runtime.

```
$ command -v hipcc; command -v hipify-clang; command -v hipify-perl
(all three: not found)
```

No ROCm/HIP toolchain is installed on this machine (NVIDIA A100, nvcc
12.9 only), and per ARTIFACT_GUIDE.md there is no sudo / no system package
installs available to add one. HIP-on-NVIDIA (compiling HIP source against
the CUDA backend via `hipcc --amd=false` / `HIP_PLATFORM=nvidia`) is a real
option in general, but it requires the ROCm HIP headers/toolchain to be
present at all — no `hip/hip_runtime.h` is available on this system,
NVIDIA-backend or otherwise, and installing ROCm is outside the "no sudo,
no system package installs" build-environment constraint.

## Scope ruling

Per ARTIFACT_GUIDE.md's 2026-08-07 scope decision ("Integration targets
NVIDIA GPU, single-card implementations only for now... [other-platform]
artifacts are SKIPPED with a one-line reason — cheap skip, no build
attempt"), an AMD-GPU-only artifact is out of scope in the same way a
CPU-only or FPGA-only one would be. No build was attempted (correctly
cheap per the rule); `source/` is kept for provenance per the layout
contract, `build.sh`/`adapter.py` are intentionally not created.

## Verdict

`windstencil: SKIPPED (AMD ROCm/HIP only — kernels #include hip/hip_runtime.h; no ROCm toolchain on this NVIDIA-only A100 machine; out of scope per NVIDIA-single-card ruling)`
