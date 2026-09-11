#!/usr/bin/env bash
# Build cuSZp's own shared library directly with nvcc (bypassing its CMake
# build, which also builds static libs/tests/Python bindings this
# integration doesn't need). Every source file compiled here is exactly
# the list CMakeLists.txt's `cuSZp_SOURCES` names -- nothing added,
# nothing removed, no kernel code touched.
# Target: A100 / sm_80 (cuSZp's own CMakeLists defaults to a multi-arch
# fatbin list 60;61;62;70;75;80;86 -- narrowed to sm_80 here, this
# machine's GPU, for a faster build; same category of arch-flag choice
# as every other GPU artifact in this benchmark).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
NVCC="${NVCC:-nvcc}"

echo "=== cuSZp build (libcuszp.so, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

SOURCES=(
  cuSZp_kernels_1D_f32.cu cuSZp_kernels_1D_f64.cu
  cuSZp_kernels_2D_f32.cu cuSZp_kernels_2D_f64.cu
  cuSZp_kernels_3D_f32.cu cuSZp_kernels_3D_f64.cu
  cuSZp_utility.cu cuSZp_timer.cu
  cuSZp_entry_1D_f32.cu cuSZp_entry_1D_f64.cu
  cuSZp_entry_2D_f32.cu cuSZp_entry_2D_f64.cu
  cuSZp_entry_3D_f32.cu cuSZp_entry_3D_f64.cu
  cuSZp.cu
)
SRCPATHS=()
for f in "${SOURCES[@]}"; do SRCPATHS+=("$SRC/src/$f"); done

# -I include        : cuSZp.cu's own `#include "cuSZp.h"`
# -I include/cuSZp   : every entry/kernel .cu's `#include "cuSZp_entry_*.h"`
#                       etc (no "cuSZp/" prefix in those quoted includes --
#                       mirrors CMakeLists.txt's own two include dirs)
"$NVCC" -O3 -arch=sm_80 -std=c++17 -Xcompiler -fPIC --shared \
    -I "$SRC/include" -I "$SRC/include/cuSZp" -I "$SRC/src" \
    "${SRCPATHS[@]}" \
    -o "$HERE/libcuszp.so"

echo "Built: $HERE/libcuszp.so"
echo "=== done ==="
