#!/usr/bin/env bash
# Build TileSpMV's tiling format construction + CUDA kernels
# (source/src/{csr2tile.h,tilespmv_cpu.h,tilespmv_cuda.h,external/CSR5_cuda})
# into a shared library via bridge.cu (this directory; see its docstring for
# why a bridge is needed instead of calling the artifact's own main.cu
# driver). Target: A100 / sm_80 (the artifact's own Makefile targets
# compute_61/sm_86; overridden here per this machine's GPU).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source/src"
NVCC="${NVCC:-nvcc}"

echo "=== TileSpMV build (bridge.so, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

# -Xcompiler -fopenmp: csr2tile.h's convert_step*/tilespmv_cpu.h use OpenMP
# pragmas (matches the artifact's own Makefile OPTIONS).
#
# -I compat/: the artifact's own Makefile points -I at a
# NVIDIA_CUDA_Samples checkout (helper_functions.h/helper_cuda.h) this
# machine doesn't have; source/src/external/CSR5_cuda/detail/cuda/
# common_cuda.h includes both but only ever uses one macro from them
# (checkCudaErrors). compat/ (this directory, NOT part of the artifact)
# supplies minimal stand-ins -- see compat/helper_cuda.h for exactly what
# was grepped-for and why. Placed first on the include path so it's found
# before anything else could shadow it.
"$NVCC" -O3 -w -arch=sm_80 -Xcompiler -fopenmp -Xcompiler -fPIC \
    -shared \
    -I "$HERE/compat" \
    -I "$SRC" \
    "$HERE/bridge.cu" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
