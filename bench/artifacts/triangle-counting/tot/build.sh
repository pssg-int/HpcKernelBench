#!/usr/bin/env bash
# Build ToT's tensor-core triangle-counting kernel as a shared library
# (tot_shim.so), compiled from tot_shim.cu (this dir, NOT part of the
# artifact) directly against the artifact's header-only tot/ library
# (source/tot/) -- same compile flags as source/tot/CMakeLists.txt's
# `totlib` INTERFACE target (C++20, --extended-lambda,
# --expt-relaxed-constexpr, -Xcompiler=-fopenmp, sm_80).
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"

CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"

echo "=== ToT build (tot_shim.so, sm_80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

nvcc -std=c++20 -O3 -arch=sm_80 \
    -ccbin "$CXX" \
    --extended-lambda --expt-relaxed-constexpr \
    -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP \
    -Xcompiler=-fopenmp,-fPIC \
    -I "$SRC/tot" \
    -shared -o "$HERE/tot_shim.so" \
    "$HERE/tot_shim.cu" \
    -lgomp

if [[ ! -f "$HERE/tot_shim.so" ]]; then
    echo "BUILD FAILED: tot_shim.so not produced" >&2
    exit 1
fi
echo "Built: $HERE/tot_shim.so"
echo "=== done ==="
