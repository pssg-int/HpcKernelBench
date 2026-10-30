#!/usr/bin/env bash
# Build the Gunrock BFS baseline as a shared library (gunrock_shim.so):
# gunrock_shim.cu (this dir) compiled straight against Gunrock v2.1.0's
# header-only include/ tree plus moderngpu's headers (source_moderngpu/), both
# cloned by toolchain.sh from the *.provenance files. Thrust/CUB come from the
# CUDA toolkit. Flags are the ones Gunrock's own CMake sets for its essentials
# target (C++17, --expt-extended-lambda, --expt-relaxed-constexpr,
# --use_fast_math, SM_TARGET). No CMake / FetchContent needed.
#
# Idempotent. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
MGPU="$HERE/source_moderngpu"
ARCH="${KB_CUDA_ARCH:-80}"
CXX="${CXX:-g++}"

echo "=== Gunrock BFS baseline build (gunrock_shim.so, sm_$ARCH) ==="
echo "gunrock commit:   $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "moderngpu commit: $(git -C "$MGPU" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

nvcc -std=c++17 -O3 -arch="sm_$ARCH" -ccbin "$CXX" \
    --expt-extended-lambda --expt-relaxed-constexpr --use_fast_math \
    -DSM_TARGET="$ARCH" -DESSENTIALS_COLLECT_METRICS=0 \
    -Xcompiler=-fPIC \
    -I "$SRC/include" -I "$MGPU/src" \
    -shared -o "$HERE/gunrock_shim.so" \
    "$HERE/gunrock_shim.cu" "$SRC/include/gunrock/io/detail/mmio.cpp" \
    -lcurand -lcusparse -lcuda

[ -f "$HERE/gunrock_shim.so" ] || { echo "STATUS: BUILD-FAILED -- gunrock_shim.so not produced"; exit 1; }
echo "Built: $HERE/gunrock_shim.so"
