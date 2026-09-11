#!/usr/bin/env bash
# Build efg's Elias-Fano-compressed-graph BFS kernel as a shared library
# (efg_shim.so), compiled from efg_shim.cu (this dir, NOT part of the
# artifact) directly against the artifact's own source/src headers plus a
# narrow, hand-vendored slice of folly (folly_vendor/, see below) -- NOT
# the artifact's own Makefile (which needs Facebook folly built+installed,
# boost, glog, double-conversion, fmt, openssl: none available on this
# machine and none installable without sudo/system packages, per
# ARTIFACT_GUIDE.md).
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source/src"
VENDOR="$HERE/folly_vendor"

CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"

echo "=== efg build (efg_shim.so, sm_80) ==="
echo "commit: $(git -C "$HERE/source" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"
echo "folly_vendor: $(find "$VENDOR" -type f | wc -l) files (see STATUS.md for provenance/scope)"

nvcc -std=c++17 -O3 -arch=sm_80 \
    -ccbin "$CXX" \
    --expt-relaxed-constexpr \
    -Xcompiler=-fopenmp,-fPIC,-Wno-unused-function \
    -I "$SRC" \
    -I "$VENDOR" \
    -shared -o "$HERE/efg_shim.so" \
    "$HERE/efg_shim.cu" \
    "$SRC/csr.cpp" \
    "$HERE/efg_util_shim.cpp" \
    -lgomp

if [[ ! -f "$HERE/efg_shim.so" ]]; then
    echo "BUILD FAILED: efg_shim.so not produced" >&2
    exit 1
fi
echo "Built: $HERE/efg_shim.so"
echo "=== done ==="
