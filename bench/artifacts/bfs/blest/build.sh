#!/usr/bin/env bash
# Build BLEST's tensor-core BFS kernel as a shared library (blest_shim.so),
# compiled from blest_shim.cu (this dir, NOT part of the artifact) directly
# against the artifact's own header-only-ish sources (source/BFS,
# source/DataStructures, source/Gorder, source/Common.cuh) -- everything
# EXCEPT main.cu/Benchmark.cuh/SuiteSparseMatrixDownloader.hpp (libcurl-only
# CLI plumbing this shim bypasses entirely, see blest_shim.cu) and
# MBFS/Closeness/CC kernels (not the bfs track).
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

echo "=== BLEST build (blest_shim.so, sm_80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

# README states CUDA >= 13.0 as a hard requirement; this machine has 12.9.
# The only CUDA-13-specific-sounding feature actually used by the BFS path
# (source/BFS/*.cuh) is the inline `mma.sync...b1.b1...and.popc` binary
# Tensor Core PTX instruction, which has been valid since sm_75/CUDA 10.2 --
# empirically it assembles fine under 12.9 for sm_80 (see STATUS.md); the
# README's stated minimum is evidently the authors' own dev/test toolchain,
# not a hard technical floor for this code path.
nvcc -std=c++20 -O3 -arch=sm_80 \
    -ccbin "$CXX" \
    -Xcompiler=-fopenmp,-fPIC \
    -I "$SRC" \
    -I "$SRC/DataStructures" \
    -I "$SRC/BFS" \
    -I "$SRC/Gorder" \
    -shared -o "$HERE/blest_shim.so" \
    "$HERE/blest_shim.cu" \
    "$SRC/Gorder/Graph.cpp" \
    "$SRC/Gorder/Util.cpp" \
    "$SRC/Gorder/UnitHeap.cpp" \
    -lgomp

if [[ ! -f "$HERE/blest_shim.so" ]]; then
    echo "BUILD FAILED: blest_shim.so not produced" >&2
    exit 1
fi
echo "Built: $HERE/blest_shim.so"
echo "=== done ==="
