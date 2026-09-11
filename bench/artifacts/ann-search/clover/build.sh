#!/usr/bin/env bash
# Build CLOVER's exact-kNN "hubs" method as a shared library (clover_shim.so),
# compiled from clover_shim.cu (this dir, NOT part of the artifact) directly
# against the artifact's own header-only include/bitonic-hubs.cuh (which
# transitively pulls in cuda_util.cuh, spatial.cuh, bitonic-shared.cuh).
# Nothing else under source/ is touched (no CMake, no FAISS, no OpenMP --
# the -DUSE_FAISS baseline path in source/CMakeLists.txt is not needed for
# CLOVER's own "hubs" method, which this shim wraps).
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"

CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"

echo "=== CLOVER build (clover_shim.so, sm_80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

# source/CMakeLists.txt hardcodes -arch=sm_89 (Ada); this machine's GPU is an
# A100 (sm_80) -- a build-system arch-flag override, allowed per
# ARTIFACT_GUIDE.md rule 3. CLOVER's "hubs" kernel uses no sm_89-specific
# feature (no wgmma/TMA -- plain global/shared-memory CUDA C++), so this is
# a straightforward retarget, not a functional change.
nvcc -std=c++20 -O3 -arch=sm_80 \
    -ccbin "$CXX" \
    -Xcompiler=-fPIC \
    -I "$SRC/include" \
    -shared -o "$HERE/clover_shim.so" \
    "$HERE/clover_shim.cu"

if [[ ! -f "$HERE/clover_shim.so" ]]; then
    echo "BUILD FAILED: clover_shim.so not produced" >&2
    exit 1
fi
echo "Built: $HERE/clover_shim.so"
echo "=== done ==="
