#!/usr/bin/env bash
# Build GraphFold's k-clique-counting kernel (CFSolver) as a shared library
# (gf_shim.so), compiled from gf_shim.cu (this dir, NOT part of the
# artifact) directly against the artifact's header-only src/ tree --
# equivalent flags to source/CMakeLists.txt's project (c++14,
# --expt-extended-lambda, sm_80 override in place of its 60/61/70 default).
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

echo "=== GraphFold build (gf_shim.so, sm_80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

# compat_include/ (this dir, NOT part of the artifact) stubs ONE missing
# header: thrust/system/cuda/experimental/pinned_allocator.h, which CTK
# 12.9's bundled Thrust no longer ships but src/utils/buffer.h still
# #includes (for an unused constructor overload -- see compat_include's own
# comment). The artifact's OWN vendored old thrust (source/thirdparty/thrust)
# is deliberately NOT used: it is incompatible with CTK 12.9's bundled CUB
# once device_vector internals are actually instantiated (verified: mixing
# them fails with cascading "THRUST_NS_QUALIFIER undefined" errors deep in
# cub/device/dispatch/dispatch_streaming_reduce.cuh). Using CTK's own
# (version-matched) Thrust+CUB throughout, patched only for this one dead
# code path, is the smaller and more correct fix.
nvcc -std=c++14 -O3 -arch=sm_80 \
    -ccbin "$CXX" \
    --expt-extended-lambda --expt-relaxed-constexpr \
    -Xcompiler=-fopenmp,-fPIC,-w \
    -w \
    -I "$HERE/compat_include" \
    -I "$SRC" \
    -shared -o "$HERE/gf_shim.so" \
    "$HERE/gf_shim.cu" \
    -lgomp

if [[ ! -f "$HERE/gf_shim.so" ]]; then
    echo "BUILD FAILED: gf_shim.so not produced" >&2
    exit 1
fi
echo "Built: $HERE/gf_shim.so"
echo "=== done ==="
