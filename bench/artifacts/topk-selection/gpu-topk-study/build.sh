#!/usr/bin/env bash
# Build gpu_topK_benchmark's GridSelect kernel as a shared library
# (topk_shim.so), compiled from topk_shim.cu (this dir, NOT part of the
# artifact) against source/include/grid_select.h and linked directly to
# the artifact's own PREBUILT source/third_party/libgridselect.so -- no
# nvcc compilation of the kernel itself is needed (or possible without its
# source, which the artifact does not ship -- only the compiled .so; see
# STATUS.md). No third-party downloads (RAFT/Faiss/gpu_selection/DrTopKSC,
# needed by the suite's OTHER algorithms) are required for this kernel.
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"

LIBDIR="$HERE/source/third_party"
if [[ ! -f "$LIBDIR/libgridselect.so" ]]; then
    echo "BUILD FAILED: $LIBDIR/libgridselect.so missing (expected checked into the artifact repo)" >&2
    exit 1
fi

echo "=== gpu-topk-study (GridSelect) build (topk_shim.so, sm_80) ==="
echo "commit: $(git -C "$HERE/source" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

cd "$HERE"
nvcc -O3 -arch=sm_80 \
    -ccbin "$CXX" \
    -Xcompiler=-fPIC \
    -I source/include \
    -L "$LIBDIR" -lgridselect \
    -shared -o topk_shim.so \
    topk_shim.cu \
    -Xlinker -rpath,'$ORIGIN/source/third_party'

if [[ ! -f "$HERE/topk_shim.so" ]]; then
    echo "BUILD FAILED: topk_shim.so not produced" >&2
    exit 1
fi
echo "Built: $HERE/topk_shim.so"
echo "=== done ==="
