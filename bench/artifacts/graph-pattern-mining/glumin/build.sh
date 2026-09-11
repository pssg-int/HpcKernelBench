#!/usr/bin/env bash
# Build GLumin's k-clique-counting kernel (CliqueSolver's LUT-accelerated
# G2Miner variant) as a shared library (gl_shim.so), compiled from
# gl_shim.cu (this dir, NOT part of the artifact) directly against the
# artifact's header-only include/ + src/clique/gpu_kernels/ trees.
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

echo "=== GLumin build (gl_shim.so, sm_80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

# -DUSE_GPU matches source's own NVFLAGS (common.mk); no vendored thrust/cub
# here (GLumin relies on the CTK's own <cub/cub.cuh>, already version-matched
# with nvcc 12.9 -- no compat shim needed, unlike graphfold/).
#
# src/common/graph.cc and src/common/VertexSet.cc are NOT header-only (unlike
# GraphFold's src/): Graph's own methods (allocateFrom, orientation,
# init_edgelist, ...) and VertexSet's static buffer-pool members are defined
# out-of-line there. source/src/common.mk's own OBJS list
# (main.o VertexSet.o graph.o) confirms every GLumin binary links both --
# so this shared library does too, in place of main.o (replaced by
# gl_shim.cu).
nvcc -std=c++17 -O3 -arch=sm_80 \
    -ccbin "$CXX" \
    --extended-lambda --expt-relaxed-constexpr \
    -Xcompiler=-fopenmp,-fPIC,-w \
    -w \
    -DUSE_GPU \
    -I "$SRC/include" \
    -I "$SRC/src/clique/gpu_kernels" \
    -shared -o "$HERE/gl_shim.so" \
    "$HERE/gl_shim.cu" \
    "$SRC/src/common/graph.cc" \
    "$SRC/src/common/VertexSet.cc" \
    -lgomp

if [[ ! -f "$HERE/gl_shim.so" ]]; then
    echo "BUILD FAILED: gl_shim.so not produced" >&2
    exit 1
fi
echo "Built: $HERE/gl_shim.so"
echo "=== done ==="
