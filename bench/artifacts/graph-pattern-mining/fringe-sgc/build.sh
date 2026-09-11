#!/usr/bin/env bash
# Build Fringe-SGC's k-clique-counting path (K4 only, via its own
# triangle-core + 1-fringe-anchored-to-all-3 reduction -- see fc_shim.cu's
# file header) as a shared library (fc_shim.so).
#
# Two steps:
#  1. Build the artifact's OWN, unmodified `fringePreprocess` CLI (host-only,
#     g++, matching source/Makefile's recipe) and run it ONCE with a K4 edge
#     list to produce k4.mo (a serialized MatchingOrder struct) -- this is
#     graph-INDEPENDENT pattern preprocessing, safe to do once at build time.
#  2. Compile fc_shim.cu (this dir, NOT part of the artifact), which
#     #include's source/src/fringeCount.cu directly (needed because its
#     core-search functions have `static` linkage -- see fc_shim.cu).
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

echo "=== Fringe-SGC build (fc_shim.so, sm_80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

echo "--- step 1: fringePreprocess (host-only, unmodified) + K4 motif ---"
"$CXX" -O3 -fopenmp "$SRC/src/fringePreprocess.cpp" -o "$HERE/fringePreprocess"
# K4: 4 nodes, all 6 edges (0-1,0-2,0-3,1-2,1-3,2-3) -- produces ./motif.mo
( cd "$HERE" && ./fringePreprocess 4 0 1 0 2 0 3 1 2 1 3 2 3 > fringePreprocess_k4.log 2>&1 )
mv "$HERE/motif.mo" "$HERE/k4.mo"
mv "$HERE/motif.dot" "$HERE/k4.dot" 2>/dev/null || true
echo "wrote $HERE/k4.mo ($(stat -c%s "$HERE/k4.mo" 2>/dev/null || echo '?') bytes)"

echo "--- step 2: fc_shim.so ---"
nvcc -std=c++17 -O3 -arch=sm_80 \
    -ccbin "$CXX" \
    --extended-lambda --expt-relaxed-constexpr \
    -Xcompiler=-fopenmp,-fPIC,-w \
    -w \
    -I "$SRC/src" \
    -shared -o "$HERE/fc_shim.so" \
    "$HERE/fc_shim.cu"

if [[ ! -f "$HERE/fc_shim.so" ]]; then
    echo "BUILD FAILED: fc_shim.so not produced" >&2
    exit 1
fi
echo "Built: $HERE/fc_shim.so"
echo "=== done ==="
