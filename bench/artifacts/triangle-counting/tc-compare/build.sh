#!/usr/bin/env bash
# Build TC-Compare's GroupTC kernel as a shared library (tcc_shim.so),
# compiled from tcc_shim.cu (this dir, NOT part of the artifact), which
# #include's the artifact's own source/approach/GroupTC/tc.cu VERBATIM
# (zero patches -- see tcc_shim.cu's docstring).
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

echo "=== TC-Compare (GroupTC) build (tcc_shim.so, sm_80) ==="
echo "commit: $(git -C "$HERE/source" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

cd "$HERE"
nvcc -O3 -arch=sm_80 \
    -ccbin "$CXX" \
    -Xcompiler=-fopenmp,-fPIC \
    -shared -o tcc_shim.so \
    tcc_shim.cu \
    -lgomp

if [[ ! -f "$HERE/tcc_shim.so" ]]; then
    echo "BUILD FAILED: tcc_shim.so not produced" >&2
    exit 1
fi
echo "Built: $HERE/tcc_shim.so"
echo "=== done ==="
