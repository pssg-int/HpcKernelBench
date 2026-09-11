#!/usr/bin/env bash
# Build ECL-SCC's signature-propagation SCC kernel as a shared library
# (ecl_scc_shim.so), compiled from ecl_scc_shim.cu (this dir, NOT part of
# the artifact) which #includes source/source/ECL-SCC_10.cu UNMODIFIED.
# Zero lines of the artifact were touched.
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

echo "=== ECL-SCC build (ecl_scc_shim.so, sm_80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

nvcc -std=c++17 -O3 -arch=sm_80 \
    -ccbin "$CXX" \
    -Xcompiler=-fPIC \
    -I "$HERE" \
    -shared -o "$HERE/ecl_scc_shim.so" \
    "$HERE/ecl_scc_shim.cu"

if [[ ! -f "$HERE/ecl_scc_shim.so" ]]; then
    echo "BUILD FAILED: ecl_scc_shim.so not produced" >&2
    exit 1
fi
echo "Built: $HERE/ecl_scc_shim.so"
echo "=== done ==="
