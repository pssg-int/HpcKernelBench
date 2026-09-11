#!/usr/bin/env bash
# Build TileSpGEMM's tilespgemm() kernel + tile2csr()/csr2tile.h preprocessing
# as a shared library (tsg_bridge.so), compiled from tsg_bridge.cu (this dir,
# NOT part of the artifact) directly against the artifact's header-only
# source/src/ tree. No file under source/ is edited; the artifact's own
# main.cu (file I/O + CLI parsing) is never compiled.
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"

echo "=== TileSpGEMM build (tsg_bridge.so, sm_80) ==="
echo "commit: $(git -C "$HERE/source" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $($NVCC --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

# Artifact's own Makefile targets an RTX-30xx dev box
# (-arch=compute_61 -code=sm_86) with CUDA 11.4; this machine is an A100
# (sm_80) with nvcc 12.9 -- arch retarget is a build-system fix
# (ARTIFACT_GUIDE.md rule 3), no kernel arithmetic touched. MAT_VAL_TYPE is
# already `double` unconditionally in source/src/common.h (no -D needed).
# -Xcompiler -fopenmp: csr2tile.h/tile2csr.h use #pragma omp parallel for.
"$NVCC" -std=c++14 -O3 -w -arch=sm_80 \
    -ccbin "$CXX" \
    --expt-relaxed-constexpr \
    -Xcompiler=-fopenmp,-fPIC \
    -I "$HERE" \
    -shared -o "$HERE/tsg_bridge.so" \
    "$HERE/tsg_bridge.cu" \
    -lgomp

if [[ ! -f "$HERE/tsg_bridge.so" ]]; then
    echo "BUILD FAILED: tsg_bridge.so not produced" >&2
    exit 1
fi
nm -D "$HERE/tsg_bridge.so" | grep -q "tsg_run" || {
    echo "tsg_run symbol missing from tsg_bridge.so" >&2
    exit 1
}
echo "Built: $HERE/tsg_bridge.so"
echo "=== done ==="
