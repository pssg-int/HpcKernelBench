#!/usr/bin/env bash
# Build PFPL's f32 ABS GPU encode/decode kernels into a shared library via
# bridge_encode.cu/bridge_decode.cu (this directory; see their docstrings
# for why two bridge files -- two separate translation units -- are needed
# instead of calling the artifact's own two standalone CLI binaries).
# Target: A100 / sm_80 (PFPL's own makefile defaults to -arch=sm_70;
# overridden here per this machine's GPU, the same kind of arch-flag
# build-system fix used by every other GPU artifact in this benchmark).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-nvcc}"

echo "=== PFPL build (bridge.so, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

"$NVCC" -O3 -arch=sm_80 -Xcompiler -fPIC -shared \
    -I "$HERE/source/src" \
    "$HERE/bridge_encode.cu" "$HERE/bridge_decode.cu" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
