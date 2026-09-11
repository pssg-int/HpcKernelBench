#!/usr/bin/env bash
# Build this artifact's HolaSpmv dense-vector SpMV kernel
# (source/hice-spmspv/hice/la/include/spmspv/csc-spmspv/detail/device/
# holaspmv.h) into a shared library via bridge.cu (this directory).
# Target: A100 / sm_80. Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV="$HERE/source/hice-spmspv/hice/la/include/spmspv/csc-spmspv/detail/device"
NVCC="${NVCC:-nvcc}"

echo "=== adaptive-spmv (HolaSpmv) build (bridge.so, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

# CUB is bundled with the CUDA 12.9 toolkit (toolchain.sh's CPATH already
# carries $CUDA_HOME/include) -- the artifact's own cmake/external/
# ExternalCub.cmake fetches a standalone copy for older CUDA toolkits that
# don't bundle it; not needed here, so not fetched (avoids an extra clone).
"$NVCC" -O3 -w -arch=sm_80 -Xcompiler -fPIC \
    -shared \
    -I "$DEV" \
    "$HERE/bridge.cu" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
