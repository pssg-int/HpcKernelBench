#!/usr/bin/env bash
# Build the cuSPARSE CG baseline (bridge.cu, this directory) into bridge.so.
# bridge.cu is OUR port of NVIDIA's CUDA samples conjugateGradient and
# conjugateGradientPrecond (see source.provenance); nothing is cloned.
# ../cusparse-ilu0-cg uses the same bridge.so.
#
# csrilu02 is deprecated (not removed) in CUDA 12.x; the deprecation warnings
# are silenced. CUDA 13 may drop it -- build with the 12.x toolkit that
# toolchain.sh pins.
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"
ARCH="${KB_CUDA_ARCH:-80}"

echo "=== cuSPARSE CG baseline build (bridge.so, sm_$ARCH) ==="
echo "nvcc: $("$NVCC" --version | tail -1)"
echo "host compiler: $("$HOST_COMPILER" --version | head -1)"
echo "math libs: $KB_MATHLIB_DIR"

"$NVCC" -ccbin "$HOST_COMPILER" -O3 -arch="sm_$ARCH" -Xcompiler -fPIC -shared \
    -Xcompiler -Wno-deprecated-declarations -diag-suppress 1444 \
    -I "$KB_CUDA_MATHLIBS/include" \
    "$HERE/bridge.cu" \
    -L "$KB_MATHLIB_DIR" -lcusparse -lcublas \
    -Xlinker -rpath -Xlinker "$KB_MATHLIB_DIR" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
