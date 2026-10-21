#!/usr/bin/env bash
# Build NVIDIA AmgX (source/, cloned from source.provenance by toolchain.sh)
# as a single-GPU, no-MPI shared library, then bridge.cu -> bridge.so.
#
# The AmgX library build is long (tens of minutes); it is skipped when
# source/build/libamgxsh.so already exists. Set AMGX_REBUILD=1 to force it.
# AmgX v2.5.0 defaults to sm_90/100/120 only, so the A100 arch is passed
# explicitly (KB_CUDA_ARCH, default 80).
#
# Idempotent. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"
HOST_CC="${HOST_CC:-$(dirname "$HOST_COMPILER")/$(basename "$HOST_COMPILER" | sed 's/g++/gcc/')}"
ARCH="${KB_CUDA_ARCH:-80}"
SRC="$HERE/source"
BUILD="$SRC/build"
LIB="$BUILD/libamgxsh.so"

echo "=== AmgX baseline build (sm_$ARCH, no MPI) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $("$NVCC" --version | tail -1)"
echo "host compiler: $("$HOST_COMPILER" --version | head -1)"
command -v cmake >/dev/null || { echo "STATUS: BUILD-FAILED -- cmake not on PATH"; exit 1; }
echo "cmake: $(cmake --version | head -1)"

if [ ! -f "$LIB" ] || [ "${AMGX_REBUILD:-0}" = 1 ]; then
  cmake -S "$SRC" -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_NO_MPI=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DCMAKE_CUDA_COMPILER="$NVCC" \
      -DCMAKE_CUDA_HOST_COMPILER="$HOST_COMPILER" \
      -DCMAKE_CXX_COMPILER="$HOST_COMPILER" \
      -DCMAKE_C_COMPILER="$HOST_CC"
  # bounded -j: an uncapped build exhausts the login node's process limit
  cmake --build "$BUILD" --target amgxsh -j"${JOBS:-8}"
fi
[ -f "$LIB" ] || { echo "STATUS: BUILD-FAILED -- $LIB not produced"; exit 1; }
echo "libamgxsh: $LIB"

"$NVCC" -ccbin "$HOST_COMPILER" -O2 -arch="sm_$ARCH" -Xcompiler -fPIC -shared \
    -I "$SRC/include" \
    "$HERE/bridge.cu" \
    -L "$BUILD" -lamgxsh -Xlinker -rpath -Xlinker "$BUILD" \
    -L "$KB_MATHLIB_DIR" -Xlinker -rpath -Xlinker "$KB_MATHLIB_DIR" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
