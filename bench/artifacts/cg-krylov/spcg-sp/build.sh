#!/usr/bin/env bash
# SPCG, the paper's method: build the SPARSIFIED ILU(0) PCG driver
# gpu_src/ilu0_gpu/sp (../spcg builds ilu0_gpu/nonsp, SPCG's own
# non-sparsified baseline). source/ is shared with ../spcg (symlink).
#   1. applies sp.patch (I/O only: external b, x dump) if not applied yet;
#   2. CMake build exactly like ../spcg/build.sh, in the sp directory.
# Idempotent. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/source"

SRC_DIR="gpu_src/ilu0_gpu/sp"
BUILD_DIR="$SRC_DIR/build"
BINARY="$BUILD_DIR/conjugateGradientPrecond"

echo "== SPCG sparsified (ilu0_gpu/sp) build =="
echo "commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
command -v nvcc >/dev/null || { echo "STATUS: BUILD-FAILED -- nvcc not on PATH"; exit 1; }
command -v cmake >/dev/null || { echo "STATUS: BUILD-FAILED -- cmake not on PATH"; exit 1; }
nvcc --version | tail -1
cmake --version | head -1

if git apply --reverse --check "$HERE/sp.patch" 2>/dev/null; then
  echo "sp.patch: already applied"
else
  # sp/main.cpp carries nothing but this patch (../spcg/source.patch touches
  # only nonsp/main.cpp): restore the pristine file, then apply
  git checkout -- "$SRC_DIR/main.cpp"
  git apply "$HERE/sp.patch"
  echo "sp.patch: applied"
fi

mkdir -p "$BUILD_DIR"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"${JOBS:-8}"

[ -x "$BINARY" ] || { echo "STATUS: BUILD-FAILED -- $BINARY not produced"; exit 1; }
echo "binary : $HERE/source/$BINARY"
echo "== SPCG sp build: OK =="
