#!/usr/bin/env bash
# Build libyacclab_buf.so: bridge_yacclab_buf.cu, which #includes YACCLAB's
# own source/cuda/src/labeling_allegretti_2019_BUF.cu VERBATIM (see that
# file's own header comment). OpenCV is not installed on this machine and
# building OpenCV+contrib (needed for the "cudafeatures2d" module the
# kernel file itself #includes) from source is out of scope per
# ARTIFACT_GUIDE ("YACCLAB needs OpenCV: only if importable/installable
# artifact-locally without building OpenCV from source; otherwise wrap the
# CUDA kernels directly with a thin shim") -- confirmed:
#   $PY -m pip install --target <tmp> opencv-python-headless --only-binary=:all:
# succeeds (a prebuilt Python wheel exists), but that wheel is a CPU-only
# build with NO CUDA support at all, let alone the "cudafeatures2d" contrib
# module (which itself requires a from-source, CUDA-enabled OpenCV+contrib
# build) -- it cannot satisfy this C++-level header dependency regardless.
# So: shim/ (this directory's sibling) instead -- see build.sh's -I order
# below and each shim file's own docstring for exactly what it stands in
# for.
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
NVCC="${NVCC:-nvcc}"

if [ ! -d "$SRC" ]; then
  echo "source/ missing -- run:" >&2
  echo "  git clone --depth 1 https://github.com/prittt/YACCLAB.git $SRC" >&2
  exit 1
fi

echo "=== YACCLAB BUF build (libyacclab_buf.so, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

# Shim dir FIRST: #include <opencv2/cudafeatures2d.hpp> / "labeling_algorithms.h"
# / "register.h" inside the verbatim-included .cu file resolve to our
# stand-ins, not YACCLAB's own (much heavier, full-benchmark-harness) copies.
"$NVCC" -O3 -arch=sm_80 -std=c++14 -Xcompiler -fPIC --shared \
    -I "$HERE/shim" \
    -I "$SRC/cuda/src" \
    "$HERE/bridge_yacclab_buf.cu" \
    -o "$HERE/libyacclab_buf.so"

echo "Built: $HERE/libyacclab_buf.so"
echo "=== done ==="
