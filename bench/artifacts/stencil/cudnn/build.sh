#!/usr/bin/env bash
# Build the cuDNN stencil baseline (bridge.cu, this directory) into bridge.so.
# bridge.cu is a port of ConvStencil's own src/cudnn/*.cu baseline programs
# (see its header); nothing is cloned. Links against the SAME libcudnn the
# harness's torch loads (the nvidia-cudnn-cu12 wheel in $PY's site-packages),
# so the process never holds two cuDNN copies. Override with CUDNN_DIR (a
# directory with include/cudnn.h and lib/ or lib64/ holding libcudnn.so*).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"

echo "=== cuDNN stencil baseline build (bridge.so, sm_80) ==="
echo "nvcc: $("$NVCC" --version | tail -1)"
echo "host compiler: $("$HOST_COMPILER" --version | head -1)"

CUDNN_ROOT="${CUDNN_DIR:-}"
if [ -z "$CUDNN_ROOT" ]; then
  CUDNN_ROOT="$("${PY:-python3}" -c '
import importlib.util, os
spec = importlib.util.find_spec("nvidia.cudnn")
if spec and spec.submodule_search_locations:
    print(list(spec.submodule_search_locations)[0])
' 2>/dev/null || true)"
fi
if [ -z "$CUDNN_ROOT" ] || [ ! -f "$CUDNN_ROOT/include/cudnn.h" ]; then
  echo "cudnn.h not found (set CUDNN_DIR, or install torch's nvidia-cudnn-cu12 wheel into \$PY)" >&2
  exit 1
fi
CUDNN_LIBDIR=""
for d in "$CUDNN_ROOT/lib" "$CUDNN_ROOT/lib64"; do
  if ls "$d"/libcudnn.so* >/dev/null 2>&1; then CUDNN_LIBDIR="$d"; break; fi
done
if [ -z "$CUDNN_LIBDIR" ]; then
  echo "libcudnn.so* not found under $CUDNN_ROOT/lib{,64}" >&2
  exit 1
fi
# the wheel ships only libcudnn.so.<major> (no unversioned symlink)
if [ -e "$CUDNN_LIBDIR/libcudnn.so" ]; then
  CUDNN_LINK="-lcudnn"
else
  CUDNN_LINK="-l:$(basename "$(ls "$CUDNN_LIBDIR"/libcudnn.so.[0-9]* | sort | head -1)")"
fi
echo "cudnn: $CUDNN_ROOT ($CUDNN_LINK from $CUDNN_LIBDIR)"
grep -m3 -E "define CUDNN_(MAJOR|MINOR|PATCHLEVEL)" "$CUDNN_ROOT/include/cudnn_version.h" 2>/dev/null || true

"$NVCC" -ccbin "$HOST_COMPILER" -O3 -arch=sm_80 -Xcompiler -fPIC -shared \
    -I "$CUDNN_ROOT/include" \
    "$HERE/bridge.cu" \
    -L "$CUDNN_LIBDIR" "$CUDNN_LINK" -Xlinker -rpath -Xlinker "$CUDNN_LIBDIR" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
