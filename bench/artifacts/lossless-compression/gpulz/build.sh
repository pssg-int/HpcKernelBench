#!/usr/bin/env bash
# Build GPULZ's own compress/decompress kernels into a shared library via
# bridge_gpulz.cu (this directory; see its docstring for why a bridge that
# splits main()'s compress/decompress halves is needed instead of calling
# the artifact's own end-to-end CLI binary).
# Target: A100 / sm_80 (GPULZ's own Makefile hardcodes `-arch sm_80` already
# -- this build keeps that value, it happens to already match this
# machine's GPU).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-nvcc}"

echo "=== GPULZ build (bridge.so, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

"$NVCC" -O3 -arch=sm_80 -Xcompiler -fPIC --shared \
    "$HERE/bridge_gpulz.cu" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
