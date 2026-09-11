#!/usr/bin/env bash
# Build FZ-GPU's own compress/decompress kernels into a shared library via
# bridge_fzgpu.cu (this directory; see its docstring for why a bridge that
# splits runFzgpu()'s compress/decompress halves is needed instead of
# calling the artifact's own end-to-end binary).
# --extended-lambda matches FZ-GPU's own Makefile flag (needed by
# include/kernel/lorenzo.cuh's use of extended __device__ lambdas).
# Target: A100 / sm_80 (FZ-GPU's own Makefile has no explicit -arch flag,
# i.e. it relies on nvcc's default; pinned explicitly here for this
# machine's GPU, same category of arch-flag choice as this benchmark's
# other GPU artifacts).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-nvcc}"

echo "=== FZ-GPU build (bridge.so, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

"$NVCC" -O3 -arch=sm_80 --extended-lambda -Xcompiler -fPIC --shared \
    -I "$HERE/source/include" \
    "$HERE/bridge_fzgpu.cu" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
