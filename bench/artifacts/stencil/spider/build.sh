#!/usr/bin/env bash
# Build SPIDER's 2D 7-radius half-precision sparse-Tensor-Core stencil
# kernel (source/src/2d_half_sparse/gpu_2d_7r_half.{h,cu}) into a shared
# library via bridge.cu (this directory; see its docstring for why a bridge
# is needed instead of the artifact's own main() driver, which additionally
# requires an uninitialized git submodule -- third_party/argparse -- this
# build never touches). Target: A100 / sm_80, which supports the mma.sp
# 2:4-structured-sparse Tensor Core instructions the kernel issues.
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source/src/2d_half_sparse"
NVCC="${NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"

echo "=== SPIDER 2d_half_sparse build (bridge.so, sm_80) ==="
echo "nvcc: $("$NVCC" --version | tail -1)"
# Host compiler pin (2026-09-09, zaratan): without an explicit -ccbin, nvcc
# does its own host-compiler search rather than simply taking the first g++
# on $PATH; on this machine that search finds this build env's own default
# g++ 13.4.0 (bundled alongside nvcc in the same conda env), which nvcc 12.x
# cannot parse (<type_traits>/<bits/hashtable.h> "identifier is undefined"
# errors -- same class of failure documented in
# bench/artifacts/convolution/hidet/STATUS.md's Reproduction section, and
# fixed the same way by most other artifacts in this repo, see `grep -l
# ccbin bench/artifacts/*/*/build.sh`). KB_GXX12 (exported by
# bench/artifacts/toolchain.sh, sourced above) is a g++ 12 known to work
# with this nvcc; harmless / a no-op default elsewhere (e.g. Perlmutter's
# NVIDIA HPC SDK nvcc, whose install dir has no competing gcc).
HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"
echo "host compiler: $("$HOST_COMPILER" --version | head -1)"

"$NVCC" -ccbin "$HOST_COMPILER" -O3 -w -arch=sm_80 -std=c++17 -Xcompiler -fPIC \
    -shared \
    -I "$SRC" \
    "$HERE/bridge.cu" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
