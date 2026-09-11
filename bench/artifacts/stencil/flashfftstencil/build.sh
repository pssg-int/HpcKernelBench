#!/usr/bin/env bash
# Build FlashFFTStencil's 2D box-stencil kernel (source/src/2D/rfft_2d/
# 2d_rfft_1_async.cu: rfft_2d_8_nwarp<>) + its FFT-plan construction
# (source/src/2D/create_fft_pfa_plan.cu: CreatePlan) into a shared library
# via bridge.cu (this directory; see its docstring for why a bridge is
# needed instead of calling the artifact's own 2d_main.cu CLI driver).
# Target: A100 / sm_80.
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"
# cuFFT ships under math_libs on this HPC SDK install, not under cuda/12.9
# itself (no libcufft.so under cuda/12.9/lib64 -- verified before writing
# this script).
CUFFT_PREFIX="${CUFFT_PREFIX:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux}"

echo "=== FlashFFTStencil-2D build (bridge.so, sm_80) ==="
echo "nvcc: $("$NVCC" --version | tail -1)"
echo "cufft prefix: $CUFFT_PREFIX"
test -f "$CUFFT_PREFIX/lib/libcufft.so" || { echo "libcufft.so not found under $CUFFT_PREFIX/lib" >&2; exit 1; }
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

"$NVCC" -ccbin "$HOST_COMPILER" -O3 -w -arch=sm_80 -Xcompiler -fPIC -shared \
    -I "$CUFFT_PREFIX/include" \
    -L "$CUFFT_PREFIX/lib" \
    "$HERE/bridge.cu" \
    -lcufft \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
