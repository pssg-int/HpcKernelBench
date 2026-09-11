#!/usr/bin/env bash
# Build ConvStencil's 2D box-stencil Tensor-Core kernel (source/src/2d/gpu.cu)
# into a shared library via bridge.cu (this directory; see its docstring for
# why a bridge is needed instead of calling the artifact's own main.cu CLI
# driver -- short version: main.cu's star2d1r/star2d3r param selection is
# dead code (always-zero weights), and the CLI bundles H2D/compute/D2H per
# call with no clean multi-timestep hook). Target: A100 / sm_80, matching
# the known-good arch flag the sibling SPIDER artifact already builds
# ConvStencil with on this exact machine (cmake -DCMAKE_CUDA_ARCHITECTURES=80).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source/src/2d"
NVCC="${NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"

echo "=== ConvStencil (2D) build (bridge.so, sm_80) ==="
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

# gpu_box_2d1r/gpu_box_2d3r use double-precision wmma (mma.h), which requires
# sm_80+ (A100). -rdc is not needed (single translation unit: bridge.cu
# includes 2d_utils.h only; gpu.cu is compiled directly alongside it, not
# via separable compilation, matching how the artifact's own CMakeLists.txt
# builds convstencil_2d: src/2d/main.cu + src/2d/gpu.cu as one executable).
"$NVCC" -ccbin "$HOST_COMPILER" -O3 -w -arch=sm_80 -Xcompiler -fPIC -shared \
    -I "$SRC" \
    "$HERE/bridge.cu" "$SRC/gpu.cu" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
