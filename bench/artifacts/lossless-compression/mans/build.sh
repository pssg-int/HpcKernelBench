#!/usr/bin/env bash
# Build MANS's own NVIDIA-backend compress/decompress device-pointer API
# (source/nv/mans_nv.{h,cpp} + source/nv/adm/mapping_uint{16,32}.cu +
# source/nv/ans/{mans_nv_ans.cu,src/GpuANS.cu}) into a shared library,
# together with bridge_mans.cu's thin extern "C" ctypes wrapper (this
# directory; see its docstring).
#
# MANS's own build system is CMake (source/CMakeLists.txt), but its
# nv/ans/CMakeLists.txt subproject hardcodes
# `set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)` (that path does not
# exist on this machine, or in general on any machine whose CUDA toolkit
# isn't installed at the conventional /usr/local/cuda prefix -- this
# machine's is under /opt/nvidia/hpc_sdk/...) and CMake's `project()`
# rejects a CMAKE_CUDA_COMPILER path override that isn't itself a real
# executable, before this integration's own `-DCMAKE_CUDA_COMPILER=...`
# cache entry gets a chance to override the subproject's own hardcoded
# `set()`. Rather than patch that CMakeLists.txt line (source/ stays
# untouched -- `git diff` inside source/ is empty), this build bypasses
# CMake entirely and invokes nvcc directly on the exact source-file list
# CMakeLists.txt's own BUILD_NV section names for `mans_nv_core`
# (mans_api.cpp is NOT included -- see bridge_mans.cu's docstring for why
# the lower-level mans::nv::*_internal_device functions are called
# directly instead), the same pattern this benchmark's other GPU compression
# artifacts (fzgpu, cuszp, gpulz, pfpl) already use to sidestep an artifact's
# own build system without patching it.
#
# The CPU backend (MANS_ENABLE_CPU, needing OpenMP + -march=native
# -mavx512f) is not built -- this integration only wraps the NVIDIA path.
#
# Target: A100 / sm_80.
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
NVCC="${NVCC:-nvcc}"

echo "=== MANS build (bridge.so, NV backend only, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

"$NVCC" -O3 -arch=sm_80 -std=c++17 -Xcompiler -fPIC --shared \
    -I "$SRC" -I "$SRC/nv" -I "$SRC/nv/ans/include" -I "$SRC/nv/ans/src" \
    "$HERE/bridge_mans.cu" \
    "$SRC/nv/mans_nv.cpp" \
    "$SRC/nv/adm/mapping_uint16.cu" \
    "$SRC/nv/adm/mapping_uint32.cu" \
    "$SRC/nv/ans/mans_nv_ans.cu" \
    "$SRC/nv/ans/src/GpuANS.cu" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
