#!/usr/bin/env bash
# Build InferFast's SpMM library, then a thin ctypes-facing wrapper (see
# wrapper.cu's header comment for why the wrapper is needed and what it does
# -- it touches zero lines of source/, it only re-exposes already-compiled
# symbols with extern "C" linkage plus explicit host<->device buffer sizing).
#
# Idempotent: safe to re-run; exit 0 = both libraries built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"   # nvcc 12.9 rejects the
    # system default g++-14 (SUSE) on <bits/alloc_traits.h> with
    # "identifier __has_construct is undefined" -- a host-compiler/libstdc++
    # version mismatch, not an InferFast problem. g++-12 is what NERSC ships
    # alongside CUDA 12.x and builds clean; this is a build-system fix
    # (compiler selection), not a kernel-code patch (ARTIFACT_GUIDE rule 3).
MATHLIB="${MATHLIB:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib}"
    # nvcc's own toolkit dir under hpc_sdk/.../cuda/12.9 ships headers/nvcc
    # only, no libcublas/libcusparse .so -- those live under the sibling
    # math_libs/12.9 tree. Located empirically (find -iname libcublas.so*).

echo "== InferFast build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

echo "-- 1/2: source/build/libSpMM_API.so (InferFast's own Makefile) --"
# The upstream Makefile compiles SASS-only (`code=sm_$(sm)`) and defaults
# `SMS ?= 80`; it is also incremental, so an sm_80 .so left from an A100 build
# is reused as-is and then fails on an H100 with cudaErrorNoKernelImageForDevice
# (=209). Pass the target arch through the Makefile's own SMS knob (same
# mechanism as HOST_COMPILER/CUDA_PATH already passed here -- no source edit)
# and `clean` first so the arch change actually takes effect.
make -C source/build clean >/dev/null 2>&1 || true
make -C source/build \
    SMS="${KB_SM:-80}" \
    HOST_COMPILER="$HOST_COMPILER" \
    CUDA_PATH="$(dirname "$(dirname "$NVCC")")" \
    LIBRARIES="-L$MATHLIB -lcublas -lcusparse"

echo "-- 2/2: libinferfast_wrapper.so (adapter glue, see wrapper.cu) --"
"$NVCC" -ccbin "$HOST_COMPILER" -O3 -std=c++17 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -I. -Isource/build -Isource/csrc \
    -shared -o libinferfast_wrapper.so wrapper.cu \
    -L source/build -lSpMM_API -L "$MATHLIB" -lcublas -lcusparse \
    -Xlinker -rpath -Xlinker "$(pwd)/source/build" \
    -Xlinker -rpath -Xlinker "$MATHLIB"

echo "-- verifying symbols resolve --"
ldd libinferfast_wrapper.so | grep -q "not found" && \
    { echo "MISSING SHARED LIBS:"; ldd libinferfast_wrapper.so | grep "not found"; exit 1; }
echo "== InferFast build: OK =="
