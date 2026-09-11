#!/usr/bin/env bash
# Build a thin ctypes-facing wrapper (see wrapper.cu's header comment for
# why it exists and what it does -- it touches zero lines of source/, it
# only compiles the artifact's own cgemm.cuh kernel + its own TurboFNO.h
# macro header into a small standalone .so with one extern "C" launcher).
#
# No TurboFNO submodule init, no cuFFT/cuBLAS link needed -- cgemm.cuh is a
# plain shared-memory-tiled kernel with no FFT/tensor-core dependency, so
# this compiles standalone against nvcc's own toolkit headers only.
#
# Idempotent: safe to re-run; exit 0 = built.
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
    # version mismatch, unrelated to TurboFNO. g++-12 is what NERSC ships
    # alongside CUDA 12.x and builds clean; this is a build-system fix
    # (compiler selection), not a kernel-code patch (ARTIFACT_GUIDE rule 3).

echo "== TurboFNO (cgemm) build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

echo "-- compiling libturbofno_wrapper.so --"
"$NVCC" -ccbin "$HOST_COMPILER" -O3 -std=c++17 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -Isource/utils -Isource/fusion_variants/1D_E_baseline \
    -shared -o libturbofno_wrapper.so wrapper.cu

echo "-- verifying symbols resolve --"
ldd libturbofno_wrapper.so | grep -q "not found" && \
    { echo "MISSING SHARED LIBS:"; ldd libturbofno_wrapper.so | grep "not found"; exit 1; }
echo "== TurboFNO (cgemm) build: OK =="
