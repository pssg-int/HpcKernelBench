#!/usr/bin/env bash
# Build SSpMM's own kernel object (unmodified) and a thin ctypes wrapper .so
# around its host entry point `spmm::SSpMM` (see wrapper.cu's header
# comment). Idempotent; exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"   # same nvcc/g++-14 mismatch
    # documented in ../inferfast/build.sh and ../mp-spmm/build.sh
    # ("identifier __has_construct is undefined" under nvcc 12.9 -ccbin
    # g++-14 for any TU including <vector>/<iostream>); g++-12 compiles clean.

echo "== SSpMM build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

echo "-- 1/2: sspmm.o (the artifact's own kernel, unmodified) --"
# Mirrors source/SSpMM/Makefile_ampere's compile rule
# ($(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu -> nvcc -std=c++11 -arch=sm_80 ... -x cu
# -c), with -std=c++17 (this repo's convention elsewhere; the source uses no
# C++11-specific construct that breaks under -std=c++17) and
# -gencode=arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} (equivalent to their -arch=sm_80,
# spelled out per this repo's other build.sh files).
"$NVCC" -ccbin "$HOST_COMPILER" -std=c++17 -O3 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} --ptxas-options=-v \
    -I source/SSpMM/include \
    -x cu -c source/SSpMM/src/sspmm.cu -o sspmm.o

echo "-- 2/2: libsspmm_wrapper.so (ctypes glue, see wrapper.cu) --"
"$NVCC" -ccbin "$HOST_COMPILER" -std=c++17 -O3 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -I source/SSpMM/include \
    -shared -o libsspmm_wrapper.so wrapper.cu sspmm.o

echo "-- verifying symbols/libs resolve --"
ldd libsspmm_wrapper.so | grep -q "not found" && \
    { echo "MISSING SHARED LIBS:"; ldd libsspmm_wrapper.so | grep "not found"; exit 1; }
echo "== SSpMM build: OK =="
