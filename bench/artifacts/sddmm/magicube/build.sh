#!/usr/bin/env bash
# Build Magicube's 16-bit quantized WMMA SDDMM kernel
# (source/SDDMM/SDDMM/src/wmma_sddmm.cu, unmodified) together with this
# directory's wrapper.cu (ctypes glue for `sddmm::wmmaSddmm_16b`, raw
# pointers, no PyTorch dependency -- same style as spmm/rode/wrapper.cu).
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): see bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}}"

echo "== Magicube SDDMM (16b) build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

# Build-system fix (rule 3): the artifact's own Makefile pins
# `-std=c++11`. Under this machine's GCC 14 host compiler, nvcc's own
# compatibility headers (<type_traits>, <bits/stl_pair.h>) use C++17
# `inline constexpr` variable templates unconditionally, which GCC 14
# rejects when the requested language standard is C++11 ("__is_pair is
# not a template" / "constexpr is not valid here" -- confirmed: compiling
# the SAME .cu file with -std=c++17 instead of the artifact's own -std=c++11
# succeeds with zero errors, everything else identical). This is a
# host-compiler/CUDA-version compatibility guard, not a change to the
# kernel: no source file is patched.
"$NVCC" -ccbin "$HOST_COMPILER" -O3 -std=c++17 -Xcompiler -fPIC \
    -gencode arch=compute_80,code=sm_80 \
    -Isource/SDDMM/SDDMM/include \
    -shared -o libmagicube_sddmm16b.so \
    wrapper.cu source/SDDMM/SDDMM/src/wmma_sddmm.cu

echo "-- verifying symbols resolve --"
ldd libmagicube_sddmm16b.so | grep -q "not found" && \
    { echo "MISSING SHARED LIBS:"; ldd libmagicube_sddmm16b.so | grep "not found"; exit 1; }
nm -D libmagicube_sddmm16b.so | grep -q " T magicube_sddmm_16b" || { echo "magicube_sddmm_16b symbol missing"; exit 1; }
echo "== Magicube SDDMM (16b) build: OK =="
