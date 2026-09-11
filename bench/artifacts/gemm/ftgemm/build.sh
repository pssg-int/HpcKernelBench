#!/usr/bin/env bash
# Build bridge.so: nvcc compiles bridge.cu, which #includes the artifact's
# own kernel/ft_sgemm/include_code_gen/{sgemm_large,ft_sgemm_large}.cuh
# verbatim (see bridge.cu's docstring) into ONE translation unit, producing
# an extern "C" shared library adapter.py dlopens via ctypes. Idempotent
# (single nvcc invocation, safe to re-run).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "== FT-GEMM bridge build (gemm) =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $(nvcc --version | tail -1)"

nvcc -O3 -std=c++14 -arch=sm_80 -Xcompiler -fPIC -shared \
    bridge.cu -o bridge.so

echo "-- verifying bridge.so was built --"
[ -f bridge.so ] || { echo "MISSING bridge.so"; exit 1; }
echo "== FT-GEMM bridge build: OK =="
