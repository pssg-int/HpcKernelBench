#!/usr/bin/env bash
# Build bridge.so: nvcc compiles bridge.cu, which #includes the artifact's
# own source/src/2d/gpu.cu verbatim (see bridge.cu's docstring) into ONE
# translation unit, producing an extern "C" shared library adapter.py
# dlopens via ctypes. Idempotent (single nvcc invocation, safe to re-run).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "== LoRAStencil bridge build (stencil) =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $(nvcc --version | tail -1)"
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

nvcc -ccbin "$HOST_COMPILER" -O3 -std=c++17 -arch=sm_80 -Xcompiler -fPIC -shared \
    bridge.cu -o bridge.so

echo "-- verifying bridge.so was built --"
[ -f bridge.so ] || { echo "MISSING bridge.so"; exit 1; }
echo "== LoRAStencil bridge build: OK =="
