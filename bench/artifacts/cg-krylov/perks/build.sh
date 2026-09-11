#!/usr/bin/env bash
# PERKS (conjugateGradient) -- build via the artifact's OWN CMakeLists.txt
# (conjugateGradient/CMakeLists.txt's single active `subcompile(...)` call
# produces library `cg_perks` from main.cu + executable `cg_perks.exe` from
# cg_driver.cu, linked together -- verified empirically, not just read off
# the CMake function; see STATUS.md). All 10 {fp32,fp64} x
# {baseline,(cmat,cvec) in FF/FT/TF/TT} template instantiations compile into
# this ONE binary; which combination runs is selected at runtime via CLI
# flags (--fp32, --baseline, --cmat, --cvec), not by which CMake target name
# was built.
#
# Idempotent: safe to re-run; exit 0 = built (binary present and executable).
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

SRC_DIR="source/conjugateGradient"
BUILD_DIR="$SRC_DIR/build/init"
BINARY="$BUILD_DIR/cg_perks.exe"

echo "== PERKS (cg-krylov / conjugateGradient) build =="
echo "commit : $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found on PATH, trying 'module load cudatoolkit' ..."
    module load cudatoolkit 2>/dev/null || true
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "STATUS: BUILD-FAILED -- nvcc not found even after module load attempt"
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "STATUS: BUILD-FAILED -- cmake not found on PATH"
    exit 1
fi

echo "nvcc   : $(command -v nvcc)"
nvcc --version | tail -1
echo "cmake  : $(command -v cmake)"
cmake --version | head -1
echo "cxx    : $(command -v c++ || command -v g++)"
echo "gpu    : $(nvidia-smi -L 2>/dev/null | head -1 || echo 'nvidia-smi unavailable')"

mkdir -p "$BUILD_DIR"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
# kernelbench build-system fix: an uncapped -j$(nproc) (128-244 on a shared
# login node) exhausts this session's process limit ("cc1plus: vfork:
# Resource temporarily unavailable") -- bounded, overridable default,
# matching the sibling spcg/build.sh's already-capped -j8.
cmake --build "$BUILD_DIR" -j"${JOBS:-8}"

if [ ! -x "$BINARY" ]; then
    echo "STATUS: BUILD-FAILED -- $BINARY not produced"
    exit 1
fi

echo "binary : $BINARY"
echo "== PERKS build: OK =="
