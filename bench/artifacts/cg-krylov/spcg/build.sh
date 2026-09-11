#!/usr/bin/env bash
# SPCG (Sparsified Preconditioned Conjugate Gradient) -- build the
# ilu0_gpu/nonsp CG driver (`conjugateGradientPrecond`). Standard CMake +
# nvcc/host-compiler build (project() only declares CXX; CUDA is used only
# via CUDAToolkit imported targets -- no .cu files to compile here).
#
# Idempotent: safe to re-run; exit 0 = built (binary present and executable).
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

SRC_DIR="source/gpu_src/ilu0_gpu/nonsp"
BUILD_DIR="$SRC_DIR/build"
BINARY="$BUILD_DIR/conjugateGradientPrecond"

echo "== SPCG (cg-krylov / ilu0_gpu/nonsp) build =="
echo "commit : $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"

# nvcc may not be on PATH in a fresh shell -- check before assuming a module
# load is needed (per task brief; this machine already has it on PATH).
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

mkdir -p "$BUILD_DIR"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
# 244 logical cores on this shared login node -- cap parallelism so a
# single-target build of one main.cpp doesn't hog it; -j8 is already
# generous for one translation unit.
cmake --build "$BUILD_DIR" -j8

if [ ! -x "$BINARY" ]; then
    echo "STATUS: BUILD-FAILED -- $BINARY not produced"
    exit 1
fi

echo "binary : $BINARY"
echo "== SPCG build: OK =="
