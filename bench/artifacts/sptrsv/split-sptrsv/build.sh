#!/usr/bin/env bash
# Split_SpTRSV: build shim.cu (this directory's own extern "C" launcher, NOT
# part of the artifact) against source/include/sptrsv_syncfree_cuda.cuh
# UNMODIFIED. No MKL / libUFget / icpc needed -- we bypass the artifact's own
# CLI (src/main.cpp) and build system (its Makefile requires icpc+MKL+
# libUFget, none available/needed here) entirely and compile only the
# self-contained CUDA kernel header, per ARTIFACT_GUIDE.md rule 1 ("wrap the
# kernel, not the paper's benchmark script").
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail

# Pin toolchain to nvcc 12.9 regardless of the login node's default loaded
# module -- see ../../toolchain.sh for why (CUDA_HOME/PATH/CPATH all need
# repointing, not just PATH; CPATH must be RE-POINTED at the 12.9 math_libs
# tree, not cleared, or cusparse.h goes missing for anything torch-adjacent).
# Resolved BEFORE the cd below: once cwd changes, a relative BASH_SOURCE[0]
# (as when this script is invoked as a relative path, e.g. from the repo
# root) would resolve against the NEW cwd and double the path.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-$(command -v nvcc || true)}"
if [ -z "$NVCC" ]; then
  echo "nvcc not found on PATH -- try 'module load cudatoolkit'" >&2
  exit 1
fi

echo "== Split_SpTRSV shim build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc  : $($NVCC --version | tail -1)"

"$NVCC" -shared -Xcompiler -fPIC -O3 -std=c++14 \
    -Isource/include \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -o shim.so shim.cu

echo "== Split_SpTRSV shim build: OK -> shim.so =="
