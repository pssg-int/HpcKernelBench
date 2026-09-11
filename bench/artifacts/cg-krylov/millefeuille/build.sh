#!/usr/bin/env bash
# Mille-feuille build: single nvcc invocation (no library, CLI binary only).
# The repo's own Makefile hardcodes CUDA_INSTALL_PATH=/usr/local/cuda-12.0,
# which does not exist on this machine -- ARTIFACT_GUIDE.md rule 3 explicitly
# allows build-system fixes like this. Rather than fight the Makefile's path
# logic, invoke the equivalent nvcc command directly with paths resolved from
# `command -v nvcc` (no kernel code touched by this build-system workaround).
#
# Idempotent: safe to re-run; exit 0 = built and usable.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")/source"

echo "== Mille-feuille (cg-krylov) build =="
echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc  : $(command -v nvcc)"
nvcc --version | tail -4
echo "gpu   : $(nvidia-smi -L 2>/dev/null | head -1 || echo 'nvidia-smi unavailable')"

mkdir -p data

CUDA_BIN="$(command -v nvcc)"
CUDA_HOME="$(dirname "$(dirname "$CUDA_BIN")")"
echo "CUDA_HOME resolved to: $CUDA_HOME"

nvcc src/main-cg.cu -O3 -w -arch=compute_80 -code=sm_${KB_SM:-80} \
    -gencode=arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -Xcompiler -fpermissive -Xcompiler -fopenmp \
    -maxrregcount=32 \
    -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" \
    -o main-cg

echo "== Mille-feuille build: OK (./source/main-cg produced) =="
