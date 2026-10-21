#!/usr/bin/env bash
# Mille-feuille, the PAPER'S method: main-cg-mixed (the repo's `make cg-mix`
# target) -- single persistent kernel + tile-wise dynamic mixed precision.
# ../millefeuille builds main-cg instead, whose >=100K-nnz path is a plain
# multi-kernel fp64 CG.
#
# source/ is shared with ../millefeuille (symlink). This script
#   1. applies mixed.patch (I/O only: external RHS, x dump, timer printout,
#      a message on the silent early return) if it is not applied yet;
#   2. compiles src/main-cg-mixed.cu with the Makefile's own cg-mix flags
#      (the Makefile hard-codes CUDA_INSTALL_PATH=/usr/local/cuda-12.0, so
#      nvcc is called directly, like ../millefeuille/build.sh does).
# No kernel code is changed.
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/source"

echo "== Mille-feuille mixed (main-cg-mixed) build =="
echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc  : $(command -v nvcc)"
nvcc --version | tail -1

if git apply --reverse --check "$HERE/mixed.patch" 2>/dev/null; then
  echo "mixed.patch: already applied"
else
  git apply "$HERE/mixed.patch"
  echo "mixed.patch: applied"
fi

CUDA_HOME_RESOLVED="$(dirname "$(dirname "$(command -v nvcc)")")"
SM="${KB_SM:-80}"
# flags = Makefile `cg-mix` target, arch made overridable
nvcc src/main-cg-mixed.cu -O3 -w -arch=compute_80 -code=sm_$SM \
    -gencode=arch=compute_$SM,code=sm_$SM \
    -Xcompiler -fpermissive -Xcompiler -fopenmp \
    -maxrregcount=32 \
    -I"$CUDA_HOME_RESOLVED/include" -L"$CUDA_HOME_RESOLVED/lib64" \
    -o main-cg-mixed

[ -x main-cg-mixed ] || { echo "STATUS: BUILD-FAILED -- main-cg-mixed not produced"; exit 1; }
echo "== Mille-feuille mixed build: OK (./source/main-cg-mixed produced) =="
