#!/usr/bin/env bash
# YuenyeungSpTRSV: build shim.cu (this directory's own extern "C" launcher,
# NOT part of the artifact) against source/CUDA/YYSpTRSV.h and
# source/CUDA/tranpose.h UNMODIFIED (the shfl_down compat fix is a scoped
# preprocessor macro defined in shim.cu itself, undef'd right after use --
# see that file's comment; nothing under source/ is edited).
#
# We bypass the artifact's own Makefile/main.cu entirely (its Makefile
# hardcodes -gencode=arch=compute_61,code=sm_61 for a Pascal GPU and a
# CUDA-10.2 include/lib path that doesn't exist here; main.cu's own CLI
# needs a .mtx file on disk) and compile only the kernel + its host-side
# preprocessing helper, per ARTIFACT_GUIDE.md rule 1 ("wrap the kernel,
# not the paper's benchmark script").
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

echo "== YuenyeungSpTRSV shim build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc  : $($NVCC --version | tail -1)"

"$NVCC" -shared -Xcompiler -fPIC -O3 -std=c++14 \
    -Isource/CUDA \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -o shim.so shim.cu

echo "== YuenyeungSpTRSV shim build: OK -> shim.so =="
