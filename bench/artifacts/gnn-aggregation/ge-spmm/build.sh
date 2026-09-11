#!/usr/bin/env bash
# Build GE-SpMM's pytorch-custom SpMM torch extension
# (source/pytorch-custom/{spmm.cpp,spmm_kernel.cu}, both UNMODIFIED).
#
# The artifact's own source/pytorch-custom/op.py JIT-compiles this via
# `torch.utils.cpp_extension.load(name='spmm', sources=['spmm.cpp',
# 'spmm_kernel.cu'])` at import time. ARTIFACT_GUIDE.md rule 9 forbids
# compiling inside adapter.py's available(), so build.sh drives the SAME
# load() call ahead of time instead, with an explicit build_directory (so
# the .so lands under this directory's own build/, never inside source/,
# per rule 3) and an explicit -lcusparse (spmm_kernel.cu's csr2cscKernel
# calls cusparseCsr2cscEx2*, and torch's cpp_extension does not link
# cuSPARSE by default the way it does cuBLAS/cudart).
#
# Idempotent: exit 0 if the compiled extension is already present and
# FORCE_REBUILD is not set.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
SRC_DIR="$HERE/source/pytorch-custom"
BUILD_DIR="$HERE/build"

if compgen -G "$BUILD_DIR"/spmm*.so > /dev/null && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[ge-spmm] extension already built, skipping (set FORCE_REBUILD=1 to rebuild)"
    exit 0
fi
mkdir -p "$BUILD_DIR"

export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"   # A100 (sm_80)
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"

echo "[ge-spmm] python: $($PY --version), torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "[ge-spmm] host compiler: $($CXX --version | head -1)"
echo "[ge-spmm] nvcc: $(nvcc --version | tail -1)"

export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}"

SRC_DIR="$SRC_DIR" BUILD_DIR="$BUILD_DIR" "$PY" - <<'PYEOF'
import os
import torch.utils.cpp_extension as cpp_ext

src_dir = os.environ["SRC_DIR"]
build_dir = os.environ["BUILD_DIR"]

cpp_ext.load(
    name="spmm",
    sources=[os.path.join(src_dir, "spmm.cpp"),
             os.path.join(src_dir, "spmm_kernel.cu")],
    build_directory=build_dir,
    extra_ldflags=["-lcusparse"],
    verbose=True,
)
PYEOF

echo "[ge-spmm] built $(compgen -G "$BUILD_DIR"/spmm*.so)"
