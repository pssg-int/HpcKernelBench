#!/usr/bin/env bash
# Build TLPGNN's GCN aggregation kernel (source/gcn/naive_kernel.cu,
# UNMODIFIED) into a torch CUDA extension.
#
# The artifact's own source/gcn/test_kernel.py JIT-compiles this kernel at
# RUN time via torch.utils.cpp_extension.load_inline(cpp_sources=[...],
# cuda_sources=[open("naive_kernel.cu").read()]). ARTIFACT_GUIDE.md rule 9
# forbids compiling inside adapter.py's available() (a JIT call there turns
# every registry scan into a rebuild), so this build.sh instead drives the
# file-based, ahead-of-time torch.utils.cpp_extension.load() against:
#   - source/gcn/naive_kernel.cu           -- the artifact's own kernel, UNMODIFIED
#   - ./gcn_glue.cpp                       -- a byte-for-byte copy of test_kernel.py's
#                                              inline `cpp_source` string (plus one
#                                              #include load_inline() would have
#                                              auto-prepended), written to this
#                                              directory (ARTIFACT_GUIDE.md rule 3:
#                                              files WE write live outside source/)
#
# Idempotent: exit 0 if the compiled extension is already present and
# FORCE_REBUILD is not set.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
SRC_DIR="$HERE/source/gcn"
BUILD_DIR="$HERE/build"

if compgen -G "$BUILD_DIR"/tlpgnn_gcn*.so > /dev/null && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[tlpgnn] extension already built, skipping (set FORCE_REBUILD=1 to rebuild)"
    exit 0
fi
mkdir -p "$BUILD_DIR"

export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"   # A100 (sm_80)
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"

echo "[tlpgnn] python: $($PY --version), torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "[tlpgnn] host compiler: $($CXX --version | head -1)"
echo "[tlpgnn] nvcc: $(nvcc --version | tail -1)"

export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}"

SRC_DIR="$SRC_DIR" BUILD_DIR="$BUILD_DIR" HERE="$HERE" "$PY" - <<'PYEOF'
import os
import torch.utils.cpp_extension as cpp_ext

src_dir = os.environ["SRC_DIR"]
build_dir = os.environ["BUILD_DIR"]
here = os.environ["HERE"]

cpp_ext.load(
    name="tlpgnn_gcn",
    sources=[os.path.join(here, "gcn_glue.cpp"),
             os.path.join(src_dir, "naive_kernel.cu")],
    build_directory=build_dir,
    # matches test_kernel.py's own extra_cuda_cflags, plus a force-include:
    # load_inline() normally prepends "#include <torch/extension.h>" to both
    # its cpp_source AND cuda_source strings before compiling (naive_kernel.cu
    # itself uses torch::Tensor/AT_DISPATCH_FLOATING_TYPES with no #include of
    # its own -- it only ever compiled because load_inline supplied this).
    # load() (file-based) does not do that, and naive_kernel.cu must stay
    # byte-identical to the clone (rule 3), so the equivalent include is
    # forced via a compiler flag instead of editing the source file.
    extra_cuda_cflags=["-Xptxas", "-O3", "-m", "64",
                        "-include", "torch/extension.h"],
    verbose=True,
)
PYEOF

echo "[tlpgnn] built $(compgen -G "$BUILD_DIR"/tlpgnn_gcn*.so)"
