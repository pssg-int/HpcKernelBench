#!/usr/bin/env bash
# Build TC-GNN's TCGNN torch CUDA extension (source/TCGNN_conv/TCGNN.cpp +
# TCGNN_kernel.cu, both UNMODIFIED) against this machine's python env
# (torch 2.8/cu128) and nvcc 12.9 / sm_80.
#
# source/TCGNN_conv/setup.py is a plain CUDAExtension (no hardcoded compiler,
# unlike StraGCN's setup.py) -- it COULD be run directly, but this build
# instead drives torch.utils.cpp_extension.load() the same way every other
# torch-extension adapter in this repo does (see
# artifacts/gnn-aggregation/stragcn/build.sh), so the compiled .so lands
# under this directory's own build/ (never inside source/, per
# ARTIFACT_GUIDE.md rule 3) and the host compiler is pinned explicitly
# rather than relying on whatever `python setup.py install` would pick up
# from this venv's own compiler_compat shim (the g++-14-vs-GCC7-cc1plus
# issue documented in every other torch-extension STATUS.md here).
#
# Idempotent: exit 0 if the compiled extension is already present and
# FORCE_REBUILD is not set.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
SRC_DIR="$HERE/source/TCGNN_conv"
BUILD_DIR="$HERE/build"

if compgen -G "$BUILD_DIR"/TCGNN*.so > /dev/null && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[tc-gnn] extension already built, skipping (set FORCE_REBUILD=1 to rebuild)"
    exit 0
fi
mkdir -p "$BUILD_DIR"

export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"   # A100 (sm_80); TCGNN_kernel.cu's wmma::precision::tf32 fragments need sm_80+
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"

echo "[tc-gnn] python: $($PY --version), torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "[tc-gnn] host compiler: $($CXX --version | head -1)"
echo "[tc-gnn] nvcc: $(nvcc --version | tail -1)"

# torch.utils.cpp_extension.load() also import-verifies the module it just
# built; this venv's python has a stale libstdc++.so.6 baked into its own
# DT_RPATH (same machine-level issue every torch-extension STATUS.md here
# documents), so the load()-time import needs the same LD_PRELOAD fix, not
# just running the adapter later.
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}"

SRC_DIR="$SRC_DIR" BUILD_DIR="$BUILD_DIR" "$PY" - <<'PYEOF'
import os
import torch.utils.cpp_extension as cpp_ext

src_dir = os.environ["SRC_DIR"]
build_dir = os.environ["BUILD_DIR"]

cpp_ext.load(
    name="TCGNN",
    sources=[os.path.join(src_dir, "TCGNN.cpp"),
             os.path.join(src_dir, "TCGNN_kernel.cu")],
    build_directory=build_dir,
    extra_cuda_cflags=["-O3"],
    verbose=True,
)
PYEOF

echo "[tc-gnn] built $(compgen -G "$BUILD_DIR"/TCGNN*.so)"
