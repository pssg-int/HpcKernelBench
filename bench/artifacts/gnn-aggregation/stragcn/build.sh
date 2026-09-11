#!/usr/bin/env bash
# Build StraGCN's GCN_ST torch CUDA extension (source/StraGCN/GNN_strassen.cpp
# + source/StraGCN/strassen.cu, both UNMODIFIED) with this machine's python
# env (torch 2.8/cu128) and nvcc 12.9 / sm_80.
#
# source/StraGCN/setup.py is NOT executed: it hardcodes
#   os.environ["CC"] = "gcc-10"; os.environ["CXX"] = "g++-10"
# at import time, which would silently override any shell-level CC/CXX we
# set (neither gcc-10 nor g++-10 exists on this machine) -- there is no way
# to honor that setup.py's own compiler choice here, so instead of patching
# it we bypass it entirely and drive torch's own JIT cpp_extension loader
# directly against the identical, unmodified .cpp/.cu source files. This is
# a build-driver substitution, not a kernel-code patch (ARTIFACT_GUIDE.md
# rule 3: "build-system fixes ... are fine; touching kernel code is not").
#
# Same host-compiler override fused3s's build.sh documents: this venv's
# python (layered on a NERSC conda env) bakes a `-B .../compiler_compat`
# flag into distutils that makes g++ pick up an ancient GCC-7-era cc1plus,
# so CXX/CC are pointed at the real system compiler directly.
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
SRC_DIR="$HERE/source/StraGCN"
BUILD_DIR="$HERE/build"

if compgen -G "$BUILD_DIR"/GCN_ST*.so > /dev/null && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[stragcn] extension already built, skipping (set FORCE_REBUILD=1 to rebuild)"
    exit 0
fi
mkdir -p "$BUILD_DIR"

export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"   # A100 (sm_80), matches bench/csrc/Makefile elsewhere in this repo
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"

echo "[stragcn] python: $($PY --version), torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "[stragcn] host compiler: $($CXX --version | head -1)"
echo "[stragcn] nvcc: $(nvcc --version | tail -1)"

# torch.utils.cpp_extension.load() also import-verifies the module it just
# built; this venv's python has a stale libstdc++.so.6 baked into its own
# DT_RPATH (same machine-level issue bench/artifacts/sddmm/fused3s/STATUS.md
# documents in full), so the load()-time import needs the same
# LD_PRELOAD fix, not just running the adapter later.
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}"

SRC_DIR="$SRC_DIR" BUILD_DIR="$BUILD_DIR" "$PY" - <<'PYEOF'
import os
import torch.utils.cpp_extension as cpp_ext

src_dir = os.environ["SRC_DIR"]
build_dir = os.environ["BUILD_DIR"]

cpp_ext.load(
    name="GCN_ST",
    sources=[os.path.join(src_dir, "GNN_strassen.cpp"),
             os.path.join(src_dir, "strassen.cu")],
    build_directory=build_dir,
    extra_cuda_cflags=["-O3"],
    verbose=True,
)
PYEOF

echo "[stragcn] built $(compgen -G "$BUILD_DIR"/GCN_ST*.so)"
