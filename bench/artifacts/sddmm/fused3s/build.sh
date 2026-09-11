#!/usr/bin/env bash
# Build the F3S torch CUDA extension (source/src/setup.py, unmodified) with
# this machine's python env (torch 2.8/cu128) and nvcc 12.9 / sm_80.
#
# Two machine-specific workarounds, neither of which touches artifact code:
#
# 1. The venv's python is built against a NERSC conda env whose sysconfig
#    bakes `CXX = "g++ -pthread -B .../compiler_compat"` into distutils.
#    That `-B` flag causes g++ to pick up an ancient (GCC 7-era) cc1plus from
#    the conda toolchain instead of the real system g++ 14, which then fails
#    torch's `#if __GNUC__ < 9` check. Fix: point CXX/CC at the system
#    compiler directly (bypassing the -B injection).
# 2. The same sysconfig also bakes `-rpath .../nersc-python/lib` into the
#    link step (via LDSHARED), which ships an OLD libstdc++.so.6 missing
#    __cxa_call_terminate (a GCC-8+ symbol the extension needs) and shadows
#    the correct system one at runtime -- and worse, RTLD_GLOBAL-preloading
#    a second libstdc++ at Python runtime to work around it (tried first)
#    SEGFAULTS from the resulting two-copies-of-libstdc++ ABI clash. The
#    real fix is to never bake that rpath in: override LDSHARED with a clean
#    link line (our own compiler, no -B, no nersc-python rpath) that instead
#    rpaths straight to this venv's own torch/lib.
#
# Idempotent: exit 0 if the compiled extension is already present and newer
# than source.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
SRC_DIR="source/src"

if compgen -G "$SRC_DIR"/F3S.cpython-*.so > /dev/null && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[fused3s] extension already built, skipping (set FORCE_REBUILD=1 to rebuild)"
    exit 0
fi

export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"   # A100 (sm_80), matches csrc/Makefile elsewhere in this repo
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export LDFLAGS=""
TORCH_LIB="$("$PY" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LDSHARED="$CXX -pthread -shared -Wl,-rpath,$TORCH_LIB"

echo "[fused3s] python: $($PY --version), torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "[fused3s] host compiler: $($CXX --version | head -1)"
echo "[fused3s] nvcc: $(nvcc --version | tail -1)"

cd "$SRC_DIR"
# Also drop the in-place F3S*.so, not just build/: 'build_ext --inplace' checks
# the final .so's timestamp against the sources and skips recompilation if it's
# newer, so an sm_80 .so left from an A100 build would be reused unchanged and
# then fail on a newer GPU with cudaErrorNoKernelImageForDevice. (We only reach
# here when a build is intended -- the skip-guard above exits first otherwise.)
# The upstream setup.py has no explicit gencode; arch follows TORCH_CUDA_ARCH_LIST.
rm -rf build F3S*.so
"$PY" setup.py build_ext --inplace

echo "[fused3s] built $(compgen -G 'F3S.cpython-*.so')"
