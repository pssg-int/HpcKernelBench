#!/usr/bin/env bash
# Build DTC-SpMM's torch extension (source/DTC-SpMM/{DTCSpMM.cpp,DTCSpMM_kernel.cu})
# via this directory's own dtc_setup.py (see its header comment for why we
# don't use the artifact's own source/DTC-SpMM/setup.py directly -- it wants
# a full prebuilt Sputnik + Glog, neither needed for the code path we wrap).
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): see bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
# This login node's bare "cc"/"c++" resolve to /usr/bin/c++ (g++-7, too old
# for torch's headers -- "GCC 9 or later" hard error); pin CC/CXX explicitly
# so distutils' customize_compiler picks up the modern compiler regardless
# of PATH/alias quirks. Same fix as artifacts/gemv/marlin/build.sh.
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"

echo "== DTC-SpMM build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $PY"

# Sputnik headers only (no CMake build -- see dtc_setup.py). Idempotent.
if [ ! -f source/third_party/sputnik/sputnik/sputnik.h ]; then
    git -C source submodule update --init --depth 1 third_party/sputnik
fi

# build_ext is incremental via build/temp; a stale sm_80 object left from an
# A100 build gets relinked unchanged and then fails at runtime on a newer GPU
# with cudaErrorNoKernelImageForDevice. When a rebuild is forced, drop the
# compiled objects so the current TORCH_CUDA_ARCH_LIST arch actually takes
# effect (build-system only; TORCH_CUDA_ARCH_LIST already parameterized).
[ "${FORCE_REBUILD:-0}" = 1 ] && rm -rf build/temp build/lib
TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}" LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" \
    "$PY" dtc_setup.py build_ext --build-lib build/lib --build-temp build/temp

ls build/lib/DTCSpMM*.so >/dev/null
echo "== DTC-SpMM build: OK =="
