#!/usr/bin/env bash
# Build FlashSparse's FS_SpMM (SpMM kernel) + FS_Block_gpu (GPU ME-BCRS-style
# preprocessing) torch extensions via this directory's own fs_setup.py --
# see its header comment for why (only 2 of the artifact's 4 extensions are
# needed for this spmm-track integration; FS_SDDMM/FS_Block are skipped).
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): see bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
# This login node's bare "cc"/"c++" resolve to /usr/bin/c++ (g++-7, too old
# for torch's headers); pin CC/CXX explicitly, same fix as
# artifacts/gemv/marlin/build.sh and artifacts/spmm/dtcspmm/build.sh.
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"

echo "== FlashSparse build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $PY"

# build_ext is incremental via build/temp; a stale sm_80 object left from an
# A100 build gets relinked unchanged and then fails at runtime on a newer GPU
# with cudaErrorNoKernelImageForDevice. When a rebuild is forced, drop the
# compiled objects so the current TORCH_CUDA_ARCH_LIST arch actually takes
# effect (build-system only; TORCH_CUDA_ARCH_LIST already parameterized).
[ "${FORCE_REBUILD:-0}" = 1 ] && rm -rf build/temp build/lib
TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}" LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" \
    "$PY" fs_setup.py build_ext --build-lib build/lib --build-temp build/temp

ls build/lib/FS_SpMM*.so build/lib/FS_Block_gpu*.so >/dev/null
echo "== FlashSparse build: OK =="
