#!/usr/bin/env bash
# Build FlashSparse's FS_SDDMM (SDDMM kernel) + FS_Block (CPU load-balanced
# preprocessing for SDDMM) torch extensions via this directory's own
# fs_sddmm_setup.py -- see its header comment. `source/` here is a SYMLINK
# to ../../spmm/flashsparse/source (read-only reuse, not re-cloned).
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): see bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

if [ ! -e source/FlashSparse/SDDMM/src/sddmmKernel.cu ]; then
    echo "source/ symlink missing or broken (expected -> ../../spmm/flashsparse/source)" >&2
    exit 1
fi

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
# This login node's bare "cc"/"c++" resolve to /usr/bin/c++ (g++-7, too old
# for torch's headers); pin CC/CXX explicitly, same fix as
# ../../spmm/flashsparse/build.sh.
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"

echo "== FlashSparse SDDMM build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $PY"

TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}" LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" \
    "$PY" fs_sddmm_setup.py build_ext --build-lib build/lib --build-temp build/temp

ls build/lib/FS_SDDMM*.so build/lib/FS_Block*.so >/dev/null
echo "== FlashSparse SDDMM build: OK =="
