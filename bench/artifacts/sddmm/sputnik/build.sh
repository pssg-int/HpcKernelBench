#!/usr/bin/env bash
# Build Sputnik's own cuda_sddmm.cu.cc (unmodified, via the source/ symlink
# to ../../spmm/sputnik/source) with this directory's wrapper.cu. See
# ../../spmm/sputnik/build.sh's header comment for why Sputnik's own
# CMakeLists.txt (which needs Glog unconditionally) is bypassed entirely.
#
# Idempotent; exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${KB_HOST_COMPILER_BIN:-/opt/cray/pe/gcc-native/12/bin}/g++}"

echo "== Sputnik (sddmm) build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

"$NVCC" -ccbin "$HOST_COMPILER" -std=c++11 -O3 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -I . -I source \
    -x cu \
    -shared -o libsputnik_sddmm_wrapper.so \
    wrapper.cu source/sputnik/sddmm/cuda_sddmm.cu.cc

echo "-- verifying symbols/libs resolve --"
# Same pipefail/SIGPIPE fix as ../../spmm/sputnik/build.sh and
# ../../spmm/rode/build.sh (see those for the full derivation):
# nm's/ldd's dumps piped straight into `grep -q` can trip
# `set -o pipefail` when grep exits early on the first match. Captured to
# variables instead. Build-script verification-step fix only.
LDD_OUT="$(ldd libsputnik_sddmm_wrapper.so)"
if grep -q "not found" <<<"$LDD_OUT"; then
    echo "MISSING SHARED LIBS:"; grep "not found" <<<"$LDD_OUT"; exit 1
fi
NM_OUT="$(nm -D libsputnik_sddmm_wrapper.so)"
for sym in sputnik_sorted_row_swizzle sputnik_sddmm_run; do
  grep -q " T $sym" <<<"$NM_OUT" || { echo "$sym symbol missing"; exit 1; }
done
echo "== Sputnik (sddmm) build: OK =="
