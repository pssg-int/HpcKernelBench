#!/usr/bin/env bash
# Build Sputnik's own cuda_spmm.cu.cc + cuda_sddmm.cu.cc (unmodified) with
# this directory's wrapper.cu (ctypes glue + a byte-for-byte port of
# SortedRowSwizzle -- see wrapper.cu's header comment for why that one
# function is reproduced rather than compiled from matrix_utils.cu.cc,
# which needs Glog/Abseil for unrelated test/benchmark data-generation code
# neither the kernel nor SortedRowSwizzle itself needs).
#
# Deliberately bypasses Sputnik's own CMakeLists.txt entirely (same
# "hand-compile just the needed .cu files" approach as rode/sspmm/nm-spmm's
# build.sh): CMakeLists.txt's `find_package(Glog REQUIRED)` is unconditional
# for the whole project even though the actual .cu.cc files this adapter
# calls never reference glog -- vendoring Glog just to satisfy an unused
# CMake dependency would add real build fragility for zero benefit here.
#
# Idempotent; exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
# g++-14 (this repo's default $CXX) trips nvcc 12.9's known
# "__has_construct"/<bits/alloc_traits.h> incompatibility on any translation
# unit including <unordered_map>/<functional> (cuda_spmm.cu.cc does) -- same
# issue documented in ../inferfast, ../mp-spmm, ../sspmm, ../nm-spmm build.sh.
HOST_COMPILER="${HOST_COMPILER:-${KB_HOST_COMPILER_BIN:-/opt/cray/pe/gcc-native/12/bin}/g++}"

echo "== Sputnik build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

# nvcc identifies source language by extension; Sputnik's own files use
# `.cu.cc` (CMakeLists.txt forces LANGUAGE CUDA via set_source_files_properties,
# which a plain nvcc invocation has no equivalent for) -- `-x cu` makes nvcc
# treat them as CUDA regardless of extension (same fix SSpMM's build.sh uses
# for its own .cu file, here needed for the extension itself, not the flag).
"$NVCC" -ccbin "$HOST_COMPILER" -std=c++11 -O3 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -I . -I source \
    -x cu \
    -shared -o libsputnik_wrapper.so \
    wrapper.cu source/sputnik/spmm/cuda_spmm.cu.cc source/sputnik/sddmm/cuda_sddmm.cu.cc

echo "-- verifying symbols/libs resolve --"
# Captured to variables rather than piped straight into grep -q: under
# `set -o pipefail`, grep -q closes its read end after the first match, and
# nm's/ldd's symbol dumps can be large enough that the producer is still
# writing when that happens, so it gets SIGPIPE (exit 141) and pipefail then
# fails the whole pipeline even when grep itself matched correctly (observed
# on this build -- see bench/artifacts/spmm/rode/build.sh for the same fix
# with the full derivation). Build-script verification-step fix only, not a
# source/ change.
LDD_OUT="$(ldd libsputnik_wrapper.so)"
if grep -q "not found" <<<"$LDD_OUT"; then
    echo "MISSING SHARED LIBS:"; grep "not found" <<<"$LDD_OUT"; exit 1
fi
NM_OUT="$(nm -D libsputnik_wrapper.so)"
for sym in sputnik_sorted_row_swizzle sputnik_spmm_run sputnik_sddmm_run; do
  grep -q " T $sym" <<<"$NM_OUT" || { echo "$sym symbol missing"; exit 1; }
done
echo "== Sputnik build: OK =="
