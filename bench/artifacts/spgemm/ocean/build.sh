#!/usr/bin/env bash
# Build Ocean's own CSR/SpGEMM sources (source/src/CSR.cpp, source/src/Utils.cpp
# -- unmodified, source/src/main.cu's CLI is skipped) together with this
# directory's wrapper.cu (see its header comment) into one shared library
# ctypes can load. No file under source/ is edited.
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"   # nvcc 12.9 rejects the
    # system default g++-14 (SUSE) on <bits/alloc_traits.h> with
    # "identifier __has_construct is undefined" -- a host-compiler/libstdc++
    # version mismatch, not an Ocean problem (same fix already used by the
    # spmm/inferfast and spmm/rassm adapters in this repo). g++-12 builds
    # clean; this is a build-system fix (compiler selection), not a
    # kernel-code patch (ARTIFACT_GUIDE.md rule 3).

echo "== Ocean build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

# Same flags as source/Makefile (CFLAGS/NVFLAGS/INCLUDES), Compute_Capability
# already sm_80 in that Makefile (unused here -- we invoke nvcc ourselves so
# the CLI/convert binaries never get built, only the objects we need).
CFLAGS="-std=c++14 -O3"
NVFLAGS="-gencode=arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} -Xptxas=-v -O3 -Xcompiler -Wall,-fPIC -D_FORCE_INLINES --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr"
INCLUDES="-Isource/include -Isource/kernels"

echo "-- compiling source/src/CSR.cpp + source/src/Utils.cpp + wrapper.cu --"
"$NVCC" -ccbin "$HOST_COMPILER" $CFLAGS $NVFLAGS $INCLUDES \
    -shared -o libocean_wrapper.so \
    source/src/CSR.cpp source/src/Utils.cpp wrapper.cu

echo "-- verifying symbols resolve --"
# Capture ldd/nm output to variables rather than piping straight into grep -q:
# on a contended login node (RLIMIT_NPROC), a `nm ... | grep -q` pipeline can
# spuriously fail (nm cannot fork, or gets SIGPIPE when grep exits on first
# match), which under `set -e`/pipefail aborts the build even though the .so
# built fine. Same build-script fix already applied to spmm/rode & spmm/sputnik.
LDD_OUT="$(ldd libocean_wrapper.so)"
if grep -q "not found" <<<"$LDD_OUT"; then
    echo "MISSING SHARED LIBS:"; grep "not found" <<<"$LDD_OUT"; exit 1
fi
NM_OUT="$(nm -D libocean_wrapper.so)"
grep -q "ocean_spgemm_run" <<<"$NM_OUT" || \
    { echo "ocean_spgemm_run symbol missing from libocean_wrapper.so"; exit 1; }

echo "== Ocean build: OK =="
