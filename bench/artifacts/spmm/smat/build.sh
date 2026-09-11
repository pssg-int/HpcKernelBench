#!/usr/bin/env bash
# Build this directory's own wrapper.cu + SMaT's unmodified mmaCBT.cu kernel
# into a shared library.
#
# NOTE on ARTIFACT_GUIDE.md's anticipated gflags fix: SMaT's own CMake build
# (source/src/cuda_hgemm/{CMakeLists.txt,build.sh}) needs system `gflags`
# (find_package(gflags REQUIRED)), unavailable here (no sudo). This build.sh
# does NOT use that CMake build at all: wrapper.cu (see its header comment)
# links only source/.../common/matrix.h (header-only, zero gflags/OpenMP/
# cublas dependency -- confirmed by tracing every #include reachable from it)
# and source/.../mma/mmaCBT.cu (SMaT's real, unmodified kernel -- the ONLY
# one main.cu's benchmark actually calls; the sibling mma*.cu kernels are
# commented out there). gflags is only used by main.cu's CLI-flag parsing,
# which this integration does not link at all (rule 1: wrap the kernel, not
# the CLI driver) -- so vendoring gflags turned out to be unnecessary once
# wrapped at this finer boundary. If a future task needs the full `hgemm` CLI
# binary (e.g. to cross-check this wrapper's numbers against SMaT's own
# driver), see STATUS.md's "Next steps" for the vendor-gflags recipe that was
# scoped but not needed here.
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}}"
SRC="source/src/cuda_hgemm/src"

echo "== SMaT build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

"$NVCC" -ccbin "$HOST_COMPILER" -O3 -std=c++17 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -I "$SRC/common" \
    -shared -o libsmat_wrapper.so wrapper.cu "$SRC/mma/mmaCBT.cu"

echo "-- verifying symbols resolve --"
ldd libsmat_wrapper.so | grep -q "not found" && \
    { echo "MISSING SHARED LIBS:"; ldd libsmat_wrapper.so | grep "not found"; exit 1; }
nm -D libsmat_wrapper.so | grep -q " T smat_run" || { echo "smat_run symbol missing"; exit 1; }
echo "== SMaT build: OK =="
