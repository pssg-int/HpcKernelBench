#!/usr/bin/env bash
# RASSM has no library target -- source/code/CMakeLists.txt only builds an
# end-to-end CLI executable (`rassm`) and a separate ASpT-baseline
# executable (`aspt`, requires Intel MKL, not needed here -- see STATUS.md).
# We still build `rassm` itself (as a build-viability / provenance check,
# and because its object files/headers are exactly what the wrapper links
# against), then compile `wrapper.cpp` (this directory, NOT under source/)
# into a small shared library ctypes can call -- see wrapper.cpp's header
# comment for why a wrapper is necessary and what it does.
#
# Idempotent: safe to re-run; exit 0 = both built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

CXX="${KB_GXX12:-/usr/bin/g++-12}"   # system default g++-14 (SUSE) is untested
                                 # here; g++-12 matches what got InferFast's
                                 # nvcc-paired build working on this machine
                                 # and RASSM's own CMakeLists.txt GNU branch
                                 # targets c++17 the same way.

echo "== RASSM build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "compiler: $($CXX --version | head -1)"
echo "boost: $(rpm -q libboost_headers1_66_0-devel 2>/dev/null || echo 'system boost, see STATUS.md')"

echo "-- 1/2: source/build/rassm (RASSM's own CMake build, GNU branch; provenance/build-viability check only -- the adapter never calls this binary, see step 2) --"
mkdir -p source/build
cmake -S source/code -B source/build \
    -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="${CXX/g++/gcc}" \
    >/dev/null
# CMakeLists.txt's `if(Boost_FOUND)` guard means the `rassm`/`aspt` targets
# simply don't exist (not a `make` error) when Boost isn't found -- true on
# a machine with no system/conda Boost at all (checked: no libboost* via
# ldconfig, no boost-cpp in any conda env here). Boost is used ONLY for
# main.cpp's CLI (program_options/iostreams/serialization) -- grep confirms
# wrapper.cpp and the .cpp files it links (common/global/util.cpp) never
# reference boost, so step 2 below (the actual functional-gate target,
# librassm_wrapper.so) does not need it. Skip step 1 gracefully instead of
# failing the whole build when Boost is absent; still attempt it (and cap
# parallelism, see below) when Boost IS present, as on Perlmutter.
if [ -f source/build/Makefile ] && grep -q "^rassm:" source/build/Makefile 2>/dev/null; then
    # only the `rassm` target -- `aspt` (ASpT baseline) needs mkl.h, unrelated
    # to RASSM's own SpMM kernel, not built here (see STATUS.md).
    # Parallelism capped well below nproc: shared login node, hard RLIMIT_NPROC
    # (256 processes) across the whole user session -- -j"$(nproc)" (128 here)
    # reliably exhausts it mid-build ("cc1plus: vfork: Resource temporarily
    # unavailable"), a resource-contention failure, not a RASSM problem.
    # Override with RASSM_BUILD_JOBS on a dedicated machine.
    make -C source/build rassm -j"${RASSM_BUILD_JOBS:-8}"
else
    echo "NOTE: Boost not found on this machine -- CMakeLists.txt's" \
         "if(Boost_FOUND) guard means the 'rassm' CLI target was never" \
         "generated. Skipping this provenance-only build; the functional" \
         "gate (librassm_wrapper.so, step 2) does not depend on it or on Boost."
fi

echo "-- 2/2: librassm_wrapper.so (adapter glue, see wrapper.cpp) --"
# Linked together with RASSM's own non-template .cpp sources (everything
# under source/code/src/ except main.cpp, which defines main() and would
# conflict): CSC.h/DCSC.h call compare2() (utils/util.h), only DEFINED in
# util.cpp, not header-only like the rest of what wrapper.cpp uses -- these
# are the artifact's own translation units, not reimplemented here.
"$CXX" -O3 -std=c++17 -march=core-avx2 -fopenmp -fPIC -shared \
    -Wno-write-strings \
    -I source/code/include \
    -DGCC_COMPILER=1 \
    -o librassm_wrapper.so wrapper.cpp \
    source/code/src/common.cpp source/code/src/global.cpp source/code/src/util.cpp

echo "-- verifying symbols --"
nm -D librassm_wrapper.so | grep -q " T rassm_prepare" || { echo "MISSING rassm_prepare"; exit 1; }
nm -D librassm_wrapper.so | grep -q " T rassm_run" || { echo "MISSING rassm_run"; exit 1; }
ldd librassm_wrapper.so | grep -q "not found" && \
    { echo "MISSING SHARED LIBS:"; ldd librassm_wrapper.so | grep "not found"; exit 1; }
echo "== RASSM build: OK =="
