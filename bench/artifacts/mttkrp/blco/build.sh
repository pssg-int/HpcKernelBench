#!/usr/bin/env bash
# Build BLCO's own MTTKRP sources (source/src/{alto_dev,blco,utils}.cu,
# source/src/{sptensor,common}.cpp, source/src/kruskal_model.{cpp,cu}
# -- all unmodified) together with this directory's wrapper.cu (see its
# header comment for exact provenance of every call) into one shared
# library ctypes can load. No file under source/ is edited.
#
# Deliberately NOT using source/Makefile: that Makefile's default target
# builds `cpd64` -- an end-to-end CLI binary that also links main.cpp,
# cpd_gpu.cu (CP-ALS, needs cuBLAS/cuSOLVER + MKL/OpenBLAS for the
# pseudoinverse step), poisson_generator.cpp/rng*.cpp (synthetic-tensor
# generation). None of that is needed to invoke BLCO's own isolated MTTKRP
# entry point (mttkrp_alto_dev_onemode<IType>, source/include/alto_dev.hpp)
# directly, so this script compiles only the object set that function
# actually depends on -- notably avoiding any MKL/OpenBLAS dependency
# entirely (grep confirms alto_dev.cu/blco.cu/kruskal_model.{cpp,cu}/
# sptensor.cpp/common.cpp/utils.cu never call a single cuBLAS/cuSOLVER/BLAS
# function; utils.cu's check_cublas/check_cusolver only switch on enum
# values from the headers, never call into the libraries).
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
    # version mismatch, not a BLCO problem (same fix already used by the
    # spmm/inferfast, spmm/rassm, and spgemm/ocean adapters in this repo).

echo "== BLCO build (isolated MTTKRP entry point, no CPD/cuBLAS/BLAS) =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

# Defines mirror source/config.mk's own defaults exactly (ALTO_MASK_LENGTH=64
# -- the "cpd64" build, i.e. LIT=IType=unsigned long long throughout;
# MAX_NUM_MODES=5; MODES_SPECIALIZED=RANKS_SPECIALIZED=0 i.e. "no
# specialization"; ALTERNATIVE_PEXT=true, REQUIRED for GPU code --
# source/include/alto_dev.hpp itself #errors without -DALT_PEXT). No -DMKL,
# no USE_*BLAS*_ define -- see header comment, none of the object files we
# compile need a BLAS symbol.
DEFINES="-DALT_PEXT -DALTO_MASK_LENGTH=64 -DMAX_NUM_MODES=5 -DALTO_MODES_SPECIALIZED=0 -DALTO_RANKS_SPECIALIZED=0"

# -Xcompiler -fopenmp: alto.hpp's gen_alto<LIT> (COO->ALTO linearization,
# called from gen_blcotensor_host) uses `#pragma omp parallel for` and
# omp_get_wtime() -- needed for both compile (pragma expansion) and link
# (libgomp), matching source/mk/include_GCC.mk's own $(OPENMP) = -fopenmp.
NVCCFLAGS="-std=c++17 -O3 -Xcompiler -fPIC,-fopenmp -gencode=arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80}"
INCLUDES="-Isource/include"

SOURCES="source/src/common.cpp source/src/sptensor.cpp source/src/utils.cu \
source/src/kruskal_model.cpp source/src/kruskal_model.cu \
source/src/blco.cu source/src/alto_dev.cu wrapper.cu"

echo "-- compiling $SOURCES --"
"$NVCC" -ccbin "$HOST_COMPILER" $DEFINES $NVCCFLAGS $INCLUDES \
    -shared -o libblco_wrapper.so $SOURCES

echo "-- verifying symbols resolve --"
ldd libblco_wrapper.so | grep -q "not found" && \
    { echo "MISSING SHARED LIBS:"; ldd libblco_wrapper.so | grep "not found"; exit 1; }
nm -D libblco_wrapper.so | grep -q "blco_run" || \
    { echo "blco_run symbol missing from libblco_wrapper.so"; exit 1; }

echo "== BLCO build: OK =="
