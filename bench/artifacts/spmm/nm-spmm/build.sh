#!/usr/bin/env bash
# Build NM-SpMM's own kernel_32x32_4x4.cu + preprocessing.cu (unmodified)
# together with this directory's wrapper.cu (ctypes glue, see wrapper.cu's
# header comment for scope: only the 32x32 tile / low-sparsity=0.5 path is
# wired). Idempotent; exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
# g++-14 (this repo's default $CXX) trips nvcc 12.9's known
# "__has_construct"/<bits/alloc_traits.h> incompatibility on ANY translation
# unit that includes <iostream> (NM-SpMM.h does) -- same issue documented in
# ../inferfast/build.sh, ../mp-spmm/build.sh, ../sspmm/build.sh. Use gcc 12
# (KB_HOST_COMPILER_BIN) as nvcc's -ccbin instead.
HOST_COMPILER="${HOST_COMPILER:-${KB_HOST_COMPILER_BIN:-/opt/cray/pe/gcc-native/12/bin}/g++}"

echo "== NM-SpMM build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

# NM-SpMM's own CMakeLists.txt targets sm_80 explicitly (arch=sm_80 in
# CUDA_NVCC_FLAGS) and uses cp.async (Ampere-generation async-copy PTX,
# source/include/ptx.h) -- no Hopper-only feature, builds fine on this
# machine's A100. -std=c++11 matches the artifact's own CMakeLists.txt.
# Only kernel_32x32_4x4.cu + preprocessing.cu are compiled (see wrapper.cu
# header for why the other 3 kernel-size files and utils.h/tests/ are not
# needed for the slice this adapter wires); this also means the artifact's
# own OpenMP dependency (utils.h's nmGEMM_on_cpu reference, tests/ only) is
# never pulled in -- the harness's own scipy reference is used instead.
"$NVCC" -ccbin "$HOST_COMPILER" -std=c++11 -O3 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -I source/include \
    -shared -o libnmspmm_wrapper.so \
    wrapper.cu source/src/kernel_32x32_4x4.cu source/src/preprocessing.cu

echo "-- verifying symbols/libs resolve --"
ldd libnmspmm_wrapper.so | grep -q "not found" && \
    { echo "MISSING SHARED LIBS:"; ldd libnmspmm_wrapper.so | grep "not found"; exit 1; }
nm -D libnmspmm_wrapper.so | grep -q " T nm_gemm_32x32_low" || \
    { echo "nm_gemm_32x32_low symbol missing"; exit 1; }
nm -D libnmspmm_wrapper.so | grep -q " T nm_preprocess_low" || \
    { echo "nm_preprocess_low symbol missing"; exit 1; }
echo "== NM-SpMM build: OK =="
