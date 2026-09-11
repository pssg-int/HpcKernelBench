#!/usr/bin/env bash
# Build GraphSet's k-clique-counting kernel (gpu_pattern_matching, driven by
# a K_k Pattern + Schedule_IEP, source/gpu/gpu_clique.cu) as a shared library
# (gs_shim.so), compiled from gs_shim.cu (this dir, NOT part of the
# artifact) together with the artifact's own gpu_clique.cu (unmodified) and
# the small set of host-side .cpp files (Pattern/Schedule_IEP/VertexSet/
# Graph/etc.) that back them.
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"

CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
# Cray MPICH: source/src/graph.cpp and graphmpi.cpp #include <mpi.h>
# unconditionally (Graph's own MPI-based pattern-matching path, unused by
# this adapter's single-GPU clique count but part of the same translation
# unit as the functions we DO need -- graph.cpp's `reduce_edges_for_clique`/
# `erase_edge`). Build-system dependency, not a kernel-code patch.
MPI_INC="${MPI_INC:-${KB_MPI_ROOT:-/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3}/include}"   # exported by toolchain.sh from KB_MPI_ROOT
MPI_LIB="${MPI_LIB:-${KB_MPI_ROOT:-/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3}/lib}"

echo "=== GraphSet build (gs_shim.so, sm_80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $($NVCC --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

# -rdc=true (relocatable device code): gs_shim.cu launches gpu_clique.cu's
# own `gpu_pattern_matching` __global__ kernel and reads/resets its
# `dev_sum`/`dev_cur_edge` __device__ globals via `extern` declarations --
# by default each .cu file's device code is compiled independently with no
# cross-TU device-symbol visibility; -rdc=true performs a device link step
# that resolves these, needing zero changes to gpu_clique.cu itself.
"$NVCC" -std=c++14 -O3 -w -arch=sm_80 -rdc=true \
    -ccbin "$CXX" \
    --extended-lambda --expt-relaxed-constexpr \
    -Xcompiler=-fopenmp,-fPIC,-w,-march=native \
    -DTHRUST_IGNORE_CUB_VERSION_CHECK \
    -I "$HERE" \
    -I "$SRC/include" \
    -I "$SRC/gpu" \
    -I "$MPI_INC" \
    -shared -o "$HERE/gs_shim.so" \
    "$HERE/gs_shim.cu" \
    "$SRC/gpu/gpu_clique.cu" \
    "$SRC/src/pattern.cpp" \
    "$SRC/src/prefix.cpp" \
    "$SRC/src/schedule_IEP.cpp" \
    "$SRC/src/vertex_set.cpp" \
    "$SRC/src/set_operation.cpp" \
    "$SRC/src/common.cpp" \
    "$SRC/src/disjoint_set_union.cpp" \
    "$SRC/src/motif_generator.cpp" \
    "$SRC/src/graphmpi.cpp" \
    "$SRC/src/graph.cpp" \
    "$SRC/src/dataloader.cpp" \
    "$SRC/src/labeled_graph.cpp" \
    -L "$MPI_LIB" -l"${KB_MPI_LIBNAME:-mpi_gnu}" -lgomp

if [[ ! -f "$HERE/gs_shim.so" ]]; then
    echo "BUILD FAILED: gs_shim.so not produced" >&2
    exit 1
fi
nm -D "$HERE/gs_shim.so" | grep -q "gs_run" || {
    echo "gs_run symbol missing from gs_shim.so" >&2
    exit 1
}
echo "Built: $HERE/gs_shim.so"
echo "=== done ==="
