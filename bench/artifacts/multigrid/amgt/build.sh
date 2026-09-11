#!/usr/bin/env bash
# Build ONLY libamgtmg_wrapper.so (mg_wrapper.cu) against the ALREADY-BUILT
# HYPRE from ../../spgemm/amgt/ (source/ here is a symlink to that build's
# source/ -- see STATUS.md). HYPRE itself is NEVER reconfigured/rebuilt by
# this script: mg_wrapper.cu calls only HYPRE's own public BoomerAMG/IJ API,
# already compiled into that build's libHYPRE.so (verified via `nm -D`, see
# STATUS.md) -- no AmgT-specific patch is needed for this path.
#
# Idempotent: safe to re-run (one .cu file, always re-links, cheap). Exit
# 0 = built. If ../../spgemm/amgt/'s libHYPRE.so is ever missing/broken,
# this script fails loudly rather than silently trying to rebuild HYPRE
# itself (rebuilding that shared dependency is spgemm/amgt/build.sh's job,
# not this artifact's -- see the task brief's "do not rebuild it unless
# broken" instruction).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}"
MATHLIB="${MATHLIB:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib}"
MPI_INC="${MPI_INC:-/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include}"
MPI_LIB="${MPI_LIB:-/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib}"
MPICH_CXX_HOST="${MPICH_CXX_HOST:-${KB_GXX12:-g++-12}}"  # matches ../../spgemm/amgt/build.sh's
                                             # own host-compiler choice; only
                                             # used here for `mpicxx -show`
                                             # sanity-printing, not the nvcc
                                             # host-compiler flag below.

SRC_HYPRE="source/AmgT_HYPRE/src"
HYPRE_DIR="$(pwd)/$SRC_HYPRE/hypre"

echo "== multigrid/amgt build (BoomerAMG wrapper; reuses spgemm/amgt's HYPRE) =="
echo "source symlink -> $(readlink -f source 2>/dev/null || echo MISSING)"
echo "nvcc: $($NVCC --version | tail -1)"

if [ ! -f "$HYPRE_DIR/lib/libHYPRE.so" ]; then
    echo "FATAL: $HYPRE_DIR/lib/libHYPRE.so not found." >&2
    echo "This artifact reuses artifacts/spgemm/amgt/'s HYPRE build; run" >&2
    echo "  artifacts/spgemm/amgt/build.sh" >&2
    echo "first (do NOT rebuild it from here)." >&2
    exit 1
fi

echo "-- verifying the public BoomerAMG/IJ API this wrapper needs is present --"
# nm's output is captured into a variable, then matched via a HERE-STRING
# (grep -q pattern <<< "$var"), never a live pipe into `grep -q`: grep -q
# exits the instant it finds a match, and a real PRODUCER PROCESS on the
# other end of a pipe (nm itself, or even `echo "$var"` re-piped) gets
# SIGPIPE if it still has unwritten output queued -- which `nm`'s ~500KB of
# symbols (this binary also exports HYPRE's CUB/thrust template
# instantiations) reliably exceeds one pipe buffer, so it reliably
# triggers. Under `set -o pipefail` bash then reports the WHOLE pipeline
# as failed even though grep itself matched -- reproduced for real while
# writing this script (both the direct `nm | grep -q` form AND the
# variable-capture-then-`echo | grep -q` form hit it identically). A
# here-string has bash write the full variable into a pipe/temp-fd
# SYNCHRONOUSLY before grep ever starts reading, so there is no live
# writer process left to receive SIGPIPE.
_hypre_syms="$(nm -D "$HYPRE_DIR/lib/libHYPRE.so")"
for sym in HYPRE_BoomerAMGCreate HYPRE_BoomerAMGSetup HYPRE_BoomerAMGSolve \
          HYPRE_BoomerAMGSetMaxIter HYPRE_BoomerAMGSetTol \
          HYPRE_BoomerAMGGetNumIterations HYPRE_BoomerAMGGetFinalRelativeResidualNorm \
          HYPRE_IJMatrixCreate HYPRE_IJMatrixSetValues HYPRE_IJMatrixAssemble \
          HYPRE_IJVectorCreate HYPRE_IJVectorSetValues HYPRE_IJVectorAssemble; do
    grep -q " T $sym\$" <<< "$_hypre_syms" || {
        echo "FATAL: symbol $sym missing from libHYPRE.so" >&2
        exit 1
    }
done
echo "   all required symbols present (no HYPRE rebuild needed)"

echo "-- building libamgtmg_wrapper.so --"
"$NVCC" -ccbin=mpicxx -O3 -std=c++14 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -I"$HYPRE_DIR/include" \
    -I"$MPI_INC" \
    -shared -o libamgtmg_wrapper.so mg_wrapper.cu \
    -L"$HYPRE_DIR/lib" -lHYPRE \
    -L"$CUDA_HOME/lib64" -lcudart \
    -L"$MATHLIB" -lcublas -lcusparse -lcurand \
    -L"$MPI_LIB" -l"${KB_MPI_LIBNAME:-mpi_gnu_123}" \
    -lm -lstdc++ \
    -Xlinker -rpath -Xlinker "$CUDA_HOME/lib64" \
    -Xlinker -rpath -Xlinker "$MATHLIB" \
    -Xlinker -rpath -Xlinker "$MPI_LIB" \
    -Xlinker -rpath -Xlinker "$HYPRE_DIR/lib"

echo "-- verifying wrapper symbols --"
# same here-string discipline as above (pipefail/grep-q SIGPIPE trap)
_wrapper_syms="$(nm -D libamgtmg_wrapper.so)"
grep -q " T amgmg_init" <<< "$_wrapper_syms" || { echo "MISSING amgmg_init"; exit 1; }
grep -q " T amgmg_boomeramg_solve" <<< "$_wrapper_syms" || { echo "MISSING amgmg_boomeramg_solve"; exit 1; }
_ldd_out="$(ldd libamgtmg_wrapper.so)"
if grep -q "not found" <<< "$_ldd_out"; then
    echo "MISSING SHARED LIBS:"; grep "not found" <<< "$_ldd_out"; exit 1
fi
echo "== multigrid/amgt build: OK =="
