#!/usr/bin/env bash
# Build AmgT's own HYPRE fork (its SpGEMM/SpMV kernels are patched into
# AmgT_HYPRE/src/seq_mv/), then a thin ctypes-facing wrapper exposing the
# standalone spgemm_amgT_fp64() kernel entry point directly (see wrapper.cu's
# header comment for the full "why" -- it touches zero lines of source/).
#
# Idempotent: safe to re-run. `./configure` is always re-run (cheap, ~15s,
# regenerates config files deterministically -- same pattern as
# rassm/build.sh's `cmake -S ... -B ...`); `make install` is GNU make's own
# incremental build (only recompiles changed objects); the final wrapper
# link always re-links (cheap, one .cu file). Exit 0 = both built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}"
MATHLIB="${MATHLIB:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib}"
MPI_INC="${MPI_INC:-${KB_MPI_ROOT:-/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3}/include}"   # exported by toolchain.sh from KB_MPI_ROOT
MPI_LIB="${MPI_LIB:-${KB_MPI_ROOT:-/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3}/lib}"
KB_MPI_LIBNAME="${KB_MPI_LIBNAME:-mpi_gnu_123}"   # Cray MPICH GNU-ABI library name; plain "mpi" elsewhere (toolchain.sh detects)
    # nvcc's own toolkit dir under hpc_sdk/.../cuda/12.9 ships libcudart but
    # not libcublas/libcusparse/libcurand -- those live under the sibling
    # math_libs/12.9 tree (same finding as spmm/inferfast/build.sh; located
    # empirically via `find -iname libcublas.so*` there, reused here).
MPICH_CC_HOST="${MPICH_CC_HOST:-${KB_GCC12:-gcc-12}}"     # underlying compiler mpicc/mpicxx
MPICH_CXX_HOST="${MPICH_CXX_HOST:-${KB_GXX12:-g++-12}}"   # invoke (via MPICH_CC/MPICH_CXX
                                              # env vars) -- the system default
                                              # gcc/g++-14 (SUSE) is untested
                                              # against nvcc 12.9's host-compiler
                                              # support here; g++-12/gcc-12
                                              # (available at /usr/bin) is the
                                              # same version that got
                                              # spmm/inferfast's nvcc build
                                              # working on this machine.
export MPICH_CC="$MPICH_CC_HOST"
export MPICH_CXX="$MPICH_CXX_HOST"
export CUDA_HOME MATHLIB

SRC_HYPRE="source/AmgT_HYPRE/src"
HYPRE_DIR="$(pwd)/$SRC_HYPRE/hypre"

echo "== AmgT build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "mpicc(host=$MPICH_CC_HOST): $(MPICH_CC=$MPICH_CC_HOST mpicc -show 2>/dev/null | head -c200)"
echo "mpicxx(host=$MPICH_CXX_HOST): $(MPICH_CXX=$MPICH_CXX_HOST mpicxx -show 2>/dev/null | head -c200)"

echo "-- 0/3: build-system fixes (permission bits only, see STATUS.md) --"
# Several of HYPRE's own helper scripts (autoconf-generated `configure`,
# `tarch`, and config/{mkinstalldirs,install-sh,*.sh}) lost their executable
# bit somewhere in this clone's transfer to this machine (git clone/tarball
# extraction quirk, unrelated to AmgT's own content -- `git diff` on any of
# these files is empty, only the mode bit changed). Restoring +x is a
# build-system fix (ARTIFACT_GUIDE.md rule 3: "Build-system fixes... are
# fine"), not a content patch.
chmod +x "$SRC_HYPRE/configure" "$SRC_HYPRE/tarch" \
    "$SRC_HYPRE/config/mkinstalldirs" "$SRC_HYPRE/config/install-sh" \
    "$SRC_HYPRE/config"/*.sh 2>/dev/null || true

echo "-- 1/3: select the AmgT_FP64 build (artifact's own documented step) --"
# The artifact's own README: "Select the compilation version by changing the
# execuative in compile.sh" -- done here via the same header-swap compile.sh
# itself performs (`cp config_files/${execuative}.h
# AmgT_HYPRE/src/seq_mv/seq_mv.h`), not a hand patch. AmgT_FP64.h differs
# from the checked-in seq_mv.h by exactly two #define toggles
# (Hypre_AMGT, ADAPTIVE_AMGT_SPMV) -- `diff` reproduced verbatim in
# STATUS.md. This governs `hypreDevice_CSRSpGemm`'s internal dispatch
# (cuSPARSE vs AmgT) and SpMV's balanced/unbalanced kernel choice; it does
# NOT gate spgemm_amgT_fp64 itself, which this wrapper calls directly and
# which is defined unconditionally in csr_spgemm_device.c (confirmed by
# grep -- no #ifdef around its definition) -- done anyway for provenance
# fidelity to the artifact's documented build convention.
cp source/config_files/AmgT_FP64.h "$SRC_HYPRE/seq_mv/seq_mv.h"

echo "-- 2/3: AmgT_HYPRE (HYPRE + AmgT's own SpGEMM/SpMV, static libHYPRE.a) --"
(
    cd "$SRC_HYPRE"
    CC=mpicc CXX=mpicxx ./configure --with-cuda --with-gpu-arch='80 80' \
        --enable-unified-memory --with-extra-ldpath="$MATHLIB" \
        --disable-fortran --enable-shared
        # --enable-shared: HYPRE's own configure flag to build libHYPRE.so
        # (PIC-compiled) instead of a static libHYPRE.a. Needed because our
        # own libamgt_wrapper.so must LINK against libHYPRE -- embedding a
        # non-PIC static .a's objects into a new .so fails at link time
        # ("relocation R_X86_64_32 against `.rodata` can not be used when
        # making a shared object; recompile with -fPIC", hit for real on
        # this machine's first build attempt). This is HYPRE's own
        # documented configure option (`./configure --help` lists it), not
        # a hand patch to any Makefile or source file.
        # --disable-fortran: this wrapper never touches HYPRE's Fortran
        # bindings; the system has no working Fortran compiler wired to
        # mpif77 on this machine (configure's Fortran name-mangling probe
        # fails otherwise) -- a build-system fix, not a content change
        # (AmgT's own compile.sh doesn't pass this flag either way, so
        # HYPRE's own configure default of enabled-but-unused Fortran
        # support is simply turned off here).
    # Capped, not $(nproc): a shared login node enforces a low per-user
    # RLIMIT_NPROC, and make -j$(nproc) (128 here) blows past it with
    # "Unable to call the compiler ... Unknown error" / "vfork: Resource
    # temporarily unavailable" from concurrent cc1plus/nvcc children --
    # resource contention, not an AmgT bug. Overridable for a machine with
    # more headroom.
    make install -j"${KB_MAKE_J:-8}"
)

echo "-- 3/3: libamgt_wrapper.so (adapter glue, see wrapper.cu) --"
"$NVCC" -ccbin=mpicxx -O3 -std=c++14 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -I"$HYPRE_DIR/include" \
    -I"$MPI_INC" \
    -shared -o libamgt_wrapper.so wrapper.cu \
    -L"$HYPRE_DIR/lib" -lHYPRE \
    -L"$CUDA_HOME/lib64" -lcudart \
    -L"$MATHLIB" -lcublas -lcusparse -lcurand \
    -L"$MPI_LIB" -l"$KB_MPI_LIBNAME" \
    -lm -lstdc++ \
    -Xlinker -rpath -Xlinker "$CUDA_HOME/lib64" \
    -Xlinker -rpath -Xlinker "$MATHLIB" \
    -Xlinker -rpath -Xlinker "$MPI_LIB" \
    -Xlinker -rpath -Xlinker "$HYPRE_DIR/lib"

echo "-- verifying symbols --"
nm -D libamgt_wrapper.so | grep -q " T amgt_init" || { echo "MISSING amgt_init"; exit 1; }
nm -D libamgt_wrapper.so | grep -q " T amgt_spgemm_run" || { echo "MISSING amgt_spgemm_run"; exit 1; }
ldd libamgt_wrapper.so | grep -q "not found" && \
    { echo "MISSING SHARED LIBS:"; ldd libamgt_wrapper.so | grep "not found"; exit 1; }
echo "== AmgT build: OK =="
