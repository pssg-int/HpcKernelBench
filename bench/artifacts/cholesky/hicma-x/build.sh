#!/usr/bin/env bash
# hicma-x (HiCMA-PaRSEC) -- build the single-node CUDA testing_potrf_tlr
# driver (dense/TLR/mixed-precision tile Cholesky on PaRSEC/DPLASMA).
#
# This is a CMake superbuild: the top-level CMakeLists.txt add_subdirectory()s
# three nested git submodules straight from source (dplasma, which itself
# nests parsec; stars-h; hcore) plus this repo's own src/ (the hicma_parsec
# library and tests/ drivers). No external StarPU/Chameleon needed at all --
# unlike the older StarPU-based HiCMA (see ../exageostat/build.sh), this
# repo's runtime IS PaRSEC/DPLASMA, built in-tree.
#
# Idempotent: safe to re-run; exit 0 = tests/testing_potrf_tlr present.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

SRC="$HERE/source"
BUILD="$SRC/build"
PREFIX="$HERE/prefix"
BINARY="$BUILD/tests/testing_potrf_tlr"
COMPAT_H="$HERE/patches/lapacke_ge_trans_compat.h"

echo "== hicma-x (HiCMA-PaRSEC) build =="
echo "commit (hicma-x) : $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"

if [ -e "$BINARY" ] && [ -x "$BINARY" ]; then
    echo "already built: $BINARY"
    exit 0
fi

# ---------------------------------------------------------------- submodules
# Shallow-cloned (git clone --depth 1); submodules are NOT fetched by that
# alone. Fetch each one shallowly too (depth 1) -- these are the exact
# forked branches (QingleiCao/{dplasma,parsec,hcore,stars-h}, "hicma_parsec"
# branch) hicma-x's own .gitmodules pins.
if [ ! -f "$SRC/dplasma/CMakeLists.txt" ]; then
    git -C "$SRC" submodule update --init --depth 1 -- dplasma hcore stars-h
fi
if [ ! -f "$SRC/dplasma/parsec/CMakeLists.txt" ]; then
    git -C "$SRC/dplasma" submodule update --init --depth 1 -- parsec
fi
if [ ! -d "$SRC/hcore/cmake_modules/ecrc/modules" ]; then
    git -C "$SRC/hcore" submodule update --init --depth 1
fi
if [ ! -d "$SRC/stars-h/cmake_modules/ecrc/modules" ]; then
    git -C "$SRC/stars-h" submodule update --init --depth 1
fi

# ------------------------------------------------------------- toolchain pin
# Beyond the CUDA_HOME/CPATH pin toolchain.sh already did: this build uses
# the Cray compiler wrappers (cc/CC/ftn), which -- when craype-accel-nvidia80
# is loaded -- inject a fixed GPU-transport-layer library
# (libmpi_gtl_cuda.so) that was built against whatever CUDA major is this
# LOGIN NODE'S DEFAULT (currently 13.2's libcudart.so.13), independent of
# which cudatoolkit module is active. That makes any MPI-linked CUDA binary
# (cc always links MPI when cray-mpich is loaded) fail at link time
# ("libcudart.so.13 ... not found" once cudatoolkit is swapped to 12.9, or
# silently ABI-mismatched if left at 13.2). Two real, independent
# environment bugs, not specific to hicma-x's own code:
#   1. craype-accel-nvidia80's own injected -I/-L paths track whichever
#      cudatoolkit module is loaded (not CUDA_HOME/CPATH), so a PLAIN cc
#      compile of a .c file using <cuda_runtime.h> (PaRSEC's
#      device_cuda_module.c does, unconditionally) picks up cudatoolkit's
#      version regardless of toolchain.sh's exports -- confirmed via
#      `cc -x c - -E -v </dev/null` showing 13.2's include dir. Fix: swap
#      the cudatoolkit module itself (module swap cudatoolkit/13.2
#      cudatoolkit/12.9) so cc's own header search matches nvcc's 12.9.
#   2. craype-accel-nvidia80 then links the CUDA-13-built GTL shim
#      regardless of (1)'s fix, and MPICH_GPU_SUPPORT_ENABLED=0 (a RUNTIME
#      toggle) does not suppress this COMPILE-TIME injection. Since this
#      build needs MPI only to link (no MPI is actually used at runtime --
#      the top-level CMakeLists.txt's own "Implementation paradigm: MPI OFF"
#      summary during configure confirms PaRSEC itself builds without
#      distributed support here), the fix is simply to NOT load
#      craype-accel-nvidia80 at all for this build; CUDA support is supplied
#      entirely via -DHICMA_PARSEC_HAVE_CUDA=ON + an explicit nvcc path.
module unload craype-accel-nvidia80 >/dev/null 2>&1 || true
module swap cudatoolkit/13.2 cudatoolkit/12.9 >/dev/null 2>&1 || true
export MPICH_GPU_SUPPORT_ENABLED=0
# Host compilers: default to the Cray PrgEnv wrappers (Perlmutter), but
# honour CC/CXX if bench/env.sh already exported them (e.g. a conda-forge
# gcc/g++ on a non-Cray machine -- KB_CC/KB_CXX). FC has no KB_FC knob
# (nothing else needs one yet): fall back to "ftn" only when CXX is itself
# a bare Cray-wrapper name (no '/'); otherwise derive gfortran from CXX's
# own directory, matching ../exageostat/build.sh's FC_PLAIN precedent.
export CC="${CC:-cc}" CXX="${CXX:-CC}"
if [ -z "${FC:-}" ]; then
    case "$CXX" in
        */*) FC="$(dirname "$CXX")/gfortran" ;;
        *)   FC="ftn" ;;
    esac
fi
export FC

echo "cc(=\$CC): $(command -v "$CC") ($("$CC" --version | head -1))"
echo "nvcc   : $(command -v nvcc) ($(nvcc --version | tail -1))"
echo "cmake  : $(command -v cmake) ($(cmake --version | head -1))"

CRAY_LIBSCI="${KB_BLAS_LIB:-${KB_BLAS_LIBDIR:-/opt/cray/pe/libsci/26.03.0/GNU/12/x86_64/lib}/libsci_gnu.so}"   # KB_BLAS_* from bench/env.sh
if [ ! -f "$CRAY_LIBSCI" ]; then
    echo "STATUS: BUILD-FAILED -- expected Cray LibSci not found at $CRAY_LIBSCI"
    exit 1
fi

# GCC 14 promotes several legacy-C diagnostics to hard errors by default
# (implicit-function-declaration, incompatible-pointer-types, int-conversion,
# declaration-missing-parameter-type) -- this ~2020-era ECRC/PaRSEC codebase
# predates that change throughout dplasma/parsec/stars-h. Downgrading these
# four back to warnings is a pure compiler-version-compat fix (matches
# ARTIFACT_GUIDE.md's own "CUDA-version guards are fine" precedent, applied
# to a GCC-version guard instead); no kernel/numeric code is touched.
#
# -include "$COMPAT_H" force-includes 3 missing prototypes
# (LAPACKE_{s,d,z}ge_trans) into every compiled C/C++ file. hcore's own
# hcore_{s,d,z}gemm.c / hcore_dgemm_fast.c call these three LAPACKE
# "extension" functions (in-place matrix transposition); Cray LibSci's
# lapacke_utils.h declares and EXPORTS them (confirmed via
# `nm -D libsci_gnu.so | grep ge_trans`), but this project's own
# include_directories() lists DPLASMA's bundled 2015 LAPACKE snapshot
# (dplasma/src/include/lapacke.h) ahead of the real vendor header, and that
# snapshot predates these three extensions -- so the compiler never sees a
# declaration for them from the "correct" header. Two more invasive fixes
# were tried and abandoned (see STATUS.md): forcing the vendor header first
# via -I (CMake emits $(C_INCLUDES) before $(C_FLAGS), so a later -I loses),
# and renaming DPLASMA's shared "_LAPACKE_H_" include guard so both headers'
# bodies run (their function signatures differ enough elsewhere --
# LAPACKE_zuncsd2by1, LAPACKE_ztprfb -- to produce genuine
# conflicting-redeclaration errors). Declaring exactly the 3 functions HCORE
# actually calls, and letting the linker resolve them against Cray LibSci
# (already on the link line via -DBLAS_LIBRARIES below), sidesteps the whole
# header-guard collision. This file lives outside every submodule and
# touches zero source in this checkout.
# -Wdeclaration-missing-parameter-type is a GCC-14-only diagnostic name;
# GCC 13 (this machine's conda-forge host compiler) doesn't recognize it at
# all and hard-errors on the unknown -Wno-error= spelling itself ("no option
# '-Wdeclaration-missing-parameter-type'"), rather than just being a no-op
# like the other three (which are older, pre-existing warning names GCC 13
# already understands, just not yet promoted to hard errors by default --
# see ../../ARTIFACT_GUIDE.md's own "CUDA-version guards are fine" precedent,
# applied to a GCC-version-specific flag: only pass the flag when the
# compiler itself accepts it, probed directly rather than by version-number
# guessing).
DECL_MISSING_FLAG=""
if echo 'int f(void);' | "$CC" -x c -Wno-error=declaration-missing-parameter-type -fsyntax-only - >/dev/null 2>&1; then
    DECL_MISSING_FLAG="-Wno-error=declaration-missing-parameter-type"
fi
COMPAT_FLAGS="-include $COMPAT_H -Wno-error=implicit-function-declaration -Wno-error=incompatible-pointer-types -Wno-error=int-conversion $DECL_MISSING_FLAG"

# Parallel-build cap (login-node resource contention, not an artifact bug):
# this login node enforces a shared, whole-session RLIMIT_NPROC (many builds
# from many users/agents run at once); a wide -j exhausts it -- observed here
# as a burst of "flex: fork failed" / "Error 127" (the shell couldn't even
# exec the compiler) across unrelated targets in the same `make` invocation,
# not a real compile error (no "error:" diagnostic anywhere in the log).
# Overridable; 8 is safe everywhere including Perlmutter's original -j24.
MAKE_J="${KB_MAKE_J:-8}"

mkdir -p "$BUILD"
cd "$BUILD"
# Skip a from-scratch reconfigure when a cache from a previous (possibly
# interrupted, e.g. by login-node fork contention -- see MAKE_J above)
# invocation already exists: CMake's own configure+generate for this
# superbuild alone takes several minutes, and re-running `make` after a
# fork-contention failure is fully idempotent against an existing cache.
if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_Fortran_COMPILER="$FC" \
        -DCMAKE_C_FLAGS="$COMPAT_FLAGS" \
        -DCMAKE_CXX_FLAGS="$COMPAT_FLAGS" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBLAS_LIBRARIES="$CRAY_LIBSCI" \
        -DHICMA_PARSEC_HAVE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
fi

# 244 logical cores on this shared login node; cap parallelism.
make -j"$MAKE_J"

if [ ! -x "$BINARY" ]; then
    echo "STATUS: BUILD-FAILED -- $BINARY missing after make"
    exit 1
fi

echo "built: $BINARY"
exit 0
