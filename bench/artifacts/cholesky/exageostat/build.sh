#!/usr/bin/env bash
# exageostat (ExaGeoStat) -- build the StarPU + Chameleon + NLopt + GSL stack
# and ExaGeoStat itself (dense/exact computation, single GPU), matching the
# task brief's general strategy for older StarPU-based ECRC codebases (see
# ../hicma-x/STATUS.md for why hicma-x itself needed a DIFFERENT, PaRSEC-
# based approach instead).
#
# Layered build, each installed into ./prefix:
#   1. StarPU 1.3.11 (source tarball -- no module on this system; --disable-mpi,
#      CUDA on)
#   2. Chameleon (submodule, release-1.1.0) on top of StarPU -- no MPI, CUDA on
#   3. NLopt 2.7.1 (source tarball -- ExaGeoStat's own auto-installer script
#      failed, see STATUS.md; a plain upstream build is simpler and faster)
#   3b. GSL 2.6 (source tarball -- same auto-installer bug as NLopt; only
#      needed on machines with no system/conda GSL, see STATUS.md)
#   4. ExaGeoStat itself, EXAGEOSTAT_USE_HICMA=OFF (dense/exact only -- see
#      STATUS.md for why the TLR/HiCMA path was out of scope for this pass)
#
# Idempotent: safe to re-run; exit 0 = prefix/bin/{chameleon_dtesting,
# synthetic_dmle_test} present.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

SRC="$HERE/source"
DEPS="$HERE/deps"
PREFIX="$HERE/prefix"
# BLAS/LAPACK: KB_BLAS_LIB (a library file) or KB_BLAS_LIBDIR (bench/env.sh); Cray LibSci on Perlmutter.
CRAY_LIBSCI="${KB_BLAS_LIB:-${KB_BLAS_LIBDIR:-/opt/cray/pe/libsci/26.03.0/GNU/12/x86_64/lib}/libsci_gnu.so}"
CRAY_LIBSCI_INC="${KB_BLAS_INCDIR:-$(dirname "$(dirname "$CRAY_LIBSCI")")/include}"
MATHLIBS="${KB_CUDA_MATHLIBS:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9}"   # exported by toolchain.sh
CC_PLAIN="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
CXX_PLAIN="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
FC_PLAIN="${FC:-$(dirname "$CXX_PLAIN")/gfortran}"
# GCC-14 promotes several legacy-C diagnostics to hard errors by default;
# this ECRC-lineage codebase (StarPU/Chameleon/ExaGeoStat, ~2018-2023 era)
# predates that change throughout. Compiler-version-compat only, matches
# ARTIFACT_GUIDE.md's "CUDA-version guards are fine" precedent applied to a
# GCC-version guard; no numerics touched. See STATUS.md for the handful of
# real source patches this build ALSO needed (cublas.h/cublas_v2.h
# mutual-exclusion, one C++-reserved-keyword parameter name, one
# vsprintf/vasprintf typo -- none of them numeric).
# -Wdeclaration-missing-parameter-type is a GCC-14-only diagnostic name;
# GCC 13 (e.g. this machine's conda-forge host compiler) doesn't recognize
# it at all and hard-errors on the unknown -Wno-error= spelling itself,
# rather than treating it as a no-op like the other three (pre-existing
# warning names GCC 13 already understands, just not yet promoted to hard
# errors by default). Probe the compiler directly rather than guessing by
# version number (see ../hicma-x/build.sh's identical probe).
DECL_MISSING_FLAG=""
if echo 'int f(void);' | "$CC_PLAIN" -x c -Wno-error=declaration-missing-parameter-type -fsyntax-only - >/dev/null 2>&1; then
    DECL_MISSING_FLAG="-Wno-error=declaration-missing-parameter-type"
fi
COMPAT_CFLAGS="-fPIC -Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration -Wno-error=int-conversion $DECL_MISSING_FLAG"

# Parallel-build cap (login-node resource contention, not an artifact bug):
# this login node enforces a shared, whole-session RLIMIT_NPROC (many builds
# from many users/agents run at once); a wide -j exhausts it ("cc1plus: vfork:
# Resource temporarily unavailable" / plain fork failures, even for unrelated
# shells). Overridable; 8 is safe everywhere including Perlmutter.
MAKE_J="${KB_MAKE_J:-8}"

# CMake-version guard (no numerics touched), set globally as an env var
# rather than only a per-invocation -DCMAKE_POLICY_VERSION_MINIMUM=3.5
# cache entry: this machine's cmake (4.x) hard-errors on any
# cmake_minimum_required() below 3.5, and NLopt's own build additionally
# shells out to standalone `cmake -P generate-{cpp,fortran}.cmake` script-
# mode invocations at BUILD time (not configure time) that do NOT inherit
# the top-level project's cache -- only the environment variable form
# (CMake's own documented mechanism for exactly this case) reaches those.
export CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"

BINARY_CHECK="$PREFIX/bin/chameleon_dtesting"
BINARY_EXAGEOSTAT_CHECK="$SRC/build/examples/synthetic_dmle_test"

echo "== exageostat build =="
echo "commit (exageostat) : $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"

if [ -x "$BINARY_CHECK" ] && [ -x "$BINARY_EXAGEOSTAT_CHECK" ]; then
    echo "already built: $BINARY_CHECK, $BINARY_EXAGEOSTAT_CHECK"
    exit 0
fi

mkdir -p "$PREFIX" "$DEPS"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:$MATHLIBS/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$CRAY_LIBSCI%/../:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:$MATHLIBS/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$(dirname "$CRAY_LIBSCI"):${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:$CUDA_HOME/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
# $CUDA_HOME/lib/pkgconfig ahead of the system default: on this machine
# CMake's own FindHWLOC.cmake (pulled in transitively by
# find_package(STARPU ... COMPONENTS HWLOC), and again by ExaGeoStat's own
# find_package(STARPU) below) does `pkg_search_module(HWLOC hwloc)`, and
# without a hwloc.pc reachable on PKG_CONFIG_PATH it silently falls through
# to the SYSTEM hwloc.pc (empty Cflags -- /usr/include is already a default
# compiler search dir) and ends up setting HWLOC_INCLUDE_DIRS=/usr/include,
# which ExaGeoStat's CMakeLists.txt then propagates via a global
# include_directories() into every compile in the project, nvcc included.
# This machine's system glibc (RHEL 8) headers (bits/floatn.h) declare
# `typedef __float128 _Float128;` whenever __cplusplus is defined (which it
# always is for nvcc's device-code frontend on a .cu file) -- and nvcc's
# restricted C++ frontend does not understand the __float128 GNU extension
# type, hard-erroring with "invalid combination of type specifiers" on
# exageostat_exact/cuda_core/compute/{cuda_conv.cu,cuda_zcmg.cu} (confirmed
# by hand: these two files compile cleanly with -I/usr/include dropped --
# they need nothing from it). Pointing PKG_CONFIG_PATH at this conda
# toolchain's own hwloc.pc first (Cflags -I$CUDA_HOME/include, matching what
# StarPU itself was already built against via --with-hwloc) makes FindHWLOC
# resolve there instead, so /usr/include never enters the CUDA include list
# at all. A CUDA-toolchain/glibc-version-compat fix, not a numerics change;
# inert on Perlmutter (no hwloc.pc under its CUDA_HOME, so this PKG_CONFIG_PATH
# entry has no .pc files and pkg-config just skips it).

# --------------------------------------------------------------- 1. StarPU
if [ ! -f "$PREFIX/lib/pkgconfig/libstarpu.pc" ]; then
    echo "--- Building StarPU 1.3.11 ---"
    STARPU_SRC="$DEPS/starpu-1.3.11"
    if [ ! -d "$STARPU_SRC" ]; then
        # curl run with LD_LIBRARY_PATH/LD_PRELOAD cleared: this build.sh (and
        # toolchain.sh before it) puts $MATHLIBS/lib64 -- the conda env's own
        # lib64 -- on LD_LIBRARY_PATH for the CUDA math libs, but that same
        # directory also ships a conda libcurl.so.4 that gets picked up ahead
        # of the system one and is ABI-mismatched with the system libssl,
        # making every curl invocation fail with "curl: (4) A requested
        # feature ... was not found built-in" before it even connects. Not a
        # curl-flag problem -- confirmed via `ldd /usr/bin/curl` before/after
        # sourcing bench/env.sh. Downloads are a build-system step, not
        # kernel code, so this is scoped to just this one command rather than
        # touching the shared toolchain.sh/env.sh LD_LIBRARY_PATH exports.
        env -u LD_LIBRARY_PATH -u LD_PRELOAD curl -sL --max-time 120 -o "$DEPS/starpu-1.3.11.tar.gz" \
            https://files.inria.fr/starpu/starpu-1.3.11/starpu-1.3.11.tar.gz
        tar xzf "$DEPS/starpu-1.3.11.tar.gz" -C "$DEPS"
    fi
    # NUMA-topology CUDA hint patch: StarPU's topology.c calls
    # hwloc_cuda_get_device_osdev_by_index() unconditionally whenever
    # CUDA+hwloc are enabled, but its own matching
    # `#include <hwloc/cuda.h>` a few lines above is (correctly) ALSO
    # guarded on a real declaration probe (HAVE_DECL_...) -- this machine's
    # system hwloc package (2.10.0) was not built with CUDA support, so
    # hwloc/cuda.h does not exist at all; the include is (correctly)
    # skipped but this call site was not, hitting an
    # implicit-function-declaration error. Widening this ONE guard to match
    # the include's disables only an optional GPU-aware NUMA-node placement
    # hint (StarPU falls back to ordinary CPU-RAM-adjacent placement, same
    # as every other topology on this single-socket build) -- not the
    # GEMM/POTRF kernels themselves. See topology.c's own inline comment
    # (left in place) for the full trace.
    TOPO="$STARPU_SRC/src/core/topology.c"
    if ! grep -q "HPC-KernelBench" "$TOPO" 2>/dev/null; then
        python3 - "$TOPO" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
old = "#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_HWLOC)\n\t\tfor (i = 0; i < config->topology.ncudagpus; i++)\n\t\t{\n\t\t\thwloc_obj_t obj = hwloc_cuda_get_device_osdev_by_index(config->topology.hwtopology, i);"
new = ("/* HPC-KernelBench / exageostat integration (build-system fix, no\n"
       " * numerics touched): widen this guard to match the include's own\n"
       " * HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX probe a few lines\n"
       " * above -- this machine's hwloc lacks CUDA support (hwloc/cuda.h\n"
       " * does not exist), so the include is skipped but this call site\n"
       " * was not, an upstream guard mismatch. Disables only an optional\n"
       " * GPU-aware NUMA placement hint, not any GEMM/POTRF kernel. */\n"
       "#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_HWLOC) && defined(HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX) && HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX\n"
       "\t\tfor (i = 0; i < config->topology.ncudagpus; i++)\n\t\t{\n"
       "\t\t\thwloc_obj_t obj = hwloc_cuda_get_device_osdev_by_index(config->topology.hwtopology, i);")
assert old in content, "StarPU topology.c patch site not found -- upstream source changed?"
content = content.replace(old, new, 1)
with open(path, "w") as f:
    f.write(content)
print("patched", path)
PYEOF
    fi

    mkdir -p "$STARPU_SRC/build"
    cd "$STARPU_SRC/build"
    rm -rf ./*
    ../configure \
        --prefix="$PREFIX" \
        --disable-mpi --disable-build-doc --disable-fortran \
        --disable-build-tests --disable-build-examples --disable-starpufft \
        --with-hwloc \
        --enable-cuda \
        --with-cuda-dir="$CUDA_HOME" \
        --with-cuda-include-dir="$CUDA_HOME/include" \
        --with-cuda-lib-dir="$CUDA_HOME/lib64" \
        CC="$CC_PLAIN" CXX="$CXX_PLAIN" NVCC="$CUDA_HOME/bin/nvcc" \
        LDFLAGS="-L$MATHLIBS/lib64 -Wl,-rpath,$MATHLIBS/lib64 -Wl,-rpath,$CUDA_HOME/lib64" \
        CPPFLAGS="-I$MATHLIBS/include"
    make -j"$MAKE_J"
    make install
    cd "$HERE"
fi

# ------------------------------------------------------------- 2. Chameleon
if [ ! -f "$PREFIX/lib/pkgconfig/chameleon.pc" ]; then
    echo "--- Building Chameleon (release-1.1.0) ---"
    CHAM="$SRC/chameleon"
    # The top-level "chameleon" submodule itself can be an EMPTY directory
    # here (a plain `git clone` of source/ from source.provenance does not
    # recurse into submodules the way the reference-machine checkout did --
    # download/checkout step, not kernel code): initialize it first if so,
    # before touching its OWN nested submodules below.
    if [ ! -f "$CHAM/CMakeLists.txt" ]; then
        git -C "$SRC" submodule update --init --depth 1 -- chameleon
    fi
    if [ ! -f "$CHAM/cmake_modules/morse_cmake/modules/find/FindMorseInit.cmake" ]; then
        git -C "$CHAM" submodule update --init --depth 1 -- cmake_modules/morse_cmake hqr
    fi
    # cublas.h / cublas_v2.h mutual-exclusion patch (CUDA-version guard, no
    # numerics touched): CUDA 12.9's cublas_v2.h hard-errors if legacy
    # cublas.h is ALSO included in the same translation unit -- a
    # restriction added after this ~2019-era code was written. Three
    # spots in this checkout unconditionally (or via the
    # CHAMELEON_USE_CUBLAS_V2=ON branch, which itself did BOTH) include
    # both; each is patched to include only cublas_v2.h when
    # CHAMELEON_USE_CUBLAS_V2 is set (this build sets it ON throughout,
    # matching what ExaGeoStat's own CUDA kernels need -- they use
    # cublasHandle_t directly). See each file's own inline comment (left in
    # place) for detail; STATUS.md has the full diffs.
    #
    # This is a NESTED submodule (chameleon/), so a fresh `git submodule
    # update --init` (above) never carries these edits -- they live only in
    # this integration's own patching, applied below (idempotent, guarded
    # by the same HPC-KernelBench marker the check loop looks for), unlike
    # the outer exageostat repo's tracked-file edits (source.patch).
    if ! grep -q "HPC-KernelBench" "$CHAM/cudablas/include/cudablas.h" 2>/dev/null; then
        python3 - "$CHAM/cudablas/include/cudablas.h" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
old = "#if defined(CHAMELEON_USE_CUBLAS_V2)\n\n#include <cublas.h>\n#include <cublas_v2.h>\n"
new = ("#if defined(CHAMELEON_USE_CUBLAS_V2)\n"
       "/* HPC-KernelBench / exageostat integration (build-system fix,\n"
       " * CUDA-version guard -- no numerics touched): CUDA 12.9's\n"
       " * cublas_v2.h hard-errors if legacy cublas.h is also included in\n"
       " * the same translation unit. This branch already selects the v2\n"
       " * API; the redundant legacy include is dropped, not replaced. */\n\n"
       "#include <cublas_v2.h>\n")
assert old in content, "cudablas.h patch site not found -- upstream source changed?"
content = content.replace(old, new, 1)
with open(path, "w") as f:
    f.write(content)
print("patched", path)
PYEOF
    fi
    if ! grep -q "HPC-KernelBench" "$CHAM/control/common.h" 2>/dev/null; then
        python3 - "$CHAM/control/common.h" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
old = "#if defined(CHAMELEON_USE_CUBLAS_V2)\n#include <cublas.h>\n#include <cublas_v2.h>\n#else\n"
new = ("#if defined(CHAMELEON_USE_CUBLAS_V2)\n"
       "/* HPC-KernelBench / exageostat integration (build-system fix,\n"
       " * CUDA-version guard -- no numerics touched): see\n"
       " * cudablas/include/cudablas.h's identical fix in this repo. */\n"
       "#include <cublas_v2.h>\n#else\n")
assert old in content, "control/common.h patch site not found -- upstream source changed?"
content = content.replace(old, new, 1)
with open(path, "w") as f:
    f.write(content)
print("patched", path)
PYEOF
    fi
    if ! grep -q "HPC-KernelBench" "$CHAM/runtime/starpu/include/chameleon_starpu.h.in" 2>/dev/null; then
        python3 - "$CHAM/runtime/starpu/include/chameleon_starpu.h.in" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
old = ("#include <cublas.h>\n#include <starpu_cublas.h>\n"
       "#if defined(CHAMELEON_USE_CUBLAS_V2)\n"
       "#include <cublas_v2.h>\n#include <starpu_cublas_v2.h>\n#endif\n")
new = ("/* HPC-KernelBench / exageostat integration (build-system fix,\n"
       " * CUDA-version guard -- no numerics touched): see exageostat's own\n"
       " * vendored copy of this file, include/chameleon_starpu.h, for the\n"
       " * identical fix. Legacy <cublas.h>/<starpu_cublas.h> were\n"
       " * unconditional and v2 was additive on top; now both sides of a\n"
       " * proper #if/#else. */\n"
       "#if defined(CHAMELEON_USE_CUBLAS_V2)\n"
       "#include <cublas_v2.h>\n#include <starpu_cublas_v2.h>\n#else\n"
       "#include <cublas.h>\n#include <starpu_cublas.h>\n#endif\n")
assert old in content, "chameleon_starpu.h.in patch site not found -- upstream source changed?"
content = content.replace(old, new, 1)
with open(path, "w") as f:
    f.write(content)
print("patched", path)
PYEOF
    fi
    for f in "$CHAM/cudablas/include/cudablas.h" \
             "$CHAM/control/common.h" \
             "$CHAM/runtime/starpu/include/chameleon_starpu.h.in"; do
        if grep -q "^#include <cublas.h>$" "$f" 2>/dev/null && ! grep -q "HPC-KernelBench" "$f"; then
            echo "STATUS: BUILD-FAILED -- $f needs the cublas_v2 patch but was not pre-patched by this integration; see STATUS.md for the exact diff to apply (upstream source changed?)"
            exit 1
        fi
    done

    mkdir -p "$CHAM/build"
    cd "$CHAM/build"
    rm -rf ./*
    # CMake-version guard (no numerics touched): this machine's cmake (4.x)
    # hard-errors on any cmake_minimum_required() below 3.5 (Chameleon
    # itself pins 3.3); -DCMAKE_POLICY_VERSION_MINIMUM=3.5 is CMake's own
    # documented escape hatch (its error message names it directly), a
    # no-op cache variable on older cmake that doesn't need it.
    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_C_FLAGS="$COMPAT_CFLAGS" \
        -DCHAMELEON_USE_MPI=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCHAMELEON_USE_CUDA=ON \
        -DCHAMELEON_USE_CUBLAS_V2=ON \
        -DCHAMELEON_ENABLE_EXAMPLE=OFF \
        -DCHAMELEON_ENABLE_TESTING=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DBLAS_LIBRARIES="$CRAY_LIBSCI" \
        -DLAPACK_LIBRARIES="$CRAY_LIBSCI" \
        -DCBLAS_LIBRARIES="$CRAY_LIBSCI" \
        -DLAPACKE_LIBRARIES="$CRAY_LIBSCI" \
        -DCBLAS_INCLUDE_DIRS="$CRAY_LIBSCI_INC" \
        -DLAPACKE_INCLUDE_DIRS="$CRAY_LIBSCI_INC" \
        -DCMAKE_C_COMPILER="$CC_PLAIN" -DCMAKE_CXX_COMPILER="$CXX_PLAIN" -DCMAKE_Fortran_COMPILER="$FC_PLAIN"
    # NOTE: unlike autotools ./configure (StarPU/GSL above), CMake does not
    # accept trailing CC=/CXX=/FC= as compiler overrides -- it silently
    # warns "Ignoring extra path from command line" and falls back to its
    # own compiler search (found this machine's broken system /usr/bin/f95
    # for Fortran; C/CXX happened to still match only because CC/CXX are
    # ALSO exported env vars from bench/env.sh, which CMake's compiler
    # detection does honour). Passing -DCMAKE_*_COMPILER explicitly above
    # is the correct, environment-independent CMake spelling.
    make -j"$MAKE_J"
    make install
    cd "$HERE"
fi

# ---------------------------------------------------------------- 3a. NLopt
if [ ! -f "$PREFIX/lib64/pkgconfig/nlopt.pc" ] && [ ! -f "$PREFIX/lib/pkgconfig/nlopt.pc" ]; then
    echo "--- Building NLopt 2.7.1 ---"
    # ExaGeoStat's own CMake auto-installer (ImportNLOPT.cmake ->
    # InstallNLOPT.sh) failed on this machine ("Please specify Installation
    # Path AND Setup Path" -- a TMP_DIR variable the top-level CMakeLists.txt
    # never sets when EXAGEOSTAT_INSTALL_DEPS=OFF). A plain upstream build
    # is simpler and faster than debugging that script.
    NLOPT_SRC="$DEPS/nlopt-2.7.1"
    if [ ! -d "$NLOPT_SRC" ]; then
        # See the StarPU curl call above: LD_LIBRARY_PATH must be cleared for
        # this invocation or curl fails with "curl: (4) ... not found built-in"
        # (conda's shadowed libcurl.so.4 mismatched with system libssl).
        env -u LD_LIBRARY_PATH -u LD_PRELOAD curl -sL --max-time 60 -o "$DEPS/nlopt-2.7.1.tar.gz" \
            https://github.com/stevengj/nlopt/archive/refs/tags/v2.7.1.tar.gz
        tar xzf "$DEPS/nlopt-2.7.1.tar.gz" -C "$DEPS"
    fi
    mkdir -p "$NLOPT_SRC/build"
    cd "$NLOPT_SRC/build"
    rm -rf ./*
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DNLOPT_PYTHON=OFF -DNLOPT_OCTAVE=OFF -DNLOPT_MATLAB=OFF \
        -DNLOPT_GUILE=OFF -DNLOPT_SWIG=OFF -DBUILD_SHARED_LIBS=ON \
        CC="$CC_PLAIN" CXX="$CXX_PLAIN"
    make -j"$MAKE_J" install
    cd "$HERE"
    export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

# ------------------------------------------------------------------ 3b. GSL
if [ ! -f "$PREFIX/lib/pkgconfig/gsl.pc" ] && [ ! -f "$PREFIX/lib64/pkgconfig/gsl.pc" ] && [ ! -x "$PREFIX/bin/gsl-config" ]; then
    echo "--- Building GSL 2.6 ---"
    # ExaGeoStat's CMakeLists.txt does `find_package(GSL 2.4.2 QUIET REQUIRED)`
    # unconditionally (not gated behind any EXAGEOSTAT_USE_* option -- see
    # "EXAGEOSTAT depends on GSL" in source/CMakeLists.txt), so a missing GSL
    # is a hard configure-time FATAL_ERROR. This machine (unlike Perlmutter,
    # which has a system GSL 2.6 package) has no GSL anywhere -- no system
    # package, nothing in the conda env. ExaGeoStat's own CMake auto-installer
    # (ImportGSL.cmake -> InstallGSL.sh) has the identical TMP_DIR bug already
    # documented for NLopt above (EXAGEOSTAT_INSTALL_DEPS=OFF means
    # ${BUILD_DEPENDENCIES} is unset/false, so ImportGSL.cmake's own
    # auto-install branch never runs and GSL_FOUND stays false ->
    # FATAL_ERROR). A plain upstream build, the exact version
    # InstallGSL.sh itself pins (2.6, also >= the 2.4.2 CMake floor), is
    # simpler and faster than debugging that script (same rationale as
    # NLopt immediately above).
    GSL_SRC="$DEPS/gsl-2.6"
    if [ ! -d "$GSL_SRC" ]; then
        # See the StarPU curl call above: LD_LIBRARY_PATH must be cleared for
        # this invocation or curl fails with "curl: (4) ... not found built-in"
        # (conda's shadowed libcurl.so.4 mismatched with system libssl).
        env -u LD_LIBRARY_PATH -u LD_PRELOAD curl -sL --max-time 120 -o "$DEPS/gsl-2.6.tar.gz" \
            https://ftp.gnu.org/gnu/gsl/gsl-2.6.tar.gz
        tar xzf "$DEPS/gsl-2.6.tar.gz" -C "$DEPS"
    fi
    cd "$GSL_SRC"
    CC="$CC_PLAIN" CXX="$CXX_PLAIN" ./configure --prefix="$PREFIX"
    make -j"$MAKE_J"
    make install
    cd "$HERE"
    export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

# ------------------------------------------------------------ 4. ExaGeoStat
if [ ! -x "$BINARY_EXAGEOSTAT_CHECK" ]; then
    echo "--- Building ExaGeoStat ---"
    # Two source patches beyond the cublas_v2 pattern above, both in
    # source/include/{common.h,chameleon_starpu.h} and
    # source/exageostat_exact/cuda_core/compute/{cuda_conv.cu,cuda_zcmg.cu}
    # -- see STATUS.md for exact diffs and rationale (a C++-reserved-keyword
    # parameter name exposed only once this header is compiled by nvcc's
    # C++ front end; a vsprintf/vasprintf typo in a debug string-formatting
    # helper unrelated to any numerical kernel). Not re-applied here since
    # they are direct edits to files this integration already committed to
    # source/, not vendored patch application -- if source/ is a fresh
    # clone without them, this build will fail with the exact errors
    # documented in STATUS.md.
    for f in "$SRC/include/common.h" "$SRC/include/chameleon_starpu.h" \
             "$SRC/exageostat_exact/cuda_core/compute/cuda_conv.cu" \
             "$SRC/exageostat_exact/cuda_core/compute/cuda_zcmg.cu"; do
        if ! grep -q "HPC-KernelBench" "$f" 2>/dev/null; then
            echo "STATUS: BUILD-FAILED -- $f is missing this integration's patch; see STATUS.md for the exact diff (upstream source changed, or source/ was re-cloned fresh?)"
            exit 1
        fi
    done

    mkdir -p "$SRC/build"
    cd "$SRC/build"
    rm -rf ./*
    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_C_FLAGS="$COMPAT_CFLAGS" \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXAGEOSTAT_SCHED_STARPU=ON \
        -DEXAGEOSTAT_USE_MPI=OFF \
        -DEXAGEOSTAT_USE_HICMA=OFF \
        -DEXAGEOSTAT_USE_NETCDF=OFF \
        -DEXAGEOSTAT_USE_CHAMELEON=ON \
        -DEXAGEOSTAT_INSTALL_DEPS=OFF \
        -DEXAGEOSTAT_USE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
        -DCUDA_NVCC_FLAGS="-Xcompiler;-fPIC" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DBLAS_LIBRARIES="$CRAY_LIBSCI" \
        -DLAPACK_LIBRARIES="$CRAY_LIBSCI" \
        -DCBLAS_LIBRARIES="$CRAY_LIBSCI" \
        -DLAPACKE_LIBRARIES="$CRAY_LIBSCI" \
        -DCBLAS_INCLUDE_DIRS="$CRAY_LIBSCI_INC" \
        -DLAPACKE_INCLUDE_DIRS="$CRAY_LIBSCI_INC" \
        -DGSL_ROOT_DIR="$PREFIX" \
        -DCMAKE_C_COMPILER="$CC_PLAIN" -DCMAKE_CXX_COMPILER="$CXX_PLAIN" -DCMAKE_Fortran_COMPILER="$FC_PLAIN"
    # See the identical Chameleon cmake call above for why this is
    # -DCMAKE_*_COMPILER rather than trailing CC=/CXX=/FC= (CMake ignores
    # those as bogus extra path arguments).
    make -j"$MAKE_J"
    cd "$HERE"
fi

if [ ! -x "$BINARY_CHECK" ] || [ ! -x "$BINARY_EXAGEOSTAT_CHECK" ]; then
    echo "STATUS: BUILD-FAILED -- $BINARY_CHECK or $BINARY_EXAGEOSTAT_CHECK missing after build"
    exit 1
fi

echo "built: $BINARY_CHECK, $BINARY_EXAGEOSTAT_CHECK"
exit 0
