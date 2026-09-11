#!/usr/bin/env bash
# Build ngAP (ASPLOS'24, conf/asplos/GeZ024) -- ngap/obat/asyncap/ppopp12 CLIs.
# Idempotent: exit 0 if already built. Only the GPU-scheme build (code/) is
# attempted -- the CPU Hyperscan baseline (hscompile/, a submodule not
# fetched by our --depth 1 non-recursive clone) is not needed for this track.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source/code"
BUILD="$HERE/build"

if [ -x "$BUILD/bin/ngap" ] && [ -x "$BUILD/bin/obat" ] && [ -x "$BUILD/bin/ppopp12" ]; then
    echo "ngap: already built ($BUILD/bin/)"
    exit 0
fi

HOST_GXX="${HOST_GXX:-${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}}"
HOST_GCC="${HOST_GCC:-${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}}"

# --- Intel oneTBB (2026-09-09, zaratan) ------------------------------------
# source/code/CMakeLists.txt does `find_package(TBB REQUIRED COMPONENTS
# tbb)` unconditionally (gpunfacommons, linked by every target). This
# system ships only the bare runtime /usr/lib64/libtbb.so.2 -- no headers,
# no CMake package (Module or Config) -- so find_package(TBB) cannot
# succeed as-is. Vendor oneTBB per-artifact from its official PyPI wheels
# (ARTIFACT_GUIDE.md rule 1's "pip install --target=<artifact>/vendor"
# convention) rather than depending on a system -devel package that may not
# exist on a given machine. NOTE: a plain `pip install --target=vendor
# tbb-devel` silently drops the wheels' `<pkg>.data/data/{include,lib,
# share}` payload (pip's --target mode only unpacks purelib/platlib, not
# the data-scheme categories these wheels use for headers/libs/cmake files)
# -- so this downloads the wheels and unpacks that payload by hand instead.
TBB_VENDOR="$HERE/vendor"
if [ ! -f "$TBB_VENDOR/lib/cmake/tbb/TBBConfig.cmake" ]; then
    echo "ngap: vendoring oneTBB (headers + lib + cmake config) into $TBB_VENDOR"
    _tbb_tmp="$(mktemp -d)"
    ( cd "$_tbb_tmp" && env -u LD_LIBRARY_PATH -u LD_PRELOAD "${PY:-python3}" -m pip download --no-deps \
          "tbb-devel==${KB_TBB_VERSION:-2023.1.0}" "tbb==${KB_TBB_VERSION:-2023.1.0}" )
    "${PY:-python3}" - "$_tbb_tmp" "$TBB_VENDOR" <<'PYEOF'
import zipfile, os, re, sys
tmp, dst = sys.argv[1], sys.argv[2]
for whl in os.listdir(tmp):
    if not whl.endswith(".whl"):
        continue
    z = zipfile.ZipFile(os.path.join(tmp, whl))
    for n in z.namelist():
        m = re.match(r"^[^/]+\.data/data/(.+)$", n)   # wheel's data-scheme payload only
        if not m:
            continue
        out = os.path.join(dst, m.group(1))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(n) as s, open(out, "wb") as d:
            d.write(s.read())
PYEOF
    rm -rf "$_tbb_tmp"
fi

mkdir -p "$BUILD"
cd "$BUILD"
# Clear any stale CMake cache so the linker flags below take effect on a
# re-run (idempotent; a fresh tree has none). oneTBB is vendored above:
# only the bare runtime soname exists system-wide (/usr/lib64/libtbb.so.2,
# no dev symlink), and ngap CMake links a plain -ltbb, so the vendored lib
# dir must be on the link search path + rpath.
rm -f "$BUILD/CMakeCache.txt"
# gcc 7.5 (system default) ships no <execution> header (C++17 parallel STL,
# used by src/commons/report_formatter.cpp); switch host compiler to the
# newer gcc-native/14 toolchain, same fix other artifacts on this machine
# use (graphfold/glumin/blest STATUS.md).
cmake "$SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_CUDA_FLAGS="-arch=sm_80" \
    -DCMAKE_C_COMPILER="$HOST_GCC" \
    -DCMAKE_CXX_COMPILER="$HOST_GXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$HOST_GXX" \
    -DCMAKE_PREFIX_PATH="$TBB_VENDOR" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$TBB_VENDOR/lib -Wl,-rpath,$TBB_VENDOR/lib" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$TBB_VENDOR/lib -Wl,-rpath,$TBB_VENDOR/lib"

# asyncap uses its own hand-rolled sub-Makefile hardcoded to
# /usr/local/cuda/bin/nvcc (not present on this machine, no CMake toolchain
# plumbing) -- it is a BASELINE (AsyncAP, SIGMETRICS'23), not this track's
# ngAP target, so it is skipped rather than patched. ngap/obat/ppopp12 use
# proper CMake CUDA targets and build cleanly.
make -j"${KB_MAKE_JOBS:-4}" ngap obat ppopp12

echo "ngap: build OK -> $BUILD/bin/"
ls -la "$BUILD/bin/"
