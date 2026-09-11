#!/usr/bin/env bash
# Build FastKron's PyTorch CUDA extension (pyfastkron.FastKronCUDA) from
# source/ via the artifact's own CMake build (PYMODULE=ON, ENABLE_CUDA=ON,
# ENABLE_X86=OFF -- we only need the CUDA backend). Three small, disclosed
# build-system patches were required (see STATUS.md "Build-system patches"):
#   1. source/setup.py: pass -DPython3_EXECUTABLE (modern CMake FindPython3
#      variable name) alongside the existing legacy -DPYTHON_EXECUTABLE.
#   2. source/CMakeLists.txt: request the Development.Embed Python3
#      component (pybind11_add_module(... SHARED ...) needs the full
#      Python3::Python link target, not just Python3::Module).
#   3. source/CMakeLists.txt: narrow gen_tuner_kernels.py's invocation to
#      -mm-type mkm (this adapter only ever calls GeMKM, never GeKMM) and
#      -batch-type cont (never strided/batched) -- cuts the generated/
#      compiled per-tile-config kernel count from ~2734 to ~136 for a
#      single (sm_80) architecture, so a from-scratch build completes in
#      under 5 minutes instead of 30-45. This project's login-node build
#      budget could not otherwise complete the artifact's own default
#      full sweep (every arch x mm-type x batch-type combination).
# No kernel/algorithm CODE is touched by any of the three patches -- every
# compiled .cu file is still gen_tuner_kernels.py's own, unmodified,
# verbatim-generated output; patch 3 only narrows WHICH of the artifact's
# own pre-existing template instantiations get generated and compiled.
# Pass -DCMAKE_CUDA_ARCHITECTURES=<arch list> below to build for other GPUs.
#
# This build bypasses setup.py's own `pip install .` (which unconditionally
# tries to ALSO build the X86 backend, needing find_package(OpenMP) +
# AVX-capable toolchain plumbing irrelevant to this GPU-only integration)
# and instead drives the SAME underlying CMakeLists.txt directly for the
# CUDA extension only, then stages it next to the artifact's own (unmodified)
# pure-Python pyfastkron/*.py wrapper in a runtime package directory this
# adapter imports from -- same "vendor the build output, don't touch the
# source" spirit as bench/artifacts/spmm/insum's sys.path.insert trick.
#
# Idempotent: safe to re-run; exit 0 = built. CMake/Make only rebuild what
# changed on a re-run.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
BUILD="$HERE/pybuild"
PKG="$HERE/pyfastkron_runtime"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CC CXX

echo "=== FastKron build (pyfastkron.FastKronCUDA, sm_50/70/80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"
echo "python: $PY"

mkdir -p "$BUILD" "$PKG/pyfastkron"

# pybind11 (`source/pybind11`) is a git SUBMODULE of the upstream repo
# (.gitmodules: url = ../../pybind/pybind11, pinned commit 58c382a...), which
# a plain `git clone --depth 1` (source.provenance's own reclone recipe, and
# toolchain.sh's kb_ensure_source) does NOT populate -- it is left as an
# empty placeholder directory (`git submodule status` shows a leading `-`,
# i.e. uninitialized), so `pybind11_add_module()` in CMakeLists.txt fails
# with "does not contain a CMakeLists.txt file". Reproduced here exactly as
# first done in this integration (see STATUS.md): a fresh, unpinned
# `git clone --depth 1` of upstream pybind11 vendored directly into
# source/pybind11 (NOT `git submodule update --init`, which would need the
# full, non-shallow-fetchable parent history) -- pybind11's CMake support is
# stable across recent releases, and the vendored tree itself is never
# modified. Idempotent: skipped if source/pybind11/CMakeLists.txt already exists.
if [[ ! -f "$SRC/pybind11/CMakeLists.txt" ]]; then
    echo "-- vendoring pybind11 (uninitialized submodule) --"
    rm -rf "$SRC/pybind11"
    git clone --depth 1 https://github.com/pybind/pybind11 "$SRC/pybind11"
fi

cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$BUILD/lib" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYMODULE=ON \
    -DENABLE_CUDA=ON \
    -DENABLE_X86=OFF \
    -DPython3_EXECUTABLE="$PY" \
    -DCMAKE_CUDA_ARCHITECTURES="${FASTKRON_CUDA_ARCH:-80}"

# Parallelism capped well below nproc: this is a shared login node with a
# hard RLIMIT_NPROC (256 processes) across the whole user session, and
# nvcc/gmake fan out several processes per translation unit -- -j"$(nproc)"
# (128 here) reliably exhausts that limit mid-build ("cc1plus: vfork:
# Resource temporarily unavailable"), a resource-contention failure, not a
# FastKron problem. Override with FASTKRON_BUILD_JOBS on a dedicated machine.
cmake --build "$BUILD" -j"${FASTKRON_BUILD_JOBS:-8}"

so=$(find "$BUILD/lib" -maxdepth 1 -iname "FastKronCUDA*.so" | head -1)
if [[ -z "$so" ]]; then
    echo "BUILD FAILED: no FastKronCUDA*.so produced under $BUILD/lib" >&2
    exit 1
fi

# Stage the artifact's own (unmodified) pure-Python wrapper next to the
# freshly built extension, in one importable package directory --
# source/pyfastkron/ itself has no compiled .so (setup.py's own build
# process would normally drop it there via `pip install .`; we replicate
# that placement without going through pip/X86).
cp "$SRC"/pyfastkron/*.py "$PKG/pyfastkron/"
cp "$so" "$PKG/pyfastkron/"

echo "Built: $so"
echo "Staged runtime package: $PKG/pyfastkron ($(ls "$PKG/pyfastkron"))"
echo "=== done ==="
