#!/usr/bin/env bash
# Build DiaQ's CPU library + Python bindings (pybind11 module `diaq`).
#
# The paper's given artifact URL (diaq_for_hamsim, see ../source_paper_repo_hamsim)
# is a benchmark/application suite that only USES DiaQ as an external dependency
# (its README says "Install DiaQ first, see srikarchundury/diaq") -- it does not
# vendor the kernel itself. The actual SpMV kernel lives in the sibling repo
# srikarchundury/diaq, cloned here as ./source. adapter.py wraps THAT library.
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

echo "=== DiaQ build (CPU + pybind11 python module) ==="
echo "nvcc:   $(nvcc --version 2>/dev/null | tail -1 || echo 'not used (CPU build)')"
echo "gcc:    $(gcc --version | head -1)"
echo "cmake:  $(cmake --version | head -1)"
echo "python: $($PY --version) ($PY)"

cd "$SRC"

# Vendor pybind11 + parallel-hashmap submodules (header-only deps; no sudo).
if [[ ! -f dependencies/pybind11/CMakeLists.txt ]] || [[ ! -f dependencies/parallel-hashmap/CMakeLists.txt ]]; then
    git submodule update --init --recursive
fi

# Configure + build directly with cmake (NOT via the repo's own build.sh,
# which would also `cmake --install` the module into plexus_env's shared
# site-packages -- we deliberately do not touch that shared venv; adapter.py
# instead sys.path-inserts this build dir).
rm -rf cpu_build && mkdir cpu_build
cmake -S . -B cpu_build \
    -DCMAKE_BUILD_TYPE=Release \
    -DMANUAL_SIMD=ON \
    -DMAKE_TESTS=OFF \
    -DAVOID_ZERO_DIAGS=ON \
    -DBUILD_PYTHON=ON \
    -DPython3_EXECUTABLE="$PY"

cmake --build cpu_build -j "$(nproc)"

SO=$(find cpu_build/lib -maxdepth 1 -name 'diaq.cpython-*.so' | head -1)
if [[ -z "$SO" ]]; then
    echo "BUILD FAILED: diaq python extension not found under cpu_build/lib" >&2
    exit 1
fi
echo "Built: $SO"
echo "=== done ==="
