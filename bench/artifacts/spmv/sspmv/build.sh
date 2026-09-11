#!/usr/bin/env bash
# Build SSpMV (the LeSpMV library, source/LeSpMV) + a thin ctypes shim
# (csr_shim.cpp, this directory) that exposes its LeSpMV_csr<int,double>
# kernel with an extern "C" entry point. See STATUS.md for the two
# build-system patches applied to source/LeSpMV/CMakeLists.txt (forced icx
# compiler with no fallback; AVX-512-only spmv_csr5.cpp on a non-AVX-512
# CPU) and why they're build-system, not kernel, fixes.
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_SRC="$HERE/source/LeSpMV"

echo "=== SSpMV build (CPU: LeSPMV_shared + csr_shim.so) ==="
echo "gcc:   $(g++ --version | head -1)"
echo "cmake: $(cmake --version | head -1)"

rm -rf "$LIB_SRC/build" && mkdir "$LIB_SRC/build"
# -static-libstdc++ -static-libgcc: this login node's Python (ctypes/CDLL)
# resolves libstdc++.so.6 from an older NERSC conda env earlier in its
# runtime search path than this g++'s own (newer) libstdc++, which lacks
# symbol version CXXABI_1.3.15 that g++-14-compiled code needs -- observed
# directly as `OSError: .../libstdc++.so.6: version 'CXXABI_1.3.15' not
# found` when ctypes.CDLL loaded libLeSPMV.so. Statically linking the C++
# runtime into both .so's sidesteps the whole runtime resolution order
# instead of fighting RPATH/RUNPATH/LD_LIBRARY_PATH precedence.
cmake -S "$LIB_SRC" -B "$LIB_SRC/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
# -j "$(nproc)" (=128 on this login node) exhausts the shared-session
# RLIMIT_NPROC=256 when several artifact builds run concurrently on the
# same login node ("cc1plus: ... vfork: Resource temporarily unavailable",
# observed directly here) -- cap parallelism, overridable, default lowered
# from nproc per bench/artifacts/toolchain.sh's login-node guidance.
cmake --build "$LIB_SRC/build" -j "${KB_MAKE_J:-8}" --target LeSPMV_shared

SO="$LIB_SRC/build/libLeSPMV.so"
if [[ ! -f "$SO" ]]; then
    echo "BUILD FAILED: $SO not produced" >&2
    exit 1
fi

echo "--- building csr_shim.so against $SO ---"
g++ -O3 -march=native -fopenmp -std=c++17 -fPIC -shared \
    -static-libstdc++ -static-libgcc \
    -I "$LIB_SRC/include" \
    "$HERE/csr_shim.cpp" \
    -L "$LIB_SRC/build" -lLeSPMV \
    -Wl,-rpath,"$LIB_SRC/build" \
    -o "$HERE/csr_shim.so"

echo "Built: $SO"
echo "Built: $HERE/csr_shim.so"
echo "=== done ==="
