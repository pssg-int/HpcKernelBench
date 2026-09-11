#!/usr/bin/env bash
# Build CB-SpMV's kernel + format-construction code (source/cb-spmv/src/)
# into a shared library via bridge.cu (this directory; see its docstring
# for why a bridge is needed instead of calling the artifact's own main.cu
# driver). Target: A100 / sm_80 (the artifact's own Makefile targets sm_89
# for an RTX 4090; overridden here per this machine's GPU, exactly the kind
# of arch-flag build-system fix the integration contract allows).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source/cb-spmv/src"
NVCC="${NVCC:-nvcc}"

echo "=== CB-SpMV build (bridge.so, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

# coo2block.h uses std::sort(std::execution::par, ...) (C++17 parallel
# algorithms); libstdc++'s implementation of the par execution policy
# dispatches to Intel TBB at link time (undefined reference to
# tbb::task's typeinfo otherwise). TBB is available system-wide at
# /usr/lib64 on the login node this artifact was originally built on, but
# NOT on this cluster's GPU compute nodes (different node image: no
# /usr/lib64/libtbb.so.2 at all -- observed directly as `OSError:
# libtbb.so.2: cannot open shared object file` from ctypes.CDLL on a gpu
# node, even though the login-node link succeeded). Since builds run on the
# login node but the gate runs on a GPU node, linking against the system
# copy is not portable here: vendor the .so into this artifact's own
# (gitignored) vendor/ dir at build time and rpath bridge.so at itself, so
# it is self-contained regardless of which node loads it.
TBB_LIB="${KB_TBB_LIB:-}"
if [[ -z "$TBB_LIB" ]]; then
    TBB_LIB="$(ldconfig -p 2>/dev/null | awk '/libtbb\.so\.2( |$)/{print $NF; exit}')"
fi
TBB_LIB="${TBB_LIB:-/usr/lib64/libtbb.so.2}"
mkdir -p "$HERE/vendor"
if [[ ! -e "$HERE/vendor/libtbb.so.2" ]]; then
    if [[ -f "$TBB_LIB" ]]; then
        cp "$TBB_LIB" "$HERE/vendor/libtbb.so.2"
    else
        echo "WARNING: libtbb.so.2 not found (looked at KB_TBB_LIB/ldconfig/$TBB_LIB);" \
             "bridge.so link or later load may fail" >&2
    fi
fi
# link against the versioned soname directly (libtbb.so.2, not the
# unversioned dev symlink libtbb.so, which needs a -devel package) --
# -Wl,-rpath so the loader finds the vendored copy on any node, not just
# the one build.sh ran on.
"$NVCC" -O3 -w -arch=sm_80 -Xcompiler -fopenmp -Xcompiler -fPIC \
    -shared \
    -I "$SRC" \
    "$HERE/bridge.cu" \
    -L "$HERE/vendor" -l:libtbb.so.2 \
    -Xlinker -rpath -Xlinker "$HERE/vendor" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
