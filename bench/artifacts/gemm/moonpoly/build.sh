#!/usr/bin/env bash
# Build MoonPoly-dev's `moonpoly` pybind extension (moonpoly/core/*.cu via
# setup.py's torch CUDAExtension). Idempotent: exit 0 if already built.
#
# MoonPoly-dev commit : 22f638da0ba5c6b2bf4a01305cee0d4c95154242
# CUTLASS commit       : 76c96b0be35cb263debe3e3d8418b80911a544ab  (pinned by
#                        MoonPoly-dev's own README/integrations/cutlass/README.md)
#
# CUTLASS is not a git submodule in this repo (no .gitmodules; 3rdparty/ is
# merely .gitignore'd) -- vendored manually below at the pinned commit, per
# ARTIFACT_GUIDE.md's "header-only deps: vendor into source/" allowance.
#
# integrations/cutlass/cutlass_4_1_pattern2_twin_gemm.patch is REQUIRED (not
# optional/vLLM-only): setup.py's own source list includes
# core/fp16/fp16_rcc.cu, which references the patch's TwinGemm/Pattern-2
# additions, so the pip build fails to compile without it. The patch is
# purely additive (new methods on existing CUTLASS device::Gemm classes +
# one new header, twin_gemm.h) -- a build-system/vendoring change, not a
# touch to MoonPoly's own kernel code, so applying it is in-scope per
# ARTIFACT_GUIDE.md rule 3.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
CUTLASS_PIN="76c96b0be35cb263debe3e3d8418b80911a544ab"

CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"   # set by toolchain.sh (KB_CUDA_HOME); Perlmutter fallback
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
# nvcc 12.9 rejects the system default g++-14 (__has_construct undefined
# against its libstdc++ headers) -- same fix as
# bench/artifacts/spmm/inferfast/build.sh.
export CC="${KB_GCC12:-/usr/bin/gcc-12}"
export CXX="${KB_GXX12:-/usr/bin/g++-12}"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

if [ ! -d "$SRC/moonpoly" ]; then
    echo "build.sh: $SRC/moonpoly missing -- clone MoonPoly-dev into source/ first" >&2
    exit 1
fi

# ---- vendor CUTLASS at the pinned commit, if not already present ----------
if [ ! -d "$SRC/3rdparty/cutlass/.git" ]; then
    mkdir -p "$SRC/3rdparty"
    rm -rf "$SRC/3rdparty/cutlass"
    mkdir -p "$SRC/3rdparty/cutlass"
    (
        cd "$SRC/3rdparty/cutlass"
        git init -q
        git remote add origin https://github.com/NVIDIA/cutlass.git
        # shallow fetch of the exact pinned SHA (GitHub allows fetching an
        # arbitrary reachable commit, not just refs) -- avoids a full-history
        # clone of a multi-hundred-MB repo for one commit.
        git fetch --depth 1 origin "$CUTLASS_PIN"
        git checkout -q FETCH_HEAD -b vendor-pin
    )
fi
got_sha="$(cd "$SRC/3rdparty/cutlass" && git rev-parse HEAD)"
if [ "$got_sha" != "$CUTLASS_PIN" ]; then
    echo "build.sh: vendored CUTLASS at $got_sha, expected $CUTLASS_PIN" >&2
    exit 1
fi

# ---- apply MoonPoly's required CUTLASS compatibility patch (idempotent) ---
# The original marker ("get_grid_shape" in gemm.h) was a false positive: that
# substring already exists in STOCK, unpatched CUTLASS at this pin (a
# pre-existing call `threadblock_swizzle.get_grid_shape(...)` at gemm.h:477,
# unrelated to the patch's own new `get_grid_shape()` method additions) -- so
# the check always read "already patched" and silently skipped `git apply` on
# a truly fresh vendor fetch, surfacing as `namespace "cutlass::gemm::device"
# has no member "TwinGemm"` when core/fp16/fp16_rcc.cu (which needs the
# patch) compiled. `git apply --check --reverse` is the robust idempotency
# test: it succeeds only if the patch is ALREADY applied.
if ! (cd "$SRC" && git apply --check --reverse integrations/cutlass/cutlass_4_1_pattern2_twin_gemm.patch 2>/dev/null); then
    (cd "$SRC" && git apply integrations/cutlass/cutlass_4_1_pattern2_twin_gemm.patch)
fi

# ---- build the pybind extension in-place ----------------------------------
if [ -f "$SRC"/moonpoly*.so ] || compgen -G "$SRC/moonpoly.cpython-*.so" > /dev/null 2>&1 || \
   compgen -G "$SRC/build/lib.*/moonpoly*.so" > /dev/null 2>&1; then
    # already built (setup.py build_ext --inplace drops moonpoly*.so next to setup.py)
    if compgen -G "$SRC/moonpoly.cpython-*.so" > /dev/null 2>&1; then
        echo "build.sh: moonpoly extension already built at $SRC -- skipping rebuild"
        exit 0
    fi
fi

cd "$SRC"
"$PY" setup.py build_ext --inplace

if ! compgen -G "$SRC/moonpoly.cpython-*.so" > /dev/null 2>&1; then
    echo "build.sh: build_ext finished but no moonpoly.cpython-*.so found under $SRC" >&2
    exit 1
fi

echo "build.sh: built OK -- $(compgen -G "$SRC/moonpoly.cpython-*.so")"
