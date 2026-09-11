#!/usr/bin/env bash
# Build MP-SpMM's preprocessing tool (their own matching+padding, unmodified)
# and a thin wrapper .so exposing the artifact's own kernel launch + metadata
# packing (see wrapper.cu's header comment). Idempotent; exit 0 = both built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"   # nvcc 12.9 -ccbin g++-14
    # (this login node's system default) fails on <bits/alloc_traits.h>
    # ("identifier __has_construct is undefined") for ANY translation unit
    # that includes <vector>/<iostream> -- an nvcc/libstdc++-14 mismatch
    # unrelated to this artifact, first hit and fixed the same way in
    # ../inferfast/build.sh. g++-12 (SUSE, ships alongside CUDA 12.x here)
    # compiles clean.
GXX="${GXX:-${KB_GXX12:-/usr/bin/g++-12}}"   # for the plain host-only preprocessing tool

echo "== MP-SpMM build =="
echo "provenance: source/ is NOT a git clone -- the paper's own GitHub repo"
echo "  (https://github.com/CGCL-codes/MP-SpMM_SC25) ships only a README"
echo "  pointing at a Zenodo artifact-evaluation release; see source.provenance."
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler (preprocessing + wrapper): $($GXX --version | head -1)"

# --- 0/2: fetch source/ from Zenodo if not already present. toolchain.sh's
#     kb_ensure_source() can't recreate this one (source.provenance has no
#     `reclone=git clone` recipe, only a `refetch=` curl+unzip recipe, since
#     this artifact's real code is a Zenodo AD/AE code-drop, not a git repo --
#     see source.provenance). Reproduces that refetch= recipe here so a fresh
#     clone of this repository builds without a manual step. Idempotent.
if [ ! -d source ]; then
    echo "-- 0/2: fetching MP-SpMM source from Zenodo (source.provenance) --"
    ZIP="$(mktemp -t mp-spmm-XXXXXX.zip)"
    # Conda's LD_LIBRARY_PATH (CUDA math libs, exported by toolchain.sh)
    # shadows the system libcurl with an ABI-mismatched one -- "curl: (4) A
    # requested feature, protocol, or option was not found built-in" -- so
    # unset it for this one curl call (build-system download-env fix, not a
    # workaround of artifact code; same fix used by other build.sh files on
    # this machine, see bench/artifacts/spmm/README.md gotchas).
    env -u LD_LIBRARY_PATH -u LD_PRELOAD curl -L -o "$ZIP" \
        "https://zenodo.org/records/16933452/files/MP-SpMM_SC25.zip?download=1"
    echo "7aacfbc60cdc0c535bf666538cbe2046  $ZIP" | md5sum -c -
    mkdir -p source
    unzip -q "$ZIP" -d source
    # The Zenodo zip extracts to source/MP-SpMM_SC25/... -- flatten it up one
    # level so build.sh's own source/mpspmm/... paths (below) resolve.
    shopt -s dotglob nullglob
    mv source/MP-SpMM_SC25/* source/
    shopt -u dotglob nullglob
    rmdir source/MP-SpMM_SC25
    rm -f "$ZIP"
fi

echo "-- 1/2: preprocessing tool (their own 2:4 matching+padding, unmodified) --"
"$GXX" -std=c++17 -Ofast -Wno-unused-result -Wno-write-strings \
    -o mpspmm_preprocess \
    source/mpspmm/preprocessing/impl-iterative-2-4.cpp \
    source/mpspmm/preprocessing/libmmio/mmio.c \
    -I source/mpspmm/preprocessing/libmmio

echo "-- 2/2: libmpspmm_wrapper.so (kernel launch + metadata-pack glue) --"
"$NVCC" -ccbin "$HOST_COMPILER" -O3 -std=c++17 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -I source/mpspmm/SpMM \
    -shared -o libmpspmm_wrapper.so wrapper.cu

echo "-- verifying symbols/libs resolve --"
ldd libmpspmm_wrapper.so | grep -q "not found" && \
    { echo "MISSING SHARED LIBS:"; ldd libmpspmm_wrapper.so | grep "not found"; exit 1; }
[ -x ./mpspmm_preprocess ] || { echo "mpspmm_preprocess did not build"; exit 1; }
echo "== MP-SpMM build: OK =="
