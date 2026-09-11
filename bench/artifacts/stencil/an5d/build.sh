#!/usr/bin/env bash
# Build the AN5D adapter: generate bridge_<shape>.cu from AN5D-Artifact's own
# pre-generated CUDA (see gen_bridge.py's docstring for why -- the PPCG-based
# generator toolchain itself could not be built here, see STATUS.md), then
# compile each of the 6 shapes into its own bridge_<shape>.so.
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$HERE/source" ]; then
    echo "=== cloning AN5D-Artifact ==="
    git clone --depth 1 https://github.com/khaki3/AN5D-Artifact "$HERE/source"
fi

echo "=== generating bridge_<shape>.cu from AN5D-Artifact's pre-generated CUDA ==="
"${PY:-python3}" "$HERE/gen_bridge.py"

NVCC="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}/bin/nvcc"
echo "nvcc: $("$NVCC" --version | tail -1)"
# Host compiler pin (2026-09-09, zaratan): without an explicit -ccbin, nvcc
# does its own host-compiler search rather than simply taking the first g++
# on $PATH; on this machine that search finds this build env's own default
# g++ 13.4.0 (bundled alongside nvcc in the same conda env), which nvcc 12.x
# cannot parse (<type_traits>/<bits/hashtable.h> "identifier is undefined"
# errors -- same class of failure documented in
# bench/artifacts/convolution/hidet/STATUS.md's Reproduction section, and
# fixed the same way by most other artifacts in this repo, see `grep -l
# ccbin bench/artifacts/*/*/build.sh`). KB_GXX12 (exported by
# bench/artifacts/toolchain.sh, sourced above) is a g++ 12 known to work
# with this nvcc; harmless / a no-op default elsewhere (e.g. Perlmutter's
# NVIDIA HPC SDK nvcc, whose install dir has no competing gcc).
HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"
echo "host compiler: $("$HOST_COMPILER" --version | head -1)"

for shape in star2d1r box2d1r star2d3r box2d3r star3d1r box3d1r; do
    echo "=== building bridge_${shape}.so (sm_80) ==="
    "$NVCC" -ccbin "$HOST_COMPILER" -O3 -w -arch=sm_80 -std=c++17 \
        -Xcompiler -fPIC,-fvisibility=hidden \
        -shared \
        -I "$HERE" \
        "$HERE/bridge_${shape}.cu" \
        -o "$HERE/bridge_${shape}.so"
done

echo "Built: $HERE/bridge_{star2d1r,box2d1r,star2d3r,box2d3r,star3d1r,box3d1r}.so"
echo "=== done ==="
