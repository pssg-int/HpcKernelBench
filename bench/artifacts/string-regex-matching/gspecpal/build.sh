#!/usr/bin/env bash
# Build GSpecPal (IPDPS'22, conf/ipps/WangWQW22) -- attempted for evidence
# only; see STATUS.md for why this artifact is SKIPPED for gating even if
# it builds (DFA-table input model does not fit this track's ANML/NFA
# workload abstraction within the integration budget).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
BUILD="$HERE/build"

HOST_GXX="${HOST_GXX:-${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}}"
HOST_GCC="${HOST_GCC:-${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}}"

mkdir -p "$BUILD"
cd "$BUILD"
cmake "$SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_CUDA_FLAGS="-arch=sm_80" \
    -DCMAKE_C_COMPILER="$HOST_GCC" \
    -DCMAKE_CXX_COMPILER="$HOST_GXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$HOST_GXX"

make -j8

echo "gspecpal: build OK -> $BUILD/"
find "$BUILD" -maxdepth 2 -type f -executable
