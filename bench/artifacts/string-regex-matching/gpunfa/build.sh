#!/usr/bin/env bash
# Build gpunfa-artifact (ASPLOS'20, conf/asplos/0002PJ20) -- obat/infant/ppopp12 CLIs.
# Idempotent: exit 0 if already built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source/gpunfa_code"
BUILD="$HERE/build"

if [ -x "$BUILD/bin/obat" ]; then
    echo "gpunfa: already built ($BUILD/bin/obat)"
    exit 0
fi

# Host compiler pin (2026-09-09, zaratan): this build env's default g++
# (13.4.0, bundled alongside nvcc in the same conda env) parses the vendored
# gpunfa_code/src/commons/SymbolStream.cpp to a broken state -- uint8_t/
# uint32_t come up "not declared" there (GCC 13's libstdc++ stopped
# transitively pulling in <cstdint> via <string>/<set>/<vector> the way
# older libstdc++ did, and the header genuinely has no direct <cstdint>
# include of its own), which then cascades into GCC's unknown-type error
# recovery silently treating the vector<uint8_t>/set<uint8_t> members as
# plain `int` for the rest of the translation unit ("request for member
# 'insert' in ..., which is of non-class type 'int'"). Not present with the
# artifact's originally-recorded g++ 7.5.0 (much looser transitive-include
# behavior). Same class of nvcc-12.x-toolchain/g++-13 friction documented in
# bench/artifacts/convolution/hidet/STATUS.md and fixed the same way
# elsewhere in this repo (see `grep -l ccbin bench/artifacts/*/*/build.sh`):
# pin to KB_GXX12 (exported by toolchain.sh, sourced above), a g++ 12 that
# does not hit this. Harmless / a no-op default on Perlmutter (system g++
# 7.5.0 there never needed this pin).
CXX_HOST="${KB_GXX12:-g++-12}"
CC_HOST="${KB_GCC12:-gcc-12}"
echo "host compiler: $("$CXX_HOST" --version | head -1)"

rm -rf "$BUILD"   # CMakeCache.txt from a prior compiler pin must not linger
mkdir -p "$BUILD"
cd "$BUILD"
cmake "$SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_CUDA_FLAGS="-arch=sm_80" \
    -DCMAKE_C_COMPILER="$CC_HOST" \
    -DCMAKE_CXX_COMPILER="$CXX_HOST" \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX_HOST"

make -j"${KB_MAKE_JOBS:-4}" obat infant ppopp12

echo "gpunfa: build OK -> $BUILD/bin/"
ls -la "$BUILD/bin/"
