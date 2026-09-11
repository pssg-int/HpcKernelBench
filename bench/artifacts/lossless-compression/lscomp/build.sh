#!/usr/bin/env bash
# Build lsCOMP's own shared library directly with nvcc (bypassing its CMake
# build, same rationale as ../../lossy-compression/cuszp/build.sh -- CMake
# also builds the example binaries this integration doesn't need). Every
# source file compiled here is exactly CMakeLists.txt's own `LSCOMP_SOURCES`
# list -- nothing added, nothing removed, no kernel code touched. lsCOMP's
# own include/lsCOMP_entry.h already wraps its declarations in `extern "C"`,
# so no bridge/shim file is needed here (unlike ../../lossy-compression/
# cuszp-v1/, whose SC'23-era header lacks that guard).
#
# Target: A100 / sm_80 (this machine's GPU; lsCOMP's own CMakeLists defaults
# to 80;86, so sm_80 is already in its intended set, just narrowed to one).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
NVCC="${NVCC:-nvcc}"

if [ ! -d "$SRC" ]; then
  echo "source/ missing -- run:" >&2
  echo "  git clone --depth 1 https://github.com/szcompressor/lsCOMP.git $SRC" >&2
  exit 1
fi

echo "=== lsCOMP build (liblscomp.so, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

SOURCES=(
  lsCOMP_kernel.cu lsCOMP_utility.cu lsCOMP_timer.cu lsCOMP_entry.cu
)
SRCPATHS=()
for f in "${SOURCES[@]}"; do SRCPATHS+=("$SRC/src/$f"); done

"$NVCC" -O3 -arch=sm_80 -std=c++17 -Xcompiler -fPIC --shared \
    -I "$SRC/include" -I "$SRC/src" \
    "${SRCPATHS[@]}" \
    -o "$HERE/liblscomp.so"

echo "Built: $HERE/liblscomp.so"
echo "=== done ==="
