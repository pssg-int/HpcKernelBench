#!/usr/bin/env bash
# Build cuSZp v1's own shared library directly with nvcc (bypassing its
# CMake build, same rationale as ../cuszp/build.sh). This is the SC'23
# ("cuSZp": kernel-fusion, single mode) code path -- tag cuSZp-V1.1 of the
# same szcompressor/cuSZp repository the SC'25 "VGC"/cuSZp3 adapter
# (../cuszp/) wraps, checked out into a SEPARATE directory (see
# source.provenance) so this adapter's build never disturbs that one's.
# Every source file compiled here is exactly CMakeLists.txt's own
# `cuSZp_SOURCES`-equivalent `target_sources()` list at this tag -- nothing
# added, nothing removed, no kernel code touched.
# Target: A100 / sm_80 (this machine's GPU), same arch-flag narrowing as
# every other GPU artifact in this benchmark (V1.1's own CMakeLists
# defaults to a multi-arch list 60;61;62;70;75).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
NVCC="${NVCC:-nvcc}"

if [ ! -d "$SRC" ]; then
  echo "source/ missing -- run:" >&2
  echo "  git clone https://github.com/szcompressor/cuSZp.git $SRC" >&2
  echo "  git -C $SRC checkout cuSZp-V1.1" >&2
  exit 1
fi

echo "=== cuSZp v1 (SC'23) build (libcuszp_v1.so, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

# f32 + f64 (f64 pulled in for free since cuSZp_utility.cu/cuSZp_timer.cu are
# shared translation units this tag's CMakeLists always builds together;
# only the f32 entry points are wrapped by adapter.py).
SOURCES=(
  cuSZp_f32.cu cuSZp_f64.cu
  cuSZp_utility.cu cuSZp_timer.cu
  cuSZp_entry_f32.cu cuSZp_entry_f64.cu
)
SRCPATHS=()
for f in "${SOURCES[@]}"; do SRCPATHS+=("$SRC/src/$f"); done

# bridge_cuszp_v1.cu: extern "C" re-export (v1's own header has no extern
# "C" guard, so its symbols come out Itanium-mangled -- see that file's
# docstring). Lives here in cuszp-v1/, not inside source/.
"$NVCC" -O3 -arch=sm_80 -std=c++17 -Xcompiler -fPIC --shared \
    -I "$SRC/include" -I "$SRC/src" \
    "${SRCPATHS[@]}" "$HERE/bridge_cuszp_v1.cu" \
    -o "$HERE/libcuszp_v1.so"

echo "Built: $HERE/libcuszp_v1.so"
echo "=== done ==="
