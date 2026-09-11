#!/usr/bin/env bash
# Build libopthuffdec.so from OptHuffDec's own unmodified sources
# (source/opt-gap-array/{src,encoder/src}) plus bridge_opthuffdec.cc (this
# directory -- a thin driver replacing demo.cc, the paper's own benchmark
# script; see that file's docstring). Every compiled artifact source file
# is exactly the file list source/opt-gap-array/Makefile's own `link`
# target names (encoder .cc files + decoder .cc files + the one .cu kernel
# file) -- nothing added, nothing removed, no kernel code touched.
#
# Host files (.cc) are compiled with g++ (matches the artifact's own
# Makefile: `CC = g++`, even though they call CUDA runtime APIs via
# templates in cuhd_gpu_memory_buffer.hxx -- host-callable calls only, no
# __global__ kernels in these files). Only cuhd_gpu_decoder.cu (the actual
# GPU kernel) is compiled with nvcc, exactly as upstream does.
#
# Target: A100 / sm_80 (this machine's GPU; upstream's own Makefile
# defaults to sm_70 for a V100 -- narrowed/updated here, a build-system
# arch-flag change per ARTIFACT_GUIDE rule 3, not a kernel-code change).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source/opt-gap-array"
CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
NVCC="${NVCC:-nvcc}"
ARCH=80

if [ ! -d "$SRC" ]; then
  echo "source/ missing -- run:" >&2
  echo "  git clone --depth 1 https://github.com/codyjrivera/ipdps22-opthuffdec.git $HERE/source" >&2
  exit 1
fi

echo "=== OptHuffDec (opt-gap-array) build (libopthuffdec.so, sm_80) ==="
echo "g++: $($CXX --version | head -1)"
echo "nvcc: $($NVCC --version | tail -1)"

BUILD="$HERE/build_obj"
mkdir -p "$BUILD"

INC="-I $SRC/include -I $SRC/encoder/include"
# -include cstdint: cuhd_constants.h uses std::uint16_t/std::uint32_t
# without including <cstdint> itself (relied on some other header
# transitively pulling it in on the older GCC/libstdc++ this was developed
# against; GCC 14's libstdc++ does not do so here) -- a force-include is a
# build-system fix (matches ARTIFACT_GUIDE rule 3: "Build-system fixes ...
# are fine; touching kernel code is not"), zero edits to source/.
CXXFLAGS="-std=c++17 -O3 -fPIC -Wno-deprecated-declarations -include cstdint"

HOST_SOURCES=(
  "$SRC/src/cuhd_codetable.cc"
  "$SRC/src/cuhd_gpu_codetable.cc"
  "$SRC/src/cuhd_gpu_decoder_memory.cc"
  "$SRC/src/cuhd_gpu_input_buffer.cc"
  "$SRC/src/cuhd_gpu_output_buffer.cc"
  "$SRC/src/cuhd_input_buffer.cc"
  "$SRC/src/cuhd_output_buffer.cc"
  "$SRC/src/cuhd_util.cc"
  "$SRC/encoder/src/llhuffman_encoder.cc"
)

OBJS=()
for f in "${HOST_SOURCES[@]}"; do
  o="$BUILD/$(basename "${f%.cc}").o"
  "$CXX" $CXXFLAGS $INC -I "$CUDA_HOME/include" -c "$f" -o "$o"
  OBJS+=("$o")
done

# The one real GPU kernel file -- OptHuffDec's actual contribution.
# --pre-include cstdint: same missing-include issue as the host files above
# (nvcc's own recognized flag for this, unlike the generic "-include X"
# two-token form which confuses its argument parser).
KERNEL_OBJ="$BUILD/cuhd_gpu_decoder.o"
"$NVCC" -O3 -arch=sm_${ARCH} -std=c++14 -Xcompiler -fPIC --pre-include cstdint \
    $INC -c "$SRC/src/cuhd_gpu_decoder.cu" -o "$KERNEL_OBJ"
OBJS+=("$KERNEL_OBJ")

# Our own bridge (not part of source/).
BRIDGE_OBJ="$BUILD/bridge_opthuffdec.o"
"$CXX" $CXXFLAGS $INC -I "$CUDA_HOME/include" -c "$HERE/bridge_opthuffdec.cc" -o "$BRIDGE_OBJ"
OBJS+=("$BRIDGE_OBJ")

"$NVCC" -shared -arch=sm_${ARCH} "${OBJS[@]}" -o "$HERE/libopthuffdec.so"

echo "Built: $HERE/libopthuffdec.so"
echo "=== done ==="
