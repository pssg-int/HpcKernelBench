#!/usr/bin/env bash
# Build RoDe's SDDMM kernel (RoDe_SDDMM/RoDeSddmm.cu, unmodified) together
# with this directory's wrapper.cu (ctypes glue + the same dependency-free
# port of RoDe's row-decomposition preprocessing used by the sibling
# spmm/rode adapter -- see wrapper.cu's header comment).
#
# `source/` here is a SYMLINK to ../../spmm/rode/source (read-only reuse,
# not re-cloned) -- see source.provenance.
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): see bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

if [ ! -e source/RoDe_SDDMM/RoDeSddmm.cu ]; then
    echo "source/ symlink missing or broken (expected -> ../../spmm/rode/source)" >&2
    exit 1
fi

NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}}"

echo "== RoDe SDDMM build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

# -include cstdint: source/utils/common_utils.h uses uint32_t without
# including <cstdint> itself (same finding as the spmm/rode build -- this
# newer nvcc/libstdc++ pairing no longer pulls it in transitively).
#
# -Ishim_include BEFORE -Isource/utils: RoDeSddmm.cu (unlike RoDeSpmm.cu)
# has `#include "matrix_utils.h"` at file scope; the real header pulls in
# abseil-cpp (unavailable -- see shim_include/matrix_utils.h's header
# comment for the full grep-verified justification that this include is
# dead code). Putting our directory first on the include path makes the
# quoted include resolve to the empty stub instead -- an include-path
# substitution (ARTIFACT_GUIDE.md rule 3: "include paths ... are fine"),
# not a patch to any file inside source/.
"$NVCC" -ccbin "$HOST_COMPILER" -O3 -std=c++17 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} --use_fast_math \
    -include cstdint \
    -I. -Ishim_include -Isource/RoDe_SDDMM -Isource/utils \
    -shared -o librode_sddmm_wrapper.so wrapper.cu source/RoDe_SDDMM/RoDeSddmm.cu

echo "-- verifying symbols resolve --"
# Same pipefail/SIGPIPE fix as ../../spmm/rode/build.sh (see that file for
# the full derivation): nm's/ldd's dumps piped straight into `grep -q` can
# trip `set -o pipefail` when grep exits early on the first match, since the
# producer is still writing and gets SIGPIPE. Captured to variables instead.
# Build-script verification-step fix only, not a source/ change.
LDD_OUT="$(ldd librode_sddmm_wrapper.so)"
if grep -q "not found" <<<"$LDD_OUT"; then
    echo "MISSING SHARED LIBS:"; grep "not found" <<<"$LDD_OUT"; exit 1
fi
NM_OUT="$(nm -D librode_sddmm_wrapper.so)"
grep -q " T sddmm_run" <<<"$NM_OUT" || { echo "sddmm_run symbol missing"; exit 1; }
echo "== RoDe SDDMM build: OK =="
