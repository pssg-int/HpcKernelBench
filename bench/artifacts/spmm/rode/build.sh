#!/usr/bin/env bash
# Build RoDe's SpMM kernel (RoDe_SpMM/RoDeSpmm.cu, unmodified) together with
# this directory's wrapper.cu (ctypes glue + a dependency-free port of RoDe's
# own row-decomposition preprocessing -- see wrapper.cu's header comment for
# why the full SPC::SparseMatrix/abseil/Glog/gflags stack is not linked: the
# numeric kernel itself has zero dependency on any of that).
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): see bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}}"

echo "== RoDe build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"

# -include cstdint: source/utils/common_utils.h uses uint32_t without
# including <cstdint> itself; this newer nvcc/libstdc++ combination no longer
# pulls it in transitively (host-compiler/stdlib-version issue, not an
# artifact bug -- build-system fix per ARTIFACT_GUIDE.md rule 3, no source
# file touched).
"$NVCC" -ccbin "$HOST_COMPILER" -O3 -std=c++17 -Xcompiler -fPIC \
    -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} --use_fast_math \
    -include cstdint \
    -I. -Isource/RoDe_SpMM -Isource/utils \
    -shared -o librode_wrapper.so wrapper.cu source/RoDe_SpMM/RoDeSpmm.cu

echo "-- verifying symbols resolve --"
# Captured to variables rather than piped straight into grep -q: under
# `set -o pipefail`, grep -q closes its read end after the first match, and
# nm/ldd's large symbol dumps here are big enough that the producer is still
# writing when that happens, so it gets SIGPIPE (exit 141) and pipefail then
# fails the whole pipeline even though grep itself matched/didn't-match
# correctly. Not a source/kernel change -- a build-script verification-step
# fix only.
LDD_OUT="$(ldd librode_wrapper.so)"
if grep -q "not found" <<<"$LDD_OUT"; then
    echo "MISSING SHARED LIBS:"; grep "not found" <<<"$LDD_OUT"; exit 1
fi
NM_OUT="$(nm -D librode_wrapper.so)"
grep -q " T rode_run" <<<"$NM_OUT" || { echo "rode_run symbol missing"; exit 1; }
echo "== RoDe build: OK =="
