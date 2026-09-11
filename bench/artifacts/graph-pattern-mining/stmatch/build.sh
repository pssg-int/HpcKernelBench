#!/usr/bin/env bash
# Build STMatch's stack-based pattern-matching kernel (_parallel_match,
# source/src/gpu_match.cu) as a shared library (st_shim.so), compiled from
# st_shim.cu (this dir, NOT part of the artifact) together with the
# artifact's own gpu_match.cu (unmodified). Everything else the artifact
# needs (Graph/Pattern/JobQueue/CallStack) is header-only
# (source/src/{graph,pattern,job_queue,callstack}.h) and pulled in via
# #include -- no separate .cpp files to compile, unlike GraphSet.
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"

CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"

echo "=== STMatch build (st_shim.so, sm_80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $($NVCC --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"

# -rdc=true (relocatable device code): st_shim.cu launches gpu_match.cu's
# own `_parallel_match` __global__ kernel via an `extern`-visible header
# declaration (source/src/gpu_match.cuh) -- this is the SAME separate-
# compilation scheme the artifact's OWN Makefile already uses for
# gpu_match.cu (`compile_gpu_match` passes `-dc`, then `compile_cu_test`
# links the resulting object against cu_test.cu), just retargeted from two
# .exe-producing steps to one shared library. Arch retargeted from the
# artifact's own `-arch=compute_86` (Ampere consumer GPU) to sm_80 (A100).
# -maxrregcount=64: _parallel_match is launched with BLOCK_DIM=1024
# threads/block (config.h, compile-time constant); sm_80's 65536
# registers/SM hard-caps a 1024-thread block at 64 registers/thread. Under
# nvcc 12.9's optimizer the unconstrained per-thread register count exceeds
# that (observed at runtime: "too many resources requested for launch",
# cudaErrorLaunchOutOfResources) even though the artifact's own Makefile
# builds this same kernel/BLOCK_DIM combination successfully with an older
# CUDA toolkit for a different (compute_86) architecture -- a compiler-
# version/arch-specific register-allocation difference, not a kernel logic
# bug. Capping registers is a standard build-system knob (no arithmetic
# changed) that trades a little occupancy/ILP for making this launch
# configuration valid at all.
"$NVCC" -std=c++17 -O3 -w -arch=sm_80 -rdc=true -maxrregcount=64 \
    -ccbin "$CXX" \
    -Xcompiler=-fPIC,-w \
    -I "$HERE" \
    -I "$SRC/src" \
    -shared -o "$HERE/st_shim.so" \
    "$HERE/st_shim.cu" \
    "$SRC/src/gpu_match.cu"

if [[ ! -f "$HERE/st_shim.so" ]]; then
    echo "BUILD FAILED: st_shim.so not produced" >&2
    exit 1
fi
nm -D "$HERE/st_shim.so" | grep -q "st_run" || {
    echo "st_run symbol missing from st_shim.so" >&2
    exit 1
}
echo "Built: $HERE/st_shim.so"
echo "=== done ==="
