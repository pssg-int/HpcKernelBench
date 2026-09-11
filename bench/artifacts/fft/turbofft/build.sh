#!/usr/bin/env bash
# Build TurboFFT's own codegen'd radix-2 kernels + our thin extern "C"
# launcher (turbofft_shim.cu, this dir, NOT part of the artifact) into
# turbofft_shim.so.
#
# TurboFFT (PPoPP'25, github.com/shixun404/TurboFFT) ships its FFT kernels
# as Python-CODE-GENERATED .cuh files (source/TurboFFT/include/code_gen/
# generated/): the repo as cloned only shipped generated/float2/*.cuh for
# logN in {7,8,9,10}, and even those defined a different, stale kernel
# (fft_10(float2*, float2*, int), 3 args) than the one
# include/TurboFFT.h's ARCH_SM==80 block #includes and calls
# (fft_radix_2<float2, N, dim_id, if_thread_ft, if_ft, if_err_injection>,
# 6 args) -- this is what the first pass (2026-08-08) recorded as
# BUILD-FAILED. Both problems are fixed by simply RUNNING the artifact's own
# generator (source/TurboFFT/include/code_gen/scripts/fft_codegen.py),
# which the artifact's own run_A100.sh calls but never actually invokes in
# its shipped (mostly-commented-out) form -- see STATUS.md "Codegen flow
# actually used" section for the exact diff against run_A100.sh.
#
# Step 1 regenerates ONLY the plain (if_thread_ft=0, if_ft=0,
# if_err_injection=0), float2, logN=1..13 (N=2..8192, TurboFFT's own
# single-kernel-launch param-table rows) base kernel files that
# turbofft_shim.cu includes. This call is idempotent: fft_codegen.py's
# save_generated_code() always opens each output file in 'w' (truncate)
# mode when if_thread_ft==0 and if_ft==0 (see the script), so re-running it
# with the same flags always reproduces the same single-specialization
# files regardless of prior runs (e.g. a fuller `--if_ft 1` etc. sweep run
# separately to prove the WHOLE TurboFFT.h still compiles -- see STATUS.md
# "full-header probe").
#
# Step 2 compiles turbofft_shim.cu (which #includes those generated .cuh
# files plus the tiny template-declaration/macro headers TurboFFT.h itself
# uses) directly with nvcc into a shared library -- NOT via TurboFFT's own
# CMakeLists.txt (which builds the full main.cu benchmark binary, cuFFT/
# VkFFT comparison code included; ARTIFACT_GUIDE.md rule 1: wrap the
# kernel, not the paper's benchmark script).
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
SCRIPTS="$SRC/TurboFFT/include/code_gen/scripts"
PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
CXX="${KB_GXX12:-g++-12}"

echo "=== turbofft build (turbofft_shim.so, sm_80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host cxx: $($CXX --version | head -1)"
echo "python: $($PY --version 2>&1)"

echo "--- step 1: codegen (fft_codegen.py --gpu A100 --datatype float2, plain flags) ---"
( cd "$SCRIPTS" && "$PY" fft_codegen.py --gpu A100 --datatype float2 )

# Sanity: the 13 base files the shim includes must exist and each define
# EXACTLY the plain specialization (no leftover FT variants from a prior
# --if_ft/--if_thread_ft sweep run against this same generated/ tree).
GEN="$SRC/TurboFFT/include/code_gen/generated/float2"
for logn in $(seq 1 13); do
    f="$GEN/fft_radix_2_logN_${logn}_upload_0.cuh"
    [[ -f "$f" ]] || { echo "BUILD FAILED: codegen did not produce $f" >&2; exit 1; }
    n=$(grep -c '^__global__' "$f")
    [[ "$n" == "1" ]] || { echo "BUILD FAILED: $f has $n specializations, expected 1 (stale generated/ tree?)" >&2; exit 1; }
done
echo "codegen OK: 13 single-specialization base files (logN 1..13) present"

echo "--- step 2: compile turbofft_shim.so ---"
nvcc -std=c++17 -O3 -gencode arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80} \
    -ccbin "$CXX" \
    -Xcompiler=-fPIC \
    -I "$SRC" \
    -shared -o "$HERE/turbofft_shim.so" \
    "$HERE/turbofft_shim.cu"

if [[ ! -f "$HERE/turbofft_shim.so" ]]; then
    echo "BUILD FAILED: turbofft_shim.so not produced" >&2
    exit 1
fi
echo "Built: $HERE/turbofft_shim.so"
echo "=== done ==="
