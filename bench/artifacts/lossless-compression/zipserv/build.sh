#!/usr/bin/env bash
# Build ZipServ's own TCA-TBE compress (CPU) + decompress (GPU kernel) into
# a shared library via bridge_zipserv.cu (this directory; see its docstring
# for exactly which two functions are wrapped and why they are the finest
# available boundary -- the standalone decompress kernel, separate from
# both the fused ZipGEMM kernel and vLLM serving internals).
#
# ZipServ's own build.sh path (`python setup.py install`) builds a
# torch CUDAExtension for the WHOLE package (csrc/*.cu + LInfer_py's
# pybind11 bindings, requiring a working PyTorch/vLLM conda env) -- this
# integration instead compiles exactly the two source files the bridge
# needs directly with nvcc, bypassing setup.py/pybind11/torch entirely
# (same pattern as source/build/Makefile, ZipServ's OWN minimal
# libL_API.so recipe for csrc/L_API.cu alone -- reused here, just retargeted
# to sm_80 and with this integration's own bridge file added).
#
# Target: A100 / sm_80 (ZipServ's own build/Makefile and kernel_benchmark/
# Makefile both hardcode `-gencode arch=compute_89,code=sm_89` for an
# RTX 4090; overridden here for this machine's A100).
#
# Idempotent: safe to re-run. Exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source"
NVCC="${NVCC:-nvcc}"

echo "=== ZipServ build (bridge.so, sm_80) ==="
echo "nvcc: $($NVCC --version | tail -1)"

"$NVCC" -O3 -arch=sm_80 -std=c++17 -Xcompiler -fPIC --shared \
    -I "$SRC/csrc" -I "$SRC/kernel_benchmark" \
    "$HERE/bridge_zipserv.cu" \
    "$SRC/csrc/L_API.cu" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
