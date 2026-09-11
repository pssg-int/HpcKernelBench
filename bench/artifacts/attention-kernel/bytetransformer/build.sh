#!/usr/bin/env bash
# Build the ByteTransformer padding-free fused-attention binding (our own
# setup.py, our own attn_binding.cu + the artifact's unmodified
# attention_fused.cu / attention_fused_long.cu -- see attn_binding.cu's
# docstring for why this needs no CUTLASS checkout).
#
# Same two machine-specific workarounds as every other torch-CUDA-extension
# build on this machine (identical derivation to
# bench/artifacts/attention-kernel/pat/build.sh / bench/artifacts/sddmm/
# fused3s/build.sh): this venv's python bakes a `-B .../compiler_compat`
# into distutils' CXX, which makes g++ silently pick up an ancient cc1plus
# and fails torch's `#if __GNUC__ < 9` check, and bakes an old-libstdc++
# rpath into LDSHARED. Fix: point CXX/CC at the real system compiler and
# override LDSHARED with a clean link line into this venv's own torch/lib.
#
# Idempotent: exit 0 if the compiled extension is already present (set
# FORCE_REBUILD=1 to force a rebuild).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

if compgen -G "$HERE"/bt_attn_binding.cpython-*.so > /dev/null && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[bytetransformer] extension already built, skipping (set FORCE_REBUILD=1 to rebuild)"
    exit 0
fi

if [[ ! -d "$HERE/source/bytetransformer" ]]; then
    echo "[bytetransformer] ERROR: $HERE/source (git clone of bytedance/ByteTransformer) is missing" >&2
    echo "  re-clone: git clone --depth 1 https://github.com/bytedance/ByteTransformer.git $HERE/source" >&2
    exit 1
fi

export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"   # A100 (sm_80); matches setup.py's explicit gencode too
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export LDFLAGS=""
TORCH_LIB="$("$PY" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LDSHARED="$CXX -pthread -shared -Wl,-rpath,$TORCH_LIB"

echo "[bytetransformer] python: $($PY --version), torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "[bytetransformer] host compiler: $($CXX --version | head -1)"
echo "[bytetransformer] nvcc: $(nvcc --version | tail -1)"

rm -rf build
"$PY" setup.py build_ext --inplace

SO="$(compgen -G 'bt_attn_binding.cpython-*.so' || true)"
if [[ -z "$SO" ]]; then
    echo "[bytetransformer] ERROR: build finished but no bt_attn_binding*.so found" >&2
    exit 1
fi
echo "[bytetransformer] built $SO"

echo "[bytetransformer] verifying import..."
"$PY" -c "
import sys
sys.path.insert(0, '$HERE')
import torch
import bt_attn_binding
print('[bytetransformer] import OK:', bt_attn_binding.fused_rm_attention)
"

echo "[bytetransformer] build.sh done."
