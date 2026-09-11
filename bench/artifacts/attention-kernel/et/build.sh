#!/usr/bin/env bash
# Build the E.T. dense self-attention binding (our own setup.py, our own
# attn_binding.cu + the artifact's unmodified kernels/attention.cu -- E.T.
# ships no python bindings of its own, only CMake C++ test binaries).
#
# Same two machine-specific workarounds as every other torch-CUDA-extension
# build on this machine (see bench/artifacts/attention-kernel/pat/build.sh /
# bench/artifacts/attention-kernel/bytetransformer/build.sh for the fuller
# derivation): this venv's python bakes a `-B .../compiler_compat` into
# distutils' CXX (routes g++ through an ancient cc1plus, fails torch's
# `#if __GNUC__ < 9` check) and bakes an old-libstdc++ rpath into LDSHARED.
#
# Idempotent: exit 0 if the compiled extension is already present (set
# FORCE_REBUILD=1 to force a rebuild).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

if compgen -G "$HERE"/et_attn_binding.cpython-*.so > /dev/null && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[et] extension already built, skipping (set FORCE_REBUILD=1 to rebuild)"
    exit 0
fi

if [[ ! -d "$HERE/source/kernels" ]]; then
    echo "[et] ERROR: $HERE/source (git clone of cctry/E.T.) is missing" >&2
    echo "  re-clone: git clone --depth 1 https://github.com/cctry/E.T..git $HERE/source" >&2
    exit 1
fi

# PATCH recorded in source.patch (ARTIFACT_GUIDE rule 3 -- minimal,
# non-kernel-logic, header-syntax-only): kernels/kernels.h's last declaration
# (sharedQK_attention_kernelLauncher) is missing its terminating semicolon in
# the artifact's own upstream repo (confirmed via `git diff` against the
# pristine clone -- `\ No newline at end of file` right after the `)`). Every
# translation unit that includes this header after it (our attn_binding.cu)
# fails to parse: the next declaration's tokens get absorbed into this one,
# producing a cascade of unrelated-looking errors ("out already declared",
# "expression must have class type", ...). A one-character build-system-level
# fix (add `;`), no computation changed. Applied idempotently here so a fresh
# `git clone` of source/ (git-ignored repo-wide) rebuilds without a manual step.
if ! grep -q "cudaStream_t stream = nullptr);" "$HERE/source/kernels/kernels.h"; then
    echo "[et] patching kernels/kernels.h: missing trailing semicolon (see source.patch)"
    sed -i '$ s/cudaStream_t stream = nullptr)$/cudaStream_t stream = nullptr);/' \
        "$HERE/source/kernels/kernels.h"
fi

export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"   # A100 (sm_80); the artifact's own CMakeLists.txt
                                    # hardcodes CMAKE_CUDA_ARCHITECTURES=70 (V100), unused
                                    # here since we don't build via its CMake.
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export LDFLAGS=""
TORCH_LIB="$("$PY" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LDSHARED="$CXX -pthread -shared -Wl,-rpath,$TORCH_LIB"

echo "[et] python: $($PY --version), torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "[et] host compiler: $($CXX --version | head -1)"
echo "[et] nvcc: $(nvcc --version | tail -1)"

rm -rf build
"$PY" setup.py build_ext --inplace

SO="$(compgen -G 'et_attn_binding.cpython-*.so' || true)"
if [[ -z "$SO" ]]; then
    echo "[et] ERROR: build finished but no et_attn_binding*.so found" >&2
    exit 1
fi
echo "[et] built $SO"

echo "[et] verifying import..."
"$PY" -c "
import sys
sys.path.insert(0, '$HERE')
import torch
import et_attn_binding
print('[et] import OK:', et_attn_binding.otf_attention)
"

echo "[et] build.sh done."
