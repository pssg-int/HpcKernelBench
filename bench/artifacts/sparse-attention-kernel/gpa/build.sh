#!/usr/bin/env bash
# Build gpa's (TomczakK25, KLab-AI3/Graph-Processing-Attention-IPDPS-2025)
# CSR sparse-FlashAttention extension (`spfa_csr`, source/Sparse_FlashAttention_CSR/).
# Of the repo's 6 kernel variants (COO/CSR explicit + Local/Local-1D-Dilated/
# Local-2D-Dilated/Global-No-Local implicit), CSR is built here: it is the
# most general (accepts ANY binary mask converted to CSR, exactly matching
# this track's arbitrary structured-mask workload) and, per this repo's own
# verification/verify.py, is cross-checked against 4 of the other 5 variants
# there (CSR==COO, CSR==Local at full density, CSR verified against dense
# PyTorch SDPA directly) -- the strongest-verified single variant to
# integrate under the login-node time budget. The other 5 variants are NOT
# built here (see adapter.py's module docstring for why CSR alone already
# generalizes over the parametric Local/Dilated/Global patterns this
# project's own workload builder produces).
#
# The README's own install instructions ("Copy the 6 folders into your
# .../site-packages/torch/ directory... python setup.py install") would
# install into the SHARED venv's site-packages -- not done here per this
# task's environment rules (no system/shared-env package installs). Instead,
# this script runs the repo's own unmodified `Sparse_FlashAttention_CSR/
# setup.py` with `build_ext --inplace`, which builds `spfa_csr*.so` directly
# inside that source directory -- importable via sys.path, no site-packages
# pollution, no repo file edited.
#
# Same machine-specific compiler workarounds as every other CUDA-extension
# build in this project (see bench/artifacts/sddmm/fused3s/build.sh for the
# full derivation): CXX/CC -> system g++14 (bypasses this venv's broken
# `-B .../compiler_compat` distutils CXX), LDSHARED rpath override (this
# venv's own torch/lib, not the stale NERSC-conda libstdc++).
#
# PATCH recorded (ARTIFACT_GUIDE rule 3 -- minimal, non-kernel-logic,
# recorded here and in STATUS.md): source/Sparse_FlashAttention_CSR/
# sp_flatt_csr_kernel.cu's dtype dispatch used the deprecated `Q.type()`
# ATen API, which this torch version's AT_DISPATCH_FLOATING_TYPES_AND_HALF
# macro can no longer compile against (no conversion to c10::ScalarType).
# Changed to `Q.scalar_type()` -- the modern, semantically-identical
# replacement (same dtype dispatch), a torch-API-version compatibility fix,
# not a change to any kernel algorithm/logic.
#
# Idempotent: exit 0 if the compiled extension is already present (set
# FORCE_REBUILD=1 to force a rebuild).
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
CSR_DIR="$HERE/source/Sparse_FlashAttention_CSR"

if [[ ! -d "$CSR_DIR" ]]; then
    echo "[gpa] ERROR: $CSR_DIR missing -- clone the repo into source/ first" >&2
    exit 1
fi

if compgen -G "$CSR_DIR"/spfa_csr*.so > /dev/null && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[gpa] spfa_csr already built, skipping (set FORCE_REBUILD=1 to rebuild)"
    exit 0
fi

export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"   # A100 (sm_80)
export MAX_JOBS="${MAX_JOBS:-1}"
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export LDFLAGS=""
TORCH_LIB="$("$PY" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LDSHARED="$CXX -pthread -shared -Wl,-rpath,$TORCH_LIB"

echo "[gpa] python: $($PY --version), torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "[gpa] host compiler: $($CXX --version | head -1)"
echo "[gpa] nvcc: $(nvcc --version | tail -1)"
echo "[gpa] repo commit: $(git -C "$HERE/source" rev-parse HEAD 2>/dev/null || echo unknown)"

cd "$CSR_DIR"
rm -rf build
"$PY" setup.py build_ext --inplace

SO="$(compgen -G 'spfa_csr*.so' || true)"
if [[ -z "$SO" ]]; then
    echo "[gpa] ERROR: build finished but no spfa_csr*.so found in $CSR_DIR" >&2
    exit 1
fi
echo "[gpa] built $SO"

echo "[gpa] verifying import..."
LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" -c "
import sys
sys.path.insert(0, '$CSR_DIR')
import spfa_csr
print('[gpa] import OK:', spfa_csr, dir(spfa_csr))
"
echo "[gpa] build.sh done."
