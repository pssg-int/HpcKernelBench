#!/usr/bin/env bash
# Build PAT's torch CUDA extension (source/setup.py, unmodified) in place,
# with this machine's python env (torch 2.8/cu128) and nvcc 12.9 / sm_80.
#
# Two machine-specific workarounds, neither of which touches artifact code
# (identical situation/fix to bench/artifacts/sddmm/fused3s/build.sh -- see
# that file's comments for the fuller derivation):
#
# 1. The venv's python is built against a NERSC conda env whose sysconfig
#    bakes `CXX = "g++ -pthread -B .../compiler_compat"` into distutils.
#    That `-B` flag causes g++ to pick up an ancient (GCC 7-era) cc1plus from
#    the conda toolchain instead of the real system g++ 14, which then fails
#    torch's `#if __GNUC__ < 9` check. Fix: point CXX/CC at the system
#    compiler directly (bypassing the -B injection).
# 2. The same sysconfig also bakes `-rpath .../nersc-python/lib` into the
#    link step (via LDSHARED), which ships an OLD libstdc++.so.6 missing a
#    GCC-8+ symbol the extension needs. Fix: override LDSHARED with a clean
#    link line (our own compiler, no -B, no nersc-python rpath) that instead
#    rpaths straight to this venv's own torch/lib.
#
# Plus one PAT-specific step: `einops` is a runtime import of
# prefix_attn/utils.py but is NOT installed in the shared venv. Per the
# task's instructions, install it into an ISOLATED --target directory next
# to this artifact rather than touching the shared venv's dependency graph.
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
SRC_DIR="$HERE/source"
CUTLASS_ROOT="$HERE/source/third_party/cutlass"
PYLIBS="$HERE/pylibs"

# ---- 1. isolated einops install (pure-python, low risk, but kept out of the
#         shared venv per instructions) -----------------------------------
if ! PYTHONPATH="$PYLIBS" "$PY" -c "import einops" >/dev/null 2>&1; then
    echo "[pat] installing einops into isolated target $PYLIBS"
    "$PY" -m pip install --target="$PYLIBS" einops
else
    echo "[pat] einops already available in $PYLIBS, skipping install"
fi

# ---- 2. idempotency check --------------------------------------------------
if compgen -G "$SRC_DIR"/prefix_attn/_prefix_attn.cpython-*.so > /dev/null && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[pat] extension already built, skipping (set FORCE_REBUILD=1 to rebuild)"
    exit 0
fi

# ---- 3. vendored CUTLASS (header-only) at the exact commit recorded in
#         STATUS.md; fetched here so a fresh clone of source/ (which is
#         git-ignored repo-wide) rebuilds without manual steps -------------
CUTLASS_SHA="dcf215af68a2d08d305076c152a06f201728cd53"
if [[ ! -d "$CUTLASS_ROOT/include" ]]; then
    echo "[pat] fetching NVIDIA/cutlass @ $CUTLASS_SHA into $CUTLASS_ROOT"
    mkdir -p "$CUTLASS_ROOT"
    git -C "$CUTLASS_ROOT" init -q
    git -C "$CUTLASS_ROOT" remote add origin https://github.com/NVIDIA/cutlass.git
    git -C "$CUTLASS_ROOT" fetch -q --depth 1 origin "$CUTLASS_SHA"
    git -C "$CUTLASS_ROOT" checkout -q FETCH_HEAD
fi
if [[ "$(git -C "$CUTLASS_ROOT" rev-parse HEAD 2>/dev/null)" != "$CUTLASS_SHA" ]]; then
    echo "[pat] ERROR: $CUTLASS_ROOT is not at the recorded commit $CUTLASS_SHA" >&2
    exit 1
fi

export CUTLASS_ROOT
export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"   # A100 (sm_80); setup.py also hardcodes -arch=compute_80
# Serialize compilation (torch's ninja backend defaults to os.cpu_count()
# parallel cicc/nvcc jobs). Each of PAT's 4 CUTLASS translation units is
# memory-heavy; on this shared login node (multiple other artifact builds
# running concurrently) a first attempt with default parallelism got every
# non-first .cu file killed by SIGKILL ("cicc died due to signal 9") --
# consistent with a memory-pressure kill, not a code error. MAX_JOBS=1 is a
# build-system concurrency knob, not a kernel-code change.
export MAX_JOBS="${MAX_JOBS:-1}"
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export LDFLAGS=""
TORCH_LIB="$("$PY" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LDSHARED="$CXX -pthread -shared -Wl,-rpath,$TORCH_LIB"

echo "[pat] python: $($PY --version), torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "[pat] host compiler: $($CXX --version | head -1)"
echo "[pat] nvcc: $(nvcc --version | tail -1)"
echo "[pat] CUTLASS_ROOT: $CUTLASS_ROOT ($(git -C "$CUTLASS_ROOT" rev-parse HEAD 2>/dev/null || echo unknown))"

cd "$SRC_DIR"
rm -rf build
PYTHONPATH="$PYLIBS${PYTHONPATH:+:$PYTHONPATH}" "$PY" setup.py build_ext --inplace

SO="$(compgen -G 'prefix_attn/_prefix_attn.cpython-*.so' || true)"
if [[ -z "$SO" ]]; then
    echo "[pat] ERROR: build finished but no _prefix_attn*.so found under prefix_attn/" >&2
    exit 1
fi
echo "[pat] built $SO"

echo "[pat] verifying import..."
PYTHONPATH="$PYLIBS:$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c "
import sys
sys.path.insert(0, '$PYLIBS')
sys.path.insert(0, '$SRC_DIR')
import torch
from prefix_attn import prefix_attn_with_kvcache, PrefixTreeCPP
from prefix_attn.data_class import create_seq_group
print('[pat] import OK:', prefix_attn_with_kvcache, PrefixTreeCPP)
"

echo "[pat] build.sh done."
