#!/usr/bin/env bash
# Build STOF's `binding_attn` CUDA extension (the fused, full/part/inner-
# bitmap block-sparse FlashAttention-family kernel -- the ONE STOF calls
# "Our Kernel" in src/benchmk_attn_unified.py's actual timed comparison; see
# adapter.py's module docstring for why the OTHER extension the repo ships,
# `block_attn_mask` (src/ops/src/block_attn_mask_cuda.cu, 6300+ lines, not
# called anywhere in the benchmark script this project's survey read), is
# deliberately NOT built here).
#
# The repo's own src/setup.py globs *.cpp + *_cuda.cu directly under
# ops/src/ and builds ONE CUDAExtension per unique basename (i.e. it would
# build BOTH binding_attn and block_attn_mask in one `build_ext --inplace`
# invocation, and setuptools aborts the whole command on the first
# extension that fails to compile). Rather than edit src/setup.py (a
# borderline build-system change this task prefers to avoid when a
# lower-risk alternative exists), this script invokes torch's own
# CUDAExtension/BuildExtension API directly, in a private throwaway
# setup.py, with EXACTLY the same extra_compile_args/architecture flags
# src/setup.py uses, restricted to binding_attn's 2 source files. No
# artifact source file is edited either way.
#
# Same two machine-specific compiler workarounds as
# bench/artifacts/sddmm/fused3s/build.sh and
# bench/artifacts/attention-kernel/pat/build.sh (read those for the full
# derivation): CXX/CC pointed at the system g++14 (bypasses this venv's
# broken `-B .../compiler_compat` distutils CXX, which picks up an ancient
# GCC-7-era cc1plus); LDSHARED overridden to rpath this venv's own
# torch/lib instead of the stale NERSC-conda libstdc++.
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
SRC_DIR="$HERE/source/src"
OPS_SRC="$SRC_DIR/ops/src"
PYLIBS="$HERE/pylibs"

# source/src/util/masks.py (unmodified artifact code -- imported by
# adapter.py for get_OuterTile_storage()) does `import matplotlib.pyplot as
# plt` at MODULE level, even though this adapter never calls any plotting
# function -- so the module fails to import at all without matplotlib
# present. Not in the shared env on this machine (was on the plexus_env used
# for the recorded run). Vendored into a per-artifact pylibs/ dir per this
# project's env rules (no shared-env installs), same pattern as
# bench/artifacts/attention-kernel/pat/build.sh. adapter.py adds PYLIBS to
# sys.path before importing util.masks.
if ! PYTHONPATH="$PYLIBS" "$PY" -c "import matplotlib" >/dev/null 2>&1; then
    echo "[sparse-transformer] installing matplotlib into $PYLIBS (needed by source/src/util/masks.py's module-level import, unused by get_OuterTile_storage itself)"
    "$PY" -m pip install --target="$PYLIBS" matplotlib
fi

if [[ -f "$OPS_SRC/binding_attn.so" ]] && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[sparse-transformer] binding_attn already built, skipping (set FORCE_REBUILD=1 to rebuild)"
    exit 0
fi

if [[ ! -d "$OPS_SRC/include/cutlass/include" ]]; then
    echo "[sparse-transformer] ERROR: $OPS_SRC/include/cutlass/include missing -- vendored cutlass absent" >&2
    exit 1
fi

export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"   # A100 (sm_80); matches this repo's own setup.py default cuda_arch=80
export MAX_JOBS="${MAX_JOBS:-1}"    # serialize nvcc/cicc jobs -- same OOM-avoidance reasoning as pat/build.sh
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export LDFLAGS=""
TORCH_LIB="$("$PY" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LDSHARED="$CXX -pthread -shared -Wl,-rpath,$TORCH_LIB"

echo "[sparse-transformer] python: $($PY --version), torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "[sparse-transformer] host compiler: $($CXX --version | head -1)"
echo "[sparse-transformer] nvcc: $(nvcc --version | tail -1)"
echo "[sparse-transformer] repo commit: $(git -C "$HERE/source" rev-parse HEAD 2>/dev/null || echo unknown)"

BUILD_TMP="$HERE/_build_binding_attn"
rm -rf "$BUILD_TMP"
mkdir -p "$BUILD_TMP"
cat > "$BUILD_TMP/setup_binding_attn_only.py" << 'PYEOF'
# Private, throwaway build script -- NOT part of the artifact's own source
# tree (lives under bench/artifacts/sparse-attention-kernel/sparse-
# transformer/_build_binding_attn/, outside source/). Mirrors
# source/src/setup.py's own extra_compile_args/arch flags exactly, scoped
# to the binding_attn extension only. See build.sh's header comment.
import os
import sys
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ops_src = sys.argv[1]
cuda_arch = sys.argv[2]
sys.argv = sys.argv[:1] + ["build_ext", "--inplace"]

extra_compile_args = {
    "nvcc": ["-O3", "-w",
             "-I/usr/local/cuda/include",
             f"-I{ops_src}/include/",
             f"-I{ops_src}/include/cutlass/include",
             f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}"],
    "cxx": ["-fPIC",
            f"-I{ops_src}/include/",
            f"-I{ops_src}/include/cutlass/include"],
}

# BuildExtension.with_options(use_ninja=True) generates its own link command
# internally (build.ninja) rather than honoring the process's $LDSHARED, so
# build.sh's LDSHARED rpath override (needed on this machine's conda python,
# see build.sh's header) is silently ignored for THIS extension -- unlike the
# fused3s/gpa extensions, which use the artifact's own setup.py without
# use_ninja and so do pick up LDSHARED. Symptom: the built .so links against
# libc10.so/libtorch.so etc. (found via torch's own -L at link time) but has
# no rpath to torch/lib, so `import binding_attn` fails at runtime with
# "libc10.so: cannot open shared object file" even though the build itself
# succeeds. Fix: pass the rpath explicitly via extra_link_args, which IS
# honored by the ninja path (computed from `torch.__file__`, not hardcoded).
torch_lib = os.path.join(os.path.dirname(os.path.abspath(torch.__file__)), "lib")
extra_link_args = [f"-Wl,-rpath,{torch_lib}"]

setup(
    name="binding_attn_standalone_build",
    ext_modules=[CUDAExtension(
        name="binding_attn",
        sources=[os.path.join(ops_src, "binding_attn.cpp"),
                 os.path.join(ops_src, "binding_attn_cuda.cu")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True, no_python_abi_suffix=True)},
)
PYEOF

cd "$BUILD_TMP"
"$PY" setup_binding_attn_only.py "$OPS_SRC" 80

SO="$(compgen -G "$BUILD_TMP"/binding_attn*.so || true)"
if [[ -z "$SO" ]]; then
    echo "[sparse-transformer] ERROR: build finished but no binding_attn*.so found in $BUILD_TMP" >&2
    exit 1
fi
cp -v "$SO" "$OPS_SRC/"
echo "[sparse-transformer] built and copied: $(basename "$SO") -> $OPS_SRC/"

echo "[sparse-transformer] verifying import..."
LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" PYTHONPATH="$OPS_SRC${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c "
import sys
sys.path.insert(0, '$OPS_SRC')
import binding_attn
print('[sparse-transformer] import OK:', binding_attn)
"
echo "[sparse-transformer] build.sh done."
