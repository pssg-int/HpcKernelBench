#!/usr/bin/env bash
# Build ONLY PathWeaver's search kernel extension (csrc/pathweaver.cu ->
# pathweaver.so), vendored under this directory -- NOT under source/.
#
# source/pathweaver/setup.py declares TWO CUDAExtension targets
# ('cpu_generate_sign_bit' and 'pathweaver'); this repo's rule 1
# (wrap the kernel, not the whole artifact build) plus the task's explicit
# "build ONLY the pathweaver extension" means we do not invoke that setup.py
# at all (it would build both, and installs into the source tree's own
# build/ directory besides). Instead this script drives
# torch.utils.cpp_extension.load(), pointing directly at
# source/pathweaver/csrc/pathweaver.cu (zero lines of source/ modified),
# with the SAME extra_cuda_cflags the artifact's own setup.py already uses
# ('-O3 -Xptxas=-v -arch=sm_80' -- sm_80 was already the artifact's own
# choice, matching this machine's A100; no arch-flag override was needed
# here, unlike CLOVER's sm_89->sm_80 retarget).
#
# cpu_generate_sign_bit.cpp (the OTHER extension) is intentionally never
# built: PathWeaver's own sign-bit pruning feature (SIGN_BIT_PRUNE) is a
# search-quality optimization the adapter does not exercise in its first,
# simplest-valid-configuration pass (see adapter.py's module docstring) --
# building it would add an OpenMP host-only extension with zero bearing on
# whether the search() kernel itself runs and gates.
#
# TOOLCHAIN PIN (host environment note, not a PathWeaver bug): this login
# node's default module set now points cudatoolkit at 26.5/CUDA 13.2 (torch
# is cu128), so bench/artifacts/toolchain.sh is sourced below to pin
# CUDA_HOME/NVHPC_CUDA_HOME/PATH to the 12.9 HPC-SDK prefix and repoint
# CPATH at 12.9's own cuda+math_libs include dirs (NOT unset -- torch's own
# ATen/cuda headers need cusparse.h, which only exists under math_libs).
# See that file's header comment for the full incident history. This
# script's own build.sh previously duplicated this logic inline (verified
# broken once: a naive `${CUDA_HOME:-default}` pattern silently kept the
# already-nonempty inherited 13.2 value); now shared, single source of
# truth for every artifact build in this repo.
#
# Idempotent: safe to re-run (removes and rebuilds the torch extension
# cache directory each time).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source/pathweaver"

# shellcheck source=../../toolchain.sh
source "$HERE/../../toolchain.sh"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

echo "=== PathWeaver search-kernel build (pathweaver.so, sm_80) ==="
echo "commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || git -C "$HERE/source" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $($PY --version)"
echo "CUDA_HOME=$CUDA_HOME"
command -v nvcc >/dev/null || { echo "nvcc not on PATH" >&2; exit 1; }
nvcc --version | tail -1
nvcc --version | grep -qE 'release 12\.' || { echo "nvcc is not CUDA 12.x (toolchain pin failed -- see header comment)" >&2; exit 1; }
echo "CC=$CC ($($CC -dumpfullversion 2>/dev/null || $CC -dumpversion))"
echo "CXX=$CXX ($($CXX -dumpfullversion 2>/dev/null || $CXX -dumpversion))"
"$PY" -c "import torch; print('torch', torch.__version__, 'torch.version.cuda', torch.version.cuda)"

BUILD_DIR="$HERE/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# LD_PRELOAD is needed here too, not just in the later verification step:
# torch.utils.cpp_extension.load() imports the freshly-built module INSIDE
# this same process right after linking (is_python_module=True, the
# default) to hand back a module object -- without the preload that import
# fails with the project's usual CXXABI_1.3.15 mismatch (this venv's base
# interpreter resolves an older conda-bundled libstdc++.so.6 ahead of the
# GCC-14-linked extension's needs). Confirmed by a build attempt without
# this preload: compile+link succeeded, then the internal import inside
# load() raised exactly that ImportError -- not a build failure, an import
# environment issue, same root cause as every other torch extension in
# this repo.
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" - <<PYEOF
import torch
from torch.utils.cpp_extension import load

ext = load(
    name="pathweaver",
    sources=["$SRC/csrc/pathweaver.cu"],
    extra_cflags=["-O3"],
    # pathweaver.cu includes <curand_kernel.h>. On this HPC SDK layout
    # cuRAND's headers live under math_libs/<ver>/include, NOT cuda/<ver>/
    # include (which only has the compiler-core CUDA headers) -- confirmed
    # by a first build attempt that failed with "curand_kernel.h: No such
    # file or directory" against cuda/12.9/include alone. toolchain.sh's
    # CPATH (sourced above) already includes math_libs/12.9/include for
    # exactly this reason (it also carries cusparse.h, needed by torch's own
    # headers), so no per-extension include-path override is needed here --
    # a real build-system fix (ARTIFACT_GUIDE rule 3), located centrally
    # rather than duplicated per artifact.
    extra_cuda_cflags=["-O3", "-Xptxas=-v", "-arch=sm_80"],
    build_directory="$BUILD_DIR",
    verbose=True,
)
print("loaded OK from", ext.__file__)
PYEOF

# torch's JIT loader names the compiled artifact <name>.so inside
# build_directory (ninja target), independent of the platform-tagged
# wheel-style name setup.py's build_ext would have produced.
SO="$BUILD_DIR/pathweaver.so"
if [[ ! -f "$SO" ]]; then
    echo "BUILD FAILED: $SO not produced" >&2
    exit 1
fi
echo "Built: $SO"

echo "-- verifying import (LD_PRELOAD needed for libstdc++ ABI, see STATUS.md) --"
# `import torch` MUST happen before `import pathweaver`: the .so links
# against -lc10/-lc10_cuda/-ltorch* by SONAME only (no baked-in rpath to
# torch/lib, unlike a setup.py/CUDAExtension-built wheel), so those symbols
# resolve only once torch's own libraries are already loaded in-process.
# Confirmed by a bare `import pathweaver` first: "ImportError: libc10.so:
# cannot open shared object file" -- not a build failure, an import-order
# requirement every adapter.py must also respect (satisfied automatically
# there since the harness always imports torch before any CUDA adapter).
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" -c "
import sys; sys.path.insert(0, '$BUILD_DIR')
import torch
import pathweaver
print('pathweaver', pathweaver.__file__, '-- import OK')
print([x for x in dir(pathweaver) if not x.startswith('_')])
"
echo "=== PathWeaver search-kernel build: OK ==="
