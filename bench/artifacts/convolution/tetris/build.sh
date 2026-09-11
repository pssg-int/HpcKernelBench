#!/usr/bin/env bash
# Build Tetris's sparse-conv kernel (SparseConv2d, source/Tetris/
# spconv2d_kernel.cu) plus a thin C-linkage wrapper (csrc/tetris_wrap.cu) into
# a shared library the adapter loads via ctypes -- the same idiom
# kernelbench/impls/gpu_cuda.py's _load_lib() uses for this project's own
# csrc/kernels.cu.
#
# Only the pieces actually needed for ONE forward SparseConv2d call are
# built: SparseFilter's CPU-side SPF packer (header-only, instantiated inside
# tetris_wrap.cu) and the SparseConv2dKernel GPU kernel + its host launcher
# SparseConv2d<...> (source/Tetris/spconv2d_kernel.cu). Neither
# cusparse_test.cc/cudnn_test.cc/sputnik_test.cc/spconv2d.cc's ~100-way
# autotuning-dispatch driver nor the sputnik/Unified-Convolution-Framework
# git submodules are needed for this -- confirmed by reading
# evalution/script/build_all.sh and source/Tetris/spconv2d_kernel.cu's own
# explicit template instantiation list (TETRIS_INSTANTIATE_TILED_FLOAT(...,
# 3, 1, 128, 2, 4) is one of the pre-compiled tile configs, and is exactly
# the one source/Tetris/spf.cc's own "ablation" driver calls). Submodules
# were left uncloned (rule: keep it minimal, only fetch what's needed).
#
# CUDA-arch fix (rule 3, build-system only, no kernel code touched): the
# artifact's own build scripts hardcode --generate-code=arch=compute_70,
# code=sm_70 (V100, the paper's eval GPU per Readme.md's "ENV: V100-PCIE-
# 32GB"). This machine's GPU is an A100 (sm_80), so this script instead
# passes --generate-code=arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80}.
#
# Idempotent: safe to re-run; exit 0 = built and loadable.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
ARCH_FLAGS="--generate-code=arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80}"
# Host compiler pin (2026-09-09, zaratan): matches the fix already used by
# most other artifacts in this repo (grep KB_GXX12 bench/artifacts/*/*/
# build.sh) -- without an explicit -ccbin, nvcc does its own host-compiler
# search rather than simply taking the first g++ on $PATH, and on this
# machine that search finds this artifact-build-env's own default g++
# (13.4.0), which nvcc 12.x can't parse (<type_traits>/<bits/hashtable.h>
# "identifier is undefined" errors, same class of failure documented in
# bench/artifacts/convolution/hidet/STATUS.md). KB_GXX12 (exported by
# bench/artifacts/toolchain.sh, sourced above) is a g++ 12 known to work
# with this nvcc.
HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"

echo "== Tetris sparse-conv build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc  : $("$NVCC" --version | tail -1)"
echo "host compiler: $("$HOST_COMPILER" --version | head -1)"
echo "python: $PY"
"$PY" - <<'EOF'
import torch
print("torch :", torch.__version__, "cuda:", torch.version.cuda,
      "available:", torch.cuda.is_available())
EOF

# cudnn.h is a compile-time-only dependency (source/Tetris/cuda_utils.h
# #includes <cudnn.h> for a CuDNNConv2d() declaration that this integration
# never calls -- see csrc/tetris_wrap.cu's header comment); no cudnn library
# is linked. Resolve the header via the site cudnn module rather than vendoring
# it (NERSC-provided, not a system package install).
CUDNN_INC="${CUDNN_INC:-}"
if [ -z "$CUDNN_INC" ]; then
  if [ -n "${CUDNN_DIR:-}" ] && [ -f "${CUDNN_DIR}/include/cudnn.h" ]; then
    CUDNN_INC="${CUDNN_DIR}/include"
  else
    # Portable fallback (2026-09-09, zaratan): no NERSC `module` system here
    # and no site-wide cudnn install -- but torch's own pip dependency
    # closure already ships cudnn 9.x headers+libs as the `nvidia-cudnn-cu12`
    # package (site-packages/nvidia/cudnn/include/cudnn.h), since torch
    # itself links against it. This is header-only use (see comment above
    # this block) so any cudnn.h of a compatible major version works; ask
    # the interpreter that will actually run the harness where its own
    # torch-provided cudnn headers live, rather than hardcoding a path.
    CUDNN_PY_INC="$("$PY" -c '
import importlib.util, os
spec = importlib.util.find_spec("nvidia.cudnn")
if spec and spec.submodule_search_locations:
    d = os.path.join(list(spec.submodule_search_locations)[0], "include")
    print(d if os.path.isfile(os.path.join(d, "cudnn.h")) else "")
' 2>/dev/null || true)"
    if [ -n "$CUDNN_PY_INC" ] && [ -f "$CUDNN_PY_INC/cudnn.h" ]; then
      CUDNN_INC="$CUDNN_PY_INC"
    fi
  fi
  if [ -z "$CUDNN_INC" ]; then
    # try loading the module non-fatally (module fn may be unavailable in
    # non-interactive shells; fall back to a known NERSC path if so)
    { module load cudnn/9.5.0 2>/dev/null || true; }
    if [ -n "${CUDNN_DIR:-}" ] && [ -f "${CUDNN_DIR}/include/cudnn.h" ]; then
      CUDNN_INC="${CUDNN_DIR}/include"
    elif [ -f /global/common/software/nersc9/cudnn/9.5.0-cuda12/include/cudnn.h ]; then
      CUDNN_INC=/global/common/software/nersc9/cudnn/9.5.0-cuda12/include
    fi
  fi
fi
if [ -z "$CUDNN_INC" ]; then
  echo "cudnn.h not found (tried \$CUDNN_DIR, the torch-bundled nvidia-cudnn-cu12 package, and the NERSC cudnn/9.5.0 module); aborting" >&2
  exit 1
fi
echo "cudnn header dir: $CUDNN_INC (header-only use, not linked)"

mkdir -p csrc/build

echo "-- compiling source/Tetris/spconv2d_kernel.cu (unmodified artifact file)"
"$NVCC" -ccbin "$HOST_COMPILER" -c -O3 -std=c++14 $ARCH_FLAGS -Xcompiler -fPIC \
  -I "$CUDNN_INC" \
  source/Tetris/spconv2d_kernel.cu -o csrc/build/spconv2d_kernel.o

echo "-- compiling csrc/tetris_wrap.cu (new wrapper file, not part of the artifact)"
"$NVCC" -ccbin "$HOST_COMPILER" -c -O3 -std=c++14 $ARCH_FLAGS -Xcompiler -fPIC \
  -DBANK_OPTIMIZE -DREORDER_OUT_CHANNEL \
  -I "$CUDNN_INC" \
  csrc/tetris_wrap.cu -o csrc/build/tetris_wrap.o

echo "-- linking csrc/build/libtetris.so"
"$NVCC" -ccbin "$HOST_COMPILER" -shared $ARCH_FLAGS \
  csrc/build/spconv2d_kernel.o csrc/build/tetris_wrap.o \
  -o csrc/build/libtetris.so

echo "-- verifying the .so loads and exposes the expected symbols"
"$PY" - <<EOF
import ctypes
lib = ctypes.CDLL("$(pwd)/csrc/build/libtetris.so")
for fn in ("tetris_pack_filter", "tetris_free_filter", "tetris_conv_forward"):
    getattr(lib, fn)
print("libtetris.so loaded OK; symbols present:",
      "tetris_pack_filter, tetris_free_filter, tetris_conv_forward")
EOF

echo "== Tetris build: OK =="
