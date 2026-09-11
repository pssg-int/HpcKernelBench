#!/usr/bin/env bash
# FlashAttention-T (PPoPP'26, conf/ppopp/Xu0BXX00000C26) -- scoped build.
#
# The artifact ships NO setup.py / pip-installable extension for the
# Ampere ("ampere/") kernel; its own workflow compiles standalone CUDA/C++
# CLI benchmark binaries via CMake (see source/1-figure8-main-results/
# README.md section 2.1). We do not build that CLI (it is the paper's own
# benchmark driver, out of scope per ARTIFACT_GUIDE.md rule 1).
#
# Instead we JIT-compile wrapper.cu (this directory, NOT under source/) --
# a NEW pybind11 boundary, analogous to bench/artifacts/spmm/rassm/wrapper.cpp
# -- via torch.utils.cpp_extension.load(). wrapper.cu #includes the
# artifact's OWN unmodified header cxx-tests/custom_api/flash_api_custom.cuh
# and instantiates its `custom_mha_fwd_{causal,noncausal}` templates for
# head_dim in {64,128} x dtype in {fp16,bf16}. See wrapper.cu's header
# comment for why this scopes down cleanly to a header-only compile with NO
# need to touch any of the 114 flash_{fwd,bwd}_hdim*.cu per-config
# instantiation files (those exist only for the STANDARD, non-tensorized
# FlashAttention-2 comparison path built by the artifact's fp16-fwd-bench-orig
# binary, which this integration does not use).
#
# Idempotent: skips compilation if the cached .so already exists and imports.
set -euo pipefail
# Toolchain pin + source/ auto-fetch (KB_* knobs): bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
SRC_AMPERE="$(pwd)/source/1-figure8-main-results/flashattention-t/ampere"
BUILD_DIR="$(pwd)/_build"
mkdir -p "$BUILD_DIR"

echo "== FlashAttention-T build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $PY ($($PY -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda)'))"
echo "nvcc:   $("$CUDA_HOME/bin/nvcc" --version | tail -1)"

# -- known compiler gotcha on this machine (venv distutils bakes a broken
#    -B compiler_compat CXX / -rpath into LDSHARED) -- fix verbatim, see
#    bench/artifacts/sddmm/fused3s/build.sh for the first occurrence.
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"   # pinned by toolchain.sh; Perlmutter fallback
export NVHPC_CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
# 2026-09-04: the default cudatoolkit/13.2 module leaks its headers through
# CPATH; 13.2's cudaTypedefs.h lacks the unversioned PFN_* aliases CUTLASS
# uses, so a 12.9 nvcc compile fails unless CPATH is re-pointed. It must be
# RE-POINTED, not cleared: torch's ATen/cuda headers include cusparse.h,
# which lives in the HPC SDK's math_libs tree, not under cuda/12.9. This is
# exactly what `module load cudatoolkit/12.9` exports. See STATUS.md
# "Environment drift" and ARTIFACT_GUIDE "Toolchain pin".
export CPATH="${CPATH:-$CUDA_HOME/include}"   # toolchain.sh already re-pointed it (math_libs include first)
export TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}"
TORCH_LIB=$($PY -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LDSHARED="$CXX -pthread -shared -Wl,-rpath,$TORCH_LIB"

echo "-- JIT-compiling wrapper.cu (head_dim in {64,128}, fp16+bf16, fwd-only) --"
# KB_SM is passed as an argv (this invocation line is OUTSIDE the quoted
# heredoc, so bash DOES expand ${KB_SM:-80} here); inside the '<<'PYEOF''
# body bash does NOT expand, so we must not write a literal ${...} into the
# Python extra_cuda_cflags (ninja treats a bare '$' as a bad build.ninja
# escape -- "bad $-escape", the H100 fresh-build failure this fixes).
"$PY" - "$SRC_AMPERE" "$BUILD_DIR" "${KB_SM:-80}" <<'PYEOF'
import sys, os
src_ampere, build_dir, kb_sm = sys.argv[1], sys.argv[2], sys.argv[3]
from torch.utils.cpp_extension import load

ext = load(
    name="flashattention_t_custom_fwd",
    sources=[os.path.join(os.getcwd(), "wrapper.cu")],
    extra_include_paths=[
        os.path.join(src_ampere, "csrc", "cutlass", "include"),
        os.path.join(src_ampere, "csrc", "flash_attn", "src"),
        os.path.join(src_ampere, "cxx-tests", "custom_api"),
    ],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3", "-std=c++17",
        "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
        f"-gencode=arch=compute_{kb_sm},code=sm_{kb_sm}",
    ],
    build_directory=build_dir,
    verbose=True,
)
print("IMPORT OK:", ext.__file__)
assert hasattr(ext, "fat_fwd"), "fat_fwd symbol missing from compiled extension"
print("BUILD-CHECK: fat_fwd symbol present")
PYEOF

echo "== FlashAttention-T build: OK =="
