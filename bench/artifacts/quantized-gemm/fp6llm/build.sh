#!/usr/bin/env bash
# FP6-LLM / Quant-LLM (github.com/usyd-fsalab/fp6_llm) -- quantized-gemm.
#
# Ships a real ahead-of-time CUDA extension (fp6_llm/csrc/{pybind.cpp,
# fp6_linear.cu}), built via torch's cpp_extension.CUDAExtension
# (source/setup.py, unmodified). setup.py's own extra_compile_args ALREADY
# hardcodes `-gencode=arch=compute_${KB_SM:-80},code=sm_${KB_SM:-80}` -- this machine's exact
# A100 target -- so no arch-flag patch is needed (unlike some other
# artifacts in this project).
#
# `pip install --target=vendor --no-deps ./source` installs into a LOCAL
# vendor/ directory, not the shared venv's site-packages -- same isolation
# rationale as tilus's/marlin's build.sh.
#
# Same CC/CXX pin as marlin's build.sh: this login node's bare `c++`
# resolves to the system /usr/bin/c++ (g++-7, too old for torch's C++17
# headers) unless CC/CXX are set to the absolute modern-compiler path.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"

echo "== fp6_llm (quantized-gemm, FP6/FP5 weight-only GEMM) build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $($PY --version)"
echo "CC=$CC ($($CC --dumpfullversion 2>/dev/null || $CC -dumpversion))"
echo "CXX=$CXX ($($CXX --dumpfullversion 2>/dev/null || $CXX -dumpversion))"
command -v nvcc >/dev/null || { echo "nvcc not on PATH"; exit 1; }
nvcc --version | tail -1

rm -rf vendor build_tmp source/build source/*.egg-info
mkdir -p vendor
"$PY" -m pip install --target=vendor --no-deps --no-build-isolation ./source

echo "-- verifying import (LD_PRELOAD needed for libstdc++ ABI) --"
LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" -c "
import sys; sys.path.insert(0, 'vendor')
import torch  # must precede fp6_llm_cuda import: fp6_llm/__init__.py does not
              # import torch itself before pulling in fp6_llm_cuda, unlike
              # marlin's/tilus's own __init__.py -- without torch's libc10.so
              # already resident, the bare extension import fails with
              # 'ImportError: libc10.so: cannot open shared object file'
              # (a real, reproducible artifact packaging gap -- see STATUS.md).
import fp6_llm
import fp6_llm_cuda
print('fp6_llm', fp6_llm.__file__, '-- import OK')
print('fp6_llm_cuda', fp6_llm_cuda.__file__, '-- import OK')
print('has eXmY interfaces:', hasattr(fp6_llm_cuda, 'linear_forward_eXmY_cuda'))
"
echo "== fp6_llm (quantized-gemm) build: OK =="
