#!/usr/bin/env bash
# Marlin (github.com/IST-DASLab/marlin) ships a real ahead-of-time CUDA
# extension (marlin/marlin_cuda.cpp + marlin/marlin_cuda_kernel.cu), unlike
# quantized-gemm's tilus/qfactory (both runtime-JIT DSLs) -- this script
# actually compiles it, once, via torch's cpp_extension.CUDAExtension
# (source/setup.py, unmodified).
#
# `pip install --target=vendor --no-deps ./source` builds the wheel and
# installs it into a LOCAL vendor/ directory, not the shared venv's
# site-packages -- same isolation rationale as tilus's build.sh (this
# machine's venv is shared with other concurrent integration work).
#
# TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}" pins the nvcc codegen target to this machine's
# A100 (sm_80) -- Marlin's own setup.py does not set an arch list itself,
# so without this torch's cpp_extension would fall back to querying the
# arch of whatever GPU is visible at BUILD time via nvidia-smi, which is
# fine here but pinning it explicitly makes the build reproducible
# regardless of which GPU happens to be visible on a shared login node.
#
# REAL ENVIRONMENT BUG FOUND (not a marlin bug, but blocks building it on
# this machine without a workaround): the host compiler that setuptools'
# UnixCCompiler resolves for the bare names "cc"/"c++" is this login
# node's SYSTEM default, /usr/bin/c++, a symlink to g++-7 (SUSE Linux
# 7.5.0) -- too old for torch's headers (c10/util/C++17.h hard-errors
# below GCC 9). This is easy to miss interactively: this shell's own
# `.bashrc` defines `alias c++=g++`, and PATH resolves the bare `g++` to
# `/opt/cray/pe/gcc-native/14/bin/g++` (14.3.0) *ahead* of `/usr/bin`, so
# a human typing `c++ --version` at an interactive prompt sees 14.3.0 and
# is misled -- but pip's build subprocess is non-interactive, so bash
# alias expansion never applies there, and PATH-only lookup of the bare
# name `c++` (not `g++`) finds `/usr/bin/c++` first because
# gcc-native/14/bin ships a `g++` binary but no `c++`-named one. Fix:
# explicitly pin CC/CXX to the modern compiler's absolute path so
# distutils' `customize_compiler` (which reads $CC/$CXX before falling
# back to sysconfig) picks it up regardless of alias/PATH quirks.
#
# Idempotent: safe to re-run (pip install overwrites vendor/ cleanly).
set -euo pipefail
# Pin the CUDA toolchain to 12.9 (matching plexus_env's torch 2.8.0+cu128) --
# needed since 2026-09-04, when this login node's default module became
# cudatoolkit/13.2 (HPC SDK 26.5): that module's nvcc/CUDA_HOME/CPATH would
# otherwise leak 13.2 headers into this build (breaks CUTLASS-style code via
# cudaTypedefs.h) even when nvcc itself is pinned. See toolchain.sh's own
# header comment for the full incident. Marlin's own kernel does not use
# CUTLASS, so this build was not observed to break, but the pin is added
# here anyway for robustness/consistency across every build.sh in this repo.
# NOTE: this must run BEFORE `cd "$(dirname "$0")"` below -- BASH_SOURCE[0]
# keeps whatever form (relative/absolute) the script was invoked with, so
# resolving it AFTER an already-changed cwd (the original bug here) doubles
# the relative path (".../gemv/marlin/bench/artifacts/gemv/marlin/../../
# toolchain.sh") and fails with "No such file or directory" whenever this
# script is invoked via a relative path (e.g. `bench/artifacts/gemv/marlin/
# build.sh` from the repo root, exactly how the reproduction harness calls
# every build.sh) -- a real, invocation-order bug, not a machine difference.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"

echo "== marlin (gemv, w4a16 decode) build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $($PY --version)"
echo "CC=$CC ($($CC --dumpfullversion 2>/dev/null || $CC -dumpversion))"
echo "CXX=$CXX ($($CXX --dumpfullversion 2>/dev/null || $CXX -dumpversion))"
command -v nvcc >/dev/null || { echo "nvcc not on PATH -- module load cudatoolkit?"; exit 1; }
nvcc --version | tail -1

rm -rf vendor build_tmp source/build source/*.egg-info
mkdir -p vendor
TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH:-8.0}" \
"$PY" -m pip install --target=vendor --no-deps --no-build-isolation ./source

echo "-- verifying import (LD_PRELOAD needed for libstdc++ ABI, see STATUS.md) --"
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" -c "
import sys; sys.path.insert(0, 'vendor')
import marlin
import marlin_cuda
print('marlin', marlin.__file__, '-- import OK')
print('marlin_cuda', marlin_cuda.__file__, '-- import OK')
"
echo "== marlin (gemv, w4a16 decode) build: OK =="
