#!/usr/bin/env bash
# Tilus is a Python DSL (like Triton): there is no fixed .so to compile ahead
# of time -- kernels are JIT-compiled by Tilus's own nvcc-based backend the
# FIRST time a kernel is actually invoked (see adapter.py's prepare(), which
# is where that JIT build happens -- that IS this artifact's preprocessing,
# per the task brief: "run the JIT once in prepare()"). This script's only
# job is to make `import tilus` (at this exact cloned commit) work.
#
# Steps:
#   1. `pip install --target=vendor --no-deps ./source` builds a wheel from
#      the cloned repo's own pyproject.toml (setuptools_scm-versioned) and
#      installs it into a LOCAL vendor/ directory -- NOT the shared venv's
#      site-packages (this machine's venv is shared with other concurrent
#      work; keeping this artifact's dependency tree self-contained under
#      bench/artifacts/quantized-gemm/tilus/vendor/ avoids touching it).
#      Installing from source/ (not PyPI's `tilus` wheel) matters: the API
#      Tilus's own examples/quantization/matmul_a16wx.py uses is pinned to
#      THIS commit, and PyPI's latest release can drift from it.
#   2. `apache-tvm-ffi==0.1.10` (its only real runtime dependency beyond
#      torch/numpy, already present on the reference machine's shared venv):
#      PINNED, not latest.
#      apache-tvm-ffi 0.1.13.post2 (latest on PyPI as of this integration)
#      ships a C++ type_traits.h whose `TypeTraits<DLTensor*>` no longer
#      inherits `TryCastFromAnyView` the way this Tilus commit's generated
#      CUDA glue code (source.cu, produced by tilus/hidet/backend/build.py)
#      expects -- a REAL artifact/dependency version-skew bug, not a build
#      typo: it fails deep inside tvm-ffi's own header with "class
#      TypeTraits<DLTensor*, void> has no member TryCastFromAnyView" when
#      compiling the auto-generated `cast` kernel. 0.1.10 (and 0.1.9,
#      0.1.8.post2 -- also tried) still use the older TypeTraitsBase
#      structure this commit's codegen was written against and compile
#      clean. No pin exists in tilus's own pyproject.toml (just bare
#      "apache-tvm-ffi") -- this is this integration's own patch,
#      build-system-only (ARTIFACT_GUIDE rule 3), no kernel code touched.
#   3. `tabulate`/`tqdm`: both are plain, declared runtime deps of tilus's
#      own pyproject.toml (`dependencies = [..., "tabulate", "tqdm"]`), just
#      not present in every shared venv (reference machine's venv happened
#      to already have them; this machine's did not, surfacing as
#      `ModuleNotFoundError: No module named 'tabulate'` on `import tilus`
#      -- tilus's own `ir/layout/register_layout.py` imports it
#      unconditionally). Installed explicitly into `vendor/` for the same
#      reason `apache-tvm-ffi` is: `--no-deps` on `./source` alone does not
#      pull them in. Both are pure-Python with no further transitive deps
#      on this platform, so `--no-deps` remains safe here too.
#   4. `cuda-python`/`cuda-bindings==12.8.0` (matching this machine's nvcc
#      12.8 toolkit): `tilus/lang/modules/cuda.py` unconditionally does
#      `import cuda.bindings.runtime as cudart` -- NVIDIA's official CUDA
#      Python bindings package, which tilus's own pyproject.toml does NOT
#      declare as a dependency at all (its own mypy config even has a
#      `[[tool.mypy.overrides]] module = ["cuda.*"]` with
#      `ignore_missing_imports = true`, i.e. tilus's own tooling already
#      expects this import can be absent -- a real, undeclared-dependency
#      gap in the artifact's own packaging, not something this integration
#      introduced). The reference machine's venv already had `cuda-python`
#      installed (for unrelated reasons); this machine's shared env does
#      not, surfacing as `ModuleNotFoundError: No module named 'cuda'` at
#      `import tilus` time. `cuda-python` itself is a thin metapackage
#      (`Requires-Dist: cuda-bindings~=12.8.0`) with no `.so` of its own --
#      the actual `cuda.bindings.runtime` extension module ships in the
#      separate `cuda-bindings` wheel, so both are installed explicitly
#      (`--no-deps` on the metapackage alone would leave `cuda.bindings`
#      missing).
#
# Idempotent: safe to re-run (pip install overwrites vendor/ cleanly).
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
TVM_FFI_VERSION="0.1.10"

echo "== tilus (quantized-gemm) build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $($PY --version)"

rm -rf vendor
mkdir -p vendor
CUDA_PYTHON_VERSION="${CUDA_PYTHON_VERSION:-12.8.0}"
"$PY" -m pip install --target=vendor --no-deps \
    ./source "apache-tvm-ffi==${TVM_FFI_VERSION}" tabulate tqdm \
    "cuda-python==${CUDA_PYTHON_VERSION}" "cuda-bindings==${CUDA_PYTHON_VERSION}"

echo "-- verifying import (LD_PRELOAD needed for libstdc++ ABI, see STATUS.md) --"
LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" -c "
import sys; sys.path.insert(0, 'vendor')
import tilus
print('tilus', tilus.__version__, '-- import OK')
"
mkdir -p cache
echo "== tilus (quantized-gemm) build: OK =="
