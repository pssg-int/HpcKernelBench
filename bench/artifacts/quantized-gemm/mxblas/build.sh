#!/usr/bin/env bash
# Build/verify MXBLAS (source/) on Hopper (sm_90a). NO kernelbench adapter.py
# exists for this artifact -- see STATUS.md's "Why no adapter.py" section:
# the quantized-gemm domain module (bench/kernelbench/domains/ml.py) only
# models weight-only INT-coded quantization (`QuantGemmWorkload.quantize()`)
# and explicitly documents, in its own docstring, that the FP8/MX-
# block-scaled `qgemm-w8a8-mx` regime MXBLAS implements "is also not
# modeled" -- wiring it up would mean adding a workload/reference to
# bench/kernelbench/domains/ml.py, which the reproduction protocol for this
# integration pass forbids editing. This build.sh therefore builds/verifies
# the artifact itself, and STATUS.md records a functional-only gate using
# the artifact's OWN test scripts (tests/test_jit.py, tests/test_mxgemm.py),
# not a kernelbench --impl run.
#
# MXBLAS ships no ahead-of-time compiled extension -- like Voltrix (same
# author, same JIT-framework design), it JIT-compiles CUDA kernels via nvcc
# (hardcoded "-gencode=arch=compute_90a,code=sm_90a" -- Hopper only, see
# REQUIRES_GPU) the first time a kernel template is actually built
# (mxblas/jit/compiler.py::build(), invoked from mxblas/gemm/kernel_pool.py
# during mx_gemm_kernel()'s own autotuned template search). This script:
#   1. login-node-safe checks that need no GPU (import only)
#   2. the artifact's own test_jit.py + a small test_mxgemm.py run, which DO
#      need a real sm_90a GPU -- run this whole script through
#      `bench/gpu_run.sh -g h100 -t 40 -- '<this path>'` (step 2 is skipped
#      gracefully, not a failure, if no GPU is visible in this shell).
#
# Idempotent: safe to re-run (JIT results are cached on disk under
# MXBLAS_CACHE_DIR); exit 0 = import OK (+ own tests OK, when a GPU was
# visible).
set -euo pipefail
# Toolchain pin (2026-09-04): see bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

echo "== MXBLAS build/verify =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $PY ($("$PY" --version 2>&1))"
echo "nvcc:   $("$CUDA_HOME"/bin/nvcc --version | tail -1)"

# MXBLAS's own JIT cache -- keep it inside this artifact dir (rule 3: every
# file WE cause to be written lives under bench/artifacts/..., never only
# under $HOME), not the artifact's own default (~/.mxblas/).
export MXBLAS_CACHE_DIR="${MXBLAS_CACHE_DIR:-$HERE/cache}"
mkdir -p "$MXBLAS_CACHE_DIR"

# Import-only sanity check: safe with NO GPU present (register_all_kernels()
# is an explicit call the README itself documents as needed "before use";
# nothing at `import mxblas` module-scope compiles or registers anything --
# confirmed by reading mxblas/__init__.py + mxblas/jit_kernels/__init__.py).
# Also confirms the Python>=3.12 requirement (typing.override in
# mxblas/gemm/naive_templates.py / tma_scales_template.py) is met -- this
# was the second, GPU-independent blocker recorded in the original
# DEFERRED-HARDWARE ruling; kb-env's Python 3.12.14 satisfies it.
LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "source")
import mxblas
assert hasattr(mxblas, "mx_gemm_kernel") and hasattr(mxblas, "register_all_kernels")
print("mxblas import OK:", mxblas.__file__)
PYEOF

# Artifact's own JIT smoke test (tests/test_jit.py) + a small MX-GEMM
# correctness/perf run (tests/test_mxgemm.py, run at a modest square shape
# instead of its 8192^3 default -- fast enough for a login-node-budget gate
# while still exercising the real WGMMA/TMA-scales mainloop template).
# NEEDS a real sm_90a GPU -- skipped (not a failure) when this shell has
# none, e.g. a plain login-node invocation.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "-- tests/test_jit.py --"
  ( cd source && PYTHONPATH="$HERE/source:${PYTHONPATH:-}" \
    LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" tests/test_jit.py )

  echo "-- tests/test_mxgemm.py (M=N=K=SM=SN=SK=2048, TT per-tensor scaling) --"
  ( cd source && PYTHONPATH="$HERE/source:${PYTHONPATH:-}" \
    LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" tests/test_mxgemm.py \
    -m 2048 -n 2048 -k 2048 -sm 2048 -sn 2048 -sk 2048 --repeats 8 --warmup 8 )

  echo "== MXBLAS build: OK (import + own test_jit.py + test_mxgemm.py on GPU) =="
else
  echo "== MXBLAS build: OK (import only -- no GPU visible in this shell;" \
       "rerun via 'bench/gpu_run.sh -g h100 -t 40 -- $0' to exercise the JIT tests) =="
fi
