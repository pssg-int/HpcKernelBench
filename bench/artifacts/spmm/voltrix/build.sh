#!/usr/bin/env bash
# Build/verify Voltrix-SpMM (source/) for the spmm-binary-adjacency-kernel
# variant. Voltrix ships no ahead-of-time compiled extension -- it JIT
# compiles its CUDA kernels (via nvcc, hardcoded
# "-gencode=arch=compute_90a,code=sm_90a" -- Hopper only, see REQUIRES_GPU)
# the first time each kernel function is actually CALLED
# (voltrix/jit/compiler.py::build(), invoked from
# voltrix/jit_kernels/tuner.py::JITTuner.compile_and_tune). This build.sh
# therefore does two things:
#   1. login-node-safe checks that need no GPU (import only -- see below for
#      why that's safe; nvcc presence/version)
#   2. a warm-up compile+launch that DOES need a real sm_90a GPU -- run this
#      whole script through `bench/gpu_run.sh -g h100 -t 40 -- '<this path>'`
#      (step 2 is skipped gracefully, not a failure, if no GPU is visible in
#      this shell -- e.g. when invoked plainly on the login node).
#
# Idempotent: safe to re-run (JIT results are cached on disk under
# VOLTRIX_CACHE_DIR, keyed by a hash of the compiled source + flags, see
# voltrix/jit/compiler.py::build()); exit 0 = import OK (+ warm-up kernel
# launch OK, when a GPU was visible).
set -euo pipefail
# Toolchain pin (2026-09-04): see bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

echo "== Voltrix-SpMM build/verify =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $PY"
echo "nvcc:   $("$CUDA_HOME"/bin/nvcc --version | tail -1)"

# Voltrix's own JIT cache -- keep it inside this artifact dir (rule 3: every
# file WE cause to be written lives under bench/artifacts/..., never only
# under $HOME), not the artifact's own default (~/.voltrix-spmm/).
# adapter.py sets the identical default so a runner process (which does not
# source this script) still shares the same warm cache.
export VOLTRIX_CACHE_DIR="${VOLTRIX_CACHE_DIR:-$HERE/cache}"
mkdir -p "$VOLTRIX_CACHE_DIR"

# Import-only sanity check: safe with NO GPU present (JIT compilation is
# triggered lazily on first kernel *call*, never on import -- voltrix's own
# __init__.py only imports function/constant DEFINITIONS, see
# voltrix/project/*.py + voltrix/jit_kernels/__init__.py; nothing at module
# scope calls compile_and_tune()).
LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "source")
import voltrix
assert hasattr(voltrix, "csr_preprocess") and hasattr(voltrix, "spmm")
print("voltrix import OK:", voltrix.__file__)
PYEOF

# Warm-up compile + kernel launch (NEEDS a real sm_90a GPU). Mirrors the
# artifact's own tests/test_spmm.py at a tiny size, just to populate the JIT
# cache and fail fast here (with a clear message) rather than inside the
# harness gate. Skipped (not a failure) when this shell has no GPU, e.g. a
# plain login-node invocation -- rerun via
# `bench/gpu_run.sh -g h100 -t 40 -- 'bench/artifacts/spmm/voltrix/build.sh'`
# to actually exercise this part.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "source")
import numpy as np, scipy.sparse as sp, torch
import voltrix

torch.manual_seed(0)
np.random.seed(0)
n = 64
A = sp.random(n, n, density=0.2, format="csr")
indptr = torch.tensor(A.indptr, dtype=torch.int32)
indices = torch.tensor(A.indices, dtype=torch.int32)
feat = torch.randn(n, 32, dtype=torch.float32).cuda()

blk_offsets, hspa_packed, hind = voltrix.csr_preprocess(indptr, indices, n)
hspa_packed.hash_tag = "build_warmup"
out = voltrix.spmm(blk_offsets, hspa_packed, hind, num_nodes=n,
                    num_edges=indices.numel(), feat=feat)
torch.cuda.synchronize()
print("voltrix warm-up SpMM OK, output shape", tuple(out.shape))
PYEOF
  echo "== Voltrix-SpMM build: OK (import + warm-up GPU launch) =="
else
  echo "== Voltrix-SpMM build: OK (import only -- no GPU visible in this shell;" \
       "rerun via 'bench/gpu_run.sh -g h100 -t 40 -- $0' to exercise the JIT warm-up) =="
fi
