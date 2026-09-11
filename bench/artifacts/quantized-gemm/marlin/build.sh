#!/usr/bin/env bash
# Marlin -- quantized-gemm track. REUSES the gemv track's build (per the
# task brief: "Already built for the gemv track... Create
# artifacts/quantized-gemm/marlin/ that REUSES that build ... do not
# rebuild"). This script does NOT compile anything -- it only:
#   1. symlinks source/ -> ../../gemv/marlin/source (same clone, same
#      commit, so provenance is identical -- ARTIFACT_GUIDE rule 3's
#      general spirit of "fetched by build.sh, not by hand" is satisfied
#      by making the reuse itself an explicit, idempotent build.sh step
#      rather than a silent out-of-band symlink);
#   2. verifies the gemv track's vendor/ (the actual compiled
#      marlin_cuda*.so extension) is present and importable.
#
# Rationale for reuse rather than a second independent build: Marlin's CUDA
# extension (marlin_cuda) has no notion of "gemv" vs "quantized-gemm" --
# it is the SAME compiled .so, the SAME INT4-weight/FP16-activation kernel,
# for either track; only the *shape mapping* in each track's adapter.py
# differs (gemv wraps it at batch m=1 with a transpose trick since gemv's
# "A" plays the role of marlin's weight; quantized-gemm wraps it directly,
# since this track's A[M,K]@W[K,N] already matches marlin's own
# A_act[m,k]@B_weight[k,n] convention with no transpose needed -- see
# adapter.py's module docstring). Building it twice would waste login-node
# compile time on an identical artifact and risk the two vendor/ trees
# drifting (different pip/torch/CUDA state at build time).
set -euo pipefail
# NOTE: source toolchain.sh BEFORE `cd` below -- BASH_SOURCE[0] keeps
# whatever form (relative/absolute) the script was invoked with; resolving
# it AFTER an already-changed cwd doubles the relative path and fails with
# "No such file or directory" whenever this script is invoked via a
# relative path (e.g. `bench/artifacts/quantized-gemm/marlin/build.sh` from
# the repo root) -- a real invocation-order bug, not a machine difference.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "${BASH_SOURCE[0]}")"

GEMV_MARLIN_DIR="../../gemv/marlin"
PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

echo "== marlin (quantized-gemm, w4a16 batched/decode GEMM) build: REUSE gemv/marlin =="

if [ ! -d "$GEMV_MARLIN_DIR/vendor/marlin" ]; then
  echo "ERROR: $GEMV_MARLIN_DIR/vendor not built -- run $GEMV_MARLIN_DIR/build.sh first" >&2
  exit 1
fi

# source/ is a symlink to the gemv track's own clone -- not a second clone.
if [ -L source ]; then
  rm -f source
elif [ -e source ]; then
  echo "ERROR: source exists and is not a symlink -- refusing to clobber" >&2
  exit 1
fi
ln -s "$GEMV_MARLIN_DIR/source" source

echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $($PY --version)"
echo "reused vendor: $(readlink -f "$GEMV_MARLIN_DIR/vendor")"

echo "-- verifying import from gemv/marlin's vendor (no local vendor/ built here) --"
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" -c "
import sys, os
vendor = os.path.join('$GEMV_MARLIN_DIR', 'vendor')
sys.path.insert(0, vendor)
import marlin
import marlin_cuda
print('marlin', marlin.__file__, '-- import OK (reused from gemv/marlin/vendor)')
print('marlin_cuda', marlin_cuda.__file__, '-- import OK')
"
echo "== marlin (quantized-gemm) build: OK (no compilation performed) =="
