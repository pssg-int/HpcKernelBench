#!/usr/bin/env bash
# TC-GNN -- spmm track. REUSES the gnn-aggregation track's build (task
# brief: "Add ... adapters ... REUSING builds that already exist for other
# tracks"). TC-GNN's compiled `TCGNN` torch extension (source/TCGNN_conv/
# {TCGNN.cpp,TCGNN_kernel.cu}) has no notion of "gnn-aggregation" vs
# "spmm" -- `forward_AGNN` is a genuine weighted WMMA/tensor-core SpMM,
# `Y = A @ X` for an arbitrary real-valued sparse `A` fed as a per-edge
# value array; only the *reference* each track compares against differs
# (gnn-aggregation: GCN-normalized aggregation; spmm: plain C = A@B with
# the workload's own random CSR values). This script does NOT compile
# anything -- it only:
#   1. symlinks source/ -> ../../gnn-aggregation/tc-gnn/source (same
#      commit, no second clone);
#   2. verifies the gnn-aggregation track's build/TCGNN*.so (the actual
#      compiled extension) is present and importable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$HERE/../../toolchain.sh"
cd "$HERE"

GNNAGG_DIR="../../gnn-aggregation/tc-gnn"
PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

echo "== tc-gnn (spmm, weighted WMMA SpMM via forward_AGNN) build: REUSE gnn-aggregation/tc-gnn =="

if ! compgen -G "$GNNAGG_DIR/build"/TCGNN*.so > /dev/null; then
  echo "ERROR: $GNNAGG_DIR/build/TCGNN*.so not built -- run $GNNAGG_DIR/build.sh first" >&2
  exit 1
fi

# source/ is a symlink to the gnn-aggregation track's own clone -- not a
# second clone.
if [ -L source ]; then
  rm -f source
elif [ -e source ]; then
  echo "ERROR: source exists and is not a symlink -- refusing to clobber" >&2
  exit 1
fi
ln -s "$GNNAGG_DIR/source" source

echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $($PY --version)"
echo "reused build: $(readlink -f "$GNNAGG_DIR/build")"

echo "-- verifying import from gnn-aggregation/tc-gnn's build/ (no local build/ compiled here) --"
LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" -c "
import sys, os, torch
build_dir = os.path.join('$GNNAGG_DIR', 'build')
sys.path.insert(0, build_dir)
import TCGNN
print('TCGNN', TCGNN.__file__, '-- import OK (reused from gnn-aggregation/tc-gnn/build)')
print('forward_AGNN:', TCGNN.forward_AGNN)
"
echo "== tc-gnn (spmm) build: OK (no compilation performed) =="
