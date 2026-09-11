#!/usr/bin/env bash
# GE-SpMM -- spmm track. REUSES the gnn-aggregation track's build (task
# brief: "Add ... adapters ... REUSING builds that already exist for other
# tracks"). GE-SpMM's compiled `spmm.so` (csr_spmm(rowptr,colind,values,
# dense), a genuine weighted CSR SpMM: plain fp32 accumulate, no tensor
# cores) has no notion of "gnn-aggregation" vs "spmm" -- it is exactly
# `Y = A @ X` for an arbitrary real-valued CSR `A`, which is precisely
# spmm's own kernel boundary too. Only the *workload feed* differs: the
# gnn-aggregation adapter feeds a GCN-normalized A_hat; this adapter feeds
# the workload's own CSR values directly (spmm's reference has no GCN
# normalization step at all). This script does NOT compile anything -- it
# only:
#   1. symlinks source/ -> ../../gnn-aggregation/ge-spmm/source (same
#      clone, same commit -- no second clone);
#   2. verifies the gnn-aggregation track's build/spmm.so (the actual
#      compiled extension) is present and importable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$HERE/../../toolchain.sh"
cd "$HERE"

GNNAGG_DIR="../../gnn-aggregation/ge-spmm"
PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

echo "== ge-spmm (spmm, general-purpose GNN-aggregation SpMM) build: REUSE gnn-aggregation/ge-spmm =="

if [ ! -f "$GNNAGG_DIR/build/spmm.so" ]; then
  echo "ERROR: $GNNAGG_DIR/build/spmm.so not built -- run $GNNAGG_DIR/build.sh first" >&2
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

echo "-- verifying import from gnn-aggregation/ge-spmm's build/ (no local build/ compiled here) --"
LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" -c "
import importlib.util, os, torch
so_path = os.path.join('$GNNAGG_DIR', 'build', 'spmm.so')
spec = importlib.util.spec_from_file_location('spmm', so_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('spmm', mod.__file__, '-- import OK (reused from gnn-aggregation/ge-spmm/build)')
print('csr_spmm:', mod.csr_spmm)
"
echo "== ge-spmm (spmm) build: OK (no compilation performed) =="
