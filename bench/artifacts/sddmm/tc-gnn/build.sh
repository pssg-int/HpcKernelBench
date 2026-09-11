#!/usr/bin/env bash
# TC-GNN -- sddmm track. REUSES the gnn-aggregation track's build (task
# brief: "Add ... adapters ... REUSING builds that already exist for other
# tracks"). TC-GNN's compiled `TCGNN` torch extension (source/TCGNN_conv/
# {TCGNN.cpp,TCGNN_kernel.cu}) has no notion of "gnn-aggregation" vs
# "sddmm" -- `forward_ef` (TC-GNN's SDDMM binding, PYBIND11 name
# "forward_ef" -> C++ `sddmm_forward` -> `sddmm_forward_cuda`) is a
# self-contained tensor-core dot-product-at-the-sparsity-pattern kernel,
# usable for any track that needs exactly that. This script does NOT
# compile anything -- it only:
#   1. symlinks source/ -> ../../gnn-aggregation/tc-gnn/source (same
#      commit, no second clone);
#   2. verifies the gnn-aggregation track's build/TCGNN*.so (the actual
#      compiled extension) is present and importable, and that it exposes
#      forward_ef.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$HERE/../../toolchain.sh"
cd "$HERE"

GNNAGG_DIR="../../gnn-aggregation/tc-gnn"
PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

echo "== tc-gnn (sddmm, forward_ef TC-block SDDMM) build: REUSE gnn-aggregation/tc-gnn =="

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
print('forward_ef:', TCGNN.forward_ef)
"
echo "== tc-gnn (sddmm) build: OK (no compilation performed) =="
