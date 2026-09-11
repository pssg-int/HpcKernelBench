#!/usr/bin/env bash
# vit-sparse (KLab-AI3/ipdps-26_vit) has NO compiled CUDA extension anywhere
# in the repo (confirmed: no setup.py/*.cu/*.cpp outside the vendored
# models/ subtrees, which this adapter does not use at all -- see
# adapter.py's module docstring). The kernel this adapter wraps
# (src/run_bench.py::attn_block, an SDPA call) and its preprocessing
# (src/extract_blocks.py::extract_blocks) are pure Python + numpy/scipy/
# torch, already satisfied by this machine's shared venv (torch 2.8/cu128,
# numpy, scipy). "Build" here is therefore just an import/smoke check --
# idempotent, exit 0 if it succeeds.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
SRC_DIR="$HERE/source"

if [[ ! -d "$SRC_DIR/src" ]]; then
    echo "[vit-sparse] ERROR: $SRC_DIR/src missing -- clone the repo into source/ first" >&2
    exit 1
fi

echo "[vit-sparse] python: $($PY --version)"
echo "[vit-sparse] repo commit: $(git -C "$SRC_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[vit-sparse] verifying imports (no compilation needed -- pure Python artifact)..."
PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}" LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" -c "
import sys
sys.path.insert(0, '$SRC_DIR')
from src.extract_blocks import extract_blocks, Block
from src.run_bench import build_block_mask_from_adj, attn_block, make_qkv, get_dtype
import torch
print('[vit-sparse] import OK; torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
"
echo "[vit-sparse] build.sh done (no compilation step -- pure Python)."
