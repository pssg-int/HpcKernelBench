#!/usr/bin/env bash
# MetaAttention has no C++/CUDA extension to compile for THIS integration --
# it is a pure-Python DSL (attn_engine/core) that lowers to GPU code at
# CALL time via TileLang (JIT, no ahead-of-time build step of our own).
# "Building" here means: install its dependencies into an ISOLATED local
# directory (never the shared venv -- see below) and verify the artifact's
# own public API imports and compiles a trivial kernel end to end.
#
# Idempotent: skips the pip install if the target dir already has tilelang.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
PYLIBS="$HERE/pylibs"
SRC="$HERE/source"

echo "== MetaAttention build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown) (PPoPP_AE branch)"
echo "python: $($PY --version), torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"

# ---- 1. isolated dependency install --------------------------------------
# The shared venv (/pscratch/sd/c/cunyang/gnn/plexus_env) already has numpy
# 2.3.3 installed and is used by many unrelated projects on this machine.
# MetaAttention's own pyproject.toml pins numpy<2, and tilelang pulls in a
# non-trivial dependency tree (apache-tvm-ffi, z3-solver, cloudpickle, ...).
# A bare `pip install` here risks silently downgrading numpy and breaking
# every other project sharing this venv. Install everything into an
# isolated --target directory instead (does not touch/resolve against the
# venv's existing packages) and sys.path-inject it from adapter.py.
if [[ ! -d "$PYLIBS/tilelang" ]]; then
    echo "[metaattention] installing deps into isolated target $PYLIBS"
    "$PY" -m pip install --target="$PYLIBS" tilelang einops jinja2 ninja sympy termcolor
else
    echo "[metaattention] deps already present in $PYLIBS, skipping install"
fi

# ---- 2. verify the artifact's own public API imports + compiles ----------
# The 3rd_parties/cutlass* git submodules (NOT initialized) are only
# referenced by the legacy core/template/cute_template* Hopper-specific
# paths (grep-confirmed: `grep -rl 3rd_parties attention_engine/attn_engine
# attention_engine/core` hits only cute_template*/); the TileLang lowering
# path examples/mha.py uses does not need them, so submodule init was
# skipped.
echo "[metaattention] verifying import + a trivial end-to-end compile..."
PYTHONPATH="$PYLIBS:$SRC/attention_engine:$SRC" "$PY" - <<'EOF'
import torch
import tilelang  # noqa: F401
from attn_engine import AttentionEngine, OnlineFunc  # noqa: F401
from core import CustomIO, SymbolScalar, Var, meta_tensor  # noqa: F401
from autotuner.arch import get_attn_device
print("[metaattention] import OK, arch:", get_attn_device().name)
from examples.mha import causal_softmax_attention
mod = causal_softmax_attention(1, 2, 32, 16, 16, dtype=torch.float16, tune=False)
q = torch.randn(1, 32, 2, 16, device="cuda", dtype=torch.float16)
k = torch.randn(1, 32, 2, 16, device="cuda", dtype=torch.float16)
v = torch.randn(1, 32, 2, 16, device="cuda", dtype=torch.float16)
out = mod(q, k, v)
assert out.shape == (1, 32, 2, 16), out.shape
print("[metaattention] trivial compile+run OK:", out.shape, out.dtype)
EOF

echo "== MetaAttention build: OK =="
