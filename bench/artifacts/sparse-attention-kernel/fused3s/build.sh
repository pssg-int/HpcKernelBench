#!/usr/bin/env bash
# This is NOT a build -- Fused3S's F3S torch extension is already built
# under bench/artifacts/sddmm/fused3s/ (its own build.sh, own STATUS.md).
# `source` here is a symlink to that build's source tree
# (`ln -s ../../sddmm/fused3s/source source`), read-only reuse: this script
# only VERIFIES the already-compiled extension is present and importable,
# per this task's instruction ("reusing that build ... do not rebuild").
#
# Why reuse rather than build a second copy: F3S.cpython-*.so is the exact
# same torch CUDA extension either way (same source, same commit, same
# toolchain) -- compiling it twice would double the build time and risk the
# two copies silently drifting if bench/artifacts/sddmm/fused3s/build.sh is
# ever re-run with a different toolchain, for zero benefit. adapter.py here
# wraps a DIFFERENT boundary of the SAME compiled kernel (the full fused
# SDDMM+softmax+SpMM pipeline, applySoftmax=True, vs. the sddmm track's
# SDDMM-stage-only wrapper, applySoftmax=False) -- no new C++/CUDA code is
# needed, so there is nothing new to compile here at all.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
SRC_DIR="$HERE/source/src"

if [[ ! -L "$HERE/source" ]]; then
    echo "[fused3s-sparse-attn] ERROR: $HERE/source must be a symlink to "\
         "../../sddmm/fused3s/source (found a real directory or nothing instead)" >&2
    exit 1
fi

if ! compgen -G "$SRC_DIR"/F3S.cpython-*.so > /dev/null; then
    echo "[fused3s-sparse-attn] ERROR: F3S*.so not found under $SRC_DIR --" >&2
    echo "  build it first: bash ../../sddmm/fused3s/build.sh" >&2
    exit 1
fi
echo "[fused3s-sparse-attn] found $(compgen -G "$SRC_DIR"/F3S.cpython-*.so)"

echo "[fused3s-sparse-attn] verifying import..."
LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" -c "
import sys
sys.path.insert(0, '$SRC_DIR')
import torch
import F3S
print('[fused3s-sparse-attn] import OK:', F3S.f3s_1tb1tcb, F3S.preprocess_gpu)
print('[fused3s-sparse-attn] cuda available:', torch.cuda.is_available())
"
echo "[fused3s-sparse-attn] build.sh done (verification only, nothing compiled)."
