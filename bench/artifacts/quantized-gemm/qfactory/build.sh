#!/usr/bin/env bash
# QFactory's `qfactory` Python package (source/qfactory/) is pure Python: no
# C-extension build step of its own. It is a "Qtile-graph compiler" (its own
# term) that emits CUDA source and JIT-compiles it with nvcc THE FIRST TIME
# a QLinear/QMatmul is actually invoked (see adapter.py's prepare(), which
# is where that JIT build happens -- this artifact's own preprocessing, same
# convention as tilus). This script only verifies `import qfactory` works
# (source/ is added to sys.path directly by adapter.py -- no `pip install`
# needed at all, since there is nothing to compile ahead of time).
#
# `requirements.txt` lists `bitblas==0.1.0`, but grep confirms nothing under
# source/qfactory/ actually imports bitblas -- it is only used by the
# separate end-to-end reproduction scripts (third_party/vllm-bitblas,
# reproduce/fig11-13), not the core kernel-compiler package this adapter
# wraps. Not installed here (would be a large, unnecessary dependency for a
# kernel-only integration).
#
# Idempotent: safe to re-run (no state to reset besides a fresh import check).
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

echo "== qfactory (quantized-gemm) build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $($PY --version)"

echo "-- verifying import (LD_PRELOAD needed for libstdc++ ABI, see STATUS.md) --"
# Login-node RLIMIT_NPROC=256 for the whole user session: numpy's OpenBLAS
# backend defaults to one thread per core (nproc, often 64-128+) and its
# pthread_create() calls exhaust that limit under concurrent session load,
# segfaulting deep inside numpy's CPU-dispatcher init (not an artifact bug --
# same class of resource contention as the `make -j` cap elsewhere in this
# repo). Cap it for this import check; overridable, harmless on any machine.
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}" "$PY" -c "
import sys; sys.path.insert(0, 'source')
import qfactory
from qfactory import QLinear
print('qfactory', qfactory.__file__, '-- import OK')
"
mkdir -p cache
echo "== qfactory (quantized-gemm) build: OK =="
