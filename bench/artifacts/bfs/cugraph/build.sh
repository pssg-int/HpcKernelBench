#!/usr/bin/env bash
# cuGraph BFS baseline: create this directory's OWN venv (deps/venv,
# git-ignored) with NVIDIA's RAPIDS wheels. Kept out of the harness env on
# purpose: RAPIDS brings its own CUDA libraries (rmm, libcugraph, cupy), and
# the adapter runs cuGraph in a separate worker process (cugraph_worker.py)
# so they never share a process with torch.
#
# Pinned: RAPIDS 25.06 (pylibcugraph-cu12, CUDA 12 wheels, Python 3.10-3.13).
# Override with KB_CUGRAPH_VERSION. Needs internet (pip) on the build node.
#
# Idempotent. Exit 0 = built.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-${KB_PY:-python3}}"
VENV="$HERE/deps/venv"
VER="${KB_CUGRAPH_VERSION:-25.6.*}"

echo "=== cuGraph BFS baseline (venv: $VENV, pylibcugraph-cu12==$VER) ==="
echo "python: $("$PY" --version)"
if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet \
    --extra-index-url=https://pypi.nvidia.com \
    "pylibcugraph-cu12==$VER" "cupy-cuda12x" "numpy"

"$VENV/bin/python" -c "import pylibcugraph, cupy; print('pylibcugraph', pylibcugraph.__version__, '| cupy', cupy.__version__)" \
  || { echo "STATUS: BUILD-FAILED -- pylibcugraph/cupy import failed in $VENV"; exit 1; }
echo "Built: $VENV"
