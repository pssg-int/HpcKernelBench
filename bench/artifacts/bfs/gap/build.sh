#!/usr/bin/env bash
# Build the GAP Benchmark Suite BFS baseline (CPU, OpenMP) as gap_shim.so:
# gap_shim.cc (this dir) #includes GAP's own unmodified src/bfs.cc (main
# renamed). Flags follow GAP's own Makefile (-std=c++11 -O3 -fopenmp), plus
# -fPIC for the shared library. source/ is cloned by toolchain.sh.
#
# Idempotent. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source/src"
CXX="${CXX:-g++}"

echo "=== GAP BFS baseline build (gap_shim.so) ==="
echo "commit: $(git -C "$HERE/source" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "host cxx: $($CXX --version | head -1)"

"$CXX" -std=c++11 -O3 -Wall -fopenmp -fPIC -shared \
    -I "$SRC" -o "$HERE/gap_shim.so" "$HERE/gap_shim.cc"

[ -f "$HERE/gap_shim.so" ] || { echo "STATUS: BUILD-FAILED -- gap_shim.so not produced"; exit 1; }
echo "Built: $HERE/gap_shim.so"
