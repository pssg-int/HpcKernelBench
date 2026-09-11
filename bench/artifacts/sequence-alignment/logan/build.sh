#!/usr/bin/env bash
# LOGAN (IPDPS'20) -- build the upstream `demo` binary (unmodified, sanity
# check only) plus `kernelbench_driver` (this integration's additive driver,
# source/src/kernelbench_driver.cu -- see that file's header comment and
# STATUS.md for why a new driver was needed instead of wrapping `demo`).
#
# Idempotent: safe to re-run; exit 0 = both binaries present and executable.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/source"

echo "== LOGAN build =="
echo "commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "STATUS: BUILD-FAILED -- nvcc not found on PATH"
    exit 1
fi
echo "nvcc   : $(command -v nvcc)"
nvcc --version | tail -1

# sm_80 (this machine's A100) instead of the Makefile's v100-only sm_70
# flag -- a build-system arch-flag override, not a kernel-code change
# (ARTIFACT_GUIDE.md rule 3). Passed on the make command line, not by
# editing the Makefile.
CUDAFLAGS="-O3 -maxrregcount=32 -std=c++14 -Isrc -Xcompiler -fopenmp -arch=sm_80"

make demo CUDAFLAGS="$CUDAFLAGS"
if [ ! -x demo ]; then
    echo "STATUS: BUILD-FAILED -- demo binary not produced"
    exit 1
fi
echo "built  : source/demo (upstream, unmodified)"

# The driver is versioned OUTSIDE source/ (source/ is git-ignored; see
# kernel-papers/.gitignore) and copied in so it can be compiled in-tree
# against LOGAN's own objects. Same file, two locations by design.
cp "$HERE/kernelbench_driver.cu" src/kernelbench_driver.cu
nvcc -c $CUDAFLAGS -dc src/kernelbench_driver.cu -o src/kernelbench_driver.o
nvcc $CUDAFLAGS src/kernelbench_driver.o src/seed.o src/score.o src/logan_functions.o -o kernelbench_driver
if [ ! -x kernelbench_driver ]; then
    echo "STATUS: BUILD-FAILED -- kernelbench_driver binary not produced"
    exit 1
fi
echo "built  : source/kernelbench_driver (this integration's additive driver)"
echo "== LOGAN build: OK =="
