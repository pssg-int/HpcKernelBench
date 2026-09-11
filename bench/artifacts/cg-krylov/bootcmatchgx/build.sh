#!/usr/bin/env bash
# This directory REUSES the already-built BootCMatchGX binary from the
# multigrid track's own integration (../../multigrid/bootcmatchgx/) --
# `source` and `fcg_bcmg.properties` here are symlinks to that directory's
# files (same clone, same build, same solver config), not a second clone
# or a second compile. There is no separate source tree to build.
#
# This script therefore does NOT rebuild anything -- it only verifies the
# shared binary is present (idempotent, exit 0 = usable) and points to the
# sibling artifact's own build.sh if it is not.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$HERE/source/BCMGX/bin/example/driverSolve"
MG_BUILD="$HERE/../../multigrid/bootcmatchgx/build.sh"

echo "== bootcmatchgx (cg-krylov track) -- shared-build check =="
echo "source/ -> $(readlink "$HERE/source" 2>/dev/null || echo '(not a symlink?)')"

if [ ! -x "$BINARY" ]; then
    echo "MISSING $BINARY" >&2
    echo "This directory reuses the multigrid track's build; run it first:" >&2
    echo "  $MG_BUILD" >&2
    exit 1
fi

_ldd_out="$(ldd "$BINARY" || true)"  # ldd exits nonzero when ANY dep is
                                      # unresolved (expected here for driver
                                      # libs); don't let `set -e` abort
                                      # before the filtered check below runs
# libcuda.so.1/libnvidia-ml.so.1 are GPU-driver runtime libs, never present
# on this GPU-less login node (see MACHINE FACTS); expected-missing here,
# resolved only when this binary actually runs on a GPU node. Any OTHER
# unresolved library still means the shared binary is stale/broken.
_missing="$(grep "not found" <<< "$_ldd_out" | grep -v -E 'libcuda\.so\.1|libnvidia-ml\.so\.1' || true)"  # grep -v
    # legitimately exits 1 when EVERY "not found" line is the expected
    # driver libs (nothing left to output) -- under `set -e -o pipefail`
    # that would otherwise abort the script right here
if [ -n "$_missing" ]; then
    echo "MISSING SHARED LIBS (shared binary is stale/broken):"; echo "$_missing"
    exit 1
fi
echo "== bootcmatchgx (cg-krylov track): OK (reusing multigrid's build) =="
