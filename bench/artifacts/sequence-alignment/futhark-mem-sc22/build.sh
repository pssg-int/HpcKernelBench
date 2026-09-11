#!/usr/bin/env bash
# futhark-mem-sc22 (SC'22) -- attempt to compile benchmarks/nw/futhark/nw.fut
# (Needleman-Wunsch, BLOSUM62, this repo's one alignment kernel among 7
# unrelated memory-bound benchmarks) with Futhark's CUDA backend, using the
# Futhark compiler binary already vendored in this repo (source/bin/futhark,
# a statically-linked Linux x86_64 ELF checked into the artifact's own git
# history -- no separate release-tarball fetch was needed).
#
# Exits 1 (BUILD-FAILED): the host-side driver compiles (after a build-system
# compiler shim -- see below and STATUS.md), but the actual GPU kernel fails
# to compile via NVRTC on this machine's CUDA 12.9. See STATUS.md for full
# evidence; this script reproduces the failure point deterministically.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

echo "== futhark-mem-sc22 (benchmarks/nw) build =="
echo "commit : $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"

FUTHARK="$(pwd)/source/bin/futhark"
if [ ! -x "$FUTHARK" ]; then
    echo "STATUS: BUILD-FAILED -- vendored futhark binary missing: $FUTHARK"
    exit 1
fi
echo "futhark: $("$FUTHARK" --version | head -1)"

# Build-system compiler shim (ARTIFACT_GUIDE.md rule 3: include/lib-path
# fixes are fine). This machine's system `cc` resolves `#include <cuda.h>`
# to the LOGIN DEFAULT module's CUDA (13.2) even when CPATH names the
# toolchain-pinned 12.9 tree first -- empirically confirmed that CPATH
# alone is NOT sufficient on this image, but an explicit `-I` flag IS (see
# STATUS.md for the isolated repro). Futhark's `cuda` backend invokes a
# bare `cc` with no CUDA-specific flags at all, so this shim (already
# checked into this artifact dir, not modifying anything system-wide)
# prepends the needed -I/-L/-rpath flags and execs the real compiler.
export PATH="$(pwd)/cc_shim:$PATH"
echo "cc     : $(command -v cc) (shim -> $(readlink -f "$(pwd)/cc_shim/cc" 2>/dev/null || echo "$(pwd)/cc_shim/cc"))"

NW_DIR="source/benchmarks/nw/futhark"
cd "$NW_DIR"
rm -f nw nw.c

echo "-- futhark cuda nw.fut (host-side compile) --"
if ! "$FUTHARK" cuda nw.fut; then
    echo "STATUS: BUILD-FAILED -- futhark cuda could not even compile the host driver"
    exit 1
fi
if [ ! -x nw ]; then
    echo "STATUS: BUILD-FAILED -- nw binary not produced"
    exit 1
fi
echo "built (host side): $NW_DIR/nw"

echo "-- functional smoke: ./nw -e mk_input (triggers device-side NVRTC compile) --"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
if echo "512i64" | ./nw -e mk_input -n >/tmp/futhark_nw_smoke.log 2>&1; then
    echo "== futhark-mem-sc22 build: OK (unexpected -- device compile succeeded) =="
    exit 0
fi
echo "device-side NVRTC compile FAILED (expected on this toolchain -- see STATUS.md):"
tail -15 /tmp/futhark_nw_smoke.log
echo "STATUS: BUILD-FAILED -- host driver compiles, but the generated GPU kernel"
echo "        does not compile under this machine's CUDA 12.9 NVRTC (mulhi/mul64hi"
echo "        intrinsics undefined -- see STATUS.md for the full trace)."
exit 1
