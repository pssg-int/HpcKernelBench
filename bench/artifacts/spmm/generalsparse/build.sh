#!/usr/bin/env bash
# Build GeneralSparse's host-side code generator/compiler ("token_test", pure
# C++11, no CUDA needed to build it -- it only EMITS .cu text and can also
# shell out to nvcc to compile+run it, which this build does NOT invoke; see
# gs_emit.cc) plus this directory's own gs_emit.cc driver, which reproduces
# ONE of token_test.cc's ~30 hardcoded operator compositions
# (test_spmm_warp_bitmap) and stops at emitting kernel_file.cu -- no compile,
# no execute_binary() -- so adapter.py can turn that text into a callable
# shared library per matrix at prepare()-time (see gs_transform.py).
#
# Idempotent: safe to re-run; exit 0 = built.
set -euo pipefail
# Toolchain pin (2026-09-04): see bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
HOST_COMPILER="${HOST_COMPILER:-$CXX}"

echo "== GeneralSparse build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "host compiler: $($CXX --version | head -1)"
echo "nvcc: $($NVCC --version | tail -1)"

# --- config path fix (build-system fix, ARTIFACT_GUIDE.md rule 3; recorded
#     in source.patch): global_config.json hardcodes the ORIGINAL author's
#     absolute path (/home/wangyaoyu/GeneralSparse) for both ROOT_PATH_STR
#     (where generated data_source/<id>/ dirs land) and spmv_header_file
#     (copied into each of those dirs) -- repoint both at this clone's own
#     source/ dir. Idempotent (only replaces the placeholder if still
#     present).
python3 - "$HERE/source/global_config.json" "$HERE/source" <<'PYEOF'
import json, sys
path, root = sys.argv[1], sys.argv[2]
with open(path) as f:
    cfg = json.load(f)
cfg["ROOT_PATH_STR"] = root
cfg["spmv_header_file"] = root + "/spmm_header_top.code"
with open(path, "w") as f:
    json.dump(cfg, f)
PYEOF
mkdir -p source/data_source

# --- token_test: GeneralSparse's own host-side codegen/compiler (only its
#     .o files are reused below; we never invoke token_test's own main(),
#     which is the ~30-strategy autotuning benchmark SCRIPT -- see gs_emit.cc
#     header comment) ---
( cd source && make token_test -j8 CXX="$CXX" LD="$CXX" >/tmp/gs_make.log 2>&1 ) \
    || { echo "make token_test FAILED (see /tmp/gs_make.log)"; tail -60 /tmp/gs_make.log; exit 1; }
test -x source/token_test

# --- gs_emit: ours, linked against the SAME .o files make just built,
#     minus token_test.o (which carries the ~30-strategy main() we don't
#     want) -- see gs_emit.cc for the single strategy it reproduces.
#
# Filtered to .o files that have a matching .cc/.cpp source (rather than a
# blind glob): source/reduction_token/total_BMT_result_reduce_to_one_
# register_with_pipeline_token.o is a checked-in-by-the-authors object file
# with NO corresponding source anywhere in the repo and is NOT part of the
# makefile's own `src` list for the `token_test` target (verified: grep
# makefile shows only the non-"_with_pipeline_" sibling in `src`) -- i.e. it
# is an orphaned build artifact from the authors' own machine, not something
# `make token_test` produces or needs. A blind `ls *.o` glob picked it up
# anyway and tried to link it into gs_emit, which fails at link time
# (`undefined reference to __libc_single_threaded`) since it was compiled by
# a different, ABI-incompatible toolchain than this machine's. Filtering to
# "has a source file" reproduces exactly the object set `token_test` itself
# links (build-system fix; source/ content untouched).
OBJS=""
for f in $(cd source && ls *.o transform_step/*.o operator/*.o kernel_token/*.o reduction_token/*.o 2>/dev/null | grep -v '^token_test\.o$'); do
    base="${f%.o}"
    if [ -f "source/${base}.cc" ] || [ -f "source/${base}.cpp" ]; then
        OBJS="$OBJS $f"
    else
        echo "gs_emit link: skipping orphaned object with no source: $f"
    fi
done
if [ -z "$OBJS" ]; then
    echo "no .o files found under source/ after make -- build-system mismatch"; exit 1
fi
"$CXX" -fno-lto -g -O3 -std=c++11 -I source \
    -Wno-unused-result -Wno-unused-value -Wno-unused-function \
    -c -o /tmp/gs_emit_$$.o gs_emit.cc
( cd source && "$CXX" -o "$HERE/gs_emit" "/tmp/gs_emit_$$.o" $OBJS -fno-lto -pthread )
rm -f "/tmp/gs_emit_$$.o"
test -x gs_emit

echo "== GeneralSparse build: OK (token_test + gs_emit) =="
