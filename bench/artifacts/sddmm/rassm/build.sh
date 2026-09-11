#!/usr/bin/env bash
# Build librassm_sddmm.so: our own thin driver (wrapper.cpp, NOT part of the
# artifact) that calls RASSM's unmodified sddmm_kstream() kernel over its own
# unmodified Residue/ATM tiling machinery. See wrapper.cpp's file header and
# STATUS.md for why a driver was needed instead of the artifact's own `rassm`
# CLI binary (its --kernel sddmm path is dead code in the default build).
#
# Idempotent: exit 0 if already built and up to date.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CXX="${CXX:-g++}"
SRC=wrapper.cpp
# util.cpp/global.cpp are unmodified artifact source: util.cpp defines
# compare1/compare2 (used by CSR/CSC/ATM's internal std::qsort calls) plus
# cache-flush helpers the headers declare as extern; global.cpp defines the
# extern globals (CACHE_NUM_WAYS, alpha_val, beta_val, ...) Residue.h's tile
# generator reads. Linking them in is the plain "build one more artifact
# .cpp alongside ours" step, not a code change.
ARTIFACT_SRC1=source/code/src/util.cpp
ARTIFACT_SRC2=source/code/src/global.cpp
OUT=librassm_sddmm.so
INCLUDE="source/code/include"

if [[ -f "$OUT" && "$OUT" -nt "$SRC" && "$OUT" -nt "$ARTIFACT_SRC1" && "$OUT" -nt "$ARTIFACT_SRC2" ]]; then
    echo "[rassm/sddmm] $OUT up to date, skipping rebuild"
    exit 0
fi

echo "[rassm/sddmm] compiler: $($CXX --version | head -1)"
# -include string: config.h uses std::string/std::map<std::string,...> (for
# its runtype<->string tables) without including <string> itself; every
# artifact .cpp that happens to pull in <string> transitively via some other
# header (as main.cpp does via boost) never notices, but global.cpp compiled
# standalone does not. Force-including <string> is a compiler-flag fix, not
# a source patch.
"$CXX" -O2 -std=c++17 -fPIC -fopenmp -Wno-unused-parameter -include string \
    -I "$INCLUDE" -I source/code/jstream \
    -shared -o "$OUT" "$SRC" "$ARTIFACT_SRC1" "$ARTIFACT_SRC2"

echo "[rassm/sddmm] built $OUT"
