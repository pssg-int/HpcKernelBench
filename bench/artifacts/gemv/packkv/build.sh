#!/usr/bin/env bash
# PackKV (github.com/BoJiang03/PackKV) ships a real ahead-of-time CUDA
# extension for its KV-cache codec + fused decompression+matvec kernels
# (packkv_cuda_ext/, built via torch.utils.cpp_extension.CUDAExtension).
#
# REAL BUILD-SYSTEM BUG FOUND (arch flag, ARTIFACT_GUIDE rule 3 allows this
# class of fix): packkv_cuda_ext/setup.py hardcodes
# `-gencode=arch=compute_120,code=sm_120` (RTX Pro 6000 / Blackwell) as the
# ONLY nvcc gencode flag; the A100 (sm_80) line is present but commented
# out in the artifact's own source (`# '-gencode=arch=compute_80,code=
# sm_80', # this works for A100`). Because setup.py sets explicit
# `-gencode` flags itself, torch's cpp_extension does NOT auto-inject
# flags from $TORCH_CUDA_ARCH_LIST (it only does that when the extension
# doesn't already specify its own -gencode/-arch flags) -- so building
# as-is on this A100 login node produces a .so with no sm_80 code, which
# would fail to load/launch at runtime. Idempotent `sed -i` patch below
# ADDS an sm_80 gencode entry alongside the existing sm_120 one (both
# targets end up in the fatbin; nothing upstream removed). Same class of
# fix ARTIFACT_GUIDE rule 3 explicitly allows ("arch flags... are fine");
# same "patch source/ directly, idempotently, in build.sh" precedent
# `gemm/moonpoly/build.sh` already establishes in this project (there via
# `git apply`, here via `sed -i` since it's a one-line flag change, not a
# multi-file patch).
#
# Idempotent: safe to re-run (sed only fires if the old line is still
# present; pip install overwrites vendor/ cleanly).
set -euo pipefail
# Pin the CUDA toolchain to 12.9 (matching plexus_env's torch 2.8.0+cu128) --
# needed since 2026-09-04, when this login node's default module became
# cudatoolkit/13.2 (HPC SDK 26.5): that module's nvcc/CUDA_HOME/CPATH would
# otherwise leak 13.2 headers into this build. See toolchain.sh's own header
# comment for the full incident (torch's ATen/cuda headers need cusparse.h
# from the matching math_libs tree, so CPATH is re-pointed, not cleared).
# NOTE: this must run BEFORE `cd "$(dirname "$0")"` below -- BASH_SOURCE[0]
# keeps whatever form (relative/absolute) the script was invoked with;
# resolving it AFTER an already-changed cwd doubles the relative path and
# fails with "No such file or directory" whenever this script is invoked
# via a relative path (e.g. `bench/artifacts/gemv/packkv/build.sh` from the
# repo root) -- a real invocation-order bug, not a machine difference.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
export CC="${CC:-/opt/cray/pe/gcc-native/14/bin/gcc}"
export CXX="${CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"

echo "== packkv (gemv, KV-cache-compressed KQ decode) build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $($PY --version)"
echo "CC=$CC CXX=$CXX"
command -v nvcc >/dev/null || { echo "nvcc not on PATH -- module load cudatoolkit?"; exit 1; }
nvcc --version | tail -1

SETUP_PY="source/packkv_cuda_ext/setup.py"
echo "-- ensuring $SETUP_PY has an ACTIVE (uncommented) sm_80 gencode flag --"
# KB_SM is passed as an argv (this invocation line is OUTSIDE the quoted
# heredoc, so bash DOES expand ${KB_SM:-80} here). It must NOT be written as a
# literal ${...} into setup.py: bash does not expand inside '<<'PYEOF'', so a
# literal '-gencode=arch=compute_${KB_SM:-80}...' would reach nvcc verbatim and
# fail (this is the H100 fresh-build failure fixed here). The cleanup step also
# strips any such broken literal line a prior run may have inserted (source/ is
# not git-reset between builds), keeping the patch idempotent.
"$PY" - "$SETUP_PY" "${KB_SM:-80}" <<'PYEOF'
import sys
path, kb_sm = sys.argv[1], sys.argv[2]
lines = open(path).readlines()
# 1) drop any line a previous (broken or prior-arch) run of THIS script added:
#    the un-expanded literal, or a resolved bench-comment line for any arch.
lines = [ln for ln in lines
         if "compute_${KB_SM" not in ln
         and "added by bench/artifacts/gemv/packkv/build.sh" not in ln]
target = f"compute_{kb_sm},code=sm_{kb_sm}',"
# idempotency check must look for an UNCOMMENTED matching line specifically --
# the artifact's own original file already has sm_80 mentioned, but only
# inside a comment (`# '-gencode=...sm_80...'`), which a naive substring
# grep would mistake for "already patched" and wrongly skip the real fix.
already_active = any(
    target in ln and not ln.strip().startswith('#')
    for ln in lines)
if already_active:
    print(f"sm_{kb_sm} gencode flag already active -- nothing to do")
    open(path, "w").writelines(lines)
else:
    marker = "'-gencode=arch=compute_120,code=sm_120', # this works for RTX Pro 6000"
    idx = next((i for i, ln in enumerate(lines) if marker in ln), None)
    assert idx is not None, "expected marker line not found -- setup.py changed upstream?"
    indent = lines[idx][:len(lines[idx]) - len(lines[idx].lstrip(' '))]
    addition = (indent + f"'-gencode=arch=compute_{kb_sm},code=sm_{kb_sm}', "
                "# added by bench/artifacts/gemv/packkv/build.sh\n")
    lines.insert(idx + 1, addition)
    open(path, "w").writelines(lines)
    print("patched:", addition.strip())
PYEOF
grep -n "gencode" "$SETUP_PY"

rm -rf vendor source/packkv_cuda_ext/build source/packkv_cuda_ext/*.egg-info
mkdir -p vendor
(cd source/packkv_cuda_ext && "$PY" -m pip install --target=../../vendor --no-deps --no-build-isolation .)

echo "-- verifying import (LD_PRELOAD needed for libstdc++ ABI, see STATUS.md) --"
# NOTE: `import torch` MUST happen before `import packkv_cuda` -- unlike
# marlin's package (whose __init__.py imports torch itself before touching
# marlin_cuda), packkv_cuda is a bare C extension with no Python wrapper;
# its .so depends on libc10.so (torch's core lib) which only resolves once
# torch has already loaded it into the process.
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" -c "
import sys; sys.path.insert(0, 'vendor')
import torch
import packkv_cuda
print('packkv_cuda', packkv_cuda.__file__, '-- import OK')
print([x for x in dir(packkv_cuda) if not x.startswith('_')])
"
echo "== packkv (gemv, KV-cache-compressed KQ decode) build: OK =="
