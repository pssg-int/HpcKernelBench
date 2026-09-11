#!/usr/bin/env bash
# Hidet has no source/ clone here: it is a proper pip package (source repo is
# archived/read-only on GitHub but still installs from PyPI). "Building"
# means: (1) pip install it into an ARTIFACT-LOCAL --target directory (2) not
# the shared venv -- see "Dependency isolation" below, (2) record the exact
# hidet/torch/nvcc versions used, (3) confirm hidet's own op-level conv2d
# path actually compiles and runs a tiny shape end to end (which is what
# proves the kernel is real and buildable on this machine's CUDA toolchain,
# not just importable).
#
# Dependency isolation (2026-09-05): `pylibs/` (git-ignored, like
# attention-kernel/pat's own pylibs/) instead of `pip install hidet` into the
# shared plexus_env. Rationale: plexus_env is shared with
# bench/artifacts/gemm/hexcute/, whose own build installs a FORK that also
# declares `name = "hidet"` in its package metadata -- whichever install
# lands last in the shared site-packages wins process-wide for every
# subsequent `import hidet`, which broke this adapter the day both builds ran
# (see STATUS.md's "Dependency isolation (2026-09-05)" section for the full
# story). `pip install --target=pylibs --no-deps` installs ONLY the hidet
# package itself into an artifact-local tree; hidet's runtime deps (torch,
# numpy, scipy, click, cuda-python, gitpython, hip-python-fork,
# importlib_metadata, lark, networkx, nvtx, packaging, psutil, requests,
# tabulate, tomlkit, tqdm) are intentionally NOT reinstalled here -- they are
# already present in plexus_env from before this fix (nothing was
# uninstalled) and are resolved normally via sys.path once pylibs/ is
# prepended (adapter.py does this ahead of `import hidet`). `--no-deps` is
# required: a plain `pip install --target=pylibs hidet==0.6.1` (tried once,
# see STATUS.md) does NOT treat plexus_env's already-installed torch/numpy/etc
# as satisfying the requirement and instead re-resolves+downloads a whole
# second ML stack (torch 2.14/cu13, ~5.2 GB) into pylibs/ -- exactly the
# venv-independence goal taken too literally; `--no-deps` avoids it since
# every actual runtime dependency is already importable from plexus_env's
# site-packages (verified once, see STATUS.md).
#
# Idempotent: safe to re-run; exit 0 = usable. This script no longer touches
# plexus_env at all (no `pip install`/`pip uninstall` into the shared venv).
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
HIDET_VERSION="0.6.1"   # pinned; matches STATUS.md's recorded wheel
PYLIBS="$(pwd)/pylibs"

# nvcc 12.9 cannot use this host's default gcc (gcc-native/14, SUSE 14.3.0)
# as a host compiler -- see adapter.py's module docstring for the exact
# parse-error signature. Prepend a compatible one (gcc-native/12, 12.3.0)
# for this build check too, matching what adapter.py does at import time.
export PATH="${KB_HOST_COMPILER_BIN:-/opt/cray/pe/gcc-native/12/bin}:$PATH"

# zaratan-specific addendum (2026-09-09): hidet's NVCC.compile() (installed
# hidet/backend/build.py) never passes -ccbin at all, so nvcc falls back to
# its OWN internal host-compiler search -- which is NOT simply "first match
# on $PATH". Empirically (see STATUS.md's Reproduction section): conda-forge
# nvcc, whose binary lives at $CUDA_HOME/bin/nvcc, always PREPENDS a few
# install-relative dirs (targets/x86_64-linux/bin, nvvm/bin, and a dir that
# resolves right back to $CUDA_HOME/bin itself) to the PATH it hands its
# child `gcc` invocation -- ahead of whatever we put in front of $PATH
# ourselves. Since $CUDA_HOME (=kb-env) is a full conda env that ALSO ships
# its own gcc/g++ 13.4.0 (KB_CC/KB_CXX), that's the compiler nvcc's internal
# search finds first, no matter the PATH order we set. This is a genuine
# difference from Perlmutter's NVIDIA HPC SDK nvcc, whose install dir holds
# no competing gcc, so a plain PATH prepend was sufficient there. Fix:
# nvcc has always supported NVCC_APPEND_FLAGS/NVCC_PREPEND_FLAGS (extra
# flags appended/prepended to every invocation, a real nvcc feature, not a
# hidet one) -- appending an explicit -ccbin bypasses nvcc's host-compiler
# auto-search entirely, with zero changes to hidet's or any kernel's code.
# Verified empirically (see STATUS.md) that plain PATH-prepend alone still
# picks up gcc 13.4.0 headers on this machine; NVCC_APPEND_FLAGS -ccbin
# fixes it. Idempotent (only appends if not already present).
case "${NVCC_APPEND_FLAGS:-}" in
    *-ccbin*) ;;
    *) export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -ccbin ${KB_HOST_COMPILER_BIN:-/opt/cray/pe/gcc-native/12/bin}/g++" ;;
esac

echo "== Hidet build check =="
echo "python: $PY"
echo "host compiler: $(g++ --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"

if ! "$PY" -c "
import sys
sys.path.insert(0, '$PYLIBS')
import hidet
assert hidet.__file__.startswith('$PYLIBS'), 'resolved outside pylibs: ' + hidet.__file__
" >/dev/null 2>&1; then
    echo "hidet not present/importable under $PYLIBS -- installing hidet==$HIDET_VERSION there (--no-deps, see header comment)"
    mkdir -p "$PYLIBS"
    "$PY" -m pip install --no-deps --target="$PYLIBS" "hidet==$HIDET_VERSION"
else
    echo "hidet already present in $PYLIBS, skipping install"
fi

# Light runtime deps hidet actually needs at import time beyond
# numpy/torch/scipy (which every machine this benchmark runs on already
# has, per the shared-venv-independence rationale above): click,
# cuda-bindings (+cuda-pathfinder), gitpython, hip-python-fork,
# importlib_metadata, lark, nvtx, tabulate, tomlkit. These happened to
# already be present in the original plexus_env (Perlmutter) this adapter
# was built against, so they were never installed here explicitly; on a
# fresh venv (e.g. zaratan's kb-env) they are missing and `import hidet`
# fails cascading through hidet/option.py (tomlkit), hidet/cuda/device.py
# (`from cuda.bindings import runtime as cudart`), hidet/utils/...
# (click/tabulate/tqdm/lark/...), etc. Installed the same way as hidet
# itself -- --no-deps, --target=pylibs -- one at a time, only if not
# already importable (from pylibs OR the base venv), so this is a no-op on
# a venv that already has them all (e.g. the original plexus_env).
#
# NOTE on cuda-python vs cuda-bindings: as of cuda-python 12.6.2/13.x on
# PyPI, the top-level `cuda-python` package split into a thin meta-package
# (cuda-python, no code, just Requires-Dist pins) plus the actual
# compiled-code packages `cuda-bindings` (provides `cuda.bindings.*`, which
# is what hidet imports) and `cuda-core`. `pip install --no-deps
# cuda-python` therefore installs an empty dist-info and leaves `import
# cuda` failing -- discovered by inspecting the downloaded wheel (8 KB,
# only METADATA, vs. cuda-bindings' ~7 MB compiled .so wheel). Installing
# cuda-bindings (+ its own light dep cuda-pathfinder, a pure-python runtime
# CUDA-library locator) directly is what actually provides the `cuda`
# namespace package hidet needs. Pinned to 12.8.0/12.9.7 to track this
# machine's CUDA 12.8 toolkit (KB_CUDA_HOME); cuda-bindings loads the
# actual CUDA libraries dynamically at runtime via cuda-pathfinder, so
# exact patch-version match to the toolkit is not required, just the same
# major CUDA series.
for _pkg_spec in "click:click" "cuda-bindings==12.8.0:cuda" "cuda-pathfinder:cuda.pathfinder" \
                 "gitpython:git" \
                 "hip-python-fork:hip" "importlib_metadata:importlib_metadata" \
                 "lark:lark" "nvtx:nvtx" "tabulate:tabulate" "tomlkit:tomlkit"; do
    _pip_name="${_pkg_spec%%:*}"
    _mod_name="${_pkg_spec##*:}"
    if ! "$PY" -c "
import sys
sys.path.insert(0, '$PYLIBS')
import $_mod_name
" >/dev/null 2>&1; then
        echo "hidet light dep '$_mod_name' (pip: $_pip_name) not importable -- installing into $PYLIBS"
        mkdir -p "$PYLIBS"
        "$PY" -m pip install --no-deps --target="$PYLIBS" "$_pip_name"
    else
        echo "hidet light dep '$_mod_name' already importable, skipping install"
    fi
done

PYTHONPATH="$PYLIBS" "$PY" - <<EOF
import sys
sys.path.insert(0, "$PYLIBS")
import torch, hidet
assert hidet.__file__.startswith("$PYLIBS"), \
    f"hidet resolved to {hidet.__file__!r}, expected under $PYLIBS -- pylibs install broken"
print("hidet  :", hidet.__version__, "(", hidet.__file__, ")")
print("torch  :", torch.__version__)
print("cuda   :", torch.version.cuda, "available:", torch.cuda.is_available())
EOF

# Functional check: hidet's own op-level conv2d API, traced/optimized into a
# FlowGraph and run once on a tiny fp32 shape, checked against a from-scratch
# fp64 numpy reference (independent of kernelbench's own reference_conv, so
# this check does not depend on kernelbench being importable/on sys.path).
"$PY" - <<EOF
import sys
sys.path.insert(0, "$PYLIBS")
import numpy as np
import torch
import hidet
assert hidet.__file__.startswith("$PYLIBS"), f"wrong hidet: {hidet.__file__}"

N, Cin, Cout, Hout, Wout, Kh, Kw, stride, groups = 1, 4, 8, 12, 12, 3, 3, 1, 1
pad_h = pad_w = (Kh - 1) // 2
Hin = (Hout - 1) * stride + Kh - 2 * pad_h
Win = (Wout - 1) * stride + Kw - 2 * pad_w

rng = np.random.default_rng(42)
X = rng.uniform(-1.0, 1.0, size=(N, Cin, Hin, Win)).astype(np.float32)
W = rng.uniform(-1.0, 1.0, size=(Cout, Cin // groups, Kh, Kw)).astype(np.float32)

x_sym = hidet.symbol(list(X.shape), dtype="float32", device="cuda")
w_sym = hidet.symbol(list(W.shape), dtype="float32", device="cuda")
y_sym = hidet.ops.conv2d(x_sym, w_sym, stride=(stride, stride), padding=(pad_h, pad_w), groups=groups)
graph = hidet.graph.optimize(hidet.trace_from(y_sym, inputs=[x_sym, w_sym]))

x_h = hidet.asarray(X, device="cuda")
w_h = hidet.asarray(W, device="cuda")
out = graph(x_h, w_h).torch().detach().to("cpu", dtype=torch.float64).numpy()

def direct_conv_fp64(X, W, stride, pad_h, pad_w, groups):
    Xp = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    Nn, _, Hp, Wp = Xp.shape
    Co, cpg, Khh, Kww = W.shape
    Ho = (Hp - Khh) // stride + 1
    Wo = (Wp - Kww) // stride + 1
    cog = Co // groups
    out = np.zeros((Nn, Co, Ho, Wo), dtype=np.float64)
    for g in range(groups):
        Xg = Xp[:, g * cpg:(g + 1) * cpg]
        Wg = W[g * cog:(g + 1) * cog]
        acc = np.zeros((Nn, cog, Ho, Wo), dtype=np.float64)
        for kh in range(Khh):
            for kw in range(Kww):
                patch = Xg[:, :, kh:kh + stride * Ho:stride, kw:kw + stride * Wo:stride]
                acc += np.einsum('ncij,oc->noij', patch, Wg[:, :, kh, kw])
        out[:, g * cog:(g + 1) * cog] = acc
    return out

ref = direct_conv_fp64(X.astype(np.float64), W.astype(np.float64), stride, pad_h, pad_w, groups)
scale = direct_conv_fp64(np.abs(X.astype(np.float64)), np.abs(W.astype(np.float64)), stride, pad_h, pad_w, groups)
err = float(np.max(np.abs(out - ref) / np.maximum(scale, 1e-300)))
print(f"functional check: hidet conv2d {X.shape} x {W.shape} -> {out.shape}, "
      f"max_scaled_err={err:.3e} (informational only -- the REAL gate is "
      f"kernelbench.runner's own reference_conv; see STATUS.md)")
assert out.shape == ref.shape, "output shape mismatch"
EOF
echo "== Hidet build check: OK (op-level conv2d compiled + ran end to end) =="
