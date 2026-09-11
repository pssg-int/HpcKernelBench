#!/usr/bin/env bash
# Build Hexcute's kernel-generating compiler (source/ = hexcute/hidet, a fork
# of hidet-org/hidet pinned at the commit hexcute-bench's .gitmodules points
# at), plus a tiny nvcc PATH shim (see bin/nvcc's own header comment for why
# it's needed instead of a hidet-exposed knob).
#
# Dependency isolation (2026-09-05): this used to `pip uninstall hidet` then
# `pip install --no-build-isolation -e source` into the shared plexus_env
# venv -- both forbidden now (plexus_env is used by the user for other work;
# do not uninstall from it or install into it) AND the actual root cause of
# a real collision: bench/artifacts/convolution/hidet/ also `pip install`s a
# package named "hidet" into the same venv, and whichever install ran last
# won process-wide for every `import hidet` (see STATUS.md's "Dependency
# isolation (2026-09-05)" section). Fix: never touch plexus_env at all.
# Reading source/setup.py's `CustomBuildCommand` (the thing `pip install -e`
# used to invoke) shows the ONLY real build step is a plain CMake build of
# two small C++ libraries (libhidet.so, libhidet_runtime.so -- host-side
# runtime glue; Hexcute's actual GEMM kernels are JIT-compiled later, at
# graph-optimize time, via bin/nvcc) followed by copying them into
# source/python/hidet/lib/. That copy step is ALSO how hidet resolves its
# own runtime library at import time with no install step at all: read
# source/python/hidet/libinfo.py's get_library_search_dirs() -- it looks for
# libhidet*.so under paths RELATIVE TO THE PACKAGE'S OWN LOCATION on disk
# (`./lib`, `../../build/lib`, ...), never via any installed-package metadata.
# So this script now just runs that CMake build directly and copies the .so
# files exactly where CustomBuildCommand would have -- entirely under
# artifacts/gemm/hexcute/, no `pip install`/`pip uninstall` of any kind.
# adapter.py then puts `source/python` first on sys.path so `import hidet`
# resolves this fork's package straight off disk (no site-packages entry at
# all), analogous to convolution/hidet's own `pylibs/` sys.path trick.
#
# One remaining wrinkle: source/python/hidet/version.py does
# `importlib.metadata.version("hidet")`, which (unlike libinfo.py) DOES
# search installed-package metadata across sys.path -- with no pip install,
# that would look past source/python (nothing there) to whatever "hidet"
# metadata plexus_env happens to have (today: convolution/hidet's own
# install), returning a foreign, misleading version string, or raising
# PackageNotFoundError outright on a venv with no hidet metadata at all
# (breaking `import hidet` entirely, since __init__.py does
# `from .version import __version__` unconditionally). Fix: synthesize a
# minimal, self-contained hidet-0.0.0.dist-info/METADATA directly under
# source/python/ (step 3 below) -- just enough for importlib.metadata to
# resolve "hidet" to THIS package without touching any real install location.
#
# Idempotent: safe to re-run; exit 0 = built and importable straight from
# source/, independent of plexus_env's own site-packages.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"
HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"   # nvcc 12.9 rejects the
    # system default g++-14 (SUSE) on <bits/alloc_traits.h> with
    # "identifier __has_construct is undefined" -- a host-compiler/libstdc++
    # version mismatch, not a Hexcute/hidet problem. Same fix already applied
    # in ../../spmm/inferfast/build.sh.
REAL_NVCC="${REAL_NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"

echo "== Hexcute (hidet fork) build =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($REAL_NVCC --version | tail -1)"
echo "host compiler: $($HOST_COMPILER --version | head -1)"
echo "python: $($PY --version), $($PY -c 'import torch; print("torch", torch.__version__)')"

echo "-- 1/4: bin/nvcc PATH shim (host-compiler override; see bin/nvcc header) --"
mkdir -p bin
cat > bin/nvcc <<EOF
#!/usr/bin/env bash
# PATH shim: hidet's own JIT backend (python/hidet/backend/build.py's NVCC
# class) resolves "nvcc" via shutil.which() and invokes it with no way to
# inject a host-compiler override -- there is no CXX/CUDAHOSTCXX/-ccbin knob
# exposed anywhere in hidet's Python build path. Putting this directory first
# on PATH (adapter.py / this script do this) makes shutil.which('nvcc')
# resolve here instead of the real toolkit nvcc, so every JIT kernel
# compilation picks up -ccbin g++-12 without any hidet source edit. This is a
# build-system/host-compiler-selection fix (ARTIFACT_GUIDE rule 3 explicitly
# allows this class of patch), not a change to hidet's compiler/kernel logic
# -- this script is pure passthrough plus one flag.
#
# Root cause: nvcc 12.9 (-ccbin g++-14, SUSE default) fails on
# <bits/alloc_traits.h> with "identifier __has_construct is undefined" --
# the same host-compiler/libstdc++ mismatch already diagnosed and fixed the
# same way in ../../spmm/inferfast/build.sh.
# NOTE: these two lines are written LITERALLY into bin/nvcc (the heredoc
# delimiter below is unquoted only for the comment interpolation above this
# point; the $ signs here are escaped) -- the generated wrapper must stay
# machine-neutral (same Perlmutter-default fallback chain as this script's
# own REAL_NVCC/HOST_COMPILER assignment above), never bake in whichever
# path THIS build happened to resolve. At actual invocation time (this
# script's own verification step below, or a later hidet JIT subprocess
# during a gate run) the wrapper reads REAL_NVCC/KB_GXX12 straight from its
# environment -- already exported by artifacts/toolchain.sh, sourced by
# both this build.sh and bench/env.sh -- falling back to the hardcoded
# Perlmutter path only if genuinely unset (e.g. no env.sh sourced at all).
REAL_NVCC="\${REAL_NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"
HOST_COMPILER="\${HOST_COMPILER:-\${KB_GXX12:-/usr/bin/g++-12}}"
exec "\$REAL_NVCC" -ccbin "\$HOST_COMPILER" "\$@"
EOF
chmod +x bin/nvcc
bin/nvcc --version >/dev/null

echo "-- 2/4: CMake build of libhidet.so / libhidet_runtime.so (source/build/) --"
# Exactly what source/setup.py's CustomBuildCommand.run() does (read
# directly from setup.py, reproduced here so no `pip install` of any kind
# touches plexus_env): `cmake ..` / `make -j8` on a handful of small C++
# files (src/hidet/runtime/*.cpp -- host-side runtime glue; hidet's own CUDA
# kernels are JIT-compiled later, at graph-optimize time, via bin/nvcc
# above), then copy the two resulting .so files into python/hidet/lib/ --
# where source/python/hidet/libinfo.py's get_library_search_dirs() looks
# for them FIRST (relative to the package's own location on disk, no
# installed-package metadata involved at all).
SRC_DIR="$HERE/source"
BUILD_DIR="$SRC_DIR/build"
LIB_DEST="$SRC_DIR/python/hidet/lib"
mkdir -p "$BUILD_DIR"
CC="${KB_GCC12:-/usr/bin/gcc-12}" CXX="$HOST_COMPILER" cmake -S "$SRC_DIR" -B "$BUILD_DIR" >/dev/null
make -C "$BUILD_DIR" -j8
mkdir -p "$LIB_DEST"
for lib in libhidet_runtime.so libhidet.so; do
    src="$BUILD_DIR/lib/$lib"
    [[ -f "$src" ]] || { echo "[hexcute] ERROR: expected $src not found after cmake build" >&2; exit 1; }
    cp -f "$src" "$LIB_DEST/$lib"
done
echo "[hexcute] copied libhidet{,_runtime}.so -> $LIB_DEST"

echo "-- 3/4: synthesizing hidet-0.0.0.dist-info (see build.sh header comment) --"
# source/python/hidet/version.py does importlib.metadata.version("hidet"),
# which -- unlike libinfo.py's on-disk relative-path lookup -- searches
# installed-package METADATA across sys.path. With no pip install at all,
# this makes __init__.py's `from .version import __version__` depend on
# whatever "hidet" metadata plexus_env happens to hold (or raise
# PackageNotFoundError on a venv with none at all, breaking `import hidet`
# outright). A minimal, self-authored dist-info directly under
# source/python/ -- found by importlib.metadata BEFORE anything later on
# sys.path once adapter.py puts source/python first -- makes this fork
# resolve its own version with zero dependency on the ambient venv.
DIST_INFO="$SRC_DIR/python/hidet-0.0.0.dist-info"
mkdir -p "$DIST_INFO"
cat > "$DIST_INFO/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: hidet
Version: 0.0.0
Summary: Hexcute fork of hidet (local build, bench/artifacts/gemm/hexcute; not pip-installed anywhere)
EOF

echo "-- 4/4: pylibs/ -- hidet's undeclared pure-Python runtime deps --"
# hidet imports several packages (tomlkit, click, lark, tabulate, tqdm,
# nvtx, psutil, gitpython/git, importlib_metadata, cuda-python's
# cuda-bindings) that are neither vendored in source/ nor guaranteed present
# in the shared venv (the reference machine's plexus_env happened to already
# have them; this machine's kb-env did not, surfacing as e.g.
# `ModuleNotFoundError: No module named 'tomlkit'` on `import hidet`).
# hidet/runtime/storage.py also does an UNCONDITIONAL `import hidet.hip`
# (AMD ROCm support, irrelevant to this artifact's CUDA-only kernel, but
# not gated behind a try/except in this codebase) -> `from hip import hip`
# -- the "hip" top-level module is published on PyPI as `hip-python`
# (AMD's official bindings); pure ctypes-style declarations, no ROCm
# runtime library needed just to import it. Installed --no-deps into an
# artifact-local pylibs/ target (same isolation rationale as tilus's/
# qfactory's vendor/ -- never touch the shared venv); adapter.py's
# _ensure_env() already appends pylibs/ to sys.path (see its own
# docstring). Idempotent: skipped once tomlkit AND hip are both importable
# from pylibs/.
PYLIBS="$HERE/pylibs"
if ! PYTHONPATH="$PYLIBS" "$PY" -c "import tomlkit, hip.hip" >/dev/null 2>&1; then
    mkdir -p "$PYLIBS"
    "$PY" -m pip install --target="$PYLIBS" --no-deps \
        tomlkit click lark tabulate tqdm nvtx psutil gitpython importlib_metadata hip-python \
        "cuda-python==${CUDA_PYTHON_VERSION:-12.8.0}" "cuda-bindings==${CUDA_PYTHON_VERSION:-12.8.0}"
else
    echo "[hexcute] pylibs/ already has hidet's runtime deps -- skipping"
fi

echo "-- verifying: import hidet (straight from source/python/, no install anywhere) resolves this fork's hexcute_matmul + get_compiled_task --"
PYTHONPATH="$SRC_DIR/python:$PYLIBS" "$PY" - <<PYEOF
import sys
sys.path.insert(0, "$SRC_DIR/python")
sys.path.append("$PYLIBS")
import hidet
assert hidet.__file__.startswith("$SRC_DIR"), f"hidet resolved to {hidet.__file__!r}, expected under $SRC_DIR"
assert hasattr(hidet.option, "hexcute_matmul"), "hexcute_matmul option missing -- wrong hidet install"
assert hasattr(hidet.graph.FlowGraph, "get_compiled_task"), "FlowGraph.get_compiled_task missing -- wrong hidet install (upstream, not this fork)"
print("hidet:", hidet.__file__, "version:", hidet.__version__)
print("hexcute_matmul default:", hidet.option.get_hexcute_matmul())
PYEOF

echo "== Hexcute build: OK (built under $HERE, plexus_env untouched) =="
