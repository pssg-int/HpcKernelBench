# bench/artifacts/toolchain.sh -- source this at the top of every build.sh.
#
# Pins the CUDA toolkit used for compiling artifacts to the one matching the
# harness's torch build (2.8.0+cu128 -> CUDA 12.x; 12.9 is the newest 12.x in
# the NVIDIA HPC SDK on this machine). Needed since 2026-09-04, when the login
# environment's default module became cudatoolkit/13.2 (HPC SDK 26.5): that
# module points CUDA_HOME, NVHPC_CUDA_HOME, the nvcc on PATH and -- through
# CPATH -- the CUDA headers at 13.2. Consequences observed: torch's JIT
# rebuilt a cached extension with the wrong nvcc; and even with nvcc 12.9,
# 13.2's cudaTypedefs.h (leaked via CPATH) breaks CUTLASS (no unversioned
# PFN_* aliases). CPATH must be RE-POINTED rather than cleared: torch's
# ATen/cuda headers include cusparse.h, which lives only in the SDK's
# math_libs tree. This is exactly what `module load cudatoolkit/12.9`
# exports in an interactive shell; the `module` function is not available in
# a plain `#!/usr/bin/env bash` script, hence this file.
#
# The runner performs the same pin in-process (kernelbench/env.py::
# pin_cuda_toolchain, printed as "[env] toolchain: ...", override with
# KB_CUDA_HOME) so gate runs and runtime-JIT DSLs are covered; shell builds
# are not, so they source this. Idempotent. Override with KB_CUDA_HOME.

# Machine defaults live in bench/env.sh (source it first on a new machine; see
# bench/ENVIRONMENT.md). Everything below honours KB_* overrides and falls back
# to the Perlmutter paths this benchmark was developed on. If neither the
# override nor the Perlmutter prefix exists, the nvcc on PATH decides.
_kb_default=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
if [ -z "${KB_CUDA_HOME:-}" ]; then
  if [ -x "$_kb_default/bin/nvcc" ]; then
    KB_CUDA_HOME="$_kb_default"
  elif command -v nvcc >/dev/null 2>&1; then
    KB_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  else
    echo "toolchain.sh: no CUDA toolkit found (set KB_CUDA_HOME, see bench/ENVIRONMENT.md)" >&2
    KB_CUDA_HOME="$_kb_default"
  fi
fi
unset _kb_default
export CUDA_HOME="$KB_CUDA_HOME"
# Compilers and interpreter: build.sh files use ${CC:-...}/${CXX:-...}/${PY:-...}
# with Perlmutter fallbacks; setting KB_CC/KB_CXX/KB_PY here overrides all of them.
[ -n "${KB_CC:-}" ]  && export CC="$KB_CC"
[ -n "${KB_CXX:-}" ] && export CXX="$KB_CXX"
[ -n "${KB_PY:-}" ]  && export PY="$KB_PY"
export NVHPC_CUDA_HOME="$CUDA_HOME"
case ":$PATH:" in
  *":$CUDA_HOME/bin:"*) ;;
  *) export PATH="$CUDA_HOME/bin:$PATH" ;;
esac
# <root>/<sdk>/cuda/<ver> -> <root>/<sdk>/math_libs/<ver>/include
_kb_ver="$(basename "$CUDA_HOME")"
_kb_sdk="$(dirname "$(dirname "$CUDA_HOME")")"
_kb_ml="$_kb_sdk/math_libs/$_kb_ver/include"
if [ -f "$_kb_ml/cusparse.h" ]; then
  export CPATH="$_kb_ml:$CUDA_HOME/include"
else
  export CPATH="$CUDA_HOME/include"
fi
# Link-time (-lcublas/-lcusolver/-lcurand/-lcusparse/-lcufft live in math_libs,
# NOT under cuda/<ver>/lib64, and nvcc only adds the latter implicitly) and
# run-time search paths, mirroring what `module load cudatoolkit/12.9` exports
# for LD_LIBRARY_PATH. Observed 2026-09-05: BootCMatchGX's link failed with
# "cannot find -lcublas" until LIBRARY_PATH carried the math_libs directory.
# KB_CUDA_MATHLIBS: where cuBLAS/cuSPARSE/... live when they are NOT under
# $CUDA_HOME (NVIDIA HPC SDK layout). A standard CUDA toolkit has them in
# $CUDA_HOME/lib64 + include, in which case this block is a no-op.
if [ -n "${KB_CUDA_MATHLIBS:-}" ]; then _kb_sdk_ml="$KB_CUDA_MATHLIBS"; else _kb_sdk_ml="$_kb_sdk/math_libs/$_kb_ver"; fi
if [ -f "$_kb_sdk_ml/include/cusparse.h" ] && [ "$_kb_sdk_ml/include" != "$_kb_ml" ]; then
  export CPATH="$_kb_sdk_ml/include:$CUDA_HOME/include"
fi
_kb_mll="$_kb_sdk_ml/lib64"
_kb_libs="$CUDA_HOME/lib64"
if [ -d "$_kb_mll" ]; then _kb_libs="$_kb_mll:$_kb_libs"; fi
export LIBRARY_PATH="$_kb_libs${LIBRARY_PATH:+:$LIBRARY_PATH}"
_kb_ld="${LD_LIBRARY_PATH:-}"   # may be unset on a machine without modules (set -u callers)
if [ "${_kb_ld#*$CUDA_HOME/lib64}" = "$_kb_ld" ]; then
  export LD_LIBRARY_PATH="$_kb_libs${_kb_ld:+:$_kb_ld}"
fi
unset _kb_ld
unset _kb_ver _kb_sdk _kb_ml _kb_mll _kb_libs _kb_sdk_ml
# libstdc++ ABI trap for torch CUDA extensions at run time (CXXABI_1.3.15):
# not exported here (LD_PRELOAD must be set before the python process starts);
# see any torch-extension STATUS.md.

# ---------------------------------------------------------------------------
# Derived variables for the per-artifact build.sh files (2026-09-08).
# The scripts read these as ${VAR:-<Perlmutter default>}; exporting them here
# from the KB_* knobs (bench/env.sh) makes one env.sh edit cover every script.
export NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
export REAL_NVCC="${REAL_NVCC:-$NVCC}"
# cuBLAS/cuSPARSE/cuSOLVER/cuRAND/cuFFT root: HPC SDK sibling tree, else the toolkit itself.
if [ -z "${KB_CUDA_MATHLIBS:-}" ]; then
  _kb_sdk_ml="$(dirname "$(dirname "$CUDA_HOME")")/math_libs/$(basename "$CUDA_HOME")"
  if [ -f "$_kb_sdk_ml/include/cusparse.h" ]; then KB_CUDA_MATHLIBS="$_kb_sdk_ml"; else KB_CUDA_MATHLIBS="$CUDA_HOME"; fi
  unset _kb_sdk_ml
fi
export KB_CUDA_MATHLIBS
if [ -d "$KB_CUDA_MATHLIBS/targets/x86_64-linux/lib" ]; then
  export KB_MATHLIB_DIR="${KB_MATHLIB_DIR:-$KB_CUDA_MATHLIBS/targets/x86_64-linux/lib}"
  export CUFFT_PREFIX="${CUFFT_PREFIX:-$KB_CUDA_MATHLIBS/targets/x86_64-linux}"
else
  export KB_MATHLIB_DIR="${KB_MATHLIB_DIR:-$KB_CUDA_MATHLIBS/lib64}"
  export CUFFT_PREFIX="${CUFFT_PREFIX:-$KB_CUDA_MATHLIBS}"
fi
export MATHLIB="${MATHLIB:-$KB_MATHLIB_DIR}"

# MPI (link-only; amgt x2, bootcmatchgx, hicma-x, exageostat, graphset).
if [ -z "${KB_MPI_ROOT:-}" ]; then
  if [ -d /opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3/include ]; then KB_MPI_ROOT=/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3
  elif command -v mpicc >/dev/null 2>&1; then KB_MPI_ROOT="$(dirname "$(dirname "$(command -v mpicc)")")"
  else KB_MPI_ROOT=/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3; fi
fi
export KB_MPI_ROOT
export MPI_ROOT="${MPI_ROOT:-$KB_MPI_ROOT}"
export MPI_INC="${MPI_INC:-$MPI_ROOT/include}"
export MPI_LIB="${MPI_LIB:-$MPI_ROOT/lib}"
# Cray MPICH names its GNU-ABI library libmpi_gnu_<ver>.so; everyone else libmpi.so.
if [ -z "${KB_MPI_LIBNAME:-}" ]; then
  KB_MPI_LIBNAME=mpi
  for _kb_f in "$MPI_LIB"/libmpi_gnu*.so; do
    [ -e "$_kb_f" ] && { KB_MPI_LIBNAME="$(basename "$_kb_f" .so)"; KB_MPI_LIBNAME="${KB_MPI_LIBNAME#lib}"; break; }
  done
  unset _kb_f
fi
export KB_MPI_LIBNAME
# BLAS/LAPACK (cholesky/*, multigrid/bootcmatchgx): a lib dir; KB_BLAS_LIB may name the library file itself.
export KB_BLAS_LIBDIR="${KB_BLAS_LIBDIR:-/opt/cray/pe/libsci/26.03.0/GNU/12/x86_64/lib}"

# gcc/g++ 12 for the artifacts whose code (or nvcc 12.x's host-compiler checks)
# rejects the default compiler. Order: KB_HOST_COMPILER_BIN, then a g++-12 on
# this system, else whatever CXX is (those artifacts may then fail to build).
if [ -z "${KB_GXX12:-}" ]; then
  for _kb_c in "${KB_HOST_COMPILER_BIN:-/opt/cray/pe/gcc-native/12/bin}/g++" /usr/bin/g++-12 "$(command -v g++-12 2>/dev/null || true)"; do
    [ -n "$_kb_c" ] && [ -x "$_kb_c" ] && { KB_GXX12="$_kb_c"; break; }
  done
  [ -n "${KB_GXX12:-}" ] || KB_GXX12="${CXX:-g++}"
  unset _kb_c
fi
export KB_GXX12
if [ -z "${KB_GCC12:-}" ]; then
  _kb_b="$(basename "$KB_GXX12")"
  KB_GCC12="$(dirname "$KB_GXX12")/${_kb_b/g++/gcc}"
  [ -x "$KB_GCC12" ] || KB_GCC12="${CC:-gcc}"
  unset _kb_b
fi
export KB_GCC12

# GPU architecture visible on this machine, for arch flags (KB_SM=90 -> sm_90,
# KB_TORCH_ARCH=9.0). The existing build.sh files still pin sm_80 (A100); see
# bench/ENVIRONMENT.md section 8 before building on another architecture.
if [ -z "${KB_SM:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  _kb_cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  case "$_kb_cc" in
    [0-9]*.[0-9]*) KB_SM="${_kb_cc%.*}${_kb_cc#*.}"; KB_TORCH_ARCH="${KB_TORCH_ARCH:-$_kb_cc}" ;;
  esac
  unset _kb_cc
fi
export KB_SM="${KB_SM:-80}" KB_TORCH_ARCH="${KB_TORCH_ARCH:-8.0}"

# ---------------------------------------------------------------------------
# Artifact checkouts (source/, source_*/) are git-ignored. When a build.sh
# sources this file and its directory has a <name>.provenance but no <name>/,
# re-create the checkout from the recorded remote + commit and re-apply the
# recorded source.patch, so a fresh clone of this repository builds without
# manual steps. Shared checkouts (source -> ../../<track>/<short>/source
# symlinks) are fetched in their owning directory. KB_NO_FETCH=1 disables.
kb_ensure_source() {   # kb_ensure_source <artifact dir>
  local dir="$1" prov name tgt remote commit owner
  for prov in "$dir"/*.provenance; do
    [ -f "$prov" ] || continue
    name="$(basename "$prov" .provenance)"
    tgt="$dir/$name"
    if [ -L "$tgt" ]; then
      owner="$(cd "$dir" && cd "$(dirname "$(readlink "$tgt")")" 2>/dev/null && pwd)" || continue
      [ "$owner" != "$dir" ] && kb_ensure_source "$owner"
      continue
    fi
    [ -e "$tgt" ] && continue
    remote="$(sed -n 's/^remote=//p' "$prov" | head -1 | sed 's/[[:space:]].*//')"
    commit="$(sed -n 's/^commit=//p' "$prov" | head -1 | sed 's/[[:space:]].*//')"
    if [ -z "$remote" ] || ! grep -q '^reclone=git clone' "$prov"; then
      echo "toolchain.sh: $tgt is missing and $prov has no git reclone recipe (see its refetch=/note= lines; build.sh may fetch it itself)" >&2
      continue
    fi
    echo "toolchain.sh: fetching $tgt from $remote @ ${commit:-HEAD}" >&2
    git clone --quiet "$remote" "$tgt" || { echo "toolchain.sh: clone of $remote failed" >&2; return 1; }
    if [ -n "$commit" ]; then
      git -C "$tgt" checkout --quiet "$commit" || { echo "toolchain.sh: checkout $commit failed in $tgt" >&2; return 1; }
    fi
    if [ "$name" = source ] && [ -s "$dir/source.patch" ]; then
      echo "toolchain.sh: applying $dir/source.patch" >&2
      git -C "$tgt" apply "$dir/source.patch" || { echo "toolchain.sh: source.patch did not apply cleanly in $tgt" >&2; return 1; }
    fi
  done
  return 0
}
if [ "${KB_NO_FETCH:-0}" != 1 ] && [ -n "${BASH_SOURCE[1]:-}" ]; then
  _kb_caller="$(cd "$(dirname "${BASH_SOURCE[1]}")" 2>/dev/null && pwd)"
  if [ -n "$_kb_caller" ]; then
    for _kb_p in "$_kb_caller"/*.provenance; do
      [ -f "$_kb_p" ] && { kb_ensure_source "$_kb_caller"; break; }
    done
    unset _kb_p
  fi
  unset _kb_caller
fi
