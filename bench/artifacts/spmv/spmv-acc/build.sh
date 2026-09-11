#!/usr/bin/env bash
# Build spmv-acc's kernel library (src/acc/) + bridge.cpp (this directory)
# into bridge.so, using hipcc (HIP_PLATFORM=nvidia -> nvcc under the hood)
# to compile the artifact's own KERNEL_STRATEGY_ADAPTIVE strategy's source
# files directly, bypassing the project's full CMake+HIP-CMake-module
# build (which needs a complete find_package(HIP) tree plus the unrelated
# `clipp` dependency for its CLI target -- neither needed for the library
# entry point this integration wraps, sparse_csr_spmv() in
# src/acc/api/spmv.h). See bridge.cpp and STATUS.md for what is wrapped.
#
# `module load` does not work inside a plain `#!/usr/bin/env bash` script
# (see bench/artifacts/toolchain.sh's own note), so a working HIP module's
# env vars are replicated explicitly below via KB_* knobs, each overridable
# and falling back to this repo's earlier (Perlmutter) HIP-module paths --
# same pattern every other build.sh in this repo uses (rule 8).
#
# Status: BUILDS on zaratan (module hip/5.6.1/gcc/11.3.0/nompi/cuda/12.3.0/
# zen2) -- see STATUS.md's "Reproduction on zaratan" entry. Was BUILD-FAILED
# on Perlmutter (only hip/5.5.1 there, incompatible with every CUDA module
# on that machine); this script's KB_* defaults still point at that
# Perlmutter module so re-running there reproduces the same failure for
# anyone auditing it, while a KB_SPMVACC_HIP_PATH/KB_SPMVACC_CUDA_PATH
# override (as set here for zaratan) picks up a working toolchain.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/source/src/acc"
COMPAT="$HERE/compat"

echo "=== spmv-acc build (hipcc, HIP_PLATFORM=nvidia) ==="

# Probe a short list of known HIP-on-CUDA install prefixes (same
# probe-a-list-of-candidates idiom bench/artifacts/toolchain.sh uses for
# CUDA_HOME) -- KB_SPMVACC_HIP_PATH overrides; the Perlmutter path (this
# repo's original reference machine, where this artifact was BUILD-FAILED)
# is kept as a candidate so the failure still reproduces there unchanged;
# the zaratan cvmfs path is what `module load
# hip/5.6.1/gcc/11.3.0/nompi/cuda/12.3.0/zen2` resolves to on this machine
# (module load itself doesn't work in a plain bash script -- see
# bench/artifacts/toolchain.sh's own note -- hence probing its install
# prefix directly instead).
for _cand in \
    "${KB_SPMVACC_HIP_PATH:-}" \
    "/global/common/software/nersc/pe/rocm/5.5.1" \
    "/cvmfs/hpcsw.umd.edu/spack-software/2023.11.20/linux-rhel8-zen2/gcc-11.3.0/hip-5.6.1-dirdplpzwq7rmslexboahlhwkb52xacr" \
    ; do
    if [[ -n "$_cand" && -x "$_cand/bin/hipcc" ]]; then HIP_PATH="$_cand"; break; fi
done
export HIP_PATH="${HIP_PATH:-/global/common/software/nersc/pe/rocm/5.5.1}"
export HIP_PLATFORM=nvidia
export HIP_COMPILER=nvcc
export PATH="$HIP_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$HIP_PATH/lib:${LD_LIBRARY_PATH:-}"

if ! command -v hipcc >/dev/null 2>&1; then
    echo "FAIL: hipcc not found even after pointing PATH at $HIP_PATH/bin" >&2
    echo "      (set KB_SPMVACC_HIP_PATH to a working HIP-on-CUDA module's" >&2
    echo "      install prefix -- module load itself doesn't work in a" >&2
    echo "      plain bash script, hence this explicit env-var replication)." >&2
    exit 1
fi
echo "hipcc: $(command -v hipcc)"

# CUDA toolkit this HIP install was built against/for -- resolved BEFORE
# the first hipcc invocation (including --version): hipcc.pl shells out to
# nvcc at CUDA_PATH/bin (default /usr/local/cuda if unset, which doesn't
# exist here) for every invocation, `--version` included. hip/5.6.1 on
# zaratan bundles its own paired cuda/12.3.0 (a version HIP 5.6.1's vendor
# shim headers are known-compatible with, unlike this repo's usual
# CUDA 12.9 pin -- see STATUS.md for the CUDA-Graph-API/cudaDeviceProp
# mismatch hit against 12.9 on Perlmutter with the older HIP 5.5.1).
for _cand in \
    "${KB_SPMVACC_CUDA_PATH:-}" \
    "/cvmfs/hpcsw.umd.edu/spack-software/2023.11.20/linux-rhel8-zen2/gcc-11.3.0/cuda-12.3.0-i4loalpvrm6df72bxd4p63emo4zm35rm" \
    ; do
    if [[ -n "$_cand" && -x "$_cand/bin/nvcc" ]]; then CUDA_PATH="$_cand"; break; fi
done
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}}"
export CPATH="$CUDA_PATH/include"   # re-point (this script does not source ../../toolchain.sh: hipcc build, not a bare nvcc one)
export PATH="$CUDA_PATH/bin:$PATH"
hipcc --version | head -3
# host compiler paired with this HIP module (its own gcc 11, not this
# repo's usual conda-forge gcc 13 -- hipcc invokes it as -ccbin).
for _cand in \
    "${KB_SPMVACC_GXX:-}" \
    "/cvmfs/hpcsw.umd.edu/spack-software/2023.11.20/linux-rhel8-x86_64/gcc-rh8-8.5.0/gcc-11.3.0-oedkmii7vhd6rbnqm6xufmg7d3jx4w6l/bin/g++" \
    ; do
    if [[ -n "$_cand" && -x "$_cand" ]]; then GXX_PAIRED="$_cand"; break; fi
done
if [[ -n "${GXX_PAIRED:-}" ]]; then
    HIPCC_EXTRA="-ccbin $GXX_PAIRED"
else
    HIPCC_EXTRA=""
fi
unset _cand

# The artifact's own strategy_picker.cpp (compiled with
# KERNEL_STRATEGY_ADAPTIVE, compat/building_config.h) dispatches to
# adaptive_sparse_spmv (hip-adaptive/adaptive.cpp), which itself calls into
# hip-flat/hip-line/hip-line-enhance/hip-vector-row (confirmed by reading
# adaptive.cpp's call sites); hip-thread-row is included for completeness
# per the original integration brief even though adaptive.cpp's own call
# into it is commented out. Every file below is the artifact's own,
# unmodified.
SOURCES=(
    "$SRC/api/spmv_imp.cpp"
    "$SRC/strategy_picker.cpp"
    "$SRC/hip-adaptive/adaptive.cpp"
    "$SRC/hip-flat/flat.cpp"
    "$SRC/hip-line/line_strategy.cpp"
    "$SRC/hip-line-enhance/line_enhance_spmv.cpp"
    "$SRC/hip-vector-row/vector_row.cpp"
    "$SRC/hip-thread-row/thread_row.cpp"
    "$SRC/hip-thread-row/native_thread_row.cpp"
    "$HERE/bridge.cpp"
)

echo "--- compiling bridge.so ---"
# -rpath $CUDA_PATH/lib64: bridge.so is built here (login node, this HIP
# module's CUDA 12.3 loaded explicitly above) but dlopen'd later from the
# harness's own Python process, whose default LD_LIBRARY_PATH points at
# this repo's usual CUDA 12.8 conda toolkit, not this module's CUDA 12.3 --
# rpath it directly so libcudart.so.12 always resolves regardless of the
# caller's environment (libcuda.so.1, the driver's own runtime lib, is
# expected to already be on the GPU node's default search path, same as
# every other CUDA bridge in this repo).
hipcc -std=c++14 -O3 -fPIC -shared $HIPCC_EXTRA \
    -I "$COMPAT" -I "$SRC" \
    "${SOURCES[@]}" \
    -Xlinker -rpath -Xlinker "$CUDA_PATH/lib64" \
    -o "$HERE/bridge.so"

echo "Built: $HERE/bridge.so"
echo "=== done ==="
