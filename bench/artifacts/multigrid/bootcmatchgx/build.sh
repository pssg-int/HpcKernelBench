#!/usr/bin/env bash
# Build BootCMatchGX's library + example binaries (bin/example/driverSolve
# is the one this track's adapter.py wraps). See STATUS.md: this build
# FAILED on this machine's login node (link-step race, most likely caused
# by the artifact's own config.mk defaulting to -j$(nproc)=244-way
# parallelism when no -j flag is given) -- this script passes a BOUNDED
# -j8 explicitly, which the failed manual attempt did not.
#
# Idempotent: safe to re-run; GNU Make's own incremental build only
# recompiles changed/missing objects. Exit 0 = built.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/source/BCMGX"
# Recorded build-system patch to config.mk (cray-libsci instead of netlib
# LAPACK / MKL; drop -lnvToolsExt, absent from the CUDA 12.9 SDK and unused
# by the sources). Idempotent: applied only if not already present.
if git -C .. apply --check "$HERE/config_mk.patch" 2>/dev/null; then
    git -C .. apply "$HERE/config_mk.patch"
    echo "applied config_mk.patch"
else
    echo "config_mk.patch already applied (or does not apply cleanly -- check manually)"
fi
# I/O-only patch (NOT kernel/solver code): Vector::print() wrote the exported
# solution with "%g" (6 significant digits), which caps any independently
# recomputed relative residual near 1e-6..1e-5 regardless of how well the
# solver converged; "%.17g" round-trips fp64 exactly. Idempotent as above.
if git -C .. apply --check "$HERE/vector_print_precision.patch" 2>/dev/null; then
    git -C .. apply "$HERE/vector_print_precision.patch"
    echo "applied vector_print_precision.patch"
else
    echo "vector_print_precision.patch already applied (or does not apply cleanly -- check manually)"
fi

NVCC="${NVCC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc}"
CUDA_DIR="${CUDA_DIR:-$CUDA_HOME}"
MPI_ROOT="${MPI_ROOT:-/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3}"
JOBS="${JOBS:-8}"   # bounded, NOT config.mk's own -j$(nproc) default
                     # (244-way on this login node -- see STATUS.md for
                     # why an uncontrolled -j is believed to have caused
                     # the previous BUILD-FAILED attempt's link-step race).

echo "== BootCMatchGX build (multigrid) =="
echo "commit: $(git -C . rev-parse HEAD 2>/dev/null || echo unknown)"
echo "nvcc: $($NVCC --version | tail -1)"

if [ -z "${CRAY_LIBSCI_PREFIX_DIR:-}" ] && [ -z "${KB_BLAS_LIB:-}" ]; then
    if ! command -v module >/dev/null 2>&1 || ! module load cray-libsci 2>/dev/null; then
        echo "NOTE: 'module load cray-libsci' unavailable in this shell (module" >&2
        echo "function only exists in an interactive login shell, not a plain" >&2
        echo "#!/usr/bin/env bash script -- same class of issue toolchain.sh" >&2
        echo "itself documents for 'module load cudatoolkit'). Set" >&2
        echo "CRAY_LIBSCI_PREFIX_DIR manually if this script is run non-" >&2
        echo "interactively, e.g.:" >&2
        echo "  export CRAY_LIBSCI_PREFIX_DIR=/opt/cray/pe/libsci/26.03.0/GNU/12/x86_64" >&2
    fi
fi
# config.mk's LAPACK+CBLAS substitution needs ONE of: CRAY_LIBSCI_PREFIX_DIR
# (Perlmutter, module-provided) or KB_BLAS_LIB (bench/env.sh's non-Cray
# OpenBLAS knob, e.g. zaratan) -- see config.mk's own comment at the
# ifdef/else-ifdef for why both substitute the same way.
if [ -z "${CRAY_LIBSCI_PREFIX_DIR:-}" ] && [ -z "${KB_BLAS_LIB:-}" ]; then
    echo "FATAL: neither CRAY_LIBSCI_PREFIX_DIR (module load cray-libsci) nor" >&2
    echo "KB_BLAS_LIB (bench/env.sh's OpenBLAS knob) is set -- config.mk's" >&2
    echo "LAPACK+CBLAS substitution needs one of the two." >&2
    exit 1
fi

# -lcuda/-lnvidia-ml (driver-API libs) are only shipped as link-time STUBS on
# a standard CUDA-toolkit layout (conda-forge; also NVIDIA's own installers,
# under lib64/stubs) since the real ones come from the GPU driver at runtime,
# absent on this login node. Perlmutter's HPC SDK/module environment already
# puts a real (or stub) libcuda.so on the link path, so this is additive and
# a no-op there. Appended, not prepended, so a real libcuda.so elsewhere on
# LIBRARY_PATH still wins.
for _stubdir in "$CUDA_DIR/lib/stubs" "$CUDA_DIR/lib64/stubs" "$CUDA_DIR/targets/x86_64-linux/lib/stubs"; do
    [ -d "$_stubdir" ] && LIBRARY_PATH="${LIBRARY_PATH:+$LIBRARY_PATH:}$_stubdir"
done
export LIBRARY_PATH

make -j"$JOBS" \
    CUDA_DIR="$CUDA_DIR" \
    MPI_DIR="$MPI_ROOT" \
    MPI_INCLUDE_DIR="$MPI_ROOT/include" \
    MPIRUN="$MPI_ROOT/bin/mpirun" \
    USE_LAPACK=1 USE_MKL=0 CC="${KB_GCC12:-gcc-12}" \
    lib example

echo "-- verifying driverSolve was built --"
[ -x bin/example/driverSolve ] || { echo "MISSING bin/example/driverSolve"; exit 1; }
_ldd_out="$(ldd bin/example/driverSolve || true)"  # ldd exits nonzero when ANY
                                                    # dep is unresolved (expected
                                                    # here for driver libs); don't
                                                    # let `set -e` abort before
                                                    # the filtered check below runs
# kernelbench build-system fix: libcuda.so.1/libnvidia-ml.so.1 are the GPU
# DRIVER's runtime libraries (distinct from the link-time stubs used above)
# -- never present on a GPU-less login node (see bench's own MACHINE FACTS:
# "The login node has NO GPU"), only resolved when this binary actually runs
# on a GPU node (bench/gpu_run.sh) via the driver's own LD_LIBRARY_PATH.
# Their absence here is expected, not a broken build; any OTHER unresolved
# library is still fatal.
_missing="$(grep "not found" <<< "$_ldd_out" | grep -v -E 'libcuda\.so\.1|libnvidia-ml\.so\.1' || true)"  # grep -v
        # legitimately exits 1 when EVERY "not found" line is the expected
        # driver libs (nothing left to output) -- under `set -e -o
        # pipefail` that would otherwise abort the script right here
if [ -n "$_missing" ]; then
    echo "MISSING SHARED LIBS:"; echo "$_missing"; exit 1
fi
if grep -q "not found" <<< "$_ldd_out"; then
    echo "NOTE: driver-runtime libs unresolved on this GPU-less login node (expected):"
    grep "not found" <<< "$_ldd_out"
fi
echo "== BootCMatchGX build: OK =="
