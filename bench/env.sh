# bench/env.sh -- ONE file to edit on a new machine, then `source bench/env.sh`.
#
# Everything the harness and the artifact build scripts need to know about the
# machine is a KB_* variable below. Defaults (when a variable is left empty)
# are the Perlmutter paths this benchmark was developed on; every build.sh
# falls back to the same defaults, so on Perlmutter nothing needs setting.
# Full explanation of each knob and of the known traps: bench/ENVIRONMENT.md.
#
# Usage:
#   source bench/env.sh              # exports KB_*, then sources artifacts/toolchain.sh
#   $PY bench/check_env.py           # verify before building anything
#   cd bench && ./smoke_all.sh       # CPU-only end-to-end check (no GPU needed)
#   bench/artifacts/<kernel>/<short>/build.sh   # per artifact

# ---- 1. Python interpreter with numpy/scipy/pyyaml/torch (see requirements.txt)
export KB_PY="${KB_PY:-/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env/bin/python}"

# ---- 2. CUDA toolkit whose MAJOR matches torch.version.cuda (torch 2.8.0+cu128 -> 12.x)
#      Standard toolkit layout: cuBLAS/cuSPARSE/... live under $KB_CUDA_HOME.
#      NVIDIA HPC SDK layout (Perlmutter): they live in a sibling math_libs tree.
export KB_CUDA_HOME="${KB_CUDA_HOME:-/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env}"
export KB_CUDA_MATHLIBS="${KB_CUDA_MATHLIBS:-$KB_CUDA_HOME}"   # standard toolkit layout (conda-forge cuda-toolkit 12.8)

# ---- 3. Host compilers. nvcc 12.9 accepts gcc <= 14; a few artifacts (marked
#      "gcc12-host" in ENVIRONMENT.md) need gcc/g++ 12 as nvcc's -ccbin.
export KB_CC="${KB_CC:-/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env/bin/gcc}"
export KB_CXX="${KB_CXX:-/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env/bin/g++}"
export KB_HOST_COMPILER_BIN="${KB_HOST_COMPILER_BIN:-/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-gcc12/bin}"   # dir holding gcc/g++ 12

# ---- 4. Optional: only some artifacts need these (table in ENVIRONMENT.md)
export KB_MPI_ROOT="${KB_MPI_ROOT:-/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env}"           # MPI headers/libs (link-only for single-GPU runs)
export KB_BLAS_LIBDIR="${KB_BLAS_LIBDIR:-/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env/lib}"
export KB_BLAS_LIB="${KB_BLAS_LIB:-$KB_BLAS_LIBDIR/libopenblas.so}"       # OpenBLAS (conda-forge), not libsci
export KB_BLAS_INCDIR="${KB_BLAS_INCDIR:-$KB_BLAS_LIBDIR/../include}" # a BLAS/LAPACK (cholesky, bootcmatchgx)

# ---- 5. Data caches (downloaded lazily on first use; safe to point at scratch)
export SHARED_KB_DATA="${SHARED_KB_DATA:-/scratch/zt1/project/bhatele-lab/shared/kernel-bench/data}"  # group-shared dataset caches
export KERNELBENCH_MATRIX_CACHE="${KERNELBENCH_MATRIX_CACHE:-$SHARED_KB_DATA/matrices}"   # SuiteSparse .mtx cache (shared)
export KERNELBENCH_TENSOR_CACHE="${KERNELBENCH_TENSOR_CACHE:-$SHARED_KB_DATA/tensors}"     # sparse-tensor (.tns) cache
export KERNELBENCH_SEQALIGN_DATA="${KERNELBENCH_SEQALIGN_DATA:-$SHARED_KB_DATA/seqalign}"  # sequence-alignment data
export KERNELBENCH_ANNSEARCH_DATA="${KERNELBENCH_ANNSEARCH_DATA:-$SHARED_KB_DATA/annsearch}" # SIFT1M / bunny etc.
export KERNELBENCH_AUTOMATA_DATA="${KERNELBENCH_AUTOMATA_DATA:-$SHARED_KB_DATA/automata}"  # ANMLZoo automata
# graphs (Planetoid, bench/graphs/), annsearch-data/, seqalign-data/, automata-data/
# follow the same convention; see the DATA_DIR variables in each domain module.
# SuiteSparse/dataset downloads go over https; RHEL8's system Python trust store
# doesn't cover the herokuapp mirror's chain, so urllib fails with SSL
# CERTIFICATE_VERIFY_FAILED. Point OpenSSL at certifi's CA bundle (shipped in the
# conda env) so kernelbench.matrices' urlretrieve verifies. Only login nodes have
# outbound internet here -- pre-fetch real matrices there; compute nodes read the
# populated $KERNELBENCH_MATRIX_CACHE.
if [ -z "${SSL_CERT_FILE:-}" ]; then
    _kb_ca="$("${KB_PY:-python3}" -c 'import certifi;print(certifi.where())' 2>/dev/null)"
    [ -n "$_kb_ca" ] && export SSL_CERT_FILE="$_kb_ca" REQUESTS_CA_BUNDLE="$_kb_ca"
    unset _kb_ca
fi

# ---- 5b. Thread oversubscription guard (shared login node). numpy/scipy/torch
#      default their BLAS + OpenMP pools to the core count (128 here); a handful
#      of concurrent python processes then blows past this node's per-user
#      RLIMIT_NPROC (256 procs+threads), and fork() starts failing with
#      "Resource temporarily unavailable" -- which surfaces as spurious numpy
#      import errors ("CPU dispatcher tracer already initialized" / "import from
#      source directory") and OpenBLAS segfaults. Functional gates do not care
#      how many BLAS threads the CPU reference uses, so cap the pools small.
#      Override for a dedicated timing run (nothing else contends there).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

# ---- 6. Run-time ABI trap for torch CUDA extensions (Perlmutter: the venv's
#      python drags in an old libstdc++ before torch's extensions load).
#      Set to the system libstdc++ that provides CXXABI_1.3.15, or leave empty
#      on machines where `python -c "import torch"` followed by importing a
#      compiled extension works without it (check_env.py tells you).
export KB_LD_PRELOAD="${KB_LD_PRELOAD:-/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env/lib/libstdc++.so.6}"   # gcc 13 libstdc++ (CXXABI_1.3.15); system one is gcc 8

# ---- derived: what the build scripts actually read
export PY="$KB_PY" CC="$KB_CC" CXX="$KB_CXX"
_kb_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_kb_here/artifacts/toolchain.sh"
unset _kb_here
if [ -n "$KB_LD_PRELOAD" ] && [ -f "$KB_LD_PRELOAD" ]; then
  export LD_PRELOAD="$KB_LD_PRELOAD"
fi
echo "kernelbench env: PY=$PY CUDA_HOME=$CUDA_HOME CC=$CC LD_PRELOAD=${LD_PRELOAD:-<none>}"
