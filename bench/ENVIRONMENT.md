# Environment: what this benchmark needs and how to set it up on a new machine

Single source of truth for machine requirements. `bench/env.sh` is the only
file to edit on a new machine; `bench/check_env.py` tells you what is missing;
every `build.sh` and the runner read the same `KB_*` variables. Per-artifact
details (exact commit, patches, gate numbers) stay in each `STATUS.md`.

## 1. Requirements at a glance

| Layer | Requirement | Reference environment (Perlmutter, 2026-09) |
|---|---|---|
| Python | 3.10+ with `numpy`, `scipy`, `PyYAML` (`requirements.txt`) | 3.11.7; numpy 2.3.3, scipy 1.16.2 |
| PyTorch | 2.8.x with CUDA support; its CUDA **major** decides the toolkit below | `torch==2.8.0+cu128` |
| CUDA toolkit | same major as torch (12.x for cu128), incl. cuBLAS/cuSPARSE/cuSOLVER/cuRAND/cuFFT headers+libs | NVIDIA HPC SDK 25.5: `cuda/12.9` + `math_libs/12.9` |
| GPU | NVIDIA; the build scripts pin sm_80 (A100), see §8 for other architectures; 2 artifacts need sm_90a (§6) | A100 80GB PCIe, driver 580.159, sm_80 |
| Host compiler | gcc/g++ 12–14 (nvcc 12.x rejects ≥15); **gcc/g++ 12 additionally required** (nvcc `-ccbin` or the artifact's own build) by 14 artifacts | gcc-native 14.3.0; gcc 12.3.0 for those 13 |
| Build tools | cmake ≥ 3.26, GNU make, git, ninja (torch JIT extensions) | cmake 3.28.3, ninja 1.10 |
| Optional | MPI headers+libs (5 artifacts, link-only), a BLAS/LAPACK (3), hipcc (1), `gh` CLI (paper-pipeline scripts only) | Cray MPICH 9.1.0, cray-libsci 26.03, hip 5.5.1 (broken vs CUDA 12.9) |
| Network | first use downloads SuiteSparse matrices, Planetoid graphs, SIFT1M, ANMLZoo, NCBI sequences; artifact clones via `git`; two artifacts fetch from Zenodo | cached under `bench/{matrices,graphs,...}` and `kernel-papers/annsearch-data/` |
| Disk | repo ≈ 20 MB; artifact clones + build products ≈ 6 GB; data caches ≈ 1.2 GB | |

`smoke_all.sh` (CPU-only, every implemented kernel) needs only the Python
layer: it runs on any machine, GPU or not.

## 2. Setting up a new machine

```bash
git clone <this repo> kernel-papers && cd kernel-papers/bench

# 1. Python: any venv; torch first (pick the wheel index matching your CUDA major)
python3 -m venv ../kb-env && . ../kb-env/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
pip install -r requirements.txt
#    exact reference set, if you want bit-for-bit the same versions:
#    pip install -r env/reference-freeze-perlmutter-2026-09-06.txt   (contains unrelated GNN packages too)

# 2. Point the harness at your toolchain: edit the KB_* lines in env.sh, then
$EDITOR env.sh
source env.sh                      # exports KB_*, PY/CC/CXX, CUDA_HOME/CPATH/LIBRARY_PATH, LD_PRELOAD

# 3. Verify
$PY check_env.py                   # OK / WARN / MISSING per requirement; lists sm_90+-only artifacts

# 4. CPU end-to-end check (minutes, no GPU)
./smoke_all.sh                     # must end with "kernels green: 35 ; failing: 0"

# 5. (optional) pre-fetch every git-ignored artifact checkout from its recorded provenance
artifacts/fetch_sources.sh         # ~6 GB; otherwise each build.sh fetches its own on first run

# 6. Build the artifacts you want, one directory at a time
artifacts/spmm/rode/build.sh       # idempotent; exit 0 = built; each dir's STATUS.md says what to expect
$PY -m kernelbench.runner --kernel spmm --list                      # discovered adapters + their availability
$PY -m kernelbench.runner --kernel spmm --variant spmm-gpu-kernel-f32 --smoke --impl rode-spmm-f32   # functional gate
```

Artifact checkouts (`source/`, `source_*/`) are git-ignored. Each
`artifacts/<k>/<short>/source.provenance` records the remote, the exact commit
and a one-line re-clone command; `source.patch` (when present) is our recorded
diff to tracked files (verified 2026-09-08: all 28 patches equal the live
diffs). When a `build.sh` runs, `toolchain.sh` re-creates a missing checkout
from those two files (clone, checkout, `git apply`; shared checkouts are
fetched in their owning directory), so a fresh clone of this repository needs
no manual fetching. `artifacts/fetch_sources.sh` does it for every directory
up front and reports what it could not fetch. One artifact (spmm/mp-spmm) comes
from Zenodo and is fetched by its own `build.sh`; third-party pins (CUTLASS,
release tarballs) are likewise fetched by the individual `build.sh`.
`KB_NO_FETCH=1` disables the automatic fetch on an offline machine.

A coding agent that has to redo all of this on a new machine gets its
instructions from `bench/REPRODUCE.md`.

## 3. The `KB_*` variables (all in `env.sh`)

| Variable | Meaning | Perlmutter default |
|---|---|---|
| `KB_PY` | interpreter with the packages above; exported as `PY` | `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python` |
| `KB_CUDA_HOME` | toolkit root (`bin/nvcc`, `include/`, `lib64/`); exported as `CUDA_HOME`, prepended to `PATH` | `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9` |
| `KB_CUDA_MATHLIBS` | where cuBLAS/cuSPARSE/… live when NOT under `CUDA_HOME` (HPC SDK layout); ignored for a standard toolkit | `…/25.5/math_libs/12.9` |
| `KB_CC`, `KB_CXX` | host compilers; exported as `CC`/`CXX`, used by nvcc `-ccbin` and CMake | gcc-native 14 |
| `KB_HOST_COMPILER_BIN` | directory with gcc/g++ **12** for the artifacts that need it | `/opt/cray/pe/gcc-native/12/bin` |
| `KB_MPI_ROOT` | MPI install root (headers + libs); linked, never launched — Cray MPICH has no `mpirun`, single-rank binaries run as MPI singletons | `/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3` |
| `KB_BLAS_LIBDIR` | a BLAS/LAPACK library directory | cray-libsci 26.03 GNU/12 |
| `KB_LD_PRELOAD` | libstdc++ to preload for torch CUDA extensions (§5); empty to disable | `/usr/lib64/libstdc++.so.6` |
| `KERNELBENCH_MATRIX_CACHE` | SuiteSparse cache dir | `bench/matrices/` |
| `KB_BLAS_LIB`, `KB_BLAS_INCDIR` | optional: the BLAS/LAPACK library *file* and its include dir when they are not `$KB_BLAS_LIBDIR/libsci_gnu.so` and `../include` (cholesky/*) | Cray LibSci layout |
| `KB_NO_FETCH` | `1` disables the automatic re-clone of missing checkouts | unset |
| `KB_SM`, `KB_TORCH_ARCH` | GPU architecture for arch flags (`90` / `9.0`); detected from `nvidia-smi` when unset | `80` / `8.0` |
| *derived by `toolchain.sh`* | `NVCC`, `REAL_NVCC`, `MATHLIB`, `CUFFT_PREFIX`, `KB_CUDA_MATHLIBS`, `MPI_ROOT`/`MPI_INC`/`MPI_LIB`, `KB_MPI_LIBNAME` (`mpi_gnu_123` on Cray, `mpi` elsewhere), `KB_GXX12`/`KB_GCC12` (the gcc 12 pair, from `KB_HOST_COMPILER_BIN`, else `g++-12`). Every `build.sh` reads these as `${VAR:-<Perlmutter default>}`: set the `KB_*` knob, not the derived name | |

The runner repeats the CUDA pin in-process at start-up
(`kernelbench/env.py::pin_cuda_toolchain`, printed as `[env] toolchain: …`),
so gate runs and runtime-JIT DSLs (tilus, qfactory, hidet) are covered even if
`env.sh` was not sourced; shell builds are not, hence `env.sh`/`toolchain.sh`.

## 4. Per-artifact extras

Beyond the common layer, these directories need something extra (from their
`build.sh`; everything is installed artifact-locally, never into the venv):

| Extra | Artifacts |
|---|---|
| g++ 12 as nvcc host compiler (`gcc12`) | convolution/hidet, fft/turbofft, gemm/hexcute, gemm/moonpoly, gemm/turbofno, mttkrp/blco, multigrid/amgt, multigrid/bootcmatchgx, spgemm/amgt, spgemm/ocean, spmm/inferfast, spmm/mp-spmm, spmm/rassm, spmm/sspmm |
| MPI headers/libs (link-only) | cholesky/hicma-x, graph-pattern-mining/graphset, multigrid/amgt, multigrid/bootcmatchgx, spgemm/amgt |
| BLAS/LAPACK | cholesky/exageostat, cholesky/hicma-x, multigrid/bootcmatchgx |
| CUTLASS (fetched at a pinned commit by `build.sh`) | attention-kernel/{bytetransformer,flashattention-t,metaattention,pat}, gemm/moonpoly, gemv/marlin, sparse-attention-kernel/sparse-transformer |
| `pip install --target=<artifact>/pylibs` (einops, tilelang, hidet, apache-tvm-ffi, …) | attention-kernel/{metaattention,pat}, connected-components/yacclab, convolution/hidet, gemm/hexcute, gemv/{marlin,packkv}, quantized-gemm/{fp6llm,qfactory,tilus}, tensor-contraction/fastkron |
| Network download at build time (Zenodo / release tarballs) | cholesky/exageostat (StarPU, NLopt), spmm/mp-spmm (code lives on Zenodo, not GitHub) |
| hipcc (HIP-on-CUDA) | spmv/spmv-acc (BUILD-FAILED here: hip 5.5.1 vs CUDA 12.9) |
| StarPU + Chameleon built from source into `<artifact>/prefix/` | cholesky/exageostat (`build.sh` does it; ~1 h) |

## 5. Known traps (all hit on the reference machine; each has a fix in place)

1. **Toolkit drift.** The site's default module moved from CUDA 12.9 to 13.2
   while torch stayed cu128. Symptoms: torch JIT recompiles a cached extension
   with the wrong nvcc; CUTLASS fails with `identifier "PFN_cuTensorMapEncodeTiled"
   is undefined` because 13.2's `cudaTypedefs.h` leaks in through `CPATH`. Fix:
   `toolchain.sh` re-points `CUDA_HOME`, `PATH`, `CPATH`, `LIBRARY_PATH`,
   `LD_LIBRARY_PATH` at the matching toolkit (`CPATH` must be re-pointed, not
   cleared: torch's ATen headers need `cusparse.h`).
2. **HPC SDK layout.** cuBLAS/cuSPARSE/cuSOLVER/cuRAND/cuFFT are in
   `math_libs/<ver>/`, not `cuda/<ver>/lib64`; nvcc adds only the latter
   implicitly. `KB_CUDA_MATHLIBS` handles it; a standard toolkit needs nothing.
3. **libstdc++ ABI.** The venv's Python loads an old libstdc++ before torch;
   extensions compiled with g++ 14 then fail with `CXXABI_1.3.15 not found`.
   Fix: `LD_PRELOAD=/usr/lib64/libstdc++.so.6` before Python starts
   (`KB_LD_PRELOAD`). `import torch` must precede importing any extension
   `.so` (libc10 must be resident).
4. **Host compiler.** nvcc 12.9 rejects gcc ≥ 15 and some ~2020 code bases
   need gcc 12 (`KB_HOST_COMPILER_BIN`). Cray `cc`/`CC` wrappers follow the
   loaded `cudatoolkit` module, not `CUDA_HOME`: use plain gcc/g++ paths.
5. **No `libnvToolsExt.so`** in CUDA 12.9 (NVTX v3 is header-only): drop
   `-lnvToolsExt` from artifact link lines (build-system fix).
6. **No MPI launcher on Cray.** `mpirun`/`mpiexec` do not exist; single-rank
   MPI programs run as singletons. Linking `libmpi.so` by hand needs
   `-lfabric` (`/opt/cray/libfabric/default/lib64`).
7. **Shared venv is off limits.** Two artifacts once `pip install`ed the same
   package name (upstream `hidet` vs the HexCute fork) into the venv and
   clobbered each other. Everything artifact-specific goes to
   `<artifact>/pylibs` or `<artifact>/source/python` on `sys.path`.
8. **Never traverse shared filesystems** (`find`/recursive `grep` over
   `/opt`, `/global`, `/pscratch`, …): an admin warning was received. Locate
   software with `command -v`, `module spider`, or the `KB_*` paths.

## 6. Artifacts that need a newer GPU (DEFERRED-HARDWARE)

Each such directory has a one-line `REQUIRES_GPU` (minimum arch + the feature
that needs it); `check_env.py` lists them. Their clones/provenance are kept so
they can be built where the hardware exists:

| Artifact | Needs | Why |
|---|---|---|
| spmm/voltrix | sm_90a (Hopper) | `cp.async.bulk.shared::cluster` + `mbarrier` PTX, hardcoded `compute_90a` |
| quantized-gemm/mxblas | sm_90a (Hopper), FP8 | FP8 tensor cores, WGMMA/TMA; also needs Python ≥ 3.12 |

## 7. Timing runs (not yet done anywhere)

All results so far are functional gates on a shared login node and are marked
`conforming:false` by the runner. Conforming measurements need: an exclusive
GPU (a batch allocation, `SLURM_JOB_*` set), locked clocks (the runner records
`clocks_locked`), and the spec's full protocol (no `--smoke`, no `--warmup`/
`--reps` overrides). On Perlmutter that means `srun` on a compute node, not
the login node; see the project memory notes on `dcgmi profile --pause` when
profiling.

## 8. GPU architecture: the build scripts pin sm_80

The reference GPU is an A100, and 78 of the 101 `build.sh` files carry an
explicit A100 arch flag. `toolchain.sh` exports `KB_SM` / `KB_TORCH_ARCH`
from the visible GPU (`nvidia-smi --query-gpu=compute_cap`), but the existing
scripts were not rewritten to use them: many generate Makefiles or patch the
artifact's own build system, where a blind substitution would break. On a
non-A100 machine, treat each script individually:

| Flag in the script | On a newer GPU (e.g. sm_90) | Action |
|---|---|---|
| `-arch=sm_80` (37 scripts) | embeds PTX for compute_80; the driver JIT-compiles it (works; first launch slower; not the SASS the paper measured) | replace with `-arch=sm_${KB_SM}` when you want native code; note the choice in STATUS.md |
| `-gencode arch=compute_80,code=sm_80` (28) | SASS only: `cudaErrorNoKernelImageForDevice` at first launch | `-gencode arch=compute_${KB_SM},code=sm_${KB_SM}` |
| `TORCH_CUDA_ARCH_LIST="8.0"` (17) | SASS only, same failure inside torch extensions | `TORCH_CUDA_ARCH_LIST="${KB_TORCH_ARCH}"` |
| `CMAKE_CUDA_ARCHITECTURES=80` (7) | CMake emits SASS + PTX for `80`: loads via JIT | `${KB_SM}` for native code |
| arch pins inside the artifact's own Makefile/CMake/setup.py (patched by `sed` in some scripts) | as above, per flag | a build-system fix under ARTIFACT_GUIDE rule 3: record it in `source.patch` |

`check_env.py` prints the count of pinned scripts when the GPU is not sm_80.
Kernels that use `__CUDA_ARCH__`-conditional code paths or Ampere-only PTX
(`cp.async`, `mma.sync` shapes) still run on Hopper; the reverse direction
(sm_90-only PTX on an A100) is what `REQUIRES_GPU` marks (§6).

## 9. zaratan (UMD): a fully set-up second machine (2026-09-09)

The whole suite was reproduced on the UMD **zaratan** cluster; it is now a
working machine, not just Perlmutter. **A collaborator continuing on zaratan
usually does nothing but `source bench/env.sh`** — every `KB_*` default in
`env.sh` already points at the toolchain below, so `check_env.py` reports
0 MISSING and `smoke_all.sh` is 35/35 out of the box. Full machine writeup and
the per-adapter recorded-vs-observed roll-up: **`bench/REPRODUCTION_zaratan.md`**.

Toolchain = two conda-forge envs, group-shared under
**`/scratch/zt1/project/bhatele-lab/shared/kernel-bench/`**:
- `envs/kb-env` — Python 3.12, torch 2.8.0+cu128, cuda-toolkit 12.8 / nvcc
  12.8.93, gcc/g++ 13.4, cmake 4.2, ninja, OpenMPI 5, OpenBLAS, torch_geometric.
- `envs/kb-gcc12` — gcc/g++ 12.4 for `KB_HOST_COMPILER_BIN`.
- `data/` — group-shared dataset caches; `env.sh` points `KERNELBENCH_*` at them
  so every collaborator reuses one copy **without re-downloading**. Already
  populated (2026-09-10, group-readable): `data/matrices/` (~6 GB, 23 real
  SuiteSparse/SNAP matrices — audikw_1, ldoor, cage14, soc-LiveJournal1,
  mip1/shipsec1/pdb1HYS, bcsstk*, email-Enron, as-caida, …), `data/tensors/`
  (FROSTT `nips.tns`/`uber.tns`/`chicago-crime.tns`), `data/annsearch/`
  (SIFT1M: `sift.tar.gz` + extracted `sift/*.fvecs`), `data/planetoid/`
  (Cora/CiteSeer/PubMed). New SuiteSparse names auto-download into
  `data/matrices/` on first use (login node only — compute nodes have no
  internet; `env.sh` sets `SSL_CERT_FILE`=certifi so the download verifies).
- **Planetoid is NOT redirectable** (the harness hardcodes `bench/graphs/`, and
  that dir is git-ignored), so point each clone at the shared copy once:
  `mkdir -p bench/graphs && ln -s "$SHARED_KB_DATA/planetoid" bench/graphs/planetoid`
  (this repo's clone is already symlinked). Everything else follows `KERNELBENCH_*`.

The site modules top out at gcc 11.3 / CUDA 12.3, which is why conda-forge is
used instead. `env.sh`'s `KB_*` defaults already point here — a collaborator in
the `bhatele-lab` group just `source bench/env.sh` and goes.

> **How `envs/kb-env` got here:** it is a real directory, **`mv`d** out of the
> original `.../user/cunyang/condaroot/.../envs/kb-env` into the shared prefix.
> On this beegfs a `mv` within the same filesystem is an instant metadata
> rename (no data copy), whereas `conda --clone` / `conda create` / `cp -a` of
> its ~62k CUDA files all crawl or hang — so `mv` is the way to relocate a
> conda env here. It works because `env.sh` only ever uses absolute paths
> (`$KB_PY`, `$CUDA_HOME`, `$CC`, ...) and never `conda activate`s, so the
> old prefix baked into conda-meta / console-script shebangs is irrelevant;
> python/nvcc/gcc resolve their own prefix from the executable location. The
> group can read it because it now lives under the group-owned `shared/` tree
> (no longer behind `user/cunyang`, which is `drwx--S---` = not group-traversable).

**Recreate the envs at the shared prefix (or on a fresh machine):**
```bash
SH=/scratch/zt1/project/bhatele-lab/shared/kernel-bench
conda create -y -p $SH/envs/kb-env -c conda-forge --override-channels python=3.12 cuda-toolkit=12.8 \
    cuda-nvcc=12.8 cuda-driver-dev=12.8 gcc=13 gxx=13 gfortran=13 cmake ninja make openmpi openblas \
    "libstdcxx-ng>=13" pkg-config autoconf automake libtool
conda create -y -p $SH/envs/kb-gcc12 -c conda-forge --override-channels gcc=12 gxx=12
E=$SH/envs/kb-env
"$E"/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
"$E"/bin/python -m pip install -r bench/requirements.txt
"$E"/bin/python -m pip install torch_geometric        # gnn/sddmm/spmm cora-graph gates
# conda-forge puts CUDA headers only under targets/x86_64-linux/include and has no lib64;
# the harness assumes the standard toolkit layout, so add the two shims once:
( cd "$E" && for f in targets/x86_64-linux/include/*; do b=$(basename "$f"); [ -e include/"$b" ] || ln -s ../targets/x86_64-linux/include/"$b" include/"$b"; done; [ -e lib64 ] || ln -s lib lib64 )
```
(The conda `create`/`cp` steps can take a long time on beegfs — run them in a
batch job, not interactively.) On a different cluster, edit the `KB_*` prefixes
at the top of `env.sh` to your own paths and re-run `$PY bench/check_env.py`.

**GPU work.** Login nodes have **no GPU**; everything GPU goes through
`bench/gpu_run.sh [-g <gres>] [-t MIN] -- '<cmd>'` (an `srun` wrapper). Its
default gres is an **A100 MIG 1g.5gb slice** (`gpu:a100_1g.5gb`, sm_80, ~5 GB) —
it allocates in seconds and is enough for almost every `--smoke` gate; full
A100s (`-g a100`) and H100s (`-g h100`, sm_90a) can queue for hours-to-days.
`env.sh` caps OMP/OpenBLAS threads to 4 so concurrent python gates don't blow
the login node's `RLIMIT_NPROC=256` (that exhaustion otherwise surfaces as
bogus numpy `CPU dispatcher tracer already initialized` errors — it is fork
starvation, not a numpy bug).

**zaratan-specific gotchas already handled** (all machine-neutral, keep the
Perlmutter fallbacks): OpenMPI lib is `libmpi.so` (`KB_MPI_LIBNAME=mpi`), BLAS
is OpenBLAS (`KB_BLAS_LIB`); the system libstdc++ is gcc-8, so `KB_LD_PRELOAD`
points at the gcc-13 one; MPI-singleton adapters strip `SLURM_/PMIX_/OMPI_/PMI_`
from the child env or deadlock on Slurm+PMIx+OpenMPI; `spmv/spmv-acc` uses the
site `module load hip/5.6.1/.../cuda/12.3.0` toolchain (probed by its build.sh);
`ngap`/`spmv-acc` need more than the MIG slice's 5 GB (use `-g a100`).
`spmm/voltrix` + `quantized-gemm/mxblas` need an sm_90a H100 (`-g h100`): built
and ready, gate pending an idle H100.
