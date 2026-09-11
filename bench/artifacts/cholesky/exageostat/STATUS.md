# exageostat — cholesky

**Status: BUILT+GATED** — `exageostat-chameleon-dpotrf` passes Chameleon's
own `--check` under `--gpus 1` on UMD zaratan's toolchain (verified at
N=256, 512, 2048), and the harness gate passes cleanly at N=256/512. See
"Reproduction on zaratan (2026-09-09)" below for the full toolchain diff
and raw evidence; the GPU-path numerical bug recorded on Perlmutter (just
below) does not reproduce here.

Earlier outcome, kept for the record: **BUILT, GATE-FAILED (GPU path)** —
`exageostat-chameleon-dpotrf` reproducibly fails Chameleon's own `--check`
under `--gpus 1` (every N tried: 256, 512, 1000, 2048); the identical
invocation under `--gpus 0` (CPU) reliably **passes**. This is recorded
as-is per ARTIFACT_GUIDE.md rule 4 — the gate is not loosened to hide it.
(This was the Perlmutter finding: nvcc 12.9, gcc 14.3, Cray LibSci,
A100-PCIE-40GB — see "The GPU-path numerical bug" section below, left
unedited.)

- Paper: "Accelerating Geostatistical Modeling and Prediction With
  Mixed-Precision Computations: A High-Productivity Approach With PaRSEC"
  (TPDS'22). `PAPER_KEY = journals/tpds/AbdulahCPBDGKLS22`.
- Artifact: https://github.com/ecrc/exageostat, commit
  `ee4a82bd5b9bf99089b76dc11d288c24c6da58e0`, `git clone --depth 1`.
  Chameleon submodule (`gitlab.inria.fr/solverstack/chameleon`,
  release-1.1.0 branch) at `4db899ca30d29927018d83964b9b6d517269abe1`, plus
  its own nested submodules `cmake_modules/morse_cmake`
  (`37bce4cbc6b44869ec837c914b731794ad9f360c`) and `hqr`
  (`aa6617c4780f00018477320e129bbd26a425b85e`). `hicma` submodule NOT
  initialized (`EXAGEOSTAT_USE_HICMA=OFF`, see "Not done" below).
- Toolchain: `nvcc` 12.9; `cmake` 3.28.3; plain `gcc`/`g++`/`gfortran` 14.3.0
  (`/opt/cray/pe/gcc-native/14/bin/*`, **not** the Cray `cc`/`CC`/`ftn`
  wrappers — see "Environment note" below); BLAS/LAPACK/LAPACKE from
  `cray-libsci/26.03.0`; GPU: single NVIDIA A100-PCIE-40GB, sm_80 (shared
  login-node GPU).

## Provenance mismatch (disclosed, see ../README.md)

AbdulahCPBDGKLS22's own title/abstract describe a **PaRSEC**-based
three-precision Cholesky. The `ecrc/exageostat` repository this project's
paper→artifact mapping points to is **StarPU+Chameleon**-based — confirmed
directly from its own `CMakeLists.txt`/`INSTALL` (no PaRSEC dependency
anywhere in this repo). The actual PaRSEC-based lineage this paper
describes appears to have been folded into `../hicma-x/`'s current
"HiCMA-PaRSEC" incarnation instead — `hicma-x`'s own `README.md` literature
list cites this exact TPDS'22 paper. This adapter wraps what the *mapped*
repository actually contains; see `../README.md`'s per-paper accounting for
how this is reflected there.

## Dependency stack built (none pre-existing as modules on this system)

1. **StarPU 1.3.11** (source tarball, `files.inria.fr/starpu` — no module
   exists here). `--disable-mpi --disable-build-tests --disable-build-examples
   --disable-starpufft --with-hwloc --enable-cuda`, plain gcc-native + nvcc
   12.9. Installed into `prefix/`.
2. **Chameleon** (submodule, release-1.1.0). `-DCHAMELEON_USE_MPI=OFF
   -DCHAMELEON_USE_CUDA=ON -DCHAMELEON_USE_CUBLAS_V2=ON`, linked against
   `prefix/`'s StarPU (`PKG_CONFIG_PATH`) and Cray LibSci
   (`-DBLAS_LIBRARIES=...libsci_gnu.so`, forced explicitly — CMake's plain
   `find_package(BLAS)` otherwise resolved an unrelated NVHPC 23.9 bundled
   `libblas.so` with no matching LAPACK, the same "BLAS found, LAPACK
   not found" failure mode `../hicma-x/STATUS.md` also hit). Installed into
   `prefix/`.
3. **NLopt 2.7.1** (source tarball, upstream GitHub release — ExaGeoStat's
   own CMake auto-installer, `ImportNLOPT.cmake` → `InstallNLOPT.sh`,
   failed here: `Please specify Installation Path AND Setup Path`, a
   `TMP_DIR` CMake variable the top-level `CMakeLists.txt` never sets when
   `EXAGEOSTAT_INSTALL_DEPS=OFF`. A plain upstream build was simpler and
   faster than debugging that script). Installed into `prefix/`
   (`lib64/pkgconfig`, note the `lib64` — CMake's default `GNUInstallDirs`
   on this distro, added to `PKG_CONFIG_PATH` alongside `lib/pkgconfig`).
4. **GSL 2.6** — already present as a system package
   (`/usr/lib64/libgsl.so`), no build needed.
5. **ExaGeoStat** itself: `EXAGEOSTAT_SCHED_STARPU=ON EXAGEOSTAT_USE_MPI=OFF
   EXAGEOSTAT_USE_HICMA=OFF EXAGEOSTAT_USE_CHAMELEON=ON
   EXAGEOSTAT_USE_CUDA=ON`. Built in `source/build/` (not installed —
   `make install` was not needed since this adapter runs binaries directly
   from the build tree).

## Environment note: plain gcc-native, not the Cray `cc`/`CC` wrappers

Unlike `../hicma-x/` (which needed the Cray wrappers specifically for their
automatic MPI/libsci linkage), this stack was built with plain
`gcc`/`g++`/`gfortran` throughout (matching `../../gemv/marlin/`'s and
`../../cg-krylov/spcg/`'s precedent) since `EXAGEOSTAT_USE_MPI=OFF` removes
the one reason to need the wrappers, and doing so sidesteps
`craype-accel-nvidia80`'s CUDA-13-built GPU-transport-layer shim entirely
(the same `libmpi_gtl_cuda.so` issue documented in `../hicma-x/STATUS.md`,
point 2) — simpler than hicma-x's `module swap`/`module unload` dance since
there is no MPI linkage to fight with here at all.

## Patches — all build-system / language-compat, no numerics touched

**Every patch below is a header/declaration-level or compiler-flag fix; none
touch a BLAS/LAPACK/StarPU/Chameleon numerical codelet.** Full diffs are in
each file (`git diff` in `source/` and `source/chameleon/`); this section
gives the substance.

1. **`cublas.h`/`cublas_v2.h` mutual exclusion** (CUDA-version guard):
   CUDA 12.9's `cublas_v2.h` hard-errors (`#error "It is an error to
   include both cublas.h and cublas_v2.h"`) if the legacy header is also
   included in the same translation unit — a restriction added after this
   ~2018-2023-era codebase was written. Found in **five** separate spots
   across two repos, each fixed to include only `cublas_v2.h` (this build
   sets `CHAMELEON_USE_CUBLAS_V2=ON` throughout, matching what
   ExaGeoStat's own CUDA kernels need — they use `cublasHandle_t` directly,
   confirmed from `exageostat_exact/cuda_core/include/exageostatcudacore.h`'s
   own function prototypes):
   - `chameleon/cudablas/include/cudablas.h` (the `#if
     CHAMELEON_USE_CUBLAS_V2` branch unconditionally included BOTH headers)
   - `chameleon/control/common.h` (same pattern)
   - `chameleon/runtime/starpu/include/chameleon_starpu.h.in` (legacy
     `<cublas.h>`/`<starpu_cublas.h>` were unconditional, v2 was additive on
     top — now both sides of a proper `#if/#else`)
   - `source/include/chameleon_starpu.h` — ExaGeoStat vendors its **own**
     static copy of the same file (not generated from the `.h.in`); same
     fix applied independently
   - `source/exageostat_exact/cuda_core/compute/{cuda_conv.cu,cuda_zcmg.cu}`
     — each did a THIRD, direct `#include <cublas.h>` at file scope, on top
     of the (now-v2-only) header chain; the redundant direct include was
     dropped from both.

   Two alternative fixes were tried and rejected before landing here: (a)
   `-DCHAMELEON_USE_CUBLAS_V2=OFF` for Chameleon builds cleanly in
   isolation, but ExaGeoStat's own CUDA kernels then fail to compile
   (`cublasHandle_t` undeclared) since the legacy API has no handle type;
   (b) leaving both headers in place and hoping CUDA 13's `#error` was
   toolkit-version-specific was not an option — CUDA 12.9 (this pinned
   toolkit, not just 13.2) already enforces it.

2. **C++-reserved-keyword parameter name** (`source/include/common.h`):
   `void chameleon_pmap(..., cham_unary_operator_t operator, ...)` — legal
   C, illegal C++ (`operator` is a keyword). This prototype sits inside an
   `extern "C" { }` block (linkage-only, not a language switch) and is now
   reached from `.cu` files compiled by nvcc's C++ front end (previously
   only plain C/gcc compiled it). Renamed the PROTOTYPE's parameter to
   `op` — a prototype's parameter names are cosmetic in C, so no matching
   change was needed at the actual definition site.

3. **`vsprintf`/`vasprintf` typo** (`source/include/common.h`,
   `chameleon_asprintf`): called `vsprintf(strp, fmt, ap)` with `strp`
   declared `char**` — a real pointer-type mismatch (`vsprintf` wants
   `char*`) that a C compiler accepts with a warning but nvcc's C++ front
   end correctly rejects as a hard error. The function's own name, its
   `__attribute__((format(printf,...)))` annotation, and its `char**`
   out-parameter all match glibc's `vasprintf(3)` signature exactly — fixed
   to call that instead. This is a debug string-formatting helper, never
   called from any Cholesky/GEMM code path.

4. **StarPU's NUMA-CUDA topology guard mismatch**
   (`deps/starpu-1.3.11/src/core/topology.c`, applied by `build.sh` on
   first build, not a `source/` patch): `hwloc_cuda_get_device_osdev_by_index()`
   is called whenever `STARPU_USE_CUDA && STARPU_HAVE_HWLOC`, but the
   MATCHING `#include <hwloc/cuda.h>` a few lines above is (correctly)
   ALSO guarded on `HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX` — a real
   declaration probe. This machine's system hwloc package (2.10.0) was not
   built with CUDA support (`hwloc/cuda.h` does not exist at all,
   confirmed), so the include is correctly skipped but the call site was
   not — an upstream guard mismatch, unrelated to this integration. Widened
   the call-site guard to match; disables only an optional GPU-aware
   NUMA-node placement hint (StarPU falls back to plain CPU-RAM-adjacent
   placement), not any GEMM/POTRF kernel.

5. **PIC for CUDA device code** (`-DCMAKE_POSITION_INDEPENDENT_CODE=ON
   -DCUDA_NVCC_FLAGS="-Xcompiler;-fPIC"`, ExaGeoStat's own `CMakeLists.txt`
   still uses the legacy `FindCUDA` module, which does not automatically
   propagate `-fPIC` from `CMAKE_C_FLAGS` to nvcc's host-code compilation
   the way modern `CMAKE_CUDA_*` support does): without this,
   `libexageostat.so`'s link failed
   (`relocation R_X86_64_32 against .bss can not be used when making a
   shared object`). Pure CMake-flag fix.

6. **GCC-14 diagnostic downgrades** (`-Wno-error=implicit-function-declaration
   -Wno-error=incompatible-pointer-types -Wno-error=int-conversion
   -Wno-error=declaration-missing-parameter-type`): same rationale as
   `../hicma-x/STATUS.md`'s identical fix — this codebase predates GCC 14's
   stricter defaults throughout.

## The GPU-path numerical bug (the actual gate result — not patched)

`chameleon_dtesting` (Chameleon's own unit-test binary, built by this
project's own build system, unmodified) isolates the failure precisely:

```
$ chameleon_dtesting -o dpotrf -n 512 -b 256 -t 4 -g 0 --check --uplo Lower
0;dpotrf;4;0;1;1;0;256;122;512;512;846930886;1.648249e-03;2.722307e+01;SUCCESS

$ chameleon_dtesting -o dpotrf -n 512 -b 256 -t 4 -g 1 --check --uplo Lower
0;dpotrf;4;1;1;1;0;256;122;512;512;846930886;9.034180e-04;4.966737e+01;FAILED
```
Reproduced identically at N=256, 1000, 2048, and with a diagonal bump of
1000 (`-z 1000`, forcing extreme diagonal dominance — still `FAILED` on
GPU). This is **not** POTRF-specific and **not** a false negative from a
broken check: the same differential (`--gpus 0` passes, `--gpus 1` fails)
reproduces on `-o dgemm`, which prints an actual residual —
`||R||=4.5` on GPU vs `||R||=4e-15` on CPU for the identical inputs — proof
the GPU numerics are genuinely wrong here, not that the checker itself is
broken (a broken checker would also reject the CPU run).

**Suspected but NOT confirmed root cause** (recorded as an open finding,
not chased further within the login-node time budget): the
`CHAMELEON_USE_CUBLAS_V2=ON` migration this build required (see Patch 1
above) changes cuBLAS's scalar (alpha/beta) argument convention; cuBLAS v2
defaults to `CUBLAS_POINTER_MODE_HOST` unless a codelet explicitly calls
`cublasSetPointerMode(CUBLAS_POINTER_MODE_DEVICE)`. If StarPU's own
`starpu_cublas_v2.h` handle-management or one of Chameleon's CUDA codelets
still assumes the legacy calling convention's implicit pointer semantics,
silently wrong GEMM/POTRF results on GPU (not a crash) is exactly what
would result. Confirming this would require instrumenting Chameleon's CUDA
codelets directly — out of scope for an artifact-integration pass whose own
rules forbid touching that code anyway (it would be numerical/kernel code,
not a build-system fix).

Per ARTIFACT_GUIDE.md rule 4, this is recorded as a genuine correctness-gate
**failure for the GPU-accelerated path**, not patched, not worked around by
silently falling back to CPU for the adapter's default `platform="cuda"`
declaration.

## adapter.py design

- `KERNEL="cholesky"`, `IMPL_NAME="exageostat-chameleon-dpotrf"`,
  `PRECISIONS=["fp64"]`.
- Wraps `chameleon_dtesting -o dpotrf --check` as a **fresh subprocess per
  `run()`** (same pattern as `../hicma-x/`'s `testing_potrf_tlr` wrap and
  `cg-krylov/spcg`'s precedent) — no separate library entry point is
  exposed by either driver for "factor exactly this externally-supplied
  matrix" (same operand-substitution situation as hicma-x; reuses the same
  `dense.py` `external_A_fro_norm` hook, with a placeholder scale of `1.0`
  since `dpotrf`'s own `--check` prints no residual number, only
  `SUCCESS`/`FAILED`).
- `to_host()` maps the parsed `RETURN` field to `0.0` (SUCCESS) or a
  `1e30` sentinel (FAILED), so the harness's existing max_scaled_err gate
  machinery faithfully passes through Chameleon's own verdict.
- **Every `run()` also launches a second, CPU-only (`--gpus 0`) control
  invocation**, recorded in `params["exageostat_dpotrf_return_cpu_control"]`
  for audit — never substituted into the actual (GPU) gate result.

## Gate verification (login node, functional checks only — no timing sweep)

Through the harness (`$PY -m kernelbench.runner --kernel cholesky --variant
cholesky-dense-single-node-fp64-kernel --impl exageostat-chameleon-dpotrf
--smoke --warmup 0 --reps 1`):

| workload | precision | gate | gpu RETURN | cpu control RETURN |
|---|---|---|---|---|
| smoke-cholesky-n256 (N=256, NB=256) | fp64 | **FAIL** (max_scaled_err=1.759e+43 vs tol 30.0) | FAILED | SUCCESS |
| smoke-cholesky-n512 (N=512, NB=256) | fp64 | **FAIL** (max_scaled_err=8.796e+42 vs tol 30.0) | FAILED | SUCCESS |

`$PY -m kernelbench.runner --kernel cholesky --list` confirms:
`ok  exageostat       exageostat-chameleon-dpotrf` under "paper artifacts"
(`available()==True` — the artifact is genuinely built and callable; the
gate result above is a real, reported FAILURE, not an availability issue).

## Not done / out of scope

- `EXAGEOSTAT_USE_HICMA=OFF`: the older (StarPU-based) HiCMA + STARS-H +
  HCORE stack (needed for this repo's own TLR/low-rank Cholesky path,
  `cholesky-tlr-lowrank-kernel`) was not built — the dense/exact path alone
  already consumed the login-node time budget chasing the GPU numerics bug
  above. `hicma-x/`'s TLR/mixed-precision coverage (already BUILT+GATED for
  the dense variant, `--kind_of_cholesky` 1/3/4/5/8/9 available but not yet
  gate-checked there either) is the nearer-term path to this track's TLR
  variant.
- No timing sweep across `recommended_subset` sizes — login-node budget
  (rule 5); no root-cause fix attempted for the GPU numerics bug (would
  require touching StarPU/Chameleon CUDA codelets — kernel code, out of
  scope per rule 3).
- `synthetic_dmle_test` (ExaGeoStat's own paper-level MLE driver) was built
  successfully and its FIRST Cholesky factorization (synthetic dataset
  generation) reliably succeeds; it was not wrapped as the adapter's gate
  target because its SECOND (MLE-reevaluation) factorization deterministically
  fails for reasons this integration could not isolate (see "What is
  wrapped" in adapter.py's own docstring for the full transcript of
  attempts) — a second, independent finding from the GPU-path bug above
  (this one reproduces even with `--gpus 0`, i.e. CPU-only, ruling out the
  same root cause).

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80; `gpu_run.sh` reported
  it as `NVIDIA A100-SXM4-40GB, 8.0`, i.e. the MIG slice's parent card
  identity), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 and
  gfortran 13.4.0 (conda-forge), cmake 4.2.3, OpenBLAS (`KB_BLAS_LIB`; this
  machine has no Cray LibSci), StarPU 1.3.11, Chameleon release-1.1.0,
  NLopt 2.7.1, GSL 2.6 (all four already built into `prefix/` by an earlier,
  interrupted session on this machine; reused as-is, not rebuilt — their
  pkg-config files were already present). `CMAKE_CUDA_ARCHITECTURES=80`
  (sm_80, matching the MIG slice).
- Build: OK. Build-system changes, all in `build.sh` (none touch `source/`):
  1. Capped `make -jN` at an overridable `${KB_MAKE_J:-8}` (was a hardcoded
     `-j24` at all 5 call sites) — this login node enforces a shared,
     whole-session `RLIMIT_NPROC` across every concurrent build/agent, and a
     wide `-j` intermittently exhausted it (`fork: retry: Resource
     temporarily unavailable`, observed even in unrelated shells during this
     session). Resource contention, not an artifact bug, matching the
     already-diagnosed machine gotcha.
  2. Prepended `$CUDA_HOME/lib/pkgconfig` to `PKG_CONFIG_PATH` (before the
     ExaGeoStat cmake configure). Root cause: CMake's `FindHWLOC.cmake`
     (pulled in by `find_package(STARPU ... COMPONENTS HWLOC)`) does
     `pkg_search_module(HWLOC hwloc)`; without this conda toolchain's own
     `hwloc.pc` on `PKG_CONFIG_PATH` it silently resolved the *system*
     `hwloc.pc` instead (empty `Cflags`, since `/usr/include` is already a
     default compiler search dir), so `HWLOC_INCLUDE_DIRS` came back as
     `/usr/include` and ExaGeoStat's global `include_directories()`
     propagated that into every compile in the project, nvcc included. This
     machine's system glibc (RHEL 8) `bits/floatn.h` declares
     `typedef __float128 _Float128;` whenever `__cplusplus` is defined
     (always true for nvcc's device-code frontend on a `.cu` file), and
     nvcc's restricted C++ frontend does not understand the `__float128` GNU
     extension type — hard error, "invalid combination of type specifiers",
     5 errors, on both `exageostat_exact/cuda_core/compute/cuda_conv.cu` and
     `cuda_zcmg.cu`. Confirmed by hand (direct `nvcc -c` invocations,
     replaying the exact flags CMake's generated build used) that both files
     compile cleanly with `/usr/include` dropped from the include list —
     they need nothing from it. Making `FindHWLOC` resolve via this conda
     toolchain's own `hwloc.pc` (`Cflags: -I$CUDA_HOME/include`, matching
     what StarPU itself already linked against via `--with-hwloc`) avoids
     `/usr/include` entirely. A CUDA-toolchain/glibc-version-compat fix, not
     a numerics change; inert on Perlmutter (no `hwloc.pc` under its
     `CUDA_HOME`, so this `PKG_CONFIG_PATH` entry has no `.pc` files and
     pkg-config just skips it).
  3. Added `-DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"` to ExaGeoStat's own cmake
     invocation (Chameleon's cmake call already had it) — defensive
     consistency fix against CMake's legacy `FindCUDA` resolving
     `CUDA_NVCC_EXECUTABLE` via this conda CUDA toolkit's
     `targets/x86_64-linux/bin/nvcc` (a symlink back to the real
     `bin/nvcc`) instead of `bin/nvcc` directly; nvcc's own relative lookup
     of its `cicc` companion tool does not follow that symlink correctly
     when invoked through the `targets/...` path (observed once during
     manual reproduction of the fix above, not confirmed to recur inside a
     full CMake-driven build, but harmless to pin explicitly either way).
- Gate: harness (`kernelbench.runner --kernel cholesky --variant
  cholesky-dense-single-node-fp64-kernel --impl exageostat-chameleon-dpotrf
  --smoke --warmup 0 --reps 1`, `bench/gpu_run.sh` default MIG slice):
  smoke-cholesky-n256 fp64: **PASS** (max_scaled_err=0.00e+00 vs tol 30.0);
  smoke-cholesky-n512 fp64: **PASS** (max_scaled_err=0.00e+00 vs tol 30.0);
  2/2 runs valid. Direct `chameleon_dtesting` corroboration (same
  methodology as the original Perlmutter finding, same job):
  `dpotrf -n 512 -b 256 -g 0 --check`: SUCCESS;
  `dpotrf -n 512 -b 256 -g 1 --check`: **SUCCESS**;
  `dpotrf -n 2048 -b 256 -g 1 --check`: **SUCCESS**;
  `dgemm -n 512 -g 0 --check`: SUCCESS, `||R||=0.000000e+00`;
  `dgemm -n 512 -g 1 --check`: **SUCCESS**, `||R||=4.440892e-16` (machine
  epsilon, i.e. genuinely correct).
- Deviation from the recorded ruling: the GPU-path correctness failure
  documented on Perlmutter (`chameleon_dtesting --gpus 1` FAILED;
  `dgemm --gpus 1` `||R||=4.5`) does **not** reproduce on zaratan's
  toolchain — every `--gpus 1` check above returns SUCCESS with a
  near-machine-epsilon residual, on the same commit and the same
  `CHAMELEON_USE_CUBLAS_V2=ON` patch set as the Perlmutter build. Toolchain
  differences between the two runs: nvcc 12.8.93 (conda-forge) here vs.
  nvcc 12.9 (NVIDIA HPC SDK) on Perlmutter; g++ 13.4.0 here vs. g++ 14.3.0
  on Perlmutter; OpenBLAS here vs. Cray LibSci on Perlmutter; an A100 MIG
  1g.5gb slice here vs. a full A100-PCIE-40GB on Perlmutter. The
  Perlmutter STATUS.md's own suspected-but-unconfirmed root cause (a
  cuBLAS-v2 scalar-pointer-mode default mismatch between StarPU's cuBLAS
  handle management and Chameleon's CUDA codelets) is consistent with a
  libcublas-version-specific default or bug that a different bundled
  cuBLAS (12.8 vs. 12.9) would plausibly not share, but this pass did not
  instrument the codelets to confirm that explanation directly — it
  remains unconfirmed on either machine.
- Verdict here: **BUILT+GATED** — does NOT equal the recorded ruling
  (recorded: BUILT, GATE-FAILED (GPU path); observed here: gate passes
  cleanly at both smoke sizes, corroborated directly against
  `chameleon_dtesting`).
