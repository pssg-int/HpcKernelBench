# hicma-x (HiCMA-PaRSEC) — cholesky

**Status: BUILT+GATED** (`cholesky-dense-single-node-fp64-kernel`, err 4.11e-03 / 1.87e-03 <= tol 30.0) — **caveat: the gated residual is the artifact's OWN `--check` output** (`testing_potrf_tlr` prints `||L'L-A||_F` and `||A||_F` for the STARS-H operand it generates internally; there is no operand/factor export, so this repo cannot recompute the residual independently). The formula and threshold are the spec's, but the number is self-reported — treat it like a library-backed self-certification (DOMAIN_GUIDE iron rule), weaker than the independently recomputed gates elsewhere in this repo.

- Paper: "Toward Capturing Genetic Epistasis From Multivariate Genome-Wide
  Association Studies Using Mixed-Precision Kernel Ridge Regression" (SC'24,
  ACM Gordon Bell Prize Finalist). `PAPER_KEY = conf/sc/LtaiefACRSKDBAK24`.
- Artifact: https://github.com/ecrc/hicma-x, commit
  `366f7e07ca9afc4149a8980a3ae9b5e13181ab00` (2026-03-18), `git clone --depth 1`.
  Nested submodules (all forks under `QingleiCao/*`, branch `hicma_parsec`,
  also shallow-cloned, `.git` kept):
  - `dplasma`: `cac89274836959b5e2ca7d40a880ebbaf320137e`
  - `dplasma/parsec` (nested submodule of dplasma): `f175789133338f42ab8ea9a28a190a7961acb745`
  - `hcore`: `24de2f4061e03fea67a0d4a08d5fcbb422188170`
  - `stars-h`: `15e9f475d6dedb4e3127a9449cdaf07fcb21b3c5`
  This repo's current state is PaRSEC/DPLASMA-based ("HiCMA-PaRSEC"), a
  significant rewrite from the StarPU+Chameleon-based HiCMA the task brief's
  general strategy assumed — see "Deviation from the assumed StarPU/Chameleon
  stack" below.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`);
  `cmake` 3.28.3; host compilers via the **Cray PrgEnv wrappers**
  `cc`/`CC`/`ftn` (gcc-native 14.3.0 underneath, `PrgEnv-gnu/8.7.0`); BLAS/
  LAPACK/LAPACKE from `cray-libsci/26.03.0`; GPU: single NVIDIA
  A100-PCIE-40GB, sm_80 (shared login-node GPU). Python:
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.

## What is wrapped

`tests/testing_potrf.c` → CMake target `testing_potrf_tlr`: an end-to-end
CLI driver that generates its own STARS-H SPD operand internally, runs one
of nine PaRSEC/DPLASMA tile-Cholesky variants, and (`--check`) prints its own
computed `||A||_F` / `||L'L-A||_F`. No separate library entry point exists
(`hicma_parsec.h`'s API is a PaRSEC task-graph interface, not a callable
"factor this matrix" function), so per ARTIFACT_GUIDE.md rule 1 this adapter
wraps the **compiled binary as a subprocess**, one fresh process per
`run()` call — see `adapter.py`'s module docstring for full rationale
(mirrors `cg-krylov/spcg`'s precedent).

## Environment findings (real bugs, not hicma-x-specific — recorded since they
will recur for any future MPI+CUDA build via the Cray wrappers on this login
node)

1. **`cc`'s own header search tracks the loaded `cudatoolkit` module, not
   `CUDA_HOME`/`CPATH`.** `toolchain.sh`'s CPATH re-point (needed for plain
   `gcc`+`nvcc` builds, see `spcg`/`marlin`) does **not** affect what `cc`/`CC`
   inject on their own — confirmed via `cc -x c - -E -v </dev/null`, which
   showed `/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/include` regardless
   of `CPATH`. This broke PaRSEC's `device_cuda_module.c`, which plainly
   `#include <cuda_runtime.h>` and uses `cudaDeviceProp::clockRate` /
   `computeMode` / `memoryClockRate` — all three **removed** in CUDA 13's
   headers (deprecated since CUDA 5, gone in 13). Fix: `module swap
   cudatoolkit/13.2 cudatoolkit/12.9` in the same shell before invoking `cc`
   (build.sh does this) — a `cudatoolkit/12.9` module genuinely exists on
   this system (unlike the NVIDIA HPC SDK path `toolchain.sh` otherwise
   pins) and realigns `cc`'s own injected `-I`/`-L` with nvcc's.
2. **`craype-accel-nvidia80` unconditionally links a fixed
   `libmpi_gtl_cuda.so` GPU-transport shim built against whatever CUDA major
   is the login node's system default (currently 13), independent of which
   `cudatoolkit` module is loaded.** `MPICH_GPU_SUPPORT_ENABLED=0` is a
   RUNTIME toggle and does **not** suppress this COMPILE-TIME injection —
   linking any MPI-using binary via `cc`/`CC` with this module loaded fails
   (`libcudart.so.13 ... not found` once `cudatoolkit` is swapped to 12.9, or
   a version-mismatched link if left at 13.2). Fix: `module unload
   craype-accel-nvidia80` entirely for this build — this project's own CMake
   configure summary confirms PaRSEC builds here with "Implementation
   paradigm: MPI OFF" (no distributed runtime is actually used for this
   single-node target), so nothing is lost.

Both are environment/toolchain issues, recorded in `build.sh`'s own comments
in full; see `bench/artifacts/toolchain.sh` for the pre-existing plain-gcc
pin these two extend for the Cray-wrapper case specifically.

## Patches — build-system only, zero submodule source touched

**No file inside `source/` or any of its four submodules was modified.**
Two build-only mechanisms in `build.sh`, both scoped narrowly:

1. **GCC-14 diagnostic downgrades** (`-Wno-error=implicit-function-declaration
   -Wno-error=incompatible-pointer-types -Wno-error=int-conversion
   -Wno-error=declaration-missing-parameter-type`): GCC 14 promotes these four
   from warnings to hard errors by default; this ~2020-era PaRSEC/DPLASMA/
   STARS-H codebase predates that change throughout. Pure compiler-version
   compat (ARTIFACT_GUIDE.md's own "CUDA-version guards are fine" precedent,
   applied to a GCC-version guard) — no semantics changed, no numerics
   touched.
2. **A 3-function LAPACKE compatibility header**
   (`patches/lapacke_ge_trans_compat.h`, force-included via `-include`),
   declaring `LAPACKE_{s,d,z}ge_trans` — see that file's own header comment
   for the full root-cause trace. Short version: `hcore`'s compute kernels
   call these three "extension" LAPACKE functions (in-place matrix
   transposition); Cray LibSci **exports** them (confirmed via
   `nm -D libsci_gnu.so | grep ge_trans`), but this project's own
   `include_directories()` lists DPLASMA's bundled 2015 LAPACKE header
   snapshot (`dplasma/src/include/lapacke.h`) ahead of the real vendor
   header, and that 2015 snapshot predates these three extensions. Two more
   invasive fixes were tried and abandoned:
   - Forcing the vendor header to win via an extra `-I` in `CMAKE_C_FLAGS`:
     doesn't work — CMake's generated compile line is
     `$(C_DEFINES) $(C_INCLUDES) $(C_FLAGS)`, so `C_INCLUDES` (from
     `include_directories()`, which lists DPLASMA's copy first) always wins
     regardless of what `CMAKE_C_FLAGS` adds.
   - Renaming DPLASMA's vendored copy's include guard (`_LAPACKE_H_` →
     something private) so BOTH headers' bodies execute: this let both be
     processed, but the two LAPACKE snapshots have genuinely different
     signatures for a few less-common functions elsewhere
     (`LAPACKE_zuncsd2by1`, `LAPACKE_ztprfb`, `LAPACKE_ilaver`, ...),
     producing real `conflicting types for 'LAPACKE_...'` errors.
   - A third attempt (scoping `-DLAPACKE_UTILS` to only the `hcore`/
     `hicma_parsec` CMake targets, relying on `hcore.h`'s own
     `#ifdef LAPACKE_UTILS #include <lapacke_utils.h> #endif` gate to pull in
     the real vendor header via an UNSHADOWED filename) got further but still
     failed non-deterministically per translation unit, because some files
     reach DPLASMA's core headers (which unconditionally `#include
     <lapacke.h>`) **before** hcore.h's own gated include runs, "poisoning"
     the shared `_LAPACKE_H_` guard with the incomplete vendored content
     first.

   Declaring exactly the 3 functions hcore actually calls (nothing else) and
   letting the linker resolve them against Cray LibSci (already on the link
   line via `-DBLAS_LIBRARIES`) sidesteps the whole header-guard collision
   entirely — no `-DLAPACKE_UTILS`, no include-order fight, no submodule
   file touched. `patches/lapacke_ge_trans_compat.h` lives outside every
   submodule (`artifacts/cholesky/hicma-x/patches/`), wired in purely via a
   compiler flag in `build.sh`.
3. `-DCMAKE_POSITION_INDEPENDENT_CODE=ON` (global): stars-h's own CMake
   build did not set `-fPIC` by default, causing the final
   `libhicma_parsec.so` link to fail (`relocation R_X86_64_PC32 against
   symbol 'LAPACKE_dlarnv_work' can not be used when making a shared
   object`) once `libstarsh.a` was linked into a shared library. Standard,
   minimal CMake-level PIC fix — not a source patch.

## Deviation from the assumed StarPU/Chameleon stack

The task brief's general strategy ("StarPU (no MPI, CUDA on) + Chameleon
... then HiCMA/ExaGeoStat's Cholesky driver") describes the **older**
StarPU-based HiCMA (still used by `../exageostat/`, see its own STATUS.md).
The `ecrc/hicma-x` repository itself has since been rewritten into
"HiCMA-PaRSEC": PaRSEC/DPLASMA is the runtime (built in-tree from its own
submodule, itself nesting a forked PaRSEC), not StarPU/Chameleon at all —
confirmed from `BUILD.md`, `.gitmodules`, and the CMake configure output's
own "Implementation paradigm" summary. No StarPU/Chameleon build was
needed for this artifact as a result; the CMake superbuild handles
dplasma/parsec/stars-h/hcore directly via `add_subdirectory()`.

## adapter.py design (see its own module docstring for full detail)

- `KERNEL="cholesky"`, `IMPL_NAME="hicma-x-potrf-tlr"`, `PRECISIONS=["fp64"]`.
- Wraps `testing_potrf_tlr` as a **fresh subprocess per `run()`** (matches
  spcg's pattern): `--kind_of_problem 2` (st-2d-sqexp, TESTS.md's own
  documented "Statistics-2d-sqexp" example) and `--kind_of_cholesky 2`
  (DENSE_TLR_DP, "full double precision" — this repo's own dense-fp64
  baseline path, the closest fit to `cholesky-dense-single-node-fp64-kernel`
  available from this binary; there is no plain non-TLR dense mode).
  `--kind_of_problem 0` (randtlr, the naive default) was tried first and
  rejected: at small N it produced a LAPACK `info=51` "may be suspicious"
  warning and a NaN in the artifact's own internal "HiCMA L vs dense L"
  cross-check (transcript below); `st-2d-sqexp` gave a clean,
  well-conditioned operand and machine-precision residuals at every size
  tried.
- **Operand substitution, disclosed, not hidden**: `testing_potrf_tlr` has
  no "load a matrix from a file" mode for any `--kind_of_problem` — it
  always generates its own SPD operand internally, sized only by
  `--N`/`--NB`/`--kind_of_problem`. There is therefore no way to feed
  `dense.py`'s seeded `_spd_matrix` into it. The residual FORMULA is still
  identical to every other cholesky impl's gate
  (`||L L^T - A||_F / (||A||_F * n * eps_fp64) < 30`); only the operand `A`
  differs (the artifact's own STARS-H covariance matrix, not
  `_spd_matrix`'s `M@M.T + n*I`). `params["hicma_x_operand_note"]` records
  this in every result.

## dense.py change (minimal, optional, documented — per the task brief's own
authorization for this exact situation)

`kernelbench/domains/dense.py`'s `reference_cholesky` gained one optional
hook: `params["external_A_fro_norm"]`. When an implementation sets this key
(to the Frobenius norm of **its own** operand, read back from the artifact's
own printed value — never invented), `reference_cholesky` uses it directly
for `scale` instead of computing `||_spd_matrix(...)||_F`. Default behavior
(`external_A_fro_norm` absent) is **completely unchanged** — every existing
caller (`ScipyCholesky`, `TorchCholesky`) never sets this key.

**No-regression check** (per DOMAIN_GUIDE.md's own verification step,
re-run after this change):
```
$PY -m kernelbench.runner --kernel cholesky --variant cholesky-dense-single-node-fp64-kernel --impl scipy-cholesky --smoke
  -> running scipy-cholesky smoke-cholesky-n256  ... (err 2.48e-03 <= 30.0)
  -> running scipy-cholesky smoke-cholesky-n512  ... (err 1.06e-03 <= 30.0)
  -> 2/2 runs valid
```
Identical to pre-change behavior (unaffected by the new hook).

## Gate verification (login node, functional checks only — no timing sweep)

Standalone binary sanity check bypassing the Python harness first
(`--kind_of_problem 0`, the rejected default, shown for the record):
```
./tests/testing_potrf_tlr --N 512 --NB 128 --kind_of_problem 0 --kind_of_cholesky 2 --adddiag 512 --check --verbose 2 --cores 4 --gpus 0
  -> Warning: Factorization may be suspicious (info = 51)
  -> -- dpotrf: ||L0||_F = 7.240829e+02, HiCMA: ||L||_F = 7.240829e+02, ||L-L0||_F = -nan, ||L-L0||_F/||L||_F = -nan
  -> -- DIFFERENT matrices !
```
Switching to `--kind_of_problem 2` (st-2d-sqexp) resolved this cleanly:
```
./tests/testing_potrf_tlr --N 900 --NB 90 --fixedacc 1e-8 --maxrank 40 --kind_of_problem 2 --check --verbose 2 --cores 4 --gpus 0
  -> -- ||A||_F = 1.513152e+02, ||L'L-A||_F = 1.624336e-14
  -> -- ||L'L-A||_F/(||A||_F) = 1.073478e-16   (normalized by N*eps: ~5.4e-4, well under 30)
```

Through the harness (`$PY -m kernelbench.runner --kernel cholesky --variant cholesky-dense-single-node-fp64-kernel --impl hicma-x-potrf-tlr ...`):

| workload | precision | protocol (non-conforming, login-node) | gate | err | wall (ms) |
|---|---|---|---|---|---|
| smoke-cholesky-n256 (N=256, NB=256) | fp64 | `--smoke --warmup 0 --reps 1` | **PASS** | 4.11e-03 <= 30.0 | 1103.2 |
| smoke-cholesky-n512 (N=512, NB=256) | fp64 | `--smoke --warmup 0 --reps 1` | **PASS** | 1.87e-03 <= 30.0 | 1167.7 |

`$PY -m kernelbench.runner --kernel cholesky --list` confirms:
`ok  hicma-x          hicma-x-potrf-tlr` under "paper artifacts".

Also checked against `cholesky-tlr-lowrank-kernel` (the more natural variant
for this artifact's actual specialty): both smoke matrices compute the exact
same low err values (4.109e-03 / 1.866e-03) but are reported `INVALID` by
the harness because that variant's spec-parsed tolerance is `None`
(unparsed prose, matching `dense.py`'s own documented blas-level1-2
precedent for the identical parser limitation) — not a defect in this
adapter or the artifact.

## Not done (out of scope / login-node budget)

- No timing sweep across `recommended_subset` sizes (N in
  {2048, 4096, 8192, 16384, 22464}) — login-node budget (rule 5); only the
  two synthetic smoke sizes were gate-checked.
- `--gpus 1` is passed in every `run()` call, but at these small smoke sizes
  the artifact's own device-utilization table shows the factorization
  actually ran on `cpu-cores` (too few tiles to justify offload) — expected
  at N=256/512, not something to fix here; larger `recommended_subset` sizes
  would need a compute-node allocation to observe real GPU engagement.
- `kind_of_cholesky` 1/3/4/5/8/9 (the genuinely low-rank/mixed-precision
  variants, closer to `cholesky-mixed-precision-kernel`) were not
  separately gate-checked — out of scope for this pass; `params` already
  exposes `kind_of_cholesky`/`kind_of_problem` as overridable knobs for
  whoever extends this to that variant next.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), gcc/g++ 13.4.0 (CC/CXX) + gfortran 13.4.0, OpenBLAS (KB_BLAS_LIB/KB_BLAS_INCDIR), OpenMPI 5.0.10 (subprocess singleton), StarPU/Chameleon/PaRSEC/HiCMA built from source, torch 2.8.0+cu128, Python 3.12.14; arch `-arch=sm_80`.
- Build: OK, exit 0 (long: builds the HiCMA/PaRSEC stack). Build-system changes: none beyond the already-committed CC/CXX/FC honouring and the gcc-14-only `-Wdeclaration-missing-parameter-type` compiler-probe. adapter.py change: strip SLURM_*/PMIX_*/OMPI_*/PMI_* from the testing_potrf_tlr child env (same Slurm+PMIx+OpenMPI singleton deadlock fix as the bootcmatchgx siblings; LD_LIBRARY_PATH preserved). Machine-neutral.
- Gate: `cholesky-dense-single-node-fp64-kernel` / `hicma-x-potrf-tlr` / --smoke: smoke-cholesky-n256 err 4.11e-03, smoke-cholesky-n512 err 1.87e-03 (both <= tol 30.0); 2/2 runs valid.
- Deviation from the recorded ruling: none — errors match the recorded values (4.11e-03 / 1.87e-03) exactly.
- Verdict here: BUILT+GATED — equals the recorded ruling.
