# amgt (AmgT) — spgemm

**Status: BUILT+GATED**

- Paper: "AmgT: Algebraic Multigrid Solver on Tensor Cores", SC'24.
  `PAPER_KEY = conf/sc/LuZWFLCY0C024` (matched by `artifact_url` in
  `../../output/included.json`).
- Artifact: https://github.com/SuperScientificSoftwareLaboratory/AmgT (a
  fork of HYPRE with AmgT's own SpGEMM/SpMV kernels patched into
  `AmgT_HYPRE/src/seq_mv/`).
- Commit cloned: `278a6dcad06a79794fc631942ce147b2ec0c1e90` (`git clone
  --depth 50`).
- Toolchain: `nvcc` 12.9, host compiler `gcc-12`/`g++-12` (via
  `MPICH_CC`/`MPICH_CXX` env vars so `mpicc`/`mpicxx` invoke it instead of
  the untested system default gcc-14), Cray `cray-mpich/9.0.1`,
  `-gencode arch=compute_80,code=sm_80` (A100), `--with-cuda
  --with-gpu-arch='80 80' --enable-unified-memory --enable-shared
  --disable-fortran`.

## What the artifact actually is / boundary wrapped

`spgemm_amgT_fp64(hypre_CSRMatrix *A, hypre_CSRMatrix *B, hypre_CSRMatrix
**C_ptr)` (`source/AmgT_HYPRE/src/seq_mv/csr_spgemm_device.c:1527`) is a
standalone C function computing `*C_ptr = A @ B` for two arbitrary
`hypre_CSRMatrix*` objects via AmgT's own mBSR tensor-core-friendly block
format — genuinely separable from the AMG solver (no AMG setup/solve runs
anywhere in this adapter; `AmgT_test/test_new.c`'s end-to-end AMG-solve
benchmark driver is never built or linked). `wrapper.cu` (this directory,
NOT under `source/`) exposes it with `extern "C"` linkage; the task brief's
own warning ("if the SpGEMM is inseparable from the multigrid hierarchy,
SKIP") does not apply here — it is separable, this is the finest available
boundary, and it is a real per-call kernel entry point, not the paper's
benchmark script (ARTIFACT_GUIDE.md rule 1).

`CSR2BSR_GPU(hypre_CSRMatrix *A)` (`csr_matvec_device.c:1198`) is
idempotent (`if (!hypre_BSRTAG(A))`), so `wrapper.cu`'s
`amgt_build_and_convert()` calls it ONCE per operand in `prepare()` — the
artifact's own format conversion, timed as preprocessing (rule 2).
`spgemm_amgT_fp64`'s own internal `CSR2BSR_GPU(A); CSR2BSR_GPU(B);` calls
then become free no-ops on every subsequent `run()` call, so the harness's
timed window covers exactly symbolic pass + `numeric_spgemm_hybrid` kernel
+ C's device allocation + the BSR-back-to-CSR pass that produces `*C_ptr` —
matching `spgemm-square-kernel-f64`'s `timing_scope`
(`benchspecs/spgemm/spec.yaml`) cleanly, unlike `../ocean/STATUS.md`'s
disclosed Workspace-overhead compromise.

`C = A @ A` (self-product, per this track's `operation` field): ONE device
`hypre_CSRMatrix` is built and passed as both operands (`A == B`), verified
safe by reading `spgemm_amgT_fp64`'s body directly — `dmatA`/`dmatB` are
by-value copies of the `bsrMAT` struct taken once at entry and never
written back into `*A`/`*B`; only `C`'s own freshly-allocated buffers are
written, so no read/write aliasing hazard exists.

## Build-system fixes (rule 3: minimal, recorded)

1. **Permission bits restored** on several of HYPRE's own autoconf-generated
   helper scripts (`configure`, `tarch`, `config/{mkinstalldirs,install-sh,
   *.sh}`) that lost their executable bit in this clone's transfer to this
   machine — `git diff` on every one of these is a 0-line content diff
   (mode-only change, confirmed above: `git diff --stat` shows `| 0` for
   all but `seq_mv.h`).
2. **`seq_mv.h` header swap** — `cp source/config_files/AmgT_FP64.h
   AmgT_HYPRE/src/seq_mv/seq_mv.h`, the artifact's own documented build-time
   selection mechanism (its README: "Select the compilation version by
   changing the execuative in compile.sh"; `compile.sh` performs the exact
   same `cp`). Full diff (2 lines):
   ```
   -// #define Hypre_AMGT
   +#define Hypre_AMGT
   -// #define ADAPTIVE_AMGT_SPMV
   +#define ADAPTIVE_AMGT_SPMV
   ```
   This governs `hypreDevice_CSRSpGemm`'s internal cuSPARSE-vs-AmgT dispatch
   and SpMV's balanced/unbalanced kernel choice; it does NOT gate
   `spgemm_amgT_fp64` itself (defined unconditionally, no `#ifdef` around
   it) — done anyway for provenance fidelity to the artifact's documented
   convention.
3. **`--disable-fortran`**: this machine has no working Fortran compiler
   wired to `mpif77`, which makes `./configure`'s Fortran name-mangling
   probe fail; the wrapper never touches HYPRE's Fortran bindings.
   `compile.sh` doesn't pass this either way (HYPRE's own default is
   enabled-but-unused Fortran support) — a build-system fix, not a content
   change.
4. **`--enable-shared`** (added after the FIRST build attempt failed):
   `compile.sh`'s own recipe builds a static `libHYPRE.a`, which cannot be
   linked into `libamgt_wrapper.so` — `ld` fails with `relocation
   R_X86_64_32 against .rodata can not be used when making a shared object;
   recompile with -fPIC` (HYPRE's default `make` does not compile with
   `-fPIC`). `--enable-shared` is HYPRE's own documented `./configure` flag
   (`./configure --help` lists it) to build a PIC `libHYPRE.so` instead —
   not a hand patch to any Makefile.
5. **`--with-extra-ldpath="$MATHLIB"`**: nvcc 12.9's own toolkit tree ships
   `libcudart` but not `libcublas`/`libcusparse`/`libcurand` (same finding
   as `../ocean/STATUS.md` and `spmm/inferfast/STATUS.md`) — those live
   under the sibling `math_libs/12.9` tree, passed to HYPRE's linker search
   path.
6. **`MPICH_CC=gcc-12 MPICH_CXX=g++-12`**: Cray's `mpicc`/`mpicxx` wrappers
   invoke the system default `gcc`/`g++` (SUSE 14.x) unless overridden;
   nvcc 12.9's host-compiler compatibility with gcc-14 was not verified on
   this machine (same class of issue `spmm/inferfast/STATUS.md` hit with
   plain `g++`), so gcc-12/g++-12 (already working for `ocean`/`inferfast`/
   `rassm` in this repo) is used instead via the documented `MPICH_CC`/
   `MPICH_CXX` env vars, not a wrapper-script edit.

No file under `source/` has any CONTENT diff — `git -C source diff
--stat` confirms every changed file is either a 0-line mode-bit change or
the 2-line `seq_mv.h` documented header-swap above.

## Runtime environment fix (login-node quirk, recorded in adapter.py)

`MPI_Init()` aborts with `MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is
requested, but GTL library is not linked` because this machine's module
environment sets `MPICH_GPU_SUPPORT_ENABLED=1` unconditionally (a
site-wide Cray-MPICH default, not an AmgT bug or a build mistake — the
GPU-transport-layer library, `-lmpi_gtl_cuda`, is a separate Cray package
this build does not link, since AmgT's own MPI usage here is a
single-process "singleton" `HYPRE_Init()` with zero actual GPU-direct MPI
transfers). `adapter.py`'s `_ensure_init()` sets
`os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"` before calling
`amgt_init()` so the adapter works regardless of the caller's shell
environment — confirmed both with and without the variable pre-set in the
shell.

## adapter.py

- `KERNEL = "spgemm"`, `IMPL_NAME = "amgt-spgemm-mbsr"`,
  `PRECISIONS = ["fp64"]` (`spgemm_amgT_fp64` only; the mixed-precision
  path, `spgemm_amgT_fp32`/`spgemm_amgT_fp16` behind `AmgT_Mixed.h`, is out
  of scope for this task).
- `prepare()`: `amgt_build_and_convert()` — H2D copy + `CSR2BSR_GPU()` once
  (the artifact's own format conversion, timed as preprocessing).
- `run()`: one `amgt_spgemm_run(A, A)` call; frees the PREVIOUS call's `C`
  first (`spgemm_amgT_fp64` allocates a fresh `C` via
  `hypre_CSRMatrixCreate` every call — hoisting that out would mean
  patching kernel code, forbidden by rule 3 — so the adapter frees between
  calls instead of leaking across the warmup+measured-reps loop).
- `to_host()`: D2H copy via `amgt_csr_copy_out()`, reindexed onto the
  shared `|A|@|A|` structural pattern via
  `kernelbench.impls.cpu_ref._canonical_pattern`/`_reindex_to_pattern` —
  same convention as `../ocean/adapter.py` and
  `kernelbench/impls/gpu_cuda.py`'s `TorchSpGEMM`.
- `timer()`: `kernelbench.impls.gpu_cuda.CudaEventTimer`. Valid here with no
  cross-stream gap: every AmgT kernel launch in `csr_spgemm_device.c`/
  `csr_matvec_device.c` omits an explicit CUDA stream argument (confirmed
  by grep — no `cudaStreamCreate`/`cudaStream_t` in either file's device
  code), i.e. everything runs on the default stream, the same stream
  torch's `CudaEventTimer` records on.
- `hypre_CSRMatrixDestroy` (the artifact's own, unmodified) is used for
  every free; it already frees the attached mBSR side-structure via its own
  `if (hypre_BSRTAG(matrix) == 1) { cudaFree(...); free(bsr_mat); }`
  branch, so no separate BSR-freeing call is needed.

## Gate verification (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
$PY -m kernelbench.runner --kernel spgemm --variant spgemm-square-kernel-f64 \
    --impl amgt-spgemm-mbsr --smoke
```

Result: **3/3 runs valid** (smoke-uniform, smoke-banded, smoke-powerlaw;
4000x4000 synthetic matrices). `max_scaled_err` (vs. scipy fp64 reference,
gated against `|A|@|A|`'s structural pattern): `2.13e-16` / `4.57e-16` /
`2.60e-16`, all `<= 1e-6` tolerance — genuinely independent agreement at
fp64 machine-epsilon precision (AmgT's own tensor-core-friendly mBSR
kernel vs. scipy's CPU computation, not self-certifying). `LD_PRELOAD=
/usr/lib64/libstdc++.so.6` was NOT needed for this adapter (tested both
with and without; no CXXABI/GLIBCXX import error occurred). Reduced-
protocol numbers only (warmup=5, reps=20, shared login-node GPU), marked
non-conforming by the runner (`conforming: False`); not a timing result
per ARTIFACT_GUIDE.md rule 5. `smoke_all.sh` re-run after this adapter
landed: still 27/27 kernels green.

## Bugs / design notes found in AmgT (research findings)

- **No correctness self-check in the artifact itself.** As the spec's
  survey (`benchspecs/spgemm/spec.yaml` evidence block) already flagged,
  AmgT validates only indirectly via downstream PCG convergence
  (`HYPRE_PCGSetTol(solver, 1e-5)`), never an elementwise SpGEMM-output
  comparison. This adapter is therefore the first elementwise numerical
  check `spgemm_amgT_fp64`'s output has been put through outside AmgT's own
  paper — it passed cleanly (see gate numbers above), which is reassuring
  but was not something the artifact could have told a user on its own.
- **HYPRE's default build is not position-independent** (`-fPIC` absent
  from `make`'s default CFLAGS even with CUDA/GPU support configured in),
  which blocks embedding `libHYPRE.a` into any shared object a downstream
  consumer might want to build (as here). `--enable-shared` is HYPRE's own
  fix for this and worked without further patching, but it is easy to miss
  — the artifact's own `compile.sh` does not use it, since its own
  `AmgT_test` consumer only ever links a static executable, never a `.so`.
- **`MPICH_GPU_SUPPORT_ENABLED=1`-by-default plus `MPI_Init()` with no
  graceful fallback** is a genuine papercut for exactly the "call one
  kernel from a generic host process" usage pattern this integration needs
  — HYPRE (and therefore AmgT) hard-requires `MPI_Init()` even for a
  single-process, no-communication use of one CSR-matrix kernel, and Cray's
  MPI aborts outright (not a warning) if GPU-aware support is requested by
  the environment but not linked. Environment-only fix, not an AmgT-code
  bug per se, but worth flagging since it will bite anyone who tries to
  embed AmgT's kernels outside the full AMG solve path on a Cray system.

## Not done

- Only the fp64 `spgemm-square-kernel-f64` variant was gated. AmgT's
  mixed-precision (`spgemm_amgT_fp16`/`fp32`, tensor-core) path was not
  wired — out of scope for this task's login-node budget; would be the
  natural candidate for `spgemm-tensorcore-mixed-precision` in a follow-up.
- No sweep across the spec's `recommended_subset` SuiteSparse matrices —
  gate check only, per the task's login-node scope. A real matrix would
  also let `time_spgemm`/`time_spgemm_preprocess` (AmgT's own internal
  counters, currently unused by this adapter) be cross-checked against the
  harness's own preprocessing_ms/kernel_ms split.


## Baseline role (2026-09-05 selection-rule revision)

**Competitor, not a SOTA baseline.** Under the revised rule (core kernel papers
evaluated on the track's own input regime first; `kernel-papers/output/
baseline_selection.md`), this artifact would not have been selected:
- kernel centrality rated `component` (the spgemm kernel is not this paper's headline, kernel-level contribution).
- evaluated regime does not match this track's inputs (tensor-core mixed-precision R.A.P coarsening products from AMG's own hierarchy, not a benchmark-chosen A^2).
Rating rationale (`output/kernel_centrality.json`): Per benchspecs/spgemm/survey.md, AmgT's SpGEMM is 'not a standalone SpGEMM microbenchmark' -- it is measured as it naturally occurs inside AMG setup (R_i.A.P_i per level), an AMG-workload-weighted mixture rather than a clean kernel-level A^2 measurement; validated only via AMG solve convergence, not elementwise SpGEMM correctness.
It stays in the registry and runs under the same gate as every other
implementation, but Phase 3 does not treat it as the human-SOTA reference for
`spgemm`.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), gcc/g++ 12.4.0 host compiler (KB_GXX12/KB_GCC12, nvcc -ccbin), OpenMPI 5.0.10 link-only (KB_MPI_LIBNAME=mpi), HYPRE built from source, torch 2.8.0+cu128, Python 3.12.14; arch `-arch=sm_80`.
- Build: OK, exit 0 (HYPRE from source is the long part, ~8 min). Build-system changes: none beyond the already-committed KB_* parameterisation (`-l"${KB_MPI_LIBNAME:-...}"`, gcc-12 host).
- Gate: `spgemm-square-kernel-f64` / `amgt-spgemm-mbsr` / --smoke: smoke-uniform err 2.13e-16, smoke-banded err 4.57e-16, smoke-powerlaw err 2.60e-16 (all <= tol 1e-6); 3/3 runs valid.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling. (Gate each amgt/MPI-linked impl in its OWN srun step: an in-process MPI_Init leaves the srun step's PMIx state so a second MPI gate in the same step is killed by the SLURM pmix_v3 error handler — a job-step artifact, not a kernel issue.)
