# amgt (AmgT) — multigrid

**Status: BUILT+GATED** (`mg-gpu-solve-kernel-fixed-iter` only; see "Not done")

- Paper: "AmgT: Algebraic Multigrid Solver on Tensor Cores", SC'24.
  `PAPER_KEY = conf/sc/LuZWFLCY0C024` (same paper as `../../spgemm/amgt/`,
  already matched there).
- Artifact: https://github.com/SuperScientificSoftwareLaboratory/AmgT.
- **No new clone.** `source/` in this directory is a **symlink** to
  `../../spgemm/amgt/source` (`ln -s ../../spgemm/amgt/source source`). No
  `git clone` was run for this directory.
- **No HYPRE rebuild.** `../../spgemm/amgt/`'s already-built
  `source/AmgT_HYPRE/src/hypre/lib/libHYPRE-2.31.0.so` (commit
  `278a6dcad06a79794fc631942ce147b2ec0c1e90`, same provenance as that
  directory's own STATUS.md) is used AS-IS. `build.sh` here only builds one
  new file, `libamgtmg_wrapper.so` (from `mg_wrapper.cu`, NEW code in this
  directory), and verifies via `nm -D` that every BoomerAMG/IJ symbol it
  calls is already present in `libHYPRE.so` before attempting the link —
  confirmed: all of `HYPRE_BoomerAMGCreate/Setup/Solve/SetMaxIter/SetTol/
  GetNumIterations/GetFinalRelativeResidualNorm`,
  `HYPRE_IJMatrixCreate/SetValues/Assemble`,
  `HYPRE_IJVectorCreate/SetValues/Assemble` are HYPRE's own standard public
  API, already compiled into that library (HYPRE always builds its full
  `parcsr_ls` module; no AmgT-specific patch gates any of these symbols,
  unlike the spgemm track's bespoke `spgemm_amgT_fp64`).
- Toolchain: identical to `../../spgemm/amgt/build.sh` (nvcc 12.9 via
  `toolchain.sh`, `mpicxx` host-compiled with `-ccbin`, Cray `cray-mpich
  9.0.1`, `-gencode arch=compute_80,code=sm_80`).

## What is wrapped

HYPRE's own **public BoomerAMG solver API**
(`source/AmgT_HYPRE/src/hypre/include/HYPRE_parcsr_ls.h`), used exactly the
way `AmgT_test/test_new.c` itself uses it (`test_new.c:392-411`, read
directly): `HYPRE_BoomerAMGCreate` + a block of `HYPRE_BoomerAMGSet*` calls,
then `HYPRE_BoomerAMGSetup(precond, A, b, x)` and
`HYPRE_BoomerAMGSolve(precond, A, b, x)` called **directly as a standalone
solver** — confirmed by reading the driver's own code: it creates a
`HYPRE_ParCSRPCGCreate` solver object in the same block but never calls
`HYPRE_ParCSRPCGSetPrecond`/`Setup`/`Solve` on it (dead code in the
artifact's own driver), i.e. AmgT's own test program does NOT actually run
BoomerAMG nested inside PCG despite creating a PCG handle — it runs
BoomerAMG as the solver itself. `mg_wrapper.cu` (this directory, new code,
touches zero lines of `source/`) exposes this through 13 `extern "C"`
functions (`amgmg_init`, `amgmg_build_matrix`, `amgmg_build_vector`,
`amgmg_parvector_zero/read`, `amgmg_boomeramg_create/setup/solve`,
`amgmg_get_num_iterations/final_residual`, `amgmg_*_destroy`).

Matrix/vector construction (`amgmg_build_matrix`/`amgmg_build_vector`)
mirrors `test_new.c`'s own IJMatrix/IJVector build sequence, including
staging column-index/value/row-count buffers through **CUDA Unified
Memory** (`cudaMallocManaged`, matching `test_new.c`'s own `gpu_malloc()` —
`source/AmgT_test/ex.h:26`, literally `cudaMallocManaged(...,
cudaMemAttachGlobal)`) — required because this HYPRE build is configured
`--with-cuda --enable-unified-memory`, which makes
`HYPRE_IJMatrixSetValues`/`HYPRE_IJVectorSetValues` expect device-visible
pointers, not plain host memory. Reproduced directly from the artifact's
own code rather than guessed at.

## Configuration (verbatim from test_new.c except one deliberate change)

`amgmg_boomeramg_create()` copies `test_new.c:392-411`'s own
PCG-preconditioner configuration block **verbatim**: `CoarsenType=8` (PMIS),
`MaxLevels=7`, `MaxRowSum=0.8`, `StrongThreshold=0.25`, `TruncFactor=0.1`,
`RelaxType=18` (l1-Jacobi family), `NumSweeps=1`, `CycleNumSweeps(3,3)`,
`InterpType=6` (extended+i), `Restriction=0` (R=P^T), `PMaxElmts=4`,
`CycleType=1` (V-cycle), `MaxCoarseSize=3` — all matching AmgT's own choices
exactly, and matching `benchspecs/multigrid/spec.yaml`'s
`protocol.hierarchy_config` text for this variant.

**One deliberate deviation, disclosed**: `NumFunctions=1`, not AmgT's
hardcoded `NumFunctions=3`. AmgT's own matrices are (per its paper) elastic/
multi-component structural problems with 3 physical DOFs per node;
`poisson-3d`/`suitesparse-real` scalar SPD matrices (this track's own
inputs) have 1. Setting `NumFunctions=3` on a scalar problem would
misinterpret which unknowns belong to the same physical node and would
degrade or corrupt AMG coarsening — not a fairness-relevant "improvement,"
a correctness fix for a mismatched input class. No other configuration
value was changed.

## The known trap (task brief, spec.yaml `notes_on_fairness`) — reproduced, not hidden

`max_iter`/`tol` are passed as **explicit runtime arguments** to
`amgmg_boomeramg_create(max_iter, tol)` (not hardcoded in the `.cu` file).
`adapter.py` passes `(50, 1e-20)` for `mg-gpu-solve-kernel-fixed-iter` —
AmgT's own exact numbers (`HYPRE_BoomerAMGSetMaxIter(precond, 50)`,
`HYPRE_BoomerAMGSetTol(precond, 1e-20)`), deliberately reproducing the trap
the spec calls out (a tolerance so tight the solve always executes the full
50 iterations) **because that is the correct behavior for THIS variant**
(fixed, comparable work regardless of convergence — see spec claim text).
The gate then checks the residual independently (see below), exactly as the
task brief specifies. `mg-gpu-e2e-pcg` (the spec's convergence-gated
counterpart, tol=1e-6, hard fail if not reached) is **not wired** by this
adapter — see "Not done".

A real, surprising finding while wiring this: `HYPRE_BoomerAMGSolve`
returns `HYPRE_ERROR_CONV` (256, `HYPRE_utilities.h`: "method did not
converge as expected") on every call under this tol=1e-20 configuration —
HYPRE's own internal convergence check correctly notices the solve didn't
reach `tol` and flags it, even though it still ran the full fixed iteration
count and returned a usable `x`. `AmgT_test/test_new.c` itself never checks
`HYPRE_BoomerAMGSolve`'s return value at all (confirmed by reading it
directly), silently absorbing this. `mg_wrapper.cu`'s
`amgmg_boomeramg_solve` makes this explicit instead of silently ignoring
every possible error code: it masks off only the `HYPRE_ERROR_CONV` bit
(expected and harmless for this deliberately-never-converging
configuration) and calls `HYPRE_ClearAllErrors()` so the bit doesn't leak
into later calls; any OTHER error bit (generic/memory/argument errors)
still propagates as a real Python-level `RuntimeError`.

## Setup/solve split

Matches `solvers.AmgSolveFixedIterCPU`'s discipline exactly (same
`kernelbench/domains/solvers.py`), so the two are apples-to-apples
comparable: `prepare()` builds the matrix/vectors, creates+configures the
solver, and calls `HYPRE_BoomerAMGSetup` **once** — timed as
`preprocessing_ms` by the harness, matching the variant's `timing_scope`
("solve kernel only, hierarchy already resident"). `run()` zeroes `x`
in-place (`amgmg_parvector_zero`, a raw `cudaMemset` on the ParVector's own
backing store via `hypre_ParVectorLocalVector`/`hypre_VectorData` — the
same internal-but-public accessor macros HYPRE's own library code uses) and
calls `HYPRE_BoomerAMGSolve` — the only thing inside the timed per-rep
window.

## Correctness gate

`to_host()` D2H-copies `x` and independently recomputes
`relres = ||b-Ax||/||b||` in fp64 via **scipy**, never HYPRE's own
`HYPRE_BoomerAMGGetFinalRelativeResidualNorm` (recorded into
`params["hypre_final_residual_internal"]` for audit/cross-check only, not
for the gate decision). Pass criterion: finite `x` and finite
independently-recomputed residual — identical wording/criterion to
`solvers.AmgSolveFixedIterCPU.to_host()`.

## Gate verification (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel multigrid --variant mg-gpu-solve-kernel-fixed-iter \
    --impl amgt-boomeramg-fixed-iter --smoke
```

Result: **3/3 runs valid** (smoke-poisson2d-24/48, smoke-poisson3d-10 — this
module's own SPD synthetic smoke matrices). Independently-recomputed
`relres_after_fixed_iters` after 50 fixed V-cycles: `4.11e-16` /
`3.50e-15` / `2.13e-16` — machine-precision agreement with
`hypre_final_residual_internal` (`4.01e-16` / `3.51e-15` / `1.71e-16`),
i.e. this is a real, working, independently-verified BoomerAMG solve, not a
degenerate no-op. `hypre_num_iterations` = 50 on every run, confirming the
fixed-work protocol actually executed (the known trap reproduced exactly).
Reduced-protocol numbers only (warmup=5, reps=20, shared login-node GPU),
correctly marked `conforming: False` by the runner; not a timing result per
ARTIFACT_GUIDE.md rule 5. `../../spgemm/amgt/`'s own adapter re-verified
unchanged after this build (`--kernel spgemm --variant
spgemm-square-kernel-f64 --impl amgt-spgemm-mbsr --smoke`): still 3/3 valid,
identical error values (`2.13e-16`/`4.57e-16`/`2.60e-16`) to before this
directory existed — confirms no HYPRE rebuild occurred and nothing in
`../../spgemm/amgt/` was disturbed. `smoke_all.sh` re-run after this
adapter landed: still all-green.

## A real bug found while writing mg_wrapper.cu (documented since it wasted
real debugging time and is a genuine, reproducible trap)

A first draft of this file's header comment wrote a prose phrase
`HYPRE_IJMatrix*/HYPRE_IJVector*` (intending a shorthand for "functions
starting with these prefixes") inside a `/* ... */` C block comment. The
literal substring `*/` inside that phrase closes the comment early,
silently turning the REST of the intended-comment prose into real source
tokens — which both `nvcc` and plain `g++` then tried to parse, producing a
long, misleading chain of "identifier undefined"/"stray backtick" errors
deep inside unrelated HYPRE headers (the corrupted parser state only
surfaced as an error once it reached code far downstream of the actual
mistake). Fixed by rewording to `HYPRE_IJMatrix.../HYPRE_IJVector...`. Not
an AmgT/HYPRE bug — a bug in this wrapper's own comment text, recorded here
because the error messages it produced pointed everywhere except at the
actual cause.

## Build-system fixes

None needed beyond `../../spgemm/amgt/`'s own (unchanged, reused as-is).
This directory's `build.sh` adds one process-hygiene fix of its own: an
`nm -D lib.so | grep -q pattern` piped check (mirroring
`../../spgemm/amgt/build.sh`'s own style) reliably fails under `set -o
pipefail` when grep's early exit-on-match sends `nm` a SIGPIPE before it
finishes writing ~500KB of symbols (this binary also exports HYPRE's
CUB/thrust template instantiations) — reproduced for real while writing
this script (both the direct `nm | grep -q` form and a
variable-capture-then-`echo | grep -q` form hit it identically). Fixed by
matching via a here-string (`grep -q pattern <<< "$var"`) instead of a live
pipe, which has bash write the full content synchronously before grep
starts reading, leaving no live producer process to receive SIGPIPE. This
is a pure shell-scripting fix, unrelated to `../../spgemm/amgt/`'s own
(smaller, never-triggered-the-bug) equivalent check, which was left
untouched.

## Not done

- **`mg-gpu-e2e-pcg` (convergence-gated PCG) is not wired.** The wrapper
  exposes everything needed at the HYPRE-API level to add it
  (`HYPRE_ParCSRPCGCreate/SetPrecond/Setup/Solve` are all present in
  `libHYPRE.so`, confirmed via the same `nm -D` check build.sh already
  runs for the fixed-iter path) — only the `mg_wrapper.cu` glue and
  `adapter.py` wiring for that specific variant were not written, given
  this session's bounded login-node scope. Natural follow-up.
- **`mg-gpu-setup-kernel-f64`** (setup-phase-only timing) is not wired as
  its own adapter either — `HYPRE_BoomerAMGSetup` is already called inside
  `prepare()` here, but nothing currently re-runs setup repeatedly inside
  `run()` the way `solvers.AmgSetupCPU` does for the CPU reference. Would
  need a small adapter variant that rebuilds the hierarchy fresh per
  `run()` call instead of once in `prepare()`.
- **`suitesparse-real` matrices** (AmgT's own 16-matrix list) were not
  exercised — only this module's own synthetic SPD smoke matrices, per
  ARTIFACT_GUIDE.md rule 5 (login node: build + minimal functional/gate
  check only).
- **Mixed-precision cascade** (fp64/fp32/fp16 per level, AmgT's own paper
  claim) is out of scope — `PRECISIONS = ["fp64"]` only.
- Operator complexity / level-count reporting (this track's own required
  secondary metric) is not read back from HYPRE's `HYPRE_BoomerAMGGetGrid
  Hierarchy`/level introspection API in this adapter — a natural addition
  for whoever wires the setup variant above.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), gcc/g++ 12.4.0 host compiler, OpenMPI 5.0.10 link-only, HYPRE from source (reused spgemm/amgt's), torch 2.8.0+cu128, Python 3.12.14; arch `-arch=sm_80`.
- Build: OK, exit 0. Build-system changes: `build.sh` — the hardcoded `-lmpi_gnu_123` (Cray-MPICH lib name) changed to `-l"${KB_MPI_LIBNAME:-mpi_gnu_123}"` so it links OpenMPI's `libmpi.so` here (KB_MPI_LIBNAME=mpi), Perlmutter default kept; link-only, no kernel change.
- Gate: `mg-gpu-solve-kernel-fixed-iter` / `amgt-boomeramg-fixed-iter` / --smoke: smoke-poisson2d-24/48, smoke-poisson3d-10 all valid (err 0.00e+00, exact structural gate); 3/3 runs valid.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
