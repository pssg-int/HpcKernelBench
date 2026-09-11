# multigrid — artifact integration outcomes

5 papers total in this track (`output/benchmark_groups.json` "multigrid"; see
`benchspecs/multigrid/survey.md` for the full evaluation-methodology survey
this section summarizes). Scope ruling (ARTIFACT_GUIDE.md): NVIDIA GPU,
single-card only; CPU-only/no-source artifacts are cheap SKIPs.

Of the 5, only 2 are genuine GPU AMG *solvers* and therefore integration
candidates at all — the other 3 are disqualified on categorical grounds
(scope or platform), independent of "newest" ranking:

| paper | key | venue/year | verdict |
|---|---|---|---|
| AmgT | `conf/sc/LuZWFLCY0C024` | SC'24 | **BUILT+GATED** (see `amgt/`) |
| BootCMatchGX | `journals/tpds/BernaschiCVD23` | TPDS'23 | **BUILT+GATED** (`mg-gpu-e2e-pcg`, 3/3 smoke valid after an I/O-only patch: `Vector::print` `%g`→`%.17g` so the exported solution round-trips fp64; the gate itself is unchanged — see `bootcmatchgx/STATUS.md`. Also reused, unmodified, by `../cg-krylov/bootcmatchgx/` as an AMG-preconditioned FCG competitor on that track's `cg-e2e-ilu0-to-convergence` variant — symlinked `source`/config, no second build.) |
| MGopt-APP | `conf/ics/YangLYDHW23` | ICS'23 | SKIP — CPU-only *and* no source |
| MGARD | `conf/ipps/ChenW0W0PTCWMFK21` | IPDPS'21 | SKIP — scope (not a solver) |
| GPTuneCrowd | `conf/ipps/ChoDKLLL23` | IPDPS'23 | SKIP — scope (autotuner, not AMG) |

"Newest single-GPU-eligible up to 3" therefore resolves to exactly the 2
rows above with a verdict of BUILT/attempted — there are only 2 GPU-solver
papers in this track's 5-paper population to begin with, not 5 minus a
newest-3 cut.

## 1. AmgT (SC'24) — `amgt/` — **BUILT+GATED**

`PAPER_KEY = conf/sc/LuZWFLCY0C024` (see `amgt/STATUS.md` for full
provenance). Wraps HYPRE's own public BoomerAMG
solver API (`HYPRE_BoomerAMGSetup`/`Solve`), reusing
`../spgemm/amgt/`'s ALREADY-BUILT HYPRE fork (`source/` here is a symlink
to `../spgemm/amgt/source`; no re-clone, no HYPRE rebuild — every symbol
this wrapper calls was already compiled into that build's `libHYPRE.so`,
confirmed via `nm -D`). One new file (`mg_wrapper.cu`) and one new adapter
wire `mg-gpu-solve-kernel-fixed-iter`: AmgT's own exact configuration
(PMIS, strong threshold 0.25, V-cycle, 3/3 sweeps, l1-Jacobi, max_iter=50,
tol=1e-20 — reproducing the spec's own documented "known trap"
deliberately, since that trap IS this variant's correct protocol).

**Gate result**: 3/3 smoke runs valid. Independently-recomputed relative
residual after the fixed 50-iteration solve: `4.11e-16` / `3.50e-15` /
`2.13e-16` across the 3 smoke matrices — machine-precision agreement with
HYPRE's own internal residual bookkeeping, confirming a real, working,
correctly-wired BoomerAMG solve (not a degenerate no-op). `mg-gpu-e2e-pcg`
and `mg-gpu-setup-kernel-f64` are NOT wired by this adapter (see
`amgt/STATUS.md` "Not done" — the HYPRE symbols for the PCG variant are
confirmed present, just not yet glued).

## 2. BootCMatchGX (TPDS'23) — `bootcmatchgx/` — **BUILT+GATED** (after the `%.17g` solution-export patch; the original `%g` 6-digit dump made any independent residual check fail)

Aggregation-based multi-GPU AMG (MPI+CUDA); a single-MPI-process/single-GPU
mode IS supported by the artifact itself (its own README: "the library
supports different low-level frameworks for computing the SpMM product ...
on a single MPI node" — i.e. 1-rank runs are a first-class, not a hacked,
usage mode), so this was a genuine integration attempt, not a scope-based
skip. Freshly cloned (`git clone --depth 1`, no prior integration existed)
into `bootcmatchgx/source/`.

**Build**: four build-system-only fixes (config.mk's LAPACK→cray-libsci
substitution; `toolchain.sh` extended to export the math_libs
`LIBRARY_PATH`/`LD_LIBRARY_PATH` cublas/cusolver/curand/cusparse actually
live under; a dead `-lnvToolsExt` flag dropped — no such library in the
CUDA 12.9 SDK and no NVTX calls in the sources; `-lfabric` added to resolve
libfabric symbols left dangling by linking `libmpi.so` directly) — full
detail in `bootcmatchgx/STATUS.md`. `bin/example/driverSolve` now builds
and links cleanly. Cray-MPICH on this login node ships no
`mpirun`/`mpiexec` at all; `adapter.py` execs the binary directly (verified
it initializes fine as an MPI singleton, `nprocs=1`).

**Gate result (current)**: 3/3 smoke runs valid, relative residual
3.9e-8 / 1.8e-7 / 7.4e-7 across the 3 smoke matrices — matching
BootCMatchGX's own internal (full-fp64) residual bookkeeping to 6-8
significant digits. This was NOT always the case, and the history is worth
keeping: BootCMatchGX's ONLY solution-export path (`--out` →
`Vector::print()` → `fprintf("%g", ...)`, 6 significant digits) originally
capped the exportable solution's precision well below what an independent
rtol=1e-6 residual check needs, making every run FAIL (0/3) regardless of
how well the solve itself converged — proven by a control experiment:
rounding the TRUE solution (scipy `spsolve`) to 6 significant digits, with
BootCMatchGX never even running, reproduced near-identical residual
magnitudes (5.14e-6/1.26e-5/2.39e-6 vs. the 5.00e-6/1.19e-5/2.58e-6
originally observed). The fix (`"%g"`→`"%.17g"` in `Vector::print`,
`vector_print_precision.patch`) is I/O-only (a format-string change
controlling `fprintf` digit count, no solver/kernel arithmetic touched) —
ARTIFACT_GUIDE.md rule 3 permits build-system/I/O fixes and forbids only
kernel/solver-code changes, so this patch is in-scope; an earlier pass had
incorrectly treated the whole file as off-limits and left the gate
failing. The gate itself was never loosened: the same rtol the spec
demands is what now passes. Full before/after evidence in
`bootcmatchgx/STATUS.md`.

## 3. MGopt-APP (ICS'23) — cheap SKIP

Geometric-MG/HPCG-style SYMGS optimization, CPU-only (per
`benchspecs/multigrid/survey.md`: "8 MPI processes x 7 OpenMP threads/rank
on ARM platforms, 4 MPI x 7 OpenMP on x86" — no GPU code path at all) —
disqualified by ARTIFACT_GUIDE.md's own scope ruling regardless of any
other consideration. Independently also has no buildable source: the
repo (`github.com/YXJ-123/MGopt-APP`) "ships only prebuilt `.exe` binaries,
no source — 'source available on request'" (survey.md item 3's `source`
line) — a second, independent disqualifying reason. No clone attempted.

## 4. MGARD (IPDPS'21) — cheap SKIP (scope)

Multigrid-based hierarchical DATA REFACTORING for lossy compression, not an
`Ax=b` solver — `benchspecs/multigrid/spec.yaml`'s own `notes_on_fairness`
and `open_questions` rule it out explicitly: "MGARD and GPTuneCrowd are
excluded from the 4 variants above ... folding MGARD's non-iterative
data-refactoring throughput ... into a 'multigrid solver kernel' spec would
blend three incompatible claims." No setup/solve split, no convergence
criterion, no `Ax=b` ever solved (survey.md item 4) — none of this track's
4 spec variants describes what this artifact does. No clone attempted;
the spec's own ruling is evidence enough.

## 5. GPTuneCrowd (IPDPS'23) — cheap SKIP (scope)

A generic crowd-sourced autotuner that treats Hypre/BoomerAMG as ONE of 5
black-box tuning targets (also PDGEQRF, NIMROD, SuperLU_DIST) — it does not
implement AMG itself at all (survey.md item 5's own scope note: "GPTuneCrowd
never implements AMG kernels itself — it treats 'run Hypre with parameter
set X, get back a runtime' as an opaque function evaluation"). Explicitly
excluded by `spec.yaml`'s `notes_on_fairness` for the same reason as MGARD.
No clone attempted.
