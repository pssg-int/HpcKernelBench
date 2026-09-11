# bootcmatchgx (BootCMatchGX) — multigrid

**Status: BUILT+GATED** (`mg-gpu-e2e-pcg`, 3/3 smoke runs valid). Getting
here required diagnosing and fixing a genuine artifact I/O precision
ceiling (fix #5 below) — see "Gate outcome" for the full story: the
control experiment that PROVED the original failure was a text-dump
truncation artifact rather than a solver defect, and the one-line I/O-only
patch (never solver/kernel logic) that fixed it.

- Paper: "A Multi-GPU Aggregation-Based AMG Preconditioner for Iterative
  Linear Solvers", TPDS'23. `PAPER_KEY = journals/tpds/BernaschiCVD23`
  (title-matched in `../../../output/benchmark_groups.json`'s "multigrid"
  list).
- Artifact: https://github.com/bootcmatch/BootCMatchGX, commit
  `420223a88d2dfef8b28571db2c9392c1af8e146c` (`git clone --depth 1`,
  cloned 2026-09-05; a genuinely fresh clone, no prior integration existed
  for this paper).
- Toolchain: nvcc 12.9 via `toolchain.sh`, host compiler gcc-12/g++-12 (same
  choice as `../amgt/`/`../../spgemm/amgt/`), Cray-MPICH 9.0.1 (link-only —
  see "No MPI launcher" below), `-gencode arch=compute_80,code=sm_80`
  (A100), `cray-libsci/26.03.0` (substituted for the upstream Makefile's
  assumed netlib `lapack-master`/Intel-MKL dependency).

## What is wrapped

BootCMatchGX exposes a clean C++ setup/solve library API
(`src/preconditioner/prec_setup.h`, `src/solver/solve.h`), but it is
templated around the library's own `CSR`/`vector<T>`/`handles` structs with
no extern-"C" boundary — writing one from scratch was judged higher-risk
than wrapping the artifact's own end-to-end CLI driver within this
session's bounded scope. `adapter.py` wraps the compiled binary
`bin/example/driverSolve` as a subprocess (same pattern as
`../../cg-krylov/spcg/adapter.py`), running the paper's own primary
configuration (FCG + BCMG, `fcg_bcmg.properties`, values copied from the
artifact's own `src/test/data/settings/{FCG_BCMG,sample}.properties`:
rtol=1e-6, itnlim=1000 matching the spec's own tolerance/cap exactly).

## Build-system / I/O fixes (rule 3: minimal, recorded; applied idempotently by `build.sh` via `git apply --check` guards)

1. **LAPACK/CBLAS → cray-libsci.** The upstream Makefile assumes a
   hand-built netlib `lapack-master` checkout or Intel MKL — neither exists
   on this AMD-CPU machine (confirmed: no `intel-oneapi-mkl` module; the
   "fast-mkl-amd" module only sets `LD_PRELOAD`, it provides no link-time
   `libmkl_core.so`). Substituted with Cray's own vendor BLAS/LAPACK
   (`cray-libsci`, LAPACKE/CBLAS-symbol-compatible), gated behind
   `ifdef CRAY_LIBSCI_PREFIX_DIR` so the upstream netlib path is untouched
   for anyone who has it.
2. **Missing `-lcublas`/`-lcusolver`/`-lcurand`/`-lcusparse` at LINK time.**
   These libraries live only under
   `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64` on this
   machine, not under the plain `cuda/12.9` tree `CUDA_DIR` points at
   (same finding as every other artifact's build.sh in this repo, e.g.
   `../../spgemm/amgt/build.sh`'s `$MATHLIB`). Fixed by having
   `bench/artifacts/toolchain.sh` (the shared pin every build.sh sources)
   export `LIBRARY_PATH`/`LD_LIBRARY_PATH` for that directory, rather than
   patching this one artifact's link line — benefits every future
   artifact build the same way.
3. **`-lnvToolsExt` dropped.** No such library ships in the CUDA 12.9 SDK
   on this machine, and BootCMatchGX's own sources contain no NVTX
   instrumentation calls that would need it (`-DUSE_NVTX` is passed to
   `NVCC_OPT` but nothing in `src/` actually calls an NVTX API) — a
   dead link flag, safe to remove.
4. **`-lfabric` added.** Linking `libmpi.so` directly (see "No MPI
   launcher" below — no `mpicc`/`mpicxx` wrapper is used at link time
   either, since none of BootCMatchGX's own translation units are compiled
   through the MPI compiler wrapper) left `fi_*@FABRIC_*` symbols
   unresolved (libfabric provides Cray-MPICH's OFI network transport, an
   indirect dependency `mpicc -show` would normally supply automatically).
   Added `-L/opt/cray/libfabric/default/lib64 -lfabric` explicitly.
5. **`Vector::print()`'s output precision, `"%g"` → `"%.17g"`**
   (`source/BCMGX/src/datastruct/vector.cu:447`,
   `vector_print_precision.patch`, applied by `build.sh` right after
   `config_mk.patch` with the same idempotent `git apply --check` guard).
   This is the fix that made the gate pass — see "Gate outcome" below for
   the full diagnosis. It is an **I/O-only** change: the format string
   controlling how many digits `fprintf` writes for a `double`, nothing
   else on the line changes, and it touches no arithmetic, no solver
   logic, no kernel code, and no other function. `"%g"` (C's default, 6
   significant digits) was silently truncating every exported solution
   value to ~6 decimal digits regardless of how accurately BootCMatchGX
   actually solved the system; `"%.17g"` round-trips a `double` exactly
   (the standard `shortest-round-trip-safe` digit count for IEEE-754
   binary64). No solver/kernel `.cu` file's COMPUTATION changed — verified
   by re-running the exact same gate before/after and observing the
   internal (pre-truncation) residual numbers were IDENTICAL in both runs
   (see "Gate outcome" table), only the externally-recomputed relres
   changed, from clearly artifact-truncation-limited values down to
   agreement with the internal figures to ~8 significant digits.

Fixes 1-4 are build-system/link-flag only; fix 5 is a one-line I/O-format
patch. No solver/kernel COMPUTATION in `source/BCMGX/src/` was ever
changed.

## No MPI launcher (adapter.py, not a build fix)

Cray-MPICH on this login node ships **no `mpirun`/`mpiexec` binary at all**
(only `srun`, which is compute-node-only and out of scope for a
login-node build+gate check). Verified empirically that `driverSolve` run
DIRECTLY (no launcher) initializes as an MPI "singleton"
(`MPI_Init`→`nprocs=1, myid=0`) without error. `adapter.py` therefore execs
the binary directly — no `mpirun -np 1` prefix (an earlier draft assumed
one would exist; removed once this was discovered).

## Build outcome: SUCCESS (after the fixes above)

`./build.sh` → `bin/example/driverSolve` builds and links cleanly;
`ldd` reports no missing shared libraries. (An EARLIER attempt, before the
4 fixes above, failed at the final `ar` link step with
`obj/datastruct/vector.o: No such file or directory` — most likely caused
by `config.mk`'s own unbounded `-j$(nproc)`=244-way default parallelism on
this shared login node; that attempt's `build.sh` already passed a bounded
`-j8`, and the object-file race did not recur once the link-flag fixes
above were also in place. Left unresolved WHICH of these ultimately fixed
the race — not reproduced again, so not chased further.)

## Diagnosis that led to fix #5 (historical: gate FAILED before this patch was applied)

This section is kept verbatim as the evidence trail for WHY fix #5 (above)
was necessary — it describes the state of this adapter BEFORE
`vector_print_precision.patch` was applied. See "Gate outcome (current)"
below for the result WITH the patch, which is what `build.sh`/`source/`
actually contain now.

`LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel
multigrid --variant mg-gpu-e2e-pcg --impl bootcmatchgx-fcg-bcmg --smoke` →
**0/3 runs valid** (this was the result BEFORE fix #5; see below for the
current, passing result). Full evidence, per matrix (`results/
multigrid_mg-gpu-e2e-pcg_1788619139.json`):

| matrix | subprocess rc | iters | internal init/final resid | **internal relative resid** | **independent relres (gated)** |
|---|---|---|---|---|---|
| smoke-poisson2d-24 | 0 | 6 | 13.779 / 5.33e-07 | **3.87e-08** | 5.00e-06 |
| smoke-poisson2d-48 | 0 | 6 | 27.914 / 4.90e-06 | **1.76e-07** | 1.19e-05 |
| smoke-poisson3d-10 | 0 | 7 | 18.433 / 1.36e-05 | **7.37e-07** | 2.58e-06 |

The subprocess runs cleanly and BootCMatchGX's OWN internal (full-fp64,
pre-truncation) bookkeeping shows it converges WELL inside the spec's
rtol=1e-6 in all 3 cases (relative residual column above — note
"Initial"/"Final residual" in the artifact's own `--info` log are ABSOLUTE
norms, `||r0||=||b||` at x0=0, not pre-normalized; the ratio is computed
here for audit, never for the gate). Yet `to_host()`'s INDEPENDENT
recompute from the exported solution — the actual, load-bearing gate,
never the solver's own bookkeeping — lands ABOVE rtol in all 3 cases.

**Root cause, proven by a control experiment, not asserted on faith.**
`--out`'s only implementation is `Vector::print()`
(`source/BCMGX/src/datastruct/vector.cu:445`): `fprintf(fp, "%g\n",
v_->val[i])` — C's `%g` defaults to **6 significant digits**, far below
fp64's ~16. Rounding the TRUE solution x* (computed independently via
`scipy.sparse.linalg.spsolve`, relres ~1e-16) to 6 significant digits with
Python's own `"%.6g"` — i.e. simulating BootCMatchGX's own print format
applied to a PERFECT answer that never touched BootCMatchGX at all —
reproduces near-identical residual magnitudes:

| matrix | relres of x* rounded to 6 sig figs | relres actually observed from BootCMatchGX's x |
|---|---|---|
| smoke-poisson2d-24 | 5.14e-06 | 5.00e-06 |
| smoke-poisson2d-48 | 1.26e-05 | 1.19e-05 |
| smoke-poisson3d-10 | 2.39e-06 | 2.58e-06 |

These match to within the same order of magnitude on every input — a
coincidence this precise, three separate times, would not occur if the
independent-gate failure came from any genuine solver inaccuracy. **No
solution BootCMatchGX could export through `--out`, however accurately
solved, can pass an independent rtol=1e-6 gate**: the boundary this adapter
is forced to use for the solution vector has a precision ceiling below
what the spec's own tolerance demands.

**The gate was NOT loosened to compensate.** `to_host()` compares the
independently-recomputed relres against the SAME rtol=1e-6 the spec
demands. The fix applied instead (`"%g"` → `"%.17g"` in `Vector::print`,
`vector_print_precision.patch`, fix #5 above) is an I/O-only format-string
change — ARTIFACT_GUIDE.md rule 3 permits build-system/I/O fixes and
forbids only kernel/solver-code changes; this line controls how many
`fprintf` digits are written for a `double`, nothing else, so it falls
squarely on the permitted side of that line (re-examined after first
drafting this section, which had incorrectly treated the whole file as
off-limits — corrected once fix #5 was written and applied, see below).

## Gate outcome (current, with `vector_print_precision.patch` applied): PASSES

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel \
    multigrid --variant mg-gpu-e2e-pcg --impl bootcmatchgx-fcg-bcmg --smoke
```

**3/3 runs valid** (`results/multigrid_mg-gpu-e2e-pcg_1788655939.json`):

| matrix | iters | independent relres (gated) | internal relative resid | gate |
|---|---|---|---|---|
| smoke-poisson2d-24 | 6 | 3.868e-08 | 3.868e-08 | PASS |
| smoke-poisson2d-48 | 6 | 1.755e-07 | 1.755e-07 | PASS |
| smoke-poisson3d-10 | 7 | 7.370e-07 | 7.370e-07 | PASS |

With full fp64 precision round-tripped through `--out`, the independently
recomputed relres now agrees with BootCMatchGX's own internal bookkeeping
to 6-8 significant digits (compare to the 6-sig-fig-truncated numbers in
the table above, which disagreed by 1-2 orders of magnitude from the
internal figures) — confirming the diagnosis: the ONLY thing fix #5
changed was the text-dump precision, and doing so alone took every smoke
matrix from FAIL to PASS with no other change. `params["gate_fail_reason"]`
is therefore no longer set on these runs; `params["bcmgx_relative_residual_
internal"]` is still recorded on every run for audit, matching what the
independent gate now confirms rather than contradicts.

## Not done

- `mg-gpu-solve-kernel-fixed-iter`/`mg-gpu-setup-kernel-f64` are not wired
  (this adapter only targets `mg-gpu-e2e-pcg`, the variant BootCMatchGX's
  own FCG+BCMG configuration most directly matches).
- `--detailed-prof`/`--summary-prof` (BootCMatchGX's own tsetup/tsolve
  breakdown) is not parsed; only the combined subprocess wall time and the
  `--info` log's iteration-count/residual fields are captured.
- A genuinely higher-precision solution export (e.g. a binary vector dump)
  does not exist anywhere in this artifact's CLI surface (confirmed by
  reading `driverSolve.cu`'s output-writing code directly — `Vector::print`
  is the only path for both `--out` and `--out-rhs`) — there is no
  adapter-side workaround available that doesn't touch `vector.cu` itself.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), gcc/g++ 12.4.0 host, OpenMPI 5.0.10 (link-only + subprocess singleton), OpenBLAS (KB_BLAS_LIB), torch 2.8.0+cu128, Python 3.12.14; arch `-arch=sm_80`.
- Build: OK, exit 0. Build-system changes: none (build.sh already ported for OpenBLAS/OpenMPI via KB_*). adapter.py change: strip SLURM_*/PMIX_*/OMPI_*/PMI_* from the driverSolve child env — see deviation. Machine-neutral (no-op on Cray).
- Gate: `mg-gpu-e2e-pcg` / `bootcmatchgx-fcg-bcmg` / --smoke: smoke-poisson2d-24/48, smoke-poisson3d-10 all valid (err 0.00e+00, ~4.4-5.6 s each e2e PCG); 3/3 runs valid.
- Deviation from the recorded ruling: none in outcome, but one adapter fix was REQUIRED here that the reference machine did not need: BootCMatchGX's `driverSolve` does MPI_Init as a singleton, and on zaratan (Slurm + PMIx + OpenMPI) the subprocess DEADLOCKS when it inherits the surrounding srun job step's PMIx/Slurm env; the adapter now strips those env prefixes before launching the child (identical fix to cg-krylov/bootcmatchgx, same binary). Confirmed on both the MIG slice and a full A100 before the fix; clean 3/3 after.
- Verdict here: BUILT+GATED — equals the recorded ruling.
