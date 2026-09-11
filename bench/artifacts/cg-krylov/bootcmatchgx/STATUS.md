# bootcmatchgx (BootCMatchGX, AMG-preconditioned FCG) — cg-krylov

**Status: BUILT+GATED — `cg-e2e-ilu0-to-convergence`, 3/3 smoke runs valid**

Paper: "A Multi-GPU Aggregation-Based AMG Preconditioner for Iterative
Linear Solvers", TPDS'23. `PAPER_KEY = journals/tpds/BernaschiCVD23`.
Artifact: `https://github.com/bootcmatch/BootCMatchGX`, commit
`420223a88d2dfef8b28571db2c9392c1af8e146c`.

## Shared-build relationship (read this first)

This directory does **NOT** re-clone or re-build BootCMatchGX. It reuses
`../../multigrid/bootcmatchgx/`'s ALREADY-BUILT integration verbatim:

- `source/` (this directory) is a symlink to
  `../../multigrid/bootcmatchgx/source` — the same clone, the same two
  applied patches (`config_mk.patch`: cray-libsci substitution;
  `vector_print_precision.patch`: `Vector::print`'s `%g`→`%.17g`, the fix
  that makes an independent fp64 residual gate pass at all — see that
  directory's `STATUS.md`/`source.patch` for the full diff and reasoning).
- `fcg_bcmg.properties` (this directory) is a symlink to that same
  directory's file — identical solver configuration (rtol=1e-6,
  itnlim=1000, prerelax/postrelax=4/4, relaxnumber_coarse=20 — BootCMatchGX's
  own defaults, matching its `src/test/data/settings/{FCG_BCMG,
  sample}.properties`).
- `bin/example/driverSolve` (reached through the `source/` symlink) is the
  SAME compiled binary the multigrid track's own `mg-gpu-e2e-pcg` variant
  calls. `build.sh` here does not compile anything — it only verifies that
  binary is present and its shared libraries resolve (see `build.sh`).

Only what genuinely differs between the two integrations is kept separate:
`adapter.py` (this track's own `KERNEL`/`IMPL_NAME`/`PAPER_KEY` and,
critically, a DIFFERENT correctness-gate formula — see below), and
`STATUS.md` (this file).

## Why this artifact competes on `cg-e2e-ilu0-to-convergence` (not a multigrid-only variant)

`benchspecs/cg-krylov/spec.yaml`'s `cg-e2e-ilu0-to-convergence` claim text
says it "generalizes SPCG's own artifact discipline... into a mandatory
requirement for every submission" — i.e. this slot is the track's general
"whole-solve to a relative-residual convergence gate" competition, not
literally restricted to ILU(0). The sibling variant `cg-hpcg-generator-amg`
explicitly cross-references "same relative-residual gate as
cg-e2e-ilu0-to-convergence" when describing its own AMG-preconditioned
loop, confirming AMG-preconditioned solvers are meant to be directly
comparable here. `spcg` (`../spcg/`, already in this directory tree)
already competes on this exact variant with ILU(0)-preconditioned PCG;
this adapter adds BootCMatchGX's AMG-preconditioned FCG as a second,
directly comparable competitor on the SAME protocol. The variant's own
`protocol.precision` text names both papers explicitly: *"fp32 (primary —
matches SPCG)... fp64 (secondary — matches BootCMatchGX's preconditioned-CG
convention...)"* — so `PRECISIONS = ["fp64"]` here is exactly what the
spec itself expects for this artifact, not a guess.

## Gate discipline — deliberately DIFFERENT from the multigrid sibling's gate

Per this integration task's own instruction ("read spcg's adapter/STATUS
for how convergence is gated there"), `to_host()` uses the cg-krylov
track's own to-convergence formula — `gate_tol = max(10*rtol, 1e-9)`,
`passed = finite and relres <= gate_tol` — mirroring
`kernelbench/domains/solvers.py`'s `ScipyCG.to_host()` "to-convergence"
branch and `../spcg/adapter.py`'s own `to_host()` EXACTLY (a 10x slack
over rtol). This is intentionally NOT the multigrid sibling's own gate for
`mg-gpu-e2e-pcg` (a harder, no-slack `relres < rtol`, per that variant's
own "failing to converge within the cap is a hard FAIL" text) — using the
cg-krylov-appropriate formula here is what makes this adapter's pass/fail
decision directly comparable to spcg's own numbers on this same variant,
rather than silently importing a stricter bar from a different track.

Everything else (subprocess-per-solve pattern, no MPI launcher needed —
Cray MPICH ships none, verified `driverSolve` runs fine as a singleton,
`.mtx`/rhs-text file writing, `--info` log parsing for audit, independent
fp64 `relres = ||b-Ax||/||b||` recompute never trusting the solver's own
bookkeeping) is structurally identical to the multigrid sibling's own
adapter — see that file's docstring for the fuller narrative; not
repeated here.

## Gate verification (login node, functional check only)

```
$PY -m kernelbench.runner --kernel cg-krylov \
    --variant cg-e2e-ilu0-to-convergence \
    --impl bootcmatchgx-cg-fcg-bcmg --smoke --precision fp64
```

```
running bootcmatchgx-cg-fcg-bcmg smoke-poisson2d-24  ... 1695.779 ms  [42.2s]
running bootcmatchgx-cg-fcg-bcmg smoke-poisson2d-48  ... 4907.131 ms  [128.5s]
running bootcmatchgx-cg-fcg-bcmg smoke-poisson3d-10  ...  929.305 ms  [24.2s]
3/3 runs valid
```

Per-matrix detail (`results/cg-krylov_cg-e2e-ilu0-to-convergence_1788657760.json`):

| matrix | iterations | relres_achieved | gate (relres <= 1.000e-05) |
|---|---|---|---|
| smoke-poisson2d-24 | 6 | 3.868e-08 | PASS |
| smoke-poisson2d-48 | 6 | 1.755e-07 | PASS |
| smoke-poisson3d-10 | 7 | 7.370e-07 | PASS |

All three comfortably inside the 10x-slack gate (1e-5), and in fact inside
the bare rtol=1e-6 too — the FCG+BCMG solve converges in 6-7 iterations on
these small Poisson systems, and the exported-solution precision issue
that once blocked the multigrid sibling's gate is fixed at the shared
source (`vector_print_precision.patch`), so it does not resurface here.

`$PY -m kernelbench.runner --kernel cg-krylov --list` confirms:
`ok  bootcmatchgx     bootcmatchgx-cg-fcg-bcmg` under "paper artifacts"
(`available()==True`).

## Timing-boundary caveat

Same posture as `../spcg/adapter.py` and the multigrid sibling's own
adapter: `timer()` is plain CPU wall-clock (`kernelbench.harness.Timer`),
not `CudaEventTimer` — the solve runs in a fresh subprocess with its own
CUDA context each call, so there is nothing in this Python process for a
CUDA event to bracket. The harness's own per-rep number therefore includes
process/CUDA-context startup and Matrix-Market parsing on top of the
actual AMG-setup+FCG-solve work.

## Not done

- No sweep across `recommended_subset` real matrices (bcsstk17, cant,
  nasasrb, ...) — out of the login-node budget (ARTIFACT_GUIDE.md rule 5);
  only the 3-matrix synthetic-smoke gate above was run, matching spcg's own
  posture on this same variant.
- `cg-hpcg-generator-amg` (the distributed weak/strong-scaling variant
  BootCMatchGX's own `--laplacian-3d`/multi-rank convention most directly
  matches) is NOT wired here — this adapter targets the single-rank
  `cg-e2e-ilu0-to-convergence` slot only, per the task's explicit scope.
- `--detailed-prof`/`--summary-prof` (BootCMatchGX's own AMG-setup-vs-
  PCG-loop time breakdown) is not parsed — only the combined subprocess
  wall time and the `--info` log's iteration-count/residual fields are
  captured, same gap as the multigrid sibling.

## Verdict

`bootcmatchgx (cg-krylov): BUILT+GATED, 3/3 smoke valid, relres
3.9e-8..7.4e-7 (gate <=1e-5, 10x slack over rtol=1e-6) — shared build with
../../multigrid/bootcmatchgx/ (symlinked source + config, no rebuild);
competes against spcg on the same cg-e2e-ilu0-to-convergence variant as an
AMG-preconditioned (not ILU0-preconditioned) alternative`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, on an A100-SXM4-40GB
  card), login-node build (shared multigrid/bootcmatchgx build, verified
  present, not rebuilt).
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, OpenMPI
  5.0.10 (`KB_MPI_LIBNAME=mpi`), OpenBLAS (`KB_BLAS_LIB`), Python 3.12.14.
- Build: OK (shared-build check only, per this directory's own build.sh;
  binary already built by `../../multigrid/bootcmatchgx/build.sh`).
  Build-system change already present from an earlier interrupted pass,
  verified still correct: `build.sh`'s `ldd` check now tolerates
  `libcuda.so.1`/`libnvidia-ml.so.1` being unresolved on this GPU-less
  login node (expected; driver libs only resolve on a GPU node) instead of
  failing on any `ldd "not found"` line.
- **New finding + fix (adapter.py, not build.sh):** `driverSolve` links
  MPI and does `MPI_Init` as a singleton (STATUS.md's existing "no MPI
  launcher needed" note, true on Perlmutter's Cray MPICH). On zaratan
  (Slurm + PMIx + OpenMPI) the FIRST such singleton subprocess inside one
  `srun` job step succeeds, but every subsequent one deterministically
  deadlocks: OpenMPI's singleton bootstrap inherits `SLURM_*`/`PMIX_*`/
  `OMPI_*` env vars from the surrounding `srun` step and tries to attach to
  that step's PMIx namespace instead of doing a genuinely independent
  singleton init, and only one client can attach at a time. Confirmed
  empirically (isolated from the harness, bypassing Python entirely): 4
  sequential `driverSolve` calls in one process hang on call #2 with
  inherited env — reproduced identically on both the a100_1g.5gb MIG slice
  and a full `-g a100` (NVIDIA A100-SXM4-40GB) allocation, ruling out a
  MIG-specific memory/SM-count cause — and succeed 4/4 (6-20s each) once
  `SLURM_*`/`PMIX_*`/`OMPI_*` are stripped from the child's env. Since the
  harness's `run_variant()` calls `impl.run()` many times per matrix (1
  correctness check + `warmup` + `reps`, e.g. 16 calls for this variant's
  own default `warmup=None`->10, `reps=5`), the gate could not complete at
  all without this fix (`subprocess.TimeoutExpired` at the adapter's own
  900s per-call cap, every time, until fixed). Fix applied in `adapter.py`
  `run()`: `env` now excludes any `SLURM_*`/`PMIX_*`/`OMPI_*` key before
  the `subprocess.run(...)` call — a no-op on a machine that doesn't set
  these (Perlmutter's Cray MPICH environment), so this is machine-neutral,
  not a gate/tolerance change, and touches no line under `source/`.
- Gate: `cg-e2e-ilu0-to-convergence`, fp64, `--smoke` (protocol overridden:
  warmup=5, reps=20 — the runner's own smoke defaults, not spec-conforming,
  same as every smoke check in this file): **3/3 runs valid.**
  smoke-poisson2d-24: relres=3.868e-08, iters=6, PASS.
  smoke-poisson2d-48: relres=1.755e-07, iters=6, PASS.
  smoke-poisson3d-10: relres=7.370e-07, iters=7, PASS.
  All three relres values are bit-for-bit identical (to the displayed
  precision) to the originally recorded Perlmutter numbers, as expected
  for a deterministic algorithm on fixed-seed synthetic inputs.
- Deviation from the recorded ruling: none, once the PMIx env fix above is
  applied. Without it, the gate cannot complete on this machine at all
  (infrastructure-level deadlock, not a correctness or build issue).
- Verdict here: **BUILT+GATED**, same as the recorded ruling.
