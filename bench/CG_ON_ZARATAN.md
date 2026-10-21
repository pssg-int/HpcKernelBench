# Running the cg-krylov track on zaratan: prompt for a coding agent

> Scope: the `cg-krylov` kernel track ONLY (4 paper adapters + 3 library
> baselines, added 2026-10-21) on the shared
> input set `sweep:cg-common`, not the full 98-artifact suite. For a
> from-scratch reproduction of everything, use `bench/REPRODUCE.md` instead.
> zaratan is already set up (see `bench/ENVIRONMENT.md` §9) — this prompt does
> not redo environment setup, it builds and runs one track.

Paste the block below into a coding agent (Claude Code or similar) started in
a fresh clone of this repository on zaratan, after `git pull` has brought in
the 2026-10-16 changes (`sweep:cg-common`, `benchspecs/cg-krylov/common_inputs.md`).

## The shared input set (added 2026-10-16)

No real matrix is in all four papers' evaluations (BootCMatchGX's paper used
only generated 3D Poisson grids), so the set mixes generated grids with real
SuiteSparse matrices from the other papers' lists. Every input is SPD and has
N divisible by 16 (Mille-feuille silently drops rows past the last multiple
of 16). Defined in `kernelbench/domains/solvers.py` `SWEEPS["cg-common"]`,
mirrored in `spec.yaml` `common_subset`; reasoning in
`benchspecs/cg-krylov/common_inputs.md`.

| Input | N | nnz | Why |
|---|---|---|---|
| poisson-3d-64 | 262,144 | 1,810,432 | generated; BootCMatchGX-style |
| poisson-3d-128 | 2,097,152 | 14,581,760 | generated; ≈ BootCMatchGX's own 130³, /16-safe |
| jnlbrng1 | 40,000 | 199,200 | 2D grid, easy (κ≈183) |
| torsion1 | 40,000 | 197,608 | 2D grid, very easy (κ≈41) |
| finan512 | 74,752 | 596,992 | PERKS Table 4 + SPCG list |
| shallow_water2 | 81,920 | 327,680 | PERKS Table 4 + SPCG list |
| apache1 | 80,800 | 542,184 | hard case |
| cfd1 | 70,656 | 1,825,580 | hard case |

## Two groups, never ranked against each other

The adapters do not look at `--variant`; each hard-codes its own mode:

| Group | Variant | Implementations | What one run is |
|---|---|---|---|
| fixed-iter | `cg-kernel-fixed-iter` (fp64) | `perks-cg-cached`, `millefeuille-cg-mixed` (the paper's method), `millefeuille-cg-reduce` (plain fp64 path, for reference), baseline `cusparse-cg` | 50 unpreconditioned CG steps (`millefeuille-cg-mixed`: 100, hard-coded in its kernel) |
| to-convergence, fp32 | `cg-e2e-ilu0-to-convergence --precision fp32` | `spcg-ilu0-cg` (ILU(0)), baseline `cusparse-ilu0-cg` | full solve to relres ≤ 1e-5 gate |
| to-convergence, fp64 | `cg-e2e-ilu0-to-convergence --precision fp64` | `bootcmatchgx-cg-fcg-bcmg` (AMG), baseline `amgx-pcg-amg` | full solve to relres ≤ 1e-5 gate |

A time from one group is not comparable to a time from the other. The
to-convergence group runs as two jobs so each paper is labelled with the
precision it really runs (SPCG is fp32-only, BootCMatchGX fp64-only).

## The baselines (added 2026-10-21)

Each paper is compared with the library its own paper claims to beat:

| Baseline | Directory | For | Paper's claim |
|---|---|---|---|
| `cusparse-cg` | `artifacts/cg-krylov/cusparse-cg/` | PERKS, Mille-feuille | Mille-feuille 3-5x over cuSPARSE; PERKS's paper used Ginkgo (not integrated) |
| `cusparse-ilu0-cg` | `artifacts/cg-krylov/cusparse-ilu0-cg/` (uses `cusparse-cg`'s `bridge.so`) | SPCG | 1.23x / 1.65x iterative phase, 1.68x / 3.73x end-to-end over non-sparsified ILU(0) PCG |
| `amgx-pcg-amg` | `artifacts/cg-krylov/amgx/` (NVIDIA AmgX v2.5.0) | BootCMatchGX | up to 2x over AmgX |

The cuSPARSE ones are our port of NVIDIA's CUDA samples
`conjugateGradient` / `conjugateGradientPrecond`. AmgX runs its own shipped
`PCG_AGGREGATION_JACOBI.json` config. All three run in-process with CUDA
event timers, so their harness time is solve time only (no process start).
None of them has been built or run yet.

---

```text
You are working in a clone of HPC-KernelBench, on zaratan. Run every
command below from the REPOSITORY ROOT unless it says otherwise.

GOAL. Run the four cg-krylov paper artifacts (PERKS, Mille-feuille, SPCG,
BootCMatchGX) and their three library baselines (cusparse-cg,
cusparse-ilu0-cg, amgx-pcg-amg) on the shared input set `sweep:cg-common`, all on one GPU
model (full A100), and report what you get, including every failure,
honestly. Do not "fix" the benchmark (spec, harness, adapters, or a paper's
own source) to make a result pass. NOTE: on zaratan a normal user cannot
lock GPU clocks (`nvidia-smi -lgc` needs root), so the runner stamps every
result `conforming: false` with reason "GPU clocks not locked" even inside
an exclusive allocation -- per bench/REPRODUCTION_zaratan.md section 4, this
is expected on this machine and is NOT something to debug or work around.

READ FIRST: CLAUDE.md; bench/CLAUDE.md; bench/ARTIFACT_GUIDE.md (rules 1-10
are binding); benchspecs/cg-krylov/spec.yaml; benchspecs/cg-krylov/
common_inputs.md (why these inputs, and every known limitation below);
bench/artifacts/cg-krylov/README.md and each adapter's own STATUS.md
(perks, millefeuille, spcg, bootcmatchgx), plus the baseline adapters'
docstrings (cusparse-cg, cusparse-ilu0-cg, amgx).

KNOWN LIMITATIONS (found 2026-10-15 by reading the artifact source; do not
rediscover them the slow way, and do NOT fix them in this pass -- record
them next to the numbers they affect):
- Mille-feuille has TWO adapters (added 2026-10-21):
  `millefeuille-cg-mixed` (artifacts/cg-krylov/millefeuille-mixed/) wraps
  `main-cg-mixed`, the paper's single-kernel tile-wise mixed-precision
  method -- this is the one that tests the paper's claim. It has never
  been built or run. Its kernel runs exactly 100 steps (hard-coded), so
  compare ms PER STEP, never whole-run time. It needs the work split to
  fit in 6912 warps (an A100); if not, it reports UNSUPPORTED. It shares
  source/ with millefeuille; its build.sh applies mixed.patch (I/O only).
  `millefeuille-cg-reduce` (artifacts/cg-krylov/millefeuille/) wraps
  `main-cg`, which for nnz >= 100K (every cg-common input) runs a plain
  multi-kernel fp64 CG, NOT the paper's method (its docstring's claim
  otherwise is wrong). Keep it as a reference; label its numbers
  "plain fp64 path, not the paper's method".
- Mille-feuille and SPCG run in a fresh subprocess per rep, so harness time
  includes process start, CUDA init, .mtx parsing and (Mille-feuille) the
  CSR->tile conversion. Each adapter also records the code's OWN internal
  timer in params (spcg_pcg_ms / spcg_precond_ms / spcg_overall_ms_internal;
  Mille-feuille's time_cg; see each STATUS.md "Timing-boundary caveat").
  Report BOTH numbers.
- SPCG stops on an ABSOLUTE squared residual (tol=1e-12f, fp32), so it
  normally runs its full 1000 iterations; its fp32 solution may miss the
  1e-5 relres gate on harder inputs. That is a result.
- BootCMatchGX timed out (rc=124) in the earlier H100 run
  (bench/H100_baselines.md); it passed smoke on A100. If it hangs here,
  record which input and the last log lines.
- On the earlier H100 real-matrix run, PERKS and Mille-feuille went 0/3 on
  bcsstk17/18/25 (50 unpreconditioned steps do not reach the reference
  tolerance on ill-conditioned matrices). Similar failures on apache1 /
  cfd1 are expected; on jnlbrng1 / torsion1 / poisson they are NOT
  expected and must be reported with the error values.
- The three baselines are NEW and have never been compiled. They are
  in-process (CUDA events, solve only), while the four paper adapters are
  subprocess wall-clock. So a baseline's harness time is NOT comparable to
  a paper adapter's harness time: compute speedups from the paper code's
  OWN internal timer (perks_reported_ms; Mille-feuille's
  millefeuille_reported_ms_per_iter; spcg_overall_ms_internal;
  BootCMatchGX's internal setup+solve time if its params record one)
  against the baseline's harness time, and say which numbers you used.
- cusparse-ilu0-cg stops at relative residual 1e-6 (cap 2000) while SPCG
  runs its own absolute rule (usually all 1000 steps). Report both
  iteration counts next to the times.
- If a baseline fails to BUILD, fixing its build is in scope: these are our
  code (bridge.cu, build.sh), not a paper's. Keep fixes to build flags,
  include/link paths and CUDA API-version guards, note each fix in a new
  STATUS.md in that directory, and do not change the solver logic or stop
  rules. AmgX v2.5.0 is a long build (tens of minutes); if its CMake cannot
  find cuBLAS/cuSPARSE under the HPC-SDK math_libs layout, pass the
  KB_CUDA_MATHLIBS paths to CMake.

GPU CHOICE: A100 FOR EVERYTHING. Builds and --smoke gates on the default
MIG slice (no -g flag; a100_1g.5gb, sm_80); every REAL run on a full card
(-g a100). Do not use -g h100 in this pass. Older result JSONs under
bench/results/ are history: do not mix them into this pass's report.

GROUND RULES (same as bench/REPRODUCE.md)
- Never loosen a correctness gate, tolerance, or protocol
  (benchspecs/cg-krylov/spec.yaml, bench/kernelbench/). A failed gate is a
  result, not a bug to work around. Do not pass --warmup/--reps overrides
  on the real runs.
- Never edit a paper's code under any artifacts/cg-krylov/*/source/
  checkout. Build-system fixes only (arch flags, include/link paths,
  CUDA-version guards); after such a fix run `git -C source diff >
  source.patch` in that artifact directory and set `patch=source.patch` in
  source.provenance. bootcmatchgx/source is a symlink to
  artifacts/multigrid/bootcmatchgx/source (shared build).
- No machine-specific paths in scripts: use the KB_* variables from
  bench/env.sh.
- All GPU work goes through bench/gpu_run.sh; nothing GPU-related runs on
  the login node (it has no GPU). Each call is one blocking `srun` job; it
  sources env.sh and runs the quoted command with cwd = bench/, so paths
  INSIDE the quotes are relative to bench/. Keep the quotes single so $PY
  expands on the compute node. `bench/gpu_run.sh [-g <gres>] [-t MIN]
  [-m MEM] -- '<cmd>'`, no SLURM account flag; if srun rejects a job for a
  missing account or a memory request, that is a finding to report, not
  something to paper over by guessing.
- Bound every filesystem search to this repository (bench/CLAUDE.md).
- Read an artifact's STATUS.md before touching it; append to it, never
  rewrite its history (see step 5).

STEP 1: ENVIRONMENT (should already be set up on zaratan)
  source bench/env.sh
  $PY bench/check_env.py       # expect 0 MISSING
  (cd bench && ./smoke_all.sh) # CPU only; confirm cg-krylov's lines are green
  (cd bench && $PY -c "from kernelbench.domains import solvers as S; \
      print(S.expand_workloads(['sweep:cg-common']))")
    expect the 8 names in the table above.
If check_env.py reports MISSING items, smoke_all.sh fails on cg-krylov, or
the sweep does not expand, STOP and report that first.

STEP 2: BUILD (multigrid/bootcmatchgx first: cg-krylov/bootcmatchgx only
verifies that shared binary)
  bench/gpu_run.sh -t 60 -- 'bash artifacts/multigrid/bootcmatchgx/build.sh'
  for a in perks millefeuille millefeuille-mixed spcg bootcmatchgx cusparse-cg; do
    bench/gpu_run.sh -t 30 -- "bash artifacts/cg-krylov/$a/build.sh"
  done
  bench/gpu_run.sh -t 180 -m 32G -- 'bash artifacts/cg-krylov/amgx/build.sh'
(cusparse-cg's build also serves cusparse-ilu0-cg.) Record exit code and the
last lines per build. A paper build failure is a result (BUILD-FAILED with
the first real error line in that STATUS.md); for a baseline build failure
see KNOWN LIMITATIONS.
  (cd bench && $PY -m kernelbench.runner --kernel cg-krylov --list)
    confirm all eight adapters show as built ("ok").

STEP 3: SMOKE GATES (MIG slice), one job per group:
  bench/gpu_run.sh -t 30 -- '$PY -m kernelbench.runner --kernel cg-krylov \
      --variant cg-kernel-fixed-iter --keep-going \
      --impl perks-cg-cached,millefeuille-cg-mixed,millefeuille-cg-reduce,cusparse-cg --smoke'
  bench/gpu_run.sh -t 30 -- '$PY -m kernelbench.runner --kernel cg-krylov \
      --variant cg-e2e-ilu0-to-convergence --precision fp32 --keep-going \
      --impl spcg-ilu0-cg,cusparse-ilu0-cg --smoke'
  bench/gpu_run.sh -t 30 -- '$PY -m kernelbench.runner --kernel cg-krylov \
      --variant cg-e2e-ilu0-to-convergence --precision fp64 --keep-going \
      --impl bootcmatchgx-cg-fcg-bcmg,amgx-pcg-amg --smoke'
The four paper adapters are expected to PASS on every smoke matrix (they
did on 2026-09-09). If one fails, stop and report the error value -- do not
proceed to the real runs with a failing implementation. A baseline that
fails smoke: debug it within the KNOWN LIMITATIONS rules (it is our code);
if it still fails, drop it from the real runs and report why.

STEP 4: THE SHARED-INPUT RUNS (full A100, one job per input per group)
One job per (group, input) keeps a hang or a slow input from killing the
rest, and lets you resume: skip any job whose --out file already exists.
Run this loop on the LOGIN node (it only submits via gpu_run.sh,
sequentially), in the background with a log (nohup ... > cg.log 2>&1 &):

  for m in poisson-3d-64 poisson-3d-128 jnlbrng1 torsion1 finan512 \
           shallow_water2 apache1 cfd1; do
    mem=32G; [ "$m" = poisson-3d-128 ] && mem=64G
    f=results/cg_common_fixed_$m.json
    [ -e bench/$f ] || bench/gpu_run.sh -g a100 -t 240 -m $mem -- \
      "\$PY -m kernelbench.runner --kernel cg-krylov \
        --variant cg-kernel-fixed-iter --keep-going \
        --impl perks-cg-cached,millefeuille-cg-mixed,millefeuille-cg-reduce,cusparse-cg \
        --matrices $m --out $f"
    f=results/cg_common_conv32_$m.json
    [ -e bench/$f ] || bench/gpu_run.sh -g a100 -t 240 -m $mem -- \
      "\$PY -m kernelbench.runner --kernel cg-krylov \
        --variant cg-e2e-ilu0-to-convergence --precision fp32 --keep-going \
        --impl spcg-ilu0-cg,cusparse-ilu0-cg \
        --matrices $m --out $f"
    f=results/cg_common_conv64_$m.json
    [ -e bench/$f ] || bench/gpu_run.sh -g a100 -t 240 -m $mem -- \
      "\$PY -m kernelbench.runner --kernel cg-krylov \
        --variant cg-e2e-ilu0-to-convergence --precision fp64 --keep-going \
        --impl bootcmatchgx-cg-fcg-bcmg,amgx-pcg-amg \
        --matrices $m --out $f"
  done

(Double quotes here so $m expands on the login node; \$PY stays literal
and expands on the compute node.) Order is cheapest first; poisson-3d-128
is the slowest (Mille-feuille's tile conversion is re-done on every rep).
If a job hits its 240-min limit, record which implementation it was on
(the log shows the "running <impl> <input>" line) and re-run that input
with the remaining implementation only, a different --out, and -t 480.
Record for every job the `[gpu_run] host=... gpu=... job=...` line, the
"N/M runs valid (k unsupported, c crashed)" line and the conforming line.

STEP 5: RECORD
Append to each artifact's STATUS.md (perks, millefeuille, spcg,
bootcmatchgx; create one for millefeuille-mixed, cusparse-cg,
cusparse-ilu0-cg and amgx) a section
  ## cg-common runs on zaratan A100 (<YYYY-MM-DD>)
with: build outcome; smoke-gate outcome; per input: harness median time,
the code's own internal time (from params), iterations_actual, achieved
relative residual, gate pass/fail, conforming; crashes/timeouts with the
last log lines. Never delete or rewrite earlier STATUS.md text -- put the
new section first and keep the old text after "Earlier outcome, kept for
the record:".

STEP 6: REPORT
  (cd bench && $PY -m kernelbench.report results/cg_common_fixed_*.json \
      --html results/cg_report_fixed_a100.html)
  (cd bench && $PY -m kernelbench.report results/cg_common_conv*_*.json \
      --html results/cg_report_conv_a100.html)
Then write a short summary for the user with four tables:
  (a) fixed-iter group: per input, PERKS vs Mille-feuille (mixed, the
      paper's method) vs Mille-feuille (plain fp64 path) vs cusparse-cg --
      ms per CG step from each code's own timer AND from the harness,
      GFLOP/s, achieved relres, pass/fail, and each paper's speedup over
      cusparse-cg (internal time per step vs baseline time per step, see
      KNOWN LIMITATIONS). Note the step counts differ (mixed: 100, others:
      50) and that mixed-precision relres must be read next to the fp64
      codes' relres.
  (b) to-convergence fp32: per input, SPCG vs cusparse-ilu0-cg -- time to
      solution (internal and harness), setup time (SPCG precond ms;
      cusparse_setup_ms_internal), iterations, achieved relres, pass/fail,
      SPCG's speedup over the baseline.
  (c) to-convergence fp64: the same for BootCMatchGX vs amgx-pcg-amg
      (amgx_setup_ms / amgx_solve_ms).
  (d) one coverage grid: 8 inputs x 8 implementations, each cell PASS /
      FAIL (relres) / CRASH / TIMEOUT.
Never rank an implementation from one group against one from the other.
For each paper, put its measured speedup over its baseline next to the
paper's own claim (Mille-feuille: 3-5x over cuSPARSE on A100; SPCG: 1.23x /
1.65x iterative phase, 1.68x / 3.73x end-to-end over non-sparsified ILU(0);
BootCMatchGX: up to 2x over AmgX; PERKS's paper used Ginkgo, which is not
integrated, so its cuSPARSE speedup has no paper number to compare to), and
flag any disagreement of more than 2x.
Do NOT commit or push anything (no git add, git commit or git push). Leave
every change (STATUS.md notes, any build fix) uncommitted in the working
tree. The deliverables are the two HTML reports above plus your summary; end
by giving both reports' paths and listing every file you changed.
```
