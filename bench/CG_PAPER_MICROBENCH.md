# cg-krylov paper-faithful micro-benchmarks on zaratan: prompt for a coding agent

> Scope: test each cg-krylov paper's OWN claim, the way that paper measured
> it, on one full A100. This is separate from the shared-input comparison in
> `bench/CG_ON_ZARATAN.md` (same inputs and rules for everyone). Here every
> paper gets its own inputs, step count, right-hand side, precision and the
> baseline it claims to beat. Added 2026-10-23.

## Why this exists

The 2026-10-23 shared-input run (`results/cg_common_*.json`) could not
answer "do the papers' claims hold", because:

- **SPCG** — the adapter wrapped `ilu0_gpu/nonsp`, SPCG's own NON-sparsified
  baseline. The paper's method is `ilu0_gpu/sp`, which needs pre-sparsified
  matrices (MATLAB in the artifact). Now wrapped as `spcg-sp-ilu0-cg`, with a
  Python port of the sparsification script.
- **Mille-feuille** — its `time_cg` timer also covers a block of
  cudaMalloc/cudaMemcpy before the solve kernel. The patch now also prints the
  upstream kernel-only timer (`time_spmv`), used as the headline number.
- **PERKS** — 50 steps instead of the paper's 10,000, and a cuSPARSE
  baseline instead of the paper's comparison. Now: 10,000 steps, plus PERKS's
  own non-persistent `--baseline` build (`perks-cg-baseline`), which isolates
  exactly what PERKS adds. A `-nan` in PERKS's output crashed the adapter; fixed.
- **BootCMatchGX** — only the subprocess wall clock (mostly start-up) was
  recorded. The adapter now reads BootCMatchGX's own profiler from its
  `--info` file: AMG setup (`prec_setup`) and solve (`TOTAL SOLVE TIME`).

## The four micro-benchmarks

| Paper | Claim tested | Runs | Settings | Inputs (`sweep:`) | Compared by |
|---|---|---|---|---|---|
| PERKS (ICS'23) | a persistent kernel caching A and vectors on chip makes memory-bound CG faster | `perks-cg-cached`, `perks-cg-baseline`, `cusparse-cg` | fp64, `KB_CG_STEPS=10000`, `KB_CG_RHS=ones` | `perks-paper` (its Table 4, 20 matrices) | ms per step, internal timers; cached vs PERKS baseline is the claim; vs cuSPARSE is context (paper used Ginkgo) |
| Mille-feuille (SC'24) | single-kernel, tile-wise mixed precision CG is 3–5x faster than cuSPARSE on A100 | `millefeuille-cg-mixed`, `cusparse-cg`, `millefeuille-cg-reduce` | fp64 interface, 100 steps (hard-coded in MF; `KB_CG_STEPS=100` for the others), `KB_CG_RHS=Aones` | `millefeuille-paper` (16 SPD, small to ~10M nnz) | ms per step: MF kernel-only timer vs cuSPARSE internal; plus relres after 100 steps |
| SPCG (SC'25) | sparsifying A before ILU(0) speeds up PCG: 1.23x / 1.65x iterative phase, 1.68x / 3.73x end-to-end | `spcg-sp-ilu0-cg` at ratios 0.01, 0.05, 0.10; `spcg-ilu0-cg` (= SPCG's nonsp baseline); `cusparse-ilu0-cg` (independent check) | fp32, `KB_CG_RHS=ones`, SPCG's own stop rule (abs 1e-12, 1000 cap) in both of its drivers | `spcg-paper` (20 from its 114-matrix list) | SPCG's own CUDA-event "PCG time" and "overall time", sp vs nonsp |
| BootCMatchGX (TPDS'23) | AMG-preconditioned FCG up to 2x faster than AmgX | `bootcmatchgx-cg-fcg-bcmg`, `amgx-pcg-amg` | fp64, rtol 1e-6, `KB_CG_RHS=ones` (paper: f = 1) | `bootcmatchgx-paper` (3D 7-pt Poisson 64³, 96³, 128³, 160³) | internal setup / solve / total, iterations |

Known gaps, by design:
- SPCG's per-matrix ratio selector (`alg_ilu0.py`) needs MATLAB-computed
  statistics and is not ported. All three ratios are run; the best valid one
  per matrix is an upper bound on what the selector could pick. Report every
  ratio, not only the best.
- BootCMatchGX's paper is multi-GPU (P100); this is one A100, with a size
  sweep in place of weak scaling. AmgX runs NVIDIA's shipped
  `PCG_AGGREGATION_JACOBI.json` (the paper does not say which AmgX config it used).
- Mille-feuille's paper used all 230 SPD SuiteSparse matrices; this is a
  16-matrix spread. Its kernel needs the work split to fit 6912 warps; inputs
  that do not are UNSUPPORTED (a result).
- PERKS's paper compared against Ginkgo, which is not integrated.

`bench/artifacts/cg-krylov/paper_summary.py` turns the result files into one
table per paper (internal timers only, geomeans, the paper's number next to
ours).

---

```text
You are working in a clone of HPC-KernelBench on zaratan, branch crylov.
Run every command from the REPOSITORY ROOT unless it says otherwise.

GOAL. Run the four paper-faithful cg-krylov micro-benchmarks described in
bench/CG_PAPER_MICROBENCH.md (above this block) on one full A100 each job,
and report, per paper, whether its own claim holds on our hardware --
including every failure, honestly. Do not "fix" the benchmark (spec,
harness, gates, a paper's own solver code) to make a result pass.
On zaratan GPU clocks cannot be locked, so every result is stamped
`conforming: false` ("GPU clocks not locked"): expected, do not debug it.

READ FIRST: CLAUDE.md; bench/CLAUDE.md; bench/ARTIFACT_GUIDE.md (rules
binding); bench/CG_PAPER_MICROBENCH.md (this file, the part above the
block); the docstrings of these adapters under bench/artifacts/cg-krylov/:
perks, perks-baseline, millefeuille-mixed, millefeuille, spcg, spcg-sp,
bootcmatchgx, cusparse-cg, cusparse-ilu0-cg, amgx; and
bench/artifacts/cg-krylov/paper_summary.py.

NEW SINCE THE LAST RUN (never built or run before -- check them first):
- spcg-sp/ (spcg-sp-ilu0-cg): SPCG's sparsified driver; shares spcg/source;
  build.sh applies sp.patch (I/O only). Its adapter writes the matrix and the
  sparsified matrix as lower-triangle, column-major "symmetric" .mtx files,
  because SPCG's sp reader accepts nothing else.
- perks-baseline/ (perks-cg-baseline): PERKS binary with --baseline.
- millefeuille-mixed/mixed.patch: now also prints time_kernel. build.sh
  swaps the older applied patch for this one and rebuilds.
- bootcmatchgx adapter: parses bcmgx_setup_ms_internal /
  bcmgx_solve_ms_internal from its --info file.
- perks adapter: no longer crashes on "-nan".
- Run-level knobs read by every cg-krylov adapter (kernelbench/domains/
  solvers.py cg_rhs / cg_steps): KB_CG_RHS = uniform | ones | Aones,
  KB_CG_STEPS = fixed-iteration step count; KB_SPCG_RATIO for spcg-sp;
  KB_MF_TIMEOUT (seconds, default 600) for millefeuille-cg-mixed.
If one of these fails to BUILD or fails its smoke gate with a clear
adapter-side error (wrong file name, parse error, wrong path), fixing it is
in scope: it is our code. Keep fixes to adapter/patch I/O, build flags and
paths; never change solver numerics or stop rules; write what you changed
into that directory's STATUS.md (create it if missing).

GROUND RULES (same as bench/CG_ON_ZARATAN.md)
- All GPU work through bench/gpu_run.sh; nothing GPU-related on the login
  node. Quote commands with single quotes so $PY expands on the compute
  node; inside the quotes paths are relative to bench/.
- Never loosen a gate or tolerance; no --warmup/--reps overrides on real runs.
- Never edit a paper's solver code under any source/ checkout.
- Bound filesystem searches to this repository.
- Do NOT commit or push. Leave every change uncommitted.

STEP 1: ENVIRONMENT
  source bench/env.sh
  $PY bench/check_env.py
  (cd bench && $PY -c "from kernelbench.domains import solvers as S; \
      [print(k, len(S.expand_workloads(['sweep:'+k]))) for k in \
      ['perks-paper','millefeuille-paper','spcg-paper','bootcmatchgx-paper']]")
    expect 20, 16, 20, 4.

STEP 2: BUILD (MIG slice is fine)
  for a in perks perks-baseline millefeuille millefeuille-mixed spcg spcg-sp \
           cusparse-cg; do
    bench/gpu_run.sh -t 30 -- "bash artifacts/cg-krylov/$a/build.sh"
  done
  (bootcmatchgx and amgx are already built; re-run their build.sh only if
  `--list` below says otherwise.)
  (cd bench && $PY -m kernelbench.runner --kernel cg-krylov --list)
    confirm perks-cg-cached, perks-cg-baseline, millefeuille-cg-mixed,
    millefeuille-cg-reduce, spcg-ilu0-cg, spcg-sp-ilu0-cg, cusparse-cg,
    cusparse-ilu0-cg, bootcmatchgx-cg-fcg-bcmg, amgx-pcg-amg are available.

STEP 3: SMOKE (MIG slice), one job per paper:
  bench/gpu_run.sh -t 30 -- 'KB_CG_STEPS=200 KB_CG_RHS=ones $PY -m kernelbench.runner \
      --kernel cg-krylov --variant cg-kernel-fixed-iter --keep-going \
      --impl perks-cg-cached,perks-cg-baseline,cusparse-cg --smoke'
  bench/gpu_run.sh -t 30 -- 'KB_CG_STEPS=100 KB_CG_RHS=Aones $PY -m kernelbench.runner \
      --kernel cg-krylov --variant cg-kernel-fixed-iter --keep-going \
      --impl millefeuille-cg-mixed,cusparse-cg --smoke'
  bench/gpu_run.sh -t 30 -- 'KB_CG_RHS=ones KB_SPCG_RATIO=0.05 $PY -m kernelbench.runner \
      --kernel cg-krylov --variant cg-e2e-ilu0-to-convergence --precision fp32 \
      --keep-going --impl spcg-sp-ilu0-cg,spcg-ilu0-cg --smoke'
  bench/gpu_run.sh -t 30 -- 'KB_CG_RHS=ones $PY -m kernelbench.runner \
      --kernel cg-krylov --variant cg-e2e-ilu0-to-convergence --precision fp64 \
      --keep-going --impl bootcmatchgx-cg-fcg-bcmg,amgx-pcg-amg --smoke'
For each: confirm in the JSON params that the new fields are present
(perks_reported_ms_per_iter; millefeuille_reported_ms with time_kernel, i.e.
no millefeuille_timer_note; spcg_pcg_ms and spcg_ratio for spcg-sp;
bcmgx_setup_ms_internal and bcmgx_solve_ms_internal) and that params["rhs"]
is the requested one. A smoke matrix that fails the GATE is a result; a
crash or a missing field is a bug to fix (in scope, see above) before the
real runs.

STEP 4: REAL RUNS (full A100, one job per input; skip a job whose --out
file exists; run the loops on the LOGIN node in the background with a log,
nohup ... > paper.log 2>&1 &). Double quotes so $m expands on the login
node; \$PY stays literal.

  # PERKS: 10,000 steps, b = ones
  for m in Trefethen_2000 msc01440 fv1 msc04515 Muu crystm02 shallow_water2 \
           finan512 cbuckle G2_circuit thermomech_dM ecology2 tmt_sym consph \
           crankseg_1 bmwcra_1 hood BenElechi1 crankseg_2 af_1_k101; do
    f=results/paper_perks_$m.json
    [ -e bench/$f ] || bench/gpu_run.sh -g a100 -t 240 -m 64G -- \
      "KB_CG_STEPS=10000 KB_CG_RHS=ones \$PY -m kernelbench.runner \
        --kernel cg-krylov --variant cg-kernel-fixed-iter --keep-going \
        --impl perks-cg-cached,perks-cg-baseline,cusparse-cg \
        --matrices $m --out $f"
  done

  # Mille-feuille: 100 steps, b = A*ones
  for m in Trefethen_2000 msc01440 fv1 msc04515 Muu crystm02 jnlbrng1 torsion1 \
           shallow_water2 finan512 cbuckle G2_circuit thermomech_dM ecology2 \
           consph af_1_k101; do
    f=results/paper_mf_$m.json
    [ -e bench/$f ] || bench/gpu_run.sh -g a100 -t 240 -m 64G -- \
      "KB_CG_STEPS=100 KB_CG_RHS=Aones \$PY -m kernelbench.runner \
        --kernel cg-krylov --variant cg-kernel-fixed-iter --keep-going \
        --impl millefeuille-cg-mixed,cusparse-cg,millefeuille-cg-reduce \
        --matrices $m --out $f"
  done

  # SPCG: fp32, b = ones; the baselines once, the sparsified driver per ratio
  for m in Muu crystm02 fv1 msc04515 msc10848 bcsstk17 Kuu LFAT5000 Dubcova2 \
           gyro jnlbrng1 torsion1 finan512 shallow_water2 thermomech_TC \
           G2_circuit parabolic_fem thermal1 apache1 cfd1; do
    f=results/paper_spcg_${m}_base.json
    [ -e bench/$f ] || bench/gpu_run.sh -g a100 -t 120 -m 64G -- \
      "KB_CG_RHS=ones \$PY -m kernelbench.runner --kernel cg-krylov \
        --variant cg-e2e-ilu0-to-convergence --precision fp32 --keep-going \
        --impl spcg-ilu0-cg,cusparse-ilu0-cg --matrices $m --out $f"
    for r in 0.01 0.05 0.1; do
      f=results/paper_spcg_${m}_sp$r.json
      [ -e bench/$f ] || bench/gpu_run.sh -g a100 -t 120 -m 64G -- \
        "KB_CG_RHS=ones KB_SPCG_RATIO=$r \$PY -m kernelbench.runner \
          --kernel cg-krylov --variant cg-e2e-ilu0-to-convergence \
          --precision fp32 --keep-going --impl spcg-sp-ilu0-cg \
          --matrices $m --out $f"
    done
  done

  # BootCMatchGX vs AmgX: 3D Poisson size sweep, fp64, b = ones
  for m in poisson-3d-64 poisson-3d-96 poisson-3d-128 poisson-3d-160; do
    f=results/paper_bcmg_$m.json
    [ -e bench/$f ] || bench/gpu_run.sh -g a100 -t 240 -m 96G -- \
      "KB_CG_RHS=ones \$PY -m kernelbench.runner --kernel cg-krylov \
        --variant cg-e2e-ilu0-to-convergence --precision fp64 --keep-going \
        --impl bootcmatchgx-cg-fcg-bcmg,amgx-pcg-amg --matrices $m --out $f"
  done

If a job hits its time limit, record which implementation it was on (the
"running <impl> <input>" line) and re-run the rest with a different --out
and a longer -t. If srun rejects -m 96G, use the largest it accepts and say so.
Record for every job the [gpu_run] host/gpu/job line and the
"N/M runs valid (k unsupported, c crashed)" line.

STEP 5: SUMMARY
  (cd bench && $PY artifacts/cg-krylov/paper_summary.py results \
      > results/cg_paper_summary.md)
  (cd bench && $PY -m kernelbench.report results/paper_*.json \
      --html results/cg_paper_report_a100.html)
Then write the user a short summary, one section per paper:
  - the paper's claim (number), what we measured (geomean and range, from
    paper_summary.py, internal timers only), and a one-line verdict:
    REPRODUCED (within 2x of the claim, same direction), PARTLY (same
    direction, more than 2x off), NOT REPRODUCED (opposite direction), or
    UNTESTABLE (with why);
  - every failure: gate FAIL (with relres), crash, timeout, UNSUPPORTED;
  - for SPCG, which ratio won per matrix and whether sparsification changed
    the iteration count;
  - for PERKS, whether the cached-vs-baseline speedup falls as the matrix
    stops fitting on chip (CSR MB column; an A100 has ~40 MB of registers +
    shared memory in total);
  - for Mille-feuille, the relres after 100 steps next to cuSPARSE fp64's.
Append a "## Paper micro-benchmark (zaratan A100, <YYYY-MM-DD>)" section to
each involved adapter's STATUS.md (new section first, old text kept).
Do NOT commit or push. End by listing every file you changed and the paths
of results/cg_paper_summary.md and results/cg_paper_report_a100.html.
```
