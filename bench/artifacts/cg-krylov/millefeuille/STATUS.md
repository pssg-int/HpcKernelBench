# millefeuille (Mille-feuille) — cg-krylov

**Status: BUILT+GATED** (smoke: 3/3 valid; representative `cant` check: gate
correctly FAILS at the spec's default maxiter=50, matching the harness's own
`scipy-cg` reference on the identical matrix/protocol — see "Representative
check" below for why this is the honest result, not a bug).

- Paper: "Mille-feuille: A Tile-Grained Mixed Precision Single-Kernel
  Conjugate Gradient Solver on GPUs", SC'24.
  `PAPER_KEY = conf/sc/YangZNJS0T024` (title-matched in
  `../../../output/included.json`).
- Artifact: https://github.com/SuperScientificSoftwareLaboratory/Mille-feuille
- Commit: `1421c29fdb38fec3ba888c4f39109b93412b9e08` (2024-10-20),
  `git clone --depth 1`, `.git` intact for provenance.
- Toolchain: `nvcc` `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc`
  (release 12.9, V12.9.41), GPU `sm_80` (A100-PCIE-40GB), python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`. No CUDA-arch build-flag
  changes from the artifact's own Makefile (`-arch=compute_80 -code=sm_80`
  already matches this machine's GPU).

## What the artifact actually is

`src/main-cg.cu` is a single ~1700-line file, `./main-cg matrix.mtx` CLI, no
library API. It dispatches by `nnz` (of the ORIGINAL, untruncated matrix) to
one of three CG solver implementations:

- `nnz < 10000` -> `cg_solve_inc`
- `10000 <= nnz < 100000` -> `cg_solve_sync`
- `nnz >= 100000` -> `cg_solve_reduce` — **the paper's actual contribution**
  (tile-grained mixed precision, single fused CUDA kernel via atomics,
  on-chip dynamic precision conversion). Every matrix in this track's
  `recommended_subset` has `nnz >= 100000`(nnz range [1e4,4e8] per the spec,
  and the subset is real structural matrices well above 1e5 nnz), so
  `cg_solve_reduce` is what any real (non-smoke) run of this adapter
  exercises. `cg_solve_inc`/`cg_solve_sync` are exercised only by this
  track's tiny synthetic smoke matrices.

`MAT_VAL_TYPE=double`, `MAT_VAL_LOW_TYPE=float` (`src/common.h`) —
`PRECISIONS=["fp64"]`: the interface/accumulation type this adapter can
observe (RHS, x, residual) is fp64. The paper's actual mixed-precision
technique — an internal fp32 `Val_Low` companion array used for tile-wise
on-chip dynamic precision downcasting inside `cg_solve_reduce`'s SpMV — is
NOT a precision this adapter selects or can disable; it runs unconditionally
whenever `cg_solve_reduce` is exercised. This is stated here rather than
modeled as a second `PRECISIONS` entry, since there is no fp64-only mode to
select instead.

## Patch 1 — the real, independently-documented artifact bug (rule 3: minimal, recorded)

`benchspecs/cg-krylov/spec.yaml`'s `cg-kernel-fixed-iter` variant exists
*specifically* to fix this bug (quoted verbatim in the spec's own
`notes_on_fairness`), and `survey.md`'s Divergences section independently
confirms it by reading this exact file. Both are quoted below for the paper
trail; this patch fixes the underlying code.

Inside `cg_solve_reduce` (the paper's actual contribution, `src/main-cg.cu`
original lines ~245/284/325):

1. **`while (iterations < 10000)` ignored the `maxiter` argument** the
   function signature already receives — `main()` calls
   `cg_solve_reduce(..., maxiter=10, ...)`, but the loop always ran 10000
   iterations regardless. `iterations` was already declared (`int iterations
   = 0;`, ~line 130) and incremented (`iterations++;`, ~line 278) — a
   one-token fix:
   ```c
   -    while (iterations < 10000)
   +    while (iterations < maxiter)
   ```
2. **`printf("time_cg=%lf ms\n",time_cg/100);`** divided by the literal
   integer `100`, not the true iteration count (10000 before this patch, or
   `maxiter` after it) — so the printed "per-iteration" time was off by
   ~100x at the artifact's own default settings. Fixed by printing the RAW
   loop time plus the true iteration count, and doing the per-iteration
   division in the Python adapter instead, where it is exact and auditable:
   ```c
   -    printf("time_cg=%lf ms\n",time_cg/100);
   +    printf("time_cg=%lf ms iterations=%d\n", time_cg, iterations);
   ```
3. **Same `time_cg/100` bug in the CSV-writing `sprintf`**, fixed the same
   way (raw `time_cg`):
   ```c
   -    sprintf(s, "%d,%.3f,%d,%e,%e\n", 100, time_cg/100, nnzR, l2_norm,sqrt(snew));
   +    sprintf(s, "%d,%.3f,%d,%e,%e\n", 100, time_cg, nnzR, l2_norm,sqrt(snew));
   ```
   The **leading `100` literal (1st CSV field) was deliberately left
   untouched**: it is NOT iteration-related. `cg_solve_inc` and
   `cg_solve_sync` (this file's other two solver paths, unpatched) write
   this identical literal `100` in the same CSV column despite having no
   `maxiter=10000`-style host loop at all (both are single persistent-kernel
   launches with a device-side iteration budget — confirmed by reading both
   functions in full; no `while (iterations < LITERAL)` host loop pattern
   exists in either). No CSV header row exists anywhere in this artifact to
   confirm the field's intended meaning, so it is left as an undocumented
   placeholder rather than guessed at.

**Only `cg_solve_reduce` was patched.** `cg_solve_inc`/`cg_solve_sync` do not
share this bug pattern (verified by reading both in full — see above), and
they are not this paper's headline contribution or the spec's cited bug, so
patching them was out of scope. Readers of a smoke-test run (which routes to
`cg_solve_inc`/`cg_solve_sync`) should not expect Patch 1's effects to be
visible there — `millefeuille_code_path` is written into every result's
`params` precisely so this is never ambiguous.

## Patch 2 — x dump (required, for independent correctness verification)

`main()`, right after the `if/else if/else if` dispatch chain, now dumps the
final host `X` array (confirmed `n` doubles, holding the solved x after any
of the three `cg_solve_*` calls returns, via `cudaMemcpy(x, k_x, ...,
cudaMemcpyDeviceToHost)` inside each solver) when `MF_X_OUT_FILE` is set —
a no-op otherwise:
```c
{
    const char* x_out = getenv("MF_X_OUT_FILE");
    if (x_out) {
        FILE* f = fopen(x_out, "wb");
        if (f) { fwrite(X, sizeof(double), n, f); fclose(f); }
    }
}
```

## Patch 3 — RHS override (applied and used by default; low-risk)

`main()`'s own RHS is manufactured: `X[i]=1` for all `i`, then
`Y_golden = A @ X` (i.e. `b = A @ ones(n)`), not this track's spec convention
(`b ~ U(-1,1)`, seed=42). Right after that computation and before the solver
dispatch, an opt-in override was added:
```c
{
    const char* rhs_path = getenv("MF_RHS_FILE");
    if (rhs_path) {
        FILE* f = fopen(rhs_path, "rb");
        if (f) { size_t nread = fread(Y_golden, sizeof(double), n, f); fclose(f); (void)nread; }
    }
}
```
**Applied and used by default** (not left as a documented-only deviation):
the adapter always sets `MF_RHS_FILE` to a freshly generated `b ~ U(-1,1)`
seed=42 vector. Two independent reasons:

1. Matches this track's spec convention directly, rather than reporting
   against the artifact's own manufactured self-check RHS.
2. **Sidesteps a separate, pre-existing artifact quirk** discovered while
   verifying this patch: `main()` truncates `n` down to a multiple of
   `BLOCK_SIZE=16` (`n = (n / BLOCK_SIZE) * BLOCK_SIZE;`) *before* allocating
   `X`/`Y_golden` at the truncated size, but the *default* (unpatched)
   `Y_golden[i] += Val[j] * X[ColIdx[j]]` loop indexes `X` with `ColIdx`
   values drawn from the ORIGINAL, untruncated matrix — an out-of-bounds
   host read on `X` whenever a matrix's dimension is not a multiple of 16
   and some in-range row has a nonzero in a truncated-away column (e.g.
   `cant.mtx`: 62451 -> 62448, 3 dropped; this is generically true for
   almost every real SuiteSparse matrix, whose dimensions are essentially
   never multiples of 16). Supplying `Y_golden` directly via `MF_RHS_FILE`
   bypasses that loop entirely and gives a well-defined, reproducible RHS.
   Not independently confirmed with ASan (out of scope for a login-node
   check), but the code-level OOB-read condition is real and easy to
   trigger; flagged here as an additional, unpatched, out-of-scope finding
   (would require touching `main()`'s array-sizing logic, not a minimal
   build-system fix).

**Consequence of the `BLOCK_SIZE` truncation** (pre-existing, not introduced
by any patch here): for any matrix whose row/col count is not a multiple of
16, Mille-feuille silently solves a slightly smaller system than requested.
The adapter computes `n_truncated = (n // 16) * 16` and performs its own
independent residual check against `A`'s top-left
`n_truncated x n_truncated` principal submatrix (the best available
approximation of what the tile format can represent), and records both `n`
values plus a `millefeuille_truncation_note` in every result's `params`
whenever truncation actually occurs. Also affects the harness's own generic
GFLOP/s cost rule (`kernelbench/domains/solvers.py`'s `_cost_cg`), which
uses the workload's ORIGINAL (untruncated) `n`/`nnz` — a slight overcount,
negligible for this track's matrices (at most 15 rows/cols out of tens of
thousands).

`git -C source diff --stat`: `src/main-cg.cu | 80 +++++++++++++++++++++++++++++++++++++++++++++++++++++-----`
(1 file changed, 74 insertions, 6 deletions — all inside `cg_solve_reduce`
and `main()`; no other function touched).

## build.sh

The artifact's own `Makefile` hardcodes
`CUDA_INSTALL_PATH=/usr/local/cuda-12.0`, which does not exist on this
machine — ARTIFACT_GUIDE.md rule 3 explicitly allows this class of fix.
Rather than fight the Makefile's path logic, `build.sh` invokes the
equivalent `nvcc` command directly (identical flags:
`-O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80
-Xcompiler -fpermissive -Xcompiler -fopenmp -maxrregcount=32`), with
`-I`/`-L` resolved from `command -v nvcc`'s own toolkit prefix rather than
the broken hardcoded path. `mkdir -p data` (the binary writes results to a
CWD-relative `data/cg_performance.csv`). Builds clean, no compiler errors or
warnings surfaced (the `-w` flag suppresses warnings — inherited from the
artifact's own Makefile, not added here).

## adapter.py — architecture

Per ARTIFACT_GUIDE.md rule 1 ("wrap at the finest boundary available"):
`main-cg` has no library entry point, so this adapter wraps the process
boundary. `prepare()` writes the workload's CSR out as a `.mtx`
(`scipy.io.mmwrite`, timed once as preprocessing) plus a `b~U(-1,1)` RHS
binary file (Patch 3). `run()` launches a fresh `main-cg` subprocess per
call — no in-process state to reset between the harness's correctness/
warmup/measured calls. `to_host()` reads back the dumped x (Patch 2),
independently recomputes `||b - A_trunc@x|| / ||b||` in fp64 (never trusting
the artifact's own internal residual bookkeeping), and applies the SAME
fixed-iter gate as `kernelbench.domains.solvers.ScipyCG.to_host()`:
`passed = finite and relres < 1.0`.

`iterations_actual` (parsed from the Patch-1-fixed `time_cg=... iterations=...`
stdout line, never assumed) feeds `solvers.py`'s `_cost_cg` GFLOP/s rule —
this is the exact quantity the spec calls out Mille-feuille's own artifact
for getting wrong. `millefeuille_reported_ms` (the artifact's own,
now-meaningful timer) is captured separately and explicitly labeled as NOT
the harness's canonical number.

### Timing-boundary contamination (documented per rule 1's escape hatch)

`timer()` is a plain CPU wall-clock `Timer` around one subprocess launch.
Because `main-cg` has no persistent state, **every** timed `run()` call
(warmup AND measured reps) re-executes: process startup, `.mtx` re-parsing
(`biio`), and the artifact's own CSR->tile-format construction
(`time_format`, timed internally by the artifact's own `gettimeofday`
bracket but NOT separated out by this adapter's `preprocessing_ms`, which
covers only writing the `.mtx`/RHS files once in `prepare()`). This is
materially more contamination than a typical subprocess-wrapped adapter: for
`cant.mtx` (62451x62451, ~4M nnz), one full subprocess call took ~6-16s wall
in testing, almost entirely tile-format construction — the actual
`cg_solve_reduce` CG loop itself, per its OWN now-correct `time_cg`, was
under 10ms for 50 iterations. Any throughput number from this adapter is
therefore measuring "subprocess + reparse + reformat + solve", not "solve"
in isolation; this is unavoidable given the artifact ships no separable
solve-only entry point, and is the single most important caveat for anyone
reading numbers from this adapter.

## Gate verification (login node, functional check only — rule 5)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
```

### 1. Raw binary sanity check (artifact's own smoke matrix)

```
cd source && ./main-cg test/add20.mtx
```
Ran cleanly: `time_cg=3.954000 ms` (nnzR=17319, so `10000 <= nnzR < 100000`
routes to `cg_solve_sync`, not `cg_solve_reduce` — expected for this tiny
shipped smoke matrix). Confirmed `MF_MAXITER`/`MF_X_OUT_FILE` env-var wiring works
standalone before touching Python (a follow-up run with
`MF_MAXITER=7 MF_X_OUT_FILE=...` produced a correctly-sized x dump).

### 2. Quick plumbing check (smoke, `cg_solve_inc`/`cg_solve_sync`)

```
$PY -m kernelbench.runner --kernel cg-krylov --variant cg-kernel-fixed-iter \
    --impl millefeuille-cg-reduce --smoke --warmup 1 --reps 1
```
**3/3 runs valid.**

| matrix | code path | n (orig->trunc) | iterations_actual | relres_achieved | gate |
|---|---|---|---|---|---|
| smoke-poisson2d-24 | cg_solve_inc | 576->576 | 50 | 2.92e-09 | PASS |
| smoke-poisson2d-48 | cg_solve_sync | 2304->2304 | 50 | 9.21e-05 | PASS |
| smoke-poisson3d-10 | cg_solve_inc | 1000->992 | 50 | 8.17e-16 | PASS |

`iterations_actual` correctly equals the configured `maxiter=50` in every
case (not a mysterious 10000) — Patch 1 confirmed working end-to-end through
the adapter, not just standalone. `millefeuille_reported_ms_per_iter` values
(~0.03-0.05 ms) are sane, no longer off by ~100x.

**Note**: none of these exercise the patched `cg_solve_reduce` function
(all nnz well under 10000/100000) — see check 3.

### 3. Representative check (`cant.mtx`, real `cg_solve_reduce` path)

```
$PY -m kernelbench.runner --kernel cg-krylov --variant cg-kernel-fixed-iter \
    --impl millefeuille-cg-reduce --matrices cant --warmup 1 --reps 1
```

Result: **correctness gate FAILS** at the spec's default `maxiter=50`:
`relres_achieved=1.0309` (>= 1.0, i.e. no net progress from x0=0 in 50
unpreconditioned iterations on this matrix). `iterations_actual=50`
(correct), `millefeuille_code_path=cg_solve_reduce` (confirmed — the
patched function). Per ARTIFACT_GUIDE.md rule 4, **the gate was not
loosened**; this is recorded as the honest result.

**This is not an adapter or artifact bug** — cross-checked against the
harness's own `scipy-cg` reference implementation on the IDENTICAL matrix
and protocol (`--impl scipy-cg --matrices cant --warmup 1 --reps 1`):
`relres_achieved=1.0208`, gate also FAILS. Both independent implementations
agree, to 2 significant figures, that unpreconditioned CG genuinely does not
decrease `cant.mtx`'s residual in 50 fixed iterations — a property of this
matrix's conditioning under the spec's `cg-kernel-fixed-iter` protocol
(no preconditioner, by design), not a defect in either implementation.

**Diagnostic (not part of the required check, one extra subprocess call,
called directly via the adapter's own `prepare/run/to_host` rather than
through the runner CLI to avoid extra warmup/rep launches)**: raising
`maxiter` to 500 makes the gate PASS: `relres_achieved=0.3167`,
`iterations_actual=500` (correct), confirming the solver genuinely
converges given more budget and that Patch 1's `maxiter` threading is
correct at multiple values, not coincidentally correct only at 50.
Additionally, the artifact's OWN internally-computed `l2_norm` for the
`maxiter=50` run (`1.030846e+00`, from `data/cg_performance.csv`) matches
this adapter's independently-computed `relres_achieved` (`1.0308455825...`)
to 6 significant figures — strong cross-validation that Patch 3's RHS
override and this adapter's residual computation are both wired correctly.

### 4. Registry check

```
$PY -m kernelbench.runner --kernel cg-krylov --list
```
`paper artifacts:` lists `ok millefeuille     millefeuille-cg-reduce` —
confirmed discovered and available.

## Other findings (out of scope, documented for the record)

- `cg_solve_inc`'s own internal `l2_norm` diagnostic (its version of the
  same post-hoc CPU-recomputed check `cg_solve_reduce` has) prints `inf` —
  observed even in a completely stock, unmodified run
  (`./main-cg test/add20.mtx` immediately after building, no env vars, no
  adapter involved: `test/add20.mtx,100,4.031,17319,inf,2.278163e-03`). This
  is a pre-existing quirk isolated to `cg_solve_inc`'s own (unpatched)
  internal `sum_ori`/`l2_norm` computation — not something introduced by any
  patch here, and not something this adapter relies on (the adapter's own
  independent residual check is a completely separate computation and does
  not exhibit this). Not investigated further (out of scope: `cg_solve_inc`
  is not this paper's contribution or the spec's cited bug).
- `main()`'s truncation of `n` to a multiple of `BLOCK_SIZE=16` (see Patch 3
  writeup) is a pre-existing artifact limitation affecting all three solver
  paths equally, not something introduced here.

## Not done

- No timing sweep across `recommended_subset` — out of scope per the
  login-node budget (rule 5); a single functional/gate check only. Full
  timed runs belong on a compute-node allocation.
- `cg_solve_inc`/`cg_solve_sync`'s identical-shaped-but-absent bug surface
  (they don't share Patch 1's specific bug, per the analysis above) was not
  further hardened; only `cg_solve_reduce` — the paper's actual contribution
  and the spec's cited function — was patched, per this task's explicit
  scope.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, on an A100-SXM4-40GB
  card), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), arch flags unchanged
  from build.sh (`-arch=compute_80 -code=sm_80`, already matches this
  GPU), Python 3.12.14. No gcc/g++ used directly (nvcc's own default host
  compiler; `-Xcompiler -fpermissive -Xcompiler -fopenmp` unchanged).
- Build: OK. Build-system changes: none beyond what was already committed
  (build.sh's direct-nvcc-invocation workaround for the artifact's own
  hardcoded, nonexistent `CUDA_INSTALL_PATH` was already in place and
  required no further changes here).
- Gate: `cg-kernel-fixed-iter`, fp64, `--smoke --warmup 1 --reps 1`
  (protocol override, matching this file's own documented command):
  **3/3 runs valid.**
  smoke-poisson2d-24 (cg_solve_inc): relres=2.915e-09, iters=50, PASS.
  smoke-poisson2d-48 (cg_solve_sync): relres=9.209e-05, iters=50, PASS.
  smoke-poisson3d-10 (cg_solve_inc): relres=8.035e-16, iters=50, PASS.
  All within noise of the originally recorded Perlmutter numbers (2.92e-09,
  9.21e-05, 8.17e-16) — the tiny difference on the third matrix is at the
  ~1e-16 machine-epsilon floor, consistent with ordinary cross-hardware
  floating-point summation-order variance, not a behavioral change. As
  before, none of these three smoke matrices exercise the patched
  `cg_solve_reduce` path (all have `nnz < 100000`); no `recommended_subset`
  real matrix was re-run here (out of scope for this reproduction pass —
  see "Not done" above, unchanged).
- Deviation from the recorded ruling: none.
- Verdict here: **BUILT+GATED**, same as the recorded ruling.
