# spcg (SPCG) — cg-krylov

**Status: BUILT+GATED**

- Paper: "Sparsified Preconditioned Conjugate Gradient Solver on GPUs" (SC'25).
  `PAPER_KEY = conf/sc/MaACSH25` (title-matched in `../../../output/included.json`,
  already confirmed before this integration started).
- Artifact: https://github.com/SwiftWare-Lab/SPCG
- Commit: `e667f3acb6e6c3bb219d4a8b520fea10e0987ff3` (2025-07-14), cloned with
  `git clone --depth 1` (history intact for provenance, working tree clean
  before patching).
- Toolchain: `nvcc` release 12.9, V12.9.41
  (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc`); `cmake`
  3.28.3 (`/usr/bin/cmake`); host compiler `g++` 14.3.0
  (`/opt/cray/pe/gcc-native/14/bin/g++`, SUSE-packaged Cray gcc-native
  module); GPU: single NVIDIA A100-PCIE-40GB, sm_80 (shared login-node GPU).
  Python: `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python` (numpy, scipy
  only needed — no torch import in this adapter).

## What is wrapped

`gpu_src/ilu0_gpu/nonsp/main.cpp` -> CMake target `conjugateGradientPrecond`
(`gpu_src/ilu0_gpu/nonsp/CMakeLists.txt`, `project(... LANGUAGES CXX)` +
`find_package(CUDAToolkit REQUIRED)`, no `.cu` files — CUDA used only via
CUDAToolkit imported targets). A single end-to-end CLI driver: load a `.mtx`
file -> cuSPARSE ILU(0) analysis+factorization (CUDA-event bracketed,
"Preconditioning Time") -> PCG loop to its own internal criterion
(`r1 > tol*tol && k < max_iter`, `tol=1e-12f` absolute unnormalized,
`max_iter=1000`, each iteration its own CUDA-event pair) -> append a row to
`results_summary_float.csv` -> exit. No separate library entry point exists,
so per ARTIFACT_GUIDE.md rule 1 this adapter wraps the **compiled binary as
a subprocess**, one fresh process per `run()` call — see `adapter.py`'s
module docstring for the full rationale (this also naturally satisfies the
spec's "5 independent runs, fresh factorization each run" requirement with
zero shared-CUDA-context risk between reps).

The `cmake/Modules` path CMakeLists.txt appends to `CMAKE_MODULE_PATH`
(`${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules`, i.e. repo-root
`cmake/Modules`) does not exist and is never `include()`d — confirmed
harmless empirically (clean configure + build, no errors referencing it).

## Patch (rule 3: minimal, recorded — no solver/algorithm code touched)

Both patches only touch (a) the RHS-vector initialization loop and (b) add
an optional post-hoc file dump after the solve is already complete. Neither
adds/removes/reorders a single cuBLAS/cuSPARSE call. `git -C source diff`:

```diff
diff --git a/gpu_src/ilu0_gpu/nonsp/main.cpp b/gpu_src/ilu0_gpu/nonsp/main.cpp
index 840df14..4938650 100644
--- a/gpu_src/ilu0_gpu/nonsp/main.cpp
+++ b/gpu_src/ilu0_gpu/nonsp/main.cpp
@@ -123,6 +123,19 @@ int main(int argc, char** argv) {
         x[i] = 0.0;    // Initial solution approximation
     }
 
+    // --- kernelbench patch (HPC-KernelBench cg-krylov integration): optional
+    // RHS override for spec conformance. Reads N little-endian float32 values
+    // from SPCG_RHS_FILE into rhs[] if the env var is set; otherwise the
+    // artifact's own rhs=1.0 default above is left completely untouched. Does
+    // not touch any solver/algorithm code (ARTIFACT_GUIDE.md rule 3).
+    {
+        const char* rhs_path = getenv("SPCG_RHS_FILE");
+        if (rhs_path) {
+            FILE* f = fopen(rhs_path, "rb");
+            if (f) { size_t nread = fread(rhs, sizeof(float), N, f); fclose(f); (void)nread; }
+        }
+    }
+
     /* Create CUBLAS context */
     cublasHandle_t cublasHandle = NULL;
     checkCudaErrors(cublasCreate(&cublasHandle));
@@ -415,6 +428,18 @@ int main(int argc, char** argv) {
 
     checkCudaErrors(cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));
 
+    // --- kernelbench patch (HPC-KernelBench cg-krylov integration): optional
+    // post-hoc solution dump for independent correctness verification by the
+    // harness (never read back by the solver itself, added after the solve
+    // is already complete). Does not touch any solver/algorithm code.
+    {
+        const char* x_out = getenv("SPCG_X_OUT_FILE");
+        if (x_out) {
+            FILE* f = fopen(x_out, "wb");
+            if (f) { fwrite(x, sizeof(float), N, f); fclose(f); }
+        }
+    }
+
     /* Check result */
     err = 0.0;
```

No new `#include` was needed: `<stdlib.h>` was already included, so
unqualified `getenv`/`FILE*`/`fopen`/`fread`/`fwrite`/`fclose` all resolved
without change (used unqualified C-style rather than `std::getenv`, the
smaller of the two options offered in the task brief).

**Patch 1 was applied** (RHS override), not skipped: it is low-risk (an
`if (env var set)` guard around the existing loop) and lets this adapter
honor the spec's `vector_operand` convention (`b: dense, U(-1,1), fixed
seed=42`) instead of SPCG's own hardcoded all-ones default — so there is
**no b=ones-vs-spec deviation** to report here.

`tol`/`max_iter` (SPCG's own hardcoded `1e-12f`/`1000`) were **not**
patched, per the task brief — SPCG's own internal loop-termination behavior
is left completely untouched. `params["spcg_internal_tol_abs_r1"]=1e-12`
and `params["spcg_internal_max_iter"]=1000` are recorded for audit in every
result record, alongside the spec's own `rtol=1e-6`/`max_iter=2000`
(`params["rtol"]`, `params["spec_max_iter"]`) — the two never conflated.

## adapter.py

- `KERNEL="cg-krylov"`, `IMPL_NAME="spcg-ilu0-cg"`, `PRECISIONS=["fp32"]`.
- `prepare()`: writes `matrix.csr` to a scratch `.mtx` via
  `scipy.io.mmwrite(..., symmetry="general")` (this **is** the artifact's own
  format-conversion-equivalent step — it only reads `.mtx` — so it is
  legitimately timed as preprocessing), plus a raw little-endian-float32
  `rhs.bin` (the spec's `U(-1,1)`, seed 42 vector, same RNG call as
  `solvers.ScipyCG.prepare()`). Writes resolved knobs back into `params` in
  place (`mode`, `rtol`, the SPCG-internal-vs-spec constants above) —
  matches `ScipyCG.prepare()`'s own discipline of never leaving a knob
  implicit.
- `run()`: launches `conjugateGradientPrecond <mtx>` as a **fresh
  subprocess** (`cwd=`a per-adapter scratch dir, `SPCG_RHS_FILE`/
  `SPCG_X_OUT_FILE` env vars set) — one full ILU(0) factorization + PCG
  solve. Parses the LAST row of the subprocess's own
  `results_summary_float.csv` (structured CSV, not stdout regex) into
  `params["iterations_actual"]` (read back from SPCG's own true loop
  counter — never assumed) plus audit fields `spcg_final_residual`,
  `spcg_precond_ms`, `spcg_pcg_ms`, `spcg_overall_ms_internal`,
  `spcg_subprocess_returncode`. Updated on **every** `run()` call (warmup
  and measured reps included), matching `ScipyCG.run()`'s own per-call
  `params["iterations_actual"]` update — by the end of the protocol these
  reflect the LAST measured rep, consistent with what the registered cost
  rule (`_cost_cg`) actually uses for GFLOP/s. Only raises (hard failure) if
  the x-dump file is genuinely missing after the subprocess exits (crash
  before the dump point); a nonzero exit code alone is NOT treated as fatal,
  since SPCG's own post-hoc `qaerr1`/`qaerr2` sanity check can legitimately
  fail even after a fine solve — the x dump (Patch 2) happens before that
  check runs, so it is always present for any run that reached the end of
  the PCG loop.
- `to_host()`: reads `x` back from the dump file, independently recomputes
  `relres = ||b - A@x||/||b||` in **fp64** using the SAME `b` written to
  `SPCG_RHS_FILE` — never trusting SPCG's own `sqrt(r1)`/tol bookkeeping.
  Gate mirrors `solvers.ScipyCG.to_host()`'s to-convergence mode **exactly**:
  `gate_tol = max(10*rtol, 1e-9)`, `rtol=1e-6`; returns
  `np.array([1.0])`/`np.array([0.0])`, matching `reference_cg()`'s sentinel
  bit-for-bit so `CORRECTNESS_MODE="exact"` (`np.array_equal`) works.
- `timer()`: plain `kernelbench.harness.Timer` (CPU wall-clock), **not**
  `CudaEventTimer` — documented prominently below.
- Precision clamp: `create(precision)` accepts any string but the class
  always sets `self.precision = "fp32"` and stores the request in
  `self.requested_precision`; if it wasn't already `"fp32"`,
  `params["spcg_precision_note"]` records the mismatch. This was **not**
  arbitrary defensiveness — it fixes a real interaction discovered while
  testing this adapter, described next.

## Finding: runner.py's default-precision heuristic mis-resolves this variant

`runner.py`'s precision auto-detection (no `--precision` flag given) does:
```python
text = (variant.protocol.precision + " " + variant.id).lower()
precision = "fp64" if "fp64" in text else ("fp32" if "fp32" in text else default)
```
`cg-e2e-ilu0-to-convergence`'s own spec text is, verbatim: *"fp32 (primary —
matches SPCG...); fp64 (secondary — matches BootCMatchGX's...)."* Both
substrings are present, and `"fp64"` is checked first, so **the default
invocation without `--precision` resolves to `"fp64"`** — confirmed by
loading the spec directly:
```
$PY -c "from kernelbench import spec; v=spec.load('cg-krylov').variant('cg-e2e-ilu0-to-convergence'); print(v.protocol.precision)"
```
This is a pre-existing `runner.py` heuristic limitation (out of scope for
this artifact-integration task to fix — flagged here since it looked at
first like an adapter bug), not something specific to SPCG. This adapter
works around it by clamping to fp32 unconditionally (see above) rather than
raising, so the task's own documented test invocation (no `--precision`
flag) still runs correctly; the actual gate/functional runs below were done
with an explicit `--precision fp32` for clarity, but a rerun without that
flag was also verified to behave identically (clamped, with the note field
set).

## Timing-boundary caveat (ARTIFACT_GUIDE.md rule 1 — read this before citing any number)

`timer()` measures **subprocess wall-clock** (`time.perf_counter` around the
whole `subprocess.run(...)` call), not a CUDA-event bracket — the solve runs
in a separate process with its own CUDA context, so there is nothing in this
Python process for a CUDA event to bracket. This means the harness's own
canonical per-rep number **includes process startup, CUDA context
init/teardown, and Matrix-Market text parsing** on top of the actual
ILU(0)+PCG work, and is measurably coarser than SPCG's own in-process
CUDA-event brackets. Concretely, on the `smoke-poisson2d-24` gate run below:
harness-measured wall time = 600.0 ms; SPCG's own internal
`Overall Time (ms)` = 59.2 ms (precond 32.1 ms + PCG 27.0 ms) — roughly
**10x** subprocess-launch overhead dominates at this tiny problem size. The
gap narrows sharply as problem size grows (SPCG-internal PCG time alone on
`cant` was 8.7s — startup overhead becomes a rounding error there). Both
numbers are always recorded in `params` (`spcg_precond_ms`, `spcg_pcg_ms`,
`spcg_overall_ms_internal` vs. the harness's own `stats_ms`), so nothing is
hidden — but any GFLOP/s or ms figure quoted from this adapter's
harness-reported `stats_ms`/`metrics` should be read as **subprocess-boundary
inclusive**, not apples-to-apples with an in-process CUDA-event number from
a different implementation of this same track.

## Preconditioner-track separability note (out of scope here, for whoever does that track)

SPCG's own `main.cpp` already brackets ILU(0) analysis+factorization
("Preconditioning Time") separately from the PCG loop ("PCG Time") via
distinct CUDA-event pairs — the ILU(0) setup/apply is cleanly separable and
could be wrapped as a standalone `preconditioner` kernel (`kind="ilu0"`)
implementation for the preconditioner track using the same
subprocess-per-solve pattern, reading `spcg_precond_ms`/`ilu_nnz_L`/
`ilu_nnz_U`-equivalent fields out of the same `results_summary_float.csv`.
Not implemented here (out of scope for this cg-krylov-only task).

## Gate verification (login node, functional checks only — no timing sweep)

Standalone binary sanity checks (bypassing the Python harness entirely, per
the task's step 1) — both run to completion, no crash:

```
$BINARY /tmp/.../tiny_spd.mtx                                    # 100x100 synthetic 2D Poisson SPD, default rhs=1.0
  -> Convergence Test OK, 19 iterations, residual 5.40e-13, 456.6 ms total

SPCG_RHS_FILE=... SPCG_X_OUT_FILE=... $BINARY /tmp/.../tiny_spd.mtx   # same matrix, patched U(-1,1) rhs + x dump
  -> 21 iterations, residual 1.99e-13, 48.2 ms total
  -> independent fp64 recompute from the dumped x: relres = 3.83e-07 (well under any reasonable gate)

$BINARY bench/matrices/cant.mtx                                  # ORIGINAL (un-round-tripped) cant.mtx, default rhs=1.0
  -> hits max_iter=1000 cap, residual 250.22 (== initial ||b||=sqrt(N)~250 -- ZERO net progress,
     confirmed via the per-iteration CSV: residual stays flat at 249.9-250.2 for all 1000 iterations,
     no NaN/Inf anywhere) -- see "Finding" below.
```

Through the harness (`$PY -m kernelbench.runner --kernel cg-krylov --variant cg-e2e-ilu0-to-convergence --impl spcg-ilu0-cg ...`):

| workload | precision | protocol (non-conforming, login-node) | gate | relres_achieved | iterations_actual | wall (ms) |
|---|---|---|---|---|---|---|
| smoke-poisson2d-24 (576x576) | fp32 | `--smoke --warmup 1 --reps 1` | **PASS** | 4.68e-07 | 41 | 600.0 |
| smoke-poisson2d-48 (2304x2304) | fp32 | `--smoke --warmup 1 --reps 1` | **PASS** | 1.16e-06 | 78 | 601.2 |
| smoke-poisson3d-10 (1000x1000) | fp32 | `--smoke --warmup 1 --reps 1` | **PASS** | 1.81e-07 | 25 | 563.2 |
| cant (62451x62451, nnz 4,007,383) | fp32 | `--matrices cant --warmup 0 --reps 1` | **FAIL** | 0.997 | 1000 (cap hit) | n/a (gate failed before timing) |

`$PY -m kernelbench.runner --kernel cg-krylov --list` confirms:
`ok  spcg             spcg-ilu0-cg` under "paper artifacts" (`available()==True`).

Gate: `passed = finite(relres) and relres <= max(10*1e-6, 1e-9) = 1e-5`. All
three smoke matrices pass comfortably (relres 2-6 orders of magnitude under
the gate). `cant` fails: `relres_achieved=0.997` means essentially **zero**
net residual reduction after the full 1000-iteration cap — not "didn't quite
reach a tight tolerance," a genuine stagnation.

### Finding: SPCG's own ILU(0)-PCG makes no progress on `cant` within its 1000-iteration cap — genuine artifact behavior, not an adapter bug

Verified two ways, both bypassing the adapter to rule out a wrapping bug:
(1) the standalone binary run above, on the *original* (not round-tripped)
`bench/matrices/cant.mtx`, with SPCG's *own default* `rhs=1.0` (no RHS
patch involved at all) — same outcome, residual flat at ~250 the whole run;
(2) the per-iteration CSV (`float_residuals_cant.csv`) shows no NaN/Inf and
a residual that is essentially constant (249.9 -> 250.2) across all 1000
iterations, not diverging — this is stagnation, not blow-up. `cant` is a
real SuiteSparse structural-mechanics matrix, in this track's own
`recommended_subset`, known in the sparse-linear-algebra literature to be
comparatively ill-conditioned for plain ILU(0) preconditioning; this is
consistent with that, not a symptom of anything this integration did.

Separately worth flagging: the artifact's own printed
`Convergence Test: OK` (main.cpp, right above a
`// TODO: Change the measurement of convergence` comment already left by
the artifact's authors) is checking `k <= max_iter`, which is **always
true** by construction (the loop's own exit condition guarantees it) — so
that message provides no actual convergence signal and printed "OK" even
for this completely-stagnated run. Not something this integration
introduced or should fix (touching the artifact's diagnostic printf is out
of scope), but worth recording since it could otherwise be misread as the
artifact vouching for a result its own harness never actually gated.

Per ARTIFACT_GUIDE.md rule 4 ("if it fails the gate, that IS a result —
record it, do not loosen the gate to make it pass"): this is recorded as-is.
No further matrices from `recommended_subset` were downloaded/run to hunt
for a passing large-scale case — out of the login-node budget (rule 5); the
three smoke passes already demonstrate the adapter/gate mechanism itself is
correct end-to-end, and the `cant` result is left as a real, reportable
data point for whoever runs the full timed protocol later on a compute
allocation (worth trying a smaller/better-conditioned `recommended_subset`
matrix there, e.g. `bcsstk17`).

## Not done

- No sweep across `recommended_subset` — out of scope per the login-node
  budget (ARTIFACT_GUIDE.md rule 5); one synthetic-smoke gate check (3
  matrices) + one real-matrix (`cant`) check only.
- The RHS override (Patch 1) and x dump (Patch 2) were verified to work
  correctly in isolation via direct binary invocation (see above) before
  being wired into `adapter.py`.
- The `sp` (sparsified) and `iluk_gpu` variants under `gpu_src/` — this
  paper's own headline contribution (sparsified ILU(K) preconditioning) —
  were not wrapped; the task scope is this track's `cg-e2e-ilu0-to-
  convergence` variant, which the plain (non-sparsified) `nonsp` driver
  already implements faithfully and is what the survey/spec archaeology was
  grounded in.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, on an A100-SXM4-40GB
  card), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (CMake's own
  default `c++`, conda-forge toolchain), cmake 4.2.3, Python 3.12.14.
- Build: OK. Build-system changes: none needed beyond what was already
  committed (`build.sh`'s `-j8` cap was already in place).
- Gate: `cg-e2e-ilu0-to-convergence`, no `--precision` flag given (matching
  this file's own documented command) — `--smoke --warmup 1 --reps 1`
  (protocol override): **3/3 runs valid.**
  smoke-poisson2d-24: relres=4.676e-07, iters=41, PASS.
  smoke-poisson2d-48: relres=1.158e-06, iters=78, PASS.
  smoke-poisson3d-10: relres=1.807e-07, iters=25, PASS.
  All match the originally recorded numbers (4.68e-07/1.16e-06/1.81e-07,
  iters 41/78/25) to the displayed precision. The documented `runner.py`
  precision-heuristic quirk reproduces identically here too: with no
  `--precision` flag the variant's own spec text resolves to a default
  request of `fp64`, and every run's `params["spcg_precision_note"]`
  confirms the adapter's fp32 clamp fired exactly as designed ("requested
  precision 'fp64' != fp32; SPCG's own kernel is float32-only (unpatched),
  clamped to fp32"). `cant` (the documented FAIL case) was not re-run in
  this pass (no local SuiteSparse matrix cache on this machine and a
  `bench/matrices/` download attempt timed out under login-node
  contention — out of scope for this reproduction per the login-node
  budget; the 3-matrix smoke gate already reproduces the adapter/gate
  mechanism end-to-end).
- Deviation from the recorded ruling: none.
- Verdict here: **BUILT+GATED**, same as the recorded ruling.
