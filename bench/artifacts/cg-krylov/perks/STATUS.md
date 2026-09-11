# perks (PERKS) — cg-krylov

**Status: BUILT+GATED**

- Paper: "PERKS: a Locality-Optimized Execution Model for Iterative
  Memory-bound GPU Applications" (ICS'23). `PAPER_KEY = conf/ics/ZhangWCMWEM23`
  (title-matched in `../../../output/included.json`, already confirmed before
  this integration started).
- Artifact: https://github.com/neozhang307/PERKS
- Commit: `b56f46513559942e1da27d9831c5e869550c8b68` (2023-05-24), already
  cloned with `git clone --depth 1` before this task started (history intact
  for provenance).
- Toolchain: `nvcc` release 12.9, V12.9.41
  (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc`, the compiler
  actually used for every `.cu` translation unit — `cg_driver.cu`, `main.cu`,
  `util/timer.cu`, `util/cub_utils.cu`); `cmake` 3.28.3 (`/usr/bin/cmake`);
  host CXX compiler as CMake's own default detection resolved it —
  `/usr/bin/c++` (GNU 7.5.0, the base-OS default; distinct from the
  interactive shell's `gcc-native/14`-module `g++` 14.3.0, since CMake's
  compiler search does not go through shell aliases) — used only for the two
  plain-C++ support libraries (`cg_cpu.cpp`, `matrix/tridiag.cpp`), not for
  any kernel code. GPU: single NVIDIA A100-PCIE-40GB, sm_80 (shared
  login-node GPU; both `-gencode arch=compute_80,code=sm_80` and
  `arch=compute_70,code=sm_70` are baked into the artifact's own
  `CMakeLists.txt`, unchanged). Python:
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python` (numpy, scipy only
  needed — no torch import in this adapter; the wrapped binary is a native
  CUDA executable, not a Python/torch kernel).

## What is wrapped, and why (contamination note, per ARTIFACT_GUIDE.md rule 1)

`conjugateGradient/main.cu` (has `int main()`) + `conjugateGradient/cg_driver.cu`
(the templated `myTest<ValueT,OffsetT,baseline,cacheMatrix,cacheVector>`
driver body, plus explicit instantiations for all 10
`{float,double} x {baseline, (cmat,cvec) in FF/FT/TF/TT}` combinations,
compiled unconditionally since `COMPILE` is never defined by the build) build
via the artifact's own (single active) `subcompile(...)` CMake call into
library `cg_perks` (from `main.cu`) + executable `cg_perks.exe` (from
`cg_driver.cu`, linked against `cg_perks`) — **verified empirically**, not
just read off the CMake function text: built the project and confirmed
`build/init/cg_perks.exe` is the only executable produced, then ran it with
every flag combination exercised below.

`main()` does device init (`findCudaDevice`), MTX-file load/parse
(`CooMatrix::InitMarket`, plain C++ text parsing), `cudaMallocManaged`
allocation for every array, THEN the CG dispatch, THEN cleanup — all inside
one process, no separate solve-only library entry point exists. Per the
parent task's explicit framing for PERKS specifically ("wrap its CG solver
binary/entry at the finest boundary available; if only whole-benchmark
binaries exist, wrap ONE solve invocation and document the contamination —
this is explicitly sanctioned, not a fallback of last resort"), `adapter.py`
launches `cg_perks.exe` as a **fresh subprocess per `run()` call**, one full
`--iters=<maxiter>` solve per call. The harness's own per-rep wall-clock
number (see "Timing-boundary caveat" below) therefore includes process
startup + CUDA context init + device query + MTX text parsing +
`cudaMallocManaged` allocation on top of the actual CG kernel loop.

### Why cg-kernel-fixed-iter, not one of the to-convergence variants

PERKS's own driver runs a hard fixed iteration count with **no** residual-based
loop termination when `--staticiter` is passed — confirmed by reading
`cg_kernels.cuh`'s loop guard:
`if((!isStaticIteration && r1 < tol*tol) || k > max_iter) break;` — with
`isStaticIteration=true` the first disjunct is always false, so the loop only
ever exits on `k > max_iter`. Confirmed **empirically**, not just by reading
the code: every run below shows `total_iter` (read back from the kernel's own
device-side counter) exactly equal to `--iters`. This is literally PERKS's own
stated research question — steady-state per-iteration throughput of a
persistent-kernel execution model — so `cg-kernel-fixed-iter` is the variant
this artifact actually implements, not a simplification of convenience.

## PERKS's own timing-granularity issue, and how this integration sidesteps it

`cg_driver.cu` brackets a **single** CUDA-event pair around the **entire**
`max_iter`-iteration dispatch call and divides by `total_iter` afterward
(confirmed by reading the file — this is exactly what
`benchspecs/cg-krylov/spec.yaml`'s `notes_on_fairness` calls out by name for
this artifact). Reproducing the full spec's per-iteration-timestamp protocol
exactly would require hand-rolling the CG recurrence per-iteration from
Python — a materially different algorithm-execution shape than calling
PERKS's own fused persistent kernel, not a timing simplification. Per the
parent task's explicit instruction (matching
`kernelbench/domains/solvers.py`'s own documented choice for `ScipyCG`): this
adapter instead treats `run()` = ONE full fixed-`maxiter`-iteration solve,
timed by the harness as ONE unit per rep — a coarser-but-honest "one call,
one full solve" convention this codebase has already established as
acceptable for this track. PERKS's own internally-reported
`total_iter`/`time` (its single-bracket-over-all-iterations number) is still
parsed and recorded into `params` (`perks_reported_ms`,
`perks_reported_iters`, `perks_reported_residual_recurrence`), clearly
labeled audit-only, never the harness's canonical timed number.

## Patches (rule 3: minimal, recorded — no CG/SpMV kernel line touched)

`git -C source diff`:

```diff
diff --git a/conjugateGradient/cg_driver.cu b/conjugateGradient/cg_driver.cu
index af85d9f..45e7129 100644
--- a/conjugateGradient/cg_driver.cu
+++ b/conjugateGradient/cg_driver.cu
@@ -19,7 +19,7 @@
 #define THRUST_IGNORE_CUB_VERSION_CHECK
 // #include <map>
 
-
+#include <cstdlib>  // getenv, for the HPC-KernelBench adapter's env-var hooks below
 
 #include "cg.h"
 #include "util/timer.cuh"
@@ -208,6 +208,23 @@ void myTest(
     // x[i] = 1;
     x[i] = 0.0;
   }
+  // --- HPC-KernelBench adapter hook (optional, additive) ---------------
+  // Spec conformance: cg-krylov's fixed-iter variant wants b ~ U(-1,1),
+  // seed=42, not this artifact's own rhs=ones(N). When the adapter writes
+  // a binary dump of such a vector and points PERKS_RHS_FILE at it, load
+  // it here (overriding rhs/r above); otherwise behavior is byte-for-byte
+  // unchanged from upstream (rhs=ones(N)).
+  {
+    const char* rhs_path = getenv("PERKS_RHS_FILE");
+    if (rhs_path) {
+      FILE* f = fopen(rhs_path, "rb");
+      if (f) {
+        size_t nread = fread(rhs, sizeof(ValueT), N, f);
+        fclose(f);
+        for (int i = 0; i < (int)nread; i++) r[i] = rhs[i];
+      }
+    }
+  }
 
 
 
@@ -308,6 +325,21 @@ void myTest(
   checkCudaErrors(cudaEventRecord(stop, 0));
   checkCudaErrors(cudaDeviceSynchronize());
 
+  // --- HPC-KernelBench adapter hook (required) --------------------------
+  // x is cudaMallocManaged and this thread has just synchronized past the
+  // solve, so it is safe to read/dump directly here (no separate D2H copy
+  // needed). Written only when the adapter sets PERKS_X_OUT_FILE; no-op
+  // (and no behavior change) otherwise. sizeof(ValueT) is 4 or 8 depending
+  // on which templated instantiation this translation unit is (fp32/fp64),
+  // so the Python side must read back with the matching dtype.
+  {
+    const char* x_out = getenv("PERKS_X_OUT_FILE");
+    if (x_out) {
+      FILE* f = fopen(x_out, "wb");
+      if (f) { fwrite(x, sizeof(ValueT), N, f); fclose(f); }
+    }
+  }
+
   float time;
   checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
   // printf("----");
@@ -354,6 +386,12 @@ void myTest(
   fprintf(stderr,"%f\t%f\t%f\t",(double)smParamsT.sm_size_coor/1024, (double)smParamsT.sm_blk_size_r/1024, (double)smParamsT.sm_size_unit_matrix*smParamsT.sm_num_matrixperblk/1024);
   fprintf(stderr,"%f\t",(double)smParamsT.sMemSize/1024);
   fprintf(stderr,"%d\t%f\t",total_iter,time);
+  // HPC-KernelBench adapter hook (optional): one unambiguous, easy-to-parse
+  // line, purely additive -- no computation changed, just an extra
+  // diagnostic print so the Python side doesn't have to scrape the
+  // multi-purpose tab-separated debug line above.
+  fprintf(stderr, "\nPERKS_ADAPTER_RESULT iters=%u time_ms=%f residual=%.17e\n",
+          total_iter, time, sqrt((double)r1));
   // size_t spmvaccess= nz*sizeof(ValueT)*2+(nz+N+1)*sizeof(OffsetT);
   // size_t totalaccess=spmvaccess+7*N*sizeof(ValueT)+max_iter*(spmvaccess+9*N*sizeof(ValueT));
   //MINIAN mem access: 1 spmv load + 3 vector update(1 update = 1 load + 1 store)
diff --git a/conjugateGradient/cub/iterator/tex_ref_input_iterator.cuh b/conjugateGradient/cub/iterator/tex_ref_input_iterator.cuh
index f02dc43..acc644e 100644
--- a/conjugateGradient/cub/iterator/tex_ref_input_iterator.cuh
+++ b/conjugateGradient/cub/iterator/tex_ref_input_iterator.cuh
@@ -42,7 +42,7 @@
 #include "../util_debug.cuh"
 #include "../config.cuh"
 
-#if (CUDART_VERSION >= 5050) || defined(DOXYGEN_ACTIVE)  // This iterator is compatible with CUDA 5.5 and newer
+#if ((CUDART_VERSION >= 5050) && (CUDART_VERSION < 12000)) || defined(DOXYGEN_ACTIVE)  // ...guard narrowed, see below
 
 #if (THRUST_VERSION >= 100700)    // This iterator is compatible with Thrust API 1.7 and newer
     #include <thrust/iterator/iterator_facade.h>
```

**Patch 0 (build-system, required to compile at all on CUDA 12.9):**
`cub/iterator/tex_ref_input_iterator.cuh` is vendored NVIDIA CUB (2018-era)
using the legacy Tesla/Fermi texture-reference API (`texture<T>`,
`cudaBindTexture`/`cudaUnbindTexture`), which CUDA 12 removed outright:
initial `cmake --build` failed with `error: texture is not a template` +
two `identifier "cudaBindTexture"/"cudaUnbindTexture" is undefined` errors.
Confirmed **dead code** before patching: `grep -rn TexRefInputIterator`
across the whole vendored tree turns up zero uses outside the class's own
definition — it is pulled in transitively via `cub/cub.cuh` (line 88) and
directly via `cg_kernels.cuh` (line 29), but never instantiated anywhere in
PERKS's own code. The guard was narrowed from `CUDART_VERSION >= 5050` to
`>= 5050 && < 12000` (skip compiling this legacy class entirely under CUDA
12+), matching the kind of version-gate later official CUB releases added
around this exact deprecated class before eventually deleting the file. This
is a CUDA-version compatibility guard on vendored, unused infrastructure —
zero kernel logic touched (ARTIFACT_GUIDE.md rule 3).

**Patch 1 (required):** `x`-dump hook in `cg_driver.cu`, right after the
dispatch call's post-solve `cudaDeviceSynchronize()`. `x` is
`cudaMallocManaged`, so this is a direct host read, no separate D2H copy.
This is the **only** way to get the solution vector out of this artifact at
all (upstream never writes `x` to a file) — without it, `to_host()` would
have nothing to gate on.

**Patch 2 (optional/additive):** one unambiguous
`PERKS_ADAPTER_RESULT iters=... time_ms=... residual=...` stderr line, added
right after the existing fragile multi-purpose tab-separated debug line — no
computation changed, purely an extra diagnostic print, verified to make
Python-side parsing trivial (single regex, no column-position scraping).

**Patch 3 (optional, applied — spec conformance):** `rhs`/`r` loaded from
`PERKS_RHS_FILE` (raw `ValueT`-sized binary, N values) if the env var is set,
else the artifact's own `rhs=ones(N)` default is untouched. **This patch was
applied and is used by every run** (`adapter.py`'s `prepare()` always writes
this file), so there is **no `b=ones(N)`-vs-spec deviation** to report for
this integration — the actual `b` used is the spec's `U(-1,1)`, seed=42 (the
identical RNG call `solvers.ScipyCG.prepare()` uses), unlike the situation
the task brief anticipated as the likely fallback.

No new build-system changes were needed beyond the CUDA-version guard above;
`<cstdlib>` was added explicitly for `getenv` (not previously included in
`cg_driver.cu`, unlike SPCG's `main.cpp` in the sibling integration, which
already had `<stdlib.h>` transitively).

## Runtime variant selection: `--cmat --cvec` (cached), not `--baseline`

Per the task's guidance ("TRY the cached/optimized combo first since it's the
paper's real contribution... fall back to `--baseline` if it doesn't
build/run cleanly"): **both were built into the same binary** (all 10
`{fp32,fp64} x {baseline,(cmat,cvec) combination}` template instantiations
compile unconditionally) and **both were run standalone** against
`bench/matrices/cant.mtx` (62,451 rows, 4,007,383 nnz) before wiring the
adapter, bypassing Python entirely:

```
$ ./build/init/cg_perks.exe --mtx=.../cant.mtx --iters=20 --baseline --staticiter --check
  residual = 6.300758e+03, total_iter=20, time=0.770290 ms, 62.07 GB/s, check-error=831.697376

$ ./build/init/cg_perks.exe --mtx=.../cant.mtx --iters=20 --cmat --cvec --staticiter --check
  residual = 6.300758e+03, total_iter=20, time=0.362955 ms, 131.73 GB/s, check-error=831.697376

$ ./build/init/cg_perks.exe --mtx=.../cant.mtx --iters=20 --fp32 --cmat --cvec --staticiter --check
  residual = 6.298725e+03, total_iter=20, time=0.364738 ms, 86.32 GB/s, check-error=831.435303
```

**Identical residual and identical `--check` error** between `--baseline` and
`--cmat --cvec` at the same iteration count, with `--cmat --cvec` ~2.1x
faster (0.363ms vs 0.770ms) — confirms algorithmic equivalence before picking
the faster, paper-representative configuration. `IMPL_NAME = "perks-cg-cached"`
wires `--cmat --cvec` by default; overridable per-run via
`params["perks_variant"] = "baseline" | "cached"` (both code paths implemented
and both were exercised manually above — `--baseline` was not merely read
about, it was run).

## adapter.py

- `KERNEL="cg-krylov"`, `IMPL_NAME="perks-cg-cached"`, `PRECISIONS=["fp64","fp32"]`
  (both genuinely supported via the templated `ValueT` + `--fp32` CLI flag;
  fp64 is the default, matching the artifact's own default when `--fp32` is
  absent).
- `prepare()`: writes `matrix.csr` to a scratch `.mtx` via
  `scipy.io.mmwrite(..., symmetry="general")` (the artifact's own
  format-conversion-equivalent step — it only reads `.mtx` text — legitimately
  timed as preprocessing), plus a raw `ValueT`-dtype `rhs.bin` (spec's
  `U(-1,1)`, seed 42, same RNG call as `solvers.ScipyCG.prepare()`). Writes
  resolved knobs back into `params` in place (`mode="fixed-iter"`, `maxiter`,
  `perks_variant`) — same discipline as `ScipyCG.prepare()`.
- `run()`: launches `cg_perks.exe --mtx=<path> --iters=<maxiter> --staticiter
  [--fp32] [--baseline | --cmat --cvec]` as a **fresh subprocess**
  (`cwd=`a per-run scratch dir, `PERKS_RHS_FILE`/`PERKS_X_OUT_FILE` env vars
  set). Parses the `PERKS_ADAPTER_RESULT` stderr line (regex, not
  column-position scraping) into `params["iterations_actual"]` (PERKS's own
  true device-side loop counter, read back correctly — confirmed equal to
  `maxiter` in every run below, since `--staticiter` disables early exit) plus
  audit fields `perks_reported_ms`, `perks_reported_residual_recurrence`.
  Only raises (hard failure) if the `x`-dump file is genuinely missing after
  the subprocess exits.
- `to_host()`: reads `x` back (dtype matched to precision), independently
  recomputes `relres = ||b - A@x||/||b||` in **fp64** using the SAME `b`
  written to `PERKS_RHS_FILE` — never trusting PERKS's own `sqrt(r1)`
  recurrence bookkeeping. Gate mirrors `solvers.ScipyCG.to_host()`'s
  **fixed-iter** branch **exactly**: `passed = finite and relres < 1.0` (not
  full convergence — a 50-iteration budget is not supposed to reach that).
  Returns `np.array([1.0])`/`np.array([0.0])` matching `reference_cg()`'s
  sentinel bit-for-bit so `CORRECTNESS_MODE="exact"` works.
- `timer()`: plain `kernelbench.harness.Timer` (CPU wall-clock), **not**
  `CudaEventTimer` — see "Timing-boundary caveat" below.
- `free()`: removes the per-run scratch directory.

## Timing-boundary caveat (read before citing any number)

`timer()` measures subprocess wall-clock around the whole `subprocess.run(...)`
call, not a CUDA-event bracket — the solve runs in a separate process with
its own CUDA context each call, so there is nothing in the harness's Python
process for an in-process CUDA event to bracket. On the `smoke-poisson2d-24`
gate run below: harness-measured (warmup=1,reps=1, non-conforming) wall time
= 363.1 ms; PERKS's own internal single-bracket `time` = 4.63 ms — process
startup + CUDA context init dominates at this tiny problem size and this
override protocol (one warmup, one measured rep — a real 5-warmup/50-rep
protocol would amortize this far better on a compute allocation). Both
numbers are always recorded in `params` (`perks_reported_ms` vs. the
harness's own `stats_ms`), so nothing is hidden — but any ms/GFLOP-s figure
quoted from this adapter's harness-reported `stats_ms`/`metrics` should be
read as subprocess-boundary-inclusive, not apples-to-apples with an
in-process CUDA-event number from a different implementation.

## Gate verification (login node, functional checks only — no timing sweep)

Standalone binary sanity checks (bypassing the Python harness, task step 1) —
see "Runtime variant selection" above; both `--baseline` and `--cmat --cvec`
ran to completion with sane, matching residuals before any Python code was
written.

Through the harness:

```
$PY -m kernelbench.runner --kernel cg-krylov --variant cg-kernel-fixed-iter \
    --impl perks-cg-cached --smoke --warmup 1 --reps 1
$PY -m kernelbench.runner --kernel cg-krylov --variant cg-kernel-fixed-iter \
    --impl perks-cg-cached --matrices cant --warmup 1 --reps 1
```

| workload | precision | gate | relres_achieved | iterations_actual | perks_reported_ms | wall (ms, harness, non-conforming) |
|---|---|---|---|---|---|---|
| smoke-poisson2d-24 (576x576, nnz 2784) | fp64 | **PASS** | 4.17e-05 | 50 | 4.63 | 363.1 |
| smoke-poisson2d-48 (2304x2304, nnz 11328) | fp64 | **PASS** | 6.55e-03 | 50 | 4.17 | 300.8 |
| smoke-poisson3d-10 (1000x1000, nnz 6400) | fp64 | **PASS** | 8.31e-12 | 50 | 4.15 | 291.5 |
| cant (62451x62451, nnz 4,007,383) | fp64 | **FAIL** | 1.0207526017052537 | 50 | 11.51 | n/a (gate failed before timing) |

`$PY -m kernelbench.runner --kernel cg-krylov --list` confirms:
`ok  perks  perks-cg-cached` under "paper artifacts" (`available()==True`).

### Finding: `cant` fails the fixed-iter gate — genuine unpreconditioned-CG behavior, cross-validated against the harness's own `scipy-cg` reference, not a PERKS/adapter bug

`relres_achieved` on `cant` is **1.0207526017052537** — i.e. after 50
unpreconditioned CG iterations starting from `x0=0`, the residual is
marginally *worse* than the initial residual (`relres0 == 1.0` exactly, by
construction). To rule out an adapter/artifact-specific bug, the harness's
own built-in `scipy-cg` reference implementation was run through the
**identical** protocol (`--impl scipy-cg --matrices cant --warmup 1 --reps 1`,
same seed=42 `b`, same `A`, same `mode=fixed-iter`, `maxiter=50`):

```
scipy-cg:    relres_achieved = 1.0207526017052542
perks-cg-cached: relres_achieved = 1.0207526017052537
```

**Agreement to ~13 significant figures** between two completely independent
implementations (CPU BLAS-based `scipy.sparse.linalg.cg` vs. PERKS's own CUDA
persistent-kernel dispatch, connected by nothing but sharing the same seed-42
`b` and the same `A`) is about as strong a correctness signal as this
integration could produce without an analytic reference: it confirms PERKS's
`--cmat --cvec` kernel computes the mathematically correct unpreconditioned
CG recurrence, and that the `cant` gate failure is a genuine property of
unpreconditioned CG on this real, moderately ill-conditioned structural
matrix within a 50-iteration budget — not a wrapping defect, RHS-patch bug,
or x-dump corruption. Per ARTIFACT_GUIDE.md rule 4 ("if it fails the gate,
that IS a result — record it, do not loosen the gate to make it pass"): this
is recorded as-is, left as a real data point for whoever runs the full timed
protocol later (worth trying a better-conditioned `recommended_subset`
matrix there, e.g. `bcsstk17`, or a larger `maxiter`).

## Not done

- No sweep across `recommended_subset` — out of scope per the login-node
  budget (ARTIFACT_GUIDE.md rule 5); the task's own two documented test
  invocations (`--smoke`, `--matrices cant`) only.
- Patch 3 (RHS override) and Patch 1 (x dump) were verified working correctly
  in isolation via direct binary invocation with `PERKS_RHS_FILE`/
  `PERKS_X_OUT_FILE` set (finite, correctly-shaped output; different residual
  than the ones(N)-rhs baseline, as expected) before being wired into
  `adapter.py`.
- `--check` (the artifact's own opt-in `A*x` vs `rhs` scan) is not passed by
  the adapter — informational only per the artifact's own README, not
  load-bearing for this integration's gate, and would add an extra
  host-side O(nnz) pass to every `run()` call for no benefit here.
- The `stencil/` subtree (PERKS's other persistent-kernel application) was
  not touched, per the parent task's explicit scope (`conjugateGradient/`
  only).

## Reuse attempt by the stencil track (2026-09-06) — SKIPPED, evidenced

`artifacts/stencil/perks/` (symlinked `source/`, same commit, no second
clone) investigated wrapping this repo's `source/stencil/{2dstencil,
3dstencil}/` kernels for the `stencil` track. Outcome: **SKIPPED** — every
stencil kernel there (naive/baseline/persistent-general, 2D and 3D, star
and box) sources its per-neighbor coefficients from a single hardcoded
`#define stencilParaT` with literal numeric constants; no CLI flag or
kernel argument anywhere accepts a caller-supplied weight set, so this
domain's own synthetic weights can never be fed in without patching
tracked source. No build was attempted there (nothing would be gate-able
regardless). See `artifacts/stencil/perks/STATUS.md` for full evidence
(file:line citations for every hardcoded weight table, both dims).
- `preconditioner`-track wiring: PERKS's own CG is unpreconditioned by
  design (no preconditioner apply exists anywhere in this driver), so there
  is nothing to wrap for the `preconditioner` kernel from this artifact.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, on an A100-SXM4-40GB
  card), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (CMake's own
  default `c++`, resolved from the conda-forge toolchain, not a base-OS
  compiler as on Perlmutter — same role, `cg_cpu.cpp`/`matrix/tridiag.cpp`
  only), cmake 4.2.3, Python 3.12.14; `-gencode arch=compute_80,code=sm_80`
  (+`compute_70,code=sm_70`) unchanged from the artifact's own CMakeLists.
- Build: OK. Build-system changes: `build.sh`'s `cmake --build ... -j8` cap
  (was `-j$(nproc)`) was already applied from an earlier interrupted pass,
  verified still correct and necessary here — an uncapped `-j$(nproc)`
  (128+ on this shared login node) exhausts the session's `RLIMIT_NPROC`
  ("cc1plus: vfork: Resource temporarily unavailable"), not an artifact
  defect. No further build-system changes needed.
- Gate: `cg-kernel-fixed-iter`, fp64 (default), `--smoke --warmup 1 --reps 1`
  (protocol override, matching this file's own documented command):
  **3/3 runs valid.**
  smoke-poisson2d-24: relres=4.174e-05, iters=50, PASS.
  smoke-poisson2d-48: relres=6.547e-03, iters=50, PASS.
  smoke-poisson3d-10: relres=8.309e-12, iters=50, PASS.
  All match the originally recorded Perlmutter numbers (4.17e-05, 6.55e-03,
  8.31e-12) to the displayed precision. `cant` (the documented FAIL case)
  was not re-run in this pass (no local SuiteSparse matrix cache on this
  machine and a `bench/matrices/` download attempt timed out under login-
  node contention — out of scope for this reproduction per the login-node
  budget; the 3-matrix smoke gate already reproduces the adapter/gate
  mechanism end-to-end).
- Deviation from the recorded ruling: none.
- Verdict here: **BUILT+GATED**, same as the recorded ruling.
