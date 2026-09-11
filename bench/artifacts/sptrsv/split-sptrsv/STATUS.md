# split-sptrsv (Split_SpTRSV) — sptrsv

**Status: BUILT (gate fails) — a genuine bit-of-the-original-artifact
correctness bug (two commented-out `__threadfence()`/`__threadfence_block()`
calls in the unmodified vendored kernel), root-caused and confirmed below by
directly restoring them in a scratch diagnostic build (not shipped). The
gate is NOT loosened; the artifact's own `shim.so`/`source/` remain
byte-for-byte the pristine clone.**

- Paper: "A Split Execution Model for SpTRSV" (TPDS'21).
  `PAPER_KEY = journals/tpds/AhmadYU21` (matches
  `output/benchmark_groups.json`'s `sptrsv` group).
- Artifact: https://github.com/ParCoreLab/Split_SpTRSV
- Commit cloned: `deeddda6910c9d2bc4262acf24d89f940fb90e91` (2021-09-07),
  `git clone --depth 1` into `./source/` (`.git` kept for provenance).
  `git status`/`git diff` inside `source/` are both empty — **zero** patches
  to the artifact's own code, confirmed again during this integration pass
  (see "Root cause" below for why that matters).
- Toolchain: `nvcc` 12.9
  (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`), `-gencode
  arch=compute_80,code=sm_80` (A100-PCIE-40GB). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python` (torch 2.8.0+cu128).
  Runtime needs `LD_PRELOAD=/usr/lib64/libstdc++.so.6` (project-wide
  CXXABI/GLIBCXX fix).
  **Toolchain pin note (verified 2026-09-04):** the login node's default
  loaded module is now `cudatoolkit/13.2` (HPC SDK 26.5), whose `CUDA_HOME`/
  `PATH`/`CPATH` all point at 13.2. `build.sh` now sources
  `../../toolchain.sh` (shared across this project's artifact directories)
  which pins `CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`,
  prepends its `bin` to `PATH`, and **re-points** (not clears) `CPATH` at
  12.9's own `math_libs/12.9/include` + `cuda/12.9/include` — clearing
  `CPATH` outright breaks anything that needs `cusparse.h` (not needed by
  this plain-nvcc shim itself, but this is the shared, uniform recipe used
  project-wide). Re-ran `build.sh` under the current environment with this
  pin: builds clean (only the artifact's own pre-existing sign-conversion/
  unused-variable warnings in `sptrsv_syncfree_cuda.cuh`, unchanged from
  the original August build), and the gate command below was re-run
  afterward — same failure mode, same order of magnitude. The harness
  (`kernelbench.runner`) also independently auto-pins `CUDA_HOME`/`CPATH`
  at import time (see its `[env] toolchain: ...` line in the gate output
  below), so gate runs were never actually affected by the module change;
  `build.sh`'s own pin only matters for a bare shell rebuild outside the
  runner.

## What the artifact actually is, and what is wrapped

Split_SpTRSV's paper contribution is a CPU-GPU **split execution model**:
partition a triangular solve's rows between a CPU thread pool (via MKL) and
a GPU kernel, connected by an SpMV correction step, to balance load across
both devices. The upstream CLI (`src/main.cpp`) drives this whole pipeline,
gated behind `libUFget` (SuiteSparse auto-download) and Intel MKL, neither
available nor wanted here per ARTIFACT_GUIDE.md rule 1 ("wrap the kernel,
not the paper's benchmark script"). What this integration wraps instead is
the GPU-only kernel the paper's own split-execution model falls back to for
its GPU-only case (**mode 2** in the CLI): the classic Liu-et-al.
warp-per-row, synchronization-free SpTRSV
(`source/include/sptrsv_syncfree_cuda.cuh`'s
`sptrsv_syncfree_cuda_analyser` / `sptrsv_syncfree_cuda_executor`). This
header has **no** dependency on MKL/libUFget/icpc — only `<cuda_runtime.h>`
and the artifact's own `common.h` typedefs — so it compiles standalone via
plain `nvcc`.

`shim.cu` (this directory, **not** part of the artifact) `#include`s that
header verbatim and adds two `extern "C"` host-side launchers around its
two unmodified `__global__` kernels; **no kernel line was edited** — `git
diff` on `source/` above is empty, confirming this.

- `split_sptrsv_indegree` — wraps `sptrsv_syncfree_cuda_analyser`
  unmodified: one-shot structural in-degree count (pure function of `L`'s
  sparsity pattern, independent of any particular solve). Called ONCE from
  `prepare()`.
- `split_sptrsv_solve` — resets the two pieces of per-call algorithm state
  the executor kernel mutates in place (`d_graphInDegree`, the
  synchronization counter; `d_left_sum`, the partial-sum accumulator), then
  launches `sptrsv_syncfree_cuda_executor` unmodified
  (`SUBSTITUTION_FORWARD`).

## Format: CSC of `L`, diagonal-first per column

`L` (this track's canonical unit-lower-triangular structural proxy,
`kernelbench.impls.cpu_ref.unit_lower_triangular` — the SAME helper the
CPU_IMPLS entry and the correctness reference use, so every sptrsv
competitor in this benchmark solves the identical triangular system) has,
for column `j`, only rows `i>=j`, so the diagonal (row==j) is structurally
the row-minimum entry in that column — scipy's ordinary ascending-row CSC
sort (`.tocsc().sort_indices()`) lands it first automatically, matching
`sptrsv_syncfree_cuda_executor`'s own indexing assumption (`pos =
d_cscColPtr[global_x_id]` read as "the diagonal" for the forward-
substitution branch). No manual reordering, no kernel code touched.

## Per-call state reset (spec `timing_scope`)

`d_graphInDegree`/`d_left_sum` are both mutated in place by the executor
during a solve, so both are reset before every call, inside the timed
region — matching `benchspecs/sptrsv/spec.yaml`'s `timing_scope` field
("Any per-call reset of algorithm-internal state ... happens INSIDE the
timed region ... as part of a genuine solve call", citing
YuenyeungSpTRSV's own correct practice as the model, which this artifact
also follows for this state). `prepare()` computes the one-shot
STRUCTURAL in-degree template once (`split_sptrsv_indegree`) and caches
it; `run()`'s `split_sptrsv_solve` does a D2D copy of that template into a
scratch buffer + a `d_left_sum` zero, both inside the timed region, before
the executor launch.

## Diagnosis process (Task 1 of this integration)

Before concluding this is an artifact defect and not an adapter/shim
mapping mistake, every assumption the executor kernel makes was checked
directly against the shipped types/constants
(`source/include/common.h`): `ind_type`=`int` (32-bit, matches this
adapter's `int32` CSC arrays — **not** an int32/int64 mismatch), `val_type`
=`double` (matches fp64), `sz_type`=`int`, `WARP_SIZE`=32,
`WARP_PER_BLOCK`=16 (`shim.cu`'s launch geometry, `threads =
WARP_PER_BLOCK*WARP_SIZE=512`, `blocks = ceil(m/16)`, matches the
executor's own `starting_x = blockIdx*WARP_PER_BLOCK` assumption exactly).
Traced the in-degree bookkeeping by hand: the analyser counts, for each
row `i`, 1 (its own diagonal, since the CSC row-index array includes
diagonal entries) + the number of off-diagonal nonzeros in row `i`; the
executor's per-column "producer" loop (`start_ptr =
d_cscColPtr[global_x_id]+1`, i.e. skips the diagonal) increments either a
shared-memory counter (`s_graphInDegree`, in-block case) or decrements the
global counter (`d_graphInDegree`, cross-block case) once per off-diagonal
dependency it resolves; the consumer wait condition
(`s_graphInDegree[local]==d_graphInDegree[global_x_id]`) is satisfied
exactly when all of a row's dependencies have been accounted for, by
construction — the bookkeeping itself is arithmetically correct. No
`ind_type`/`val_type` ctypes mismatch, no unit-vs-stored-diagonal
confusion (the executor divides by `d_cscVal[pos]`, whatever is stored at
the diagonal position — this integration's `L` always stores an explicit
`1.0` there), no missing `d_x`/state-reset issue (all confirmed by direct
code reading, not just gate-number-watching).

### Hand-checkable small systems: PASS (rules out an adapter/shim mapping bug)

Built a tiny 6x6 lower-triangular system by hand (structurally random 0.6
density, U(0.1,1.0) values) and ran it through the **actual production
adapter path** (`adapter.create("fp64").prepare()/.run()/.to_host()`,
`shim.so` unmodified), comparing against an independent dense-loop numpy
forward substitution (not `_forward_substitute`, not scipy — a third,
from-scratch implementation):

| case | rows | blocks | max relative err |
|---|---|---|---|
| tiny-6x6 (hand-built) | 6 | 1 | **0.0** (bit-exact) |
| just-over-one-block-17x17 | 17 | 2 | 1.15e-16 |
| cross-block-40x40 | 40 | 3 | 1.36e-15 |

All pass at fp64 machine-epsilon precision, including a case that
deliberately spans a cross-block dependency (row 16+ depending on a
column processed by block 0). **This rules out a mapping/typing/format
bug in the adapter or shim** for small inputs — the wiring is correct.

### Scale sweep: finds the actual failure boundary

Re-ran `adapter.prepare()/.run()` directly (bypassing the runner, same
`matrices.synthetic` generator the domain's own smoke workloads use) across
a size sweep, `pattern="uniform"`/`"powerlaw"` (same patterns as the
failing smoke workloads), holding everything else fixed:

| rows | blocks | max_rel_err (uniform) | n_bad_rows | max_rel_err (powerlaw) | n_bad_rows |
|---|---|---|---|---|---|
| 16 | 1 | 0.0 | 0 | 0.0 | 0 |
| 64 | 4 | 3.6e-15 | 0 | 4.9e-15 | 0 |
| 128 | 8 | 5.2e-15 | 0 | 2.7e-15 | 0 |
| 256 | 16 | **6.3e+00** | 182 | **2.2e+01** | 211 |
| 1000 | 63 | 7.5e+00 | 871 | 8.5e+01 | 872 |
| 4000 | 250 | 5.2e+00 | 3277 | 5.4e+01 | 3471 |

Correctness holds through 128 rows / 8 blocks and breaks down sharply once
the matrix is large/dependency-dense enough (this specific instance:
256 rows / 16 blocks) — i.e. this is **not** a fixed logic bug that would
reproduce identically regardless of scale (which small hand-built tests
above already ruled out); it is scale/traffic-dependent, the signature of
a genuine concurrency race, not a mapping error.

**Determinism check** (same 256-row case, same seed, 6 back-to-back runs):
5 of 6 runs produced byte-identical error/bad-row sets; 1 of 6 differed by
1 row (182 vs. 183 bad rows, slightly different max error). Reproducible
enough to look "mostly deterministic" for a fixed launch on an
otherwise-similar-load GPU (typical of a GPU race with a narrow timing
window), but the run-to-run variation is real, not zero — confirming a
genuine hardware race rather than a pure, always-identical logic bug.

### Root cause: two commented-out memory fences in the artifact's own executor kernel

`source/include/sptrsv_syncfree_cuda.cuh`'s `sptrsv_syncfree_cuda_executor`
(the exact kernel this shim wraps) has, in its "Producer" phase:

```c++
if (cond) {                                            // in-block (shared mem)
    atomicAdd((val_type *)&s_left_sum[pos], xi * d_cscVal[j]);
    //__threadfence_block();                             <-- commented out
    atomicAdd((int *)&s_graphInDegree[pos], 1);
}
else {                                                  // cross-block (global mem)
    atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
    //__threadfence();                                    <-- commented out
    atomicSub(&d_graphInDegree[rowIdx], 1);
}
```

(lines 219-220 and 224-225 of the vendored file, confirmed via
`grep -n threadfence` against the pristine, unmodified clone — `git diff`
on `source/` is empty, so these lines are exactly as JiyaSu/ParCoreLab
shipped them, not something touched during this integration.) The
`atomicSub(&d_graphInDegree[rowIdx], 1)` (or the shared-memory equivalent)
is the **signal** a dependent row's busy-wait consumer loop polls to
decide its predecessor's contribution has landed; the preceding
`atomicAdd(&d_left_sum[rowIdx], ...)` is the **data** that contribution
actually delivers. These are two independent atomic read-modify-write
operations on two *different* memory addresses — without a fence between
them, the memory subsystem is free to make the signal (`d_graphInDegree`
decrement) visible to a thread on a different SM **before** the data
(`d_left_sum` add) is visible to that same thread. A consumer that reads
`d_left_sum[global_x_id]` right after its wait condition is satisfied can
then read a **stale value that doesn't yet include this producer's
contribution** — silently understating the accumulated partial sum, never
crashing, never hanging, just numerically wrong. This exactly explains
every observed symptom: correct at small scale (few concurrent producer/
consumer pairs, narrow race window rarely hit in practice), wrong and
growing worse at larger scale (more concurrent cross-block traffic, wider
window), and *mostly*-but-not-perfectly deterministic for a fixed input on
a fixed GPU (a real hardware race, not a fixed logic error).

Notably, the artifact's OTHER three synchronization-free kernels in the
same header (`SLFCKernel`, `ELMRKernel`, `ELMCKernel` — level-set-style
kernels used by the paper's other execution modes, not wrapped by this
adapter) all call an ACTIVE, uncommented `__threadfence()` at the
analogous point in their own producer/signal sequences (lines 36, 86, 128).
Only the two fences inside `sptrsv_syncfree_cuda_executor` — the exact
kernel this integration's GPU-only path uses — are commented out. This
strongly suggests an accidental omission/incomplete edit (e.g. a disabled
"try without the fence for a performance experiment" that was never
restored) rather than an intentional design choice, since the closely
analogous kernels in the same file get it right.

### Confirmatory diagnostic (NOT a shipped fix): restoring the two fences eliminates the failure

To confirm this exact mechanism (and rule out any remaining alternative
explanation, e.g. an occupancy/scheduling-order hazard unrelated to memory
ordering), copied `source/include/sptrsv_syncfree_cuda.cuh` to a scratch
directory
(`/tmp/claude-106793/.../scratchpad/fence_probe/`, **not** `source/` or
this directory's shipped `shim.so`), uncommented exactly those two lines
(`__threadfence_block();` / `__threadfence();`, nothing else changed), and
rebuilt a scratch `shim_fenced.so` with the identical `nvcc` flags
`build.sh` uses. Re-ran the exact same scale sweep against that
fence-restored build, adapter/RHS-pool construction otherwise unchanged:

| rows | blocks | max_rel_err (uniform), fences restored | max_rel_err (powerlaw), fences restored |
|---|---|---|---|
| 128 | 8 | 5.7e-15 | 3.6e-15 |
| 256 | 16 | 9.4e-15 | 5.2e-14 |
| 512 | 32 | 5.6e-15 | 1.5e-14 |
| 1000 | 63 | 9.1e-15 | 3.8e-14 |
| 2000 | 125 | 9.8e-15 | 1.4e-13 |
| 4000 | 250 | 1.2e-14 | 1.2e-13 |

**Zero bad rows at every size, all at fp64 machine-epsilon precision** —
the same order of magnitude as `yysptrsv-fused`'s passing gate numbers.
Restoring exactly the two fences the artifact's own source has commented
out, with nothing else touched, fully resolves the correctness failure
across the entire range that failed before. This is conclusive: the root
cause is the artifact's own missing memory fences, not this integration's
adapter/shim mapping.

**This fix is intentionally NOT applied to the shipped `source/`/
`shim.so`.** Per ARTIFACT_GUIDE.md rule 3 ("patch minimally... touching
kernel code is not [fine]") and rule 4 ("the gate is never loosened"), the
shipped artifact stays byte-for-byte the pristine clone and the gate
result stands as a genuine, documented finding — exactly the disposition
used for the `zipserv-decompress` bit-exactness bug found in the
lossless-compression track (see that track's STATUS.md for the same
"root-cause, don't patch the kernel, record the gate failure" pattern).

## Gate verification (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel sptrsv --variant sptrsv-solve-kernel \
    --impl split-sptrsv-syncfree --smoke
```

Result across 3 separate invocations during this integration (numbers
vary run to run, consistent with the race-condition root cause above —
never anywhere near the `1e-9` tolerance):

| run | smoke-uniform max_scaled_err | smoke-powerlaw max_scaled_err |
|---|---|---|
| pre-diagnosis (task brief) | 1.225e+00 | 9.188e+00 |
| post-rebuild (old CUDA_HOME) | 8.525e-01 | 9.932e+00 |
| post-toolchain.sh rebuild | 1.076e+00 | 1.070e+01 |

tol = `1e-9` (fp64, `sptrsv-solve-kernel`'s spec tolerance). **0/2 smoke
workloads valid**, every run. No crash, no hang, no timeout — the gate
runs cleanly to completion and reports a genuine correctness FAILURE.
`conforming: False` is expected (`--smoke`, protocol-overridden
`warmup=5, reps=20`, shared login-node GPU).

## Candidate accounting (sptrsv track, 5 surveyed papers)

See `../yysptrsv/STATUS.md`'s "Candidate accounting" section for the full
table (identical for both files, reproduced in summary here): of the 5
papers in `benchspecs/sptrsv/survey.md` / `output/benchmark_groups.json`'s
`sptrsv` group, only 2 pass ARTIFACT_GUIDE.md's single-NVIDIA-GPU-card
scope filter — this artifact and `yysptrsv-fused` — and **both are
already attempted**. The remaining 3 are scope-skips, not unattempted
candidates: SuperLU_DIST/3D-SpTRSV (`conf/sc/01790S0L23`, SC'23) is
multi-node/multi-GPU (MPI) despite carrying an `nvidia-gpu` platform tag —
the whole paper claim is inter-GPU communication, no single-card entry
point exists; HDagg (`conf/ipps/ZarebavaniCLSD22`, IPDPS'22) and
partially-strided-codelet (`conf/sc/CheshmiCD22`, SC'22) are both CPU-only
(OpenMP/MKL, no GPU code path at all). There is no unattempted third (or
newer) single-GPU candidate in this track. This artifact and
`yysptrsv-fused`, both TPDS'21, are — despite being chronologically the
two OLDEST papers in the 5-paper group — the complete single-GPU-scope
candidate pool, hence trivially "the newest single-GPU artifacts" once
platform scope is applied first, per ARTIFACT_GUIDE.md's own filtering
order.

## Not done (login-node budget, ARTIFACT_GUIDE.md rule 5)

- No real SuiteSparse `recommended_subset` sweep — only smoke workloads and
  the ad hoc diagnostic sizes above were exercised.
- No timing run (compute-node allocation needed) — moot regardless, since
  this impl does not pass the correctness gate; ARTIFACT_GUIDE.md rule 4
  forbids reporting timing for a gate-failing implementation.
- The fence-restored diagnostic build was NOT swept to the same breadth as
  the failing build (6 sizes, both patterns, one seed each) — sufficient
  to confirm the mechanism, not an exhaustive characterization of exactly
  where the race first becomes observable in general (that boundary is
  itself launch-order/timing dependent, as the determinism check shows,
  and is not the point of this diagnosis).
- The CPU-GPU split-execution mode (the paper's actual headline
  contribution — dividing rows between MKL-threaded CPU and this GPU
  kernel) is not exercised at all; this integration only ever targeted the
  GPU-only fallback path (mode 2) per the task's own framing, and that
  path is what turned out to have the bug.
- `sptrsv-analysis-phase` was not separately gated (the one-shot in-degree
  analyser kernel itself has no cross-block synchronization and is not
  suspected of any correctness issue — the bug is isolated to the executor
  kernel's producer/consumer memory ordering).

## Verdict

`split-sptrsv-syncfree: BUILT (gate fails)`. `smoke-uniform`/
`smoke-powerlaw` both INVALID, `max_scaled_err` in the `O(1)`-`O(10)` range
against tol `1e-9`, every run. Root-caused to two commented-out
`__threadfence()`/`__threadfence_block()` calls in the artifact's own,
unmodified `sptrsv_syncfree_cuda_executor` kernel
(`source/include/sptrsv_syncfree_cuda.cuh` lines 219-225) — a missing
memory-ordering guarantee between the "data" (`d_left_sum` atomicAdd) and
"signal" (`d_graphInDegree` atomicSub) halves of its cross-block
producer/consumer protocol, letting a consumer observe the ready signal
before the corresponding data write is globally visible. Confirmed by (1)
hand-checked small systems through the actual adapter path passing at
machine-epsilon precision, ruling out an adapter/shim mapping bug; (2) a
size sweep showing the failure is scale/traffic-dependent, not a fixed
logic error; (3) a determinism check showing near-but-not-perfect
reproducibility, consistent with a genuine hardware race; and (4) a
scratch diagnostic rebuild with exactly those two fences restored (nothing
else changed) passing at fp64 machine-epsilon precision across the entire
failing size range. The fix is NOT applied to the shipped artifact or
`shim.so` (ARTIFACT_GUIDE.md rules 3/4); the gate failure stands as the
recorded result.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, on an
  A100-SXM4-40GB physical card, driver 595.71.05, node gpu-b11-6),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), default host compiler
  found by nvcc's own search (kb-env g++ 13.4.0, no `-ccbin` needed --
  unlike some other artifacts in this repo, this header has no dependency
  on `<type_traits>`/libstdc++ internals that trip up nvcc 12.x + g++ 13,
  so `build.sh` needed zero edits beyond the pre-existing
  `toolchain.sh`-ordering fix already in the working tree), torch
  2.8.0+cu128, Python 3.12.14; `-gencode arch=compute_80,code=sm_80`
  unchanged.
- Build: OK. Only the pre-existing `#68-D` sign-conversion / unused-`start`
  warnings in the vendored `sptrsv_syncfree_cuda.cuh`, same as recorded
  above. Build-system changes: none beyond the `source ... ; cd ...`
  ordering fix already present in `build.sh` before this pass (a prior
  interrupted pass fixed `BASH_SOURCE` resolving against the wrong cwd).
- Gate: `sptrsv-solve-kernel` / `split-sptrsv-syncfree` --
  **HANGS** (does not crash, does not return) partway through
  `smoke-uniform`, confirmed twice: once by hitting a 15-minute `srun`
  wall-clock limit while still printing "running ... smoke-uniform ...",
  and once more deterministically by wrapping the same command in
  `timeout 480` inside the allocation, which killed it at RC=124 after
  8 minutes with no further output past the "running ... smoke-uniform
  ..." line. `smoke-powerlaw` was never reached (workloads run
  sequentially within one process). No GFLOP/s or error numbers were
  produced either time.
- Deviation from the recorded ruling: the FAILURE MODE differs from the
  Perlmutter record above, which states "No crash, no hang, no timeout --
  the gate runs cleanly to completion and reports a genuine correctness
  FAILURE" with finite `max_scaled_err` in the `O(1)`-`O(10)` range. On
  zaratan's A100 (MIG 1g.5gb slice, sm_80, driver 595.71.05) the identical
  `--smoke` invocation against the identical, unmodified `shim.so` instead
  hangs indefinitely inside `smoke-uniform` and never reaches a
  reported-error state. This is consistent with, and does not contradict,
  the same root cause documented above (the two commented-out
  `__threadfence()`/`__threadfence_block()` calls in
  `sptrsv_syncfree_cuda_executor`): the kernel's cross-block producer/
  consumer protocol is a busy-wait spin loop that additionally assumes all
  launched blocks are co-resident for forward progress; without the
  fences, a different GPU/driver/occupancy combination (a 14-SM MIG slice
  here vs. a full A100-PCIE-40GB on Perlmutter) can turn the same missing
  memory-ordering guarantee into a livelock (a consumer spinning on a
  write that never becomes visible to it) rather than a merely-stale read
  that eventually resolves -- both are direct, expected consequences of
  the identical missing-fence bug, not a new or different defect. Per
  rule 9 (bound effort; this defect is already fully root-caused above)
  this was not chased further with a full (non-MIG) `-g a100` allocation,
  which the reproduction protocol reserves for OOM-class failures, not
  hangs whose mechanism is already understood.
- Verdict here: **BUILT (gate fails)** -- equals the recorded ruling.
  `split-sptrsv-syncfree` still does not pass the correctness gate on
  zaratan; only the concrete symptom (hang vs. finite wrong values) is
  hardware-dependent, as expected for an unsynchronized cross-block race.
