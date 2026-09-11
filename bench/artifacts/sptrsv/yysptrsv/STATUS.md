# yysptrsv (YuenyeungSpTRSV) — sptrsv

**Status: BUILT (gate fails on zaratan: hangs, does not return) — build is
byte-for-byte clean and identical to the Perlmutter record below;
`smoke-uniform` never completes (confirmed by two independent deterministic
timeouts, one internal `timeout 1200`). Suspected but NOT confirmed root
cause: SM/occupancy starvation of the artifact's own cross-block busy-wait
synchronization (`d_get_value` "ready" flag spin-wait, see below) on the
14-SM A100 MIG 1g.5gb slice this reproduction ran on; not re-verified on a
full (non-MIG) A100 (queues for hours on this cluster; not attempted here,
see "Reproduction on zaratan" section below for the full reasoning). This
is the SAME failure signature (hang at the same point) independently
observed for `../split-sptrsv` on the identical slice this same session —
see that artifact's own already-confirmed missing-memory-fence root cause,
which is a different but analogous busy-wait hazard.
Earlier outcome, kept for the record: BUILT+GATED — smoke-uniform err
1.87e-15, smoke-powerlaw err 8.18e-15 (both <= tol 1e-9), 2/2 smoke runs
valid, on Perlmutter's full A100-PCIE-40GB.**

- Paper: "YuenyeungSpTRSV: A Thread-Level and Warp-Level Fusion
  Synchronization-Free Sparse Triangular Solve" (TPDS'21).
  `PAPER_KEY = journals/tpds/ZhangSLHWDW21` (matches
  `output/benchmark_groups.json`'s `sptrsv` group and
  `output/included.json`'s artifact URL for this key).
- Artifact: https://github.com/JiyaSu/YuenyeungSpTRSV
- Commit cloned: `e3b4b5f9f11cd8d8eb6463933e009603c60ae4c3` (2021-03-19),
  `git clone --depth 1` into `./source/` (`.git` kept for provenance).
  `git status`/`git diff` inside `source/` are both empty — zero patches to
  the artifact's own code (the one compatibility fix lives entirely in
  `shim.cu`, see below).
- Toolchain: `nvcc` 12.9
  (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`), `-gencode
  arch=compute_80,code=sm_80` (A100-PCIE-40GB). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python` (torch 2.8.0+cu128).
  Runtime needs `LD_PRELOAD=/usr/lib64/libstdc++.so.6` (project-wide
  CXXABI/GLIBCXX fix).
  **Toolchain pin note (verified 2026-09-04, after this artifact was
  originally built in August):** the login node's default loaded module is
  now `cudatoolkit/13.2` (HPC SDK 26.5), so a bare `command -v nvcc` on a
  fresh shell resolves to 13.2, not 12.9, and that module also exports
  `CPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/math_libs/13.2/include:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/include`
  — so even pinning `PATH` to the 12.9 `nvcc` binary is not sufficient by
  itself, since 12.9's compiler would still pick up 13.2's headers ahead of
  its own via that `CPATH`. `build.sh` now `source`s a shared
  `bench/artifacts/toolchain.sh` (used project-wide, not written for this
  track) which pins `CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`,
  prepends its `bin` to `PATH`, and **re-points** (not clears) `CPATH` at
  12.9's own `math_libs/12.9/include` + `cuda/12.9/include` — clearing
  `CPATH` outright would break anything needing `cusparse.h` (not required
  by this plain-nvcc shim itself, but this is the uniform recipe used
  across every artifact directory). Re-ran `build.sh` under the current
  (13.2-default) environment with this pin in place: builds clean (only
  the artifact's own pre-existing `tranpose.h` sign-conversion/unused-
  variable warnings, unchanged from the original build), and the gate
  command below was re-run afterward with identical results — the
  already-built `shim.so` from August was unaffected by the module change,
  this pin only matters for any future rebuild. (The harness's own
  `kernelbench.runner` separately auto-detects and corrects a `CUDA_HOME`/
  `CPATH` mismatch at import time — see its `[env] toolchain: ...` line in
  the gate output below — but `build.sh` pins explicitly rather than
  relying on that, since a bare `nvcc` build.sh invocation is outside the
  runner's process.)

## What the artifact actually is

YuenyeungSpTRSV fuses two synchronization-free SpTRSV design points into
one kernel: within a group of `WARP_SIZE=32` consecutive rows, rows whose
average nnz/row falls below a `border` threshold (10, the artifact's own
`main.cu` default) are solved **thread-level** (one thread per row, cheap
rows), and rows above the threshold are solved **warp-level** (one warp per
row, expensive/dense rows) — `yySpTRSV_csr_kernel`
(`source/CUDA/YYSpTRSV.h`) picks the strategy per row-group via a
precomputed partition table. The host-side partitioning step
(`matrix_warp`, `source/CUDA/tranpose.h`) that decides this split IS the
paper's "analysis" contribution (its own abstract: "does not need long
preprocessing time to calculate levels") — pure CPU code, no CUDA. The
upstream repo's own CLI (`CUDA/main.cu`) reads a `.mtx` file from disk and
calls `YYSpTRSV_csr(...)`, a C++ function that ALSO owns its own internal
`BENCH_REPEAT=100` mean-timing loop (survey.md: mean not median, no
warmup) — not called here at all, per ARTIFACT_GUIDE.md rule 1.

## Kernel entry points wrapped (not the CLI/benchmark loop)

`shim.cu` (this directory, **not** part of the artifact) `#include`s
`source/CUDA/tranpose.h` and `source/CUDA/YYSpTRSV.h` verbatim and exposes
3 `extern "C"` functions:

- `yy_warp_partition` — calls the artifact's own `matrix_warp` (host-side,
  CPU-only) unmodified, `border=10` (the artifact's own `main.cu` default).
  Called ONCE, from `prepare()` — this is the "one-time analysis" cost
  `benchspecs/sptrsv/spec.yaml`'s `sptrsv-analysis-phase` variant gives its
  own first-class timing to; folded here into `prepare()`'s single timed
  preprocessing cost for `sptrsv-solve-kernel`.
- `yy_solve` — resets the three pieces of per-call algorithm state
  `yySpTRSV_csr_kernel` mutates in place (`d_get_value`, the per-row "ready"
  flag array the thread-level path spin-waits on; `d_x`; `d_id_extractor`,
  an atomic counter for dynamic warp-level work-item assignment), then
  launches the artifact's own `yySpTRSV_csr_kernel` unmodified. The reset
  trio matches the artifact's OWN correct per-iteration practice inside
  `YYSpTRSV_csr`'s `BENCH_REPEAT` loop (`source/CUDA/YYSpTRSV.h` lines
  213-215 do the identical 3 `cudaMemset`s before every timed launch), and
  is explicitly sanctioned by `benchspecs/sptrsv/spec.yaml`'s
  `timing_scope` field as belonging INSIDE the timed region ("this matches
  YuenyeungSpTRSV's correct cudaMemset-before-timing practice"). Both the
  reset and the kernel launch are inside the one function
  `CudaEventTimer` wraps.
- `yy_sync` — `cudaDeviceSynchronize()` + `cudaGetLastError()`, used once
  after the analysis call in `prepare()` (not used per-solve; `run()`
  relies on the harness's own event-based timer to synchronize).

No kernel line in either header was edited except the one build-compat
macro below.

## Patch: `__shfl_down` CUDA-12 compatibility macro (build-system fix, not a kernel change)

`source/CUDA/YYSpTRSV.h` line 91 calls the pre-CUDA-9 **unmasked**
`__shfl_down(val, offset)` inside `yySpTRSV_csr_kernel`'s warp-level
reduction path. CUDA 9+ removed the unmasked warp-shuffle intrinsics in
favor of the `_sync` forms that take an explicit participation mask; CUDA
12 (this project's toolchain) has no fallback declaration for the old
name at all. `shim.cu` scopes a macro around the two `#include`s only:

```c++
#define __shfl_down(val, offset) __shfl_down_sync(0xffffffff, (val), (offset))
#include "tranpose.h"
#include "YYSpTRSV.h"
#undef __shfl_down
```

`0xffffffff` (full-warp mask) is the correct semantic equivalent here: the
call site in `yySpTRSV_csr_kernel`'s warp-level branch has no divergent
branch before it — every lane of the warp reaches the shuffle
unconditionally — so the full mask reproduces exactly what the
pre-Volta unmasked form implicitly assumed (independent-thread-scheduling
hardware did not exist yet when this code was written; all lanes
participating was the only possibility). `<cuda_fp16.h>` is `#include`d
in `shim.cu` **before** this macro is defined specifically so CUDA's own
genuine 3-argument `__shfl_down(__half/__half2, delta, width)` overloads
get declared first and are not shadowed by the macro (the macro only
applies inside the scoped region, and only to the unqualified 2-argument
call form the vendored header actually uses). The vendored file on disk
(`source/CUDA/YYSpTRSV.h`) is never modified — confirmed by `git diff`
above being empty.

**Probe build without the macro, run for this STATUS.md** (this directory's
own claim, verified rather than just asserted): copied `shim.cu` to a
scratch file with the `#define __shfl_down(...)` / `#undef __shfl_down`
lines deleted (nothing else changed) and compiled it with the exact same
`nvcc` invocation `build.sh` uses, into
`/tmp/claude-106793/.../scratchpad/yysptrsv_probe/`. Result: compilation
**fails**, exactly as documented:

```
YYSpTRSV.h(91): error: identifier "__shfl_down" is undefined
              sum += __shfl_down(sum, offset);
1 error detected in the compilation of "shim_nomacro.cu".
```

Confirms the macro is load-bearing for this toolchain, not defensive
boilerplate.

## Format (no conversion needed — the artifact's own native layout)

Plain CSR of `L`, diagonal LAST per row — this IS the artifact's own
native format (its abstract: "users do not need to conduct format
conversion") and exactly what
`kernelbench.impls.cpu_ref.unit_lower_triangular` already produces after
`.sort_indices()`: `L` is lower-triangular, so within any row every
off-diagonal entry has `col < row` and the diagonal has `col == row` (the
row-maximum column), hence sorted last. `prepare()` therefore does **less**
format work than `split-sptrsv`'s CSR->CSC step — no reordering beyond the
one `sort_indices()` call already present in the shared derivation helper.

**Diagonal-value note (a real artifact quirk, disclosed rather than
silently sidestepped):** `yySpTRSV_csr_kernel` computes
`x_i = (b_i - sum) / d_csrVal[rowend-1]` — it divides by whatever nonzero
happens to be the LAST stored entry in the row, not an assumed-unit
diagonal, despite the artifact's own `change2tran` comment calling its
output "the unit-lower triangular sparsity structure" (survey.md flagged
this exact inconsistency independently). This integration's canonical `L`
(`unit_lower_triangular`, shared by every sptrsv competitor in this
benchmark) always stores an EXPLICIT diagonal value of `1.0` as the
row-maximum entry, so in practice `d_csrVal[rowend-1] == 1.0` here and the
artifact's "whatever sits last" behavior and a genuine unit-diagonal
convention coincide for this integration's inputs — but this is a property
of the shared workload derivation, not something `yySpTRSV_csr_kernel`
itself guarantees or checks (it would silently divide by a wrong value if
fed a factor whose row-maximum happened not to be the diagonal).

## Preprocessing split (ARTIFACT_GUIDE.md rule 2)

`prepare()` = `unit_lower_triangular()` (shared workload derivation,
timed as ordinary Python/scipy preprocessing) + `yy_warp_partition()`
(the artifact's own one-shot thread/warp analysis) + H2D upload of
`rowptr`/`colidx`/`val`/`warp_num` + fixed RHS-pool construction. `run()`
= **one** `yy_solve()` call (state reset + kernel launch, both timed).

## RHS pool

Identical construction to `kernelbench.impls.cpu_ref.ScipySpTRSV.prepare()`
(same seed/count read from the same `params` dict every impl in a run
receives): `X ~ U(0,1)` draws, `B = L @ X.T` via one fp64 scipy sparse
matvec (never via the technique under test), cycled by a cursor across
`run()` calls. A real, adapter-level bug was found and fixed here during
this integration: `(L64 @ X.T).T` is a **transposed view** (strided, not
C-contiguous) — confirmed via a standalone repro — and `torch.as_tensor`
preserves that non-contiguous layout, while `shim.cu`'s `yy_solve` treats
`d_b` as a flat contiguous pointer. An uncopied strided view would have fed
the solver scrambled right-hand sides. Fixed with
`np.ascontiguousarray(...)` before the device upload — an adapter-side
operand-construction bug, not an artifact bug (see `adapter.py`'s own
docstring, same fix pattern used in `split-sptrsv`).

## Gate verification (login node, functional/gate check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel sptrsv --variant sptrsv-solve-kernel \
    --impl yysptrsv-fused --smoke
```

Result (both the original build's run, and re-run after this STATUS.md's
toolchain-pin verification, agree to the digit): **2/2 smoke workloads
valid** (`smoke-uniform`, `smoke-powerlaw` — `smoke-banded` is excluded
from this kernel's own smoke set by `kernelbench/domains/sparse.py`,
independent of this artifact, because the canonical unit-lower-triangular
derivation makes that pattern catastrophically ill-conditioned for every
sptrsv competitor, not specific to this impl):

| smoke workload | max_scaled_err | tol |
|---|---|---|
| smoke-uniform | 1.87e-15 | 1e-9 |
| smoke-powerlaw | 8.18e-15 | 1e-9 |

Both at fp64 machine-epsilon-level agreement — comfortably within the
spec's `sptrsv-solve-kernel` tolerance (`< 1e-9` fp64), and notably tighter
than the artifact's own validation practice (a 1% relative tolerance per
survey.md — the loosest of the 5 surveyed sptrsv papers). `conforming:
False` is expected and correct (`--smoke`, protocol-overridden
`warmup=5, reps=20`, shared login-node GPU — not a publishable timing run,
per ARTIFACT_GUIDE.md rule 5).

## Candidate accounting (sptrsv track, 5 surveyed papers)

`benchspecs/sptrsv/survey.md` and `output/benchmark_groups.json`'s
`sptrsv` group both list the same 5 papers. Per ARTIFACT_GUIDE.md's scope
ruling (NVIDIA GPU, single-card only), only **2 of 5** pass the platform
filter — this artifact and `split-sptrsv` — and **both are already
attempted**:

| paper | key | year | platform | verdict |
|---|---|---|---|---|
| YuenyeungSpTRSV | journals/tpds/ZhangSLHWDW21 | 2021 | nvidia-gpu | **attempted (this dir)** |
| Split_SpTRSV | journals/tpds/AhmadYU21 | 2021 | cpu, nvidia-gpu | **attempted (`../split-sptrsv`)** |
| SuperLU_DIST (3D SpTRSV) | conf/sc/01790S0L23 | 2023 | cpu, nvidia-gpu, **distributed** | scope-skip: despite an `nvidia-gpu` tag, the paper's actual claim is a multi-node/multi-GPU (MPI) 3D communication-avoiding algorithm scaling to 256 GPUs (survey.md #1) — there is no single-card entry point; the whole point of the artifact is inter-GPU communication. Out of "single-card" scope. |
| HDagg | conf/ipps/ZarebavaniCLSD22 | 2022 | cpu | scope-skip: CPU-only (OpenMP/MKL, 40-core Cascade Lake nodes per survey.md #2), no GPU code path. |
| partially-strided-codelet | conf/sc/CheshmiCD22 | 2022 | cpu | scope-skip: CPU-only (same Sympiler `FusionDemo` CPU harness as HDagg per survey.md #3), no GPU code path. |

So this track has exactly **2** single-GPU-card candidates in its entire
5-paper survey, both already integrated — there is no unattempted third
(or newer) single-GPU candidate to add. Note the two ARE, chronologically,
the two OLDEST papers in the group (both TPDS'21, vs. 2022/2023 for the 3
scope-excluded ones) — "newest single-GPU artifacts" here means newest
*within the scope-filtered pool*, which after platform filtering happens
to equal "all of them," not a coincidence of recency ranking ignoring
scope. This mirrors the `mttkrp`/`blco` track's own thinness finding (that
track had exactly 1 GPU candidate out of 4).

## Not done (login-node budget, ARTIFACT_GUIDE.md rule 5)

- No real SuiteSparse `recommended_subset` sweep (`apache2`, `ecology2`,
  `thermal2`, ... — 21 matrices) — only the 2 synthetic smoke workloads
  were gated.
- No timing run (compute-node allocation needed; this integration is
  login-node build+gate only). The numbers in the gate table above are
  from a `--smoke`-overridden, reduced-protocol, shared-GPU run, explicitly
  marked `conforming: False` by the runner — not publishable timing.
- `sptrsv-analysis-phase` and `sptrsv-e2e-amortized` variants were not
  separately exercised (no analysis-phase-only or k-solve-amortized gate
  run); `sptrsv-distributed-scaling` does not apply to this artifact
  (single-GPU only).
- fp32 (the spec's secondary precision) was not exercised — `VALUE_TYPE`
  is compiled as `double` unconditionally in this build (`source/CUDA/
  common.h`'s default); building a second fp32 `shim.so` was not attempted.

## Verdict

`yysptrsv-fused: BUILT+GATED`. `smoke-uniform`/`smoke-powerlaw` both valid
at fp64 machine-epsilon precision (`1.87e-15`/`8.18e-15`, tol `1e-9`). One
adapter-level bug found and fixed (transposed-view RHS buffer, not an
artifact bug). One build-system compatibility macro (`__shfl_down` ->
`__shfl_down_sync`, CUDA-9+ API removal), verified load-bearing via a
probe build that fails without it. Zero lines of the artifact's own source
touched (`git diff` on `source/` is empty). Toolchain pinned explicitly in
`build.sh` against this login node's now-default `cudatoolkit/13.2` module
(`CUDA_HOME`/`PATH`/`unset CPATH`), verified by a clean rebuild + identical
gate re-run under the current environment.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, on an
  A100-SXM4-40GB physical card, driver 595.71.05, node gpu-b11-6),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), default host compiler
  found by nvcc's own search (kb-env g++ 13.4.0, no `-ccbin` needed, same
  situation as `../split-sptrsv`), torch 2.8.0+cu128, Python 3.12.14;
  `-gencode arch=compute_80,code=sm_80` unchanged. `build.sh` needed zero
  edits beyond the pre-existing `toolchain.sh`-ordering fix already in the
  working tree (a prior interrupted pass fixed `BASH_SOURCE` resolving
  against the wrong cwd).
- Build: OK. Only the pre-existing `#68-D` sign-conversion (`tranpose.h`
  line 31, `unsigned int min=-1`) / unused-`min2`-variable warnings, same
  as recorded above. Build-system changes: none.
- Gate: `sptrsv-solve-kernel` / `yysptrsv-fused` -- **HANGS**, does not
  return, confirmed twice: once by hitting a 12-minute `srun` wall-clock
  limit while still printing "running yysptrsv-fused smoke-uniform ...",
  and once more with a `timeout 1200` (20 min) wrapper inside a 22-minute
  allocation, which killed it at RC=124 with no further output past that
  same line. `smoke-powerlaw` was never reached.
- Deviation from the recorded ruling: significant -- the Perlmutter record
  above is a clean 2/2-valid pass at fp64 machine-epsilon precision
  (`1.87e-15`/`8.18e-15` vs tol `1e-9`). On zaratan's A100 MIG 1g.5gb slice
  the identical, unmodified `shim.so` (built from the identical, unpatched
  `source/` -- `git diff` on `source/` is empty here too) instead hangs
  indefinitely inside `smoke-uniform` and never reports a result of either
  kind. Two candidate explanations were considered:
  1. **SM/occupancy starvation (suspected, most likely)**: this artifact's
     own thread-level path uses a per-row `d_get_value` "ready" flag array
     that a spin-wait busy-loop polls across blocks (documented above,
     "Kernel entry points wrapped" section) -- the same class of
     cross-block busy-wait design as `../split-sptrsv`'s executor kernel,
     which hung identically on the SAME slice this same session (see that
     artifact's own confirmed missing-`__threadfence()` root cause). Such
     designs typically assume enough blocks are co-resident on the SM
     array simultaneously for forward progress; the MIG 1g.5gb slice here
     has only ~14 SMs vs. a full A100-PCIE-40GB's 108 on Perlmutter, so a
     workload whose block count fits full-card occupancy but not a 14-SM
     slice's could genuinely deadlock on the slice while completing
     normally on a full card. This is circumstantial (matching failure
     signature + a documented busy-wait design in this artifact's own
     code), not confirmed by a fence-restoration-style direct experiment
     the way `split-sptrsv`'s bug was.
  2. **Node/environment issue**: considered and judged less likely --
     `squeue` during this session showed other `cunyang` jobs concurrently
     running on the SAME node (`gpu-b11-6`) on presumably different MIG
     instances (28 independent `a100_1g.5gb` slices are configured on that
     node per `sinfo`/`scontrol show node`), which should be SLURM-isolated
     from this job's own slice; a driver-level fault would more typically
     surface as an explicit CUDA error rather than a silent, content-free
     hang reaching the identical line both times.
  A full (non-MIG) `-g a100` re-check was **not attempted**: `sinfo`/
  `squeue -p gpu-a100` showed several other users' jobs already pending
  on that partition (`PD`, reasons `Priority`/`Resources`/
  `AssocGrpBillingMinutes`), consistent with the gpu_run.sh header's own
  "queues for hours" warning; combined with rule 9's effort-bounding and
  five more directories still queued in this reproduction pass, spending
  an open-ended queue wait to test a hypothesis about a HANG (the
  documented `-g a100` exception is phrased for a `CUDA OOM` failure
  specifically, not a hang) was judged out of scope for this pass. This is
  flagged here explicitly as an open question for a follow-up pass with a
  larger time budget, not silently resolved either way.
- Verdict here: **BUILT (gate fails: hangs, does not complete)** -- this is
  the REVERSE of the recorded ruling (a previously BUILT+GATED impl whose
  gate does not complete on this hardware). The top bold line above has
  been updated accordingly per the reproduction protocol, with the
  Perlmutter PASS kept for the record. This should NOT be read as
  disproving the Perlmutter result (which has its own independent,
  detailed gate table) -- the most likely explanation is a hardware/
  occupancy mismatch specific to the MIG slice used here, not a newly
  discovered defect in `yysptrsv-fused` itself, but that explanation is
  unconfirmed and this reproduction's own observed outcome (hang, no
  result) is recorded as-is per instructions.
