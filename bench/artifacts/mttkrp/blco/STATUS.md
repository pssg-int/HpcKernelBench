# blco (BLCO) — mttkrp

**Status: BUILT+GATED**

- Paper: "Efficient, Out-of-Memory Sparse MTTKRP on Massively Parallel
  Architectures" (ICS'22). `PAPER_KEY = conf/ics/NguyenHCLTSRPC22` (matched by
  title in `../../../output/included.json`).
- Artifact: https://github.com/jeewhanchoi/blocked-linearized-coordinate
- Commit cloned: `1a6e4fcaabab0a0611acc5896fd304e7bacc47b4`, `git clone
  --depth 1` into `./source/` (`.git` kept for provenance).
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  host compiler `g++-12` (SUSE 12.3.0) — nvcc 12.9 rejects the system default
  g++-14 on `<bits/alloc_traits.h>` (`__has_construct is undefined`, a
  host-compiler/libstdc++ mismatch, same fix already used by
  `spmm/inferfast`, `spmm/rassm`, `spgemm/ocean` in this repo).
  `-gencode=arch=compute_80,code=sm_80` (A100). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`. Runtime needs
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` (same CXXABI/GLIBCXX fix documented
  throughout this repo's other CUDA adapters). No MKL/OpenBLAS, no
  cuBLAS/cuSOLVER linkage — see "What was built" below.

## What the artifact actually is

BLCO (Blocked Linearized CoOrdinate) is the paper's own GPU sparse-tensor
format for MTTKRP: nonzero coordinates are bit-interleaved into one
linearized ALTO-style index (`source/include/alto.hpp`'s `gen_alto`), then
re-split into fixed-size, GPU-resident "blocks"
(`source/include/blco.hpp`'s `gen_blcotensor_host`/`gen_blcotensor_device`).
The upstream repo builds this into `cpd64`/`cpd128`, end-to-end CLI binaries
for CP-ALS tensor decomposition (`-p`/`--bench` mode restricts them to a
pure MTTKRP benchmark loop).

## What was built (not the CLI binary)

`build.sh` does **not** use `source/Makefile`'s default target (`cpd64`,
which also links `main.cpp`, `cpd_gpu.cu` — CP-ALS's pseudoinverse step,
needing cuBLAS/cuSOLVER + MKL/OpenBLAS — and the synthetic-tensor-generation
files). Instead it compiles only the object set BLCO's own isolated MTTKRP
entry point actually needs: `source/src/{common,sptensor}.cpp`,
`source/src/{utils,kruskal_model,blco,alto_dev}.cu`,
`source/src/kruskal_model.cpp`, plus this directory's `wrapper.cu`, into one
`libblco_wrapper.so` ctypes loads. Verified by grep: none of these files
call a single cuBLAS/cuSOLVER/BLAS function (`utils.cu`'s
`check_cublas`/`check_cusolver` only switch on enum values from the headers)
— so this build has **zero** BLAS/cuBLAS/cuSOLVER dependency, a genuine
simplification versus the artifact's own default build.

## Kernel entry point wrapped (ARTIFACT_GUIDE.md rule 1)

`wrapper.cu` (this directory, **not** under `source/`) exposes 6 `extern
"C"` functions that call BLCO's own, unmodified functions directly:
`blco_build` (`CreateSparseTensor` -> `gen_blcotensor_host<IType>`),
`blco_set_factor`, `blco_upload` (`gen_blcotensor_device` +
`send_blcotensor_over` + `send_masks_over` + `make_device_copy(KruskalModel*)`),
`blco_run` (**one** call to `mttkrp_alto_dev_onemode<IType>` —
`source/include/alto_dev.hpp`'s own doc comment: "Single MTTKRP execution,
used by both benchmark MTTKRP and CPD driver" — the exact function BLCO's
own benchmark driver, `mttkrp_alto_dev`, calls inside its per-iteration
timing loop; called here directly instead of through that loop or the CLI's
`.tns` file I/O), `blco_get_output`, `blco_free`. `wrapper.cu` touches zero
lines of `source/` beyond the one documented bug fix below.

**Kernel variant: BLCO Level-1** (`kernel_id=1`), not `AUTO`(`10`, the
README's own recommended default, which needs additional partial-matrix
allocation/reduction plumbing for small-target-mode-length tensors). Level-1
is one of BLCO's own already-implemented, always-valid kernel variants,
exercising the SAME BLCO format (the paper's actual contribution) as
Level-3/AUTO — only the per-block reduction strategy differs. See
`adapter.py`'s docstring for the full reasoning; `kernel_id` could become a
`params` knob in a future revision.

## Preprocessing split (rule 2)

`prepare()` = `blco_build()` (COO -> BLCO, the artifact's own format
conversion) + `blco_set_factor()` x(order-1) + `blco_upload()` (host->device
transfer). `run()` = **one** `blco_run()` call (pure MTTKRP compute over an
already GPU-resident BLCO tensor, no data movement).

## In-memory path only (task brief: "use the in-memory path if available")

`blco_build()` uses `max_block_size = nnz` (**one** block, matching
`source/src/main.cpp`'s own `if (!stream_data) max_block_size = X->nnz;`);
`blco_upload()`/`blco_run()` both hardcode `stream_data=false`, matching
`mttkrp_alto_dev`'s own non-streaming branch (`num_streams =
at_host->block_count`, one synchronous `send_blcotensor_over` upfront,
never re-transferred). BLCO's own **out-of-memory streaming path**
(`stream_data=true`, re-transferring each block from host on EVERY
`mttkrp_alto_dev_onemode` call — the paper's actual headline contribution
for tensors that overflow GPU memory: Amazon/Patents/Reddit, Table 2) is
**not exercised** by this adapter — disclosed, not hidden, per the task
brief's explicit single-GPU-scope instruction.

## Correctness gate discipline

`prepare()` imports `kernelbench.domains.tensor._make_factors` directly
(not reimplemented) so this adapter's factor matrices are numerically
IDENTICAL to `reference_mttkrp`'s own (`np.random.default_rng(seed +
m).uniform(-1, 1, ...)` per mode) before being copied into BLCO's
`KruskalModel` via `blco_set_factor()`. `BLCO`'s own
`KruskalModelRandomInit` (a `rand_r`-based generator) is never called. This
matters: `kernelbench.impls.gpu_cuda.TorchMTTKRP` (this domain's own,
"never instantiated/run" CUDA wiring) generates factors via a *different*
RNG (`torch.Generator`/`torch.rand`) than `_make_factors`'s numpy
`default_rng` — had it ever been run through the gate as-is, its output
would be computed against different factor values than the reference used,
and the gate would fail for a reason having nothing to do with MTTKRP
correctness. Flagged here as a latent harness-side (not BLCO) issue; this
adapter avoids it by construction, per ARTIFACT_GUIDE.md rule 4 (gate
against the SAME independent reference every other impl uses, never a
self-certifying or mismatched one).

## Real artifact bugs found (both fixed at the wrapper/`source/` boundary,
zero MTTKRP kernel-arithmetic code touched)

**Bug 1 — `gen_alto()` frees uninitialized pointers (`free(): invalid
pointer` crash), `source/include/alto.hpp`.** `AltoTensor<LIT>` declares
in-class default member initializers (`prtn_ptr = nullptr`,
`prtn_intervals = nullptr`, `cr_masks = nullptr`, and 3 more under
`OPT_ALTO`), but `gen_alto()` allocates the struct via `AlignedMalloc`
(plain `posix_memalign` — no constructor runs, so those initializers never
take effect) and never explicitly sets those 3 fields; only `create_alto()`
(a *different* code path, not used by BLCO's own GPU MTTKRP flow) allocates
them. `destroy_alto()` unconditionally frees all of them regardless of
which path built the tensor. `blco.hpp`'s `gen_blcotensor_host()` — called
by every BLCO GPU MTTKRP run, including the artifact's own `main.cpp` CLI —
calls `gen_alto()` then unconditionally `destroy_alto()`, so it frees 3
garbage pointers on every single invocation. Reproduced 100% of the time
inside this project's ctypes-embedded process (a heap with prior
allocation/free activity from Python/torch/CUDA-context init reliably
returns non-zero "fresh" memory from `posix_memalign`, unlike a pristine
freshly-`exec`'d CLI process, which likely explains why the upstream
artifact's own minimal-process usage may not always surface this). **Fix**:
one line, `memset(_at, 0, sizeof(AltoTensor<LIT>));` immediately after
allocation in `gen_alto()` (`source/include/alto.hpp`, +14 lines total incl.
comment; `git -C source diff --stat`: `include/alto.hpp | 14 ++`). Pure
memory-safety fix — zero change to any algorithm, numerical result, or
kernel-launch code. Kept in `source/` (not the wrapper) because the bug is
inside a header-only function template our wrapper calls as a black box;
there is no way to intercept "between" `gen_alto` and `destroy_alto` from
outside without duplicating `gen_blcotensor_host`'s ~80-line body verbatim
into `wrapper.cu`, which would be a larger, less honest footprint than this
1-line fix.

**Bug 2 — MTTKRP output silently corrupted by uninitialized-buffer
atomicAdd (all-NaN or wrong-but-not-NaN rows), affects BOTH this wrapper
AND the artifact's own CLI usage.** Every `mttkrp_lvl1_*_kernel` (and every
other kernel variant) in `source/src/alto_dev.cu` accumulates into `output`
via `atomicAdd(output + output_row * rank + i, value)` — never a plain
store. The output buffer is `KruskalModel::U[target_mode]`
(`CreateKruskalModel`, `source/src/kruskal_model.cpp`), allocated via
`AlignedMalloc` and **never zeroed by BLCO's own code**: `main.cpp`'s CLI
flow calls `KruskalModelRandomInit` (fills ALL factor matrices, including
the target-mode slot, with pseudorandom `U(0,1)` noise) and then calls
`mttkrp_alto_dev` directly — with no `memset` in between, in EITHER the
plain `-p`/`--bench` path or the `-c`/`do_check` correctness-verification
path. Two distinct consequences: (a) `-p -m <iters>` (their own benchmark
mode, `iters>1`) re-runs `mttkrp_alto_dev_onemode` in a loop that
atomicAdd's onto the SAME buffer every iteration without ever re-zeroing
between iterations, so the reported "MTTKRP time" is real, but the output
values it computes are the sum of `iters` independent MTTKRP passes plus the
original random noise, not one correct MTTKRP result — irrelevant for
*their* timing-only use case, but not something a caller could safely reuse
the output from; (b) even `do_check`'s single-iteration
(`max_iters=1`) correctness-verification path adds the true MTTKRP result on
top of that same leftover `U(0,1)` noise before comparing against a
from-scratch CPU reference, meaning their own verification tolerance would
need to absorb a small additive bias per row.

Directly observed in this integration: `blco_build()` does not call
`KruskalModelRandomInit` at all (this adapter supplies its own,
gate-matching factors via `blco_set_factor()` — see "Correctness gate
discipline" above), so the target-mode buffer here starts as **raw,
un-randomized `AlignedMalloc` heap garbage**. Reproduced directly against
`kernelbench.domains.tensor.reference_mttkrp`: 3 of 30 output rows on
`smoke-3d-tiny` (shape `(30,25,20)`, `nnz=400`, mode 0) came back entirely
`NaN` despite every row having >=1 contributing nonzero (confirmed:
`np.unique(w.indices[:,0])` covers all 30 rows) — consistent with
uninitialized bytes happening to decode as a NaN bit pattern for those 3
specific rows, then `NaN + (correct partial sum) = NaN` for every
subsequent atomicAdd into that same row. **Fix, in `wrapper.cu`'s
`blco_upload()`** (not `source/`): `cudaMemset(h->M_dev->U[h->target_mode],
0, sizeof(FType) * h->tmode_length * h->rank)` immediately after
`make_device_copy(h->M)`, before any kernel call. After this fix, all rows
match `reference_mttkrp` to `2e-16`-`4.6e-16` (fp64 machine-epsilon-level
agreement), confirmed across every mode of both smoke tensors and R in
{16, 32, 64} (21/21 combinations checked directly, see "Gate verification"
below) — proving the true MTTKRP computation itself (the atomicAdd
accumulation, the ALTO/BLCO coordinate decode, the per-mode gather) is
correct; only the missing zero-init was ever wrong.

## Gate verification (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel mttkrp --variant mttkrp-general-kernel-fp64 \
    --impl blco-mttkrp-lvl1 --smoke
```
Result: **3/3 runs valid** (`smoke-3d-small`, `smoke-3d-tiny`,
`smoke-4d-small`, all mode 0, R=32 default): `max_scaled_err = 3.09e-16 /
3.56e-16 / 3.69e-16`, all `<= tol 1e-6`.

Same command with `--variant mttkrp-general-e2e-preproc`: **3/3 runs
valid**, same error magnitudes (this variant amortizes preprocessing over
k=100 calls but uses the same correctness gate).

Beyond the required `--smoke` invocation, directly verified (standalone
script, not wired into the runner) across **every mode** of both smoke
tensors and **all 3 spec-recommended ranks** (R in {16, 32, 64}) — 21/21
combinations, `max_scaled_err` in `[2.3e-16, 4.6e-16]` throughout, i.e.
uniformly at fp64 machine-epsilon precision regardless of which mode is the
MTTKRP target or which rank is used:

| tensor | modes checked | R checked | err range |
|---|---|---|---|
| smoke-3d (60,50,40) nnz=2000 | 0,1,2 | 16,32,64 | 2.48e-16 – 4.05e-16 |
| smoke-4d (20,18,16,14) nnz=1200 | 0,1,2,3 | 16,32,64 | 2.30e-16 – 4.62e-16 |

Reduced-protocol numbers only (warmup=5, reps=20, shared login-node GPU),
explicitly marked non-conforming by the runner (`conforming: False`); not a
timing result per ARTIFACT_GUIDE.md rule 5. No timing sweep was run.

## Track thinness (mttkrp)

mttkrp is the smaller of this integration pass's two thin tracks. Its
survey (`benchspecs/mttkrp/survey.md`) covers 4 papers; **BLCO is the
track's only NVIDIA-GPU single-card artifact** (this integration's scope
per `ARTIFACT_GUIDE.md`'s scope ruling). The other 3 are **CPU-only,
platform-scope skips — cheap skip, no build attempt** (per the scope
ruling: "CPU-only ... artifacts are SKIPPED with a one-line reason...no
build attempt"), evidenced directly by each paper's own stated hardware in
`survey.md`:

- **SySTeC** (`conf/cgo/PatelAA25`, CGO'25) — "single core[d], ... 12-core
  dual-socket Intel Xeon E5-2680 v3 @ 2.50GHz, 128GB RAM. ... Julia v1.10,
  Finch v0.6.22" (survey.md). No GPU code path exists in the paper's own
  evaluation.
- **SymProp** (`conf/ipps/0001S0K25`, IPDPS'25) — "Single node, 32 cores (2x
  AMD EPYC 7302 16-core), OpenMP with `OMP_PROC_BIND=spread`" (survey.md).
  CPU-only S³TTMc/S³TTMcTC implementation, no GPU variant.
- **WACO** (`conf/asplos/WonMEA23`, ASPLOS'23) — "code generated via TACO,
  compiled with icc `-O3 -qopenmp`... Hardware: dual-socket 24-core/
  48-thread Intel Xeon E5-2680 v3, `numactl --interleave=all`" (survey.md).
  CPU-only learned format/schedule autotuner, no GPU kernel.

With BLCO integrated, this track's GPU-candidate list is exhausted for this
pass (0 other GPU artifacts to evaluate).

## Not done

- Only the smoke synthetic-tensor set was gated (login-node budget, rule
  5) — no run against the spec's real FROSTT `recommended_subset` (nips,
  uber, chicago-crime, ...); those `.tns` files are not cached locally
  (`kernelbench.domains.tensor.load_workload` never auto-downloads them,
  by design) and would need a compute-node timing allocation regardless.
- `cpd128` (`LIT=unsigned __int128`, needed for tensors whose modes need
  >64 total ALTO-linearization bits, e.g. `nell-1` per BLCO's own paper
  Table 2) was not built — `MAX_NUM_MODES=5`/`ALTO_MASK_LENGTH=64` only,
  sufficient for every smoke workload and the general-kernel variant's
  order-3/4 FROSTT tensors this size class covers.
- `kernel_id` is hardcoded to Level-1 (`1`); AUTO(`10`)/Level-3(`3`) are not
  wired (would need `create_da_mem_dev`/`zero_partials_Async`/
  `partial_matrix_reduction` plumbing in `wrapper.cu`) — see adapter.py's
  docstring.
- The out-of-memory streaming path (`stream_data=true`) is not wrapped —
  see "In-memory path only" above.
- No timing sweep (login-node rule 5) — only functional/gate checks.

## Verdict

`blco-mttkrp-lvl1: BUILT+GATED`. Tensor/mode/rank gated: all 3 smoke
tensors (order 3-4) x every mode x R in {16,32,64} (21/21 combinations
directly verified, plus the required `--smoke` runner invocation on both
`mttkrp-general-kernel-fp64` and `mttkrp-general-e2e-preproc`, 3/3 each) —
`max_scaled_err` uniformly at fp64 machine-epsilon precision
(`2.3e-16`-`4.6e-16`), tol `1e-6`. In-memory (non-streaming) path wrapped,
NOT BLCO's out-of-memory streaming contribution — disclosed above. Two real
artifact bugs found and fixed: (1) `gen_alto()`/`destroy_alto()` in
`include/alto.hpp` free uninitialized pointers on every GPU-MTTKRP call
(reliable `free(): invalid pointer` crash embedded in a larger process;
1-line `source/` memset fix, documented, minimal); (2) every MTTKRP kernel
variant `atomicAdd`s into an output buffer BLCO's own code never zeroes
(silent NaN/wrong-value corruption for any output row not touched by a
prior write in the SAME call, affecting the artifact's own CLI usage too,
not just this wrapper; fixed via one `cudaMemset` in `wrapper.cu`, no
`source/` change needed for this one).

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, on an
  A100-SXM4-40GB physical card, driver 595.71.05), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 12.4.0 (conda-forge
  `kb-gcc12`, via `KB_GXX12` -- same host-compiler-version-mismatch reason
  as recorded above, just a different g++ 12.x build), torch 2.8.0+cu128,
  Python 3.12.14, no cmake needed (nvcc direct compile). `build.sh` needed
  zero edits: its `NVCC="${NVCC:-...}"`/`HOST_COMPILER="${KB_GXX12:-...}"`
  defaults already read the exported knobs first (rule 8), and
  `bench/artifacts/toolchain.sh` exports `NVCC` before `build.sh`'s own
  default line ever evaluates, so the Perlmutter HPC-SDK path in the script
  is never reached here. `-gencode=arch=compute_80,code=sm_80` unchanged
  (matches this GPU).
- Build: OK (`libblco_wrapper.so` links clean, `blco_run` symbol present,
  no missing shared libs). Build-system changes: none.
- Gate: `mttkrp-general-kernel-fp64`: PASS 3/3 (smoke-3d-small err
  2.69e-16, smoke-3d-tiny err 2.54e-16, smoke-4d-small err 3.69e-16, all
  <= tol 1e-6). `mttkrp-general-e2e-preproc`: PASS 3/3, same error
  magnitudes (2.69e-16 / 2.54e-16 / 3.69e-16). Both runs `conforming:
  False` (reduced protocol warmup=5/reps=20, GPU clocks not locked,
  shared-device variance flag) -- not a timing result, per protocol.
- Deviation from the recorded ruling: none. Error magnitudes match the
  Perlmutter run's fp64-machine-epsilon range (recorded 3.09e-16/3.56e-16/
  3.69e-16) to within the expected run-to-run noise at this precision.
- Verdict here: BUILT+GATED -- equals the recorded ruling.
