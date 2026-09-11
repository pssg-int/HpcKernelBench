# ocean (Ocean-SpGEMM) — spgemm

**Status: BUILT+GATED**

- Paper: "Ocean: Fast Estimation-Based Sparse General Matrix-Matrix
  Multiplication on GPU", ICS'26. `PAPER_KEY = conf/ics/LiG26` (matched by
  `artifact_url` in `../../output/included.json`).
- Artifact: https://github.com/CornellHPC/Ocean-SpGEMM
- Commit cloned: `cb093963f6e45d9deabf7848bc05a538af796315` (2026, `git clone
  --depth 50`). `git -C source status --short` / `git -C source diff
  --stat` are both EMPTY — zero lines of `source/` were touched.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`), host compiler
  `g++-12` (SUSE 12.3.0) — same fix already used by `spmm/inferfast` and
  `spmm/rassm` in this repo (nvcc 12.9 + the system default g++-14 fails
  `<bits/alloc_traits.h>` with `__has_construct is undefined`, a
  host-compiler/libstdc++ mismatch unrelated to Ocean). `-gencode
  arch=compute_80,code=sm_80` (A100). No cuBLAS/cuSPARSE dependency (Ocean
  is a from-scratch CUDA implementation using only `cub`).

## What the artifact actually is

`somespgemm::SpGEMM::run(A, B)` (`source/kernels/SpGEMM.cuh`) is Ocean's own
library entry point — the exact class `source/src/main.cu`'s CLI
constructs and calls. `wrapper.cu` (this directory, NOT under `source/`)
calls it directly (ARTIFACT_GUIDE.md rule 1), together with
`somespgemm::CSR`/`cuCSR`/`convert()` (`source/include/CSR.h`,
`source/src/CSR.cpp`) for host↔device format conversion. `wrapper.cu`
touches **zero** lines of `source/` — it only adds `extern "C"` entry points
around already-defined, unmodified classes/functions, compiled together
with `source/src/CSR.cpp` and `source/src/Utils.cpp` (unmodified) into
`libocean_wrapper.so`; `source/src/main.cu` (the CLI) is never compiled
into this library.

## adapter.py

- `KERNEL = "spgemm"`, `IMPL_NAME = "ocean-spgemm"`, `PRECISIONS = ["fp64"]`
  (Ocean's `data_t` is `double` throughout, no other value type).
- `prepare()`: two independent device `cuCSR` operands are built from our
  workload's CSR via `ocean_build_csr()` (host `CSR::alloc()` fill + Ocean's
  own `convert(cuCSR&, const CSR&)` H2D copy) — this **is** the artifact's
  format conversion, timed as preprocessing per ARTIFACT_GUIDE.md rule 2.
  `C = A @ A` per `benchspecs/spgemm/spec.yaml`'s `operation` field; two
  separate device copies are built (rather than aliasing one pointer as
  both A and B) to avoid relying on any unaudited A!=B assumption inside
  Ocean's ~2500-line kernel implementation.
- `run()`: one `ocean_spgemm_run()` call = one `SpGEMM::run(A,B)` call, with
  `Config{warmup_iters=0, bench_iters=1}` (see "Timing scope" below for why).
  The previous call's result handle is freed at the top of each call (device
  memory would otherwise leak one fresh `cuCSR` per rep across the
  warmup+measured-reps loop — only the first call's result is ever read via
  `to_host()`).
- `to_host()`: D2H copy via `ocean_csr_to_host()`, wrapped into a
  `scipy.sparse.csr_matrix`, then reindexed onto the shared `|A|@|A|`
  structural pattern via `kernelbench.impls.cpu_ref._canonical_pattern` /
  `_reindex_to_pattern` — the same convention `TorchSpGEMM.to_host()`
  (`kernelbench/impls/gpu_cuda.py`) uses, so the gate compares apples to
  apples across every spgemm impl regardless of each one's own internal
  output order.
- `timer()`: `kernelbench.impls.gpu_cuda.CudaEventTimer`, reused unmodified.

## Bug found in Ocean's own code (research finding)

**`cuCSR::row_offsets`'s terminal sentinel (`row_offsets[rows]`, i.e. the
"total nnz" entry every CSR consumer requires — `indptr[-1] == nnz`) is
never written by Ocean's SpGEMM kernel.** Every OTHER entry
`row_offsets[0..rows-1]` is correct (cross-checked against an independent
scipy computation on a hand-built 5x5 test matrix — see below), and the
correct total nnz IS tracked separately in `cuCSR::nnz` (a host-side
`size_t`), so this is not a correctness bug in the SpGEMM computation
itself, only in what the *device row_offsets array* contains at its last
slot — but it is a real, reproducible artifact bug: any consumer that reads
`cuCSR`/its `CSR` D2H conversion as a standard CSR array (as scipy, and
essentially every downstream SpGEMM/SpMV/SpMM library, does) gets a
**malformed CSR structure** (`indptr[-1] == 0` instead of `nnz`), which
promptly corrupts `scipy.sparse`'s C-level sparsetools routines
(`sum_duplicates`/`sort_indices`/`eliminate_zeros`) — observed directly as
a nondeterministic `free(): invalid pointer` / `free(): invalid size`
glibc abort when this adapter's `to_host()` first tried to canonicalize
Ocean's raw output.

Repro / isolation trail (kept minimal, not shipped as a test — this file is
the record):
1. A standalone C++ program and a minimal ctypes script calling
   `ocean_build_csr`/`ocean_spgemm_run`/`ocean_csr_to_host` directly (no
   Python harness, no scipy) ran cleanly with **no crash**, on both a
   hand-built 5x5 matrix and the 4000x4000 smoke matrix, including 26
   repeated calls (matching the harness's warmup=5+reps=20+1-gate-check
   count) and with `torch.cuda.init()` called first — ruling out a
   CUDA-context/allocator conflict with torch, a repeated-call leak, and a
   ctypes marshaling bug.
2. Printing the D2H-copied `row_offsets` array directly showed the last
   entry was always `0` regardless of matrix, while `cuCSR::nnz` (read via
   a separate `ocean_csr_dims()` call) was always correct.
3. Comparing `row_offsets[0..rows-1]` against an independent
   `scipy.sparse` computation of the same tiny matrix showed EXACT
   agreement (`[0, 3, 7, 12, 16]` both ways) except for the missing last
   entry (`19` expected, `0` observed) — confirming the SpGEMM computation
   itself is correct and this is purely a missing-sentinel bug in the
   output array construction.
4. The crash reproduced only once `scipy.sparse.csr_matrix(...)` was built
   from the buggy array and put through `sum_duplicates()`/`sort_indices()`
   (exactly what `cpu_ref._canonical_pattern`/`_reindex_to_pattern` do) —
   confirming the malformed `indptr` array, not Ocean's GPU kernels or this
   project's ctypes plumbing, was the crash's root cause.

**Fix (in `wrapper.cu`, not `source/` — ARTIFACT_GUIDE.md rule 3):**
`ocean_csr_to_host()` now overwrites `h_row_offsets[rows]` with the
authoritative `cuCSR::nnz` value after the D2H copy. One line, documented
inline in `wrapper.cu`. This is a wrapper-boundary fix to output
formatting, not a change to Ocean's SpGEMM algorithm or kernel code.

## Disclosed timing-scope contamination

`somespgemm::Workspace`'s constructor (`source/kernels/SpGEMM.cuh`) creates
20 `cudaStream`s and 200 `cudaEvent`s on **every** call to `SpGEMM::run()` —
this happens before Ocean's own internal `cuTimer` starts, so it is not
counted in Ocean's own reported number, but it **is** included in this
adapter's externally-measured per-call time (the `CudaEventTimer` wraps the
whole `ocean_spgemm_run()` call). Unlike AmgT's `CSR2BSR_GPU`
(idempotent, tag-guarded, cleanly hoistable into `prepare()` — see
`../amgt/STATUS.md`), Ocean's per-call Workspace setup is intrinsic to how
`SpGEMM::run()` is written; hoisting it out would require restructuring
Ocean's own control flow (touching its algorithm code, discouraged by
ARTIFACT_GUIDE.md rule 3), so it is disclosed here instead. This adapter's
measured per-call time is therefore a documented UPPER BOUND on
`spgemm-square-kernel-f64`'s `timing_scope` (symbolic+numeric+
C-allocation only), not an exact match — relevant for any future timed
(non-smoke) run of this adapter, not for the gate check below.

`Config{warmup_iters=0, bench_iters=1}` disables Ocean's OWN internal
warmup+bench averaging loop (its default config does 10+10 internal
iterations inside a SINGLE `run()` call) so that one `ocean_spgemm_run()`
call performs exactly one prologue→analysis→symbolic/estimation→numeric
pass; the harness's own warmup(5)/measured(20) reps (smoke protocol; spec
protocol is warmup=10/reps=20) govern repetition instead, per the spec.

## Gate verification (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel spgemm --variant spgemm-square-kernel-f64 --impl ocean-spgemm --smoke
```

Result: **3/3 runs valid** (smoke-uniform, smoke-banded, smoke-powerlaw;
4000x4000 synthetic matrices). `max_scaled_err` (vs. scipy fp64 reference,
gated against `|A|@|A|`'s structural pattern per DOMAIN_GUIDE.md):
`2.21e-16` / `4.80e-16` / `4.13e-16`, all `<= 1e-6` tolerance — genuinely
independent agreement at essentially fp64 machine-epsilon precision (not
self-certifying: Ocean's own CUDA kernels vs. scipy's CPU computation).
Reduced-protocol numbers only (warmup=5, reps=20, shared login-node GPU),
explicitly marked non-conforming by the runner (`conforming: False`); not a
timing result per ARTIFACT_GUIDE.md rule 5. No timing sweep was run.

## Not done

- Only the fp64 `spgemm-square-kernel-f64` variant was gated. Ocean has no
  tensor-core/mixed-precision mode (fp64 `data_t` only), so
  `spgemm-tensorcore-mixed-precision` does not apply to this adapter.
- No sweep across the spec's `recommended_subset` SuiteSparse matrices —
  out of scope per the task's login-node budget; a single gate check only.
  A real matrix (e.g. `cant`, `pwtk`) would exercise Ocean's HLL-estimation
  path more thoroughly than the synthetic smoke matrices do; left for a
  compute-node timing run.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), gcc/g++ 12.4.0 host compiler, torch 2.8.0+cu128, Python 3.12.14; arch `-arch=sm_80`.
- Build: OK, exit 0. Build-system changes: `build.sh` — the symbol-verification step (`nm -D ... | grep -q`) now captures nm/ldd output to a variable first, so it cannot spuriously fail under login-node RLIMIT_NPROC contention (nm cannot fork / SIGPIPE), same fix as spmm/rode & spmm/sputnik. The `.so` itself built fine on the first attempt; only the verification pipeline was fragile.
- Gate: `spgemm-square-kernel-f64` / `ocean-spgemm` / --smoke: 3/3 runs valid (max err 4.13e-16 <= tol 1e-6).
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
