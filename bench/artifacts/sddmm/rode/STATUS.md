# rode (RoDe) — sddmm

**Status: BUILT+GATED**

- Paper: "RoDe: A Row Decomposition-based Approach for Sparse Matrix-Matrix
  Multiplication on GPUs", PPoPP'24. `PAPER_KEY = conf/ppopp/PangFQZL24`
  (matches `benchspecs/sddmm/spec.yaml`'s own `evidence` entry for this key).
- Artifact: https://github.com/CRAFT-THU/RoDe. `source/` in this directory
  is a SYMLINK to `../../spmm/rode/source` (read-only reuse, not a re-clone,
  per this task's instructions) — same commit as the spmm track's rode
  adapter: `f446e29802015df3d2fd9cc904f4cbe0cce97f3a` (2023-11-27).
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`), host compiler
  `/opt/cray/pe/gcc-native/14/bin/g++` (GCC 14.3.0), `-gencode
  arch=compute_80,code=sm_80`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.

## Selection rationale

Core general-GPU-SDDMM baseline under the revised kernel-centrality rule
(2026-09-05): RoDe's row-decomposition CSR SDDMM is exactly
`sddmm-csr-kernel-f32`'s claimed common denominator ("the common denominator
most GPU papers in the track report" — spec's own `claim` field), fp32,
general (non-block-structured) SuiteSparse input — the spec's default-
calibrated regime, not a recency/tiebreak pick. Same paper/artifact as the
spmm track's rode adapter (which already carries the full selection
narrative); this is its SDDMM sibling.

## What the artifact actually is

`source/RoDe_SDDMM/RoDeSddmm.cu` + `RoDeSddmm.h`: two host launcher
functions, `RoDeSDDMM_n32`/`RoDeSDDMM_n128` (dense-feature-width-specialized;
RoDe ships exactly these two instantiations, no general-K template), each
launching two CUDA kernels — `SDDMMKernel4Block` (main "segments", same
row-decomposition idea as RoDe_SpMM's Kernel1) and `SDDMMKernel4Residue`
(tail nonzeros that don't fill a full `KBLOCK=32` group). Both host
launchers take raw pointers/ints only — confirmed by direct grep: zero
references to `SPC::SparseMatrix`/`CudaSparseMatrix`/`absl::` anywhere in
`RoDeSddmm.cu` (the actual numeric kernels are independent of that class
hierarchy, same situation as RoDe_SpMM).

Unlike Fused3S's SDDMM stage, RoDe's kernels already multiply by the sparse
value `S[i,j]` INSIDE the kernel (`accumulator_fragment[x] *
values_tile[x]`, confirmed by reading `RoDeSddmm.cu`) and `Store()` each
output directly at its own CSR position (`out + row_offset + n_idx` /
`out + row_offset + threadIdx.x`) — no post-hoc "`* S.data`" step or
TC-block layout reindex is needed in `adapter.py`, unlike the fused3s
adapter.

## What was wrapped (rule 1)

`RoDeSDDMM_n32`/`RoDeSDDMM_n128`, compiled and linked completely unmodified
into `librode_sddmm_wrapper.so` together with this directory's `wrapper.cu`
(ctypes glue). Wrapped at the same level as the spmm/rode adapter: one call
= one SDDMM kernel invocation (both of RoDe's own internal Kernel-block +
Kernel-residue launches, exactly as its own eval driver issues them).

## Build-system workaround: unused `#include "matrix_utils.h"` (not a kernel patch)

`RoDeSddmm.cu` (unlike `RoDeSpmm.cu`) has `#include "matrix_utils.h"` at
file scope. The real header (`source/utils/matrix_utils.h`) pulls in
`absl/random/random.h` (abseil-cpp) — an uninitialized git submodule in
this clone and unavailable as a NERSC module (same finding as the sibling
spmm/rode adapter hit for `SPC::SparseMatrix`'s dependency chain; see that
STATUS.md). Direct grep of `RoDeSddmm.cu` confirms the include is DEAD
CODE: zero references anywhere in the file to any symbol
`matrix_utils.h` declares. `shim_include/matrix_utils.h` in this directory
is an empty stub (same include guard, `MATRIX_UTILS_H_`) placed on the
`-I` path BEFORE `source/utils/` in `build.sh`, so the quoted include
resolves there instead — an include-path substitution
(ARTIFACT_GUIDE.md rule 3 explicitly allows "include paths" as a
build-system fix), not a patch to any tracked file inside `source/`. No
`source.patch` needed (`source/` is untouched, byte-identical to the
spmm/rode symlink target).

Separately, the same `-include cstdint` compiler-flag fix as the spmm/rode
build (for `source/utils/common_utils.h`'s unguarded `uint32_t` use) is
reapplied here, since `RoDeSddmm.cu` also transitively includes
`common_utils.h`.

## Preprocessing (rules 2/3, disclosed)

RoDe's own SDDMM eval drivers call the identical
`SparseMatrix::RowDivide2Segment` preprocessing RoDe_SpMM uses
(`source/utils/matrix_utils.cu:989-1041`) — same reason as the spmm/rode
adapter for porting rather than linking the class (abseil/Glog/gflags for
unrelated host-side logging/random-fill). `wrapper.cu::row_divide_to_segment`
is a byte-for-byte copy of the same port already used and verified by
`../../spmm/rode/wrapper.cu` (not shared via a cross-track header, per this
task's "do not touch other tracks" instruction — each artifact directory
is self-contained). This is timed as preprocessing in `adapter.py::prepare()`.

## Real bug found and fixed: per-K SegmentLength mismatch (own wrapper bug, not RoDe's kernel)

**This is the interesting finding of this integration.** RoDe's own eval
drivers use a **different** `SEG_LENGTH` constant for `RowDivide2Segment`
depending on the dense-feature-width variant — confirmed by direct grep of
both drivers:

```
source/eval/eval_sddmm_f32_n32.cu:30:  #define SEG_LENGTH 512
source/eval/eval_sddmm_f32_n128.cu:30: #define SEG_LENGTH 32
```

each feeding `sm1.RowDivide2Segment(SEG_LENGTH,4,32)`, and each exactly
matching that variant's own kernel-internal compile-time constant
(`RoDeSDDMMKernel_n32<float4,4,32,32,8,32,512>` vs.
`RoDeSDDMMKernel_n128<float4,4,32,32,8,128,32>` — the trailing template
argument is `SEG_LENGTH`, which directly sets
`grid_dim1 = ((m1+kBlockItemsY-1)/kBlockItemsY, SEG_LENGTH/kBlockItemsX, 1)`).

An earlier version of `sddmm_prepare()` in this wrapper hardcoded
`SegmentLength=512` for both K=32 and K=128 (copying the spmm/rode
convention, which has only one SEG_LENGTH since RoDeSpmm's two variants
happen to share it). For K=128 this makes `RowDivide2Segment` produce
segment windows (up to 512 wide) that `RoDeSDDMMKernel_n128`'s actual grid
(`SEG_LENGTH/kBlockItemsX = 32/32 = 1` block-column) never covers beyond the
first 32 entries — everything past index 32 within such a segment is
**silently never written** by the kernel (left at whatever `out` was
pre-zeroed to).

**Symptom observed** (`--kernel sddmm --variant sddmm-csr-kernel-f32
--impl rode-sddmm --smoke`, dim=128 only): `smoke-powerlaw` failed with
`max_scaled_err=4.82e-01` (tol `1e-4`); `smoke-uniform`/`smoke-banded`
happened to pass because their near-constant row length (24) never produces
a >32-wide segment at K=128 to begin with. Root-caused via a standalone
ctypes debug script (per-nnz error breakdown by row, not checked in): every
bad entry belonged to a row with `nnz>=32` (segment-path rows), and within
each bad row the FIRST 32 entries were fp32-clean (`~1e-9` scaled err) while
every entry from index 32 onward matched exactly the signature of an
unwritten (still-zero) output compared against a nonzero fp64 reference —
confirmed identical (`grid_dim1.y=1` for K=128) once cross-checked against
`eval_sddmm_f32_n128.cu`'s own `SEG_LENGTH 32`.

**Fix**: `sddmm_prepare()` now takes `K` and selects
`seg_length = (K==32) ? 512 : 32` before calling `row_divide_to_segment` —
matching RoDe's own two eval drivers exactly, per ARTIFACT_GUIDE.md rule 3
(a preprocessing-parameter correction to OUR wrapper code, not a patch to
any file inside `source/`; the compiled kernel itself is untouched
throughout). This is a genuine bug that would have been in THIS wrapper
either way — it is emphatically not a defect being attributed to RoDe's own
kernel, which behaves exactly as its own eval driver instructs it to given
the segment layout it actually receives.

## adapter.py

- `KERNEL = "sddmm"`, `IMPL_NAME = "rode-sddmm"`, `PRECISIONS = ["fp32"]` —
  same rationale as spmm/rode (RoDe's fp64 overloads exist in source but
  are never instantiated by its own eval driver).
- `prepare()`: CSR arrays cast to native int32/float32; `K` outside
  `{32, 128}` raises `NotImplementedError` naming the constraint (rule 8) —
  RoDe ships no other dense-feature-width instantiation, so
  `sddmm-csr-kernel-f32`'s only two swept `K` values are exactly the ones
  supported (no UNSUPPORTED cases for this spec's own dims sweep, unlike
  spmm's N=256/512 gap).
- Dense operands: `A` (row-indexed, `M x K`, seed 42), `B` (col-indexed,
  `N x K`, seed 43) via `numpy.random.default_rng`, matching
  `cpu_ref.reference_sddmm`'s `_dense_operand` exactly.
- `run()`: zeros `out` first (defensive hygiene across repeated timed reps;
  RoDe's SDDMM kernels `Store()` rather than `atomicAdd`, and
  `RowDivide2Segment`'s block/residue partition covers every nonzero
  exactly once, so this isn't strictly required for correctness the way it
  is for RoDe_SpMM's atomic accumulation, but is kept for consistency and
  as a safety net), then ONE `sddmm_run()` call on the default CUDA stream
  for both of RoDe's internal launches (same stream-0 rationale as the
  spmm/rode adapter's `run()` — torch's `CudaEventTimer` records on the
  current/default stream).
- `to_host()`: `out` is already in the caller's original CSR nnz order (see
  "What the artifact actually is" above) — a plain fp64 cast, no reindex.
- `timer()` reuses `kernelbench.impls.gpu_cuda.CudaEventTimer`.

## Gate verification (login node, functional check + one real matrix)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel sddmm \
    --variant sddmm-csr-kernel-f32 --impl rode-sddmm --smoke --warmup 1 --reps 3
```

Result (after the SegmentLength fix above): **6/6 valid**. `max_scaled_err`
6.3e-08 – 1.4e-07 (tol `1e-4`) — clean fp32 FMA noise, no Tensor-Core
rounding. Before the fix: 5/6 valid, `smoke-powerlaw dim=128` failed at
`4.82e-01` (see finding above). Reduced-protocol numbers only (warmup=1,
reps=3, shared login-node GPU; `conforming: False`), not a timing result
per ARTIFACT_GUIDE.md rule 5.

Also verified on one real SuiteSparse matrix (cached):

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel sddmm \
    --variant sddmm-csr-kernel-f32 --impl rode-sddmm --matrices cant \
    --dims 32,128 --warmup 1 --reps 3
```

Result: **2/2 valid**. `cant` (62451x62451, nnz=4,007,383):
K=32: `0.334 ms, 768.5 GFLOP/s, err=1.56e-07`;
K=128: `0.463 ms, 2216.9 GFLOP/s, err=7.92e-08`. Reduced-protocol,
non-conforming numbers only — no timing sweep was run.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge,
  `$KB_CXX`), torch 2.8.0+cu128, Python 3.12.14; `-gencode
  arch=compute_80,code=sm_80` (unchanged from build.sh).
- Build: OK. Build-system changes: none this pass (the ldd/nm pipefail-safety
  fix for this build.sh was already committed in `aa30f86`; carried
  unchanged, re-ran idempotently; `source/` symlink to `../../spmm/rode/
  source` resolved correctly).
- Gate: `sddmm-csr-kernel-f32 --smoke`: 6/6 valid, `max_scaled_err`
  6.29e-08–1.43e-07 <= tol 1e-4, PASS. `sddmm-csr-kernel-f32 --matrices cant
  --dims 32,128`: 2/2 valid, K=32 err=1.56e-07, K=128 err=7.92e-08, PASS.
- Deviation from the recorded ruling: none — the `cant` errors match the
  Perlmutter record exactly (1.56e-07 / 7.92e-08); the per-K SegmentLength
  fix (K=128 → `SEG_LENGTH=32`) is still load-bearing here, confirmed by the
  passing K=128 `smoke-powerlaw` run, the specific case that exercised the
  bug documented above.
- Verdict here: BUILT+GATED — same as the recorded ruling.

## Not done

- No sweep across the full 26-matrix `recommended_subset` or the full
  warmup=5/reps=50 protocol — out of scope per the task's login-node
  budget (build + gate check only).
- `sddmm-tensorcore-blocked-fp16`/`sddmm-quantized-dlmc`/
  `sddmm-attention-e2e` variants not exercised — RoDe's SDDMM is plain
  fp32 CSR, so only `sddmm-csr-kernel-f32` applies.
- `RoDe_SpMM` is the sibling artifact, already integrated in the spmm
  track; not touched here (read-only symlink reuse of its `source/` only).
