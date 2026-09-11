# rode (RoDe) — spmm

**Status: BUILT+GATED**

- Paper: "RoDe: A Row Decomposition-based Approach for Sparse Matrix-Matrix
  Multiplication on GPUs", PPoPP'24. `PAPER_KEY = conf/ppopp/PangFQZL24`
  (matches `benchspecs/spmm/spec.yaml`'s own `evidence` entry for this key).
- Artifact: https://github.com/CRAFT-THU/RoDe
- Commit cloned: `f446e29802015df3d2fd9cc904f4cbe0cce97f3a` (2023-11-27),
  `git clone --depth 1`. `source/` is byte-identical to upstream (no patch).
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`), host compiler
  `/opt/cray/pe/gcc-native/14/bin/g++` (GCC 14.3.0), `-gencode
  arch=compute_80,code=sm_80`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.

## Selection rationale

Core general-GPU-SpMM baseline under the revised kernel-centrality rule
(2026-09-05): RoDe's row-decomposition CSR SpMM is literally
`spmm-gpu-kernel-f32`'s claimed common denominator ("The common denominator
most GPU SpMM papers (RoDe, GE-SpMM, SMaT, DTC-SpMM's inner kernel) actually
report" — spec's own `claim` field), fp32, general (non-block-structured)
SuiteSparse input — the spec's *default*-calibrated regime, not a recency or
tiebreak pick. RoDe is also the baseline every later paper in this line
(DTC-SpMM, FlashSparse, and the sibling agent's Voltrix/SSpMM/MP-SpMM) cites
and compares against.

## What the artifact actually is

`source/RoDe_SpMM/RoDeSpmm.cu` + `RoDeSpmm.h`: two host launcher functions,
`RoDeSpmm_n32`/`RoDeSpmm_n128` (dense-width-specialized; RoDe ships exactly
these two instantiations, no general-N template), each launching two CUDA
kernels — `RoDeComputeKernel1` (main "segments": each row's nonzeros split
into <=512-element contiguous chunks, processed by a persistent CTA — the
paper's actual contribution, "row decomposition" for load balance regardless
of per-row nnz skew) and `RoDeComputeKernel2` (residue: the tail nonzeros of
each row that don't fill a full `KBLOCK=32` group). Both take **raw
pointers/ints only** — confirmed by `RoDeSpmm.cu`'s own `#include` list
(`basic_utils.h`, `cuda_runtime.h`, `common_utils.h` — **not**
`matrix_utils.h`) — i.e. the numeric kernel has zero dependency on RoDe's
`SPC::SparseMatrix`/`CudaSparseMatrix` class hierarchy.

## What was wrapped (rule 1)

`RoDeSpmm_n32`/`RoDeSpmm_n128`, compiled and linked completely unmodified
into `librode_wrapper.so` together with this directory's `wrapper.cu`
(ctypes glue). Wrapped at the same level as `inferfast`/`insum`: one call =
one SpMM kernel invocation (both of RoDe's own internal Kernel1+Kernel2
launches, exactly as its own eval driver issues them).

## Preprocessing / dependency-chain finding (rules 2/3, disclosed, not hidden)

RoDe's row-decomposition preprocessing lives in a class method,
`SPC::SparseMatrix::RowDivide2Segment` (`source/utils/matrix_utils.cu:
989-1041`). That class pulls in **abseil-cpp + Google Glog + gflags**
(`source/cmake/Dependencies.cmake`: `find_package(Glog REQUIRED)`,
`add_subdirectory(third_party/abseil-cpp)`, itself an uninitialized git
submodule) purely for host-side logging macros (`CHECK_LE`/`CHECK_GE`) and a
random-fill constructor path this adapter never needs (real matrix values
are supplied, not RoDe's own `absl::Uniform` random fill). Building three
extra dependency chains (abseil, Glog, gflags — none available as NERSC
modules; confirmed via `module spider glog|gflags|abseil`, all "unable to
find") solely to reach one pure-array host loop that the numeric kernel
itself never calls would be a large, load-bearing-for-nothing addition to
this integration.

**What was done instead**: `wrapper.cu::row_divide_to_segment` is a
byte-for-byte port of `RowDivide2Segment`'s algorithm (identical loop
structure, same `SegmentLength=512, vectorLen=4, KBLOCK=32` — RoDe's own
`eval/eval_spmm_f32_n128.cu`'s exact call:
`sm1.RowDivide2Segment(SEG_LENGTH,4,32)` with `#define SEG_LENGTH 512`),
operating directly on the CSR `row_offsets` array — no GPU, no randomness, no
logging. This reproduces the artifact's own preprocessing **algorithm**
exactly (verified byte-for-byte against `matrix_utils.cu` while writing it);
only the surrounding class scaffolding (built to serve RoDe's own `.mtx`
CLI driver, irrelevant to the numeric kernel) is not linked. The actual SpMM
computation always calls RoDe's real, compiled `RoDeSpmm_n32`/`RoDeSpmm_n128`
— nothing about the kernel itself is reimplemented. This is timed as
preprocessing in `adapter.py::prepare()`, per the harness contract.

Separately, a genuine build-system fix was needed: `source/utils/
common_utils.h` uses `uint32_t` without including `<cstdint>`; this
nvcc/libstdc++ pairing no longer pulls it in transitively. Fixed via
`nvcc -include cstdint` (a compiler flag, ARTIFACT_GUIDE.md rule 3 —
no source file touched).

## Implementation finding: zero-block kernel launch (documented, not a gate workaround)

Confirmed by direct ctypes testing (`/tmp/.../rode_debug.py`, not checked
in): for matrices where every row has fewer nonzeros than `KBLOCK=32` (this
track's low-degree GNN-style graphs), `RowDivide2Segment` produces
`n_segs==0` — every row's nonzeros land entirely in the residue set. RoDe's
own `RoDeSpmmKernel` (`RoDeSpmm.cu:492-506`, unmodified) unconditionally
launches **both** Kernel1 (grid = `(0, *, *)` in this case) and Kernel2
(a valid, non-empty launch). Kernel1's zero-block launch deterministically
raises `cudaErrorInvalidConfiguration` (9) on this CUDA/driver pair at the
launch call itself; Kernel2 still executes correctly regardless (verified:
`max abs err ~ 2e-6` against an fp64 CPU reference, both with and without
this condition — see the two standalone debug runs below). This is RoDe's
own kernel launcher's behavior on an input shape its own eval driver (always
real, higher-degree SuiteSparse `.mtx` files) never exercises; the numeric
result is unaffected. `wrapper.cu::rode_run` treats this ONE specific,
structurally-explained error code (`cudaErrorInvalidConfiguration` **and**
`n_segs==0`) as benign rather than surfacing a false kernel failure; any
other error code, or this same code when `n_segs != 0`, is still propagated
and would fail the gate (rule 4 — nothing about the correctness gate itself
was loosened).

```
# all-residue case (nnz_per_row=24 < KBLOCK=32, matches smoke-uniform/-banded/-powerlaw)
n_segs=0 n_segs_residue=4000 ; run err (before fix): 9 ; max abs err: 1.69e-06
# mixed case (nnz_per_row=40 > KBLOCK=32)
n_segs=2000 n_segs_residue=2000 ; run err: 0 ; max abs err (N=32): 2.18e-06 ; (N=128): 2.10e-06
```

## adapter.py

- `KERNEL = "spmm"`, `IMPL_NAME = "rode-spmm"`, `PRECISIONS = ["fp32"]` —
  RoDe's double-precision overloads exist in source but are never
  instantiated by its own eval driver; not wired here.
- `prepare()`: CSR arrays cast to the artifact's native int32/float32 layout,
  `rode_prepare()` builds RoDe's row-decomposition metadata on the GPU (timed
  as preprocessing). `N` outside `{32, 128}` raises `NotImplementedError`
  naming the constraint (rule 8) — RoDe ships no other dense-width
  instantiation, so `spmm-gpu-kernel-f32`'s `N=256,512` are UNSUPPORTED for
  this impl (confirmed in the gate run below).
- `B`: `numpy.random.default_rng` matching `cpu_ref.reference_spmm`'s
  `_dense_operand` exactly (see `insum/adapter.py`'s docstring for why this
  matters).
- `run()`: zeros `C` first (RoDe's kernels `atomicAdd` into `C` for
  segment-split rows and the residue tail — same convention as
  insum/inferfast), then ONE `rode_run()` call on the default CUDA stream
  (stream 0) for both of RoDe's internal launches. Simplification: RoDe's own
  eval driver issues Kernel1/Kernel2 on two separate real streams for
  intra-call overlap; this adapter uses stream 0 for both instead, because
  torch's `CudaEventTimer` records start/stop on torch's *current* (default)
  stream and `torch.cuda.Stream()` objects are created CUDA-non-blocking —
  issuing RoDe's raw-pointer kernel launches on such streams would not be
  guaranteed ordered against those timing events (a correctness/timing
  hazard). Both kernels still run exactly as compiled, just serialized on
  one stream rather than overlapped across two.
- `to_host()`: row-major `(M, N)` fp64 cast — confirmed via
  `output_offset = m_idx * n + n_idx` in `RoDeSpmm.cu`, i.e. standard
  row-major, same convention as `gpu_cuda.py`'s built-in impls (no transpose
  needed, unlike InferFast's column-major output).
- `timer()` reuses `kernelbench.impls.gpu_cuda.CudaEventTimer`.

## Gate verification (login node, functional check + one real matrix)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-gpu-kernel-f32 --impl rode-spmm --smoke --warmup 1 --reps 3
```

Result: **6/6 valid, 6 unsupported** (N=256/512 correctly rejected with
`NotImplementedError`). `max_scaled_err` 2.0e-07 – 2.6e-07, all `<= tol
1e-4` — comfortably fp32-clean (RoDe's kernel is plain fp32 FMA, no
Tensor-Core mixed-precision accumulation). Reduced-protocol numbers only
(warmup=1, reps=3, shared login-node GPU; `conforming: False`), not a timing
result per ARTIFACT_GUIDE.md rule 5.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge,
  `$KB_CXX`), torch 2.8.0+cu128, Python 3.12.14; `-gencode
  arch=compute_80,code=sm_80` (unchanged from build.sh).
- Build: OK. Build-system changes: none this pass (the LD_PRELOAD/ldd-nm
  pipefail-safety fixes for this build.sh were already committed in
  `aa30f86`; carried unchanged, re-ran idempotently).
- Gate: `spmm-gpu-kernel-f32 --smoke`: 6/6 valid, 6 unsupported (N=256/512),
  `max_scaled_err` 1.98e-07–2.56e-07 <= tol 1e-4, PASS. `spmm-gpu-kernel-f32
  --matrices cant --dims 32,128`: 2/2 valid, err 2.99e-07/3.11e-07, PASS.
  `spmm-binary-adjacency-kernel --smoke`: 6/6 valid, err 1.88e-07–2.74e-07,
  PASS. `spmm-binary-adjacency-kernel --matrices cora --dims 32,128`:
  **FAILED TO RUN** — `ModuleNotFoundError: No module named
  'torch_geometric'` (`kernelbench/domains/sparse.py`'s Planetoid loader;
  this machine's `kb-env` has no `torch_geometric` installed — a
  harness/environment dependency gap, not an artifact or kernel failure; not
  fixed here since it is a shared-harness dependency, out of scope for a
  per-artifact build-system fix per this task's directory scope).
- Deviation from the recorded ruling: none for the kernel itself — every
  number that ran matches the Perlmutter record (identical `cant` errors,
  2.99e-07/3.11e-07, bit-for-bit; smoke ranges consistent). The only gap is
  that the `cora` real-matrix check (binary-adjacency competitor variant)
  could not run on zaratan because this environment's Python lacks
  `torch_geometric`, whereas the Perlmutter env used for the original record
  had it — an environment gap unrelated to RoDe's kernel.
- Verdict here: BUILT+GATED — same as the recorded ruling.

Also verified on one real SuiteSparse matrix (cached, per the task's matrix
set):

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-gpu-kernel-f32 --impl rode-spmm --matrices cant \
    --dims 32,128 --warmup 1 --reps 3
```

Result: **2/2 valid**. `cant` (62451x62451, nnz=4,007,383):
dim=32: `0.122 ms, 2102.5 GFLOP/s, err=2.99e-07`;
dim=128: `0.377 ms, 2722.9 GFLOP/s, err=3.11e-07`. Again reduced-protocol,
non-conforming numbers only — no timing sweep was run.

## Not done

- No sweep across the full 24-matrix `recommended_subset` or the full
  warmup=10/reps=100 protocol — out of scope per the task's login-node
  budget (build + gate check only).
- `spmm-gpu-e2e-preproc-f32` (amortized k=100 protocol) and
  `spmm-cpu-kernel-f32`/`spmm-gpu-quantized-int` variants not exercised —
  RoDe is GPU-fp32-only, so only `spmm-gpu-kernel-f32` applies; the e2e
  variant would reuse the same `prepare()` unchanged but was not separately
  gated here.
- `RoDe_SDDMM` (the same repo also ships an SDDMM kernel) is out of scope
  for this spmm-track task.

## 2026-09-06: also gated under spmm-binary-adjacency-kernel (competitor)

Run as a weighted-kernel competitor for the new pattern-only-kernel variant
(`spec.yaml`'s `spmm-binary-adjacency-kernel`, see `../README.md`'s
"Pattern-only SpMM variant"): `--variant spmm-binary-adjacency-kernel --impl
rode-spmm --precision fp32 --smoke --dims 32,128 --warmup 1 --reps 3` ->
**6/6 valid**, `max_scaled_err` 1.9e-07 – 2.7e-07; `--matrices cora --dims
32,128` -> **2/2 valid**, `max_scaled_err` 1.4e-07 – 1.5e-07. RoDe reads
values it does not need on this variant (no special-casing in the adapter),
and is simply fp32-fast regardless — a fair, expected pass. Reduced
-protocol, non-conforming numbers only (login-node GPU).
