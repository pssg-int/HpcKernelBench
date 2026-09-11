# dtcspmm (DTC-SpMM) — spmm

**Status: BUILT+GATED on spmm-binary-adjacency-kernel (its actual regime —
tf32, unweighted 0/1-adjacency SpMM) since 2026-09-06; general weighted
variants (`spmm-tensorcore-fp16`): gate FAILS (values hardcoded to 1.0, see
finding below)**

## 2026-09-06: re-gated under spmm-binary-adjacency-kernel

`benchspecs/spmm/spec.yaml` gained a dedicated `spmm-binary-adjacency-kernel`
variant (see `../README.md`'s "Pattern-only SpMM variant" section) whose
harness-level `kernelbench.domains.sparse.variant_transform` hook forces the
sparse operand's stored values to 1.0 for EVERY implementation, including
the fp64 CSR reference, before either sees it — the exact regime this
adapter's IMPORTANT FINDING below shows the artifact is structurally limited
to. No adapter code change was needed: `dtcspmm-tcf-spmm` already computes
`A_pattern @ B` unconditionally (it never reads `A`'s values at all), so
once the reference is binarized too, the two agree everywhere, not just on
already-binary inputs like `cora`.

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-binary-adjacency-kernel --impl dtcspmm-tcf-spmm \
    --precision tf32 --smoke --warmup 1 --reps 3
```
Result: **12/12 valid** (3 synthetic smoke matrices x N in {32,128,256,512}),
`max_scaled_err` 1.95e-04 – 2.92e-04, all `<= tol 0.01` — the SAME smoke
matrices that were 0/9 INVALID under `spmm-tensorcore-fp16` (err 2.96–5.67)
now pass cleanly, because the reference is binarized alongside the kernel.

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-binary-adjacency-kernel --impl dtcspmm-tcf-spmm \
    --precision tf32 --matrices cora --dims 128 --warmup 1 --reps 3
```
Result: **1/1 valid**, `max_scaled_err = 4.87e-04` (identical to the
`spmm-tensorcore-fp16`/cora result below, since `cora` was already binary --
`cora+bin` binarizes a no-op there). Reduced-protocol, non-conforming
numbers only (warmup=1, reps=3, shared login-node GPU) — no timing sweep.

The `spmm-tensorcore-fp16` gate result documented further below ("Gate verification") is UNCHANGED and
kept as the on-record general-weighted-variant failure this variant exists
to explain, per the user's 2026-09-06 decision (never delete a documented
FAIL, add the correct home alongside it).

- Paper: "DTC-SpMM: Bridging the Gap in Accelerating General Sparse Matrix
  Multiplication with Tensor Cores", ASPLOS'24. `PAPER_KEY =
  conf/asplos/Fan0024` (matches `benchspecs/spmm/spec.yaml`'s own `evidence`
  entry).
- Artifact: https://github.com/HPMLL/DTC-SpMM_ASPLOS24
- Commit cloned: `66eca266bde1f3faf5f8ef7a054db2391b50e27d` (2024-06-19),
  `git clone --depth 1`. One tracked-file patch (see `source.patch`).
- Toolchain: `nvcc` 12.9, host compiler
  `/opt/cray/pe/gcc-native/14/bin/g++` (GCC 14.3.0) — pinned explicitly via
  `CC`/`CXX` (this login node's bare `c++` resolves to `/usr/bin/c++`,
  g++-7, too old for torch's headers: `"You're trying to build PyTorch with
  a too old version of GCC. We need GCC 9 or later."`; same fix as
  `artifacts/gemv/marlin/build.sh`). `TORCH_CUDA_ARCH_LIST=8.0` (A100;
  upstream's own `DTC-SpMM/build.sh` targets `"8.6 8.9"`, RTX 4090/3090).
  Python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch
  `2.8.0+cu128`.

## Selection rationale

Core general-GPU-SpMM baseline under the revised kernel-centrality rule
(2026-09-05): DTC-SpMM's tensor-core ME-TCF kernel is the baseline every
later paper in this line (FlashSparse, Voltrix, SSpMM, MP-SpMM) compares
against, and is named explicitly in `spmm-gpu-kernel-f32`'s own `claim`
field ("DTC-SpMM's inner kernel").

## What was wrapped (rule 1)

`preprocess_gpu` (CSR -> ME-TCF: 16x8-tiled, bitmap-encoded compressed
tensor-core format) + `run_DTCSpMM` (the WMMA SpMM kernel against that
format), both `source/DTC-SpMM/{DTCSpMM.cpp,DTCSpMM_kernel.cu}` symbols,
called directly and unmodified except for the one disabled `#define` below.
`preprocess_gpu`'s conversion is timed as preprocessing in `prepare()`.

## Build (rules 3/6)

Built via this directory's own `dtc_setup.py` (a `torch.utils.cpp_extension.
CUDAExtension`), **not** the artifact's own `source/DTC-SpMM/setup.py`,
which requires prebuilt Sputnik + Glog libraries (`SPUTNIK_PATH`/`GLOG_PATH`
env vars -> `build/{sputnik,glog}` from full CMake builds of each).

**Why avoided**: traced every use of Sputnik/Glog in the two source files —
1. `DTCSpMM_kernel.cu`'s ONLY Sputnik call site is `spmm_forward_sputnik()`
   (a Sputnik-baseline comparison function this repo bundles for its OWN
   paper evaluation, structurally unrelated to DTC-SpMM's own ME-TCF
   kernel), gated behind `#ifdef USE_SPUTNIK` (which the file `#define`s
   unconditionally at the top). **One-line patch**: commented out that
   `#define` (`source.patch`) — the function body becomes a no-op, so
   `libsputnik.so` is never referenced at LINK time. The unconditional
   `#include <sputnik/spmm/cuda_spmm.h>`/`<sputnik/sputnik.h>` at the top of
   the file still needs Sputnik's HEADERS to exist (compile-time only), so
   `build.sh` runs `git submodule update --init --depth 1
   third_party/sputnik` — headers only, no CMake build of Sputnik itself.
2. Glog: `grep -rn "glog\|CHECK_\|LOG(" DTC-SpMM/*.cpp DTC-SpMM/*.cu` found
   zero references to any Glog macro/header in either file (`CHECK_INPUT` is
   a local `TORCH_CHECK` macro, unrelated to Glog) — the upstream
   `setup.py`'s Glog link appears to be vestigial. Confirmed by grepping
   Sputnik's own header chain (`sputnik.h` -> `spmm/cuda_spmm.h` ->
   `cuda_utils.h`) for Glog too: none (only an unrelated `test_utils.h`, not
   on our include path, uses `glog/logging.h`). Not fetched, not linked.

No source code touched beyond the one `#define` line; the ME-TCF
preprocessing and WMMA kernel are 100% upstream.

## IMPORTANT FINDING: the artifact's exposed Python API is structurally unweighted (binary-pattern) SpMM

Traced every SpMM binding in `DTCSpMM.cpp`'s `PYBIND11_MODULE` (`run_DTCSpMM`,
`run_DTCSpMM_balance`, `DTCSpMM_gcn`, and their `_clock`-instrumented
siblings) back to their C++ implementations in `DTCSpMM_kernel.cu`: **every
one** builds its Tensor-Core value operand as

```cpp
auto val = torch::ones({TCblocktile_id.size(0)}, options);
float *valuesA = val.data<float>();
```

**inside** the wrapper function — there is no parameter anywhere in the
exposed API for a caller to supply real nonzero values
(`grep -rn "torch::Tensor value" DTC-SpMM/` — zero hits in either file). The
underlying `__global__` kernels genuinely accept an arbitrary `valuesA`
device pointer (their `_with_value_` naming is literal, and the
multiply-accumulate math does use whatever is in that buffer) — this is
**not** a hardcoded-in-silicon binary kernel — but `preprocess_gpu` /
`seg_sort_dequ` (its internal CUDA sort/dedup pass building the compressed
tile structure) tracks only STRUCTURE, never a value permutation, so nothing
in the released code can hand a real value through to that pointer. This is
consistent with the paper's own evaluation suite (this spec's own `evidence`
line for this key: "8-graph GNN suite (origin+TCA-reordered)") being
entirely unweighted GNN-adjacency graphs. Reverse-engineering
`seg_sort_dequ`'s sort order to recover a value permutation ourselves was
judged out of this integration's budget, and would stop being "the kernel
the artifact ships" (rule 1) — it would be a new capability neither the
paper nor the released code demonstrates.

**Consequence (rule 4 — a genuine failure IS a result, not something to
paper over)**: this adapter computes `A_pattern @ B` (every stored nonzero
counted as 1.0). This equals the spec's `A @ B` (real values) exactly when
A's stored values are already all 1 — true for cora/citeseer/pubmed via
this codebase's own Planetoid loader (`sparse.py::_load_planetoid`
explicitly sets `A.data[:] = 1.0`) — and false for general numerically
-valued matrices (cant, consph, pdb1HYS, ...) and for this track's synthetic
smoke matrices (`kernelbench.matrices.synthetic` draws `U(-1,1)`, never 1).

## Precision (confirmed by reading the kernel, not assumed)

`DTCSpMM_kernel.cu` uses `wmma::fragment<..., wmma::precision::tf32, ...>`
with `wmma::fragment<wmma::accumulator, ..., float>` throughout every
`spmm_forward_cuda_kernel_improved_ptx_1684_*` variant — TF32 Tensor-Core
compute, fp32 accumulate. This is `spmm-tensorcore-fp16`'s documented
"tf32 secondary" row (the variant explicitly carries both "fp16 primary /
tf32 secondary ... reported as separate rows, never averaged"), not the
fp16 row. `PRECISIONS = ["tf32"]`; gated with `--precision tf32`.

## adapter.py

- `KERNEL = "spmm"`, `IMPL_NAME = "dtcspmm-tcf-spmm"`, `PRECISIONS =
  ["tf32"]`.
- `prepare()`: CSR -> ME-TCF via `preprocess_gpu(col_idx, row_ptr, M,
  BLK_H=16, BLK_W=8, ...)`, timed as preprocessing. Square-matrix
  constraint (`M == K`, since `preprocess_gpu` takes one `num_nodes` for
  both dimensions) and `N % BLK_H == 0` are enforced with
  `NotImplementedError` naming the constraint (rule 8) — neither triggers
  for this track's `recommended_subset`/dims sweep in practice.
  `B`: numpy `default_rng` matching `cpu_ref.reference_spmm`'s
  `_dense_operand` exactly.
- `run()`: ONE `run_DTCSpMM(...)` call with `exeplan="float_nonsplit"` —
  the simplest of DTC-SpMM's 5 autotuned code paths (float/float2/float4 x
  split/nonsplit) and the only one valid for every N in this spec's
  dense-dims sweep without an extra `N%32==0` constraint; the paper's own
  eval autotunes `exeplan` per `(dataset, N)` via a hardcoded lookup table
  (`scripts/DTCSpMM/run_DTC_SpMM.py`'s `ExecutionPlan`) — not reproduced
  here (a disclosed simplification, not a correctness issue: all 5 paths
  compute the same operation at different vectorization widths). No manual
  output-zeroing needed (`run_DTCSpMM` allocates a fresh `zeros_like(input)`
  internally, no atomic accumulation across calls).
- `timer()` reuses `kernelbench.impls.gpu_cuda.CudaEventTimer`.

## Gate verification (login node, functional check only)

**Smoke (expected FAIL — evidences the IMPORTANT FINDING above, not a bug):**

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-tensorcore-fp16 --impl dtcspmm-tcf-spmm --precision tf32 \
    --smoke --warmup 1 --reps 3
```

Result: **0/9 INVALID**. `max_scaled_err` ranged 2.96–5.67 against tol
`0.01` — smoke matrices carry real `U(-1,1)` values, and the artifact
computes `A_pattern @ B` regardless, so the mismatch is exactly the missing
value-weighting, not numerical noise (errors of O(1), not O(1e-3)).

**Real binary graph (cora — this track's own recommended_subset entry,
already known-unweighted via this codebase's Planetoid loader):**

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-tensorcore-fp16 --impl dtcspmm-tcf-spmm --precision tf32 \
    --matrices cora --warmup 1 --reps 3
```

Result: **3/3 valid**. `max_scaled_err = 4.87e-04 – 4.88e-04`, comfortably
`<= tol 0.01` — and strikingly close to TF32's unit roundoff (`2^-11 ≈
4.9e-4`), confirming the ME-TCF tiling + WMMA kernel mechanism is correct
for the class of input the artifact's released API actually supports.
Reduced-protocol, non-conforming numbers only (warmup=1, reps=3, shared
login-node GPU; `0.09–0.19 GFLOP/s` at this tiny graph's scale is not a
meaningful throughput figure) — no timing sweep was run.

## Not done

- No value-permutation recovery for `seg_sort_dequ` (see IMPORTANT
  FINDING) — out of budget; would need independent verification against the
  kernel's actual tile-fill order to avoid silently producing wrong results
  through a NEW code path this integration would own, which is a bigger
  claim than "wraps the artifact's kernel."
- No sweep across the rest of the 8-graph GNN suite / full protocol — one
  representative binary graph (cora) was sufficient to evidence the finding
  within the login-node budget.
- `run_DTCSpMM_balance` (dynamic load-balancing variant) and the TCA
  block-reordering preprocessing (`TCA-reordering/`, needs `minhashcuda` +
  `cugraph`) were read but not wired — `run_DTCSpMM` on `preprocess_gpu`'s
  direct (non-reordered) output was the simplest genuine kernel path
  sufficient to gate.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 (full SXM4 40GB, `gpu-b11-6`, sm_80) via
  `bench/gpu_run.sh`, login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14. `dtc_setup.py` build succeeded cleanly
  (Sputnik headers already present from an earlier `git submodule update`
  on this checkout), no build-system changes.
- Build: OK. Build-system changes: none.
- Gate: `spmm-binary-adjacency-kernel` smoke tf32: 12/12 valid, err
  1.95e-04–2.92e-04 (all `<= 0.01`). `spmm-binary-adjacency-kernel` cora
  dim=128: 1/1 valid, err 4.87e-04. `spmm-tensorcore-fp16` smoke tf32: 0/9
  INVALID, err 2.96–5.67 (matches the recorded finding exactly). `spmm-
  tensorcore-fp16` cora dim=128/256/512: 3/3 valid, err 4.87e-04/4.87e-04/
  4.88e-04 — every number matches the STATUS.md values above bit-for-bit.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED on spmm-binary-adjacency-kernel (spmm-
  tensorcore-fp16 gate fails, as recorded) — equals the recorded ruling.
