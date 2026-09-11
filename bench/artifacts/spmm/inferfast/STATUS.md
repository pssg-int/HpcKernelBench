# inferfast (InferFast) — spmm

**Status: BUILT (gate blocked: fp16 kernel vs. an fp32-calibrated tolerance —
expected, not a bug; see "Gate result" below)**

- Paper: "InferFast: Bridging the Gap Between Unstructured LLM Sparsity and
  Practical GPU Throughput", ICS'26. `PAPER_KEY = conf/ics/ShenBSCH26`
  (matched by title in `../../output/included.json`).
- Artifact: https://github.com/MLsys-HPC/InferFast
- Commit cloned: `dc5a7f179009b4ed9a3fd50c9c3e384a19352eea` (2025-12-17),
  `git clone --depth 50`.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`), host compiler
  `g++-12` (SUSE 12.3.0), `-gencode arch=compute_80,code=sm_80`,
  `libcublas`/`libcusparse` 12.9 from
  `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/...`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.

## What the artifact actually is

A Flash-LLM/SpInfer-lineage bitmap-tiled Tensor-Core SpMM kernel for
unstructured-sparse fp16 weights (`source/csrc/SpMM_API.cu` +
`SpMM_Kernel.cuh`): `InferFast_InitSparseMatrixA` compresses a dense fp16
matrix into a per-tile bitmap + compressed-value + tile-offset format on the
host; `InferFast_SpMM_SplitK_API` runs the Tensor-Core kernel against that
format. Both are called unmodified (rule 1: wraps the kernel entry points
directly, not `kernel_benchmark/spmm_test.cu`'s benchmark driver).

## Build (rules 3/6: minimal patches, all recorded; no kernel code touched)

1. **Host compiler**: the system default `g++` is SUSE's `g++-14`, and
   `nvcc 12.9 -ccbin g++-14` fails compiling `<bits/alloc_traits.h>`
   (`identifier "__has_construct" is undefined` — 53 errors, a
   libstdc++-version/nvcc-version mismatch unrelated to InferFast).
   `g++-12` (available at `/usr/bin/g++-12`) compiles clean. Passed via
   `Makefile`'s existing `HOST_COMPILER` variable — no Makefile edits.
2. **Math libs**: `source/build/Makefile`'s default `-lcublas -lcusparse`
   fails to link (`cannot find -lcublas`) because the `nvcc`-toolkit-only
   path under `hpc_sdk/.../cuda/12.9` ships no `libcublas.so`. Located the
   matching libs under the sibling
   `hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib` (found
   empirically via `find -iname libcublas.so*`) and passed via the
   Makefile's existing `LIBRARIES` variable. `source/build/libSpMM_API.so`
   builds clean with these two overrides — zero lines of `source/` changed.
3. **`wrapper.cu`** (new file, `bench/artifacts/spmm/inferfast/wrapper.cu`,
   outside `source/`): the artifact's host functions
   (`InferFast_InitSparseMatrixA`, `reorder_matrices`) and device API
   (`InferFast_SpMM_SplitK_API`) have C++ (not `extern "C"`) linkage and a
   two-phase malloc/realloc-sized output contract that ctypes can't call
   directly. `wrapper.cu` re-declares `reorder_matrices` with an *identical*
   signature (resolves to the already-compiled symbol via the linker — no
   reimplementation) and adds `extern "C"` entry points
   (`inferfast_compress` / `inferfast_compress_sizes` / `inferfast_copy_out`
   / `inferfast_free_compress` / `inferfast_run`) that only marshal buffer
   sizes and pointers; every actual computation still happens inside the
   artifact's own compiled functions. Built into a second small `.so`
   (`libinferfast_wrapper.so`), linked against `source/build/libSpMM_API.so`
   via rpath. Full reasoning is in the file's header comment.

Both `make -C source/build` and the wrapper compile are idempotent
(`build.sh` re-runnable, exits 0).

## adapter.py

- `KERNEL = "spmm"`, `IMPL_NAME = "inferfast-spmm-splitk"`,
  **`PRECISIONS = ["fp16"]`** — the kernel is Tensor-Core `half` throughout;
  there is no fp32 code path to wrap honestly. (The task's generic adapter
  template defaults to fp32; this artifact genuinely doesn't have one — see
  ARTIFACT_GUIDE.md's own instruction to record "what the artifact actually
  supports.")
- `prepare()` (the artifact's own format conversion, timed as
  preprocessing): our CSR workload is densified to fp16 and zero-padded to
  InferFast's fixed tile grid (`M % 128 == 0`, `K % 64 == 0` —
  `InferFast_InitSparseMatrixA` uses integer division
  `num_global_tiles_M = M / tile_M_global` and silently *drops* any
  non-tile-aligned tail rather than erroring, so the adapter pads explicitly
  instead of letting that truncation happen invisibly), then
  `reorder_matrices` + `InferFast_InitSparseMatrixA` (both unmodified)
  compress it, then every compressed buffer is H2D-copied to CUDA tensors.
  `B`'s real `K` rows are drawn with `numpy.random.default_rng` matching
  `cpu_ref.reference_spmm`'s `_dense_operand` exactly (see `insum/STATUS.md`
  for why this matters — `gpu_cuda.py`'s torch-RNG `_dense` helper draws a
  *different* B than the numpy-RNG reference the gate compares against, a
  pre-existing harness/built-in-impl issue independent of this artifact);
  the `K..Kp-1` padding rows are zero (irrelevant either way, since they
  only multiply against `A`'s zero-padded columns).
- `run()`: ONE `InferFast_SpMM_SplitK_API` call, `split_k=1` (no reduction
  pass — `A` itself is a dead parameter in this API, confirmed by tracing
  `SpMM_API.cu`: only `Compressed_A` reaches the kernel, `A` is passed
  through unread — so `nullptr` is passed for it).
- `to_host()`: InferFast's own storage convention writes `C`
  **column-major** `(Mp, N)` (`source/csrc/SpMM_Kernel.cuh`:
  `BlockGlobalPTR[j + i * M_Global]`) — not something this integration
  introduced. Read back as row-major `(N, Mp)`, transposed, then cropped
  from the padded `Mp` back to the true `M`.
- `timer()` reuses `kernelbench.impls.gpu_cuda.CudaEventTimer`.

## Gate result (login node, functional check only)

The spmm spec (`benchspecs/spmm/spec.yaml`) has no fp16 variant — only
`spmm-gpu-kernel-f32` (tol `1e-4`), `-e2e-preproc-f32`, `-cpu-kernel-f32`,
and `-gpu-quantized-int` (structural, no numeric tolerance). Gating this
fp16-only kernel therefore means running `spmm-gpu-kernel-f32`'s protocol
with `--precision fp16` overriding the variant-name-derived default — the
closest available comparison, not a perfect fit (recorded honestly, not
hidden).

`--matrices cant` (the task's suggested example matrix, 62451x62451,
nnz=4,007,383) was **not used for this artifact specifically**: InferFast's
own preprocessing (`InferFast_InitSparseMatrixA`) requires a fully
*densified* host copy of `A`, twice over transiently (once in `adapter.py`'s
own `A_dense` build, once again inside `inferfast_compress`'s internal
`malloc` before `InferFast_InitSparseMatrixA`'s own initial
`M*K*sizeof(half)` allocation before it `realloc`s down) — for `cant` padded
to 62464x62464 that is ~7.8 GB per copy, ~20+ GB transiently, on a *shared*
login node. Substituted the harness's own synthetic smoke matrices
(`--smoke`, 4000x4000, ~32 MB dense) instead — a real, honest functional
check of the same kernel path, just at a size that doesn't risk starving
other users' processes on shared host RAM. This O(M·K) host-memory
preprocessing cost is an actual property of InferFast's tile-bitmap
compression approach (any dense unstructured matrix must be examined
tile-by-tile on the CPU to build the bitmap), not an adapter shortcut.

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
$PY -m kernelbench.runner --kernel spmm --variant spmm-gpu-kernel-f32 \
    --impl inferfast-spmm-splitk --precision fp16 --smoke --warmup 1 --reps 3
```

Result: **0/12 runs INVALID** — `max_scaled_err` ranged `4.39e-04` (dim=32,
uniform) to `7.21e-04` (dim=256, powerlaw), all against the fp32 variant's
`tol=1e-4`. This is the expected magnitude for fp16 (`half`) accumulation:
unit roundoff `2^-11 ≈ 4.9e-4`, and the observed scaled errors are all within
~1.5x of that — i.e. the kernel is numerically well-behaved for fp16, it
simply cannot satisfy a tolerance calibrated for fp32. **Independently
verified this is not an integration bug**: a standalone ctypes smoke test
(256x256, dim=128, outside the harness) gave `max_scaled_err=5.87e-04`,
consistent with the harness numbers, and the raw output values were close to
the fp64 reference elementwise (`max abs err=1.4e-3` against `max |ref|=3.95`
— a ~3.5e-4 relative error on the largest entries, again fp16-typical).

**Per ARTIFACT_GUIDE.md rule 4 ("if it fails the gate, that IS a result —
record it, do not loosen the gate to make it pass"): no tolerance override
was applied.** This is recorded as a structural finding: the spmm spec would
need a dedicated fp16 variant (with an fp16-appropriate tolerance, the way
the existing `-gpu-quantized-int` variant already carves out int4/int8 into
its own structural/no-tolerance bucket) for this class of Tensor-Core
`half` kernel to be gated fairly. Until then, `inferfast-spmm-splitk` is
correctly wired and produces numerically sane fp16 output, but cannot pass
`spmm-gpu-kernel-f32`'s fp32 bound by construction.

## Not done

- No sweep across matrices/dims/Split_K values — out of scope per the
  task's login-node budget.
- `Split_K > 1` (the artifact's multi-pass reduction path,
  `SplitK_Reduction` kernel) is wired in `wrapper.cu`'s signature but never
  exercised (`self.split_k = 1` hardcoded in `adapter.py`) — the
  single-pass path is the simplest genuine kernel call and was sufficient to
  gate.


## Baseline role (2026-09-05 selection-rule revision)

**Competitor, not a SOTA baseline.** Under the revised rule (core kernel papers
evaluated on the track's own input regime first; `kernel-papers/output/
baseline_selection.md`), this artifact would not have been selected:
- kernel centrality rated `component` (the spmm kernel is not this paper's headline, kernel-level contribution).
- evaluated regime does not match this track's inputs (unstructured-sparse pruned LLM weights, ~30-90% density, fp16/bf16 Tensor Cores via CDP-TCBE bitmap decoding).
Rating rationale (`output/kernel_centrality.json`): Headline is an SpMM kernel (CDP-TCBE format) but built and evaluated for LLM inference weight sparsity; spec's general variant is SuiteSparse fp32, only the tensorcore-fp16 variant is in range.
It stays in the registry and runs under the same gate as every other
implementation, but Phase 3 does not treat it as the human-SOTA reference for
`spmm`.

## 2026-09-06: also gated under spmm-binary-adjacency-kernel (competitor)

Run as a weighted-kernel competitor for the new pattern-only-kernel variant
(`spec.yaml`'s `spmm-binary-adjacency-kernel`, see `../README.md`'s
"Pattern-only SpMM variant"): `--variant spmm-binary-adjacency-kernel --impl
inferfast-spmm-splitk --precision fp16 --smoke --dims 32,128 --warmup 1
--reps 3` -> **6/6 valid**, `max_scaled_err` 3.9e-04 – 5.2e-04 (fp16-typical,
comfortably under this variant's 1e-2 bound — unlike the `spmm-gpu-kernel-f32`
gate above, which is fp32-calibrated and blocks this fp16 kernel by
construction); `--matrices cora --dims 128` -> **1/1 valid**,
`max_scaled_err = 3.93e-03`. A fair, expected pass. Reduced-protocol,
non-conforming numbers only (login-node GPU; smoke matrices used throughout,
not `cant`, per the memory-footprint note above).

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 12.4.0 (conda-forge
  `kb-gcc12` env, via `KB_GXX12`), torch 2.8.0+cu128, Python 3.12.14; arch
  `compute_80,code=sm_80` as recorded.
- Build: OK. `source/build/libSpMM_API.so` was already built from an earlier
  pass (`make: Nothing to be done for 'all'`, idempotent); `libinferfast_wrapper.so`
  rebuilt clean. Build-system changes: none (build.sh's existing
  `${NVCC:-...}`/`${MATHLIB:-...}`/`${HOST_COMPILER:-${KB_GXX12:-...}}`
  fallbacks already read this machine's exported `NVCC`/`MATHLIB`/`KB_GXX12`
  from `bench/artifacts/toolchain.sh`, no edits needed).
- Gate: `spmm-gpu-kernel-f32` (fp16, smoke, warmup=1, reps=3) — 0/12 valid,
  all INVALID, `max_scaled_err` 4.387e-04 (dim=32, uniform) – 7.206e-04
  (dim=256, powerlaw) vs `tol 1e-4` — same expected fp16-vs-fp32-tolerance
  mismatch. `spmm-binary-adjacency-kernel` (fp16, smoke, dims 32/128,
  warmup=1, reps=3) — 6/6 valid, `max_scaled_err` 3.88e-04 – 5.24e-04 <= 0.01.
  `spmm-binary-adjacency-kernel` (fp16, `cora`, dim=128) — 1/1 valid,
  `max_scaled_err = 3.93e-03 <= 0.01` (exact match to the recorded value).
- Deviation from the recorded ruling: none — every PASS/FAIL and every error
  magnitude reproduces the ruling above almost exactly (the smoke-set ranges
  differ in the last significant digit or two from GPU/driver noise, same
  conclusion throughout).
- Verdict here: BUILT (gate FAILS on `spmm-gpu-kernel-f32` as recorded, an
  expected fp16-vs-fp32-tolerance mismatch, not a bug; PASSES on
  `spmm-binary-adjacency-kernel` as recorded) — same as the recorded ruling.
