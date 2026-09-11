# Sputnik (spmm) — STATUS

**Status: BUILT+GATED — clean pass on smoke matrices and on both mandated
real matrices (`cant`, `cora`).**

- Paper: "Sparse GPU Kernels for Deep Learning" (Gale, Zaharia, Young,
  Elsen), SC'20. `PAPER_KEY = conf/sc/GaleZYE20`.
- Toolchain: `nvcc` 12.9, host compiler g++-12
  (`/opt/cray/pe/gcc-native/12/bin/g++`, SUSE 12.3.0 — same nvcc-12.9/g++-14
  `<bits/alloc_traits.h>` mismatch documented in `../inferfast`,
  `../mp-spmm`, `../sspmm`, `../nm-spmm` `build.sh`), `-gencode
  arch=compute_80,code=sm_80`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.
- Selection rationale: core baseline under the revised kernel-centrality
  rule — Sputnik's kernels ARE this paper's headline contribution and are
  cited as a baseline by essentially every later paper in this track (RoDe,
  DTC-SpMM, FlashSparse, SSpMM, MP-SpMM all compare against Sputnik or
  cuSPARSE).

## Provenance ruling: listed artifact URL vs. canonical repository

The paper table's listed artifact URL is
`https://github.com/luckylsk34/Sparse-Kernels`; the canonical library is
`https://github.com/google-research/sputnik`. Cloned and inspected both
(`sputnik-check/` scratch clone, since removed after this ruling — the
comparison is preserved here and is reproducible from the two commits
below).

**Finding: the listed repo is a stripped, renamed, single-commit
("added cmake") derivative of `google-research/sputnik`.** Its
`kernels/sparsekernel/{spmm,sddmm}/*.cu.cc` are semantically IDENTICAL to
sputnik's own `sputnik/{spmm,sddmm}/*.cu.cc` — confirmed by `diff` after
normalizing `s/sparsekernel/sputnik/` on the namespace/include paths: the
only remaining differences are (a) pure line-wrapping/clang-format-style
reformatting of multi-line template typedefs and (b) the rename itself; the
class/function bodies and logic are byte-for-byte the same. However:

1. **No license notice preserved.** Sputnik is Apache-2.0-licensed Google
   code; every file in the canonical repo carries a 13-line copyright/
   license header. The listed fork has NO `LICENSE` file and NO per-file
   header on any kept source file — the license notice was stripped when
   the files were copied/renamed.
2. **Incomplete subset, not an active fork.** Only `spmm/` and `sddmm/`
   were kept (dropped: `CudaSpmmBiasRelu`/half-precision overloads,
   `softmax/`, `depthwise/`, `bias_relu/`); single commit, no evidence of
   independent maintenance or verification against a current toolchain.

Per ARTIFACT_GUIDE.md's instruction ("prefer the listed URL if it is a
faithful copy, else google-research/sputnik at a pinned commit, documenting
why"): **this integration clones `google-research/sputnik` instead**,
avoiding redistributing Apache-2.0 code with its license notice stripped,
and gets the complete, canonical, properly-attributed source. The two
kernel files this integration actually wraps are functionally identical
either way, so this ruling changes provenance/attribution, not behavior.

- Commit cloned: `bbf5840ba5efccf01f862855c785f71bcc6ff1f0` (2020-11-04),
  `git clone --depth 1` from `google-research/sputnik`.
- Listed URL checked (not used): `luckylsk34/Sparse-Kernels`, commit
  `79104ac1f866bea52f32e42e23e90d3a063e7483` (fetched 2026-09-06).

## Build: bypassing Sputnik's own CMakeLists.txt

Sputnik's `CMakeLists.txt` calls `find_package(Glog REQUIRED)`
UNCONDITIONALLY (not gated by `BUILD_TEST`/`BUILD_BENCHMARK`), and Glog is
not installed on this machine (`module spider glog` fails; no
`libglog`/`pkg-config` entry). Checked whether the actual kernel code needs
it: grepped `cuda_utils.h`, `common.h`, `spmm/*.h`, `sddmm/*.h`, and both
`cuda_spmm.cu.cc`/`cuda_sddmm.cu.cc` for `glog`/`LOG(`/`CHECK(` — **zero
hits**. Glog is pulled in only by `matrix_utils.cu.cc`'s `SparseMatrix`/
`CudaSparseMatrix` classes (used exclusively by Sputnik's OWN tests/
benchmarks, which this integration does not build). Abseil is likewise only
needed for `BUILD_TEST`/`BUILD_BENCHMARK` (random-number generation in
`SparseMatrix`'s constructor) — not needed here either.

**`build.sh` bypasses CMake entirely** (same approach as `../rode`,
`../sspmm`, `../nm-spmm`'s `build.sh`): a single `nvcc` invocation compiles
`sputnik/spmm/cuda_spmm.cu.cc` + `sputnik/sddmm/cuda_sddmm.cu.cc`
(UNMODIFIED) together with this directory's `wrapper.cu` into
`libsputnik_wrapper.so`. One build-system fix needed: nvcc identifies
source language by file EXTENSION, and Sputnik's own files use `.cu.cc`
(CMake forces `LANGUAGE CUDA` via `set_source_files_properties`, which a
plain nvcc invocation has no equivalent for) — fixed with `-x cu` (same
class of fix as SSpMM's build.sh, there needed for a different reason).

## What was wrapped

- **`sputnik::CudaSpmm`** (`source/sputnik/spmm/cuda_spmm.h`/`.cu.cc`,
  UNMODIFIED) — fp32 CSR x dense, called with `bias=nullptr` (the
  artifact's own `if (bias != nullptr)` guard skips the optional bias/ReLU
  epilogue, giving plain SpMM). Dispatches internally on `n % 4`/`n % 2`
  (`cuda_spmm.cu.cc:415-497`) to select a vectorized or scalar kernel, so
  **every N in the spec's sweep (32, 128, 256, 512) works directly** — no
  `NotImplementedError` needed for the dense width, unlike every other
  hand-tiled artifact in this directory (RoDe: N in {32,128} only; SSpMM: N
  multiple of 64; NM-SpMM: fixed 32-wide tile).
- **`SortedRowSwizzle`** (`source/sputnik/matrix_utils.cu.cc:302-317`) —
  Sputnik's own row-length-descending argsort preprocessing, used to
  balance the persistent-CTA grid's work. REPRODUCED byte-for-byte in
  `wrapper.cu` rather than compiled from `matrix_utils.cu.cc` directly,
  because that file's OTHER functions (`SparseMatrix`) need Glog+Abseil for
  unrelated test-data-generation code neither the kernel nor
  `SortedRowSwizzle` itself ever touches (confirmed:
  `SortedRowSwizzle` is a completely self-contained ~15-line function using
  only `<vector>`/`<algorithm>`/`<numeric>`/`<cstring>`) — same precedent
  as `../rode/wrapper.cu`'s `row_divide_to_segment` port. Timed as
  preprocessing in `prepare()` per ARTIFACT_GUIDE.md rule 2.

Dense operand `B` drawn with `numpy.random.default_rng(seed).uniform(-1,1)`,
matching `cpu_ref.reference_spmm`'s `_dense_operand` exactly.

## Gate result (login node, functional check only)

```
source bench/env.sh
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" -m kernelbench.runner \
    --kernel spmm --variant spmm-gpu-kernel-f32 --impl sputnik-spmm-f32 \
    --smoke --warmup 1 --reps 2
```

**12/12 smoke runs valid** (3 synthetic matrices x N in {32,128,256,512}),
`max_scaled_err` in `2.0e-7..2.7e-7`, three orders of magnitude under the
`1e-4` tolerance — a genuinely correct fp32 kernel.

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" -m kernelbench.runner \
    --kernel spmm --variant spmm-gpu-kernel-f32 --impl sputnik-spmm-f32 \
    --matrices cant --dims 128 --warmup 1 --reps 2
```
`cant` (62451x62451, nnz~4M): **1/1 valid**, `max_scaled_err=3.94e-07`.

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" -m kernelbench.runner \
    --kernel spmm --variant spmm-gpu-kernel-f32 --impl sputnik-spmm-f32 \
    --matrices cora --dims 128 --warmup 1 --reps 2
```
`cora` (Planetoid, nnz=10556): **1/1 valid**, `max_scaled_err=1.52e-07`.

All `gflops`/`ms` figures are login-node/non-conforming (shared GPU,
`warmup=1, reps=2` overrides) and are NOT a performance claim.

## Regime note (per the task brief)

Sputnik's own SC'20 evaluation targets DNN-pruned weights at moderate
sparsity (3,012 matrices from 49 real pruned ResNet-50/Transformer models,
V100, cuSPARSE baseline — see `benchspecs/spmm/spec.yaml`'s own `evidence`
entry for this `PAPER_KEY`). This track's `spmm-gpu-kernel-f32` gates on
general SuiteSparse matrices instead — core/mismatch-leaning relative to
Sputnik's own regime, same situation as most artifacts in this directory.
Gated anyway (ARTIFACT_GUIDE.md rule 4: never skip or loosen the gate for a
regime mismatch) — a pass is a pass, as the numbers above show.

## Not done

- No sweep across the full `recommended_subset` or timing runs (login-node
  budget; gate-only per ARTIFACT_GUIDE.md rule 5).
- `sputnik::CudaSpmmEx`/`CudaSpmmBiasRelu` (bias/ReLU variant),
  half-precision overload, `softmax/`, `depthwise/`, `bias_relu/` modules
  not wrapped (out of scope: this track needs only plain fp32 SpMM/SDDMM).
- Row-length-sorted `row_indices` was verified to be produced (not just
  passed through as an identity array like `../sspmm`'s dead parameter) —
  not independently benchmarked for its load-balancing effect (a
  performance question, out of scope for a login-node functional gate).

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 12.4.0 (conda-forge,
  `$KB_HOST_COMPILER_BIN`/`$KB_GXX12`), torch 2.8.0+cu128, Python 3.12.14;
  `-gencode arch=compute_80,code=sm_80` (unchanged from build.sh).
- Build: OK. Build-system changes: none this pass (the ldd/nm pipefail-safety
  fix and `KB_HOST_COMPILER_BIN` default for this build.sh were already
  committed in `aa30f86`; carried unchanged, re-ran idempotently).
- Gate: `spmm-gpu-kernel-f32 --smoke`: 12/12 valid, `max_scaled_err`
  1.98e-07–2.70e-07 <= tol 1e-4, PASS. `--matrices cant --dims 128`: 1/1
  valid, err=3.94e-07, PASS. `--matrices cora --dims 128`: **FAILED TO RUN**
  — `ModuleNotFoundError: No module named 'torch_geometric'`
  (`kernelbench/domains/sparse.py`'s Planetoid loader; this machine's
  `kb-env` has no `torch_geometric` installed — a harness/environment
  dependency gap, not an artifact or kernel failure; not fixed here since it
  is a shared-harness dependency, out of scope for a per-artifact
  build-system fix per this task's directory scope).
- Deviation from the recorded ruling: none for the kernel itself — the
  `cant` error matches the Perlmutter record exactly (3.94e-07); smoke range
  is consistent. The only gap is that the `cora` real-matrix check could not
  run on zaratan because this environment's Python lacks `torch_geometric`,
  whereas the Perlmutter env used for the original record had it — an
  environment gap unrelated to Sputnik's kernel.
- Verdict here: BUILT+GATED — same as the recorded ruling.
