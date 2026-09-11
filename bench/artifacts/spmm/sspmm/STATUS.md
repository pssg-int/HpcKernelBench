# SSpMM — STATUS

**Status: BUILT+GATED — passes at N=128/256/512 on synthetic smoke matrices
and on a real SuiteSparse/GNN graph (cora); a real, large padding-overhead
finding on irregular graphs is documented below.**

- Paper: "SSpMM: A Scalable and Efficient Tensor-Core-Based SpMM Library
  Across GPU Generations", TPDS'25.
  `PAPER_KEY = journals/tpds/XueWYTLFSCSL25`.
- Artifact: https://github.com/xuezy-mmi/SSpMM
- Commit cloned: `6b7fec5a369f6e470ae667a385e7dcc5f1b6c55e` (2025-06-24),
  `git clone --depth 1`.
- Toolchain: `nvcc` 12.9, host compiler `g++-12` (SUSE 12.3.0, same
  nvcc/g++-14 `<bits/alloc_traits.h>` mismatch documented in
  `../inferfast/build.sh` and `../mp-spmm/build.sh`) —
  `-gencode arch=compute_80,code=sm_80`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.
- Selection rationale: core general-GPU-SpMM baseline under the revised
  kernel-centrality rule — a Tensor-Core SpMM library explicitly designed to
  be portable/scalable across GPU generations (Transpose Mapping), evaluated
  on the spec's own SuiteSparse/GNN input regime.

## What the artifact actually is / what was wrapped

`source/SSpMM/src/sspmm.cu`'s `spmm::SSpMM(...)` (the artifact's own
contribution — its test driver names it "Ro-SpMM", distinct from the
vendored vectorSparse baseline `spmm::SpMM`/"mma884" in the same file, which
this adapter does NOT wrap) is a `mma.sync`-based Tensor-Core SpMM kernel
over a "vector-sparse" CSR variant (rows grouped into fixed-size blocks that
share one `column_indices` array per group — see `adapter.py`'s module
docstring for the full format and why this adapter had to implement the
general-CSR-to-vector-sparse conversion itself: the artifact only ships
already-vectorized DLMC benchmark files, no general converter).

- `libsspmm_wrapper.so` (`wrapper.cu`, this directory, NOT under `source/`)
  links unmodified against `sspmm.o` (compiled unmodified from
  `source/SSpMM/src/sspmm.cu` by `build.sh`, using the artifact's own
  `Makefile_ampere` compile rule/flags) and only adds `extern "C"` linkage
  around `spmm::SSpMM`.
- `adapter.py`'s `prepare()` groups CSR rows into blocks of `vec_length=8`
  (chosen over the artifact's other supported value, 16, for less padding
  on irregular matrices — see below), takes the UNION of each group's
  nonzero columns, and explicitly materializes a 0.0 value for any row
  missing a nonzero at a column another row in its group has. This padding
  is always an exact literal zero (never a dropped or approximated real
  value, unlike MP-SpMM's 2:4 pruning), so it can never change the true
  product — the gate compares directly against the reference computed on
  the ORIGINAL, unpadded `matrix.csr`.
- A real finding while reading the kernel source: **`row_indices` (the
  row-swizzle/reorder array `spmm::SSpMM` takes) is a dead parameter as
  compiled** — every one of the 5 `__ldg(row_indices + ...)` call sites in
  `src/sspmm.cu` is commented out. The artifact's own test driver's
  `sorted`/`unsorted` CLI flag is therefore a no-op for both correctness and
  performance in this compiled version; this adapter passes an identity
  array (matching the driver's own default) purely for documentation
  honesty.
- `B` drawn with `numpy.random.default_rng(seed).uniform(-1,1,(K,N))`,
  matching `cpu_ref.reference_spmm`'s `_dense_operand` (see `insum`/
  `inferfast`/`mp-spmm` STATUS.md for why this specific RNG match matters —
  a pre-existing, documented harness trap for any adapter that instead used
  `gpu_cuda.py`'s torch-RNG helper).
- **PRECISIONS = ["fp16"]** (mixed fp16-compute/fp32-accumulate — the
  artifact's fp32*fp32 overload just prints "doesn't support float input"
  and returns without computing anything); gated under
  `spmm-tensorcore-fp16` (`--precision fp16`) per the task, tolerance `0.01`
  (parsed cleanly per `--list`).
- N constraint: `spmm::SSpMM`'s dispatch hardcodes `Tile_N=64`
  (`SSpMM_ex<float4,int,1,32,64,32>`), so N must be a multiple of 64 —
  `prepare()` raises `NotImplementedError` otherwise. All three of the
  spec's swept N values (128, 256, 512) satisfy this, so nothing is
  UNSUPPORTED under the default sweep.

## Gate result (login node, functional check only)

```
source bench/artifacts/toolchain.sh
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-tensorcore-fp16 --impl sspmm-rospmm-v8 \
    --precision fp16 --smoke --warmup 1 --reps 2
```

**9/9 smoke runs valid** (3 synthetic matrices x N in {128,256,512}):

| matrix | N | max_scaled_err | tol | GFLOP/s (non-conforming) |
|---|---|---|---|---|
| smoke-uniform | 128 | 3.24e-04 | 0.01 | 75.70 |
| smoke-uniform | 256 | 3.46e-04 | 0.01 | 148.02 |
| smoke-uniform | 512 | 3.63e-04 | 0.01 | 279.58 |
| smoke-banded | 128 | 3.43e-04 | 0.01 | 77.29 |
| smoke-banded | 256 | 3.31e-04 | 0.01 | 154.19 |
| smoke-banded | 512 | 3.57e-04 | 0.01 | 305.92 |
| smoke-powerlaw | 128 | 4.15e-04 | 0.01 | 95.67 |
| smoke-powerlaw | 256 | 4.38e-04 | 0.01 | 187.23 |
| smoke-powerlaw | 512 | 4.65e-04 | 0.01 | 357.45 |

All errors are fp16-typical magnitude (unit roundoff `~5e-4`), well inside
tolerance — a genuinely correct kernel at every tested N, not a marginal
pass. Also checked against a real SuiteSparse/GNN-suite matrix (`cora`,
N=128): `max_scaled_err=3.93e-03` — still comfortably under `0.01`, though
notably higher than the synthetic matrices (see finding below for why).

## Finding: vector-sparse padding overhead is severe on unstructured graphs

Instrumented `prepare()`'s own `padded_vec_nnz` (the vector-sparse format's
actual stored nonzero count, including padding) against the matrix's real
`nnz`, per the tensorcore-fp16 spec's disclosure requirement ("report both
raw-nnz and any zero-padded nnz introduced by tiling"):

| matrix | raw nnz | padded nnz (`padded_vec_nnz * 8`) | ratio |
|---|---|---|---|
| cora | 10,556 | 78,088 | **7.40x** |
| citeseer | 9,104 | 70,480 | **7.74x** |

At `vec_length=8`, the theoretical worst case is 8x (every row in a group
has a disjoint column set, so the union is the sum of all 8 rows' nonzero
counts). Citation-graph adjacency matrices land at 92-97% of that worst
case — essentially no two rows in a random group of 8 consecutive rows
share a nonzero column, which is exactly what an unstructured, low-degree
sparse graph looks like. This is a genuine structural property of applying
a vectorSparse-lineage format to matrices very different from the pruned,
block-regular DNN weights (DLMC) this format family was designed for — not
an artifact of this adapter's conversion (the conversion is a standard,
minimal-padding construction: it takes the exact union per group, no more).
It also explains why `cora`'s measured error (`3.93e-03`) is higher than the
synthetic smoke matrices': with ~7x more terms summed per output element
(most of them exact zeros, but still real fp16 additions/roundings in the
accumulation), more opportunities for fp16 rounding to accumulate — still
well inside the `0.01` tolerance, but a visible, real effect of the format's
fit to this input class.

## Not done

- No sweep across the full recommended_subset or timing runs (out of scope
  per the task's login-node budget: gate-only checks).
- `vec_length=16` was not gated (chosen 8 instead, see above); the artifact
  supports both.
- The Python-side CSR->vector-sparse conversion in `adapter.py` is a
  straightforward per-group Python loop (dict-based union/lookup per row
  group) — correct and adequate for a gate check, but not fast enough for a
  large recommended_subset matrix (e.g. `cant`, nnz~4M) at scale; a real
  timed run would need a vectorized (numpy/Cython/small CUDA kernel)
  version of this conversion. Flagged rather than optimized, matching this
  task's login-node-only scope.

## 2026-09-06: also gated under spmm-binary-adjacency-kernel (competitor)

Run as a weighted-kernel competitor for the new pattern-only-kernel variant
(`spec.yaml`'s `spmm-binary-adjacency-kernel`, see `../README.md`'s
"Pattern-only SpMM variant"): `--variant spmm-binary-adjacency-kernel --impl
sspmm-rospmm-v8 --precision fp16 --smoke --dims 128,256 --warmup 1 --reps 2`
-> **6/6 valid**, `max_scaled_err` 2.0e-04 – 2.9e-04; `--matrices cora --dims
128` -> **1/1 valid**, `max_scaled_err = 3.93e-03` (identical to the
`spmm-tensorcore-fp16`/`cora` result above, since `cora` was already
binary). A fair, expected pass. Reduced-protocol, non-conforming numbers
only (login-node GPU).

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 12.4.0 (conda-forge
  `kb-gcc12` env, via `KB_GXX12`), torch 2.8.0+cu128, Python 3.12.14, arch
  `compute_80,code=sm_80` as recorded.
- Build: OK (only `ptxas info` register/smem reports, no errors/warnings).
  Build-system changes: none (build.sh's existing
  `${NVCC:-...}`/`${HOST_COMPILER:-${KB_GXX12:-...}}` fallbacks already
  read this machine's exported knobs).
- Gate: `spmm-tensorcore-fp16` (fp16, smoke, warmup=1, reps=2) — 9/9 valid,
  `max_scaled_err` 3.24e-04 - 4.65e-04, bit-for-bit identical to the
  recorded table. `cora` (N=128) — 1/1 valid, `max_scaled_err = 3.93e-03`,
  exact match. `spmm-binary-adjacency-kernel` (fp16, smoke, dims 128/256,
  warmup=1, reps=2) — 6/6 valid, `max_scaled_err` 1.98e-04 - 2.92e-04,
  within the recorded 2.0e-04-2.9e-04 range. `cora+bin` (N=128) — 1/1
  valid, `max_scaled_err = 3.93e-03`, exact match (identical to the
  tensorcore-fp16 `cora` result, as recorded, since `cora` is already
  binary).
- Deviation from the recorded ruling: none — every error value reproduces
  exactly or within the recorded range.
- Verdict here: BUILT+GATED — same as the recorded ruling.
