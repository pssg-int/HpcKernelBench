# NM-SpMM — STATUS

**Status: BUILT+GATED — the wired 32x32/sparsity=0.5 slice passes cleanly on
a synthetic N:M-structured workload (the artifact's mandated input class);
general SuiteSparse matrices are correctly rejected with `NotImplementedError`
(regime: PARTIAL, per the task brief).**

- Paper: "NM-SpMM: Efficient N:M Sparsity for Deep Learning" (IPDPS'25).
  `PAPER_KEY = conf/ipps/MaWDCH0ZWZCDWFP25`.
- Artifact: https://github.com/CMa-X/NM-SpMM
- Commit cloned: `9a926e08c309d853bb0db3651689799dc4f75c33` (2024-10-31),
  `git clone --depth 1`.
- Toolchain: `nvcc` 12.9, host compiler g++-12 (`/opt/cray/pe/gcc-native/12/bin/g++`,
  SUSE 12.3.0 — the same nvcc-12.9/g++-14 `<bits/alloc_traits.h>`
  `__has_construct` mismatch documented in `../inferfast/build.sh`,
  `../mp-spmm/build.sh`, `../sspmm/build.sh`; triggered here by
  `NM-SpMM.h`'s `#include <iostream>`), `-gencode arch=compute_80,code=sm_80`.
  Python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.
- Selection rationale: core baseline under the revised kernel-centrality
  rule — N:M structured sparsity is exactly this paper's headline
  contribution, evaluated at kernel level against nmSPARSE and cuBLAS dense
  GEMM (paper's own claims: 2.1x over nmSPARSE, 1.4-6.3x over cuBLAS dense).
  No Hopper-only feature: `CMakeLists.txt` targets `sm_80` explicitly and
  the only non-portable PTX used is `cp.async.ca/cg` (`source/include/ptx.h`),
  an Ampere-generation (sm_80+) async-copy instruction — this machine's A100
  runs it natively, so this artifact is **not** DEFERRED-HARDWARE.

## Precision — real finding vs. the task brief's expectation

The task brief anticipated an fp16 Tensor-Core kernel ("fp16 tensor cores
-> fp16 bound 1e-2"). **That does not match the artifact.** Grepped every
`source/src/*.cu` and `source/include/ptx.h` for `half`/`wmma`/`mma.sync`/
`tensor`: none found. Every kernel buffer (`A`, `B`, `C`) is `float*`; the
kernels are CUDA-core FMA loops with `cp.async`-based double-buffered
shared-memory pipelines (a well-tuned SGEMM-style design), not Tensor-Core
MMA. **`PRECISIONS = ["fp32"]`**, gated under `spmm-gpu-kernel-f32`'s
tolerance family (parsed `1e-4`), not `spmm-tensorcore-fp16`.

## Does an existing quantized/structured-sparsity variant fit? No — checked and ruled out

`benchspecs/spmm/spec.yaml`'s `spmm-gpu-quantized-int` variant names NM-SpMM
explicitly in its claim text (alongside Magicube/InferFast/GeneralSparse),
but its protocol is int4/int8/int16 **packed integer** arithmetic with
**EXACT integer-match correctness** ("no tolerance, since this is integer
arithmetic, not floating point"). NM-SpMM computes real fp32 arithmetic —
comparing its output to an exact-integer-match gate is a category error, not
a looser or tighter version of the same claim. Ruling: **does not fit**;
this appears to be a spec-authoring conflation of "structured sparsity" with
"quantized" rather than a literal claim that NM-SpMM does integer math. No
other existing variant is a better fit for "fp32 GEMM against an N:M-pruned
weight" than `spmm-gpu-kernel-f32`'s own tolerance family — this adapter
targets that variant's correctness bound while substituting its own
synthetic input for the variant's `recommended_subset` (see next section).

## Regime: PARTIAL — what NM-SpMM actually needs as input

Traced `source/tests/test_nmspmm.cu` + `source/include/utils.h::init_data`
(the artifact's ONLY data-generation path; it ships no general-CSR-to-NM
converter, the same situation `../sspmm/STATUS.md` documents for
vector-sparse conversion). For `C[M,N] = A[M,K](dense) @ Wsparse[K,N]`:
group the `N` axis into chunks of `VEC_LEN=32`; for every chunk and every
contiguous `pruning_M`-wide block of `K`, exactly `pruning_N` of the
`pruning_M` rows survive — **and the surviving row-SET is identical across
all 32 columns in the chunk** (`init_data`'s
`DT[(k+u)*Q+a] = tmp_index[u]` is indexed by `(compressed-row, column-chunk)`
only, never by an individual column). This is a coarse, *vector-wise* N:M
pattern, fundamentally incompatible with per-row-independent sparsity (what
every general SuiteSparse matrix has).

Reconciling with this track's `C[M,N] = A[M,K](sparse) * B[K,N](dense)`
convention (sparse operand on the LEFT, opposite of NM-SpMM's own
convention) via the transpose identity `A@B = (B^T @ A^T)^T`
(`M_nm=Nh, K_nm=Kh, N_nm=Mh, A_nm=B_dense^T, Wsparse_nm=matrix.csr^T`,
`C = C_nm^T`) means the ORIGINAL `matrix.csr`'s requirement becomes: **every
group of 32 CONSECUTIVE ROWS must share one identical nonzero-column
pattern per `pruning_M`-wide column block** (values may differ per row).
`adapter.py::verify_and_extract_nm_pattern()` checks this explicitly and
raises `NotImplementedError` naming the constraint when it fails — verified
directly against `smoke-uniform` (a `sparse.py` smoke matrix): row 0 of
group 0 has only 1 nonzero in its first 32-column block (a normal,
independent-per-row sparse pattern), so the very first check fails
immediately and cleanly:

```
nmspmm-32x32-n2m1to2: N:M structured sparsity required; general SuiteSparse
matrices do not satisfy it -- row-group 0 (rows 0..31), column block 0
(cols 0..31): row 0 has 1 nonzeros in this block, expected exactly
pruning_N=16 (sparsity=0.5)
```

## What N:M configs does the artifact actually support? (survey finding)

`source/tests/test_nmspmm.cu` hardcodes exactly 4 `sparsity` values, each
with a FIXED compile-time block width (`pruning_M` is a CLI argument but is
**never forwarded into any kernel launch** — the real block width is the
template constant `Ks`, baked per-dispatcher in `source/src/kernel_*.cu`):

| sparsity | kept : block (`Ks`) | reduced ratio | preprocessing path |
|---|---|---|---|
| 0.5   | 16 : 32 | 1:2 | low  (`PreProcessing_low_sparsity` only) |
| 0.625 | 12 : 32 | 3:8 | low  (`PreProcessing_low_sparsity` only) |
| 0.75  | 16 : 64 | 1:4 | high (`transIndex` + `PreProcessing_high_sparsity` + `column_info` side array) |
| 0.875 |  8 : 64 | 1:8 | high (same as above) |

**None of these is the fine-grained "2:4" pattern** (block-of-4) most N:M
literature means — NM-SpMM's own contribution IS this coarser block width
(32 or 64), amortizing per-element metadata across a whole vector of 32
output columns (the paper's "hierarchical blocking" claim), at a real cost
in pruning granularity relative to per-row 2:4.

Each of the 4 sparsity levels is also available at 4 kernel tile sizes
(`kernel_32x32_4x4`, `kernel_32x64_8x4`, `kernel_64x64_8x8`,
`kernel_64x128_8x8`), selected by matrix-size dispatch in
`test_nmspmm.cu`'s `main()`.

## What was wrapped

**Scope wired here (login-node integration budget): sparsity=0.5 (16:32,
the `low` preprocessing path) on the smallest tile, `kernel_32x32_4x4`.**
`wrapper.cu` (this directory) adds `extern "C"` linkage around two of the
artifact's own, UNMODIFIED functions:

- `nm_preprocess_low` -> `PreProcessing_low_sparsity`
  (`source/src/preprocessing.cu`) — the artifact's own layout-transform
  preprocessing for the index array, called via ctypes exactly as the
  artifact's own test driver calls it (ARTIFACT_GUIDE.md rule 2: this is
  the artifact's format conversion, invoked unmodified, not reimplemented).
- `nm_gemm_32x32_low` -> `nmGEMM_small_matrices_low_sparsity`
  (`source/src/kernel_32x32_4x4.cu`) — dispatches to
  `kernel_32x32_4x4_low_sparsity<Ms=32,Ns=32,Ks=32,Ws=16,Mt=4,Nt=4>`, the
  paper's own kernel.

`adapter.py::verify_and_extract_nm_pattern()` is THIS adapter's own code
(not artifact code): it verifies the structural contract and — only if it
holds — extracts the per-(row-group, column-block) shared pattern and
per-row values needed to build the artifact's compressed `B` (row-major
dense values) and raw index array `DT` (local block offsets, matching
`init_data()`'s own `DT[(k+u)*Q+a] = tmp_index[u]` construction exactly,
independently re-derived in Python since the artifact has no general
converter — same situation as `../sspmm/adapter.py::_vectorize_csr`). The
artifact's OWN `PreProcessing_low_sparsity` C function is then called
(unmodified, via ctypes) to apply the final shared-memory-friendly layout
swizzle — nothing about that transform was reimplemented or guessed at.

Dense operand `B_dense` (harness's `[Kh,Nh]` operand) drawn with
`numpy.random.default_rng(seed).uniform(-1,1)`, matching
`cpu_ref.reference_spmm`'s `_dense_operand` exactly (same RNG-match
convention as every other adapter in this directory).

## Gate result (login node, functional check only)

Real matrices from `spmm-gpu-kernel-f32`'s `recommended_subset` cannot
satisfy the structural contract (confirmed above on a `sparse.py` smoke
matrix; every real SuiteSparse/GNN matrix has independent per-row sparsity,
so the same rejection is certain without needing to download and check
each one individually). Gated instead on a **standalone synthetic N:M
workload** built to satisfy the exact constraint
(`gate_synthetic_nm.py`, this directory — precedent:
`../../spmv/diaq/STATUS.md`'s substitute gate), run through the SAME code
path the runner CLI uses (`kernelbench.harness.run_variant`) against
`spmm-gpu-kernel-f32`'s own tolerance and
`kernelbench.impls.cpu_ref.reference_spmm` (the harness's independent fp64
CPU reference — never a loosened tolerance):

```
source bench/env.sh
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" \
    bench/artifacts/spmm/nm-spmm/gate_synthetic_nm.py
```

```
variant: spmm-gpu-kernel-f32  tolerance: 0.0001 (parsed)
matrix: 128x128, nnz=8192 (density=0.500, expect 0.500)
  N=  32  valid=True  PASS  max_scaled_err=1.485e-07  tol=0.0001  gflops=5.80
  N= 128  valid=True  PASS  max_scaled_err=1.779e-07  tol=0.0001  gflops=31.61
  N= 256  valid=True  PASS  max_scaled_err=1.757e-07  tol=0.0001  gflops=71.31
  N= 512  valid=True  PASS  max_scaled_err=2.056e-07  tol=0.0001  gflops=108.95

ALL PASS
```

**4/4 substitute-gate runs valid**, all four of `spmm-gpu-kernel-f32`'s
swept `N` values. Error is `~1.5-2e-7`, three orders of magnitude under the
`1e-4` bound — a genuinely correct fp32 kernel (plain FMA accumulation, no
reduced-precision rounding to explain away), not a marginal pass.
`gflops` figures are login-node/non-conforming (shared GPU, `warmup=1,
reps=2`) and are NOT a performance claim.

Also confirmed the rejection path is exercised for real (not just
theoretically correct): running `adapter.py`'s `prepare()` directly on
`sparse.py`'s `smoke-uniform` matrix raises `NotImplementedError` on the
very first row-group/column-block it checks (quoted above).

## Not done

- **0.625/0.75/0.875 sparsity** and the **3 larger kernel tiles**
  (`kernel_32x64_8x4`, `kernel_64x64_8x8`, `kernel_64x128_8x8`) exist in the
  artifact and were surveyed (table above) but not wired into
  `wrapper.cu`/`adapter.py` — the 0.75/0.875 "high sparsity" path
  additionally needs `transIndex` + `PreProcessing_high_sparsity` +
  building a `column_info` side array, meaningfully more preprocessing
  logic than the login-node integration budget for one artifact covers.
- No sweep across the real `recommended_subset` (impossible by construction
  for this artifact — see "Regime: PARTIAL" above) or across matrix sizes
  beyond the one 128x128 synthetic gate matrix.
- No timing runs (login-node budget; gate-only per ARTIFACT_GUIDE.md rule 5).
- `nmGEMM_on_cpu` (the artifact's OWN CPU reference in `utils.h`) was not
  ported or called — this integration uses the harness's own
  `cpu_ref.reference_spmm` exclusively, per the correctness-gate rule
  (rule 4: gate against the harness's independent reference, not the
  artifact's self-check).

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 12.4.0 (conda-forge
  `kb-gcc12` env, via `KB_HOST_COMPILER_BIN`), torch 2.8.0+cu128, Python
  3.12.14, arch `compute_80,code=sm_80` as recorded.
- Build: OK. Build-system changes: none (build.sh's existing
  `${HOST_COMPILER:-${KB_HOST_COMPILER_BIN:-...}/g++}` fallback already
  reads this machine's `KB_HOST_COMPILER_BIN`, exported by
  `bench/artifacts/toolchain.sh`).
- Gate: `gate_synthetic_nm.py` substitute gate (128x128 synthetic N:M
  workload, sparsity=0.5, `spmm-gpu-kernel-f32` tolerance 1e-4) — 4/4 valid,
  `max_scaled_err` 1.485e-07 (N=32) / 1.779e-07 (N=128) / 1.757e-07 (N=256) /
  2.056e-07 (N=512) — bit-for-bit identical to the recorded table (GFLOP/s
  figures differ, expected given the different GPU/protocol, and are
  non-conforming numbers either way). Did not separately re-verify the
  `NotImplementedError` rejection on a real SuiteSparse-shaped matrix (its
  exact triggering text is already quoted above and depends only on
  `verify_and_extract_nm_pattern`'s pure-Python structural check, unaffected
  by this machine's toolchain).
- Deviation from the recorded ruling: none — the substitute-gate errors
  reproduce exactly, confirming a genuinely correct fp32 kernel on this
  machine too.
- Verdict here: BUILT+GATED (PARTIAL regime, substitute-gate PASS) — same
  as the recorded ruling.
