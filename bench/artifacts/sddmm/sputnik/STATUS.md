# Sputnik (sddmm) — STATUS

**Status: BUILT+GATED — clean pass on smoke matrices and on both mandated
real matrices (`cant`, `cora`).**

- Paper: "Sparse GPU Kernels for Deep Learning" (Gale, Zaharia, Young,
  Elsen), SC'20. `PAPER_KEY = conf/sc/GaleZYE20`.
- `source/` is a SYMLINK to `../../spmm/sputnik/source` (same convention as
  `artifacts/sddmm/rode/source -> ../../spmm/rode/source`) — read-only reuse
  of the spmm track's already-cloned canonical `google-research/sputnik`
  checkout (commit `bbf5840ba5efccf01f862855c785f71bcc6ff1f0`). See
  `../../spmm/sputnik/STATUS.md` for the full provenance ruling (listed
  fork `luckylsk34/Sparse-Kernels` vs. canonical repo — stripped Apache-2.0
  notices, not re-derived here) and toolchain details (g++-12, nvcc 12.9,
  sm_80, same `-x cu` build-system fix, same reasoning applies verbatim).
- Selection rationale: core baseline under the revised kernel-centrality
  rule — same paper/artifact as the spmm track's Sputnik integration;
  `sddmm-csr-kernel-f32`'s own claim text names "the Sputnik/Gale-2020
  kernels every later paper baselines against" explicitly.

## What was wrapped

**`sputnik::CudaSddmm`** (`source/sputnik/sddmm/cuda_sddmm.h`/`.cu.cc`,
UNMODIFIED) — fp32, computes a pure pattern-restricted dot product
(`output_values[nz] = dot(lhs_matrix[row(nz)], rhs_matrix[col(nz)])`, NO
separate "sparse value" operand in Sputnik's own signature at all).
Dispatches internally on `k % 4`/`k % 2` (`cuda_sddmm.cu.cc:169-195`), so
every K in the spec's sweep works directly, no `NotImplementedError`
needed.

**Convention match, not a bug workaround:** `benchspecs/sddmm/spec.yaml`'s
`operation` field is `"P[i,j] = S[i,j] * dot(A[i,:], B[j,:])"` and
`cpu_ref.reference_sddmm` literally multiplies the dot product by the
original matrix's stored value (`out[lo:hi] = (B[cols] @ A[i]) *
data[lo:hi]`). Since Sputnik's kernel has no such operand, this adapter's
`run()` multiplies Sputnik's raw output by the matrix's own values
on-device AFTER the kernel call — the exact same convention this
codebase's own `kernelbench.impls.gpu_cuda.TorchSDDMM`/`CustomSDDMM`
adapters already use (`TorchSDDMM.run()`: `out.values() * h["Sdata"]`), so
this is matching an established pattern, not inventing a special case.

**`SortedRowSwizzle`** — ported byte-for-byte in this directory's
`wrapper.cu` (duplicated from `../../spmm/sputnik/wrapper.cu`'s copy rather
than shared via a header, to keep this directory's `build.sh`
self-contained and independent of the spmm directory's build products —
see that file's header comment for the full Glog/Abseil-avoidance
rationale, not re-derived here).

Dense operands: `A` (M x K) seed 42, `B` (N x K) seed 43 —
`numpy.random.default_rng`, matching `cpu_ref.reference_sddmm`'s
`_dense_operand` seed/seed+1 convention exactly (same as `../rode/adapter.py`).

## Gate result (login node, functional check only)

```
source bench/env.sh
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" -m kernelbench.runner \
    --kernel sddmm --variant sddmm-csr-kernel-f32 --impl sputnik-sddmm-f32 \
    --smoke --warmup 1 --reps 2
```
**6/6 smoke runs valid** (3 synthetic matrices x K in {32,128} -- sddmm's
smoke set has no `dim=256/512`, `DIM_KEY["sddmm"]="K"` and
`dense_dims()` for this variant returns `{32,128}` per spec), `max_scaled_err`
in `6.5e-8..1.4e-7`.

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" -m kernelbench.runner \
    --kernel sddmm --variant sddmm-csr-kernel-f32 --impl sputnik-sddmm-f32 \
    --matrices cant --dims 128 --warmup 1 --reps 2
```
`cant`: **1/1 valid**, `max_scaled_err=7.98e-08`.

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$PY" -m kernelbench.runner \
    --kernel sddmm --variant sddmm-csr-kernel-f32 --impl sputnik-sddmm-f32 \
    --matrices cora --dims 128 --warmup 1 --reps 2
```
`cora`: **1/1 valid**, `max_scaled_err=6.54e-08`.

All `gflops`/`ms` figures are login-node/non-conforming (shared GPU,
`warmup=1, reps=2` overrides) and are NOT a performance claim.

## Regime note

Same as `../../spmm/sputnik/STATUS.md`: Sputnik's own evaluation regime is
DNN-pruned weights at moderate sparsity, not general SuiteSparse — gated
anyway per ARTIFACT_GUIDE.md rule 4, and passes cleanly as shown above.

## Not done

- No sweep across the full `recommended_subset` or timing runs (login-node
  budget).
- `CudaSddmmEx`'s bias/relu-free template parameters beyond the default
  dispatch, half-precision variants: not wrapped (out of scope).

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 12.4.0 (conda-forge,
  `$KB_HOST_COMPILER_BIN`/`$KB_GXX12`), torch 2.8.0+cu128, Python 3.12.14;
  `-gencode arch=compute_80,code=sm_80` (unchanged from build.sh).
- Build: OK. Build-system changes: none this pass (the ldd/nm pipefail-safety
  fix for this build.sh was already committed in `aa30f86`; carried
  unchanged, re-ran idempotently; `source/` symlink to `../../spmm/sputnik/
  source` resolved correctly).
- Gate: `sddmm-csr-kernel-f32 --smoke`: 6/6 valid, `max_scaled_err`
  6.51e-08–1.43e-07 <= tol 1e-4, PASS. `--matrices cant --dims 128`: 1/1
  valid, err=7.98e-08, PASS. `--matrices cora --dims 128`: **FAILED TO RUN**
  — `ModuleNotFoundError: No module named 'torch_geometric'` (same
  Planetoid-loader environment gap as `../../spmm/sputnik` and
  `../../spmm/rode`; not an artifact/kernel issue, not fixed here — shared
  harness dependency, out of scope for this directory).
- Deviation from the recorded ruling: none for the kernel itself — the
  `cant` error matches the Perlmutter record exactly (7.98e-08); smoke range
  is consistent. Only the `cora` check is untested here due to the missing
  `torch_geometric` dependency in this environment.
- Verdict here: BUILT+GATED — same as the recorded ruling.
