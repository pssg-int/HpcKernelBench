# tc-gnn (TC-GNN) — sddmm

**Status: BUILT+GATED (reused build) — cora (dim 32, 128) passes at
tf32-level error (2.67e-04, 1.19e-04, tol 1e-2); every smoke workload is
UNSUPPORTED via the shared `num_nodes % 16 == 0` guard; a NEW, previously
undocumented artifact bug found and guarded: `embedding_dim % 8 != 0`
silently corrupts output via cross-row data bleed.**

- Paper: "TC-GNN: Bridging Sparse GNN Computation and Dense Tensor Cores on
  GPUs" (USENIX ATC'23). `PAPER_KEY = conf/usenix/WangFWHD23`.
- Artifact: https://github.com/YukeWang96/TC-GNN_ATC23.
- Commit: `0ff60b2f0acbce25bc0b137d3909c41a520da0fa` (identical to
  `artifacts/gnn-aggregation/tc-gnn` — see "Reuse" below, no second clone).
- Toolchain: `nvcc` 12.9, GPU `sm_80` (A100-PCIE-40GB, `TORCH_CUDA_ARCH_LIST
  =8.0`). Python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch
  `2.8.0+cu128`. `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required.

## Reuse, not a rebuild (task brief's explicit instruction)

Already built for the gnn-aggregation track (`artifacts/gnn-aggregation/
tc-gnn/`). This artifact wraps a DIFFERENT entry point from the SAME
compiled `.so` — `TCGNN.forward_ef` (SDDMM), not `forward_AGNN` (SpMM,
used by `../../spmm/tc-gnn/`). `build.sh` does not compile anything:
symlinks `source/` -> `../../gnn-aggregation/tc-gnn/source`, imports
`TCGNN` directly from `../../gnn-aggregation/tc-gnn/build/`. See
`../../gnn-aggregation/tc-gnn/STATUS.md`'s reuse note.

## What `forward_ef` computes (read directly from TCGNN_kernel.cu, per the task brief's instruction)

`sddmm_forward_cuda_kernel` takes ONE dense input matrix `in_mat` plus the
same structural preprocessing as `forward`/`forward_AGNN`, and computes,
per edge `(row,col)` in the sparsity pattern, `edgeFeature[eIdx] =
dot(in_mat[row], in_mat[col])` — a tf32 WMMA `X @ X^T`, SAMPLED at the
sparsity pattern, using the SAME matrix for both the row and column
operand (no second tensor argument exists in `sddmm_forward`'s signature
at all). Output lands at `edgeFeature[eIdx]` in the ORIGINAL edge-list
(CSR nnz) order — no TC-block-layout reindex on the way out (confirmed by
reading the store loop), unlike Fused3S's `saveSddmmResult` path. It is an
UNWEIGHTED dot product (never reads `S.data`), same situation as Fused3S
— `S[i,j]` is applied post-hoc in `to_host()` here, identical pattern to
`../../sddmm/fused3s/adapter.py`.

## The "self-SDDMM only" gap, bridged via a bipartite double-cover embedding

`forward_ef` structurally cannot take two INDEPENDENT dense operands (one
`input` tensor only), but this track's own reference
(`reference_sddmm`) draws A (M x K, row-indexed, seed) and B (N x K,
col-indexed, seed+1) INDEPENDENTLY. **Bridge (adapter-side only, no
kernel code touched)**: build a combined feature matrix `Z` of shape
`(M+N) x K` with `Z[0:M]=A`, `Z[M:M+N]=B`, and re-index every edge
`(i,j)` to `(i, M+j)` — treating A's M rows and B's N rows as `M+N`
distinct nodes in one node space, with `nodePointer` extended by `N` more
permanently-zero-out-degree rows for the B side. `forward_ef(Z, ...)`
then computes exactly `dot(A[i], B[j])` at exactly the original nnz
position (edge order preserved by construction — column ids are just
shifted by `M`). This is pure preprocessing (index remap + concatenation),
computing the SAME mathematical quantity the general two-operand SDDMM
does, not an approximation.

**Verified independently before wiring into the adapter** (standalone
script, `/tmp/.../validate_tcgnn_sddmm.py`, not checked in — brute-force
numpy vs. TC-GNN's own kernel through the embedding):

| M | N | K | num_nodes | max_scaled_err |
|---|---|---|---|---|
| 50 | 70 | 32 | 120 (non-square, off-boundary window) | 2.22e-04 |
| 2708 | 2708 | 32 | 5416 (cora-sized, square) | 2.86e-04 |
| 2708 | 2708 | 128 | 5416 | 1.63e-04 |

All at TF32-level scaled error (using the domain's own `scale =
|B[cols]|@|A[i]|` denominator, not a naive `|ref|` one, matching
`cpu_ref.reference_sddmm`'s own convention) — confirms the embedding is
correct, not merely "runs without crashing."

Cost note (not a correctness issue): TC-GNN's own `preprocess()`/kernel
iterate over `M+N` nodes and `(M+N+15)//16` row windows — roughly double
the bookkeeping a hypothetical native two-operand SDDMM kernel would need.
Irrelevant here (no timing measurement performed, ARTIFACT_GUIDE.md rule
5), noted for whoever reads these numbers as throughput later.

## Two guards

**1. `(M+N) % 16 == 0`** -> `NotImplementedError` — the SAME shared
`preprocess()` off-by-one bug already confirmed (canary-buffer test) in
`../../gnn-aggregation/tc-gnn/STATUS.md` finding #2, now triggered on the
COMBINED node count `M+N` (since both operands are folded into one node
space by the embedding above) rather than on `M` alone. Every SMOKE
workload here is `M=N=4000` -> `M+N=8000`, `8000 % 16 == 0` -> guard fires
on **every** `--smoke` workload. `cora` (`M=N=2708` -> `M+N=5416`,
`5416 % 16 == 8`) is safe.

**2. NEW finding (not previously documented anywhere in this repo):
`K % BLK_W (8) != 0` silently corrupts output via cross-row data bleed.**
`sddmm_forward_cuda_kernel`'s dense-tile staging loop bounds-checks each
fetch only against the TOTAL matrix size
(`source_idx >= numNodes*embedding_dim`), not the current row's own valid
range. For the LAST `warp_iter` chunk when `embedding_dim % BLK_W != 0`,
`dense_dimIdx` values past `embedding_dim % BLK_W` push `source_idx` into
the START of the NEXT node's feature row (row-major, stride
`embedding_dim`) instead of being zero-padded — silently mixing a
DIFFERENT node's values into the dot product rather than crashing or
reading garbage. **Confirmed empirically** (standalone script, random
`M=50,N=70` pattern, same validation harness as above):

| K | max_scaled_err |
|---|---|
| 20 (not mult. of 8) | 5.48e-01 |
| 33 (not mult. of 8) | 2.48e-01 |
| 24 (mult. of 8) | 2.29e-04 |

Guarded: `prepare()` raises `NotImplementedError` for `K % 8 != 0`. This
track's `sddmm-tensorcore-blocked-fp16` dims sweep is `{32, 128}` — both
multiples of 8 — so this guard is never exercised by a spec-conforming run
of this variant; reported because it was found while reading/testing the
exact kernel this adapter wraps (ARTIFACT_GUIDE.md rule 8's spirit: name a
constraint even when the track's own sweep does not currently hit it).

## Compute precision

Same TF32 WMMA family as `forward`/`forward_AGNN` (`wmma::precision::
tf32`, fp32 accumulate). Gated under `sddmm-tensorcore-blocked-fp16`
(tolerance `1e-2`) — same class of variant `fused3s`/`flashsparse` use in
this track. `PRECISIONS = ["fp16"]`.

## Gate results (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel sddmm --variant sddmm-tensorcore-blocked-fp16 --precision fp16 \
    --impl tcgnn-ef-sddmm-f16 --smoke
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel sddmm --variant sddmm-tensorcore-blocked-fp16 --precision fp16 \
    --impl tcgnn-ef-sddmm-f16 --matrices cora --dims 32,128 \
    --warmup 2 --reps 3
```

| workload | K | outcome | max_scaled_err |
|---|---|---|---|
| smoke (6/6) | 32,128 | UNSUPPORTED (`M+N=8000 % 16 == 0`) | -- |
| cora | 32 | **VALID** | 2.67e-04 (tol 1e-2) |
| cora | 128 | **VALID** | 1.19e-04 (tol 1e-2) |

**2/2 serviceable workloads pass.** Reduced-protocol numbers only (login
node, shared GPU; `conforming: False`), not a timing result per
ARTIFACT_GUIDE.md rule 5.

## Not done

- No compute-node timing sweep (out of scope, rule 5).
- No sweep across the full `recommended_subset` registry — smoke + `cora`
  only, per the task brief's login-node/low-cost scope.
- Finding #1's canary-buffer memory-corruption reproduction was not
  independently re-run here (reused from the gnn-aggregation build's own
  test, same compiled code path); finding #2's empirical test used a
  standalone non-checked-in script rather than the actual runner harness.
- The bipartite double-cover embedding was validated on 3 synthetic
  shapes, not swept across the full dims/matrix registry.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05; job
  ran on gpu-b11-6.zaratan.umd.edu), login-node build (reused
  `gnn-aggregation/tc-gnn`'s compiled `TCGNN.so`, no recompilation here).
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14.
- Build: OK (reuse-verify only, same pattern as `spmm/tc-gnn`; succeeded on
  the first attempt here, no `RLIMIT_NPROC` contention hit this time).
  Build-system changes: none.
- Gate: `cora` in the recorded command could not be loaded on this machine
  (`torch_geometric` missing; see `../../gnn-aggregation/tc-gnn/STATUS.md`'s
  reproduction note for the full explanation). Substituted `cant`
  (`M=N=62451`, combined node count `M+N=124902`, `124902 % 16 == 6`, safe
  from guard #1 exactly like `cora`'s `M+N=5416`):
  - `sddmm-tensorcore-blocked-fp16` smoke (6 workloads, `M+N=8000`): 0/0
    valid, 6/6 UNSUPPORTED via guard #1 (`(M+N) % 16 == 0`) -- matches
    recorded.
  - `sddmm-tensorcore-blocked-fp16` on `cant`, K=32: **VALID**,
    err=3.43e-04 <= tol 1e-2.
  - `sddmm-tensorcore-blocked-fp16` on `cant`, K=128: **VALID**,
    err=1.70e-04 <= tol 1e-2.
- Deviation from the recorded ruling: none in substance -- `cant`
  substituted for `cora` (torch_geometric unavailable, see above); 2/2
  serviceable workloads pass at TF32-level error, same pattern as the
  recorded `cora`-based verification (2.67e-04 / 1.19e-04 there vs
  3.43e-04 / 1.70e-04 here).
- Verdict here: BUILT+GATED -- equals the recorded ruling.
