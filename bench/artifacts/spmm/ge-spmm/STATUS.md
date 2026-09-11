# ge-spmm (GE-SpMM) — spmm

**Status: BUILT+GATED (reused build) — clean pass, err ~1.5e-07..4.2e-07,
20/20 workloads valid; no N restriction found (unlike RoDe or TC-GNN).**

- Paper: "GE-SpMM: General-Purpose Sparse Matrix-Matrix Multiplication on
  GPUs for Graph Neural Networks", SC'20. `PAPER_KEY = conf/sc/HuangD0Y20`.
- Artifact: https://github.com/hgyhungry/ge-spmm.
- Commit: `f62f51169eb26c0d4411f6d9744eb585854410e1` (identical to
  `artifacts/gnn-aggregation/ge-spmm` — see "Reuse" below, no second clone).
- Toolchain: `nvcc` 12.9, GPU `sm_80` (A100-PCIE-40GB). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required (CXXABI mismatch, same as
  every other torch-extension artifact in this project).

## Reuse, not a rebuild (task brief's explicit instruction)

This artifact is **already built for the gnn-aggregation track**
(`artifacts/gnn-aggregation/ge-spmm/`: `build.sh` compiled GE-SpMM's real
`source/pytorch-custom/{spmm.cpp,spmm_kernel.cu}` torch CUDA extension into
`build/spmm.so`, byte-identical to the clone). GE-SpMM's compiled
`csr_spmm(rowptr, colind, values, dense)` has no notion of
"gnn-aggregation" vs "spmm" — it is exactly the same weighted CSR SpMM
kernel either way, `C[m,k] = sum_n A_csrVal[...] * B_dnVal[n,k]`, plain
fp32 multiply-accumulate, no tensor cores. This artifact's `build.sh`
therefore does **not** compile anything:
1. symlinks `source/` -> `../../gnn-aggregation/ge-spmm/source` (same
   commit, one clone);
2. imports the compiled module directly from
   `../../gnn-aggregation/ge-spmm/build/spmm.so` — **no local `build/`
   exists here at all**.

`source.provenance` documents the reuse pointer instead of a `reclone`
command (there is nothing to re-clone independently — see that file).
`artifacts/gnn-aggregation/ge-spmm/STATUS.md` has a line noting this reuse.

## Shape mapping — simpler than the gnn-aggregation adapter (no normalization)

`csr_spmm` is exactly `C = A @ B` for an arbitrary real-valued CSR `A` —
this track's own reference (`kernelbench.impls.cpu_ref.reference_spmm` /
`ScipySpMM`, `C = A @ B`, no normalization). Unlike the gnn-aggregation
adapter for this same kernel (which must compute `A_hat =
D^-1/2(A+I)D^-1/2` before feeding `csr_spmm`, since that track's own
reference is GCN-normalized aggregation), this adapter feeds the
workload's own CSR `indptr`/`indices`/`data` straight in, cast to
int32/float32 — no sandwich trick, no normalization, no per-edge-value
detour needed.

## N-sweep (DIM_KEY note, task brief) — checked, no restriction found

`spmm_cuda`'s dispatch (`spmm_kernel.cu:425-454`) is a fully general
3-tier ladder on `k = dense.size(1)` (`k<32` -> `spmm_test0`, `k<64` ->
`spmm_test1`, else -> `spmm_test2`), each tier computing ceiling-divided
tiles of the K dimension with in-kernel bounds checks — there is no N
value the dispatch itself refuses (confirmed by reading the source, not
just the paper's "general-purpose" framing). Checked empirically against
every N `spmm-gpu-kernel-f32` sweeps ({32,128,256,512}, see gate results
below): all 20/20 workloads pass at ordinary fp32 accumulation error.
`prepare()` only asserts `N>0` defensively; no `NotImplementedError` path
is exercised anywhere in this integration — unlike RoDe (N in {32,128}
only, `artifacts/spmm/rode/`) or TC-GNN (F<=128 only,
`artifacts/spmm/tc-gnn/`), GE-SpMM's paper claim of "general-purpose,
arbitrary GNN hidden width" is genuinely reflected in the dispatch code.

## Compute precision

`spmm_test0/1/2` operate on `float*` throughout (`torch::kFloat32`
asserted in `spmm.cpp`); no tensor-core or reduced-precision path exists.
`PRECISIONS = ["fp32"]`, matching `DEFAULT_PRECISION["spmm"] == "fp32"`
already — no explicit `--precision` flag needed on the CLI.

## Gate results (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel spmm --variant spmm-gpu-kernel-f32 \
    --impl gespmm-spmm-f32 --smoke
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel spmm --variant spmm-gpu-kernel-f32 \
    --impl gespmm-spmm-f32 --matrices cant,cora --dims 32,128,256,512 \
    --warmup 2 --reps 3
```

Result: **20/20 valid** (tolerance `1e-4`):

| workload | dims | max_scaled_err |
|---|---|---|
| smoke-uniform/banded/powerlaw (12) | 32,128,256,512 | 1.98e-07 – 2.70e-07 |
| cant (62451x62451, nnz=4,007,383) | 32,128,256,512 | 3.91e-07 – 4.19e-07 |
| cora (2708x2708, nnz=13,264) | 32,128,256,512 | 1.45e-07 – 1.82e-07 |

All comfortably within tolerance, ordinary fp32 accumulation noise, same
order of magnitude as the gnn-aggregation adapter's own gate for the same
compiled kernel (1.93e-07 – 2.88e-07). Reduced-protocol numbers only
(login-node, shared GPU; `conforming: False`), not a timing result per
ARTIFACT_GUIDE.md rule 5.

## Not done

- No compute-node timing sweep (out of scope, ARTIFACT_GUIDE.md rule 5).
- No sweep across the full `spmm-gpu-kernel-f32` `recommended_subset`
  registry — smoke + two representative real matrices (cant, cora), per
  the task brief's login-node/low-cost scope.
- `csr_spmm_no_edge_value` (unweighted path) not wrapped — the weighted
  path already covers this track's one reference exactly (same reasoning
  as `../../gnn-aggregation/ge-spmm/STATUS.md`).
- `spmm-tensorcore-fp16`/`spmm-binary-adjacency-kernel` not attempted for
  this adapter — GE-SpMM has no tensor-core path; those variants are
  covered by `artifacts/spmm/tc-gnn/` instead (see its STATUS.md).

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05; job
  ran on gpu-b11-6.zaratan.umd.edu), login-node build (reused
  `gnn-aggregation/ge-spmm`'s compiled `spmm.so`, no recompilation here).
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14.
- Build: OK (reuse-verify only). Build-system changes: none.
- Gate: `cora` in the recorded command (`--matrices cant,cora`) could not
  be loaded on this machine (`torch_geometric` missing; see
  `../../gnn-aggregation/tc-gnn/STATUS.md`'s reproduction note for the full
  explanation); ran `cant` only:
  - `spmm-gpu-kernel-f32` smoke (12/12 valid): err 1.98e-07 - 2.70e-07
    <= tol 1e-4.
  - `spmm-gpu-kernel-f32` on `cant`, dims 32/128/256/512 (4/4 valid):
    err 3.91e-07 - 4.19e-07 <= tol 1e-4.
- Deviation from the recorded ruling: none -- 16/16 workloads valid here
  (recorded: 20/20 including `cora`); the `cant` numbers reproduce almost
  exactly (recorded `cant` row: 3.91e-07 - 4.19e-07; here: 3.91e-07 -
  4.19e-07), and no N restriction was found here either, consistent with
  the recorded ruling.
- Verdict here: BUILT+GATED -- equals the recorded ruling.
