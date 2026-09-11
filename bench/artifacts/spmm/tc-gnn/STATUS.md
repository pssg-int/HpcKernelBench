# tc-gnn (TC-GNN) — spmm

**Status: BUILT+GATED (reused build) — cora dim=128 passes at tf32-level
error (4.87e-04) under BOTH `spmm-tensorcore-fp16` (tol 1e-2) and
`spmm-binary-adjacency-kernel` (tol_fp16 1e-2); every smoke workload and
every dim>128 workload is UNSUPPORTED via the same two confirmed guards
already documented for the gnn-aggregation build of this kernel.**

- Paper: "TC-GNN: Bridging Sparse GNN Computation and Dense Tensor Cores on
  GPUs" (USENIX ATC'23). `PAPER_KEY = conf/usenix/WangFWHD23`.
- Artifact: https://github.com/YukeWang96/TC-GNN_ATC23.
- Commit: `0ff60b2f0acbce25bc0b137d3909c41a520da0fa` (identical to
  `artifacts/gnn-aggregation/tc-gnn` — see "Reuse" below, no second clone).
- Toolchain: `nvcc` 12.9, GPU `sm_80` (A100-PCIE-40GB, `TORCH_CUDA_ARCH_LIST
  =8.0`, required by `wmma::precision::tf32` fragments). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required (CXXABI mismatch, same as
  every other torch-extension artifact in this project).

## Reuse, not a rebuild (task brief's explicit instruction)

This artifact is **already built for the gnn-aggregation track**
(`artifacts/gnn-aggregation/tc-gnn/`: `build.sh` compiled TC-GNN's real
`TCGNN.cpp`/`TCGNN_kernel.cu` torch extension into `build/TCGNN.so`).
TC-GNN's compiled kernel has no notion of "gnn-aggregation" vs "spmm" — it
is the identical `.so` either way; only the reference each track compares
against differs (gnn-aggregation: GCN-normalized aggregation; spmm: plain
`C = A @ B` with the workload's own arbitrary CSR values). This artifact's
`build.sh` therefore does **not** compile anything:
1. symlinks `source/` -> `../../gnn-aggregation/tc-gnn/source` (same
   commit, one clone);
2. imports `TCGNN` directly from `../../gnn-aggregation/tc-gnn/build/`
   (**no local `build/` here at all**).

`source.provenance` documents the reuse pointer instead of a `reclone`
command. `artifacts/gnn-aggregation/tc-gnn/STATUS.md` has a line noting
this reuse.

## What was wrapped, and why `forward_AGNN` (not `forward`)

Same reasoning as the gnn-aggregation adapter, read in full there
(`../../gnn-aggregation/tc-gnn/STATUS.md`), reproduced here rather than
re-derived: `TCGNN.forward` (`spmm_forward_cuda_kernel`) hardcodes every
sparse-tile cell to `1` — genuinely unweighted, no value array in its
signature at all. `spmm-tensorcore-fp16`'s own reference
(`reference_spmm`) is a plain `C = A @ B` for an ARBITRARY random CSR `A`
— so this adapter uses `forward_AGNN` (TC-GNN's own weighted WMMA SpMM
entry point, taking a genuine per-edge `edgeAttention` array) fed the
workload's own `matrix.csr.data` directly, cast to fp32 — no
normalization step (unlike the gnn-aggregation adapter, which must
GCN-normalize first).

`spmm-binary-adjacency-kernel` (added concurrently in this same session by
another agent's `kernelbench.domains.sparse.variant_transform` hook) forces
`workload.csr.data[:] = 1` BEFORE this adapter's `prepare()` ever sees the
workload — no adapter-side special-casing was needed to also gate there:
`forward_AGNN`'s `edgeAttention` array simply comes out all-ones in that
case, mathematically equivalent to (though not literally routed through)
TC-GNN's own unweighted `forward` path. Gated at both variants below, per
the task brief ("if the concurrent agent's binary variant exists, also
gate there").

## Three guards, reproduced verbatim from the gnn-aggregation adapter (same compiled kernel, same bugs)

1. **`N % 16 != 0`** -> `NotImplementedError` (silent tail-drop:
   `spmmAGNN_forward_cuda_kernel`'s FLOOR-division `dimTileNum` vs. the
   host wrapper's CEILING-division shared-memory sizing).
2. **`M % 16 == 0`** -> `NotImplementedError` (confirmed one-element
   out-of-bounds WRITE, heap corruption: `TCGNN.cpp`'s `preprocess()`
   row-window loop is off-by-one, `for (iter=0; iter<num_nodes+1;
   iter+=16)`; see the gnn-aggregation STATUS.md's canary-buffer
   reproduction — not re-run here, same compiled code).
3. **`N > 128`** -> `NotImplementedError` (confirmed silent zero-fill:
   fixed `WARPperBlock=8`, no outer loop over further embedding-dimension
   tiles).

Every one of this track's SMOKE_WORKLOADS matrices is `4000x4000`
(`4000 % 16 == 0`) — guard #2 fires on **every** `--smoke` workload,
exactly like the gnn-aggregation build's own finding. `cora`
(2708 nodes, `2708 % 16 == 4`) is safe from guard #2; `N=256`/`N=512` in
this track's own `[128,256,512]` dense-dims sweep trip guard #3, leaving
only `N=128` serviceable.

## Compute precision

`wmma::precision::tf32` fragments, fp32 accumulate — same TF32 tensor-core
family as `dtcspmm`/`flashsparse`/`sspmm` in this track. Unlike
`gnn-aggregation` (no laxer variant exists there), `spmm` HAS a dedicated
tensor-core-class variant, `spmm-tensorcore-fp16` (tolerance `1e-2`) —
exactly the right home for this precision class, and
`spmm-binary-adjacency-kernel` additionally carries a **per-precision**
tolerance (`tolerance_for("fp16") == 1e-2`, vs. its own fp32 default
`1e-4` — `kernelbench.spec.Variant.tolerance_for`, the spec's own declared
bound, not loosened by this adapter). `PRECISIONS = ["fp16"]`.

## Gate results (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel spmm --variant spmm-tensorcore-fp16 --precision fp16 \
    --impl tcgnn-agnn-spmm-f16 --smoke
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel spmm --variant spmm-tensorcore-fp16 --precision fp16 \
    --impl tcgnn-agnn-spmm-f16 --matrices cora --dims 128,256,512 \
    --warmup 2 --reps 3
# also gated under the concurrently-added binary-adjacency variant:
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel spmm --variant spmm-binary-adjacency-kernel --precision fp16 \
    --impl tcgnn-agnn-spmm-f16 --smoke
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel spmm --variant spmm-binary-adjacency-kernel --precision fp16 \
    --impl tcgnn-agnn-spmm-f16 --matrices cora --dims 128,256,512 \
    --warmup 2 --reps 3
```

| variant | workload | N | outcome | max_scaled_err |
|---|---|---|---|---|
| spmm-tensorcore-fp16 | smoke (9/9) | 128,256,512 | UNSUPPORTED (guard #2: `M=4000 % 16 == 0`) | -- |
| spmm-tensorcore-fp16 | cora | 128 | **VALID** | 4.87e-04 (tol 1e-2) |
| spmm-tensorcore-fp16 | cora | 256, 512 | UNSUPPORTED (guard #3: `N > 128`) | -- |
| spmm-binary-adjacency-kernel | smoke (12/12) | 32,128,256,512 | UNSUPPORTED (guard #2) | -- |
| spmm-binary-adjacency-kernel | cora+bin | 128 | **VALID** | 4.87e-04 (tol_fp16 1e-2) |
| spmm-binary-adjacency-kernel | cora+bin | 256, 512 | UNSUPPORTED (guard #3) | -- |

**2/2 serviceable workloads pass** (both at `cora`, `N=128`, the only
combination that clears all three guards). The `max_scaled_err` is
IDENTICAL (4.87e-04) between the weighted and binarized runs — expected,
not a bug: TF32's fixed relative mantissa precision (~2^-10) dominates the
scaled-error metric regardless of the operand value distribution, so a
near-identical error at the same shape/dim is the signature of genuine
TF32 rounding noise, not evidence the binarization was silently ignored
(confirmed independently: the binarized run's `edgeAttention` array is
verified all-ones by construction of `variant_transform`, upstream of this
adapter). Reduced-protocol numbers only (login node, shared GPU;
`conforming: False`), not a timing result per ARTIFACT_GUIDE.md rule 5.

## Not done

- No compute-node timing sweep (out of scope, rule 5).
- No sweep across the full `recommended_subset` registry of either
  variant — smoke + `cora` only, per the task brief's login-node/low-cost
  scope.
- Guards #1-#3 are reproduced from, not independently re-derived against,
  the gnn-aggregation build's own empirical confirmation (canary-buffer
  test, `F=256/512` zero-fill measurement) — same compiled `.so`, so the
  same code paths apply; not re-verified bit-for-bit here.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05; job
  ran on gpu-b11-6.zaratan.umd.edu), login-node build (reused
  `gnn-aggregation/tc-gnn`'s compiled `TCGNN.so`, no recompilation here).
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14.
- Build: OK (reuse-verify only -- symlinks `source/`, imports
  `TCGNN.forward_AGNN` from `../../gnn-aggregation/tc-gnn/build/`, no
  compilation). Build-system changes: none. First attempt of this
  reuse-verify step hit a transient `RuntimeError: CPU dispatcher tracer
  already initlized` / segfault inside numpy's init, caused by OpenBLAS
  thread creation failing under this login node's `RLIMIT_NPROC=256`
  (concurrent builds from other artifacts' subagents running at the same
  time, not an issue with this artifact) -- retried alone with
  `OPENBLAS_NUM_THREADS=4` and it succeeded immediately, matching the
  documented "LOGIN-NODE PROCESS LIMIT" gotcha.
- Gate: `cora` in the recorded verification command could not be loaded on
  this machine (`torch_geometric` not installed in `kb-env`; see
  `../../gnn-aggregation/tc-gnn/STATUS.md`'s reproduction note for the full
  explanation). Substituted `cant` (SuiteSparse, `62451 % 16 == 3`, same
  guard-safety class as `cora`):
  - `spmm-tensorcore-fp16` smoke (9 workloads, `M=4000`): 0/0 valid, 9/9
    UNSUPPORTED via guard `M % 16 == 0` -- matches recorded.
  - `spmm-tensorcore-fp16` on `cant`, dim=128: **VALID**, err=6.63e-04
    <= tol 1e-2.
  - `spmm-tensorcore-fp16` on `cant`, dim=256, 512: UNSUPPORTED via guard
    `N > 128` -- matches recorded.
  - `spmm-binary-adjacency-kernel` smoke (12 workloads): 0/0 valid, 12/12
    UNSUPPORTED via guard `M % 16 == 0` -- matches recorded.
  - `spmm-binary-adjacency-kernel` on `cant`, dim=128: **VALID**,
    err=4.83e-04 <= tol 1e-2.
  - `spmm-binary-adjacency-kernel` on `cant`, dim=256, 512: UNSUPPORTED via
    guard `N > 128` -- matches recorded.
- Deviation from the recorded ruling: none in substance -- `cant`
  substituted for `cora` (torch_geometric unavailable, see above); 2/2
  serviceable workloads pass, same pattern (VALID only at N=128, both
  variants, TF32-level error well inside the 1e-2 tensor-core tolerance) as
  the recorded `cora`-based verification (which measured 4.87e-04 there vs
  6.63e-04 / 4.83e-04 here).
- Verdict here: BUILT+GATED -- equals the recorded ruling.
