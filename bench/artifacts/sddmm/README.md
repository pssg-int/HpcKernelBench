# sddmm track — artifact adapters

Sampled Dense-Dense Matrix Multiplication: `P[i,j] = S[i,j] * dot(A[i,:], B[j,:])`
for `(i,j)` in `nnz(S)`. Spec: `benchspecs/sddmm/spec.yaml`. Domain module:
`kernelbench/domains/sparse.py` (kernel `"sddmm"`).

## Baseline selection rule (user decision, 2026-09-05)

Baselines are chosen from `kernel-papers/output/baseline_selection.md`
(produced by `select_baselines.py` from the per-(paper, track) ratings in
`output/kernel_centrality.json`), not "newest N artifacts":

1. **kernel centrality** `core` (the kernel IS the paper's headline
   contribution, evaluated at kernel level) > `component` (part of a
   bigger system/pipeline/codegen output) — `tangential` is never a
   baseline;
2. **regime match** with the spec's inputs: `matches` > `partial` >
   `mismatch`;
3. **single-NVIDIA-GPU path** required (current scope);
4. **recency** only as the tiebreak; up to 5 per track.

`output/baseline_selection.md`'s sddmm section names 4 papers to integrate
(`fused3s`, `flashsparse`, `rode`, `hp-spmm-sddmm`, `magicube` — 5 listed,
one already `[integrated]`) plus `rassm`, demoted from an earlier
"newest 3" pass to a labelled competitor (`component`, regime matches, but
**no single-GPU path** — RASSM is CPU-only). Already-integrated adapters
the revised rule would not have picked (rassm) stay in the registry as
competitors under the same gate, labelled `component`/off-regime here and
in their own STATUS.md, and are not the "human SOTA" reference for Phase 3.

## Artifacts

| dir | paper (PAPER_KEY) | centrality / regime | outcome |
|---|---|---|---|
| `fused3s` | Fused3S, ICS'25 (`conf/ics/LiC25`) | core / matches | **BUILT+GATED** — wraps the `f3s_1tb1tcb` ablation kernel's isolated SDDMM stage (fp16 tensor-core, `applySoftmax=False`); gated under `sddmm-tensorcore-blocked-fp16`, `max_scaled_err=1.70e-04` (tol `1e-2`) on `cant` @ K=128. Square-matrix constraint only. |
| `rode` | RoDe, PPoPP'24 (`conf/ppopp/PangFQZL24`) | core / matches | **BUILT+GATED** — wraps `RoDeSDDMM_n32`/`n128` (fp32 CSR), `source/` symlinked read-only from `../../spmm/rode/source`. 6/6 smoke + 2/2 real-matrix (`cant`) valid under `sddmm-csr-kernel-f32`, err ~1e-7. Found and fixed a real bug in this integration's own preprocessing wrapper (wrong `SegmentLength` for K=128 — RoDe's own eval drivers use different constants per dense-width variant; the compiled kernel itself is correct). K in {32,128} only. |
| `flashsparse` | FlashSparse, PPoPP'25 (`conf/ppopp/ShiLXFWW25`) | core / matches | **BUILT, gate INCOMPLETE** — wraps `FS_SDDMM.forward_gen_fp16_gnn` (fp16, general two-operand SDDMM), `source/` symlinked read-only from `../../spmm/flashsparse/source`. Verified the task's flagged `values=1.0`-preprocessing concern is a non-issue (the kernel never consumes edge weights from any preprocessing path, weighting is applied in Python post-hoc, same as fused3s). Derived and independently verified the output tensor-core layout for full 16-wide tiles and small residue tiles; found and evidenced (via differential testing against the compiled kernel) a genuine DATA-LOSS bug in the kernel's own `Store()` pointer arithmetic for residue tiles in `[8,16)` nonzeros — affects most real inputs, so the fp16 gate cannot be completed responsibly. Not a GPU-arch issue (builds and runs on this A100). |
| `hp-spmm-sddmm` | HP-SpMM-SDDMM, IPDPS'23 (`conf/ipps/FanWC23`) | core / matches | **SKIPPED — no SDDMM code** (rule 7). Verified: no `sddmm/` directory anywhere in the cloned snapshot, README claims "FP32 SpMM implementations" only, and the only 2 "sddmm" string hits in the whole source tree are 6 unreferenced struct fields in `spmm_common.h` (vestigial, never wired to any kernel). Confirms `benchspecs/sddmm/spec.yaml`'s own `open_questions` suspicion. The repo's real SpMM kernel is a spmm-track candidate instead. |
| `magicube` | Magicube, SC'22 (`conf/sc/LiOH22`) | core / matches | **BUILT, gate not attempted** — `wmmaSddmm_16b` (16-bit fixed-point quantized WMMA kernel, packed `int32` operands + `int32` accumulator, NOT IEEE fp16) compiles and links cleanly (`-arch=sm_80`, no Hopper features — confirmed not a DEFERRED-HARDWARE case). Found and fixed a real build-system issue (artifact's `-std=c++11` fails under this machine's GCC 14; `-std=c++17` fixes it, no kernel change). Correctness gate not completed: the artifact's own packed-16-bit-subword convention is ambiguous even in its own source (its host reference treats sub-words as *unsigned*, but its own test-data generator produces meaningless filler, not real quantized values) and a quick empirical probe of the compiled kernel didn't resolve it within budget — flagged for follow-up rather than shipping a guessed dequantization. 4-bit/8-bit paths explicitly out of scope per this task's instructions (a different, non-fp16-adjacent precision class). |
| `rassm` | RASSM, ASPLOS'26 (`conf/asplos/JainGC25`) | component / matches, **no single-GPU path** | **BUILT+GATED** (pre-existing, this task did not touch it) — CPU-only adaptive-tiling SDDMM; kept in the registry as a competitor per the scope ruling (CPU-only artifacts already integrated stay as backups), not a core-rule pick. |
| `insum` | Insum, ASPLOS'26 (`conf/asplos/WonAAE26`) | component / partial | **SKIPPED (pre-existing, not touched)** — documented in its own STATUS.md as a single unrelated SpMM-COO Triton-codegen demo, no SDDMM benchmark code recoverable (ASPLOS'26 had not yet occurred at survey time). No `adapter.py`. |
| `tc-gnn` | TC-GNN, USENIX ATC'23 (`conf/usenix/WangFWHD23`) | core / partial | **BUILT+GATED (reused build)** — `source/`/compiled `.so` REUSED from `../../gnn-aggregation/tc-gnn/` (no second clone or compile); wraps `TCGNN.forward_ef` (tf32 WMMA self-SDDMM, `X@X^T` sampled at the pattern) via a bipartite double-cover node-doubling embedding (adapter-side only) to bridge its single-operand kernel onto this track's two-independent-operand reference. 2/2 valid on `cora` (K=32,128), `max_scaled_err` 2.67e-04/1.19e-04 (tol `1e-2`) under `sddmm-tensorcore-blocked-fp16`; every `--smoke` workload UNSUPPORTED (`M+N=8000 % 16 == 0`, the same shared `preprocess()` heap-corruption bug documented for the gnn-aggregation build). New finding: `K % 8 != 0` silently corrupts output via cross-row data bleed (confirmed empirically, not exercised by this track's own K in {32,128}). |
| `sputnik` | Sputnik (Gale et al.), SC'20 (`conf/sc/GaleZYE20`) | core / mismatch (own regime is DNN-pruned weights, not general SuiteSparse) | **BUILT+GATED, clean pass** — wraps `sputnik::CudaSddmm` (fp32, `source/` symlinked read-only from `../../spmm/sputnik/source`) under `sddmm-csr-kernel-f32`. Ruling: the paper-table's listed artifact URL (`luckylsk34/Sparse-Kernels`) is a stripped, renamed, single-commit fork of `google-research/sputnik` with its Apache-2.0 license notices removed — cloned the canonical repo instead (see `../../spmm/sputnik/STATUS.md` for the full diff evidence). 6/6 smoke + `cant` 1/1 + `cora` 1/1 valid, `max_scaled_err` 6.5e-8..1.4e-7 (tol 1e-4). Sputnik's own `CudaSddmm` has no separate values operand (pure pattern-restricted dot product); `run()` multiplies by the matrix's own values afterward, matching this codebase's existing `TorchSDDMM`/`CustomSDDMM` convention exactly. `SortedRowSwizzle` preprocessing ported byte-for-byte (Glog/Abseil-free port, same precedent as `rode`'s row-decomposition port) since `matrix_utils.cu.cc`'s other functions need Glog+Abseil (unvendored here) for unrelated test-data code. |

## Gate summary (login node, reduced protocol — not spec-conforming)

```
$PY -m kernelbench.runner --kernel sddmm --list
implementations: cpu=['scipy-csr-sddmm']
                 cuda=['custom-warp-csr-sddmm', 'torch-sampled-addmm-sddmm']
paper artifacts:
  ok flashsparse      flashsparse-sddmm
  ok fused3s          fused3s-1tb1tcb-sddmm
  -- insum                                     no adapter.py
  ok rassm            rassm-adaptive-tiled-sddmm
  ok rode             rode-sddmm
  ok sputnik          sputnik-sddmm-f32
  ok tc-gnn           tcgnn-ef-sddmm-f16
```

(`magicube`/`hp-spmm-sddmm` ship no `adapter.py` — see their STATUS.md;
`--list` correctly does not claim them as registered implementations.)

- `tcgnn-ef-sddmm-f16` (reused build, no rebuild): `sddmm-tensorcore-
  blocked-fp16 --precision fp16`, smoke 0/6 (all UNSUPPORTED, shared
  `num_nodes%16==0` guard) + `cora` 2/2 valid (K=32,128), err
  2.67e-04/1.19e-04. See `tc-gnn/STATUS.md` for the bipartite-embedding
  bridge and the new `K%8!=0` finding.

Real functional/gate checks performed this pass (all reduced-protocol,
non-conforming per ARTIFACT_GUIDE.md rule 5 — login node, no timing sweeps):

- `rode-sddmm`: `sddmm-csr-kernel-f32`, smoke 6/6 valid + `cant` 2/2 valid.
- `sputnik-sddmm-f32`: `sddmm-csr-kernel-f32`, smoke 6/6 valid + `cant` 1/1
  valid + `cora` 1/1 valid, err 6.5e-8..1.4e-7.
- `flashsparse-sddmm`: `sddmm-tensorcore-blocked-fp16`, `prepare()` raises a
  clearly-named `RuntimeError` (the documented kernel data-loss finding),
  not a silent failure or a crash.
- `fused3s-1tb1tcb-sddmm`: previously gated (see its own STATUS.md),
  unchanged by this task.
