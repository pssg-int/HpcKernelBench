# gnn-aggregation — paper artifacts

Every paper artifact integrated (or evaluated and declined) for this
track's kernel: `H'[V,F] = A_hat[V,V](sparse, CSR) @ H[V,F](dense
row-major)`, `A_hat` either the raw adjacency or its GCN symmetric
normalization `D^-1/2(A+I)D^-1/2` (`benchspecs/gnn-aggregation/spec.yaml`).

## Baseline selection rule (2026-09-05)

Baselines are chosen from `kernel-papers/output/baseline_selection.md`
(produced by `select_baselines.py` from the per-(paper, track) ratings in
`output/kernel_centrality.json`), replacing the retired "newest 3
open-source artifacts per track" rule:

1. **kernel centrality** `core` (the kernel IS the paper's headline
   contribution, evaluated at kernel level) > `component` (a part of a
   bigger system/pipeline) — `tangential` is never a baseline;
2. **regime match** with the track spec's inputs: `matches` > `partial` >
   `mismatch`;
3. **single-NVIDIA-GPU path** required (current scope);
4. **recency** only as the tiebreak; up to 5 per track.

`kernel_centrality.json`'s `gnn-aggregation` entries rated 5 papers
`core`+`matches` (TC-GNN, TLPGNN, GE-SpMM, EPPGCN, FeatGraph — exactly the
5 this directory's non-already-integrated adapters cover, in the order
given by the baseline-selection task) plus StraGCN at `core`+`partial`
(kept in the registry, outside the top-5 cutoff) and two clear non-fits
(MaxK-GNN `core`+`partial` structurally off-regime; FASTEN `tangential`
+`mismatch`, already-integrated adapters that the rule would not have
picked, retained as competitors per ARTIFACT_GUIDE.md but not the "human
SOTA" reference).

## Artifacts

| dir | paper | centrality / regime | outcome |
|---|---|---|---|
| `tc-gnn/` | TC-GNN (ATC'23) | core / matches | **BUILT** — gate FAILS at measured tf32 error (~7e-4 vs 1e-4 tol); two workload-crashing kernel bugs found & guarded |
| `tlpgnn/` | TLPGNN (HPDC'22) | core / matches | **BUILT** — gate FAILS by construction (kernel computes row-mean D^-1 A, not this track's D^-1/2(A+I)D^-1/2 reference) |
| `ge-spmm/` | GE-SpMM (SC'20) | core / matches | **BUILT+GATED** — 16/16 valid, err ~2e-7 |
| `eppgcn/` | EPPGCN (TPDS'22) | core / matches | **SKIPPED** — no plain (unfused) forward-aggregation entry point; every sparse-touching pybind call bakes in a dense GEMM |
| `featgraph/` | FeatGraph (SC'20) | core / matches | **SKIPPED** — TVM installs via pip --target, but FeatGraph's own kernel modules need TVM v0.7 (2020); only 0.25.x/0.26.0 are pip-installable, ~6 years of breaking API changes later |
| `stragcn/` | StraGCN (SC'25) | core / partial | **BUILT+GATED** (pre-existing) — outside the top-5 cutoff, kept as a competitor |
| `maxk-gnn/` | MaxK-GNN (ASPLOS'24) | core / partial | **SKIPPED** (pre-existing) — no Python binding for the differentiated kernel anywhere in the repo |
| `fasten/` | FASTEN (ICS'24) | tangential / mismatch | **SKIPPED** (pre-existing) — segmented dense GEMM, no sparse adjacency operand at all |

See each `<dir>/STATUS.md` for full evidence, patches, gate numbers, and
provenance. `tc-gnn`, `tlpgnn`, `ge-spmm`, `eppgcn`, `featgraph` were
integrated in this pass (2026-09-05); `stragcn`/`maxk-gnn`/`fasten` predate
it and are summarized here for completeness.

## Harness fix made during this integration

`kernelbench/domains/sparse.py`'s `smoke_workloads(kernel=...)` now takes
the absolute value of the synthetic smoke matrices' edge weights
specifically for `gnn-aggregation` (every other kernel sharing this smoke
set — spmv/spmm/sddmm/spgemm/sptrsv — is unaffected; SIGNED weights are
fine for those). `matrices.synthetic()`'s default SIGNED `U(-1,1)` weights
made a meaningful fraction of rows' `D^-1/2(A+I)D^-1/2` degree sum land at
or near zero-magnitude; the CPU reference correctly zeroes those rows
(masking `deg<=0`), giving a gate `scale = |A_hat|@|X|` that is exactly `0`
there — any artifact whose own kernel algebra does not preserve exact-zero
the same way a direct multiply does (confirmed for real: StraGCN's
Strassen-decomposed SpMM leaves a `~1e-8` rounding residue at such rows,
see `stragcn/STATUS.md`) divides that residue by a `~0` floor and reports
`max_scaled_err` in the `1e290+` range, regardless of the artifact's actual
accuracy. Real graphs in `recommended_subset` are nonnegative by
construction and never hit this degeneracy. This is a one-line change (only
in `smoke_workloads`'s `gnn-aggregation` branch, gated on `kernel ==
"gnn-aggregation"`); no other kernel's smoke behavior changed (verified:
`./smoke_all.sh` stayed 35/35 green before and after).
