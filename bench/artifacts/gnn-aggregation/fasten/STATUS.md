# FASTEN — gnn-aggregation — STATUS

**Outcome: SKIPPED — the kernel is segmented dense-dense GEMM, not sparse
adjacency aggregation; no adjacency/CSR operand exists anywhere in the API.**

Paper: "FASTEN: Fast GPU-accelerated Segmented Matrix Multiplication for
Heterogeneous Graph Neural Networks" (ICS'24). `PAPER_KEY = conf/ics/0001SLFY024`
(confirmed via `output/included.json`, matched by `artifact_url`).
Repo: `https://github.com/Deep-Learning-Profiling-Tools/fasten`, commit
`784bd907aa3685770fb14608fbbd620e21261937` (2024-04-23; `git clone --depth 1`
into `./source/`).

## Why SKIPPED (evidence)

This track's `operation` (`benchspecs/gnn-aggregation/spec.yaml`) is
`H'[V,F] = A_hat[V,V](sparse, CSR) @ H[V,F](dense)` — a sparse adjacency
times a dense feature matrix. FASTEN's actual operator, read from its own
source, is a **segmented (grouped-by-type) dense GEMM** with no sparse
adjacency operand at all, at any boundary — not even the "single relation"
degenerate case the task's own guidance anticipated as a possible mapping.

- `fasten/ops.py`'s public entry point,
  `fasten_segment_matmul(input, other, tensor_slice, engine)`, dispatches to
  `execute_engine(..., op_name='segment_matmul_forward')`. `input`/`other`
  are both **dense** tensors; there is no sparse/CSR argument anywhere in
  `FastenSegmentMatmul.forward`'s signature.
- `fasten/tensor_slice.py`'s `TensorSlice` — the only "graph-shaped" object
  in the API — is a `(type_index, type, start, end, next)` table of
  **contiguous row RANGES** grouping `input`'s rows by relation type; it
  carries no edge list, no neighbor structure, no adjacency matrix of any
  kind. It is a batching/grouping index, not a graph.
- `test/test_ops.py::test_perf` (the artifact's own benchmark, confirming
  the real operand shapes) constructs
  `data = torch.randn((M, K))` and `other = torch.randn((T, K, K))`
  (`T` = number of relation types, each a **dense** `K x K` weight matrix),
  then calls `ops.fasten_segment_matmul(data, other, tensor_slice, ...)`.
  The computed output is, per row `i`, `data[i] @ other[type_of(i)]` — this
  is the R-GCN/HGT "per-relation-type linear transform" step (`W_r @ x`),
  the stage that happens **before/instead of** neighbor aggregation in a
  heterogeneous GNN layer, not the aggregation (scatter/gather over edges)
  itself.
- Even the degenerate "single relation, one segment covering every row"
  case the task's guidance suggested as a possible mapping ("a single-relation
  call equals plain aggregation") reduces `fasten_segment_matmul` to
  `output = input @ other[0]` — a **plain dense GEMM** (`gemm` kernel shape,
  already covered by this repo's own `gemm` track), not `A_hat @ X` with any
  sparse operand. There is no boundary at which FASTEN's kernel becomes a
  sparse-times-dense product.

This matches `benchspecs/gnn-aggregation/spec.yaml`'s own
`notes_on_fairness`, written during the Phase-2 survey before this
integration task began:

> "FASTEN's segmented matmul for heterogeneous GNNs ... [is a] structurally
> different problem (grouped-by-relation-type GEMM ...) from the
> single-adjacency-matrix, single-device aggregation this spec's four
> variants target. Neither is folded in; see open_questions."

No build attempted (cheap skip, per ARTIFACT_GUIDE.md's scope ruling: a
genuine kernel-identity mismatch, not a platform/toolchain issue).

## Provenance

- Artifact commit: `784bd907aa3685770fb14608fbbd620e21261937`
- Source kept at `./source/` (`git clone --depth 1`) for provenance; no
  `build.sh`/`adapter.py` (nothing to wrap).
