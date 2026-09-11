# connected-components — artifact integrations

Domain module: `kernelbench/domains/graph.py`. Spec:
`benchspecs/connected-components/spec.yaml`. Four variants across THREE
genuinely distinct sub-problems: biconnected components (BCC, undirected,
2 variants), strongly connected components (SCC, directed, 1 variant), and
binary-image 2D/3D connected-component LABELING (CCL, regular grid — a
related-but-distinct primitive, NOT a fourth data point on the same
sparse-graph "connectivity kernel" curve; see the spec's own
`notes_on_fairness`).

## Baseline selection rule (user decision, 2026-09-05)

Baselines are chosen from `kernel-papers/output/baseline_selection.md`,
produced by `select_baselines.py` from the per-(paper, track) ratings in
`output/kernel_centrality.json`:

1. **kernel centrality** `core` (the kernel IS the paper's headline
   contribution) > `component` (part of a bigger pipeline) — `tangential`
   is never a baseline;
2. **regime match** with the track spec's inputs: `matches` > `partial` >
   `mismatch`;
3. **single-NVIDIA-GPU path** required (current scope);
4. **recency** only as the tiebreak; up to 5 per track.

Already-integrated adapters the rule would not have picked stay in the
registry as competitors (still run under the same gate) but are labelled
`component`/off-regime here and are not the "human SOTA" reference in
Phase 3.

## Directories

| dir | paper (PAPER_KEY) | centrality | regime | sub-problem | status |
|---|---|---|---|---|---|
| `fast-bcc` | GPU BCC, IPDPS'26 (`conf/ipps/SahuARKB26`) | core | matches | BCC | SKIPPED (no clean CC-only entry point — see its STATUS.md) |
| `ecl-scc` | ECL-SCC, SC'23 (`conf/sc/AlabandiSBB23`) | core | matches | SCC | BUILT+GATED |
| `yacclab` | YACCLAB (BUF), TPDS'20 (`journals/tpds/AllegrettiBG20`) | core | matches | image-CCL | BUILT+GATED |

**No centrality-rating correction is being reported.** The spec is NOT
graph-only for this track — it has a dedicated `cc-image-ccl-2d3d` variant
built specifically around this paper's own richer test taxonomy, so
`journals/tpds/AllegrettiBG20`'s "regime: matches" verdict in
`output/kernel_centrality.json` is consistent (it matches ONE of the
track's 4 variants, the one built for it), even though it would not match
the other 3 (sparse-graph) variants. See `yacclab/STATUS.md`'s "Regime:
images, not graphs" section for the full reasoning trail.

## New this pass

- **`yacclab`**: BUF (Block-based Union-Find), 2D 8-connectivity, wrapped
  against a THIN SHIM (not real OpenCV — the CUDA-enabled `cudafeatures2d`
  contrib module OpenCV needs is not available as a prebuilt wheel and
  building it from source was out of scope; per ARTIFACT_GUIDE's own
  fallback for exactly this case). The upstream kernel file
  (`labeling_allegretti_2019_BUF.cu`) is `#include`d verbatim, unmodified
  — the shim only supplies the small `PtrStepSz`/`GpuMat`/benchmark-base-
  class types the file itself references.

  **Real bug found (in the shim, fixed there, not in the kernel)**: the
  shim's first `GpuMat::create()` used a flat, unpadded device allocation,
  which produced a genuine 16-bit-alignment device fault the moment an
  odd-width image was tried — the kernel's own paired-pixel read
  (`reinterpret_cast<int16_t*>`) assumes real OpenCV's `cudaMallocPitch`-
  guaranteed row alignment, which the shim was missing. Fixed by switching
  the shim to `cudaMallocPitch`; confirmed via `yacclab/debug_buf.py`
  (5/5 random-shape trials, exact match against an independent scipy
  `ndimage.label` reference).

  **`kernelbench/domains/graph.py` gained a small, additive `Image2D`
  workload type** (binary 2D images, an independent 8-connectivity
  union-find reference reusing the module's existing
  `_union_find_components` helper) specifically to give this variant a
  workload to run against — every other kernel/variant's behavior in that
  module is unchanged (verified: `bfs` and the other 3
  connected-components variants still run against the original graph
  smoke set). See `yacclab/STATUS.md`'s "Domain change" section.

  BUILT+GATED, exact match on 3/3 smoke workloads.
