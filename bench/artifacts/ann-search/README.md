# ann-search track — artifact integration summary

Four candidates attempted (all 4 GPU-eligible candidates in the track — see
"Newest-three verification" below):

- **clover** (ICS'25, `conf/ics/KamelYC25`) — **BUILT+GATED**,
  `exact-spatial-knn-kernel`, exact top-k distance-multiset match, err
  7.98e-08..1.31e-07 (<= 1e-4), CLOVER's own "hubs" spatio-graph exact kNN
  method (D=3 only, self-query-only kernel — see `clover/STATUS.md` for the
  structural finding and its consequences). See `clover/STATUS.md`.
- **pathweaver** (ATC'25, `conf/usenix/KimPNHKLL25`) — **BUILT+GATED**,
  `ann-highdim-recall-qps-pareto`, recall@10 = 0.9906 (D=32) / 0.9766
  (D=64), both above the spec's 0.9 floor, PathWeaver's own `search()`
  graph-traversal kernel run in its simplest single-pass configuration
  (no ghost-stage seeding, no sign-bit pruning) over an independently-built
  exact k-NN graph substituting for CAGRA's own graph construction (RAFT
  out of scope — a disclosed, fairness-relevant substitution, see
  `pathweaver/STATUS.md`).
- **rtnn** (PPoPP'22, `conf/ppopp/000122`) — **BUILD-FAILED**, a genuine
  Thrust/CUB versioned-ABI-namespace incompatibility between this
  2022-era artifact and CUDA 12.9's much newer bundled Thrust, surviving
  past OptiX configuration and CUDA/PTX compilation, failing only at the
  final link step. See `rtnn/STATUS.md`.
- **xhd** (ICS'26, `conf/ics/GengYLWZ26`) — **SKIPPED**, scope mismatch
  (its end-to-end deliverable, a Hausdorff distance VALUE, is explicitly
  out of `exact-spatial-knn-kernel`'s scope per the spec's own fairness
  ruling) compounded by a build (RMM `FetchContent` + vcpkg/ITK)
  disproportionate to extracting its one in-scope k-NN sub-routine — a
  cheap skip, no build attempt. See `xhd/STATUS.md`.

`clover` and `xhd` (and, had it built, `rtnn`) target
`exact-spatial-knn-kernel`; `pathweaver` is the **only** candidate in this
track targeting `ann-highdim-recall-qps-pareto` (the high-dimensional
SIFT/DEEP/SPACEV-family approximate-search variant) — see below for why
the track's other two high-dim candidates were never attempted.

## Newest-three verification (against `benchspecs/ann-search/survey.md`)

`survey.md` surveys all 7 papers in the track's input set, with publication
years/venues: **X-HD** (ICS'26), **DRIM-ANN** (SC'25), **UpANNS** (SC'25),
**PathWeaver** (ATC'25), **CLOVER** (ICS'25), **FANNS** (SC'23), **RTNN**
(PPoPP'22). By venue chronology alone, the three newest are {X-HD,
DRIM-ANN, UpANNS} — **not** what was attempted above (which includes
CLOVER 2025-ICS and RTNN 2022 instead of DRIM-ANN/UpANNS).

**DRIM-ANN and UpANNS were not attempted, and correctly so**: both are
DRAM-PIM (UPMEM) systems, not NVIDIA GPU implementations of the paper's own
contribution. `survey.md`'s own entries are explicit: DRIM-ANN's hardware
is "CPU Intel Xeon Gold 5218 ..., GPU NVIDIA A100 PCIe 80GB HBM2e, **PIM
UPMEM up to 2,560 DPUs / 159GB**" — the A100 appears only as a GPU-IVF-PQ
*baseline* the paper compares against (Faiss-GPU), never as where DRIM-ANN's
own novel kernel runs; UpANNS's hardware is "7x UPMEM PIM modules, 128GB
memory, Faiss 1.8.0 (**for the CPU/GPU baseline comparison**)" — same
pattern, GPU is a baseline, not the paper's own contribution. Per
`ARTIFACT_GUIDE.md`'s scope ruling (2026-08-07, "Integration targets
NVIDIA GPU, single-card implementations only... multi-GPU/distributed-only... 
artifacts are SKIPPED with a one-line reason"), a PIM-DPU architecture is
squarely out of scope on the same footing as the gemv track's QIGen
(CPU-only) precedent — this note in this README constitutes that one-line
reason; no artifact directories were created for either.

**FANNS** (SC'23) is also a high-dim candidate but was not attempted: its
own novel contribution is an FPGA auto-design framework (Xilinx Alveo
U55c, Vitis HLS) that searches a hardware+algorithm design space to meet a
user-given recall target; per `survey.md`, its GPU numbers come from
**Faiss-GPU as a baseline** (V100), not from a kernel FANNS itself wrote —
there is no separable "FANNS's own GPU kernel" to wrap, the same category
of exclusion as GPU-DPF was for the gemv track's non-separable-kernel
reasoning, just on the hardware-target axis instead.

Restricting to candidates whose OWN novel contribution is an NVIDIA GPU
kernel — {X-HD, PathWeaver, CLOVER, RTNN} — **all four** were attempted in
this pass, exceeding rather than merely meeting the "three newest" floor,
because the track has only four such candidates total (the other three,
DRIM-ANN/UpANNS/FANNS, are excluded by platform/scope as shown above, not
by recency). Within those four, chronology is X-HD (2026) > PathWeaver
(ATC'25) > CLOVER (ICS'25) > RTNN (PPoPP'22) — RTNN is the oldest and would
have been the one candidate droppable under a strict "three newest" rule,
but was attempted anyway (and yielded a genuine, actionable BUILD-FAILED
finding) since all four were in scope and none was disproportionately
expensive to attempt except X-HD, which was cheaply skipped on scope
grounds alone (see its STATUS.md) rather than for being older.
