# Survey: matrix-function-evaluation

SINGLETON track — 1 paper (`conf/sc/LassSKP20`, the submatrix method for
CP2K). Surveyed via arXiv fulltext (ar5iv mirror of
`arxiv.org/abs/2004.10811`, since the plain `arxiv.org/abs` and
`arxiv.org/html` pages returned abstract-only) plus the artifact repo
(`github.com/pc2/SubmatrixMethod`) root listing and README.

## Submatrix method for CP2K (conf/sc/LassSKP20)

- **Matrix function evaluated**: the density-matrix **sign function**, via
  Löwdin symmetric orthogonalization, needed inside CP2K's density-functional
  theory (DFT) electronic-structure pipeline. (Not, as the repo name might
  suggest, a generic inverse-p-th-root routine in this paper's own
  evaluation — the repo's README cites a separate 2018 PASC paper for the
  inverse-p-th-root application; the SC'20 paper surveyed here targets the
  sign function specifically.)
- **Core idea benchmarked**: convert operations on a large, sparse,
  distributed matrix into independent computations on many small, nearly
  dense **submatrices**, extracted so each fits comfortably in cache /
  on a single accelerator — trading exactness for full utilization of dense
  floating-point hardware (CPU vector units, GPU tensor cores, FPGA
  pipelines).
- **Workloads/inputs**: CP2K's own standard liquid-water DFT benchmark
  family — a fixed 32-H₂O-molecule unit cell repeated `NREP` times per
  dimension. Paper's own literal sweep: `NREP` from 2 to 8, i.e. **768 to
  49,152 atoms**. Basis sets: `SZV-MOLOPT-SR-GTH` (primary), `DZVP-MOLOPT-SR-GTH`
  (used for a larger-basis-set comparison point).
- **Linear scaling claim**: submatrix size is independent of overall system
  size once the system exceeds roughly 200 water molecules — the paper's own
  stated condition for the method's linear-scaling behavior to kick in.
  Exact numeric submatrix dimension not recovered from the fetched excerpt.
- **Accuracy / correctness**: results compared against a reference computed
  with truncation threshold `eps_filter = 1e-15`; the method's own tunable
  `eps_filter` swept from `1e-5` to `1e-12` (paper's Figure 7), reporting
  energy error vs. this reference across the sweep. Paper's own conclusion:
  the submatrix approximation "does not negatively impact the results too
  much" — exact numeric error bound not extracted from the fetched excerpt
  (flagged in spec.yaml open_questions).
- **Hardware / performance**:
  - CPU: Intel Xeon Gold 6148 "Skylake", 40 cores/node, scaled to 8 nodes
    (320 cores); submatrix method run at **1 thread per MPI rank**.
  - GPU: NVIDIA RTX 2080 Ti, comparing FP16/FP32/FP64 tensor-core paths;
    headline number **35.2 TFLOP/s at 250W** in FP16 for the sign-function
    kernel.
  - FPGA: Intel Stratix 10 GX 2800, practical ceiling ~3.4 TFLOP/s.
  - Strong scaling: **83% parallel efficiency** scaling from 2 to 8 nodes.
- **Baseline**: 2nd-order Newton-Schulz iteration on sparse DBCSR matrices —
  CP2K's own existing production method — but run with a **different
  parallel mapping** (8 MPI ranks/node × 5 threads/rank) than the submatrix
  method's own 1 thread/rank. This is an asymmetry in the paper's own
  reported comparison, not something this survey can resolve without
  re-running both at matched configurations.

Source: ar5iv fulltext of arXiv:2004.10811; repo `pc2/SubmatrixMethod` root
listing + README.md (GitHub API).

## Divergences

Not applicable — singleton track, one paper. The one internal tension
flagged in spec.yaml's `notes_on_fairness` is the paper's own asymmetric
rank/thread mapping between the submatrix method and its Newton-Schulz
baseline, which this survey reports as-is (it is the paper's own reported
configuration) rather than silently treating it as an equalized comparison.
