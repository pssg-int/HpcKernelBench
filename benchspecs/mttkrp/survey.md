# MTTKRP track — evaluation-methodology survey

Surveyed 4/4 papers, all via fulltext (3 via arXiv PDF, 1 via OSTI PDF; two of
the four have no arXiv copy and were retrieved from OSTI/arXiv mirrors of the
peer-reviewed PDF).

## BLCO — Nguyen et al., "Efficient, Out-of-Memory Sparse MTTKRP on Massively
Parallel Architectures" (ICS'22), arXiv:2201.12523

Source: arXiv fulltext (`arxiv.org/pdf/2201.12523`), Sections 6.1–6.5.

- **Workloads**: 14 real-world sparse tensors from the FROSTT and HaTen2
  repositories, explicitly listed (Table 2): NIPS, Uber, Chicago, Vast-2015,
  DARPA, Enron, NELL-2, FB-M, Flickr, Delicious, NELL-1 (fit in GPU memory,
  3.1M–143.6M nnz) plus Amazon, Patents, Reddit (1.7B–4.7B nnz, explicitly
  treated as **out-of-memory** because they overflow device memory on
  competing frameworks). No stated inclusion threshold beyond "representative
  set… that cover a wide range of tensor properties" — this is a fixed,
  named list, not a filtered subset of a larger public suite.
- **Rank**: fixed R = 32 for MTTKRP ("as per prior work [MM-CSF]").
- **Mode sweep**: **all modes** — Figure 8 reports all-mode MTTKRP time;
  Figure 9 additionally reports **per-mode** speedup for every tensor mode
  individually (the paper's own motivation, Fig. 1, is that mode-specific
  formats vary 2–12× across modes for the same tensor, so all-mode reporting
  is treated as the fair basis for comparison).
- **Format conversion / preprocessing**: format construction (COO → BLCO)
  timed **separately** (Fig. 11–12, broken down into setup / initial
  linearization / sorting / re-encoding / blocking stages). Not included in
  per-iteration MTTKRP time. Authors report BLCO needs ~12 full (all-mode)
  MTTKRP iterations on average to amortize its own construction cost,
  vs. up to an order of magnitude more for competing GPU formats.
- **Timing protocol**: double-precision values, 64-bit integers; performance
  = **average over 25 iterations**. No explicit warmup step stated; timer
  not explicitly named (elapsed wall time via profiler-adjacent
  instrumentation) — Nsight Compute/Nsight Systems used for memory-metric
  collection, not for the primary timing.
- **Precision & correctness**: fp64 (double) values with 64-bit integer
  indices. No explicit numerical-tolerance correctness gate described in the
  fetched sections (framework compared for functional correctness is not
  detailed — treated as an **open question**).
- **Metric**: execution time (s), speedup vs. MM-CSF (geomean 2.12–2.6×
  across 3 GPU platforms, up to 33.35× per-mode), memory volume (GB, via
  `l1tex__t_bytes.sum`), memory throughput (TB/s).
- **Baselines**: MM-CSF (state-of-the-art mode-specific tree format,
  ported to DPC++ from CUDA), GenTen, F-COO.
- **Hardware**: Intel discrete GPU (single tile), NVIDIA A100, NVIDIA V100;
  host: dual-socket AMD EPYC 7662 (128 threads) for format construction.

## SySTeC — Patel, Ahrens, Amarasinghe, "SySTeC: A Symmetric Sparse Tensor
Compiler" (CGO'25), arXiv:2406.09266

Source: arXiv fulltext, Section 5 (Evaluation) + Artifact Appendix.

- **Workloads**: For MTTKRP specifically (unlike SSYMV/SYPRD/SSYRK, which use
  the Vuduc et al. SuiteSparse-derived matrix suite, Table 2, 29 named
  matrices), the paper states **no public database of symmetric tensors
  exists**, so MTTKRP is evaluated on **synthetic, uniformly-distributed
  symmetric random sparse tensors generated via an Erdős–Rényi
  distribution**, sweeping sparsity and numerical rank (Fig. 11: rank/
  sparsity pairs (0.1,10), (0.0001,10), (0.1,~40–500), (0.0001,~40–500),
  varied per dimensionality). Dense factor matrix B is also randomly
  generated; "numerical rank" = number of columns of B.
- **Rank**: swept, not fixed — from 10 up to 500 depending on the
  experiment (3D/4D/5D MTTKRP each swept over 4 sparsity×rank points).
- **Mode sweep**: kernel is inherently **mode-agnostic / single-mode**: for
  a *fully symmetric* input tensor, all N mode-n MTTKRP computations are
  mathematically identical, so only one (canonical) mode needs to be
  computed and reused. This is a structural property of the symmetric
  sub-problem, not an evaluation shortcut — it does not carry over to
  general (asymmetric) MTTKRP.
- **Format conversion / preprocessing**: explicitly **excluded** — "the time
  to rearrange data before or after each kernel is not included in the
  timings, including transposition or replicating the output."
- **Timing protocol**: **single core**, single-threaded, 12-core dual-socket
  Intel Xeon E5-2680 v3 @ 2.50GHz, 128GB RAM. Timings = **minimum of 10,000
  runs or 5 seconds of measurement, whichever comes first** (min-of-many, not
  median). Julia v1.10, Finch v0.6.22.
- **Precision & correctness**: not explicitly stated (Julia/Finch default is
  double / Float64); no explicit tolerance given, but artifact validates
  numerical output against comparison methods programmatically.
- **Metric**: speedup normalized to naive Finch baseline (bar charts); for
  3-/4-/5-D MTTKRP the paper reports theoretical max speedups (2×, 6×, 24×
  from 1/n! access reduction × 1/(n-1)! compute reduction) vs. measured
  maximal speedups of 3.38×, 7.35×, 29.8×.
- **Baselines**: naive Finch (unoptimized dense-looking sparse code),
  TACO (column-major), SPLATT (MTTKRP-specific library).
- **Format**: A is CSF (`Dense(Sparse(Sparse(...)))`), B/C dense.

## SymProp — Li, Shivakumar, Li, Kannan, "SymProp: Scaling Sparse Symmetric
Tucker Decomposition via Symmetry Propagation" (IPDPS'25), OSTI PDF (OSTI
ID 3002148; no arXiv copy)

Note: SymProp targets S³TTMc / S³TTMcTC (symmetric Tensor-Times-Matrix-chain,
the HOOI/HOQRI Tucker-decomposition kernels), not literal MTTKRP, but is
grouped in this track's input set as a closely related
symmetric-tensor-times-dense-matrix(-chain) kernel family.

- **Workloads**: 4 synthetic tensors reused from prior CSS-format work
  (6D/L6, 7D/L7, 10D/L10, 12D/H12 — order, dimension, and UNNZ given
  explicitly in Table III) **plus** 5 real-world tensors built from published
  hypergraph datasets (contact-school, trivago-clicks, walmart-trips,
  stackoverflow, amazon-reviews — order 5 to 12, dims up to 2.5M, UNNZ up to
  740K). Hyperedges become nonzeros; dummy nodes pad non-uniform hyperedge
  cardinality up to the tensor order (except contact-school, capped at
  order 5).
- **Rank**: fixed **per dataset**, not swept in the main comparison (Table
  III: 2, 3, 5, 3 for synthetic; 12, 4, 10, 4, 3 for real) — "the ranks are
  set to the maximum values that allow the baseline algorithms to run… and
  SymProp to run within memory constraints." A dedicated rank-sweep
  ({2,4,8,12,16,20}) is run separately as a parameter study (Fig. 5a).
- **Mode sweep**: mode-agnostic by construction (fully symmetric input; same
  factor matrix used for every mode in the Tucker decomposition).
- **Format conversion / preprocessing**: not discussed as a separate cost;
  CSS-format construction cost is not reported/excluded explicitly (open
  question).
- **Timing protocol**: each operation run **10 times**, average runtime
  reported. No explicit warmup step described. Single node, 32 cores (2×
  AMD EPYC 7302 16-core), OpenMP with `OMP_PROC_BIND=spread`.
- **Precision & correctness**: **explicit double-precision** ("Floating-point
  numbers are represented in double-precision"). Correctness validated via
  convergence-curve comparison (relative reconstruction error vs.
  iteration) against HOOI, not a fixed numeric tolerance.
- **Metric**: wall time (s), speedup vs. baselines, memory footprint (GB),
  thread-scalability speedup (up to 32 cores).
- **Baselines**: TTMc-SPLATT (general sparse TTMc, I/O modified to accept
  IOU-format input directly), S³TTMc-CSS (state-of-the-art symmetric
  sparse-tensor format prior to this work).
- **Format**: CSS (Compressed Sparse Symmetric, tree-based, memoized) for
  the sparse input; compact IOU (index-ordered-unique) linear storage for
  dense symmetric intermediates.

## WACO — Won, Mendis, Emer, Amarasinghe, "WACO: Learning Workload-Aware
Co-optimization of the Format and Schedule of a Sparse Tensor Program"
(ASPLOS'23), ACM/MIT-CSAIL PDF (no arXiv copy)

Source: `commit.csail.mit.edu/papers/2023/WACO_ASPLOS23.pdf`, Sections 5.1–5.2
and 5.6.

- **Workloads**: MTTKRP is one of 4 algorithms (SpMV, SpMM, SDDMM, MTTKRP)
  jointly benchmarked. Training data: **2,893 real SuiteSparse matrices**
  arbitrarily resized into **21,400 sparse matrices** (rows < 131,072, nnz <
  10M); ~2M (matrix, format+schedule, runtime) tuples collected per
  algorithm via random sampling (100 SuperSchedules/matrix). For MTTKRP
  specifically, 3-D sparse tensors are generated **following the prior
  SpTFS paper's approach** (not FROSTT). Held-out test set: **726 real-world
  SuiteSparse matrices** not in training (<10M nnz, <100K rows).
- **Rank**: dense-operand column count |j| = **16** (MTTKRP), fixed, not
  swept, both factor matrices row-major.
- **Mode sweep**: **mode-0 only** — the algorithm is fixed to
  `D[i,j] = A[i,k,l]*B[k,j]*C[l,j]`; other modes are not evaluated.
- **Format conversion / preprocessing / autotuning search**: WACO's format
  and schedule are chosen by a **learned cost model + approximate-nearest-
  neighbor search**; the top-10 SuperSchedules from the search are then
  actually measured on hardware and the fastest is reported as "WACO's"
  runtime — i.e., the reported kernel time explicitly **excludes** the
  (expensive, learned-model) search/tuning cost and any format-conversion
  cost. That search+format-conversion cost is reported **separately** as an
  amortization problem (Section 5.6, Table 8): SpMV/SpMM must be re-run 919
  / 101 times on average to amortize WACO's tuning overhead versus a naive
  baseline, and end-to-end evaluation of real workloads (PageRank, GMRES,
  mesh simulation, GNN training, pruned-NN inference) explicitly folds in
  `T_tuning + T_format_convert + T_tuned_kernel × N_runs`.
- **Timing protocol**: code generated via TACO, compiled with icc
  `-O3 -qopenmp`; each sampled program **repeated 50 rounds, median time
  reported**; samples exceeding 1 minute excluded from training data. Final
  evaluation measures the top-10 candidates on real hardware and reports
  the fastest. Hardware: dual-socket 24-core/48-thread Intel Xeon E5-2680
  v3, `numactl --interleave=all`.
- **Precision**: **single-precision (fp32)** — "All the algorithms were
  performed with single-precision data." This is the one paper in the
  track that does NOT use fp64.
- **Correctness**: not explicitly described (TACO-generated code is assumed
  correct by construction; no numeric tolerance stated).
- **Metric**: speedup vs. baselines (geomean over test matrices), GFLOPS in
  supplementary figures for SpMV/SpMM/SDDMM (not explicitly plotted for
  MTTKRP GFLOPS in the fetched sections), search-time-vs-speedup tradeoff
  curves.
- **Baselines** (MTTKRP-specific; MKL and ASpT are "Not Impl." for MTTKRP):
  **BestFormat** (SpTFS-derived format classifier — format-only tuner) and
  **Fixed CSF** (CCC format, TACO default schedule, OpenMP chunk size 32).
  WACO achieves 1.27× geomean over the better of these two.

## Divergences

- **Tensor suite — no shared corpus across all 4 papers.** BLCO uses named
  real FROSTT/HaTen2 tensors; SySTeC and SymProp's synthetic-tensor
  experiments use randomly generated symmetric tensors (Erdős–Rényi /
  reused prior synthetic sets); SymProp additionally uses real
  *hypergraph*-derived tensors (a different application domain than
  FROSTT); WACO uses SuiteSparse-*matrices* resized/converted into 3-D
  tensors following a third paper's (SpTFS) methodology. **This reflects a
  genuine problem split, not sloppiness**: SySTeC/SymProp target
  *symmetric* input tensors, for which no public real-world corpus with
  FROSTT's scale exists, so synthetic generation is close to unavoidable.
  The spec below keeps FROSTT (BLCO's suite, the one public/citable
  standard real-tensor corpus in the track) as the general/asymmetric
  variant's suite, and defines an explicit synthetic-generator protocol
  (matching SySTeC's Erdős–Rényi method, since it is the most precisely
  specified) for the symmetric variant, with real hypergraph tensors as an
  optional supplementary set.
- **Mode sweep: all-modes vs. mode-0-only.** BLCO's own headline motivation
  (Fig. 1) is that per-mode MTTKRP time on the same tensor can vary by
  **up to 12×** under mode-specific formats, so all-mode reporting is
  treated as necessary for a fair, format-agnostic claim. WACO evaluates
  mode-0 only. SySTeC/SymProp's mode-agnosticism is a genuine mathematical
  property of symmetric inputs (all modes are equal), not a shortcut. The
  spec below makes an explicit all-modes-reported variant the
  kernel-only default for general/asymmetric MTTKRP, and treats mode-0-only
  results (à la WACO) as an insufficient basis for a "MTTKRP performance"
  claim without the other modes disclosed.
- **Rank R: wildly different, 2–500 across papers.** BLCO fixes R=32; WACO
  fixes R=16; SymProp fixes R per-dataset (2–12, chosen to keep baselines
  runnable) with a separate sweep to {2,4,8,12,16,20}; SySTeC sweeps rank
  from 10 to 500. No single rank is comparable across all four. The spec
  fixes a small canonical rank set ({16, 32, 64}) for the general kernel,
  matching the two papers (BLCO, WACO) that fix a single rank value, and
  keeps a wider low-rank sweep for the symmetric variant since symmetric
  Tucker/MTTKRP work explicitly studies rank-dependent speedup.
- **Precision: fp64 vs. fp32.** BLCO explicitly uses fp64 + 64-bit integer
  indices; SymProp explicitly uses fp64; SySTeC's default (Julia
  `Float64`) is fp64 by omission; **WACO explicitly uses fp32** for all
  four algorithms. This is a real, stated precision divergence, not an
  omission — the spec makes fp64 the default (3 of 4 papers) and calls out
  fp32 as a secondary, explicitly-labeled variant axis rather than silently
  mixing results.
- **Preprocessing / autotuning-search inclusion and amortization count.**
  BLCO reports format-construction cost separately and gives an explicit
  amortization count (~12 iterations) to break even against a competing
  format. WACO's "preprocessing" is qualitatively heavier — a *learned
  cost-model search* over an enormous format+schedule space — and the paper
  itself analyzes amortization in terms of hundreds to thousands of kernel
  invocations (Table 8), explicitly warning that WACO-style autotuning is
  "only advantageous… in applications requiring repetitive runs." SySTeC and
  SymProp exclude preprocessing (transposition/replication, and CSS
  construction respectively) from all reported numbers without an
  amortization analysis. The spec below requires preprocessing to always be
  reported, and standardizes on FIXED k=100 amortized iterations for the
  end-to-end variant (a middle ground: large enough to remain fair to
  WACO-style search costs while representative of the tens-of-iterations
  CP-ALS/HOOI loops the underlying decomposition algorithms actually run).
- **Timing protocol never matches across papers.** BLCO: avg-of-25, no
  stated warmup. SySTeC: min-of-(10,000 runs or 5s), single-threaded.
  SymProp: avg-of-10. WACO: median-of-50 (during data collection) then
  single "fastest of top-10 candidates measured" for the headline result
  (not a statistic over repeated runs of one fixed program at all — it is
  a max-over-candidates statistic, arguably a mild form of the "best-of-N"
  practice this instruction set explicitly asks us to fix). None specify a
  device-side/steady-state timer explicitly for MTTKRP. The spec below
  fixes warmup=10, reps=50, statistic=median (report min/max too), and
  flags WACO's "fastest-of-top-10" practice as a fairness issue to avoid
  in the benchmark (report the schedule chosen by the tuner's own ranking,
  not the best of several it produced).
- **Correctness gate absent in all four papers' reported text.** No paper
  in this track states an explicit numerical tolerance against a reference
  MTTKRP for the reported performance numbers (SySTeC/SymProp validate via
  artifact scripts or convergence curves, not a stated tolerance threshold).
  This is flagged as an open question / gap that the spec closes by
  mandating an explicit tolerance.
