# SpMM track — evaluation methodology survey

Track input: `data/track_inputs/spmm.json`, 36 papers. 12 surveyed below in depth
(6 via arXiv fulltext, 6 via artifact-repo benchmark scripts read with
`gh api`), well above the 5-paper / 6-paper minimum. The remaining 24 are used
only for their abstracts to cross-check the "Divergences" and "evidence"
sections (their claimed speedups, e.g. "up to Nx", are noted but not treated
as verified methodology).

Operation surveyed throughout: `C[M,N] = A[M,K] (sparse) * B[K,N] (dense)`,
occasionally batched/repeated inside GCN/GNN training loops.

---

## 1. Gale, Zaharia, Young, Elsen — "Sparse GPU kernels for deep learning" (SC'20)

- key: `conf/sc/GaleZYE20`, arXiv:2006.10901 (fulltext via ar5iv)
- **workloads**: 3,012 sparse weight matrices harvested from 49 real pruned
  models — ResNet-50/ImageNet (kept ≥70% top-1 accuracy) and Transformer/WMT14
  En-De (kept ≥20 BLEU). Sparsity patterns come from actual magnitude pruning,
  not synthetic/random matrices — moderate sparsity levels (the paper's whole
  point is that DNN sparsity, unlike scientific-computing sparsity, is often
  too low for naive sparse kernels to beat dense).
- **timing protocol**: not stated explicitly (no warmup/repeat counts found in
  the available text); GPU is V100, CUDA 10.1.
- **timing scope**: kernel-only. For SpMM used as `im2col`+SpMM for
  convolutions, the `im2col` transform time is explicitly EXCLUDED from the
  benchmark. For SDDMM, the cuBLAS transpose of one operand IS included in the
  timed region (an inconsistency the paper documents rather than hides).
- **precision**: FP32 primary; also a mixed variant (FP16 data / FP32
  accumulate, 16-bit indices).
- **correctness**: not documented in the evaluation section excerpt available.
- **metric**: TFLOP/s and % of FP32 peak; also geometric-mean speedup (3.58x
  SpMM, 2.19x SDDMM vs. cuSPARSE).
- **baselines**: cuSPARSE (`cusparseSpMM`, `cusparseConstrainedGeMM`), plus
  MergeSpMM and ASpT for the RNN case.
- source: arXiv fulltext (ar5iv.labs.arxiv.org/html/2006.10901).

## 2. Rahman, Sujon, Azad — "FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and GNNs" (IPDPS'21)

- key: `conf/ipps/RahmanSA21`, arXiv:2011.06391 (fulltext via ar5iv)
- **workloads**: named real graphs — Cora (2,708v/5,278e), Harvard (15,126v/
  824,617e), Pubmed (19,717v/44,324e), Flickr, ogbn-proteins (132,534v/
  39,561,252e), Amazon, Youtube (1,138,499v/2,990,443e), Orkut (3,072,441v/
  117,185,083e) — plus synthetic RMAT graphs (PaRMAT, 100K vertices) for
  sensitivity sweeps.
- **timing protocol**: "time for 10 iterations and report the average" —
  mean-of-10, no warmup mentioned, no min/max/median reported.
- **timing scope**: kernel time only, explicitly excluding I/O and
  preprocessing; measured from the Python interface (adds call overhead vs. a
  pure C++ measurement, noted separately by the authors).
- **precision**: FP32; embedding dimension d swept over {32, 64, 128, 256, 512}.
- **correctness**: not detailed in the excerpt.
- **metric**: GFLOP/s, arithmetic intensity + roofline analysis, speedup vs.
  baselines.
- **baselines**: DGL v0.5.2 (PyTorch 1.5.1), Intel MKL v2019.5.281 (SpMM only).
- **hardware**: 3 CPU architectures — Intel Skylake 8160 (48c/2s, 2.10GHz),
  AMD EPYC 7551 (64c/2s, 2GHz), ARM ThunderX CN8890 (48c/1s, 1.9GHz).
- source: arXiv fulltext.

## 3. Gianinazzi et al. — "Arrow Matrix Decomposition" (PPoPP'24)

- key: `conf/ppopp/GianinazziZHLAS24`, arXiv:2402.19364 (fulltext via
  arxiv.org/html)
- **workloads**: 13 of the largest SuiteSparse matrices, each decomposed into
  2-4 arrow matrices. Named examples: GenBank (68M rows), MAWI (69M), Webbase
  (118M), OSM-Europe (51M), GAP-twitter (62M rows).
- **scale**: strong/weak scaling up to 256 compute nodes; communication-volume
  comparison specifically at 128 GPUs.
- **timing protocol**: not detailed in the fetched text (no warmup/repeat
  counts visible).
- **timing scope**: this is a **distributed-memory** algorithm — the metric of
  interest is communication volume and runtime growth under weak scaling, not
  a single-node kernel GFLOP/s number. Decomposition (preprocessing) cost vs.
  amortization is not separated out in the excerpt.
- **metric**: communication-volume reduction (3-5x vs. 1.5D decomposition on
  128 GPUs), runtime growth of only 2.3-6.2% scaling from 18M to 200M+ rows at
  fixed vertices/GPU, speedups of 5.3-14.3x vs. the 1.5D A-stationary baseline
  and 1.7-58x vs. a 1D hypergraph-partitioning baseline.
- **baselines**: 1.5D A-stationary SpMM, 1D hypergraph partitioning.
- source: arXiv fulltext.

## 4. Xu et al. — EPPGCN, "Accelerating Backward Aggregation in GCN Training with Execution Path Preparing on GPUs" (TPDS'22)

- key: `journals/tpds/XuSYLJ22`, arXiv:2204.02662 (fulltext via ar5iv)
- **workloads**: 8 real graphs — Cora (2,708), Citeseer (3,327), Pubmed
  (19,717), Twitter (81,306), Blog (88,784), Amazon (410,236), Google
  (875,713), YouTube (1,134,890) vertices — sourced from PyG, SNAP, and
  Network Data Repository. Training-vertex ratio default 10% (also tested at
  80%).
- **timing protocol**: 100 epochs of GCN training per run; backward
  aggregation time is isolated from forward/combination/misc stages.
- **timing scope**: whole-training-loop decomposition, not a standalone
  kernel microbenchmark — SpMM (aggregation) time is one component reported
  separately from total epoch time.
- **precision & correctness**: accuracy delta vs. baseline reported as exactly
  "0.000" — an end-to-end numerical-equivalence check, not a kernel-level
  tolerance.
- **hardware**: NVIDIA Tesla P100 (3,584 cores/16GB) primary, V100S (5,120
  cores/32GB) secondary; host 2x Intel Xeon E5-2680 v4, 256GB RAM.
- **metric**: speedup normalized to GNNAdvisor: 1.48-5.68x (P100) / 1.12-6.57x
  (V100S) for backward aggregation specifically.
- **baselines**: GNNAdvisor (primary), DGL v0.9.0 (secondary).
- source: arXiv fulltext.

## 5. Dias, Sivaram, Strout, Sadayappan — SparseLNR, "Accelerating Sparse Tensor Computations Using Loop Nest Restructuring" (ICS'22)

- key: `conf/ics/DiasSS022`, arXiv:2205.11622 (fulltext via ar5iv)
- **workloads**: mixed corpus — SuiteSparse, Network Repository, FROSTT, and
  the 1998 DARPA Intrusion Detection dataset. Named 2D matrices: cora
  (2.7K×2.7K), bcsstk17 (11K×11K), pdb1HYS (36K×36K), circuit5M
  (5.56M×5.56M). Named 3D tensors: flickr-3d, nell-2, darpa1998.
- **hardware**: single-socket AMD Ryzen Threadripper 3990X, 64 cores @2.2GHz
  (32KB L1d, 512KB L2, 16MB shared L3). Compiled with GCC 7.5.0 `-O3
  -ffast-math`, OpenMP 4.5, all 64 physical cores used.
- **timing protocol**: not explicitly stated (no warmup/repeat/statistic
  found in the evaluation text available).
- **timing scope**: this is a compiler paper — the measured programs are the
  fused-loop-nest kernels TACO/SparseLNR generates; compile/codegen time is
  not part of the reported per-run numbers (loop fusion is a code
  transformation applied ahead of time, like format conversion elsewhere).
- **precision & correctness**: not discussed in the evaluation excerpt.
- **metric**: speedup vs. two TACO baselines — 1.23-1997x vs. baseline TACO
  schedules (this huge range is a red flag: likely includes cases where the
  unfused TACO baseline is pathologically slow/near-zero, not representative
  of a typical library; treated as a divergence below).
- **baselines**: TACO-original (single fused kernel, perfectly nested loops),
  TACO-separate (manually decomposed into multiple kernels).
- source: arXiv fulltext.

## 6. Tripathy, Yelick, Buluç — CAGNET, "Reducing Communication in Graph Neural Network Training" (SC'20)

- key: `conf/sc/TripathyYB20`, arXiv:2005.03300 (fulltext via ar5iv)
- **workloads**: Reddit, a protein network with >1B edges, and other
  unspecified datasets (full list truncated in the available fulltext).
- **scale**: trained on over a hundred GPUs.
- **hardware**: Summit supercomputer (ORNL).
- **timing scope**: distributed 1D/1.5D/2D/3D sparse-dense matmul algorithms
  built on top of **existing single-node kernels from cuSPARSE** called via
  PyTorch — i.e. the SpMM kernel itself is not novel here, the contribution is
  the communication schedule around it. Communication is analyzed with an
  α-β model (constant per-message latency α + per-word bandwidth term β).
- **metric**: asymptotic communication-volume reduction (e.g. 2D algorithm:
  O(√P) fewer words than vertex-partitioning baselines); measured 72% total
  communication reduction and 29% max-per-process reduction in a Reddit
  partitioning experiment.
- **correctness/timing protocol**: not detailed in the fetched excerpt.
- source: arXiv fulltext.

## 7. Huang, Ding, Yu — GE-SpMM (SC'20)

- key: `conf/sc/HuangD0Y20`, artifact: github.com/hgyhungry/ge-spmm
- **workloads**: SNAP graph collection (`data/snap/*`, one `.mtx` per named
  graph directory) plus a `data/misc/*.mtx` set. CSR-native, matches GNN
  framework format directly (a design goal: zero format-conversion overhead).
- **timing protocol** (from `spmm_test.cu`, code-verified): `#define ITER
  200`; one dummy `warmup<<<1,1>>>()` kernel before each measured block;
  `cudaEventRecord`/`cudaEventElapsedTime` wraps the **entire 200-iteration
  loop**, then GFLOP/s = `ITER * nnz * 2 * N / time_ms / 1e6`. This yields only
  a **mean**, no per-iteration variance, min, or median.
  `run_test.sh` sweeps output width N over {32, 64, 128, 256, 512}.
- **timing scope**: kernel-only, no format conversion (CSR taken as given).
- **correctness**: a `#define VALIDATE` block exists in the source but is
  **commented out / disabled by default** in the shipped test — correctness
  checking is opt-in, not gating.
- **metric**: GFLOP/s, reported per (dataset, N) as a CSV row comparing
  cuSPARSE-gflops vs. gespmm-gflops.
- **baselines**: cuSPARSE, and merge-spmm/Gunrock (`gbspmm`) run at the same N
  values in the same script.
- source: `gh api repos/hgyhungry/ge-spmm/contents/{run_test.sh,spmm_test.cu}`.

## 8. Pang, Fan, Qiu, Zhu, Li — RoDe (PPoPP'24)

- key: `conf/ppopp/PangFQZL24`, artifact: github.com/CRAFT-THU/RoDe
- **workloads**: user-supplied directory of `.mtx` matrices (the paper's own
  set is SuiteSparse-derived, consistent with the row-decomposition method
  being format/dataset-agnostic); `eval.sh` iterates `$DATA_PATH/*`.
- **timing protocol** (code-verified, `eval/eval_spmm_f32_n128.cu`): `int ITER
  = 10`; a single `cudaEventRecord`/`cudaEventElapsedTime` pair brackets the
  whole 10-iteration loop (mean, not median); GFLOPS = `ITER * nnz * 2 * N /
  tot_ms / 1e6`. Separate eval files fix N to 32 or 128 (`#define BN 128`),
  and separate f32 vs. f64 files exist.
- **timing scope**: **preprocessing is explicitly reported separately** —
  `eval.sh` runs a dedicated preprocessing-only pass
  (`ASpT_SpMM_GPU/pure_preprocess`, `Preprocess_opt/preprocess`) writing
  `result_preprocess.csv` with per-matrix preprocessing time for
  ASpT/Sputnik/RoDe, kept out of the per-iteration kernel GFLOPS number. This
  is the cleanest amortization discipline of the surveyed set.
- **correctness**: a custom `MatrixDiff` CUDA kernel sums `|A-B|` over the
  whole output matrix and prints the aggregate — there is no normalized
  relative-error threshold or automatic pass/fail gate in the code; a human
  has to read the printed number.
- **baselines**: Sputnik and cuSPARSE compared in the same driver file as RoDe.
- **metric**: GFLOP/s.
- source: `gh api repos/CRAFT-THU/RoDe/contents/{eval.sh,eval/eval_spmm_f32_n128.cu}`.

## 9. Fan, Wang, Chu — DTC-SpMM (ASPLOS'24)

- key: `conf/asplos/Fan0024`, artifact: github.com/HPMLL/DTC-SpMM_ASPLOS24
- **workloads** (code-verified, `run_DTC_SpMM.py`): 8 real GNN graphs —
  YeastH, OVCAR-8H, Yeast, DD, web-BerkStan, reddit, ddi, protein — each
  tested in an "origin" (raw CSR) and a "reorder" (TCA-reordered) variant.
  This exact 8-graph suite recurs across several tensor-core SpMM papers
  (descends from the TC-GNN/GNNAdvisor benchmark set).
- **timing protocol**: `DTCSpMM.preprocess_gpu` (block partitioning +
  metadata construction, i.e. sparse→Tensor-Core format conversion) is called
  **once, outside** the per-feature-size loop; the SpMM kernel itself
  (`DTCSpMM.run_DTCSpMM[_balance]`) is then re-run at N ∈ {128, 256, 512} —
  confirming preprocessing is amortized/excluded from the per-call kernel
  number by construction, though the exact warmup/repeat count inside the
  library binding is not visible from the Python driver.
- **hardware**: RTX 4090 (preferred) or RTX 3090, CUDA 12.1.
- **baselines**: cuSPARSE, Sputnik, SparseTIR, TC-GNN — each has its own
  parallel run script in `scripts/`.
- **metric**: execution time + throughput CSV per (dataset, feature-size).
- source: `gh api repos/HPMLL/DTC-SpMM_ASPLOS24/contents/{README.md,scripts/DTCSpMM/run_DTC_SpMM.py}`.

## 10. Okanovic et al. — SMaT (SC'24)

- key: `conf/sc/OkanovicKLBVH24`, artifact: github.com/spcl/smat
- **workloads** (code-verified, `src/matrix_list.csv`): a small, explicitly
  curated set of ~9 SuiteSparse matrices with id/group/name/dims/nnz recorded
  directly in the repo — e.g. mip1 (66,463×66,463, nnz 10,352,819), shipsec1,
  pdb1HYS, consph, cant, cop20k_A, dc2, rma10, conf5_4-8x8-10 — plus
  synthetic "band matrices" generated for controlled-density studies.
- **timing protocol** (`src/run_smat.sh`): `-warmup_iterations=1
  -profiling_iterations=10 -sleep_duration=100` (100ms pause between
  configurations, presumably to avoid GPU clock-boost carryover between
  runs); M=N=K=512 fixed in the swept driver.
- **correctness**: an `-enable_check` boolean flag exists in the binary's CLI
  but is set to `false` in the shipped sweep script — correctness verification
  is again opt-in and off by default in the actual benchmark run.
- **baselines/metric**: uses the raw CUDA MMA API directly (Tensor Cores);
  also ships an `ncu --set full` profiling path for deep per-kernel metrics
  separate from the throughput sweep.
- source: `gh api repos/spcl/smat/contents/src/{matrix_list.csv,run_smat.sh}`.

## 11. Li, Osama, Huang — Magicube (SC'22, Best Paper Finalist)

- key: `conf/sc/LiOH22`, artifact: github.com/ParCIS/Magicube
- **workloads**: `SpMM/SpMM/eval_matrices` (curated set, not fully enumerated
  in the driver excerpt) swept across 5 quantized precision configurations:
  16b8b, 8b8b, 8b4b, 4b4b (mixed and matched A/B bit-widths).
- **timing protocol** (code-verified, `spmm_benchmark.cpp`): 16 warmup
  iterations, then `NUM_PROFILES` profiling iterations — but distinctively, a
  **new CUDA event pair is created/recorded/destroyed inside every single
  profiling iteration** (`for(iter<NUM_PROFILES){ cudaEventCreate...
  cudaEventRecord(start); kernel; cudaEventRecord(end);
  cudaEventElapsedTime...}`), i.e. per-call timing rather than one timer
  around a batched loop like RoDe/GE-SpMM/SMaT. This is the methodologically
  strongest timer pattern of the surveyed GPU papers, since it permits
  reporting median/min/max instead of only a batch-mean — though the paper
  itself does not confirm which statistic it reports over the `NUM_PROFILES`
  samples.
- **precision**: quantized low-precision INT4/INT8/mixed on Tensor Cores.
- **correctness**: a full CPU reference (`compute_ref_integers`) re-implements
  the quantized integer arithmetic bit-for-bit (mask/shift decode, integer
  MAC) and is diffed against the GPU output — since the arithmetic is integer,
  the natural check is **exact match**, not a floating-point tolerance.
- **hardware**: NVIDIA A100-SXM4-40GB, CUDA 11.4.0, GCC 8.4.1.
- source: `gh api .../SpMM/SpMM/spmm_benchmark.cpp`, `README.md`.

## 12. Jain, Gaikwad, Chennupati — RASSM (ASPLOS'25)

- key: `conf/asplos/JainGC25`, artifact: github.com/gt-tinker/RASSM
- **workloads**: user-supplied `$RASSM_DATASET/*` directory of `.mtx`
  matrices (paper compares against ASpT and J-Stream on matrices drawn from
  that prior literature; the exact list isn't enumerated in the run script
  itself, only the driving loop over the dataset directory).
- **timing protocol** (`scripts/run-rassm.sh`, code-verified): `nruns=50`;
  the driver `grep`s **"Median Time"** and **"GFLOPS"** directly out of the
  program's stdout — i.e. the tool itself computes and reports the median
  over 50 runs, the most rigorous statistic of any surveyed artifact (vs. the
  batched-mean pattern common on the GPU side).
- **timing scope**: CPU kernel-only; `layers=4` suggests SpMM is invoked
  repeatedly per matrix run (mimicking a multi-layer GNN forward pass) inside
  each of the 50 timed repetitions.
- **threading**: `OMP_NUM_THREADS` = total physical cores across all sockets
  (from `lscpu`), `OMP_PROC_BIND=true`, threads pinned via `OMP_PLACES=cores`.
  Cache-aware tile sizes (Ri, Rj) and target-cache-size are auto-derived from
  `lscpu`'s reported L2 size.
- **metric**: GFLOPS + median time, per (matrix, feature-size).
- source: `gh api repos/gt-tinker/RASSM/contents/scripts/run-rassm.sh`.

---

## Divergences

- **Timer methodology splits three ways**: (a) batched-loop-then-divide mean
  with a single CUDA-event pair around N=10-200 iterations (RoDe, GE-SpMM,
  SMaT) — cannot report median/variance, hides jitter; (b) per-iteration CUDA
  events inside the loop (Magicube) — enables median/min/max but papers don't
  always report it; (c) wall-clock/std::chrono averages on CPU (FusedMM: mean
  of 10; RASSM: **median** of 50, the outlier in rigor). The spec fixes this
  by mandating per-iteration device-side timestamps and reporting median +
  min/max, following RASSM's practice rather than the more common
  batched-mean.
- **Correctness checking is frequently opt-in and disabled in the shipped
  benchmark script**: GE-SpMM (`VALIDATE` macro commented out) and SMaT
  (`-enable_check=false` in the sweep script) both ship with correctness
  verification off by default in the exact script used to produce reported
  numbers. RoDe checks but only prints an unnormalized `sum|A-B|` with no
  pass/fail gate. Only Magicube (exact integer match, quantized arithmetic)
  and EPPGCN (exact accuracy-delta match, end-to-end) enforce a hard
  correctness gate. The spec makes correctness-before-timing mandatory and
  quantitative (relative error threshold per precision), which several
  papers' own artifacts do not currently do.
- **Preprocessing/format-conversion treatment differs by paper family**:
  RoDe and DTC-SpMM cleanly separate one-shot preprocessing (block
  partitioning, TCA-reordering) from the timed per-call kernel, reporting
  preprocessing cost as its own artifact (RoDe: separate CSV column per
  baseline). GE-SpMM's whole design goal is to need **zero** preprocessing
  (CSR-native), so the question doesn't arise for it. Quantized/CPU papers
  (Magicube, FusedMM, RASSM) don't do runtime format conversion at all
  (data is pre-packed offline). No surveyed paper silently folds format
  conversion into the per-iteration number, but none of them state an
  explicit amortization count (how many repeated SpMM calls "pay for" one
  conversion) either — the spec makes k explicit (k=100, following EPPGCN's
  100-epoch training convention and RASSM's `layers=4 × nruns=50` repeated-
  invocation pattern).
- **Input suites don't overlap much across papers, and none state a single
  principled inclusion rule.** Two rough clusters exist: (1) general
  SuiteSparse matrices selected by nnz range and hand-curated for tensor-core
  friendliness (SMaT's explicit 9-matrix list, DTC-SpMM's origin matrices,
  RoDe/RASSM's directory-of-.mtx pattern), and (2) a recurring "standard GNN
  benchmark" octet — YeastH, OVCAR-8H, Yeast, DD, web-BerkStan, reddit, ddi,
  protein — that traces back to the TC-GNN/GNNAdvisor lineage and reappears
  in DTC-SpMM and (by convention) several of the abstract-only papers in this
  track (FlashSparse, Voltrix, GeneralSparse, MP-SpMM all cite RoDe/DTC-SpMM
  as baselines and by convention inherit overlapping matrix sets, though this
  wasn't independently verified for those abstract-only entries). SparseLNR
  and FusedMM instead draw from Network Repository/FROSTT/graph-embedding
  corpora with almost no overlap with the tensor-core cluster. The spec's
  `recommended_subset` unions both clusters and states the selection rule
  explicitly rather than deferring to "whatever the artifact ships with."
- **Metric units and scope diverge by hardware class**: single-node GPU/CPU
  papers report GFLOP/s or TFLOP/s (7 of 12 surveyed); the two distributed
  papers (Arrow Matrix, CAGNET) report communication volume and scaling
  efficiency instead, because at their scale communication — not FMA
  throughput — is the bottleneck being optimized. These are not
  commensurable with a single-node kernel-GFLOP/s benchmark; distributed SpMM
  is treated as **out of scope** for this harness (see open_questions) rather
  than forced into an ill-fitting variant.
- **"Up to Nx" best-case-only reporting is the norm in the abstracts of the
  24 not-fully-surveyed papers** (e.g. GeneralSparse "up to 20.82x over
  cuSPARSE", MP-SpMM "up to 8.6x over RoDe", DTC-SpMM "up to 1.91x
  end-to-end"), consistent with SparseLNR's fully-surveyed 1.23-1997x range
  which almost certainly includes non-representative best cases. The spec
  requires reporting the full per-matrix distribution (median speedup +
  IQR/range across the fixed input suite), not a single headline number.
- **Precision is not a single axis**: fp32 dominant (RoDe, GE-SpMM, SMaT,
  FusedMM, SparseLNR), fp64 offered as an alternate build (RoDe), fp16-mixed
  (Gale et al.), and integer-quantized 4-16 bit (Magicube) are all present in
  the surveyed set and are not directly comparable on the same GFLOP/s scale
  (quantized ops do integer MACs, not FMAs) — handled as a separate
  precision-specific variant rather than folded into the fp32 kernel variant.

## Recommended-subset provenance

Matrices proposed for `spec.yaml`'s `recommended_subset` are the union of
named instances that recur across ≥1 code-verified surveyed artifact (not
cherry-picked from abstracts): SMaT's 9-matrix SuiteSparse list (mip1,
shipsec1, pdb1HYS, consph, cant, cop20k_A, dc2, rma10, conf5_4-8x8-10),
DTC-SpMM's 8-graph GNN suite (YeastH, OVCAR-8H, Yeast, DD, web-BerkStan,
reddit, ddi, protein), SparseLNR's circuit5M as a large-scale outlier, and
FusedMM/EPPGCN's small standard graphs (cora, citeseer, pubmed, amazon,
youtube) for the low-nnz end of the range.
