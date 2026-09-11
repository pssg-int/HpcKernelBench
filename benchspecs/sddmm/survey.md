# SDDMM track — evaluation-methodology survey

Track input: `data/track_inputs/sddmm.json` (13 papers). 10 of the 13 were
inspected; 8 yielded concrete, source-grounded methodology facts (via arXiv
fulltext or artifact-repo benchmark/run scripts read through `gh api`); 2
(Insum, HP-SpMM-SDDMM) turned out to have artifact repos too thin to recover
a protocol from, which is itself a useful finding (see Divergences /
Open questions). RASSM, FusedMM and distributed_sddmm/HnH are algorithmic
papers whose main claim is for SpMM but whose artifact/benchmark explicitly
also drives an SDDMM code path — included because the track's papers overlap
heavily with the spmm track and their SDDMM protocol is concretely
recoverable.

---

## conf/sc/GaleZYE20 — "Sparse GPU kernels for deep learning" (SC'20)

Source: arXiv fulltext `2006.10901` (ar5iv HTML) — the AI-summarized extract
below should be treated as approximate paraphrase, not verbatim quote; the
sparsity-bucket and matrix-count figures should be cross-checked against the
PDF table before being hard-coded into a harness.

- **workloads/inputs**: introduces the "DLMC" (deep-learning sparse matrix)
  dataset: sparse weight matrices from ResNet-50 (ImageNet) and Transformer
  (WMT14 En-De), pruned by 4 different algorithms at multiple target
  sparsities (~71%-95% observed range, later organized by Magicube into
  50/70/80/90/95/98% buckets — see below). Also uses SuiteSparse (~2,833
  matrices) as a scientific-computing sparsity/structure comparison, not as
  a benchmark suite for the kernel itself.
- **dense operand**: N reflecting real workload shapes — RNN batch 16-128,
  Transformer hidden dim 256-2048, conv-derived N from 64 to 14,400.
- **timing protocol**: not recovered in detail from the extract (ar5iv
  summarization dropped the numeric loop code); known from the ecosystem
  this paper spawned (its released kernel library is "Sputnik", used as a
  baseline by nearly every later paper below) to use CUDA-event timing.
  Flag as **not independently verified** here.
- **precision**: fp32, plus mixed fp16-storage/fp32-accumulate.
  Hardware: V100, CUDA 10.1.
- **metric**: TFLOP/s and % of fp32 peak (kernels reach 27% of V100 fp32
  peak); geometric-mean speedup (3.58x / 2.19x) vs. baselines for two
  different workload classes.
  Also reports **speedup on complete models** (sparse Transformer, sparse
  MobileNet) — an end-to-end amortization argument layered on top of the
  raw kernel numbers.
- **baselines**: cuSPARSE (`cusparseSpMM`, `cusparseConstrainedGeMM`),
  MergeSpMM, ASpT.
- **correctness**: not recovered from the extract.

## conf/ipps/RahmanSA21 — "FusedMM" (IPDPS'21)

Source: arXiv fulltext `2011.06391` (ar5iv HTML).

- **workloads/inputs**: real graphs used as the sparsity pattern: Cora
  (2,708v/5,278e), Pubmed (19,717v/44,324e), YouTube (1.14M v/2.99M e),
  Orkut (3.07M v/117.2M e), OGBprot (132.5K v/39.6M e), plus Harvard,
  Flickr, Amazon, and synthetic RMAT graphs (generated via PaRMAT) for
  parameter-sensitivity sweeps.
- **dense operand**: embedding dim d in {32, 64, 128, 256, 512}; most
  detailed tables use d=128.
- **timing protocol**: "measure the time for 10 iterations and report the
  average time"; IO/preprocessing excluded from the timed region. No
  explicit warmup phase is described (10-iteration average appears to be
  the entire measurement).
- **precision**: fp32 primary; the code generator can also emit fp64.
- **metric**: wall-clock kernel time (seconds), speedup vs. baseline.
- **baselines**: DGL v0.5.2 (+PyTorch v1.5.1), Intel MKL v2019.5.281
  (SpMM only), native PyTorch for end-to-end training comparisons.
- **correctness**: indirect — validated via **downstream task accuracy**
  (Force2Vec F1-micro scores match between reference and FusedMM-based
  implementation, 0.78 vs 0.79) rather than a direct per-element numerical
  tolerance on the SDDMM output itself. This is a materially weaker
  correctness gate than an elementwise tolerance check (see Divergences).
- **platforms**: Intel Skylake 8160, AMD EPYC 7551, ARM ThunderX CN8890 —
  CPU/ARM only, vectorized kernels, code-generated per architecture.

## conf/ipps/BharadwajBD22 — "Distributed-Memory Sparse Kernels for ML" / HnH (IPDPS'22)

Source: artifact repo `PASSIONLab/distributed_sddmm`, file
`local_kernel_benchmark.cpp` (single-node local-kernel benchmark harness
inside the distributed-memory library), read via `gh api`.

- **workloads/inputs**: synthetic Erdos-Renyi random sparse matrices,
  parameterized by `logM` (matrix dimension, log2) and `nnz_per_row`; not
  SuiteSparse or DLMC — this paper's SDDMM benchmark is purely
  scale-driven (varying M, nnz/row, R) rather than "real matrix" driven, to
  isolate the effect of 1.5D/2.5D communication algorithms.
- **dense operand**: R (embedding width) swept over a list of values
  (`rValues`).
- **timing protocol**: **min-time / dynamic-repetition**, not fixed
  iteration count: a `do { ...; num_trials++; } while (elapsed < min_time)`
  loop runs the CPU (MKL/OpenMP) SDDMM kernel repeatedly until a
  wall-clock threshold `min_time` is exceeded, then reports
  `throughput = nnz * 2 * R * num_trials / elapsed` in GFLOP/s. This is a
  materially different protocol family from the fixed-N-iteration GPU
  papers below (see Divergences).
- **precision**: fp64 (`double` throughout, MKL `mkl_sparse_d_*` calls).
- **metric**: GFLOP/s, FLOP count = `2 * nnz * R` (same 2-FLOPs-per-nonzero
  convention as the GPU papers).
- **baselines**: Intel MKL CSR SpMM (`mkl_sparse_d_mm`) as the reference
  dense-adjacent kernel; PETSc cited in the one-liner as the ~10x-slower
  distributed baseline for the full 1.5D/2.5D algorithm (not reproduced in
  this local-kernel file).
- **correctness**: not present in this benchmark file (no diff/validate
  step visible; likely validated elsewhere in the test suite, not in the
  timed path).
- **platform**: CPU, MPI + OpenMP, designed for distributed clusters
  (module load script `modules.sh` targets NERSC Cori).

## conf/sc/LiOH22 — "Magicube" (SC'22, Best Paper Finalist)

Source: artifact repo `ParCIS/Magicube`, files `README.md` and
`SDDMM/SDDMM/sddmm_benchmark.cpp`, read via `gh api`.

- **workloads/inputs**: the **DLMC dataset** (same corpus introduced by
  Gale et al. SC'20), fetched via
  `wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz`. The
  artifact organizes evaluation matrices into `eval_matrices/s{50,70,80,
  90,95,98}.txt` — one file list per target sparsity level (50/70/80/90/
  95/98%), each listing per-layer `.smtx` matrices from `rn50/` (ResNet-50)
  and (by symmetry with the SpMM side) transformer pruning subdirectories,
  e.g. `rn50/magnitude_pruning/0.5/bottleneck_2_block_group2_1_1.smtx`.
  This file-list-per-sparsity-bucket structure is a directly reusable,
  concrete selection mechanism.
- **dense operand**: tested via separate launcher scripts per bit-width
  (`launch_sddmm_magicube_4b4b.py`, `_8b8b.py`, `_16b16b.py`).
- **timing protocol**: **warmup = 32 iterations (discarded)**, then
  **NUM_PROFILES = 512 timed iterations**, each iteration individually
  timed with a fresh `cudaEvent_t` start/stop pair, accumulated and
  reported as `sddmm_ms_avg / NUM_PROFILES` — i.e. **mean**, not median,
  over 512 reps.
- **precision**: quantized low-bit — SDDMM tested at 4-bit x 4-bit,
  8-bit x 8-bit, and 16-bit x 16-bit (mixed-precision int formats on Tensor
  Cores); this is a genuinely different numerical regime from the fp32/
  fp16 kernels in the rest of the track.
- **metric**: ms runtime -> implied GFLOP/s (consistent 2*nnz*K
  convention used elsewhere in repo/paper).
- **baselines**: (from repo structure) a `baselines/` directory and
  `end2end_eval/` comparing against cuDNN fp16 and vectorSparse for the
  end-to-end sparse-Transformer figure.
- **correctness**: not visible in the excerpted benchmark file; the
  container-based AE workflow (Zenodo docker image) is the paper's stated
  reproducibility path rather than an in-code tolerance check.
- **hardware**: NVIDIA A100-SXM4-40GB (paper-stated).

## conf/ppopp/PangFQZL24 — "RoDe" (PPoPP'24)

Source: artifact repo `CRAFT-THU/RoDe`, files `README.md`, `AE.md`,
`eval.sh`, `script/eval_sddmm_call.sh`, `eval/eval_sddmm_f32_n128.cu`, read
via `gh api`.

- **workloads/inputs**: SuiteSparse matrices selected via `ssgetpy` with
  filter `rowbounds=(10000,None), colbounds=(10000,None),
  nzbounds=(100000,None), limit=5000` (`script/download_script.py`) — i.e.
  **rows >= 1e4, cols >= 1e4, nnz >= 1e5**, effectively the whole
  size-filtered SuiteSparse collection. A 6-matrix "small" smoke-test
  subset is also shipped explicitly by name: `192bit, wv2010, xenon2,
  Zd_Jac2_db, Zhao2, mip1` (`script/download_data_small.sh`).
- **dense operand**: N in {32, 128} (separate compiled kernel variants
  `eval_sddmm_f32_n32.cu` / `eval_sddmm_f32_n128.cu`), plus f64 N=128
  variants for SpMM (SDDMM appears f32-only in this repo).
- **timing protocol**: **ITER = 10**, `cudaEventRecord`/`cudaEventElapsedTime`
  bracketing the whole 10-launch loop (`cudaDeviceSynchronize()` once
  before the loop, once after) — **no separate discarded warmup phase**;
  the reported GFLOP/s is `ITER * nnz * 2 * K / tot_ms`, i.e. an
  **average over the 10 launches**, not a median, and the first launch
  (potential cold-cache effects) is included in the timed region.
- **precision**: fp32 for SDDMM (RoDe repo has separate f64 kernels for
  SpMM but not SDDMM).
- **correctness**: an optional `#ifdef VALIDATE` path exists
  (`MatrixDiff` kernel, sums absolute per-element differences against a
  naive dense-reference `StandKernel`) but **is not compiled into the
  default evaluation binary** — the numbers `eval.sh`/AE reproduces are
  produced *without* the validation gate active by default. This is a
  concrete instance of "correctness check exists but is off during the
  timed run" that the spec should not repeat.
- **metric**: GFLOP/s = `ITER * nnz * 2 * K / time_ms / 1e6`.
- **preprocessing**: format conversion/preprocessing cost is measured
  **separately** and reported on its own (`result_preprocess.csv` in
  `AE.md`, comparing ASpT/Sputnik/RoDe preprocessing cost) — never mixed
  into the per-iteration kernel time. Good practice, matches the spec's
  fairness principle directly.
- **baselines**: Sputnik (Gale et al.'s SDDMM kernel, vendored as
  `Sputnik_SDDMM/`), ASpT (`ASpT_SDDMM_GPU/`).

## conf/ppopp/ShiLXFWW25 — "FlashSparse" (PPoPP'25)

Source: artifact repo `ParCIS/FlashSparse`, files `README.md`,
`dataset.txt`, `eva/kernel/sddmm/test_sddmm_shell.sh`, read via `gh api`.

- **workloads/inputs**: 515 sparse matrices total — SuiteSparse matrices
  (downloaded as `.tar.gz`, converted to `.npz`, then block-preprocessed
  via `FS_Block_gpu.preprocess_gpu_fs()` into the paper's tensor-core
  block format) plus GNN datasets from DGL/PyG official sources.
- **dense operand**: N in {32, 128} for SDDMM (`test_sddmm_shell.sh` calls
  `sddmm_fp16_test_args.py 32`, `128` and `sddmm_tf32_test_args.py 32`,
  `128`).
- **precision**: fp16 and tf32 (two explicit Tensor Core precision
  variants), each run separately.
- **preprocessing**: matrices must be partitioned/preprocessed into TC
  blocks before the kernel runs (`preprocess_gpu_fs`); the repo reproduces
  a dedicated "Table 7 / format ablation" measuring this cost
  (`eva/plot/ablation/format/format.py`) separately from steady-state
  kernel throughput — same separate-and-reported pattern as RoDe.
- **metric**: results collected into `result/FlashSparse/sddmm/*.csv`
  (throughput-style, consistent with the rest of the paper's GFLOP/s
  reporting) and later summarized to reproduce Figure 13 / Table 6 /
  Table 5 (median/left-right split for two GPUs: H100 PCIe and RTX4090).
- **baselines**: RoDe, Sputnik, cuSPARSE, GNNAdvisor, TC-GNN, GE-SpMM,
  DTC-SpMM (SpMM only) — RoDe itself vendored as `Baseline/RoDe/`.
- **hardware**: two GPUs used side by side throughout, H100 PCIe and
  RTX4090 — cross-architecture comparison is part of the paper's own
  design, not just single-GPU numbers.
- **correctness**: not recovered from the surface-level files read; a
  `eva/accuracy/gcn/eva_gcn.py` script exists for end-to-end model
  accuracy (analogous to FusedMM's downstream-task validation), separate
  from a raw numerical kernel-output check.

## conf/ics/LiC25 — "Fused3S" (ICS'25)

Source: artifact repo `HPCForge/Fused3S`, files `README.md`,
`scripts/baseline_comp/baseline_comp_kernel_only.py`,
`scripts/baseline_comp/sddmm_comp.sh`, `scripts/downloadDataset.py`, read
via `gh api`. This is the **cleanest, most spec-compliant protocol found
in the whole track** and is treated as the reference implementation for
the timing-protocol fairness principles below.

- **workloads/inputs**: 15 named real graph datasets in the standard
  kernel-only comparison — `citeseer, cora, pubmed, Ell, github, Artist,
  com-amazon.ungraph, Blog, amazon0505, igb_small, yelp, reddit,
  igb_medium, ogbn-products, amazonProducts` — plus 2 "large" datasets run
  separately (`igb_large, ogbn-papers100M`) and 9 **batched-graph**
  datasets for a distinct batched-attention setting (`ZINC, PascalVOC-SP,
  COCO-SP, Peptides-func, Peptides-struct, ogbg-molhiv, ogbg-ppa,
  ogbg-molpcba, ogbg-code2`). Graphs downloaded via PyG/OGB/IGB loaders
  (`torch_geometric.datasets`, `ogb.nodeproppred`, `igb.dataloader`).
- **dense operand**: embedding dim 128 as the shell-script default
  (`featureDim=128`), fp32 host tensors cast to fp16 for the Tensor-Core
  kernels.
- **timing protocol**: **warmup = 3 iterations (discarded)**, then
  **niter = 10 timed iterations**, each bracketed by
  `torch.cuda.Event(enable_timing=True)` start/stop with
  `end_event.synchronize()` before recording; **both median AND mean AND
  std** of the 10 timed samples are computed and logged, with **median**
  used as the value written into the results table
  (`perf.runtime_pd.loc[...] = median_time`). This directly matches the
  spec instructions' "prefer median + report min/max/std" principle.
- **FLOP accounting**: explicit and separable per stage —
  `sddmm_flops = nnz * embedding_dim * 2`,
  `softmax_flops = nnz * 5`, `spmm_flops = nnz * embedding_dim * 2`; the
  fused-kernel FLOPs reported is the **sum of all three stages**, i.e.
  Fused3S numbers are for the fused SDDMM+softmax+SpMM pipeline, not
  isolated SDDMM. Isolating SDDMM-only cost requires per-kernel profiling
  (the repo does this via `ncu --kernel-name "regex:...sddmm_kernel..."`
  for Figure-level profiling, not for the main throughput table).
- **precision**: fp16/fp32 mixed (Tensor Core GEMM path), correctness
  checked separately (`scripts/tests/test_f3s_accuracy.py`) — i.e.
  correctness verification is a **distinct script from the perf run**,
  not gated into the timed path (same soft spot as RoDe's optional
  VALIDATE, though at least a dedicated accuracy test exists and is
  documented as a required reproduction step).
- **baselines**: DF-GNN (tiling/hyper variants), FlashSparse
  (no-softmax/naive-softmax/stable-softmax variants), PyG's native
  `GTConv` (`pyg_gtconv`) — 11 named algorithm variants total in the
  comparison matrix.
- **hardware**: A30 and H100 GPUs (Ampere-tuned kernels, Hopper support
  described as "ongoing work").

## conf/asplos/JainGC25 — "RASSM" (ASPLOS'25) — CPU representative

Source: artifact repo `gt-tinker/RASSM`, files `README.md`,
`code/src/main.cpp` (`--kernel` flag), `scripts/download_large_40M.sh`,
read via `gh api`.

- **workloads/inputs**: SuiteSparse matrices, ~45GB uncompressed,
  downloaded by explicit name in `download_large_40M.sh` — 47 matrices
  including `pre2, twotone, pwtk, hcircuit, scircuit, lp_ken_18, xenon2,
  cfd2, gearbox, pkustk14, kim2, lung2, stomach, torso1, torso2, torso3,
  cage12, cage13, cage14, af_shell1..af_shell9, c-73, matrix_9,
  matrix-new_3, barrier2-1..barrier2-12, ohne2, para-4..para-10, Stanford,
  Stanford_Berkeley, language, Hamrle3`. **cage12/13/14 and xenon2
  overlap directly with RoDe's suite** — a rare cross-paper, cross-platform
  (CPU vs GPU) shared-matrix anchor point, useful for the spec's
  recommended_subset.
- **kernel scope**: `main.cpp` exposes `--kernel {spmm, sddmm}` (default
  `spmm`) with dedicated `data_movement_experiment_sddmm<...>()` — SDDMM
  is a real, distinct code path, not just mentioned in the abstract.
- **timing protocol**: not fully recovered (main.cpp only partially read);
  README documents reproduction via `scripts/run-all.sh` (hours-long full
  sweep) -> `extract_stats.sh` -> a Jupyter notebook that regenerates
  Figure 6, and explicitly warns "exact numbers will vary... general
  trends should be similar" — i.e. the paper's own reproducibility
  statement does not claim tight numerical reproducibility, only
  trend-level.
- **platform**: CPU only, ICC-preferred build, boost dependency; adaptive
  2D tiling is the core contribution (residue-matrix-guided tile-size
  selection), directly analogous to the GPU-side format/tiling
  optimizations (RoDe's row decomposition, FlashSparse's TC blocking).
- **baselines** (from repo dirs `aspt/`, `jstream/` and scripts
  `run-aspt.sh`, `run-csf-uo.sh`, `run-csf-us.sh`, `run-csr.sh`,
  `run-jstream.sh`): ASpT, J-Stream, plain CSR, CSF (uniform-order and
  uniform-stride variants) — a CPU-specific baseline set, disjoint from
  the GPU papers' cuSPARSE/Sputnik/RoDe baseline set.

## conf/ipps/FanWC23 — "HP-SpMM-SDDMM" (IPDPS'23) — thin artifact, partial

Source: artifact repo `fan1997/HP-SpMM-SDDMM`, `README.md`, `run.sh`,
read via `gh api`.

- **workloads/inputs**: 16 GNN graph datasets by name — `Yelp, Flickr,
  AIFB, BGS, AM, MUTAG, AmazonCoBuyComputer, AmazonCoBuyPhoto, CoauthorCS,
  CoauthorPhysics, arxiv, collab, ddi, ppa, products, reddit` (`.smtx` +
  `.cugraph_reorder` file pairs), run at K in {32, 64, 128}.
- **baselines**: cuSPARSE 12.2, GE-SpMM.
- **metric**: "average GFLOPS" reported (repetition count for the average
  not recovered).
- **caveat (important)**: despite the repo name and the original IPDPS'23
  paper covering both SpMM and SDDMM, **the currently hosted repository
  only contains an SpMM benchmark path** (`spmm/` and `test/` directories,
  `run.sh` only invokes `test_all` with what the README states are
  "FP32 SpMM implementations"). No `sddmm/` directory, launcher, or timing
  code is present in this snapshot to recover an SDDMM-specific protocol
  from. Recorded here as an artifact-completeness gap, not a methodology
  fact — see Open questions.

## conf/asplos/WonAAE26 — "Insum" (ASPLOS'26) — thin artifact, not usable

Source: artifact repo `nullplay/IndirectEinsum`, `README.md`, read via
`gh api`.

- The repo currently contains a single `example.py` (SpMM-COO demo using
  `torch.compile`/Triton codegen) and no benchmark/dataset/run scripts of
  any kind. ASPLOS'26 has not yet occurred as of the survey date, so no
  camera-ready artifact or public fulltext was available to cross-check.
  **No methodology facts could be recovered**; retained in the track input
  as a forward-looking codegen approach (GroupCOO format, indirect
  einsums) but excluded from the spec's evidence base. Flagged in Open
  questions — re-survey once the artifact/paper are published.

---

## Divergences

1. **No shared input suite across the track.** Three largely disjoint
   matrix universes are used and almost never overlap:
   - *DLMC* (sparse deep-learning weight matrices, sparsity-bucketed
     50-98%): Gale et al. SC'20 (origin), Magicube.
   - *General SuiteSparse* (size-filtered, scientific/structural
     matrices): RoDe (`rows,cols>=1e4, nnz>=1e5` via `ssgetpy`),
     FlashSparse (515 matrices), RASSM (CPU, 47 named matrices). RoDe and
     RASSM concretely share `cage12/13/14` and `xenon2` — the only
     verified cross-paper overlap found.
   - *Real GNN graphs* (citation networks, social networks, OGB/IGB node-
     and graph-property-prediction sets): Fused3S, FusedMM,
     HP-SpMM-SDDMM. These three each use different, only partially
     overlapping graph lists (e.g. `reddit`, `ogbn-products`/`products`,
     `pubmed` appear in more than one; `Orkut`, `YouTube`, `igb_*`,
     `ZINC` do not).
   The spec resolves this by defining **separate variants per suite
   tradition** rather than forcing an artificial single union (a forced
   union would silently favor whichever tradition happens to dominate the
   combined list by matrix count).

2. **Dense operand width K/N/d is inconsistent but clusters at two
   points.** RoDe and FlashSparse both hard-split their kernels at
   N in {32, 128}; HP-SpMM-SDDMM tests {32,64,128}; FusedMM sweeps
   {32,64,128,256,512}; Fused3S defaults to 128. **32 and 128 are the
   two values every GPU paper surveyed actually ran** — adopted as the
   spec's required pair, with 256 as an optional third point for
   transformer-scale coverage (FusedMM, Gale2020 test up to 512-2048).

3. **Timing protocol has four distinct families, not one.** (a) RoDe:
   fixed ITER=10, **no discarded warmup**, mean over the loop. (b)
   Magicube: 32-iteration discarded warmup + 512 timed reps, **mean**.
   (c) Fused3S: 3-iteration discarded warmup + 10 timed reps, **median**
   (+ mean/std also reported) — closest to the spec's fairness principles.
   (d) HnH/distributed_sddmm: **min-wall-time-based dynamic repetition**
   (run until `elapsed >= min_time`, not a fixed count) — a CPU/MPI-style
   protocol incompatible with a fixed-ITER GPU harness. FusedMM (CPU,
   different paper) uses yet another point protocol: exactly 10 iterations
   averaged, no stated warmup. The spec fixes this by mandating an
   explicit warmup count and reporting median with min/max, closest to
   Fused3S's practice, and treats HnH's min-time protocol as
   variant-specific (distributed track) rather than folding it into the
   GPU kernel-only variant.

4. **Correctness gating is frequently weak or off by default.** RoDe ships
   a validation kernel but does **not compile it into the default AE
   binary** — the published numbers are never checked against a reference
   during the actual timed run. FusedMM validates indirectly via
   downstream classification F1 score equivalence rather than a
   per-element numerical tolerance on the SDDMM output itself. Fused3S at
   least has a dedicated accuracy-test script, but it is a separate
   invocation from the performance run, not a gate before timing counts.
   Magicube's correctness path was not recoverable from the excerpted
   file. **None of the 8 papers with recovered protocols hard-gate
   correctness before accepting a timed run** — the spec makes this
   mandatory (see `correctness` field per variant), which is a deliberate
   deviation from all 8 papers' actual practice.

5. **Preprocessing/format-conversion time is sometimes reported separately
   (good) and sometimes invisible.** RoDe and FlashSparse both explicitly
   separate and report format-conversion cost (RoDe's
   `result_preprocess.csv`, FlashSparse's Table 7 format ablation).
   Magicube, FusedMM, Fused3S, HnH give no visible accounting of
   preprocessing cost in the files inspected — it is unclear whether their
   headline throughput numbers include or exclude the (potentially
   substantial, for Tensor-Core block formats) one-time format-conversion
   step.

6. **Precision scope differs qualitatively, not just quantitatively.**
   Magicube's whole contribution is sub-fp32 quantized SDDMM (4b/8b/16b
   Tensor Core int formats) — this cannot be meaningfully merged into an
   fp32 or fp16 comparison table; it needs its own variant with its own
   correctness tolerance (dequantized error, not raw numerical error).

7. **SDDMM is sometimes measured standalone, sometimes only as part of a
   fused pipeline.** Fused3S's headline numbers are for the fused
   SDDMM+softmax+SpMM ("3S") kernel; isolating SDDMM-only cost requires
   `ncu` kernel-name-regex profiling, which the repo does only for
   supplementary figures, not the main comparison table. RASSM's SDDMM
   path and RoDe/FlashSparse/Magicube's SDDMM kernels are standalone.
   The spec provides both an isolated-kernel variant and a
   fused-pipeline end-to-end variant to make both kinds of paper claims
   comparable on their own terms.

8. **Artifact completeness varies enough to block a protocol survey.**
   HP-SpMM-SDDMM's current repo snapshot has no SDDMM code despite the
   paper and repo name; Insum's repo is a single unrelated SpMM-COO demo.
   Both are excluded from the evidence base (kept in `open_questions`).

## Open questions

- Gale et al. SC'20's exact DLMC matrix count/sparsity-bucket boundaries
  were only recovered via an AI-summarized ar5iv extract, not a direct
  quote from the PDF/HTML tables — re-verify against the paper's Table 1
  (or Magicube's `eval_matrices/s*.txt` file contents, which are a
  faithful machine-readable proxy) before hard-coding into the harness.
- RASSM's exact SDDMM timing protocol (warmup/rep count/statistic) was not
  fully recovered — `main.cpp` confirms the `--kernel sddmm` code path
  exists and is driven by `data_movement_experiment_sddmm`, but the
  measurement loop itself (likely in `code/src/global.cpp` or
  `common.cpp`, not read) needs a follow-up pass before RASSM numbers are
  cited as a fp32-CPU-kernel reference point.
  Also unresolved: **whether RASSM's SDDMM path is a first-class
  contribution or an SpMM-adjacent add-on** — the paper's own abstract
  frames the headline number ("1.36x over MKL SpMM... beats ASpT and
  J-Stream") entirely in SpMM terms; SDDMM may be secondary.
- Gale et al. SC'20's precise timing protocol (warmup/rep/statistic) was
  not recovered at all from the arXiv extract; needs a PDF-level re-read
  or a check of the (separately maintained) Sputnik GitHub repo's
  benchmark harness before being cited beyond "uses CUDA events, V100,
  27% of peak."
  Note also: the `artifact_url` listed in the track input
  (`luckylsk34/Sparse-Kernels`) is a third-party mirror/fork, not the
  original `google-research/sputnik` repo — worth flagging for the
  artifact-verification pipeline, not just this survey.
- HP-SpMM-SDDMM: does an SDDMM benchmark exist anywhere in the repo's
  git history / a different branch, or was the SDDMM half of the paper
  never open-sourced? Needs a `gh api repos/.../branches` and
  `gh api repos/.../commits?path=...` check before concluding the artifact
  is genuinely SpMM-only.
- Insum (ASPLOS'26): re-survey once the conference (2026) has occurred and
  either fulltext or a fuller artifact is published; currently zero
  benchmark facts are recoverable.
- distributed_sddmm/HnH's min-time dynamic-repetition protocol needs an
  explicit `min_time` value (not visible in the excerpted benchmark
  function signature — passed in from a caller not read) before it can be
  turned into a concrete spec parameter for a possible future
  distributed-SDDMM variant.
