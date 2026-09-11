# SpMV track — evaluation methodology survey

Track input: `data/track_inputs/spmv.json`, 20 papers. 10 surveyed below in
depth (5 via arXiv fulltext, matched against 6 code-verified artifact-repo
benchmark scripts read with `gh api` — some papers used both), well above the
5-paper / 6-paper minimum. The remaining 10 are used only for their abstracts
to cross-check the "Divergences" section; their claimed speedups are noted but
not treated as verified methodology.

Operation surveyed throughout: `y[M] = A[M,K](sparse) * x[K](dense)`,
occasionally embedded inside a solver loop (AMG V-cycle, Krylov sequence) or
specialized to Top-K similarity search / symmetric storage.

---

## 1. Niu, Li, Duan, Jiang, Tang — TileSpMV (IPDPS'21)

- key: `conf/ipps/NiuLDJ0T21`, artifact: github.com/SuperScientificSoftwareLaboratory/TileSpMV
- **workloads**: "all matrices in the SuiteSparse Matrix Collection" (~2,757 at
  time of writing), no explicit nnz filter found in the repo. Values in the
  timing harness are synthetic (`csrValA[i]=i%10`, `x[i]=i%10`), not the
  matrix's real numeric values.
- **timing protocol** (code-verified, `src/tilespmv_cuda.h`, `src/main.cu`):
  `WARMUP_NUM=200` warmup launches, then a 4×`BENCH_REPEAT` stabilization pass,
  then a **1000-iteration** timed loop (`BENCH_REPEAT=1000`), mean divided by
  `BENCH_REPEAT`. Timer: `gettimeofday` wrapped around kernel launch +
  `cudaDeviceSynchronize()`.
- **timing scope**: kernel-only — H2D copy done once before the loop, D2H once
  after; each iteration re-zeroes `y` via `cudaMemset`. Format conversion
  (CSR→Tile, `Tile_create`) is **never timed at all** — no `gettimeofday`
  brackets it anywhere in `main.cu`, so construction cost is invisible to the
  artifact, not merely excluded-and-reported.
- **precision & correctness**: fp64 (`MAT_VAL_TYPE=double`). Correctness:
  relative check `|y_golden[i]-y[i]| > 0.01*|y[i]|` — a **1% relative
  tolerance**, looser than most other surveyed papers.
- **metric**: `GFLOPS = 2*nnz*1e-6/time_ms` (standard 2 flops/nnz).
- **baselines**: cuSPARSE BSR-SpMV, cuSPARSE CSR-SpMV, CSR5 (all v11.1-era);
  claims up to 426.6x/3.96x/2.61x respectively.
- source: repo `src/main.cu`, `src/tilespmv_cuda.h`, `src/common.h`,
  `src/Makefile`; paper text via web-search summary (PDF itself would not
  render through WebFetch).

## 2. Du, Li, Wu, Li, Tan, Sun — AlphaSparse (SC'22)

- key: `conf/sc/DuLWLTS22`, arXiv:2212.10432 (fulltext, text-extracted via
  pypdf), artifact: github.com/AnonymousRepo123/AlphaSparse (mirrored at
  PAA-NCIC/AlphaSparse)
- **workloads**: 843 SuiteSparse matrices, filtered: rows > 9K, nnz ∈
  [50K, 60M], no empty rows.
- **timing protocol** (code-verified, `code_builder.cc`): a calibration run of
  `pre_repeat_num=2000` iterations estimates per-iteration time, then
  `repeat_num=(8000/time_ms)*2000` is computed so the real timed run totals
  **~8 seconds wall time** — i.e. repeat count adapts to matrix speed rather
  than being fixed. GFLOPS = `2*nnz*repeat_num/(timeuse_us/1e6)/1e9`. Timer:
  `gettimeofday`.
- **timing scope**: kernel-only (H2D/`cudaDeviceSynchronize()` before
  `gettimeofday(&start)`, D2H after `gettimeofday(&end)`). **Format+kernel
  search time is entirely excluded from the reported GFLOPS** and can take
  **up to 8 hours per matrix** (search capped at 8h; average 3.2h with
  pruning, 8.0h without, per the paper's Table III) — i.e. one-shot
  construction cost can be >1000x a single SpMV call's runtime, with **no
  amortization argument** offered anywhere in the text.
- **precision & correctness**: fp32, explicitly stated ("single-precision for
  floating-point values in experiments"); `x` initialized to constant 100,
  `y` to 0. No explicit numeric tolerance found in the extracted text.
- **metric**: GFLOP/s as above.
- **baselines**: ACSR (self-implemented), CSR-Adaptive (ViennaCL 1.7.1), CSR5,
  Merge-based CSR, HYB (cuSPARSE 9.2) as "artificial formats"; PFS
  (format-selector autotuner); TACO. Hardware: A100 and RTX 2080.
- source: arXiv fulltext (2212.10432, pypdf-extracted); repo `code_builder.cc`,
  `executor.cc`, `spmv_header_top.code`, `README.md`.

## 3. Qiu, Xu, Fang, Zhang, Duan, et al. — DCS-SpMV (SC'24)

- key: `conf/sc/QiuXFZD0WCC024`, artifact: github.com/qleonardo/DCS-SpMV
- **workloads**: 318 large numerically-symmetric SuiteSparse matrices (rows >
  50,000, nnz > 100,000), per the Artifact-Description appendix. The repo's
  `matrix_list.csv` enumerates 305 rows — a minor discrepancy against the
  paper's stated 318.
- **timing protocol** (code-verified, `common.h`, ADAE appendix text): each
  implementation run **100 times per matrix**, averaged GFLOPS reported
  (`BENCH_REPEAT=100`). Timer: `omp_get_wtime()`.
- **timing scope**: preprocessing (DC-tree construction + row reordering, or
  MKL handle creation) is timed **separately** into `pre_time_add` and
  excluded from the headline averaged GFLOPS; no explicit amortization
  discussion.
- **precision & correctness**: fp64 (`VALUE_TYPE=double`). Correctness:
  combined relative+absolute check, `fabs(diff) > 1e-5*max(|ref|,|y|) AND
  fabs(diff) > 1e-5`.
- **metric**: `GFLOPS = 2*nnz/(1e6*time_ms)`; also a bandwidth figure
  (`dataSize/(1e6*time_ms)` GB/s) via CSR-array byte accounting.
- **baselines**: naive OpenMP CSR-SpMV, Intel oneMKL SpMV, oneMKL symmetric
  SpMV, RACE (external library). Hardware: Intel Xeon Gold 6258R (28c, x86)
  and HUAWEI Kunpeng 920 (32c, ARM); threads pinned to physical cores.
- source: Artifact-Description appendix (pypdf-extracted); `common.h`,
  `MKL_SpMV.h`, `CSR_SpMV.h`, `DC_SymSpMV.h`, `DC_Hybrid_SymSpMV.h`,
  `matrix_list.csv`, `profiling.cpp`.

## 4. Cong, Xing — CB-SpMV (ICS'25)

- key: `conf/ics/CongSC00Q25`, arXiv:2605.18515 (fulltext — resolves; "today"
  is 2026-08-06 so a May-2026 id is not future-dated), artifact:
  github.com/xing-cong/CB-Sparse
- **workloads**: 2,843 SuiteSparse matrices (paper text); repo ships only 3
  toy examples (`1138_bus.mtx`, `heart2.mtx`, `sme3Da.mtx`) for quick-start,
  no size/nnz range stated in the extracted text.
- **timing protocol** (code-verified, `cb-spmv/src/cb-spmv.cuh` L372-402): one
  untimed correctness-check launch (doubles as warmup), then `ITER=10`
  (macro in `macros.h`) timed launches wrapped in `cudaEventRecord`, mean =
  `elapsed/ITER`. **The paper's own text claims "each kernel executed 1000
  times"** — a real discrepancy between the released reproducer (`ITER=10`)
  and the paper's methodology section.
- **timing scope**: kernel-only. Format conversion (`coo2block_gather()`)
  runs once, outside the timed region — and its cost is **never printed,
  measured, or discussed anywhere** in the released code or extracted paper
  text; not even reported-separately, simply absent.
- **precision & correctness**: fp64. Validation (`utils.h`):
  `abs(y_computed - y_ref) < abs_tol + rel_tol*|y_ref|` with
  `abs_tol=1e-3, rel_tol=1e-4`.
- **metric**: `GFLOPS = 2.0*nnz*1e-6/runtime_ms`. The cuSPARSE-BSR baseline
  itself sweeps block sizes {2,4,8,16} and reports only the **best** — the
  baseline number is already a best-of-4-configs result before comparison.
- **baselines**: cuSPARSE-BSR v12.4, TileSpMV, DASP; claims up to 3.95x/10.34x
  over TileSpMV/cuSPARSE-BSR on A100.
- source: arXiv 2605.18515 (`/abs/`, `/html/`, `/pdf/` all resolved); repo
  `README.md`, `cb-spmv/README.md`, `cb-spmv/src/{main.cu,cb-spmv.cuh,
  utils.h,macros.h}`.

## 5. Chen, Sung, So, Talati, Blaauw, et al. — Bit-GraphBLAS (IPDPS'22)

- key: `conf/ipps/ChenSSTBL22`, arXiv:2201.08560 (fulltext via
  ar5iv.labs.arxiv.org), artifact: github.com/hsung2/Bit-GraphBLAS
- **workloads**: all 521 binary square matrices in SuiteSparse — rows/cols 2
  to 214,005,017, nnz 2 to 11,588,725,964; classified into 6 structural
  pattern types (Dot, Diagonal, Block, Stripe, Road, Hybrid).
- **timing protocol** (code-verified, `bmv_eval/test_spmv_baseline_a100.cu`,
  `Makefile`): `TEST_TIMES=5` — "average of 5 runs" per the paper. Timed with
  a CUDA-event wrapper (`GpuTimer`); `beta=1.0` is deliberately set when
  `TEST_TIMES>1` (accumulating into `y`) to prevent the compiler/driver from
  short-circuiting repeated identical calls. No separate warmup pass visible.
- **timing scope**: kernel-only. Format conversion (CSR→B2SR) is measured
  and reported **separately**: 3-34 ms, explicitly called a "one-time cost"
  that is "greatly amortized" — one of the few surveyed papers to make an
  explicit amortization argument, distinguishing "kernel" time from full
  multi-iteration "algorithm" (graph-traversal) time in its tables.
- **precision & correctness**: 1-bit matrix representation, fp32 output
  vector. Baseline is `cusparseScsrmv()` (fp32 CSR).
- **metric**: speedup = baseline_time/B2SR_time vs. matrix density; no
  explicit GFLOP/s formula. Claims up to 40x for SpMV kernels alone (6555x
  for SpGEMM).
- **baselines**: cuSPARSE (`cusparseScsrmv`). Hardware: GTX1080, TITAN V,
  CUDA 10.0.
- source: arXiv fulltext (2201.08560, via ar5iv); repo
  `bmv_eval/{Makefile,testbmv.sh,test_bin_bin_full.cu,
  test_spmv_baseline_a100.cu}`.

## 6. Lu, Zhang, Wang, Fu, Liu, Chen, Yuan, Chen — AmgT (SC'24)

- key: `conf/sc/LuZWFLCY0C024`, artifact:
  github.com/SuperScientificSoftwareLaboratory/AmgT
- **workloads**: 16 hand-picked "representative" SuiteSparse matrices
  (thermal/CFD/structural/power-network domains), e.g. spmsrtls (29,995
  rows), ldoor (952,203 rows, 46,522,475 nnz) — a much smaller, curated set
  than the sweep-style papers above.
- **timing protocol** (code-verified, `AmgT_test/test_new.c`): full AMG
  V-cycle, max 50 iterations (no convergence cutoff); SpMV invoked 31x per
  iteration (30 across 6 levels + 1 residual) → 1,551 SpMV calls total per
  run. Wall-clock via `gettimeofday` around whole
  `HYPRE_BoomerAMGSetup`/`Solve` calls, but internally instrumented with
  fine-grained counters (`time_spmv_sum`, `time_spmv_preprocess`,
  `spmv_times`, `time_spgemm(_preprocess)`, `csr2bsr_step{1,2,3}`,
  `bsr2csr_step{1,2,3}`) printed at the end.
- **timing scope**: setup-phase (SpGEMM) vs. solve-phase (SpMV) explicitly
  separated. Format conversion (CSR↔mBSR, invoked 2x#levels-1 times) is
  measured and reported separately (Fig. 10): "generally...longer than 5% of
  overall execution time" — an honest, non-trivial disclosed overhead, not
  folded silently into headline speedups.
- **precision & correctness**: fp64 baseline; mixed-precision variant uses a
  **level-dependent cascade** — fp64 at the finest level, fp32 at the second
  level, fp16 at remaining coarser levels (not a single tensor-core dtype).
  No explicit residual/convergence tolerance reported; the paper states it
  focuses on demonstrating performance, not solver-correctness guarantees.
- **metric**: speedups only. SpMV alone: 1.34x/1.19x avg (up to 2.21x/2.09x)
  vs cuSPARSE on A100/H100; 2.92x avg (up to 6.70x) vs rocSPARSE on MI210.
  Full solve-phase: 1.24x/1.13x avg vs HYPRE.
- **baselines**: HYPRE v2.31.0-GPU calling cuSPARSE v12.2 (NVIDIA) /
  rocSPARSE v6.1.2 (AMD MI210).
- source: repo `README.md`, `AmgT_test/test_new.c`; author-hosted paper PDF
  (www.ssslab.cn, no arXiv exists) text-extracted via proxy after direct
  WebFetch failed on the binary PDF.

## 7. Song, Chi, Guo, Cong — Serpens (DAC'22)

- key: `conf/dac/SongCGC22`, arXiv:2111.12555 (fulltext via ar5iv; note the
  track-input arXiv field 2103.04808 actually belongs to the Top-K SpMV
  paper below, a data-entry mismatch), artifact: github.com/UCLA-VAST/Serpens
- **workloads**: two suites — (a) 12 large matrices from SNAP/OGB/SuiteSparse,
  45K-2.45M vertices, up to 124M edges (e.g. googleplus, crankseg_2,
  Si41Ge41H72); (b) 2,519 SuiteSparse matrices filtered to nnz ∈
  [1,000, 100M], geomean density 1.4e-3.
- **timing protocol** (code-verified, `src/serpens-host.cpp`): the FPGA
  kernel is invoked once per matrix, but internally repeats `rp_time=100`
  times on-device; the host divides total `tapa::invoke` wall time by
  `rp_time` — i.e. amortized-over-100 mean, not host-side relaunch
  repetition. No explicit warmup step or best/median reporting. GPU baseline
  uses `cudaEventElapsedTime`; CPU reference uses
  `std::chrono::steady_clock`.
- **timing scope**: the measured value is explicitly labeled "**Kernel
  time**" in code — buffer construction, CSR→CSC conversion, and per-PE
  edge-list generation happen host-side, untimed, before `tapa::invoke`;
  PCIe/HBM transfer inclusion is not explicitly discussed in the paper text
  (ambiguous, unlike the "Kernel time" label's clarity about compute).
- **precision & correctness**: fp32 throughout. Correctness is a **loose
  two-level tolerance**: per-element relative error
  `|cpu-fpga|/(min(|cpu|,|fpga|)+1e-4) > 1e-4` flags a mismatch, and the run
  "passes" if **fewer than 2% of elements mismatch** — markedly more
  permissive than an elementwise numeric gate.
- **metric**: `GFLOPS = 2*(nnz+M)/1e9/time` (2 flops/nnz plus the beta*y
  axpy term); also MTEPS (`nnz/time`) and "bandwidth efficiency"
  (throughput/peak-bandwidth).
- **baselines**: K80 GPU (480GB/s, 130W), GraphLily (285GB/s, 43W), Sextans
  (417GB/s, 52W). Claims 2.10x over K80 (2,519-matrix suite), 1.91x/1.76x
  over GraphLily/Sextans (12-matrix suite).
- source: repo `src/serpens-host.cpp`, `README.md`; arXiv fulltext
  (2111.12555, via ar5iv) for dataset/baseline specs.

## 8. Parravicini, Cattaneo, Sironi, Santambrogio, et al. — Top-K SpMV FPGA (DAC'21)

- key: `conf/dac/ParraviciniCSS21`, arXiv:2103.04808 (fulltext via ar5iv),
  artifact: github.com/AlbertoParravicini/approximate-spmv-topk
- **workloads**: mostly **synthetic** matrices (`test_spmv_topk.py`): rows ∈
  {5M, 10M, 15M}, embedding cols ∈ {512, 1024}, avg nnz/row ∈ {20, 40},
  distributions {uniform, left-skewed Γ}, K=100 top-k, plus one real matrix
  (sparsified GloVe embeddings, 2.196M rows x 300 cols, 54.9M nnz).
- **timing protocol** (code-verified, `test_spmv_topk.py`): `NITER=30`
  repetitions per config; no explicit warmup flag in the harness. Repeated
  across FPGA precision builds (`24core_219mhz_32bit`,
  `24core_235mhz_26bit`, `32core_172mhz_21bit`, `24core_229mhz_float`), GPU
  variants (CSR/COO x fp16/fp32), and a Python CPU baseline
  (`sparse_dot_topn`).
- **timing scope**: reduced-precision packet-wise CSR compression is a
  build-time/host preprocessing step; not clearly stated whether it (or
  host<->FPGA DMA) is inside the timed window — unlike Serpens, no explicit
  "Kernel time" label was found in the reviewed material.
- **precision & correctness**: reduced fixed-point precision is the paper's
  headline technique — tested at 20-bit (Q1.19), 25-bit (Q1.24), 32-bit
  fixed (Q1.31), and 32-bit float. Correctness is evaluated with
  **ranking-quality metrics** (Precision, Kendall's tau, NDCG) against an
  exact CPU baseline, not elementwise numeric tolerance — appropriate since
  the output is an approximate top-K index set, not a full `y` vector.
- **metric**: execution time / speedup; a roofline-model plot shows bandwidth
  efficiency (GB/s attained vs. peak).
- **baselines**: CPU = 2x Intel Xeon Gold 6248, multithreaded
  `sparse_dot_topn`; GPU = Tesla P100 (549GB/s), cuSPARSE+Thrust radix sort.
  Claims 100x vs CPU, 2x vs GPU with 20% higher bandwidth, 14.2x
  power-efficiency.
- source: repo README + `test_spmv_topk.py`; arXiv fulltext (2103.04808, via
  ar5iv).

## 9. Zhang, Liu, Yao, Duan, Yin, Wu — FBMPK (IPDPS'23)

- key: `conf/ipps/ZhangLYDYLW23`, artifact: github.com/Fliange/FBMPK
- **workloads**: SuiteSparse matrices via Matrix Market; repo ships
  `audikw_1.zip` (real SuiteSparse structural matrix, pre-reordered by ABMC)
  as the canonical verification example. Full paper-level matrix list was
  not recoverable — both the ACM/IEEE-hosted PDF and a White Rose OA copy
  used font encodings that defeated stream-level text extraction; the
  reported "CPU + ARM" platform claim also could not be independently
  confirmed (the Makefile shows only an x86 g++/icc build path).
- **timing protocol** (code-verified, `ABMC_version.cpp`): `NTIMES=30`
  explicit macro — the loop runs the **full k-step sequence** 30 times,
  arithmetic mean reported (`baseline_time/NTIMES`, etc.); no separate
  warmup exclusion (the first of the 30 is included in the average). Timer:
  `mytimer()` → `gettimeofday`.
- **timing scope — the key divergence of this paper**: this is a SEQUENCE
  workload — `Ax, A²x, ..., A^k x` and linear combinations, not a single
  SpMV call. The measured region is only the SpMV-sequence compute (baseline
  loop / forward-backward ABMC kernel / MKL loop over `power_k` calls). The
  ABMC graph partitioning/coloring/reordering **preprocessing is excluded
  entirely** — the README states outright "This part of the code does not
  include preprocessing," and the repo ships a pre-reordered matrix
  specifically so users skip that cost. Unlike the format-conversion papers
  above (which at least measure-and-report-separately or amortize), FBMPK's
  public artifact simply drops this cost from anything that gets timed.
  Reported time is the **total time for the whole k-call sequence** (mean
  over 30 trials), not divided by k.
- **precision & correctness**: fp64 (`VALUE_TYPE=double`,
  `ABMC_MPK.h`). Correctness (`ResultCheck.h`): combined absolute+relative
  tolerance `EPS=1e-6`, gated by a `-DRESULTCHECK` compile flag that is
  **off by default**.
- **metric**: raw wall-clock time (seconds) for baseline vs. MKL vs.
  forward-backward (ABMC) vs. ablation (no-BtB) variants; no GFLOP/s formula
  found in the reachable code.
- **baselines**: Intel MKL (`mkl_sparse_d_mv`), naive per-call SpMV loop
  (with a Kahan-summation ablation).
- source: repo `README.md`, `ABMC_version.cpp`, `ABMC_MPK.h`,
  `spmv_MKL.h`, `ResultCheck.h`, `ABMCpre.h`. Paper-level facts (matrix
  suite, ARM specs, k values tested) are unverified — both PDF sources
  fetched but unreadable; flagged as a gap, not guessed.

## 10. Jain, Gaikwad, Chennupati — RASSM (ASPLOS'25) — negative finding

- key: `conf/asplos/JainGC25`, artifact: github.com/gt-tinker/RASSM
- The paper's abstract motivates "Single-Sparse-Matrix Kernels (SSMKs) such
  as SpMM, SDDMM, SpMV, and SpTS," but **the public artifact does not
  implement SpMV or SpTS at all**: `code/src/main.cpp`'s kernel dispatch only
  branches on `kernel=="spmm"` or `"sddmm"`, `code/include/` has only
  `spmm/` and `sddmm/` subdirectories, and a repo-wide search for "spmv"
  returns zero hits. RASSM is therefore excluded from this track's spec —
  it is already surveyed in the SpMM track (`benchspecs/spmm/survey.md`
  entry #12), where its residue-based adaptive-tiling method actually has
  code. Recorded here only so a future reader doesn't re-attempt this
  paper for SpMV and hit the same dead end.

---

## Divergences

- **Preprocessing/format-conversion treatment splits into (at least) four
  distinct practices, not two.** (a) Never timed or discussed at all —
  TileSpMV's CSR→Tile conversion and CB-SpMV's `coo2block_gather()` are
  invisible to their artifacts, not merely excluded-and-reported. (b) Timed
  but reported separately as a modest one-time cost — DCS-SpMV's DC-tree
  construction, Bit-GraphBLAS's CSR→B2SR conversion (3-34ms, explicit
  amortization argument), AmgT's CSR↔mBSR conversion (>5% of runtime,
  disclosed). (c) Timed but excluded *entirely* from anything reported —
  FBMPK's ABMC reordering is not even given a "preprocessing" line item; the
  public artifact ships a pre-reordered matrix so the cost never appears.
  (d) An open-ended autotuning/design SEARCH, qualitatively different in
  scale from (a)-(c) — AlphaSparse's per-matrix format+kernel search can
  take up to 8 hours, dwarfing any plausible per-call amortization; the
  paper reports zero amortization argument for this. The spec (Step 2) keeps
  these as separate concerns: mechanical format conversion goes into the
  `spmv-e2e-preproc` variant's amortized number, while unbounded
  autotuning/design search is flagged separately rather than blended in,
  since no single realistic `k` makes an 8-hour search "fair" the way a
  34ms format conversion amortizes trivially.
- **Timer/repetition rigor varies by roughly an order of magnitude in rep
  count, and several artifacts under-warm or don't separate warmup from
  the timed loop at all.** TileSpMV: 200 warmup + 1000 timed (mean).
  AlphaSparse: adaptive rep count targeting ~8s wall time (mean).
  DCS-SpMV: 100 reps (mean). CB-SpMV: code has `ITER=10` while the *paper
  text* claims 1000 — a genuine paper-vs-artifact discrepancy. Bit-GraphBLAS:
  `TEST_TIMES=5` (mean) — the thinnest rep count of any surveyed GPU paper.
  Serpens: on-device `rp_time=100`, amortized mean, no host-side repetition.
  Top-K SpMV: `NITER=30`, no warmup flag found. FBMPK: `NTIMES=30`,
  explicitly **no warmup exclusion** — the first (cold) iteration is folded
  into the mean along with the other 29. None of the 9 surveyed
  SpMV-implementing papers report median+min/max; all report only a mean
  (or an amortized/divided mean). The spec fixes this: uniform warmup=20,
  reps=100 with per-iteration timestamps, statistic=median with min/max/
  stdev also reported — stricter than the thinnest artifacts (Bit-GraphBLAS,
  FBMPK) and providing a real distribution instead of every surveyed paper's
  mean-only reporting.
- **Correctness tolerance spans four orders of magnitude, and is
  frequently opt-in / disabled by default.** Tightest: FBMPK's `1e-6`
  absolute+relative (but gated behind a `-DRESULTCHECK` flag that is *off*
  in the default build) and DCS-SpMV's `1e-5`. Middling: CB-SpMV's
  `abs=1e-3, rel=1e-4`. Loosest: TileSpMV's flat **1% relative** tolerance
  and Serpens's "**<2% of elements may mismatch** at 1e-4 relative" — a
  pass/fail criterion on the *count* of bad elements, not a bound on any
  single element's error. AlphaSparse states no explicit tolerance in the
  extracted text at all. Top-K SpMV departs from elementwise checking
  altogether, using ranking-quality metrics (Kendall's tau, NDCG) since its
  output is an approximate top-K index set. The spec fixes a single
  tolerance per precision (1e-9 relative for fp64, 1e-4 relative for fp32,
  both vs. an fp64 reference), gated as a hard pass/fail before any timing
  counts — tighter than 3 of the 9 surveyed papers' own practice.
- **The "shape" of the workload itself diverges, not just its parameters.**
  Eight of the nine SpMV-implementing papers benchmark a single isolated
  `y = Ax` call. FBMPK instead optimizes a *bounded sequence* of calls on
  the same matrix (`Ax, A²x, ..., A^k x`), reporting only total-sequence
  time, never per-call. AmgT embeds SpMV inside a real AMG solve (31 calls
  per V-cycle x up to 50 iterations = up to 1,551 calls), separately
  instrumenting per-call SpMV time via internal counters. This is different
  enough from single-call throughput (cache/reuse effects across the
  sequence, and reordering that pays off only over multiple calls) that
  folding it into the single-call kernel variant would misrepresent both:
  the spec adds a dedicated `spmv-sequence-krylov` variant.
- **Symmetric-matrix SpMV is a genuinely different computational kernel,
  not a data-selection subset of general SpMV.** DCS-SpMV halves storage
  (numerically symmetric matrices only) but must resolve write-write
  conflicts on the output vector across threads — a correctness dimension
  (race-freedom, not just numeric tolerance) that none of the general-SpMV
  papers need to address, and its GFLOP/s accounting differs (flops per
  *stored* nonzero vs. flops per full/unfolded nonzero). The spec keeps this
  as its own variant (`spmv-symmetric-kernel`) rather than a subset filter
  of `spmv-csr-kernel`, and mandates full-unfolded-nnz flop counting so the
  two variants' GFLOP/s numbers stay honestly non-comparable rather than
  superficially matching.
- **Precision splits roughly evenly and is not always disclosed as a single
  axis.** fp64: TileSpMV, DCS-SpMV, CB-SpMV, FBMPK, AmgT's baseline. fp32:
  AlphaSparse, Bit-GraphBLAS's baseline, Serpens. Reduced/mixed: Top-K
  SpMV's fixed-point sweep (20/25/32-bit), AmgT's level-dependent
  fp64/fp32/fp16 cascade inside one solve. The mixed/cascade precision
  patterns (AmgT, Top-K SpMV) don't fit a single-precision-per-call model at
  all and are left as an open question rather than forced into the fp32/fp64
  split (see `open_questions` in spec.yaml).
- **FPGA papers leave host<->accelerator transfer scope ambiguous except
  when explicitly labeled.** Serpens's code explicitly labels its measured
  region "Kernel time" (buffer construction untimed, host-side), making the
  scope unambiguous. Top-K SpMV's harness has no equivalent label, and
  neither the paper text nor the reviewed script confirms whether PCIe/HBM
  DMA is inside the timed window. The spec requires FPGA implementations to
  report kernel-only time (device-resident operands, matching Serpens's
  practice) with host-DMA measured and reported as a separate, explicit
  number — never left ambiguous.
- **"Up to Nx" headline-only reporting is common** in this track's
  abstracts even among the fully-surveyed set: TileSpMV "up to 426.6x",
  Bit-GraphBLAS "up to 40x/6555x", CB-SpMV "up to 3.95x", DAC/Top-K-SpMV
  "100x vs CPU". Only DCS-SpMV and AmgT report both average and max
  speedups across their matrix set in the reviewed text. The spec requires
  reporting the median and full range (min/max) across the fixed
  `recommended_subset`, not a single best-case number.
- **Input suites cluster into three unrelated selection philosophies with no
  shared inclusion rule stated by any paper**: (1) "the whole SuiteSparse
  collection" with no filter (TileSpMV, near-exhaustive Bit-GraphBLAS
  binary-matrix subset); (2) a moderate nnz-range filter yielding hundreds
  to low-thousands of matrices (AlphaSparse: 843, DCS-SpMV: 318 symmetric,
  CB-SpMV: 2,843, Serpens: 2,519); (3) a small hand-curated set of ~10-16
  "representative" matrices chosen for domain relevance (AmgT: 16;
  Top-K SpMV: mostly synthetic + 1 real). No paper states a principled
  cross-paper inclusion criterion — each picks its own filter and matrix
  count independently. The spec's `recommended_subset` unions matrices
  actually named across ≥1 code-verified surveyed artifact.

## Recommended-subset provenance

Matrices proposed for `spec.yaml`'s `recommended_subset` are the union of
named instances that recur across ≥1 code-verified surveyed artifact or
appear in AmgT/Serpens/FBMPK's own curated lists: `audikw_1` (FBMPK's
canonical example, also an AmgT-adjacent structural matrix), `ldoor`, `Serena`,
`StocF-1465`, `Bump_2911`, `Flan_1565`, `Hook_1498`, `Geo_1438`,
`dielFilterV3real`, `Long_Coup_dt6`, `Cube_Coup_dt6` (large symmetric FEM/CFD
matrices, matching AmgT's and DCS-SpMV's domain and the size range both
target), `crankseg_2` (Serpens's 12-matrix suite), `cage15`, `delaunay_n24`,
`roadNet-CA`, `hugetrace-00020`, `wiki-Talk`, `soc-LiveJournal1`, `com-Orkut`
(irregular/graph-like matrices at the low-density end, matching Serpens's and
TileSpMV's SuiteSparse-wide sweeps), `webbase-1M`, `in-2004`, `nlpkkt240`,
`Queen_4147` (large irregular matrices recurring in general SpMV literature),
`mip1`, `pwtk` (small/medium matrices that recur across the SpMM track's
survey too, useful for cross-track sanity checks).
