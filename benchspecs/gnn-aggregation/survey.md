# Survey: gnn-aggregation

Track: SpMM-like GNN neighbor-aggregation / scatter-gather kernels (11 papers).
All 11 papers were surveyed (exceeds the 5-paper minimum); sources below are
code-level (artifact repo README + benchmark/driver source read via `gh api`)
for 10/11 papers, and arXiv abstract + repo README for the two papers whose
arXiv abstract page did not expose fulltext (ar5iv/arxiv.org/html both 404'd
for these two IDs; only the abstract-page fields were retrievable via
WebFetch) and whose repos do not expose a bare timing loop within the file
budget of this survey (StraGCN).

---

## CBM — "Accelerating GNNs Using a Novel Computation-Friendly Matrix
Compression Format" (IPDPS'25) — `conf/ipps/AlvesMBFGR25`

- **Platform**: CPU only (shared-memory, OpenMP; Intel oneAPI MKL used as the
  CSR baseline). Serial and parallel (thread-count sweep) settings.
- **Workloads**: `ca-HepPh`, `ca-AstroPh` (SNAP collaboration networks),
  `Cora`, `PubMed` (Planetoid), `COLLAB` (TUDataset), `coPapersCiteseer`,
  `coPapersDBLP` (SuiteSparse), `ogbn-proteins-raw` (OGB). 8 named graphs,
  no stated size-based inclusion rule — just "the datasets we had."
- **Operation variants benchmarked**: `A@X` (plain adjacency), `A@D^-1/2@X`,
  `D^-1/2@A@D^-1/2@X` (full GCN normalization) — each in both CBM and CSR/MKL
  format, i.e. the paper treats the *normalized* aggregation as the
  first-class benchmark target, not bare adjacency SpMM.
- **Timing protocol** (`benchmark/benchmark_matmul.py`): `--warmup 10`
  (default) + `--iterations 250` (default, script overrides to 250);
  `time.time()` (Python wall clock) around each `a.matmul(x, y)` call;
  reports **mean, std, min, max** over the post-warmup iterations — the most
  complete statistic reporting of any paper in this track.
- **Timing scope**: kernel-only per call; format conversion to CBM (the
  "compression" step) is timed SEPARATELY by a dedicated script
  (`compression_metrics.sh` → reports compression time + compression ratio
  vs CSR), never mixed into the matmul-loop numbers. This is the cleanest
  separation of preprocessing vs steady-state kernel time in the track.
- **Dense operand**: N=500 columns (fixed), fp32 (`torch.rand`, i.e. U(0,1)
  not U(-1,1)), threads swept explicitly ({1, 16} in the shipped script,
  `MAX_THREADS`/`GOMP_CPU_AFFINITY` pinned).
- **Correctness**: separate `validate.sh` / `benchmark/validate.py` compares
  CBM output element-wise against MKL/CSR output with explicit `--rtol`/
  `--atol` CLI flags — run as its own pass, not concurrently with the timed
  benchmark.
- **Metric**: raw time (seconds), NOT GFLOP/s — the paper does not normalize
  by nnz/N in its own harness output.
- **Source**: `github.com/cbm4scale/CBM-IPDPS25` — README.md,
  `benchmark/benchmark_matmul.py`, `benchmark/utilities.py`,
  `scripts/matmul.sh`, `scripts/inference.sh`.

---

## StraGCN — "GPU-Accelerated Strassen's Sparse-Dense Matrix Multiplication
for GCN Training" (SC'25) — `conf/sc/HeLDL0M025`

- **Platform**: GPU (evaluated on RTX 4090; CPU host Intel Xeon Gold 5220).
- **Workloads**: graph datasets distributed as preprocessed `.npy` via a
  Zenodo record (`StraGCN-graphs.tar.gz`); exact dataset names not visible
  from the README (only download instructions, no manifest).
- **Baselines**: GNNAdvisor (`GNNA/`), PyG, DGL — all three vendored as
  subdirectories with their own `main.py`-style entry points
  (`gcn.py`, `pyg_main.py`, `GNNA_main.py`).
- **What is measured**: full GCN training (`python gcn.py` in each
  baseline's directory) — no isolated single-kernel benchmark script is
  exposed at the README level; the paper's own repository README does not
  document warmup/repetition counts or the timer used, unlike every other
  artifact surveyed here.
- **Note on source depth**: this is the one paper in the track surveyed only
  at the README/abstract level — the actual timing loop lives inside
  `StraGCN/gcn.py` / `dgl/gcn.py`, which was not read in this pass (marked
  as an open question below rather than guessed at).
- **Source**: `github.com/CGCL-codes/StraGCN` — README.md only; abstract.

---

## MaxK-GNN — "Extremely Fast GPU Kernel Design for Accelerating GNN
Training" (ASPLOS'24) — `conf/asplos/PengXSHZHKKD24`

- **Platform**: GPU (A100 80GB tested; requires compute capability ≥ 8.0).
- **Workloads — kernel-level**: a custom 24-graph benchmark suite (figure
  `24graphs.png` in repo, e.g. `reddit.dgl`; exact list not machine-readable
  from the README) distributed via Google Drive; **no inclusion criterion
  stated**.
- **Workloads — full training pipeline**: `flickr`, `ogbn_products`,
  `ogbn_proteins`, `reddit`, `yelp` (5 OGB/GraphSAGE-standard datasets),
  models `{sage, gcn, gin}`.
- **Operation**: this is NOT plain unstructured SpMM. Forward is a row-wise
  product SpGEMM against a MaxK-sparsified (top-k per row, "CBSR" format)
  feature matrix; backward is an outer-product SSpMM (sampled sparse-dense).
  k is swept over `{16, 32, 64, 96, 128, 192}` against a fixed
  `dim_origin=256`. This is a structured-sparsity variant of aggregation, not
  the dense-N-column SpMM most other papers in the track benchmark.
- **Timing protocol** (`kernels/spmm_base.h::timing_body`, shared by all
  kernel variants including the cuSPARSE and GNNAdvisor baselines compiled
  into the same binary): **4 warmup iterations + 4 measured iterations**,
  each measured iteration individually wrapped in
  `timestamp(t0); run(); cudaDeviceSynchronize(); timestamp(t1)` and
  averaged — no median/min/max, and only 4 samples. This is the weakest
  statistical protocol of any paper surveyed in this track.
- **Correctness**: a `check_err()` function exists (sum of abs error,
  threshold `err_sum/len < 0.001`) but in the shipped `main.cu::test_graph`
  the call site that invokes it is commented out — **the benchmark binary as
  shipped does not validate correctness during the timed run.**
- **Baselines**: cuSPARSE SpMM (`spmm_cusparse.cu`), GNNAdvisor
  (`spmm_gnna.cu`). Reported speedup: 6.93x/5.39x/2.55x/1.46x over cuSPARSE
  at k=8/16/32/64 (for degree>50 graphs); 9.57x/7.46x/3.55x/2.04x over
  GNNAdvisor.
- **Source**: `github.com/xiexi51/MaxK-GNN` — README.md, `kernels/main.cu`,
  `kernels/spmm_base.h`, `kernels/spmm_maxk.cu`.

---

## FASTEN — "Fast GPU-accelerated Segmented Matrix Multiplication for
Heterogeneous GNNs" (ICS'24) — `conf/ics/0001SLFY024`

- **Platform**: GPU (NVIDIA, compute capability ≥ 7.0).
- **Operation**: segmented matrix multiplication — grouped/batched GEMM where
  each segment corresponds to one relation type in a heterogeneous graph
  (RGCN/HGT/RGAT). This is the "gather-by-type, GEMM-per-segment" analogue of
  aggregation for HGNNs, not a 2-operand sparse-times-dense SpMM.
- **Workloads**: `AIFB`, `MUTAG`, `BGS`, `AM` (RGCN/RGAT relational
  benchmarks), `DBLP`, `Freebase` (HGT heterogeneous-graph benchmarks) — all
  small/medium standard relational-GNN benchmark graphs, not SuiteSparse.
- **Timing protocol** (`test/test_ops.py::test_perf`, uses Triton's `proton`
  profiler): explicit **warmup pass** (runs forward once to capture output,
  a **second warmup pass** "to trigger backward kernels" before backward is
  profiled), then a `proton.scope(...)` region records the timed call with
  `flops` metadata attached for automatic throughput computation. Swept over
  `phase in {forward, backward}`, `dtype`, `engine` (fasten vs baseline), and
  `K` (embedding width).
- **Baselines**: CUTLASS grouped GEMM, cuBLAS (looped per-segment GEMM).
  Reported average speedup: 13.65x vs CUTLASS, 4.72x vs cuBLAS
  (operator-wise, i.e. kernel-only, not end-to-end model time).
- **Correctness**: not visible in the perf-test file itself (separate
  correctness assertions likely live in `test_op.py`'s non-perf tests, not
  re-examined here).
- **Source**: `github.com/Deep-Learning-Profiling-Tools/fasten` — README.md,
  `test/test_ops.py`.

---

## TC-GNN — "Bridging Sparse GNN Computation and Dense Tensor Cores on
GPUs" (USENIX ATC'23) — `conf/usenix/WangFWHD23`

- **Platform**: GPU, requires sm ≥ 80 (Ampere, e.g. RTX 3090).
- **Workloads**: 14 named graphs spanning citation graphs (`citeseer`,
  `cora`, `pubmed`, `ppi`) through protein/molecule graphs (`PROTEINS_full`,
  `OVCAR-8H`, `Yeast`, `DD`, `YeastH`) to large SNAP-style graphs
  (`amazon0505`, `artist`, `com-amazon`, `soc-BlogCatalog`, `amazon0601`) —
  this is (a superset intersecting with) the "GNNAdvisor benchmark suite"
  reused, with small variations, by several other papers below.
- **Timing protocol — epoch-level** (`main_tcgnn.py`): **9-epoch dry run**
  (untimed, `for epoch in range(1,10): train()`), then
  `torch.cuda.synchronize()` + `time.perf_counter()` bracket around
  `args.epochs=200` full training epochs (forward+backward+optimizer step);
  reports `train_time*1e3/epochs` — i.e. **ms/epoch, full training**, not an
  isolated kernel time.
- **Timing protocol — single-kernel** (separate scripts
  `0_run_tcgnn_single_kernel.sh`, `2_tcgnn_single_kernel.py`,
  `3_cnt_TC_blk_SpMM.py`/`3_cnt_TC_blk_SDDMM.py`): TC-GNN is one of the few
  papers in the track that explicitly ships BOTH an epoch-level model
  benchmark AND an isolated single-SpMM/SDDMM-kernel benchmark, plus a
  "dense-tile-count" script measuring the tensor-core tile utilization of
  its sparse→dense-tile graph translation.
- **Preprocessing note**: the sparse→dense-tile-format graph translation
  (the paper's core contribution for TCU-mapping) is a real one-shot cost;
  none of the shipped benchmark scripts report ITS latency — only the
  resulting tile COUNT (`3_cnt_TC_blk_*.py`), not translation time. The
  translation cost is silently excluded from both the epoch-level and
  single-kernel numbers.
- **Baselines**: DGL (`dgl_baseline/`), PyG (`pyg-baseline/`), cuSPARSE
  bSpMM (`TCGNN-bSpmm/cusparse`), tSparse, Triton — six separate baseline
  harnesses, each producing its own CSV, reflecting a fairly rigorous
  multi-baseline comparison design.
- **Source**: `github.com/YukeWang96/TC-GNN_ATC23` — README.md,
  `main_tcgnn.py`, `1_bench_gcn.py`.

---

## TLPGNN — "A Lightweight Two-Level Parallelism Paradigm for GNN
Computation on GPU" (HPDC'22) — `conf/hpdc/FuJH22`

- **Platform**: GPU (V100 tested).
- **Workloads**: dataset directories under `data/` following a
  `features.npy` + `csr.npz` convention; only `citeseer` is shown as the
  worked example in the README/script default, but the loader is
  dataset-agnostic (any dataset following the same file layout works).
- **Timing protocol** (`gcn/test_kernel.py`): **kernel-only, Python-side
  wall clock** — one untimed call to warm up JIT/compilation
  (`load_inline` custom CUDA extension), then **10 measured repetitions**,
  each individually bracketed by `time.perf_counter()` +
  `th.cuda.synchronize()`, averaged (`run_time*1e3/10`) — mean only, no
  min/max/median, and the timer is host-side `perf_counter` around a
  synchronize rather than device-side CUDA events (courser-grained than
  GE-SpMM/MaxK-GNN's device-timer approach).
- **Scope**: this measures ONE aggregation kernel call in isolation (the
  custom CUDA extension's `forward`), not a full layer or training epoch —
  the narrowest, purest "kernel-only" scope of the papers surveyed.
- **Sweep**: `--size` (feature dimension), default 32 in the example
  invocation.
- **Source**: `github.com/charlifu/TLPGNN` — README.md,
  `gcn/test_kernel.py`.

---

## QGTC — "Accelerating Quantized GNN via GPU Tensor Core" (PPoPP'22) —
`conf/ppopp/WangFD22`

- **Platform**: GPU (Tensor Core, Ampere-class for INT8 paths).
- **Operation**: aggregation on QUANTIZED (1/2/4/8-bit) adjacency/feature
  representations via bit-decomposed Tensor-Core MMA — a precision axis none
  of the other papers in this track vary.
- **Workloads**: `artist`, `soc-BlogCatalog`, `ppi`, `ogbn-arxiv` (Cluster-GCN
  full-model epoch benchmark); separate synthetic-shape sweep
  (`X1_height/X1_width/X2_width` in {1024,2048,4096} × {16,32,64,...,1024})
  for the raw GEMM-level Tensor-Core comparison against cuBLAS INT8 — the
  latter is NOT tied to any real graph's adjacency structure at all.
- **Timing protocol**: epoch-level for the model benchmarks (README shows an
  "Epoch (ms)" results table for Cluster-GCN/batched-GIN, bit-width swept
  `{1,2,4,8}`); the raw-GEMM comparison reports TFLOPs directly from
  `bench_cuBLAS_INT8.py` output (exact warmup/rep counts not extracted from
  the README table; would need the referenced `.py` scripts' internals).
- **Baselines**: DGL (full-precision epoch time), cuBLAS GEMM-EX INT8
  (raw-GEMM-level only, Fig. 8 "additional study" not tied to any graph).
- **Source**: `github.com/YukeWang96/QGTC_PPoPP22` — README.md; arXiv
  abstract (`2111.09547`) fulltext not retrievable via WebFetch (404 on both
  `arxiv.org/html/...` and would need PDF parsing not attempted here).

---

## EPPGCN — "Accelerating Backward Aggregation in GCN Training With
Execution Path Preparing on GPUs" (TPDS'22) — `journals/tpds/XuSYLJ22`

- **Platform**: GPU.
- **Workloads**: 10 graphs — `cora`, `citeseer`, `pubmed`, `youtube`,
  `amazon`, `corafull`, `catalog`, `twitter`, `google`, `dblp` — again
  overlapping the "GNNAdvisor-lineage" suite seen in TC-GNN, with different
  large-graph members (`twitter`/`google`/`dblp`/`catalog` vs TC-GNN's
  `amazon0505`/`artist`/`soc-BlogCatalog`), underscoring that no two papers
  in this track use the identical suite despite sharing a common ancestor
  benchmark harness.
- **Key variable being studied**: this paper's whole contribution is that
  GCN BACKWARD aggregation only needs a partial-vertex ("active-vertex")
  computation; its own benchmark sweeps `train_ratio` (fraction of vertices
  active) ∈ `{0.1, 0.3, 0.5, 0.8}` and GCN depth `layers` ∈ `{3,4,5,6}` — the
  benchmark script (`bench_EPPGCN.py`) is a driver over `gcn_main.py`, which
  is a fork of the **GNNAdvisor (OSDI'21) benchmark harness** (the script
  literally invokes a path
  `/home/yc/OSDI21_AE-master/GNNAdvisor/gcn_main.py` in one version), i.e.
  EPPGCN's own timing protocol (warmup epochs, measured epochs, timer) is
  INHERITED from GNNAdvisor's harness rather than defined fresh — consistent
  with GNNAdvisor-derived epoch-level timing used by TC-GNN/QGTC above.
- **Reported speedup**: 1.48x-5.65x for backward aggregation alone;
  1.05x-1.37x (with preprocessing) / 1.03x-1.35x (on-the-fly, no
  preprocessing) for overall training — i.e. EPPGCN is one of the only
  papers in the track that explicitly reports BOTH a
  preprocessing-amortized and a preprocessing-free end-to-end number side by
  side, which is exactly the kernel-only vs e2e-with-preprocessing split
  this spec adopts.
- **Baseline**: GNNAdvisor (its own parent harness).
- **Source**: arXiv `2204.02662` (abstract page only — fulltext 404'd on
  both `arxiv.org/html` and `ar5iv`); `github.com/Catriminal/EPPGCN` —
  `EPPGCN/bench_EPPGCN.py`.

---

## FeatGraph — "A Flexible and Efficient Backend for GNN Systems" (SC'20) —
`conf/sc/HuYWYZL0ZW20`

- **Platform**: CPU (x86, TVM-scheduled) AND GPU (CUDA, TVM-scheduled) — the
  only paper in the track with a single benchmark driver that autotunes and
  measures both backends.
- **Workloads**: any adjacency matrix given as a `scipy` CSR `.npz` file
  (`--dataset` CLI arg); repo ships a `download_reddit_dataset.py` helper,
  implying Reddit is the canonical worked example, but the harness itself is
  dataset-agnostic (no fixed suite shipped).
- **Timing protocol** (`benchmark/bench_vanilla_spmm.py`): uses TVM's own
  `module.measure_average_time(inputs, num_runs)` utility — **5 runs**,
  average only reported (`"average time of {} runs: {} sec"`); TVM's timer
  internally handles device synchronization but the script does not surface
  warmup-iteration count or min/max separately.
- **What is swept, not the feature dim alone**: this benchmark is really an
  AUTOTUNING sweep over schedule parameters — `num_col_partitions` ×
  `num_feat_partitions` (CPU) or `num_cuda_blocks` × `threads_per_block`
  (GPU), each a power-of-2 grid — with a fixed `feat_len` (default 128) per
  sweep point. I.e. FeatGraph's own "benchmark" IS an autotuning search, not
  a single steady-state number; the paper's reported "up to 32x" speedup is
  presumably the BEST point found in this search, not a fixed-config number.
- **Source**: `github.com/amazon-science/FeatGraph` (default branch
  `master`) — `benchmark/bench_vanilla_spmm.py`.

---

## GE-SpMM — "General-purpose SpMM on GPUs for GNNs" (SC'20) —
`conf/sc/HuangD0Y20`

- **Platform**: GPU (CUDA 10.1 tested).
- **Workloads**: the full SNAP collection mirrored via `sparse.tamu.edu`
  (SuiteSparse's SNAP category) — ~66 real graphs (`soc-Epinions1` through
  `wiki-topcats`, see `data/download_SNAP.sh`, com-Friendster/com-Orkut/
  twitter7/soc-Pokec explicitly commented OUT as "too large") plus a small
  `data/misc/*.mtx` set. This is the largest and most systematically
  gathered (whole-category) input suite of any paper in this track — the
  closest to a "SuiteSparse-style inclusion rule," though still informal
  (category-level, not size-thresholded).
- **Timing protocol** (`spmm_test.cu`): `#define ITER 200`; a device
  "warmup" kernel (`warmup<<<1,1>>>()`) is launched **200 times** before
  timing starts; the actual measured region wraps the ENTIRE 200-call loop
  in a SINGLE CUDA-event pair
  (`cudaEventRecord(start); for(i<ITER) kernel(); cudaEventRecord(stop);`)
  and divides by `ITER` — i.e. only the batched MEAN is ever computed, no
  per-iteration timestamps and therefore no median/min/max/variance
  possible from this protocol. Same batched-mean pattern applied identically
  to the cuSPARSE baseline call for a fair head-to-head.
- **Dense operand sweep**: `B_ncols` doubles from 128 up to `max_ncols`
  (128→256→512), fp32.
- **Correctness**: guarded by a compile-time `#ifdef VALIDATE` block
  (elementwise compare vs. a golden dense reference, `1e-2` absolute
  tolerance) that is DISABLED by default — the actual `run_test.sh`
  performance run does not compile with `VALIDATE` defined, so **the
  numbers papers report from this harness are not concurrently
  correctness-checked**, mirroring MaxK-GNN's gap above.
- **Metric**: GFLOP/s = `nnz*2/1e6*B_ncols / (time_ms/ITER)`.
- **Baselines**: cuSPARSE (`cusparseScsrmm2`), merge-spmm/GraphBLAST
  (`gbspmm`), GunRock (separate build). Reported: up to 1.41x over cuSPARSE.
- **Source**: `github.com/hgyhungry/ge-spmm` — README.md, `spmm_test.cu`,
  `run_test.sh`, `data/download_SNAP.sh`.

---

## CAGNET — "Reducing Communication in Graph Neural Network Training"
(SC'20) — `conf/sc/TripathyYB20`

- **Platform**: distributed multi-GPU (`torch.distributed`), evaluated on
  OLCF Summit; the paper's own headline claim is "trained on over a hundred
  GPUs."
- **Operation**: this is a family of 4 PARALLEL DECOMPOSITIONS of the SAME
  SpMM (1D, 1.5D, 2D, 3D sparse-dense matmul across ranks), not a
  single-GPU kernel design — the "kernel" per rank is a local SpMM (backed
  by whatever local library is available), and the paper's actual
  contribution and measured quantity is COMMUNICATION VOLUME/TIME, not
  local kernel throughput.
- **Workloads**: `Reddit` (via PyG's built-in downloader), `Amazon`,
  `Protein`/`subgraph3` (custom COO pickles) — Reddit and a >1-billion-edge
  protein network are the headline scale points from the abstract.
- **Timing protocol** (`gcn_distr.py`): `time.time()` (host wall clock)
  brackets used at TWO granularities: (a) fine-grained per-phase barriers
  (`stop_time()` around every comm/comp sub-step, accumulated into
  `comp_time`/`comm_time`/`bcast_comm_time`/etc. arrays, gated by a
  `--timing True/False` flag) and (b) a single coarse bracket
  (`tstart`/`tstop`) around the ENTIRE `epochs`-length training loop
  starting from epoch 1 — **no warmup epochs are excluded from the coarse
  measurement** (epoch 1, which typically includes first-touch/allocation
  overhead, is included in the timed region). `--runcount` allows repeating
  the whole training run N times for averaging across runs (not
  iterations).
- **Scope**: full end-to-end training, decomposed into comm vs comp time —
  the only paper in the track that treats communication as a first-class,
  separately-reported metric.
- **Source**: `github.com/PASSIONLab/CAGNET` — README.md, `gcn_distr.py`;
  arXiv abstract (`2005.03300`) fulltext not retrievable via WebFetch (404).

---

## Divergences

1. **Timing scope spans four incompatible levels**, all reported under the
   same "SpMM"/"aggregation" umbrella: (a) isolated device-kernel time via
   CUDA events (GE-SpMM, MaxK-GNN), (b) isolated kernel time via host-side
   Python timer + explicit sync (TLPGNN, CBM), (c) full training-epoch time
   including optimizer step and all non-aggregation layers (TC-GNN, QGTC,
   EPPGCN/GNNAdvisor-lineage), (d) distributed epoch time decomposed into
   comm/comp (CAGNET). A paper's claimed "Nx speedup" is only meaningful
   within its own scope-tier; comparing e.g. MaxK-GNN's kernel-level 6.93x
   to TC-GNN's epoch-level numbers directly would be an apples-to-oranges
   error the spec must not repeat.
2. **Statistical rigor of the timing protocol varies by >100x in sample
   count** and reported statistics: CBM (10 warmup + 250 measured,
   mean/std/min/max) and GE-SpMM (200 warmup-kernel launches + a 200-call
   batched-mean region) sit at the rigorous end; MaxK-GNN (4 warmup + 4
   measured, mean only, no variance reported) and TLPGNN (1 untimed call +
   10 measured, mean only) sit at the weak end. GE-SpMM's batched-mean-only
   CUDA-event pattern (identical to the sibling `spmm` track's own
   observation about RoDe/SMaT) structurally cannot report median/min/max
   even if the paper wanted to.
3. **Correctness validation is frequently NOT wired into the timed run**:
   GE-SpMM's validation path is behind a disabled-by-default `#ifdef
   VALIDATE`; MaxK-GNN ships a `check_err()` function whose call site is
   commented out in the active benchmark path. CBM is the clean exception —
   correctness is checked by a dedicated script with explicit tolerances,
   run separately from (not concurrently with) the timed benchmark. This
   spec makes the correctness gate mandatory and pre-timing, following CBM's
   pattern, not GE-SpMM's/MaxK-GNN's opt-in-and-disabled pattern.
4. **Preprocessing/format-conversion handling diverges three ways**:
   excluded and separately reported with an explicit cost (CBM's
   `compression_metrics.sh`); excluded and reported only as a proxy metric,
   not latency (TC-GNN's dense-tile-count scripts report tile COUNT, not the
   graph-translation TIME); or excluded and never reported at all (GE-SpMM,
   TLPGNN, MaxK-GNN's meta-data generation step, which is a one-time
   `generate_meta.py` run whose cost is absent from every kernel-benchmark
   number the repo produces). EPPGCN is notable for being the only paper
   that reports BOTH a with-preprocessing and a without-preprocessing
   end-to-end number side by side (1.05-1.37x vs 1.03-1.35x), which is
   direct evidence that preprocessing amortization visibly moves the
   headline number in this track.
5. **No two papers use the same dataset suite**, despite most tracing to a
   common "GNNAdvisor (OSDI'21) benchmark harness" lineage: TC-GNN's 14
   graphs, EPPGCN's 10 graphs, and QGTC's 4 graphs overlap only partially
   (e.g. `artist`/`soc-BlogCatalog`/`ppi` appear in ≥2 of these three, but
   `amazon0505` only in TC-GNN and `dblp`/`google`/`twitter`/`catalog` only
   in EPPGCN). CBM and GE-SpMM instead draw from SNAP/SuiteSparse with
   synthetic dense features. MaxK-GNN and CAGNET use large OGB/GraphSAGE
   production datasets with REAL feature dimensions. FASTEN's heterogeneous
   relational graphs (AIFB/MUTAG/BGS/AM/DBLP/Freebase) are a disjoint
   category entirely (different operator: segmented matmul, not 2-operand
   SpMM). **No paper states an explicit inclusion/exclusion criterion** —
   every suite is "whatever the authors' pipeline already had," typically
   inherited by citation from GNNAdvisor or SNAP's SC20-era popularity, not
   re-justified. This spec fixes an explicit criterion (below) instead.
6. **Feature-dimension (N) treatment diverges**: swept as the primary
   independent variable (GE-SpMM {128,256,512}; FeatGraph autotuned at fixed
   feat_len=128; MaxK-GNN's structured-k swept {16..192} against fixed
   dim_origin=256) vs fixed to whatever the model's hidden width happens to
   be (TC-GNN/EPPGCN's `--hidden 16`) vs entirely dataset-driven with no
   control (CAGNET's `--midlayer 16`, real per-dataset input dims). A
   benchmark that wants comparable throughput numbers needs to decouple "N
   swept for the aggregation kernel itself" from "hidden width of a specific
   published GCN configuration," which the literature conflates.
7. **Precision**: fp32 dominates, but QGTC is a dedicated quantized-precision
   paper (1/2/4/8-bit) not commensurable with any fp32 GFLOP/s number above,
   and MaxK-GNN's backward SSpMM operates on a structurally sparse
   (top-k-per-row) feature matrix rather than a fully dense one — both are
   flagged as their own axis, not folded into the main fp32 dense-N variant.
8. **Metric units diverge**: GFLOP/s (GE-SpMM, computed from nnz), raw
   seconds with no FLOP normalization (CBM), ms/epoch (TC-GNN/QGTC/EPPGCN
   lineage), TFLOPs on shapes untethered to any real graph (QGTC's Fig. 8
   cuBLAS-INT8 comparison), and communication volume/time as a first-class
   metric distinct from any throughput number (CAGNET). These are not
   pooled into one ranking in this spec's design.
9. **Single-node vs distributed** is a scale discontinuity, not just a
   configuration knob: CAGNET's 16-100+-GPU `torch.distributed` setting
   optimizes communication volume, a fundamentally different bottleneck from
   every single-GPU/single-CPU-node kernel above. Folding it into the same
   variant as the others would compare incommensurable things (see
   `open_questions`).
