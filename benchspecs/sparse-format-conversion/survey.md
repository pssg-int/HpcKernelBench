# sparse-format-conversion — survey

Singleton track: 1 paper.

## conf/hpdc/PengTPK25 — LiteForm (HPDC 2025)

"LiteForm: Lightweight and Automatic Format Composition for Sparse
Matrix-Matrix Multiplication on GPUs." LiteForm picks, per input matrix, a
composed sparse storage format — CELL (a partitioned ELL-family format
built via SparseTIR's `column_part_hyb`) vs. BCSR — using two small
pretrained RandomForest models (one classifies CELL-vs-BCSR, one predicts
the CELL partition count), then bucket-searches a per-partition config
before running SpMM. This "predict format, predict partitioning, bucket
search, then run" pipeline is exactly the format-conversion-plus-search
preprocessing this track is meant to cover.

- **workloads/inputs**: `playground_pipline/proc05.spmm.end-to-end.py`'s
  `NAMES` list — 7 real graphs from the GNN literature: `cora, citeseer,
  pubmed, ppi, arxiv, proteins, reddit` (loaded via `MTX`, i.e. Matrix
  Market format, likely GNN adjacency matrices in the SparseTIR-artifact
  convention this repo explicitly builds on:
  "Please set up the environment according to SparseTir Artifact"). Despite
  the repo also shipping a `scripts/download_suitesparse.v3.curl.py`
  utility that can pull the full ~900GB SuiteSparse collection, the actual
  released end-to-end benchmark driver (`drive05.spmm.end-to-end.sh` →
  `proc05...py`) only exercises these 7 named graphs, not the general
  collection — an important scope note: the "wide suite" capability exists
  in the tooling but is not what the paper's own headline pipeline runs.
- **format pair(s)**: source is the matrix's native CSR-equivalent
  (loaded via `MTX`); target is either **CELL** (`build_hyb_format` /
  `column_part_hyb`, an ELL-style partitioned hybrid format with a
  bucket-config search over partition boundaries) or **BCSR**
  (`bench_bsrmm`), selected per-matrix by the format-selection model.
  Both are one-way conversions for the purpose of running SpMM; no
  roundtrip (target format back to source) conversion path is exercised
  anywhere in the code read.
- **preprocessing cost accounting**: the pipeline explicitly separates and
  times THREE preprocessing stages via `time.perf_counter()`: (1) format
  selection (`predict_format_selection`, a ~microsecond RandomForest
  `.predict` call over 7 matrix-level features: num_rows/num_cols/nnz/
  avg,min,max,stddev nnz-per-row), (2) partition-count prediction
  (`predict_num_partitions`, a second RandomForest call over
  density-based features), and (3) bucket-config search
  (`search_bucket_config`) — a genuinely separate, more expensive search
  step over CELL's own hyperparameters. This is one of the more disciplined
  preprocessing-cost breakdowns among the survey corpus for other tracks —
  LiteForm's own artifact already treats "how long does picking+building
  the format take" as first-class, timed output, not folded silently into
  a headline GFLOP/s number.
- **timing protocol / correctness**: the actual SpMM kernel timing
  (`bench_hyb_with_config`, `bench_bsrmm`, `bench_naive`) is delegated to
  `sparsetir_artifact.profile_tvm_ms`, an external helper from the
  SparseTIR artifact package this repo depends on but does not vendor —
  its exact warmup/repetition/statistic convention was not confirmed from
  the files read (flagged in open_questions). `FLUSH_L2=ON` is set as an
  environment variable before the driver script runs, indicating the
  benchmark deliberately flushes the GPU L2 cache between timed calls
  (a cold-cache discipline, notable since many other tracks' surveyed
  papers don't state a cache policy at all).
- **precision**: fp32 (SparseTIR's `float32` sparse-buffer convention
  visible in the `ell`/`csrmm` TIR prim-func signatures read).
- **baselines**: `bench_naive` (presumably a plain/default cuSPARSE-class
  reference, name suggests an un-composed baseline format) alongside CELL
  and BCSR; the paper's own headline claim (per `benchmark_groups.json`'s
  one-liner) is "2.06x over cuSPARSE."

Source: repo `README.md`, `playground_pipline/{drive05.spmm.end-to-end.sh,
proc05.spmm.end-to-end.py, format_algos.py}` (read in full/large excerpt).

## Divergences / open items

- Single-paper track: no cross-paper divergence. The main internal tension
  worth flagging is that the repo's tooling *supports* full-SuiteSparse-
  scale evaluation (the 900GB download script) but the actual released
  end-to-end benchmark only runs 7 named GNN graphs — this spec's
  `selection` field is written to make that scope explicit rather than
  imply a broader suite than what was actually run.
- No roundtrip (convert-then-convert-back) correctness check exists in the
  reviewed code; the track's own axis list calls for "roundtrip
  correctness," so this spec adds it as a new, stricter requirement not
  evidenced in LiteForm's own artifact (see notes_on_fairness).
