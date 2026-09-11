# Survey: embedding-ops

Track input: `data/track_inputs/embedding-ops.json` (4 papers, all 4 surveyed).
No paper in this track has a public arXiv id/OA link *except* FULL-W2V; for the
other three, Semantic Scholar (`api.semanticscholar.org/graph/v1/paper/DOI:...`)
also reported no open-access PDF, so methodology for cuKE, RecFlex, and EL-Rec
below is reconstructed entirely from their artifact repos (benchmark/run
scripts), per the instructions' fallback order.

---

## 1. cuKE (IPDPS 2024) — `parallel-group/cuke`

- **workloads/inputs**: Knowledge-graph-embedding (KGE) score functions —
  TransE, TransH, TransR, TransF, RESCAL, neg_TransR, neg_TransF — evaluated
  on knowledge graphs (default dataset in the driver script is `FB15k`,
  standard stats ~14,951 entities / 1,345 relation types / ~483K training
  triples; not re-confirmed in the code slice read, cited from common
  knowledge of the dataset). Two embedding tables per run: entity table
  `Eemb[nnodes, dim]` and relation table `Remb[nedges, dim]`. Default
  `batch_size=1024`, `dim=512`, `neg_sample_size=64` (all CLI-overridable).
- **pooling / access pattern**: not pooled-sum (DLRM-style); each score
  function does a **single-index gather** per tensor (`Eemb[h]`, `Eemb[t]`,
  `Remb[r]`) followed by fused elementwise/bilinear arithmetic (e.g. TransE:
  `Eemb[h] - Eemb[t] + Remb[r]`). Relation ids `r` exhibit heavy **reuse
  skew** (few relation types shared by many edges in a batch) which cuKE
  explicitly exploits via a `reuse` attribute + runtime "inspector" that
  sorts/dedups repeated relation indices and caches the unique relation rows
  in GPU shared memory — a cache/hit-rate optimization triggered by access
  skew, though no hit-rate number is reported in the code, only the
  optimization's existence.
- **timing protocol**: one **untimed warm-up call**, then 100 timed calls to
  `run.gpu.compile_and_run(...)` wrapped in `torch.cuda.Event` start/end;
  reported statistic is the **mean** (`elapsed_time_ms/100`); no min/max/median
  reported. Timer: CUDA events (device-side).
- **timing scope**: kernel-only (the compiled fused CUDA kernel call);
  CUDA-graph/JIT compilation of the generated code happens once before
  warm-up and is not counted; no H2D/D2H in the timed region beyond what the
  kernel call itself triggers.
- **precision & correctness**: fp32 (`torch.empty(..., at::kFloat)` in
  generated code). No numeric-tolerance correctness check is visible in the
  benchmark driver itself; correctness is argued by construction (code is
  compiler-generated from the same computation graph run in PyTorch) and
  compared informally against PyTorch/TVM outputs elsewhere in the repo.
- **metric**: average latency per call (ms); paper's headline is speedup vs.
  TorchScript (14.9x, per `one_liner`).
- **baselines**: PyTorch eager, TVM (Fig 9a-d, Fig 10a-b compare cuke vs.
  pytorch vs. tvm scripts in `apps/kge/scripts/`).
- **source**: repo `apps/kge/README.md`, `apps/kge/test_cuke.py`
  (`parallel-group/cuke`, read via `gh api`).

---

## 2. RecFlex (SC 2024) — `PanZaifeng/RecFlex`

- **workloads/inputs**: synthetic DLRM-style recommendation models —
  "simplified DLRMs with no bottom MLP", **reduce-sum pooling for all
  features**, embedding-dim concatenation after lookup. Two table-config
  regimes seen in the repo:
  - `examples/models/A` (small/synthetic test config): tables with fixed
    `rows=1024`, `embed_dim` in `{4,8,16,32,64,128}`.
  - `examples/models/mlperf/table_config.txt`: a **26-table MLPerf/DLRM-Terabyte-
    derived config**, row counts spanning `3` to `40,000,000`
    (several tables pinned at exactly 40,000,000 rows — a synthetic cap
    matching MLPerf's known Terabyte-scale table-size convention), uniform
    `embed_dim=64`.
- **access-distribution skew**: **explicitly parameterized** — index values
  for each table are sampled via `get_power_law_probabilities(size, alpha=3.0)`
  (`np.random.power(a=3.0)`) in `data_synthesis/data_generate.py`, i.e. a
  tunable power-law skew with `alpha` as a first-class knob. Pooling-factor
  (hot-count) distribution also varies by declared feature type in
  `data_config.txt`: `one-hot` (factor=1), `multi-hot-static` (fixed factor),
  `multi-hot` (truncated-normal(mean, stddev, max)), `multi-hot-one-side`.
  The `mlperf/generate_dist.py` script additionally reproduces **MLPerf's own
  query-length quantile table** (quantiles `[100,100,200,...,700]`) for batch
  construction.
- **batch size**: batch sizes themselves are drawn from a **log-normal**
  distribution (`query_mean=256`, `query_stddev=128`, `query_max=512`,
  `num_queries=128` sample batches per run by default).
- **timing protocol / scope — two separate measurements**:
  1. *Kernel-only*: `examples/run_e2e.sh` wraps the fused kernel in
     `nsys profile -c cudaProfilerApi`, then `nsys stats --report gpukernsum`
     sums **only GPU kernel durations**, host-side work excluded entirely.
  2. *Batch end-to-end*: `examples/end_to_end.py` — for each of 128 sample
     batches, an **untimed** `preprocess(...)` (host-side variable-length
     feature parsing) is run, then `time.perf_counter()` brackets only the
     `recom_model(...)` call; per-batch times are accumulated (ms) over all
     128 batches. No separate warm-up iterations are visible in this script
     (the first of the 128 batches is not excluded).
- **precision**: not stated in the read scripts; standard PyTorch fp32
  tables implied (TorchRec/HugeCTR baselines use fp32 embedding tables by
  default in their reference configs).
- **metric**: accumulated latency (ms); kernel-only GPU-time sum (ms).
- **baselines**: TorchRec, HugeCTR, RECom — reported average speedups of
  1.95x / 11.57x / 7.92x respectively (from README).
- **source**: `PanZaifeng/RecFlex` — `README.md`,
  `data_synthesis/data_generate.py`, `examples/end_to_end.py`,
  `examples/run_e2e.sh`, `examples/models/mlperf/{table_config.txt,
  generate_dist.py}` (all via `gh api`).

---

## 3. EL-Rec (SC 2022) — `Ash-Zheng/SC_artifacts_eval`

- **workloads/inputs**: three **real, standard DLRM click-log datasets**,
  with exact per-table row counts read directly from
  `Figure11/ELRec_train.py`:
  - **Kaggle** (Criteo Display-Ads Challenge): 26 tables, `dense_num=13`,
    `feature_size=16`; row counts range 4 (smallest) to 9,214,729 (largest).
  - **Terabyte** (Criteo Terabyte logs): 26 tables, `feature_size=64`; row
    counts range 4 to 39,991,895 (~40M, largest table).
  - **Avazu**: 20 tables, `feature_size=16`; row counts range 5 to 6,385,037.
  (Full `table_length` lists captured verbatim in the code, e.g. Kaggle:
  `[1461, 581, 9214729, 2031648, 306, 24, 12471, 634, 4, 90948, 5633,
  7607629, 3183, 28, 14825, 4995567, 11, 5606, 2172, 4, 6431684, 18, 16,
  272266, 105, 138045]`.)
- **pooling / access skew**: standard DLRM `EmbeddingBag`-style pooled
  lookups per categorical feature (bag sizes implicit in the click-log data,
  not separately controlled); access skew is **whatever the real ad-click
  logs contain** (naturally power-law / hot-cold, uncontrolled/unlabeled).
  EL-Rec's core contribution is a **tensor-train (TT) compressed** embedding
  table plus an **index-reordering** technique explicitly designed to
  "harvest performance gains from both local and global information of
  training inputs" — i.e., a locality/hit-rate-oriented reordering of row
  indices before TT-lookup, though no explicit cache-hit-rate metric is
  printed by the benchmark script itself.
- **batch size**: 4096 (fixed, `Figure11/ELRec_train.py`).
- **timing protocol / scope**: **full training-step wall-clock**, not
  kernel-isolated: `time_wrap()` = `torch.cuda.synchronize(); return
  time.time()`; a single un-warmed-up loop of `num_iters=1000` (data load →
  forward → BCE loss → `optimizer.zero_grad()` → `backward()` →
  `optimizer.step()`) is timed start-to-end; the reported number is total
  wall time for the 1000 iterations (`out.log`: `"ELRec, {dataset}, time:
  {:.3f}"`). No warm-up iterations are excluded and no repetition beyond the
  single 1000-iteration run is used, so no mean/median/variance is reported.
- **precision & correctness**: fp32 training; correctness is validated as
  **downstream task test accuracy** (Table 4: accuracy of DLRM / TT-Rec /
  TorchRec / HugeCTR / EL-Rec on the same datasets), *not* a numerical
  tolerance on kernel output.
- **metric**: total wall time for a fixed iteration count (→ derivable
  training throughput / iterations-per-second); also 1/2/4-GPU scaling
  (Figure 12/13).
- **baselines**: DLRM (Facebook reference, CPU-GPU and GPU), TT-Rec,
  TorchRec, HugeCTR.
- **source**: `Ash-Zheng/SC_artifacts_eval` — `README.md`,
  `Figure11/ELRec_train.py`, `Figure13/README.md` (via `gh api`).

---

## 4. FULL-W2V (ICS 2021, Best Paper) — `tlranda/FULL-W2V`

- **workloads/inputs**: Word2Vec (SGNS) on two standard NLP corpora (from
  arXiv 2312.07743 full text, fetched via `ar5iv.labs.arxiv.org/html/2312.07743`):
  - **Text8**: vocabulary 71,291, 16,718,845 words/epoch, 17,006 sentences.
  - **One Billion Words**: vocabulary 555,514, 804,269,957 words/epoch,
    30,607,795 sentences.
  Both filtered to words occurring ≥5 times; sentences capped at 1,000
  words. Embedding table here is the vocab×dim word/context table (this is
  the "embedding" being gathered), `dim=128`, negative samples `N=5`,
  context window `W=5`; training epochs: 20 (Text8) / 5 (1B-Word).
- **access-distribution skew**: not artificially parameterized (unlike
  RecFlex); relies on the **natural Zipfian word-frequency skew** inherent
  to any text corpus. FULL-W2V's core contribution ("Lifetime Reuse of
  Context Words") is precisely a data-reuse/caching strategy that exploits
  this skew — hot (frequent) context words are kept resident and reused
  across the negative-sampling window rather than re-fetched — analogous to
  the embedding-row cache/hit-rate concern in DLRM-style pooling, but for a
  gather-then-SGD-update kernel rather than pooled-sum.
- **timing protocol**: **mean and standard deviation of 5 identical
  executions** ("reduce variance"); no explicit warm-up iteration is
  isolated (the first training epoch of the 5 runs is not excluded).
  Timer/statistic collection method beyond "5 runs, mean+std" not further
  specified in the text obtained.
  Memory-traffic figures separately obtained via **Nsight Systems**
  profiling (GB/epoch), not from the wall-clock timing runs.
- **precision & correctness**: precision not explicitly stated (standard
  fp32 for W2V weights). Correctness/quality is validated **not** by a
  numerical tolerance against a reference kernel but by **downstream
  embedding-quality metrics**: Spearman's rank correlation on WS-353 and
  SimLex-999 word-similarity benchmarks, plus analogy-reconstruction
  accuracy on Mikolov's analogy set.
- **metric**: throughput in **words/second**; memory demand in GB/epoch.
- **baselines**: pWord2Vec (CPU), pSGNScc (CPU), accSGNS (GPU), Wombat
  (GPU).
- **hardware**: three GPU generations — V100 (Volta, 80 SM, 900 GB/s HBM2),
  Titan Xp (Pascal, 60 SM, 548 GB/s), P100 (Pascal, 56 SM, 549 GB/s); CPU
  baselines use one thread per logical core.
- **source**: arXiv 2312.07743 full text (via ar5iv render); repo
  `tlranda/FULL-W2V` `README.md` corroborates the replication harness
  (`replication/sweep.py`, `template.pbs`, hierarchical corpus/system/impl
  qualifier files) but adds no new numeric detail beyond the arXiv text.

---

## Divergences

- **Timing scope varies from "single kernel" to "full training step" across
  the four papers**, with no common denominator reported by all of them:
  cuKE times one fused-kernel call (mean of 100, CUDA events); RecFlex
  splits kernel-only (nsys GPU-kernel-sum) from batch-accumulated
  wall-clock-with-untimed-preprocessing; EL-Rec times an entire
  1000-iteration **training step** (forward+backward+optimizer, data
  loading included, no warm-up); FULL-W2V times **whole training epochs**
  (mean of 5 full runs) including the entire SGD/negative-sampling loop.
  None of the four isolates "just the embedding gather/pooling kernel" in
  a way directly comparable to the others — the spec below has to define
  that isolation itself.
- **No paper reports GB/s (memory bandwidth) as its primary metric**, despite
  all four being explicitly memory/access-pattern bound problems; they
  report wall time, mean latency, or throughput in domain-specific units
  (words/sec, training wall time) instead. This is a real gap for a
  memory-bound-kernel benchmark and the spec fixes it by mandating GB/s as a
  co-primary metric, computed from *unique* rows actually touched (which
  depends on skew — see next point).
- **Access-skew handling ranges from fully parameterized to fully implicit**:
  RecFlex exposes skew as a tunable `alpha` (power-law); EL-Rec and
  FULL-W2V inherit whatever skew is latent in real click-log/text data
  (uncontrolled, dataset-dependent); cuKE exploits (but doesn't parameterize)
  skew only for the small relation-type table, not the large entity table.
  A GB/s metric computed under skew is ambiguous unless the unique-row
  count / hit-rate is fixed and reported — none of the four papers reports
  a hit-rate number, so the spec introduces one explicitly.
- **Correctness validation differs categorically, not just in tolerance**:
  cuKE/RecFlex rely on informal/compiler-correctness arguments (no visible
  numeric-tolerance gate in the benchmark scripts read); EL-Rec validates
  **downstream task accuracy** (Table 4); FULL-W2V validates **downstream
  embedding quality** (WS-353/SimLex/analogy). None reports a
  max-relative-error-vs-reference-gather check, which is the natural
  correctness gate for a pure memory-bound kernel benchmark (as opposed to
  a full training pipeline) — the spec adds this as a precondition.
- **Warm-up/repetition protocol is inconsistent and mostly thin**: cuKE (1
  warm-up call, 100 reps, mean only); RecFlex (no isolated warm-up, 128
  accumulated batches); EL-Rec (**no warm-up at all**, single un-repeated
  1000-iteration run); FULL-W2V (no isolated warm-up, 5 full-run
  repetitions, mean+std). The spec fixes an explicit warm-up + repetition
  count rather than inheriting any one paper's (weak) choice.
  the pipeline
- **Pooling model differs**: RecFlex and EL-Rec are genuine DLRM-style
  pooled (`EmbeddingBag`-equivalent) multi-hot lookups; cuKE and FULL-W2V
  are single-index (non-pooled) gathers followed by fused elementwise/SGD
  arithmetic. These are different enough operations that the spec defines
  separate variants rather than forcing one shape.
