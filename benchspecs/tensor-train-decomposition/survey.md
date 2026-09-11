# Tensor-train-decomposition track — evaluation-methodology survey

Surveyed 2/2 papers, both primarily via the artifact repo (run scripts +
source), since neither paper's fulltext was reachable through arXiv/OA
mirrors: STTID (IPDPS 2026) has no arXiv id and its only public preprint
mirror (Zenodo) hosts a software archive with metadata only, not the paper
text; EL-Rec (SC'22) has no arXiv id and IEEE Xplore / Unpaywall / Semantic
Scholar all report it closed-access with no OA location. In both cases the
artifact's own run scripts turned out to be exceptionally concrete
(exact CLI invocations with every numeric argument spelled out for STTID;
exact hyperparameters read from source for EL-Rec), consistent with this
project's general finding that timing-loop code reveals protocol details
paper text often omits.

## STTID — Meng, Stoudenmire, Pierce, Mueller, Li, "STTID: High-Performance
Sparse Tensor-Train Interpolative Decomposition" (IPDPS 2026)

Source: `github.com/tensorworld/STTID` (C++/CUDA + Python, MIT license, 1
star, pushed 2026-02-19) — `README.md`, `run.sh` (exact CLI invocations,
reproduced verbatim below), `Py/tt_id.py` (reconstruction-error/tolerance
math), `src/sttid/util/timer.cpp` (timing mechanism). Zenodo record
(`zenodo.org/records/18702524`) confirmed the paper/software pairing but
contributed no additional fulltext.

STTID computes Tensor-Train Interpolative Decomposition (TT-ID): unlike
TT-SVD, ID selects a subset of actual rows/columns (skeleton decomposition)
rather than computing dense orthogonal factors, which lets TT-cores of a
SPARSE input tensor remain genuinely sparse — the paper's stated motivation
is that conventional (TT-SVD-based) methods "produce dense TT-cores" even
for sparse input, defeating memory/compute savings.

- **Workloads**: three named categories, every one used in `run.sh` with
  the FULL concrete invocation (verbatim, `<binary> <path> <nnz> <dims...>
  <rmax> <epsilon> <spthres> <binary-flag> <idx_offset>`):
  - random synthetic: `Rnd5.tns` (order 4, 100^4, 12,729 nnz, rmax=5000),
    `Rnd6.tns` (order 5, 50^5, 27,843 nnz, rmax=15000) — spthres=0
  - knowledge-graph tensors: `KG_JF17K` and `KG_WikiPeople`, each in a
    3D and a 4D variant (e.g. KG_JF17K_3: 66x12270x12270, 25,820 nnz;
    KG_WikiPeople_4: 23x6536x6536x6536, 9,509 nnz), rmax=100,000,
    spthres=0.1, `binary`-entry flag set (entries are 0/1) — these come
    from real knowledge-graph hyperedge data
  - **FROSTT tensors** (`frostt.io`): uber (183x24x1140x1717, 3.31M nnz),
    nips (2482x2862x14036x17, 3.10M nnz), chicago-crime-comm
    (6186x24x77x32, 5.33M nnz, order 4), chicago-crime-geo
    (6185x24x380x395x32, 6.33M nnz, order 5) — rmax=500, spthres=0.3
- **TT-rank / accuracy target as a spec parameter**: `epsilon` (relative
  accuracy tolerance) is FIXED at 1e-10 across every single run.sh
  invocation regardless of tensor category — this is the actual accuracy
  target from which the realized TT-ranks are derived (not the reverse:
  `rmax` is only a CEILING). `Py/tt_id.py` shows the exact mechanism: a
  per-mode cutoff `delta = (epsilon / sqrt(dim-1)) * ||X||_2` is computed
  once from the global epsilon and tensor norm, then each mode's realized
  rank is `r_i = min(r_max, r_delta_i)` where `r_delta_i` comes from a
  rank-revealing decomposition (three interchangeable criteria in the
  code: PRRLDU/partial-rank-revealing-LU, a nuclear-norm-score variant,
  and QRCP/QR-with-column-pivoting) run against that cutoff. This IS the
  compression-vs-error tradeoff the track's key axes call for: epsilon is
  the accuracy knob, the resulting per-mode TT-ranks (and hence TT-core
  memory footprint) are the dependent/compression-side outcome, capped by
  `rmax`.
- **Precision/format**: CPU and GPU (CUDA) implementations both exist
  (binary names carry `_cpu` suffix; a hybrid CPU/GPU path is claimed in
  the abstract: "developing a hybrid CPU..." — truncated in the fetched
  abstract text, likely "...CPU/GPU pipeline").
- **Timing protocol**: `src/sttid/util/timer.cpp` implements a
  `std::chrono::high_resolution_clock`-based scope-timer (RAII: starts on
  construction, stops on destruction) that accumulates call-count, total,
  mean, min, max, and standard deviation into a static map keyed by label —
  i.e. the artifact's OWN internal instrumentation already computes exactly
  the statistic set this project's specs standardize on (mean/min/max +
  implicit repetition via however many times each labeled region is
  entered), rather than a single-shot wall-clock read. No separate,
  explicit warmup phase was found in `run.sh` itself (each dataset is
  invoked once per script line), so any repetition/warmup happens INSIDE
  the timed binary via the labeled-region mechanism, not visible from the
  shell-level script.
- **Correctness**: `Py/tt_id.py`'s reconstruction-error check is
  `rerror = ||C@X - W||_2 / ||W||_2` (relative Frobenius-norm error of the
  low-rank approximation against the reshaped original), computed against
  the SAME `epsilon` used to derive the TT-ranks — i.e. correctness here
  IS the accuracy-target gate, not a separate fixed tolerance. A
  `test_quality.py` Python driver additionally reports core density,
  realized TT-ranks, and reconstruction error across methods
  (TT-SVD/TT-cross-LU/TT-cross-QR/TT-ID) for side-by-side comparison.
- **Metric**: the one_liner's claimed headline is "up to 550x GPU speedup"
  (speedup of the GPU sparse-TTID kernel over an unstated CPU/dense
  baseline; exact baseline and per-dataset breakdown not recovered from
  the sources reached — flagged as an open question).
- **Baselines**: dense TT-ID (`build.sh` explicitly builds "STTID and
  dense TT-ID" side by side) is the paper's own primary internal baseline;
  the Python quality-comparison script additionally names TT-SVD,
  TT-cross-LU, and TT-cross-QR as comparison methods for the
  quality/reconstruction-error axis specifically (not necessarily for the
  timing axis).

## EL-Rec — Wang et al., "EL-Rec: Efficient Large-Scale Recommendation
Model Training via Tensor-Train Embedding Table" (SC 2022)

Source: `github.com/Ash-Zheng/SC_artifacts_eval` (the paper's official
SC'22 Artifacts-Evaluation repo, 9 stars, no declared license) —
`README.md`, `Figure11/README.md` + `run.sh` + `ELRec_train.py`,
`Table4/README.md`, `models/ELRec_dlrm.py`, `models/Efficient_TT/`.

EL-Rec uses Tensor-Train decomposition NOT as a one-shot analysis of a
given sparse tensor (unlike every other paper in this project's tensor
tracks), but as a FIXED, TRAINED compression format for a Deep Learning
Recommendation Model's (DLRM) embedding tables: each large embedding table
is factored into 2-3 small TT-cores that are trained end-to-end via SGD;
the timed cost is the on-the-fly TT-core-product ("reconstruction") of an
embedding row on every forward/backward pass, amortized into overall DLRM
training throughput — this maps directly onto the track's
"decomposition vs. reconstruction timing" axis as the RECONSTRUCTION side
(STTID above is the DECOMPOSITION side).

- **Workloads**: three named, standard DLRM benchmark datasets (all
  real, not synthetic): Kaggle Display Advertising Challenge (via
  figshare), Criteo Terabyte (Criteo Labs), Avazu CTR Prediction (Kaggle)
  — the README requires each pre-organized under a `dlrm_dataset/`
  directory.
- **TT-rank / accuracy tradeoff as a spec parameter**: `tt_rank=128` is
  the default in `models/ELRec_dlrm.py`'s `Eff_TTEmbedding` constructor
  (`tt_ranks=[self.tt_rank, self.tt_rank]`, i.e. a 3-core TT-decomposition
  of each embedding table with two internal bond dimensions both set to
  128). The embedding dimension determines the TT-SHAPE the table is
  factored into: `feature_size=16 -> q_shape=[2,2,4]`,
  `feature_size=64 -> q_shape=[4,4,4]` (Avazu/Kaggle use feature_size=16,
  Criteo Terabyte uses 64, per `Figure11/ELRec_train.py`). TT-rank here is
  a TRAINING hyperparameter fixed BEFORE training starts (unlike STTID,
  where rank is a numerical-tolerance-derived OUTPUT of decomposing an
  already-given tensor) — this is a structurally different use of
  "TT-rank" from STTID's, flagged in Divergences below.
  `models/Efficient_TT/` contains custom CUDA kernels
  (`efficient_tt_cuda.cu`, `hashtbl_cuda_utils.cuh`) for the batched
  TT-core-product embedding lookup, i.e. the "reconstruction" kernel
  itself is hand-optimized CUDA, not a generic tensor-algebra call.
- **Timing protocol**: `Figure11/ELRec_train.py` records wall-clock
  training time over exactly 1000 iterations
  (`num_iters = 1000`, `torch.cuda.synchronize()` bracketing
  `start = time_wrap()` / `end = time_wrap()` around the WHOLE loop, i.e.
  a single elapsed-time measurement over 1000 iterations, not a
  per-iteration distribution) — batch size 4096. No warmup iterations are
  excluded from the 1000; iteration 1 (including any first-batch
  data-loading/cuDNN-autotune cost) is included in the same timed window
  as iteration 1000.
- **Precision**: not explicitly stated in the fetched training script
  (PyTorch default fp32 is the working assumption, unconfirmed).
- **Correctness/quality gate**: `Table4/README.md` states "To get the
  accuracy of each model, please run `./run.sh`", which produces a
  `loss.png` — i.e. EL-Rec validates the TT-compressed embedding table via
  DOWNSTREAM TASK LOSS/accuracy (recommendation quality after training),
  not via a numerical reconstruction-error check of the embedding TABLE
  itself against an uncompressed reference. The training objective is
  binary cross-entropy (`torch.nn.BCELoss`); no AUC computation was found
  in the fetched `ELRec_dlrm.py`, though `Table4`'s separate
  `*_train_and_test.py` scripts (not fetched in full) are named to suggest
  a held-out test-set evaluation exists beyond training loss alone.
- **Metric**: training wall-clock time per 1000 iterations (Figure 11);
  downstream model quality/loss (Table 4); the paper's own headline claim
  (per this project's Phase-1 one_liner) is "speeds DLRM training 3x on 1
  GPU."
- **Baselines**: DLRM (dense embedding table, CPU-GPU and multi-GPU
  variants), FAE, TT-Rec (a prior TT-compressed-embedding system), and
  (per the top-level README, though not present in the `Figure11`
  directory's own script) TorchRec and HugeCTR as additional
  industrial-strength baselines used elsewhere in the artifact's full
  figure set. Multi-GPU configurations (1, 2, 4 GPUs) are tested for
  scaling.

## Divergences

- **What "TT-rank" means is structurally different between the two
  papers.** STTID's TT-rank is an OUTPUT: given a sparse tensor and an
  accuracy target (`epsilon`), the rank-revealing decomposition determines
  the SMALLEST rank per mode that meets that target, capped by `rmax`.
  EL-Rec's TT-rank is an INPUT: a fixed hyperparameter (128) chosen before
  training starts, with no accuracy-target-driven rank selection at all —
  a fixed-shape compressed embedding table is trained via SGD from that
  starting point regardless of whether a smaller or larger rank would
  better trade off model quality vs. memory. A benchmark spec that
  reported "TT-rank" as one comparable axis across both without this
  distinction would misrepresent EL-Rec's practice as if it searched for a
  target-accuracy rank the way STTID does. This spec's two variants keep
  the parameter's role (output vs. input) explicit rather than treating
  it as one universal knob.
- **Decomposition timing vs. reconstruction timing.** STTID times the
  ONE-SHOT decomposition of a given sparse tensor into TT-cores. EL-Rec
  NEVER decomposes an existing tensor at all in the timed region — its
  TT-cores are randomly initialized and learned via gradient descent; what
  is timed is the repeated on-the-fly "reconstruction" (TT-core product)
  of embedding rows during training, amortized over 1000 iterations. These
  are genuinely the two different timing scopes the track's key axes ask
  for ("decomposition vs. reconstruction timing"), not two measurements of
  the same thing under different names.
- **Correctness philosophy**: STTID validates a NUMERICAL reconstruction-
  error bound against the exact input tensor (a pure numerical-
  linear-algebra correctness notion). EL-Rec validates DOWNSTREAM TASK
  QUALITY (recommendation loss/accuracy) with no numerical
  reconstruction-error check of the embedding table against an
  uncompressed reference at all — there is no "ground truth" uncompressed
  tensor EL-Rec's TT-cores are approximating, since the table is learned
  from scratch in TT-form. This spec keeps a numerical gate for the
  decomposition variant and a task-quality gate for the reconstruction/
  training variant, rather than inventing a numerical tolerance for
  EL-Rec that its own methodology has no notion of.
- **Warmup practice**: neither paper's own artifact excludes a warmup
  region from its reported timing window (STTID: single invocation per
  config in `run.sh`, no explicit warmup step visible at the shell level;
  EL-Rec: `num_iters=1000` includes iteration 1). This spec flags both as
  divergences from this project's general warmup convention rather than
  silently imposing one.

## Open questions

- STTID's "up to 550x GPU speedup" headline figure's exact baseline
  (CPU sparse-TTID? dense TT-ID? which dataset?) was not recovered from
  the sources reached (the paper's fulltext itself, via Zenodo, was not
  accessible — only metadata).
- STTID's `run.sh` shows no explicit warmup/repetition count at the shell
  level; whether the labeled-region timer in `timer.cpp` is invoked
  multiple times per `run.sh` line (i.e. repetition happens INSIDE the
  binary) was not confirmed by reading the `main()` driver code itself.
- STTID's exact GPU model/hardware used for the "550x" claim was not
  recovered.
- EL-Rec's precision (fp32 assumed, PyTorch default) was not explicitly
  confirmed in the fetched training script.
- EL-Rec's exact accuracy metric (AUC? test BCE loss? something else) for
  the Table 4 comparison was not recovered — only that `loss.png` is
  produced by `./run.sh`; the underlying `*_train_and_test.py` scripts
  were not fetched in full.
- EL-Rec's GPU model/hardware and exact per-baseline Figure-11 numbers
  were not recovered (the README references a specific Docker image,
  `happy233/zheng_dlrm:latest`, suggesting a fixed provided environment,
  but not a disclosed hardware spec in the fetched text).

## evidence

- sttid: >
    IPDPS'26, sparse Tensor-Train Interpolative Decomposition (TT-ID) with
    sparsity-preserving TT-cores (vs. dense-TT-core competitors); random
    synthetic + real knowledge-graph (KG_JF17K, KG_WikiPeople) + real
    FROSTT tensors (uber, nips, chicago-crime-comm/-geo), order 3-5;
    epsilon (accuracy target) FIXED at 1e-10 across every run, rmax as a
    per-category rank CEILING (500-100000), realized per-mode TT-rank
    derived from epsilon via a rank-revealing decomposition (PRRLDU /
    nuclear-norm / QRCP, three interchangeable criteria); CPU and GPU
    (CUDA) implementations; internal
    std::chrono-based labeled-region timer tracking mean/min/max/stddev;
    relative-Frobenius-norm reconstruction-error correctness check tied
    directly to epsilon; dense TT-ID as primary baseline plus
    TT-SVD/TT-cross-LU/TT-cross-QR for the quality-comparison script; "up
    to 550x GPU speedup" headline (baseline/dataset breakdown not
    recovered).
- elrec: >
    SC'22, Tensor-Train-compressed DLRM embedding table trained end-to-end
    via SGD (TT-rank=128 fixed hyperparameter, NOT accuracy-target-derived);
    custom CUDA kernels for the batched TT-core-product embedding lookup
    ("reconstruction"); 3 real DLRM datasets (Kaggle, Criteo Terabyte,
    Avazu); training wall-clock over exactly 1000 iterations (batch 4096,
    no warmup exclusion), single elapsed-time window (not a
    per-iteration distribution); downstream task loss/accuracy (not
    numerical reconstruction error) as the correctness/quality gate via a
    separate Table-4 script producing loss.png; DLRM/FAE/TT-Rec (and
    TorchRec/HugeCTR elsewhere in the artifact) as baselines; multi-GPU
    (1/2/4) scaling tested; "speeds DLRM training 3x on 1 GPU" headline
    (per Phase-1 one_liner).
