# SVD track — evaluation-methodology survey

Surveyed 2/2 papers. Both are TPDS 2020 papers from the same lab (Zhang,
Liu, Wang / Walid, "TensorLet"/"Tensor and Deep Learning Lab") sharing ONE
artifact repository, so the survey draws on one repo for both, plus
whatever abstract-level detail could be recovered — neither paper has an
arXiv id or an open-access PDF location (confirmed closed via IEEE Xplore
direct fetch [blocked/empty response], Semantic Scholar
`openAccessPdf: null`, and Unpaywall `is_oa: false, oa_status: closed` for
both DOIs). This is the track with the thinnest paper-text evidence in this
project's Phase-2 survey pass so far; the artifact repo's own MATLAB/CUDA
test/benchmark scripts, which DO contain concrete sizes, sweep ranges, and
timing code, are therefore the primary evidence source rather than a
secondary cross-check.

Both papers operate on **t-SVD** (tensor SVD via the t-product): for a 3rd-
order tensor T(m,n,k), t-SVD applies a DFT along the 3rd (tube) mode, then
computes k independent ordinary matrix SVDs of the resulting m x n
frequency-domain slices, then an inverse DFT. This is the mechanism by
which "matrix shapes (square/tall-skinny)" and "full vs. truncated SVD"
this track's key axes ask about actually arise in these two papers: the
per-slice SVD shape is the m x n "matrix shape" axis, and k (the tube
count / batch depth) is an additional batching axis neither a plain-matrix
SVD track would have.

## cuTensor-Tubal — Zhang, Liu, Wang, Walid, "cuTensor-Tubal: Efficient
Primitives for Tubal-Rank Tensor Learning Operations on GPUs" (TPDS 2020,
DOI 10.1109/TPDS.2019.2940192)

Source: `github.com/YangletLiu/cuTensor_CUDA_Library_for_Transform_based_
Tensors` (C/CUDA + MATLAB reference, no declared license, 2 stars, last
push 2021-05-12) — `README.md`, `GPU/test/*` (per-primitive C++ test
drivers), `GPU/test/test.sh` (exact sweep script), `CPU/testsvd.m`
(MATLAB reference sweep script) — plus abstract-level detail from IEEE/
ResearchGate/DeepAI mirrors (no fulltext reached).

cuTensor-tubal is a LIBRARY paper: it implements and evaluates 7 primitives
(t-FFT, inverse t-FFT, t-product, **t-SVD**, **t-QR**, t-inverse,
t-normalization) each in three execution-strategy variants — "based"
(reference/naive), "streamed" (CUDA-stream-parallel across tube slices),
and "batched" (single batched-GEMM/batched-SVD-style call across all tube
slices at once). This based/streamed/batched execution-strategy split is
itself a required, disclosed parameter in every one of its own benchmark
invocations (confirmed directly from `GPU/test/test.sh`, which passes a
literal `"batched"`/`"streamed"` string argument to each test binary).

- **Workloads**: synthetic dense random tensors T(m,n,k) (both the CPU
  MATLAB reference script `CPU/testsvd.m` and the GPU `test.sh` sweep
  script confirm this — no real/named dataset is used for the core-library
  micro-benchmarks; this is a pure primitive-throughput paper, not an
  application paper).
- **Matrix shapes (this track's key axis)**: `CPU/testsvd.m` sweeps
  SQUARE frontal slices only, m=n in {50, 100, ..., 500} (step 50), 10
  points; `GPU/test/test.sh` (confirmed for the t-FFT primitive, and by
  the shared driver-file convention likely representative of the tsvd/tqr
  scripts too, though not independently re-fetched for tsvd specifically)
  sweeps m=n in {200, 300, ..., 2000} (step 100), also SQUARE. **No
  tall-skinny (m != n) configuration was found in any fetched script** —
  this track's key-axes instruction calling for both square AND
  tall-skinny shapes is NOT something either paper's own practice
  actually covers; the spec below adds tall-skinny as a variant-internal
  sweep point rather than silently reporting only the square case both
  papers use.
- **Batch depth k**: swept independently of m,n — `CPU/testsvd.m`: k in
  {128, 256, ..., 1024} (step 128), 8 points, giving an 8x10=80-point
  (m=n, k) grid; `GPU/test.sh`'s confirmed t-FFT sweep uses a fixed k=128
  while varying m=n, i.e. the GPU and CPU scripts sweep DIFFERENT axes as
  their primary variable (GPU: size at fixed batch depth; CPU: both size
  and batch depth) — not necessarily reconciled into one single sweep in
  the paper's own headline figures.
- **Full vs. truncated (this track's key axis)**: t-SVD here is always
  computed FULL (every singular value/vector of each m x n frontal slice)
  — no truncation/rank-target parameter was found in the `test.cpp`
  driver for `tsvd`; this is a primitive-throughput paper, not a
  low-rank-approximation paper (see cuTensor-completion below for the
  truncated case).
- **Timing protocol**: `GPU/test/tsvd/test.cpp` uses the C standard-
  library `clock()` function (NOT CUDA events), single-shot per
  invocation — no loop, no repetition, no warmup inside the timed test
  binary itself; a SEPARATE `GPU/test/warmup/` binary exists in the repo
  (calling an unexamined `warmup()` function), suggesting a warmup pass is
  run once before the whole sweep rather than once per configuration, but
  this was not confirmed from `test.sh`'s own invocation order.
  `CPU/testsvd.m` uses MATLAB's `clock`/`etime`, also single-shot per
  (m,k) grid point, appended to a results file (`newtsvd_time.txt`) —
  notably, in the fetched copy of `testsvd.m`, the actual `[U,S,V] =
  t_svd(T)` call is commented out, so that particular script variant as
  fetched measures loop/IO overhead only, not the algorithm — flagged
  as an open question (a different, uncommented version presumably
  produced the paper's actual numbers).
- **Precision**: not explicitly stated in any fetched script; MATLAB
  double (fp64) is the default and no single/fp32 cast was found, so fp64
  is the working assumption for the CPU reference path; the GPU C++ path's
  precision was not independently confirmed.
- **Correctness**: `GPU/test/tsvd/test.cpp`'s "streamed" execution branch
  ONLY compares its U/S/V output against the "based" (reference)
  implementation's output via element-wise differences — this IS the
  paper's actual correctness-checking mechanism (execution-strategy
  cross-validation, not comparison against an external/vendor SVD), but no
  explicit numeric tolerance was found printed/checked against in the
  fetched excerpt (raw differences appear to be printed, not gated). The
  "batched" branch has NO correctness check at all in the fetched code —
  only the "based" reference output is printed.
- **Metric**: speedup vs. a CPU baseline (dual 10-core Xeon CPUs, per the
  abstract) — up to 27.03x for t-SVD specifically (16.91x t-product,
  38.97x t-QR, 22.36x t-inverse, 15.43x t-normalization, per the same
  source), i.e. wall-clock speedup, not a GFLOP/s number, is this paper's
  own headline metric family.
- **Baselines**: the CPU reference implementation (MATLAB, on dual
  10-core Xeon) is the ONLY baseline found; no comparison against
  cuBLAS/cuSOLVER/MAGMA batched-SVD routines was found in any source
  reached (flagged as an open question — a batched-SVD library comparison
  would be the natural external baseline this project's other specs
  favor, but this paper's own practice appears to be CPU-vs-GPU-of-its-
  own-code only).

## High Performance GPU Tensor Completion With Tubal-Sampling Pattern —
Zhang, Liu, Wang (TPDS 2020, DOI 10.1109/TPDS.2020.2975196)

Source: same repo as above, specifically `GPU/apps/app1/` (the tensor-
completion application built on top of the base t-SVD/t-FFT primitives:
`fft.cu`, `one_step.cu`/`one_step.h` — an alternating-minimization
"one step" update — `norm.cpp`) and `CPU/alter_min_LS.m` /
`CPU/alter_min_LS_one_step.m` (the CPU MATLAB reference for the same
algorithm) — plus abstract-level detail (IEEE Xplore direct fetch
returned empty content; ResearchGate blocked at 403; DeepAI/Semantic
Scholar mirrors used instead).

This paper uses t-SVD as a SUB-KERNEL inside an iterative tensor-completion
(low-tubal-rank matrix/tensor recovery from partial observations) pipeline
— it needs the TRUNCATED, low-tubal-rank case, called repeatedly (once per
alternating-minimization iteration), which is the direct counterpart to
cuTensor-Tubal's one-shot FULL-SVD primitive-throughput case above.

- **Workloads**: the abstract specifically motivates the "tubal-sampling
  pattern" (as opposed to uniform element-wise sampling) as arising from
  real acquisition constraints (sensor failures, downsampling, packet
  loss in IoT); the concrete tensor sizes/sources used in the paper's own
  evaluation were NOT recovered from any source reached (only the
  algorithm's implementation — `app1/` — was inspected, not a
  paper-side dataset table).
- **TT-rank/truncation as a spec parameter (this track's tolerance axis)**:
  the algorithm is alternating-minimization for LOW-TUBAL-RANK recovery,
  so a target tubal-rank (or equivalently, a fraction-of-frontal-slice-
  rank truncation applied at every t-SVD sub-call) is intrinsic to the
  method; the exact CLI/parameter surface exposing this truncation target
  was not confirmed from the fetched `one_step.cu`/`one_step.h` headers
  (only their existence and role — "one alternating-minimization step" —
  was determined, not their parameter list).
- **Timing protocol**: not independently confirmed for this application
  beyond the base library's `clock()`-based, single-shot-per-call
  convention documented above (the completion app calls the same
  primitives).
- **Precision**: same working assumption as above (fp64 MATLAB reference,
  GPU precision unconfirmed).
- **Correctness**: for a tensor-COMPLETION algorithm the natural
  correctness notion is RECOVERY ACCURACY (relative error of the
  completed tensor vs. the known ground truth on the withheld/sampled-out
  entries), not a numerical-kernel-output check like cuTensor-Tubal's
  based-vs-streamed comparison above — this is a structurally different
  correctness axis from the sibling paper's, even though both papers are
  literally the same lab's same t-SVD implementation underneath. The
  exact tolerance/metric (RSE? RMSE? relative Frobenius error?) was not
  recovered from the sources reached.
- **Metric**: speedup, not GFLOP/s — the headline figures (per this
  project's Phase-1 one_liner and corroborated by the WebSearch abstract
  excerpt) are maximum 248.18x over CPU MATLAB, 7,403.27x over GPU
  element-sampling tensor completion (the naive/uniform-sampling GPU
  baseline, run within the SAME cuTensor-tubal library), and 33.27x over
  "GPU high-performance matrix completion" (a 2D/matrix-only competitor,
  not tensor-aware).
- **Baselines**: CPU MATLAB (own reference implementation), GPU
  element-sampling tensor completion (an ablation/naive-sampling variant
  of the SAME library, not an external system), and an unnamed "GPU
  high-performance matrix completion" system (external, 2D-only).

## Divergences

- **Full vs. truncated SVD (this track's key axis) maps cleanly onto the
  two papers, not within either one.** cuTensor-Tubal's t-SVD primitive is
  always FULL (every singular value of every frontal slice, no rank
  target). The tensor-completion paper's t-SVD sub-calls are always
  TRUNCATED (low-tubal-rank, called iteratively as part of
  alternating-minimization). Neither paper itself sweeps both regimes;
  the track-level spec must synthesize both regimes as separate variants,
  each anchored on the paper that actually exercises it, rather than
  treating "full vs. truncated" as a parameter either individual paper
  varies.
- **Matrix shape**: both papers' own evaluations (as far as recovered)
  use SQUARE frontal slices (m=n) exclusively; neither exercises
  tall-skinny shapes despite this track's key axes calling for both. This
  is flagged explicitly rather than silently matched — the spec adds a
  tall-skinny sweep point as a spec-level addition, not evidence these
  papers cover it.
- **Correctness philosophy differs by application, not by
  implementation.** cuTensor-Tubal (the primitives library) validates
  NUMERICAL AGREEMENT between its own execution strategies (streamed vs.
  based) — a strategy cross-check, not a comparison against an external
  ground truth. The tensor-completion paper validates RECOVERY ACCURACY
  against known ground-truth entries — an application-level statistical
  correctness notion. Both use literally the same underlying t-SVD CUDA
  code, so this divergence is about WHAT the correctness check is
  checking against, not a difference in numerical method.
- **No baseline outside the authors' own library for the primitives
  paper.** cuTensor-Tubal compares its GPU code only against its own CPU
  MATLAB reference, never against cuSOLVER/cuBLAS/MAGMA batched-SVD
  routines — a real gap this project's other tracks generally flag when a
  paper's own baseline set omits the obvious vendor-library competitor
  (see notes_on_fairness).

## Open questions

- Neither paper's fulltext was reached (both closed-access, no OA
  location per Unpaywall/Semantic Scholar, IEEE Xplore fetch
  blocked/empty) — all evaluation detail above comes from the shared
  artifact repo's scripts/source plus third-party abstract mirrors, not
  the papers' own Methodology/Evaluation section text. This is a thinner
  evidentiary base than every other track surveyed in this project so far.
- The exact GPU model(s) used for either paper's headline numbers were
  not recovered (only "dual 10-core Xeon CPUs" for the CPU-side baseline,
  from the abstract).
- `CPU/testsvd.m`'s fetched copy has its actual `t_svd(T)` call commented
  out — whether this reflects the script actually used to produce the
  paper's numbers, or a debug/scratch copy left in the repo, is unresolved.
- The tensor-completion paper's own concrete dataset/tensor-size table,
  its exact tubal-rank truncation parameter surface, and its exact
  recovery-accuracy metric/tolerance were not recovered from any source
  reached.
- Whether cuTensor-Tubal's `test.sh`-confirmed t-FFT sweep pattern
  (m=n in {200..2000}, k=128 fixed) is representative of the `tsvd`-
  specific sweep (not independently re-fetched) is an inference, not a
  confirmed fact — `CPU/testsvd.m`'s sweep (m=n in {50..500}, k in
  {128..1024}) is the only DIRECTLY confirmed t-SVD-specific sweep found.
- No comparison against vendor batched-SVD libraries (cuSOLVER
  `cusolverDnXgesvd`/batched APIs, MAGMA) was found for either paper —
  unconfirmed whether this is a true omission in the papers themselves or
  an artifact of the incomplete fulltext access.

## evidence

- cutensor_tubal: >
    TPDS'20 (DOI 10.1109/TPDS.2019.2940192), 7-primitive GPU library for
    tubal-rank (t-product-based) tensor operations (t-FFT/t-product/t-SVD/
    t-QR/t-inverse/t-normalization), each in based/streamed/batched
    execution-strategy variants (a required, disclosed parameter per its
    own test.sh); synthetic square-frontal-slice tensors T(m,n,k), m=n
    swept 50-2000 across two scripts, k (batch depth) swept 128-1024
    independently; t-SVD always FULL (no truncation); clock()-based
    single-shot timing (no confirmed warmup-per-config, no confirmed
    repetition); correctness = streamed-vs-based cross-check (no external
    ground truth, no stated numeric tolerance); speedup-vs-CPU-MATLAB
    metric (up to 27.03x for t-SVD specifically, on dual 10-core Xeon
    CPU baseline); no vendor-library (cuSOLVER/MAGMA) baseline found.
- gpu_tensor_completion: >
    TPDS'20 (DOI 10.1109/TPDS.2020.2975196), tensor completion for the
    tubal-sampling pattern via alternating-minimization with t-SVD as a
    repeated, TRUNCATED (low-tubal-rank) sub-kernel; same underlying
    library/repo as cuTensor-Tubal (app1/ + CPU MATLAB alter_min_LS.m
    reference); dataset table and exact truncation-parameter surface not
    recovered; correctness = recovery accuracy vs. ground truth on
    withheld entries (exact metric/tolerance not recovered); speedup
    metric (up to 248.18x vs CPU MATLAB, 7403.27x vs GPU element-sampling
    completion, 33.27x vs an external GPU matrix-completion system).
