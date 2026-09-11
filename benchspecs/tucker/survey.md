# Tucker track — evaluation-methodology survey

Surveyed 2/2 papers. SymProp via a DOE/OSTI-hosted fulltext PDF plus artifact
repo (README + directory layout); SGD_Tucker via arXiv abstract/HTML (the
fulltext evaluation section — Section 5/6 — could not be extracted from the
public copies reached: arXiv HTML truncates before Section 5, and the two
open-access mirrors found via Semantic Scholar/Unpaywall, TU Delft's
repository, returned 404s on the direct PDF links resolved through the
landing page) plus a direct read of the artifact's C++ source
(`SGD__Tucker.cpp`), which is unusually informative for protocol questions
(exact CLI argument parsing, the hardcoded 100-iteration training loop,
`gettimeofday`-based per-iteration timer, and the RMSE/MAE convergence
reporting) even though the paper's own dataset table was not recovered.

## SymProp — Sanjay et al. (tensorworld group), "SymProp: Scaling Sparse
Symmetric Tucker Decomposition via Symmetry Propagation" (IPDPS 2025)

Source: OSTI-hosted fulltext PDF (`osti.gov/servlets/purl/3002148`, matched
by exact title from web search — OSTI mirrors DOE-funded IPDPS/SC papers)
+ `github.com/tensorworld/SymProp` README/CMakeLists (C++, MIT license,
PaRSEC-free, header-only-style `include/`+`src/` layout with `extern/` for
third-party deps, 0 stars, pushed 2025-03-27).

SymProp optimizes two kernels that ARE the per-iteration compute of Tucker
decomposition on sparse **symmetric** tensors: S3TTMc (Sparse Symmetric
Tensor-Times-Same-Matrix-chain) for Higher-Order Orthogonal Iteration (HOOI),
and S3TTMcTC (...-chain-Times-Core) for Higher-Order QR Iteration (HOQRI).
Symmetry means only ONE canonical-mode TTMc call is mathematically needed per
HOOI/HOQRI sweep, not N — the same simplification this project's mttkrp
track's `mttkrp-symmetric-kernel-fp64` variant already encodes for SymProp's
sibling MTTKRP-adjacent kernel (S3TTMc/S3TTMcTC also appear in that track's
survey, sourced from a full prior fulltext read of this same paper).

- **Workloads**: real hypergraph-derived symmetric tensors — the mttkrp
  track's survey (a full fulltext pass done earlier in this project)
  recorded 5 named real datasets (contact-school, trivago-clicks,
  walmart-trips, stackoverflow, amazon-reviews) plus 4 synthetic
  Erdos-Renyi-style generated tensors, order N swept 3-8. Today's
  second-pass OSTI extraction (a lower-fidelity PDF-to-markdown conversion)
  surfaced a partially-overlapping but not identical list — "Frossar,
  NELL-2, Delicious, Enron, DBLP" — that could not be reconciled with the
  first list; **this is flagged as an open question below** rather than
  silently picking one. NELL-2/Delicious/DBLP are canonical FROSTT tensor
  names, which is suspicious for a paper about SYMMETRIC tensors (FROSTT's
  general tensors are not symmetric) — likely OCR/table-extraction noise
  from the PDF-to-text pass, or a comparison against general (non-symmetric)
  baselines using ordinary tensors alongside the symmetric ones. Not
  independently resolved.
- **Rank**: target Tucker rank swept 2-20 (2, 4, 8, 12, 16, 20 per the prior
  fulltext pass), fixed per dataset for headline numbers.
- **Format**: CSS (compressed sparse symmetric), the paper's own novel
  storage format for sparse symmetric tensors, built once (not re-timed
  per HOOI/HOQRI call); compared against dense-general CSF-style baselines
  which must materialize the full non-canonical triangle.
- **Timing protocol**: warmup runs performed before measurement (both
  passes agree); mean execution time (2nd-pass extraction) vs. the prior
  pass's more specific "min-of-(10000 runs or 5s)" statistic reported
  ALONGSIDE mean/median in the fulltext (both are apparently reported,
  matching SySTeC's convention in the sibling mttkrp track, since these two
  papers are close collaborators/share evaluation style); 32-core OpenMP,
  single-threaded and multi-threaded numbers both reported per the prior
  pass.
- **Precision**: fp64 throughout (both passes agree; no fp32 variant
  reported).
- **Correctness**: relative-error comparison against a reference
  implementation (exact tolerance value not stated in either extraction
  pass).
- **Metric**: GFLOP/s using the SYMMETRY-REDUCED flop count actually
  performed, plus wall-clock speedup vs. baselines.
- **Baselines**: TTMc-SPLATT (general/asymmetric SPLATT-based TTMc, no
  symmetry exploitation) and S3TTMc-CSS (the paper's own prior/simpler
  symmetric format, an ablation more than an external baseline).
  Headline: S3TTMc-SP (SymProp) beats S3TTMc-CSS by 1.44-285.49x and
  TTMc-SPLATT by up to 4.98x on cases SPLATT can run at all (SPLATT
  materializes the full asymmetric tensor and runs out of memory on the
  largest symmetric cases SymProp handles, per the one_liner's "up to
  50.9x" headline figure appearing elsewhere in secondary sources)  —
  SymProp specifically enables Tucker decompositions at orders/ranks/sizes
  that were previously intractable for the general (non-symmetry-aware)
  pipeline. This "runs where baselines cannot run at all" pattern matters
  for the benchmark design: a variant that only reports geometric-mean
  speedup over the intersection of runnable cases would hide this.

## SGD_Tucker — Li, Li, Li, Rellermeyer, Chen, Li, "SGD_Tucker: A Novel
Stochastic Optimization Strategy for Parallel Sparse Tucker Decomposition"
(TPDS 2021), arXiv:2012.03550

Source: arXiv abstract page + arXiv HTML (`arxiv.org/html/2012.03550`,
truncated before Section 5) + direct inspection of
`github.com/ZixuanLi-China/SGD__Tucker` (C++/OpenMP/Armadillo, MIT license,
3 stars, single-file `SGD__Tucker.cpp`, Makefile `demo` target).

This paper is methodologically a different animal from SymProp: it treats
sparse Tucker decomposition the way matrix-factorization recommender
systems treat SGD-based collaborative filtering, not the way dense-tensor
HPC libraries treat HOSVD/HOOI. Its own stated contribution is avoiding
"intermediate variable explosion" (materializing the full Khatri-Rao/
Kronecker product across ALL nonzeros every ALS sweep) by mini-batching:
"only follows the randomly selected small samples rather than the whole
elements."

- **Workloads (from the artifact, since the paper's own dataset table was
  not recovered)**: the repo ships exactly one real dataset,
  `movielens_tensor.{train,test}` (MovieLens-derived ratings tensor,
  15.9MB train / 156KB test, tab-separated `idx1 ... idxN value` format,
  1-indexed). The paper's abstract targets "High-Order, High-Dimension, and
  Sparse Tensor (HOHDST)" generically; the demo invocation
  (`./SGD__Tucker ./Data/movielens_tensor.train ./Data/movielens_tensor.test
  4 3 4 4 4`) fixes order=3, core rank 4 per mode. Whether the paper's own
  headline numbers use additional real tensors (Netflix, Yelp, Enron,
  NELL-2 are common in this sub-literature) is an **open question** — not
  resolved from any source reached.
- **Rank**: `core_dimen[i]`, one Tucker rank per mode, passed as CLI
  arguments (`argv[5..5+order-1]`); a separate `core_kernel` argument
  controls the mini-batch/core-unfolding width used internally by the SGD
  update — both are required, disclosed, per-run parameters in the
  artifact's own CLI, which is a useful protocol fact independent of the
  paper text.
- **Iteration count**: hardcoded to exactly 100 iterations in `main()`
  (`for (int i = 0; i < 100; i++)`), each iteration running one
  `Update_Parameter_A()` sweep with a decaying learning rate
  (`learn_rate_a = learn_rate_a_init / (1 + 1.0 * pow(i, 1.5))`,
  `learn_rate_a_init = 0.002`).
  This is the artifact's literal analogue of "HOSVD/HOOI iteration count"
  for an SGD-driven decomposition — 100 SGD sweeps, not 100 HOOI power
  iterations, so it is NOT directly comparable to SymProp's HOOI/HOQRI
  iteration semantics; the two papers' "iteration" is a different unit of
  work (see Divergences below).
- **Timing protocol**: wall-clock via `gettimeofday` (`seconds()` helper),
  accumulated per-iteration into a running `time_spend` total; printed once
  per iteration alongside RMSE/MAE. No warmup iteration is excluded — the
  first (iteration 0) timing includes any first-touch/cache-cold effects,
  and is reported like every other iteration.
- **Precision**: fp64 (Armadillo `mat` = double by default; no precision
  flag in the CLI).
- **Correctness/convergence gate**: train RMSE, test RMSE, train MAE, test
  MAE computed and printed EVERY iteration (not just at the end) — this is
  a convergence-curve report, not a single pass/fail numerical-tolerance
  gate; no explicit target-RMSE stopping criterion is coded (the loop
  always runs exactly 100 iterations regardless of convergence state).
- **Metric**: per-iteration wall-clock time (cumulative), RMSE/MAE
  convergence curve; the paper's own headline claim ("at least 2x faster
  than the state of the art") is a wall-clock-to-equivalent-accuracy
  comparison, not a raw FLOP/s throughput number — this paper does not
  report GFLOP/s at all in any source reached.
- **Baselines**: not recovered from any source reached (paper text/dataset
  table unavailable); "state of the art" is unnamed in the abstract. Common
  baselines in this sub-literature (P-Tucker, Vest, CP-ALS-style sparse ALS)
  are a plausible but UNVERIFIED guess, not asserted here.

## Divergences

- **What counts as one "iteration" is incompatible between the two
  papers.** SymProp's unit is one HOOI/HOQRI sweep (dominated by one
  S3TTMc/S3TTMcTC kernel call per mode, or ONE call under symmetry);
  SGD_Tucker's unit is one stochastic-mini-batch SGD update over a randomly
  sampled subset of nonzeros. A benchmark spec that reported both under a
  single "time per iteration" number without labeling the iteration
  semantics would silently compare a full-batch power-iteration step
  against a mini-batch gradient step — this spec keeps them in separate
  variants rather than forcing one iteration-count axis across both.
- **Symmetric vs. general.** SymProp is symmetric-tensor-only (its entire
  contribution is exploiting symmetry); SGD_Tucker's MovieLens artifact
  tensor and stated HOHDST target are general (asymmetric) sparse tensors.
  This maps cleanly onto the mttkrp track's existing
  general-vs-symmetric split and this spec reuses that split.
- **Metric family**: SymProp reports GFLOP/s + speedup (classic HPC-kernel
  style); SGD_Tucker reports only wall-clock-to-accuracy (ML-convergence
  style) and never states a FLOP count or GFLOP/s number in any source
  reached. The spec's kernel-only variant (SymProp-shaped) reports
  GFLOP/s; the end-to-end variant (SGD_Tucker-shaped) reports
  time-to-reconstruction-error-target instead, rather than forcing a FLOP/s
  number SGD_Tucker's own paper never computes.
- **Correctness style**: SymProp checks numerical correctness of the
  KERNEL OUTPUT against a reference (a single-shot relative-error gate,
  the HPC-kernel-paper norm); SGD_Tucker reports a full RMSE/MAE
  CONVERGENCE CURVE against held-out test entries (the recommender-systems
  norm, closer to a reconstruction-quality track than a kernel-correctness
  track). Neither is dropped: the spec's kernel variant uses SymProp's
  gate, the end-to-end variant uses SGD_Tucker's target-RMSE style gate.

## Open questions

- SymProp's real-tensor list could not be reconciled between two extraction
  passes of the same OSTI PDF (see above); needs a direct read of the
  PDF's dataset table, not a markdown-summarized extraction.
- SymProp's numeric correctness tolerance (relative-error threshold) was
  not stated in either extraction pass.
- SGD_Tucker's own Section 5 dataset table, hardware, and named baselines
  were not recovered from any of the 4 sources attempted (arXiv HTML,
  DeepAI mirror, ar5iv, two TU Delft open-access PDF links resolved via
  Semantic Scholar/Unpaywall that both 404'd) — only the abstract and the
  artifact's C++ source were usable. A direct fetch of the arXiv PDF
  (binary, not HTML) was not attempted and may succeed where the HTML
  mirrors failed.
- Whether SGD_Tucker's paper reports any tensors beyond MovieLens (the
  only dataset shipped in the artifact) is unknown.
- SGD_Tucker's `core_kernel` CLI parameter's exact mathematical role (per
  the source, it sizes the mode-unfolding width `core_dimen[i] x
  core_kernel` used in `parameter_g[i]`, but the paper's own name/
  definition for it was not recovered) should be confirmed against the
  paper text before being treated as a stable, well-defined tuning knob in
  a harness.

## evidence

- symprop: "IPDPS'25, S3TTMc/S3TTMcTC for HOOI/HOQRI on symmetric sparse
  tensors, CSS format, real hypergraph-derived + synthetic Erdos-Renyi
  tensors, order 3-8, rank swept 2-20, fp64, warmup + mean/min-of-N
  statistic (two extraction passes partially disagree), relative-error
  correctness gate (tolerance unstated), GFLOP/s (symmetry-reduced count)
  + speedup, TTMc-SPLATT/S3TTMc-CSS baselines, up to 285x vs CSS / 4.98x
  vs SPLATT and enables previously-intractable sizes."
- sgd_tucker: "TPDS'21/arXiv:2012.03550, SGD-mini-batch ALS-style sparse
  Tucker decomposition avoiding full Khatri-Rao/Kronecker materialization,
  MovieLens artifact dataset (paper's full dataset table not recovered),
  Tucker rank + mini-batch width as required CLI parameters, hardcoded
  100-iteration SGD loop, gettimeofday wall-clock per iteration (no
  warmup excluded), fp64, per-iteration train/test RMSE+MAE convergence
  curve as the correctness/quality report (no single numerical-tolerance
  gate), no GFLOP/s reported anywhere, >=2x wall-clock-to-accuracy speedup
  vs an unnamed 'state of the art' baseline."
