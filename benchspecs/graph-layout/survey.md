# graph-layout — survey

Singleton track: 1 paper.

## Li et al., "Rapid GPU-Based Pangenome Graph Layout", SC 2024
(`conf/sc/LiSDHGGPGZ24`)

- **Source**: arXiv fulltext, `arxiv.org/html/2409.00876` (WebFetch),
  cross-checked against `arxiv.org/abs/2409.00876` abstract and the GitHub
  README (`tonyjie/gpu_pangenome_layout`, fetched via `gh api`).
- **Workloads**: Human Pangenome Reference Consortium (HPRC) Year-1
  dataset — 24 whole-chromosome pangenome graphs (chr1-22, chrX, chrY),
  built with the `odgi`/PG-SGD pangenome pipeline. Per-chromosome sizes
  range: nodes 3.2x10^5 - 1.1x10^7, paths (haplotype sequences through the
  graph) 307 - 3029, total nucleotides 8.8x10^7 - 1.1x10^9, average node
  degree ~1.4 (very sparse). This is the standard, non-cherry-picked
  "one instance per human chromosome" suite — there is no ambiguity about
  selection since it's the complete chromosome set.
- **Timing protocol**: reports wall-clock layout time (hours:minutes:
  seconds format in the CPU baseline case, minutes for GPU). 15
  independent runs per pangenome were used specifically to confirm
  run-to-run layout repeatability/stability (stochastic algorithm: uses
  random node/path sampling), not explicitly stated as the number
  averaged for the headline timing numbers.
- **Timing scope**: layout computation only (Path-Guided Stochastic
  Gradient Descent, PG-SGD, run for a fixed iteration schedule);
  graph loading from the `.og` (optimized dynamic graph) file format is a
  separate step not folded into the reported layout time.
- **Precision & correctness / quality**: this is a non-convex layout
  optimization (2D embedding minimizing path-weighted stress), so there
  is no single "correct" answer — quality is assessed via the paper's own
  proposed **Sampled Path Stress (SPS)** metric: samples node pairs along
  each path, computes stress with a 95%-CI estimate (central-limit-
  theorem based) instead of exhaustive O(sum of path-length^2) stress,
  making it tractable at chromosome scale. Reported average GPU/CPU SPS
  ratio = 1.08x ("no quality loss"). Visual inspection of rendered
  layouts is used as a secondary qualitative check.
- **Metric**: primary = wall-clock layout time (with the 57.3x headline
  speedup being time-based); secondary = SPS quality score (must be
  reported alongside time, not instead of it, since this is a heuristic
  optimization problem).
- **Baselines**: `odgi layout`, the state-of-the-art multithreaded CPU
  implementation, run on a 32-core Intel Xeon Gold 6246R @ 3.4GHz.
- **Hardware**: NVIDIA RTX A6000 (primary, CUDA 11.7) and NVIDIA A100
  (secondary/cross-check, CUDA 12.2).

## Divergences

N/A — only one paper in this track. The one methodological choice worth
flagging explicitly as a spec decision (not a disagreement, since there's
only one source) is that SPS is a *sampled* stress estimator, not exact
stress — the spec requires the sample size / CI to be disclosed whenever
SPS is reported, since a paper could otherwise shrink the sample to make
the CI look tighter without disclosing precision loss.
