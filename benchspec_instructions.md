# Benchmark-spec design instructions — HPC KernelBench Phase 2

You are designing a FAIR benchmark specification for one kernel track, grounded
in how the track's own papers actually evaluate. Your input file (given in the
task prompt) lists the papers: title/venue/year, DOI, arXiv id (may be empty),
OA link, and the open-source artifact repo.

## Step 1 — survey the papers' evaluation methodology

For AT LEAST 5 papers of the track (or all, if fewer), determine how they
benchmark. Sources, in order of preference:
1. arXiv fulltext: WebFetch `https://arxiv.org/abs/<id>` /
   `https://arxiv.org/html/<id>` (also try ar5iv.labs.arxiv.org/html/<id>) —
   read the Evaluation/Experimental-Setup section.
2. The artifact repo: use Bash `gh api repos/OWNER/REPO/contents[/dir]` to list
   files and `gh api repos/OWNER/REPO/contents/<path> -H "Accept: application/vnd.github.raw+json"`
   to read benchmark/run scripts (run.sh, bench*.py, Makefile targets, README
   usage). Timing loops in code reveal warmup/repetition/timer placement more
   reliably than paper text.
3. The OA link (WebFetch) or, failing everything, the abstract.

Record per paper (be concrete; "SuiteSparse" alone is not enough — note the
selection criterion and count):
- **workloads/inputs**: dataset names + sizes (e.g. "SuiteSparse, 2,893 matrices
  with nnz in [100K, 200M]"; "GEMM shapes from DeepBench + LLM GEMM M,N,K list";
  "3D grid 512^3, 7-point star")
- **timing protocol**: warmup iterations, measured repetitions, statistic
  (mean/median/min), timer (CUDA events / std::chrono / nvbench / google-bench)
- **timing scope**: kernel-only? includes host-device transfer? includes format
  conversion / preprocessing / autotuning search? amortization argument used?
- **precision & correctness**: fp64/fp32/fp16/int8; how results are validated
  (vs cuSPARSE/cuBLAS/reference CPU, error tolerance)
- **metric**: GFLOP/s, GB/s, time, speedup-vs-baseline, throughput unit
- **baselines** they compare against (helps define the benchmark's reference set)

## Step 2 — synthesize the track's benchmark spec

Design 1-4 benchmark variants that make DIFFERENT legitimate claims comparable:
- a **kernel-only** variant (steady-state, preprocessing excluded but REPORTED
  separately) — the common denominator most papers use;
- an **end-to-end** variant when the track's literature disagrees about
  preprocessing (e.g. sparse format conversion, tuning search, graph
  reordering): includes one-shot preprocessing + k executions, k chosen from
  real usage (report k explicitly);
- optionally precision-specific or size-class variants (small/medium/large)
  when papers cluster that way.

Fairness principles (apply and note where a paper's own setup violates them):
- input suite must not be cherry-picked: prefer the union/standard suite the
  papers share, with an explicit inclusion criterion; fix the exact list.
- fixed timing protocol: explicit warmup count, repetition count, statistic
  (prefer median + report min/max), one timer definition.
- preprocessing NEVER silently mixed into per-iteration time; if amortized,
  the amortization count is part of the spec.
- correctness gate before timing counts (tolerance per precision).
- report both time and the throughput metric customary in the track.

## Step 3 — write two files (create the directory)

`benchspecs/<slug>/survey.md`:
- one subsection per paper surveyed: methodology facts (bullet list) + source
  (arxiv/repo file path/abstract)
- a "Divergences" section: where papers disagree (scope, inputs, metrics)

`benchspecs/<slug>/spec.yaml` (follow this schema; concrete values, no
placeholders):

```yaml
kernel: spmm
summary: one sentence on what is being measured
operation: "C[M,N] = A[M,K](sparse) * B[K,N](dense row-major)"
variants:
  - id: spmm-csr-kernel-f32
    claim: steady-state kernel throughput, preprocessing excluded
    inputs:
      suite: SuiteSparse
      selection: "all real square matrices, 1e5 <= nnz <= 2e8 (~2800)"
      recommended_subset: [list 10-30 named canonical instances]
      dense_operand: "N in {32, 128, 256}, fp32, initialized U(-1,1)"
    protocol:
      format: CSR (given, conversion not timed)
      warmup: 10
      reps: 100
      statistic: median
      timer: device-side events (CUDA events or equivalent)
      timing_scope: kernel only; H2D/D2H excluded
      preprocessing_reported: separately, once
    metric: {primary: GFLOP/s (2*nnz*N flops), secondary: ms}
    correctness: "max rel err vs cuSPARSE fp64 reference < 1e-4"
  - id: spmm-e2e-preproc
    claim: end-to-end incl. one-shot format conversion, amortized over k=100 runs
    ...
notes_on_fairness:
  - "papers X,Y exclude conversion, Z includes it: variant split resolves this"
open_questions:
  - anything you could not resolve from the sources
evidence: {<paper key>: "what this paper did, one line", ...}
```

Be an engineer, not a diplomat: if the literature's common practice has a flaw
(e.g. only reporting best-of-N), the spec should fix it and say so. Reply with
one line when done: "<slug>: surveyed N papers (M via fulltext/code), K variants".
