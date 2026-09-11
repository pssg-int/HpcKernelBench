# Kernel-centrality tagging — instructions for the rating agents

Goal: for every (paper, kernel-group) pair in your input chunk, decide how much
the paper is *about* that kernel, and whether the paper's evaluation regime
matches what this benchmark track measures. These ratings drive which paper
artifacts become the "human SOTA" baselines of HPC-KernelBench, so be strict
and evidence-based. Do not rate popularity or quality — only centrality and
regime match.

## Inputs

- Your chunk: `data/centrality/input_chunk_NN.json`. For each group you get
  the kernel slug, a digest of `benchspecs/<slug>/spec.yaml` (what the track
  measures, its variants and their input suites/selection rules), and the
  papers (title, venue, year, platform, one_liner, abstract, assigned kernels,
  artifact URL).
- For anything the digest leaves unclear, read `benchspecs/<slug>/spec.yaml`
  and `benchspecs/<slug>/survey.md` (the survey has per-paper notes on
  workloads and timing scope for the papers surveyed in depth). You may also
  use `gh api repos/<owner>/<repo>/readme` (base64 decode) to read the
  artifact's README when the abstract is ambiguous about what ships. Do not
  clone repositories. Do not search the filesystem outside `kernel-papers/`.

## Labels

`centrality` — one of:
- `core`: the paper's headline contribution IS an implementation of this
  kernel (or a family of kernels of which this one is a primary member), and
  the paper evaluates it as a kernel (kernel-level throughput/time against
  kernel-level baselines such as vendor libraries or prior kernels).
  Examples: RoDe for spmm; a new SpTRSV algorithm for sptrsv; a CUDA FFT
  library paper for fft.
- `component`: the kernel is real and separable in the artifact, but it is one
  part of a larger system or pipeline (an inference engine, a compression
  codec, a solver stack, a geostatistics application, a GNN training system),
  OR the paper's evaluation is end-to-end rather than kernel-level, OR the
  paper is a compiler/DSL/autotuner that *generates* this kernel among others.
  Examples: PackKV's KV-cache kernels for gemv; a DSL whose example set
  includes SpMM; an AMG solver paper for spgemm.
- `tangential`: the kernel is only touched (used as a library call, a
  baseline, a sub-step never optimized), or the assignment to this group looks
  wrong. Examples: a paper that calls cuFFT for fft; a Hausdorff-distance
  paper for ann-search.

`regime_match` — one of `matches`, `partial`, `mismatch`: does the paper's
evaluated input regime match the track spec's input suite/selection?
- `matches`: same class of inputs (e.g. SuiteSparse/GNN graphs for spmm;
  standard SPD matrices for cholesky; SIFT/DEEP-style vectors for ann-search).
- `partial`: overlapping but narrower/different emphasis (e.g. only GNN graphs
  when the spec covers SuiteSparse ∪ GNN; only N:M structured sparsity when
  the spec's headline is unstructured).
- `mismatch`: a different regime entirely (e.g. 50-70%-dense pruned LLM
  weight matrices vs 1e-3-density SuiteSparse; fp16 tensor-core semi-dense vs
  the spec's fp32 general variant — say which variant, if any, it does match).

`regime` — a short free-text phrase naming the paper's actual evaluated
inputs/precision regime (e.g. "SuiteSparse + Reddit/OGB graphs, fp32",
"pruned LLaMA weights 50-70% density, fp16 tensor cores", "N:M structured").

`gpu_single_card` — true/false: the artifact has a single-NVIDIA-GPU kernel
path (the current scope of the benchmark). CPU-only, FPGA-only, multi-GPU-only
→ false.

`rationale` — one or two sentences citing the evidence (abstract wording,
survey note, README line). Plain English.

## Output

Write `data/centrality/chunk_NN.json` (same NN as your input) as a raw JSON
array, one object per (paper, group) pair, in the same order as the input:

```json
{"key": "conf/ics/ShenBSCH26", "group": "spmm",
 "centrality": "component", "regime_match": "mismatch",
 "regime": "unstructured-sparse pruned LLM weights, ~50-70% density, fp16 tensor cores",
 "gpu_single_card": true,
 "rationale": "Headline is an SpMM kernel (CDP-TCBE format) but built and evaluated for LLM inference weight sparsity; spec's general variant is SuiteSparse fp32, only the tensorcore-fp16 variant is in range."}
```

Every pair must appear exactly once. Validate the JSON before finishing
(`python -c "import json; json.load(open('data/centrality/chunk_NN.json'))"`).
Do not modify any other file.
