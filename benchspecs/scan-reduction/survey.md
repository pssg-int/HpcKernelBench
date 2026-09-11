# Scan/reduction track — evaluation methodology survey

Track input: `data/track_inputs/scan-reduction.json`, 3 papers, all 3
surveyed (2 via code-verified artifact + arXiv fulltext, 1 via code-verified
artifact + partial arXiv fulltext — appendix inaccessible), above the
>=2-paper minimum.

Operation surveyed: an associative(-commutative) reduction of n values to a
scalar, or to H << n segment-keyed accumulators (generalized histogram), or
a CASCADED/fused reduction feeding a downstream op within one kernel launch
(e.g. softmax normalization -> GEMM). **None of the three papers implements
a literal Blelloch/Hillis-Steele all-prefixes SCAN** (an output array of n
partial sums) despite the track's name — flagged explicitly in Divergences
and open_questions rather than silently forced into the spec.

---

## 1. Navarro et al. — GPU Tensor Cores for Fast Arithmetic Reductions (TPDS'21)

- key: `journals/tpds/NavarroCBRV21`, arXiv:2001.05585 (fulltext via
  ar5iv), artifact: github.com/crinavar/gpu-tensor-core. The track input's
  own listed repo (`crinavar/xxxxx`) is a placeholder/redacted name that
  404s; the real repo was located by searching `user:crinavar` on GitHub
  and confirmed by matching its README's `./prog <dev> <n> <factor_ns>
  <seed> <REPEATS> <method>` CLI signature and MMA-reduction source files
  against the paper's algorithm description ("chained m x m matrix
  multiply-accumulate operations").
- **workloads/inputs** (code-verified, `reduction/tensor-cores-MMA/src/
  main.cu`, `scripts/benchmark-comparison.sh`, `scripts/benchmark-cub.sh`):
  array size n swept via `seq STARTN DN ENDN` (caller-supplied range,
  invoked per-experiment by the plotting scripts); paper text (via ar5iv)
  states initial variant comparisons around "~1 million" elements, extended
  to "n >= 10x10^6" for the error-vs-n curves, with Figure 8 spanning
  multiple orders of magnitude. Two input distributions: Normal(mean=0,
  variance=1) and Uniform(0,1), matching the `DISTRIBUTION=("Normal"
  "Uniform")` arrays present in every benchmark script.
- **timing protocol** (code-verified, `main.cu` L31-190,
  `benchmark-cub.sh`): TWO-LEVEL structure. Inner: a `REPEATS` CLI argument
  (CUDA-event timed via `cudaEventElapsedTime`, divided by `REPEATS`
  inside the binary — `time/(REPEATS)` printed). Outer: the shell harness
  invokes the binary `SAMPLES` times and computes MEAN/VAR/STDEV/STDERR
  across the `SAMPLES` invocations via a Welford-style running-mean
  bash/`bc` computation — i.e. `SAMPLES` independent process launches, each
  already an internal `REPEATS`-iteration mean. No warmup phase found in
  `main.cu` (`REPEATS` includes the first, cold iteration in its average).
- **timing scope**: kernel-only. `main.cu` allocates and H2D-copies the
  input once (`Ad`->`Adh`) before the timed region, device-resident
  throughout the loop; the CPU gold-reduction (`gold_reduction`, used only
  for the correctness check) is separately timed with `omp_get_wtime()`
  and NOT included in the GPU number.
- **precision & correctness**: input array is fp32 (`float A[]`); the
  tensor-core MMA path internally uses FP16 inputs to the matrix-multiply
  (per the paper text: "the operation A x B is done in FP16 precision" —
  the hardware constraint of Volta-generation tensor cores) with FP32
  accumulation, while an `omp_reduction<double>` variant (alg 5) provides
  an fp64 CPU baseline. Correctness = `fabs(100*(gpu_sum-cpu_sum)/cpu_sum)`
  — a **percentage relative error**, not a fixed tolerance gate; the
  paper's own text reports this stays "below 1 percent" once n >= 10x10^6
  for Normal-distributed input, and "below 0.001%" for Uniform(0,1) input
  in the single-pass fp32 variant — i.e. the acceptable error is itself
  n-dependent and distribution-dependent, never a single fixed number. One
  documented FAILURE MODE: "the FP16 CUB implementation did not pass this
  test as the result overflowed at an early stage" — a genuine correctness
  bug in a baseline, reported as such rather than silently omitted.
- **metric**: primary is **BEPS (billion elements per second)**, e.g.
  "~420 BEPS" for the single-pass variant on a V100 — NOT GB/s; a
  throughput-of-elements metric, convertible to GB/s only by fixing an
  element byte-width (4 bytes for fp32 input -> BEPS*4 = GB/s). Also
  reports raw runtime (ms) and speedup ratios.
- **baselines**: `warpshuffle_reduction` (standard non-tensor-core GPU
  reduction), `recurrence_reduction` and `split_reduction` (the paper's own
  intermediate variants), CUB (fp16 and fp32 builds, `../CUB/prog_cub16`/
  `prog_cub32`), `omp_reduction` (CPU OpenMP, fp32 and fp64). Claims ~3.2x
  over standard GPU reduction, "slightly faster" than CUB-fp16 and 2x
  faster than CUB-fp32 for the single-pass variant. Hardware: Tesla V100
  primary, TITAN RTX in an appendix.
- source: repo `reduction/tensor-cores-MMA/src/main.cu`,
  `scripts/benchmark-comparison.sh`, `scripts/benchmark-cub.sh`,
  `scripts/plot-all.sh`, `TODO`; arXiv fulltext (2001.05585, via ar5iv).

## 2. Henriksen et al. — Compiling Generalized Histograms for GPU (SC'20)

- key: `conf/sc/HenriksenHSO20`, no arXiv (author-hosted/Zenodo only),
  artifact: github.com/diku-dk/futhark-sc20 (Zenodo-archived, DOI badge in
  README).
- **workloads/inputs** (code-verified, `benchmarks/cub/genhisto_*.cu`,
  `prototype/histo-main.cu`): the core `genhisto` microbenchmark
  (`prototype/histo-main.cu`, matching Section IV.B / Table II) fixes
  `INP_LEN = 50,000,000` (50M) input elements and sweeps the **histogram
  size H** (`Hmax = 4,000,000`) — this is a SEGMENTED/keyed reduction where
  H (number of output segments) is the swept axis, not just the input
  size n. Confirmed via the released raw-data dump `histograms.json`:
  logged configs include H in {63, 127, 505, 2041, ..., 1,572,863} at fixed
  input length 50,000,000 — spanning ~4 orders of magnitude of H, from
  heavy-contention (few bins, many threads write the same bin) to
  near-conflict-free (H approaching the theoretical max). Three
  race-resolution/kernel strategies compared per H: `cas` (compare-and-swap
  atomics), `hdw` (hardware atomic add), `xcg` (exchange-based). Five
  further application-level benchmarks reuse the same generalized-histogram
  primitive at their own natural sizes: 435.gromacs (SPEC CPU2006 port),
  image registration (`img-reg`, PyTorch-vs-Futhark), k-means (`kmeans`,
  vs. the kmcuda library), and Parboil's `histo`/`tpacf`.
- **timing protocol** (code-verified, `prototype/histo-main.cu` L13-43):
  `#define GPU_RUNS 100` (**100 GPU-timed repetitions**),
  `#define CPU_RUNS 1` (CPU reference computed once, for correctness only,
  not timed competitively). The released raw-data file
  (`benchmarks/cub/histograms.json`) confirms this empirically: every
  logged config's `"runtimes"` array has exactly 100 entries (in
  microseconds). No explicit warmup phase separates a "cold" iteration
  from the 100 — inspection of the raw traces shows the first few samples
  of several configs running measurably slower in an irregular pattern
  (e.g. one trace opens `496,497,499,1025,482,487,...` — an outlier at
  position 4, not position 1, suggesting GPU clock/thermal transients
  rather than a simple cold-start effect), then settling; all 100 are kept,
  not trimmed.
- **timing scope**: kernel-only; the SPEC/Parboil/kmeans/img-reg
  application-level benchmarks each measure their own natural timed region
  (a Makefile producing the relevant Table III row) — the `img-reg` README
  explicitly warns "this benchmark runs on random data and does not
  perform any validation," an honest disclosure that this one downstream
  benchmark's numbers are timing-only, uncorroborated by correctness.
- **precision & correctness**: `genhisto` operates on user-supplied
  associative-commutative operators (paper text: "arbitrary associative
  and commutative operators") — the released microbenchmark's raw data
  (histogram counts, effectively an integer add-reduction per bin) is
  exact/deterministic-in-value regardless of thread interleaving (integer
  addition is associative and commutative in finite-precision hardware
  arithmetic, unlike floating-point addition — a distinction this
  benchmark's own choice of operator sidesteps rather than stress-tests).
  No explicit tolerance/error-check code was located for `genhisto` itself
  beyond an implicit "total count must equal input length" invariant.
- **metric**: runtime (ms/us); the paper's own Table II reports raw
  runtimes per (H, strategy) cell, not a derived GB/s or elements/sec
  number for the core `genhisto` benchmark, though the design-space
  exploration in `prototype` (Figure 7) explicitly reasons about L2 cache
  occupancy and bandwidth when choosing among race-resolution strategies.
- **baselines**: CUB-based hand-written histogram implementations
  (`genhisto_cas.cu`/`genhisto_hdw.cu`/`genhisto_xcg.cu`, one per
  race-resolution strategy) are the direct baseline for the
  Futhark-generated code in the `cub/` comparison; kmcuda (k-means
  library), PyTorch (image registration), and the original SPEC/Parboil
  C/CUDA implementations are the baselines for the five downstream
  application benchmarks. Abstract claims: "outperforms similar primitives
  from CUB."
- source: repo `README.md`, `benchmarks/README.md`,
  `benchmarks/cub/README.md`, `prototype/README.md`,
  `prototype/histo-main.cu`, `benchmarks/cub/histograms.json` (raw data),
  `benchmarks/img-reg/README.md`, `benchmarks/kmeans/README.md`,
  `library/README.md`.

## 3. Tang, Li, Wang, Shu, Ling, Xing, Zhou, Liu — RedFuser (ASPLOS'26)

- key: `conf/asplos/TangLWSLXZL26`, arXiv:2603.10026 (fulltext partially
  retrieved — main-body evaluation prose extracted via arxiv.org/html and
  ar5iv-equivalent rendering; **Appendix A.5, which the main text
  explicitly points to for exact benchmark shapes/configs, was not
  reachable through either fetch path** — flagged as a gap below rather
  than guessed), artifact: github.com/alibaba/redfuser (built on Apache
  TVM; 21 stars; author list confirmed via the repo's own README bibtex
  entry).
- **workloads/inputs**: the paper's own text (per WebFetch extraction)
  names four representative cascaded-reduction subgraph classes: (1)
  Multi-Head Attention (BERT, ViT, LLaMA-65B shapes) and Multi-Latent
  Attention (DeepSeek-R1) — softmax-then-GEMM, the paper's headline
  example; (2) Flash-Decoding; (3) MoE routing — GEMM followed by softmax
  and **top-k selection** (Switch Transformer, ERNIE, DeepSeek-V2-Lite,
  Qwen3 configs) — the same cascaded-reduction-then-select pattern this
  project's topk-selection track also covers, from the opposite side (here
  the reduction/normalization is the object of study, top-k is the
  downstream consumer); (4) FP8 per-token-quantization + GEMM (dynamic
  scaling, itself a reduction, feeding a quantized GEMM). Code-verified
  concrete example shapes from the repo's own codegen entry points (used
  as the `main()` default config in each example script, which generates
  a Tile-lang kernel — NOT necessarily the exact shape used in the paper's
  own timed evaluation table, which per the paper text lives in an
  inaccessible appendix): `flash_attention.py` — q/k/v =
  [128, 16, 512, 64] (batch=128, heads=16, seq_len=512, head_dim=64), fp16
  with fp32 accumulation; `moe_routing.py` — A = [4096, 4096] fp16,
  softmax + top-k(k=8) fused in one kernel (`topi.tir_reduce_topk`).
- **timing protocol**: **not located** in the reachable paper text or
  repository code within this survey's search — no timing harness (e.g.
  `do_bench`, a CUDA-event wrapper) was found via GitHub code search
  (`repo:alibaba/redfuser do_bench` and `... cuda.Event` both returned
  zero hits) inside the example/codegen scripts reviewed
  (`flash_attention.py`, `moe_routing.py` generate Tile-lang kernel SOURCE
  CODE; they do not themselves time it). The paper text (per WebFetch)
  states results only as "RedFuser achieves 2.8x and 2.6x speedup over
  PyTorch Dynamo and TVM" for MHA — no warmup count, repetition count, or
  timer identity confirmed from either source. This is the
  weakest-verified paper of the 6 surveyed across both tracks in this
  delivery.
- **timing scope**: unknown from reachable sources (open question).
- **precision & correctness**: fp16 compute with fp32 accumulation is
  code-confirmed in both example shapes (`accum_dtype="float32"` in
  `flash_attention.py`); an fp8 quantization case study is discussed in
  the paper's abstract/text but the exact dtype used per-benchmark in the
  timed evaluation table could not be confirmed (appendix-gated). No
  correctness/tolerance-check code was found in the reviewed example
  scripts (they are codegen drivers, not correctness-checked benchmarks);
  the paper text does not describe a validation methodology in the
  reachable excerpt.
- **metric**: latency and speedup ratios only (per WebFetch extraction of
  the paper text) — no GB/s or bandwidth-utilization number found anywhere
  in the reachable material, despite this being the track's only paper
  whose headline operation (online-softmax's running-max/running-sum
  recurrence, fused into a GEMM consumer) is structurally closest to a
  genuine memory-bound cascaded reduction. This is consistent with an
  end-to-end-fused-kernel framing where the point is eliminating an
  intermediate HBM round-trip rather than characterizing one kernel's own
  bandwidth.
- **baselines**: PyTorch v2.7 Eager, PyTorch Dynamo + Inductor (Triton
  codegen), Apache TVM v0.21 (Relax frontend, no CUTLASS/FlashInfer), and
  hand-optimized libraries FlashAttention2 and FlashMLA. Hardware: NVIDIA
  A10 (24GB) and H800 (80GB), CUDA 12.8.
- source: arXiv 2603.10026 (`/abs/`, `/html/` — main-body prose only,
  appendix not retrieved); repo `README.md`,
  `python/tvm/redfuser/example/{flash_attention.py, moe_routing.py}`;
  GitHub code search (`do_bench`, `cuda.Event`) returned no results within
  the artifact.

---

## Divergences

- **None of the three papers implements a literal all-prefixes scan.** The
  track is named "scan-reduction" but the three input papers cover a
  scalar reduction (Navarro), a segmented/keyed reduction into H << n
  output bins (Henriksen), and a cascaded reduction fused with a
  downstream consumer op inside one kernel (RedFuser's online-softmax
  running-max/running-sum recurrence, structurally scan-LIKE — it carries
  a running accumulator across sequential tile steps — but never emits an
  output array of n partial sums the way a Hillis-Steele/Blelloch scan
  does). This is flagged rather than silently patched: the spec's three
  variants below are grounded in what these three papers actually measure
  (flat reduction, segmented reduction, fused/online reduction), and a
  dedicated inclusive/exclusive all-prefixes-scan variant is left as an
  explicit gap in `open_questions` rather than invented with no grounding
  in this input set.
- **The throughput metric unit differs across all three papers, and none
  uses the same one.** Navarro: BEPS (billion elements/sec), convertible
  to GB/s only by fixing an assumed byte-width. Henriksen: raw runtime
  (ms/us) per (H, strategy) cell, no throughput derivation for the core
  `genhisto` benchmark. RedFuser: latency + speedup ratio only, no
  bandwidth number at all despite its operation being memory-bound in
  nature. The spec fixes GB/s (bytes read + bytes written, standard
  roofline accounting) as the primary metric for the flat and segmented
  variants (both are genuinely memory-bound single-pass-over-input kernels
  where this is meaningful) and keeps latency/speedup as primary for the
  fused variant (where the point is eliminating a round-trip, not
  characterizing one kernel's own bandwidth).
- **Determinism/reproducibility of floating-point reduction is a real,
  under-discussed correctness axis in this track.** Navarro's tensor-core
  MMA path changes the REDUCTION ORDER relative to a tree or warp-shuffle
  reduction (accumulation happens inside systolic matrix-multiply units,
  not a simple binary tree) — floating-point addition is non-associative,
  so this is not merely a speed difference but a genuine "different
  algorithm, different bit pattern" situation, which the paper itself
  acknowledges by reporting a PERCENTAGE ERROR (not requiring bit-identity)
  and by documenting a genuine baseline FAILURE (FP16 CUB overflowing).
  Henriksen's three race-resolution strategies (cas/hdw/xcg) resolve
  concurrent writes to the SAME histogram bin via atomics, whose
  accumulation order across threads is run-to-run non-deterministic for
  any non-associative operator (the released microbenchmark sidesteps this
  by using an associative integer-count operator, but the paper's own text
  claims support for "arbitrary" operators, so a float-sum histogram would
  inherit this non-determinism — untested in the released artifact).
  RedFuser's online/blockwise softmax rescaling likewise changes
  accumulation order relative to a single-pass sum. The spec makes an
  explicit run-to-run bit-identity check (same input, same algorithm,
  repeated `reps` times) a REQUIRED reported field for all three variants
  — none of the three papers reports whether their OWN technique is
  run-to-run deterministic, only whether it matches a reference within
  tolerance once.
- **Array-size vs. segment-count are different sweep axes conflated under
  one track name.** Navarro sweeps n (input size) with H implicitly = 1
  (reduces to one scalar). Henriksen fixes n = 50,000,000 and sweeps H
  (output segment count) from 63 to 1,572,863 — the opposite axis. Neither
  paper sweeps both simultaneously. The spec's `scan-reduction-flat-kernel`
  and `scan-reduction-segmented-histogram` variants keep these as
  genuinely separate variants rather than forcing a shared 2D sweep grid
  neither paper's artifact actually explores.
- **Warmup/repetition rigor ranges from "documented two-level statistic"
  to "not locatable at all."** Navarro: REPEATS (internal, CUDA-event,
  divided-mean) x SAMPLES (external, shell-level, mean/var/stdev/stderr) —
  no explicit warmup, first iteration folded into the REPEATS average.
  Henriksen: GPU_RUNS=100 fixed, no explicit warmup separation (raw traces
  show occasional mid-run outliers, not a clean cold-start pattern).
  RedFuser: no timing harness locatable in the reachable code or paper
  text at all — the weakest-verified paper of the six surveyed across both
  tracks in this delivery. The spec fixes warmup=20/reps=100/median+range
  uniformly (the project's standing convention across other kernel
  tracks), which is stricter than any of the three papers' own documented
  practice and does not depend on RedFuser's unlocatable protocol.

## Open questions carried into spec.yaml

See `open_questions` in `spec.yaml`: the missing literal-scan variant,
RedFuser's unlocatable timing protocol and appendix-gated benchmark shapes,
and the untested float-sum-histogram non-determinism question are all
carried forward rather than resolved by guessing.
