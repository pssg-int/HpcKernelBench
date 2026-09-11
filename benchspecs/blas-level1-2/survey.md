# BLAS Level-1/2 track — evaluation-methodology survey

Kernel scope: vector-vector (level-1: dot, axpy, norm, scale) and
matrix-vector (level-2: gemv, symv/ssymv, trsv) operations, as generated
or accelerated by compiler/DSL/generator systems (this track's 4 papers
are all *tools that produce or accelerate* BLAS L1/L2 kernels, not
hand-written BLAS libraries themselves — a different flavor of "track"
than gemv, which is papers about the kernel directly). All 4 papers in
`data/track_inputs/blas-level1-2.json` were surveyed. fBLAS was surveyed
via its own artifact repo's dedicated BLAS L1/L2 evaluation harness
(the strongest and most directly on-topic source of the four, and the
only one whose evaluation code was read in full); SySTeC via its own
symmetric-matrix-vector (ssymv) kernel source + README; Exo 2 and VeGen
via arXiv/ACM search summaries plus repository structure (their PDFs
could not be extracted as text by the tools available in this pass — see
Divergences/open_questions).

---

## 1. fBLAS — conf/sc/MatteisLH20 (SC 2020; arXiv 1907.07929; repo
   `spcl/FBLAS`) — also cited as evidence in `benchspecs/gemm/spec.yaml`
   for its GEMM-side evaluation; this survey focuses on its **DOT** and
   **GEMV** evaluation code specifically, read in full from
   `evaluation/individual_modules/` and `evaluation/cpu_comparison/`.

- **Workload**: FPGA (Intel/Altera OpenCL, streaming HLS) implementations
  of `sdot`/`ddot` (level-1) and `sgemv`/`dgemv` (level-2), chosen by the
  paper explicitly "as representative samples of BLAS Level 1, 2, and 3,
  respectively" (per `evaluation/individual_modules/README.md`) alongside
  GEMM. For DOT/GEMV specifically (unlike GEMM), input data is generated
  directly on the FPGA rather than streamed from host/HBM, "to test the
  scaling behavior of these memory bound applications" at vectorization
  widths that exceed what the testbed's real memory interface (e.g. HBM)
  could sustain — i.e. these are compute-fabric/on-chip-bandwidth
  ceiling experiments, deliberately decoupled from off-chip memory
  bandwidth for this specific pair of experiments.
- **Vector-length sweep**: `n` is a CLI argument to every host program
  (`gemv_host -n 2048 -m 1024 ...`, `dot -n <length>`); the module
  JSON descriptions fix a SIMD/streaming width of 64 (`"width": 64`) for
  both `sdot` and `sgemv`, and GEMV additionally fixes tile sizes
  (`"tile N size": 2048, "tile M size": 2048`) independent of the overall
  problem size swept via CLI.
- **Precision**: float and double, both compiled as separate kernel
  variants (`sdot`/`ddot`, `sgemv`/`dgemv`) rather than templated at
  runtime — precision is a build-time choice on FPGA, unlike the other
  3 papers in this track (all CPU codegen systems where precision is
  more often a runtime/template parameter).
- **Hardware**: Intel/Altera FPGA (OpenCL/HLS toolchain,
  `CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA` env var for emulation mode);
  for the `cpu_comparison` subfolder specifically, an Intel MKL-backed
  CPU run of the same DOT/GEMV computation is measured on the SAME
  host as a baseline, plus a Mammut-library-based power measurement for
  that CPU baseline (the only paper across BOTH tracks surveyed so far,
  gemv and blas-level1-2, to report a power metric alongside
  throughput).
- **Timing protocol** (`host_dot.cpp`, read in full): per-run timing via
  `current_time_usecs()` (host-side wall-clock, microsecond resolution)
  wrapping `queues[i].enqueueTask(kernels[i])` for every kernel/helper
  module followed by `queues[i].finish()` for every queue — i.e. an
  explicit full-pipeline drain before stopping the clock, preventing
  async-queue leakage into the next timed iteration. Buffer-transfer
  time (`enqueueWriteBuffer`/`enqueueReadBuffer`) is measured
  SEPARATELY from compute time in the same loop iteration
  (`transfer_times` vs. `fblas_times`, two distinct vectors) — i.e.
  the harness enforces "preprocessing/transfer never silently mixed into
  per-iteration compute time" at the code level, matching this
  benchmark suite's own stated fairness principle exactly.
  **Repetition count**: CLI argument `-r <runs>`, no fixed default
  (README's own worked example uses `-r 1` for a quick emulation smoke
  test; the paper's real evaluation runs presumably use more, not
  stated in the README). **Statistic: mean** over `runs`, with **sample
  standard deviation** also computed and reported alongside the mean
  (`host_dot.cpp`: `stddev=sqrt(stddev/runs)`) — this is the only paper
  across both tracks surveyed so far that reports a formal dispersion
  statistic (stddev) rather than just min/max or nothing.
- **Correctness**: validated against `cblas_sdot`/`cblas_ddot` (OpenBLAS/
  MKL CPU reference) with an explicit **precision-dependent absolute
  tolerance**, `flteps`, hardcoded in `host_gemv.cpp` as `1e-4` for
  float and `1e-6` for double (read directly from source, lines ~194-198)
  — the only paper in either track surveyed so far to hardcode a
  concrete numeric tolerance value in the benchmark code itself, rather
  than only asserting "compared against a reference" with no number.
- **Metric**: GOps/s (`2*n/time` for DOT — a multiply-add pair per
  element, matching this suite's own FLOP-counting convention) and GB/s
  (`data_bytes/time`, where `data_bytes = 2*n*sizeof(T)` for DOT — the
  two input vectors, output scalar not counted since it's O(1)).
  Transfer time and transfer bandwidth are computed but reported
  separately, never folded into the compute-time metric.
- **Baselines**: OpenBLAS (correctness reference for DOT/GEMV), Intel
  MKL (performance baseline for the CPU-only comparison programs in
  `cpu_comparison/`).
- **Source**: repo `evaluation/individual_modules/README.md`,
  `evaluation/cpu_comparison/README.md`, `evaluation/cpu_comparison/
  {sdot,sgemv}.json`, `evaluation/cpu_comparison/host_dot.cpp` (all read
  in full) + arXiv 1907.07929 (cited in benchspecs/gemm/spec.yaml's own
  survey, not re-fetched here).

## 2. SySTeC — conf/cgo/PatelAA25 (CGO 2025; arXiv 2406.09266; repo
   `radha-patel/SySTeC`)

- **Workload**: a symmetry-exploiting sparse/dense tensor compiler
  (built on the Finch tensor-compiler framework) that specializes
  kernels to skip redundant work across the `n!` transpositions of an
  order-`n` symmetric tensor. Directly relevant to this track: `ssymv.jl`
  — a **symmetric matrix-vector product** kernel (`y = A*x` with `A`
  symmetric, only the canonical upper/lower triangle read), a genuine
  BLAS level-2 routine (`SSYMV`/`DSYMV` in standard BLAS naming), plus
  `syprd.jl` (symmetric outer/rank update, closer to level 2/3
  boundary).
- **Symmetrization strategy** (from the README's own worked SpMV
  example, generalizes directly to SYMV): the compiler rewrites
  `y[i] += A[i,j]*x[j]` into a loop over `i<=j` only, emitting BOTH
  `y[i] += A_ij*x[j]` and `y[j] += A_ij*x[i]` when `i != j` (a single
  read of `A_ij` updates two output elements), and only the single
  diagonal-safe update when `i==j` — this is the textbook symmetric-GEMV
  optimization (halving matrix reads), here auto-generated rather than
  hand-coded.
- **Reported savings** (README, application-level, not GEMV-specific):
  "savings ranging from 1.3x for SpMV to 7.8x for a 4-dimensional MTTKRP
  kernel" — the 1.3x figure is explicitly attributed to SpMV (sparse),
  not dense SYMV; the dense-SYMV-specific speedup number was not found
  in the README text read in this pass (would require the CGO'25 paper
  body itself, not recovered — see open_questions).
- **Precision, hardware, timing protocol, vector-length sweep,
  correctness validation**: NOT recoverable from the README (a
  usage/functionality doc, not an evaluation doc) or from the
  `ssymv.jl` kernel source itself (a Finch kernel *definition*, not a
  benchmark driver — no timing loop, no problem-size parameter, no
  correctness assertion visible in the ~15-line snippet read). The
  repo's `tests/` directory (listed but not read in this pass) is the
  more likely location for this information — flagged as an open
  question.
- **Source**: repo `README.md` (fulltext) + `level_two/ssymv.jl`
  (fulltext, kernel definition only) + WebSearch summary confirming
  venue/arXiv id/authors and the general 1.3x-7.8x savings range.

## 3. VeGen — conf/asplos/ChenMCA21 (ASPLOS 2021; repo `ychen306/vegen`)

- **Workload**: a vectorizer generator targeting non-SIMD (and SIMD)
  vector instructions, evaluated on DSP/image/ML kernels. The one
  concretely-named benchmark kernel recoverable in this pass is
  `dot_16x1x16_uint8_int8_int32`, described as coming from "TVM's 2D
  convolution layer" — a **level-1-style dot-product-with-reduction
  microkernel** (the accumulate-into-int32 structure is exactly a
  BLAS-level-1 `dot`/`sdsdot`-shaped inner loop, generalized to the
  mixed-int8/uint8-input pattern AVX512-VNNI's `vpdpbusd` instruction
  targets), plus x265's `idct4` kernel (not level-1/2, listed for
  context).
- **Target ISA**: Intel AVX2 and AVX512-VNNI specifically for the
  dot-product kernel (VNNI is the ISA extension whose whole purpose is
  fused 8-bit-multiply/32-bit-accumulate dot-product instructions,
  making this the natural non-SIMD-vector-instruction target for a
  level-1 reduction kernel).
- **Precision**: uint8 x int8 inputs, int32 accumulator — this track's
  only surveyed paper exercising an explicitly mixed-precision,
  quantized-integer level-1 kernel (relevant to this track's
  "reduction-accuracy considerations for dot/norm" axis per the task
  instructions, though from the compiler/ISA-targeting angle rather
  than a numerical-accuracy angle).
- **Metric**: speedup relative to reference auto-vectorizing compilers,
  measured in instruction count and a derived speedup ratio for at
  least one concrete example (Figure 2, per the search-summary source):
  reference 273 instructions/1.0x, ICC 106 instructions/1.5x, GCC 61
  instructions/2.2x, LLVM 4 instructions/11.0x (LLVM's own
  auto-vectorizer already reaches high speedup on this particular
  example via a different code shape), VeGen reaching AVX512-VNNI
  utilization beyond what any of the three reference compilers achieve
  (exact instruction count/speedup number for VeGen's own output not
  captured in the extracted summary). A separate, previously-known
  result (from the paper's own abstract, not re-verified in this pass)
  is "speedup 3x compared to LLVM's vectorizer" on x265's idct4 kernel.
- **Timing protocol, warmup/repetitions, statistic, correctness
  validation**: NOT recoverable — every fetch attempt against the
  paper's actual text (ACM DL PDF via WebFetch: HTTP 403; adapt.cs.
  illinois.edu direct PDF via WebFetch: binary/undecodable to the tool;
  ResearchGate: HTTP 403; docslib.org: partial/summarized text only, no
  methodology section) failed to surface a methodology section. The
  instruction-count-based metric itself suggests VeGen's evaluation may
  be closer to a static-cost-model comparison than a wall-clock
  benchmark for at least some of its reported numbers, but this is an
  inference from the one figure recovered, not a confirmed reading of
  the paper's actual evaluation-section methodology text.
- **Source**: WebSearch summary of `docslib.org`'s extracted excerpt +
  repo top-level directory listing (`gslp`, `sema`, `test-with-csmith.py`,
  `test`, `utils` — no dedicated `benchmark/`/`evaluation/` directory
  was found at the top level, unlike fBLAS/SySTeC; the benchmark harness
  is presumably inside `test/`, not explored further in this pass).

## 4. Exo 2 — conf/asplos/IkarashiQDRBR25 (ASPLOS 2025; arXiv 2411.07211;
   repo `exo-lang/exo`)

- **Workload**: a user-schedulable-language (USL) system where users
  define custom scheduling primitives; evaluated, per the WebSearch
  summary of the paper's own abstract/introduction, across "80+
  high-performance kernels, including BLAS level 1, 2, GEMM, blur, and
  unsharp mask" — i.e. BLAS L1/L2 kernels are explicitly named as part
  of the evaluated kernel set, but the repo's own `apps/` directory
  (`conv`, `halide`, `resnet`, `sgemm`, `ssyrk` under `apps/x86`) does
  NOT show a visible level-1/level-2-specific subdirectory at the paths
  explored in this pass — the BLAS L1/L2 kernels are most likely
  defined via the scheduling-library mechanism itself (Exo 2's core
  contribution) rather than as static example files, or live in a path
  not explored (e.g. inside `tests/`, or a separate companion
  benchmark-suite repo referenced by the paper but not linked from
  `artifact_url` in the track's input data).
- **Reported comparison**: "delivering performance comparable to or
  better than MKL, OpenBLAS, BLIS, and Halide on AVX2, AVX512, and
  Gemmini platforms" (WebSearch summary) — three hardware/ISA targets,
  at least two of which (AVX2, AVX512) are the same x86 SIMD ISAs VeGen
  also targets, and BLIS/OpenBLAS/MKL are exactly the vendor/reference
  BLAS libraries this track's other papers (SySTeC via Finch's own
  typical baselines, fBLAS via its `cpu_comparison` MKL baseline) also
  compare against — establishing MKL/OpenBLAS/BLIS as this track's
  de facto standard reference-library baseline set across 3 of its 4
  papers.
- **Vector/matrix-length sweep, precision, timing protocol, warmup/
  repetitions, statistic, correctness validation, exact metric
  (speedup? GFLOPS? both?)**: NOT recoverable in this pass. Every
  fetch attempt against the paper's own PDF or HTML rendering failed
  (`arxiv.org/pdf/2411.07211`: binary/undecodable to WebFetch;
  `arxiv.org/html/2411.07211` and versioned variants v3/v4: HTTP 404,
  the html mirror does not exist for this submission at the versions
  tried; `ar5iv.labs.arxiv.org`: redirected back to the abstract page
  rather than serving HTML; ResearchGate: HTTP 403); no local
  PDF-to-text tool (`pdftotext`/`pdftoppm`) was available in this
  environment to fall back on. The repo itself also predates or is
  organized independently of the specific BLAS L1/L2 benchmark files
  the paper's own evaluation used, as far as could be determined from
  directory listings alone.
- **Source**: WebSearch summary of the abstract/introduction (multiple
  independent search-result snippets corroborate the "80+ kernels
  including BLAS level 1, 2" and "MKL, OpenBLAS, BLIS, Halide... AVX2,
  AVX512, Gemmini" claims) + repo `README.md` and `apps/x86` directory
  listing (fulltext/listing, but did not surface BLAS L1/L2-specific
  benchmark code).

---

## Divergences

- **What "the kernel" even measures differs across all 4 papers**:
  fBLAS benchmarks BLAS L1/L2 as literal, named routines (`sdot`,
  `sgemv`) against their literal BLAS-library counterparts (OpenBLAS/
  MKL) — the most direct, apples-to-apples case in this track. SySTeC
  and Exo 2 instead evaluate a *code-generation system* whose output
  happens to include BLAS-L1/L2-shaped kernels among a larger and more
  diverse benchmark set (SySTeC: symmetric tensor kernels up to
  4-dimensional MTTKRP; Exo 2: "80+" kernels spanning GEMM, image
  filters, and BLAS L1/2) — for these two, an isolated BLAS-L1/2-only
  number may not even be separately reported in the paper (SySTeC's
  README only gives an aggregate range, "1.3x for SpMV to 7.8x for
  MTTKRP", not a SYMV-specific figure). VeGen evaluates a *vectorizer*
  whose target kernel happens to have a dot-product-reduction shape,
  measured (in the one recovered figure) by static instruction count
  rather than a runtime benchmark at all. This is a much less
  homogeneous track than gemv or gemm: 1 of 4 papers is "benchmark this
  literal BLAS routine", 3 of 4 are "benchmark this generator/compiler,
  one of whose outputs happens to be a BLAS L1/L2-shaped kernel".
- **Timing statistic**: fBLAS's own harness uses mean + stddev
  (explicit dispersion reporting, not just central tendency) — no other
  paper in this track states a statistic at all in the sources
  recoverable in this pass, so there is no cross-paper disagreement to
  resolve here, only a near-total absence of stated protocol outside
  fBLAS.
- **Preprocessing/transfer separation**: fBLAS's own harness already
  enforces this benchmark suite's "never silently mixed into
  per-iteration time" principle in code (separate `transfer_times` vs.
  `fblas_times` vectors) — the strongest and most literal match to this
  suite's stated fairness principle found in either track surveyed so
  far. No evidence either way for the other 3 papers.
- **Correctness tolerance**: fBLAS hardcodes a concrete, precision-
  dependent absolute tolerance (`1e-4` float / `1e-6` double) directly
  in the benchmark source — again, the only paper in this track (and
  one of only 2 papers across both tracks surveyed today, alongside
  MARLIN in the gemv track) to state a concrete numeric tolerance
  anywhere in its own artifact rather than only asserting "validated
  against a reference" with no number.
- **Reduction precision for dot/norm-style kernels not actually
  addressed by any of the 4 papers**: the task instructions call out
  "reduction-accuracy considerations for dot/norm" as a key axis for
  this track, but none of the 4 surveyed papers discusses summation
  order, pairwise/Kahan/compensated summation, or reduction-tree shape
  as a correctness or accuracy concern — fBLAS's dot-product kernel is
  presumably a simple sequential (or fixed-width-streaming) accumulation
  with no compensated-summation variant evident in its JSON config;
  VeGen's int32-accumulator dot kernel sidesteps floating-point
  reduction-order concerns entirely by accumulating in exact integer
  arithmetic. This spec therefore has to introduce the
  reduction-accuracy axis itself rather than generalize it from the
  literature — flagged explicitly in `notes_on_fairness` below.

## Open questions carried into spec.yaml

- SySTeC's own CGO'25 paper body (not just its README) was not read in
  this pass; the dense-SYMV-specific speedup number (as opposed to the
  README's SpMV-specific 1.3x and MTTKRP-specific 7.8x figures), its
  timing protocol, hardware target, and correctness-validation method
  are all unconfirmed. The repo's `tests/` directory (listed but not
  opened) is the most likely place to find a SYMV benchmark driver.
- VeGen's and Exo 2's full evaluation-section methodology (timing
  protocol, warmup/repetition counts, statistic, correctness
  validation, precise per-kernel metrics) could not be recovered from
  any source reachable by the tools available in this pass — every PDF
  fetch attempt for both papers returned undecodable binary content or
  an HTTP error (403/404), and no local PDF-to-text utility was
  available. Both papers' repos were explored at the top-level/one-
  level-deep only; a deeper search (e.g. `apps/`'s full recursive
  listing for Exo 2, `test/` for VeGen) might surface dedicated BLAS
  L1/L2 benchmark drivers not found in this pass.
- fBLAS's own top-level evaluation README references "sec. 6.B" /
  "sec. 6.D" of its paper for the individual-modules and
  cpu-comparison experiments respectively; the paper's own stated
  repetition count (`-r <runs>` value actually used to produce its
  published numbers, as opposed to the README's illustrative `-r 1`
  smoke-test example) was not confirmed from the paper text itself in
  this pass (arXiv 1907.07929 was cited via the earlier gemm-track
  survey, not independently re-fetched for this specific number).
