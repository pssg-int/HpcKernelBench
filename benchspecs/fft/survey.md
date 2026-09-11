# FFT track — evaluation-methodology survey

8 of the 10 papers in `data/track_inputs/fft.json` were surveyed with concrete,
sourced facts (well above the ≥5 bar): 3 primarily via the GitHub artifact
repo's actual benchmark/driver source read with `gh api` (TurboFFT, TurboFNO,
FlashFFTStencil — reading the timing loop in `main.cu` directly, which is more
reliable than the paper text per the task instructions), 1 via arXiv HTML
fulltext (cuHPX), 2 via a mix of arXiv abstract text + README (TurboFNO's
arXiv preprint, CLAIRE/Brunn), and 2 via README only where no arXiv id/OA
fulltext was reachable (FLUPS, cuFalcon, FFCz — FLUPS's own arXiv abstract page
also contributed). The two remaining papers (ZhangLW20 tensor completion,
ZhangLWW20 cuTensor-tubal) share one artifact repo
(`YangletLiu/cuTensor_CUDA_Library_for_Transform_based_Tensors`) whose README
lists a `t-fft` primitive with three execution modes (based/streamed/batched)
but gives no timing-protocol detail; they are recorded briefly under
"Evidence" but were not surveyed in the same depth as the other 8 (no readable
PDF, no arXiv id, and the repo's benchmark driver was not read this pass — see
`open_questions`).

`arxiv.org/pdf/<id>` and `ar5iv.labs.arxiv.org/html/<id>` both returned raw PDF
binary for the two 2020-era papers checked (CLAIRE, and indirectly the CLAIRE
ar5iv mirror) rather than parseable text — `poppler-utils` (`pdftoppm`) is not
installed in this environment, so those PDFs could not be rendered page-by-page
either. Where this happened, the `arxiv.org/abs/<id>` plain abstract page
(which DOES render as text) was used instead, at the cost of losing
methodology detail that lives only in the PDF body.

---

## TurboFFT — co-designed high-performance and fault-tolerant FFT on GPUs
PPoPP 2025, `conf/ppopp/WuZ0HJDDCC25`, no arXiv id, artifact `shixun404/TurboFFT`

- **Workloads**: 1D complex-to-complex (C2C) power-of-two FFT, `N = 2^1 .. 2^25`.
  Two sweep modes selected by `--if_bench`: mode 1 fixes `N` and sweeps batch
  `bs` upward by powers of 2 up to a per-N cap; mode 2 fixes the **total** work
  `N * bs = 2^25` and trades size against batch (`bs = 2^(param1 - logN)`) —
  i.e. the repo's own benchmark explicitly separates a "size-scaling" study
  from a "fixed-total-work, batch-vs-size-tradeoff" study (`TurboFFT/main.cu`).
- **Precision**: both fp32 (`float2`) and fp64 (`double2`), fully templated —
  every benchmark mode is run at both precisions (`run_A100.sh`).
- **Timing protocol**: `ntest = 10` (`TurboFFT_main`, `main.cu`). ONE untimed
  warmup pass through the full radix-decomposition kernel sequence runs before
  `cudaEventRecord(fft_begin)`; the timed region then runs the kernel sequence
  `ntest` times, with a `cudaDeviceSynchronize()` call **inside** the loop
  after every individual kernel launch (not just once at the end); elapsed
  time is `cudaEventElapsedTime(...) / ntest` — i.e. the reported statistic is
  the **mean** of 10 reps, not median, and no min/max is recorded.
- **Timer**: CUDA events (`cudaEventRecord`/`cudaEventElapsedTime`).
- **Timing scope**: kernel only; H2D/D2H excluded. Per-`N` kernel code is
  **generated at build time** by a Python codegen step
  (`fft_codegen_stride.py`) driven from a param file — this is TurboFFT's
  analogue of "plan creation," but it happens once at compile/build time, not
  per-run, and is not counted anywhere in the reported per-call timing.
- **Correctness**: `utils::compareData<DataType>(output_turbofft, output_cufft,
  N*bs, 1e-4)` — max-abs/rel tolerance `1e-4` against cuFFT as the reference,
  same precision (fp32 vs fp32 cuFFT, fp64 vs fp64 cuFFT).
- **Metric**: `GFLOP/s = 5 * N * log2(N) * bs / elapsed_time` (the standard
  radix-2 Cooley–Tukey FLOP-count convention) and memory bandwidth
  `GB/s = N * bs * sizeof(DataType) * 2 / elapsed_time`.
- **Baselines**: cuFFT (via `profiler::cufft::test_cufft`, same benchmark
  harness/timer) and VkFFT (via a separate PyTorch+VkFFT-backend script,
  `test_VkFFT.py`, run out-of-band and merged into the same plot data).
- **Fault tolerance (secondary contribution, not core FFT throughput)**:
  ABFT checksum-based error detection/correction fused into the FFT kernel,
  benchmarked at `thread_bs ∈ {1,2,4}` against an "offline" (separate
  recompute-based) fault-tolerant FFT baseline, both with and without
  injected errors.
- **Hardware**: A100-SXM4-40GB + AMD EPYC 7713 (System A, ~2 hr full sweep);
  Tesla T4 + Intel Xeon Silver 4216 (System B, ~30 min full sweep) — both
  documented with exact CUDA/gcc/cmake versions in the PPoPP artifact-appendix
  README.
- **Source**: `gh api repos/shixun404/TurboFFT/contents/TurboFFT/main.cu`,
  `.../run_A100.sh`, `.../readme.md` (PPoPP25 artifact-appendix doc).

---

## TurboFNO — fused FFT-GEMM-iFFT GPU kernel for Fourier Neural Operators
SC 2025, `conf/sc/WuZDZHC25`, arXiv 2504.11681 (companion preprint), artifact
`shixun404/TurboFNO` (git submodules the `TurboFNO_dev` branch of TurboFFT)

- **Workloads**: complex-to-complex (C2C) FFT only, frequency domain
  **truncated to size 64** after the high-frequency filter (the FNO use case,
  not a general FFT benchmark). 1D config
  (`benchmark_config/problem_size_1d.txt`): `bs ∈ {1,2,4,...,32768}` (15
  powers-of-two values), `dimX=1`, `DY ∈ {128,256}`, `N ∈ {64,128}`,
  `K ∈ {8,16,...,128}` (step 8). 2D config
  (`benchmark_config/problem_size_2d.txt`): `bs ∈ {1,8,16,...,128}` (17
  values), `DX,DY ∈ {128,256}`, `N ∈ {64,128}`, same `K` sweep. `N` here is
  the FFT length; `K` is the GEMM hidden/channel dimension the FFT output
  feeds into — this is an FFT-GEMM co-design benchmark, not FFT in isolation.
- **Precision**: fp32 only — "the baseline PyTorch implementation uses single
  precision" (arXiv text); no fp64 path evaluated.
- **Timing protocol**: "reported performance data are averaged over tens of
  runs to minimize fluctuations" (arXiv text) — mean statistic, no explicit
  warmup-discard count, no min/max, and no stated timer type in the extracted
  text (the TurboFFT submodule it's built on uses CUDA events, so this is the
  likely mechanism but is not independently confirmed for TurboFNO itself —
  flagged in `open_questions`).
- **Baselines**: PyTorch (eager-mode cuFFT+cuBLAS), NVIDIA cuBLAS+cuFFT called
  manually (removing PyTorch dispatch overhead), and a hand-written CUDA
  baseline replicating PyTorch's exact operation sequence.
- **Ablation design**: 5 kernel-fusion variants benchmarked head-to-head —
  E (unfused baseline) → A (FFT+GEMM+iFFT, separate kernels) → B (fused
  FFT+GEMM) → C (FFT + fused GEMM + iFFT) → D (fully fused FFT-GEMM-iFFT) —
  i.e. the paper's own headline claim is a *speedup ladder* across fusion
  levels on the identical problem-size sweep, not a single number.
- **Metric**: speedup vs. PyTorch baseline ("up to 150%, average 67% gain"
  across the sweep, arXiv text); also raw ms per configuration (README sample
  output: `1D_A, bs=1, dimX=1, DY=128, N=64, K=8, TIME=0.026ms`).
- **Correctness**: not addressed in either the README or the extracted arXiv
  text (flagged in `open_questions`).
- **Hardware**: A100-PCIE-40GB, host 1×64-core AMD EPYC 7763 @ 3.5GHz boost.
- **Source**: `gh api repos/shixun404/TurboFNO/contents/README.md`,
  `.../benchmark_config/problem_size_{1d,2d}.txt`; arXiv abs/html text for
  2504.11681 (fulltext PDF page-render unavailable, no `pdftoppm`).

---

## FlashFFTStencil — bridging FFT to memory-efficient stencil computation on TCUs
PPoPP 2025, `conf/ppopp/HanLCBZYCZCY25`, no arXiv id, artifact
`KevinWu2017/FlashFFTStencil`

- **Workloads**: stencil computation reformulated as FFT-domain convolution
  via a Prime Factor Algorithm (PFA) decomposition — the FFT sizes used are
  NOT plain powers of two but a mixed-radix pair: `rfft_size = 64`,
  `fft_size = 128`, combined via `pfa_size = 56` (= `64 - (2*(kernel_radix))`
  overlap-save-style tiling, `KERNEL_SIZE ∈ {29, 29, 31}` for the 3 named 1D
  stencils, so the usable sub-tile is `pfa_size - (KERNEL_SIZE - 1) = 54` for
  the 29-tap kernels). Stencil shapes: 1D (`Heat-1D`, `1D5P`, `1D7P`), 2D
  (`Heat-2D`, `Box2D9P`, `Star2D9P`, `Box2D25P`), 3D (`Heat-3D`, `Box3D27P`).
  CLI: `{1d,2d,3d}.out shape input_size time_iteration_size [--custom]`.
- **Precision**: fp64 (`double`) in the 1D driver (`1d_main.cu`); random
  U(0,1) input fill, random stencil weights (unless `--custom`).
- **Preprocessing**: `CreatePlan(kernel_weights, KERNEL_SIZE, ...)` is called
  ONCE, before the malloc/copy/launch sequence and outside any timed
  iteration loop in `1d_main.cu` — i.e. FFT-plan/PFA-decomposition setup for
  the (fixed, small) stencil kernel is explicitly a one-shot preprocessing
  step in this codebase's own structure, separate from the per-tile FFT
  execution.
- **Correctness**: `areArraysEqual(gpu_result, cpu_result, length, epsilon)` —
  compared against a **direct spatial-domain convolution reference**
  (`stencil1D`, periodic wraparound), NOT against a reference FFT — i.e. this
  validates the stencil OUTPUT, not FFT numerical accuracy per se, since FFT
  is only an internal implementation detail of this codebase's stencil op.
- **Baselines** (from `benchmarks/` directory layout): `cufft-by-pytorch`
  (naive FFT-based stencil convolution built on PyTorch/cuFFT) and `cudnn`
  (direct-convolution stencil via cuDNN) — i.e. FlashFFTStencil is compared
  against both an FFT-based competitor and a non-FFT direct-convolution
  competitor.
- **Metric**: speedup — "2.57× average speedup over SOTA... 103.0× speedup
  in 1D cases vs. cuFFT-based stencil implementations" (README headline).
- **Hardware**: developed/validated on A100 and H100 (Ampere/Hopper), CUDA
  12.1, NVIDIA PyTorch 23.05 container; README states it "should work
  seamlessly" on other Ampere/Hopper cards (3090/4090) but those are not the
  benchmarked hardware.
- **Source**: `gh api repos/KevinWu2017/FlashFFTStencil/contents/README.md`,
  `.../src/1D/1d_main.cu`, `.../src/1D/check_correct.hpp`.

---

## FLUPS — flexible and performant massively parallel Fourier transform library
TPDS 2023, `journals/tpds/BaltyCG23`, arXiv 2211.07777 (abstract page only —
fulltext PDF page-render unavailable), artifact `vortexlab-uclouvain/flups`

- **Workload**: distributed 3D unbounded/semi-unbounded/periodic Poisson
  solves via FFT (C++ library, FFTW backend on CPU, MPI for distribution).
  Supports both cell-centered and node-centered data layouts (the v2.x
  extension this paper describes) and vector-field (`lda>1`) solves including
  a Biot–Savart mode.
- **Scale**: "performance metrics ... analyzed and detailed on various
  top-tier European facilities up to 49,152 cores" (arXiv abstract) — weak
  scaling is the implied study design given the unbounded-domain use case
  (Green's-function-based solves scale with per-core problem size), though
  the exact per-core grid size at that core count was not confirmed from the
  abstract text alone.
- **Baseline**: AccFFT, "using a periodic case" (arXiv abstract) — i.e. the
  cross-library comparison is restricted to the domain type AccFFT itself
  supports (periodic), not the unbounded/semi-unbounded cases that are
  FLUPS's actual novel contribution.
- **Communication pattern (distributed all-to-all, directly relevant to this
  track's distributed-FFT axis)**: TWO alternative implementations, selected
  at COMPILE time via `-DCOMM_NONBLOCK`: (1) an MPI all-to-all — "the default
  robust option" per the README — and (2) non-blocking point-to-point
  (further split into persistent-request and `MPI_Datatype`-based variants).
  README's own performance guidance: prefer non-blocking specifically when
  per-core unknowns are high (~128³) AND total core count is not too high
  (~<10k); all-to-all is recommended otherwise.
- **Plan-creation / setup time (this track's FFT analogue of "preprocessing,"
  explicitly separated)**: FFTW's own planner effort is a build/run-time
  choice (`FFTW_FLAG ∈ {FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT,
  FFTW_EXHAUSTIVE}`), and a precomputed FFTW "wisdom" file can be loaded via
  `HAVE_WISDOM="path"` to skip planning entirely on repeat runs. Separately,
  FLUPS precomputes the Green's function AND the communication pattern once
  per solver instance ("solve the Poisson problem many times ... at low cost
  using precomputed Green's function and communication patterns" — README)
  — i.e. the library's own design already treats one-shot setup (FFT
  planning + Green's function + comm-pattern construction) as amortized
  across repeated solves, matching this track's plan-creation-vs-transform
  split almost exactly.
- **Built-in profiler**: a `PROF` compile flag enables per-function timing
  breakdown via the `h3lpr` profiler dependency (used for the paper's own
  performance analysis, not further detailed in the extracted text).
- **Precision**: not explicitly stated in the extracted text; FFTW-based C++
  double arithmetic is the library's implicit default (flagged as an open
  question rather than assumed).
- **Correctness**: "validated against analytical solutions for unbounded,
  semi-unbounded, and periodic domains" (README) — a continuous-PDE analytic
  reference, not a numerical FFT-vs-FFT comparison; also a large CI/validation
  test matrix (1000 boundary-condition combinations × 8 Green's-function
  kernels × 2 data layouts = 16,000 correctness tests, run as "daily" nightly
  regression, per README's "Testing" section) — a notably more thorough
  correctness regime than any other paper surveyed in this track.
- **Source**: `gh api repos/vortexlab-uclouvain/flups/contents/README.md`;
  `arxiv.org/abs/2211.07777` (abstract text).

---

## CLAIRE (Brunn et al.) — multi-node multi-GPU diffeomorphic image registration
SC 2020, `conf/sc/BrunnHBMM20`, arXiv 2008.12820, artifact
`andreasmang/claire`

- **Workload**: FFT is one of three co-optimized computational kernels
  (interpolation, high-order finite-difference operators, and 3D FFT) inside
  a full Gauss–Newton–Krylov PDE-constrained solver for diffeomorphic image
  registration — this is an end-to-end application benchmark, not a
  standalone FFT microbenchmark; the FFT itself is used for the
  Poisson-like preconditioner/regularizer inside the solver.
- **Sizes**: 256³-resolution registration solved in 5 seconds on a single
  NVIDIA V100; largest run scales to **2048³** (~25 billion unknowns,
  "approximately 152× larger than the largest problem solved in
  state-of-the-art GPU implementations") on **64 nodes × 256 V100 GPUs**
  (TACC Longhorn) — a weak-scaling design (problem size grows with GPU
  count). A CLARITY microscopy dataset at 1024×768×768 is also mentioned as
  a real (non-power-of-two, non-cubic) test image.
- **Distributed communication**: "device direct communication" (GPUDirect,
  implied NCCL/MPI-CUDA-aware) for the main computational kernels including
  FFT — the multi-GPU analogue of FLUPS's CPU-side all-to-all, but the exact
  FFT communication primitive (alltoall vs. alltoallv vs. custom) could not
  be confirmed from the abstract-page text alone.
  distributed all-to-all pattern used for the FFT specifically could not be
  confirmed beyond "device direct communication" from the reachable text.
- **Baseline / metric**: "a performance speedup of 70% compared to the
  state-of-the-art" on the single-GPU 256³ case (comparison target — prior
  CLAIRE CPU/AccFFT-based version, or a competing GPU registration code —
  not confirmed from the extracted text).
- **Precision, timing protocol (warmup/reps/statistic/timer), and explicit
  FFT-only vs. end-to-end timing scope**: NOT recoverable from the reachable
  text — both `arxiv.org/pdf/2008.12820` and the ar5iv mirror returned raw
  PDF binary rather than parseable text in this environment (no
  `pdftoppm`/poppler-utils installed to render pages), and the plain
  `arxiv.org/abs/2008.12820` page only exposes the abstract, not the
  methodology section. Flagged in `open_questions`.
- **Source**: `arxiv.org/abs/2008.12820` (abstract text only).

---

## cuHPX — GPU-accelerated differentiable spherical harmonic transforms on HEALPix
IPDPS 2026, `conf/ipps/ChengSWB26`, arXiv 2510.01785, artifact `NVlabs/cuHPX`

- **Workload**: spherical harmonic transform (SHT) and inverse SHT on the
  HEALPix grid. A full SHT decomposes into a 1D FFT along each iso-latitude
  ring (the longitude direction) followed by an associated-Legendre-transform
  along latitude — cuHPX's benchmarked "FFT-adjacent" component is this ring
  FFT plus the surrounding data-remapping/regridding step, not a standalone
  1D/2D/3D FFT in isolation.
- **Sizes**: `nside` up to 1024 (HEALPix resolution parameter, itself NOT a
  power of two in general — `nside` need only be a positive integer for
  general HEALPix, though power-of-two `nside` is the historically common
  case); band-limit `ellmax = 3*nside - 1` (≈3071 at `nside=1024` — a
  mixed-radix / non-power-of-two transform length by construction). Memory
  for the `(N_theta, ellmax, mmax)` coefficient tensor "exceeds 70 GB" at
  `nside=1024, ellmax=2048`.
- **Precision**: not explicitly stated in the extracted text (flagged open
  question); paper emphasizes reaching "machine precision" agreement with
  reference implementations rather than stating a working dtype.
- **Baselines**: `healpy` (single-threaded CPU reference implementation),
  `ducc` (20-thread CPU, AMD EPYC 9454), `S2FFT` (JAX-based GPU library),
  `torch-harmonics` (equiangular-grid-only — used specifically for the
  cross-grid regridding comparison, not the native-HEALPix SHT comparison).
- **Metric**: elapsed time (ms) and speedup ratios — "20× vs. ducc (20
  threads), 200× vs. healpy, 7× vs. S2FFT" at `nside=1024`; "300× speedup on
  H100 vs. multithreaded ducc" specifically for the data-remapping/regridding
  step.
- **Correctness**: round-trip error `‖f − iSHT(SHT(f))‖` in both L2 and L∞
  norms; spectral-convergence check across resolutions; numerical agreement
  with `healpy` "to machine precision"; sensitivity analysis across 3
  quadrature-weighting schemes (equal/ring/pixel weights). A differentiability
  check (gradient-descent convergence on real WMAP cosmic-microwave-background
  sky data) validates the autograd path specifically, beyond pure numerics.
- **Hardware**: RTX 4090, A100, H100 (H100 is the primary platform for the
  headline speedup numbers).
- **Source**: `arxiv.org/html/2510.01785` (HTML fulltext).

---

## cuFalcon — adaptive parallel GPU implementation of Falcon post-quantum signing
TPDS 2026, `journals/tpds/LiWSYDZ26`, no arXiv id, artifact
`encryptorion-lab/cuFalcon`

- **Workload**: FFT is an internal kernel of the Falcon lattice-based
  signature scheme's floating-point fast-Fourier sampling (`ffSampling`) step
  — a SMALL, fixed-size, high-throughput-batched complex FFT
  (`Falcon_N ∈ {512, 1024}`, the two NIST Falcon parameter sets), not a
  general-purpose FFT library benchmark. This is the track's clearest
  "NTT-adjacent" representative even though Falcon itself deliberately uses
  a real/complex FFT rather than a modular NTT (its ring is not
  NTT-friendly), because the size/batch/precision regime — small power-of-two
  N, very high batch count, custom floating-point representation
  (`fpr.cuh`) — mirrors what an actual NTT-based scheme (Kyber/Dilithium)
  would also stress.
- **Timing protocol**: CUDA events (`cudaEventRecord`/`cudaEventElapsedTime`)
  wrap ONE untimed-loop, single-shot measurement of a full signing batch
  spread across `stream_num = 16` concurrent CUDA streams × `BATCH`
  signatures per stream (`cuFalcon_512/main.cu`). No explicit repeated-trial
  loop, no discarded warmup, and no min/max/median reported anywhere in the
  read excerpt — total wall time for the one batch is printed directly
  (`"sign_opt-512::Total execution time for all streams: %.2f ms"`), implying
  the paper's reported throughput metric is signatures/second computed from
  that single aggregate measurement.
- **Correctness**: a verification path is referenced in the README ("perform
  signing and verify correctness") but the specific tolerance/comparison
  logic was not located in the portion of `main.cu` read this pass (flagged
  as an open question).
- **Source**: `gh api repos/encryptorion-lab/cuFalcon/contents/README.md`,
  `.../cuFalcon_512/main.cu`.

---

## FFCz — Fast Fourier Correction for spectrum-preserving lossy compression
IPDPS 2026, `conf/ipps/RenUDKLYCG26`, no arXiv id (arXiv 2601.01596 cited in
the repo's own future-dated BibTeX entry but not independently checked),
artifact `rcrcarissa/FFCz`

- **Workload**: 1D/2D/3D real-valued forward+inverse FFT used as a
  frequency-domain error-correction pass layered on top of an existing lossy
  compressor's decompressed output — both a CPU path (FFTW-based,
  `ffcz_cpu`) and a GPU path (CUDA, `ffcz`) are provided from the same CLI.
- **Precision**: BOTH single (`-f`) and double (`-d`) precision are supported
  as a top-level CLI flag — the only paper surveyed in this track besides
  TurboFFT to explicitly benchmark both precisions on the SAME workload
  rather than picking one.
- **Example workload** (README usage example): 3D grid `512×512×512`,
  single precision.
- **Dual-domain correctness gate (the paper's core contribution, and directly
  relevant to this track's correctness field)**: TWO simultaneous error
  bounds are enforced at once — a SPATIAL bound (`-M ABS|REL <epsilon>`) and
  a FREQUENCY-domain bound (`-F ABS|REL|PTW <delta>`, where `PTW` is a
  pointwise-in-frequency bound) — i.e. this paper's whole premise is that
  spatial-domain-only error bounds (which is what every other lossy
  compressor guarantees) are insufficient, and it adds an explicit
  frequency-domain guarantee via FFT-domain correction.
- **Target domains** (from the abstract, motivating the grid-size class):
  cosmological simulation power spectra, turbulent-combustion energy
  spectra, and X-ray/ptychography diffraction patterns — all real-valued,
  regular-grid, 3D scientific data, structurally the same input class FLUPS
  and CLAIRE target.
- **Timing protocol / baselines / metric**: not determined from the README
  alone (the CPU/GPU source directories were not read in depth this pass —
  flagged as an open question); the tool is framed as a one-shot
  edit/decompress operation, not an iterated microbenchmark, so a
  repetition-count convention may not even apply the way it does for the
  library-style FFT benchmarks above.
- **Source**: `gh api repos/rcrcarissa/FFCz/contents/README.md`.

---

## (Light coverage) Tensor-completion / cuTensor-tubal pair
TPDS 2020, `journals/tpds/ZhangLW20` + `journals/tpds/ZhangLWW20`, shared
artifact `YangletLiu/cuTensor_CUDA_Library_for_Transform_based_Tensors`

- The shared README documents a `cuTensor` GPU library implementing 6
  transform-based tensor primitives (`t-fft`, `t-product`, `t-svd`, `t-qr`,
  `t-inv`, `t-normalization`), each with THREE execution-mode variants
  explicitly listed in a table: **based** (single-stream baseline),
  **streamed** (CUDA-stream-overlapped), and **batched** (batched-plan
  execution) — this 3-way based/streamed/batched split is a genuine and
  useful axis for this track's own batch-dimension variant design, distinct
  from how TurboFFT/TurboFNO sweep batch size as a plain scalar parameter.
- No timing protocol, problem sizes, precision, or baseline detail was
  recoverable from the README alone; the repo's `GPU/` source tree (not read
  this pass) would be needed to extract the actual benchmark driver, the
  same way it was for TurboFFT/TurboFNO/FlashFFTStencil. Recorded here as a
  named data point (the `based/streamed/batched` axis) rather than a fully
  surveyed paper — see `open_questions`.

---

## Divergences

1. **Statistic (mean vs. median)**: TurboFFT explicitly averages `ntest=10`
   reps (`elapsed_time / ntest`); TurboFNO explicitly averages "tens of
   runs"; cuFalcon reports a SINGLE untimed measurement with no repetition at
   all. None of the three code-level-verified papers in this track report a
   median or a min/max spread, unlike some other tracks' surveyed papers
   (e.g. stencil's AN5D, which at least reports mean-of-5). This spec departs
   from all three by requiring median + reported min/max (see `spec.yaml`).

2. **Synchronization granularity inside the timed region**: TurboFFT's own
   timed loop calls `cudaDeviceSynchronize()` after EVERY individual kernel
   launch inside the `ntest`-rep loop (not once at the end) — this adds
   `kernel_launch_times * ntest` sync-barrier stalls to the measured time
   rather than `ntest` (one sync per full transform). This is a real
   measurement-methodology artifact in the reference implementation, not
   necessarily a flaw in TurboFFT's own reported relative comparisons (cuFFT
   is timed with the same harness/convention so the comparison is
   internally consistent), but it means TurboFFT's absolute GFLOP/s numbers
   are NOT directly comparable to a differently-synchronized harness without
   correcting for this.

3. **Preprocessing / "plan creation" means something different in each
   codebase's own architecture**: cuFFT/FFTW-style libraries (used as
   baselines throughout) have an explicit runtime `plan` object
   (`cufftPlan*`, `fftw_plan_dft*`) created once and reused — this is the
   textbook "preprocessing" analogue. TurboFFT and TurboFNO instead
   template-generate their kernels at BUILD/COMPILE time via a Python codegen
   script reading a param file — there is no runtime "plan creation" call at
   all in their own benchmark harness; the analogous cost is a one-time,
   off-critical-path code-generation + compilation step, never measured
   inside any of their reported numbers. FlashFFTStencil sits in between: it
   has an explicit runtime `CreatePlan()` call for its fixed PFA
   decomposition, called once outside the timed loop. FLUPS is the most
   FFTW-textbook case, additionally separating THREE one-shot costs (FFTW
   planner effort, Green's-function precomputation, communication-pattern
   construction) that can each be individually cached/reused/wisdom-loaded
   across repeated solves. This track's spec (see `spec.yaml`,
   `fft-plan-creation` variant) treats "plan creation" broadly enough to
   cover both the classical runtime-plan case AND the build-time-codegen
   case, and requires whichever cost is nonzero to be measured and reported
   once rather than assumed away.

4. **Precision coverage**: TurboFFT and FFCz explicitly benchmark BOTH fp32
   and fp64 on the same workload; TurboFNO benchmarks fp32 ONLY; FLUPS's
   precision is not stated in the reachable text (FFTW-double is the
   library's likely implicit default, but this is inferred, not confirmed);
   FlashFFTStencil's own driver uses fp64 (`double`) even though the paper
   targets Tensor Core Units, which more commonly run fp16/tf32 — i.e. this
   specific driver file may not represent the paper's own headline
   TCU-precision numbers (only the 1D CPU-reference-comparison driver was
   read, not the TCU kernel itself). This track's spec mandates evaluating
   BOTH fp32 and fp64 as separate reported numbers per the union-suite
   fairness principle, rather than picking whichever one convenience
   dictates per paper.

5. **Batch axis is expressed differently across papers**: TurboFFT/TurboFNO
   treat batch (`bs`) as a first-class, independently-swept scalar parameter.
   FLUPS/CLAIRE express "batch" implicitly as vector-field components (`lda`
   in FLUPS's API) rather than independently-batched transforms. The
   cuTensor-tubal README documents batch as a distinct EXECUTION MODE
   (`based` vs. `streamed` vs. `batched`) rather than a size parameter swept
   within one mode — i.e. "batched" there means "uses cuFFT's/cuBLAS's
   batched-plan API," a qualitatively different code path, not just a larger
   number.

6. **Distributed communication pattern and transport differ by
   platform**: FLUPS is CPU+MPI, offering a compile-time choice between
   classic MPI all-to-all and non-blocking point-to-point (persistent or
   `MPI_Datatype`-based) communication, with README-documented guidance on
   which regime favors which core-count/per-core-size combination. CLAIRE is
   multi-GPU+MPI, described only as "device direct communication" (implying
   GPU-aware MPI / NCCL-style transport) — the specific all-to-all primitive
   used for its FFT communication could not be confirmed from the reachable
   abstract-page text. This track's distributed variant (see `spec.yaml`)
   keeps CPU-side all-to-all-vs-non-blocking and GPU-side device-direct
   communication as separate, explicitly labeled sub-cases rather than
   conflating them.

7. **What "correctness" even means differs by paper**: TurboFFT validates
   numerically against cuFFT at a stated tolerance (1e-4) — a true
   FFT-vs-FFT numerical check. FlashFFTStencil validates the STENCIL output
   against a direct spatial-domain convolution reference, not FFT output
   against a reference FFT — correctness there is one level removed from FFT
   accuracy itself. cuHPX validates SHT round-trip self-consistency
   (`‖f − iSHT(SHT(f))‖`) plus agreement with an independent CPU
   implementation (`healpy`) to machine precision, plus an
   application-level differentiability/gradient-descent check — a
   substantially richer correctness regime than a single scalar tolerance.
   FLUPS validates against continuous analytic PDE solutions, not a
   numerical FFT reference at all, backed by a very large (16,000-case)
   nightly regression suite. FFCz's whole contribution IS a dual-domain
   (spatial + frequency) error-bound guarantee, explicitly rejecting the
   spatial-only correctness notion every other lossy-compression paper uses.
   No single tolerance number is universal across this track's own papers;
   this spec's correctness field is deliberately per-variant rather than
   track-wide.

## open_questions

- CLAIRE's precision, exact timing protocol (warmup/reps/statistic/timer),
  and FFT-only vs. end-to-end timing scope could not be recovered — both
  `arxiv.org/pdf/2008.12820` and its ar5iv mirror returned raw PDF binary in
  this environment (no `pdftoppm`/poppler-utils available to render pages),
  and the plain abstract page has no methodology section.
- TurboFNO's exact timer mechanism (CUDA events vs. host wall clock) and
  warmup-discard count were not confirmed independently of TurboFFT's own
  (confirmed) convention; the arXiv text only states "averaged over tens of
  runs."
- cuFalcon's correctness-verification tolerance/comparison logic exists
  (README references it) but was not located in the read portion of
  `main.cu`.
- FFCz's timing protocol, baselines, and metric are undetermined — only the
  README was read; the CPU/GPU source directories were not opened this pass.
- The `cuTensor` (`t-fft`) primitive's `based`/`streamed`/`batched` 3-way
  split is recorded as a useful axis, but no problem sizes, precision, or
  timing protocol could be recovered without reading the `GPU/` source tree,
  which was not done this pass.
- FLUPS's precision (single vs. double) is inferred (FFTW-double, the
  library's likely default) but not confirmed from any reachable text.
- CLAIRE's specific distributed-communication primitive for FFT ("device
  direct communication") is not confirmed to be all-to-all, all-to-allv, or
  a custom pattern.

## evidence

- `conf/ppopp/WuZ0HJDDCC25` (TurboFFT): "1D C2C power-of-two FFT, N=2^1..2^25,
  fp32+fp64, ntest=10 mean via CUDA events (1 warmup, per-launch device
  sync), vs. cuFFT (1e-4 tol) + VkFFT, GFLOP/s (5*N*log2N*bs) + GB/s,
  A100/T4, size-sweep and fixed-total-work-tradeoff sweep both included."
- `conf/sc/WuZDZHC25` (TurboFNO): "C2C FFT truncated to 64, fused into
  FFT-GEMM-iFFT for FNO, fp32 only, bs/DX/DY/N/K swept per
  benchmark_config/*.txt, averaged over tens of runs, vs. PyTorch/cuBLAS+
  cuFFT, 5-stage fusion ablation (E/A/B/C/D), speedup up to 150%, A100."
- `conf/ppopp/HanLCBZYCZCY25` (FlashFFTStencil): "FFT-as-convolution for
  stencils via PFA (rfft=64,fft=128,pfa=56 mixed-radix), fp64, plan created
  once outside timed loop, correctness vs. direct-conv reference (not vs.
  FFT reference), vs. cuFFT-based and cuDNN-based baselines, 2.57x avg /
  103x 1D speedup, A100/H100."
- `journals/tpds/BaltyCG23` (FLUPS): "distributed CPU FFTW-based unbounded/
  semi-unbounded/periodic Poisson FFT, up to 49152 cores, vs. AccFFT
  (periodic only), all-to-all vs. non-blocking comm (compile-time choice),
  FFTW planner effort + wisdom caching + precomputed Green's function/comm
  pattern all explicitly separated from per-solve time, validated vs.
  analytic solutions (16000-case regression suite)."
- `conf/sc/BrunnHBMM20` (CLAIRE): "FFT as one of 3 co-optimized kernels in a
  multi-GPU diffeomorphic registration solver, 256^3 single-V100 (5s) up to
  2048^3 on 64 nodes x 256 V100 (weak scaling), device-direct comm, 70%
  speedup vs. SOTA on the single-GPU case; precision/timing-protocol
  unrecoverable from reachable text (PDF-only)."
- `conf/ipps/ChengSWB26` (cuHPX): "ring-FFT + Legendre SHT on HEALPix,
  nside up to 1024 (ellmax=3*nside-1, mixed-radix), vs.
  healpy/ducc/S2FFT/torch-harmonics, round-trip L2/Linf + machine-precision
  agreement + differentiability check, 20x/200x/7x speedups, H100 primary."
- `journals/tpds/LiWSYDZ26` (cuFalcon): "small fixed-N (512/1024) batched
  complex FFT inside Falcon post-quantum signing (ffSampling step),
  single-shot CUDA-event-timed batch across 16 streams, no repeated trials
  or warmup observed, NTT-adjacent size/batch/precision regime."
- `conf/ipps/RenUDKLYCG26` (FFCz): "1D/2D/3D real FFT for frequency-domain
  lossy-compression error correction, CPU(FFTW)+GPU(CUDA) both fp32/fp64,
  example workload 512^3 fp32, dual spatial+frequency error-bound gate
  (ABS/REL/PTW), timing protocol/baselines undetermined (README only)."
- `journals/tpds/ZhangLW20` + `journals/tpds/ZhangLWW20` (tensor
  completion / cuTensor-tubal): "shared repo documents t-fft primitive with
  based/streamed/batched execution-mode axis; no sizes/timing/precision
  recoverable from README alone (light coverage only)."
