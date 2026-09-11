# GEMV track — evaluation-methodology survey

Kernel: dense/mixed-precision matrix-vector multiply, `y = alpha*A*x + beta*y`
(and its batched/streaming/compressed variants). All 5 papers in
`data/track_inputs/gemv.json` were surveyed; 4 via arXiv fulltext + artifact
repo code (the strongest, most concrete source in every case), 1
(TLR-MVM/Cerebras) via its artifact repo's own benchmark harness source code
after the SC'23 PDF itself could not be extracted as text.

---

## 1. QIGen — conf/cgo/PegolottiAP26 (CGO 2026; arXiv 2307.03738, earlier
   preprint of the same code/idea; repo `IST-DASLab/QIGen`)

- **Workload**: LLaMA 7B/13B/30B decode; the timed kernels are the custom
  matrix-vector products for each nonuniformly-quantized `nn.Linear` layer
  swapped in via `infergen.swap_modules_llama(...)`. Exact per-layer M,K
  dims are not given in the extracted text (they follow LLaMA's own
  attention/MLP projection shapes, standard for that model family).
- **Precision**: 4-bit, 3-bit, 2-bit weights (GPTQ-quantized) vs FP32
  baseline; quantization group sizes swept in {16, 32, 64, 128}.
- **Hardware**: single-socket AMD EPYC 7742 (64 cores/64 threads, AVX2),
  compiled with `gcc 9.4.0 -O3 -mavx -mavx2 -mfma -march=native -ffast-math
  -ftree-vectorize`, OpenMP 4.5 for threading.
- **Timing protocol**: not stated in the extractable text — no explicit
  warmup/repetition count or timer API is given; the reported number is
  end-to-end decode throughput, not an isolated per-kernel time.
- **Timing scope**: end-to-end generation (tokens/s over a 128-token
  generation), i.e. includes every layer's kernel plus all non-GEMV
  overhead — not kernel-isolated.
- **Correctness**: perplexity on wikitext2 vs the FP32/llama.cpp reference
  model (an accuracy-drop check, not a numeric per-kernel tolerance).
- **Metric**: tokens/s (generation throughput); memory footprint in MiB for
  a 128-token generation is reported as a secondary metric.
- **Baselines**: llama.cpp (int4 CPU reference kernel).
- **Headline result**: up to 2.6x speedup over llama.cpp on 13B with full
  (non-grouped) quantization; ~2x with grouped quantization.
- **Source**: arXiv 2307.03738 fulltext (html) + repo `README.md`/
  `generate.py`/`demo.ipynb`.

## 2. MARLIN — conf/ppopp/FrantarCCHA25 (PPoPP 2025; arXiv 2408.11743;
   repo `IST-DASLab/marlin`)

- **Workload**: FP16-activation x INT4-weight GEMV/thin-GEMM. Two shape
  sources, both reproduced verbatim from `bench.py`:
  - a single "ideal" shape sized to perfectly tile the GPU's SM count:
    `(4*256*SMS, 256*SMS)` (SMS = 108/72/82/84 for A100/A10/3090/A6000);
  - real LLM linear-layer shapes, `MODELS` dict in `bench.py`: Llama7B
    `(4096,3*4096),(4096,4096),(4096,2*10752),(10752,4096)`; Llama13B/33B/
    65B analogous; Falcon180B `(14848,14848*5+1024),(14848*5,14848)`.
- **Batch sweep**: `[1,2,4,8,16,32,64,128]` by default, `[1,2,...,2048]`
  in the repo's own "ALL" full-sweep mode — this is the axis MARLIN's
  central claim (near-ideal 4x speedup up to batch 16-32, degrading after)
  is built on.
- **Precision**: INT4 weight (`groupsize` in {-1 (per-column), 128}), FP16
  activation/output, FP16 group scales.
- **Hardware**: NVIDIA A10 (primary), RTX 3090, A6000, A100; ECC disabled
  on A10 (`nvidia-smi -e 0`) to reach spec-sheet bandwidth; base-clock
  locking (`nvidia-smi --lock-gpu-clocks=...`) used for the "sustained
  performance" experiment.
- **Timing protocol** (`bench.py`, `benchmark()`): `warmup=1` iteration,
  `iter=10` measured; NOT synchronized between individual iterations
  (deliberately, "to hide kernel launch overhead" like real serving would);
  `torch.cuda.synchronize()` once after the warmup point and once at the
  end; **statistic = mean** (`(time.time()-tick)/iter`), not median; a
  1-second `time.sleep()` is inserted between successive benchmark configs
  to let the GPU "cool down" and avoid clock throttling from biasing later
  runs — an explicit, code-level anti-throttling protocol step no other
  paper in this survey states.
- **Timing scope**: kernel only (no packing/reshuffling of weights timed;
  that is a one-shot `Layer.pack(...)` preprocessing step done before
  benchmarking).
- **Correctness** (`test.py`): `torch.mean(torch.abs(C - C_ref)) /
  torch.mean(torch.abs(C_ref)) < 0.001`, i.e. a **mean** (not max) relative
  error tolerance of 0.1%, checked against a dense FP16 matmul using the
  *same* dequantized reference weights (not the pre-quantization original
  weights) — isolates kernel-implementation bugs from quantization error.
- **Metric**: TFLOP/s (`2*A.numel()*C.shape[1]/time`), GB/s (bytes = 2 bytes/
  elem x (A+B+C) for the dense baseline; for the quantized kernel, bytes =
  2*A.numel() [FP16 activation] + 4*B.numel() [B stored packed, 8 int4
  values/int32, so numel=K*N/8, thus 4*numel = 0.5 byte/element] +
  2*C.numel() [FP16 output] + 2*s.numel() [FP16 scales]), and speedup vs.
  a dense FP16 `torch.matmul` baseline measured in the same script.
- **Baselines**: dense FP16 matmul (own script), other 4-bit inference
  kernels (unnamed in the extracted README, shown only as a peak-perf plot),
  vLLM standard-precision kernel (for the reported 2.8x end-to-end serving
  speedup).
- **Source**: arXiv 2408.11743 (html) + repo `bench.py`, `test.py`,
  `README.md`.

## 3. PackKV — conf/ipps/JiangYLHDJ26 (IPDPS 2026; arXiv 2512.24449; repo
   `BoJiang03/PackKV`)

- **Workload**: LLM KV-cache-compressed GEMV — the timed op is `cuBLAS
  matrix-vector multiplication on the original (uncompressed) K/V data`
  vs. PackKV's `fusion computation+decompression kernel` on the compressed
  data. The "matrix" here is the per-request KV cache (context-length x
  head-dim); context lengths tested reach 32K and 128K tokens for the
  longest-context models. 6 models (Llama2-7B/13B, Llama3.1-8B,
  DeepSeek-R1-Llama-8B, Ministral-8B-2410, Phi-4) x 6 downstream benchmarks
  (CoQA, GSM8K, MMLU, Winogrande, GPQA, SQuAD Completion) for the
  accuracy side; separate `throughput_run_a100.py` /
  `throughput_run_rtx_pro.py` scripts drive the perf side.
- **Precision**: FP16/BF16 (standard LLM inference dtype; the paper does
  not state a separate precision specifically for the throughput runs).
- **Hardware**: workstation with 1x NVIDIA RTX PRO 6000 (98 GB), cluster
  node with 4x NVIDIA A100 (40 GB) — separate gen/run/figure scripts exist
  per GPU (`*_a100.py` / `*_rtx_pro.py`), plus a dedicated multi-GPU
  scaling experiment (`throughput_multi_gpu_scaling_run_a100.py
  --gpu_count {1,2,3,4}`).
- **Timing protocol**: not recoverable in detail from the fulltext extract
  (warmup/repetition counts, timer API not stated); the repo's
  `evaluation/evaluation.py` (53KB, not read line-by-line in this pass)
  almost certainly contains the actual loop — flagged as an open question.
- **Timing scope**: kernel execution time only, explicitly contrasting
  "cuBLAS matrix-vector multiplication on the original data" against
  PackKV's fused decompression+compute kernel on the compressed data — by
  construction, decompression is INSIDE the timed region for PackKV's own
  kernel (it cannot be separated out without changing what is measured,
  since compressed-format decode is inherent to reading the operand at
  all).
- **Correctness**: accuracy-drop comparison against SOTA quantization
  methods "under the same and minimum accuracy drop" (application-level,
  not a numeric per-kernel tolerance).
- **Metric**: throughput improvement percentage vs. cuBLAS GEMV (75.7% for
  K, 171.7% for V averaged across A100/RTX Pro 6000 in the abstract;
  75.6%/171.6% in the fulltext extract — same claim, minor rounding),
  GB/s broken out per-model per-GPU (Figures 15-16 per the extracted
  summary), plus 153.2%/179.6% memory-reduction numbers for K/V
  respectively (a compression-ratio metric, separate from the throughput
  metric).
- **Baselines**: cuBLAS GEMV on uncompressed KV cache (PyTorch call).
- **Source**: arXiv 2512.24449 (html) fulltext + repo
  `ae_run_all_for_a100.sh`, `scripts/throughput_run_a100.py`,
  `evaluation/evaluation.py` (listing only).

## 4. GPU-DPF — conf/asplos/LamJ0MGLLLRLRW024 (ASPLOS 2024; arXiv
   2301.10904; repo `facebookresearch/GPU-DPF`)

- **Workload**: Distributed Point Function evaluation — structurally a
  one-hot-indicator-vector times dense table product (`table[N,
  entry_size]`, result = `one_hot(k) @ table`), which is *why* this paper
  is in the gemv track's input set, but the GPU kernel never materializes
  the one-hot vector or reads the full N x entry_size table from memory —
  it computes each output pseudorandomly via a PRF (AES-128/Salsa20/
  ChaCha20) expansion, so its performance ceiling is **compute-bound
  (PRF evaluations/sec)**, not memory-bandwidth-bound like every other
  paper in this survey. See Divergences below.
- **Table sizes** (`benchmark.py`): N (table rows) in {16384, 65536,
  262144, 1048576}; entry_size fixed at 16 (32-bit words/entry, so 64
  bytes/row) in the shipped benchmark; `test_gpu_dpf_sweep()` additionally
  sweeps N in {128,256,512,1024,8192} with random batch/entrysize for
  correctness (not perf) testing.
- **Precision**: 32-bit integer table entries; DPF keys are ~2KB regardless
  of table size (only `log2(N)` scales the key size).
- **Hardware**: P100 (README's own sample output), V100 (paper's reported
  headline numbers), tested on "cuda > 11.4".
- **Timing protocol** (`test_gpu_dpf_perf` in `dpf.py`): NO warmup;
  `reps=10`; wall-clock `time.time()` around the whole 10-call loop (not
  per-call); throughput = `batch*reps/elapsed`. No explicit
  `torch.cuda.synchronize()` shown inside the timed region — the CUDA
  calls could in principle overlap with Python-side bookkeeping between
  reps, which the code does not guard against. This is the weakest-stated
  timing protocol among the 5 papers surveyed.
- **Correctness**: exact-match check, `np.linalg.norm(rec - gt) <= 1e-8`
  after reconstructing from the two secret-shared DPF evaluations — since
  entries are exact integers this is effectively a bit-exact
  reconstruction check, not a floating-point tolerance.
- **Metric**: throughput in "dpfs/sec" (keys evaluated per second), NOT
  GB/s or GFLOP/s — consistent with the operation being compute-bound
  rather than memory-bound.
- **Baselines**: CPU DPF (1-thread and 32-thread), Google's
  `distributed_point_functions` CPU reference implementation.
- **Headline result**: >200x speedup vs. 1-thread CPU DPF, >15x vs.
  32-thread CPU DPF, on V100 with AES-128 PRF and 16 32-bit values/entry.
- **Source**: WebSearch summary of arXiv 2301.10904 (PDF fetch exceeded
  the fetch tool's size limit; abstract page also yielded little
  structured text) + repo `README.md`, `benchmark.py`, `dpf.py`
  (fulltext).

## 5. TLR-MVM / Cerebras seismic — conf/sc/LtaiefHWJRK23 (SC 2023, Gordon
   Bell Prize finalist; repo `ecrc/tlrmvm`)

- **Workload**: Tile Low-Rank Matrix-Vector Multiplication (TLR-MVM) for
  3D Multi-Dimensional Deconvolution (seismic redatuming); datasets are a
  standard seismic redatuming dataset and a MAVIS adaptive-optics
  covariance-matrix dataset (both linked from Zenodo in the README, exact
  matrix dimensions not given in the README text). Deployed on 48 Cerebras
  CS-2 systems (35,784,000 processing elements total) for the paper's
  headline MDD run.
- **Precision**: the repo builds against MKL/BLIS/cuBLAS/rocBLAS depending
  on target (single-threaded BLAS required for the TLR-compressed kernel
  itself; the dense-GEMV comparison harness below uses `float`/`cublasSgemv`
  and a `singlecomplex` variant also exists).
- **Hardware classes supported** (from `CMakeLists.txt`/README build
  instructions, illustrating the breadth of this kernel's target
  platforms across the wider track): Intel CPU, AMD EPYC CPU (BLIS),
  NVIDIA GPU (cuBLAS), AMD GPU (rocBLAS), NEC Aurora Vector Engine,
  Fujitsu A64FX — plus the Cerebras CS-2 wafer-scale target for the
  paper's own headline result (not built through this generic CMake path).
- **Timing protocol** — this is the artifact's *own* dense-GEMV comparison
  benchmark (`benchmark/cuda/bench_mvm_cuda_fp32.cpp`, a direct primary
  source since the SC'23 PDF text could not be extracted): CUDA-event
  timer (`cudaEventRecord`/`cudaEventElapsedTime`) wrapping a single
  `cublasSgemv` call per iteration, `cudaDeviceSynchronize()` +
  `cudaEventSynchronize(stop)` before/after each timed call (no leakage
  across iterations); `loopsize` iterations run (CLI arg, no fixed
  default in source — no explicit warmup iterations are discarded, all
  `loopsize` samples go into the statistic); **statistic = median**
  (`std::sort(rawtime)` then `rawtime[size()/2]`).
- **Timing scope**: kernel only; H2D/D2H happen once before the loop
  (`cudaMemcpy` outside the timed region), matching this track's
  kernel-only convention.
- **Correctness**: mean relative error vs. a single-threaded naive CPU
  GEMV reference (`checkcorrectness`: `mean(|y_gpu - y_cpu| / |y_cpu|)`)
  — reported as a raw number, no pass/fail threshold enforced in the
  benchmark binary itself.
- **Metric**: GB/s, computed as `bytes / median_time` where `bytes = M*N*
  sizeof(float) + (M+N)*sizeof(float)` — i.e. the standard GEMV
  memory-traffic model (matrix read once + input vector read once +
  output vector written once); this is the exact byte-counting convention
  this spec adopts (generalized to arbitrary dtype width) for its primary
  metric.
- **Headline paper result** (from the SC'23 abstract/press coverage, PDF
  text not recoverable): TLR-MVM on 48 CS-2 systems sustains 92.58 PB/s
  aggregate memory bandwidth on 35,784,000 processing elements for the
  MDD application — ~3x the aggregate theoretical bandwidth of Leonardo
  or Summit. This number reflects the *compressed, SRAM-resident* TLR
  kernel at wafer scale, not the dense cuBLAS comparison harness above.
- **Baselines**: dense (uncompressed) MVM via vendor BLAS (MKL/BLIS/
  cuBLAS/rocBLAS) is TLR-MVM's own "dense counterpart" baseline per the
  README's framing ("Our TLR-MVM outperforms its dense counterpart on
  many vendor architectures").
- **Source**: repo `README.md` (fulltext) + `benchmark/cuda/
  bench_mvm_cuda_fp32.cpp` (fulltext, primary source for timing protocol)
  + WebSearch summaries of the SC'23 paper (PDF fetch via WebFetch failed
  — server returned corrupted/unreadable stream both from the ACM link
  and the eScholarship mirror; no local PDF text extractor was available
  in this environment to recover it, see open_questions).

---

## Divergences

- **Memory-bound vs. compute-bound "GEMV"**: 4 of 5 papers (QIGen,
  MARLIN, PackKV, TLR-MVM) are unambiguously memory-bandwidth-bound
  matrix-vector kernels and report GB/s or a GB/s-derived metric as (or
  alongside) their headline number. GPU-DPF is structurally a
  matrix-vector product but is compute-bound (PRF evaluation cost, no
  matrix ever read from memory) and reports "dpfs/sec" instead — it is
  kept in this survey/spec as evidence and as a boundary case flagged in
  `notes_on_fairness`, but its protocol is NOT used as a template for any
  variant, since GB/s roofline-fraction reporting (this track's primary
  metric) is not meaningful for it.
- **Timing statistic**: TLR-MVM's own harness uses **median**; MARLIN's
  own harness uses **mean**; GPU-DPF uses a single wall-clock delta over
  a 10-call loop (neither mean nor median of per-call samples, effectively
  a mean but computed differently); QIGen and PackKV do not state a
  statistic in the extractable text. This spec standardizes on median
  (reporting min/max) throughout, per the same fairness principle used in
  `benchspecs/gemm/spec.yaml`, which is a deviation from MARLIN's own
  choice (mean) — flagged in `notes_on_fairness`.
- **Preprocessing/decompression scope**: TLR-MVM keeps
  tile-rank-decomposition (an offline, one-shot preprocessing step) out
  of the per-call timed region, matching this track's "preprocessing
  reported separately" convention. PackKV's kernel-vs-cuBLAS comparison
  times fused decompression+compute together *by construction* — it is
  not something that can be factored out without changing what the
  kernel actually computes (there is no "already-decompressed" state to
  start the timer at, since decompression is not a separate materialized
  step in PackKV's design). This is exactly the kind of disagreement the
  design instructions call for a dedicated end-to-end variant to resolve
  rather than silently averaging over.
- **Correctness tolerance basis**: MARLIN checks *mean* relative error
  (<0.1%) against a dense reference using the *same dequantized* weights
  (isolating kernel bugs from quantization error); TLR-MVM reports *mean*
  relative error with no enforced threshold; GPU-DPF requires *exact*
  reconstruction (its values are exact integers, not floating point);
  QIGen and PackKV validate at the *application* level (perplexity /
  downstream task accuracy), not per-kernel numeric error. None of the 5
  papers uses a *max* relative error gate. This spec's dense-kernel
  variant nonetheless adopts max-relative-error (following
  `benchspecs/gemm/spec.yaml`'s convention, for consistency across this
  benchmark suite and because max error catches per-element failures a
  mean can hide) — flagged as a deliberate, stated deviation from every
  surveyed paper's own practice.
- **Cache-state framing does not transfer cleanly from GEMM**: the GEMM
  track's "hot vs. cold whole-matrix-in-cache" framing (from LibShalom)
  does not apply naturally here — GEMV's dominant O(M*N) operand (the
  matrix) is essentially never small enough to be cache-resident across
  independent calls at any of the shapes this track's papers actually
  use (LLM linear layers, KV caches, seismic/covariance matrices,
  DPF tables are all >> LLC size), while the O(M+N) vector operand is
  *always* cache-resident and its reuse (or lack thereof) inside a single
  kernel invocation is exactly what MARLIN's design optimizes for. This
  spec therefore replaces a naive hot/cold matrix binary with a
  3-state axis (`cold-matrix`, `hot-vector-reuse`,
  `sram-resident-extreme`) — see `spec.yaml` and `notes_on_fairness`.

## Open questions carried into spec.yaml

- QIGen's and PackKV's exact warmup/repetition counts and timer APIs were
  not recoverable from the extracted text in this pass (QIGen: no
  benchmark loop code was inspected beyond `generate.py`/`README.md`;
  PackKV: `evaluation/evaluation.py`, 53KB, was listed but not read
  line-by-line).
- TLR-MVM's SC'23 PDF text could not be recovered via WebFetch (both the
  ACM DL link and the eScholarship OA mirror returned undecodable/binary
  content, and no local PDF-to-text tool — `pdftotext`/`pdftoppm` — was
  available in this environment), so the paper's own stated warmup/
  repetition protocol for the *TLR-compressed* (as opposed to the dense
  cuBLAS comparison) kernel, and the exact seismic/MAVIS matrix
  dimensions, are not confirmed from primary text — only from the
  repo's benchmark source code and secondary (search-summary) sources.
- GPU-DPF's `dpf_cpp.BATCH_SIZE` constant (the default batch size used in
  `test_gpu_dpf_perf`) is compiled into the C++ extension and not visible
  from the Python source read in this pass.
