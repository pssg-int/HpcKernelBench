# Survey: lossless-compression track evaluation methodology

Track input: `data/track_inputs/lossless-compression.json` (7 papers, bit-exact
compressors spanning generic byte streams, HPC/scientific multi-byte integer
data, and LLM-weight tensors). None of the 7 papers ship an arXiv id in the
track input except Rapidgzip; all 7 are surveyed below — 2 via arXiv/PDF
fulltext (Rapidgzip via ar5iv, GPULZ via its own repo-hosted PDF extracted
with `pypdf`) and 5 via artifact-repo README + raw benchmark/harness source
(`gh api repos/OWNER/REPO/contents/...`). Reading the actual timing-loop code
(not just READMEs) was essential here: 4 of the 7 repos hardcode their
warmup/repetition counts as C/C++ `#define`/CLI-default constants that are
nowhere stated in any README or abstract.

---

## 1. Rapidgzip (Knespel & Brendel, HPDC'23) — arXiv 2308.08955, fulltext via ar5iv

- **workloads/inputs**: 3 corpora, each scaled to fill core counts by
  concatenating copies: (1) base64-encoded random-content data (128–512 MiB
  per core for rapidgzip itself, 1 GiB for the single-threaded baselines,
  compression ratio 1.315 — a near-incompressible control case since pugz's
  restrictive byte-9–126 requirement made this the only content pugz can
  handle); (2) Silesia corpus (426 MiB uncompressed per unit, ratio 3.1,
  scaled by concatenating 2–256 gzip tarballs); (3) FASTQ biological
  sequence data from SRA accession SRR22403185 (362 MiB uncompressed, ratio
  3.74) — a genomic-data control distinct from Silesia's generic text/binary
  mix.
- **timing protocol**: benchmarks **repeated 20–100 times**, results reported
  as mean **± 1 standard deviation**. Output is written to `/dev/null` and
  the input file resides in `/dev/shm` (tmpfs) specifically to remove disk
  I/O from the measurement; process pinning via `taskset`. Core counts swept
  at {1, 16, 48, 64, 96, 128, 256}.
- **timing scope**: decompression only — Rapidgzip is a decompressor, not a
  compressor (input files are pre-existing standard-gzip output at various
  tool/levels: bgzip, gzip, igzip, pigz levels 0–9, zstd, bzip2, lz4). I/O is
  explicitly excluded by construction (tmpfs source, `/dev/null` sink).
- **precision & correctness**: byte-stream, format-agnostic gzip/DEFLATE.
  Correctness is not explicitly re-verified in the disclosed evaluation
  methodology text — the paper's correctness argument is structural (the
  cache+prefetcher architecture "can safely handle faulty decompression
  results" from trial-and-error mid-stream starts), not an empirical
  byte-for-byte comparison reported per benchmark run.
- **metric**: decompression bandwidth (GB/s or MB/s), speedup vs. GNU gzip
  and igzip, parallel scaling efficiency across core counts.
- **baselines**: the widest baseline set in the track — GNU gzip 1.12, igzip
  (Intel ISA-L), pigz 2.7, pugz, bgzip 1.17, pzstd 1.5.4, lbzip2 2.5, lz4
  1.9.4, zstd 1.5.4 (9 tools).
- **hardware**: single AMD Rome node, 2-socket AMD EPYC 7702 (64 cores/socket,
  256 hardware threads with SMT), 2 GHz base clock, 512 GiB RAM.
- **source**: arXiv fulltext (`ar5iv.labs.arxiv.org/html/2308.08955`).

## 2. GPULZ (Zhang et al., ICS'23) — repo-hosted PDF fulltext (hpdps-group/ICS23-GPULZ)

- **workloads/inputs**: 6 named datasets from 2 benchmark suites. TPC-H:
  `tpch-int32` (int32 integers), `tpch-string` (UTF-8 strings). SDRBench-
  derived: `hurr-quant`, `hacc-quant`, `nyx-quant` — these are **NOT** raw
  scientific floating-point fields; they are cuSZ's own intermediate
  **uint16 quantization-code streams**, generated from the HACC/Hurricane
  ISABEL/Nyx SDRBench fields at a relative error bound of 1e-3, i.e. GPULZ is
  evaluated as a drop-in replacement/companion for cuSZ's lossless
  entropy-coding back-end, not as a general floating-point compressor.
  `rtm` (float32, seismic imaging, petroleum exploration) is the one direct
  floating-point input.
- **timing protocol**: a full grid ablation over chunk size C ∈
  {2048, 4096, 8192, 16384} bytes, sliding window W ∈ {32, 64, 128, 255}, and
  symbol length S ∈ {1, 2, 4} bytes (default reported configuration:
  C=2048, W=128, S=2). **No warmup/repetition count or statistic (mean vs.
  median) is disclosed anywhere in the Experimental-Setup section** — Table
  1/2's numbers read as single-run results.
- **timing scope**: the README's own CLI output labels the throughput
  "compression e2e throughput" / "decompression e2e throughput" computed as
  `original_bytes / process_wall_time` — i.e. a coarse **host-side wall-clock
  around the whole CLI binary invocation**, not a device-side kernel-only
  measurement, unless the user separately profiles with `nsys` (the README
  explicitly recommends `nsys profile --stats=true` "to obtain more accurate
  timing for the compression kernel," implying the default number already
  includes host/launch overhead beyond the 3 GPU kernels).
- **precision & correctness**: byte/symbol-oriented LZSS is lossless by
  construction (dictionary-match, no numeric approximation); the paper
  reports compression ratio = raw_size/compressed_size as the round-trip
  fidelity proxy but does not print an explicit bit-exact diff check in the
  disclosed evaluation section (unlike MANS or the Huffman-decoder paper
  below, which both hard-gate on exact equality inside the harness itself).
- **metric**: compression ratio; compression throughput (GB/s) = original
  size / compress-or-decompress time. Decompression throughput is reported
  **only as a single average across all 6 datasets** (16.4 GB/s on A4000,
  29.1 GB/s on A100) — the paper itself states "we do not describe and
  compare it [decompression] with other compressors in detail," a real
  reporting-granularity gap versus its much more detailed compression-ratio
  and compression-throughput tables.
- **baselines**: CULZSS (open-source GPU LZSS, CPU-encodes/GPU-matches),
  nvCOMP's LZ4 2.6.0 (closed-source).
- **hardware**: 2 platforms — HPC node (dual 64-core AMD EPYC 7742, 4×
  NVIDIA A100 108SM/40GB, CentOS 7.4, CUDA 11.4.120) and workstation
  (24-core Intel Xeon W-2265, 2× NVIDIA A4000 40SM/16GB, Ubuntu 20.04.5,
  CUDA 11.7.99).
- **use-case validation**: GPULZ is inserted as a drop-in lossless
  post-stage ahead of cuSZ's own Huffman step (replacing/augmenting it) on 4
  SDRBench-family fields (CESM, Hurricane, Nyx, +rtm-adjacent) at relative
  error bound 1e-2, improving compression ratio 1.9×–8.7× at a throughput
  cost (Table 3) — methodologically the same "improve cuSZ's lossless
  back-end" framing as the IPDPS'22 Huffman-decoder paper below; the two are
  directly comparable competitors for the same architectural slot.
- **source**: `ICS23-GPULZ.pdf` (repo-hosted local copy, downloaded and
  text-extracted with `pypdf` since no `pdftotext`/`ar5iv` conversion was
  available) — Sections 4.1–4.5 read in full — + README.md.

## 3. MANS (Huang et al., SC'25) — repo README + raw source (hpdps-group/MANS)

- **workloads/inputs**: multi-byte integer data (u16/u32). `testdata/exafel`
  ships real EXAFEL light-source X-ray detector data (photon-science domain,
  same application class as lsCOMP below). The README's headline
  compression-ratio/throughput numbers are described as "real-world
  scientific datasets" and "quantization-based datasets" (SZ/cuSZ-style
  quant-code streams, same data class as GPULZ's `*-quant` inputs above),
  though the exact named list is not pinned beyond the shipped `exafel` set.
  Separately, `cpu_mans_autotune.cpp` generates **synthetic** u16 data
  (configurable size list, 4 tunable block-type ratios: smooth/spike/
  constant/random) purely to pick a per-input-size thread-count config for
  the CPU path — this synthetic data is NOT part of the reported
  ratio/throughput numbers, only a calibration input.
- **timing protocol**: `cpu_mans_bench.cpp` CLI exposes `--warmup` (default
  **5**) and `--runs`/`--iters` (default **10**) directly as documented
  flags (`Usage: ... [--warmup 5] [--runs 10]`). Critically, the benchmark
  loop reports the **best**-of-N throughput per direction
  (`best_comp_mbps = std::max(best_comp_mbps, comp_mbps)` across all
  post-warmup iterations), not a mean or median — an explicit best-of-N
  reporting practice that inflates the headline number relative to a typical
  run.
- **timing scope**: kernel-only, via an internal scoped timing collector
  (`mans::TimingCollector`, core-only phase timers:
  `adm_encode_core`/`entropy_encode_core`/`entropy_decode_core`/
  `adm_decode_core`) — no I/O or file-loading time folded in.
- **precision & correctness**: the bench harness itself **hard-gates on
  bit-exact equality**: `std::memcmp(recovered.data(), input.data(),
  expected_bytes)`; any mismatch sets `stats.ok=false`, prints
  "Decompression mismatch." and aborts the run. This is the strongest,
  most explicitly machine-enforced correctness gate found anywhere in this
  track's surveyed artifacts.
- **metric**: throughput MB/s (`original_bytes / time`, both directions
  separately), compression ratio = `total_bytes / avg_compressed_bytes`
  (averaged compressed size across the bench_iters runs, even though the
  *time* metric is best-of-N — a further asymmetry: ratio is averaged,
  throughput is maxed).
- **platforms**: 3-way cross-platform — CPU (OpenMP+SIMD, with a `-p`
  portable/GPU-consistent mode and an `-r` max-ratio FSE-ANS mode), NVIDIA
  GPU (CUDA), AMD GPU (HIP/ROCm). The README claims the `-p` CPU mode
  "maintains consistency with GPU behavior" but does not claim strict
  bit-identical CPU==GPU output (weaker than PFPL's explicit cross-platform
  bit-identity guarantee in the lossy-compression track).
- **baselines**: FSE-ANS (CPU reference ANS implementation), 16-bit Huffman,
  nvCOMP Huffman (GPU).
- **hardware**: Intel Xeon Gold 5220S (CPU numbers), NVIDIA A100 (GPU),
  AMD MI210 (GPU).
- **source**: `gh api repos/hpdps-group/MANS/contents/README.md` +
  `cpu/cpu_mans_bench.cpp` (raw source, ~330 lines read in full) +
  `testdata/` directory listing.

## 4. Huffman decoding for cuSZ (Rivera et al., IPDPS'22) — repo README + raw source (codyjrivera/ipdps22-opthuffdec)

- **workloads/inputs**: 5 named HPC datasets, each first passed through
  cuSZ's own quantizer (`cusz --skip huffman`) so the paper's actual input
  is cuSZ's **intermediate quantization-code stream** (same architectural
  role as GPULZ above): HACC `vx.f32` (280,953,867 elements), EXAALT
  `dataset2-2338x106711.x.f32.dat` (2,338×106,711), CESM "Large"
  `CLDICE_1_26_1800_3600.f32` (26×1800×3600), Nyx `baryon_density.dat`
  (512³), Hurricane ISABEL `HURR-CAT.bin.f32` (400×500×500, a concatenation
  of the CLOUD/PRECIP/QCLOUD/QRAIN fields into one file via `getdata.sh`).
  Error mode `r2r` (value-range-relative), the disclosed sweep in `run.sh`
  fixes eb=1e-3.
- **timing protocol**: `demo.cc` performs **1 untimed warmup decode call**,
  then wraps **`NROUNDS`=10 timed decode calls** inside a single
  `TIMER_START`/`TIMER_STOP` pair and reports the **mean** (`timeUs =
  i.second / NROUNDS`) — the most rigorous, explicitly-in-code
  warmup+repetition+averaging protocol found in the track (matched only by
  Rapidgzip's paper-level disclosure, but here it's directly in the
  benchmark source).
- **timing scope**: kernel-only. `GPU memcpy HtD` (H2D) and `GPU memcpy DtH`
  (D2H) are each wrapped in their **own separate** `TIMER_START`/`STOP`
  block, distinct from the `"decoding"` block — decode throughput is
  computed strictly from the decode-kernel-only interval, transfer excluded
  by construction. This is a clean, code-verified example of the
  kernel-only/end-to-end split this track's spec needs to enforce
  everywhere.
- **precision & correctness**: `cuhd::CUHDUtil::equals(buffer.data(),
  out_buf->get_decompressed_data().get(), size)` — an explicit **bit-for-bit
  equality check** printed as `"mismatch"` on failure. Compression ratio is
  also computed and printed:
  `size*sizeof(SYMBOL_TYPE) / (compressed_size*sizeof(UNIT_TYPE))`.
- **metric**: decode time (µs, mean over 10 rounds), throughput GB/s =
  `original_size / decode_time`, compression ratio.
- **comparison structure**: 4 decoders share one harness — `orig-self-sync`
  (Weissenberger & Schmidt, ICPP'18 / CUHD), `orig-gap-array` (Yamamoto et
  al., ICPP'20), `opt-self-sync` (their optimization of ICPP'18),
  `opt-gap-array` (their optimization of ICPP'20) — plus a 5th baseline,
  cuSZ's own built-in Huffman decoder, run via a separate `runcusz.sh`.
  Reported headline: 2.43× average decompression speedup for cuSZ overall
  (abstract).
- **hardware**: x86_64 host + single NVIDIA Tesla V100 32GB, GCC 8.3.1,
  NVCC 11.1.
- **source**: `gh api repos/codyjrivera/ipdps22-opthuffdec/contents/README.md`
  + `getdata.sh` (raw) + `opt-self-sync/run.sh` (raw) +
  `opt-self-sync/src/demo.cc` (raw, ~200 lines read in full).

## 5. hZCCL (Huang et al., SC'24) — repo README + raw scripts (ZCCLorg/zccl)

- **workloads/inputs**: standalone-compressor microbenchmark
  (`jobs/evaluate_compressor_NYX.sh`) uses the Nyx 512³ cosmology
  simulation, **6 named fp32 fields**: `baryon_density`,
  `dark_matter_density`, `temperature`, `velocity_x`, `velocity_y`,
  `velocity_z`, block size **36** (SZ-family-style block config). Separately,
  the MPI collective (allreduce) benchmark (`jobs/different_sizes.sh`) sweeps
  synthetic data sizes from 50 MiB upward across up to 64 nodes.
- **timing protocol**: the standalone compressor test driver
  (`testfloat_homocompress_fastmode1_arg.c`) hardcodes `#define ITERATIONS
  1` — a **single-shot** measurement via `gettimeofday`-based
  `cost_start()`/`cost_end()` macros, **no repetition, no warmup at all**
  — the single weakest-disclosed timing protocol found across this track
  (and the lossy-compression track surveyed previously). The MPI-collective
  benchmark is invoked with `-i 10 -w 0` (10 iterations, 0 warmup) —
  slightly better but still zero warmup.
- **error bound / correctness — a genuine track-fit problem**: hZCCL's
  compressor is an **error-bounded LOSSY homomorphic compressor**
  (`evaluate_compressor_NYX.sh` sweeps relative error bound {1E-1, 1E-2,
  1E-3, 1E-4}), not a bit-exact lossless codec. Its own
  `error_evaluation()` function computes PSNR, max absolute/relative
  error, and NRMSE against the error bound — **explicitly NOT a bit-exact
  gate**. hZCCL appears in this track's input list because its
  contribution ("operations directly on compressed data") is architecturally
  adjacent to compression, but its correctness contract contradicts the
  track name. See Divergences/Open questions below — it is **excluded from
  the core spec variants**.
- **metric**: compression/communication time, PSNR/NRMSE/max-error (not
  throughput-vs-ratio in the sense the rest of the track uses).
- **baselines**: traditional DOC (decompress-operate-compress) workflow vs.
  their homomorphic "hZ-dynamic" direct-operate-on-compressed-data mode.
- **source**: `gh api repos/ZCCLorg/zccl/contents/README.md` +
  `jobs/evaluate_compressor_NYX.sh` (raw) + `jobs/different_sizes.sh` (raw)
  + `examples/testfloat_homocompress_fastmode1_arg.c` (raw, error-evaluation
  function read in full).

## 6. ZipServ (Fan et al., ASPLOS'26) — repo README + raw scripts (xxyux/ZipServ)

- **workloads/inputs**: LLM weight tensors (bf16) from named model families
  with real (M, K) projection shapes swept across batch size N ∈ {8, 16,
  32}: Llama3-8B/70B/405B, Qwen3-32B/14B/8B, Qwen2.5-72B/32B/14B/7B,
  Gemma-3-27B/12B, Mistral3-24B/123B. Standard transformer layer shapes
  (Q/K/V/O_proj, FFN gate/up/down, Embedding) plus fused merged-QKV and
  merged-FFN-gate-up variants. A separate `run_all_main.sh` sweeps generic
  GEMM (M, K, N) triples against a `SPARSITY` parameter ∈ {40, 50, 60, 70}%
  — this is the TCA-TBE bitmap-encoding's high-frequency-bit density knob,
  **not** matrix zero-sparsity.
- **timing protocol**: `kernel_benchmark/test_mm.cu` and `test_decompress.cu`
  hardcode `#define WARM_UP_ITERATION 0` and `#define BENCHMARK_ITERATION 1`
  — a CUDA-event-timed **single-shot, zero-warmup** measurement for the
  microbenchmark binaries. Separately, `kernel_benchmark/run.sh` drives a
  per-layer-shape sweep through **Nsight Compute**
  (`ncu --metrics gpu__time_duration.sum --csv`) — i.e. a single kernel
  launch measured by the hardware performance-counter-based profiler rather
  than repeated wall-clock averaging, which is a legitimate (if differently
  documented) substitute for repetition, but the plain CUDA-event path
  (`test_mm`/`test_decompress` run directly, as in `run_all_main.sh`) has no
  such substitute and is genuinely single-shot/no-warmup.
- **timing scope**: kernel-only — the fused "ZipGEMM" kernel (decompress +
  Tensor-Core GEMM in one launch, the paper's core "load-compressed,
  compute-decompressed" design) is timed device-pointer-to-device-pointer;
  H2D/D2H is out of scope for the microbenchmarks (weights are already
  resident on-device via the quantization-config-driven loading path).
- **precision & correctness — a genuine gap**: `LInfer_py/
  verify_compression.py` verifies correctness **only at the GEMM-output
  level**: `cos_sim = cosine_similarity(ground_truth_C, custom_C)`, passing
  if `cos_sim > 0.999`, alongside printed MSE and relative error against a
  bf16 `cuBLAS` reference. **The decompressed WEIGHT tensor itself is never
  compared byte-for-byte against the original weight in the disclosed
  verification script** — only the downstream matmul's numerical output is
  checked, and that check already has an inherent bf16-Tensor-Core rounding
  tolerance baked in, so it cannot detect a subtly-wrong-but-numerically-
  close decompression bug. For a system that markets itself as **lossless**
  weight compression, this is a real verification gap in the public
  artifact — flagged explicitly in Divergences below.
- **metric**: kernel-level speedup vs. cuBLAS dense GEMM (2.21× claimed,
  abstract), end-to-end inference speedup vs. vLLM (1.22× avg, via
  `end2end_inference/` batch/input-len/output-len sweeps), weight size
  reduction (up to 30%, abstract).
- **baselines**: cuBLAS (dense GEMM kernel level), vLLM (dense end-to-end
  inference level).
- **hardware**: NVIDIA GPU, compute capability ≥ 8.0 (Ampere-class: A100,
  A6000, RTX 4090 all listed as supported in the README's requirements;
  no single fixed benchmark GPU is pinned in the README itself).
- **source**: `gh api repos/xxyux/ZipServ/contents/README.md` +
  `kernel_benchmark/{run.sh,run_all_main.sh,test_mm.cu,test_decompress.cu,
  utils.h}` (raw) + `LInfer_py/verify_compression.py` (raw, full
  verification function read).

## 7. lsCOMP (Huang et al., SC'25) — repo README (szcompressor/lsCOMP), lossless-mode facts

lsCOMP is also in the lossy-compression track input (same repo, same single
fused kernel supports both modes); this entry focuses on the **lossless**
configuration specifically, cross-checked against the fuller lossy-track
survey already on file at `benchspecs/lossy-compression/survey.md`.

- **workloads/inputs**: light-source/X-ray detector data, `uint16`/`uint32`
  (NOT floating-point) — e.g. `cssi.bin` 600×1813×1558 (X-ray coherent
  scattering imaging), same photon-science domain as MANS's `exafel`
  testdata above.
- **lossless configuration**: lsCOMP has no separate lossless codec — its
  "Adaptive Scalar Quantization" (`-b`, 4 quant-bin levels) and "Selective
  Pooling" (`-p`, pooling threshold) lossy knobs, when set to
  **`-b 1 1 1 1 -p 1`**, reproduce exact lossless compression through the
  **same single fused GPU kernel** as the lossy path — lossless is a
  degenerate parameter setting, architecturally identical to the lossy mode
  rather than a separate code path.
- **timing protocol**: README explicitly states 3 GPU warmup iterations
  ("Section 1: GPU Warmup — Performing GPU warmup runs for **3
  iterations**") before the timed run.
  Throughput = `original_bytes / kernel-only time`, cross-checked
  arithmetically against the worked example (6,779,169,600 B / 0.012865 s ≈
  527 GB/s vs. the printed 490.76 GB/s — the small gap is on-device
  verification overhead, not a PCIe round-trip).
- **precision & correctness**: for the lossless configuration, the worked
  README example prints **`Max abs diff: 0`** — the clearest, most explicit
  self-reported bit-exact confirmation string found anywhere in this track's
  artifacts (stronger than a bare compression-ratio proxy, though still a
  one-off printed example rather than a hard-coded assertion/gate in the
  benchmark harness itself, unlike MANS or the Huffman-decoder paper).
- **metric**: throughput GB/s (compress/decompress reported separately),
  compression ratio (16.17× lossless on the worked `cssi` example, 23.33× on
  a smaller Python-binding slice example).
- **baselines**: "up to 20× higher performance than industry-leading GPU
  compressors" (abstract; nvcomp/bitshuffle-class tools implied, not
  individually named).
- **hardware**: NVIDIA A100 (40 GB).
- **source**: `gh api repos/szcompressor/lsCOMP/contents/README.md` (already
  read in full for the lossy-compression track survey; re-used here,
  lossless-mode facts extracted).

---

## Divergences

1. **Track-inclusion tension: not everything in this track's input list is
   actually bit-exact.** hZCCL's core compressor is an **error-bounded
   lossy** homomorphic compressor with a PSNR/NRMSE/max-error correctness
   contract, not a bit-exact one — its role in the track is "compression
   accelerates a communication primitive," architecturally adjacent to
   lossless compression but not itself lossless. ZipServ *claims* lossless
   weight compression but its own public verification script never checks
   the decompressed weight tensor byte-for-byte — only a downstream
   GEMM-output cosine-similarity (>0.999, an inherently lossy-tolerant
   check given bf16 Tensor Core rounding). lsCOMP's lossless mode is a
   degenerate parameter setting of a fundamentally lossy-first codec. Only
   MANS, GPULZ, Rapidgzip, and the IPDPS'22 Huffman-decoder paper are
   unambiguous, purpose-built bit-exact byte/integer-stream compressors with
   hard bit-exact gates already present in their own harnesses.

2. **Data-domain split has no shared corpus, unlike the lossy-compression
   track's SDRBench convergence.** Generic byte-stream/text/genomic
   corpora (Rapidgzip: Silesia, FASTQ, base64-random) vs. multi-byte
   scientific/sensor **integer** data — mostly SZ/cuSZ **quantization-code
   streams**, not raw floating point (GPULZ's `*-quant` sets, the
   Huffman-decoder paper's cuSZ-preprocessed inputs) or detector data (MANS's
   `exafel`, lsCOMP's `cssi.bin`) — vs. LLM weight tensors as a GEMM operand
   (ZipServ) vs. raw fp32 simulation fields under an error-bound contract
   (hZCCL, excluded from core scope per point 1). This maps directly onto
   the 3-way variant split below (generic-byte-stream / scientific-multibyte
   / structured-GEMM-operand) instead of one shared input suite.

3. **Repetition/warmup discipline spans the full range from best-practice to
   none at all, and where a paper both discloses AND enforces it in code,
   the disclosed number often contradicts a "fair" statistic.** The
   Huffman-decoder paper (IPDPS'22) is the most rigorous: 1 untimed warmup +
   10 timed rounds, mean. Rapidgzip is the most statistically thorough at
   the paper-writing level: 20–100 repeats, mean ± 1 std-dev. MANS discloses
   concrete CLI defaults (`--warmup 5 --runs 10`) but its own benchmark
   binary reports **best**-of-10, not mean/median — an explicit
   optimistic-bias practice. GPULZ discloses no warmup/repetition/statistic
   anywhere in its Experimental Setup. ZipServ's plain CUDA-event
   microbenchmark path hardcodes **zero warmup, one measured iteration**
   (its `ncu`-profiled sweep substitutes a hardware-counter single-launch
   measurement instead, which is a different and more defensible kind of
   "no repetition"). hZCCL's standalone-compressor test driver hardcodes
   `ITERATIONS 1` with no warmup at all via a single `gettimeofday` span —
   the single weakest protocol found in this track (and the lossy-compression
   track surveyed previously).

4. **Timing scope divergence, worse than "end-to-end means two things"
   (the lossy-compression track's finding) — here it spans at least four
   distinct scopes.** (a) kernel-only-with-transfer-timed-separately
   (Huffman-decoder: H2D/D2H each in their own `TIMER_START` block distinct
   from "decoding"; lsCOMP: explicit disk/H2D/kernel/D2H/disk breakdown);
   (b) whole-process wall-clock including CLI/host overhead (GPULZ's default
   README-printed "e2e throughput," which the README itself says needs
   `nsys` to get a truer kernel-only number); (c) in-memory-only with I/O
   deliberately routed around rather than measured-and-subtracted
   (Rapidgzip: `/dev/shm` source, `/dev/null` sink); (d) full MPI-collective
   -inclusive time, where compression is one stage inside a distributed
   communication primitive's total latency (hZCCL's `compression_allreduce`
   benchmark) — a scope that isn't comparable to a standalone kernel call at
   all. The spec must name each variant's scope explicitly, never reuse a
   bare "end-to-end."

5. **Correctness-gate rigor ranges from machine-enforced-and-fatal to
   entirely absent.** `std::memcmp`-based hard abort (MANS) and
   `CUHDUtil::equals`-based hard "mismatch" print (Huffman-decoder) are the
   two genuine hard gates found. lsCOMP prints `Max abs diff: 0` for one
   worked example, not a gate wired into the benchmark loop itself. GPULZ
   relies on the implicit exactness of dictionary-based LZ compression
   without printing an explicit round-trip diff in the evaluation section.
   ZipServ has no bit-exact check anywhere in its disclosed verification
   path — only a >0.999 cosine-similarity check on downstream GEMM output.
   hZCCL has no bit-exact gate by design (its correctness contract is an
   error bound, not exactness). Given the track's name, the spec below
   imposes a **universal hard bit-exact roundtrip gate**, independent of
   (and in ZipServ's case, strictly stronger than) what each paper's own
   public artifact actually enforces.

6. **Throughput numerator convention is nonetheless consistent, matching the
   lossy-compression track's finding.** Every paper surveyed here that
   states its throughput formula explicitly (MANS's `comp_mbps`/
   `decomp_mbps` code, GPULZ's stated "ratio of original data size to
   compression/decompression time," the Huffman-decoder's `thruGbs` code,
   lsCOMP's worked-example cross-check) uses **original (uncompressed) byte
   count** as the numerator for BOTH compression and decompression
   throughput — never compressed-byte count, and never switched per
   direction. This is a genuine point of continuity with the
   lossy-compression track and is adopted as-is.

7. **"Compress AND decompress" as a joint metric breaks down for
   decompression-only tools.** Rapidgzip has no compression path at all —
   it decompresses standard-format files produced by other tools. For this
   sub-domain the spec reports compression ratio as a **fixed property of
   the input corpus** (produced once, by a reference encoder at a stated
   level) rather than a per-tool re-measured number, while throughput
   remains decompression-only and is never allowed to stand in silently for
   "the tool's overall compression performance."

## Open questions

- hZCCL is excluded from all 3 core spec variants because its correctness
  contract (error-bounded, PSNR/NRMSE-checked) contradicts "lossless"; it is
  architecturally closer to the lossy-compression track's collective-
  communication-adjacent fringe than to this track's core. Flagged here
  rather than force-fit, matching how the lossy-compression track's own
  survey scoped out lsCOMP's integer-quantization mode and the AMR-specific
  TAC/TAC+ family.
- No repo-disclosed number could be found for ZipServ's benchmark GPU model
  (the README lists 3 *supported* GPUs — A100/A6000/RTX 4090 — as minimum
  compute-capability requirements, not a single pinned benchmark machine);
  the abstract's "2.21× over cuBLAS" figure's hardware is unconfirmed from
  the sources read here.
- GPULZ's paper PDF (repo-hosted local copy) had to be fetched by direct
  download + `pypdf` text extraction because neither `WebFetch` on the
  arXiv abstract page nor `ar5iv.labs.arxiv.org/html/2304.07342` returned
  usable content (ar5iv reported a fatal LaTeX-to-HTML conversion error).
  This is noted as a reproducibility caveat for future survey passes on
  this same paper.
- ZipServ's `run_all_main.sh` invokes a `./spmm_test` binary whose source
  was not located in the browsed portion of the repo (only `test_mm.cu` and
  `test_decompress.cu` were found under `kernel_benchmark/`); its exact
  internal warmup/repetition constants are therefore unconfirmed and the
  `WARM_UP_ITERATION=0`/`BENCHMARK_ITERATION=1` values cited above are taken
  from the two source files that WERE located, not from `spmm_test` itself.
- MANS's exact named real-world dataset list beyond the shipped
  `testdata/exafel` sample (the README's "quantization-based datasets" and
  broader "real-world scientific datasets" claims) could not be pinned down
  further without the paywalled SC'25 paper PDF; the `recommended_subset`
  below uses the shipped sample plus the SZ/cuSZ-quant-code convention
  already confirmed via GPULZ/the Huffman-decoder paper as a reasonable
  stand-in.
- No independent confirmation of bit-exactness could be obtained for GPULZ
  beyond its algorithmic argument (LZSS is dictionary-based and inherently
  lossless) — no printed round-trip diff was found in the Experimental
  Setup or Evaluation sections read.
