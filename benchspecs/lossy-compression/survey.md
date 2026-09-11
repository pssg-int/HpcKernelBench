# Survey: lossy-compression track evaluation methodology

Track input: `data/track_inputs/lossy-compression.json` (20 papers, SZ/ZFP-family
error-bounded compressors for scientific floating-point data). 9 papers surveyed
below (5 via arXiv/ar5iv fulltext, 4 via artifact repo README + benchmark
scripts). Two more sources (VGC/SC25, sdrbench.github.io dataset catalog) are
cited for context but ACM DL blocked fulltext access (paywalled, abstract-only).

---

## 1. FZ-GPU (Zhang et al., HPDC'23) — arXiv 2304.12557, fulltext via ar5iv

- **workloads/inputs**: 6 SDRBench fields — HACC (cosmology, 1D, 1,123.81 MB,
  280,953,867 particles, fields `xx`,`vx`, log-transformed before compression
  because of huge dynamic range), CESM-ATM (climate, 2D 1800×3600, 25.92 MB,
  fields `CLDICE`,`RELHUM`), NYX (cosmology, 3D 512³, 536.87 MB, field
  `baryon_density`), Hurricane ISABEL (climate, 3D 100×500×500, 100 MB, fields
  `CLDICE`,`QRAIN`), QMCPACK (quantum, 3D 7935×69×288, 630.74 MB, field
  `einspline`), RTM (petroleum/seismic, 3D 449×449×235, 189.50 MB, field
  `snapshot_1200`). All single-precision (fp32) only.
- **error bound**: relative error bound (fraction of value range) swept at
  `{1e-2, 5e-3, 1e-3, 5e-4, 1e-4}`. cuZFP has no error-bounded mode, so it is
  instead tuned to match the PSNR achieved by the error-bounded runs — an
  explicit methodological workaround the spec must account for.
- **timing protocol**: "kernel time" reported for compression throughput;
  no explicit warmup/repetition count or statistic disclosed in the text.
  Separately, PCIe host<->device transfer is measured (11.4 GB/s effective per
  GPU) and folded into a distinct "overall throughput" number — i.e. the paper
  itself reports two throughput numbers, kernel-only and end-to-end, as
  different metrics for different claims.
- **timing scope**: kernel-only throughput is the headline metric; end-to-end
  (incl. H2D/D2H) is reported separately, not silently mixed in.
- **precision & correctness**: fp32 only. Compressed output must respect the
  error-bounded mode's contract; no numeric tolerance discussed beyond the
  bound itself (SZ-family convention: the bound is a hard pointwise
  guarantee, not a statistical one).
- **metric**: compression throughput (GB/s), compression ratio, bitrate
  (32 bits / CR), PSNR + SSIM for rate-distortion, plus the separate overall
  (H2D/D2H-inclusive) throughput number.
- **baselines**: cuZFP (fixed-rate only), cuSZ, cuSZx, MGARD-GPU, and their
  own multi-threaded CPU implementation (FZ-OMP, dual 28-core Xeon Gold
  6238R). GPUs: 2×A100 (Ampere, 108 SMs, 40GB, CUDA 11.4.120) and 2×RTX A4000
  (40 SMs, 16GB, CUDA 11.7.99).
- **source**: arXiv fulltext (ar5iv.labs.arxiv.org/html/2304.12557).

## 2. TAC+ (Wang et al., TPDS'24) — arXiv 2301.01901, fulltext

- **workloads/inputs**: 10 AMR (Adaptive Mesh Refinement) datasets from 3
  simulation codes, NOT regular-grid SDRBench data: Nyx (6 sets: Run1_Z10,
  Run1_Z5, Run1_Z2 — 2-level, finest 512³, 3.3–6.7 GB; Run2_T3/T4, Run3_Z1 —
  3-4 level, finest 512³–1024³, 169 MB–416 MB), WarpX (2 sets, 2-level,
  128²×1024/256²×2048, 1.4–2 GB), IAMR (2 sets, 3-level, finest 512³,
  336 MB–2 GB). Each dataset is a multi-resolution AMR hierarchy, not a
  single flat array.
- **error bound**: not uniform — varies per figure/experiment; relative error
  bound examples of `6.7e-3` and `4.8e-4` appear for power-spectrum analysis.
- **timing protocol**: reports "OpST" (their pre-process strategy) and
  "AKDTree" time cost separately from the underlying SZ compression time —
  i.e. preprocessing (AMR-level unification/reordering) is REPORTED
  separately from compression, never silently merged.
- **precision**: fp32/fp64 both supported (32/64 bits/value before
  compression, stated explicitly).
- **metric**: compression ratio, bitrate, PSNR, rate-distortion curves, plus
  application-specific downstream metrics (cosmology power spectrum error,
  halo-finder mass/cell differences) — i.e. the paper validates that the
  compression error doesn't corrupt a downstream scientific analysis, not
  just a pointwise numeric bound.
- **baselines**: 3 internal baselines only — 1D-naive (each AMR level
  compressed independently as 1D), 1D-zMesh, and 3D (levels resampled to
  uniform resolution then compressed as one 3D volume). Does not compare
  directly against ZFP or MGARD in the evaluation.
- **hardware**: 2×28-core Intel Xeon Gold 6238R, 384 GB RAM (CPU only).
- **source**: arXiv fulltext (ar5iv.labs.arxiv.org/html/2301.01901).

## 3. TAC (Wang et al., HPDC'22) — arXiv 2204.00711, fulltext

- **workloads/inputs**: 7 AMR datasets, all from the Nyx cosmology code (64
  Mpc region), 2 real simulation runs: Run1_Z10/Z5/Z3/Z2 (2-level,
  512/256 grids, density 23–77%), Run2_T2/T3/T4 (3–4 level, finest
  256–1024, density down to 3e-5%). Predecessor dataset to TAC+'s 10-set
  suite (same Nyx family, fewer levels/apps).
- **error bound**: absolute error bound swept at `{1E+08, 1E+09, 1E+10}`
  (density field has huge dynamic range, hence large absolute bounds);
  relative bound examples `6.7e-3`/`4.8e-4` for power-spectrum checks.
- **timing protocol**: throughput = original data size / TOTAL time, where
  total time explicitly INCLUDES "pre-processing, compression, and
  decompression" — i.e. TAC (unlike TAC+) reports one merged end-to-end
  number rather than separating preprocessing out. This is a direct
  methodological divergence from TAC+ (its own successor paper).
- **precision**: fp32/fp64 (32/64 bits/value, stated explicitly).
- **metric**: compression ratio, bitrate, PSNR, throughput (MB/s), power
  spectrum error, halo-finder differences.
- **baselines**: same 3 internal baselines as TAC+ (1D-naive, 1D-zMesh,
  3D-unified). ZFP/MGARD mentioned only as background, not benchmarked.
- **hardware**: identical to TAC+ — 2×28-core Xeon Gold 6238R, 384 GB RAM.
- **source**: arXiv fulltext (ar5iv.labs.arxiv.org/html/2204.00711).

## 4. PFPL (Fallin et al., IPDPS'25) — repo README (burtscher/PFPL)

- **workloads/inputs**: SDRBench ("or IEEE 754 floating-point input of your
  choice"); no fixed subset pinned in the README — evaluated on whatever
  SDRBench fields the paper's Table uses (paper PDF not machine-readable via
  WebFetch; abstract confirms "SDRBench inputs").
- **error bound**: 3 modes supported — absolute (ABS), relative (REL), and a
  novel "normalized-absolute" (NOA) bound. Abstract example: absolute error
  bound of `1E-3` yields 5 GB/s (CPU) / 423 GB/s (GPU) single-precision
  throughput.
- **timing protocol**: README states explicitly — **"the codes will be run 9
  times and the runtimes reported in seconds"** to match the paper. This is
  the single most concrete repetition count found across the surveyed track.
  No warmup mentioned; statistic (mean/median/min of the 9) not stated.
- **timing scope**: compress/decompress binaries operate on a file already on
  disk; timing wraps the compress/decompress call, not process startup.
- **precision & correctness**: fp32 and fp64, IEEE 754 binary. Compressor is
  "guaranteed error bound" and produces "bit-for-bit identical" output on CPU
  vs GPU — an explicit cross-platform correctness contract, stronger than
  the norm (most SZ-family compressors don't guarantee CPU==GPU bit-identity).
- **metric**: throughput (GB/s), compression ratio. Explicit claim: "at least
  4.6 times higher throughput than 7 leading compressors" — i.e. the paper's
  own baseline set has 7 members (not individually named in the abstract).
- **hardware**: 16-core AMD Threadripper 2950X CPU @3.5GHz (hyperthreaded,
  32 threads) for CPU numbers; NVIDIA RTX 4090 for GPU numbers.
- **source**: `gh api repos/burtscher/PFPL/contents/README.md`.

## 5. cuSZp / cuSZp2 / cuSZp3 (Huang et al., SC'23/SC'24/SC'25) — repo README (szcompressor/cuSZp)

- **workloads/inputs**: generic fp32/fp64 arrays, 1D/2D/3D processing modes;
  README examples use named HPC fields (`pressure_3000`, `velocity_x`,
  `xx.f32`) consistent with SDRBench naming. SC'23 abstract states 6
  representative scientific datasets on A100.
- **error bound**: `abs` and `rel` both first-class CLI flags
  (`-eb abs 1E-4`, `-eb rel 1e-3`), examples span `1e-2` to `1e-4`.
- **timing protocol**: the repo's own C API pattern is the clearest artifact
  found in this survey for a device-side kernel timer:
  ```c
  timer_GPU.StartCounter();
  cuSZp_compress(...);
  float cmpTime = timer_GPU.GetCounter();
  ```
  i.e. a GPU wall-clock (`TimingGPU`, CUDA-event-based) wraps ONLY the
  compress/decompress call — H2D/D2H is explicitly outside this timed region
  in the C API example (data pointers are already device pointers, `d_*`).
- **timing scope**: the headline numbers ARE explicitly labeled
  "**end-to-end** speed" in every README example (e.g. "cuSZp compression
  end-to-end speed: 410 GB/s") — but "end-to-end" here means the single
  fused CUDA kernel's wall time (quantization+encoding all fused into one
  kernel launch), NOT host<->device transfer. This is a source-of-confusion
  the spec must disambiguate: "end-to-end" in the SZ-GPU-compressor
  literature usually means "one kernel, no separate passes," not "incl. PCIe."
- **precision**: fp32 and fp64, selectable per run.
- **metric**: compression/decompression throughput (GB/s, computed against
  *original* uncompressed byte count — confirmed by cross-checking lsCOMP's
  worked example below), compression ratio. 3 encoding modes trade ratio vs.
  speed (fixed / plain / outlier).
- **baselines**: predecessor versions of itself across the 3 papers (cuSZp1
  SC'23 kernel-fusion baseline, cuSZp2 SC'24 new encoding modes, cuSZp3/VGC
  SC'25 dimensionality+versatility) plus a sibling ICS'25 paper
  (hierarchical-delta AaTrox mode) from the same group.
- **hardware**: NVIDIA A100 (~300-550 GB/s depending on mode/dim).
- **source**: `gh api repos/szcompressor/cuSZp/contents/README.md`.

## 6. QoZ (Liu et al., SC'22) — repo README (robertu94/QoZ) + abstract

- **workloads/inputs**: "comprehensive evaluation using 6 real-world
  scientific applications" (not individually named in the abstract/README;
  full paper is paywalled on computer.org, blocked from WebFetch). QoZ is
  explicitly built as an extension of SZ3 (same input/config format).
- **error bound**: inherits SZ3's abs/rel/PSNR/PW-relative error-bound modes
  (QoZ CLI is drop-in compatible with `sz3`; `-q 0` disables QoZ's
  quality-tuning to fall back to plain SZ3 for apples-to-apples comparison —
  a useful built-in ablation switch).
- **timing protocol / hardware**: not disclosed in README; not independently
  verifiable without the paywalled PDF.
- **precision**: fp32/fp64 (SZ3-family convention). 2D/3D only for QoZ's own
  optimizations — other dimensionalities silently fall back to plain SZ3.
- **metric**: compression ratio at fixed error bound (rate-distortion
  framed as "dynamic quality metric" — PSNR, SSIM, or autocorrelation can
  each be the optimization target depending on config), claims "up to 270%
  ratio gain vs SOTA" (abstract).
- **baselines**: "6 other state-of-the-art error-bounded lossy compressors"
  (abstract; not named — almost certainly includes SZ2/SZ3, ZFP, MGARD, FPZIP
  based on standard practice in this group's other papers, but not
  independently confirmed here).
- **source**: `gh api repos/robertu94/QoZ/contents/README.md` + abstract.

## 7. SPERR (Li et al., IPDPS'23) — repo README + scripts (NCAR/SPERR)

- **workloads/inputs**: 2D/3D structured scientific floating-point data;
  README explicitly recommends SDRBench "for testing and evaluation
  purposes." Repo ships its own `test_data/` (e.g. `wmag128.float`,
  `wmag256.float`, `wmag512.float` — 128³/256³/512³ synthetic cubes) used by
  its own evaluation script.
- **error bound / quality control**: THREE alternative targets, not just one
  error-bound axis — (1) fixed bit-per-pixel (BPP), (2) target PSNR, (3)
  point-wise error (PWE) tolerance. This is a genuine three-way divergence
  from the SZ-family's abs/rel-only convention.
- **timing protocol**: `evaluations/evaluate_compress.sh` sweeps a FIXED
  bpp list `(0.125 0.25 0.5 1 2 4)` for compression, calling
  `compressor_3d <file> <dims> XYZ tmp <bpp>` once per point and appending to
  a `.result` file — i.e. the canonical evaluation protocol for SPERR is a
  **rate-distortion sweep at fixed bitrates**, not a fixed-error-bound sweep.
  `evaluations/compare_zfp_sz/` further confirms the output artifact is a
  `PSNR-BPP.data` table fed into a gnuplot rate-distortion curve
  (`PSNR_to_BPP_plot.gp`) — direct evidence of the field's standard
  rate-distortion reporting convention.
- **precision**: fp32 (`.float` test files); double supported via API.
- **metric**: PSNR at fixed BPP (rate-distortion curve is the primary
  deliverable, not a single throughput number); abstract's headline claim is
  "best rate-distortion trade-off among current popular lossy scientific
  data compressors" — a comparative claim ACROSS the whole RD curve, not one
  operating point.
- **baselines**: implied by `compare_zfp_sz` directory name — ZFP and SZ
  compared on the same PSNR-vs-BPP axes.
- **source**: `gh api repos/NCAR/SPERR/contents/README.md` +
  `evaluations/evaluate_compress.sh` (raw content read) +
  `evaluations/compare_zfp_sz/` directory listing.

## 8. FAZ (Liu et al., ICS'23) — repo README (JLiu-1/FAZ) + abstract

- **workloads/inputs**: "comprehensive evaluation using 6 real-world
  scientific applications" (abstract; same phrasing/scale as QoZ — same
  author, same group, same evaluation convention reused across their
  papers, strong circumstantial evidence both draw from the same 6-app
  suite even though neither README names it explicitly).
- **error bound**: FAZ CLI is declared "identical to the sz3 command" —
  i.e. same abs/rel/PSNR/PW-rel modes as SZ3/QoZ. `-F 0` deactivates FAZ's
  auto-tuning to compare against plain SZ3, mirroring QoZ's `-q 0` ablation
  switch — confirms this Liu/Di/Cappello-group lineage shares one evaluation
  harness convention (auto-tuned mode vs. baseline-mode-of-same-binary).
- **timing / hardware**: not disclosed in README; full ICS PDF not
  independently fetched (paywalled).
- **metric**: compression ratio at fixed error bound vs. "6 other
  state-of-the-art" compressors (abstract).
- **approach tag**: `autotuning` (per track_inputs.json) — FAZ's contribution
  is the modular framework that searches over predictor/encoder
  combinations per-block, not a single fixed algorithm; timing therefore has
  a THIRD scope beyond kernel-only/end-to-end: the one-shot autotuning
  search cost, which the spec must treat like a preprocessing cost
  (reported separately, not amortized silently into steady-state numbers).
- **source**: `gh api repos/JLiu-1/FAZ/contents/README.md` + abstract.

## 9. lsCOMP (Huang et al., SC'25) — repo README (szcompressor/lsCOMP)

- **workloads/inputs**: light-source/X-ray detector data (`uint32`/`uint16`,
  NOT floating-point — a genuine divergence from the rest of the track,
  since this is integer sensor data with different statistics), e.g.
  `cssi.bin` 600×1813×1558 (X-ray coherent scattering imaging). Also targets
  "Open SciVis Datasets" as a generic integer-compression benchmark beyond
  light sources.
- **error bound**: NOT an abs/rel numeric bound — instead 2 lossy knobs:
  "Adaptive Scalar Quantization" (`-b` gives 4 quant-bin levels, monotonic
  x≤y≤z≤w) and "Selective Pooling" (`-p` pooling threshold on a data block).
  Setting `-b 1 1 1 1 -p 1` reproduces LOSSLESS mode exactly — i.e. lsCOMP's
  lossy and lossless paths share one kernel, lossless is just a lossy-mode
  parameter setting. This is architecturally different from the abs/rel
  bound convention used everywhere else in the track.
- **timing protocol**: printed breakdown in the worked example is the most
  granular I/O-vs-kernel split found in this survey — explicit separate
  timers for: disk read, CPU→GPU transfer, GPU compression kernel, GPU→CPU
  transfer (compressed, optional/verification-only), GPU decompression
  kernel, GPU→CPU transfer (decompressed, optional), disk write. README
  explicitly states "Section 1: GPU Warmup — Performing GPU warmup runs for
  **3 iterations**" before the timed run — confirmed warmup count.
- **timing scope**: the reported "compression end-to-end speed" (490.76
  GB/s) is computed over ORIGINAL byte count / measured kernel-only time
  (verified: 6,779,169,600 B / 0.012865 s ≈ 527 GB/s vs. the printed 490.76
  GB/s — close enough that the discrepancy is the small GPU-side
  verification overhead bundled in, not a full PCIe round-trip). Confirms
  the community throughput convention: `GB/s = original_bytes / time`, not
  compressed_bytes / time.
- **precision**: unsigned integer 32/16-bit (not fp32/fp64).
- **metric**: throughput GB/s (compress and decompress reported separately),
  compression ratio (e.g. 16.17× lossless on the worked cssi example; a
  second Python-binding worked example on a smaller slice reports 23.33×).
  Correctness check: `Max abs diff: 0` printed for the lossy-disabled
  configuration.
- **baselines**: "up to 20× higher performance than industry-leading GPU
  compressors" (abstract; nvcomp/bitshuffle-class tools implied, not
  individually named in README).
- **hardware**: NVIDIA A100 (40 GB).
- **source**: `gh api repos/szcompressor/lsCOMP/contents/README.md`.

---

## Context: SDRBench canonical dataset catalog

`WebFetch https://sdrbench.github.io` — 15 standard datasets are hosted:
CESM-ATM (climate, 1.47–17 GB), EXAALT (MD, 60 MB–2.4 GB), Hurricane ISABEL
(weather, 1.25 GB), EXAFEL (X-ray imaging, 51 MB–8.5 GB), HACC (cosmology,
5–19 GB), NYX (cosmology, 2.7 GB), NWChem (molecular integrals, 16 GB),
SCALE-LETKF (climate DA, 4.9 GB), QMCPACK (quantum MC, 1 GB), Miranda
(hydrodynamics, 1.87–106 GB), S3D (combustion, 44 GB), XGC (fusion,
unstructured, 1.2 GB), NSTX GPI (fusion imaging, 4.1 GB), plus 2 synthetic
"Brown" regularity-test sets. This is the field's de-facto standard suite —
7 of the 9 surveyed papers use a named subset of it directly (FZ-GPU by
name; PFPL, cuSZp, SPERR by explicit README pointer; QoZ/FAZ/TAC-family use
Nyx/S3D-class simulation outputs consistent with the same catalog even where
not literally downloaded from the SDRBench portal).

---

## Divergences

1. **"End-to-end" means two different things across papers.** In the
   TAC/TAC+/PFPL/FZ-GPU convention, "end-to-end" (or "total time") includes
   preprocessing + compression + decompression, sometimes + host<->device
   transfer. In the cuSZp/cuSZp2/cuSZp3 convention, "end-to-end" means
   "single fused CUDA kernel, no multi-pass pipeline" — it explicitly
   EXCLUDES PCIe transfer (their C API times only the device-pointer-to-
   device-pointer kernel call). The spec's variant names must not reuse the
   bare word "end-to-end" without also specifying whether H2D/D2H is inside
   or outside the timed region.

2. **Error-bound-target vs. rate-target compressors.** SZ-family
   (SZ3/QoZ/FAZ/cuSZp/FZ-GPU/PFPL/TAC) all take the error bound as the input
   knob and report the resulting compression ratio. SPERR (wavelet/SPECK)
   and ZFP natively take the bit-rate as the input knob and report the
   resulting PSNR. lsCOMP takes neither — it takes quantization-bin/pooling
   parameters with no closed-form error-bound guarantee. A fair track spec
   needs a rate-distortion (PSNR-vs-bpp) reporting mode that both families
   can be plotted on, in addition to a fixed-error-bound mode that only the
   SZ-family and cuSZp/FZ-GPU support natively.

3. **Preprocessing amortization differs even within the SAME lineage.** TAC
   (HPDC'22) merges preprocessing (AMR-level unification) into the reported
   throughput; its own successor TAC+ (TPDS'24) explicitly separates
   OpST/AKDTree preprocessing time from compression time. FAZ's autotuning
   search is a third kind of one-shot cost (parameter search over
   predictor/encoder combos) not comparable to either.

4. **Data type divergence.** 8 of 9 surveyed papers operate on IEEE 754
   floating-point (fp32 mostly, fp32/fp64 mixed in PFPL/TAC/TAC+/cuSZp).
   lsCOMP operates on unsigned integers (`uint16`/`uint32`) with a
   fundamentally different lossy-parameter surface (quant bins + pooling,
   not abs/rel error bound). It cannot share the same error-bound axis as
   the rest of the track; it is scoped out of the core variants below and
   flagged as an open question.

5. **Repetition/warmup counts are rarely disclosed, and where disclosed they
   disagree.** PFPL: 9 runs, no warmup stated. lsCOMP: 3 GPU warmup
   iterations, then apparently 1 timed run in the worked example (unclear if
   the paper itself uses more). FZ-GPU, TAC, TAC+, QoZ, FAZ, SPERR: no
   explicit count in any source read. The spec fixes a single protocol
   (warmup=3, reps=9 taking the median) as the fair default, borrowing
   PFPL's disclosed repetition count and lsCOMP's disclosed warmup count
   since neither paper's own choice is unreasonable and both are the most
   concrete numbers found in the track.

6. **Correctness contract is a hard pointwise guarantee, not a tolerance.**
   Unlike (e.g.) SpMM's "< 1e-4 relative error" statistical tolerance
   vs. a reference, EVERY error-bounded compressor in this track promises
   that 100% of reconstructed values satisfy the bound — this is the
   product's core claim, not an approximation. The correctness gate in the
   spec is therefore "zero pointwise violations," not an average-error
   threshold. PFPL additionally promises CPU==GPU bit-identical output,
   which is a strictly stronger (optional, PFPL-specific) guarantee we do
   not require track-wide.

## Open questions

- No full text was retrievable for QoZ (SC'22, computer.org paywall) or FAZ
  (ICS'23, ACM DL paywall) or VGC (SC'25, ACM DL paywall, redirect blocked
  by WebFetch) or lsCOMP's own paper text (ACM DL) — all 4 rely on
  README + abstract only, so their exact named dataset list, repetition
  count, and hardware are unconfirmed (marked accordingly above).
- The AMR-specific variant (TAC/TAC+ family) uses fundamentally different
  input shapes (multi-level hierarchical grids, not flat regular arrays)
  than the rest of SDRBench. We do not currently define a dedicated AMR
  benchmark variant in spec.yaml (out of scope for a first cut — no public
  AMR benchmark corpus as standardized as SDRBench was found); it is left as
  a follow-up track extension.
- lsCOMP's integer-quantization lossy mode has no abs/rel error-bound
  equivalent; it is excluded from the core spec's error-bound sweep and
  flagged rather than force-fit into the SZ-family convention.
- Exact per-paper repetition/warmup counts beyond PFPL (9 runs) and lsCOMP
  (3 warmup iterations) could not be confirmed from available sources; the
  spec's fixed protocol is therefore a reasonable synthesis rather than a
  literature-wide consensus value.
