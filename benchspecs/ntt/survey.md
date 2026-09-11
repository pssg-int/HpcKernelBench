# NTT track survey

SINGLETON track: 1 paper (`conf/asplos/00150A26`, Cheddar). Sources used:
arXiv fulltext (`arxiv.org/pdf/2407.13055`, the "arXiv version" the paper's
own README points to for the ASPLOS'26 camera-ready — downloaded and
text-extracted locally since neither `pdftoppm` nor the WebFetch markdown
converter reliably surfaced body-text/table content from the PDF stream),
plus the artifact repo `scale-snu/cheddar-fhe` (README, `unittest/Testbed.h`,
`unittest/BasicTest.cpp`, `parameters/bootparam_40.json`).

## Cheddar: A Swift Fully Homomorphic Encryption Library Designed for GPU Architectures (ASPLOS 2026)

Cheddar is a GPU CKKS-FHE library. NTT/INTT is not the paper's end-user-facing
deliverable (CKKS ciphertext ops are), but the paper explicitly identifies
(I)NTT (together with BConv, basis conversion) as "the most compute-intensive
component of CKKS, encompassing nearly all NTT computation" and devotes a
dedicated optimization section (Sec. 4.2: SMR-based lazy reduction, extended
on-the-fly twiddle-factor generation "EOT") to it. No track paper (there is
only this one) ever reports a standalone, isolated NTT/INTT number in
absolute time — it is always measured either (a) as a labeled slice of a
stacked execution-time-breakdown bar chart inside a larger fused kernel, or
(b) implicitly, inside whole-ciphertext-op or whole-FHE-workload timings.
This is the central methodological fact this spec has to work around (see
Divergence 1 below).

- **Ring dimension / polynomial degree**: fixed at N = 2^16 (`log_degree=16`
  in every shipped parameter file, confirmed in
  `parameters/bootparam_40.json`) across the entire paper — no other N is
  evaluated.
- **Modulus / RNS system**: Cheddar's headline contribution is 32-bit
  execution: primes q_i restricted to [0, 2^31) (31-bit primes packed in a
  32-bit word, "SMR" = Signed Montgomery Reduction), vs. the 62–64-bit primes
  used by the compared open-source libraries' standard 64-bit RNS. Default
  parameter set: N=2^16, PQ < 2^1776, dnum=4 (confirmed:
  `bootparam_40.json` has 43 main + 4 terminal + 12 auxiliary primes ≈ 48
  limbs total at α=12, matching Table 7's "Cheddar 48 12" row). A reduced
  48→24-limb point is also reported.
- **Batch / limb count**: NTT is applied per-limb (independently, for each
  RNS prime) and per-digit during key-switching (α limbs per digit, dnum
  digits) — the natural "batch count" axis for an NTT microbenchmark is
  exactly this (L limbs) × (α-sized digit groups during HRot/key-switch).
  Table 7 sweeps L ∈ {48, 24} at α ∈ {12, 6} for Cheddar and the same L at
  α=6 for all three baseline libraries (baselines' limb count is *halved*
  from their native 64-bit-word count "for fair comparison" against
  Cheddar's 32-bit words — i.e., matched *bit-width of data processed*, not
  matched *limb count*).
- **Timing protocol**: `unittest/BasicTest.cpp` uses a fixed
  `warm_up = 5` and a `std::chrono::high_resolution_clock` + explicit
  `cudaDeviceSynchronize()` before/after (macro `__ProfileStart`/`__ProfileEnd`
  in `unittest/Testbed.h`). Reading the macro closely: the timed region syncs
  once after the warm-up loop and then executes+times exactly one more
  invocation of the region of interest — i.e. **one measured execution per
  test-binary run**, not an internally-repeated-and-averaged loop. Table 7 in
  the paper reports "**median** execution times," which is stronger than
  what any other paper we have seen in this project reports, but the
  external harness that repeats the whole test binary N times to get that
  median (what N is, whether the binary is re-launched or the C++ test
  re-invoked) is not in the reachable artifact files read this pass — open
  question.
- **Correctness / precision**: this is CKKS (*approximate* FHE) — the
  artifact's own gate is a **message-level** tolerance,
  `static constexpr double max_error_ = 1e-3;` (`Testbed.h`), applied after a
  full Encode→Encrypt→Op→Decrypt→Decode round trip via `CompareMessages`,
  comparing complex-valued plaintext slots. This tolerance covers the
  cumulative noise of the *entire* CKKS pipeline (encryption noise +
  rescaling + the operation itself), not the NTT/INTT transform pair in
  isolation. The paper never states an NTT-only round-trip tolerance (e.g.
  INTT(NTT(x)) == x mod q_i). Because NTT/INTT over Z_{q_i}[x]/(x^N+1) is an
  *exact*, invertible, integer-modular operation (not a floating-point
  approximation), the natural correctness gate for an isolated NTT benchmark
  is bit-exact equality mod q_i, not an epsilon tolerance — this spec adopts
  that instead of borrowing the message-level 1e-3 (see Divergence 2).
- **Metric / hardware**: μs (Table 7, elementary ops: HMult/HRot/HAdd/Rescale,
  each of which internally issues one or more NTT/INTT calls), ms (bts,
  ResNet-20 CIFAR-10 inference), ms/it (HELR training). GPUs: V100, A100
  40GB, A100 80GB, H100, RTX 4090, RTX 5090 (own hardware), MI100 (baseline
  GME's custom-silicon comparison point only). Peak int32 TOPS and DRAM
  bandwidth per GPU are given in Table 4 and used to explain scaling.
- **Baselines**: at the elementary-op / (I)NTT-adjacent granularity —
  Liberate.FHE, HEonGPU, Phantom (Table 7, all open-source GPU CKKS
  libraries using standard 64-bit RNS NTT); at the full-FHE-workload level —
  100×, TensorFHE, HEaaN-GPU, WarpDrive, GME, plus FPGA implementations FAB,
  Poseidon, EFFACT (Table 5). Speedups range 2.18×–19.6× depending on
  workload/baseline pair.
- **What Cheddar's own (I)NTT-specific ablation actually measures**: Fig. 3b
  is an execution-time breakdown (μs) of HRot (N=2^16, α=12, 48 limbs, RTX
  4090) into NTT / BConv / Automorphism / Elementwise / Memcpy components
  across four optimization stages (Base, +LR, +OT, +EOT); EOT is reported to
  give "a 1.16× speedup of NTT for HRot" specifically (vs. 1.10× for the
  prior on-the-fly, non-extended, twiddle scheme). This is the *only* place
  in the reachable text where an NTT-specific (not NTT+BConv+... combined)
  performance number appears, and even there it is a component slice inside
  a fused, multi-op kernel launch — not a standalone NTT() call.

Source: arXiv fulltext (arxiv.org/pdf/2407.13055, local text extraction);
`scale-snu/cheddar-fhe` README, `unittest/Testbed.h`, `unittest/BasicTest.cpp`,
`parameters/bootparam_40.json` (via `gh api`).

## Divergences / gaps this spec has to resolve

1. **No paper in this track ever times bare NTT/INTT alone.** Every number
   is either an elementary-CKKS-op time (HMult/HRot/HAdd/Rescale — each a
   fused kernel sequence of INTT→BConv→NTT plus elementwise math, per the
   paper's own Sec. 2's "common computational routine of
   INTT→BConv→NTT") or a whole-FHE-workload time (bts/HELR/ResNet/Sort).
   This spec therefore defines an `ntt-kernel-isolated` variant that the
   *track's own artifact does not currently expose as a standalone
   benchmark* — running it requires disabling kernel fusion (or writing a
   microbenchmark harness around Cheddar's or a comparable library's raw
   NTT device function) — and flags this explicitly rather than presenting
   borrowed fused-kernel numbers as if they were pure-NTT numbers.
2. **Correctness tolerance mismatch.** The artifact's own correctness gate
   (`max_error_ = 1e-3` on decrypted message values) is a CKKS-approximation
   tolerance, appropriate for judging whether an *optimization* broke
   accuracy, but not appropriate as an NTT-track correctness gate, since a
   buggy-but-"close enough" NTT could pass it by accident (the FHE noise
   budget can mask a small NTT error). This spec requires exact modular
   equality against a reference NTT for the isolated-kernel variant instead.
3. **"Batch count" has two different, non-interchangeable meanings** in this
   paper: (a) number of RNS limbs L processed per ciphertext (a data-layout
   axis, since Cheddar fuses per-limb NTTs into one kernel launch), and (b)
   number of ciphertexts / independent polynomials processed together (a
   true batching axis in the classical sense). The paper's own comparisons
   only vary (a); it never reports a multi-ciphertext-batch sweep. This spec
   keeps (a) as the "batch count" axis (since that is what the track's paper
   actually varies) but flags that a multi-ciphertext axis is unaddressed by
   the source paper.
4. **Modulus bit-width comparison is deliberately asymmetric.** Cheddar's
   headline claim requires comparing its 31-bit-prime/32-bit-word primes
   against baselines' 62–64-bit primes; the paper's own fairness
   normalization is to halve the baseline's *limb count* (not match its
   bit-width) so that both sides process the same total bit-width of
   modulus data. This spec preserves that normalization explicitly as a
   named protocol field rather than leaving "L=24 vs L=48" unexplained.
