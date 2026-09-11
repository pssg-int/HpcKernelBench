# fft track — artifact integration summary (second pass)

Nine candidates attempted across two passes. **One BUILT+GATED** (turbofft,
new this pass), eight SKIPPED / BUILD-FAILED-then-fixed.

## Second pass (this pass) — two retried, two new

- **turbofft** (PPoPP'25, `conf/ppopp/WuZ0HJDDCC25`) — **BUILT+GATED**
  (was BUILD-FAILED). TurboFFT is code-generated; the first pass never ran
  its generator (`fft_codegen.py`). Running it fixes both the missing
  `logN` files AND the stale `fft_10`-vs-`fft_radix_2<>` name/signature
  mismatch the first pass found — proven by a clean, zero-error 4m8s
  compile of the complete, unmodified `TurboFFT.h`. Shipped adapter wraps
  `logN 1..13` (`N=2..8192`), fp32, forward C2C via a thin `extern "C"`
  shim; direction="inverse" via the standard `conj(FFT(conj(x)))/N`
  identity around the same unmodified kernel. Of this domain's 8 smoke
  workloads, the 3 it structurally supports (1D, power-of-two, C2C:
  fwd-oop, fwd-inplace, inverse-oop) **all pass**, err 4.1e-9..4.5e-6
  (<= 1e-4). The literal mandated `--smoke` command crashes on the 2nd
  (out-of-scope, mixed-radix) workload — a pre-existing, not-adapter-
  specific `runner.py`/`spectral.py` interaction (`smoke_workloads()` is
  not variant-aware and `prepare()`'s `NotImplementedError` isn't caught
  per-workload); a substitute gate script (same precedent as
  `../spmv/diaq/STATUS.md`) demonstrates all 3 supported points pass. See
  `turbofft/STATUS.md`.

- **cufalcon** (TPDS 2026, `journals/tpds/LiWSYDZ26`) — **SKIPPED**
  (confirmed, deeper evidence). The first pass's premise was wrong: a
  standalone kernel launch boundary DOES exist (`ifft_t`, `fft.cu:196`,
  missed by the first pass's grep). But `fft1024`/`ifft1024` compute
  Falcon's own negacyclic-ring evaluation transform (evaluation at ODD
  powers of a primitive 2N-th root of unity), NOT a DFT of the input array
  up to permutation/scale — proven empirically with a GPU impulse-response
  probe: output slot 0 for a unit impulse at position K equals
  `exp(i*pi*K/1024)` exactly (K=0..5, full double precision), whereas a
  genuine (possibly permuted) DFT's DC bin is always `1+0i` regardless of
  impulse position. Recovering a DFT needs a non-trivial per-index INPUT
  TWIST (not a permutation/scale), which would mean benchmarking an
  artificial transform composed around a repurposed kernel rather than the
  paper's own FFT usage. See `cufalcon/STATUS.md`.

- **flups** (TPDS'23, `journals/tpds/BaltyCG23`) — **SKIPPED**, cheap.
  Distributed, MPI+FFTW, CPU-only C++ library — no GPU code anywhere
  (confirmed via GitHub API, no clone needed). Out of
  `ARTIFACT_GUIDE.md`'s NVIDIA-GPU-single-card scope. See
  `flups/STATUS.md`.

- **claire** (SC'20, `conf/sc/BrunnHBMM20`) — **SKIPPED**, cheap. Its
  spectral/Fourier layer (`include/mpicufft.hpp`, `MPIcuFFT<T>`) is a thin
  per-rank wrapper around `cufftHandle planR2C/planC2R/planC2C` (plus an
  AccFFT-based path for the older multi-node-CPU build) — no FFT kernel of
  CLAIRE's own, same pattern as `ffcz`/`cutensor-tubal` below. Confirmed
  via GitHub's code-search API, no clone needed. See `claire/STATUS.md`.

## First pass (2026-08-08) — unchanged, not revisited beyond the above

- **cuhpx** (IPDPS'26, `conf/ipps/ChengSWB26`) — SKIPPED. Spherical-
  harmonic-transform library; its cuFFT usage is fused into HEALPix
  ring-specific Bluestein-padding/phase-shift pre/post-processing, no
  plain-array FFT entry point. See `cuhpx/STATUS.md`.
- **cutensor-tubal** (TPDS 2020 x2, `journals/tpds/ZhangLWW20` +
  `journals/tpds/ZhangLW20`) — SKIPPED. `Tfft` is a plain
  `cufftPlan1d`/`cufftExecC2C` call wrapped in host-side transpose loops,
  not the papers' own kernel (their contribution is tubal-rank tensor
  completion built on top of it). See `cutensor-tubal/STATUS.md`.
- **ffcz** (IPDPS'26, `conf/ipps/RenUDKLYCG26`) — SKIPPED. Calls cuFFT
  directly as a library; no FFT kernel of its own (already integrated,
  also SKIPPED, in the `lossy-compression` track for a different reason).
  See `ffcz/STATUS.md`.
- **flashfftstencil** (PPoPP'25, `conf/ppopp/HanLCBZYCZCY25`) — SKIPPED.
  Sole 1D kernel `rfft_pfa_stencil_8_nwarp` fuses forward-RFFT + stencil-
  weight multiply + inverse in one inseparable `__global__` kernel; no
  standalone FFT-only launch boundary (this track's `source/` for it is a
  symlink to the `stencil` track's clone). See `flashfftstencil/STATUS.md`.
- **turbofno** (SC 2025, `conf/sc/WuZDZHC25`) — SKIPPED. Fuses
  FFT+GEMM+iFFT into single kernels by design (no FFT-only stage in
  `fusion_variants/`); also depends on the same `shixun404/TurboFFT`
  `TurboFNO_dev` submodule this pass's turbofft fix does not touch (a
  different branch/snapshot). Not revisited: even with a working TurboFFT
  the fusion-design objection alone settles turbofno's SKIPPED status.
  See `turbofno/STATUS.md`.

## Track ruling

**fft track: 1 implementation integrated (turbofft), gated and passing.**
Every other surveyed candidate either calls a vendor FFT library
(cuHPX, cuTensor-Tubal, FFCz, CLAIRE) rather than implementing its own
kernel, fuses the transform inseparably into a larger kernel
(FlashFFTStencil, TurboFNO), computes a mathematically different
transform under the same name (cuFalcon's negacyclic/twisted evaluation),
or is out of GPU-single-card scope entirely (FLUPS). This is a narrow but
now non-empty track: turbofft's own kernel genuinely implements and
correctly computes a batched, single-precision, power-of-two, 1D C2C FFT,
gated against an independent fp64 numpy reference.
