# cuHPX — STATUS

**Outcome: SKIPPED — no standard FFT callable on plain arrays; SHT != FFT**

Paper: "cuHPX: GPU-Accelerated Differentiable Spherical Harmonic Transforms
on HEALPix Grids", IPDPS 2026. PAPER_KEY = `conf/ipps/ChengSWB26`.
Repo: `https://github.com/NVlabs/cuHPX`
(commit `94e128bcf3abb5807fe2c31b9c20221a0c458d98`, 2026-04-03;
`git clone --depth 50` into `./source/`).

## Why SKIPPED (evidence)

cuHPX's own README positions it as a spherical-harmonic-transform (SHT) and
regridding library for HEALPix grids, not an FFT library:

```python
from cuhpx import SHTCUDA, iSHTCUDA
sht = SHTCUDA(nside, lmax=lmax, mmax=mmax, quad_weights='ring')
coeff = sht(signal)
```

Its only FFT usage is entirely internal to the SHT pipeline, and is not
exposed as a general "transform this N-point array" call:

- `source/cuhpx/hpx_sht.py` calls `torch.fft.rfft`/`torch.fft.fft` (and a
  Bluestein-algorithm path, `healpix_rfft_bluestein`/`healpix_irfft_bluestein`,
  for ring lengths that aren't power-of-two-friendly) as one step inside
  per-isolatitude-**ring** real-FFT helpers (`healpix_rfft_torch`,
  `healpix_irfft_torch`) — every ring on a HEALPix grid has a DIFFERENT
  length (`nphi` varies with latitude), so this is inherently a ragged,
  per-ring transform, not a fixed-shape batched FFT.
- The CUDA-accelerated path (`source/src/harmonic_transform/hpx_fft.cpp` +
  `hpx_fft_cuda.cu`, `healpix_rfft_batch`/`healpix_irfft_batch`/
  `healpix_rfft_class` — the actual `pybind11` entry points, see
  `hpx_fft.cpp:711-714`) wraps a `cufftPlan1d(..., padding_, ...)` call, but
  `padding_` is a HEALPix-specific "pad every ring's `nphi` up to a common
  length via a Bluestein-convolution trick" size (see
  `hpx_fft.h`'s `HealpixFFT`/`HealpixIFFT` classes: `x_pad`/`y_pad` buffers,
  `rfft_pre_process_dispatch`/`rfft_phase_shift_dispatch`/
  `irfft_post_process_dispatch` custom CUDA kernels around the cuFFT call) —
  fused, HEALPix-ring-specific pre/post-processing (padding + phase shift for
  non-power-of-two `nphi`), not a general N-point transform on a plain dense
  array matching this track's `FFTWorkload` (fixed `dims`/`batch`, no
  ring-dependent-length or Bluestein-padding machinery).

There is no function in the package that takes a plain array of a
caller-chosen size and returns its DFT; every FFT call site is bound to the
HEALPix ring-geometry pre/post-processing. Per the task brief's own
pre-registered condition ("only if a standard FFT callable on plain arrays
exists (SHT != FFT); else SKIP with evidence") and ARTIFACT_GUIDE.md rule 7
("If the artifact genuinely does not implement the track's kernel... mark
SKIPPED with evidence and move to the next candidate"), this is SKIPPED
without a build attempt.

## Verdict

`cuhpx: SKIPPED (SHT library; its cuFFT calls are fused into HEALPix
ring-specific Bluestein-padding/phase-shift pre/post-processing
(hpx_fft.cpp/hpx_fft_cuda.cu), no standalone plain-array FFT entry point)`
