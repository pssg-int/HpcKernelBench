# FFCz (fft track) — STATUS

**Outcome: SKIPPED — not an FFT kernel implementation; its FFT usage is a
plain cuFFT library call**

Paper: "FFCz: Fast Fourier Correction for Spectrum-Preserving Lossy
Compression of Scientific Data", IPDPS 2026.
PAPER_KEY = `conf/ipps/RenUDKLYCG26`.
Repo: `https://github.com/rcrcarissa/FFCz`
(commit `96649b36b2a155075682d86fbabeb925e7d9e29a`, 2026-06-13;
`git clone --depth 20` into `./source/`).
Found via `../../output/benchmark_groups.json`'s `fft` group as a "further"
candidate, since the 4 pre-registered candidates all resolved to
SKIPPED/BUILD-FAILED (see sibling STATUS.md files in this directory).

## Why SKIPPED (evidence)

This artifact is **already integrated into the `lossy-compression` track**
(`../lossy-compression/ffcz/STATUS.md`, also SKIPPED there, for a different
but consistent reason: it requires an externally-produced base-compressor
reconstruction as input, not a standalone `compress(D, eb) -> C` kernel).

For the `fft` track specifically, the additional and more basic problem: FFCz
does not implement its own FFT kernel at all. Its correction algorithm
(`GPU/projection_algorithm.cu`, `GPU/projection_algorithm.cuh`) calls cuFFT
directly as a library:

```
$ grep -rln "cufft\|fft(" --include="*.cu" --include="*.cuh" GPU/
GPU/projection_algorithm.cuh
GPU/config.cuh
GPU/projection_algorithm.cu
GPU/main.cu
GPU/projection_algorithm_impl.cuh
```

There is no FFT kernel of FFCz's own to wrap — doing so would just re-time
NVIDIA's own cuFFT, which this project already carries as the `torch-fft`
CUDA baseline (`kernelbench/impls/gpu_cuda.py`). Per ARTIFACT_GUIDE.md rule 7
("If the artifact genuinely does not implement the track's kernel... mark
SKIPPED with evidence and move to the next candidate"), and consistent with
the paper's own framing (its contribution is a spectrum-preserving
*correction* algorithm for lossy compression, not an FFT implementation).

## Verdict

`ffcz: SKIPPED (uses cuFFT as a plain library call in
GPU/projection_algorithm.cu, not its own FFT kernel; the paper's actual
contribution -- dual-domain error-bound correction -- is a lossy-compression
kernel, already evaluated and SKIPPED for a different reason under
../lossy-compression/ffcz/STATUS.md)`
