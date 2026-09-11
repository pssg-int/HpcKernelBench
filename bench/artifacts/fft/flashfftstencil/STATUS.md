# FlashFFTStencil (fft track) — STATUS

**Outcome: SKIPPED — the FFT stage is not separable from the stencil fusion**

Paper: "FlashFFTStencil: Bridging Fast Fourier Transforms to Memory-Efficient
Stencil Computations on Tensor Core Units", PPoPP 2025.
PAPER_KEY = `conf/ppopp/HanLCBZYCZCY25`.
Repo: `https://github.com/KevinWu2017/FlashFFTStencil`
(commit `4579ea11ccb490aaa5d6d6246e1345c0e9b48c1f`, 2025-06-05).

`./source` is a **symlink** to `../../stencil/flashfftstencil/source`
(cloned once by the sibling agent integrating this artifact into the
`stencil` track; this directory adds no second clone, per the task brief).

## Why SKIPPED (evidence)

FlashFFTStencil's entire design goal (per its own abstract: "Kernel Tailoring
on HBM fuses distinct kernels to enhance parallelism while reducing memory
transfer and footprint") is to fuse the FFT, the stencil-weight
multiplication, and the inverse transform into a SINGLE kernel — it is not
possible to isolate "just the FFT" without touching kernel code.

Concretely, for the 1D case (`src/1D/`, `1d_main.cu`), the entire pipeline —
forward real-FFT via Prime-Factor-Algorithm decomposition (`pfa_size=56 =
8*7`, matching this track's own smoke workload
`smoke-1d-mixedradix56-c2c-fwd-oop`, chosen for exactly this reason per
`kernelbench/domains/spectral.py`'s comment on `_FFT_SMOKE`), elementwise
multiplication by the stencil weights in the frequency domain, and (implied)
the inverse transform, is ONE `__global__` kernel:

```
$ grep -n "__global__\|__device__" src/1D/rfftstencil_pfa/rfft_8_pfa_fastcomplex.cu
18:__global__ void rfft_pfa_stencil_8_nwarp(
```

There is exactly one kernel entry point in the whole 1D pipeline
(`rfft_pfa_stencil_8_nwarp`), and it is already named for what it does: an
RFFT fused with the stencil computation. No separate "forward-transform-only"
kernel exists to call in isolation; writing one would mean extracting a
sub-portion of `rfft_pfa_stencil_8_nwarp`'s device code into a new kernel —
exactly the kind of kernel-code change ARTIFACT_GUIDE.md rule 3 disallows.
(2D/3D directories follow the same fused-kernel pattern per the paper's
"Architecture Aligning on SMEM" and "Computation Streamlining on TCU"
techniques, not inspected further since the 1D case already settles the
verdict.)

## Verdict

`flashfftstencil: SKIPPED (sole kernel rfft_pfa_stencil_8_nwarp fuses
forward-RFFT + stencil-weight multiply + inverse in one inseparable
__global__ kernel; no standalone FFT-only launch boundary, matching the
paper's own "Kernel Tailoring" fusion design)`
