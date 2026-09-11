# CLAIRE — STATUS

**Outcome: SKIPPED — ships no FFT kernel of its own; its Fourier/spectral
layer is a distributed wrapper around cuFFT (single-GPU) / AccFFT
(multi-GPU), confirmed without cloning**

Paper: "Multi-node multi-GPU diffeomorphic image registration for
large-scale imaging problems", SC 2020. PAPER_KEY = `conf/sc/BrunnHBMM20`.
Repo: `https://github.com/andreasmang/claire`

Untried candidate from `output/benchmark_groups.json`'s "fft" group,
`platform: ["nvidia-gpu", "distributed"]`. Checked via GitHub's code-search
API (`gh api search/code?q=repo:andreasmang/claire+cufft` /
`+AccFFT`), no clone needed:

- `cufft` appears in 13 files, `AccFFT` in 35, including
  `include/mpicufft.hpp` / `src/Spectral/mpicufft.cpp` (a `MPIcuFFT<T>`
  class), `cmake/FindACCFFT.cmake` (a CMake find-module for the external
  AccFFT library), `include/Spectral.hpp`, `src/Spectral/Spectral.cpp`.
- `include/mpicufft.hpp` (fetched directly via `gh api
  repos/andreasmang/claire/contents/include/mpicufft.hpp`) is unambiguous:

  ```cpp
  #include <cufft.h>
  #include <cuda.h>
  #include <mpi.h>
  ...
  class MPIcuFFT {
    ...
    virtual void execR2C(void *out, const void *in);
    virtual void execC2R(void *out, const void *in);
    ...
  protected:
    cufftHandle planR2C;
    cufftHandle planC2R;
    cufftHandle planC2C;
    ...
  };
  ```

  This is a thin per-rank cuFFT wrapper (create `cufftHandle` plans, call
  cuFFT's exec functions) plus CLAIRE's own MPI data-redistribution
  (pencil/slab decomposition, `changeSize`/`restrictTo`/`prolongFrom`)
  around it — the FFT computation itself is NVIDIA's own cuFFT, not
  CLAIRE's. `cmake/FindACCFFT.cmake` shows the (older/CPU-cluster) build
  path uses the third-party AccFFT library instead, for the same reason.

This matches the fft spec's own framing of CLAIRE (`benchspecs/fft/
spec.yaml`'s "an FFT embedded inside a larger scientific pipeline... FFCz,
CLAIRE" and "Plan creation... a runtime plan object (cufftPlan*/
fftw_plan_dft*)... FLUPS" -- survey.md Divergence 3) and the SAME pattern
already ruled SKIPPED in this track for `ffcz`/`cutensor-tubal`
(`../ffcz/STATUS.md`, `../cutensor-tubal/STATUS.md`): the paper's actual
contribution (multi-GPU diffeomorphic image-registration solver /
preconditioner) is built on top of a vendor FFT library call, not an FFT
kernel of its own. Per ARTIFACT_GUIDE.md rule 7, this is a clean, cheap
SKIP — no build attempted.

## Verdict

`claire: SKIPPED (Fourier/spectral layer is MPIcuFFT -- a thin
cufftHandle planR2C/planC2R/planC2C wrapper, include/mpicufft.hpp -- plus
an AccFFT-based multi-node path, cmake/FindACCFFT.cmake; no FFT kernel of
CLAIRE's own to wrap, confirmed via GitHub API without cloning, same
pattern as ../ffcz and ../cutensor-tubal)`
