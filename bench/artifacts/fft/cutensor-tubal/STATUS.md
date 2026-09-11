# cuTensor-Tubal (fft track) — STATUS

**Outcome: SKIPPED — not an FFT kernel implementation; its FFT usage is a
thin, host-transpose-heavy wrapper around cuFFT**

Papers (this one repo backs both `benchmark_groups.json` "fft" entries):
- "cuTensor-Tubal: Efficient Primitives for Tubal-Rank Tensor Learning
  Operations on GPUs", TPDS 2020. PAPER_KEY = `journals/tpds/ZhangLWW20`.
- "High Performance GPU Tensor Completion With Tubal-Sampling Pattern",
  TPDS 2020. PAPER_KEY = `journals/tpds/ZhangLW20` (status: "likely", per
  `output/included.json`).

Repo: `https://github.com/YangletLiu/cuTensor_CUDA_Library_for_Transform_based_Tensors`
(commit `a53ac3fff554a5a8272f1266d658dadf4d367f02`, 2021-05-12;
`git clone --depth 20` into `./source/`).
Found via `../../output/benchmark_groups.json`'s `fft` group as a "further"
candidate, since the 4 pre-registered candidates and 2 additional
same-list entries all resolved to SKIPPED/BUILD-FAILED (see sibling
STATUS.md files in this directory).

## Why SKIPPED (evidence)

The library's per-tube FFT helper (`GPU/apps/app1/Tfft.h`, `fft.cu`) is a
direct `cufftPlan1d`/`cufftExecC2C` call, bracketed by a host-side
double-loop transpose and an explicit H2D/D2H round trip on every
invocation:

```cpp
void Tfft(float *t,int l,int bat,cufftComplex *tf) {
    cufftComplex *t_f = new cufftComplex[l*bat];
    for(int i=0;i<bat;i++) for(int j=0;j<l;j++) { /* host transpose into t_f */ }
    cudaMalloc(&d_fftData, ...);
    cudaMemcpy(d_fftData, t_f, ..., cudaMemcpyHostToDevice);
    cufftHandle plan = 0;
    cufftPlan1d(&plan, l, CUFFT_C2C, bat);
    cufftExecC2C(plan, d_fftData, d_fftData, CUFFT_FORWARD);
    cudaMemcpy(t_f, d_fftData, ..., cudaMemcpyDeviceToHost);
    for(int i=0;i<bat;i++) for(int j=0;j<l;j++) { /* host transpose back */ }
}
```

This is not the paper's own FFT kernel — it is NVIDIA's cuFFT, wrapped with
an inefficient host-side transpose that would not represent the library's
actual performance claim even if wrapped (the papers' real contribution is
the t-SVD-based tubal-rank tensor-completion/learning algorithm built on top
of this, not the FFT step itself). Wrapping `Tfft` for this benchmark would
just re-time cuFFT through an artificially slow host-transpose harness,
which is worse than useless for a "paper's own kernel" comparison and
duplicates the project's existing `torch-fft` CUDA baseline
(`kernelbench/impls/gpu_cuda.py`). Per ARTIFACT_GUIDE.md rule 7.

## Verdict

`cutensor-tubal: SKIPPED (Tfft in GPU/apps/app1/fft.cu is a plain
cufftPlan1d/cufftExecC2C call wrapped with host-side transpose loops, not
the papers' own kernel -- their contribution is tubal-rank tensor
completion/learning built on top of this cuFFT call, not an FFT
implementation)`
