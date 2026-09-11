# TurboFNO (fft track) — STATUS

**Outcome: SKIPPED — fuses FFT+GEMM+iFFT into one kernel by design, and
depends on the same broken TurboFFT codegen system as a submodule**

Paper: "TurboFNO: High-Performance Fourier Neural Operator with Fused
FFT-GEMM-iFFT on GPU", SC 2025. Not itself in `output/included.json` under a
`conf/sc/...` key at the time of this check (found via
`../../output/benchmark_groups.json`'s `fft` group,
`key: conf/sc/WuZDZHC25`) — treated as a "further" candidate per the task
brief's fallback instruction, since the 4 pre-registered candidates all
resolved to SKIPPED/BUILD-FAILED (see sibling STATUS.md files in this
directory).
Repo: `https://github.com/shixun404/TurboFNO`
(commit `215d916419b9a7745b803e20d20a45afd335ce14`, 2026-07-01;
`git clone --depth 20` into `./source/`).

## Why SKIPPED (evidence)

TurboFNO's own README states its entire point up front: "Unlike standard FNO
implementations that execute FFT, filtering, GEMM, and iFFT as separate
kernels... TurboFNO introduces the **first fully fused FFT-GEMM-iFFT GPU
kernel**." `fusion_variants/` (its main kernel directory, "All kernel fusion
variants (stepwise E->A->B->C->D for 1D/2D)") is organized entirely around
degrees of fusion between the FFT and the GEMM stages — there is no fusion
variant whose output is "just an FFT," by the project's own framing.

Separately, and independently disqualifying: the README requires
`TurboFFT/` as a **git submodule pinned to the `TurboFNO_dev` branch** of the
`shixun404/TurboFFT` repo — the same repository this directory's `turbofft/`
STATUS.md documents as BUILD-FAILED on its `main` and `artifact` branches
(missing generated kernel files, header/kernel-name mismatches). Even setting
the fusion-design objection aside, building TurboFNO would first require a
working TurboFFT, which was not available.

## Verdict

`turbofno: SKIPPED (fuses FFT+GEMM+iFFT into single kernels by design --
fusion_variants/ has no FFT-only stage; also depends on the shixun404/TurboFFT
TurboFNO_dev submodule, the same broken codegen system documented BUILD-FAILED
in ../turbofft/STATUS.md)`
