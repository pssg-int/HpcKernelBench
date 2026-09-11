# FLUPS — STATUS

**Outcome: SKIPPED — distributed, CPU-only library (no GPU code at all);
out of current scope per ARTIFACT_GUIDE.md's scope ruling**

Paper: "FLUPS - A Flexible and Performant Massively Parallel Fourier
Transform Library", TPDS 2023. PAPER_KEY = `journals/tpds/BaltyCG23`.
Repo: `https://github.com/vortexlab-uclouvain/flups`

Untried candidate from `output/benchmark_groups.json`'s "fft" group,
`platform: ["distributed", "cpu"]`. Confirmed cheaply via GitHub API
(`gh api repos/vortexlab-uclouvain/flups`, `gh api
repos/vortexlab-uclouvain/flups/readme`), no clone needed:

- Repo language: C++ (MPI), no CUDA/HIP anywhere.
- README's own "Installation"/"Dependencies" section requires an MPI
  compiler (`mpicc`/`mpic++`) and **FFTW3** (`--enable-mpi
  --enable-openmp`, `fftw3 > v3.3.8`) plus `h3lpr` (a separate MPI utility
  library) — FLUPS computes distributed FFTs by calling FFTW across MPI
  ranks (a "node-centered data layout and non-blocking comms" library,
  per the group's own `one_liner`), not a GPU kernel.
- README explicitly states its purpose is a **distributed Poisson solver**
  built on precomputed Green's functions and FFTW-based transforms across
  MPI ranks — the paper's contribution (per its own citation blurb, "a
  flexible and performant massively parallel Fourier transform library")
  is the communication/data-layout design for scaling FFTW across many
  CPU nodes, not a novel single-GPU FFT kernel.

Per ARTIFACT_GUIDE.md's scope ruling ("Integration targets NVIDIA GPU,
single-card implementations only for now. CPU-only, multi-GPU/
distributed-only... artifacts are SKIPPED with a one-line reason") this is
a clean, cheap SKIP — no build attempted.

## Verdict

`flups: SKIPPED (CPU-only, MPI-distributed FFTW wrapper library, no GPU
code anywhere in the repo -- confirmed via GitHub API without cloning; out
of ARTIFACT_GUIDE.md's NVIDIA-GPU-single-card scope)`
