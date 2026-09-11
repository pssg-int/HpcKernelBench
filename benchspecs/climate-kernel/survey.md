# Climate-kernel track — evaluation-methodology survey

`output/benchmark_groups.json["climate-kernel"]` has exactly 2 papers, matching
`data/track_inputs/climate-kernel.json` exactly (this track's input file DOES
exist, unlike `lattice-boltzmann`/`fdtd-seismic`). Both papers are surveyed
(2/2, above the "≥5 or all" bar). Note up front: these 2 papers are not
"the same kind of climate kernel" at all — one is a numerical PDE dynamical
core (a stencil, cross-listed in the `stencil` track), the other is a
statistical/ML-adjacent climate *emulator*'s linear-algebra kernel. This
divergence is the defining fact of this small track and shapes its 2 variants
directly (see Divergences).

---

## ozHOPE — Ozaki-scheme FP16 Tensor Core acceleration of a shallow-water dynamical core
IPDPS 2026, `conf/ipps/YaoZLX26`, no arXiv/OA; repo README + abstract
(`gh api`). **Reused from `benchspecs/stencil/survey.md`** (same paper,
cross-listed in both tracks — `benchmark_groups.json` places it in both
`stencil` and `climate-kernel`); summarized here for this track's own axes:

- **Model class**: HOPE, a **research dynamical core** — the fluid-dynamics
  integration engine of a shallow-water atmospheric model, NOT a full Earth
  System Model (no radiation, no microphysics, no land/ocean coupling). This
  is a genuine "dynamics-only, no physics parameterization" kernel in this
  track's physics-vs-dynamics axis sense.
- **Technique**: convolution-reformulated stencil computation on FP16 Tensor
  Cores via the Ozaki scheme (emulates higher working precision, tunable up
  to FP64, from FP16 hardware).
- **Benchmark cases**: the **Williamson standard test suite** for
  atmospheric dynamical cores — steady-state geostrophic flow,
  Rossby-Haurwitz wave, perturbed jet flow. These are domain-standard
  validation problems with known analytic/reference solutions, not a
  synthetic grid-size sweep — i.e. "resolution" in this paper means the
  Williamson suite's own standard resolution, not an arbitrary chosen size.
- **Hardware/software**: NVIDIA A40/RTX A6000, CUDA 11.8, cuDNN 9.1.0 (via a
  cuDNN-frontend Python interface), PyTorch 2.6.
- **Correctness**: diagnostic variables compared against an FP64 reference
  per benchmark case; convergence-order preservation is itself a first-class
  reported result (not just pass/fail) — directly relevant to this track's
  "throughput vs. accuracy" tension for any low-precision climate kernel.
- **Metric**: end-to-end speedup (1.82x average, up to 4.34x) — NOT
  simulated-days/day (SYPD), the metric a production GCM community would
  recognize; this paper reports neither wall-clock-per-simulated-day nor
  cell-updates/s directly, only a relative speedup number.
- **Source**: `github.com/jnyao/ozHOPE` README + abstract (already fetched
  for `stencil` track).

---

## Abdulah et al. — Boosting Earth System Model Outputs / Exascale Climate Emulators (HiCMA-PaRSEC)
SC 2024, `conf/sc/AbdulahBBCCGKKL24`, no arXiv/OA link recorded; **ACM Gordon
Bell Prize for Climate Modelling winner**; artifact `github.com/ecrc/hicma-x`
(`artifact_status: likely` in Phase-1's own record; confirmed and deepened
via `gh api` for this survey — README, TESTS.md, and
`tests/testing_climate_emulator.c` all read directly)

- **What this actually is**: NOT a physical GCM/dynamical-core kernel. It is
  a **statistical spatio-temporal climate emulator** — uses the spherical
  harmonic transform (SHT) to stochastically model spatio-temporal
  variation in climate fields, trained on real Earth System Model output
  (318 billion hourly temperature points from a 35-year, 31-billion-daily-
  point 83-year ensemble) to generate cheap statistical replicas at
  ultra-high spatial resolution (0.034 degrees, ~3.5km) without re-running
  the expensive physical model. The "kernel" being benchmarked is the
  emulator's own compute-heavy core: forward/inverse SHT + a
  mixed-precision, tile-low-rank (TLR) Cholesky factorization (POTRF) of the
  resulting spatial covariance structure.
- **CLI/parameters** (`tests/testing_climate_emulator.c`, confirmed via `gh
  api`): `--latitude L` sets the grid via **N = L^2** (spherical-harmonic
  grid, latitude x longitude points collapse to one size parameter);
  `--NB` tile size; `--gpus`; `--nruns` (repetition count for the POTRF
  phase specifically, see Timing below); `--adaptive_decision` /
  `--adaptive_memory` (mixed-precision tile-format selection); mixed
  precision bands controllable via `--band_dense_dp/sp/fp8`
  (double/single/FP8 arithmetic bands within one factorization).
  Confirmed example: `--latitude 120 --NB 1440 --N 14400`.
- **Timing structure** (`performForwardSHT`, `performMatrixOperations`,
  `performCholeskyFactorization`, `performInverseOperations` in the C
  source, each independently wrapped in `SYNC_TIME_START`/`SYNC_TIME_PRINT`
  macros): forward SHT (reported in its own Gflop/s), forward-SHT reshape,
  matrix memory allocation, SYRK (C = A A^T), matrix-norm computation, THEN
  **the Cholesky factorization loop specifically is repeated `nruns` times**
  (`for (int i = 0; i < params->nruns; i++) hicma_parsec_potrf(...)`) while
  every other phase (SHT forward/inverse, SYRK, norm) runs exactly once —
  i.e. the harness's OWN convention already separates one-shot preprocessing
  from a repeatable kernel phase, matching this whole survey effort's
  "preprocessing reported separately, kernel phase repeated" principle
  without this spec having to impose it.
- **Precision**: mixed — double/single/half/FP8 tile-level bands within one
  factorization, "adaptive memory management and precision selection" is
  the paper's own headline contribution (not a fixed global precision the
  way most other tracks' papers use).
- **Correctness**: Mean Square Error (MSE) computed against a MATLAB
  reference result (`climate_emulator_diff_double`, `climate_emulator_mse`)
  — this is a genuinely quantitative, automated correctness check, not just
  a conserved-quantity drift check.
- **Metric**: the paper's own headline number (from `output/included.json`'s
  one_liner, not independently re-verified against fulltext) is **0.976
  EFlop/s on Frontier** — i.e. this paper's primary claimed metric is raw
  sustained FLOP/s of the mixed-precision solver at exascale, not a
  climate-domain-native throughput unit (not SYPD, not grid-points/s).
- **Source**: `github.com/ecrc/hicma-x` — `README.md`, `TESTS.md`,
  `tests/testing_climate_emulator.c` (all fetched directly via `gh api`).

---

## Divergences

1. **This track's 2 papers are not comparable kernels at all — they share
   only the word "climate."** ozHOPE is a numerical PDE dynamical-core
   stencil (Tensor-Core-accelerated shallow-water fluid dynamics).
   HiCMA-PaRSEC's climate emulator is a statistical/ML-adjacent
   spherical-harmonic-transform + TLR-Cholesky solver that does not solve
   any physical equation of motion at all — it emulates the STATISTICS of
   an ensemble of prior GCM runs. The spec below gives each its own variant
   rather than forcing a shared input suite or metric, matching this
   track's genuine bimodality (analogous to how `lattice-boltzmann`'s
   regular-grid vs. complex-geometry papers get separate variants, but more
   extreme here — these 2 papers aren't even solving the same class of
   mathematical problem).

2. **Neither paper reports simulated-days/day (SYPD)**, the metric a
   production climate-modeling community (CESM, E3SM, ICON, etc.) would
   recognize as the standard throughput unit for comparing a climate kernel
   against real deployment requirements. ozHOPE reports a relative speedup
   number; HiCMA-PaRSEC reports raw EFlop/s. This is flagged as an open
   question rather than silently converted to SYPD, since neither paper's
   own simulated-time-per-step is confirmable from the sources read (ozHOPE
   IS a real physical timestep, but its physical-time-per-simulated-day
   conversion was not extracted; HiCMA-PaRSEC's emulator doesn't have a
   "simulated day" concept in the same sense at all — it produces one
   statistical realization per SHT+Cholesky pass, not a timestepped
   trajectory).

3. **"Physics-vs-dynamics split" (the parent task's own suggested axis)
   applies cleanly to ozHOPE (pure dynamics, no physics parameterization —
   explicitly a research dynamical core) but does not apply to HiCMA-PaRSEC
   at all** — there is no dynamics/physics distinction in a statistical
   emulator; the relevant analogous axis for HiCMA-PaRSEC is
   precision-band allocation (which tiles get fp64 vs fp32 vs fp16 vs fp8)
   traded against reconstruction MSE, which the spec surfaces as its own
   axis for variant 2 instead of forcing physics-vs-dynamics onto it.

4. **HiCMA-PaRSEC's own timing harness already implements the
   preprocessing/kernel separation this whole survey effort has had to
   argue FOR in every other track** (`stencil`'s Tensor-Core variant,
   `fdtd-seismic`'s autotuning-exclusion, `lattice-boltzmann`'s
   geometry-indexing exclusion) — its `nruns` loop around POTRF specifically,
   with SHT/SYRK/norm each timed once outside that loop, is exactly the
   discipline this spec would otherwise have to impose from outside. This
   is noted as a positive finding, not a divergence to fix.

5. **Both papers are outliers relative to the general `stencil` track's
   own convention of a synthetic microbenchmark suite** — ozHOPE uses the
   Williamson standard test suite (domain-standard, not synthetic) and
   HiCMA-PaRSEC uses a size parameter (`--latitude`) tied to a specific
   real spherical-harmonic grid convention, not a generic cube_edge/T
   sweep. Both variants below are paper-native rather than forcing either
   into `stencil`'s star/box{2d,3d}{r} taxonomy, which simply does not
   apply to either paper's problem domain.
