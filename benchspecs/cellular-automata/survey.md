# Survey — cellular-automata (singleton track: 1 paper)

## journals/tpds/NavarroQMFH25 — CAT: Cellular Automata on Tensor Cores (TPDS 2025)

- **What it does**: reformulates the update of large-neighborhood-radius cellular automata
  (CA) whose transition function is a *weighted summation of the neighborhood* (the Larger
  Than Life, LTL, family generalizes Conway's Game-of-Life to arbitrary radius `r` and
  configurable birth/survival threshold intervals over that weighted sum) as a matrix
  multiplication, so it can run on GPU tensor cores (and, in a CPU port, Intel AMX) instead
  of on ordinary CUDA cores. Six solver implementations live in one codebase and are
  compared head-to-head: `BASESolver` (naive baseline), `COARSESolver`, `SHAREDSolver`
  (shared-memory-tiled baseline), `MCELLSolver`, `PACKSolver` (bit/byte-packed
  representation), and the paper's own `CATSolver` / `CATMultiStepSolver` /
  `CATMultiStepSolver2` (tensor-core variants, the last two fusing multiple CA steps per
  kernel launch to amortize halo-exchange overhead).
- **workloads/inputs** (confirmed from `tests/test_CG.cu`): a single square grid of side `N`
  (CLI-configurable, default `N=32` in the dev-test harness; the paper's own reported curves
  sweep grid size and radius `1 <= r <= 16`, per the abstract), depth `Z` (number of
  independent CA "layers" simulated in parallel, default 1), cell density `density` for
  random 0/1 initialization (default 0.3, i.e. 30% live cells — CLI signature:
  `test_CG [N] [density] [radius] [Z] [steps] [inner] [regions_x] [regions_y]`), and
  `numSteps` simulation steps (default 2 in the harness; paper figures use larger step
  counts not pinned down in the source read).
- **timing protocol** (confirmed from `tests/test_CG.cu`): `cudaEvent_t` device-side timer,
  wrapping the entire `numSteps`-step simulation loop (`cudaEventRecord` immediately after a
  `cudaDeviceSynchronize()`, before the step loop; `cudaEventRecord`+`cudaEventSynchronize`
  after) for each of the four timed solvers (CATSolver, CATMultiStepSolver,
  CATMultiStepSolver2, PACKSolver) run back-to-back on the same input. **Single trial per
  solver, no discarded warmup, no repetition** visible in the public dev-test harness (the
  same pattern seen across most singleton-track artifacts surveyed in this project). Data
  preparation (`prepareData`, e.g. uint8-to-half/tensor-core-layout packing, and the
  PACKSolver's bit-packing) happens once before the timed loop and its cost is not
  included in `totalKernelMs`, i.e. the harness already separates one-shot preprocessing
  from per-step kernel time.
- **precision & correctness**: CA cells are boolean/low-integer state (uint8 in the
  reference representation); CATSolver's tensor-core path packs cells into GPU tensor-core
  input types (fp16/int8-class, exact for boolean automaton state) and PACKSolver bit-packs
  8 cells per byte. Correctness is checked via `CompareCatPack` (byte/value-exact comparison
  between the CAT solver's grid state and the PACK solver's grid state after simulation) —
  an inter-implementation agreement check, not a comparison against an independent
  reference implementation or analytical solution.
- **metric**: the harness computes `Gcells/s = totalCells / (totalKernelMs/1000) / 1e9`
  where `totalCells = N*N*Z*numSteps` (i.e. **giga cell-updates per second**), plus raw
  kernel ms and per-step average ms, plus a dimensionless speedup of each tensor-core
  variant over the plain `CATSolver`. The paper's own headline numbers are speedup ratios
  (up to 101x over a GPU baseline, ~14x over the fastest prior GPU approach) and, separately,
  energy efficiency (J or W, from the abstract, not directly visible in the timing harness
  read).
- **baselines**: the paper's own prior-state-of-the-art GPU CA implementations
  (`BASESolver`/`COARSESolver`/`SHAREDSolver`/`MCELLSolver`/`PACKSolver`, spanning naive to
  bit-packed shared-memory approaches) plus a CPU AMX port (separate repo,
  `kezada94/CAT_cpu`, referenced in the README but not surveyed here as a distinct
  artifact).
- **source**: `github.com/temporal-hpc/CAT` — `README.md` (abstract, build instructions),
  `tests/test_CG.cu` (CLI args, timing loop, correctness check, throughput formula),
  `tests/main.cu` (a second, simpler benchmark driver with hardcoded `N=2048`, `RADIUS=1`
  constants), `src/*.cu` (six solver implementations, not read in depth beyond file names).

## Divergences

- Single-paper track: no cross-paper divergence. The main internal tension is between the
  **abstract's claimed evaluation range** (`1 <= r <= 16`, multiple grid sizes, energy
  measurements, CPU-AMX comparison) and **what the public dev-test harness actually runs**
  (a single hardcoded default configuration unless overridden via CLI args, no energy
  instrumentation visible, no automated sweep script found in the repository listing). This
  spec's protocol section follows the harness's own timing methodology (CUDA events, no
  warmup/repeat) since that is what is verifiable, but recommends multi-trial repetition as
  a fairness fix (flagged in `notes_on_fairness`) rather than presenting single-trial timing
  as if it were already best practice.
- The correctness check found (`CompareCatPack`) validates internal consistency between two
  of the paper's own implementations, not against an independent reference (e.g. a serial
  CPU implementation of the same LTL rule). This spec's correctness gate therefore adds an
  explicit reference-implementation requirement beyond what the artifact itself checks,
  flagged in `open_questions`.
