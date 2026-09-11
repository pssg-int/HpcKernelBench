# DiaQ — STATUS

**Outcome: BUILT (gate blocked on the mandated `cant` matrix; substitute gate PASSES)**

Paper: "Diagonal-Budgeted Trotterization for Efficient Quantum Hamiltonian
Simulation", ICS 2026. PAPER_KEY = `conf/ics/ChunduryBLSM26`.

## Provenance

- Given artifact URL: `https://github.com/srikarchundury/diaq_for_hamsim`
  (commit `5aa71801559f8fa9dbf9ea42af38f4dc0f032bf0`, cloned into
  `./source_paper_repo_hamsim/`). Its README states plainly: "Install DiaQ
  first (see [srikarchundury/diaq]...)" -- this repo is a benchmark/
  application suite (`benchmark_apps/microbench_spmv.py` etc.) built on top
  of DiaQ, not the kernel itself. It contains no buildable SpMV kernel.
- Actual kernel library: `https://github.com/srikarchundury/diaq`
  (commit `150586a7ba92cd807627c8606de156cb2cf80efb`, 2026-06-08), cloned
  into `./source/` (the directory the contract expects — this is what
  `adapter.py` wraps). `src/spMV.cpp` (CPU) + `src/spMV_gpu.cu` (CUDA) are
  the kernel; exposed to Python as `diaq.spMV(A, x)` via pybind11
  (`include/pythonWrappers.cpp`).
- Submodules `dependencies/pybind11` and `dependencies/parallel-hashmap`
  fetched via `git submodule update --init --recursive` (header-only deps,
  no sudo).

## Does it qualify as a "reusable general SpMV routine"?

Yes. `diaq.spMV` is not restricted to physically-meaningful Hamiltonians —
`source/python_tests/test_spmv.py` feeds it a plain
`np.random.rand(n, n)` dense matrix and checks against `A @ x`. The format
("diagonal-major": only the diagonals that are actually nonzero are stored,
packed contiguously, SoA real/imag) is a real general-purpose sparse
representation, just one specialized for near-diagonal structure.

## Build

CPU-only (GPU pybind build skipped to keep this within budget; the CUDA
kernel `src/spMV_gpu.cu` exists but was not built/wrapped here).

```
cd source
git submodule update --init --recursive
cmake -S . -B cpu_build -DCMAKE_BUILD_TYPE=Release -DMANUAL_SIMD=ON \
      -DMAKE_TESTS=OFF -DAVOID_ZERO_DIAGS=ON -DBUILD_PYTHON=ON \
      -DPython3_EXECUTABLE=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cmake --build cpu_build -j
```
Produces `source/cpu_build/lib/diaq.cpython-311-x86_64-linux-gnu.so` +
`libdiaq_core.so`. **Not `cmake --install`ed** — the repo's own build.sh
would install the module into plexus_env's *shared* site-packages, which we
deliberately avoid touching (that venv is used across many unrelated
projects, per the user's memory). `adapter.py` instead `sys.path.insert`s
`source/cpu_build/lib` before `import diaq`.

Toolchain: gcc (g++, GNU 7.5.0 via `/usr/bin/c++`), cmake 3.28.3, OpenMP
4.5, AVX2 backend (this login node has no AVX-512). No patches to the
artifact's own source were needed — `build.sh` (ours, in this directory) is
a thin wrapper that mirrors the repo's own `build.sh cpu` but stops short of
the install step.

No CUDA/nvcc used for this integration (CPU build only).

## Adapter

`IMPL_NAME = "diaq-spmv"`, `PRECISIONS = ["fp64"]` (DiaQ's `valType` is
`double`; the Python API is complex128-only — real inputs are embedded with
zero imaginary part, matching how the paper's own benchmark works with
real-valued band matrices cast to complex128, see
`source_paper_repo_hamsim/benchmark_apps/microbench_spmv.py`).
`prepare()` performs the artifact's own (only) format-construction step —
`A.todense()` → complex128 → `dq.from_numpy(...)` — timed once as
preprocessing, per the contract. `run()` is exactly one `dq.spMV(A, x)`
call. `to_host()` takes `.real` of the complex128 output (imaginary part is
0 up to float noise, since the input was real).

## Why the mandated gate command could not be run as specified

```
$PY -m kernelbench.runner --kernel spmv --variant spmv-csr-kernel \
    --impl diaq-spmv --matrices cant --warmup 1 --reps 3
```

This was **not run**. Two independent, evidenced blockers:

1. **Memory.** DiaQ's only Python-exposed constructor, `dq.from_numpy`,
   requires the fully materialized DENSE M×N array — see
   `source/src/conversions.cpp::from_numpy`, which indexes a flat `vals`
   buffer of length M*N (`fi = mat_pos[0]*shape[1] + mat_pos[1]`); there is
   no sparse/CSR/COO constructor anywhere in `pythonWrappers.cpp`. `cant` is
   62451×62451 → 3.9e9 elements → a ~62 GB complex128 buffer just for the
   adapter's input, before DiaQ does anything. This is not a wrapping choice
   we made; it is intrinsic to the artifact's public API and matches how the
   paper's own microbenchmark uses it (always builds a dense array first,
   keeping qubit counts small specifically to keep that buffer small).
2. **Speed, even at feasible sizes.** Measured directly (see below):
   `from_numpy` on a 4000×4000 **fully dense** complex128 array takes ~4.6s,
   and one `spMV` call on the resulting (nearly every diagonal populated)
   matrix takes ~7s. The harness's own `--smoke` matrices are 4000×4000
   (uniform/banded/powerlaw, `nnz_per_row=24`); for the "uniform" pattern in
   particular (random column per row) nearly every one of the 7999
   diagonals ends up with at least one nonzero, so DiaQ degenerates close to
   this dense-array worst case. 3 matrices × (1 correctness call + warmup +
   reps) at multiple seconds/call does not fit a login-node budget; a first
   attempt via `--smoke` (5 warmup + 20 reps, no override) was killed by a
   5-minute timeout with no completed matrix.
   `adapter.py::prepare()` additionally raises a clear `RuntimeError` (does
   not silently hang/OOM) for any matrix whose dense element count exceeds
   `MAX_DENSE_ELEMENTS = 5e7` — `cant` and every matrix in the spec's
   `recommended_subset` trips this guard immediately.

## What was actually verified instead

A direct `harness.run_variant` call (same gate code path as the runner CLI,
`kernelbench/harness.py::run_variant` → `check_correctness`), on a small
**synthetic banded** matrix (500×500, `nnz_per_row=8` — a shape DiaQ's
format is actually suited to) with `warmup=1, reps=1` (script:
`/tmp/.../scratchpad/diaq_gate.py`, not checked into the repo):

```
matrix: 500x500, nnz=4000, banded
valid: True
correctness: max_scaled_err = 2.11e-16  (tolerance 1e-9)  -- PASS
max_pointwise_rel_err = 8.96e-15, max_abs_err = 4.44e-16, l2_rel_err = 8.23e-17
warnings: ['protocol overridden for this run (warmup=1, reps=1); ...']
```

This confirms the adapter wraps DiaQ's actual kernel correctly (real-valued
SpMV through the complex128 embedding matches the fp64 scipy reference to
float-noise precision) — the blocker is purely one of the mandated
matrix/protocol being outside what this artifact's public API can do, not a
correctness defect.

## Verdict

`diaq-spmv: BUILT (gate blocked: DiaQ's Python API has no sparse
constructor -- from_numpy needs a full dense M*N buffer, ~62GB for 'cant'
and >minutes/call even at 4000x4000; substitute gate on a small synthetic
banded matrix PASSES, err=2.1e-16)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, CPU-only (login-node build; no GPU needed for this
  artifact).
- Toolchain: g++ 13.4.0 (conda-forge), cmake 4.2.3, Python 3.12.14, pybind11
  3.0.4; no CUDA involved (CPU build).
- Build: OK. Build-system changes: none (`PY="${PY:-/pscratch/.../python}"`
  already reads the exported `$PY` first, so it picked up this machine's
  kb-env Python without any edit).
- Gate: mandated command (`--matrices cant --warmup 1 --reps 3`) reconfirmed
  blocked the same way -- `adapter.py`'s `MAX_DENSE_ELEMENTS` guard raises
  immediately for a 62451x62451-shaped matrix ("3.900e+09 dense elements
  ... exceeds the MAX_DENSE_ELEMENTS=5e+07 safety cap"), reproduced directly
  against a same-shaped synthetic CSR matrix (bypassing the real `cant.mtx`
  load, which repeatedly stalled for minutes under this session's shared-
  login-node contention from concurrently running sibling reproduction
  agents -- an environment issue, not an artifact one; see cb-spmv's entry
  below for the same contention class). Substitute gate (500x500 synthetic
  banded matrix, `warmup=1 reps=1`, same code path as the CLI via
  `harness.run_variant` with `correctness_mode="max_scaled_err"` to match
  `sparse.py`'s `CORRECTNESS_MODE`): PASS, `max_scaled_err = 2.48e-16 <=
  1e-9` (recorded value 2.11e-16 -- same order of magnitude, fp64
  unit-roundoff on a freshly RNG-seeded synthetic matrix, no algorithmic
  difference).
- Deviation from the recorded ruling: none. Same blocker (the artifact's own
  API constraint, not a build/environment defect here), same clean PASS on
  the substitute check.
- Verdict here: BUILT (gate blocked, same reason) -- same as the recorded
  ruling.

## Baseline role (2026-09-05 selection-rule revision)

**Competitor, not a SOTA baseline.** Under the revised rule (core kernel papers
evaluated on the track's own input regime first; `kernel-papers/output/
baseline_selection.md`), this artifact would not have been selected:
- kernel centrality rated `component` (the spmv kernel is not this paper's headline, kernel-level contribution).
- evaluated regime does not match this track's inputs (diagonal-structured Hamiltonian matrices (HamLib suite), custom diagonal-sparse layout, not general CSR).
Rating rationale (`output/kernel_centrality.json`): HamSim builds a diagonal-sparse matvec kernel but the headline contribution and evaluation are the full Hamiltonian-simulation speedup vs Qiskit-Aer, not isolated y=Ax throughput vs SpMV libraries; the diagonal-only structure and quantum-sim workload (HamLib) differ fundamentally from SuiteSparse.
It stays in the registry and runs under the same gate as every other
implementation, but Phase 3 does not treat it as the human-SOTA reference for
`spmv`.
