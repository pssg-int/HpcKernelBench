# RASSM — sddmm — STATUS: BUILT+GATED

- Paper: RASSM: Residue-based Acceleration of Single Sparse Matrix
  Computation via Adaptive Tiling (ASPLOS'25). `PAPER_KEY = conf/asplos/JainGC25`
- Repo: https://github.com/gt-tinker/RASSM, commit `3224b4466e9e40b15cb94826c3c062963c87a6dd`
  (2024-12-02). `source/` is a symlink to `../../spmm/rassm/source` — the
  sibling agent integrating this same repo for the spmm track cloned it
  first; we reuse that checkout rather than cloning a second copy, per the
  task instructions.
- Build: `g++ (SUSE Linux) 14.3.0`, `-std=c++17 -O2 -fopenmp`, no CUDA
  involved (CPU artifact). No cmake/Boost needed for our path (see below).
- GPU/arch: n/a (CPU).

## What "wrapping the kernel" required here

`code/src/main.cpp` exposes `--kernel sddmm --type RASSM`, which is supposed
to build the paper's own Adaptive Tile Matrix (ATM) via its Residue-based
tile generator and run `sddmm_kstream` over it. Reading
`code/include/experiments.h::data_movement_experiment_sddmm()` shows this
whole path — the `S_atm` construction AND the `sddmm_kstream` call — is
guarded by `#ifdef RUN_CSR_ATM_KSTREAM`. Grepping the entire `code/include`
tree confirms that macro is **never `#define`d anywhere**, in `config.h` or
elsewhere. So the shipped `rassm` binary silently no-ops for
`--kernel sddmm` regardless of `--type` — there is no way to exercise
RASSM's own SDDMM contribution through its CLI as shipped. (This matches
what the earlier benchspecs/sddmm/spec.yaml survey pass flagged as an open
question: "unclear whether RASSM treats SDDMM as a first-class contribution
... its own abstract frames the headline speedup entirely in SpMM terms.")

Rather than edit `config.h`/`experiments.h` to flip that macro (patching the
artifact's own driver files), `wrapper.cpp` (new file, NOT part of the
artifact, lives next to this STATUS.md) calls the exact same sequence of
**unmodified** artifact functions that `#ifdef` block would have called:

1. `Residue::adaptive_2d_greedy_Ti_greedy_Tj_tile_generator` — the paper's
   own residue-based adaptive tile generator (`code/include/Residue.h`,
   untouched).
2. The `ATM` constructor — the paper's own Adaptive Tile Matrix format
   (`code/include/matrices/ATM.h`, untouched).
3. `sddmm_kstream` — the paper's own kernel
   (`code/include/sddmm/tiled.h`, untouched), called once per `run()`.

Two small **build-flag** (not source-patch) workarounds, documented in
`build.sh`:
- `util.cpp`/`global.cpp` (unmodified artifact `.cpp`s) are linked in
  alongside `wrapper.cpp` because they define `compare1`/`compare2` (used by
  `CSR`/`CSC`/`ATM`'s internal `std::qsort` calls) and the extern globals
  (`CACHE_NUM_WAYS`, ...) `Residue.h`'s tile generator reads.
  `wrapper.cpp::rassm_sddmm_prepare()` sets `CACHE_NUM_WAYS = 8` before
  calling the tile generator, matching the CLI's own `--numways` default
  (main.cpp) since nothing else initializes that global outside the CLI's
  own arg-parsing path.
- `-include string`: `config.h` uses `std::string`/`std::map<std::string,...>`
  without including `<string>` itself; every artifact `.cpp` that happens to
  pull it in transitively (main.cpp, via boost headers) never notices, but
  `global.cpp` compiled standalone does. Force-including `<string>` is a
  compiler flag, not a source edit.

## The reordering problem (and how it's handled)

`ATM`'s constructor reorders nonzeros into its own `row_ptr`/`cols`/`vals`
arrays, organized by (panel, column-tile) for cache locality — NOT
necessarily the caller's original CSR nnz order. `sddmm_kstream` writes its
output `O[j]` indexed by that same internal order `j`, not by the input
matrix's `(row_idx[i], col_idx[i])` position `i`.

`wrapper.cpp::rassm_sddmm_export_layout()` exports the ATM's own
`row_ptr`/`cols` arrays. `adapter.py::prepare()` uses them to compute, ONCE
per matrix (a structural, value-independent step — the same spirit as
prepare()'s format-conversion role), a permutation `perm` such that
`O[perm]` reproduces the caller's original nnz order — via a sort+searchsorted
match on `(row,col)` keys, with a hard `RuntimeError` if any key fails to
match exactly (never silently misaligning the gate). `to_host()` applies
`perm` before the harness's correctness check runs.

## Precision

RASSM's `TYPE` (`config.h`) is a compile-time `using TYPE = double;` — there
is no fp32 compute path in the artifact at all. `adapter.py` services both
`fp32` and `fp64` precision *labels* by rounding the dense A/B operands with
the same recipe `kernelbench.impls.cpu_ref.reference_sddmm` uses for its
fp32 variant (round to float32, then widen to float64) before handing them
to the (always-double) kernel — so the requested-precision label controls
input rounding, matching the reference's own inputs bit-for-bit, while the
kernel itself always computes in double.

## Verification (login node, correctness gate only, reduced protocol)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
$PY -m kernelbench.runner --kernel sddmm --variant sddmm-csr-kernel-f32 \
    --impl rassm-adaptive-tiled-sddmm --matrices cant --dims 128 \
    --warmup 1 --reps 3
```

Result: **1/1 valid**, `max_scaled_err = 4.41e-16` (tolerance `1e-4`) —
essentially machine-epsilon, as expected for a double-precision kernel
compared against a double-precision reference. Not spec-conforming (login
node, reduced warmup/reps) — timing numbers from this run must not be
published, only the correctness result.

## Provenance

- Artifact commit: `3224b4466e9e40b15cb94826c3c062963c87a6dd`
- Host compiler: g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`)
- Python: `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python` (numpy via its
  own deps only — no torch needed for this CPU artifact)


## Baseline role (2026-09-05 selection-rule revision)

**Competitor, not a SOTA baseline.** Under the revised rule (core kernel papers
evaluated on the track's own input regime first; `kernel-papers/output/
baseline_selection.md`), this artifact would not have been selected:
- kernel centrality rated `component` (the sddmm kernel is not this paper's headline, kernel-level contribution).
- no single-NVIDIA-GPU kernel path (current scope).
Rating rationale (`output/kernel_centrality.json`): RASSM's own abstract frames its headline speedup entirely in SpMM terms with SDDMM as a secondary code path (--kernel sddmm exists but is SpMM-adjacent per the spec's open_questions), so SDDMM is real but not the paper's headline; its SuiteSparse selection matches the general-suite variant's input tradition closely.
It stays in the registry and runs under the same gate as every other
implementation, but Phase 3 does not treat it as the human-SOTA reference for
`sddmm`.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan; CPU-only artifact, gate run via `bench/gpu_run.sh`
  (an otherwise-idle A100 job on `gpu-b11-6`, GPU itself unused) rather than
  the shared login node — see `../../spmm/rassm/STATUS.md`'s "Machine
  gotchas" section for why (the login node was under sustained
  `RLIMIT_NPROC`/swap pressure from concurrent sibling agents that made an
  equivalent CPU-only gate hang for close to an hour there; running the
  identical command on an `srun`-isolated compute node instead completed in
  a few seconds). Login-node build.
- Toolchain: g++ 13.4.0 (conda-forge; this directory's `build.sh` reads
  plain `${CXX:-g++}`, no gcc-12 pin needed here unlike the spmm/rassm
  wrapper), no cmake/Boost involved (this build never touches
  `source/code/CMakeLists.txt`). Python 3.12.14, numpy only.
- Build: OK. Build-system changes: none — this directory's `build.sh`
  already builds via a single direct `g++ -shared` invocation with no
  `-j`-parallel `make`/`cmake` step, so it was never exposed to the
  RLIMIT_NPROC-from-`-j$(nproc)` gotcha the sibling `spmm/rassm/build.sh`
  needed fixing for.
- Gate: `sddmm-csr-kernel-f32`, `cant` dim=128: **1/1 valid**, `max_scaled_err
  = 4.41e-16 <= tol 1e-4` — matches the "Verification" section above
  bit-for-bit.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
