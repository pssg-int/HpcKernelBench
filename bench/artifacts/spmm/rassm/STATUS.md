# rassm (RASSM) — spmm

**Status: BUILT+GATED** (on the task's required verification matrix; one
edge-case caveat found and documented below, not on the gating run)

- Paper: "RASSM: Residue-based Acceleration of Single Sparse Matrix
  Computation via Adaptive Tiling", ASPLOS'25. `PAPER_KEY = conf/asplos/JainGC25`
  (matched by title in `../../output/included.json`).
- Artifact: https://github.com/gt-tinker/RASSM (CPU; implements SpMM and
  SDDMM with its own residue-based adaptive tiling preprocessing, per the
  task brief).
- Commit cloned: `3224b4466e9e40b15cb94826c3c062963c87a6dd` (2024-12-02),
  `git clone --depth 50`.
- Toolchain: `g++-12` (SUSE 12.3.0), `-march=core-avx2 -fopenmp -O3`,
  cmake 3.28.3, system Boost 1.66.0 (`libboost_headers1_66_0-devel`,
  in-range of the README's stated 1.65-1.74). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.

## Contamination note (ARTIFACT_GUIDE.md rule 1) — read this first

RASSM ships **no library**, only an end-to-end CLI binary (`rassm`) that:
reads a matrix from an `.mtx` file it opens itself; generates its dense
operand internally via `generate_dense()` (`(i % 100) / 100.0` — a fixed
deterministic pattern, not something a correctness gate's fp64 reference
could be matched against); runs its own internal warmup/timing loop; and
prints only aggregate `"Median Time"`/`"GFLOPS"` text. The one
correctness-check code path (`RUN_CHECK` macro in `experiments.h`) only ever
prints a boolean and is compiled out by default (`#ifdef
RUN_CORRECTNESS_CHECK`, never defined by the CMakeLists.txt) — there is no
way to get the actual output matrix `C` out of the executable, and no way to
hand it a caller-chosen `B`.

So this adapter wraps at the finest available boundary instead:
`wrapper.cpp` (this directory, **not** under `source/`) calls RASSM's own
header-only preprocessing pipeline — `CSR` → `CSC` → `Residue` →
`adaptive_2d_greedy_Ti_greedy_Tj_tile_generator` → `ATM` — and its k-stream
SpMM kernel (`spmm_atm_kstream_compiler_vectorized`) **directly**, bypassing
`main.cpp`'s CLI parsing / `.mtx` file I/O / timing loop / `generate_dense`
entirely. Every function called is the artifact's own unmodified code,
copied from nowhere — see `wrapper.cpp`'s header comment and the "Provenance
of each call" note below. This preprocessing pipeline construction (Residue
matrix + adaptive greedy panel generation + `ATM`) **is RASSM's own tiling
preprocessing** (ARTIFACT_GUIDE.md rule 2 / the task brief's own hint) —
it lives in `adapter.py`'s `prepare()`, timed as preprocessing.

## Build (rule 3: minimal patches, all recorded; no kernel code touched)

`git -C source diff --stat`: `CMakeLists.txt | 13 +-`, `config.h | 5 +`,
`experiments.h | 21 +-`.

1. **`config.h`**: added `#include <string>`. `std::map<std::string,...>`
   a few lines down no longer compiles under gcc-12's libstdc++ (`'string'
   is not a member of 'std'`) — newer libstdc++ stopped transitively
   pulling `<string>` in via `<map>`. Header-only include fix.
2. **`experiments.h`**: the `runtype::INTEL_MKL` branch (a comparison
   *baseline*, vendor Intel MKL — not any part of RASSM's own kernel, which
   is the `runtype::RASSM` branch a few lines above, untouched) uses MKL
   types (`sparse_matrix_t`, `MKL_INT`, ...) unconditionally, even though
   this same file already gates *including* `mkl_spblas.h` on `#ifdef
   INTEL_COMPILER` a few lines up — so building with GCC (no Intel compiler,
   no MKL) always fails here regardless of which `run_mode` is selected at
   runtime. Mirrored the existing include-guard onto this branch's body
   (`#ifdef INTEL_COMPILER ... #else print_error_exit(...) #endif`) — the
   MKL codepath itself is untouched when compiled with `INTEL_COMPILER`,
   and RASSM's own kernel is not on this branch at all.
3. **`CMakeLists.txt`**: the GNU (`CMAKE_CXX_COMPILER_ID STREQUAL "GNU"`)
   branch was missing `-march=core-avx2` — the non-GNU/non-Intel fallback
   branch a few lines below already has it, and the Intel branch has its
   ICC equivalent, so this reads as an oversight specific to the GNU
   branch. Without it, GCC refuses to inline the raw AVX2/FMA intrinsics
   `config.h`'s non-Intel codepath uses unconditionally on x86_64
   (`_mm256_fmadd_pd`, ...) with `"target specific option mismatch"`.
   Added the same flag the artifact's own other branches already use.
4. **Not built**: the `aspt` target (`source/code/aspt/SpMM_ASpT_DP.cc`,
   `#include "mkl.h"`) — an ASpT-baseline comparison executable, unrelated
   to RASSM's own kernel, requires Intel MKL headers not installed here.
   `build.sh` only builds the `rassm` target.

`wrapper.cpp` itself needed one more thing, found by tracing a SIGFPE
under `gdb` (see next section) rather than a source patch:
`CACHE_NUM_WAYS` (`extern ITYPE CACHE_NUM_WAYS;` in `config.h`, defined
zero-initialized in `global.cpp`) is **only ever assigned inside
`main.cpp`**, from CLI option `--numways` (`default_value(8)`). Bypassing
`main.cpp` entirely (as this wrapper does) leaves it at its zero default,
and `Residue.h`'s `adaptive_2d_greedy_Ti_greedy_Tj_tile_generator` divides
by it unguarded (`max_output_cache_volume = (cache_size*cache_split) /
CACHE_NUM_WAYS`) — an integer divide-by-zero, `SIGFPE`, on the very first
call. `wrapper.cpp`'s `rassm_prepare()` sets it to the artifact's own
documented CLI default (8) before calling in — not a behavior change, just
supplying the initialization `main.cpp` would have done. Also linked
`source/code/src/{common,global,util}.cpp` (RASSM's own non-template
translation units — `CSC.h`/`DCSC.h` call `compare2()`, only *defined* in
`util.cpp`) alongside `wrapper.cpp`; `main.cpp` itself is excluded (defines
`main()`, would conflict).

## adapter.py

- `KERNEL = "spmm"`, `IMPL_NAME = "rassm-spmm-kstream"`,
  **`PRECISIONS = ["fp64"]`** — RASSM's kernel is hardcoded to
  `TYPE = double` in `config.h`, not a template parameter the CLI (or this
  wrapper) can vary; gating needs `--precision fp64`.
- `prepare()`: `wrapper.cpp`'s `rassm_prepare()` — RASSM's own
  `CSR`/`CSC`/`Residue`/`adaptive_2d_greedy_Ti_greedy_Tj_tile_generator`/
  `ATM` pipeline, called with the artifact's own CLI defaults
  (`Ri=Rj=64`, `targCache=DEFAULT_LLC=1MiB`, `cache-split=4`,
  `oi-aware=true`, `temporal-input/output=false`, `resolution=1` — from
  `main.cpp`'s `boost::program_options` defaults / `scripts/run-rassm.sh`).
  `B` is generated with `numpy.random.default_rng` matching
  `cpu_ref.reference_spmm`'s `_dense_operand` exactly (see
  `insum/STATUS.md`'s docstring for why this matters — RASSM's own
  `generate_dense()` is a fixed non-random pattern, unusable as a
  gate-matching operand, so it's bypassed entirely and `wrapper.cpp`'s
  `rassm_run()` is handed our own `B` buffer directly).
- `run()`: ONE `spmm_atm_kstream_compiler_vectorized` call via
  `rassm_run()`. `Ti`/`Tj` parameters to that function are dead (confirmed
  at build time: `-Wunused-parameter` fires on both — the `ATM` object
  already carries its own per-panel tiling), passed as 0. `O` accumulates
  (`+=`), zeroed by the adapter before each call for a clean `C = A@B`.
  Row-major indexing confirmed by reading `spmm/kstream.h`
  (`O_data[i*(feature+PADDING_C)+k]`, `I_data[col*(feature+PADDING_B)+k]`,
  `PADDING_B=PADDING_C=0` in `config.h`) — matches this track's `B: K×N`
  / `C: M×N` row-major convention with no transpose needed either side.
- `timer()`: plain CPU wall-clock (`kernelbench.harness.Timer`), same as
  `cpu_ref.ScipySpMM`.

## Gate verification (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
$PY -m kernelbench.runner --kernel spmm --variant spmm-cpu-kernel-f32 \
    --impl rassm-spmm-kstream --precision fp64 --matrices cant --dims 128 \
    --warmup 1 --reps 3
```

Result: **1/1 runs valid.** `max_rel_err = 7.92e-16 <= tol 1e-6` — at fp64
machine-epsilon level, effectively exact. Reproduced 3x (cant/dims=128 twice
across separate invocations, `err=7.92e-16` and re-run consistent; plus
cant/dims=32 `err=7.57e-16` and pdb1HYS/dims=128 `err=1.29e-15`) — all
clean, all real SuiteSparse matrices from the spec's own
`recommended_subset`. `54-97 ms`, reduced-protocol numbers only
(warmup=1-2, reps=2-3, shared login-node GPU/CPU — `conforming: False`,
correctly marked non-conforming by the runner); not a timing result, per
ARTIFACT_GUIDE.md rule 5.

## Finding: synthetic-matrix edge case (not on the gating matrix, disclosed anyway)

While building this adapter, `harness`'s own tiny synthetic smoke matrix
(`kernelbench.matrices.synthetic("smoke-uniform", rows=512, cols=512,
nnz_per_row=8)`, 4096 nnz) at `N=32` gave `max_scaled_err≈5.6` — clearly
wrong (not a rounding-level discrepancy), reproduced deterministically
across repeated runs. Traced further with a 4×4 hand-built example
(`{(0,0):1,(1,1):2,(2,2):3,(3,3):4,(0,3):5}`): the artifact's **own**
internal self-check (`verify_matrix_structure()`, `Reader.h`, invoked the
same way `main.cpp` does for its `atm_correct` debug print) reported a
structural mismatch — `(row 0, col 3)` and `(row 3, col 3)` values swapped
(`E:5 G:4` / `E:4 G:5`) — and the kernel's actual output reproduced exactly
that swap. On the 512×512 case the same self-check reported the ATM
structure *correct*, yet the kernel output was still wrong — i.e. two
distinct symptoms, at least one a genuine bug in RASSM's own `ATM`
construction (`matrices/ATM.h`), triggered by small/synthetic matrices;
not something introduced by this wrapper (only unmodified artifact
functions are called, in the same sequence `main.cpp` uses). Per
ARTIFACT_GUIDE.md rule 3 ("touching kernel code" disqualifies a minimal
patch), this was **not patched** — RASSM's own `RUN_CHECK`/
`RUN_CORRECTNESS_CHECK` machinery being dead code by default is itself
suggestive that this path may not have been exercised much upstream either.
Recorded here as a genuine finding, not hidden, per rule 4 — but it does not
change the gate result above, which used the task's specified real matrix
(`cant`) and the spec's `recommended_subset` matrices, all of which passed
cleanly and reproducibly.

## Not done

- No sweep across matrices/dims/`Ri`/`Rj`/`targCache` — out of scope per
  the task's login-node budget.
- SDDMM (RASSM implements both, per the task brief) — only the `spmm` track
  entry point (`spmm_atm_kstream_compiler_vectorized`) was wired; SDDMM
  would need a separate adapter under `bench/artifacts/sddmm/rassm/`.
- Root-causing the synthetic-matrix `ATM` construction bug above — would
  require modifying `matrices/ATM.h` itself (kernel-adjacent data-structure
  code), out of scope for a "wrap, don't fix" integration per rule 3.


## Baseline role (2026-09-05 selection-rule revision)

**Competitor, not a SOTA baseline.** Under the revised rule (core kernel papers
evaluated on the track's own input regime first; `kernel-papers/output/
baseline_selection.md`), this artifact would not have been selected:
- no single-NVIDIA-GPU kernel path (current scope).
Rating rationale (`output/kernel_centrality.json`): RASSM's residue-based adaptive tiling is evaluated with dedicated SpMM-specific GFLOPS/median-time numbers against MKL/ASpT/J-Stream/CSF-4; spec's CPU variant (RASSM's own median-of-50 protocol) is explicitly modeled on this paper.
It stays in the registry and runs under the same gate as every other
implementation, but Phase 3 does not treat it as the human-SOTA reference for
`spmm`.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan; CPU-only artifact, no GPU used, but the actual gate
  run was executed through `bench/gpu_run.sh` (an otherwise-idle A100 MIG
  slice/full A100 job on `gpu-b11-6`) rather than the shared login node —
  see the machine-gotchas paragraph below for why. Login-node build.
- Toolchain: g++ 12.4.0 (conda-forge, `$KB_HOST_COMPILER_BIN`), cmake 4.2.3,
  `-march=core-avx2 -fopenmp -O3`. Python 3.12.14, numpy (no torch needed).
  **No system or conda Boost is installed on this machine at all** (checked:
  `ldconfig -p`, `/usr/include`, `/usr/lib64`, and all three conda envs —
  zero hits), unlike Perlmutter's system `libboost_headers1_66_0-devel`.
- Build: OK. Build-system change:
  `bench/artifacts/spmm/rassm/build.sh` — RASSM's `CMakeLists.txt` only
  defines the `rassm`/`aspt` executable targets inside an
  `if(Boost_FOUND)` guard; with no Boost available the CMake configure
  step succeeds but the Makefile it generates simply has no `rassm` target,
  so the unconditional `make -C source/build rassm` failed with
  "No rule to make target 'rassm'". Traced Boost's actual use: it is
  needed ONLY by `main.cpp`'s CLI (`program_options`/`iostreams`/
  `serialization`); `wrapper.cpp` and the artifact `.cpp`s it links
  (`common.cpp`/`global.cpp`/`util.cpp`) reference no Boost header at all —
  confirmed by grep. Step 1 (building the full `rassm` CLI binary, which
  this integration never calls — see "Contamination note" above, it's a
  provenance/build-viability check only) now detects a missing `rassm`
  Makefile target and skips itself with a clear NOTE instead of failing the
  whole build; step 2 (`librassm_wrapper.so`, the actual functional-gate
  target) is unaffected and unchanged. Also capped step 1's `make -j` at
  `${RASSM_BUILD_JOBS:-8}` (was unconditional `-j"$(nproc)"` = 128, which
  exhausts this shared login node's `RLIMIT_NPROC=256` — the same
  documented resource-contention gotcha as `fastkron`/`bootcmatchgx`, not
  an artifact issue). Neither change touches Boost-present behavior
  (Perlmutter): there step 1 still builds normally.
- Gate: `spmm-cpu-kernel-f32`, `cant` dim=32: **1/1 valid**, `max_rel_err =
  7.57e-16 <= tol 1e-6`. `cant` dim=128: **1/1 valid**, `err = 7.92e-16`.
  `pdb1HYS` dim=128: **1/1 valid**, `err = 1.29e-15`. All three numbers
  match the Perlmutter-recorded values in this file's "Gate verification"
  section bit-for-bit.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.

### Machine gotchas hit while reproducing (recorded here in detail; referenced from the other four STATUS.md files in this batch)

1. **Shared login-node contention made this specific gate hang for ~1 hour
   before it was diagnosed and worked around** — not an artifact bug, but
   worth recording precisely since it looked at first like one. The login
   node was shared with several concurrent sibling agents each rebuilding/
   gating other artifacts, driving `RLIMIT_NPROC` (256 processes/threads for
   the whole user session) and node-wide memory into sustained exhaustion
   (`free -h` showed 4.0/4.0 GiB swap fully used for an extended period).
   Under that pressure, `import scipy.sparse.linalg` (pulled in by
   `kernelbench.impls.cpu_ref`, needed even to just resolve `--impl`) would
   itself hang or crash: OpenBLAS's thread-pool init inside that import
   calls `pthread_create`, which fails with `EAGAIN` under `RLIMIT_NPROC`
   pressure and can leave the interpreter stuck for the run's entire
   lifetime with **zero** CPU usage (confirmed via `ps -o pcpu` on the
   stuck PID: 0.4-1.6% over 15+ minutes) rather than erroring — a swap-
   thrashing/thread-starvation deadlock symptom, reproduced independently
   with a bare `--smoke` run that hit the identical
   `OpenBLAS blas_thread_init: pthread_create failed ... RLIMIT_NPROC 256`
   message before a `timeout` wrapper killed it. Fix: `OPENBLAS_NUM_THREADS=1
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` avoids the extra `pthread_create`
   calls entirely (single-threaded BLAS needs none beyond the main thread).
   Even with that fix, the same run on the **shared login node** still
   barely progressed (0.4-1.1% CPU for 14+ minutes on the same `cant`
   matrix) — genuine swap-thrashing from other users'/agents' concurrent
   memory pressure, not something a thread-count env var fixes. Moving the
   *identical* command to an `srun`-allocated compute node via
   `bench/gpu_run.sh` (CPU-only workload, GPU unused — just to get an
   isolated node) made it complete in **1.5-2.5 seconds** per matrix/dim,
   confirming the login node itself, not RASSM or this integration, was the
   bottleneck. Recommendation for any future CPU-only gate on a
   heavily-shared login node: run it through `gpu_run.sh` (or an
   equivalent CPU-partition `srun`) rather than directly on the login node,
   and always set `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` around any
   Python process that imports scipy/numpy on a `RLIMIT_NPROC`-capped
   machine, even for a single-threaded workload.
2. **`bench/gpu_run.sh` compute nodes have no outbound internet.** A
   real-matrix gate that needs a SuiteSparse download must have the
   `.mtx`/`.tar.gz` already cached under the matrix-cache directory before
   the `srun` job starts (fetch it from the login node first, e.g. via
   `kernelbench.matrices.fetch_suitesparse(name)`); otherwise
   `urllib.request.urlretrieve` inside the compute-node process blocks for
   several minutes before raising `TimeoutError: [Errno 110] Connection
   timed out` (observed directly on `pdb1HYS` — see also `flashsparse`
   STATUS.md files' `cora` download reproducing the analogous
   `aiohttp.client_exceptions.ConnectionTimeoutError` for
   `torch_geometric`'s Planetoid loader). A **race** is also possible: if
   the compute-node process has *already entered* `urlretrieve` before a
   concurrent login-node prefetch finishes, the compute-node attempt is
   still doomed (it already committed to the offline path) even though the
   file exists moments later — a plain retry of the same command, now that
   the file is cached, succeeds immediately (`0.37s` CSR construction,
   `err=1.29e-15` — exactly what happened for `pdb1HYS` here).
3. **`bench/env.sh`'s `KERNELBENCH_MATRIX_CACHE` default is a latent
   portability bug, independent of (1)/(2) above and worth fixing
   centrally** (not fixed here — out of this task's per-directory scope,
   `env.sh` is shared infrastructure): `export
   KERNELBENCH_MATRIX_CACHE="${KERNELBENCH_MATRIX_CACHE:-}"` sets the
   variable to an explicit EMPTY STRING in the environment (not unset).
   `kernelbench/matrices.py` does `os.environ.get("KERNELBENCH_MATRIX_CACHE",
   <default path>)`, and `os.environ.get` returns `""` (not the default)
   for a key that is present-but-empty — so `CACHE` resolves to
   `os.path.normpath("")` = `"."`, i.e. the process's CURRENT WORKING
   DIRECTORY, not the intended `bench/matrices/`. In practice this means
   every downloaded `.mtx`/`.tar.gz`/`ssstats.csv` lands directly in
   `bench/` (confirmed: `cant.mtx`, `pdb1HYS.mtx`, `ssstats.csv`, etc. were
   all found sitting in `bench/` itself, not `bench/matrices/`, both from
   this session and an earlier one) rather than the dedicated cache
   directory `bench/README.md`/`ENVIRONMENT.md` describe. Harmless
   functionally (caching still works, files just end up in the wrong,
   messier place and clutter `git status` if `bench/*.mtx`/`*.tar.gz`
   aren't already covered by `.gitignore` — they are, checked) but worth a
   one-line fix in `bench/env.sh` (`${KERNELBENCH_MATRIX_CACHE:+...}` or
   simply not exporting the var at all when empty) so a fresh clone's first
   real-matrix gate caches where the docs say it will.
