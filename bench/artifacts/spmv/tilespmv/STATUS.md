# TileSpMV — STATUS

**Outcome: BUILT+GATED (partial) — clean pass on structured inputs; real
memory-corruption bug in the artifact's own preprocessing crashes on the
real SuiteSparse matrices tried, and a silent wrong-answer on high tile-
column-fanout patterns.**

Paper: "TileSpMV: A Tiled Algorithm for Sparse Matrix-Vector Multiplication
on GPUs", IPDPS 2021. PAPER_KEY = `conf/ipps/NiuLDJ0T21`. Selected as a
**core baseline under the revised kernel-centrality rule**
(`output/kernel_centrality.json` key `spmv|conf/ipps/NiuLDJ0T21`: centrality
`core`, regime `matches` — "general tiled GPU SpMV algorithm evaluated
across the full SuiteSparse collection").
Repo: `https://github.com/SuperScientificSoftwareLaboratory/TileSpMV`
(commit `d8bd7e41d0e527c64b0fffb261064e615143d675`, 2022-08-12;
`git clone --depth 50` into `./source/`).

## What was wrapped

The artifact ships only an end-to-end binary (`src/main.cu`) that loads a
`.mtx` file itself, overwrites its values with a synthetic `i%10` pattern,
and times a WARMUP_NUM(200) + BENCH_REPEAT(1000)x4-then-1000 loop inside one
function (`call_tilespmv_cuda`, `src/tilespmv_cuda.h`) wrapped in a single
`gettimeofday()` pair — exactly the "own binary reading .mtx and timing
internally" case this integration was told to bypass. `bridge.cu` (this
directory) splits it:

- **`tilespmv_prepare()`** (timed once, as preprocessing): calls the
  artifact's own `Tile_create` (`src/csr2tile.h` — the paper's tiled-format
  selection: CSR/COO/ELL/HYB/dense/dense-row/dense-col per 16x16 tile,
  unmodified) and `tilespmv_cpu` (`src/tilespmv_cpu.h` — despite its name
  this derives the row-balance bookkeeping `ptroffset1/2`,
  `blkcoostylerowidx*`, `rowblkblock` the GPU kernels need, unmodified),
  every device malloc/H2D copy `call_tilespmv_cuda` would do, and ONE
  untimed launch of `stir_spmv_cuda_kernel_v5` — see `bridge.cu`'s "why v5"
  section for why this is genuinely one-time (v5 builds a per-warp
  *structural* prefetch schedule, independent of x's numeric values, that
  every later `v6` call reads back).
- **`tilespmv_run()`** (timed per iteration): re-zeros `d_y`, launches
  `stir_spmv_cuda_kernel_v6` — the artifact's real steady-state kernel —
  exactly once.

## A real finding, corrected mid-integration: the vendored CSR5 path is dead weight, not a missing correction

`csr2tile.h`'s format selection tags any 16x16 tile with <=`COO_NNZ_TH`(12)
nonzeros as Format=1 ("very sparse" -> in-tile COO), handled by `v6`'s own
`case 1` branch. Separately, the SAME entries are ALSO copied
(`convert_step3`/`convert_step4`'s `case 1` block, `new_coocount`/
`offset_new`) into a second, matrix-wide "deferred COO" structure
(`matrix->deferredcoo_*`) meant to be processed by a vendored copy of
Weifeng Liu's CSR5 SpMV (`src/external/CSR5_cuda/`). The artifact's own
WARMUP and first "cache-warm" loops call `A.spmv(alpha, d_y)` after `v6`
(`tilespmv_cuda.h:1079-1081,1104-1105`) to add this — but since CSR5
accumulates via `d_y[...] +=`, and `v6` ALREADY counted those same entries
via its own Format=1 handling, this **double-counts**. An earlier version
of this bridge, reading only the artifact's SECOND (actually-measured) loop
— where the equivalent call is commented out
(`tilespmv_cuda.h:1131-1132`) — assumed that omission was a bug the
artifact's real reported numbers silently benefit from, and added the CSR5
call to every `run()`. **Direct measurement disproved this**: on a
synthetic 64x64 matrix engineered so every nonzero lands in its own
Format=1 tile (`coototal == nnz`), the CSR5-including bridge returned
`y = 2*y_ref` EXACTLY (verified element-by-element against NumPy). The
artifact's own measured/reported loop — the one that drops the CSR5 call —
is the CORRECT one: `v6` alone already accounts for every nonzero.
This bridge does **not** build or call CSR5 at all (confirmed: removing it
took the same 64x64 test from `max_scaled_err=1.0` to exactly `0.0`).
Full account left in `bridge.cu`'s "why NOT CSR5" section, including the
double-count evidence, since it directly documents an artifact behavior
future integrators of this codebase should not repeat.

## Build-system / compat fixes (none touch kernel arithmetic)

1. **Arch flag**: `-arch=compute_61 -code=sm_86` (artifact's own Makefile,
   an RTX-30-series dev box) -> `-arch=sm_80` for this machine's A100.
2. **Missing NVIDIA-CUDA-Samples headers**: `common_cuda.h` (vendored CSR5
   code, still `#include`d transitively by `tilespmv_cuda.h` even though
   this bridge never calls into it) does
   `#include <helper_functions.h>` / `<helper_cuda.h>`, assuming a
   `NVIDIA_CUDA_Samples` checkout the artifact's own Makefile points `-I`
   at (`/home/usr/NVIDIA_CUDA-11.1_Samples/common/inc`), absent here.
   Grepping the entire vendored tree: only ONE macro from these headers is
   actually used, `checkCudaErrors`. `compat/helper_cuda.h` (this
   directory, NOT the artifact) supplies just that macro with the same
   semantics; `compat/helper_functions.h` is an empty stub (nothing else is
   used). `build.sh` adds `compat/` to `-I`, ahead of `source/src` — no
   tracked file touched for this fix.
3. **Pre-Volta warp-shuffle intrinsics**: the vendored CSR5 code
   (`external/CSR5_cuda/detail/cuda/*.h`, written for CUDA ~8/9) calls
   `__shfl`/`__shfl_up`/`__shfl_down`/`__shfl_xor` with no mask argument;
   CUDA 12.9 no longer declares these names at all targeting sm_70+
   ("identifier '__shfl' is undefined"). `compat/shfl_compat.h` (this
   directory) provides templated shims forwarding to the `_sync` forms with
   a full-warp mask (`0xffffffffu`) — reproduces the old implicit-all-lanes
   semantics exactly, since every call site here launches full warps with
   no divergent early exit. No tracked file touched.
4. **One inline-PTX CUDA-version fix, in a TRACKED file** (recorded in
   `source.patch`): `external/CSR5_cuda/detail/cuda/utils_cuda.h`'s
   `scan_32_shfl(double)` hand-writes `shfl.up.b32` PTX directly (not
   through the C intrinsic, so fix #3 above doesn't cover it); ptxas
   targeting sm_70+ rejects pre-`.sync` `shfl` PTX outright. Added
   `.sync` + a full-warp membermask operand (`0xffffffff`) to both
   `shfl.up.b32` instructions — same arithmetic, newer PTX syntax, no
   value/algorithm change. This is the one tracked-file diff in
   `source.patch`.
5. **Missing `return` in two vendored CSR5 functions — a genuine
   undefined-behavior bug, not a build-system knob, but recorded the same
   way since it involves zero arithmetic/algorithm change**: initially the
   bridge DID build and call the CSR5 handle (before the finding above
   showed that was wrong), and hit a real hang: `format_cuda.h`'s
   `format_warmup()` and `anonymouslib_cuda.h`'s
   `anonymouslibHandle::warmup()` are both declared to return `int` but had
   no `return` on any path (UB in C++). Under nvcc 12.9 `-O3` this
   manifested as control never reaching the caller after
   `format_warmup()`'s own last statement completed (confirmed by
   instrumenting every intermediate call: the final
   `checkCudaErrors(cudaFree(d_scan))` inside `format_warmup` prints its
   success and then the process never proceeds). Added
   `return ANONYMOUSLIB_SUCCESS;` / `return format_warmup();` — the
   convention every OTHER method in the same class already uses. This fix
   is now moot (CSR5 is never called at all, per the finding above) but is
   left in `source.patch` since it's harmless, correct, and documents a
   real bug hit along the way. No arithmetic changed.

No kernel code was modified anywhere in either patch.

## Build

```
./build.sh
```
`nvcc -O3 -w -arch=sm_80 -Xcompiler -fopenmp -Xcompiler -fPIC -I compat -I source/src -shared bridge.cu -o bridge.so`.
nvcc 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`). Idempotent.

## Adapter

`IMPL_NAME = "tilespmv"`, `PRECISIONS = ["fp64"]` (artifact's `MAT_VAL_TYPE`,
`common.h`, is `double`). `prepare()` sorts the harness's CSR (`sort_indices`,
required — Tile_create's per-tile scan expects sorted column indices within
each row, matching `main.cu`'s own usage pattern), generates `x` via
`np.random.default_rng(seed).uniform(-1,1,size=cols)` exactly matching
`cpu_ref.reference_spmv`'s RNG (numpy `default_rng`, NOT `torch.rand` —
the documented RNG-mismatch trap), then calls `tilespmv_prepare`. `run()` is
one `tilespmv_run` call (stream=0, matching `CudaEventTimer`'s recording
stream). `to_host()` D2H-copies `y`.

## Gate verification

### Small controlled correctness check (bypassing the harness, to isolate the CSR5 finding above)
64x64 synthetic matrix, one nonzero per row at a random unique column (forces
`coototal == nnz`, i.e. every entry routed through Format=1/in-tile-COO):
against a NumPy `A @ x` reference, `max_scaled_err = 0.0` (bit-exact) with
CSR5 excluded (this bridge's final form); `max_scaled_err = 1.0`
(`y = 2*y_ref` exactly) with CSR5 included (the earlier, wrong form) —
direct proof of the double-counting finding above.

### Harness smoke (`--smoke`, synthetic 4000x4000, reduced protocol)
```
$PY -m kernelbench.runner --kernel spmv --variant spmv-csr-kernel --impl tilespmv --smoke
```
```
  running tilespmv smoke-uniform  ...  INVALID: max_scaled_err=8.350e-01 vs tol 1e-09
  running tilespmv smoke-banded   ...  0.013 ms  14.83 GFLOP/s  (err 3.20e-16 <= 1e-09)
  running tilespmv smoke-powerlaw ...  INVALID: max_scaled_err=1.000e+00 vs tol 1e-09
1/3 runs valid
```
`smoke-banded` (nonzeros clustered near the diagonal — few active
tile-columns per tile-row) passes at fp64 unit-roundoff. `smoke-uniform`/
`smoke-powerlaw` (nonzeros spread across the full column range — MANY
active tile-columns per tile-row, forcing `tilespmv_cpu`'s row-balance
splitter, `tilespmv_cpu.h`'s `balancenumblk > PREFETCH_SMEM_TH(4)` branch,
to split many tile-rows across multiple `rowblkblock` entries) FAIL with
large errors. This is NOT the CSR5 issue (already excluded) and NOT a
gate-tolerance issue — it traces to the "split tile-row" (`signbit=1`,
`blkcoostylerowidx_colstart/colstop`) code path inside
`stir_spmv_cuda_kernel_v5`/`_v6` themselves, which is genuine artifact
kernel logic this integration does not have license to patch (rule 3).

### Real SuiteSparse matrices (mandated command)
```
$PY -m kernelbench.runner --kernel spmv --variant spmv-csr-kernel --impl tilespmv --matrices webbase-1M --warmup 1 --reps 3
$PY -m kernelbench.runner --kernel spmv --variant spmv-csr-kernel --impl tilespmv --matrices cant --warmup 1 --reps 3
```
Both **crash** during `tilespmv_free()`/next allocation with heap-corruption
errors (`corrupted double-linked list (not small)` on `webbase-1M`;
`free(): invalid pointer` on `cant`, and with `OMP_NUM_THREADS=1` forced
(ruling out an OpenMP race) `cant` instead crashes deterministically inside
glibc's allocator itself: `Fatal glibc error: malloc.c:4369 (_int_malloc):
assertion failed: (unsigned long)(size) >= (unsigned long)(nb)` — this is a
**buffer-overflow inside Tile_create's own format-conversion arrays**
(`src/csr2tile.h`'s `convert_step1-4`, unmodified), not a threading race,
not a bridge/adapter bug (the same `Tile_create`/`tilespmv_cpu` call
sequence, on the SAME small smoke matrices and my isolated 64x64 test,
completes and produces bit-exact output). Both `cant` and `webbase-1M` are
real, ordinary CSR matrices — nothing pathological about either as inputs;
the artifact's own preprocessing arrays are undersized for matrices at this
scale on this build (CUDA 12.9 / glibc on this machine expose it; the
artifact's own 2021-era testing on an older toolkit evidently did not
catch it, or used a build where the corruption stayed silent).

**Per rule 4 ("failures are results; never loosen a gate") and rule 3 (no
kernel-code patches), this is reported as-is rather than patched.** The
bridge and adapter are verified mechanically correct (bit-exact on every
case that doesn't hit the artifact's own bugs); the artifact's own
`Tile_create`/`tilespmv_cpu`/`stir_spmv_cuda_kernel_v5/_v6` genuinely cannot
be gated successfully on this track's real-matrix subset as published.

## Verdict

`tilespmv: BUILT+GATED (partial) — exact on structured/banded inputs
(err=3.2e-16); FAILS (wrong answer) on wide-column-fanout synthetic inputs;
CRASHES (heap corruption in the artifact's own Tile_create) on both real
SuiteSparse matrices tried (webbase-1M, cant). No timing number obtained.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80` (unchanged); `source.patch` (the inline-PTX
  `.sync` fix + the two missing-`return` fixes) re-applied automatically by
  `toolchain.sh` when `source/` was recreated.
- Build: OK. Build-system changes: none.
- Gate: smoke: 1/3 valid -- smoke-uniform INVALID (max_scaled_err=8.350e-01),
  smoke-banded PASS (err=3.20e-16 <= 1e-9), smoke-powerlaw INVALID
  (max_scaled_err=1.000e+00) -- identical numbers to the recorded ones.
  `--matrices webbase-1M --warmup 1 --reps 3`: CRASH, `corrupted
  double-linked list (not small)` (glibc abort, exit 134) -- matches the
  recorded webbase-1M crash signature exactly. `--matrices cant --warmup 1
  --reps 3`: CRASH, `malloc.c:4106: _int_malloc: Assertion '(unsigned long)
  (size) >= (unsigned long) (nb)' failed.` (glibc abort, exit 134) -- same
  assertion the recorded ruling documented for `cant` under
  `OMP_NUM_THREADS=1` (`malloc.c:4369`, this machine's glibc: `malloc.c:4106`
  -- different line number, same assertion, same buffer-overflow-in-
  Tile_create root cause per the recorded analysis, not a new finding).
- Deviation from the recorded ruling: none -- same partial pass/fail/crash
  pattern, same error magnitudes.
- Verdict here: BUILT+GATED (partial) -- same as the recorded ruling.
