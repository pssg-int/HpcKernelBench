# TileSpGEMM -- STATUS

**Outcome: BUILT+GATED (partial) -- builds cleanly, wraps the artifact's own
kernel at the finest available boundary, and the mechanical wrapping
(dimensions, nnz counts, symbolic tile-column pattern) is verified correct
via a synthetic controlled test; but the artifact's own numeric SpGEMM
kernel produces WRONG VALUES on a purely-positive synthetic matrix and a
genuine INDEXING BUG (nonzero columns outside the structurally guaranteed
`|A|@|A|` superset) on two real SuiteSparse matrices. Fails the exact-match
correctness gate; recorded here as a real finding, not patched.**

Paper: "TileSpGEMM: A Tiled Algorithm for Parallel Sparse General
Matrix-Matrix Multiplication on GPUs" (PPoPP'22). `PAPER_KEY =
conf/ppopp/NiuLJS0022`. Selected as a **core baseline under the revised
kernel-centrality rule** (`output/kernel_centrality.json` key
`spgemm|conf/ppopp/NiuLJS0022`: centrality `core`, regime `matches` --
"SuiteSparse matrices (A^2, general A x B), fp64, GPU"; see
`output/baseline_selection.md`'s spgemm section, "one of the deeply-surveyed
papers that directly defines the spec's square-kernel variant").
Repo: `https://github.com/SuperScientificSoftwareLaboratory/TileSpGEMM`
(commit `fe3a3457cec078fddd73c04f4ffed14edee7fb21`, 2022-11-10;
`git clone --depth 1` into `./source/`).

Sibling lesson consulted before starting (per task instructions):
`../../spmv/tilespmv/STATUS.md` (same lab/codebase family) -- primed the
expectation of CUDA-12 compat fixes and a real memory/numeric bug in the
artifact's own preprocessing/kernel code. Both materialized here too, in a
different form (a tile-column-indexing bug rather than heap corruption).

## What was wrapped

The artifact ships only an end-to-end binary (`source/src/main.cu`) that
parses a `.mtx` file itself (`mmio_allinone`), overwrites A's values with a
synthetic `i % 10` pattern (`main.cu:100-101`), builds the tiled format, and
calls `tilespgemm()` (`source/src/tilespgemm-cuda.h`) -- the artifact's real
kernel function, invoked inside an `#ifdef DEBUG` block that
`source/src/common.h` already enables **unconditionally**
(`#define DEBUG 1`, `TIMING`/`SPACE`/`CHECK_RESULT` likewise, no `-D` flag
needed -- confirmed this is the artifact's actual default/measured path, not
a debug-only stub; `REPEAT_NUM` is also fixed at `1` in `common.h:69`,
matching `benchspecs/spgemm/spec.yaml`'s own `notes_on_fairness`: "The
artifact ships with REPEAT_NUM=1").

`tsg_bridge.cu` (this directory, **NOT** part of the artifact) calls
`tilespgemm()` directly from our own CSR arrays -- no file I/O, no synthetic
value overwrite (real workload fp64 values used throughout, required for a
meaningful correctness gate):

- **`tsg_prepare(n, nnz, rowptr, colidx, vals)`** -- builds the artifact's
  own `SMatrix` A directly from our CSR (host malloc + memcpy), aliases B to
  the SAME host arrays (verified safe by reading `csr2tile_row_major`/
  `csr2tile_col_major`'s bodies: both only READ `rowpointer`/`columnindex`/
  `value`, matching main.cu's own `-aat 0` self-product code path exactly),
  then calls the artifact's own `csr2tile_row_major(A)`/
  `csr2tile_col_major(B)` (`source/src/csr2tile.h`, unmodified) -- this IS
  the CSR->tile conversion `spgemm-square-kernel-f64`'s spec excludes from
  the timed window (rule 2), plus `main.cu`'s own (untimed, host-side)
  intersection-bitmask construction and `nnzCub` count, reproduced verbatim
  since `tilespgemm()`'s signature requires both.
- **`tsg_run(handle)`** -- ONE call to `tilespgemm()` (unmodified). See
  "Disclosed timing contamination" below.
- **`tsg_finalize_csr`/`tsg_to_host`** -- `tile2csr()` (`source/src/
  tile2csr.h`, unmodified), host-only, converts the tiled C back to plain
  CSR; called once from Python `to_host()`, never inside the timed `run()`
  path.

Zero lines of `source/` were modified.

## Build-system / compat fixes (none touch kernel arithmetic)

1. **Arch retarget**: artifact's own Makefile targets an RTX-30xx dev box
   (`-arch=compute_61 -code=sm_86`, CUDA 11.4) -> `-arch=sm_80` for this
   machine's A100, nvcc 12.9.
2. **`__shfl` without `_sync`**: `source/src/spgemm_nsparse_kernel.h:858-859`
   call the pre-Volta unmasked `__shfl()` on `int`/`double` inside the
   nsparse hash-bin kernel path (only reachable for matrices with more than
   `NUMCOLC_SPA_OR_HASH_TH` = 16384 tile-columns, i.e. > ~262144 rows/cols --
   NOT exercised by any matrix actually gated below, all of which take the
   SPA-only `step1` path instead). CUDA 12.9 no longer declares unmasked
   `__shfl` at all targeting sm_70+. `tsg_bridge.cu` defines a templated
   shim forwarding to `__shfl_sync` with a full-warp mask (0xffffffffu),
   same fix already used by `../../spmv/tilespmv/compat/shfl_compat.h`.
3. **Vendored `cuda_fp16.h`/`cuda_fp16.hpp` conflict with CUDA 12.9's
   libcu++**: `source/src/common.h:15` does a plain quoted
   `#include "cuda_fp16.h"`, resolving to the artifact's own vendored
   ~CUDA-11-era copy (no `__nv_bfloat16` interop). Under nvcc 12.9, `thrust`
   (pulled in by `utils_cuda_scan.h`) transitively needs the CURRENT
   `cuda_fp16.h`/`cuda_bf16.h` pair (`cuda/std/__cmath/traits.h`,
   `nvfp16.h`) and fails against the vendored one (`__half::__half` ctor
   mismatch; `__hisnan`/`__hisinf`/`__habs`/`__hmax` "device function called
   from host device function" errors). Both the vendored and the system
   header use the identical include guard (`__CUDA_FP16_H__`, unchanged
   since CUDA 11) -- `tsg_bridge.cu` includes the SYSTEM `<cuda_fp16.h>`/
   `<cuda_bf16.h>` first, making the artifact's later quoted include a
   guarded no-op. This codebase's fp64 SpGEMM path never uses half
   precision at all -- the vendored header is dead weight for this build
   regardless.

No tracked file in `source/` needed a line-level patch (unlike TileSpMV's
one inline-PTX fix); `source.patch` is empty/absent.

## Build

```
./build.sh
```
`nvcc -std=c++14 -O3 -w -arch=sm_80 --expt-relaxed-constexpr
-Xcompiler=-fopenmp,-fPIC -shared tsg_bridge.cu -o tsg_bridge.so -lgomp`.
Idempotent. `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required at runtime
(CXXABI/GLIBCXX version mismatch, same as every other artifact in this repo).

## Disclosed timing contamination (per-call, not patched -- rule 3)

`tilespgemm()` mallocs+H2D-copies matrixA/matrixB's ALREADY-TILED host
arrays into fresh device buffers AND D2H-copies the tiled C result back, ALL
INSIDE the same call it also does the symbolic+numeric computation in, then
frees every device buffer it allocated (verified by reading the function's
own ~600-line body and its tail, which `cudaFree`s every `d_*` buffer for A,
B, and C before returning) -- no separate, already-idempotent entry point
exists to do the H2D setup once and the compute repeatedly. This adapter's
measured per-call time would therefore be a disclosed UPPER BOUND on
`spgemm-square-kernel-f64`'s `timing_scope` (A/B's persistent tile-structure
H2D transfer is supposed to be excluded, "operands resident on device before
timing starts") -- same posture as `../ocean/adapter.py`'s Workspace-
construction disclosure. Moot for this artifact since no valid timing number
was obtained (see below).

## Real bug found in the artifact's own numeric kernel

### Synthetic controlled test (bypassing the harness, isolating the finding)

20x20 random matrix, density 0.3, all-positive values `U(0.1, 1.0)` (seed
0) -- purely positive values, so `C = A@A`'s true nonzero pattern IS exactly
`|A|@|A|`'s pattern (no cancellation possible), making this the simplest
possible ground truth check against an independent `scipy.sparse` product:

```
ref (scipy) nnz: 339
tsg (TileSpGEMM) nnz: 335
structural pattern mismatch: 4 positions present in ref, absent in C (0 extra positions in C)
values wrong at 69 of the 335 shared (row,col) positions (max abs diff 2.11,
several off by 100%+, not a rounding-level discrepancy)
```

The **symbolic** (nonzero-position) computation is very nearly right (335/339
= 98.8% of positions, zero false positives) -- most of the bug lives in the
**numeric** (value-accumulation) step. Ruled out as causes, each verified
directly:

1. **Not a compiler-optimization artifact**: identical wrong output at
   `-O3` (build.sh's own flags) and `-O0`.
2. **Not an OpenMP race**: identical wrong output with `OMP_NUM_THREADS=1`
   and with the bridge rebuilt entirely WITHOUT `-fopenmp` (pragmas become
   no-ops, `csr2tile.h`/`tile2csr.h`'s loops run strictly sequentially).
3. **Not this bridge's A/B aliasing**: `csr2tile_row_major`/
   `csr2tile_col_major`'s full bodies were read line-by-line; both only
   READ `rowpointer`/`columnindex`/`value` (the fields aliased between A and
   B) and write exclusively into each `SMatrix`'s own `tile_*` fields --
   confirmed no cross-mutation is possible.
4. **Deterministic**, not a device race: bit-identical wrong output across
   every one of the above rebuild/env configurations, run repeatedly.
5. **Not a raw-CLI-vs-bridge artifact**: a from-scratch build of the
   artifact's *original, completely unmodified* `main.cu`/Makefile (arch
   retargeted to sm_80 only, same compat shims) was attempted for an
   independent check via its own internal cuSPARSE cross-check
   (`spgemm_cu.h`/`CHECK_RESULT`); it segfaults immediately on this login
   node (inside glibc's `printf`, confirmed via `gdb bt`, before any
   program output) on both the 20x20 test matrix and `cant.mtx`, for
   reasons unrelated to TileSpGEMM's own logic (this login node's shared/
   module environment, not reproducible via the ctypes/Python path this
   integration actually uses -- see ARTIFACT_GUIDE.md rule 5's login-node
   caveat). This dead-end was abandoned in favor of the working ctypes path
   above once it was clear the crash predated `main()`'s first `printf`.

### Real SuiteSparse matrices: a genuine indexing bug (not just wrong values)

Mandated gate commands:
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel spgemm --variant spgemm-square-kernel-f64 --impl tilespgemm \
    --matrices cant --warmup 1 --reps 2
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel spgemm --variant spgemm-square-kernel-f64 --impl tilespgemm \
    --matrices pdb1HYS --warmup 1 --reps 1
```
Both **raise inside the gate itself** (`kernelbench/impls/cpu_ref.py`'s
`_reindex_to_pattern`, shared with every other spgemm adapter -- same gate
`../ocean/adapter.py` and `scipy-csr-spgemm` pass cleanly):
```
cant:     row 0 has 53 nonzero(s) at column(s) [252, 253, 254, 255, 259]... outside |A|@|A|'s structural pattern
pdb1HYS:  row 21 has 3 nonzero(s) at column(s) [1008, 5136, 5168] outside |A|@|A|'s structural pattern
```
`|A|@|A|` (all-nonnegative operands, per `cpu_ref._canonical_pattern`'s own
docstring) is a **mathematically guaranteed superset** of any correct
`A@A`'s true output support -- a value appearing OUTSIDE it is not a
formatting or tolerance issue, it is a real indexing bug in the producing
implementation. On `cant`, 252-255 are the last four columns of tile-column
15 (`BLOCK_SIZE=16`, tile 15 = columns 240-255) and 259 is the fourth column
of the NEXT tile-column (16, columns 256-271) -- the same class of
tile-column-boundary confusion visible in the 20x20 synthetic test above
(that test's own 4 missing/wrong-adjacent positions clustered at columns 16
and 19, exactly its own tile-1 boundary, `BLOCK_SIZE=16, n=20`). This is
consistent with a bug in the numeric kernel's per-row **tile-column offset
bookkeeping** (which of the row's several output tiles a given accumulated
value/column gets written into) rather than in the symbolic tile-count
computation, which is the part `step1` gets almost entirely right.

Per ARTIFACT_GUIDE.md rule 4 ("failures are results; never loosen a gate")
and rule 3 (no kernel-code patches): reported as-is. The bridge and
build are verified mechanically correct (dimensions, nnz counts, and the
tile-conversion/H2D/D2H plumbing all behave as the artifact's own code
dictates); the artifact's own `tilespgemm()` numeric kernel genuinely
produces structurally invalid output on real matrices under this build.

## Gate verification

### Harness smoke (`--smoke`, synthetic 4000x4000, reduced protocol)
```
$PY -m kernelbench.runner --kernel spgemm --variant spgemm-square-kernel-f64 --impl tilespgemm --smoke
```
```
  running tilespgemm smoke-uniform   ... INVALID: max_scaled_err=9.969e+03 vs tol 1e-06
  running tilespgemm smoke-banded    ... INVALID: max_scaled_err=8.152e+02 vs tol 1e-06
  running tilespgemm smoke-powerlaw  ... INVALID: max_scaled_err=2.397e+04 vs tol 1e-06
0/3 runs valid
```

### Real SuiteSparse matrices (mandated command)
Both `cant` and `pdb1HYS` raise inside the gate's `_reindex_to_pattern`
(out-of-superset column indices -- see above), rather than producing an
in-range-but-wrong-valued result; treated as INVALID/failed runs, same as
the smoke set. No timing number was obtained for any input.

## Not done

- No sweep across the spec's `recommended_subset` beyond `cant`/`pdb1HYS` --
  both already demonstrate the same class of failure; a broader sweep would
  not change the verdict, and this integration is login-node/gate-check
  budgeted (ARTIFACT_GUIDE.md rule 5).
- `spgemm-e2e-preproc-f64`/`spgemm-tensorcore-mixed-precision`/
  `spgemm-distributed-strong-scaling-f64` variants not exercised (the
  kernel-only variant already fails the gate; TileSpGEMM has no tensor-core
  path and no distributed mode regardless).
- No attempt was made to locate or fix the numeric kernel's tile-column
  bookkeeping bug itself -- that would be a kernel-code change, forbidden by
  ARTIFACT_GUIDE.md rule 3.

## Verdict

`tilespgemm: BUILT+GATED (partial) -- clean build (2 build-system/compat
fixes, 0 tracked-file patches), bridge/wrapping verified mechanically
correct (dims, nnz, near-perfect symbolic tile-column pattern: 335/339 on a
20x20 positive-only control), but the artifact's own tilespgemm() numeric
kernel produces WRONG VALUES on synthetic input and a genuine
tile-column-indexing bug (output positions outside the mathematically
guaranteed |A|@|A| superset) on two independent real SuiteSparse matrices
(cant, pdb1HYS). 0/3 smoke runs valid; both real-matrix gate attempts raise
inside the correctness gate itself. No timing number obtained. Confounders
(compiler optimization level, OpenMP threading, this bridge's own A/B
aliasing) were individually ruled out.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch 2.8.0+cu128, Python 3.12.14; arch `-arch=sm_80`.
- Build: OK, exit 0. Build-system changes: none.
- Gate: `spgemm-square-kernel-f64` / `tilespgemm` / --smoke: 0/3 runs valid (max_scaled_err 2.881e+04 vs tol 1e-6 on smoke-uniform).
- Deviation from the recorded ruling: none — reproduces the recorded finding exactly: TileSpGEMM produces wrong values on a purely-positive synthetic matrix (the exact-match gate fails), documented as a real artifact finding, not patched. The kernel builds and runs; its numeric output is simply wrong for this regime.
- Verdict here: BUILT+GATED (partial; exact-match gate FAILS on synthetic, as recorded) — equals the recorded ruling.
