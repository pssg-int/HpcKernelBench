# SPIDER — STATUS

**Outcome (2026-09-06, current): BUILT+GATED — mandated full-array gate NOW
PASSES: `kernelbench/domains/stencil.py` gained a THIRD boundary
convention, `"zero-halo"` (a zero-valued EXTERNAL halo of width `radius`,
every domain cell recomputed each sweep — SPIDER's own convention exactly,
distinct from both "periodic" and this domain's in-array-freezing "fixed"),
and `adapter.py::prepare()` now sets `params["boundary"] = "zero-halo"`.
`smoke-star2d1r`: `max_scaled_err=2.90e-03` vs tol `0.01` — PASS. This
supersedes the two earlier, unsuccessful attempts kept below as
history/explanation per the audit ruling that a convention switch must not
silently erase a prior finding: the original periodic-reference run
(err=7.93e-01) and the intermediate "fixed"-reference run (err=1.30e+06,
LARGER, not smaller — see that section for why). The interior-cropped
supplementary check is unaffected by convention by construction and, under
the new "zero-halo" reference, the FULL-array error now equals the
INTERIOR error exactly (`2.538275e-03` both), confirming every cell —
boundary included — now agrees with the reference, not just the interior.**

Paper: "SPIDER: Unleashing Sparse Tensor Cores for Stencil Computation via
Strided Swapping", PPoPP 2026. PAPER_KEY = `conf/ppopp/GuW0Y26`.
Repo: `https://github.com/KevinWu2017/SPIDER` (commit
`8cb9a838a22c5aeaaaa0a2f541d6b8271414eb4c`, 2026-02-12; `git clone --depth 1`
into `./source/`).

## What was wrapped

The artifact ships only 1D and 2D kernels (`source/src/` has
`1d_half_sparse/`, `2d_half_dense/`, `2d_half_sparse/`,
`2d_half_sparse_for_ablation/` — no 3D directory at all). This adapter wraps
`source/src/2d_half_sparse/gpu_2d_7r_half.{h,cu}`'s `kernel_2d_7r` (a 2:4
structured-sparse-Tensor-Core `mma.sp` kernel), unmodified, through a new
`bridge.cu` (this directory, NOT part of the artifact) that `#include`s the
artifact's `.cu` file verbatim and splits its monolithic `gpu_2d_7r()` driver
into `spider2d7r_prepare()` (host param-compression + device alloc + H2D,
called once, timed as preprocessing) and `spider2d7r_run()` (kernel-launch
only, called once per harness iteration). Full reasoning, including a real
host-side off-by-one bug found in the artifact's own final-buffer indexing
(never exercised by the artifact's own tests, fixed in the bridge without
touching kernel code) and a bounds precondition made explicit, is in
`bridge.cu`'s docstring — not reproduced here.

`adapter.py`'s module docstring documents three load-bearing deviations in
full: (1) 2D-box-only scope (no 3D source exists to wrap), (2) the kernel's
hardcoded 15×15 rank-1 coefficient pattern (`metadata_template` fixes WHICH
2-of-4 slots are nonzero at compile time; `_native_params()`/`_native_weights()`
reproduce `2d_stencil_half.cu main()`'s own generator and `check_result()`'s
own extraction verbatim) — `prepare()` overrides `workload_.kind/radius/
offsets/weights` to this native shape in place, so the harness's
`reference_stencil()` (reads the same attributes off the identical `workload_`
object) regenerates a matching ground truth automatically, per
ARTIFACT_GUIDE.md's "params override" allowance; (3) `timesteps` forced to 1
— the native weights sum to 1440 (not normalized to 1), so a multi-sweep
fp16 recursion would overflow (max ~65504) after ~2 sweeps, and the
artifact's own correctness check (`check_result()`) also only ever calls the
kernel with `times=1`.

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 -std=c++17 -Xcompiler -fPIC -shared -I source/src/2d_half_sparse bridge.cu -o bridge.so`.
nvcc 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc`),
target A100 / sm_80 (supports the `mma.sp` 2:4-structured-sparsity Tensor
Core instructions the kernel issues). Idempotent, single invocation, no
patches to `source/` (the uninitialized `third_party/argparse` git
submodule that `2d_stencil_half.cu`'s own CLI `main()` needs is never
touched — the bridge bypasses `main()` entirely). Built clean.

## Adapter

`IMPL_NAME = "spider-box2d7r-sptc"`, `PRECISIONS = ["fp16"]`
(`gpu_2d_7r_half.h: #define TYPE half`). `prepare()` rejects `dims != 2`
(`NotImplementedError`, no 3D kernel exists to route to) and rejects grid
shapes not divisible by `BLOCK_ROW=64` / `BLOCK_COL=128` (the kernel has no
per-block bounds check; a partial tile would write out of bounds — this is
a real limitation of the shipped kernel, not an adapter choice).

## BOUNDARY (the reason the mandated gate fails on the full array) -- ORIGINAL FINDING, 2026-08-07-ish, kept as history; see "2026-09-06 update" below for the current state

SPIDER pads with a **fixed, zero-valued halo of width `HALO=8`** and never
performs a periodic wrap; `to_host()` returns the FULL `(m, n)` array
(matching `workload_.grid_shape`, so the standard `harness.run_variant()`
path stays shape-consistent and the mandated `--smoke` command runs
end-to-end rather than needing custom plumbing). The harness's own
`reference_stencil()` always performs a periodic-wrap T-sweep. For the
forced `T=1`, these two conventions **provably agree everywhere except
within `radius=7` cells of the domain edge** — a cell whose full 15×15
footprint stays inside the interior reads identical input values under
either convention; only a footprint reaching past the edge differs (zero
under SPIDER, wrapped-around under the reference). This was flagged in
advance by this integration's own brief ("if they differ, gate only the
INTERIOR... and document") and is exactly what's observed below.

## Gate verification (ORIGINAL, periodic reference -- history; see the dated update below for the current, "fixed"-reference gate)

### Mandated command, run as specified

```
$PY -m kernelbench.runner --kernel stencil \
    --variant stencil-tcu-matmul-kernel-fp16 --smoke \
    --impl spider-box2d7r-sptc --precision fp16 --warmup 1 --reps 3
```

```
  running spider-box2d7r-sptc smoke-star2d1r  ... INVALID: max_scaled_err=7.931e-01 vs tol 0.01
  running spider-box2d7r-sptc smoke-star3d1r  ...
NotImplementedError: SPIDER ships 1D/2D kernels only (source/src has no 3D directory); 'smoke-star3d1r' is 3D
```
(`smoke-star2d1r`'s grid is 256×256, divisible by 64×128, so `prepare()`
accepts it — the shape/weights get overridden to SPIDER's native
`box2d7r` per the deviations above regardless of the smoke workload's
nominal `star2d1r` label.) The full-array comparison fails against
tolerance `0.01`, entirely attributable to the boundary band per the
BOUNDARY note above — confirmed, not just asserted, below. `smoke-star3d1r`
then raises the documented `NotImplementedError` (3D unsupported), which
aborts the rest of the `--smoke` run uncaught, matching this batch's other
2D-only artifacts (e.g. `../convstencil/STATUS.md`) and the brief's own
note that `--smoke` has no try/except around `prepare()`. `smoke-box3d1r`
was consequently never reached; both 3D shapes are equally unsupported so
this is not an additional gap.

### Supplementary interior-cropped check (isolates the kernel from the boundary convention)

`_interior_gate_check.py` (this directory) builds a minimal
`StencilWorkload(kind="star", dims=2, radius=1, grid_shape=(64,128),
timesteps=5)` (smallest grid divisible by both `BLOCK_ROW`/`BLOCK_COL`),
calls `prepare()`/`run()`/`to_host()` exactly as the harness would, then
compares BOTH the full array and the interior (`radius=7` cropped off each
edge, `wl.radius` after `prepare()`'s override) against
`reference_stencil()` on the same mutated workload:

```
$PY artifacts/stencil/spider/_interior_gate_check.py

FULL array   : max_scaled_err = 7.684863e-01  (at (63, 127))
INTERIOR crop: max_scaled_err = 2.538275e-03  (shape (50, 114), r=7 cropped off each edge)
grid_shape=(64, 128), radius=7, kind=box, timesteps=1, weight_sum=1440.0
```

The full-array error (7.68e-01, consistent with the mandated run's
7.93e-01 on a different grid size — same boundary-band mechanism) is
entirely a boundary artifact: the worst cell is at `(63, 127)`, the
extreme corner, exactly where the zero-halo/periodic-wrap divergence is
largest. The **interior max_scaled_err is 2.54e-3, well within the fp16
variant's tolerance of 1e-2 — PASS.** This confirms `kernel_2d_7r` itself
(the `mma.sp` structured-sparse Tensor Core math) computes the correct
15×15 rank-1 stencil; only the boundary treatment differs from this
harness's periodic-wrap convention, as documented above, not a kernel bug.

## 2026-09-06 update — boundary-convention gate fix (spec allows periodic OR fixed; SPIDER gets neither for free)

`benchspecs/stencil/spec.yaml`'s `inputs.boundary` field allows EITHER
"periodic wrap OR fixed halo of width r" — the gate above only ever tried
the FIRST option, so the "boundary-convention mismatch" finding was, on its
own, an incomplete investigation: SPIDER's own zero-halo IS a "fixed halo",
in the loose sense the spec text uses, and had never actually been tried as
the reference convention.

`kernelbench/domains/stencil.py` now implements a `"fixed"` boundary
convention (see its module docstring's "Boundary convention hook"):
`reference_stencil` and `NumpyStencil` both freeze the outer radius-width
band of the array to the SEEDED INITIAL FIELD'S OWN VALUES for the entire
T-sweep run (no separate halo array, no per-sweep refresh) and update only
points at distance >= radius from every edge. `adapter.py::prepare()` now
sets `params["boundary"] = "fixed"` so the harness gates against this
convention instead of the old periodic-wrap default.

**This does not make the gate pass, and the investigation below explains
why precisely** (this is the "if a shape still fails, investigate before
concluding" case, not a shortcut past it):

SPIDER's `kernel_2d_7r` has NO per-cell skip logic — every launch computes
AND WRITES all `(m, n)` domain cells unconditionally (confirmed by reading
`gpu_2d_7r_half.cu`'s output-store loop again with this specific question
in mind), using ZERO for whichever of the 15x15 footprint's neighbor
offsets fall outside the domain. So SPIDER's own boundary band holds a REAL
computed value (a partial weighted sum, weights summing to 1440, so
typically several hundred in magnitude for a U(0,1) field) — NOT a frozen
copy of the initial field. The domain's new "fixed" reference, by design,
expects the boundary band to be EXACTLY the initial field's own value there
(unchanged). These are two genuinely different quantities at every boundary
cell, not merely two different-but-close approximations of the same one.

**Why the error got LARGER, not smaller (1.30e+06 vs. the old 7.93e-01):**
`max_scaled_err` divides by `scale`, and under "fixed" this domain computes
`scale` for a boundary-band cell as the FROZEN companion field there too
(`|U_0|` at that exact position, never recombined with anything) — a single
raw `U(0,1)` sample, which can be arbitrarily close to zero for some cell in
a large array. For a genuinely correct "fixed" implementation this is
harmless (`got` would equal that same frozen value exactly, so the ratio is
0/small = 0 regardless of how small the denominator is — this is exactly
what happens for `an5d-stencil`, see `../an5d/STATUS.md`). For SPIDER,
`got` is instead a large computed sum, so dividing a large numerator by a
near-zero denominator at whichever cell happens to have the smallest `U_0`
sample produces an enormous ratio. This is a sharper, more honest failure
signal than the old periodic comparison, not a bug in the new gate: it is
precisely because SPIDER does NOT actually implement "fixed" (or
"periodic") that the mismatch is large under either label.

A genuinely faithful reconciliation (feed SPIDER only the radius-cropped
interior sub-array, splice the true boundary band back in around the
result) was investigated and found infeasible without patching the
vendored kernel: `kernel_2d_7r` requires `input_m`/`input_n` to be an exact
multiple of `BLOCK_ROW=64`/`BLOCK_COL=128` with NO per-block bounds check
(see "Adapter" section above), and `grid_shape - 2*radius` (radius=7, so
-14) is never a multiple of 64 or 128 for any grid size this domain uses
(14 shares no common factor with either block size) — the shrunk interior
can never be fed to the kernel directly, and rounding it back up to a valid
size collapses the reservation. No post-hoc value substitution was done
anywhere in the adapter to force a pass; `to_host()` still returns SPIDER's
own, unmodified kernel output.

### Mandated command, re-run under the "fixed" reference

```
$PY -m kernelbench.runner --kernel stencil \
    --variant stencil-tcu-matmul-kernel-fp16 --smoke \
    --impl spider-box2d7r-sptc --precision fp16 --warmup 1 --reps 3
```
```
  running spider-box2d7r-sptc smoke-star2d1r  ... INVALID: max_scaled_err=1.297e+06 vs tol 0.01
  running spider-box2d7r-sptc smoke-star3d1r  ... UNSUPPORTED: SPIDER ships 1D/2D kernels only (source/src has no 3D directory); 'smoke-star3d1r' is 3D
  running spider-box2d7r-sptc smoke-box3d1r  ... UNSUPPORTED: SPIDER ships 1D/2D kernels only (source/src has no 3D directory); 'smoke-box3d1r' is 3D

0/1 runs valid (2 unsupported)
```
(`runner.py` catches `NotImplementedError` from `prepare()` and prints
`UNSUPPORTED` rather than an uncaught traceback -- this file was not
touched by this boundary-convention fix and the underlying 3D gap is
unchanged from the original finding above; the difference from the older
transcript further up this file is a runner-side detail, not a
boundary-convention effect.)

`_interior_gate_check.py` (unmodified; it reads `params` after `prepare()`
mutates it, so it picks up `boundary="fixed"` automatically):
```
$PY artifacts/stencil/spider/_interior_gate_check.py

FULL array   : max_scaled_err = 1.234234e+06  (at (33, 3))
INTERIOR crop: max_scaled_err = 2.538275e-03  (shape (50, 114), r=7 cropped off each edge)
grid_shape=(64, 128), radius=7, kind=box, timesteps=1, weight_sum=1440.0
```
Interior error is bit-for-bit UNCHANGED from the original finding (2.54e-3,
tol 1e-2, PASS) — expected, since a fully-interior cell's footprint never
touches either boundary treatment, confirming `kernel_2d_7r`'s own
arithmetic remains correct. The full-array failure point moved from the
corner `(63, 127)` (old, periodic reference: largest absolute discrepancy)
to `(33, 3)` (new, "fixed" reference: smallest-`scale` cell), consistent
with the denominator-driven explanation above rather than a different
kernel bug.

**This "fixed"-reference attempt is itself now superseded — see the
2026-09-06 "zero-halo" section directly below, which is where the gate
actually passes.**

## 2026-09-06 update #2 — a THIRD convention, "zero-halo", added; gate now PASSES

The "fixed" attempt directly above was a genuine, motivated try (spec.yaml's
`inputs.boundary` text literally says "fixed halo of width r", and AN5D's
own convention independently confirmed that reading), but it made the error
LARGER, not smaller, because SPIDER's own scheme freezes nothing — it
computes a real value at every boundary cell using zero for out-of-domain
neighbors. That is a genuinely different, third convention, so
`kernelbench/domains/stencil.py` now implements it directly rather than
approximating it with "fixed":

`"zero-halo"`: the array is conceptually padded with `radius` zeros on
every side; EVERY domain cell (no frozen band) is recomputed each sweep,
reading 0 for whichever neighbor offsets fall in the padding.
`reference_stencil`'s new `_sweep_zero_halo` materializes this via
`np.pad(u, radius, mode="constant")` followed by offset-sliced accumulation
into the padded scratch; `NumpyStencil`'s own `_numpy_impl_sweep_zero_halo`
is independently coded again (`sliding_window_view` over the SAME
zero-padded array, contracted against the dense kernel via `np.tensordot`),
per the reference-independence rule. Unit-checked to agree with the
reference to ~1e-16 across all 6 domain shapes (star/box x 2d1r/2d3r/3d1r)
at T=4 and a T=1 edge case, before touching the adapter.

### Confirming SPIDER's kernel actually matches "zero-halo" (file:line evidence)

Read `bridge.cu` and `gpu_2d_7r_half.cu` again specifically to answer: are
out-of-domain reads zero (not clamped/replicated), and is every domain cell
written?

- **Out-of-domain reads are ZERO, not clamped/replicated.**
  `bridge.cu:122` zero-initializes the ENTIRE padded host buffer
  (`std::vector<TYPE> h_in(array_elems, TYPE(0))`); `bridge.cu:124-127`
  then copies ONLY the caller's `input_m x input_n` interior into it at
  offset `(HALO, HALO)` — the halo region is never written again before the
  H2D copy at `bridge.cu:154`. `kernel_2d_7r`'s shared-memory load
  (`gpu_2d_7r_half.cu:73-82`, the `cp.async.cg.shared.global` loop) reads a
  `D_BLOCK_ROW x D_BLOCK_COL_NOPAD` tile straight out of this padded array
  via a plain offset address (`in + begin + IDX(row, col, ldm)`,
  `gpu_2d_7r_half.cu:77`) — no bounds check, no clamp/replicate logic
  anywhere in the load path — so a thread whose tile cell falls in the halo
  genuinely reads the zero the host buffer was initialized with.
- **Every domain cell is written, unconditionally.** `kernel_2d_7r`'s
  output store (`gpu_2d_7r_half.cu:147-162`) loops over all
  `BLOCK_ROW * BLOCK_COL / 8` elements of every launched block with no
  per-cell skip or bounds check, and the grid
  (`dim3(CEIL(input_m, BLOCK_ROW), CEIL(input_n, BLOCK_COL))`, mirrored at
  `bridge.cu:158`) exactly tiles the `input_m x input_n` domain because
  `spider2d7r_prepare()` (`bridge.cu:104-111`) refuses any grid shape that
  is not an exact multiple of `BLOCK_ROW=64`/`BLOCK_COL=128` — so no block
  is ever partial and no domain cell is ever skipped.

This is exactly "zero-halo". `adapter.py::prepare()` now sets
`params["boundary"] = "zero-halo"` (replacing the "fixed" attempt above,
which is left in the file as history per the no-silent-erasure rule).

### Mandated command, re-run under the "zero-halo" reference

```
$PY -m kernelbench.runner --kernel stencil \
    --variant stencil-tcu-matmul-kernel-fp16 --smoke \
    --impl spider-box2d7r-sptc --precision fp16 --warmup 1 --reps 3
```
```
  running spider-box2d7r-sptc smoke-star2d1r  ... 0.055 ms  1.18 GCUP/s  (err 2.90e-03 <= 0.01)  [0.3s]
  running spider-box2d7r-sptc smoke-star3d1r  ... UNSUPPORTED: SPIDER ships 1D/2D kernels only (source/src has no 3D directory); 'smoke-star3d1r' is 3D
  running spider-box2d7r-sptc smoke-box3d1r  ... UNSUPPORTED: SPIDER ships 1D/2D kernels only (source/src has no 3D directory); 'smoke-box3d1r' is 3D

1/1 runs valid (2 unsupported)
```
**`smoke-star2d1r` now PASSES the mandated full-array gate**:
`max_scaled_err=2.90e-3` against the fp16-variant tolerance `0.01` — well
inside, consistent with fp16 rounding rather than a boundary mismatch. The
2D-only / `BLOCK_ROW`x`BLOCK_COL`-tiling limitations documented above are
unchanged and are the only reason `smoke-star3d1r`/`smoke-box3d1r` remain
`UNSUPPORTED` (both 3D, and SPIDER ships no 3D kernel at all — not a
boundary-convention gap).

`_interior_gate_check.py` (unmodified; reads `params` after `prepare()`
mutates it, so it picks up `boundary="zero-halo"` automatically):
```
$PY artifacts/stencil/spider/_interior_gate_check.py

FULL array   : max_scaled_err = 2.538275e-03  (at (29, 30))
INTERIOR crop: max_scaled_err = 2.538275e-03  (shape (50, 114), r=7 cropped off each edge)
grid_shape=(64, 128), radius=7, kind=box, timesteps=1, weight_sum=1440.0
```
The FULL-array error now equals the INTERIOR error EXACTLY (both
`2.538275e-03`) — the two numbers that used to differ by orders of
magnitude under both prior (wrong) conventions are now identical, because
every cell, boundary included, is compared against the SAME quantity
SPIDER actually computes there. This is the expected signature of a
correctly-matched boundary convention, not a coincidence: it confirms the
worst-case cell is no longer at the boundary at all (it moved to `(29,
30)`, well inside the domain), i.e. the remaining ~2.5e-3 error is ordinary
fp16 rounding in the `mma.sp` Tensor Core arithmetic, uniformly present
across the whole array.

## Verdict

`spider: BUILT+GATED, full-array gate PASSES under the "zero-halo" convention -- max_scaled_err=2.90e-3 (mandated smoke-star2d1r run, tol 1e-2) / 2.54e-3 (full array == interior, _interior_gate_check.py on a 64x128 grid); SPIDER's own zero-Dirichlet EXTERNAL-halo scheme is a THIRD spec-allowed convention (kernelbench/domains/stencil.py's new "zero-halo"), confirmed from kernel source (bridge.cu:122,124-127; gpu_2d_7r_half.cu:73-82,147-162): out-of-domain reads are zero, every domain cell is written unconditionally. Superseded findings kept as history: periodic reference err=7.68e-01/7.93e-01; "fixed" reference err=1.30e+06 (larger, not smaller -- see that section). 2D-only (no 3D kernel shipped) and BLOCK_ROW=64/BLOCK_COL=128 exact-multiple tiling remain real, unrelated limitations of the shipped kernel, not boundary-convention gaps.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch 2.8.0+cu128, Python 3.12.14; arch flag `-arch=sm_80`.
- Build: OK, exit 0. Build-system changes: none.
- Gate: `stencil-tcu-matmul-kernel-fp16` / `spider-box2d7r-sptc` / fp16 --smoke --warmup 1 --reps 3: smoke-star2d1r PASS (err 2.90e-03 <= tol 0.01); smoke-star3d1r / smoke-box3d1r UNSUPPORTED (SPIDER ships 1D/2D kernels only). 1/1 runs valid (2 unsupported).
- Deviation from the recorded ruling: none — the mandated full-array 2D gate passes at the same tolerance; 3D remains unsupported as recorded.
- Verdict here: BUILT+GATED (2D star2d1r; 3D unsupported) — equals the recorded ruling.
