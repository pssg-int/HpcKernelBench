# AN5D — STATUS

**Outcome (2026-09-06, current): Generator toolchain BUILD-FAILED (documented
below) — FALLBACK: BUILT+GATED by wrapping the artifact's own PRE-GENERATED
CUDA for all 6 of this domain's shapes. Mandated full-array `--smoke` gate
now PASSES on all 3 smoke shapes (max_scaled_err ~1e-8, tol 1e-5) after
gating against the spec's OTHER allowed boundary convention ("fixed halo of
width r", `params["boundary"]="fixed"`) instead of periodic wrap -- AN5D's
own generated code was ALREADY implementing "fixed" natively (its own
`common.h::init_grid` fills the array once; the generated sweep's loop
bounds never touch the outer radius-width band again), so once the harness
gates against the SAME convention the artifact actually computes, the
full-array and interior-cropped numbers converge to the same ~1e-8 figure.
See the dated section below for what changed and why; the original
periodic-reference finding (full-array FAILS at err 0.31-0.38, interior
PASSES at ~1e-8) is kept below as history.**

Paper: "AN5D: Automated Stencil Framework for High-Degree Temporal Blocking
on GPUs", CGO 2020. `PAPER_KEY = conf/cgo/MatsumuraZWEM20`.
Artifact repo: `https://github.com/khaki3/AN5D-Artifact` (commit
`fbebf5f2bb2fa1aae2bfc8fd975947a2ff791752`, 2021-09-15; `git clone --depth 1`
into `./source/`, untouched — read-only, no patch to tracked files).

## 1. Generator toolchain — BUILD-FAILED (exact blocker)

AN5D itself (`https://github.com/khaki3/AN5D`, separate from the artifact
repo above) is a PPCG/isl-based source-to-source generator: given a plain C
stencil with `#pragma scop`/`#pragma endscop` and `--bt/--bs1/--bs2/--sl`
tuning flags, it emits the CUDA that is the paper's actual contribution.
Investigated (clone deleted after investigation; commit hashes below for
reproducibility — not kept in this checkout, see `source.provenance`):

- `AN5D`'s `.gitmodules` pulls in `isl` (`git://repo.or.cz/isl.git`) and a
  fork of `pet` (`https://github.com/khaki3/pet`, commit
  `03facd30b683b294789d2ddf148f84c4d8138db9`, 2019-04-10) — `pet` is PPCG's
  C-source frontend, built directly against Clang's C++ LibTooling AST API.
- `pet`'s own `README` states its requirement explicitly: **"LLVM/clang
  libraries, 2.9 or higher... it's best to install the latest release
  (3.8)... development versions from before the latest release are not
  supported."** — i.e. even the *pet upstream* only ever supported clang up
  through the ~3.8 era (2016), and explicitly disclaims anything newer as
  untested/unsupported, before accounting for this fork's own additional
  2019 pin.
- This machine's only compiler-toolchain modules are `module spider llvm` ->
  `llvm/{18.1.0, 20.1.3, 21.1.4, 22.1.2}` (no `clang` module exists at all —
  `module spider clang` returns "Unable to find: clang"; confirmed no
  `clang`/`clang++` binary on `PATH` via `command -v`). That is a ~15-major-
  version gap from pet's own supported ceiling. Clang's internal C++ AST API
  (the `RecursiveASTVisitor`/`Stmt`/`Decl`/`Attr`/`TypeLoc` classes pet's
  `.cc` files construct scops from) has changed signature repeatedly across
  that span — porting `pet` across it is source-porting work on a 2019
  research-quality codebase, not a build flag or a version pin, and is
  explicitly out of scope per ARTIFACT_GUIDE.md rule 3 (no patching
  kernel/tool internals to make a build succeed).
- No sudo / no system package installs (`ENVIRONMENT.md` §5.8); building an
  LLVM/Clang release from source in the 3.x-8.x range compatible with `pet`
  would itself be a multi-hour-plus from-scratch toolchain build, separate
  from and larger than AN5D's own `isl`+`ppcg`+`pet` build — well outside
  this integration's ~2h budget for the generator path.

**Verdict: BUILD-FAILED.** Blocker: `pet` (AN5D's PPCG C-frontend
dependency) requires a clang/LLVM release in the ~2.9-3.8 range (per pet's
own README); only llvm/{18,20,21,22} are available via `module spider llvm`
on this machine, no `clang` module or binary exists at all, and there is no
sudo path to install an old one. This is the documented fallback case the
integration brief anticipated ("if the generator toolchain cannot be built
in ~2h... check whether the repo ships pre-generated CUDA sources").

## 2. Fallback: wrapping AN5D-Artifact's own pre-generated CUDA

`source/README.md`'s own "Tuned" evaluation path documents exactly this:
`source/compiled/{double,float}/` ships **438 pairs** of
`<shape>-<bS1>-<bT>-<sl>_{host.cu,kernel.cu,kernel.hu}` files per precision —
the actual CUDA AN5D's generator produced for its own benchmark sweep,
checked into the artifact repo (`compiled/nvcc_compile_top5.sh` compiles
these directly with plain `nvcc`, no generator involved, to reproduce the
paper's own "Tuned" table). These ARE the paper's own generated kernels
(produced once upstream, not by us) — wrapping them is the artifact's own
sanctioned evaluation path, not a workaround.

Crucially, `compiled/double/` covers **all 6** of this domain's shapes
natively, in AN5D's own naming (`kernelbench/domains/stencil.py`'s shape
parser adopts this convention verbatim): `star2d1r`, `box2d1r`, `star2d3r`,
`box2d3r`, `star3d1r`, `box3d1r` — unlike the spider/convstencil/lorastencil
siblings (each 2D-only, 1-2 shapes). `box3d1r.c`'s own coefficients contain
no division (`source/box3d1r.c`); `j3d27pt.c` is the same 27-point 3x3x3-box
shape but WITH a final `/159` — matching `stencil.py`'s own comment ("AN5D's
own named 27-point 3D kernel is a division-free variant of box3d1r") and its
`ALIASES = {"j3d27pt": "box3d1r"}`. Since `load_workload("j3d27pt")` already
resolves to `kind="box", dims=3, radius=1` before this adapter ever sees the
workload, no separate handling was needed — `box3d1r.c`'s own kernel is used
uniformly for that shape/radius/kind combination.

### 2.1 Generator flow replaced by mechanical extraction (`gen_bridge.py`)

`gen_bridge.py` (this directory) does NOT hand-transcribe or reimplement
AN5D's generated dispatch logic. For each of 6 chosen tuning configs (below),
it:

1. Locates the enclosing `{ ... }` compound statement in the artifact's own
   `<shape>-<tuning>_host.cu` (every AN5D-generated file has the identical
   skeleton: `cudaMalloc` -> H2D `cudaMemcpy` -> a self-contained
   `{ #ifndef AN5D_TYPE ... }` block containing ONLY the temporal-blocking
   kernel-launch cascade -> `cudaCheckKernel()` -> D2H `cudaMemcpy` ->
   `cudaFree`) via brace-matching, and extracts that block + the following
   `cudaCheckKernel();` verbatim.
2. Asserts the extracted block's primary `__side0Len` (temporal blocking
   degree `bt`) matches what this script expects for that shape — catches a
   wrong file being picked.
3. Splices the extracted text (unmodified byte-for-byte) into a small
   `extern "C"` shim (`bridge_<shape>.cu`) exposing separate
   `an5d_<shape>_{prepare,run,copy_out,free}` entry points — the same
   split-the-monolithic-driver pattern the spider/convstencil siblings' own
   hand-written bridges use, here done mechanically because AN5D's own
   `kernel_stencil()` bundles device malloc + H2D + dispatch + D2H + free
   into ONE function call, but this harness times H2D (prepare) and kernel
   launches (run) separately.
4. `#include`s the matching `<tuning>_kernel.cu` (the artifact's own
   `__global__` kernel definitions, entirely unmodified) directly above.

`build.sh` runs `gen_bridge.py` then compiles each `bridge_<shape>.cu` with
`-Xcompiler -fvisibility=hidden` (only the 4 `an5d_<shape>_*` entry points
are exported `default` visibility) into its own `bridge_<shape>.so` — needed
because all six shapes' pre-generated files independently define
`kernel0_1`, `kernel0_2`, ... at file scope; without hidden visibility,
loading more than one shape's `.so` into the same process risks the dynamic
linker resolving one shape's kernel launch to a DIFFERENTLY-SHAPED kernel of
the same name in another already-loaded `.so` (silently wrong results, not a
link error).

### 2.2 Tuning configs chosen (smallest available temporal-blocking degree `bt`, for the shortest/simplest remainder cascade)

| shape | config used | `bt` (`__side0Len`) | other configs available (not used) |
|---|---|---|---|
| star2d1r | `star2d1r-256-10-128` | 10 | 512-{10,13,16}-{128,256,512} |
| box2d1r  | `box2d1r-512-8-128`   | 8  | 256-10-{128,256,512}, 512-16-{128,256,512} |
| star2d3r | `star2d3r-512-4-128`  | 4  | 512-{6,7}-{128,256,512} |
| box2d3r  | `box2d3r-128-2-128`   | 2  | {256,512}-2-128, 512-{6,7}-{256,512} |
| star3d1r | `star3d1r-32x32-4-128`| 4  | 32x32-{5,6,7}-*, 32x32 variants |
| box3d1r  | `box3d1r-32x32-3-128` | 3  | 32x16-3-*, 32x32-4-128, 64x16-3-* |

All 6 verified against every grid size this integration actually gates
(64/24 generic sizes, and the exact `--smoke` sizes 256/32/24 — see §4) via
a standalone ctypes probe before wiring into `adapter.py`.

## 3. Load-bearing deviations (all documented in `adapter.py`'s module docstring too)

1. **Native coefficients, not injectable weights.** Unlike convstencil/
   lorastencil (which pass a runtime weight array), AN5D's generated CUDA has
   the coefficients baked in as C `float` literals directly inside the
   polyhedral-scheduled update statement — there is no params pointer to
   inject this domain's own synthetic weights into. `adapter.py::prepare()`
   overrides `workload_.offsets`/`weights` in place to AN5D's OWN
   coefficients — same "params override" pattern as spider/lorastencil, per
   ARTIFACT_GUIDE.md — but here the coefficients are **parsed directly from
   AN5D's own `source/<shape>.c` reference file by regex**
   (`_parse_native_weights`, matches on `#pragma scop`..`#pragma endscop`),
   not hand-transcribed. Verification (before wiring into `adapter.py`):

   ```
   star2d1r.c: n_terms=5  (expected 5)   sum=1.00000  center=0.25
   box2d1r.c:  n_terms=9  (expected 9)   sum=0.99997  center=0.25001
   star2d3r.c: n_terms=13 (expected 13)  sum=0.99984  center=0.25002
   box2d3r.c:  n_terms=49 (expected 49)  sum=1.00000  center=0.25424
   star3d1r.c: n_terms=7  (expected 7)   sum=1.00030  center=0.25
   box3d1r.c:  n_terms=27 (expected 27)  sum=0.98550  center=0.0355
   ```
   Every shape's parsed term count matches the shape's own support size
   exactly (star-r has `2*dims*r+1`, box-r has `(2r+1)^dims`).

   Kind/dims/radius are NOT overridden (unlike spider) — AN5D's pre-generated
   set happens to cover this domain's 6 shapes exactly, so the requested
   shape already matches the wrapped kernel; only the coefficient VALUES
   differ from the domain's own synthetic weight generator.

2. **Boundary (ORIGINAL finding, history — see §4.3 "2026-09-06 update" for
   the current, fixed state).** AN5D's own `source/common.h::init_grid`
   fills a **fixed halo of width `radius`, ONCE**, and the generated sweep's
   loop bounds (`[radius, dimsize-radius)`) never write to it again across
   timesteps — confirmed by reading `common.h` directly (`init_grid` is
   called once per `benchmark()` trial, not per sweep). This harness's own
   `reference_stencil()` used to ONLY implement a periodic-wrap convention,
   which re-wraps the halo EVERY sweep — a fundamentally different
   convention from what AN5D's own generated code computes. The ORIGINAL
   `prepare()` filled the halo via **one periodic wrap** (not zero, unlike
   spider) from the same `U(0,1)` interior field the reference independently
   regenerates (same seed, same mutated `workload_`) as a partial
   workaround — so sweep 1 matched the reference EVERYWHERE, including the
   boundary band, but from sweep 2 onward AN5D's frozen halo diverged from
   the reference's freshly-rewrapped one; because the stencil support has
   radius `r`, a boundary-driven discrepancy can only propagate `r` cells
   inward per additional sweep — after `T` sweeps, a crop of `r*T` cells off
   every edge was unaffected by the boundary convention and isolated the
   kernel's own arithmetic. This was the same class of finding as spider's
   fixed-zero-halo (`spider/STATUS.md` "BOUNDARY"), generalized to AN5D's
   native multi-timestep dispatch — see §4.1/4.2 for the original numbers
   (interior error ~1e-8 regardless of `T`, full-array error growing with
   `T`) and §4.3 for the fix that made the periodic-wrap workaround
   unnecessary.

3. **Grid shape.** AN5D's generated code takes a single `dimsize` int reused
   for every axis (square/cube grids only — see `gen_bridge.py`'s
   `malloc_expr`); `prepare()` raises `NotImplementedError` for a non-square/
   cube `grid_shape`. Never hit here: this domain's own default AND `--smoke`
   grids are all square (2D) / cube (3D).

4. **Temporal blocking handles arbitrary T natively — no adapter loop
   needed.** Read `<shape>-<tuning>_host.cu` directly: the generated dispatch
   computes `__c0Len = timestep - 0` at RUNTIME (not baked into the tuning
   config) and decomposes it into `⌊T/bt⌋` full-`bt` macro-steps (the main
   loop, `kernel0_<bt>`) plus a `switch`-like `else if` cascade over
   `T mod bt` handling the remainder with progressively smaller kernels
   (`kernel0_1` .. `kernel0_(bt-1)`) down to single-step granularity. `T`
   need NOT divide `bt` — confirmed both by reading the cascade and
   empirically (T=1 and T=5 both ran correctly against `bt` in {2,3,4,8,10}
   above, none of which divide 5 except bt=... — actually none divide 5
   exactly other than trivially, yet every (bt, T=5) combination above
   produced a bit-for-bit-with-the-cascade-decomposed result matching the
   reference to ~1e-8). Unlike convstencil/lorastencil (which found AN5D-
   style native T-loops that DON'T refresh the halo mid-loop and so had to
   write their own single-sweep-at-a-time adapter loop), AN5D's own
   generated cascade is what's called directly, once, per `run()` — no
   adapter-side loop.

## 4. Gate verification (§4.1/4.2 are the ORIGINAL periodic-reference run -- history; §4.3 is the current, "fixed"-reference gate)

### 4.1 Mandated command, run as specified (history — periodic reference)

```
$PY -m kernelbench.runner --kernel stencil \
    --variant stencil-cpu-gpu-kernel-fp64 --smoke \
    --impl an5d-stencil --precision fp64
```
```
running an5d-stencil smoke-star2d1r  ... INVALID: max_scaled_err=3.848e-01 vs tol 1e-05
running an5d-stencil smoke-star3d1r  ... INVALID: max_scaled_err=3.468e-01 vs tol 1e-05
running an5d-stencil smoke-box3d1r  ... INVALID: max_scaled_err=3.131e-01 vs tol 1e-05

0/3 runs valid
```
All 3 smoke shapes are natively supported (no `NotImplementedError` — a
first among this batch's stencil-TCU-style adapters, since AN5D's shape
catalog covers 2D AND 3D). All 3 FAIL the mandated full-array gate, entirely
attributable to the boundary-convention mismatch (§3 point 2) — confirmed,
not just asserted, below.

### 4.2 Supplementary interior-cropped check (`_interior_gate_check.py`, this directory)

Isolates the boundary band from the kernel's own arithmetic for all 6
shapes (the 3 `--smoke` shapes at their exact mandated grid sizes, plus
box2d1r/star2d3r/box2d3r — not in `stencil.py`'s `--smoke` set — on a small
custom 64x64 grid, same precedent as lorastencil's own direct
`run_variant()`-equivalent check for a shape `--smoke` doesn't cover):

```
smoke-star2d1r (mandated --smoke shape)    full_err=3.848e-01  interior_err(crop=  5, shape=(246, 246))=5.7626e-09  tol=1e-05  PASS
smoke-star3d1r (mandated --smoke shape)    full_err=3.468e-01  interior_err(crop=  5, shape=(22, 22, 22))=3.1577e-08  tol=1e-05  PASS
smoke-box3d1r  (mandated --smoke shape)    full_err=3.131e-01  interior_err(crop=  5, shape=(14, 14, 14))=4.1093e-08  tol=1e-05  PASS
box2d1r  (not in --smoke; custom grid)     full_err=4.923e-01  interior_err(crop=  5, shape=(54, 54))=6.2441e-08  tol=1e-05  PASS
star2d3r (not in --smoke; custom grid)     full_err=2.539e-01  interior_err(crop= 15, shape=(34, 34))=1.8035e-08  tol=1e-05  PASS
box2d3r  (not in --smoke; custom grid)     full_err=2.127e-01  interior_err(crop= 15, shape=(34, 34))=2.8919e-08  tol=1e-05  PASS

ALL INTERIOR CHECKS PASS
```
The mandated run's exact full-array numbers (3.848e-01 / 3.468e-01 /
3.131e-01) reproduce bit-for-bit here, confirming this script exercises the
identical code path. **Every shape's interior-cropped error is ~1e-8,
roughly three orders of magnitude under the variant's own `1e-5` tolerance**
— consistent across all 6 shapes and both crop widths (5 for radius-1
shapes at T=5, 15 for radius-3 shapes at T=5), which is exactly what the
"boundary error propagates `radius` cells per sweep" bound predicts (crop
= `radius * timesteps`). This confirms AN5D's generated kernel arithmetic
(coefficients, temporal-blocking dispatch cascade, ping-pong buffer
indexing) is correct for all 6 shapes; only the fixed-vs-periodic boundary
convention differs from this harness's reference, exactly as documented.

The residual ~1e-8 (not full fp64 machine epsilon ~1e-16) is itself
explained and expected: AN5D's coefficients are written as C `float`
literals (`0.09371f`) even in the "double"-typed generated kernel — the
literal itself is rounded to float32 precision (~7 decimal digits) before
being widened to `double` for the multiply-accumulate, while this harness's
reference parses the same decimal text directly to full `double` precision.
This is a property of the artifact's OWN generated source, not an adapter
bug, and is three orders of magnitude inside tolerance regardless.

### 4.3 2026-09-06 update — boundary-convention gate fix: full-array gate now PASSES

`benchspecs/stencil/spec.yaml`'s `inputs.boundary` field allows EITHER
"periodic wrap OR fixed halo of width r" — §4.1/4.2 above only ever gated
against the FIRST option, using a periodic-wrap-for-sweep-1-only workaround
that necessarily diverged from sweep 2 onward, which is exactly what
produced the 0.21-0.49 full-array errors above. AN5D's own generated code
was never actually implementing periodic wrap at all — §3 point 2
establishes it fills its outer radius-width band ONCE and never touches it
again, which is precisely `kernelbench/domains/stencil.py`'s newly-added
`"fixed"` convention (module docstring "Boundary convention hook"), not an
approximation of it.

**The fix**: `adapter.py::prepare()` now (a) sets
`params["boundary"] = "fixed"` so `reference_stencil()` gates against the
same convention AN5D's kernel actually computes, and (b) passes
`dimsize = grid_shape` DIRECTLY to `an5d_<shape>_prepare()` instead of the
old `grid_shape + 2*radius` — since AN5D's own `[radius, dimsize-radius)`
loop bound is computed FROM whatever `dimsize` it is given, handing it
`grid_shape` directly makes the domain's own outer radius-width band double
as AN5D's native halo, with NO extra allocation and NO padding/wrap step:
both ping-pong buffers are seeded with the plain `grid_shape`-sized initial
field via `np.copyto`-free direct assignment (`buf[0] = field; buf[1] =
field`), and the previous `np.pad(..., mode="wrap")` call is REMOVED
entirely (not merely no-op'd) — there is nothing left to imitate periodic
wrap for, since sweep 1 no longer needs to "start out agreeing" with a
different convention. `to_host()` no longer crops `radius` cells off each
edge either, since `dimsize` already equals `grid_shape`.

Mandated command, re-run after the fix:

```
$PY -m kernelbench.runner --kernel stencil \
    --variant stencil-cpu-gpu-kernel-fp64 --smoke \
    --impl an5d-stencil --precision fp64
```
```
  running an5d-stencil smoke-star2d1r  ... 0.063 ms  5.16 GCUP/s  (err 9.54e-09 <= 1e-05)
  running an5d-stencil smoke-star3d1r  ... 0.081 ms  2.03 GCUP/s  (err 3.52e-08 <= 1e-05)
  running an5d-stencil smoke-box3d1r   ... 0.123 ms  0.56 GCUP/s  (err 4.11e-08 <= 1e-05)

3/3 runs valid
```
**All 3 mandated smoke shapes now PASS the FULL-ARRAY gate directly**, with
errors matching the OLD interior-cropped numbers almost exactly
(9.54e-09/3.52e-08/4.11e-08 vs. the §4.2 interior figures
5.76e-09/3.16e-08/4.11e-08 for the same 3 shapes) — expected, since the
full array and the interior are now the SAME comparison under a convention
both sides actually implement; there is no longer a boundary band where the
two sides disagree. Re-running `_interior_gate_check.py` (unmodified; it
picks up `boundary="fixed"` automatically via the same `params` dict
`prepare()` mutates) confirms this for all 6 shapes:

```
$PY artifacts/stencil/an5d/_interior_gate_check.py

smoke-star2d1r (mandated --smoke shape)    full_err=9.541e-09  interior_err(crop=  5, shape=(246, 246))=5.7626e-09  tol=1e-05  PASS
smoke-star3d1r (mandated --smoke shape)    full_err=3.522e-08  interior_err(crop=  5, shape=(22, 22, 22))=3.1577e-08  tol=1e-05  PASS
smoke-box3d1r  (mandated --smoke shape)    full_err=4.109e-08  interior_err(crop=  5, shape=(14, 14, 14))=4.1093e-08  tol=1e-05  PASS
box2d1r  (not in --smoke; custom grid)     full_err=6.486e-08  interior_err(crop=  5, shape=(54, 54))=6.2441e-08  tol=1e-05  PASS
star2d3r (not in --smoke; custom grid)     full_err=1.803e-08  interior_err(crop= 15, shape=(34, 34))=1.8035e-08  tol=1e-05  PASS
box2d3r  (not in --smoke; custom grid)     full_err=2.892e-08  interior_err(crop= 15, shape=(34, 34))=2.8919e-08  tol=1e-05  PASS

ALL INTERIOR CHECKS PASS
```
Full-array and interior-cropped error now agree to 3-4 significant figures
for every shape (the tiny remaining gap is the crop region itself moving by
a few cells, not a new discrepancy). The residual ~1e-8 (vs. fp64 machine
epsilon ~1e-16) is the SAME float32-literal-rounding artifact documented
above (§4.2) — unrelated to the boundary-convention fix, still three orders
of magnitude inside the `1e-5` tolerance.

No shape needed further investigation (radius/band mismatch, corner
handling, and T handling were all checked and found consistent: the
`dimsize = grid_shape` substitution worked cleanly at every grid size and
`bt` tuning config already in use, with no assertion failures from AN5D's
own generated bounds-safety checks).

- nvcc 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`), via
  `bench/artifacts/toolchain.sh`. Arch flag `-arch=sm_80` (A100).
- `-Xcompiler -fPIC,-fvisibility=hidden` — see §2.1 for why hidden visibility
  is required (kernel0_N symbol collisions across the 6 shapes' `.so`s).
- Python: `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.
- Build: `./build.sh` — clones `source/` if missing, runs `gen_bridge.py`,
  compiles 6 independent `bridge_<shape>.so`. Idempotent; no patches to any
  tracked file in `source/`.

## 6. Not done

- Generator toolchain not built (§1) — BUILD-FAILED, not attempted further
  within the ~2h budget (see the specific clang/LLVM version blocker).
- `compiled/float/` (fp32-generated sources) not wired up — this domain's
  primary variant (`stencil-cpu-gpu-kernel-fp64`) is fp64; fp32 would be a
  secondary pass this integration did not have budget for.
- 1D shapes (`source/compiled/double/j2d5pt`-style 1D analogues do not
  exist; AN5D's own 1D benchmarks are not in this artifact's `compiled/`
  listing at all — only 2D/3D) and radius-2/4 stencils
  (`star{2,3}d{2,4}r`, `box{2,3}d{2,4}r}` — also present in
  `compiled/double/` but outside this domain's 6-shape catalog) were not
  wired up: out of scope for `kernelbench/domains/stencil.py`'s
  `_parse_shape_name`, which only recognizes radius 1 and 3.
- No timing sweep beyond the functional gate above (ARTIFACT_GUIDE.md rule
  5 — login node, correctness only).
- The device-memory allocated by `prepare()` is freed only by `free()`; no
  leak was found (each of the 6 `.so`s allocates exactly one `dev_A` buffer
  per `prepare()` call, freed by the matching `an5d_<shape>_free`).

## Verdict

`an5d: generator BUILD-FAILED (pet's PPCG C-frontend needs clang <=~3.8;
only llvm/{18,20,21,22} modules exist here, no clang at all, no sudo) |
FALLBACK BUILT+GATED via AN5D-Artifact's own pre-generated CUDA, all 6
domain shapes (star2d1r, box2d1r, star2d3r, box2d3r, star3d1r, box3d1r) |
(2026-09-06) mandated full-array --smoke gate now PASSES on all 3 smoke
shapes under the spec's "fixed halo of width r" convention, which is what
AN5D's own generated code actually implements (err 9.5e-09 to 4.1e-08, tol
1e-05) | interior-cropped supplementary check still PASSES on all 6 shapes,
now numerically consistent with the full-array numbers (err ~1e-8, tol
1e-5) | history: the ORIGINAL periodic-reference gate was INVALID on all 3
smoke shapes (err 0.31-0.38) because of a boundary-convention mismatch this
update resolves, not a kernel defect`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, parent card reported by
  `nvidia-smi` as "NVIDIA A100-SXM4-40GB, 8.0" on `gpu-b11-6`), login-node
  build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8, `$KB_CUDA_HOME`), host
  compiler for all 6 `bridge_<shape>.so` compiles: g++ 12.4.0 (conda-forge
  `kb-gcc12` env, via `-ccbin`, added — see below), Python 3.12.14
  (`kb-env`). No change to the generator-toolchain verdict (§1): not
  re-attempted here (already exhaustively diagnosed as BUILD-FAILED —
  `pet` needs clang ~2.9-3.8, none available, no sudo; reproducing that
  investigation again was out of scope for this pass, which reproduces the
  FALLBACK path).
- Build: OK, all 6 shapes (`gen_bridge.py` regenerated all 6
  `bridge_<shape>.cu` from the same `source/compiled/double/` files at the
  same tuning configs listed in §2.2; unchanged since `source/` here is the
  same `khaki3/AN5D-Artifact` checkout per `source.provenance`).
  Build-system changes: `build.sh` added
  `HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"` and
  `-ccbin "$HOST_COMPILER"` to the per-shape `nvcc -shared` compile —
  this artifact's `nvcc` call previously had no `-ccbin`, so `nvcc` did its
  own host-compiler search rather than following `$PATH`, and on this
  machine that search finds `$KB_CUDA_HOME`'s own bundled g++ 13.4.0 first
  (nvcc 12.8 cannot parse it — same failure class documented in
  `bench/artifacts/convolution/hidet/STATUS.md`'s Reproduction section, and
  the same `-ccbin "$HOST_COMPILER"`/`KB_GXX12` fix already used by most
  other artifacts in this repo). No change to `gen_bridge.py`, any
  `bridge_<shape>.cu`, or anything under `source/`.
- Gate: mandated command
  (`--variant stencil-cpu-gpu-kernel-fp64 --smoke --impl an5d-stencil
  --precision fp64`) and `_interior_gate_check.py` (unmodified), one GPU
  allocation:
  - smoke-star2d1r: `err 9.54e-09 <= 1e-05` **PASS**
  - smoke-star3d1r: `err 3.52e-08 <= 1e-05` **PASS**
  - smoke-box3d1r: `err 4.11e-08 <= 1e-05` **PASS**
  - `3/3 runs valid` — identical error figures to the recorded §4.3 numbers
    to 3 significant digits.
  - `_interior_gate_check.py`: all 6 shapes PASS, full/interior errors
    identical to §4.3's table (star2d1r 9.541e-09/box2d1r 6.486e-08/
    star2d3r 1.803e-08/box2d3r 2.892e-08/star3d1r 3.522e-08/box3d1r
    4.109e-08).
- Deviation from the recorded ruling: none. Numbers match the §4.3
  ("2026-09-06 update", fixed-boundary convention) figures essentially
  bit-for-bit.
- Verdict here: generator toolchain still not (re-)attempted (BUILD-FAILED
  stands as recorded, not re-investigated); FALLBACK BUILT+GATED, all 6
  shapes build, mandated full-array `--smoke` gate PASSES on all 3 smoke
  shapes, interior-cropped supplementary check PASSES on all 6 — equals the
  recorded ruling.
