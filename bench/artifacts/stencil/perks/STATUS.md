# perks (PERKS) — stencil

**Status: SKIPPED — every PERKS stencil kernel (2D and 3D, star and box,
naive/baseline/persistent-general) hardcodes its own fixed numeric
coefficients via a compile-time preprocessor macro (`stencilParaT`), with
NO runtime-configurable weight-array entry point anywhere in either
driver. This domain's `_build_weights()` produces a DIFFERENT synthetic
coefficient set per shape that cannot be fed to any of these kernels
without editing the numeric literals baked into `source/` — disallowed by
ARTIFACT_GUIDE.md rule 3 ("if the kernel itself must change to run, mark
SKIPPED and say why"). No build was attempted (nothing would be gate-able
regardless of whether it compiled).**

- Paper: "PERKS: a Locality-Optimized Execution Model for Iterative
  Memory-bound GPU Applications" (ICS'23). `PAPER_KEY = conf/ics/ZhangWCMWEM23`
  — same artifact already integrated for the `cg-krylov` track.
- Artifact: https://github.com/neozhang307/PERKS.
- Commit: `b56f46513559942e1da27d9831c5e869550c8b68` (identical to
  `artifacts/cg-krylov/perks` — `source/` here is a SYMLINK to that
  directory's clone, no second `git clone`; see `source.provenance`).

## Reuse, not a rebuild (task brief's explicit instruction)

`source/` is a symlink to `../../cg-krylov/perks/source` — same commit,
one clone. `artifacts/cg-krylov/perks/build.sh` only configures/builds the
`conjugateGradient/` subdirectory of this repo (`SRC_DIR="source/
conjugateGradient"`); the `stencil/` subdirectory (`source/stencil/
{2dstencil,3dstencil}/`) — PERKS's OTHER, and in fact primary, persistent-
kernel application, per its own paper title ("Locality-Optimized Execution
Model for Iterative Memory-bound GPU Applications") — was never configured
or compiled by that build (`cg-krylov/perks/STATUS.md`'s own "Not done"
section says so explicitly: "The `stencil/` subtree ... was not touched,
per the parent task's explicit scope"). This directory therefore has no
`build.sh`/`adapter.py` (matching the layout convention for a SKIPPED
artifact — see e.g. `../../sddmm/hp-spmm-sddmm/`, `source/` +
`source.provenance` + `STATUS.md` only): nothing was built, because the
finding below (hardcoded weights) makes the correctness gate unsatisfiable
regardless of whether compilation succeeds.

## What PERKS ships (read directly, per the task's instruction)

`source/stencil/2dstencil/` and `source/stencil/3dstencil/` implement
Jacobi-style stencil sweeps for exactly the shapes this domain names:
2D `2d5pt` (star2d1r), `2d9pt`/`2ds9pt` (box2d1r-family), `2d13pt`/
`2d17pt`/`2d21pt` (star2d3r-family, higher radius), `2ds25pt`/`2d25pt`
(box2d3r-family); 3D `3d7pt` (star3d1r), `3d13pt`/`3d17pt` (higher-radius
star), `3d27pt` (box3d1r — the domain's own `j3d27pt` alias resolves to
exactly this: "AN5D's own named 27-point 3D kernel is a division-free
variant of box3d1r", `kernelbench/domains/stencil.py:39-41`, and PERKS's
own `poisson/` subdirectory under `3dstencil/` is precisely this same
27-point box compute, confirmed by reading `poisson_gold.cpp` below).
Each shape has multiple generated variants (`naive`, `naivenvcc`,
`baseline`, `baseline_cm`, `persistent`, `gen`, `genwr` — per
`source/stencil/2dstencil/README.md`) built via `./config.sh && ./build.sh`
(CMake, executables land in `./build/init/`) — the same "clone once,
`config.sh`+`build.sh` produces the actual binaries" shape as
`cg-krylov/perks`'s own `cg_perks.exe`, and the same finest-boundary
"wrap one kernel launch" contract this project already uses for that CLI
binary.

## The blocking finding: EVERY PERKS stencil kernel hardcodes its coefficients

Read `source/stencil/{2dstencil,3dstencil}/common/cuda_computation.cuh`
(the ONLY place any of PERKS's driver-facing kernels obtain their per-
neighbor weight): the compute template `computation<...>()` DOES accept
`west/east/north/south/(top/bottom)/center` (star) or `filter[...]` (box)
as ordinary C++ function parameters (`#define stencilParaList ...`) — this
part IS genuinely parameterized, by design. But at every one of the THREE
places any actual `__global__` kernel invokes it (`jacobi-general-
kernel.cuh:40`, `jacobi-naive.cu:16`, `jacobi-baseline.cu:76` for 2D;
`j3d-naive.cu`, `j3d-baseline.cu`, `j3d-baseline-memwarp.cu` for 3D — six
sites total, confirmed by `grep -rn "stencilParaT;"`), the values actually
supplied come from the SAME macro, `stencilParaT`, and its definition is a
plain, unconditional `#define` with LITERAL numeric constants — not a
kernel-launch argument, not a value read from `argv`/a file, not anything
a caller (including our own new glue code, since it is invoked from
inside files we do not control the internals of) can override without
editing the macro body itself:

```
# 2D star (2dstencil/common/cuda_computation.cuh:19-25)
const REAL west[6]={12.0/118,9.0/118,3.0/118,2.0/118,5.0/118,6.0/118};
const REAL east[6]={12.0/118,9.0/118,3.0/118,3.0/118,4.0/118,6.0/118};
const REAL north[6]={5.0/118,7.0/118,5.0/118,4.0/118,3.0/118,2.0/118};
const REAL south[6]={5.0/118,7.0/118,5.0/118,1.0/118,6.0/118,2.0/118};
const REAL center=15.0/118;

# 2D box, HALO=1 (2dstencil/common/cuda_computation.cuh:33-40)
const REAL filter[3][3] = {{7/118,5/118,9/118},{12/118,15/118,12/118},{9/118,5/118,7/118}};

# 3D star, HALO=1 -- i.e. star3d1r (3dstencil/common/cuda_computation.cuh:62-68)
const REAL center=-1.67f; west=0.162f east=0.161f north=0.163f south=0.164f
bottom=0.166f top=0.165f;   # note: NEGATIVE center, asymmetric per-face weights

# 3D box (the "poisson" 27-point variant -- box3d1r / j3d27pt)
# poisson/poisson_gold.cpp:19-21 (the CPU gold reference, confirms the same
# literal convention as the GPU kernel's own stencilParaT):
output[i][j][k] = 2.666*C - 0.166*(6 face neighbors) - 0.0833*(12 edge neighbors)
```

Every one of these is a FIXED, shape-specific literal constant set,
different from and unrelated to `kernelbench.domains.stencil._build_weights()`
(this domain's own synthetic, sign-varying, `sum(|w|)==1`-normalized
coefficient set — `stencil.py:95-127`, deliberately different from any
specific PDE's physical coefficients, generic over `(kind,dims,radius)`).
Confirmed there is no CLI escape hatch either: `grep -in "weight|coef"` over
`source/stencil/2dstencil/jacobi.driver.cpp` and
`source/stencil/3dstencil/j3d.driver.cpp` (the actual `main()` argument
parsers) returns **zero matches** — every documented CLI flag
(`--fp32`, `--check`, `--usesm`, `--bdim`, `--blkpsm`, `--iter`, `--warmup`,
`--doubletile`, per each stencil README) controls tiling/precision/timing
knobs, never the stencil coefficients themselves. This is confirmed
structurally, not merely by absence of a flag: `computation()`'s
parameters ARE real function arguments, but the ONLY three call sites in
the entire artifact that invoke it always do so via the SAME hardcoded
`stencilParaT` macro expansion — there is no second, alternative call site
anywhere that threads a caller-supplied array through instead.

This is a strictly worse situation than the sibling
`../../stencil/lorastencil/STATUS.md` precedent this task named: there, 3
of 4 shapes were unwrappable for analogous "kernel ignores/mishandles its
own params" reasons, but ONE (`star2d3r`) genuinely accepted this domain's
`workload.weights` through a real runtime array argument and was
`BUILT+GATED`. Here, PERKS's driver-facing kernels (`naive`, `baseline`,
`general`) have **no such argument anywhere in any of the six call sites**
across BOTH 2D and 3D, star AND box, radius 1 AND higher — the
`computation()` template's own parameterization is an internal
implementation-organization choice (letting `jacobi-naive.cu`,
`jacobi-baseline.cu`, and the persistent `jacobi-general-kernel.cuh` share
one device function), never exposed past a single, fixed, compile-time
macro. Reaching a runtime-configurable weight set would require writing
new source that either (a) edits the tracked macro body inside
`cuda_computation.cuh` itself (a kernel-code patch, rule 3), or (b)
authors an entirely new preprocessor-macro override plus a from-scratch
kernel-launch driver replicating PERKS's own grid/block/shared-memory
sizing and (for `general`/`persistent`) its cooperative-groups
persistent-kernel synchronization from the ground up — at that point the
work is reimplementing a stencil kernel using PERKS's helper functions as
a library, not wrapping PERKS's own kernel (rule 1's "wrap the kernel, not
[[[reimplement your own using its pieces]]]"). Neither is in scope.

## Why the boundary-condition question doesn't even arise

Separately (documented for completeness, though moot given the finding
above already blocks every shape): PERKS's `naive`/`baseline` kernels use
**clamped** boundary indexing (`max(0, l_x-1-hl)`, `min(width_x-1,
l_x+1+hl)`, `jacobi-naive.cu`), not this domain's **periodic-wrap**
convention (`np.roll`, `stencil.py:340`). This mismatch alone would be
bridgeable with adapter-side periodic-wrap halo padding before each call
(the same technique `../../stencil/lorastencil/STATUS.md` already uses:
"re-padding the periodic-wrap halo from the CURRENT field before each
single-sweep call") — it is NOT what blocks this integration; the
hardcoded-weight finding above is dispositive on its own, and would remain
so even if the boundary condition matched exactly.

## Verdict

`perks: SKIPPED (all shapes: star2d1r, box2d1r, star2d3r, box2d3r,
star3d1r, box3d1r/j3d27pt) -- every PERKS stencil kernel (naive, baseline,
persistent/general; 2D and 3D) sources its per-neighbor weights from a
single, unconditional, compile-time #define (stencilParaT) with literal
numeric constants specific to PERKS's own benchmark convention; no CLI
flag, kernel argument, or generator step anywhere in the artifact accepts
a caller-supplied coefficient set. This domain's own synthetic weights
(kernelbench.domains.stencil._build_weights()) can therefore never be fed
to any PERKS stencil kernel without editing tracked source (disallowed,
ARTIFACT_GUIDE.md rule 3) -- not attempted, not gated, no build performed.`

## Not done

- No `build.sh`/`adapter.py` written (nothing to build toward — see
  above); `source/` symlink + this STATUS.md + `source.provenance` are the
  only files in this directory, matching the established SKIPPED-artifact
  layout (e.g. `../../sddmm/hp-spmm-sddmm/`).
- The register-autotuning codegen scripts (`ProcessCompilerLog.py`,
  `ProcessRsts2cuh.py`) were read enough to confirm they generate
  `perksconfig.cuh` (max-register-per-thread tuning), NOT stencil
  coefficients — not run.
- 1D shapes: PERKS ships no `stencil/1dstencil/` directory at all (only
  `2dstencil/`, `3dstencil/`); this domain's `1d1r`/`1d2r` shapes have no
  PERKS candidate regardless of the weight finding.
