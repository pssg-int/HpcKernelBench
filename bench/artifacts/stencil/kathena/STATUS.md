# K-Athena — STATUS

**Outcome: SKIPPED — no standalone stencil kernel with a plain-array
interface; the paper's actual contribution (the Kokkos-parallelized MHD
finite-volume update) is a fused multi-physics solver, and even the one
candidate that superficially resembles a diffusion/stencil operator is
written against the same Kokkos-backed array class, so any build at all
needs Kokkos built from source (no `module spider kokkos` on this machine,
and K-Athena vendors it only as an uninitialized git submodule) — explicitly
out of this integration's budget per the task brief.**

Paper: "K-Athena: A Performance Portable Structured Grid Finite Volume
Magnetohydrodynamics Code", TPDS 2021. `PAPER_KEY = journals/tpds/GreteGO21`.
Repo: `https://gitlab.com/pgrete/kathena` (commit
`5da8e587380aea4de4a50a4982d66b72124589cd`, 2023-02-22; `git clone --depth 1`
into `./source/`, untouched — read-only, no build attempted).

Time spent: well under the 45-minute budget (clone + read `src/` + two grep
passes settled the question; no build attempted at all, per the task's own
"do not build Kokkos unless ... available via `module spider`" instruction).

## What K-Athena actually is

A Kokkos port of Athena++ (a widely-used astrophysical MHD code): the
paper's contribution is making Athena++'s finite-volume hydro/MHD update
performance-portable across CPU/GPU via Kokkos, while keeping the original
Athena++ algorithms (reconstruction, Riemann solvers, constrained transport
for the magnetic field, RK time integration) unchanged. `src/` layout:
`hydro/`, `field/`, `reconstruct/`, `eos/`, `coordinates/`, `mesh/`, `bvals/`,
`gravity/`, `multigrid/`, `fft/`, `task_list/`.

## Is the array class actually Kokkos-backed? (checked directly)

`src/athena_arrays.hpp`'s `AthenaArray<T>` (used EVERYWHERE in this codebase,
including `hydro_diffusion`/`field_diffusion` below) is not the original
Athena++ CPU array — it wraps `Kokkos::View<T*,...,DevSpace>` internally
(`KView1D_`..`KView5D_` members, `get_KView4D()` accessors,
`Kokkos::deep_copy` in the copy constructor — `athena_arrays.hpp:93-148`).
So EVERY file in this codebase that includes `athena_arrays.hpp` -- which is
all of them -- requires Kokkos headers to compile at all, regardless of
whether that particular file's own loops use `Kokkos::parallel_for`.

## Candidate 1: the paper's actual contribution — `hydro/calculate_fluxes.cpp` + `hydro/add_flux_divergence.cpp`

`Hydro::CalculateFluxes` (reconstruction -> Riemann solver -> flux array,
`hydro/calculate_fluxes.cpp`, 21 occurrences of
`Kokkos`/`par_for`/`MDRangePolicy`) and `Hydro::AddFluxDivergenceToAverage`
(`hydro/add_flux_divergence.cpp`, reads `x1flux`/`x2flux`/`x3flux` via
`get_KView4D()` then applies the flux-difference update, plus a per-
coordinate-system geometric source term) are the genuinely
Kokkos-parallelized, GPU-portable core this paper is about. This is
architecturally a FUSED finite-volume MHD solver — reconstruction, an
approximate Riemann solve (choice of `rsolvers/` — HLLE/HLLC/HLLD/Roe),
constrained-transport electric-field averaging (`field/calculate_corner_e.cpp`),
and the flux-divergence update are separate stages that all run inside one
`Hydro::CalculateFluxes` call and its sibling functions, each taking the full
`MeshBlock`/`Coordinates`/`FaceField` object graph as context (`pmb->is/ie/...`,
`pco_->dx1v(...)`, `b.x1f`/`b.x2f`/`b.x3f` for the staggered magnetic field).
This is NOT a `U_out[p] = sum_d w_d*U_in[p+d]` weighted-neighborhood sweep —
it is the same class of finding as `flashfftstencil/STATUS.md` (fused
kernel, no standalone sub-kernel boundary) and `ozhope/STATUS.md` (paper's
actual numerical contribution entangled with the surrounding solver/mesh
object graph), not attempted further.

## Candidate 2: `hydro/hydro_diffusion/{conduction,viscosity}.cpp`, `field/field_diffusion/*.cpp`

These looked like the most promising candidate at first glance — thermal
conduction / viscosity / Ohmic-and-ambipolar resistivity are, physically,
Laplacian-like diffusion operators, structurally closer to a plain stencil
than the Riemann-solver machinery above. Read `hydro_diffusion/conduction.cpp`
in full (137 lines): `HydroDiffusion::ThermalFlux_iso(prim, cons, cndflx)`
computes a face-centered flux CONTRIBUTION
(`x1flux(k,j,i) -= kappaf*denf*dTdx`, one term added into a flux array that
ALSO receives the hyperbolic/Riemann-solver flux from candidate 1 above,
later combined by `AddFluxDivergenceToAverage`) — not a complete
"read neighbors, write updated grid" sweep on its own, and it is a member
function of `HydroDiffusion` taking `pmb_`(`MeshBlock*`)/`pco_`(`Coordinates*`)
as implicit context for grid indices (`pmb_->is/ie/js/je/ks/ke`,
`pmb_->block_size.nx2/nx3`) and non-Cartesian metric factors
(`pco_->dx1v`/`h2v`) — not a plain-array interface.

More importantly: **`grep -c "Kokkos\|par_for\|MDRangePolicy"` over
`hydro/hydro_diffusion/*.cpp` and `field/field_diffusion/*.cpp` returns 0** —
these files use plain nested `for` loops (`#pragma omp simd` on the innermost
loop only) over `AthenaArray<Real>` operands. Per the check above,
`AthenaArray` is STILL Kokkos-backed at the storage level (so this code only
compiles/links against Kokkos headers/libs), but the loops themselves are
CPU-only, unparallelized-by-Kokkos legacy Athena++ code path — consistent
with the repo's own README: **"Please note that not all features of
Athena++ have been ported yet. Currently supported are non-relativistic
hydro and MHD simulations..."** — i.e. this is exactly one of the
NOT-YET-PORTED, non-GPU-portable pieces the paper is not claiming credit
for, not the paper's own performance-portable contribution.

## Why no build was attempted (Kokkos availability, checked first)

Per the task's own instruction ("do not build Kokkos unless a separable
kernel exists AND a Kokkos install is available via `module spider kokkos`
-- building Kokkos from source is out of budget"): checked
`module spider kokkos` FIRST, before any further code reading — **"Unable to
find: kokkos"** (no module of any name/version). `source/.gitmodules` shows
`kokkos` is vendored ONLY as a git submodule
(`https://github.com/kokkos/kokkos.git`, path `./kokkos`), and
`configure.py --kokkos_path` defaults to `./kokkos` with no apparent
system-install / `find_package(Kokkos)` alternative in its argument list —
building Kokkos from source is the only path this codebase's own build
system supports, and it was never fetched by the `--depth 1` clone (no
submodule init was run). This settles the question independent of whether
Candidate 1 or 2 above turned out separable: even if either did, there is no
way to compile ANY file in this repository without Kokkos, and building
Kokkos from source is explicitly out of this integration's budget.

## Not done

- No submodule init / Kokkos build attempted (out of budget, per above).
- No further candidates investigated inside `mesh/`, `multigrid/`, `fft/`,
  `gravity/` -- these are progressively less likely to contain a
  star/box-shaped stencil kernel than the two candidates above (multigrid's
  own smoother IS a stencil in principle, but `multigrid/` here is a
  self-gravity Poisson solver riding on the same Kokkos/MeshBlock machinery
  as candidate 1, not evaluated further within the time budget once the
  Kokkos-availability blocker above was confirmed to apply repo-wide).

## Verdict

`kathena: SKIPPED (no module spider kokkos on this machine; K-Athena vendors
Kokkos only as an uninitialized git submodule with no system-install build
path, so no file in this repo -- AthenaArray<Real> itself is Kokkos::View-
backed, athena_arrays.hpp:93-148 -- can be compiled at all without a
from-source Kokkos build, out of budget; separately, the paper's actual
Kokkos-parallelized contribution, hydro/calculate_fluxes.cpp +
add_flux_divergence.cpp, is a fused reconstruction+Riemann-solve+CT+flux-
divergence MHD update with no standalone plain-array stencil boundary, and
the one structurally-simpler candidate, hydro_diffusion/conduction.cpp +
viscosity.cpp + field_diffusion/*.cpp, is confirmed by
`grep -c "Kokkos\|par_for\|MDRangePolicy"` == 0 to be UNPORTED legacy
Athena++ CPU code -- not the paper's own contribution -- and is itself only
a flux CONTRIBUTION requiring MeshBlock/Coordinates context, not a complete
stencil sweep)`
