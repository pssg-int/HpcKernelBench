# stencil — artifact integration outcomes

## Baseline selection rule (user decision, 2026-09-05)

Baselines are chosen from `kernel-papers/output/baseline_selection.md`,
produced by `select_baselines.py` from the per-(paper, track) ratings in
`output/kernel_centrality.json`: (1) kernel **centrality** `core` (the
kernel IS the paper's headline contribution) beats `component`
(`tangential` is never a baseline); (2) **regime match** with the track
spec's inputs (`matches` > `partial` > `mismatch`); (3) **single-NVIDIA-GPU
path** required; (4) **recency** only as the tiebreak, up to 5 per track.
Already-integrated adapters the rule would not have picked stay in the
registry as competitors (still gated) but are labelled `component`/
off-regime here rather than treated as the human-SOTA reference.

## Directory listing (`kernel_centrality.json` keys are `stencil|<paper_key>`)

| dir | paper | key | centrality/regime | verdict |
|---|---|---|---|---|
| `spider/` | SPIDER (PPoPP'26) | `conf/ppopp/GuW0Y26` | core / matches | **BUILT+GATED** — full-array mandated gate PASSES under the domain's `zero-halo` convention (2026-09-06; its kernel uses a zero-valued external halo); earlier interior-only pass kept as history in STATUS.md |
| `flashfftstencil/` | FlashFFTStencil (PPoPP'25) | `conf/ppopp/HanLCBZYCZCY25` | core / matches | BUILT+GATED (box2d1r, fp64, err 4.40e-15) |
| `convstencil/` | ConvStencil (PPoPP'24) | `conf/ppopp/ChenLWBWMYZCY24` | core / matches | BUILT+GATED (2D star2d1r only, err 1.14e-16; 1D/3D not wired, time budget) |
| `windstencil/` | WindStencil (ICS'26) | `conf/ics/ZhangZLLJZYLL26` | core / matches (AMD-only) | SKIPPED — AMD ROCm/HIP only, no NVIDIA build path on this machine |
| `lorastencil/` | **LoRAStencil (SC'24)** | `conf/sc/ZhangLYCZCY24` | core / matches | **BUILT+GATED** (star2d3r only, fp64, err 2.92e-16; box2d1r/box2d3r/star2d1r genuinely unwrappable — see below) |
| `ozhope/` | **ozHOPE (IPDPS'26)** | `conf/ipps/YaoZLX26` | core / matches | **SKIPPED** — tensor-core reconstruction kernel not separable with a plain-array interface (see below) |
| `perks/` | **PERKS (ICS'23)** | `conf/ics/ZhangWCMWEM23` | core / matches | **SKIPPED (reused build)** — `source/` REUSED from `../../cg-krylov/perks/` (no second clone); every PERKS stencil kernel (2D/3D, star/box) hardcodes its own coefficients via a fixed compile-time macro with no runtime injection path anywhere — see below |
| `an5d/` | **AN5D (CGO'20)** | `conf/cgo/MatsumuraZWEM20` | core / matches | **BUILT+GATED** — generator toolchain unbuildable (needs clang<=3.8); the paper's own pre-generated CUDA wrapped for all 6 shapes; full-array gate PASSES under `fixed` halo (1e-8) |
| `kathena/` | **K-Athena (TPDS'21)** | `journals/tpds/GreteGO21` | core / matches | **SKIPPED** — no `module spider kokkos` on this machine and K-Athena vendors Kokkos only as an uninitialized git submodule with no system-install path, so nothing in this repo compiles without a from-source Kokkos build (out of budget); separately, the paper's own Kokkos-parallelized contribution (`hydro/calculate_fluxes.cpp` + `add_flux_divergence.cpp`) is a fused reconstruction+Riemann-solve+CT+flux-divergence MHD update with no standalone plain-array stencil boundary — same class of finding as `flashfftstencil`/`ozhope`. The one candidate that looked simpler (`hydro_diffusion`/`field_diffusion`) is confirmed (`grep -c "Kokkos\|par_for\|MDRangePolicy" == 0`) to be UNPORTED legacy Athena++ CPU code, not the paper's own contribution, and only a flux CONTRIBUTION (not a complete sweep) entangled with `MeshBlock`/`Coordinates` context. See `kathena/STATUS.md`. |

`spider`/`flashfftstencil`/`convstencil`/`windstencil` predate this
integration pass (already GATED/SKIPPED per their own STATUS.md); this
pass added `lorastencil/` and `ozhope/`; a later pass added `perks/`
(cross-track reuse from `cg-krylov`).

## LoRAStencil (SC'24) — `lorastencil/` — **BUILT+GATED (star2d3r only)**

`PAPER_KEY = conf/sc/ZhangLYCZCY24`. Wraps `gpu_star_2d3r`
(`source/src/2d/gpu.cu`) through a new `bridge.cu`, fp64 throughout (same
situation as the convstencil/flashfftstencil siblings — genuine fp64
DMMA, not fp16, despite appearing in spec.yaml's fp16-labeled variant-2
`suite` field), gated against variant 1 (`stencil-cpu-gpu-kernel-fp64`).

**Gate result**: `max_scaled_err = 2.92e-16` (tol 1e-5) on a 128x128
star2d3r workload, T=3 — PASS by nine orders of magnitude, consistent with
an independent 64x64 probe (3.33e-16) run before wiring the adapter.

**Why only ONE of the four 2D kernels is wrapped** (verified empirically,
not a coverage shortcut — see `lorastencil/STATUS.md`/`bridge.cu` for full
evidence): `gpu_box_2d3r` (dispatched for both `box2d1r`/`box2d3r`) is
NOT a general low-rank decomposition — it is a hardcoded 4-level "onion-
ring" peel matching LoRAStencil's own synthetic benchmark pattern, and
divides by zero (`0.0/0.0` → NaN) for both a true zero-padded box2d1r
(corner entry exactly 0) AND this domain's own box2d3r weight construction
(residual concentrates in a row the factorization's level-1 division does
not expect). `gpu_star_2d1r` ignores its own `params` argument entirely
(hardcoded `{0,1,2,4,2,1,0}` local arrays). Only `gpu_star_2d3r` builds its
factors by plain assignment (no division) — mathematically exact for a
star-shaped support, confirmed empirically. 1D/3D not wired (time budget,
matching convstencil's own precedent).

## ozHOPE (IPDPS'26) — `ozhope/` — **SKIPPED**

`PAPER_KEY = conf/ipps/YaoZLX26`. `recon.py`'s `tpp_recon_class.forward()`
looked separable at first (a genuine stencil-as-convolution kernel: plain
cuDNN conv at `prec_mode=0`, Ozaki-split fp16-Tensor-Core conv + custom
torch-op accumulate at `prec_mode=1`, the paper's actual contribution) —
but reading the custom op's C++ body (`pyt_ozaki_kernel.cu:479-583`) shows
its correctness depends on: (1) compile-time-hardcoded channel-count
constants (`OZCUDNN_CHNLS=16`, `OZCUDNN_OUTCHNLS=64`) tuned to ozHOPE's own
reconstruction operator; (2) THREE undocumented environment variables
(`NUMSPLIT`, `BITS_PER_SLICE`, `OZCUDNN_OUTCHNLS`) read with no validation
and no default anywhere in this repository; (3) a constructor
(`tpp_recon_class.__init__`) that requires the full cubed-sphere `mesh`
object, not a bare shape. This is the functional equivalent of "fused with
the dynamical core, not separable" even though the Python call itself is
not textually inlined into the RK loop — see `ozhope/STATUS.md` for the
full investigation, including confirming the `nvidia-cudnn-frontend`
Python dependency and ATen/cuDNN C++ headers WOULD have been available
(isolated `pip install --target` probe, not installed into the shared
project venv) before the channel/env-var/mesh blockers were found.

## PERKS (ICS'23) — `perks/` — **SKIPPED (reused build)**

`PAPER_KEY = conf/ics/ZhangWCMWEM23`. Same artifact already integrated for
the `cg-krylov` track (`artifacts/cg-krylov/perks/`, `cg_perks.exe`) — this
directory REUSES that clone (`source/` symlinked, no second `git clone`;
no second build either, since nothing here compiles — see below) rather
than re-fetching PERKS a second time, per this task's cross-track-reuse
instruction. PERKS ships genuine 2D/3D star/box stencil kernels
(`source/stencil/{2dstencil,3dstencil}/`, covering exactly this domain's
`star2d1r`/`box2d1r`/`star2d3r`/`box2d3r`/`star3d1r`/`box3d1r`
(=`j3d27pt`) shape family) alongside the CG solver it was originally
wrapped for.

**Why it is SKIPPED, not gated**: every one of PERKS's driver-facing
stencil kernels (`naive`, `baseline`, and the persistent `general`
variant, both 2D and 3D) obtains its per-neighbor coefficients from a
SINGLE, unconditional, compile-time `#define stencilParaT` with literal
numeric constants specific to PERKS's own benchmark convention (e.g. 2D
star: `west/east/north/south` in `{2,3,5,6,7,9,12}/118`, `center=15/118`;
3D star radius-1: `center=-1.67`, six asymmetric per-face weights
`~0.16`-`0.166`; the `poisson/` 27-point box variant —
`box3d1r`/`j3d27pt` — hardcodes `2.666`/`-0.166`/`-0.0833`). Grepping both
drivers' argument parsers (`jacobi.driver.cpp`, `j3d.driver.cpp`) for any
weight/coefficient CLI flag returns zero matches, and the ONLY three
(2D)/three (3D) call sites of the underlying `computation()` template
device function all route through this SAME hardcoded macro — there is no
second call site, CLI flag, or generator step anywhere in the artifact
that threads a caller-supplied coefficient array through instead. This
domain's own synthetic weights
(`kernelbench.domains.stencil._build_weights()`) can therefore never be
fed to any PERKS stencil kernel without patching numeric literals inside
`source/` — disallowed by ARTIFACT_GUIDE.md rule 3 ("if the kernel itself
must change to run, mark SKIPPED and say why"), so no build was attempted
here (nothing would be gate-able regardless of whether it compiled). This
is a strictly more pervasive version of the `lorastencil/star2d1r`/
`box2d1r`/`box2d3r` finding above (which had at least ONE shape,
`star2d3r`, genuinely wrappable via a real runtime `params` array) — PERKS
has no such shape anywhere, in either 2D or 3D. Full file:line evidence
in `perks/STATUS.md`.

## Boundary conventions (2026-09-06)

| dir (impl) | convention the kernel implements | evidence / gate |
|---|---|---|
| `spider/` (`spider-box2d7r-sptc`) | Zero-Dirichlet EXTERNAL halo — now modeled EXACTLY by the domain's new `"zero-halo"` convention | INVALID, `max_scaled_err=1.30e+06` (smoke-star2d1r, LARGER than even the original periodic-reference failure — see STATUS.md) | **PASSES**: `max_scaled_err=2.90e-03` (mandated `smoke-star2d1r` run, tol `0.01`). Supplementary `_interior_gate_check.py`: full-array error now EQUALS the interior-cropped error exactly (`2.538275e-03` both, on a 64x128 grid) — the worst-case cell moved off the boundary entirely, confirming the remaining error is ordinary fp16 rounding, not a boundary mismatch. `adapter.py::prepare()` now sets `params["boundary"] = "zero-halo"`. |
| `an5d/` (`an5d-stencil`) | Fixed halo of width `radius`, filled ONCE by `common.h::init_grid`; generated sweep's `[radius, dimsize-radius)` loop bounds never touch it again across timesteps — this domain's new "fixed" convention, EXACTLY | INVALID on all 3 smoke shapes, `max_scaled_err` 0.31-0.38 (from a periodic-wrap-for-sweep-1-only workaround that diverged from sweep 2 onward), tol 1e-5. Interior-cropped check PASSED (~1e-8). | **PASSES on all 3 smoke shapes directly on the full array**: `max_scaled_err` 9.5e-09 / 3.5e-08 / 4.1e-08 (star2d1r / star3d1r / box3d1r), tol 1e-5 — three-to-four orders of magnitude inside tolerance. Fix: pass `dimsize = grid_shape` directly (not `+2*radius`) so AN5D's own native halo coincides with the domain's own outer band; removed the periodic-wrap-fill workaround entirely (no longer needed); set `params["boundary"]="fixed"`. All 6 domain shapes verified via the existing `_interior_gate_check.py` (now full-array == interior to 3-4 sig figs, since both sides agree on the convention). |


# stencil — boundary-convention gate fix (2026-09-06)

For the coordinator to merge into `README.md` (not edited directly here per
current ownership). Task: make the stencil track's correctness gate honor
BOTH boundary conventions `benchspecs/stencil/spec.yaml`'s `inputs.boundary`
field allows ("periodic wrap OR fixed halo of width r"), since the
pre-existing gate only ever implemented periodic wrap while SPIDER and
AN5D's own kernels implement a fixed-halo-style convention instead.

## What changed

`kernelbench/domains/stencil.py` gained a `"fixed"` boundary convention
(module docstring "Boundary convention hook"), implemented independently in
BOTH `reference_stencil` (new `_sweep_fixed` helper) and `NumpyStencil`
(new `_numpy_impl_sweep_fixed` helper, a genuinely different numpy
primitive — `sliding_window_view` + `tensordot` vs. the reference's
offset-loop slicing — per the reference-independence rule). Semantics:
each sweep updates only points at distance >= radius from every edge;
the width-radius boundary band keeps the seeded initial field's own values
for the whole run (no separate halo array, no per-sweep refresh). Selection
is via `params["boundary"] = "fixed"`, set by an implementation's own
`prepare()` on the same `params` dict the harness passes to the reference
right after — absent, the workload's default ("periodic") applies, so every
pre-existing caller is unaffected. The cost model (`_cost_stencil`,
GCUP/s) is UNCHANGED — it still counts the full `cells * timesteps`
regardless of convention; `describe()`'s `boundary` field discloses which
band a "fixed" run actually left untouched.

## Per-adapter rows (full-array `--smoke`/mandated-command gate, before -> after)


## 2026-09-06 update #2 — a THIRD convention, `"zero-halo"`, added (SPIDER)

The "fixed" convention above turned out not to be what SPIDER computes:
`"fixed"` freezes an IN-ARRAY band to its t=0 value, but SPIDER's own
`kernel_2d_7r` recomputes EVERY domain cell each launch, reading zero for
whichever neighbor offsets fall outside the domain (confirmed from source:
`bridge.cu:122` zero-fills the entire padded host buffer before only the
interior is copied in at `bridge.cu:124-127`; `gpu_2d_7r_half.cu:73-82`
reads straight out of that padded array with no clamp/replicate logic; the
output store at `gpu_2d_7r_half.cu:147-162` writes the full block tile
unconditionally with no per-cell skip). That is a genuinely third
spec-allowed convention (spec.yaml's "fixed halo of width r" also covers a
zero-valued EXTERNAL halo, not only an in-array frozen band), so
`kernelbench/domains/stencil.py` gained `"zero-halo"` directly (new
`_sweep_zero_halo` reference helper + independently-coded
`_numpy_impl_sweep_zero_halo` CPU-impl helper, unit-checked to agree to
~1e-16 across all 6 domain shapes) rather than continuing to approximate it
with "fixed".


## Unaffected (verified, no change needed)

- `convstencil/` (`convstencil-tcu`): genuinely implements periodic wrap
  (re-pads via `np.pad(mode="wrap")` every sweep); explicitly guards
  `workload_.boundary != "periodic"`. Gate unchanged, still PASSES
  (`err=1.14e-16`, tol 0.01).
- `lorastencil/` (`lorastencil-star2d3r`): also genuinely periodic wrap,
  same guard pattern. Gate unchanged, still PASSES (`err=2.92e-16`,
  tol 1e-5).
- `perks/`: SKIPPED entirely, unrelated to boundary (every PERKS stencil
  kernel hardcodes its own numeric coefficients with no runtime-injectable
  weight array, so no shape can be gated regardless of boundary). For the
  record, PERKS's own `naive`/`baseline` kernels use CLAMPED
  (replicate-edge / Neumann-style) boundary indexing — a fourth convention,
  neither periodic, fixed, nor SPIDER's zero-Dirichlet — but this is moot
  since the coefficient mismatch blocks gating on its own (see
  `perks/STATUS.md`, "Why the boundary-condition question doesn't even
  arise"). Not edited (owned by a concurrent agent / already finalized as
  SKIPPED before this task started).
- CPU `numpy-stencil` / `scipy-convolve-stencil` (default periodic):
  unchanged, `err=0.00e+00` / `~1e-16`, still PASS.

## Files touched by this fix (both rounds: `"fixed"` then `"zero-halo"`)

- `kernelbench/domains/stencil.py` — `"fixed"` convention (reference +
  `NumpyStencil`), then `"zero-halo"` convention (same pair of code paths:
  `_sweep_zero_halo` in the reference, `_numpy_impl_sweep_zero_halo` in
  `NumpyStencil`), module docstring hook (updated for all three
  conventions), `describe()` text.
- `DOMAIN_GUIDE.md` — new "An implementation-selected convention (params
  override)" section generalizing the hook pattern; updated to list all
  three stencil conventions once `"zero-halo"` was added.
- `artifacts/stencil/an5d/adapter.py`, `artifacts/stencil/an5d/STATUS.md`
  (round 1, `"fixed"` — unaffected by round 2)
- `artifacts/stencil/spider/adapter.py`, `artifacts/stencil/spider/STATUS.md`
  (round 1: `"fixed"`, still INVALID; round 2: `"zero-halo"`, now PASSES)
- This file (new in round 1) — NOT `artifacts/stencil/README.md`, which is
  currently owned by another agent.

`./smoke_all.sh` re-verified green after BOTH rounds (35/35; smoke uses
periodic by default and none of the smoke-covered CPU/GPU impls set
`params["boundary"]`).
