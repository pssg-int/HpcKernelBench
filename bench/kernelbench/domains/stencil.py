"""
Stencil / structured-grid sweeps: a fixed weighted sum over a small
neighborhood (star or box support) applied to every point of a dense
D-dimensional grid, repeated for T sequential time steps with double-buffered
(ping-pong) storage.

Follows the shape of sparse.py (the reference domain module):

  WORKLOADS   — grid + shape + timestep-count descriptor, smoke set + spec-sized
                loader keyed by AN5D's shape-name convention
  COST        — GCell-updates/s, NOT a flop rate (spec.yaml refuses GFLOP/s as
                primary: FLOP-per-point-update conventions differ per paper and
                are not comparable; see benchspecs/stencil/spec.yaml notes_on_fairness)
  REFERENCE   — an fp64 reference performing the identical T-sweep recursion,
                returning (result, scale) where `scale` runs the SAME recursion
                with |weights| on the abs-field -- the T-fold generalization of
                the sparse domain's |A|@|B| cancellation-robust denominator
  IMPLS       — CPU (numpy ping-pong sweep, scipy.ndimage.convolve) and a
                torch-based CUDA sweep (wired, not run on the login node)

Boundary convention hook (2026-09-06): benchspecs/stencil/spec.yaml's
`inputs.boundary` field explicitly allows EITHER "periodic wrap OR fixed halo
of width r" -- both are spec-conforming, not just "periodic". An
implementation declares which one its own kernel actually computes by
setting `params["boundary"] = "fixed"` inside its own `prepare()` (the same
`params` dict the harness passes on, unmodified, to `reference(matrix,
params)` right after -- precedent: dense.py's `reference_gemv` `quant_bits`
key, ml.py's `reference_qgemm` `dequantized_W_override` key). When absent,
the workload's own default (`StencilWorkload.boundary`, "periodic") is used,
so every existing caller/impl is unaffected by this hook's addition. Under
"fixed", both `reference_stencil` and `NumpyStencil` update only points at
distance >= radius from every edge each sweep and leave the width-radius
boundary band holding the seeded initial field's own values for the entire
T-sweep run (no separate halo array, no per-sweep refresh) -- see
`_sweep_fixed` below.

A third spec-conforming convention, `"zero-halo"` (2026-09-06, added for
SPIDER, PPoPP 2026 -- see `artifacts/stencil/spider/STATUS.md`): read
literally, spec.yaml's "fixed halo of width r" covers not only an IN-ARRAY
frozen band (this domain's own "fixed", AN5D's convention) but also an
EXTERNAL halo of zeros appended outside the array, which is what SPIDER's
own kernel actually computes against. Under "zero-halo" EVERY domain cell is
recomputed each sweep (no frozen band at all): the array is conceptually
padded with `radius` zeros on every side and the stencil is applied to every
original-domain position, reading zero for whichever neighbor offsets land
in the padding. `reference_stencil`'s `_sweep_zero_halo` materializes this
via `np.pad(u, radius, mode="constant")` followed by offset-sliced
accumulation into the zero-padded scratch; `NumpyStencil`'s own
`_numpy_impl_sweep_zero_halo` is independently coded again (per the
reference-independence rule below): `sliding_window_view` over that SAME
zero-padded array, contracted against the dense kernel via `np.tensordot` --
the technique `_numpy_impl_sweep_fixed` also uses, but applied to the padded
array so the window position axis covers the FULL domain instead of only
the unpadded interior.

The correctness gate always compares the FULL grid_shape array under the
SAME convention the implementation declared (the harness never crops), so
an implementation whose kernel does not actually implement the convention
it (or this domain, by default) claims still fails honestly. The cost model
is UNCHANGED by this hook: `_cost_stencil` still counts `cells * timesteps`
(the FULL grid_shape cell count) regardless of boundary convention -- it
does NOT subtract the width-radius band that a "fixed" run leaves untouched
(this is exact, not an overcount, for "periodic" and "zero-halo", both of
which update every cell every sweep), so GCUP/s stays comparable in
denominator across all three conventions at the cost of slightly
overcounting a "fixed" run's true update count by that band; this is
disclosed via `describe()`'s `boundary` field, never silently normalized
away.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .. import workload
from ..harness import Timer

KERNELS = ["stencil"]
# declared but not yet implemented here; listed so `--list` can say so honestly
PLANNED = ["lattice-boltzmann", "fdtd-seismic"]

ITEMSIZE = {"fp64": 8, "fp32": 4, "fp16": 2, "tf32": 4}

# AN5D's own named 27-point 3D kernel is a division-free variant of box3d1r --
# same support/shape, per spec.yaml's variant-1 `suite` field.
ALIASES = {"j3d27pt": "box3d1r"}

# spec.yaml variant-1 grid_sizes (verbatim): 2D 16384x16384; 3D 512^3, with a
# "small-class" 256^3 alternative for resource-constrained targets. 1D size is
# carried from variant 2 (the only place a 1D grid size is given).
_GRID_1D = (10_240_000,)
_GRID_2D = (16384, 16384)
_GRID_3D = (512, 512, 512)
_GRID_3D_SMALL = (256, 256, 256)

# T=1000 back-to-back sweeps is AN5D's primary-benchmark convention, which
# spec.yaml's variant 1 (stencil-cpu-gpu-kernel-fp64) adopts as its default.
# Variant 2 (Tensor-Core matmul reformulations) uses different T per its own
# CLI convention (10240 for 2D, 100000 for 1D, 1000 for 3D) -- callers wanting
# that protocol must pass `timesteps=` explicitly; a bare shape name alone
# does not carry which variant asked for it.
DEFAULT_TIMESTEPS = 1000

_SHAPE_RE = re.compile(r"^(star|box)?(\d+)d(\d+)r$")


def _parse_shape_name(name: str) -> tuple[str, int, int]:
    m = _SHAPE_RE.match(name)
    if not m:
        raise ValueError(
            f"stencil: unrecognized shape id {name!r}; expected an AN5D-convention "
            "name like star2d1r, box2d1r, star2d3r, box2d3r, star3d1r, box3d1r, "
            "1d1r, 1d2r (rtm3d25p8th and other application-specific high-order "
            "stencils are not implemented here -- see PLANNED)")
    kind = m.group(1) or "star"  # 1D star/box coincide (2*1*r+1 == (2r+1)^1)
    dims = int(m.group(2))
    radius = int(m.group(3))
    return kind, dims, radius


def _support_offsets(kind: str, dims: int, radius: int) -> list[tuple[int, ...]]:
    """
    star-r: center + each axis independently displaced by 1..r, both signs
            -> 2*dims*r + 1 points (spec.yaml `operation` field, literally).
    box-r : full (2r+1)^dims Chebyshev ball around the center.
    """
    center = tuple([0] * dims)
    if kind == "box":
        return list(itertools.product(range(-radius, radius + 1), repeat=dims))
    offsets = [center]
    for axis in range(dims):
        for k in range(1, radius + 1):
            for sign in (1, -1):
                o = [0] * dims
                o[axis] = sign * k
                offsets.append(tuple(o))
    return offsets


def _build_weights(offsets: list[tuple[int, ...]], dims: int) -> dict[tuple[int, ...], float]:
    """
    Synthetic coefficient set, generic over shape/dims/radius. Chosen to satisfy
    two properties simultaneously:

      1. sum(|w_d|) == 1 exactly, so the sweep is non-expansive in the max-norm
         (||U_new||_inf <= ||U||_inf) regardless of sign pattern -- bounded for
         any T, no NaN/Inf risk from a synthetic weight set that was never meant
         to model a specific physical PDE.
      2. genuine sign variation (center weight positive, neighbor weights split
         with alternating sign by axis direction, forward/backward), so the
         correctness gate actually exercises cancellation rather than summing
         all-positive terms -- this is what makes the cancellation-robust
         `scale` (see reference_stencil) a meaningful check and not a no-op.

    Half the weight mass sits on the center point; the other half is split
    evenly across the non-center support points.
    """
    center = tuple([0] * dims)
    non_center = [o for o in offsets if o != center]
    weights: dict[tuple[int, ...], float] = {center: 0.5}
    if not non_center:
        weights[center] = 1.0
        return weights
    per_neighbor = 0.5 / len(non_center)
    for o in non_center:
        # sign follows the direction of the first nonzero coordinate, so
        # "forward" neighbors and "backward" neighbors partially cancel --
        # a discrete-derivative-like pattern, not a plain positive average.
        first_nonzero = next(c for c in o if c != 0)
        sign = 1.0 if first_nonzero > 0 else -1.0
        weights[o] = sign * per_neighbor
    return weights


def _default_grid_shape(dims: int, small: bool = False) -> tuple[int, ...]:
    if dims == 1:
        return _GRID_1D
    if dims == 2:
        return _GRID_2D
    if dims == 3:
        return _GRID_3D_SMALL if small else _GRID_3D
    raise ValueError(f"stencil: no spec-given grid size for dims={dims}")


# ------------------------------------------------------------------ workload
@dataclass
class StencilWorkload:
    """
    A dense D-dimensional grid + a stencil shape + a timestep count. `timesteps`
    is the number of sweeps performed inside ONE timed `impl.run()` call (the
    spec bulk-times all T sweeps as a single interval, never per-sweep -- see
    spec.yaml protocol.timing_scope) and is an explicit field here (not a
    hardcoded harness constant) precisely because the surveyed papers disagree
    on T: AN5D times T=1000, SPIDER's own family times T=10240 (2D) or T=100000
    (1D). It is echoed in describe() so every result record states unambiguously
    how many sweeps its throughput number was computed over.
    """

    name: str
    kind: str                       # "star" | "box"
    dims: int
    radius: int
    grid_shape: tuple[int, ...]
    timesteps: int
    precision: str = "fp64"
    boundary: str = "periodic"      # spec allows periodic wrap or fixed halo of width r
    seed: int = 20260806
    source: str = "synthetic"       # every workload here is synthetic random-fill

    offsets: list[tuple[int, ...]] = field(init=False, repr=False)
    weights: dict[tuple[int, ...], float] = field(init=False, repr=False)

    def __post_init__(self):
        self.grid_shape = tuple(int(s) for s in self.grid_shape)
        if len(self.grid_shape) != self.dims:
            raise ValueError(
                f"stencil {self.name!r}: grid_shape {self.grid_shape} has "
                f"{len(self.grid_shape)} dims, shape id implies {self.dims}")
        self.offsets = _support_offsets(self.kind, self.dims, self.radius)
        self.weights = _build_weights(self.offsets, self.dims)

    @property
    def cells(self) -> int:
        return int(np.prod(self.grid_shape))

    @property
    def support_size(self) -> int:
        return len(self.offsets)

    def initial_field(self, dtype=np.float64) -> np.ndarray:
        """U(0,1) random fill -- no cherry-picked "nice" data, per spec.yaml's
        dense_operand field (both AN5D's and ConvStencil's harnesses default to
        random fill)."""
        rng = np.random.default_rng(self.seed)
        return rng.uniform(0.0, 1.0, size=self.grid_shape).astype(dtype)

    def dense_kernel(self, dtype=np.float64, *, flip: bool = False) -> np.ndarray:
        """
        Dense (2r+1)^dims array built from the sparse offset->weight map, for
        implementations that want a library convolution/correlation call
        instead of an explicit offset loop.

        flip=False (default): kernel[radius+o] = weights[o] -- CROSS-CORRELATION
        layout (out[p] = sum_c k[c]*u[p+c-radius]). This is what PyTorch's
        conv1d/2d/3d actually compute (they do not flip the kernel despite the
        name) and what scipy.ndimage.correlate computes.

        flip=True: kernel[radius-o] = weights[o] -- point-reflected, TRUE
        convolution layout (out[p] = sum_c k[c]*u[p-c+radius]), what
        scipy.ndimage.convolve computes.

        Both flavors reproduce the identical U_out[p] = sum_d weights[d]*U_in[p+d]
        update (verified against the offset-loop sweep; only the array layout
        handed to each library differs).
        """
        size = 2 * self.radius + 1
        k = np.zeros((size,) * self.dims, dtype=dtype)
        for o, w in self.weights.items():
            idx = tuple((self.radius - c if flip else self.radius + c) for c in o)
            k[idx] = w
        return k

    def describe(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "shape": f"{self.kind}{self.dims}d{self.radius}r",
            "kind": self.kind,
            "dims": self.dims,
            "radius": self.radius,
            "support_points": self.support_size,
            "grid_shape": list(self.grid_shape),
            "cells": self.cells,
            # explicit per the harness requirement: a throughput number is
            # meaningless without stating how many sweeps it was computed over.
            "timesteps_per_call": self.timesteps,
            "boundary": (
                (
                    f"{self.boundary} wrap, halo width {self.radius}; halo fill is "
                    "implicit in the periodic-wrap sweep (np.roll / mode='wrap') and "
                    "is NOT a separate timed step"
                    if self.boundary == "periodic" else
                    f"fixed halo of width {self.radius}: only points at distance "
                    f">= {self.radius} from every edge are updated each sweep; the "
                    "boundary band keeps the seeded initial field's own values for "
                    "the whole T-sweep run (no separate halo array, no per-sweep "
                    "refresh)"
                    if self.boundary == "fixed" else
                    f"zero-halo of width {self.radius}: the array is conceptually "
                    "padded with zeros on every side and EVERY domain cell is "
                    "recomputed each sweep (no frozen band), reading 0 for whichever "
                    "neighbor offsets fall outside the domain"
                ) + " -- this is the workload's own DEFAULT; an implementation may "
                "select any of the three conventions per-run via params['boundary'] "
                "in prepare() (see kernelbench/domains/stencil.py module docstring), "
                "independent of this field. Single-node kernel-only claim either "
                "way, no distributed halo exchange (spec.yaml inputs.boundary)."
            ),
            "precision": self.precision,
            "init": "U(0,1) random fill, no cherry-picked data (spec.yaml dense_operand)",
            "seed": self.seed,
        }


# ------------------------------------------------------------------ workloads
SMOKE = [
    ("smoke-star2d1r", dict(kind="star", dims=2, radius=1, grid_shape=(256, 256), timesteps=5)),
    ("smoke-star3d1r", dict(kind="star", dims=3, radius=1, grid_shape=(32, 32, 32), timesteps=5)),
    ("smoke-box3d1r", dict(kind="box", dims=3, radius=1, grid_shape=(24, 24, 24), timesteps=5)),
]


def smoke_workloads():
    return [StencilWorkload(name=n, **kw) for n, kw in SMOKE]


def load_workload(name: str, *, timesteps: int | None = None,
                   grid_shape: tuple[int, ...] | None = None) -> StencilWorkload:
    """
    Build a spec-sized workload from an AN5D-convention shape id (the naming
    variant 1 and variant 2 of spec.yaml both use, so this one loader covers
    the union): star2d1r, box2d1r, star2d3r, box2d3r, star3d1r, box3d1r, 1d1r,
    1d2r, plus the box3d1r-equivalent alias j3d27pt. Append "-256" to a 3D name
    (e.g. "star3d1r-256") for the spec's small-class 256^3 grid instead of the
    default 512^3.

    Grid sizes come verbatim from spec.yaml's variant-1 `grid_sizes` field.
    `timesteps` defaults to T=1000 (variant 1's AN5D-derived default); pass an
    explicit value to reproduce a different variant's protocol -- see
    DEFAULT_TIMESTEPS's docstring note on why this is not inferred from the
    bare shape name.
    """
    base = ALIASES.get(name, name)
    small = base.endswith("-256")
    if small:
        base = base[: -len("-256")]
    kind, dims, radius = _parse_shape_name(base)
    shape = grid_shape or _default_grid_shape(dims, small=small)
    T = timesteps if timesteps is not None else DEFAULT_TIMESTEPS
    return StencilWorkload(name=name, kind=kind, dims=dims, radius=radius,
                            grid_shape=tuple(shape), timesteps=T)


# ----------------------------------------------------------------- cost rule
def _cost_stencil(w: StencilWorkload, params: dict) -> tuple[int, int]:
    """
    GCell-updates/s per spec.yaml metric.primary, literally:
    `grid_points * T / elapsed_s / 1e9`. `elapsed_s` (computed by the harness)
    is the wall/device time of ONE impl.run() call, which performs all T
    sweeps as a single bulk-timed interval -- so work_count must be the FULL
    T-sweep cell-update count, not a per-sweep count, to match what is
    actually being divided by.

    This is deliberately NOT a flop count: spec.yaml's notes_on_fairness
    section refuses GFLOP/s as a primary metric because the FLOP-per-update
    convention differs per paper (PERKS: 17 FLOP/cell for one specific 9-point
    kernel; ConvStencil: an unresolved per-shape multiplier) and is therefore
    not comparable across papers without a stated convention. GCell-updates/s
    depends only on grid size and T, so it is registered as the ONLY primary
    unit here.

    `bytes` follows metric.secondary's own GB/s formula: 2*grid_points*itemsize
    (read+write of the ping-pong buffers) per sweep, summed over T sweeps --
    a compulsory-traffic LOWER BOUND, not measured traffic.
    """
    timesteps = int(params.get("timesteps", w.timesteps))
    itemsize = ITEMSIZE[params.get("precision", w.precision)]
    cells = w.cells
    work = cells * timesteps
    byts = 2 * cells * itemsize * timesteps
    return work, int(byts)


workload.register_cost("stencil", _cost_stencil, unit="GCUP/s")


# ----------------------------------------------------------------- sweep core
def _sweep(u: np.ndarray, offsets: list[tuple[int, ...]],
           weights: dict[tuple[int, ...], float], dims: int,
           out: np.ndarray | None = None) -> np.ndarray:
    """
    One periodic-wrap stencil sweep: out[p] = sum_d weights[d] * u[p+d].

    `out`, when given, is written in place (filled then accumulated into) so
    the ping-pong buffer discipline holds: the OUTER grid buffers are
    allocated once by the caller (impl.prepare()) and never reallocated across
    timesteps here. The per-offset `np.roll` / elementwise-multiply temporaries
    are ordinary numpy working memory, the same category of allocation scipy's
    own sparse matmul (`A @ B`) performs internally in the sparse domain's CPU
    implementations -- not something this vectorized path avoids either.
    """
    axes = tuple(range(dims))
    if out is None:
        out = np.zeros_like(u)
    else:
        out.fill(0.0)
    for o in offsets:
        w = weights[o]
        if w == 0.0:
            continue
        shifted = np.roll(u, shift=tuple(-c for c in o), axis=axes)
        out += w * shifted
    return out


def _sweep_fixed(u: np.ndarray, offsets: list[tuple[int, ...]],
                  weights: dict[tuple[int, ...], float], dims: int, radius: int,
                  out: np.ndarray | None = None) -> np.ndarray:
    """
    One fixed-halo stencil sweep (see module docstring "Boundary convention
    hook"): out[p] = sum_d weights[d] * u[p+d] for every p at distance >=
    radius from every edge; the width-radius boundary band is copied
    UNCHANGED from `u` -- no separate halo array, no per-sweep refresh, so a
    band cell keeps whatever value it held when this function first saw it
    (the seeded initial field's own boundary value, for sweep 1 -- and
    therefore for every subsequent sweep too, since this function never
    writes to it).

    Deliberately a DIFFERENT code path from `_sweep` (periodic/np.roll):
    every interior access here stays in-bounds by construction (no
    wraparound is needed or performed), so this is direct sliced
    accumulation, not a shift-based one. `NumpyStencil`'s own fixed-boundary
    sweep (`_numpy_impl_sweep_fixed`, further below) is independently coded
    again from this one, per DOMAIN_GUIDE.md's reference-independence rule.
    """
    shape = u.shape
    if out is None:
        out = np.empty_like(u)
    np.copyto(out, u)  # boundary band settled here; interior overwritten below
    interior_shape = tuple(s - 2 * radius for s in shape)
    if any(n <= 0 for n in interior_shape):
        return out  # grid too small for this radius along some axis -- all boundary
    acc = np.zeros(interior_shape, dtype=u.dtype)
    for o in offsets:
        w = weights[o]
        if w == 0.0:
            continue
        src = tuple(slice(radius + c, radius + c + n) for c, n in zip(o, interior_shape))
        acc += w * u[src]
    dst = tuple(slice(radius, radius + n) for n in interior_shape)
    out[dst] = acc
    return out


def _sweep_zero_halo(u: np.ndarray, offsets: list[tuple[int, ...]],
                      weights: dict[tuple[int, ...], float], dims: int, radius: int,
                      out: np.ndarray | None = None) -> np.ndarray:
    """
    One zero-halo stencil sweep (see module docstring "Boundary convention
    hook"): out[p] = sum_d weights[d] * u_ext[p+d] for EVERY domain cell p,
    where u_ext is `u` conceptually extended by `radius` zeros on every side
    -- no frozen band (unlike `_sweep_fixed`), every cell is recomputed every
    sweep, using 0 for whichever neighbor offsets fall outside the domain.
    This is SPIDER's own convention (PPoPP 2026): a zero-valued EXTERNAL
    halo, distinct from both "periodic" (wraps around) and this domain's
    "fixed" (freezes an IN-ARRAY band to its t=0 value and never recomputes
    it) -- see `artifacts/stencil/spider/STATUS.md`.

    Implemented by materializing the zero-padded array explicitly
    (`np.pad(u, radius, mode="constant")`, value 0.0) and slicing offset
    windows INTO it -- a THIRD code path, independent of both `_sweep`
    (np.roll-based) and `_sweep_fixed` (direct sliced accumulation on the
    unpadded array, no padding at all).
    """
    padded = np.pad(u, radius, mode="constant", constant_values=0.0)
    if out is None:
        out = np.zeros_like(u)
    else:
        out.fill(0.0)
    for o in offsets:
        w = weights[o]
        if w == 0.0:
            continue
        # domain cell p sits at padded index p+radius; its neighbor at
        # offset o sits at padded index p+radius+o -- so slicing `padded`
        # starting at (radius+o) for u.shape[i] elements along each axis
        # yields, at position p, exactly padded[p+radius+o], which equals
        # u[p+o] when that neighbor is in-bounds and 0 when it fell in the
        # zero padding.
        src = tuple(slice(radius + c, radius + c + n) for c, n in zip(o, u.shape))
        out += w * padded[src]
    return out


# ----------------------------------------------------------------- reference
def reference_stencil(workload_: StencilWorkload, params: dict):
    """
    fp64 reference: the identical T-sweep recursion the timed implementation
    performs, starting from the same U(0,1) initial field.

    `scale` runs the SAME recursion in parallel with |weights| applied to the
    abs-field (seeded from |U_0|). This is the T-fold generalization of the
    sparse domain's |A|@|B| cancellation-robust denominator: a stencil sweep is
    applied T times, so an error (or a legitimate near-zero result from
    cancelling terms) at sweep t can be amplified or attenuated by sweeps
    t+1..T. A single-sweep magnitude bound is not enough to certify T
    compounded sweeps; the scale has to be carried through the same number of
    applications as the result it is meant to bound.

    `boundary` (module docstring "Boundary convention hook"): read from
    `params["boundary"]`, falling back to `workload_.boundary` (the
    workload's own default, "periodic") when the caller/impl did not set it.
    Both the result `u` and its companion scale `s` run under the SAME
    convention, sweep for sweep -- a "fixed" boundary band is frozen for both
    equally, so `s` still legitimately bounds `u`'s error there too (both are
    pinned to their own t=0 value, not recomputed); under "zero-halo" both
    `u` and `s` are recomputed at every cell every sweep against the SAME
    zero-padded extension, so `s` bounds `u`'s error there too.
    """
    timesteps = int(params.get("timesteps", workload_.timesteps))
    boundary = params.get("boundary", workload_.boundary)
    offsets, dims, radius = workload_.offsets, workload_.dims, workload_.radius
    u = workload_.initial_field(dtype=np.float64)
    s = np.abs(u)
    absw = {o: abs(w) for o, w in workload_.weights.items()}
    if boundary == "periodic":
        for _ in range(timesteps):
            u = _sweep(u, offsets, workload_.weights, dims)
            s = _sweep(s, offsets, absw, dims)
    elif boundary == "fixed":
        for _ in range(timesteps):
            u = _sweep_fixed(u, offsets, workload_.weights, dims, radius)
            s = _sweep_fixed(s, offsets, absw, dims, radius)
    elif boundary == "zero-halo":
        for _ in range(timesteps):
            u = _sweep_zero_halo(u, offsets, workload_.weights, dims, radius)
            s = _sweep_zero_halo(s, offsets, absw, dims, radius)
    else:
        raise ValueError(
            f"stencil: unknown boundary convention {boundary!r} for "
            f"{workload_.name!r} (expected 'periodic', 'fixed', or "
            "'zero-halo', spec.yaml inputs.boundary)")
    return u, s


REFERENCES = {"stencil": reference_stencil}
CORRECTNESS_MODE = {"stencil": "max_scaled_err"}
DEFAULT_PRECISION = {"stencil": "fp64"}


def _numpy_impl_sweep_fixed(u: np.ndarray, dense_kernel: np.ndarray, radius: int,
                             dims: int, out: np.ndarray | None = None) -> np.ndarray:
    """
    NumpyStencil's OWN fixed-boundary sweep -- a code path INDEPENDENT of
    `reference_stencil`'s `_sweep_fixed` (DOMAIN_GUIDE.md's reference-
    independence rule: same formula, never the same helper). Instead of an
    offset loop with per-offset sliced accumulation, this gathers every
    (2r+1)^dims neighborhood window in ONE `sliding_window_view` call and
    contracts all of them against the dense kernel array
    (`StencilWorkload.dense_kernel(flip=False)`, the cross-correlation layout
    documented there: `dense_kernel[radius+o] == weights[o]`) via a single
    `np.tensordot` -- a different numpy primitive entirely, not a
    re-typed copy of the reference's loop.

    `sliding_window_view(u, (2r+1,)*dims)`'s first `dims` axes are the window
    POSITION (one per valid top-left corner, length N_i-2r along axis i) and
    its last `dims` axes are the window CONTENT (length 2r+1 each); the
    position axis for a window starting at offset p corresponds to the
    stencil's OUTPUT point p+radius, so contracting the content axes against
    `dense_kernel` produces exactly the interior region
    `[radius, N_i-radius)` per axis -- the boundary band (never covered by
    any window's position axis) is left at whatever `out` already held,
    which this function fills via `np.copyto(out, u)` first, matching
    `_sweep_fixed`'s "no separate halo array, no refresh" contract.
    """
    if out is None:
        out = np.empty_like(u)
    np.copyto(out, u)  # boundary band settled here; interior overwritten below
    window = 2 * radius + 1
    if any(s < window for s in u.shape):
        return out  # grid too small for this radius along some axis -- all boundary
    windows = sliding_window_view(u, (window,) * dims)
    content_axes = tuple(range(dims, 2 * dims))
    interior = np.tensordot(windows, dense_kernel, axes=(content_axes, tuple(range(dims))))
    dst = tuple(slice(radius, radius + n) for n in interior.shape)
    out[dst] = interior
    return out


def _numpy_impl_sweep_zero_halo(u: np.ndarray, dense_kernel: np.ndarray, radius: int,
                                 dims: int, out: np.ndarray | None = None) -> np.ndarray:
    """
    NumpyStencil's OWN zero-halo sweep -- a code path INDEPENDENT of
    `reference_stencil`'s `_sweep_zero_halo` (DOMAIN_GUIDE.md's reference-
    independence rule: same formula, never the same helper). Pads `u` with
    `radius` zeros on every axis (`np.pad(..., mode="constant")`, same
    padding rule as the reference, but a separately-written call, not a
    shared helper), then gathers every (2r+1)^dims neighborhood window over
    the PADDED array in ONE `sliding_window_view` call and contracts all of
    them against the dense kernel (`StencilWorkload.dense_kernel(flip=False)`,
    cross-correlation layout, `dense_kernel[radius+o] == weights[o]`) via a
    single `np.tensordot` -- the same technique `_numpy_impl_sweep_fixed`
    uses, mirrored onto the zero-padded array instead of the raw one.

    `sliding_window_view(padded, (2r+1,)*dims)`'s position-axis length along
    axis i is `padded.shape[i] - (2r+1) + 1 == (u.shape[i]+2r) - 2r - 1 + 1
    == u.shape[i]` -- i.e. the window position axis covers the FULL original
    domain exactly (not just the interior, unlike the "fixed" convention's
    version above), because the zero padding supplies whatever neighbor
    values would otherwise fall outside the array. No band copy, no
    cropping: every output cell comes from a window contraction.
    """
    if out is None:
        out = np.empty_like(u)
    padded = np.pad(u, radius, mode="constant", constant_values=0.0)
    window = 2 * radius + 1
    windows = sliding_window_view(padded, (window,) * dims)
    content_axes = tuple(range(dims, 2 * dims))
    result = np.tensordot(windows, dense_kernel, axes=(content_axes, tuple(range(dims))))
    np.copyto(out, result)
    return out


# --------------------------------------------------------------------- impls
class NumpyStencil:
    """
    Vectorized numpy sweep with explicit double buffering.

    prepare() allocates the pristine initial field AND both ping-pong grid
    buffers once. run() resets buf[0] from the pristine field (an in-place
    `np.copyto`, not a reallocation) and then performs `timesteps` sweeps,
    swapping which buffer is "current" each step -- so every call to run()
    (correctness check, warmup, and every timed rep alike) performs the exact
    same, already-gated T-sweep computation, rather than continuing to evolve
    the field further on each successive call.

    `boundary` (module docstring "Boundary convention hook"): read from
    `params["boundary"]`, falling back to `workload_.boundary` ("periodic")
    when unset -- same lookup `reference_stencil` performs on the identical
    `params` dict, so both agree on which convention a given run gates
    against. The "fixed" branch uses `_numpy_impl_sweep_fixed`, and the
    "zero-halo" branch uses `_numpy_impl_sweep_zero_halo` -- both
    independently-coded sweeps (see each function's own docstring), never
    `reference_stencil`'s own `_sweep_fixed` / `_sweep_zero_halo`.
    """

    name = "numpy-stencil"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        self.precision = precision
        self.dtype = np.float64 if precision == "fp64" else np.float32

    def prepare(self, workload_: StencilWorkload, params: dict):
        u0 = workload_.initial_field(dtype=self.dtype)
        bufs = [np.empty_like(u0), np.empty_like(u0)]
        weights = {o: self.dtype(w) for o, w in workload_.weights.items()}
        boundary = params.get("boundary", workload_.boundary)
        if boundary not in ("periodic", "fixed", "zero-halo"):
            raise ValueError(
                f"stencil: unknown boundary convention {boundary!r} for "
                f"{workload_.name!r} (expected 'periodic', 'fixed', or "
                "'zero-halo', spec.yaml inputs.boundary)")
        # both "fixed" and "zero-halo" impl sweeps contract a sliding-window
        # view against this same dense kernel array -- only what they window
        # (unpadded array vs. zero-padded array) differs.
        dense_kernel = (workload_.dense_kernel(dtype=self.dtype, flip=False)
                        if boundary in ("fixed", "zero-halo") else None)
        return {
            "u0": u0, "bufs": bufs,
            "offsets": workload_.offsets, "weights": weights, "dims": workload_.dims,
            "timesteps": int(params.get("timesteps", workload_.timesteps)),
            "boundary": boundary, "radius": workload_.radius, "dense_kernel": dense_kernel,
        }

    def run(self, h):
        bufs = h["bufs"]
        np.copyto(bufs[0], h["u0"])
        offsets, weights, dims = h["offsets"], h["weights"], h["dims"]
        cur, nxt = 0, 1
        if h["boundary"] == "periodic":
            for _ in range(h["timesteps"]):
                _sweep(bufs[cur], offsets, weights, dims, out=bufs[nxt])
                cur, nxt = nxt, cur
        elif h["boundary"] == "fixed":
            dense_kernel, radius = h["dense_kernel"], h["radius"]
            for _ in range(h["timesteps"]):
                _numpy_impl_sweep_fixed(bufs[cur], dense_kernel, radius, dims, out=bufs[nxt])
                cur, nxt = nxt, cur
        else:  # "zero-halo"
            dense_kernel, radius = h["dense_kernel"], h["radius"]
            for _ in range(h["timesteps"]):
                _numpy_impl_sweep_zero_halo(bufs[cur], dense_kernel, radius, dims, out=bufs[nxt])
                cur, nxt = nxt, cur
        return bufs[cur]

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


class ScipyConvolveStencil:
    """
    Same math via scipy.ndimage.convolve with a dense kernel array and
    periodic ('wrap') boundary mode -- a genuinely different code path from
    the offset-loop sweep above (exercises the correctness gate against real
    implementation diversity, not just the same helper called twice).

    Same double-buffering discipline as NumpyStencil: pristine field + two
    grid buffers allocated once in prepare(); run() writes into the
    preallocated `output=` buffer each sweep, nothing reallocated per step.
    """

    name = "scipy-convolve-stencil"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        self.precision = precision
        self.dtype = np.float64 if precision == "fp64" else np.float32

    def prepare(self, workload_: StencilWorkload, params: dict):
        from scipy import ndimage  # local import: only this impl needs it
        u0 = workload_.initial_field(dtype=self.dtype)
        kernel = workload_.dense_kernel(dtype=self.dtype, flip=True)
        bufs = [np.empty_like(u0), np.empty_like(u0)]
        return {
            "u0": u0, "bufs": bufs, "kernel": kernel, "ndimage": ndimage,
            "timesteps": int(params.get("timesteps", workload_.timesteps)),
        }

    def run(self, h):
        bufs, kernel, ndimage = h["bufs"], h["kernel"], h["ndimage"]
        np.copyto(bufs[0], h["u0"])
        cur, nxt = 0, 1
        for _ in range(h["timesteps"]):
            ndimage.convolve(bufs[cur], kernel, output=bufs[nxt], mode="wrap")
            cur, nxt = nxt, cur
        return bufs[cur]

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


CPU_IMPLS = {
    "stencil": {
        "numpy-stencil": NumpyStencil,
        "scipy-convolve-stencil": ScipyConvolveStencil,
    }
}


def cuda_impls():
    from ..impls import gpu_cuda as g
    return {"stencil": {"torch-conv-stencil": g.TorchStencilConv}}
