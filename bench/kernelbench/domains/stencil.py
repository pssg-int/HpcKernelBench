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
  PAPER-NATIVE— each integrated paper's own metric (AN5D GFLOP/s, the Tensor-
                Core lineage's GStencil/s + execution time, SPIDER's plotted
                normalization), attached as metrics["paper_native"] alongside
                GCUP/s -- see _native_stencil
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

import dataclasses
import itertools
import os
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .. import workload
from ..harness import Timer

KERNELS = ["stencil"]
# declared but not yet implemented here; listed so `--list` can say so honestly
PLANNED = ["lattice-boltzmann", "fdtd-seismic"]

ITEMSIZE = {"fp64": 8, "fp32": 4, "fp16": 2, "tf32": 4}

# spec.yaml variant-1 grid_sizes (verbatim): 2D 16384x16384; 3D 512^3, with a
# "small-class" 256^3 alternative for resource-constrained targets. 1D size is
# carried from variant 2 (the only place a 1D grid size is given).
_GRID_1D = (10_240_000,)
_GRID_2D = (16384, 16384)
_GRID_3D = (512, 512, 512)
_GRID_3D_SMALL = (256, 256, 256)

# T=1000 back-to-back sweeps is AN5D's primary-benchmark convention, which
# spec.yaml's variant 1 (stencil-cpu-gpu-kernel-fp64) adopts as its default.
DEFAULT_TIMESTEPS = 1000

# Per-variant defaults by dimensionality: (grid_shape, timesteps). Variant 2
# (Tensor-Core lineage) uses its papers' shared sizes -- spec.yaml
# stencil-tcu-matmul-kernel-fp16 inputs.named_kernels / protocol.iteration_count:
# 1D 10,240,000 x T=100,000; 2D 10240^2 x T=10,240; 3D 1024^3 x T=1,024.
VARIANT_DEFAULTS = {
    "stencil-cpu-gpu-kernel-fp64": {
        1: (_GRID_1D, DEFAULT_TIMESTEPS),
        2: (_GRID_2D, DEFAULT_TIMESTEPS),
        3: (_GRID_3D, DEFAULT_TIMESTEPS),
    },
    "stencil-tcu-matmul-kernel-fp16": {
        1: ((10_240_000,), 100_000),
        2: ((10240, 10240), 10_240),
        3: ((1024, 1024, 1024), 1_024),
    },
}

# AN5D's five named kernels (CGO'20 Table 3), transcribed from the artifact's
# own benchmark sources at the commit in artifacts/stencil/an5d/
# source.provenance (j2d5pt.c, j2d9pt.c, j2d9pt-gol.c, gradient2d.c,
# j3d27pt.c, `#pragma scop` region). Offsets are (i, j[, k]) in array-axis
# order. The sources write every coefficient as a float literal (`5.1f`), so
# even AN5D's double build multiplies by the float-rounded value; _f32 keeps
# that. The four linear kernels divide the weighted sum by an integer AFTER
# summing, and the reference does the same.
#
# NOTE: these coefficients sum to less than the divisor (j2d5pt 0.42,
# j2d9pt/-gol 0.70, j3d27pt 0.15 per step), so the field decays toward zero;
# at AN5D's own T=1000, j2d5pt and j3d27pt underflow to exactly 0 in fp64.
# reference_stencil warns when that makes the gate vacuous -- the coefficients
# are the paper's and are kept as-is.
def _f32(x: float) -> float:
    return float(np.float32(x))


NAMED = {
    "j2d5pt": dict(dims=2, radius=1, kind="star", divisor=118, coeffs={
        (-1, 0): 5.1, (0, -1): 12.1, (0, 0): 15.0, (0, 1): 12.2, (1, 0): 5.2}),
    "j2d9pt": dict(dims=2, radius=2, kind="star", divisor=118, coeffs={
        (-2, 0): 7.1, (-1, 0): 5.1, (0, -2): 9.2, (0, -1): 12.1, (0, 0): 15.0,
        (0, 1): 12.2, (0, 2): 9.1, (1, 0): 5.2, (2, 0): 7.2}),
    "j2d9pt-gol": dict(dims=2, radius=1, kind="box", divisor=118, coeffs={
        (-1, -1): 7.1, (-1, 0): 5.1, (-1, 1): 9.2, (0, -1): 12.1, (0, 0): 15.0,
        (0, 1): 12.2, (1, -1): 9.1, (1, 0): 5.2, (1, 1): 7.2}),
    "j3d27pt": dict(dims=3, radius=1, kind="box", divisor=159, coeffs={
        (-1, 0, 0): 1.500, (-1, -1, -1): 0.500, (-1, -1, 0): 0.700,
        (-1, -1, 1): 0.900, (-1, 0, -1): 1.200, (-1, 0, 1): 1.201,
        (-1, 1, -1): 0.901, (-1, 1, 0): 0.701, (-1, 1, 1): 0.501,
        (0, 0, 0): 1.510, (0, -1, -1): 0.510, (0, -1, 0): 0.710,
        (0, -1, 1): 0.910, (0, 0, -1): 1.210, (0, 0, 1): 1.211,
        (0, 1, -1): 0.911, (0, 1, 0): 0.711, (0, 1, 1): 0.511,
        (1, 0, 0): 1.520, (1, -1, -1): 0.520, (1, -1, 0): 0.720,
        (1, -1, 1): 0.920, (1, 0, -1): 1.220, (1, 0, 1): 1.221,
        (1, 1, -1): 0.921, (1, 1, 0): 0.721, (1, 1, 1): 0.521}),
    # u + 1 / sqrt(eps + sum over the 4 axis neighbors of (u - u_nb)^2):
    # nonlinear, so it has no weights and weight-taking adapters must refuse it.
    "gradient2d": dict(dims=2, radius=1, kind="gradient", eps=0.0001),
}
for _spec in NAMED.values():
    if "coeffs" in _spec:
        _spec["coeffs"] = {o: _f32(c) for o, c in _spec["coeffs"].items()}
    if "eps" in _spec:
        _spec["eps"] = _f32(_spec["eps"])

# Workload-name syntax (see load_workload): <shape>[-256][@AxB[xC]][:T=<n>]
_NAME_RE = re.compile(r"^(?P<base>[A-Za-z0-9-]+?)(?P<small>-256)?"
                      r"(?:@(?P<grid>\d+(?:x\d+)*))?(?::T=(?P<T>\d+))?$")

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
    three properties simultaneously:

      1. sum(w_d) == 1 and every w_d > 0 (a convex combination), so the sweep
         is non-expansive in the max-norm (||U_new||_inf <= ||U||_inf) AND
         preserves a constant field exactly. The solution therefore stays
         O(mean(U_0)) ~ 0.5 for any T -- it can neither blow up nor decay
         toward zero. This matters for the correctness gate: an earlier
         version used signed weights summing to 0.5, so the true answer
         shrank like 0.5^T (~1e-153 at T=1000, exactly 0.0 at T=10240) while
         the gate's scale stayed O(1), and an all-zeros output passed the
         1e-5 tolerance from about T=20 on. Do not reintroduce a weight sum
         below 1 (or a signed set whose sum is not 1).
      2. asymmetry: neighbors in the "forward" direction (first nonzero
         coordinate > 0) carry 3x the weight of "backward" ones, so a
         flipped/reflected kernel or a mis-signed offset in an implementation
         changes the answer and is caught. (Two levels only: it does NOT
         distinguish permutations among equally-weighted offsets, e.g. an
         axis swap on a star -- as before.)
      3. half the total weight sits on the center point; the rest is split
         across the non-center support points in the 3:1 forward:backward
         ratio above.

    Trade-off: with all weights positive the sweep is a smoother, so on a
    SMALL grid at very large T the field mixes toward a constant and
    misplaced-data bugs (e.g. a one-cell shift) stop changing the answer
    (see the flat-field warning in reference_stencil). Gate on a grid/T where
    the field is still varying; zeros and wrong-scale outputs are caught at
    every T.
    """
    center = tuple([0] * dims)
    non_center = [o for o in offsets if o != center]
    weights: dict[tuple[int, ...], float] = {center: 0.5}
    if not non_center:
        weights[center] = 1.0
        return weights
    raw = {o: (1.5 if next(c for c in o if c != 0) > 0 else 0.5) for o in non_center}
    total = sum(raw.values())
    for o in non_center:
        weights[o] = 0.5 * raw[o] / total
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
    kind: str                       # "star" | "box" | "gradient" (AN5D gradient2d)
    dims: int
    radius: int
    grid_shape: tuple[int, ...]
    timesteps: int
    precision: str = "fp64"
    boundary: str = "periodic"      # spec allows periodic wrap or fixed halo of width r
    seed: int = 20260806
    source: str = "synthetic"       # every workload here is synthetic random-fill
    named: str | None = None        # AN5D named kernel id (key of NAMED), else None

    offsets: list[tuple[int, ...]] = field(init=False, repr=False)
    weights: dict[tuple[int, ...], float] = field(init=False, repr=False)

    def __post_init__(self):
        self.grid_shape = tuple(int(s) for s in self.grid_shape)
        if len(self.grid_shape) != self.dims:
            raise ValueError(
                f"stencil {self.name!r}: grid_shape {self.grid_shape} has "
                f"{len(self.grid_shape)} dims, shape id implies {self.dims}")
        if self.named is None:
            self.offsets = _support_offsets(self.kind, self.dims, self.radius)
            self.weights = _build_weights(self.offsets, self.dims)
            return
        spec = NAMED[self.named]
        if "coeffs" in spec:
            # Effective linear weights c/divisor, for implementations that take
            # a weight map. The reference itself sums with the raw coefficients
            # and divides afterwards, as AN5D's source does.
            self.offsets = list(spec["coeffs"])
            self.weights = {o: c / spec["divisor"] for o, c in spec["coeffs"].items()}
        else:
            self.offsets = _support_offsets("star", self.dims, self.radius)
            self.weights = {}

    @property
    def linear(self) -> bool:
        """False for nonlinear named kernels (gradient2d): no weight map exists."""
        return self.named is None or "coeffs" in NAMED[self.named]

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
        if not self.linear:
            raise NotImplementedError(
                f"stencil {self.name!r}: {self.named} is nonlinear (sqrt of "
                "squared differences), not a weighted sum -- no dense kernel exists")
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
            "shape": self.named or f"{self.kind}{self.dims}d{self.radius}r",
            **({"named_kernel": {
                "paper": "AN5D (CGO'20) Table 3; coefficients from the artifact's "
                         f"{self.named}.c",
                "linear": self.linear,
                "divisor": NAMED[self.named].get("divisor"),
            }} if self.named else {}),
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
    # sized for the narrow paper adapters: LoRAStencil runs only star2d3r and
    # needs rows % 32 == 0 and cols % 64 == 0; FlashFFTStencil runs only
    # box2d1r on a square grid whose width is a multiple of 6.
    ("smoke-star2d3r", dict(kind="star", dims=2, radius=3, grid_shape=(64, 128), timesteps=5)),
    ("smoke-box2d1r", dict(kind="box", dims=2, radius=1, grid_shape=(96, 96), timesteps=5)),
    # AN5D named kernel with its paper-native division, small enough to gate
    ("smoke-j2d5pt", dict(kind="star", dims=2, radius=1, grid_shape=(128, 128), timesteps=5,
                          named="j2d5pt")),
]


def smoke_workloads():
    return [StencilWorkload(name=n, **kw) for n, kw in SMOKE]


def load_workload(name: str, *, timesteps: int | None = None,
                   grid_shape: tuple[int, ...] | None = None,
                   variant: str | None = None) -> StencilWorkload:
    """
    Build a workload from a name of the form

        <shape>[-256][@<grid>][:T=<steps>]

    <shape> is an AN5D-convention id (star2d1r, box3d2r, 1d3r, ...: any star
    or box, any dims, any radius) or one of AN5D's named kernels (NAMED:
    j2d5pt, j2d9pt, j2d9pt-gol, gradient2d, j3d27pt). "-256" selects the
    spec's optional small-class 256^3 grid for 3D. "@AxB[xC]" sets the grid
    and ":T=n" the time steps, e.g. "box2d2r@5120x5120:T=10000", so any size
    a paper swept can be run from the CLI (`--matrices`).

    Defaults when the name does not say: the variant's own sizes
    (VARIANT_DEFAULTS; the runner passes `variant`), else variant 1's
    16384^2 / 512^3 and T=1000. Explicit keyword arguments override the name.
    """
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"stencil: cannot parse workload name {name!r}; "
                         "expected <shape>[-256][@AxB[xC]][:T=n]")
    base, small = m.group("base"), bool(m.group("small"))
    named = base if base in NAMED else None
    if named:
        spec = NAMED[named]
        kind, dims, radius = spec["kind"], spec["dims"], spec["radius"]
    else:
        kind, dims, radius = _parse_shape_name(base)

    v_grid, v_T = VARIANT_DEFAULTS.get(variant, {}).get(dims, (None, None))
    if grid_shape is None and m.group("grid"):
        grid_shape = tuple(int(g) for g in m.group("grid").split("x"))
    if grid_shape is None:
        grid_shape = (_GRID_3D_SMALL if small and dims == 3 else
                      v_grid or _default_grid_shape(dims, small=small))
    if timesteps is None and m.group("T"):
        timesteps = int(m.group("T"))
    if timesteps is None:
        timesteps = v_T or DEFAULT_TIMESTEPS
    return StencilWorkload(name=name, kind=kind, dims=dims, radius=radius,
                            grid_shape=tuple(grid_shape), timesteps=timesteps,
                            named=named)


# Paper experiments whose shapes, sizes AND step counts are all stated
# (spec.yaml variant 2 + extended_experiments). `--matrices sweep:<id>`
# expands to the list. Sweeps whose paper leaves T unstated (ConvStencil
# Figure 8, LoRAStencil Figure 9, FlashFFTStencil Figures 7-9) are left out on
# purpose: run them with an explicit ":T=" instead of an invented value.
def _sizes(shapes, grids, T):
    return [f"{sh}@{'x'.join(map(str, g))}:T={T}" for sh in shapes for g in grids]


SWEEPS = {
    # AN5D (CGO'20) §6.1: 16 synthetic shapes + 5 named kernels, 16384^2 / 512^3, T=1000
    "an5d-native": (
        _sizes([f"{k}2d{r}r" for k in ("star", "box") for r in (1, 2, 3, 4)]
               + ["j2d5pt", "j2d9pt", "j2d9pt-gol", "gradient2d"], [(16384, 16384)], 1000)
        + _sizes([f"{k}3d{r}r" for k in ("star", "box") for r in (1, 2, 3, 4)]
                 + ["j3d27pt"], [(512, 512, 512)], 1000)),
    # ConvStencil (PPoPP'24) Table 4; its 3D T is unstated, so 3D is omitted
    "convstencil-native": (
        _sizes(["1d1r", "1d2r"], [(10_240_000,)], 100_000)
        + _sizes(["star2d1r", "box2d1r", "star2d3r", "box2d3r"], [(10240, 10240)], 10_240)),
    # LoRAStencil (SC'24) Table II
    "lorastencil-native": (
        _sizes(["1d1r", "1d2r"], [(10_240_000,)], 10_000)
        + _sizes(["star2d1r", "box2d1r", "star2d3r", "box2d3r"], [(10240, 10240)], 10_240)
        + _sizes(["star3d1r", "box3d1r"], [(1024, 1024, 1024)], 1_024)),
    # FlashFFTStencil (PPoPP'25) Table 3
    "flashfftstencil-native": (
        _sizes(["1d1r", "1d2r", "1d3r"], [(536_870_912,)], 1000)
        + _sizes(["star2d1r", "box2d1r"], [(16384, 16384)], 1000)
        + _sizes(["star3d1r", "box3d1r"], [(768, 768, 768)], 1000)),
    # SPIDER (PPoPP'26) Figure 10 (scripts/Figure10_run.sh)
    "spider-native": (
        _sizes(["1d1r", "1d2r"], [(10_240_000,)], 100_000)
        + _sizes([f"{k}2d{r}r" for k in ("box", "star") for r in (1, 2, 3)],
                 [(10240, 10240)], 10_240)),
    # SPIDER Figure 11 (scripts/Figure11_run.sh)
    "spider-1d-scaling": _sizes(["1d1r", "1d2r"],
                                [(1024 * n,) for n in [1024] + [2048 * i for i in range(1, 21)]],
                                10_000),
    "spider-2d-scaling": _sizes(["box2d1r", "box2d2r", "box2d3r"],
                                [(512 * i, 512 * i) for i in range(1, 21)], 10_240),
    # SPIDER Figure 12 (scripts/Figure12_run.sh)
    "spider-ablation": _sizes(["box2d2r"], [(n, n) for n in (1280, 2560, 5120, 10240)], 10_000),
}


def expand_workloads(names: list[str]) -> list[str]:
    """Runner hook: replace every `sweep:<id>` with that sweep's workload names."""
    out = []
    for n in names:
        if n.startswith("sweep:"):
            key = n[len("sweep:"):]
            if key not in SWEEPS:
                raise ValueError(f"stencil: unknown sweep {key!r}; have {sorted(SWEEPS)}")
            out.extend(SWEEPS[key])
        else:
            out.append(n)
    return out


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


# ------------------------------------------------------ paper-native metrics
# Each integrated paper's OWN performance metric, recomputed from this
# harness's timed region and attached to the result record as
# metrics["paper_native"] -- ALONGSIDE the cross-paper GCUP/s, never in place
# of it (spec.yaml metrics.reporting_rule). Definitions are taken from each
# paper's evaluation section (fulltext) and cross-checked against the
# artifact's own print statement at the commit pinned in source.provenance;
# see spec.yaml metrics.per_paper_native for the citations.
#
# Anything a paper reports that this harness cannot measure (profiler
# counters, models, baseline speedups) is listed under "not_collected" with
# the reason, never dropped or zero-filled.

# AN5D Table 3 "FLOP/Cell". The synthetic shapes are one multiply per support
# point plus the adds between them: star2d 8x+1, star3d 12x+1, box
# 2*(2x+1)^d - 1, i.e. 2*points - 1. The named kernels carry their own counts
# (division / sqrt included), so they are looked up, not derived.
_AN5D_NAMED_FLOP_PER_CELL = {
    "j2d5pt": 10, "j2d9pt": 18, "j2d9pt-gol": 18, "gradient2d": 19, "j3d27pt": 54,
}

# SPIDER's plotting scripts (scripts/Figure10_draw.py, Figure11_draw.py,
# `unify_gstencil_sptc`) turn the one radius-7 fp16 SpTC run into the
# per-radius bars of Figures 10/11: divide by half2double_precision_ratio = 4
# (fp16 -> fp64 normalization, paper §4.1 "we scale the results by a factor of
# 4") and multiply by 7/r (one radius-7 sweep == 7/r fused radius-r steps).
_SPIDER_HALF2DOUBLE = 4
_SPIDER_PLOTTED_RADII = (1, 2, 3)


def _an5d_flop_per_cell(w: StencilWorkload) -> int:
    if w.named in _AN5D_NAMED_FLOP_PER_CELL:
        return _AN5D_NAMED_FLOP_PER_CELL[w.named]
    points = (2 * w.dims * w.radius + 1 if w.kind == "star"
              else (2 * w.radius + 1) ** w.dims)
    return 2 * points - 1


def _metric(name, value, unit, cls, definition, source) -> dict:
    return {"name": name, "value": value, "unit": unit, "class": cls,
            "definition": definition, "source": source}


def _not_collected(name, cls, reason) -> dict:
    return {"name": name, "class": cls, "status": "not collected", "reason": reason}


# timed-region note shared by the two adapters whose bridge adds a halo refresh
_HALO_REFRESH_NOTE = (
    "Since 2026-09-23 the whole T-step run stays on the GPU (the bridge "
    "launches the artifact's kernel directly). The timed region adds, beyond "
    "the paper's kernel launches, one device-to-device reset of the initial "
    "field per call and an O(perimeter) periodic halo refresh per step -- the "
    "refresh the artifact's own loop omits, needed for a correct periodic "
    "T-step sweep. Results timed before that date included host re-padding "
    "and H2D/D2H every step and are not comparable to the paper.")


def _native_stencil(impl_name: str, w: StencilWorkload, params: dict,
                    stats_ms: dict, statistic: str) -> dict | None:
    T = int(params.get("timesteps", w.timesteps))
    sec = stats_ms[statistic] / 1e3
    if sec <= 0:
        return None
    updates = w.cells * T
    gstencil = updates / sec / 1e9
    exec_ms = stats_ms[statistic]
    gst_def = ("T * prod(N_i) / (t * 1e9): grid points updated per second, "
               "T = real time steps performed, t = execution time")
    def exec_time(src):
        return _metric("execution time", exec_ms, "ms", "timing",
                       f"elapsed time of one call performing all T={T} steps", src)

    if impl_name == "an5d-stencil":
        fpc = _an5d_flop_per_cell(w)
        src = "AN5D, CGO'20, §6.1 Table 3 + Figures 5-6 (fulltext, arXiv 2001.01473)"
        out = {
            "paper": "AN5D (CGO'20, conf/cgo/MatsumuraZWEM20)",
            "primary": _metric(
                "GFLOP/s", fpc * updates / sec / 1e9, "GFLOP/s", "timing",
                f"FLOP/Cell (Table 3) x cell updates / kernel time; FLOP/Cell = {fpc} "
                "for this shape", src),
            "metrics": [
                _metric("GCell/s", gstencil, "GCell/s", "timing",
                        "cell updates / kernel time (right-hand axis of Figures 5-6)", src),
                _metric("FLOP/Cell", fpc, "FLOP", "timing",
                        "Table 3 per-point FLOP count used for GFLOP/s", src),
            ],
            "not_collected": [
                _not_collected("percent of peak", "hw",
                               "needs the device's FP64 peak; the harness does not "
                               "carry a peak table"),
                _not_collected("registers/thread, spills, shared memory bytes", "profiler",
                               "needs ptxas -v / ncu"),
                _not_collected("model-predicted GFLOP/s and model accuracy", "model",
                               "AN5D's own performance model is not run"),
            ],
            "notes": ["Paper statistic is the MEAN of 5 runs after 1 warm-up; the "
                      "paper-statistic value below uses the same timed reps."],
        }
        if statistic != "mean" and stats_ms.get("mean"):
            out["metrics"].append(_metric(
                "GFLOP/s (paper statistic: mean)",
                fpc * updates / (stats_ms["mean"] / 1e3) / 1e9, "GFLOP/s", "timing",
                "same formula, mean of the timed reps instead of the median", src))
        return out

    if impl_name == "convstencil-tcu":
        src = "ConvStencil, PPoPP'24, §5.1 Eq. 16 (fulltext) + src/2d/gpu.cu"
        return {
            "paper": "ConvStencil (PPoPP'24, conf/ppopp/ChenLWBWMYZCY24)",
            "primary": _metric("GStencils/s", gstencil, "GStencils/s", "timing",
                               gst_def, src),
            "metrics": [exec_time(src)],
            "not_collected": [
                _not_collected("uncoalesced global accesses (%)", "profiler", "needs ncu"),
                _not_collected("shared-memory bank conflicts per request", "profiler",
                               "needs ncu"),
                _not_collected("speedup vs cuDNN / Brick / DRStencil / TCStencil", "timing",
                               "those baselines are not integrated in this harness"),
            ],
            "notes": [
                "The artifact prints input_m*input_n*times*3 for radius-1 shapes because "
                "each launch there is a 3-step fused kernel (7x7 = 3-fold self-convolution "
                "of the 3x3, main.cu param_box_2d1r); `times` counts launches. This adapter "
                "runs ONE unfused step per launch, so T already counts real steps and no "
                "multiplier applies -- same Eq. 16 quantity.",
                _HALO_REFRESH_NOTE,
            ],
        }

    if impl_name == "lorastencil-star2d3r":
        src = "LoRAStencil, SC'24, §V-A Eq. 18 (fulltext) + src/2d/gpu.cu:478"
        return {
            "paper": "LoRAStencil (SC'24, conf/sc/ZhangLYCZCY24)",
            "primary": _metric("GStencil/s", gstencil, "GStencil/s", "timing",
                               gst_def, src),
            "metrics": [exec_time(src)],
            "not_collected": [
                _not_collected("shared-memory load/store/total requests", "profiler",
                               "needs ncu"),
                _not_collected("Compute (SM) throughput (%)", "profiler", "needs ncu"),
                _not_collected("arithmetic intensity (FLOP/byte)", "profiler", "needs ncu"),
                _not_collected("speedup vs cuDNN / AMOS / Brick / DRStencil / TCStencil / "
                               "ConvStencil", "timing",
                               "those baselines are not integrated in this harness"),
            ],
            "notes": [
                "gpu_star_2d3r (the wrapped kernel) prints T*m*n/t with no fusion "
                "multiplier; the x3 in other LoRAStencil kernels is the same 3-step fusion "
                "ConvStencil uses for small kernels.",
                _HALO_REFRESH_NOTE,
            ],
        }

    if impl_name == "flashfftstencil-box2d1r":
        src = "FlashFFTStencil, PPoPP'25, §5.1 Metrics + Figure 6 (fulltext) + src/2D/2d_main.cu:185"
        return {
            "paper": "FlashFFTStencil (PPoPP'25, conf/ppopp/HanLCBZYCZCY25)",
            "primary": _metric("GStencil/s", gstencil, "GStencil/s", "timing",
                               gst_def, src),
            "metrics": [exec_time(src)],
            "not_collected": [
                _not_collected("speedup over state of the art", "timing",
                               "the compared systems are not all integrated; compute "
                               "from paired runs on the same GPU"),
                _not_collected("memory footprint (GB) / OOM boundary", "hw",
                               "device memory high-water mark is not sampled"),
                _not_collected("uncoalesced global accesses (%), shared-store bank "
                               "conflicts per request, TCU pipeline utilization (%)",
                               "profiler", "needs ncu"),
                _not_collected("arithmetic intensity, zero fraction in TC fragments",
                               "profiler", "needs ncu"),
            ],
            "notes": [
                "Paper uses execution time and GStencil/s together as its key metrics; its "
                f"Table 3 runs 1000 steps, while this adapter performs T={T} step(s) per "
                "call (the artifact's 2D T-loop re-applies the same sweep, not a "
                "recursion), so execution time here is not the paper's 1000-step time.",
            ],
        }

    if impl_name == "spider-box2d7r-sptc":
        src = "SPIDER, PPoPP'26, §4.1 Metrics (fulltext, arXiv 2506.22035) + src/*/gpu_*_7r_half.cu"
        plot_src = "SPIDER scripts/Figure10_draw.py, Figure11_draw.py (unify_gstencil_sptc)"
        derived = [
            _metric(f"GStencils/s, Box-2D{r}R-equivalent, fp64-normalized",
                    gstencil / _SPIDER_HALF2DOUBLE * (w.radius / r),
                    "GStencils/s", "derived",
                    f"measured radius-{w.radius} GStencils/s / {_SPIDER_HALF2DOUBLE} "
                    f"(fp16->fp64) x {w.radius}/{r} (one radius-{w.radius} sweep == "
                    f"{w.radius}/{r} radius-{r} steps); what the paper PLOTS, not a "
                    f"separate radius-{r} run", plot_src)
            for r in _SPIDER_PLOTTED_RADII
        ]
        return {
            "paper": "SPIDER (PPoPP'26, conf/ppopp/GuW0Y26)",
            "primary": _metric(f"GStencils/s (radius-{w.radius} SpTC sweep, fp16)",
                               gstencil, "GStencils/s", "timing",
                               "points updated per second, as the artifact prints it: "
                               "m*n*times / t / 1e9", src),
            "metrics": [exec_time(src), *derived],
            "not_collected": [
                _not_collected("row-swap bandwidth (GB/s), row-swap instruction count and "
                               "duration", "profiler", "needs ncu"),
                _not_collected("modeled operations / input + parameter accesses per "
                               "update (Table 1)", "model", "analytical model, not measured"),
                _not_collected("speedup vs cuDNN / TCStencil / ConvStencil / LoRAStencil / "
                               "FlashFFTStencil", "timing",
                               "compute from paired runs on the same GPU"),
            ],
            "notes": ["Derived rows reproduce the paper's own normalization so the number "
                      "can be read against Figures 10/11; they are not independent "
                      "measurements and must not be ranked against measured radius-r runs."],
        }

    return None


workload.register_native_metrics("stencil", _native_stencil)


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


def _named_windows(u: np.ndarray, radius: int, boundary: str):
    """Pad `u` by `radius` per the boundary convention; return (padded, slicer)."""
    padded = np.pad(u, radius, mode="wrap" if boundary == "periodic" else "constant")

    def at(o):
        return padded[tuple(slice(radius + c, radius + c + n) for c, n in zip(o, u.shape))]
    return padded, at


def _restore_band(out: np.ndarray, u: np.ndarray, radius: int) -> np.ndarray:
    """"fixed" convention: the width-radius band keeps its values from `u`."""
    keep = np.copy(u)
    inner = tuple(slice(radius, n - radius) for n in u.shape)
    if all(n > 2 * radius for n in u.shape):
        keep[inner] = out[inner]
    return keep


def _sweep_named_linear(u, coeffs, divisor, radius, boundary):
    """AN5D linear named kernel, verbatim: (sum_o c_o * u[p+o]) / divisor."""
    _, at = _named_windows(u, radius, boundary)
    acc = np.zeros_like(u)
    for o, c in coeffs.items():
        acc += c * at(o)
    out = acc / divisor
    return _restore_band(out, u, radius) if boundary == "fixed" else out


def _gradient2d_term(u, eps, boundary):
    """AN5D gradient2d's added term 1/sqrt(eps + sum_nb (u - u_nb)^2)."""
    _, at = _named_windows(u, 1, boundary)
    sq = np.full_like(u, eps)
    for o in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        d = u - at(o)
        sq += d * d
    return 1.0 / np.sqrt(sq)


def _reference_named(w: StencilWorkload, timesteps: int, boundary: str):
    """
    Reference for AN5D's named kernels, following each kernel's source
    arithmetic (sum, then divide; or gradient2d's sqrt). `scale` carries the
    same recursion with |c| (linear) or accumulates the always-positive added
    term (gradient2d: |u_next| <= |u| + term), so it bounds the result the
    way the synthetic path's |weights| recursion does.
    """
    spec = NAMED[w.named]
    u = w.initial_field(dtype=np.float64)
    s = np.abs(u)
    if "coeffs" in spec:
        absc = {o: abs(c) for o, c in spec["coeffs"].items()}
        for _ in range(timesteps):
            u = _sweep_named_linear(u, spec["coeffs"], spec["divisor"], w.radius, boundary)
            s = _sweep_named_linear(s, absc, spec["divisor"], w.radius, boundary)
    else:
        for _ in range(timesteps):
            term = _gradient2d_term(u, spec["eps"], boundary)
            u_next, s_next = u + term, s + term
            if boundary == "fixed":
                u_next, s_next = _restore_band(u_next, u, 1), _restore_band(s_next, s, 1)
            u, s = u_next, s_next
    return u, s


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
    if boundary not in ("periodic", "fixed", "zero-halo"):
        raise ValueError(
            f"stencil: unknown boundary convention {boundary!r} for "
            f"{workload_.name!r} (expected 'periodic', 'fixed', or "
            "'zero-halo', spec.yaml inputs.boundary)")
    if workload_.named:
        u, s = _reference_named(workload_, timesteps, boundary)
        _warn_if_gate_weak(workload_, timesteps, boundary, u, s)
        return u, s
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
    _warn_if_gate_weak(workload_, timesteps, boundary, u, s)
    return u, s


def _warn_if_gate_weak(workload_, timesteps, boundary, u, s):
    # Underflow guard: a kernel whose coefficients sum below 1 (AN5D's named
    # kernels do) decays the field geometrically; once it underflows, the
    # reference is all zeros and an all-zeros output passes the gate.
    peak = float(np.max(s)) if s.size else 0.0
    if peak < 1e-250:
        warnings.warn(
            f"stencil gate for {workload_.name!r} (T={timesteps}): the reference "
            f"field has decayed to ~0 (max scale {peak:.1e}); an all-zeros output "
            "would pass. The gate is VACUOUS at this T -- gate at a smaller T "
            "(e.g. ':T=50') to check correctness.", stacklevel=3)
        return
    # Flat-field guard: with smoothing (all-positive) weights a small grid at
    # large T mixes toward a constant, and then a misplaced-data bug (one-cell
    # shift, wrong offset) no longer changes the answer. Measured on this
    # domain's weights: shift bugs were caught whenever ptp(u)/max(scale) >=
    # ~3e-4 and missed below ~5e-7, so warn well above that.
    spread = float(np.ptp(u)) / max(peak, 1e-300)
    if spread < 1e-3:
        warnings.warn(
            f"stencil gate for {workload_.name!r} (grid {workload_.grid_shape}, "
            f"T={timesteps}, boundary={boundary}): the reference field has gone "
            f"nearly flat (ptp/scale={spread:.1e}). The gate still catches "
            "zero/wrong-scale outputs but CANNOT catch misplaced-data bugs "
            "(shifted or wrong-offset neighbors). Gate on a larger grid or a "
            "smaller T.", stacklevel=3)


REFERENCES = {"stencil": reference_stencil}


# Correctness-gate budget, in point-updates (cells x support points x steps)
# per reference recursion. The fp64 numpy reference runs ~1e8-2e8 point-
# updates/s, and it runs twice (result + scale), so 3e10 keeps the gate at
# roughly 5-10 minutes -- while a full paper-sized run (10240^2 x T=10240, or
# 16384^2 x T=1000 for a radius-3 box) would take hours to days on the CPU.
# Override with KB_STENCIL_GATE_POINT_OPS.
GATE_POINT_OPS = float(os.environ.get("KB_STENCIL_GATE_POINT_OPS", 3e10))


def gate_workload(kernel: str, w: StencilWorkload) -> StencilWorkload | None:
    """
    Runner hook: the workload the correctness gate runs on instead of `w`,
    or None to gate on `w` itself.

    Same grid, same shape, same coefficients, same boundary -- only fewer
    time steps, as many as the budget allows. Keeping the full grid means
    the gate still exercises the implementation's full-size indexing, tiling
    and alignment; only the step count shrinks. The count keeps T's parity
    (an adapter's final-buffer selection depends on it) and is at least 2 when
    T allows, so ping-pong across steps is exercised. At the paper sizes it
    stays >= one AN5D temporal block for every bridged shape (AN5D's
    compiled blocking is 10/8/4/2/4/3 steps for star2d1r/box2d1r/star2d3r/
    box2d3r/star3d1r/box3d1r).
    """
    T = w.timesteps
    per_step = w.cells * max(w.support_size, 1)
    t = int(GATE_POINT_OPS // per_step)
    if t >= T:
        return None
    t = max(t, min(T, 2))
    if (T - t) % 2:
        t = t - 1 if t > 2 else t + 1
    return dataclasses.replace(w, timesteps=t)
# Common yardstick for the report's speedup column, in order of preference:
# the library GPU sweep runs every shape any paper adapter can, so each paper
# gets a speedup on its own shape even where no other paper overlaps it.
BASELINE_IMPLS = {"stencil": ["torch-conv-stencil", "numpy-stencil"]}
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


def _numpy_impl_named_step(u: np.ndarray, h: dict) -> np.ndarray:
    """
    NumpyStencil's OWN sweep for AN5D's named kernels -- independent of
    `_reference_named` (which slices a padded array per offset): this one
    gathers every neighborhood in one `sliding_window_view` over the padded
    array and contracts it with np.tensordot (linear kernels, then divides),
    or indexes the window's axis-neighbor positions (gradient2d).
    """
    r, boundary = h["radius"], h["boundary"]
    padded = np.pad(u, r, mode="wrap" if boundary == "periodic" else "constant")
    win = sliding_window_view(padded, (2 * r + 1,) * u.ndim)
    if h["raw_kernel"] is not None:
        axes = tuple(range(u.ndim, 2 * u.ndim))
        out = np.tensordot(win, h["raw_kernel"], axes=(axes, tuple(range(u.ndim))))
        out = out / h["divisor"]
    else:  # gradient2d: window centre is (1,1); axis neighbors at (0,1),(2,1),(1,0),(1,2)
        c = win[..., 1, 1]
        sq = h["eps"] + sum((c - win[..., a, b]) ** 2
                            for a, b in ((0, 1), (2, 1), (1, 2), (1, 0)))
        out = c + 1.0 / np.sqrt(sq)
    if boundary == "fixed":
        res = u.copy()
        inner = tuple(slice(r, n - r) for n in u.shape)
        res[inner] = out[inner]
        return res
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
        if workload_.named:
            spec = NAMED[workload_.named]
            raw = None
            if "coeffs" in spec:
                raw = np.zeros((2 * workload_.radius + 1,) * workload_.dims, dtype=self.dtype)
                for o, c in spec["coeffs"].items():
                    raw[tuple(workload_.radius + x for x in o)] = c
            return {
                "u0": u0, "bufs": bufs, "named": True, "boundary": boundary,
                "radius": workload_.radius, "raw_kernel": raw,
                "divisor": spec.get("divisor"), "eps": spec.get("eps"),
                "timesteps": int(params.get("timesteps", workload_.timesteps)),
            }
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
        if h.get("named"):
            cur = 0
            for _ in range(h["timesteps"]):
                np.copyto(bufs[1 - cur], _numpy_impl_named_step(bufs[cur], h))
                cur = 1 - cur
            return bufs[cur]
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
