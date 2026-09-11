"""
Adapter for AN5D (CGO'20)'s automated high-degree-temporal-blocking CUDA
stencil kernels. PAPER_KEY = conf/cgo/MatsumuraZWEM20.
Repo: https://github.com/khaki3/AN5D-Artifact.

AN5D is a CODE GENERATOR (PPCG/isl-based source-to-source): its contribution
is the CUDA it emits for a given (stencil, tuning) pair. The generator
toolchain itself could not be built on this machine within budget -- see
STATUS.md ("Generator toolchain: BUILD-FAILED"). AN5D-Artifact (this
directory's `source/`) ships its own PRE-GENERATED CUDA for its whole
benchmark suite under `source/compiled/double/<shape>-<tuning>_{host,kernel}.cu`
-- the paper's own generated kernels, produced once upstream and checked in
for its "Tuned" evaluation path (source/README.md). This adapter wraps six of
them, one per shape this domain names (star2d1r, box2d1r, star2d3r, box2d3r,
star3d1r, box3d1r -- exactly AN5D's own naming convention, which
kernelbench/domains/stencil.py's shape parser adopts verbatim).

`gen_bridge.py` (this directory) mechanically extracts each shape's generated
kernel-launch dispatch cascade from its `_host.cu` (brace-matched, not
hand-transcribed) into `bridge_<shape>.cu`, split into prepare() (device
malloc + H2D once) / run() (kernel launches only -- AN5D's dispatch natively
handles ANY `timestep` value at runtime, see STATUS.md's "Temporal blocking"
section) / copy_out() (D2H) / free(), the same split-the-monolithic-driver
pattern the sibling spider/convstencil adapters use. `build.sh` compiles each
shape into its own `bridge_<shape>.so`.

Three load-bearing deviations, documented in full in STATUS.md:

1. NATIVE COEFFICIENTS: AN5D's benchmark stencils' coefficients are C `float`
   literals baked directly into the polyhedral-scheduled update statement of
   the ALREADY-GENERATED CUDA -- there is no runtime params array to inject
   this domain's own weights into (unlike convstencil/lorastencil). prepare()
   overrides workload_.offsets/weights in place to AN5D's OWN coefficients,
   parsed directly from its own source/<shape>.c reference file by regex (not
   hand-transcribed -- see _parse_native_weights and STATUS.md's verification
   transcript), so the harness's reference_stencil() (reads these same
   attributes off the identical workload_ object) regenerates a matching fp64
   ground truth automatically -- the same "params override" pattern
   spider/lorastencil already use, per ARTIFACT_GUIDE.md.
2. BOUNDARY (2026-09-06 update -- boundary-convention gate fix): AN5D's own
   common.h::init_grid fills the array ONCE (`dimsize = compsize +
   BENCH_RAD*2` in AN5D's OWN vocabulary) and the generated sweep's loop
   bounds ([radius, dimsize-radius)) never touch the outer radius-width band
   again across timesteps -- confirmed by reading common.h and every
   generated bridge_<shape>.cu's own `__c1Len = dimsize - radius - radius`
   loop-length expression directly. This is EXACTLY
   kernelbench/domains/stencil.py's "fixed" boundary convention (module
   docstring "Boundary convention hook"): a width-radius band that keeps its
   t=0 value for the whole run, no separate halo array, no per-sweep
   refresh. The FIX (previously this adapter passed `dimsize = compsize +
   2*radius` -- i.e. treated AN5D's `compsize` argument as this domain's own
   grid_shape and ADDED an extra radius-width ring OUTSIDE it -- and filled
   that extra ring via one periodic wrap of the interior, trying to
   approximate the harness's periodic-wrap reference for sweep 1 only,
   diverging from sweep 2 onward) is to pass `dimsize = grid_shape` DIRECTLY
   (no added ring) and set `params["boundary"] = "fixed"`: AN5D's own
   `[radius, dimsize-radius)` bound then naturally coincides with this
   domain's own "fixed"-convention interior of the SAME grid_shape-sized
   array, and both ping-pong buffers are seeded with the plain
   grid_shape-sized initial field (no np.pad, no wrap) -- see prepare()
   below. The previous "one periodic wrap, valid for sweep 1 only" workaround
   is REMOVED entirely; it was compensating for a fixed/periodic mismatch
   that no longer exists once both sides agree on "fixed". `to_host()` no
   longer crops radius cells off each edge either -- dimsize IS grid_shape
   now, nothing extra to crop. See STATUS.md for the full-array gate numbers
   before and after this change, and for why the ~1e-8 interior-cropped
   check (a still-valid, independent confirmation that AN5D's coefficients
   and dispatch cascade are arithmetically correct) is retained as
   supplementary evidence rather than replaced.
3. GRID SHAPE: AN5D's generated code takes a single `dimsize` and reuses it
   for every axis (see gen_bridge.py's malloc_expr) -- only square (2D) /
   cube (3D) grids are supported; prepare() raises NotImplementedError
   otherwise (never hit by this domain's own default or smoke grids, which
   are all square/cube).

Unlike the spider/convstencil/lorastencil siblings (each wraps ONE shape,
2D-only), AN5D-Artifact's pre-generated set happens to cover exactly this
domain's 6 shapes in BOTH 2D and 3D, and needs no per-workload shape
override (kind/dims/radius already match what was requested) -- the only
override is the coefficient VALUES (point 1 above).
"""
from __future__ import annotations

import ctypes
import os
import re

import numpy as np

KERNEL = "stencil"
IMPL_NAME = "an5d-stencil"
PAPER_KEY = "conf/cgo/MatsumuraZWEM20"
PRECISIONS = ["fp64"]  # compiled/double/*_kernel.hu: literal `double *A` (see build.sh)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "source")

# (kind, dims, radius) -> (bridge shape name, native-coefficient .c source)
_SHAPES = {
    ("star", 2, 1): ("star2d1r", "star2d1r.c"),
    ("box", 2, 1): ("box2d1r", "box2d1r.c"),
    ("star", 2, 3): ("star2d3r", "star2d3r.c"),
    ("box", 2, 3): ("box2d3r", "box2d3r.c"),
    ("star", 3, 1): ("star3d1r", "star3d1r.c"),
    ("box", 3, 1): ("box3d1r", "box3d1r.c"),
}

_AXES = "ijk"


def _parse_native_weights(c_source_path: str, dims: int) -> dict:
    """
    Extract AN5D's own hardcoded stencil coefficients directly from its
    benchmark .c source (e.g. source/box2d3r.c) via regex over the
    `#pragma scop ... #pragma endscop` region -- the literal update formula
    the paper's generator was run against (source/README.md's own "AN5D can
    be used as follows: an5d --bt=4 ... star3d1r.c" -- these ARE the input
    files). This is a mechanical extraction, not a hand-transcription: every
    shape's coefficient count/sum was cross-checked against this parser
    before wiring it in (STATUS.md's verification transcript).
    """
    with open(c_source_path) as f:
        text = f.read()
    m = re.search(r"#pragma scop(.*?)#pragma endscop", text, re.S)
    region = m.group(1) if m else text
    axis_pat = "".join(r"\[(" + a + r")([+-]\d+)?\]" for a in _AXES[:dims])
    term_re = re.compile(r"([0-9]+\.[0-9]+)f\s*\*\s*A\[t%2\]" + axis_pat)
    weights: dict = {}
    for mo in term_re.finditer(region):
        coeff = float(mo.group(1))
        offs = []
        for gi in range(dims):
            off_str = mo.group(3 + 2 * gi)  # groups: (axis,off) pairs, off is group 2*gi+3
            offs.append(int(off_str) if off_str else 0)
        key = tuple(offs)
        weights[key] = weights.get(key, 0.0) + coeff
    if not weights:
        raise RuntimeError(f"_parse_native_weights: no terms found in {c_source_path}")
    return weights


def available() -> tuple[bool, str]:
    try:
        for shape_name, _ in _SHAPES.values():
            so = os.path.join(_HERE, f"bridge_{shape_name}.so")
            if not os.path.exists(so):
                return False, f"not built: {so} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


_LOADED: dict = {}


def _load(shape_name: str):
    if shape_name in _LOADED:
        return _LOADED[shape_name]
    lib = ctypes.CDLL(os.path.join(_HERE, f"bridge_{shape_name}.so"))
    dp = ctypes.POINTER(ctypes.c_double)
    p = f"an5d_{shape_name}_"
    getattr(lib, p + "prepare").argtypes = [dp, ctypes.c_int]
    getattr(lib, p + "prepare").restype = ctypes.c_void_p
    getattr(lib, p + "run").argtypes = [ctypes.c_void_p, ctypes.c_int]
    getattr(lib, p + "run").restype = None
    getattr(lib, p + "copy_out").argtypes = [ctypes.c_void_p, dp]
    getattr(lib, p + "copy_out").restype = None
    getattr(lib, p + "free").argtypes = [ctypes.c_void_p]
    getattr(lib, p + "free").restype = None
    _LOADED[shape_name] = lib
    return lib


class An5dStencil:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                "AN5D-Artifact's compiled/double/* pre-generated sources are "
                f"double(fp64)-typed CUDA (literal `double *A` in every "
                f"_kernel.hu, not a SB_TYPE macro); requested {precision!r}. "
                "compiled/float/* also exists but was not wired up in this "
                "integration pass (fp64 is the domain's primary variant).")
        self.precision = precision

    def prepare(self, workload_, params: dict):
        key = (workload_.kind, workload_.dims, workload_.radius)
        if key not in _SHAPES:
            raise NotImplementedError(
                f"AN5D-Artifact's pre-generated compiled/double/ sources wired "
                f"up here cover exactly this domain's 6 shapes (star2d1r, "
                f"box2d1r, star2d3r, box2d3r, star3d1r, box3d1r); "
                f"(kind={workload_.kind!r}, dims={workload_.dims}, "
                f"radius={workload_.radius}) is not one of them (1D and "
                f"radius-2/4 stencils exist in the artifact too but were not "
                f"wired up in this integration pass).")
        shape_name, cfile = _SHAPES[key]
        lib = _load(shape_name)

        dims, radius = workload_.dims, workload_.radius
        grid_shape = tuple(int(s) for s in workload_.grid_shape)
        compsize = grid_shape[0]
        if any(s != compsize for s in grid_shape):
            raise NotImplementedError(
                f"AN5D-Artifact's generated code takes a single `dimsize` "
                f"reused for every axis (square/cube grids only -- see "
                f"gen_bridge.py's malloc_expr); grid_shape {grid_shape} is not.")
        # --- boundary (module docstring point 2, 2026-09-06 update): dimsize
        # IS grid_shape now, not grid_shape + 2*radius -- see the docstring's
        # BOUNDARY section for why this makes the domain's own outer
        # radius-width band double as AN5D's native halo, matching the
        # "fixed" convention exactly rather than approximating it.
        dimsize = compsize

        # --- deviation 1: native coefficient override (precedent: spider,
        # lorastencil) -- see module docstring point 1.
        native_weights = _parse_native_weights(os.path.join(_CSRC, cfile), dims)
        workload_.offsets = list(native_weights.keys())
        workload_.weights = native_weights

        # --- deviation 2: boundary -- see module docstring point 2 (rewritten
        # 2026-09-06). No periodic-wrap workaround: `params["boundary"]` is
        # set to "fixed" so the harness's reference_stencil() also freezes
        # its own outer radius-width band to the seeded initial field's
        # values, matching what AN5D's generated code already does natively
        # (its `[radius, dimsize-radius)` loop bounds never touch that band,
        # confirmed by reading common.h/the generated dispatch -- see module
        # docstring). Both ping-pong buffers are seeded with the SAME
        # grid_shape-sized initial field (no extra padding array, no wrap):
        # the untouched band is simply whichever value was already in
        # bufs[cur] at that position, which is u0's own value from t=0
        # onward since neither buffer's band is ever overwritten.
        params["boundary"] = "fixed"
        field = workload_.initial_field(dtype=np.float64)  # shape == grid_shape == (dimsize,)*dims
        buf = np.empty((2,) + field.shape, dtype=np.float64)
        buf[0] = field
        buf[1] = field
        buf = np.ascontiguousarray(buf)

        dp = ctypes.POINTER(ctypes.c_double)
        handle = getattr(lib, f"an5d_{shape_name}_prepare")(
            buf.ctypes.data_as(dp), dimsize)
        if not handle:
            raise RuntimeError(f"an5d_{shape_name}_prepare returned a null handle")

        timesteps = int(params.get("timesteps", workload_.timesteps))
        return {
            "lib": lib, "shape_name": shape_name, "handle": handle,
            "dimsize": dimsize, "radius": radius, "dims": dims,
            "compsize": compsize, "timesteps": timesteps,
            "_buf": buf,  # keep alive: prepare()'s H2D copy is synchronous, precautionary
        }

    def run(self, h):
        # ONE call: AN5D's own dispatch cascade handles the full `timesteps`
        # count natively (main loop of `bt`-sized macro-steps + a remainder
        # switch down to single-step kernels for any leftover -- verified by
        # reading the generated _host.cu, see STATUS.md "Temporal blocking").
        # No adapter-side T-sweep loop needed, unlike convstencil/lorastencil.
        getattr(h["lib"], f"an5d_{h['shape_name']}_run")(h["handle"], h["timesteps"])
        return h

    def to_host(self, out) -> np.ndarray:
        # dimsize == compsize == grid_shape now (see prepare()'s boundary
        # note) -- the returned array already has the full requested shape,
        # outer radius-width band included and untouched; no crop needed
        # (unlike the pre-2026-09-06 version, which allocated dimsize =
        # compsize + 2*radius and cropped the extra padding back off here).
        dimsize, dims = out["dimsize"], out["dims"]
        result = np.ascontiguousarray(np.empty((2,) + (dimsize,) * dims, dtype=np.float64))
        dp = ctypes.POINTER(ctypes.c_double)
        getattr(out["lib"], f"an5d_{out['shape_name']}_copy_out")(
            out["handle"], result.ctypes.data_as(dp))
        t_final = out["timesteps"] % 2
        return result[t_final].astype(np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        getattr(h["lib"], f"an5d_{h['shape_name']}_free")(h["handle"])
        h.clear()


def create(precision: str):
    return An5dStencil(precision)
