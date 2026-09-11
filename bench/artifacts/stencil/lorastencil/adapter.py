"""
Adapter for LoRAStencil (SC'24, "LoRAStencil: Low-Rank Adaptation of Stencil
Computation on Tensor Cores", conf/sc/ZhangLYCZCY24).

Wraps ONE of the artifact's four 2D Tensor-Core kernels --
`gpu_star_2d3r` (source/src/2d/gpu.cu) -- through bridge.cu/.so (this
directory, NOT part of the artifact). The other three 2D kernels
(`gpu_box_2d3r` used for both box2d1r/box2d3r, and `gpu_star_2d1r`) were
investigated and found to be genuinely UNWRAPPABLE with this domain's
arbitrary caller-supplied weights -- not a shape-coverage shortcut, but a
real limitation of the artifact's own unmodified host-side code, verified
empirically (see bridge.cu's docstring and STATUS.md for the full
derivation; summarized here):

  - box2d1r: `gpu_box_2d3r`'s own "Factorize parameter matrix" step
    (gpu.cu:290, `prop = params[(i+3)*7] / params[0]`) divides by the
    weight matrix's OWN CORNER entry (offset (-3,-3)). A true zero-padded
    radius-1 box has a zero corner -> 0.0/0.0 = NaN, poisoning the whole
    factorization. Confirmed empirically: feeding an IDENTITY kernel
    (single nonzero at the center, corner exactly 0) through
    `lorastencil2d_box2d3r_sweep` returns all-NaN output.
  - box2d3r: the SAME function, fed this domain's own (non-onion-ring)
    box3r weight pattern (`stencil.py::_build_weights`), also returns
    all-NaN -- a DIFFERENT division inside the same factorization (level 1,
    gpu.cu:307, `prop = temp[...][1*7+1] / temp[...][7+1]`) divides zero by
    zero once the residual from level 0 concentrates in a row this
    artifact's factorization does not expect. `gpu_box_2d3r`'s host-side
    scheme is not a general low-rank decomposition: it is hardcoded to
    LoRAStencil's own synthetic "concentric-square-ring, one constant value
    per ring" benchmark pattern (see main.cu's `param_box_2d1r`
    construction) and silently produces NaN, not merely a lossy
    approximation, for any weight matrix that does not share that specific
    structure -- including this domain's own principled, cancellation-
    exercising synthetic weights.
  - star2d1r: `gpu_star_2d1r` (gpu.cu:484-487) ignores its own `params`
    argument entirely -- `param_u`/`param_v` are hardcoded local arrays
    (`{0,1,2,4,2,1,0}`), never read from `params`. There is no way to
    inject this domain's star2d1r weights into this kernel at all.

`gpu_star_2d3r` has none of this: it builds its two "line" factors by
PLAIN ASSIGNMENT from `params` (no division anywhere, gpu.cu:429-444),
which is mathematically EXACT for a star-shaped support (a star sum
decomposes losslessly into an independent vertical-line convolution plus an
independent horizontal-line convolution -- no cross terms exist to
approximate). Verified empirically (bridge/probe script, 64x64 random
field, this domain's own star3r weight construction): max abs error
3.33e-16 against `stencil.py`'s own periodic-wrap offset-loop reference --
fp64 rounding only, not an approximation artifact.

PRECISION: DATA_TYPE is `double` throughout 2d_utils.h/gpu.cu (genuine fp64
Tensor-Core DMMA), same situation as the convstencil/flashfftstencil
siblings -- PRECISIONS = ["fp64"], gated against variant 1
(`stencil-cpu-gpu-kernel-fp64`), not variant 2 (fp16-labeled), for the same
reason those two siblings are: the artifact's own arithmetic never touches
fp16 despite appearing in variant 2's `suite` listing.

WEIGHT LAYOUT + OUTPUT OFFSET (verified empirically before being wired in
here, not just derived on paper -- see bridge.cu's probe transcript in
STATUS.md): `params[i*7+j]` weights offset `(i-3, j-3)` from the stencil's
center (cross-correlation / `dense_kernel(flip=False)` layout, matching
convstencil's own convention for the same paper family); valid output
starts at buffer offset `(row=4, col=4)` within a `(m+8) x (n+8)`
periodic-wrap-padded host buffer -- a SYMMETRIC HALO=4 margin on all four
sides (simpler than convstencil's asymmetric (3,4) case, because
LoRAStencil's own `#define HALO 4` applies uniformly).

TIMESTEPS: `gpu_star_2d3r`'s own internal `times`-argument loop ping-pongs
device buffers WITHOUT refreshing the periodic-wrap halo between sweeps
(same class of finding as convstencil's `gpu_box_2d1r` and
flashfftstencil's `rfft_2d_8_nwarp`). `bridge.cu` always calls with
times=1; this adapter's `run()` performs its OWN T-sweep loop, re-padding
the periodic-wrap halo from the CURRENT field before each single-sweep
call -- identical discipline to convstencil's adapter.

DEVICE-MEMORY LEAK (real artifact bug, not introduced by this
integration): `gpu_star_2d3r`/`gpu_box_2d3r`/`gpu_star_2d1r` never
`cudaFree()` their internal `array_d[0]`/`array_d[1]` device buffers --
confirmed by grep (zero `cudaFree` calls anywhere in gpu.cu). Each `run()`
timestep therefore leaks two small device buffers; acceptable for this
integration's bounded, small-grid login-node gate (a handful of calls) but
would need patching (out of scope: this is the artifact's own driver code,
not a build-system fix) before any large-scale/long-running use.

SHAPE ALIGNMENT CONSTRAINT (inherited from the artifact, not introduced):
the kernel grid is `ceil(m/32) x ceil(n/64)` blocks with no tail/boundary
guard in the shared-memory load loop, so `m` must be a multiple of 32 and
`n` a multiple of 64 or a block will read/write past the padded
allocation. `prepare()` raises `NotImplementedError` otherwise.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "stencil"
IMPL_NAME = "lorastencil-star2d3r"
PAPER_KEY = "conf/sc/ZhangLYCZCY24"
PRECISIONS = ["fp64"]  # DATA_TYPE double throughout 2d_utils.h/gpu.cu

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "bridge.so")

_KERNEL_RADIUS = 3
_HALO = 4
_ROW_MULT = 32
_COL_MULT = 64


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    dp = ctypes.POINTER(ctypes.c_double)
    lib.lorastencil2d_star2d3r_sweep.argtypes = [dp, dp, dp, ctypes.c_int, ctypes.c_int]
    lib.lorastencil2d_star2d3r_sweep.restype = None
    return lib


def _build_params(weights: dict) -> np.ndarray:
    k = np.zeros((7, 7), dtype=np.float64)
    for (dr, dc), w in weights.items():
        if abs(dr) > _KERNEL_RADIUS or abs(dc) > _KERNEL_RADIUS:
            raise ValueError(
                f"lorastencil-star2d3r: offset {(dr, dc)} exceeds the "
                f"kernel's fixed radius-{_KERNEL_RADIUS} (7x7) support")
        k[_KERNEL_RADIUS + dr, _KERNEL_RADIUS + dc] = w
    return np.ascontiguousarray(k.reshape(-1))


class LoRAStencilStar2D3R:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"lorastencil-star2d3r's kernel (DATA_TYPE, 2d_utils.h) is "
                f"double; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, workload_, params: dict):
        if workload_.kind != "star" or workload_.dims != 2 or workload_.radius != 3:
            raise NotImplementedError(
                "lorastencil-star2d3r only implements the 2D radius-3 star "
                f"kernel (star2d3r); got {workload_.kind}{workload_.dims}d"
                f"{workload_.radius}r -- see adapter.py's module docstring "
                "for why box2d1r/box2d3r/star2d1r are not wrapped (genuine "
                "artifact limitations, not a coverage shortcut)")
        if workload_.boundary != "periodic":
            raise NotImplementedError(
                "lorastencil-star2d3r adapter only implements periodic-wrap "
                f"boundary; workload requested {workload_.boundary}")
        m, n = workload_.grid_shape
        if m % _ROW_MULT != 0 or n % _COL_MULT != 0:
            raise NotImplementedError(
                f"lorastencil-star2d3r's kernel grid has no tail guard "
                f"(BLOCK_SIZE_ROW=32, BLOCK_SIZE_COL=64, gpu.cu); grid_shape "
                f"must be a multiple of ({_ROW_MULT},{_COL_MULT}), got ({m},{n})")

        u0 = workload_.initial_field(dtype=np.float64)
        params49 = _build_params(workload_.weights)
        timesteps = int(params.get("timesteps", workload_.timesteps))
        return {
            "field": np.ascontiguousarray(u0),
            "params49": params49,
            "m": m, "n": n,
            "timesteps": timesteps,
        }

    def run(self, h):
        field = h["field"]
        m, n = h["m"], h["n"]
        params49 = h["params49"]
        dp = ctypes.POINTER(ctypes.c_double)
        for _ in range(h["timesteps"]):
            padded = np.pad(field, ((_HALO, _HALO), (_HALO, _HALO)), mode="wrap")
            padded = np.ascontiguousarray(padded, dtype=np.float64)
            out = np.zeros((m, n), dtype=np.float64)
            self.lib.lorastencil2d_star2d3r_sweep(
                padded.ctypes.data_as(dp), out.ctypes.data_as(dp),
                params49.ctypes.data_as(dp), m, n)
            field = out
        h["field"] = field
        return field

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        h.clear()


def create(precision: str):
    return LoRAStencilStar2D3R(precision)
