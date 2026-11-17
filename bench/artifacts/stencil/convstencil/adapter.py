"""
Adapter for ConvStencil (PPoPP 2024, "ConvStencil: Transform Stencil
Computation to Matrix Multiplication on Tensor Cores",
conf/ppopp/ChenLWBWMYZCY24).

Wraps the artifact's own 2D box-stencil Tensor-Core kernel (gpu_box_2d1r,
source/src/2d/gpu.cu) through bridge.cu/.so (this directory) -- see that
file's docstring for the full reasoning. Short version:

  - main.cu's CLI switch never actually populates weights for the
    star2d1r/star2d3r shapes (param_star_2d1r is declared, zero-initialized,
    and never written to anywhere in main.cu -- verified by reading the
    whole file). We bypass main.cu and call gpu_box_2d1r() directly with a
    weight array built from the harness's own StencilWorkload.weights,
    sidestepping that bug without touching kernel code.
  - gpu_box_2d1r's internal multi-`times` loop ping-pongs device buffers
    WITHOUT refreshing the halo band between sweeps, so it cannot reproduce
    our domain's periodic-wrap T-sweep recursion for T>1. The bridge
    therefore launches the artifact's own `kernel2d` itself and refreshes the
    periodic halo ON THE DEVICE after each step (../periodic_halo.cuh), so
    the whole T-step run stays GPU-resident: prepare() uploads once, run()
    is T x (kernel + halo refresh) with no host transfer, to_host() copies
    the interior back. (Until 2026-09-23 run() re-padded on the host and
    round-tripped H2D/D2H every step, so its timings mostly measured PCIe,
    not the kernel -- see bridge.cu "History".)

Coverage: 2D ONLY. ConvStencil's kernel always operates on a fixed 7x7
(radius-3) support internally (HALO=3 hardcoded in gpu.cu regardless of
shape name), so every launch costs the same whatever the shape.

Temporal fusion (2026-11-17, matches the paper): for radius-1 shapes
(star2d1r, box2d1r) each launch applies the 3-fold self-convolution of the
3x3 weights, i.e. 3 real time steps per launch, exactly what the artifact's
main.cu does (param_box_2d1r) and what its GStencil/s formula (times*3)
assumes. T % 3 leftover steps run as one extra launch with the (T % 3)-fold
kernel. Radius-2/3 shapes run one step per launch (zero-padded into 7x7 for
radius 2). KB_CONVSTENCIL_FUSION=0 restores the old one-step-per-launch
mode. Before this change every shape measured ~61 GCell/s on an A100,
because the radius-1 shapes wasted 2/3 of each launch. 1D
(source/src/1d) and 3D (source/src/3d) were NOT wired up in this
integration pass's time budget -- their kernel/buffer layouts were not
reverse-engineered, so prepare() raises NotImplementedError for
workload.dims != 2 rather than risk silently-wrong output. See STATUS.md.

Precision: ConvStencil's kernel operates entirely in `double` (DATA_TYPE,
2d_utils.h; all wmma fragments in gpu.cu are double-templated) -- fp64 only.

Boundary offsets (row margin 3/3, column margin 4/4, both periodic-wrap)
were determined by reading gpu.cu's kernel2d output-store index arithmetic
and independently verified against a hand-written periodic-wrap numpy
reference (identity, pure-axis-shift, and 5-point-star weight sets, all
max-abs-err 0 to ~2e-16) before being wired in here -- see bridge.cu's
docstring for the derivation and STATUS.md for the verification transcript.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "stencil"
IMPL_NAME = "convstencil-tcu"
PAPER_KEY = "conf/ppopp/ChenLWBWMYZCY24"
PRECISIONS = ["fp64"]  # gpu_box_2d1r's DATA_TYPE / wmma fragments are all `double`

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "bridge.so")

_KERNEL_RADIUS = 3


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


_ROW_MULT = 32   # BLOCK_SIZE_ROW: kernel2d's grid has no tail guard
_COL_MULT = 64   # BLOCK_SIZE_COL


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    dp = ctypes.POINTER(ctypes.c_double)
    vp = ctypes.c_void_p
    lib.convstencil2d_prepare.argtypes = [dp, dp, dp, ctypes.c_int, ctypes.c_int]
    lib.convstencil2d_prepare.restype = vp
    lib.convstencil2d_run.argtypes = [vp, ctypes.c_int, ctypes.c_int]
    lib.convstencil2d_run.restype = None
    lib.convstencil2d_copy_out.argtypes = [vp, dp]
    lib.convstencil2d_copy_out.restype = None
    lib.convstencil2d_free.argtypes = [vp]
    lib.convstencil2d_free.restype = None
    return lib


_FUSE = 3  # steps per launch for radius-1 shapes (paper: 7x7 = 3-fold 3x3)


def _fusion_enabled() -> bool:
    # KB_CONVSTENCIL_FUSION=0 restores the old one-step-per-launch mode
    # (useful to compare against pre-2026-11-17 results).
    return os.environ.get("KB_CONVSTENCIL_FUSION", "1") != "0"


def _self_convolve(weights: dict, k: int) -> dict:
    """Weights of k consecutive applications of the stencil, as ONE stencil.

    out[x] = sum_d w[d] in[x+d] applied twice gives sum_{d1,d2} w[d1] w[d2]
    in[x+d1+d2], so the k-step weight at offset d is the k-fold convolution
    of the weight table. Exact on a periodic grid (up to fp rounding order).
    """
    out = {(0, 0): 1.0}
    for _ in range(k):
        nxt: dict = {}
        for (a, b), wa in out.items():
            for (c, d), wb in weights.items():
                key = (a + c, b + d)
                nxt[key] = nxt.get(key, 0.0) + wa * wb
        out = nxt
    return out


def _radius(weights: dict) -> int:
    return max((max(abs(dr), abs(dc)) for dr, dc in weights), default=0)


def _build_params(weights: dict) -> np.ndarray:
    k = np.zeros((7, 7), dtype=np.float64)
    for (dr, dc), w in weights.items():
        if abs(dr) > _KERNEL_RADIUS or abs(dc) > _KERNEL_RADIUS:
            raise ValueError(
                f"convstencil-tcu: offset {(dr, dc)} exceeds the kernel's "
                f"fixed radius-{_KERNEL_RADIUS} (7x7) support")
        k[_KERNEL_RADIUS + dr, _KERNEL_RADIUS + dc] = w
    return np.ascontiguousarray(k.reshape(-1))


class ConvStencilTCU:
    name = "convstencil-tcu"
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"convstencil-tcu's kernel (DATA_TYPE, 2d_utils.h) is "
                f"double; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, workload_, params: dict):
        if not getattr(workload_, "linear", True):
            raise NotImplementedError(
                f"convstencil-tcu: {workload_.name!r} is a nonlinear kernel; "
                "ConvStencil computes weighted sums only")
        if workload_.dims != 2:
            raise NotImplementedError(
                "convstencil-tcu: only the 2D kernel is wired up in this "
                "integration pass; 1d and 3d were not bridged. dims=" +
                str(workload_.dims))
        if workload_.boundary != "periodic":
            raise NotImplementedError(
                "convstencil-tcu adapter only implements periodic-wrap "
                "boundary; workload requested " + str(workload_.boundary))
        m, n = workload_.grid_shape
        if m % _ROW_MULT or n % _COL_MULT:
            raise NotImplementedError(
                f"convstencil-tcu: kernel2d launches ceil(m/32) x ceil(n/64) "
                f"blocks with no tail guard; grid_shape must be a multiple of "
                f"({_ROW_MULT},{_COL_MULT}), got ({m},{n})")
        field = np.ascontiguousarray(workload_.initial_field(dtype=np.float64))
        timesteps = int(params.get("timesteps", workload_.timesteps))
        w = workload_.weights
        # Paper-faithful temporal fusion: radius-1 shapes run 3 real steps per
        # launch with the 3-fold self-convolved 7x7 kernel (main.cu
        # param_box_2d1r; README's box2d1r example only reproduces with
        # times*3 steps). T % 3 leftover steps run as ONE extra launch with
        # the (T % 3)-fold kernel. Radius 2-3 shapes are not fused (the paper
        # does not either: their 3-fold kernel would not fit in 7x7).
        fuse = _FUSE if (_fusion_enabled() and _radius(w) <= 1) else 1
        n_main, rem = divmod(timesteps, fuse)
        p_main = _build_params(_self_convolve(w, fuse) if fuse > 1 else w)
        p_tail = _build_params(_self_convolve(w, rem)) if rem else p_main
        n_tail = 1 if rem else 0
        params["temporal_fusion"] = fuse
        params["kernel_launches"] = n_main + n_tail
        dp = ctypes.POINTER(ctypes.c_double)
        handle = self.lib.convstencil2d_prepare(
            field.ctypes.data_as(dp), p_main.ctypes.data_as(dp),
            p_tail.ctypes.data_as(dp), m, n)
        if not handle:
            raise RuntimeError("convstencil2d_prepare returned a null handle")
        return {"handle": handle, "m": m, "n": n, "timesteps": timesteps,
                "n_main": n_main, "n_tail": n_tail}

    def run(self, h):
        # all T real steps on the device (n_main fused launches + an optional
        # tail launch); the CUDA event timer syncs on its stop event
        self.lib.convstencil2d_run(h["handle"], h["n_main"], h["n_tail"])
        return h

    def to_host(self, out):
        result = np.empty((out["m"], out["n"]), dtype=np.float64)
        self.lib.convstencil2d_copy_out(
            out["handle"], result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return result

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.convstencil2d_free(h["handle"])
        h.clear()


def create(precision: str):
    return ConvStencilTCU(precision)
