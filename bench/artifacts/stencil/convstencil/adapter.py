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
shape name); radius-1 shapes (star2d1r, box2d1r) are expressed by
zero-padding the unused outer ring of the 7x7, done in _build_params(). 1D
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
    lib.convstencil2d_prepare.argtypes = [dp, dp, ctypes.c_int, ctypes.c_int]
    lib.convstencil2d_prepare.restype = vp
    lib.convstencil2d_run.argtypes = [vp, ctypes.c_int]
    lib.convstencil2d_run.restype = None
    lib.convstencil2d_copy_out.argtypes = [vp, dp]
    lib.convstencil2d_copy_out.restype = None
    lib.convstencil2d_free.argtypes = [vp]
    lib.convstencil2d_free.restype = None
    return lib


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
        params49 = _build_params(workload_.weights)
        dp = ctypes.POINTER(ctypes.c_double)
        handle = self.lib.convstencil2d_prepare(
            field.ctypes.data_as(dp), params49.ctypes.data_as(dp), m, n)
        if not handle:
            raise RuntimeError("convstencil2d_prepare returned a null handle")
        return {"handle": handle, "m": m, "n": n,
                "timesteps": int(params.get("timesteps", workload_.timesteps))}

    def run(self, h):
        # all T steps on the device; the CUDA event timer syncs on its stop event
        self.lib.convstencil2d_run(h["handle"], h["timesteps"])
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
