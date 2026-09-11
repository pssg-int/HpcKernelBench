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
    our domain's periodic-wrap T-sweep recursion for T>1. This adapter's
    run() performs its own T-sweep loop instead, calling the kernel once per
    timestep (times=1 each) and re-padding the periodic-wrap halo from the
    CURRENT field between calls (host-side, matches
    kernelbench/domains/stencil.py's np.roll-based reference exactly).

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
_ROW_HALO = 3
_COL_HALO = 4


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
    lib.convstencil2d_sweep.argtypes = [dp, dp, dp, ctypes.c_int, ctypes.c_int]
    lib.convstencil2d_sweep.restype = None
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
        u0 = workload_.initial_field(dtype=np.float64)
        params49 = _build_params(workload_.weights)
        timesteps = int(params.get("timesteps", workload_.timesteps))
        h = {}
        h["field"] = np.ascontiguousarray(u0)
        h["params49"] = params49
        h["m"] = m
        h["n"] = n
        h["timesteps"] = timesteps
        return h

    def run(self, h):
        field = h["field"]
        m = h["m"]
        n = h["n"]
        params49 = h["params49"]
        dp = ctypes.POINTER(ctypes.c_double)
        for _ in range(h["timesteps"]):
            padded_in = np.pad(field, ((_ROW_HALO, _ROW_HALO), (_COL_HALO, _COL_HALO)), mode="wrap")
            padded_in = np.ascontiguousarray(padded_in, dtype=np.float64)
            padded_out = np.zeros_like(padded_in)
            self.lib.convstencil2d_sweep(padded_in.ctypes.data_as(dp), padded_out.ctypes.data_as(dp), params49.ctypes.data_as(dp), m, n)
            field = padded_out[_ROW_HALO:_ROW_HALO + m, _COL_HALO:_COL_HALO + n]
            field = np.ascontiguousarray(field)
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
    return ConvStencilTCU(precision)
