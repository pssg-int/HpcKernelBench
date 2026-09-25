"""
The cuDNN stencil baseline the Tensor-Core stencil papers compare against:
ConvStencil (PPoPP'24), LoRAStencil (SC'24), FlashFFTStencil (PPoPP'25) and
SPIDER (PPoPP'26) all report speedup over "cuDNN", and the programs they run
for it are ConvStencil's src/cudnn/conv_*.cu (SPIDER's Figure10_run.sh calls
them from its ConvStencil submodule). bridge.cu ports that call sequence:
one cudnnConvolutionForward per step with CUDNN_CONVOLUTION_FWD_ALGO_
IMPLICIT_PRECOMP_GEMM, fp64, zero padding of `radius`, ping-pong buffers, no
sync inside the loop. See bridge.cu's header for what is kept verbatim and
what differs.

This is a library BASELINE, not a paper's contribution; PAPER_KEY names the
paper whose artifact ships the baseline programs.

Boundary: cuDNN pads with zeros outside the grid and recomputes every cell,
i.e. the domain's "zero-halo" convention, selected in prepare(). The
torch-conv-stencil baseline pads circularly (periodic); the two compute the
same stencil with different boundaries and both are gated against their own.

Precision: fp64 (what the papers run) and fp16 (data half, compute float;
added so SPIDER's fp16 result has a same-precision library comparator).
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "stencil"
IMPL_NAME = "cudnn-stencil"
PAPER_KEY = "conf/ppopp/ChenLWBWMYZCY24"   # ships src/cudnn/conv_*.cu
ROLE = "baseline"
PRECISIONS = ["fp64", "fp16"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "bridge.so")
_PREC = {"fp64": (0, np.float64), "fp16": (1, np.float16)}


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    vp, ip = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
    lib.cudnn_stencil_prepare.argtypes = [vp, vp, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int]
    lib.cudnn_stencil_prepare.restype = vp
    lib.cudnn_stencil_run.argtypes = [vp, ctypes.c_int]
    lib.cudnn_stencil_run.restype = ctypes.c_int
    lib.cudnn_stencil_copy_out.argtypes = [vp, vp]
    lib.cudnn_stencil_copy_out.restype = ctypes.c_int
    lib.cudnn_stencil_free.argtypes = [vp]
    lib.cudnn_stencil_free.restype = None
    lib.cudnn_stencil_last_error.restype = ctypes.c_char_p
    lib.cudnn_stencil_last_error_code.restype = ctypes.c_int
    lib.cudnn_stencil_version.restype = ctypes.c_size_t
    lib.cudnn_stencil_algo.argtypes = [vp]
    lib.cudnn_stencil_algo.restype = ctypes.c_int
    return lib


# cudnnConvolutionFwdAlgo_t values, for the result record
_ALGO_NAMES = ["IMPLICIT_GEMM", "IMPLICIT_PRECOMP_GEMM", "GEMM", "DIRECT", "FFT",
               "FFT_TILING", "WINOGRAD", "WINOGRAD_NONFUSED"]


class CudnnStencil:
    name = IMPL_NAME
    platform = "cuda"
    # 0: IMPLICIT_PRECOMP_GEMM, as ConvStencil's src/cudnn programs (and so
    # LoRAStencil's and SPIDER's cuDNN numbers); 1: the fastest measured
    # algorithm, as FlashFFTStencil's benchmarks/cudnn/cudnn-test.cpp
    ALGO_MODE = 0

    def __init__(self, precision: str = "fp64"):
        if precision not in _PREC:
            raise NotImplementedError(
                f"cudnn-stencil is built for {sorted(_PREC)}; requested {precision!r}")
        self.precision = precision
        self.lib = _load_lib()

    def _error(self) -> str:
        return (self.lib.cudnn_stencil_last_error() or b"").decode(errors="replace")

    def prepare(self, workload_, params: dict):
        if not getattr(workload_, "linear", True):
            raise NotImplementedError(
                f"cudnn-stencil: {workload_.name!r} is nonlinear; a convolution "
                "computes weighted sums only")
        if not 1 <= workload_.dims <= 3:
            raise NotImplementedError(f"cudnn-stencil: dims={workload_.dims} not supported")
        params["boundary"] = "zero-halo"
        code, dtype = _PREC[self.precision]
        field = np.ascontiguousarray(workload_.initial_field(dtype=dtype))
        kernel = np.ascontiguousarray(workload_.dense_kernel(dtype=np.float64, flip=False)
                                      .astype(dtype))
        shape = (ctypes.c_int * workload_.dims)(*workload_.grid_shape)
        handle = self.lib.cudnn_stencil_prepare(
            field.ctypes.data, kernel.ctypes.data, workload_.dims, shape,
            workload_.radius, code, self.ALGO_MODE)
        if not handle:
            msg = f"cudnn-stencil prepare: {self._error()}"
            if self.lib.cudnn_stencil_last_error_code() == 1:
                raise NotImplementedError(msg)
            raise RuntimeError(msg)
        algo = int(self.lib.cudnn_stencil_algo(handle))
        # recorded with the result (the runner stores `params`)
        params["cudnn_algo"] = _ALGO_NAMES[algo] if algo < len(_ALGO_NAMES) else str(algo)
        params["cudnn_version"] = int(self.lib.cudnn_stencil_version())
        return {"handle": handle, "shape": tuple(workload_.grid_shape), "dtype": dtype,
                "timesteps": int(params.get("timesteps", workload_.timesteps))}

    def run(self, h):
        if self.lib.cudnn_stencil_run(h["handle"], h["timesteps"]):
            raise RuntimeError(f"cudnn-stencil run: {self._error()}")
        return h

    def to_host(self, out) -> np.ndarray:
        result = np.empty(out["shape"], dtype=out["dtype"])
        if self.lib.cudnn_stencil_copy_out(out["handle"], result.ctypes.data):
            raise RuntimeError(f"cudnn-stencil copy_out: {self._error()}")
        return result.astype(np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.cudnn_stencil_free(h["handle"])
        h.clear()


def create(precision: str):
    return CudnnStencil(precision)
