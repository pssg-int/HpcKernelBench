"""
Adapter for FZ-GPU (HPDC 2023, "FZ-GPU: A Fast and High-Ratio Lossy
Compressor for Scientific Computing Applications on GPUs",
conf/hpdc/ZhangTD0F0TC23).

Wraps FZ-GPU's own compression kernels (launch_construct_LorenzoI_var +
compressionFusedKernel, source/src/fz.cu) and decompression kernels
(decompressionFusedKernel + launch_reconstruct_LorenzoI_var, same file)
through bridge_fzgpu.cu (this directory) -- see that file's docstring for
exactly what had to be split out of FZ-GPU's own runFzgpu() (which
performs compression AND decompression together in one call, by the
artifact's own design/limitation per its README) and why.

Direction-fixed per the compression domain's contract
(kernelbench/domains/compression.py): this is the "-compress" registration.
Decompression is used ONLY for the correctness gate (to_host()/free()),
outside the timed region -- see those methods' docstrings.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "lossy-compression"
IMPL_NAME = "fzgpu-compress"
PAPER_KEY = "conf/hpdc/ZhangTD0F0TC23"
PRECISIONS = ["fp32"]  # FZ-GPU's kernels are float32-only (fz.cu never instantiates fp64)

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "bridge.so")


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    p = ctypes.c_void_p
    f32p = ctypes.POINTER(ctypes.c_float)
    lib.fzgpuc_prepare.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, f32p, ctypes.c_double]
    lib.fzgpuc_prepare.restype = p
    lib.fzgpuc_run.argtypes = [p]
    lib.fzgpuc_run.restype = None
    lib.fzgpuc_compressed_bytes.argtypes = [p]
    lib.fzgpuc_compressed_bytes.restype = ctypes.c_size_t
    lib.fzgpuc_free.argtypes = [p]
    lib.fzgpuc_free.restype = None
    lib.fzgpud_run_and_copy.argtypes = [p, f32p]
    lib.fzgpud_run_and_copy.restype = ctypes.c_int
    return lib


class FzgpuCompress:
    name = "fzgpu-compress"
    platform = "cuda"
    direction = "compress"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"fzgpu-compress wraps FZ-GPU's float32-only kernels; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        # H2D + workspace alloc (bridge_fzgpu.cu::fzgpuc_prepare), which
        # also zero-pads to FZ-GPU's own chunk-aligned buffer length --
        # that padding IS the artifact's own preprocessing requirement,
        # timed once here.
        eb = w.correctness_tolerance
        if eb is None:
            raise ValueError(
                f"{w.name}: no error bound attached (correctness_tolerance is "
                "None) -- fzgpu-compress needs an absolute eb")
        # Note on FZ-GPU's own convention: its CLI only ever takes one bare
        # number (README: `./fz-gpu ... 1e-3`), used internally as a
        # fraction of the data's own value range -- there is no abs-mode
        # branch anywhere in fz.cu. This adapter passes the harness's own
        # already-resolved ABSOLUTE bound straight into the kernels
        # instead (see bridge_fzgpu.cu's docstring for why that sidesteps
        # FZ-GPU's internal range computation entirely), so it is correct
        # regardless of `w.error_bound_mode`.
        data = np.ascontiguousarray(w.data, dtype=np.float32)
        shape = w.data.shape
        if len(shape) == 3:
            dimz, dimy, dimx = shape  # row-major C order: last axis fastest
        elif len(shape) == 2:
            dimz = 1
            dimy, dimx = shape
        elif len(shape) == 1:
            dimz, dimy, dimx = 1, 1, shape[0]
        else:
            raise NotImplementedError(f"fzgpu-compress: unsupported shape {shape}")
        self._dtype = w.data.dtype
        self._shape = shape
        self._eb = float(eb)
        params["direction"] = self.direction
        params["error_bound_mode"] = w.error_bound_mode
        params["error_bound_value"] = w.error_bound_value
        params["error_bound_abs"] = eb
        params["fzgpu_dims_xyz"] = [dimx, dimy, dimz]
        handle = self.lib.fzgpuc_prepare(
            dimx, dimy, dimz, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_double(self._eb))
        if not handle:
            raise RuntimeError("fzgpuc_prepare returned a null handle")
        nbEle = dimx * dimy * dimz
        return {"handle": handle, "nbEle": nbEle, "params": params, "_data": data}

    def run(self, h):
        # exactly one compress call: launch_construct_LorenzoI_var +
        # compressionFusedKernel (FZ-GPU's own kernels, unmodified) --
        # see bridge_fzgpu.cu::fzgpuc_run for the required per-call reset
        self.lib.fzgpuc_run(h["handle"])
        return h

    def to_host(self, out) -> np.ndarray:
        # Correctness gate only: decompress (FZ-GPU's own kernels,
        # unmodified) + D2H copy. DELIBERATELY OUTSIDE the timed region --
        # run() above is the only call the harness's CudaEventTimer ever
        # brackets; this method's cost never enters times_ms.
        nbEle = out["nbEle"]
        buf = np.empty(nbEle, dtype=np.float32)
        got = self.lib.fzgpud_run_and_copy(
            out["handle"], buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if got != nbEle:
            raise RuntimeError(f"fzgpu decode size mismatch: got {got}, expected {nbEle}")
        return buf.astype(np.float64).reshape(self._shape)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h) -> None:
        # Ratio/error/PSNR bookkeeping happens here (after all timed reps)
        # so it never inflates the timed region -- same discipline as this
        # benchmark's other lossy-compression GPU adapters.
        cmp_bytes = int(self.lib.fzgpuc_compressed_bytes(h["handle"]))
        orig_bytes = h["nbEle"] * 4
        params = h["params"]
        params["achieved_compressed_bytes"] = cmp_bytes
        params["compression_ratio"] = orig_bytes / cmp_bytes if cmp_bytes else float("inf")

        # one more decode (untimed, same call as to_host) to record
        # achieved max error / PSNR from whatever the LAST timed run()
        # actually produced
        nbEle = h["nbEle"]
        buf = np.empty(nbEle, dtype=np.float32)
        got = self.lib.fzgpud_run_and_copy(
            h["handle"], buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if got == nbEle:
            recon = buf.astype(np.float64)
            orig = h["_data"].astype(np.float64).ravel()
            diff = recon - orig
            mse = float(np.mean(diff * diff))
            value_range = float(orig.max() - orig.min())
            psnr = float("inf") if mse == 0.0 else \
                20.0 * np.log10(value_range) - 10.0 * np.log10(mse)
            params["achieved_max_abs_error"] = float(np.max(np.abs(diff)))
            params["psnr_db"] = psnr

        self.lib.fzgpuc_free(h["handle"])
        h.clear()


def create(precision: str):
    return FzgpuCompress(precision)
