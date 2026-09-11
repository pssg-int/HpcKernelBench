"""
Adapter for PFPL (IPDPS 2025, "Fast and Effective Lossy Compression on GPUs
and CPUs with Guaranteed Error Bounds", conf/ipps/FallinADCB25).

Wraps PFPL's own single-precision absolute-error-bound GPU kernels
(d_encode / d_decode, source/src/f32_abs_{comp,decomp}_gpu.cu) through
bridge_encode.cu / bridge_decode.cu (this directory) -- see those files'
docstrings for exactly what had to be split out of PFPL's own main()
(file I/O + its own NUM_RUNS=9 timing loop) and why.

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
IMPL_NAME = "pfpl-compress"
PAPER_KEY = "conf/ipps/FallinADCB25"
PRECISIONS = ["fp32"]  # this adapter wraps PFPL's f32_abs_{comp,decomp}_gpu kernels

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
    u8p = ctypes.POINTER(ctypes.c_ubyte)
    lib.pfplc_prepare.argtypes = [ctypes.c_int, u8p, ctypes.c_float]
    lib.pfplc_prepare.restype = p
    lib.pfplc_run.argtypes = [p]
    lib.pfplc_run.restype = None
    lib.pfplc_encoded_size.argtypes = [p]
    lib.pfplc_encoded_size.restype = ctypes.c_int
    lib.pfplc_free.argtypes = [p]
    lib.pfplc_free.restype = None
    lib.pfpld_run_and_copy.argtypes = [p, u8p]
    lib.pfpld_run_and_copy.restype = ctypes.c_int
    return lib


class PfplCompress:
    name = "pfpl-compress"
    platform = "cuda"
    direction = "compress"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"pfpl-compress wraps PFPL's f32_abs GPU kernels; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        # H2D + workspace alloc (bridge_encode.cu::pfplc_prepare) -- PFPL's
        # own device allocation/H2D-copy block, lifted from its main(),
        # timed once here as preprocessing.
        eb = w.correctness_tolerance
        if eb is None:
            raise ValueError(
                f"{w.name}: no error bound attached (correctness_tolerance is "
                "None) -- pfpl-compress needs an absolute eb")
        data = np.ascontiguousarray(w.data, dtype=np.float32)
        self._dtype = w.data.dtype
        self._shape = w.data.shape
        self._eb = float(eb)
        insize = int(data.nbytes)
        params["direction"] = self.direction
        params["error_bound_mode"] = w.error_bound_mode
        params["error_bound_value"] = w.error_bound_value
        params["error_bound_abs"] = eb
        handle = self.lib.pfplc_prepare(
            insize, data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.c_float(self._eb))
        if not handle:
            raise RuntimeError("pfplc_prepare returned a null handle")
        # keep `data` alive: ctypes doesn't, and pfplc_prepare only reads
        # it synchronously (no async H2D), so this is precautionary
        return {"handle": handle, "insize": insize, "params": params, "_data": data}

    def run(self, h):
        # exactly one d_encode launch (PFPL's own kernel, unmodified) --
        # see bridge_encode.cu::pfplc_run for the required per-call reset
        self.lib.pfplc_run(h["handle"])
        return h

    def to_host(self, out) -> np.ndarray:
        # Correctness gate only: decompress (PFPL's own d_decode kernel,
        # unmodified, bridge_decode.cu) + D2H copy of the reconstructed
        # array. DELIBERATELY OUTSIDE the timed region -- run() above is
        # the only call the harness's Timer ever brackets; this method's
        # cost never enters times_ms.
        insize = out["insize"]
        buf = np.empty(insize, dtype=np.uint8)
        decsize = self.lib.pfpld_run_and_copy(
            out["handle"], buf.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        if decsize != insize:
            raise RuntimeError(
                f"pfpl decode size mismatch: got {decsize}, expected {insize}")
        arr = np.frombuffer(buf.tobytes(), dtype=self._dtype).reshape(self._shape)
        return arr.astype(np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h) -> None:
        # Ratio/error/PSNR bookkeeping happens here (after all timed reps,
        # before the harness copies `params` for the cost model) so it
        # never inflates the timed region -- same discipline as the CPU
        # quantize-zlib codec in kernelbench/domains/compression.py.
        encsize = self.lib.pfplc_encoded_size(h["handle"])
        orig = h["insize"]
        params = h["params"]
        params["achieved_compressed_bytes"] = int(encsize)
        params["compression_ratio"] = orig / encsize if encsize else float("inf")

        # one more decode (untimed, same call as to_host) to record
        # achieved max error / PSNR from whatever the LAST timed run()
        # actually produced
        buf = np.empty(orig, dtype=np.uint8)
        decsize = self.lib.pfpld_run_and_copy(
            h["handle"], buf.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        if decsize == orig:
            recon = np.frombuffer(buf.tobytes(), dtype=self._dtype).astype(np.float64)
            orig_arr = h["_data"].astype(np.float64).ravel()
            diff = recon - orig_arr
            mse = float(np.mean(diff * diff))
            value_range = float(orig_arr.max() - orig_arr.min())
            psnr = float("inf") if mse == 0.0 else \
                20.0 * np.log10(value_range) - 10.0 * np.log10(mse)
            params["achieved_max_abs_error"] = float(np.max(np.abs(diff)))
            params["psnr_db"] = psnr

        self.lib.pfplc_free(h["handle"])
        h.clear()


def create(precision: str):
    return PfplCompress(precision)
