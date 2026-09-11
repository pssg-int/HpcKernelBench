"""
Adapter for GPULZ (ICS 2023, "GPULZ: Optimizing LZSS Lossless Compression
for Multi-byte Data on Modern GPUs", conf/ics/ZhangTDYSTC23).

Wraps GPULZ's own compress kernels (compressKernelI + cub::DeviceScan
exclusive-sum + compressKernelIII, source/gpulz.cu) and decompress kernel
(decompressKernel, same file) through bridge_gpulz.cu (this directory) --
see that file's docstring for exactly what had to be split out of GPULZ's
own main() (which performs compression AND decompression together in one
call, by the artifact's own design/limitation per its README) and why.

Direction-fixed per the compression domain's contract
(kernelbench/domains/compression.py): this is the "-compress" registration.
Decompression is used ONLY for the correctness gate (to_host()/free()),
outside the timed region -- see those methods' docstrings. This mirrors the
fzgpu-compress/cuszp-compress adapters already integrated for the
lossy-compression track (same domain module, same split-timing discipline).

Byte semantics: GPULZ's kernels operate on a flat, dtype-agnostic multi-byte
symbol stream (INPUT_TYPE, compiled here as uint32_t, GPULZ's own shipped
default -- source/gpulz.cu line 15) -- they only ever compare/copy raw
symbol bit patterns, never interpret them as IEEE floats, so re-purposing
the compression domain's fp32 Field workload as a raw byte stream (its
in-memory bit pattern, not its numeric value) is exactly what MANS's/GPULZ's
own "multi-byte scientific/sensor integer data" input class already is
(benchspecs/lossless-compression/spec.yaml's variant 2) -- no semantic
mismatch, no special-casing of NaN/Inf needed.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "lossless-compression"
IMPL_NAME = "gpulz-compress"
PAPER_KEY = "conf/ics/ZhangTDYSTC23"
PRECISIONS = ["fp32"]  # matches the compression domain's DEFAULT_PRECISION;
                        # GPULZ itself is byte/symbol-agnostic (see docstring)

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
    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.gpulzc_prepare.argtypes = [u8p, ctypes.c_uint32]
    lib.gpulzc_prepare.restype = p
    lib.gpulzc_run.argtypes = [p]
    lib.gpulzc_run.restype = None
    lib.gpulzc_compressed_bytes.argtypes = [p]
    lib.gpulzc_compressed_bytes.restype = ctypes.c_size_t
    lib.gpulzc_free.argtypes = [p]
    lib.gpulzc_free.restype = None
    lib.gpulzd_run_and_copy.argtypes = [p, u8p, ctypes.c_uint32]
    lib.gpulzd_run_and_copy.restype = ctypes.c_int
    return lib


class GpulzCompress:
    name = "gpulz-compress"
    platform = "cuda"
    direction = "compress"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"gpulz-compress wraps a byte-stream codec via the fp32 smoke "
                f"workload's raw bytes; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        # H2D + workspace alloc (bridge_gpulz.cu::gpulzc_prepare), which also
        # zero-pads to GPULZ's own BLOCK_SIZE-aligned buffer length -- that
        # padding IS the artifact's own preprocessing requirement, timed
        # once here (same discipline as fzgpu-compress's chunk padding).
        data = np.ascontiguousarray(w.data)
        raw = data.tobytes()
        self._dtype = w.data.dtype
        self._shape = w.data.shape
        self._nbytes = len(raw)
        params["direction"] = self.direction
        buf = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
        handle = self.lib.gpulzc_prepare(buf, ctypes.c_uint32(len(raw)))
        if not handle:
            raise RuntimeError("gpulzc_prepare returned a null handle")
        return {"handle": handle, "nbytes": len(raw), "params": params}

    def run(self, h):
        # exactly one compress pass: compressKernelI + 2 cub exclusive-sum
        # scans + compressKernelIII (GPULZ's own kernels, unmodified) -- see
        # bridge_gpulz.cu::gpulzc_run
        self.lib.gpulzc_run(h["handle"])
        return h

    def to_host(self, out) -> np.ndarray:
        # Correctness gate only: decompress (decompressKernel, GPULZ's own
        # kernel, unmodified) + D2H copy. DELIBERATELY OUTSIDE the timed
        # region -- run() above is the only call the harness's
        # CudaEventTimer ever brackets; this method's cost never enters
        # times_ms.
        nbytes = out["nbytes"]
        buf = (ctypes.c_uint8 * nbytes)()
        got = self.lib.gpulzd_run_and_copy(out["handle"], buf, ctypes.c_uint32(nbytes))
        if got != nbytes:
            raise RuntimeError(f"gpulz decode size mismatch: got {got}, expected {nbytes}")
        arr = np.frombuffer(bytes(buf), dtype=self._dtype).reshape(self._shape)
        return arr.astype(np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h) -> None:
        # Ratio bookkeeping happens here (after all timed reps, before the
        # harness copies `params` for the cost model) so it never inflates
        # the timed region -- same discipline as this benchmark's other
        # lossless/lossy-compression GPU adapters.
        cmp_bytes = int(self.lib.gpulzc_compressed_bytes(h["handle"]))
        orig_bytes = h["nbytes"]
        params = h["params"]
        params["achieved_compressed_bytes"] = cmp_bytes
        params["compression_ratio"] = orig_bytes / cmp_bytes if cmp_bytes else float("inf")
        self.lib.gpulzc_free(h["handle"])
        h.clear()


def create(precision: str):
    return GpulzCompress(precision)
