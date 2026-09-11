"""
Adapter for MANS (Huang et al., SC 2025, "MANS: Efficient and Portable ANS
Encoding for Multi-Byte Integer Data on CPUs and GPUs",
conf/sc/HuangYYLGLJWFHT25).

Wraps MANS's own NVIDIA device-pointer API
(mans::nv::compress_internal_device / decompress_internal_device,
source/nv/mans_nv.{h,cpp}) directly -- no split of an end-to-end CLI main()
needed here, unlike this benchmark's other single-file-CLI GPU compression
artifacts (gpulz, fzgpu, pfpl): MANS genuinely ships a reusable
device-pointer library entry point for its NVIDIA backend. `bridge_mans.cu`
(this directory) is a thin ctypes-callable extern "C" forwarder (needed only
because the underlying C++ functions take a `std::size_t&` out-parameter
ctypes cannot call directly) -- see that file's docstring.

Direction-fixed per the compression domain's contract
(kernelbench/domains/compression.py): this is the "-compress" registration.
Decompression is used ONLY for the correctness gate (to_host()/free()),
outside the timed region -- mirrors the gpulz-compress/fzgpu-compress/
cuszp-compress adapters already integrated in this benchmark.

Device buffers are allocated with torch (`ctypes.c_void_p(tensor.data_ptr())`),
the same pattern `kernelbench/impls/gpu_cuda.py`'s CustomSpMV/CustomSpMM/
CustomSDDMM and this benchmark's cuszp-compress adapter already use.

Byte semantics: MANS's ADM+ANS pipeline operates on a flat multi-byte
integer stream (uint16 here, MANS's own default dtype -- mans_api.cpp's
`default_params()`) -- it only ever bit-maps/entropy-codes the integer
VALUES, no floating-point interpretation anywhere in the pipeline, so
re-purposing the compression domain's fp32 Field workload's raw bytes
(reinterpreted as a uint16 stream) is exactly the same "multi-byte
scientific/sensor integer data" input class MANS's own paper targets
(spec variant 2's EXAFEL/quant-code domain) -- same convention already used
by this benchmark's gpulz-compress adapter for its own byte-stream input.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "lossless-compression"
IMPL_NAME = "mans-compress"
PAPER_KEY = "conf/sc/HuangYYLGLJWFHT25"
PRECISIONS = ["fp32"]  # matches the compression domain's DEFAULT_PRECISION;
                        # MANS's own codec is dtype-agnostic over its raw
                        # bytes (see docstring) -- reinterpreted as uint16

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "bridge.so")

_MANS_DTYPE_U16 = 0   # mans::DataType::U16 (source/mans_defs.h)
_MANS_MODE_P = 0      # mans::Mode::P ("portable, GPU-consistent" -- README)


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
    szp = ctypes.POINTER(ctypes.c_size_t)
    lib.mans_max_compress_bytes.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t]
    lib.mans_max_compress_bytes.restype = ctypes.c_size_t
    lib.mans_compress_device.argtypes = [
        p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32, p, szp]
    lib.mans_compress_device.restype = ctypes.c_int
    lib.mans_decompress_device.argtypes = [
        p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32, p, szp]
    lib.mans_decompress_device.restype = ctypes.c_int
    return lib


class MansCompress:
    name = "mans-compress"
    platform = "cuda"
    direction = "compress"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"mans-compress wraps a uint16-stream codec via the fp32 smoke "
                f"workload's raw bytes; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        import torch

        # MANS's ADM+ANS pipeline needs a whole number of uint16 elements;
        # fp32 data is always an even number of bytes, so no padding is
        # needed here (unlike gpulz-compress's BLOCK_SIZE-aligned padding).
        data = np.ascontiguousarray(w.data)
        raw = data.tobytes()
        if len(raw) % 2 != 0:
            raise ValueError(f"{w.name}: odd byte count {len(raw)}, cannot "
                              f"view as a uint16 stream for mans-compress")
        u16 = np.frombuffer(raw, dtype=np.uint16)
        num_elements = int(u16.size)

        self._dtype = w.data.dtype
        self._shape = w.data.shape
        self._nbytes = len(raw)
        self._num_elements = num_elements
        params["direction"] = self.direction
        params["mans_dtype"] = "u16"
        params["mans_mode"] = "p"

        max_cmp_bytes = int(self.lib.mans_max_compress_bytes(
            _MANS_DTYPE_U16, _MANS_MODE_P, ctypes.c_size_t(num_elements)))
        if max_cmp_bytes <= 0:
            raise RuntimeError(
                f"mans_max_compress_bytes returned {max_cmp_bytes} for "
                f"{num_elements} elements")

        d_in = torch.from_numpy(u16.copy()).to("cuda")          # uint16, device
        d_cmp = torch.empty(max_cmp_bytes, dtype=torch.uint8, device="cuda")
        d_dec = torch.empty(num_elements, dtype=torch.uint16, device="cuda")
        return {
            "d_in": d_in, "d_cmp": d_cmp, "d_dec": d_dec,
            "num_elements": num_elements, "cmp_size": ctypes.c_size_t(0),
            "params": params,
        }

    def run(self, h):
        # exactly one mans_compress_device call = MANS's own
        # compress_internal_device (ADM mapping + entropy-code stage,
        # unmodified) -- see bridge_mans.cu
        rc = self.lib.mans_compress_device(
            ctypes.c_void_p(h["d_in"].data_ptr()), ctypes.c_size_t(h["num_elements"]),
            _MANS_DTYPE_U16, _MANS_MODE_P,
            ctypes.c_void_p(h["d_cmp"].data_ptr()), ctypes.byref(h["cmp_size"]))
        if rc != 0:
            raise RuntimeError(f"mans_compress_device failed (rc={rc})")
        return h

    def to_host(self, out) -> np.ndarray:
        # Correctness gate only: decompress (mans_decompress_device, MANS's
        # own kernel, unmodified) + D2H copy. DELIBERATELY OUTSIDE the timed
        # region -- run() above is the only call the harness's
        # CudaEventTimer ever brackets; this method's cost never enters
        # times_ms.
        import torch
        dec_size = ctypes.c_size_t(0)
        rc = self.lib.mans_decompress_device(
            ctypes.c_void_p(out["d_cmp"].data_ptr()), out["cmp_size"],
            _MANS_DTYPE_U16, _MANS_MODE_P,
            ctypes.c_void_p(out["d_dec"].data_ptr()), ctypes.byref(dec_size))
        if rc != 0:
            raise RuntimeError(f"mans_decompress_device failed (rc={rc})")
        torch.cuda.synchronize()
        got_elements = dec_size.value // 2
        if got_elements != out["num_elements"]:
            raise RuntimeError(
                f"mans decode size mismatch: got {got_elements} u16 elements, "
                f"expected {out['num_elements']}")
        u16_host = out["d_dec"].detach().to("cpu").numpy()
        arr = np.frombuffer(u16_host.tobytes(), dtype=self._dtype).reshape(self._shape)
        return arr.astype(np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h) -> None:
        # Ratio bookkeeping happens here (after all timed reps, before the
        # harness copies `params` for the cost model) so it never inflates
        # the timed region -- same discipline as this benchmark's other
        # lossless/lossy-compression GPU adapters.
        cmp_bytes = int(h["cmp_size"].value)
        orig_bytes = h["num_elements"] * 2
        params = h["params"]
        params["achieved_compressed_bytes"] = cmp_bytes
        params["compression_ratio"] = orig_bytes / cmp_bytes if cmp_bytes else float("inf")
        h.clear()
        import torch
        torch.cuda.empty_cache()


def create(precision: str):
    return MansCompress(precision)
