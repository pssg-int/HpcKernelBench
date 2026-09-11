"""
Adapter for cuSZp3 / "VGC" (SC 2025, "GPU Lossy Compression for HPC Can Be
Versatile and Ultra-Fast", conf/sc/HuangDLC25).

The repo at https://github.com/szcompressor/cuSZp is one continuously
developed codebase spanning three SC papers -- SC'23 (cuSZp1, kernel
fusion), SC'24 (cuSZp2, new lossless modes), SC'25 (cuSZp3/"VGC",
dimensionality (1D/2D/3D) x 3-mode (fixed/plain/outlier) versatility,
README's own "cuSZp Versatile Support" table) -- and the commit cloned
here (see STATUS.md for the hash) IS the SC'25 code: its API already
exposes the dim x mode matrix that paper introduces (`cuszp_dim_t`
1D/2D/3D times `cuszp_mode_t` fixed/plain/outlier), not the single-mode
SC'23 original. PAPER_KEY is set to the SC'25 key accordingly (see
STATUS.md for the disambiguation evidence).

Wraps cuSZp's own generic device-pointer C API (cuSZp_compress /
cuSZp_decompress, include/cuSZp.h, README's "Use cuSZp as C/C++ Internal
API" section) directly -- no bridge.cu split needed at all, unlike this
benchmark's other CUDA artifacts. cuSZp ships as a genuine library
(source/CMakeLists.txt builds `cuSZp_shared`), so `libcuszp.so` (this
directory) is compiled straight from cuSZp's own unmodified sources
(build.sh) and called through ctypes, with device buffers allocated via
torch (the same `ctypes.c_void_p(tensor.data_ptr())` pattern already used
by kernelbench/impls/gpu_cuda.py's CustomSpMV/CustomSpMM/CustomSDDMM).

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
IMPL_NAME = "cuszp-compress"
PAPER_KEY = "conf/sc/HuangDLC25"  # SC'25 "VGC"/cuSZp3 -- see adapter docstring
PRECISIONS = ["fp32"]  # cuSZp also ships fp64 kernels (CUSZP_TYPE_DOUBLE),
                        # not wrapped here -- this adapter fixes fp32 to match
                        # the compression domain's DEFAULT_PRECISION and the
                        # smoke workloads' dtype

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "libcuszp.so")

# include/cuSZp.h enum values (mirrored here since ctypes doesn't parse C
# headers; kept in lockstep with that file by name -- see its docstring)
_CUSZP_DIM_1D = 1
_CUSZP_MODE_OUTLIER = 2   # "always highest compression ratios" per README
_CUSZP_TYPE_FLOAT = 0


class _Uint3(ctypes.Structure):
    """Mirrors CUDA's built-in uint3 (x,y,z uint32) -- cuSZp_compress's
    `dims` parameter. Unused content for 1D processing (passed as {0,0,0},
    exactly as cuSZp's own examples/cuSZp_test_f32_1D.cpp does), but the
    struct must still match layout/size for the ctypes call ABI to line up
    with the by-value uint3 argument."""
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]


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
    sig = [p, p, ctypes.c_size_t, None, ctypes.c_float,
           ctypes.c_int, _Uint3, ctypes.c_int, ctypes.c_int, p]
    sig_compress = list(sig)
    sig_compress[3] = ctypes.POINTER(ctypes.c_size_t)   # size_t* cmpSize (out)
    lib.cuSZp_compress.argtypes = sig_compress
    lib.cuSZp_compress.restype = None
    sig_decompress = list(sig)
    sig_decompress[3] = ctypes.c_size_t                  # size_t cmpSize (in)
    lib.cuSZp_decompress.argtypes = sig_decompress
    lib.cuSZp_decompress.restype = None
    return lib


class CuszpCompress:
    name = "cuszp-compress"
    platform = "cuda"
    direction = "compress"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"cuszp-compress wraps cuSZp's f32/outlier/1D kernel; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        # H2D + workspace alloc. cuSZp_compress internally cudaMallocs/
        # cudaFrees its own small offset/flag scratch buffers PER CALL
        # (src/cuSZp_entry_1D_f32.cu) -- that is the artifact's own design,
        # left untouched; only the ORIGINAL-data buffer, the compressed-
        # output buffer, and the decompressed-output buffer are allocated
        # here, once, as preprocessing.
        import torch
        eb = w.correctness_tolerance
        if eb is None:
            raise ValueError(
                f"{w.name}: no error bound attached (correctness_tolerance is "
                "None) -- cuszp-compress needs an absolute eb")
        data = np.ascontiguousarray(w.data, dtype=np.float32)
        self._dtype = w.data.dtype
        self._shape = w.data.shape
        self._eb = float(eb)
        nbEle = int(data.size)
        params["direction"] = self.direction
        params["error_bound_mode"] = w.error_bound_mode
        params["error_bound_value"] = w.error_bound_value
        params["error_bound_abs"] = eb
        params["cuszp_mode"] = "outlier"
        params["cuszp_processing_dim"] = "1D"

        d_ori = torch.from_numpy(data).to("cuda")
        # d_cmpBytes sized sizeof(float)*nbEle -- the worst-case bound
        # cuSZp's own examples (examples/cuSZp_test_f32_1D.cpp) allocate,
        # since fixed-length encoding never expands past the original size
        d_cmp = torch.empty(nbEle * 4, dtype=torch.uint8, device="cuda")
        d_dec = torch.empty(nbEle, dtype=torch.float32, device="cuda")
        return {
            "d_ori": d_ori, "d_cmp": d_cmp, "d_dec": d_dec,
            "nbEle": nbEle, "cmp_size": ctypes.c_size_t(0),
            "dims0": _Uint3(0, 0, 0), "params": params,
        }

    def run(self, h):
        # exactly one cuSZp_compress call (cuSZp's own single fused kernel
        # for this mode/dim -- see cuSZp_compress_1D_outlier_f32)
        self.lib.cuSZp_compress(
            ctypes.c_void_p(h["d_ori"].data_ptr()),
            ctypes.c_void_p(h["d_cmp"].data_ptr()),
            h["nbEle"], ctypes.byref(h["cmp_size"]),
            ctypes.c_float(self._eb), _CUSZP_DIM_1D, h["dims0"],
            _CUSZP_TYPE_FLOAT, _CUSZP_MODE_OUTLIER, ctypes.c_void_p(0))
        return h

    def to_host(self, out) -> np.ndarray:
        # Correctness gate only: decompress (cuSZp's own kernel, unmodified)
        # + D2H copy of the reconstructed array. DELIBERATELY OUTSIDE the
        # timed region -- run() above is the only call the harness's
        # CudaEventTimer ever brackets; this method's cost never enters
        # times_ms.
        import torch
        self.lib.cuSZp_decompress(
            ctypes.c_void_p(out["d_dec"].data_ptr()),
            ctypes.c_void_p(out["d_cmp"].data_ptr()),
            out["nbEle"], out["cmp_size"],
            ctypes.c_float(self._eb), _CUSZP_DIM_1D, out["dims0"],
            _CUSZP_TYPE_FLOAT, _CUSZP_MODE_OUTLIER, ctypes.c_void_p(0))
        torch.cuda.synchronize()
        arr = out["d_dec"].detach().to("cpu", dtype=torch.float64).numpy()
        return arr.reshape(self._shape)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h) -> None:
        # Ratio/error/PSNR bookkeeping happens here (after all timed reps)
        # so it never inflates the timed region -- same discipline as the
        # CPU quantize-zlib codec in kernelbench/domains/compression.py and
        # this benchmark's other lossy-compression GPU adapter (pfpl).
        import torch
        cmp_bytes = int(h["cmp_size"].value)
        orig_bytes = h["nbEle"] * 4
        params = h["params"]
        params["achieved_compressed_bytes"] = cmp_bytes
        params["compression_ratio"] = orig_bytes / cmp_bytes if cmp_bytes else float("inf")

        # one more decode (untimed, same call as to_host) to record
        # achieved max error / PSNR from whatever the LAST timed run()
        # actually produced
        self.lib.cuSZp_decompress(
            ctypes.c_void_p(h["d_dec"].data_ptr()),
            ctypes.c_void_p(h["d_cmp"].data_ptr()),
            h["nbEle"], h["cmp_size"],
            ctypes.c_float(self._eb), _CUSZP_DIM_1D, h["dims0"],
            _CUSZP_TYPE_FLOAT, _CUSZP_MODE_OUTLIER, ctypes.c_void_p(0))
        torch.cuda.synchronize()
        recon = h["d_dec"].detach().to("cpu", dtype=torch.float64).numpy().ravel()
        orig = h["d_ori"].detach().to("cpu", dtype=torch.float64).numpy().ravel()
        diff = recon - orig
        mse = float(np.mean(diff * diff))
        value_range = float(orig.max() - orig.min())
        psnr = float("inf") if mse == 0.0 else \
            20.0 * np.log10(value_range) - 10.0 * np.log10(mse)
        params["achieved_max_abs_error"] = float(np.max(np.abs(diff)))
        params["psnr_db"] = psnr

        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return CuszpCompress(precision)
