"""
Adapter for cuSZp v1 (SC 2023, "cuSZp: An Ultra-fast GPU Error-bounded Lossy
Compression Framework with Optimized End-to-End Data Movement", conf/sc/
HuangD0LC23).

The repo at https://github.com/szcompressor/cuSZp is one continuously
developed codebase spanning three SC papers -- see ../cuszp/STATUS.md's
"PAPER_KEY disambiguation" section for the full evidence trail. That sibling
adapter's clone (HEAD, no tag pinned at clone time) turned out to already be
the SC'25 ("VGC"/cuSZp3, dim x mode versatility matrix) code, not the SC'23
original the spec survey's evidence entry actually cited. This adapter closes
that gap: it wraps tag `cuSZp-V1.1`, the single-mode, no-dim-parameter API
(`SZp_compress_deviceptr_f32` / `SZp_decompress_deviceptr_f32`,
include/cuSZp_entry_f32.h) that IS the SC'23 kernel -- confirmed a genuinely
distinct code path, not just a version bump: different source files
(src/cuSZp_f32.cu, a single fused compress/decompress kernel pair with no
`cuszp_dim_t`/`cuszp_mode_t` concept at all) and a different, narrower entry
signature (no `dim`/`dims`/`mode` params -- see that header).

`source/` here is a git WORKTREE of the same local clone ../cuszp/source
checked out to `cuSZp-V1.1` (`git worktree add ../cuszp-v1/source
cuSZp-V1.1`, run from inside ../cuszp/source) rather than a second network
clone or a plain symlink: a plain symlink would point at ../cuszp/source's
shared working directory and checked-out HEAD, so checking out V1.1 there
would also switch the SC'25 adapter's own already-BUILT+GATED source tree
out from under it. A linked worktree gives this directory its own
independent HEAD/index while still sharing the same `.git` object database
(no re-download) -- see source.provenance for the exact recipe, including a
plain-clone fallback if the worktree link ever breaks (e.g. ../cuszp/source
is deleted or re-cloned fresh).

Wraps the SC'23 kernel's C API directly via ctypes (libcuszp_v1.so, built by
build.sh from this tag's own unmodified `.cu` sources -- zero patches to
`source/`), same pattern as ../cuszp/adapter.py and
kernelbench/impls/gpu_cuda.py's other native-CUDA ctypes kernels.

Direction-fixed per the compression domain's contract
(kernelbench/domains/compression.py): this is the "-compress" registration.
Decompression is used ONLY for the correctness gate (to_host()/free()),
outside the timed region.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "lossy-compression"
IMPL_NAME = "cuszp-v1-compress"
PAPER_KEY = "conf/sc/HuangD0LC23"  # SC'23 original "cuSZp" -- see adapter docstring
PRECISIONS = ["fp32"]  # v1 also ships fp64 kernels (SZp_*_f64), not wrapped here --
                        # fixes fp32 to match the compression domain's
                        # DEFAULT_PRECISION and the smoke workloads' dtype

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "libcuszp_v1.so")


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    # Calls go through bridge_cuszp_v1.cu's extern "C" trampolines
    # (cuszpv1_compress_f32/cuszpv1_decompress_f32), not the mangled
    # SZp_*_deviceptr_f32 symbols directly -- v1's own header
    # (source/include/cuSZp_entry_f32.h) has no extern "C" guard, so nvcc
    # emits Itanium-mangled C++ symbols ctypes cannot look up by name
    # (confirmed via `nm -D libcuszp_v1.so`). See bridge_cuszp_v1.cu.
    lib = ctypes.CDLL(_SO_PATH)
    p = ctypes.c_void_p
    lib.cuszpv1_compress_f32.argtypes = [
        p, p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_float, p]
    lib.cuszpv1_compress_f32.restype = None
    lib.cuszpv1_decompress_f32.argtypes = [
        p, p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, p]
    lib.cuszpv1_decompress_f32.restype = None
    return lib


class CuszpV1Compress:
    name = "cuszp-v1-compress"
    platform = "cuda"
    direction = "compress"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"cuszp-v1-compress wraps cuSZp v1's f32 kernel; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        # H2D + workspace alloc, mirroring ../cuszp/adapter.py's discipline:
        # only the ORIGINAL-data buffer, the compressed-output buffer, and the
        # decompressed-output buffer are allocated here, once, as
        # preprocessing. Unlike the SC'25 kernel, this v1 entry point does its
        # own cmpOffset/flag scratch alloc+free INSIDE the call every time
        # (src/cuSZp_entry_f32.cu's SZp_compress_deviceptr_f32) -- that is
        # this kernel's own design (a real, if minor, per-call allocation
        # cost the SC'25 successor's README explicitly says it optimized
        # away), left completely untouched per ARTIFACT_GUIDE rule 3.
        import torch
        eb = w.correctness_tolerance
        if eb is None:
            raise ValueError(
                f"{w.name}: no error bound attached (correctness_tolerance is "
                "None) -- cuszp-v1-compress needs an absolute eb")
        data = np.ascontiguousarray(w.data, dtype=np.float32)
        self._dtype = w.data.dtype
        self._shape = w.data.shape
        self._eb = float(eb)
        nbEle = int(data.size)
        params["direction"] = self.direction
        params["error_bound_mode"] = w.error_bound_mode
        params["error_bound_value"] = w.error_bound_value
        params["error_bound_abs"] = eb
        params["cuszp_variant"] = "v1-plain (SC'23)"

        d_ori = torch.from_numpy(data).to("cuda")
        # d_cmp sized sizeof(float)*nbEle -- the same worst-case bound v1's
        # own hostptr wrapper (src/cuSZp_entry_f32.cu) allocates
        d_cmp = torch.empty(nbEle * 4, dtype=torch.uint8, device="cuda")
        d_dec = torch.empty(nbEle, dtype=torch.float32, device="cuda")
        return {
            "d_ori": d_ori, "d_cmp": d_cmp, "d_dec": d_dec,
            "nbEle": nbEle, "cmp_size": ctypes.c_size_t(0),
            "params": params,
        }

    def run(self, h):
        # exactly one SZp_compress_deviceptr_f32 call (v1's own single fused
        # compress kernel -- SZp_compress_kernel_f32)
        self.lib.cuszpv1_compress_f32(
            ctypes.c_void_p(h["d_ori"].data_ptr()),
            ctypes.c_void_p(h["d_cmp"].data_ptr()),
            h["nbEle"], ctypes.byref(h["cmp_size"]),
            ctypes.c_float(self._eb), ctypes.c_void_p(0))
        return h

    def to_host(self, out) -> np.ndarray:
        # Correctness gate only: decompress (v1's own kernel, unmodified) +
        # D2H copy. DELIBERATELY OUTSIDE the timed region -- run() above is
        # the only call the harness's CudaEventTimer ever brackets.
        import torch
        self.lib.cuszpv1_decompress_f32(
            ctypes.c_void_p(out["d_dec"].data_ptr()),
            ctypes.c_void_p(out["d_cmp"].data_ptr()),
            out["nbEle"], out["cmp_size"],
            ctypes.c_float(self._eb), ctypes.c_void_p(0))
        torch.cuda.synchronize()
        arr = out["d_dec"].detach().to("cpu", dtype=torch.float64).numpy()
        return arr.reshape(self._shape)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h) -> None:
        # Ratio/error/PSNR bookkeeping happens here (after all timed reps) --
        # same discipline as ../cuszp/adapter.py and the CPU quantize-zlib
        # codec in kernelbench/domains/compression.py.
        import torch
        cmp_bytes = int(h["cmp_size"].value)
        orig_bytes = h["nbEle"] * 4
        params = h["params"]
        params["achieved_compressed_bytes"] = cmp_bytes
        params["compression_ratio"] = orig_bytes / cmp_bytes if cmp_bytes else float("inf")

        self.lib.cuszpv1_decompress_f32(
            ctypes.c_void_p(h["d_dec"].data_ptr()),
            ctypes.c_void_p(h["d_cmp"].data_ptr()),
            h["nbEle"], h["cmp_size"],
            ctypes.c_float(self._eb), ctypes.c_void_p(0))
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
    return CuszpV1Compress(precision)
