"""
Adapter for lsCOMP (SC 2025, "lsCOMP: Efficient Light Source Compression",
conf/sc/HuangDUMCLSC25) -- the LOSSLESS-compression registration.

This is a DIFFERENT integration than ../../lossy-compression/lscomp/, which
is SKIPPED there (uint32/uint16-only API, quantBins+pooling lossy knobs have
no closed-form error-bound equivalent to that track's abs/rel fp32 contract
-- see that directory's STATUS.md). The lossless-compression track has no
such mismatch: lsCOMP's own README states its lossy pipeline (Adaptive
Scalar Quantization + Selective Pooling) can be disabled entirely --
"To disable Adaptive Scalar Quantization, set `-b 1 1 1 1`; to disable
Selective Pooling, set `-p 1`" -- "which makes lsCOMP operate in a lossless
mode", and its own shipped example demonstrates exactly this configuration
producing "Max abs diff: 0". kernelbench/domains/compression.py's
lossless-compression CORRECTNESS_MODE is "exact" (bit-for-bit round trip),
which this configuration is designed to satisfy.

Byte-layout note (disclosed, per ARTIFACT_GUIDE rule 8's "convert in
prepare() and disclose" instruction): lsCOMP's API takes uint32_t*/uint16_t*
arrays of DETECTOR COUNTS, not arbitrary byte streams. The lossless-
compression domain's `Field` workload is a synthetic fp32 scientific array,
not integer detector data. This adapter REINTERPRETS the fp32 array's raw
bit pattern as **uint16** (a `float32.view(uint16)`, which doubles the
element count -- 2 uint16 halves per fp32 word -- not a value-rounding
cast); every bit of the original array is preserved, so a bit-exact round
trip of the reinterpreted buffer is equivalent to a bit-exact round trip of
the original fp32 data.

## Why uint16, not uint32 (a real bug found in lsCOMP's uint32 kernel)

The first version of this adapter used the uint32 entry points
(`lsCOMP_compression/decompression_uint32_bsize64`) with a straight
`float32.view(uint32)` (1:1 element mapping). That FAILED the round-trip
gate for any field containing a negative value (2 of this domain's 3 smoke
workloads: `smoke-turbulent-3d`, `smoke-multiscale-3d`; only the
strictly-nonnegative `smoke-smooth-3d` passed) -- reproduced, isolated, and
root-caused directly in `source/src/lsCOMP_kernel.cu` (evidence kept in
`debug_roundtrip.py`, this directory):

- The compressor computes a per-64-element-block bit-width
  `temp_rate = 32 - __clz(max_quantized_val)` (lines 120/129), whose full
  legitimate range is **0..32 inclusive** (`__clz` of a 32-bit value with
  the top bit set returns 0, giving `temp_rate = 32`) -- i.e. `temp_rate`
  needs 6 bits to represent every value it can actually take.
- That value is packed into the per-block metadata byte as
  `fixed_rate[j] = (bin_choice << 5) | temp_rate` (lines 122/131) and
  re-extracted later via `fixed_rate[j] & 0x1f` (line 261, both the
  compressor's own second pass AND the decompressor, `lsCOMP_kernel.cu`
  ~line 622) -- a **5-bit** field, representable range 0..31 only.
- `temp_rate == 32` is `0b100000` in binary: `& 0x1f` (`0b011111`) zeroes
  it out completely, aliasing to the SAME bit pattern as a legitimate
  `temp_rate == 0` ("this block is uniformly one value, nothing to store").
  Any 64-element block containing a value whose top bit (bit 31) is set --
  i.e. any block containing at least one value >= 2^31, which for a
  reinterpreted IEEE-754 bit pattern means simply "any negative float" --
  triggers this collision. Once one block anywhere aliases like this, the
  compressor's own second pass writes 0 payload bytes for that block (its
  local copy of `fixed_rate[j]` is masked identically), but the WARP-level
  exclusive prefix sum that positions every LATER warp's output region
  (`thread_ofs`/`locOffset`/`cmpOffset`, computed in the compressor's FIRST
  pass, lines 121/130, from the UNMASKED `temp_rate`) still reserved the
  full (up to 256-byte) span for it -- so every later warp's actual write
  position (computed from the deflated, masked count) ends up misaligned
  from where later reads expect it, corrupting the entire remainder of the
  buffer, not just the one offending block. Confirmed empirically
  (`debug_roundtrip.py`): bounding synthetic uint32 data to `< 2**31 - 1`
  round-trips perfectly; introducing even one value `>= 2**31` corrupts the
  whole array; a missing-preinitialization red herring (a genuinely
  all-zero block also decodes to garbage on an uninitialized output buffer,
  since decode leaves value-0 blocks untouched and expects the caller to
  have pre-zeroed `d_decData` -- same convention this repo's cuSZp lineage
  uses, see ../../lossy-compression/cuszp-v1/STATUS.md) was isolated and
  ruled out separately by testing with a zeroed buffer.

This is a genuine, reproducible defect in the released kernel's uint32
lossless path for ANY input distribution that can produce a value
`>= 2**31` in some 64-element block -- never triggered by lsCOMP's own
target workload (light-source photon counts, always small and
non-negative) but immediately triggered by this domain's synthetic fp32
scientific fields once reinterpreted as uint32.

**Workaround (not a patch to source/, a different, equally-documented use
of the artifact's own public API)**: the uint16 entry points
(`lsCOMP_compression/decompression_uint16_bsize64`) share the exact same
kernel structure and the exact same 5-bit `temp_rate` field, but a uint16
input's per-block max value can never need more than 16 bits, so
`temp_rate` never approaches the 32-vs-31 collision boundary -- confirmed
empirically immune across full-range random uint16 data, constant
`0xFFFF`, and this domain's actual sign-mixed fp32 fields reinterpreted as
uint16 pairs (`debug_uint16.py`). This adapter therefore reinterprets each
fp32 word as **two uint16 halves** (`float32.view(uint16)`, native byte
order, 2x the element count) and calls the uint16 API -- still lsCOMP's own
unmodified kernel, still the lossless (`quantBins=1,1,1,1`, `poolingTH=1`)
configuration, just the dtype entry point that this artifact's own bit-width
budget can actually support for arbitrary data. Per ARTIFACT_GUIDE rule 3,
no kernel code was touched (`git -C source diff` is empty); this is an
adapter-side input-conversion choice, disclosed here and in `params`.

`uint3 dims`: lsCOMP treats a 3D array as `dims.x` 2D slices of shape
`(dims.y, dims.z)`, `dims.z` the fastest-varying axis (README: "dim.z is
the fastest dimension"). The Field workload's data is row-major (C-order),
whose last axis is already fastest-varying; after the uint16
reinterpretation the last axis has 2x as many (uint16) elements as the
original fp32 axis had, so `dims = (shape[0], shape[1], 2*shape[2])`.

Wraps lsCOMP's own generic device-pointer C API directly via ctypes
(liblscomp.so, built by build.sh from lsCOMP's own unmodified `.cu`
sources -- zero patches to `source/`; include/lsCOMP_entry.h already wraps
its declarations in `extern "C"`, so no bridge/shim file is needed here,
unlike ../../lossy-compression/cuszp-v1/). Same device-buffer-via-torch
pattern as this benchmark's other native-CUDA ctypes kernels.

Direction-fixed per the compression domain's contract: this is the
"-compress" registration. Decompression is used ONLY for the correctness
gate (to_host()/free()), outside the timed region.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "lossless-compression"
IMPL_NAME = "lscomp-lossless-compress"
PAPER_KEY = "conf/sc/HuangDUMCLSC25"
PRECISIONS = ["fp32"]  # matches the lossless-compression domain's
                        # DEFAULT_PRECISION; the fp32 bit pattern is
                        # reinterpreted as uint16 pairs before compression
                        # (see module docstring), not value-cast

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "liblscomp.so")


class _Uint3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]


class _Uint4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint),
                ("z", ctypes.c_uint), ("w", ctypes.c_uint)]


# quantBins=(1,1,1,1) + poolingTH=1.0 -- lsCOMP's own documented lossless
# configuration (README: "To disable Adaptive Scalar Quantization, set
# `-b 1 1 1 1`; to disable Selective Pooling, set `-p 1`... which makes
# lsCOMP operate in a lossless mode"). Not something this adapter invented.
_LOSSLESS_QUANT_BINS = _Uint4(1, 1, 1, 1)
_LOSSLESS_POOLING_TH = 1.0


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
    lib.lsCOMP_compression_uint16_bsize64.argtypes = [
        p, p, ctypes.POINTER(ctypes.c_size_t), _Uint3, _Uint4, ctypes.c_float, p]
    lib.lsCOMP_compression_uint16_bsize64.restype = None
    lib.lsCOMP_decompression_uint16_bsize64.argtypes = [
        p, p, ctypes.c_size_t, _Uint3, _Uint4, ctypes.c_float, p]
    lib.lsCOMP_decompression_uint16_bsize64.restype = None
    return lib


class LscompLosslessCompress:
    name = "lscomp-lossless-compress"
    platform = "cuda"
    direction = "compress"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"lscomp-lossless-compress wraps lsCOMP's uint16 kernel via an "
                f"fp32-bitpattern reinterpretation; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        import torch
        data = np.ascontiguousarray(w.data, dtype=np.float32)
        if data.ndim != 3:
            raise NotImplementedError(
                f"{w.name}: lscomp-lossless-compress needs a 3D field "
                f"(dims.x slices of dims.y x dims.z); got shape {data.shape}")
        self._dtype = w.data.dtype
        self._shape = data.shape

        # Bit-level reinterpretation (view, not cast): every bit of the
        # original fp32 array is preserved. uint16, NOT uint32 -- see
        # module docstring's "Why uint16" section for the real lsCOMP
        # uint32-kernel bug this sidesteps.
        u16 = data.view(np.uint16)  # 2x elements vs. data.size
        nbEle = int(u16.size)
        dims = _Uint3(self._shape[0], self._shape[1], 2 * self._shape[2])

        params["direction"] = self.direction
        params["lscomp_mode"] = "lossless (quantBins=1,1,1,1 poolingTH=1)"
        params["lscomp_entry_dtype"] = "uint16 (NOT uint32 -- see adapter docstring bug note)"
        params["lscomp_byte_reinterpretation"] = (
            "fp32 bit pattern viewed as 2x uint16, NOT value-cast -- see adapter docstring")

        d_ori = torch.from_numpy(u16).to("cuda")
        # worst-case compressed-size bound: lsCOMP's own fixed-rate
        # encoding never expands past ~1 flag byte + 2 bytes per uint16
        # element in this configuration -- sizeof(uint16)*nbEle plus
        # headroom is a safe upper bound, same worst-case-allocation
        # convention as this benchmark's cuSZp adapters.
        d_cmp = torch.empty(nbEle * 2 + 4096, dtype=torch.uint8, device="cuda")
        # MUST be zero-initialized: lsCOMP's decompressor leaves a
        # zero-valued block's output untouched (it never explicitly writes
        # 0), relying on the caller to have pre-zeroed the destination --
        # same convention as this repo's cuSZp-lineage examples (see
        # ../../lossy-compression/cuszp-v1/STATUS.md's decompress note).
        d_dec = torch.zeros(nbEle, dtype=torch.uint16, device="cuda")
        return {
            "d_ori": d_ori, "d_cmp": d_cmp, "d_dec": d_dec,
            "nbEle": nbEle, "dims": dims, "cmp_size": ctypes.c_size_t(0),
            "params": params,
        }

    def run(self, h):
        self.lib.lsCOMP_compression_uint16_bsize64(
            ctypes.c_void_p(h["d_ori"].data_ptr()),
            ctypes.c_void_p(h["d_cmp"].data_ptr()),
            ctypes.byref(h["cmp_size"]), h["dims"],
            _LOSSLESS_QUANT_BINS, ctypes.c_float(_LOSSLESS_POOLING_TH),
            ctypes.c_void_p(0))
        return h

    def _decompress_to(self, h, dst_tensor):
        import torch
        dst_tensor.zero_()  # see prepare()'s zero-init note
        self.lib.lsCOMP_decompression_uint16_bsize64(
            ctypes.c_void_p(dst_tensor.data_ptr()),
            ctypes.c_void_p(h["d_cmp"].data_ptr()),
            h["cmp_size"], h["dims"],
            _LOSSLESS_QUANT_BINS, ctypes.c_float(_LOSSLESS_POOLING_TH),
            ctypes.c_void_p(0))
        torch.cuda.synchronize()

    def to_host(self, out) -> np.ndarray:
        # Correctness gate: decompress (lsCOMP's own kernel, unmodified),
        # view the recovered uint16 buffer back as fp32 bits, D2H copy.
        # DELIBERATELY OUTSIDE the timed region -- run() above is the only
        # call the harness's CudaEventTimer ever brackets.
        self._decompress_to(out, out["d_dec"])
        u16 = out["d_dec"].detach().to("cpu").numpy().view(np.float32)
        return u16.reshape(self._shape).astype(np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h) -> None:
        import torch
        cmp_bytes = int(h["cmp_size"].value)
        orig_bytes = h["nbEle"] * 2
        params = h["params"]
        params["achieved_compressed_bytes"] = cmp_bytes
        params["compression_ratio"] = orig_bytes / cmp_bytes if cmp_bytes else float("inf")

        self._decompress_to(h, h["d_dec"])
        # .reshape(-1): d_ori keeps numpy .view()'s natural (shape[0],
        # shape[1], 2*shape[2]) shape while d_dec is allocated flat --
        # same bytes/order, different ndarray shape, so a bare
        # np.array_equal (which requires equal SHAPES, not just equal
        # values) would spuriously report a mismatch on shape alone.
        recon_u16 = h["d_dec"].detach().to("cpu").numpy().reshape(-1)
        orig_u16 = h["d_ori"].detach().to("cpu").numpy().reshape(-1)
        exact = bool(np.array_equal(recon_u16, orig_u16))
        params["achieved_bit_exact"] = exact
        params["achieved_max_abs_error"] = 0.0 if exact else float("nan")

        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return LscompLosslessCompress(precision)
