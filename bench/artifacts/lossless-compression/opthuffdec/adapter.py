"""
Adapter for OptHuffDec (IPDPS 2022, "Optimizing Huffman Decoding for
Error-Bounded Lossy Compression on GPUs", conf/ipps/RiveraDTYTC22) --
the lossless-compression track, `opt-gap-array` variant (ICPP'20 gap-array
decoder plus this paper's optimizations; the repo's own README names this
as one of its two headline optimized decoders alongside `opt-self-sync`).

The artifact's own scope is narrower than a general byte-stream compressor:
it is a Huffman DEcoder for cuSZ-style QUANTIZATION-CODE streams (SYMBOL_TYPE
= uint16_t per source/opt-gap-array/include/cuhd_constants.h -- exactly the
dtype cuSZ's own quantization codes use). This benchmark's
lossless-comp-gpu-multibyte-dual-scope variant's own recommended_subset
lists cuSZ quant-code workloads by name ("HACC/vx quant-code ... uint16 --
used verbatim by both GPULZ and the Huffman-decoder paper") -- this is
exactly this artifact's own target input shape, not a repurposing.

Per ARTIFACT_GUIDE's task description for this artifact ("Wrap encode
(their reference encoder or cuSZ's) in prepare() and the optimized decode
in run(), gating the round-trip"): the artifact SHIPS a usable reference
encoder (`llhuff::LLHuffmanEncoder`, LGPL-3.0, its own
encoder/{include,src}/llhuffman_encoder.{h,cc} -- a length-limited Huffman
encoder, "LL" = length-limited, MAX_CODEWORD_LENGTH=14) that produces
exactly the codeword-stream + gap-array format `opt-gap-array`'s GPU decoder
consumes -- so NO independent encoder needed to be written for this
integration.

Byte-layout note (disclosed, per ARTIFACT_GUIDE rule 8): the
lossless-compression domain's `Field` workload is a synthetic fp32
scientific array, not a pre-existing quant-code stream. This adapter
reinterprets the fp32 bit pattern as uint16 (`float32.view(uint16)`, 2
halves per fp32 word, a bit-preserving VIEW, never a value-rounding cast)
before Huffman-encoding it -- this is a DIFFERENT (and more information-
dense/harder-to-compress) symbol distribution than cuSZ's own quant codes
(which cluster tightly around a bias value), so the achieved COMPRESSION
RATIO here is not comparable to the paper's own reported ratios; the
round-trip correctness gate and the measured DECODE kernel throughput
(what this paper's optimizations target) are both still legitimate --
Huffman coding is lossless by construction for any symbol actually present
in its own table, regardless of the source distribution's compressibility.

Wraps the artifact via bridge_opthuffdec.cc (this directory), a thin C API
driver replacing demo.cc (the paper's own benchmark script) -- see that
file's docstring for exactly which upstream calls it makes and why the
split matches ARTIFACT_GUIDE rule 1/2 (kernel-not-script; encode is
preprocessing). libopthuffdec.so is built by build.sh from OptHuffDec's own
unmodified `.cc`/`.cu` sources plus that one bridge file -- zero patches to
`source/` (`git -C source diff` is empty; the only fixes were build-flag
level, see build.sh's `--pre-include cstdint` comment).

Direction-fixed per the compression domain's contract: this is the
"-decompress" registration (the artifact's headline contribution IS the
decoder; encode happens in prepare(), untimed, exactly as instructed).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "lossless-compression"
IMPL_NAME = "opthuffdec-gap-decompress"
PAPER_KEY = "conf/ipps/RiveraDTYTC22"
PRECISIONS = ["fp32"]  # matches the lossless-compression domain's
                        # DEFAULT_PRECISION; the fp32 bit pattern is
                        # reinterpreted as uint16 symbols before encoding
                        # (see module docstring), not value-cast

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "libopthuffdec.so")


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
    lib.opthuffdec_prepare.argtypes = [ctypes.POINTER(ctypes.c_uint16),
                                       ctypes.c_size_t, ctypes.c_int]
    lib.opthuffdec_prepare.restype = p
    lib.opthuffdec_run.argtypes = [p]
    lib.opthuffdec_run.restype = None
    lib.opthuffdec_fetch_output.argtypes = [p, ctypes.POINTER(ctypes.c_uint16)]
    lib.opthuffdec_fetch_output.restype = None
    lib.opthuffdec_compressed_bytes.argtypes = [p]
    lib.opthuffdec_compressed_bytes.restype = ctypes.c_size_t
    lib.opthuffdec_free.argtypes = [p]
    lib.opthuffdec_free.restype = None
    return lib


class OptHuffDecGapDecompress:
    name = "opthuffdec-gap-decompress"
    platform = "cuda"
    direction = "decompress"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"opthuffdec-gap-decompress wraps a uint16-symbol Huffman "
                f"decoder via an fp32-bitpattern reinterpretation; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()
        self._handle = None

    def prepare(self, w, params: dict):
        # PREPROCESSING (untimed): bit-reinterpret the field as uint16
        # symbols, then run the artifact's OWN reference encoder
        # (llhuff::LLHuffmanEncoder, called from bridge_opthuffdec.cc's
        # opthuffdec_prepare) plus GPU buffer allocation + H2D transfer --
        # exactly what demo.cc itself times as separate ("encoding"/"GPU
        # buffer allocation"/"GPU memcpy HtD") phases BEFORE its own
        # "decoding" phase.
        data = np.ascontiguousarray(w.data, dtype=np.float32)
        self._dtype = w.data.dtype
        self._shape = data.shape

        u16 = data.view(np.uint16).reshape(-1)  # bit-preserving view, 2x elements
        n_symbols = int(u16.size)
        self._n_symbols = n_symbols

        params["direction"] = self.direction
        params["opthuffdec_variant"] = "opt-gap-array"
        params["opthuffdec_byte_reinterpretation"] = (
            "fp32 bit pattern viewed as 2x uint16 symbols, NOT value-cast -- see adapter docstring")

        u16_c = np.ascontiguousarray(u16)
        buf = u16_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        device_id = 0
        handle = self.lib.opthuffdec_prepare(buf, ctypes.c_size_t(n_symbols),
                                             ctypes.c_int(device_id))
        if not handle:
            raise RuntimeError("opthuffdec_prepare returned NULL")

        return {"handle": handle, "params": params}

    def run(self, h):
        # exactly one call to the artifact's own optimized GPU decode
        # kernel (cuhd::CUHDGPUDecoder::decode, opt-gap-array's
        # cuhd_gpu_decoder.cu, unmodified) -- the only call the harness's
        # CudaEventTimer brackets.
        self.lib.opthuffdec_run(h["handle"])
        return h

    def to_host(self, out) -> np.ndarray:
        # Correctness gate: D2H copy of the decoded uint16 symbol stream
        # (DELIBERATELY outside the timed region, same discipline as
        # demo.cc's own separately-timed "GPU memcpy DtH" phase), then view
        # back as fp32 bits.
        buf = np.empty(self._n_symbols, dtype=np.uint16)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        self.lib.opthuffdec_fetch_output(out["handle"], ptr)
        return buf.view(np.float32).reshape(self._shape).astype(np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h) -> None:
        cmp_bytes = int(self.lib.opthuffdec_compressed_bytes(h["handle"]))
        orig_bytes = self._n_symbols * 2
        params = h["params"]
        params["achieved_compressed_bytes"] = cmp_bytes
        params["compression_ratio"] = orig_bytes / cmp_bytes if cmp_bytes else float("inf")
        # Bit-exactness itself is recorded in the result's own `correctness`
        # field (the harness's to_host()-vs-reference gate, already the
        # authoritative check) -- not re-derived a second time here.

        self.lib.opthuffdec_free(h["handle"])
        h.clear()


def create(precision: str):
    return OptHuffDecGapDecompress(precision)
