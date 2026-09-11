"""
Adapter for ZipServ (Fan et al., ASPLOS 2026, "ZipServ: Fast and
Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression",
conf/asplos/FanYPLLW0026).

Wraps ZipServ's own TCA-TBE ("Tensor-Core-Aware Triple Bitmap Encoding")
lossless weight codec at the FINEST available boundary: the standalone
decompress kernel (csrc/L_API.cu::BF16TripleBitmap_Decompress_API ->
BF16TripleBitmap_Decompress_Kernel) -- NOT the LLM-serving pipeline
(third_party/vllm/, LInfer_py/backend/, never touched by this adapter) and
NOT the fused "ZipGEMM" decompress+GEMM kernel
(BF16TripleBitmap_MM_API/_Kernel, ZipServ's headline contribution, also
never called here). Confirmed separable by reading csrc/L_API.cu directly:
the decompress kernel reconstructs the full M_Global x K_Global bf16 matrix
on its own, taking no B/C GEMM operands at all -- see bridge_zipserv.cu's
docstring for the full separability argument.

Direction-fixed per the compression domain's contract
(kernelbench/domains/compression.py), but "-decompress" this time (unlike
gpulz-compress/fzgpu-compress/cuszp-compress): ZipServ's own compress step
(kernel_benchmark/utils.h::InitBF16MatrixTripleBitmap_Host, the exact
function ZipServ's real production Python binding calls --
LInfer_py/linfer_lib.cu) is host-only CPU code with no CUDA kernel at all,
matching the spec's own framing (benchspecs/lossless-compression/
spec.yaml, lossless-comp-structured-operand-fused: "format: TCA-TBE
fixed-length triple-bitmap encoding (given; one-shot compression not timed
in this variant)"). prepare() therefore does the one-shot CPU compress
(untimed preprocessing, per ARTIFACT_GUIDE rule 2); run() times ONLY the
GPU decompress kernel, the one genuine kernel invocation this artifact
separably offers.

Byte semantics: TCA-TBE only ever inspects/repacks BF16 SIGN/EXPONENT/
MANTISSA BIT FIELDS via bitwise ops (InitBF16MatrixTripleBitmap_Host's own
`(bf16_bits >> 15) & 0x1` / `(bf16_bits >> 7) & 0xFF` / `bf16_bits & 0x7F`
-- no floating-point arithmetic on the values anywhere in compress or
decompress), so re-purposing the compression domain's fp32 Field workload's
raw bytes (viewed as a stream of bf16 BIT patterns, not semantically
interpreted as bf16 numbers) is safe and lossless regardless of content --
same convention already used by this benchmark's gpulz-compress/
mans-compress adapters for their own byte-stream inputs.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "lossless-compression"
IMPL_NAME = "zipserv-decompress"
PAPER_KEY = "conf/asplos/FanYPLLW0026"
PRECISIONS = ["fp32"]  # matches the compression domain's DEFAULT_PRECISION;
                        # ZipServ's own codec is bit-pattern-agnostic (see
                        # docstring) -- reinterpreted as a bf16 bit stream

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
    lib.zipserv_prepare.argtypes = [u8p, ctypes.c_size_t]
    lib.zipserv_prepare.restype = p
    lib.zipserv_run.argtypes = [p]
    lib.zipserv_run.restype = ctypes.c_int
    lib.zipserv_copy_output.argtypes = [p, u8p, ctypes.c_size_t]
    lib.zipserv_copy_output.restype = ctypes.c_int
    lib.zipserv_compressed_bytes.argtypes = [p]
    lib.zipserv_compressed_bytes.restype = ctypes.c_size_t
    lib.zipserv_free.argtypes = [p]
    lib.zipserv_free.restype = None
    return lib


class ZipservDecompress:
    name = "zipserv-decompress"
    platform = "cuda"
    direction = "decompress"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"zipserv-decompress wraps a bf16-bit-pattern codec via the "
                f"fp32 smoke workload's raw bytes; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        # One-shot CPU compress (bridge_zipserv.cu::zipserv_prepare ->
        # ZipServ's own InitBF16MatrixTripleBitmap_Host, unmodified) +
        # H2D copy of every compressed piece the decompress kernel needs +
        # allocation of the device output buffer -- all untimed
        # preprocessing, per ARTIFACT_GUIDE rule 2 and this domain's own
        # "-decompress" direction contract (kernelbench/domains/
        # compression.py's _LosslessCodec.prepare()).
        data = np.ascontiguousarray(w.data)
        raw = data.tobytes()
        self._dtype = w.data.dtype
        self._shape = w.data.shape
        self._nbytes = len(raw)
        params["direction"] = self.direction
        buf = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
        handle = self.lib.zipserv_prepare(buf, ctypes.c_size_t(len(raw)))
        if not handle:
            raise RuntimeError("zipserv_prepare returned a null handle")
        return {"handle": handle, "nbytes": len(raw), "params": params}

    def run(self, h):
        # exactly one BF16TripleBitmap_Decompress_API call -- ZipServ's own
        # standalone decompress kernel, unmodified, no GEMM fusion. This is
        # the only call the harness's CudaEventTimer ever brackets.
        rc = self.lib.zipserv_run(h["handle"])
        if rc != 0:
            raise RuntimeError(f"zipserv_run (decompress kernel) failed (rc={rc})")
        return h

    def to_host(self, out) -> np.ndarray:
        # run() above IS the decompress call already -- per this domain's
        # "-decompress" direction contract, to_host() just converts its
        # output, no extra decode needed (unlike this benchmark's
        # "-compress"-direction GPU adapters, which need a separate
        # untimed decode here for the gate).
        nbytes = out["nbytes"]
        buf = (ctypes.c_uint8 * nbytes)()
        got = self.lib.zipserv_copy_output(out["handle"], buf, ctypes.c_size_t(nbytes))
        if got != nbytes:
            raise RuntimeError(f"zipserv decode size mismatch: got {got}, expected {nbytes}")
        arr = np.frombuffer(bytes(buf), dtype=self._dtype).reshape(self._shape)
        return arr.astype(np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h) -> None:
        # Ratio bookkeeping: the compressed size is already fully known
        # from prepare() (ZipServ's compress step is deterministic and was
        # already run once, untimed) -- no extra decode needed here, unlike
        # this benchmark's "-compress"-direction adapters that must re-run
        # an untimed decode in free() to capture the LAST timed run's
        # output (decompress here is idempotent: same compressed input,
        # same reconstructed output, every call).
        cmp_bytes = int(self.lib.zipserv_compressed_bytes(h["handle"]))
        orig_bytes = h["nbytes"]
        params = h["params"]
        params["achieved_compressed_bytes"] = cmp_bytes
        params["compression_ratio"] = orig_bytes / cmp_bytes if cmp_bytes else float("inf")
        self.lib.zipserv_free(h["handle"])
        h.clear()


def create(precision: str):
    return ZipservDecompress(precision)
