"""
FP6-LLM / Quant-LLM adapter for the quantized-gemm track.

Paper: "Quant-LLM: Accelerating the Serving of Large Language Models via
FP6-Centric Algorithm-System Co-Design" (artifact/kernel name: FP6-LLM /
TC-FPx), USENIX ATC'24. `PAPER_KEY = conf/usenix/XiaZWCYYBWZZRHS24`.
Artifact: https://github.com/usyd-fsalab/fp6_llm -- a real ahead-of-time
CUDA extension (`fp6_llm_cuda`, built from `fp6_llm/csrc/{pybind.cpp,
fp6_linear.cu}`, setup.py already pins `-gencode=arch=compute_80,
code=sm_80`, this machine's exact GPU) implementing a fused FPx-weight x
FP16-activation Tensor-Core GEMM for x in {5,6} bits (E2M2/E3M2 floating
MINIFLOAT weights, one scale per output channel -- NOT the group-scaled
INTEGER scheme every other impl in this track shares).

REGIME MISMATCH -- WHY THIS NEEDS A NEW HOOK, NOT `w.quantize()` (per this
integration's explicit brief, and `kernel_centrality.json`'s own
`quantized-gemm|conf/usenix/XiaZWCYYBWZZRHS24` regime note: "FP6/arbitrary-
bit Tensor-Core GEMM... batch-crossover finding... directly motivated the
spec's batched-kernel variant"): `ml.py`'s `QuantGemmWorkload.quantize()`
ONLY ever produces symmetric per-GROUP INTEGER codes (see its own
docstring) -- FP6-LLM's kernel instead consumes a genuinely different
arithmetic FORMAT: an IEEE-754-style floating MINIFLOAT (sign + E exponent
bits + M mantissa bits, non-uniform step size that shrinks near zero and
widens near the max representable magnitude) with a single scale per
OUTPUT CHANNEL (this track's N, spanning the ENTIRE K -- not a 128-column
group). Neither the numeric FORMAT nor the scaling GRANULARITY match
`QuantGemmWorkload`'s scheme, so per the task brief this adapter does NOT
bend the workload (`w.bits`/`w.group_size` are IGNORED -- see below) and
instead:
  1. regenerates the SAME pre-quantization continuous weight
     `QuantGemmWorkload.quantize()` itself draws (`np.random.default_rng
     (w.seed + _W_OFFSET).standard_normal((w.K, w.N))` -- `_W_OFFSET =
     1_000_003` copied verbatim from `ml.py`, since it is not a public
     constant; same cross-module-replication convention every other
     adapter in this project already uses for private RNG helpers) --
     "derived from the workload's own fp16 weight" per the task brief,
     not an unrelated random draw;
  2. quantizes THAT weight into FP6 (E3M2, the paper's dominant/headline
     format -- `_EXPONENT=3, _MANTISSA=2` below) with a genuine,
     independently-written minifloat quantizer (`_quantize_fpx_code`,
     see its own docstring -- this is legitimate "pack from our
     quantized values" work per ARTIFACT_GUIDE rule 3, since the artifact
     itself ships no continuous-fp16-to-FPx quantizer at all: its own
     `csrc/utils/weight_quant.h::cast_fp16_fp6` is UNCOMPILED, UNBOUND
     dead code -- not in `setup.py`'s `sources` list, not in `pybind.cpp`
     -- explicitly marked "To be used in the future as a tool"; and its
     own `tests/python/kernel_test_fpx.py` benchmark script feeds the
     kernel RANDOM BITS (`torch.randint(...)`), never a real quantized
     weight, confirming no such function is exercised anywhere in this
     repository);
  3. gates against a reference built from THOSE SAME dequantized FP6
     values, via a new small, documented, OPTIONAL params hook in
     `ml.py`'s `reference_qgemm` -- `params["dequantized_W_override"]`,
     the exact "dequantized_A_override" pattern `dense.py::reference_gemv`
     already established for `gemv/packkv`'s own regime-mismatched
     quantization scheme (see that function's and `ml.py`'s own updated
     docstrings). This does NOT loosen the gate: an FP6-LLM kernel bug
     that computed the wrong output from the SAME dequantized FP6 weights
     would still fail it -- only the SOURCE of the ground-truth weight
     values changes (FP6-dequantized instead of INT-dequantized), exactly
     mirroring what a real FP6-LLM deployment actually computes.

THE MINIFLOAT QUANTIZER (`_quantize_fpx_code`) -- WHAT IT IS AND ISN'T:
a numpy port of `cast_fp16_fp6`'s OWN documented bit-manipulation algorithm
(sign bit copied, exponent REBIASED by a constant offset, mantissa
TRUNCATED -- not rounded -- from fp16's 10 bits down to M bits, subnormal
handling for values below FPx's normal range), generalized from that
function's hardcoded (E=3,M=2) to arbitrary (E,M) via the exact bias-remap
identity `weight_dequant.h::DeQuantMatrix_FPx_To_FP16`'s own `BIAS_OFFSET
= (1<<4) - (1<<(EXPONENT-1))` constant already encodes (verified: for
E=3, BIAS_OFFSET=16-4=12=15-3, i.e. fp16's bias(15) minus FPx's own
IEEE bias(2^(E-1)-1=3) -- the SAME arithmetic identity, confirming this
adapter's encoder and the ARTIFACT'S OWN COMPILED decoder agree on the
format). Verified empirically before writing this adapter (round-trip
through the artifact's own `weight_dequant_eXmY_cpu`, independent of the
GEMM kernel under test): E3M2 gives ~8-10% median relative quantization
error, consistent with a 2-bit mantissa; E2M2 saturates at |value|=7
(vs. E3M2's 28), both matching `cast_fp16_fp6`'s own documented
`absmax_fp6=28`. This quantizer is used ONLY to construct inputs (the
codes fed to the kernel AND, via the artifact's own SEPARATE
`weight_dequant_eXmY_cpu` function -- never `linear_forward_eXmY_cuda`
itself -- the reference override) -- never to independently verify the
GEMM kernel's output, preserving DOMAIN_GUIDE's reference-independence
rule (the reference and the impl-under-test share no code, only the
data both are fed).

BIT-PACKING LAYOUT (`_pack_bits`): `weight_prepacking_eXmY_cpu`'s own
`weight_matrix_prepacking_x_bit` (`csrc/utils/weight_prepacking.h`) and
`weight_dequant_eXmY_cpu`'s own `DeQuantMatrix_FPx_To_FP16` both read the
raw FPx codes via `Extract_X_Bits_To_A_Byte`, which numbers bits MSB-first
starting from BYTE 0 of the row -- i.e. a big-endian BIT stream. Naively
`.view(np.int32)`-ing a manually bit-shifted `uint32` array (treating bit 0
of the stream as the array's OWN most-significant bit) is WRONG on this
little-endian machine: numpy's native `int32` view puts a value's low byte
at the LOWEST memory address, but `Extract_X_Bits_To_A_Byte` expects the
row's FIRST (most-significant-in-the-stream) byte at the LOWEST address.
The fix (and the one real bug this adapter's own development caught,
NOT an artifact bug -- see "Real bugs found" in STATUS.md): build the
actual byte SEQUENCE first (byte 0 = the row's first 8 stream-bits,
MSB-first within each byte -- standard big-endian bit-packing), THEN
`.view(np.int32)` that byte array -- numpy's native little-endian int32
view of an already-correctly-ordered byte buffer reproduces exactly what
`reinterpret_cast<unsigned char*>` on that same buffer sees in C++.
Verified against the artifact's own `weight_dequant_eXmY_cpu` (see above).

SHAPE MAPPING: FP6-LLM's own convention is weight `[OC, IC]`, activation
`[B, IC]`, output `[B, OC]` (`fp6_linear.cu`'s own comments) -- `OC<->N`,
`IC<->K`, `B<->M` in this track's own `C[M,N]=A[M,K]@W[K,N]` convention
(the SAME per-output-channel/per-N-column mapping marlin's own gemv
adapter needed, but FP6-LLM's own kernel signature already takes activation
first, weight second, so no transpose trick is needed here either --
just an axis-name relabeling). HARD KERNEL CONSTRAINTS (`fp6_linear.cu`'s
own `assert`s inside `fpx_linear_kernel`, not merely the test script's
convenience checks): `OC (this track's N) % 256 == 0`, `IC (this track's
K) % 64 == 0`. Every real `recommended_subset` N is already a multiple of
256 and every K a multiple of 64 (in fact 128, checked against `ml.py`'s
`_QGEMM_SHAPES` -- same table `marlin/adapter.py` checked) -- zero padding
ever fires on the real registry; only 2 of the 3 `--smoke` shapes need it
(`N=384->512`; `K=96->128,N=128->256`), exact for the gate (padded K
columns are zero-coded -> decode to exactly 0.0, contribute nothing;
padded N/OC rows are sliced off before `to_host()`).

`w.bits` / `w.group_size` are IGNORED entirely (documented, not a bug):
every incoming workload -- regardless of its own nominal INT bit-width --
is re-quantized into FP6 here, per the task brief's explicit design
("feed the kernel FP6-quantized weights derived from the workload's fp16
weight" is independent of whatever `w.bits` says). This is why
`PRECISIONS` deliberately does not gate on `w.bits` the way `tilus`'s/
`marlin`'s adapters do on `w.bits==4`.
"""

from __future__ import annotations

import os

import numpy as np

KERNEL = "quantized-gemm"
IMPL_NAME = "fp6llm-fpx-gemm"
PAPER_KEY = "conf/usenix/XiaZWCYYBWZZRHS24"
PRECISIONS = ["fp16"]

HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(HERE, "vendor")

_EXPONENT = 3     # FP6 (e3m2), the paper's dominant/headline format
_MANTISSA = 2
_BIT_WIDTH = 1 + _EXPONENT + _MANTISSA
_OC_TILE = 256    # fp6_linear.cu's fpx_linear_kernel: assert(M_Global % 256 == 0)
_IC_TILE = 64     # fp6_linear.cu's fpx_linear_kernel: assert(K_Global % 64 == 0)

_FP16_EXP_BITS = 5
_FP16_MANT_BITS = 10
_FP16_EXP_BIAS = 15
_W_OFFSET = 1_000_003   # ml.py's own QuantGemmWorkload._W_OFFSET, replicated
                        # verbatim (private to that module) -- see docstring.


def available() -> tuple[bool, str]:
    if not os.path.isdir(os.path.join(_VENDOR, "fp6_llm")):
        return False, "vendor/ not built -- run build.sh"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        import sys
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        import fp6_llm  # noqa: F401  (imports torch first internally -- see build.sh)
        import fp6_llm_cuda  # noqa: F401
        if not hasattr(fp6_llm_cuda, "linear_forward_eXmY_cuda"):
            return False, "fp6_llm_cuda missing eXmY interfaces (stale build?)"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg or "__cxa_call_terminate" in msg:
            msg = ("libstdc++ ABI mismatch -- run with "
                   "LD_PRELOAD=/usr/lib64/libstdc++.so.6 (see STATUS.md)")
        elif "libc10.so" in msg:
            msg = ("libc10.so not found -- import torch before fp6_llm_cuda "
                   "(see STATUS.md's 'Real bugs found')")
        return False, msg
    return True, ""


def create(precision: str):
    return Fp6LlmQuantGemm(precision)


def _activation(w, params, dtype) -> np.ndarray:
    """Identical recipe to kernelbench.domains.ml._make_activation (private
    to that module; replicated verbatim, same convention every other
    adapter in this track already uses for this exact helper)."""
    seed = params.get("seed", w.seed)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((w.M, w.K)).astype(dtype)


def _true_weight(w) -> np.ndarray:
    """Regenerates the SAME pre-quantization continuous weight
    QuantGemmWorkload.quantize() itself draws (ml.py, verbatim RNG
    formula) -- see module docstring point 1."""
    rng = np.random.default_rng(w.seed + _W_OFFSET)
    return rng.standard_normal((w.K, w.N))


def _fpx_max(exponent_bits: int, mantissa_bits: int) -> float:
    """Largest representable FPx magnitude, treating an all-ones exponent
    field as a regular (not inf/NaN) value -- fp6_llm's own convention,
    see csrc/utils/weight_quant.h::cast_fp16_fp6's `absmax_fp6 = 28`
    literal, reproduced generally here and cross-checked against it for
    E=3,M=2 in this adapter's own pre-integration verification."""
    bias = (1 << (exponent_bits - 1)) - 1
    max_exp_field = (1 << exponent_bits) - 1
    return (1.0 + ((1 << mantissa_bits) - 1) / (1 << mantissa_bits)) * (2.0 ** (max_exp_field - bias))


def _quantize_fpx_code(x_fp16: np.ndarray, exponent_bits: int, mantissa_bits: int) -> np.ndarray:
    """Numpy port of fp6_llm's own (uncompiled, unbound, but authoritative)
    csrc/utils/weight_quant.h::cast_fp16_fp6, generalized from its
    hardcoded (E=3,M=2) to arbitrary (E,M) -- see module docstring for why
    this port is necessary (no bound quantizer exists in the artifact) and
    how its correctness was verified (round-trip through the artifact's
    own weight_dequant_eXmY_cpu). Truncates (does not round) the mantissa,
    matching cast_fp16_fp6's own documented behavior exactly.
    """
    bits = x_fp16.view(np.uint16).astype(np.int64)
    sign = (bits >> 15) & 0x1
    exp = (bits >> _FP16_MANT_BITS) & ((1 << _FP16_EXP_BITS) - 1)
    mant = bits & ((1 << _FP16_MANT_BITS) - 1)
    exp_bias_x = (1 << (exponent_bits - 1)) - 1
    shift = _FP16_MANT_BITS - mantissa_bits
    new_mant = mant >> shift
    new_exp = exp - _FP16_EXP_BIAS + exp_bias_x
    min_x_exp_val = -exp_bias_x + 1
    target_exp_val = exp - _FP16_EXP_BIAS
    is_fp16_subnormal = (exp == 0)
    is_x_subnormal = (~is_fp16_subnormal) & (target_exp_val < min_x_exp_val)
    new_exp = np.where(is_fp16_subnormal, 0, new_exp)
    new_mant = np.where(is_fp16_subnormal, 0, new_mant)
    shift_amt = np.clip(min_x_exp_val - target_exp_val, 0, 31)
    subnorm_mant = (new_mant | (1 << mantissa_bits)) >> shift_amt
    new_exp = np.where(is_x_subnormal, 0, new_exp)
    new_mant = np.where(is_x_subnormal, subnorm_mant, new_mant)
    new_mant = np.clip(new_mant, 0, (1 << mantissa_bits) - 1)
    new_exp = np.clip(new_exp, 0, (1 << exponent_bits) - 1)
    code = (sign << (exponent_bits + mantissa_bits)) | (new_exp << mantissa_bits) | new_mant
    return code.astype(np.uint8)


def _pack_bits(codes: np.ndarray, bit_width: int):
    """Row-major, MSB-first bit-packing into int32 words matching
    Extract_X_Bits_To_A_Byte's own big-endian-bitstream convention -- see
    module docstring's "BIT-PACKING LAYOUT" for why a naive bit-shifted
    view(int32) is wrong on this little-endian machine, and how this was
    caught (a real bug found during this adapter's own development, not
    an artifact bug -- documented in STATUS.md)."""
    import torch
    OC, IC = codes.shape
    assert IC % 32 == 0
    total_bits = IC * bit_width
    assert total_bits % 8 == 0
    nbytes = total_bits // 8
    bitstream = np.zeros((OC, total_bits), dtype=np.uint8)
    for b in range(bit_width):
        bitstream[:, b::bit_width] = (codes >> (bit_width - 1 - b)) & 1
    bitstream_r = bitstream.reshape(OC, nbytes, 8)
    byte_vals = np.zeros((OC, nbytes), dtype=np.uint8)
    for i in range(8):
        byte_vals |= (bitstream_r[:, :, i] << (7 - i)).astype(np.uint8)
    byte_vals = np.ascontiguousarray(byte_vals)
    words = byte_vals.view(np.int32)
    return torch.from_numpy(np.ascontiguousarray(words))


class Fp6LlmQuantGemm:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp16 activation/output "
                f"(fp6_llm's kernel is FP16-activation x FPx-weight only); "
                f"requested {precision!r}")
        self.precision = precision
        self._N_true = None

    def prepare(self, workload, params: dict):
        import sys
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        import torch
        import fp6_llm

        w = workload
        K_pad = ((w.K + _IC_TILE - 1) // _IC_TILE) * _IC_TILE
        N_pad = ((w.N + _OC_TILE - 1) // _OC_TILE) * _OC_TILE

        # --- derive FP6-quantized weight from the workload's OWN
        # pre-quantization continuous weight (module docstring point 1-2;
        # w.bits/w.group_size are deliberately ignored, see docstring).
        Wtrue = _true_weight(w)                       # (K, N) fp64
        W_ocic = Wtrue.T                               # (N=OC, K=IC)
        fpx_max = _fpx_max(_EXPONENT, _MANTISSA)
        amax = np.abs(W_ocic).max(axis=1)              # per-OC (per-N-column) scale
        scale = np.maximum(amax, 1e-12) / fpx_max
        W_scaled_fp16 = (W_ocic / scale[:, None]).astype(np.float16)
        codes = _quantize_fpx_code(W_scaled_fp16, _EXPONENT, _MANTISSA)   # (N, K) uint8

        codes_pad = np.zeros((N_pad, K_pad), dtype=np.uint8)   # code 0 == 0.0 exactly
        codes_pad[: w.N, : w.K] = codes
        scale_pad = np.ones(N_pad, dtype=np.float32)            # padded rows: all-zero codes, scale irrelevant
        scale_pad[: w.N] = scale

        packed_raw = _pack_bits(codes_pad, _BIT_WIDTH)           # CPU int32 tensor, pre-interleaving layout
        scale_t = torch.as_tensor(scale_pad.astype(np.float16))

        # --- ground truth for the fairness hook (module docstring point 3):
        # the artifact's OWN (separate from the compute kernel) host-side
        # dequantizer -- independent of linear_forward_eXmY_cuda, per
        # DOMAIN_GUIDE's reference-independence rule.
        dequant_full = fp6_llm.weight_dequant_eXmY_cpu(_EXPONENT, _MANTISSA, packed_raw, scale_t)
        W_dequant = dequant_full.numpy().astype(np.float64)[: w.N, : w.K].T   # (K, N)
        params["dequantized_W_override"] = np.ascontiguousarray(W_dequant)

        # --- the artifact's own tile-interleaved packing (this IS the
        # artifact's preprocessing -- ARTIFACT_GUIDE rule 2, timed as such).
        packed_kernel = fp6_llm.weight_prepacking_eXmY_cpu(_EXPONENT, _MANTISSA, packed_raw)

        A_np = _activation(w, params, np.float16)
        A_pad = np.zeros((w.M, K_pad), dtype=np.float16)
        A_pad[:, : w.K] = A_np
        A_cuda = torch.as_tensor(A_pad, device="cuda")
        weights_cuda = packed_kernel.to("cuda")
        scale_cuda = scale_t.to("cuda")

        self._N_true = w.N
        return {"A": A_cuda, "W": weights_cuda, "s": scale_cuda}

    def run(self, h):
        import fp6_llm
        return fp6_llm.linear_forward_eXmY_cuda(_EXPONENT, _MANTISSA, h["A"], h["W"], h["s"], 1)

    def to_host(self, out):
        import torch
        y = out.detach().to("cpu", dtype=torch.float64).numpy()
        return y[:, : self._N_true]

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
