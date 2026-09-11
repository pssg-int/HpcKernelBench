"""
Tilus adapter for the quantized-gemm track.

Paper: "Tilus: A Tile-Level GPGPU Programming Language for Low-Precision
Computation", ASPLOS'26. `PAPER_KEY = conf/asplos/DingHZL0Y0P26`.
Artifact: https://github.com/NVIDIA/tilus -- a tile-level GPU-kernel DSL (the
"finest boundary" here is the DSL's own shipped quantized-GEMM EXAMPLE, per
the task brief: "wrap its shipped/generated quantized-GEMM kernel example").

Kernel wrapped: `examples/quantization/matmul_a16wx.py`'s `QuantizedLinear`
module -- Tilus's own fp16-activation x b-bit-weight (b in 1..8, uniform
per-group symmetric quantization) fused-dequant GEMM, run UNMODIFIED
(`sys.path`-imported directly from source/examples/quantization/, no copy,
no edit). `QuantizedMatmul` (the actual compute kernel) is decorated with
`@tilus.autotune(...)` across warp_spatial/warp_repeat/num_stages/
split_k_factor, BUT the example class ships its own
`debug_schedule = dict(warp_spatial=[2,2], warp_repeat=[2,4,2],
num_stages=3, split_k_factor=1)` class attribute, and Tilus's own
`generate_schedules()` (python/tilus/lang/instantiated_script.py:132) uses
ONLY that single schedule whenever `debug_schedule` is set -- so importing
the example AS-IS already avoids a full autotune sweep (would otherwise be
3*5*2*3=90 configs); the JIT only ever builds ONE. Measured cold-build time
for the three kernels this module uses (a `cast` kernel, the layout-shuffle
kernel, and the compute kernel) is ~25-40s total on this A100 -- well inside
a login-node "run the JIT once in prepare()" budget per the task brief, not
the multi-hour tuning case that would warrant SKIPPED.

KEY FAIRNESS POINT (per this integration's task brief -- domains/ml.py's
QuantGemmWorkload docstring states the identical rule): every impl of this
track must consume the SAME quantized values as the harness's reference
(`reference_qgemm`), which dequantizes `workload.quantize()`'s codes/scale.
Tilus's own `QuantizedLinear.load_and_quantize(weight)` computes its OWN
scale/codes from a floating weight tensor -- calling it as-shipped would
requantize independently and (correctly) fail the gate for a
quantization-scheme mismatch, not a kernel bug. Since Tilus's own scheme
(symmetric per-group affine, weight is a plain uint8-packed tensor +
per-group fp16 scale) is STRUCTURALLY IDENTICAL to `QuantGemmWorkload`'s own
scheme, this adapter bypasses `load_and_quantize`'s round/clamp/scale-compute
steps and instead:
  1. takes the SAME `codes`/`scale` array `w.quantize()` returns (shared with
     every other impl and the reference, not reimplemented here);
  2. casts `codes` (already-integral values in [-qmax, qmax], a strict
     subset of Tilus's own signed-int range, e.g. int4b's [-8,7]) through
     Tilus's OWN `tilus.from_torch(...).to(w_dtype)` cast kernel -- a
     LOSSLESS pack (rounding an already-integral float to its nearest
     integer is a no-op) that reproduces exactly the bit-packed buffer
     `load_and_quantize` would have produced from these same codes;
  3. feeds that buffer through Tilus's own `change_layout_kernel` (the SAME
     call `load_and_quantize` makes) to shuffle it into the tile layout
     `QuantizedMatmul` expects;
  4. writes `scale` directly into `self.scales` (already the exact
     (n_groups, N) shape Tilus's own parameter expects).
This is "do the packing from OUR quantized values in prepare()" (this
integration's default rule), not the "gate against ITS OWN dequantized
values" exception -- Tilus's uniform per-group scheme makes the former
straightforward, unlike e.g. Quantix's non-uniform clustering codebook (see
that adapter's docstring for why IT needs the exception instead).

PRECISION: PRECISIONS = ["fp16"] -- `QuantizedMatmulCommon.__init__` asserts
`a_dtype in [float16, bfloat16]`; this adapter only wires fp16 (matching
DEFAULT_PRECISION["quantized-gemm"]).

Bit-widths: {3, 4, 6, 8} via Tilus's `int3b`/`int4b`/`int6b`/`int8` dtypes
(the domain's own smoke shapes use bits in {3, 4}; the real recommended_
subset registry uses bits=4). Tilus's floating quantized formats
(float3_e1m1 etc.) are NOT wired -- QuantGemmWorkload only ever generates
integer-coded weights (see its `quantize()`), so there is nothing to feed a
float-coded kernel path here.
"""

from __future__ import annotations

import os

import numpy as np

KERNEL = "quantized-gemm"
IMPL_NAME = "tilus-quantized-matmul"
PAPER_KEY = "conf/asplos/DingHZL0Y0P26"
PRECISIONS = ["fp16"]

HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(HERE, "vendor")
_EXAMPLES_DIR = os.path.join(HERE, "source", "examples", "quantization")
_CACHE_DIR = os.path.join(HERE, "cache")

_BITS_SUPPORTED = (3, 4, 6, 8)


def available() -> tuple[bool, str]:
    if not os.path.isdir(_VENDOR) or not os.path.isdir(
            os.path.join(_VENDOR, "tilus")):
        return False, "vendor/ not built -- run build.sh"
    if not os.path.exists(os.path.join(_EXAMPLES_DIR, "matmul_a16wx.py")):
        return False, "source/ not cloned (examples/quantization/matmul_a16wx.py missing)"
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
        import tilus  # noqa: F401
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg:
            msg = ("libstdc++ ABI mismatch -- run with "
                   "LD_PRELOAD=/usr/lib64/libstdc++.so.6 (see STATUS.md)")
        return False, msg
    return True, ""


def create(precision: str):
    return TilusQuantGemm(precision)


def _activation(w, params, dtype) -> np.ndarray:
    """Identical recipe to kernelbench.domains.ml._make_activation (private
    to that module, so replicated verbatim here rather than importing an
    underscore-prefixed symbol across modules -- turbofno's/fused3s's own
    adapters use this same convention)."""
    seed = params.get("seed", w.seed)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((w.M, w.K)).astype(dtype)


class TilusQuantGemm:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp16 activation "
                f"(QuantizedMatmulCommon asserts a_dtype in [float16, "
                f"bfloat16]); requested {precision!r}")
        self.precision = precision

    def prepare(self, workload, params: dict):
        import sys
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        if _EXAMPLES_DIR not in sys.path:
            sys.path.insert(0, _EXAMPLES_DIR)
        import tilus
        import torch
        from matmul_a16wx import QuantizedLinear  # unmodified artifact example

        tilus.option.cache_dir(_CACHE_DIR)

        w = workload
        if w.bits not in _BITS_SUPPORTED:
            raise NotImplementedError(
                f"{IMPL_NAME}: bits={w.bits} not in {_BITS_SUPPORTED} "
                "(int3b/int4b/int6b/int8 are the only dtypes this adapter "
                "wires -- see module docstring)")
        w_dtype = {3: tilus.int3b, 4: tilus.int4b,
                  6: tilus.int6b, 8: tilus.int8}[w.bits]
        group_size = w.group_size_eff

        ql = QuantizedLinear(
            x_dtype=tilus.float16, w_dtype=w_dtype, group_size=group_size,
            in_features=w.K, out_features=w.N).cuda()

        # SHARED quantized values -- the fairness point (see module
        # docstring): codes/scale come from the workload's own quantize(),
        # identical to what reference_qgemm and every other impl consume.
        codes, scale, group_idx = w.quantize()

        # Pack OUR codes through Tilus's OWN cast kernel (lossless -- codes
        # are already-integral values inside w_dtype's representable range)
        # instead of letting QuantizedLinear.load_and_quantize() recompute
        # its own scale/codes from a floating weight tensor.
        codes_t = torch.as_tensor(
            np.ascontiguousarray(codes.astype(np.float32)), device="cuda")
        packed = tilus.from_torch(codes_t).to(w_dtype).storage
        ql.change_layout_kernel(
            w.K, w.N, packed.data_ptr(), ql.quantized_weight.data_ptr())
        ql.scales[:] = torch.as_tensor(
            scale.astype(np.float16), device="cuda")

        A_np = _activation(w, params, np.float16)
        A = torch.as_tensor(A_np, device="cuda")

        return {"ql": ql, "A": A}

    def run(self, h):
        return h["ql"](h["A"])

    def to_host(self, out):
        return out.detach().to("cpu", dtype=__import__("torch").float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
