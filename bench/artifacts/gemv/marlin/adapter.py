"""
Marlin adapter for the gemv track.

Paper: "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large
Language Models", PPoPP'25. `PAPER_KEY = conf/ppopp/FrantarCCHA25`.
Artifact: https://github.com/IST-DASLab/marlin -- a highly optimized
FP16-activation x INT4-weight (symmetric, per-group) CUDA GEMM kernel
purpose-built for LLM decode (batch=1..~32, the "thin-GEMM" regime).
Wrapped here UNMODIFIED via its own high-level `marlin.Layer` API
(`Layer.pack()` / `Layer.forward()`, source/marlin/__init__.py) -- no copy,
no kernel edit.

SHAPE MAPPING (marlin's "m,k,n" -> this domain's gemv "M,K"): this domain's
gemv workload (`dense.py`'s `DenseShape`) models y[M] = A[M,K] @ x[K] with
an implicit batch of 1 (N=1 in every _gemv_shapes() entry -- ONE output
vector). Marlin's own convention is C[m,n] = A_act[m,k] @ B_weight[k,n]
with `m` = batch (decode: m=1). Setting m=1 (a single activation row = our
x), k = this domain's K (contraction), n = this domain's M (output rows,
i.e. marlin's "weight" B is the TRANSPOSE of this domain's A) reproduces
y = A@x exactly: C[0,n] = sum_k x[k]*B[k,n] = sum_k x[k]*A[n,k] = y[n].
This IS "wrap at M=1 (decode)" per the task brief -- gemv's own N=1 batch
dimension already equals marlin's m=1, not something this adapter chooses.

QUANTIZATION / VALUE-SHARING (the task brief's "follow the quantized-gemm
fairness pattern"): marlin's kernel computes INT4-weight x FP16-activation
GEMM; it cannot consume a continuous fp32/fp64 weight at all. Per
ARTIFACT_GUIDE.md's fairness rule (identical to quantized-gemm's own,
domains/ml.py's QuantGemmWorkload docstring), the correctness gate must
compare marlin's output against a reference built from the SAME quantized
values marlin itself consumes, not against the pre-quantization continuous
matrix (comparing against the continuous matrix would conflate ordinary
quantization noise with kernel-implementation error -- verified empirically
before writing this adapter: symmetric per-group INT4-quantizing this
domain's own gemv operands and comparing against the CONTINUOUS-A reference
produces 0.15%-6.6% max_scaled_err from quantization noise ALONE,
shape-dependent, which would swamp gemv-quantized-weight-kernel's 1e-3
parsed tolerance regardless of whether the kernel itself is correct).

`dense.py`'s gemv workload has no built-in notion of quantization (unlike
ml.py's QuantGemmWorkload), and `runner.py` calls `domain.REFERENCES[kernel]`
uniformly for every impl -- there is no per-impl reference override (same
constraint tilus's/qfactory's own STATUS.md documents for quantized-gemm).
This is therefore a GENUINELY REQUIRED small fix (ARTIFACT_GUIDE.md's own
allowance), made directly in `kernelbench/domains/dense.py`:
`reference_gemv` now accepts two OPTIONAL params keys, "quant_bits" and
"quant_group_size" -- when absent (every existing caller), behavior is
bit-for-bit unchanged; this adapter's `prepare()` sets them (params is the
SAME dict object `harness.run_variant` later passes to `reference_gemv`,
called AFTER `prepare()` -- see dense.py's docstring for the exact
ordering) so the gate compares marlin's output against A AFTER the SAME
symmetric per-(row,group) quantize-dequantize step
(`dense.quantize_dequantize_groupwise`, a new small PUBLIC helper both
this adapter and reference_gemv call) marlin itself is fed.

Marlin's OWN packing API (`Layer.pack(linear, scales)`) does not take raw
integer codes directly -- it takes a DEQUANTIZED fp16 `nn.Linear` weight
(its own term: "a fake-quantized linear layer") plus a matching per-group
fp16 scale tensor, and internally re-derives `round(w / s)` to get its
packed integer representation. Since this adapter constructs that
dequantized weight AS `codes * scale` from the SHARED codes/scale (not
independently re-quantized), marlin's own `round(w/s)` recovers exactly
those same integer codes (lossless, `w/s` is already integral by
construction) -- this IS "feed it shared quantized codes/scales where its
packing API allows" (this is literally the SAME pattern marlin's own
`test.py::gen_quant4` uses to build ITS OWN ground truth for `test.py`'s
correctness check).

PRECISION CONVENTION (documented per the task brief): weights are INT4
(marlin is hard-coded 4-bit only, `maxq = 2**4 - 1` in `Layer.pack`),
activation and output are FP16, exactly matching MARLIN's own paper/README
convention ("W4A16"). `PRECISIONS = ["fp16"]` -- `dense.py`'s
`DEFAULT_PRECISION["gemv"]` is unconditionally "fp64" (see that module's
own docstring point 3: the spec's parsed tolerance is fp64-scoped), so
running this impl REQUIRES an explicit `--precision fp16` on the CLI (the
generic `--smoke` template in ARTIFACT_GUIDE.md does not by itself select
fp16) -- documented again in STATUS.md's exact verified command.

SHAPE-COMPATIBILITY PADDING: marlin's `Layer.__init__` hard-asserts
`infeatures % 128 == 0` and `outfeatures % 256 == 0` (its own tile-size
floor). This domain's gemv shapes were derived from TLR-MVM's bandwidth
sweep and GPU-DPF's/PackKV's table shapes (see benchspecs/gemv/spec.yaml),
NOT an LLM-decode shape list, so the tall-skinny (K=16 or 32) and
short-fat (M=32 or 128) shapes each violate ONE of marlin's two tile
constraints (never both at once for this domain's actual shapes -- see the
assert in `prepare()` below). This adapter zero-pads up to the nearest
valid tile boundary: padding K with zero columns of the weight (and
correspondingly the reference's SAME-seeded x is padded too, though this
is immaterial -- zero weight columns contribute 0 regardless of x) and
padding M with zero output rows, sliced off before `to_host()`. This is
EXACT for the correctness gate (padding contributes provably zero) but
readers of a FUTURE timed run at tall-skinny/short-fat shapes should note
the kernel is doing padded-tile work the workload's own declared M*K*bytes
byte-accounting does not reflect -- flagged here and in STATUS.md, not
silently absorbed into a bandwidth number.
"""

from __future__ import annotations

import os

import numpy as np

KERNEL = "gemv"
IMPL_NAME = "marlin-w4a16-decode"
PAPER_KEY = "conf/ppopp/FrantarCCHA25"
PRECISIONS = ["fp16"]

HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(HERE, "vendor")

_BITS = 4                 # marlin is INT4-only (Layer.pack hardcodes maxq=2**4-1)
_IN_TILE = 128             # Layer.__init__: infeatures % 128 == 0
_OUT_TILE = 256            # Layer.__init__: outfeatures % 256 == 0
_GROUPSIZE = 128           # Layer.__init__: groupsize must be -1 or literally 128
_B_OFFSET = 1_000_003      # dense.py's own arbitrary seed offset for x, replicated verbatim


def available() -> tuple[bool, str]:
    if not os.path.isdir(os.path.join(_VENDOR, "marlin")):
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
        import marlin  # noqa: F401
        import marlin_cuda  # noqa: F401
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg:
            msg = ("libstdc++ ABI mismatch -- run with "
                   "LD_PRELOAD=/usr/lib64/libstdc++.so.6 (see STATUS.md)")
        return False, msg
    return True, ""


def create(precision: str):
    return MarlinGemv(precision)


def _rng_operand(rows: int, cols: int, seed: int, dtype) -> np.ndarray:
    """Verbatim copy of kernelbench.domains.dense._rng_operand (private to
    that module; replicated here rather than imported -- same convention
    every other artifact adapter in this project uses for these RNG
    helpers, e.g. quantized-gemm/tilus's adapter.py docstring)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(dtype)


def _rng_vector(n: int, seed: int, dtype) -> np.ndarray:
    """Verbatim copy of kernelbench.domains.dense._rng_vector."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=n).astype(dtype)


class MarlinGemv:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp16 activation/output "
                f"(marlin's kernel is FP16-activation x INT4-weight only); "
                f"requested {precision!r}. dense.py's "
                f"DEFAULT_PRECISION['gemv'] is 'fp64' (see that module's "
                f"docstring point 3), so pass --precision fp16 explicitly "
                f"on the CLI -- see this adapter's module docstring / "
                f"STATUS.md for the exact verified command.")
        self.precision = precision
        self._M_true = None

    def prepare(self, workload, params: dict):
        import sys
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        import torch
        import marlin as mlib
        from kernelbench.domains.dense import quantize_dequantize_groupwise

        w = workload
        M, K = w.M, w.K   # gemv's (output rows, contraction dim); N is
                           # always 1 for every gemv-track shape (single
                           # output vector) -- see dense.py's _gemv_shapes()
        assert K < _IN_TILE or K % _IN_TILE == 0, (
            f"{IMPL_NAME}: K={K} is neither < {_IN_TILE} nor a multiple of "
            f"it -- the group-size-sharing assumption below (see module "
            f"docstring) needs one or the other; true for every gemv-track "
            f"shape as of this integration, revisit if the shape registry "
            f"grows a K in this gap")

        # --- shared-quantized-value fairness (see module docstring): tell
        # reference_gemv (called AFTER prepare() with this SAME params
        # dict -- harness.run_variant) to gate against the dequantized
        # values, using the SAME group size this adapter uses below.
        group_size = _IN_TILE if K % _IN_TILE == 0 else K
        params["quant_bits"] = _BITS
        params["quant_group_size"] = group_size

        # --- regenerate A, x EXACTLY as reference_gemv will: generate at
        # the RUN's own precision (fp16) first, THEN widen to fp64 -- see
        # dense.py's module docstring for why this order matters (skipping
        # the fp16-rounding step would desynchronize this adapter's codes
        # from reference_gemv's, since round-to-fp16 can flip which side
        # of a quantization boundary a value falls on).
        seed = params.get("seed", w.seed)
        A16 = _rng_operand(M, K, seed, np.float16)
        A64 = A16.astype(np.float64)
        x16 = _rng_vector(K, seed + _B_OFFSET, np.float16)

        codes, scale, dequant = quantize_dequantize_groupwise(A64, _BITS, group_size)

        # --- marlin's own hard tile constraints; zero-pad up to the
        # nearest valid boundary (see module docstring's "SHAPE-
        # COMPATIBILITY PADDING"). n_groups_marlin == scale.shape[1]
        # always holds given the assert above: either K already divides
        # _IN_TILE (no K padding, group_size == _GROUPSIZE, 1:1) or K <
        # _IN_TILE (K_pad == _IN_TILE == _GROUPSIZE, exactly 1 group on
        # both sides, and zero-padding never changes an amax-derived
        # scale) -- both cases keep the reference's and marlin's per-group
        # scale identical.
        K_pad = ((K + _IN_TILE - 1) // _IN_TILE) * _IN_TILE
        M_pad = ((M + _OUT_TILE - 1) // _OUT_TILE) * _OUT_TILE
        n_groups_marlin = K_pad // _GROUPSIZE
        assert scale.shape[1] == n_groups_marlin, (
            f"{IMPL_NAME}: group-count mismatch (reference {scale.shape[1]} "
            f"vs marlin {n_groups_marlin}) -- see module docstring assert above")

        W_dequant = np.zeros((M_pad, K_pad), dtype=np.float32)
        W_dequant[:M, :K] = dequant
        W_fp16 = torch.as_tensor(W_dequant, dtype=torch.float16, device="cuda")

        scale_full = np.ones((M_pad, n_groups_marlin), dtype=np.float32)
        scale_full[:M, :] = scale
        s_fp16 = torch.as_tensor(scale_full, dtype=torch.float16, device="cuda")

        # marlin's own high-level packing API: linear.weight must be the
        # DEQUANTIZED ("fake-quantized") fp16 weight; pack() internally
        # re-derives round(w/s) == our shared codes exactly (lossless, w
        # is codes*scale by construction) -- see module docstring.
        linear = torch.nn.Linear(K_pad, M_pad, bias=False)
        linear.weight.data = W_fp16
        layer = mlib.Layer(K_pad, M_pad, groupsize=_GROUPSIZE).cuda()
        layer.pack(linear, s_fp16)

        x_pad = np.zeros(K_pad, dtype=np.float32)
        x_pad[:K] = x16.astype(np.float64)
        A_act = torch.as_tensor(x_pad.reshape(1, K_pad), dtype=torch.float16,
                                device="cuda")

        self._M_true = M
        return {"layer": layer, "A": A_act}

    def run(self, h):
        return h["layer"](h["A"])

    def to_host(self, out):
        import torch
        y = out.detach().to("cpu", dtype=torch.float64).numpy()
        return y[0, : self._M_true]

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
