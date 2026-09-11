"""
QFactory adapter for the quantized-gemm track.

Paper: "QFactory: Accelerating Quantized Large Language Model Serving with
Qtile Graphs", USENIX ATC'25. `PAPER_KEY = conf/usenix/ZhangZSZ25`.
Artifact: https://github.com/zqh-wz/QFactory-AE -- a Qtile-graph JIT
compiler for quantized GEMM kernels. `qfactory.QLinear`/`GPTQLinear`
(source/qfactory/qlinear.py) is the library's own high-level entry point
(the same one `tests/acc_gptq.py`, the artifact's own accuracy test, calls),
wrapped here UNMODIFIED -- no copy, no source edit.

`qfactory/` (the package) is pure Python + a runtime nvcc JIT (like
`tilus`): `GPTQLinear.__init__` calls `matmul.compile_profile(...)`, which
emits a CUDA source file and compiles it the first time a given
(shape, dtype, group_size) config is requested -- see prepare() below,
which is exactly where that JIT build happens (this artifact's own
preprocessing, per the task brief). `compile_flags={"enable_transform":
False, "enable_schedule": False, "enable_lower": False}` -- copied verbatim
from `tests/acc_gptq.py`'s own accuracy-test invocation -- disables
QFactory's autotuning search passes, so only ONE kernel variant is ever
compiled (measured ~11s cold-build for the smoke shape below), the same
"the artifact's own accuracy-test config already avoids expensive
autotuning" pattern `tilus`'s `debug_schedule` class attribute provides.

KEY FAIRNESS POINT (per this integration's task brief -- domains/ml.py's
QuantGemmWorkload docstring states the identical rule): every impl must
consume the SAME quantized values as `reference_qgemm`. QFactory's own
`GPTQLinear` accepts EXTERNALLY SUPPLIED `(weight, scale, zero)` -- it never
re-quantizes from a continuous source itself (`tests/acc_gptq.py`'s own
`asymmetric_quant`/`dequant` free functions are TEST-SCRIPT convenience code,
not part of the `qfactory` package -- never imported by this adapter). This
makes value-sharing straightforward, like `tilus`, UNLIKE `quantix` (SKIPPED,
see that adapter's STATUS.md) -- but QFactory's own quantization convention
is ASYMMETRIC (unsigned code in `[0, 2^nbits)`, `dequant = (code - (zero+1))
* scale`, GPTQ's own zero-point convention -- see `convert.py::dequant` in
this same repo, again not imported here), while `QuantGemmWorkload.quantize()`
is SYMMETRIC (signed code in `[-qmax, qmax]`, `dequant = code * scale`, no
zero point). These are reconciled with an exact affine SHIFT, not an
approximation:

    qweight = code + qmax            (unsigned code in [0, 2*qmax] subset of [0, 2^nbits))
    zero    = qmax - 1               (constant, same for every group/column)
    =>  (qweight - (zero + 1)) * scale = (code + qmax - qmax) * scale = code * scale

i.e. QFactory's dequantized value is made to equal `reference_qgemm`'s
EXACTLY (not approximately) for every element, by construction -- a
zero-point reparameterization of the SAME shared codes/scale, not a
different quantization. Verified: gate error lands at the same ~1e-4 scale
as `tilus`'s and `numpy-dequant-gemm`'s (see STATUS.md), confirming the
mapping is exact and the residual is ordinary fp16/kernel-arithmetic error.

GROUP-COUNT ALIGNMENT (an artifact constraint, not a value-sharing issue):
`GPTQLinear.__init__` asserts `in_features % (group_size * 2) == 0` -- K
must split into an EVEN number of groups. `QuantGemmWorkload`'s OWN group
size (`w.group_size_eff`) does not always satisfy this (e.g. this domain's
own per-column-fallback smoke shape has exactly 1 group). Rather than
raising (which would crash `--smoke`'s whole run -- `runner.py` has no
per-shape try/except) or touching the shared smoke-shape list (would need
re-verifying every OTHER impl against a changed list), this adapter picks
its OWN, finer group size purely for what it hands to QFactory: if
`w.n_groups` is already even, use `w.group_size_eff` unchanged; otherwise
halve it (`w.group_size_eff` is even for every shape this domain currently
generates -- 128/96/64 -- so halving always yields an integer). Halving a
group and REPEATING its one shared scale value across both halves does not
change any element's dequantized value (dequant is elementwise `code*scale`;
two adjacent sub-groups sharing the identical scale value dequantize
identically to one group spanning both) -- so this is a bookkeeping choice,
not a numerical approximation, and every element QFactory computes still
matches `reference_qgemm`'s exactly by the shift identity above.
"""

from __future__ import annotations

import os

import numpy as np

KERNEL = "quantized-gemm"
IMPL_NAME = "qfactory-gptq-matmul"
PAPER_KEY = "conf/usenix/ZhangZSZ25"
PRECISIONS = ["fp16"]

HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_DIR = os.path.join(HERE, "source")
_CACHE_DIR = os.path.join(HERE, "cache")

_BITS_SUPPORTED = (2, 4, 8)   # QFactory's own PERMUTE_ORDER table (convert.py) only defines 2/4/8


def available() -> tuple[bool, str]:
    if not os.path.isdir(os.path.join(_SOURCE_DIR, "qfactory")):
        return False, "source/ not cloned"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        import sys
        if _SOURCE_DIR not in sys.path:
            sys.path.insert(0, _SOURCE_DIR)
        import qfactory  # noqa: F401
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg:
            msg = ("libstdc++ ABI mismatch -- run with "
                   "LD_PRELOAD=/usr/lib64/libstdc++.so.6 (see STATUS.md)")
        return False, msg
    return True, ""


def create(precision: str):
    return QFactoryQuantGemm(precision)


def _activation(w, params, dtype) -> np.ndarray:
    """Identical recipe to kernelbench.domains.ml._make_activation (private
    to that module, replicated verbatim rather than imported across modules
    -- same convention turbofno's/fused3s's/tilus's own adapters use)."""
    seed = params.get("seed", w.seed)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((w.M, w.K)).astype(dtype)


class QFactoryQuantGemm:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp16 activation "
                f"(qfactory's kernel_gptq is fp16-activation only); "
                f"requested {precision!r}")
        self.precision = precision

    def prepare(self, workload, params: dict):
        import sys
        if _SOURCE_DIR not in sys.path:
            sys.path.insert(0, _SOURCE_DIR)
        os.environ.setdefault("QFACTORY_ARCH", "80")
        os.environ.setdefault("QFACTORY_CACHE_DIR", _CACHE_DIR)
        import torch
        from qfactory import QLinear

        w = workload
        if w.bits not in _BITS_SUPPORTED:
            raise NotImplementedError(
                f"{IMPL_NAME}: bits={w.bits} not in {_BITS_SUPPORTED} "
                "(qfactory's convert.py PERMUTE_ORDER table only defines "
                "2/4/8-bit packing)")

        codes, scale, group_idx = w.quantize()   # SHARED -- see module docstring
        qmax = (1 << (w.bits - 1)) - 1
        K, N = w.K, w.N
        gs_eff, n_groups_eff = w.group_size_eff, w.n_groups

        # pick a QFactory-side group size with an EVEN group count (see
        # module docstring's "GROUP-COUNT ALIGNMENT" note); scale is
        # repeated across the split, never recomputed.
        if n_groups_eff % 2 == 0:
            gs_q, repeat = gs_eff, 1
        else:
            if gs_eff % 2 != 0:
                raise NotImplementedError(
                    f"{IMPL_NAME}: group_size_eff={gs_eff} is odd with an "
                    f"odd group count ({n_groups_eff}) -- no even-group-"
                    "count split available; qfactory's GPTQLinear requires "
                    "K %% (2*group_size) == 0")
            gs_q, repeat = gs_eff // 2, 2
        n_groups_q = K // gs_q

        # exact zero-point shift: (code+qmax) - (qmax-1+1) == code (see
        # module docstring's derivation) -- makes qfactory's dequantized
        # value equal reference_qgemm's dequantized value elementwise.
        qweight_np = (codes.astype(np.int64) + qmax).astype(np.int32)
        zero_val = qmax - 1
        scale_q_np = np.repeat(scale, repeat, axis=0)          # (n_groups_q, N)
        zeros_q_np = np.full((n_groups_q, N), zero_val, dtype=np.int32)

        qweight = torch.as_tensor(np.ascontiguousarray(qweight_np),
                                  dtype=torch.int32, device="cuda")
        scales_t = torch.as_tensor(scale_q_np.astype(np.float16), device="cuda")
        zeros_t = torch.as_tensor(zeros_q_np, dtype=torch.int32, device="cuda")

        ql = QLinear.init(
            w.M, K, N, w.bits, qweight, gs_q, scales_t, zeros_t,
            compile_flags={"enable_transform": False,
                           "enable_schedule": False, "enable_lower": False})

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
