"""
Adapter for MetaAttention (A Unified and Performant Attention Framework
across Hardware Backends, PPoPP'26, SJTU-IPADS/MetaAttention).
`PAPER_KEY = conf/ppopp/ChenC0XMM0X00W026`.

MetaAttention is a Python DSL + code-generation framework: an attention
variant (score function, mask, online-softmax recurrence) is described in
Python and `attn_engine.AttentionEngine(...)` compiles it -- via TileLang as
the runtime backend -- into a callable GPU kernel
(`mod(q, k, v) -> out`). This is fundamentally different from PAT/
FlashAttention-T (explicit hand-written CUDA template instantiations): the
compiled module is produced by calling the artifact's OWN unmodified public
API (`AttentionEngine`, `OnlineFunc`, `CustomIO`, `SymbolScalar`, `Var`,
`meta_tensor`), the same classes `source/examples/mha.py` and
`source/examples/mha_v2.py` use -- this adapter's `_build_softmax_attention`
below is a NEW, small, adapter-local helper (like RASSM's wrapper.cpp
precedent) that calls that SAME public API with the mask/online-softmax
closures spelled out inline, so ONE helper serves both the causal (mask_mod:
q_idx>=kv_idx) and bidirectional (mask_mod: always-true) cases -- the
artifact's own `examples/mha.py::causal_softmax_attention` only wires the
causal case, and `examples/` has no ready-made bidirectional factory to
import directly. Every symbol referenced (`AttentionEngine`, `OnlineFunc`,
`online_fwd`/`online_fwd_epilogue`/`forward`/`backward` protocol, `CustomIO`,
`SymbolScalar`, `Var`, `meta_tensor`, `get_attn_device`) is copied verbatim
(same field names, same recurrence math) from `examples/mha.py`'s and
`examples/mha_v2.py`'s own online-softmax implementation -- nothing here is
invented attention math, only the mask predicate is parameterized.

`prepare()` does two things beyond generating operands, both legitimate
preprocessing under ARTIFACT_GUIDE rule 2:
1. COMPILES the kernel for this exact (B,H,S,D,DV) shape by calling
   `AttentionEngine(...)` -- MetaAttention's kernels are shape-specialized
   (B/H/S/D are compile-time constants baked into the TileLang lowering),
   so "compile a kernel for this workload's shape" genuinely IS this
   artifact's own preprocessing step, timed as such (a few seconds per
   shape, confirmed empirically -- see STATUS.md).
2. Permutes operands from this harness's `(B,H,S,d)` layout to the
   artifact's own `(B,S,H,d)` layout (confirmed from
   `source/testing/test.py::test_softmaxattention`'s
   `query = torch.randn(B, S, H, D, ...)` and `o = attention_module(q,k,v)`
   call pattern) -- this permute is the artifact's own required calling
   convention, not a math change.

RNG discipline (the trap documented in
bench/artifacts/spmm/insum/STATUS.md's "Finding, not an Insum bug"
postmortem): Q/K/V are generated with the EXACT numpy `np.random.default_rng`
formula `kernelbench.domains.ml._qkv` uses -- same seed, same shape order,
same call order -- so this adapter's operands are bit-identical to what the
fp64 reference (and every other impl under test) uses. Re-typed here as a
separate copy rather than importing ml.py's private helper, matching the
insum/fused3s/pat precedent.

Coverage (see STATUS.md for the exact shapes tried): MHA (Hkv==H) CAUSAL
prefill only, verified working INCLUDING at d=16 (every kernelbench --smoke
synthetic shape uses d=16, and unlike PAT/FlashAttention-T -- hand-tuned
CUDA templates hard-restricted to head_dim in {64,128} -- this TileLang-
codegen artifact was empirically confirmed to compile and run CORRECTLY at
d=16 against the harness's fp64 reference, not merely assumed to work).
NOT wired, each documented as a clean `NotImplementedError` rather than a
silent wrong answer:
  * bidirectional mask -- attempted (an always-true mask_mod predicate, both
    infer_mask=True/False), compiles and runs without error but produces
    numerically WRONG output (~1.27 max abs error against an independent
    fp64 reference, vs ~5e-4 for causal at the same shape); not chased
    further within budget, see STATUS.md.
  * GQA (Hkv<H) -- `examples/mha.py`/`mha_v2.py`'s single-`H` factory builds
    Q,K,V all with the same head count; a GQA-capable factory
    (`examples/sparse_gqa_decode.py`) exists but is a block-SPARSE decode
    kernel, not a drop-in dense-GQA replacement, out of budget to adapt.
  * decode (Sq=1) -- not wired, prefill was the priority given budget.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "source")
ATTN_ENGINE_DIR = os.path.join(SOURCE, "attention_engine")
PYLIBS = os.path.join(HERE, "pylibs")

KERNEL = "attention-kernel"
IMPL_NAME = "metaattention-softmax"
PAPER_KEY = "conf/ppopp/ChenC0XMM0X00W026"
PRECISIONS = ["fp16"]

_NP_DTYPE = {"fp16": np.float16}  # only fp16 exercised/gated -- see STATUS.md


def _ensure_paths():
    for p in (PYLIBS, ATTN_ENGINE_DIR, SOURCE):
        if p not in sys.path:
            sys.path.insert(0, p)


def available() -> tuple[bool, str]:
    try:
        _ensure_paths()
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        import tilelang  # noqa: F401
        from attn_engine import AttentionEngine, OnlineFunc  # noqa: F401
        from core import CustomIO, SymbolScalar, Var, meta_tensor  # noqa: F401
        from autotuner.arch import get_attn_device  # noqa: F401
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _build_softmax_attention(B, H, S, D, DV, causal: bool, dtype):
    """
    Compile a MetaAttention softmax-attention module for this exact shape.
    Copied/parameterized from source/examples/mha.py::causal_softmax_attention
    and source/examples/mha_v2.py -- see module docstring. `causal=False`
    uses an always-true mask predicate (bidirectional / full attention),
    `causal=True` the standard lower-triangular one; both share the SAME
    online-softmax recurrence (the artifact's own, unmodified).
    """
    from attn_engine import AttentionEngine, OnlineFunc
    from core import CustomIO, SymbolScalar, Var, meta_tensor
    from autotuner.arch import get_attn_device

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx if causal else (q_idx >= 0)

    softmax_scale = 1.0 / math.sqrt(D)

    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        return score * softmax_scale

    class OnlineSoftmax(OnlineFunc):
        def __init__(self):
            online_rowscales = {
                "m": SymbolScalar("m", Var("-inf")),
                "r": SymbolScalar("r", Var("0.0")),
            }
            final_rowscales = {"lse": SymbolScalar("lse", Var("0.0"))}
            super().__init__(online_rowscales, final_rowscales, CustomIO())

        @staticmethod
        def online_fwd(scores, online_rowscales, b, h, q_idx):
            m, r = online_rowscales["m"], online_rowscales["r"]
            m_new = m.max(scores.get_reduce("max"))
            scale_tmp = (m - m_new).exp()
            r = r * scale_tmp
            scores = (scores - m_new).exp()
            r = r + scores.get_reduce("sum")
            return scores, {"m": m_new, "r": r}, scale_tmp

        @staticmethod
        def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
            o_new = o / online_rowscales["r"]
            lse = online_rowscales["r"].log() + online_rowscales["m"]
            return o_new, {"lse": lse}

        @staticmethod
        def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
            lse = final_rowscales["lse"]
            return (scores - lse).exp()

        @staticmethod
        def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):
            return (dp - doosum_rowscales) * scores

    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )
    online = OnlineSoftmax()
    attn_device = get_attn_device()
    mod = AttentionEngine(
        qkv_meta, CustomIO({}), score_mod=score_mod, mask_mod=mask_mod,
        online_func=online, tune=False,
        tune_file=f"tuned_config/{attn_device.name}/attn_tl.json",
        tune_bwd=False,
        tune_file_bwd=f"tuned_config/{attn_device.name}/attn_tl_bwd.json",
        infer_mask=True,
    )
    return mod


class MetaAttentionSoftmax:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"{IMPL_NAME}: only {PRECISIONS} exercised/gated, got {precision!r}")
        self.precision = precision

    def prepare(self, workload, params: dict):
        _ensure_paths()
        import torch

        w = workload
        if w.Hkv != w.H:
            raise NotImplementedError(
                f"{IMPL_NAME}: GQA (Hkv={w.Hkv} < H={w.H}) not wired -- "
                "examples/mha.py's single-H factory builds Q,K,V with the "
                "same head count; see STATUS.md's coverage section")
        if w.Sq != w.Sk or w.Sq <= 1:
            raise NotImplementedError(
                f"{IMPL_NAME}: only prefill (Sq==Sk>1) wired, not decode "
                f"(Sq={w.Sq}, Sk={w.Sk}); see STATUS.md's coverage section")
        if w.mask != "causal":
            # Tried: mask_mod returning an always-true predicate (q_idx>=0)
            # with both infer_mask=True and infer_mask=False. Both compile
            # and run without error but produce numerically WRONG output
            # (max_abs_err ~1.27 against an independent fp64 bidirectional
            # reference, vs ~5e-4 for the causal path at the same shape) --
            # some assumption in the artifact's block-mask codegen (or in
            # how this adapter's predicate gets symbolically traced) does
            # not hold for a non-causal mask. Not chased further within
            # budget; raising cleanly rather than shipping a silently wrong
            # bidirectional path. See STATUS.md's coverage section.
            raise NotImplementedError(
                f"{IMPL_NAME}: mask={w.mask!r} not supported -- bidirectional "
                "was attempted and produces incorrect output (see adapter.py "
                "comment + STATUS.md); only causal is gated")

        B, H, S, d = w.B, w.H, w.Sq, w.d
        causal = True
        seed = params.get("seed", w.seed)
        np_dtype = _NP_DTYPE[self.precision]

        # Operand generation: EXACT match to kernelbench.domains.ml._qkv's
        # np.random.default_rng formula -- see module docstring's RNG
        # discipline note. Generated in this harness's (B,H,S,d) layout,
        # THEN permuted to the artifact's own (B,S,H,d) calling convention.
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((B, H, S, d)).astype(np_dtype)
        K = rng.standard_normal((B, H, S, d)).astype(np_dtype)
        V = rng.standard_normal((B, H, S, d)).astype(np_dtype)

        torch_dtype = torch.float16
        Qt = torch.from_numpy(Q).to(device="cuda", dtype=torch_dtype)
        Kt = torch.from_numpy(K).to(device="cuda", dtype=torch_dtype)
        Vt = torch.from_numpy(V).to(device="cuda", dtype=torch_dtype)

        # Artifact's own layout is (B, S, H, d), not this harness's (B, H, S, d)
        # -- the permute+contiguous() IS the artifact's calling convention,
        # timed here as preprocessing (ARTIFACT_GUIDE.md rule 2).
        Qt = Qt.permute(0, 2, 1, 3).contiguous()
        Kt = Kt.permute(0, 2, 1, 3).contiguous()
        Vt = Vt.permute(0, 2, 1, 3).contiguous()

        # Compile the kernel for THIS shape (MetaAttention's own
        # preprocessing step -- see module docstring point 1).
        mod = _build_softmax_attention(B, H, S, d, d, causal, torch_dtype)

        return {"mod": mod, "q": Qt, "k": Kt, "v": Vt}

    def run(self, h):
        return h["mod"](h["q"], h["k"], h["v"])

    def to_host(self, out):
        import torch
        # artifact's (B, Sq, H, d) -> ml.py reference's (B, H, Sq, d)
        return out.detach().permute(0, 2, 1, 3).contiguous().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return MetaAttentionSoftmax(precision)
