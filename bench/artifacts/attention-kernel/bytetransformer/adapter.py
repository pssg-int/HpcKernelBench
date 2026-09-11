"""
Adapter for ByteTransformer (bytedance/ByteTransformer, IPDPS'23,
"ByteTransformer: A High-Performance Transformer Boosted for Variable-Length
Inputs"). `PAPER_KEY = conf/ipps/ZhaiJWJZCLZ23`.

Wraps `bytetransformer::Attention<OperationType::HALF>::fused_rm_infer` /
`::fused_long_rm_infer` -- the artifact's fused, PADDING-FREE (remove-padding)
multi-head self-attention kernel -- via `attn_binding.cu`, a thin pybind11
glue file WE wrote (see its own docstring for the full derivation: why this
boundary needs no CUTLASS, and why linking still needs
attention_nofused.cu/attention_nofused_utils.cu/gemm.cu even though those
code paths are never executed at runtime). This is ONE kernel invocation per
`run()` call, not the paper's own benchmark harness or the encoder-layer
`th_op/` torch binding (which additionally runs QKV/output GEMMs, LayerNorm
and the FFN block -- an encoder LAYER, not the attention kernel).

Boundary chosen: the "remove-padding" (padding-free) kernel family is the
paper's headline contribution ("Boosted for Variable-Length Inputs"). Per
this task's own instruction, this adapter feeds EQUAL-length sequences --
the degenerate, no-padding-to-remove case -- since
`kernelbench.domains.ml.AttentionWorkload` models a dense (B,H,S,d) batch
with no per-sequence length variation. This is still the kernel the paper
ships (its own README documents exactly this "no padding" special case),
not a different algorithm; batch_idx = [0, S, 2S, ..., B*S] is simply the
correct cu-seqlens-style offsets for that degenerate input, not a workaround.

Real, load-bearing constraints of the artifact's OWN dispatch code (see
attn_binding.cu's docstring, cross-checked from attention_fused.cu /
attention_fused_long.cu), enforced here as NotImplementedError per
ARTIFACT_GUIDE rule 8:
  - size_per_head (d) must be EXACTLY 64 -- every WMMA_ATTENTION_RM /
    WMMA_ATTENTION_LONG_RM macro invocation hardcodes the SIZE_PER_HEAD
    template argument to the literal `64`, independent of what's passed to
    the Attention<> constructor.
  - seq_len (S) must be in [1, 352] -- fused_rm_infer's switch only has
    cases for S in {16,32,48,64,80} and fused_long_rm_infer's only for
    split_count 6..22 (S in (80,352]); outside that, NEITHER function
    launches any kernel and `output` stays uninitialized.
  - H == Hkv (no GQA) -- the kernel has a single `head_num`, no separate KV
    head count.
  - mask must be "bidirectional" -- fused_rm_kernel's own softmax loop
    applies NO masking beyond each sequence's own length (the `attention_mask`
    parameter is accepted but never dereferenced in the rm kernel body); it
    is unconditionally full self-attention within a sequence. A causal
    workload would silently get the wrong (unmasked) answer, so causal is
    refused rather than silently mismeasured.
  - variant_kind must be "prefill" (Sq==Sk==S>1) -- this is a batched
    self-attention kernel over one packed sequence set, not a KV-cache
    decode kernel.
  - precision fp16 only (Traits<HALF>::DataType=__half; the FP32
    specialization forces use_fused_attention_=false in the artifact's own
    Attention constructor, i.e. FP32 never even reaches this kernel family).

RNG discipline: Q/K/V generated bit-identically to `kernelbench.domains.ml
._qkv` (same np.random.default_rng seed, same shapes, same call order), per
the precedent in pat/gpa/fused3s's adapters. QKV bias is fed as ZERO (the
harness's AttentionWorkload/reference_attention model no bias term; the
kernel always adds a QKV bias internally, so a zero bias makes its computed
math equal ours exactly, not an approximation).

Layout: the kernel wants ONE packed `qkv` tensor of shape
[B*S, 3*hidden_dim], token-major, with Q/K/V concatenated along the last
axis and heads concatenated within each segment (`hidden_dim = H*d`) --
derived from attention_fused.cu's own pointer arithmetic
(`pos = token*(hidden_dim/2 half2 units)*3 + head*(d/2) + ...`), cross-
checked against th_op/ths_op.h's own qkv_kernel weight layout comment
(`hidden_dim, hidden_dim*3`). `to_host()` reshapes the flat
(B*S, H*d) output back to (B,H,S,d) to compare against
`reference_attention`'s layout.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

KERNEL = "attention-kernel"
IMPL_NAME = "bytetransformer-fused-rm-attn"
PAPER_KEY = "conf/ipps/ZhaiJWJZCLZ23"
# Traits<OperationType::HALF>::DataType = __half; FP32 forces
# use_fused_attention_=false in the artifact's own Attention constructor, so
# fp32 never reaches this kernel family at all -- fp16 only.
PRECISIONS = ["fp16"]

HEAD_SIZE = 64          # hardcoded SIZE_PER_HEAD in every dispatch macro
MAX_SEQ_LEN = 352        # fused_long_rm_infer's last dispatched bucket


def _ensure_path():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)


def available() -> tuple[bool, str]:
    so_present = any(f.startswith("bt_attn_binding.cpython") and f.endswith(".so")
                      for f in os.listdir(HERE)) if os.path.isdir(HERE) else False
    if not so_present:
        return False, "bt_attn_binding*.so not built -- run build.sh"
    try:
        _ensure_path()
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        import bt_attn_binding  # noqa: F401
        return True, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg:
            msg += (" -- re-run with LD_PRELOAD=/usr/lib64/libstdc++.so.6 "
                    "set BEFORE the python process starts; see STATUS.md")
        return False, msg


def _qkv_like_ml(B, H, Hkv, Sq, Sk, d, seed, np_dtype):
    """Bit-identical to kernelbench.domains.ml._qkv -- see this adapter's
    module docstring's RNG discipline note."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, Sq, d)).astype(np_dtype)
    K = rng.standard_normal((B, Hkv, Sk, d)).astype(np_dtype)
    V = rng.standard_normal((B, Hkv, Sk, d)).astype(np_dtype)
    return Q, K, V


class ByteTransformerFusedRmAttn:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"{IMPL_NAME} only services {PRECISIONS} (Traits<HALF> tensor-core "
                f"WMMA kernel; the artifact's own FP32 specialization disables "
                f"fused attention entirely); requested {precision!r}")
        self.precision = precision

    def prepare(self, workload, params: dict):
        _ensure_path()
        import torch
        import bt_attn_binding

        w = workload
        if w.kernel != "attention-kernel":
            raise NotImplementedError(f"{IMPL_NAME} only services attention-kernel")
        if w.variant_kind != "prefill" or w.Sq != w.Sk or w.Sq <= 1:
            raise NotImplementedError(
                f"{IMPL_NAME} only implements prefill self-attention (Sq==Sk>1); "
                f"got variant_kind={w.variant_kind!r}, Sq={w.Sq}, Sk={w.Sk}")
        if w.mask != "bidirectional":
            raise NotImplementedError(
                f"{IMPL_NAME}'s fused_rm_infer kernel applies NO masking beyond "
                f"each sequence's own length (attention_mask is a dead parameter "
                f"in wmma_attention_rm_kernel) -- it is unconditionally full "
                f"bidirectional self-attention; got mask={w.mask!r}")
        if w.H != w.Hkv:
            raise NotImplementedError(
                f"{IMPL_NAME}: the kernel has a single head_num, no GQA support; "
                f"got H={w.H}, Hkv={w.Hkv}")
        if w.d != HEAD_SIZE:
            raise NotImplementedError(
                f"{IMPL_NAME}: SIZE_PER_HEAD is hardcoded to {HEAD_SIZE} in every "
                f"WMMA_ATTENTION_(LONG_)RM macro invocation; got d={w.d}")
        if not (1 <= w.Sq <= MAX_SEQ_LEN):
            raise NotImplementedError(
                f"{IMPL_NAME}: fused_rm_infer/fused_long_rm_infer's own dispatch "
                f"switches only cover seq_len in [1,{MAX_SEQ_LEN}]; got Sq={w.Sq}")

        B, H, d, S = w.B, w.H, w.d, w.Sq
        hidden = H * d
        seed = params.get("seed", w.seed)
        np_dtype = np.float16
        Q, K, V = _qkv_like_ml(B, H, H, S, S, d, seed, np_dtype)

        # ---- artifact's own packed-QKV layout: [B*S, 3*hidden], heads
        # concatenated within each of the Q/K/V segments (see module
        # docstring's "Layout" note). This packing IS this adapter's
        # preprocessing (ARTIFACT_GUIDE rule 2), timed once in prepare().
        Qf = np.ascontiguousarray(Q.transpose(0, 2, 1, 3)).reshape(B * S, hidden)
        Kf = np.ascontiguousarray(K.transpose(0, 2, 1, 3)).reshape(B * S, hidden)
        Vf = np.ascontiguousarray(V.transpose(0, 2, 1, 3)).reshape(B * S, hidden)
        qkv_np = np.concatenate([Qf, Kf, Vf], axis=-1)  # (B*S, 3*hidden)

        qkv = torch.from_numpy(qkv_np).to("cuda", torch.float16).contiguous()
        qkv_bias = torch.zeros(3 * hidden, dtype=torch.float16, device="cuda")
        batch_idx = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device="cuda")

        self._last_bhsd = (B, H, S, d)

        return {
            "qkv": qkv, "qkv_bias": qkv_bias, "batch_idx": batch_idx,
            "B": B, "S": S, "H": H, "d": d,
        }

    def run(self, h):
        import bt_attn_binding
        return bt_attn_binding.fused_rm_attention(
            h["qkv"], h["qkv_bias"], h["batch_idx"], h["B"], h["S"], h["H"], h["d"])

    def to_host(self, out):
        import torch
        B, H, S, d = self._last_bhsd
        # out: (B*S, H*d) -> (B,S,H,d) -> (B,H,S,d), matching
        # reference_attention's layout.
        reshaped = out.reshape(B, S, H, d).transpose(1, 2)
        return reshaped.detach().contiguous().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return ByteTransformerFusedRmAttn(precision)
