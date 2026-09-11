"""
Adapter for E.T. (cctry/E.T., SC'21, "E.T.: Re-Thinking Self-Attention for
Transformer Models on GPUs"). `PAPER_KEY = conf/sc/ChenHPLG0D021`.

Wraps `OTF_attention_kernelLauncher` -- E.T.'s "On-The-Fly" DENSE self-
attention kernel (`kernels/attention.cu`) -- via `attn_binding.cu`, a thin
pybind11 glue file WE wrote (E.T. ships NO python bindings at all, only
CMake C++ test binaries per its own README: "a few examples of encoder").
This is the DENSE self-attention path only, per this task's instructions:
the repo also ships `Prune_attention_kernelLauncher` (attention-aware
structured-pruned V, a separate contribution) and
`sharedQK_attention_kernelLauncher` (consumes an EXTERNALLY precomputed QK
product via a separate cuBLAS strided-batched GEMM, used by the
"sequence-aware" DistilBERT-style encoder) -- neither is wrapped here.

Boundary: `OTF_attention_kernelLauncher` is a standalone function call --
one CUDA kernel launch (`__kernel_multi_head_full_skew_warpSFM`) computing
`softmax(QK^T/sqrt(d) + mask) @ V` for ONE (batch-less) sequence, confirmed
from `encoder/encoder.cu`'s `Encoder_tile::run()`, which calls it directly
(no wrapping class needed -- this is finer-grained than even that encoder's
own attention step, which additionally needs the QKV/output projection
GEMMs, LayerNorms and the FFN -- an encoder LAYER, not this kernel).

Real, load-bearing constraints of the kernel's OWN implementation
(`attention.cu`, read in full), enforced as `NotImplementedError` per
ARTIFACT_GUIDE rule 8:
  - seq_len (S) must be a multiple of 16 -- the kernel's grid is
    `dim3(seq_len/16, nhead)` (integer division, no remainder handling); a
    non-multiple would silently leave the last `seq_len % 16` rows of the
    output uninitialized.
  - head_dim (d) must be a multiple of 16 -- the WMMA 16x16x16 fragment
    tiling loop (`for i=0;i<head_dim;i+=16`) requires this exactly.
  - H == Hkv (no GQA) -- the launcher takes one `nhead`, used identically
    for Q/K/V's head-slicing offset.
  - dynamic shared memory must fit the CUDA default 48KB limit WITHOUT an
    opt-in `cudaFuncSetAttribute` call -- `OTF_attention_kernelLauncher`
    itself never raises the limit (unlike, e.g., ByteTransformer's
    long-sequence kernel, which explicitly does), so for large
    `head_dim`/`seq_len` combinations the kernel launch would itself fail
    at the CUDA runtime level; this adapter computes the artifact's own
    `smem_size` formula and refuses upfront rather than let that happen.
  - precision fp16 only -- every buffer in `attention.cu` is `half`; the
    kernel has no other arithmetic path.
  - `variant_kind=="prefill"`, `Sq==Sk>1` -- this is a batched-over-nothing
    (single sequence) self-attention kernel, not a KV-cache decode kernel.

Unlike bytetransformer's fused_rm kernel, this kernel's masking IS fully
general (an additive (seq_len,seq_len) matrix, 0 keep / -inf masked,
confirmed from `__kernel_multi_head_full_skew_warpSFM`'s
`temp_row2[...] += mask_base[...]` line, applied to the ALREADY-SCALED
QK^T scores) -- so BOTH "causal" and "bidirectional" masks from this
track's `AttentionWorkload` are supported.

Batching: the kernel has no batch dimension in its own grid (confirmed from
`encoder/encoder.cu`'s `Encoder_tile::run()`, which calls it once per
single sequence) -- `run()` loops over the workload's B in Python, calling
the binding once per batch item, matching the gpa/spfa_csr adapter's
identical "loop over independent per-item kernel calls" precedent (the same
fixed (seq_len,seq_len) additive mask is reused, unscaled, across every
batch item and every head, matching the kernel's own per-call mask
argument and this track's own "same mask for every batch/head" convention).

RNG discipline: Q/K/V generated bit-identically to `kernelbench.domains.ml
._qkv` (same seed/shape/call-order), per the pat/gpa/bytetransformer
precedent.

PATCH recorded (see build.sh / source.patch): `kernels/kernels.h`'s last
declaration (`sharedQK_attention_kernelLauncher`) is missing its
terminating semicolon in the artifact's own upstream repo -- a one-
character header-syntax fix, not a change to any kernel algorithm/logic.

**CONFIRMED ARTIFACT BUG, gate FAILS (see STATUS.md for the full isolation):**
`kernels/attention.cu`'s `softmax_blk()` helper (used by this kernel) calls
`h2exp2()` (base-2 exponential, `2^x`) where the softmax definition needs
`h2exp()` (`e^x`) -- with no compensating `log2(e)` factor anywhere. Every
row still sums to 1 (a valid, but WRONG, probability distribution,
systematically flatter than the true softmax by an effective `ln(2)`
logit-scale factor). This is not a bug in this adapter's wrapping -- it
was isolated with an "identity-V" readout trick and hand-computed exact
logits (STATUS.md point 4): the kernel's output for logits `[0,1,2,3]` is
bit-for-bit `2^[0,1,2,3]` normalized, not `e^[0,1,2,3]` normalized. The
correctness gate on this track's real workloads (bert-base-short/
bert-large/gpt2-small) fails at `max_scaled_err` 0.47-1.7 against tolerance
1e-2, exactly as expected from this root cause. Reported honestly per
ARTIFACT_GUIDE rule 4 (a failing gate IS the result); not patched (rule 3
reserves kernel-logic changes for a SKIPPED verdict, not a silent fix).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

KERNEL = "attention-kernel"
IMPL_NAME = "et-otf-attn"
PAPER_KEY = "conf/sc/ChenHPLG0D021"
# every buffer in attention.cu is `half` -- no other arithmetic path exists.
PRECISIONS = ["fp16"]

FP16_SKEW = 16          # attention.cu's own constant
MAX_SMEM_BYTES = 49152  # CUDA's default max dynamic shared mem/block (48KB);
# OTF_attention_kernelLauncher never calls cudaFuncSetAttribute to raise this
# (unlike bytetransformer's long-sequence kernel), so a launch beyond this is
# a genuine artifact limitation, not a gate to work around.


def _ensure_path():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)


def available() -> tuple[bool, str]:
    so_present = any(f.startswith("et_attn_binding.cpython") and f.endswith(".so")
                      for f in os.listdir(HERE)) if os.path.isdir(HERE) else False
    if not so_present:
        return False, "et_attn_binding*.so not built -- run build.sh"
    try:
        _ensure_path()
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        import et_attn_binding  # noqa: F401
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


def _smem_bytes(seq_len: int, head_dim: int) -> int:
    """OTF_attention_kernelLauncher's own formula (attention.cu):
    smem = sizeof(half) * (16*(head_dim+FP16_skew) + 16*(seq_len+FP16_skew))."""
    temp_q = 16 * (head_dim + FP16_SKEW)
    temp_row = 16 * (seq_len + FP16_SKEW)
    return 2 * (temp_q + temp_row)  # sizeof(half) == 2


class EtOtfAttn:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"{IMPL_NAME} only services {PRECISIONS} (every buffer in "
                f"attention.cu is `half`); requested {precision!r}")
        self.precision = precision

    def prepare(self, workload, params: dict):
        _ensure_path()
        import torch

        w = workload
        if w.kernel != "attention-kernel":
            raise NotImplementedError(f"{IMPL_NAME} only services attention-kernel")
        if w.variant_kind != "prefill" or w.Sq != w.Sk or w.Sq <= 1:
            raise NotImplementedError(
                f"{IMPL_NAME} only implements prefill self-attention (Sq==Sk>1); "
                f"got variant_kind={w.variant_kind!r}, Sq={w.Sq}, Sk={w.Sk}")
        if w.H != w.Hkv:
            raise NotImplementedError(
                f"{IMPL_NAME}: OTF_attention_kernelLauncher takes a single "
                f"nhead, no GQA support; got H={w.H}, Hkv={w.Hkv}")
        S, d = w.Sq, w.d
        if S % 16 != 0:
            raise NotImplementedError(
                f"{IMPL_NAME}: the kernel's grid is dim3(seq_len/16, nhead) with "
                f"no remainder handling; seq_len must be a multiple of 16, got S={S}")
        if d % 16 != 0:
            raise NotImplementedError(
                f"{IMPL_NAME}: the WMMA 16x16x16 fragment tiling requires "
                f"head_dim to be a multiple of 16; got d={d}")
        smem = _smem_bytes(S, d)
        if smem > MAX_SMEM_BYTES:
            raise NotImplementedError(
                f"{IMPL_NAME}: dynamic shared memory needed ({smem} bytes) "
                f"exceeds CUDA's default 48KB ({MAX_SMEM_BYTES}) limit; "
                f"OTF_attention_kernelLauncher never calls cudaFuncSetAttribute "
                f"to raise this (S={S}, d={d})")

        B, H = w.B, w.H
        seed = params.get("seed", w.seed)
        np_dtype = np.float16
        Q, K, V = _qkv_like_ml(B, H, H, S, S, d, seed, np_dtype)
        hidden = H * d

        # Packed per-batch (S, hidden) layout, heads concatenated -- matches
        # OTF_attention_kernelLauncher's own K_ptr/Q_ptr offset arithmetic
        # (head_dim * blockIdx.y within a d_model-wide row).
        Qf = np.ascontiguousarray(Q.transpose(0, 2, 1, 3)).reshape(B, S, hidden)
        Kf = np.ascontiguousarray(K.transpose(0, 2, 1, 3)).reshape(B, S, hidden)
        Vf = np.ascontiguousarray(V.transpose(0, 2, 1, 3)).reshape(B, S, hidden)

        Qc = torch.from_numpy(Qf).to("cuda", torch.float16).contiguous()
        Kc = torch.from_numpy(Kf).to("cuda", torch.float16).contiguous()
        Vc = torch.from_numpy(Vf).to("cuda", torch.float16).contiguous()

        # ---- artifact's own additive-mask convention: build the (S,S)
        # 0/-inf matrix ONCE (workload data, per this track's "same mask for
        # every batch/head" convention -- see reference_attention's own
        # docstring), reused unscaled for every batch item in run().
        if w.mask == "causal":
            i = np.arange(S)[:, None]
            j = np.arange(S)[None, :]
            keep = j <= i
        elif w.mask == "bidirectional":
            keep = np.ones((S, S), dtype=bool)
        else:
            raise NotImplementedError(f"{IMPL_NAME}: unknown mask {w.mask!r}")
        mask_np = np.where(keep, 0.0, float("-inf")).astype(np.float16)
        mask_c = torch.from_numpy(mask_np).to("cuda", torch.float16).contiguous()

        self._last_bhsd = (B, H, S, d)

        return {"Q": Qc, "K": Kc, "V": Vc, "mask": mask_c, "B": B, "S": S,
                "hidden": hidden, "H": H}

    def run(self, h):
        import et_attn_binding
        B, S, hidden, H = h["B"], h["S"], h["hidden"], h["H"]
        Q, K, V, mask = h["Q"], h["K"], h["V"], h["mask"]
        # One kernel launch per batch item (no batch dim in the artifact's
        # own grid -- see module docstring's "Batching" note); the SAME
        # mask buffer is passed to every call, matching the kernel's own
        # per-call mask argument.
        outs = [et_attn_binding.otf_attention(Q[b], K[b], V[b], mask, S, hidden, H)
                for b in range(B)]
        return outs

    def to_host(self, outs):
        import torch
        B, H, S, d = self._last_bhsd
        stacked = torch.stack(outs, dim=0)          # (B, S, hidden)
        reshaped = stacked.reshape(B, S, H, d).transpose(1, 2)  # (B,H,S,d)
        return reshaped.detach().contiguous().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return EtOtfAttn(precision)
