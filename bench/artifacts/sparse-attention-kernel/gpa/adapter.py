"""
Adapter for gpa (TomczakK25, "Longer Attention Span: Increasing Transformer
Context Length With Sparse Graph Processing Techniques", IPDPS'25,
KLab-AI3/Graph-Processing-Attention-IPDPS-2025).
`PAPER_KEY = conf/ipps/TomczakK25`.

Of the repo's 6 kernel variants (COO/CSR explicit-mask + Local/
Local-1D-Dilated/Local-2D-Dilated/Global-No-Local implicit-mask), this
adapter wraps `spfa_csr` -- the CSR explicit-mask kernel
(`source/Sparse_FlashAttention_CSR/`) -- the most general variant: it
accepts an ARBITRARY binary mask in CSR form, so it is a drop-in receiver
for every pattern this track's `SparseAttentionWorkload` can build, not
restricted to one parametric family. The other 5 variants are not built
here; see build.sh's header for the rationale (compile-time/risk budget).

**Read the kernel source in full** (`sp_flatt_csr_kernel.cu`) before writing
this adapter, which turned up two things worth recording:

1. `W_val` (the CSR mask's VALUE array) is accepted as a kernel parameter
   but **never read** by the compute -- the kernel always recomputes the
   raw score `dot(Q_i,K_j)/sqrt(d)` fresh into its own shared-memory
   scratch buffer (also confusingly named `W_val_shared`) and never
   dereferences the `W_val` global-memory accessor at all. This means the
   kernel implements EXACTLY this track's operation -- unweighted 0/1-mask
   softmax attention, `O = softmax(QK^T/sqrt(d) restricted to the mask) @
   V` -- with the "weight" concept in its own CSR format being vestigial.
   This adapter passes `W_val = ones(nnz)` (any value would give the
   identical result; ones is the clearest expression of "these are the
   nonzero positions").
2. Confirmed via `verification/verify.py` (the artifact's own, unmodified
   correctness script) that CSR's own call convention is `spfa_csr.forward(
   q, k, v, w_row_off, w_col_ind, w_val, m, l, out, use_nan)` with `q,k,v`
   as 2D `(N,d)` tensors (README: "single-headed, single-batched
   attention" -- no batch/head dims in this kernel's own interface). This
   adapter therefore loops over the workload's `B*H` independent attention
   problems in `run()`, calling the SAME kernel entry point once per
   (batch, head) slice -- exactly how a real caller with more than one
   head/batch would have to use this artifact (nothing hidden from the
   timed region).

`m`/`l` are FlashAttention-style online-softmax running-state buffers the
CALLER must pre-initialize (`m = -inf`, `l = 0.0`, per `verify.py`'s own
convention) -- this is preprocessing-adjacent bookkeeping, allocated once in
`prepare()` and reset per (b,h) slice in `run()` (the reset itself is O(N),
part of this kernel's genuine per-call state, not hidden).

PATCH recorded (see build.sh's header comment and STATUS.md): a
torch-version compatibility fix (`Q.type()` -> `Q.scalar_type()`) was
required in `sp_flatt_csr_kernel.cu` for the extension to even COMPILE
against this machine's torch 2.8 (`Q.type()` is deprecated ATen API that a
current `AT_DISPATCH_FLOATING_TYPES_AND_HALF` can no longer implicitly
convert). Semantically identical dtype dispatch, no kernel-algorithm change
-- the same class of fix ARTIFACT_GUIDE rule 3 sanctions ("CUDA-version
guards are fine").
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSR_DIR = os.path.join(HERE, "source", "Sparse_FlashAttention_CSR")

KERNEL = "sparse-attention-kernel"
IMPL_NAME = "gpa-spfa-csr"
PAPER_KEY = "conf/ipps/TomczakK25"
# AT_DISPATCH_FLOATING_TYPES_AND_HALF -- fp32/fp16 (and fp64, not serviced
# here: this track's operand convention widens from run precision, and the
# CPU reference is already the fp64 ground truth).
PRECISIONS = ["fp16", "fp32"]

_TORCH_DTYPE_NAME = {"fp16": "float16", "fp32": "float32"}


def _ensure_path():
    if CSR_DIR not in sys.path:
        sys.path.insert(0, CSR_DIR)


def available() -> tuple[bool, str]:
    so_present = os.path.isdir(CSR_DIR) and any(
        f.startswith("spfa_csr") and f.endswith(".so") for f in os.listdir(CSR_DIR))
    if not so_present:
        return False, "spfa_csr*.so not built -- run build.sh"
    try:
        _ensure_path()
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        import spfa_csr  # noqa: F401
        return True, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg:
            msg += (" -- re-run with LD_PRELOAD=/usr/lib64/libstdc++.so.6 "
                    "set BEFORE the python process starts; see STATUS.md")
        return False, msg


def _qkv_like_ml(B, H, S, d, seed, np_dtype):
    """Bit-identical to kernelbench.domains.ml._qkv_sparse -- see this
    track's other adapters (vit-sparse, sparse-transformer) for the same,
    independently-typed pattern."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    K = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    V = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    return Q, K, V


class GpaSpfaCsr:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"{IMPL_NAME} only services {PRECISIONS}; requested {precision!r}")
        self.precision = precision

    def prepare(self, workload, params: dict):
        _ensure_path()
        import torch

        w = workload
        if w.kernel != "sparse-attention-kernel":
            raise NotImplementedError(f"{IMPL_NAME} only services sparse-attention-kernel")

        B, H, S, d = w.B, w.H, w.S, w.d
        seed = params.get("seed", w.seed)
        np_dtype = {"fp16": np.float16, "fp32": np.float32}[self.precision]
        Q, K, V = _qkv_like_ml(B, H, S, d, seed, np_dtype)
        torch_dtype = getattr(torch, _TORCH_DTYPE_NAME[self.precision])

        # (B,H,S,d) -> B*H independent (S,d) slices, this kernel's own
        # single-headed/single-batched interface (see module docstring).
        q_slices = [torch.from_numpy(np.ascontiguousarray(Q[b, h])).to("cuda", torch_dtype)
                    for b in range(B) for h in range(H)]
        k_slices = [torch.from_numpy(np.ascontiguousarray(K[b, h])).to("cuda", torch_dtype)
                    for b in range(B) for h in range(H)]
        v_slices = [torch.from_numpy(np.ascontiguousarray(V[b, h])).to("cuda", torch_dtype)
                    for b in range(B) for h in range(H)]

        # ---- artifact's own mask-format conversion: the workload's fixed
        # (S,S) bool mask -> CSR (ARTIFACT_GUIDE rule 2 preprocessing,
        # timed once; shared across every (b,h) slice, exactly matching
        # this track's "same mask for every batch/head" convention).
        mask_t = torch.from_numpy(np.ascontiguousarray(w.mask.astype(np.float32))).to("cuda")
        csr = mask_t.to_sparse_csr()
        w_row_off = csr.crow_indices().to(torch.uint64).contiguous()
        w_col_ind = csr.col_indices().to(torch.uint32).contiguous()
        # W_val is never read by the kernel (see module docstring point 1);
        # ones is the clearest expression of "these are the nonzero
        # positions", cast to the run dtype since the C++ side expects
        # W_val's dtype to match Q/K/V's scalar_t for the templated kernel.
        w_val = torch.ones(csr.col_indices().shape[0], dtype=torch_dtype, device="cuda")

        outs = [torch.zeros((S, d), dtype=torch_dtype, device="cuda") for _ in range(B * H)]
        m_init = torch.full((S,), float("-inf"), dtype=torch_dtype, device="cuda")
        l_init = torch.zeros((S,), dtype=torch_dtype, device="cuda")

        # to_host() needs B/H to reshape the flat list of (S,d) slices back
        # into (B,H,S,d); stashed on self here since prepare()/run()/
        # to_host()/free() all run on the same instance within one
        # run_variant() call (matches the fused3s/rassm adapters' pattern).
        self._last_bh = (B, H)

        return {
            "q_slices": q_slices, "k_slices": k_slices, "v_slices": v_slices,
            "outs": outs, "m_init": m_init, "l_init": l_init,
            "w_row_off": w_row_off, "w_col_ind": w_col_ind, "w_val": w_val,
            "B": B, "H": H, "S": S, "d": d,
        }

    def run(self, h):
        import spfa_csr
        outs = h["outs"]
        for i in range(h["B"] * h["H"]):
            m = h["m_init"].clone()
            l = h["l_init"].clone()
            outs[i].zero_()
            spfa_csr.forward(
                h["q_slices"][i], h["k_slices"][i], h["v_slices"][i],
                h["w_row_off"], h["w_col_ind"], h["w_val"],
                m, l, outs[i],
                False,   # use_nan -- fully-masked rows (none exist here; the
                         # workload's mask always keeps the diagonal, see
                         # ml.py's _build_sparse_mask) get 0.0 rather than NaN
            )
        return outs

    def to_host(self, outs):
        import torch
        B, H = self._last_bh
        S, d = outs[0].shape
        stacked = torch.stack(outs, dim=0).reshape(B, H, S, d)
        return stacked.detach().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return GpaSpfaCsr(precision)
