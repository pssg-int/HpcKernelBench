"""
Adapter for STOF (PPoPP'26 "Accelerating Sparse Transformer Inference on
GPU", HeyDavid633/PPoPP26-pap161-AE). `PAPER_KEY = conf/ppopp/DaiDRYLL0CS26`.

STOF's real "Our Kernel" path (confirmed the one benchmarked in
`src/benchmk_attn_unified.py`, per the track's own survey.md) is a
CUTLASS/FlashAttention-family fused kernel (`src/ops/src/binding_attn.cpp` +
`binding_attn_cuda.cu`, compiled as the `binding_attn` torch extension) that
consumes an ARBITRARY binary mask converted into a full/part/inner-bitmap
block-sparse format via `src/util/masks.py::get_OuterTile_storage()` (also
unmodified) -- a fully general, lossless block encoding of any [B,N,N]
0/1 mask (each 64x64 outer tile is classified full/empty/mixed; mixed tiles
keep an exact 8x8-sub-tile bitmap), NOT restricted to STOF's own 5 named
mask families. This is exactly the mask-format-conversion ARTIFACT_GUIDE
rule 2 describes as legitimate preprocessing.

This adapter calls `binding_attn.forward(...)` -- the compiled extension's
actual C++ entry point -- DIRECTLY, rather than going through
`src/ops/package_op.py`'s `binding_attn_func`/`BindingAttnFunc` autograd
wrapper. Reason: `package_op.py` unconditionally does `import
block_attn_mask` at module level (STOF's OTHER, separate ~6300-line legacy
kernel extension, not exercised by `benchmk_attn_unified.py`'s actual timed
comparison and deliberately NOT built by this integration's build.sh -- see
build.sh's own header comment), so importing `package_op.py` would fail
without also building that unrelated extension. Calling `binding_attn.
forward()` directly is a MORE minimal wrap of the same real kernel entry
point (`package_op.py`'s own `_binding_attn_forward()` is a thin,
autograd-irrelevant pass-through to this exact call for a forward-only
inference benchmark -- no logic is skipped, see this adapter's `run()` for
the literal same argument list/order `_binding_attn_forward` uses, read
from unmodified source).

Precision/shape constraints, from `binding_attn.cpp`'s own `run_mha_fwd()`
(unmodified, read in full): `if (params.d != 64) AT_ERROR("Only head_dim=64
is supported")` and `if (params.is_bf16) AT_ERROR("Only fp16 is supported,
not bf16")` -- the compiled kernel is hard head_dim=64, fp16-only (matches
`binding_attn_cuda.cu`'s explicit-instantiation comment: only
`cutlass::half_t, 64, {true,false}` are compiled). `PRECISIONS = ["fp16"]`,
and `prepare()` raises `NotImplementedError` for any other `d`.

Mask/head-batching semantics, confirmed by reading `binding_attn.cpp`'s
`flashattn_binding_gpu()` signature: `full_row_ptr`/`part_row_ptr`/etc. carry
NO head dimension at all -- the SAME per-batch mask is applied to every
attention head, exactly matching this track's own `SparseAttentionWorkload`
convention (one (S,S) mask shared across B and H). `get_OuterTile_storage`
does expect one mask PER BATCH ELEMENT (its own loop is `for b in
range(batch_size): ...`), so this adapter tiles the workload's single fixed
mask across the B axis before conversion -- a faithful (not hacked)
preprocessing step, since our workload's own semantics ARE "the same mask
for every batch element".

Unlike vit-sparse's block-tiled artifact (bench/artifacts/
sparse-attention-kernel/vit-sparse/), STOF's kernel is a REAL fused
FlashAttention-style CUDA kernel with proper online-softmax bookkeeping
across ALL of a row's blocks within one kernel launch -- there is NO
single-block-containment restriction here; any mask this domain module can
build is a legitimate input.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "source")
SRC_DIR = os.path.join(SOURCE, "src")
OPS_SRC = os.path.join(SRC_DIR, "ops", "src")
# source/src/util/masks.py does `import matplotlib.pyplot as plt` at module
# level (unused by get_OuterTile_storage itself) -- vendored here by
# build.sh into a per-artifact pylibs/ dir (not the shared env); see
# build.sh's header comment.
PYLIBS = os.path.join(HERE, "pylibs")

KERNEL = "sparse-attention-kernel"
IMPL_NAME = "stof-binding-attn"
PAPER_KEY = "conf/ppopp/DaiDRYLL0CS26"
# binding_attn.cpp's run_mha_fwd() hard-errors on bf16 and on d!=64 -- see
# module docstring.
PRECISIONS = ["fp16"]

BLOCK_M = 64  # benchmk_attn_unified.py's own --block_m/--block_n CLI default
BLOCK_N = 64


def _ensure_paths():
    # PYLIBS is APPENDED, not prepended: `pip install --target=` pulled in
    # its own numpy (a matplotlib dependency) alongside matplotlib itself,
    # and that copy must never shadow the shared env's numpy (the one torch
    # was actually built/imported against) -- appending means `import numpy`
    # resolves to the already-on-sys.path shared site-packages copy first;
    # PYLIBS is only consulted for names not found there (matplotlib,
    # contourpy, cycler, fonttools, kiwisolver, pillow, pyparsing,
    # python-dateutil, six -- none of which the shared env provides).
    if PYLIBS not in sys.path:
        sys.path.append(PYLIBS)
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    if OPS_SRC not in sys.path:
        sys.path.insert(0, OPS_SRC)


def available() -> tuple[bool, str]:
    so_present = os.path.exists(os.path.join(OPS_SRC, "binding_attn.so"))
    if not so_present:
        return False, "binding_attn.so not built -- run build.sh"
    try:
        _ensure_paths()
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        import binding_attn  # noqa: F401
        from util.masks import get_OuterTile_storage  # noqa: F401
        return True, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg:
            msg += (" -- re-run with LD_PRELOAD=/usr/lib64/libstdc++.so.6 "
                    "set BEFORE the python process starts; see STATUS.md")
        return False, msg


def _qkv_like_ml(B, H, S, d, seed, np_dtype):
    """Bit-identical to kernelbench.domains.ml._qkv_sparse -- see
    vit-sparse/adapter.py's `_qkv_like_ml` for the same, independently-typed
    pattern used across this track's adapters."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    K = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    V = rng.standard_normal((B, H, S, d)).astype(np_dtype)
    return Q, K, V


class StofBindingAttn:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"{IMPL_NAME}: binding_attn.cpp hard-errors on non-fp16 "
                f"dtypes ('Only fp16 is supported, not bf16'); requested {precision!r}")
        self.precision = precision

    def prepare(self, workload, params: dict):
        _ensure_paths()
        import torch
        from util.masks import get_OuterTile_storage

        w = workload
        if w.kernel != "sparse-attention-kernel":
            raise NotImplementedError(f"{IMPL_NAME} only services sparse-attention-kernel")
        if w.d != 64:
            raise NotImplementedError(
                f"{IMPL_NAME}: binding_attn.cpp only instantiates head_dim=64 "
                f"(binding_attn_cuda.cu's explicit template instantiations); got d={w.d}")

        B, H, S, d = w.B, w.H, w.S, w.d
        seed = params.get("seed", w.seed)
        Q, K, V = _qkv_like_ml(B, H, S, d, seed, np.float16)
        torch_dtype = torch.float16

        # (B,H,S,d) -> (B,S,H,d), the layout binding_attn.forward expects
        # (confirmed from binding_attn.cpp's CHECK_SHAPE comments and
        # benchmk_attn_unified.py's own query = query1.permute(0,2,1,3)).
        q = torch.from_numpy(np.ascontiguousarray(Q.transpose(0, 2, 1, 3))).to("cuda", torch_dtype)
        k = torch.from_numpy(np.ascontiguousarray(K.transpose(0, 2, 1, 3))).to("cuda", torch_dtype)
        v = torch.from_numpy(np.ascontiguousarray(V.transpose(0, 2, 1, 3))).to("cuda", torch_dtype)

        # ---- artifact's own mask-format conversion (ARTIFACT_GUIDE rule 2:
        # timed as preprocessing here, not per-call cost). The workload's
        # single fixed (S,S) mask is tiled across B (get_OuterTile_storage's
        # own per-batch loop; see module docstring on why this is faithful,
        # not a hack, given our workload's "same mask for every batch
        # element" semantics).
        mask_np = np.broadcast_to(w.mask, (B, S, S)).astype(np.float16)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).to("cuda", torch_dtype)
        (_nnz_pct, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx,
         load_row_ptr, load_col_idx, inner_bitmaps) = get_OuterTile_storage(
            mask_t, BLOCK_M, BLOCK_N)

        causal = (w.pattern == "causal")
        softmax_scale = 1.0 / (d ** 0.5)

        out = torch.empty_like(q)

        return {
            "q": q, "k": k, "v": v, "out": out,
            "full_row_ptr": full_row_ptr, "full_col_idx": full_col_idx,
            "part_row_ptr": part_row_ptr, "part_col_idx": part_col_idx,
            "inner_bitmaps": inner_bitmaps,
            "load_row_ptr": load_row_ptr, "load_col_idx": load_col_idx,
            "causal": causal, "softmax_scale": softmax_scale,
        }

    def run(self, h):
        import binding_attn
        # Literal same call/arg order as ops/package_op.py's
        # _binding_attn_forward() (unmodified source, read in full) -- see
        # module docstring for why this adapter calls the compiled
        # extension directly instead of going through that Python wrapper.
        out, softmax_lse, S_dmask, rng_state = binding_attn.forward(
            h["q"], h["k"], h["v"],
            h["full_row_ptr"], h["full_col_idx"],
            h["part_row_ptr"], h["part_col_idx"], h["inner_bitmaps"],
            h["load_row_ptr"], h["load_col_idx"],
            None,               # out_ (optional preallocated output; let the kernel allocate)
            None,               # alibi_slopes_
            0.0,                # dropout_p
            h["softmax_scale"],
            h["causal"],
            -1, -1,             # window_size_left, window_size_right (no local windowing here --
                                 # the mask itself already encodes the sparsity pattern)
            0.0,                # softcap
            False,              # return_softmax
            None,               # generator
        )
        del softmax_lse, S_dmask, rng_state
        return out

    def to_host(self, out):
        import torch
        # binding_attn's own (B,S,H,d) -> this track's (B,H,S,d) convention.
        return out.detach().permute(0, 2, 1, 3).contiguous().to(
            "cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return StofBindingAttn(precision)
