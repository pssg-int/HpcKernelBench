"""
Adapter for PAT (Prefix-Aware Attention for LLM Decoding, ASPLOS'26,
flashserve/PAT). `PAPER_KEY = conf/asplos/YiZHYSWZZLL26`.

PAT is a CUTLASS-based decode-attention kernel: for a batch of paged-KV-cache
sequences it identifies shared prefixes and schedules them into dedicated
CTAs to cut redundant KV-cache reads. Its Python entry point,
`prefix_attn.prefix_attn_with_kvcache(q, k_cache_paged, v_cache_paged, tree,
softmax_scale, out)`, is the ONE call this adapter's run() makes -- not
PAT's own benchmark scripts (`benchmark/`, `test/test.py::tree_benchmark`),
per ARTIFACT_GUIDE rule 1. `tree` is a `PrefixTreeCPP` built from
`(seq_lens, block_table)` via `build_radix_tree` + `pack_schedule`; this
adapter's workload has NO shared prefixes across the batch (our
AttentionWorkload models independent decode queries), so the tree degenerates
to "one leaf per sequence" -- a legitimate, PAT-supported input, not a
degenerate/unsupported case (PAT's own `create_seq_group` + plain
`block_tables`/`seq_lens` path, used verbatim by `test/test.py`'s
`test_tree_attn_manual`, exercises exactly this).

Two things prepare() does beyond "call the kernel", both legitimate
preprocessing under ARTIFACT_GUIDE rule 2 (the artifact's own format
conversion, timed as preprocessing):

1. Builds PAT's paged KV-cache layout -- `k_cache_paged`/`v_cache_paged`
   shape `(num_blocks, block_size, Hkv, d)` plus a per-sequence
   `block_table` -- from the harness's flat `(B,Hkv,Sk,d)` K/V. Every
   sequence in our workload has the SAME Sk (no per-sequence length
   variation is modeled), so the simplest correct choice is used: each
   sequence gets its OWN contiguous block range (`num_blocks = B *
   ceil(Sk/block_size)`, sequence b's blocks are `[b*nblk, b*nblk+1, ...]`)
   -- no prefix sharing is invented since our workload has none.
   `block_size=16` matches `benchspecs/attention-kernel/spec.yaml`'s decode
   variant (`layout: "paged, block_size=16"`).
2. Builds the `SeqGroup`/`PrefixTreeCPP` scheduling metadata
   (`build_radix_tree` + `pack_schedule` + `kernel_info.to_gpu`) that PAT's
   kernel needs to know which CTA processes which (query, KV-block) pairs.

RNG discipline (the trap documented in
bench/artifacts/spmm/insum/STATUS.md's "Finding, not an Insum bug"
postmortem): Q/K/V are generated with the EXACT numpy `np.random.default_rng`
formula `kernelbench.domains.ml._qkv` uses -- same seed, same shape order,
same call order (Q then K then V) -- so this adapter's operands are bit-
identical to what the fp64 reference (and every other impl under test) uses.
This is intentionally re-typed here as a separate copy (not imported from
ml.py) rather than sharing a code object with the harness, matching the
precedent in fused3s/insum's adapters -- operand generation is meant to be
identical everywhere per ml.py's own docstring, but the reference's *compute
path* must stay independent of any implementation under test.
PAT's own `prefix_attn.generate_random_kv_cache` (torch.random.manual_seed +
torch.randn) is deliberately NOT used -- it draws a different bit sequence
from the same integer seed and would silently compare PAT's output against
operands nobody else in the harness sees.

Layout gotcha: PAT's q/out tensors are `(B, Sq, H, d)` -- head dim is axis 2,
NOT axis 1 like ml.py's `(B,H,Sq,d)` convention. Confirmed from
`prefix_attn/_prefix_attn.pyi`'s tensor comments and `test/test.py`'s
`q = torch.randn(len(seq_group), 1, nheads_q, head_dim, ...)`. `to_host()`
transposes axes 1,2 back before returning, so the harness compares like-for-
like against `reference_attention`'s `(B,H,Sq,d)` output.

Coverage: PAT's csrc/ only instantiates head_dim in {64, 128} (see
csrc/pat_fwd_split_hdim{64,128}_{fp16,bf16}_sm80.cu -- no hdim=16
instantiation exists), and only implements decode (Sq=1, unconditionally
causal over the full KV history -- see module docstring in ml.py's
_causal_mask on why decode causal masking removes nothing). prepare() raises
NotImplementedError for anything else (wrong head_dim, Sq!=1, non-causal
mask) -- the sanctioned "shape-constrained artifacts" behavior, not a bug.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "source")
PYLIBS = os.path.join(HERE, "pylibs")

KERNEL = "attention-kernel"
IMPL_NAME = "pat-decode-attn"
PAPER_KEY = "conf/asplos/YiZHYSWZZLL26"
# csrc/ instantiates both fp16 and bf16 sm80 kernels for hdim 64/128.
PRECISIONS = ["fp16", "bf16"]

BLOCK_SIZE = 16  # spec's own convention for the decode variant's paged layout

_NP_DTYPE = {"fp16": np.float16, "bf16": np.float32}  # numpy has no bf16
_TORCH_DTYPE = None  # resolved lazily (needs torch imported)


def _ensure_paths():
    if PYLIBS not in sys.path:
        sys.path.insert(0, PYLIBS)
    if SOURCE not in sys.path:
        sys.path.insert(0, SOURCE)


def available() -> tuple[bool, str]:
    so_present = False
    pkg_dir = os.path.join(SOURCE, "prefix_attn")
    if os.path.isdir(pkg_dir):
        so_present = any(f.startswith("_prefix_attn.cpython") and f.endswith(".so")
                          for f in os.listdir(pkg_dir))
    if not so_present:
        return False, "prefix_attn/_prefix_attn*.so not built -- run build.sh"
    try:
        _ensure_paths()
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        from prefix_attn import prefix_attn_with_kvcache, PrefixTreeCPP  # noqa: F401
        from prefix_attn.data_class import create_seq_group  # noqa: F401
        return True, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI_1.3.15" in str(e):
            # Same environment quirk as flashattention-t (see its STATUS.md
            # "Finding: a second, distinct libstdc++ ABI trap"): this venv's
            # python is a symlink to a NERSC-provided interpreter that
            # transitively loads an old bundled libstdc++ (missing
            # CXXABI_1.3.15) as soon as `torch` is imported anywhere in the
            # process; the extension (compiled with system g++-14) then
            # can't resolve that symbol against the already-resident old
            # copy. Fix is a process-startup env var, not fixable from
            # inside this already-running interpreter.
            msg += (" -- re-run with LD_PRELOAD=/usr/lib64/libstdc++.so.6 "
                    "set BEFORE the python process starts; see STATUS.md")
        return False, msg


def _qkv_like_ml(B, H, Hkv, Sq, Sk, d, seed, np_dtype):
    """
    Bit-identical to kernelbench.domains.ml._qkv: same default_rng seed, same
    shapes, same call order (Q, then K, then V) -- see module docstring's RNG
    discipline note.
    """
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, Sq, d)).astype(np_dtype)
    K = rng.standard_normal((B, Hkv, Sk, d)).astype(np_dtype)
    V = rng.standard_normal((B, Hkv, Sk, d)).astype(np_dtype)
    return Q, K, V


class PATDecodeAttn:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"{IMPL_NAME} only services precisions {PRECISIONS} "
                f"(fp16/bf16 tensor-core sm80 kernels); requested {precision!r}")
        self.precision = precision

    def _torch_dtype(self):
        import torch
        return {"fp16": torch.float16, "bf16": torch.bfloat16}[self.precision]

    def prepare(self, workload, params: dict):
        _ensure_paths()
        import torch
        from prefix_attn import PrefixTreeCPP
        from prefix_attn.data_class import create_seq_group

        w = workload
        if w.variant_kind != "decode" or w.Sq != 1:
            raise NotImplementedError(
                f"{IMPL_NAME} only implements decode (Sq=1); got "
                f"variant_kind={w.variant_kind!r}, Sq={w.Sq}")
        if w.mask != "causal":
            raise NotImplementedError(
                f"{IMPL_NAME}'s decode kernel is unconditionally causal over "
                f"the full KV history; got mask={w.mask!r}")
        if w.d not in (64, 128):
            raise NotImplementedError(
                f"{IMPL_NAME}: PAT's csrc/ only instantiates head_dim in "
                f"(64, 128) (pat_fwd_split_hdim{{64,128}}_*_sm80.cu); got d={w.d}")

        B, H, Hkv, d, Sk = w.B, w.H, w.Hkv, w.d, w.Sk
        seed = params.get("seed", w.seed)
        np_dtype = _NP_DTYPE[self.precision]

        Q, K, V = _qkv_like_ml(B, H, Hkv, 1, Sk, d, seed, np_dtype)

        # ---- artifact's own format conversion: flat (B,Hkv,Sk,d) K/V into
        # PAT's paged KV-cache layout (num_blocks, block_size, Hkv, d), one
        # contiguous block range per sequence -- see module docstring point 1.
        block_size = BLOCK_SIZE
        nblk = math.ceil(Sk / block_size)
        padded_len = nblk * block_size

        Kt = np.ascontiguousarray(K.transpose(0, 2, 1, 3))  # (B, Sk, Hkv, d)
        Vt = np.ascontiguousarray(V.transpose(0, 2, 1, 3))
        if padded_len != Sk:
            pad = ((0, 0), (0, padded_len - Sk), (0, 0), (0, 0))
            Kt = np.pad(Kt, pad)
            Vt = np.pad(Vt, pad)

        k_paged_np = Kt.reshape(B * nblk, block_size, Hkv, d)
        v_paged_np = Vt.reshape(B * nblk, block_size, Hkv, d)
        Qt = np.ascontiguousarray(Q.transpose(0, 2, 1, 3))  # (B, Sq=1, H, d) -- PAT's own layout

        torch_dtype = self._torch_dtype()

        def _to_cuda(arr):
            t = torch.from_numpy(arr)
            if t.dtype != torch_dtype:
                t = t.to(torch_dtype)
            return t.to("cuda")

        q = _to_cuda(Qt).contiguous()
        k_cache_paged = _to_cuda(k_paged_np).contiguous()
        v_cache_paged = _to_cuda(v_paged_np).contiguous()

        block_table = [[b * nblk + i for i in range(nblk)] for b in range(B)]
        seq_lens = [Sk] * B

        # ---- PAT's own scheduling metadata (SeqGroup / PrefixTreeCPP) --
        # no shared prefixes across the batch (our workload models none);
        # this is PAT's supported "independent sequences" path, exercised
        # verbatim by source/test/test.py::test_tree_attn_manual.
        create_seq_group(block_table, seq_lens, block_size)  # validates shapes; unused beyond that
        table_tensor = torch.tensor(block_table, dtype=torch.int32)  # CPU, per test.py's `table = block_table.cpu()`
        tree = PrefixTreeCPP(block_size)
        tree.build_radix_tree(seq_lens, table_tensor)
        tree.pack_schedule(None, H // Hkv, Hkv)
        tree.kernel_info.to_gpu(torch.device("cuda"))

        out = torch.empty_like(q)
        scale = 1.0 / math.sqrt(d)

        return {
            "q": q, "k_cache_paged": k_cache_paged, "v_cache_paged": v_cache_paged,
            "tree": tree, "scale": scale, "out": out,
        }

    def run(self, h):
        from prefix_attn import prefix_attn_with_kvcache
        prefix_attn_with_kvcache(
            q=h["q"], k_cache_paged=h["k_cache_paged"], v_cache_paged=h["v_cache_paged"],
            tree=h["tree"], softmax_scale=h["scale"], out=h["out"],
        )
        return h["out"]

    def to_host(self, out):
        # PAT's (B, Sq, H, d) -> ml.py reference's (B, H, Sq, d).
        import torch
        return out.detach().transpose(1, 2).contiguous().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return PATDecodeAttn(precision)
