"""
Marlin adapter for the quantized-gemm track.

Paper: "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large
Language Models", PPoPP'25. `PAPER_KEY = conf/ppopp/FrantarCCHA25`.
Artifact: https://github.com/IST-DASLab/marlin -- a highly optimized
FP16-activation x INT4-weight (symmetric, per-group) CUDA GEMM kernel.
This track's `benchspecs/quantized-gemm/spec.yaml` recommended_subset shapes
are copied VERBATIM from marlin's own `bench.py` (per
`kernel_centrality.json`'s own note: "its exact per-layer LLaMA/
Falcon-180B shapes are copied verbatim into the spec's recommended_subset"),
so this is close to marlin's native regime, not a bent-to-fit one.

REUSE, NOT A REBUILD (per task brief): this artifact is ALREADY built for
the gemv track at `bench/artifacts/gemv/marlin/` (build.sh compiled the
real `marlin_cuda` CUDA extension into that track's `vendor/`). Marlin's
compiled kernel has no notion of "gemv" vs "quantized-gemm" -- it is the
identical `.so`; only the shape mapping each track's adapter performs
differs. This adapter's own `build.sh` therefore does not compile
anything: it symlinks `source/` to the gemv track's clone (same commit)
and imports `marlin`/`marlin_cuda` directly from
`../../gemv/marlin/vendor/` (no local `vendor/` here at all).

SHAPE MAPPING -- markedly SIMPLER than gemv's, no transpose trick needed:
this track's own workload convention (`ml.py`'s `QuantGemmWorkload`) is
already `C[M,N] = A[M,K] @ dequantize(W[K,N])`, i.e. A is the activation
and W is the weight -- exactly marlin's own `C[m,n] = A_act[m,k] @
B_weight[k,n]` convention, with M<->m (batch), K<->k (contraction,
marlin's "infeatures"), N<->n (marlin's "outfeatures"). Unlike the gemv
adapter (whose domain models y=A@x with A as an implicit *weight*, forcing
a role-swap onto marlin's m=1 batch slot), here M is marlin's own batch
dimension `m` directly -- this track's own decode (M in {1..32}) and
batched (M in {64..4096}) sweeps map onto marlin's `m` with NO adapter-side
reinterpretation, and (per `marlin_cuda_kernel.cu`'s `marlin_cuda()`) `m`
has NO tile-alignment constraint at all -- the kernel internally chunks any
`m` into blocks of 16 rows (see the `for (i = 0; i < tot_m_blocks; i += 4)`
loop and its own internal padding for the last partial block), so **no
M-padding is needed here** (unlike gemv's M-padding for its unrelated
outfeatures role).

FAIRNESS: the SAME `w.quantize()` codes/scale (`ml.py`'s
`QuantGemmWorkload`, shared with `reference_qgemm` and every other impl in
this track) are fed to marlin's own `Layer.pack(linear, scales)`, which
takes a DEQUANTIZED fp16 weight + matching scale and internally re-derives
`round(w/s)` -- lossless here since `w = codes*scale` is already exactly
integral divided by `s` (identical technique to `gemv/marlin/adapter.py`
and `quantized-gemm/tilus/adapter.py`'s docstrings, and to marlin's own
`test.py::gen_quant4` convention for building ITS OWN reference).

HARD CONSTRAINTS (marlin's `Layer.__init__`, unmodified, not relaxed):
`infeatures % 128 == 0`, `outfeatures % 256 == 0`, `groupsize in {-1, 128}`.
Mapped onto this track's fields:
  - K (marlin's infeatures) must be zero-padded up to a multiple of 128.
    Every real `recommended_subset` K value (4096, 5120, 6144, 8192, 11008,
    13824, 14848, 24576, 28672, 74240) is ALREADY a multiple of 128, so no
    padding ever happens on the real registry -- only the K=96 smoke shape
    (`smoke-qgemm-w4-percol-decode`) needs it.
  - N (marlin's outfeatures) must be zero-padded up to a multiple of 256.
    Every real `recommended_subset` N value is ALSO already a multiple of
    256 (checked exhaustively against `ml.py`'s `_QGEMM_SHAPES`) -- padding
    only fires for the N=384 smoke shape (`smoke-qgemm-w4-g128-decode`).
  - `group_size_eff` must equal either 128 (marlin's own "grouped" mode,
    `groupsize=128`) or K itself, i.e. the domain's per-column fallback
    (`group_size<=0` or doesn't divide K -- `QuantGemmWorkload.
    group_size_eff`'s own documented fallback), mapped to marlin's
    "ungrouped" mode `groupsize=-1` (one scale per output column, spanning
    the padded K). Any OTHER effective group size (e.g. the
    `smoke-qgemm-w3-g64-batched` shape's `group_size_eff=64`) is not
    representable by marlin's 2-value `{-1, 128}` groupsize enum --
    `NotImplementedError`, not a silent approximation.
  - bits must be exactly 4 (marlin is hard-coded INT4, `maxq = 2**4-1` in
    `Layer.pack`) -- every OTHER bit-width `NotImplementedError`s.
Padding is EXACT for the correctness gate (extra K rows are zero-weight,
contribute 0 to the dot product regardless of the corresponding activation
value; extra N columns are simply sliced off `to_host()` before compare) --
same reasoning `gemv/marlin/adapter.py`'s "SHAPE-COMPATIBILITY PADDING"
section documents, and the SAME caveat applies to any future timed run: the
kernel does slightly more tile work than the workload's own M*K*bytes
byte-accounting reflects at the two padded smoke shapes (never at any real
`recommended_subset` shape, where K and N are always tile-aligned already).

PRECISION: `PRECISIONS = ["fp16"]`, matching `DEFAULT_PRECISION
["quantized-gemm"] == "fp16"` already -- no explicit `--precision` flag
needed on the CLI (unlike the gemv adapter, whose domain defaults to fp64).
"""

from __future__ import annotations

import os

import numpy as np

KERNEL = "quantized-gemm"
IMPL_NAME = "marlin-w4a16-gemm"
PAPER_KEY = "conf/ppopp/FrantarCCHA25"
PRECISIONS = ["fp16"]

HERE = os.path.dirname(os.path.abspath(__file__))
_GEMV_MARLIN = os.path.join(HERE, "..", "..", "gemv", "marlin")
_VENDOR = os.path.join(_GEMV_MARLIN, "vendor")   # reused, not local

_BITS = 4
_IN_TILE = 128     # Layer.__init__: infeatures % 128 == 0
_OUT_TILE = 256    # Layer.__init__: outfeatures % 256 == 0
_GROUPSIZE = 128   # Layer.__init__: groupsize must be -1 or literally 128


def available() -> tuple[bool, str]:
    if not os.path.isdir(os.path.join(_VENDOR, "marlin")):
        return False, (f"reused vendor/ not found at {_VENDOR} -- run "
                        f"../../gemv/marlin/build.sh (this track's own "
                        f"build.sh only symlinks source/, it does not "
                        f"compile) then this artifact's build.sh")
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
        if "CXXABI" in msg or "GLIBCXX" in msg or "__cxa_call_terminate" in msg:
            msg = ("libstdc++ ABI mismatch -- run with "
                   "LD_PRELOAD=/usr/lib64/libstdc++.so.6 (see STATUS.md)")
        return False, msg
    return True, ""


def create(precision: str):
    return MarlinQuantGemm(precision)


def _activation(w, params, dtype) -> np.ndarray:
    """Identical recipe to kernelbench.domains.ml._make_activation (private
    to that module; replicated verbatim here, same convention tilus's/
    qfactory's adapters already use for this exact helper)."""
    seed = params.get("seed", w.seed)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((w.M, w.K)).astype(dtype)


class MarlinQuantGemm:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp16 activation/output "
                f"(marlin's kernel is FP16-activation x INT4-weight only); "
                f"requested {precision!r}")
        self.precision = precision
        self._N_true = None

    def prepare(self, workload, params: dict):
        import sys
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        import torch
        import marlin as mlib

        w = workload
        if w.bits != _BITS:
            raise NotImplementedError(
                f"{IMPL_NAME}: bits={w.bits} unsupported -- marlin is "
                f"hard-coded INT4 only (Layer.pack's maxq=2**4-1)")

        gs_eff = w.group_size_eff
        if gs_eff == _GROUPSIZE:
            marlin_groupsize = _GROUPSIZE
            K_pad = w.K   # group_size_eff==128 implies K % 128 == 0 already
        elif gs_eff == w.K:
            marlin_groupsize = -1   # marlin's "ungrouped", one scale/column
            K_pad = ((w.K + _IN_TILE - 1) // _IN_TILE) * _IN_TILE
        else:
            raise NotImplementedError(
                f"{IMPL_NAME}: group_size_eff={gs_eff} (K={w.K}) is neither "
                f"128 nor per-column (K itself) -- marlin's Layer only "
                f"supports groupsize in {{-1, 128}}, see module docstring")

        N_pad = ((w.N + _OUT_TILE - 1) // _OUT_TILE) * _OUT_TILE

        # --- SHARED quantized values (the fairness point, see module
        # docstring): codes/scale come from the workload's own quantize(),
        # identical to what reference_qgemm and every other impl consume.
        codes, scale, group_idx = w.quantize()
        W_dequant = codes.astype(np.float64) * scale[group_idx]   # (K, N)

        W_pad = np.zeros((K_pad, N_pad), dtype=np.float64)
        W_pad[:w.K, :w.N] = W_dequant
        # linear.weight has shape (outfeatures, infeatures) = (N_pad, K_pad)
        # per torch.nn.Linear's own convention; Layer.pack() internally
        # does linear.weight.data.t() to recover the (K_pad, N_pad) layout
        # this track's W already uses -- see marlin/__init__.py::Layer.pack.
        linear = torch.nn.Linear(K_pad, N_pad, bias=False)
        linear.weight.data = torch.as_tensor(
            W_pad.T, dtype=torch.float16, device="cuda").contiguous()

        n_groups_layer = K_pad // (marlin_groupsize if marlin_groupsize != -1 else K_pad)
        assert n_groups_layer == w.n_groups, (
            f"{IMPL_NAME}: group-count mismatch (workload {w.n_groups} vs "
            f"marlin layer {n_groups_layer}) -- see module docstring assert")
        # Layer.pack()'s `scales` argument has shape (outfeatures, groups) --
        # the TRANSPOSE of w.quantize()'s own (n_groups, N) scale, see the
        # gemv adapter's identical transpose for the analogous reason.
        scale_full = np.ones((N_pad, n_groups_layer), dtype=np.float32)
        scale_full[:w.N, :] = scale.T
        s_fp16 = torch.as_tensor(scale_full.astype(np.float16), device="cuda")

        layer = mlib.Layer(K_pad, N_pad, groupsize=marlin_groupsize).cuda()
        layer.pack(linear, s_fp16)

        A_np = _activation(w, params, np.float16).astype(np.float64)
        A_pad = np.zeros((w.M, K_pad), dtype=np.float64)
        A_pad[:, :w.K] = A_np
        A_act = torch.as_tensor(A_pad, dtype=torch.float16, device="cuda")

        self._N_true = w.N
        return {"layer": layer, "A": A_act}

    def run(self, h):
        return h["layer"](h["A"])

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
