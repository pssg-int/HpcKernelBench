"""
ML-inference kernels: convolution (CNN forward layers), attention-kernel
(fused scaled-dot-product attention, prefill and KV-cache decode), and
sparse-attention-kernel (attention restricted to a 0/1 sparsity mask --
sliding-window/local, Longformer, BigBird, dilated, and block-local/regional
patterns, per benchspecs/sparse-attention-kernel/spec.yaml's
sparse-attn-structured-mask-kernel-fp16 variant ONLY). The spec's OTHER
variant, sparse-attn-graph-adjacency-kernel-fp16 (Fused3S's own claim: fused
SDDMM+softmax+SpMM over a REAL graph's adjacency, not a swept ratio), is
already integrated as `fused3s-1tb1tcb-sddmm` under the sddmm track
(bench/artifacts/sddmm/fused3s/) and is NOT re-implemented here -- see
bench/artifacts/sparse-attention-kernel/*/STATUS.md for the cross-track
pointer. This module owns the structured/swept-ratio variant only.

Also: quantized-gemm (LLM-inference-shaped fused-dequantize GEMM: fp16
activation x low-bit weight, per benchspecs/quantized-gemm/spec.yaml's
qgemm-w4a16-decode-kernel / qgemm-w4a16-batched-kernel variants -- see this
module's own quantized-gemm section below for exactly what is and is not
wired, and why).

Follows sparse.py's shape (WORKLOADS / COST / REFERENCE / IMPLS), and
dense.py's kernel-aware `smoke_workloads(kernel=...)` convention (this module
owns four kernels with MUTUALLY INCOMPATIBLE workload types -- a conv layer
shape is nothing like an attention (B,H,S,d) shape is nothing like a masked
sparse-attention (B,H,S,d)+mask shape is nothing like a quantized-gemm
(M,K,N,bits,group_size) shape -- exactly the situation that convention was
built for).

  WORKLOADS   -- ConvWorkload: a single NCHW conv layer shape (Cout, Hout,
                 Wout, Cin, Kh, Kw, stride, groups), transcribed verbatim
                 from benchspecs/convolution/spec.yaml's 30-instance
                 layer_shape_suite. AttentionWorkload: a (B,H,Hkv,d,Sq,Sk,
                 mask) shape, covering both the prefill (Sq==Sk) and decode
                 (Sq=1, always-causal, GQA-heavy) variants with the SAME
                 dataclass and cost/reference/impl code, since decode is
                 mathematically just attention with Sq=1 -- not a separate
                 algorithm.
  COST        -- convolution: flops_ref = 2*N*Cout*(Cin/groups)*Kh*Kw*Hout*
                 Wout, the spec's ALGORITHM-INVARIANT formula (computed from
                 the direct-conv op count regardless of which algorithm class
                 actually ran, so Winograd/FFT-conv's legitimately-smaller
                 executed-op count is never what gets reported -- see
                 _cost_convolution). attention-kernel: the spec's own
                 causal-aware prefill formula and un-causal decode formula,
                 picked by shape -- see _cost_attention for the exact
                 convention and why "GFLOP/s" (not the spec's own "TFLOP/s")
                 is registered here.
  REFERENCE   -- fp64, independently coded from every CPU impl below (no
                 shared code path with either). Operands are generated AT THE
                 RUN'S OWN PRECISION first, then widened to fp64 -- same
                 discipline dense.py's reference_gemm uses, so the gate
                 compares "same rounded inputs, fp64 accumulation" against
                 "same rounded inputs, run-precision accumulation", not
                 accidentally conflating operand-rounding error with the
                 kernel's own arithmetic error. Both references return
                 (result, scale) for harness.py's cancellation-robust
                 max_scaled_err gate; see reference_conv/reference_attention
                 for what `scale` is in each case.
  IMPLS       -- convolution: numpy-im2col-conv (implicit-GEMM) vs.
                 scipy-direct-conv (explicit sliding-window correlation) --
                 two genuinely different algorithm classes, matching the
                 spec's mandatory algorithm_class disclosure. attention-
                 kernel: numpy-attention (naive, materializes the full
                 (Sq,Sk) score matrix) vs. numpy-flash-attention (tiled,
                 online-softmax, NEVER materializes (Sq,Sk)) -- the exact
                 contrast this benchmark track exists to measure.

sparse-attention-kernel additions (WORKLOADS/COST/REFERENCE/IMPLS, same
shape as above):

  WORKLOAD    -- SparseAttentionWorkload: (B,H,d,S) + a named `pattern`
                 (causal/sliding_window/longformer_local_global/
                 bigbird_local_global_random/dilated_1d/block_local/
                 coo_arbitrary) + `pattern_params`. The mask itself is
                 DETERMINISTIC WORKLOAD DATA -- built once (seeded) by
                 `_build_sparse_mask`, lazily cached on `.mask`, and
                 reported in `describe()` with a sha256 checksum (of the
                 packed bit array) plus the MEASURED density (nnz/S^2),
                 since the spec's own `measured_sparsity_reported` clause
                 says no named pattern lands exactly on its nominal Sf
                 target -- see SparseAttentionWorkload's docstring for why
                 sharing this mask ARRAY across the reference and every
                 impl is correct (it is workload data, like a real graph's
                 adjacency matrix or this module's own `_qkv`-generated
                 Q/K/V), while the mask APPLICATION code (the np.where step
                 inside each softmax pipeline) is independently written in
                 reference_sparse_attention vs. NumpySparseAttention -- no
                 shared "_apply_mask" helper, per DOMAIN_GUIDE's
                 reference-independence ruling.
  COST        -- spec's structured-mask variant metric.primary, literal:
                 flops = 2*2*B*H*nnz(mask)*d (effective/nnz-based, NOT
                 S^2-based) -- see _cost_sparse_attention's docstring for
                 why the spec's mandatory secondary dense-equivalent number
                 is reported as a derivable-by-hand quantity (measured
                 density is always in describe()) rather than invented into
                 a metrics field the harness's single-(flops,bytes)-per-
                 kernel cost-rule contract has no slot for.
  REFERENCE   -- fp64, masked softmax attention; mask application inlined
                 independently of NumpySparseAttention (see above). scale =
                 P@|V|, the identical convention reference_attention uses,
                 independently re-typed here (not calling or importing
                 reference_attention) per this track's instruction to
                 inline rather than share the dense reference's helper.
  IMPLS       -- ONE numpy masked-attention impl (numpy-masked-attention),
                 not a naive/flash pair -- this track's interesting
                 contrast is the mask's structure and the paper artifacts'
                 native sparse formats (bench/artifacts/
                 sparse-attention-kernel/), not a CPU tiling algorithm.

PLANNED, and why each stays there rather than being stubbed in:

  * gnn-aggregation -- cross-listed here (it is the GNN-adjacent ML kernel
    in this track) but actually OWNED and tracked by sparse.py's own
    PLANNED list (domains/__init__.py routes it to "sparse": neighbor
    aggregation is a segment-reduce / SpMM-shaped sparse pattern, not a
    dense-tensor one). Listed here too purely as a pointer so a reader of
    this module isn't left wondering where it went -- sparse.py is the
    actual owner and the actual place to implement it.
  * winograd -- had NO domain owner at all before this change (an orphan
    OWNER-dict gap, fixed alongside this module: domains/__init__.py now
    routes "winograd" -> "ml", since Winograd convolution is a
    convolution-algorithm variant and belongs next to `convolution` here).
    Not implemented: it needs its own spec-driven shape suite and a
    from-scratch F(m,r) transform, genuinely separate work from either
    kernel above, not a quick add-on.

quantized-gemm's KERNEL-LEVEL scope, and what is deliberately NOT wired:
benchspecs/quantized-gemm/spec.yaml makes `qgemm-model-accuracy` -- a
perplexity-DELTA measurement on a GPTQ-calibrated Llama-2-7B against
WikiText2 -- a MANDATORY companion to every reported speed number ("only 2
of 13 surveyed papers treat this as first-class... the task's own
instructions require accuracy alongside speed"). That needs a multi-GB real
LLM checkpoint, a calibration pass and a full-document perplexity eval --
genuinely out of scope for a login-node kernel-benchmark integration pass
(no model-serving stack, no calibration corpus, no eval harness here). Per
this integration's explicit brief, `qgemm-model-accuracy` is NOT wired: no
Workload, no reference, no CPU impl exists for it, and `--list` still
describes its protocol text (spec.py parses every variant regardless of
which ones a domain module implements numerically) but nothing in this
module can execute it. Any report built from this module's quantized-gemm
results MUST NOT be presented as if it came with its spec-mandated accuracy
companion -- that gap applies to EVERY result this module produces, not a
caveat specific to one run. `qgemm-w8a8-mx` (both-operand FP8/INT8
quantization, a structurally distinct arithmetic regime from weight-only
W4A16) is also not modeled: QuantGemmWorkload only quantizes the WEIGHT
operand (activation stays fp16/bf16, per this integration's brief: "weights
quantized to the spec's headline bit-width/group-size, activations fp16").

All four CPU_IMPLS families are pure numpy/scipy, matching this track's CPU
verification requirement (torch is available on this machine and is a
legitimate CPU baseline too, but is deliberately not used for the CPU impls
here so the numpy-vs-scipy / naive-vs-flash / masked-vs-dense / dequant-cast
contrasts stay the point). cuda_impls() wires torch GPU library baselines
(F.conv2d / F.scaled_dot_product_attention, the latter reused with an
explicit boolean attn_mask for sparse-attention-kernel; a dequantize-then-
cuBLAS-GEMM baseline for quantized-gemm, the "naive/explicit baseline every
fused kernel is compared against" per the spec's own `operation` field) but
is never instantiated/run here -- the login node's GPU is shared (see
DOMAIN_GUIDE.md). The REAL GPU competitors for sparse-attention-kernel are
the paper artifacts under bench/artifacts/sparse-attention-kernel/ (native
sparse formats: ILP-scheduled adjacency tiles, fused mask-aware kernels,
COO/CSR explicit kernels); for quantized-gemm they are the paper artifacts
under bench/artifacts/quantized-gemm/ (fused-dequant kernels) -- not these
torch fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import workload
from ..harness import Timer

KERNELS = ["convolution", "attention-kernel", "sparse-attention-kernel", "quantized-gemm"]
PLANNED = ["gnn-aggregation", "winograd"]   # see module docstring

ITEMSIZE = {"fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2, "tf32": 4, "int8": 1}
# numpy has no native bf16/tf32 dtype; both alias to fp32 for the CPU
# reference/impl path here -- same convention dense.py uses for tf32 (neither
# has meaning without real tensor-core hardware).
_NP_DTYPE = {"fp64": np.float64, "fp32": np.float32, "fp16": np.float16,
             "bf16": np.float32, "tf32": np.float32}


def _dtype(precision: str):
    return _NP_DTYPE.get(precision, np.float32)


# ===================================================================== conv
def _infer_in_dim(out_dim: int, k: int, stride: int, pad: int) -> int:
    """
    Solve Hin/Win from the direct-conv output-size formula
    `Hout = floor((Hin + 2*pad - K)/stride) + 1` for the exact Hin that makes
    the floor divide evenly: `Hin = (Hout-1)*stride + K - 2*pad`. The spec's
    layer_shape_suite states only Cout/Hout/Wout/Cin/Kh/Kw/stride/groups (no
    Hin/Win) -- this is a genuine degree of freedom (several Hin/pad pairs
    reproduce the same Hout), and this choice (paired with the pad=(K-1)//2
    convention ConvWorkload uses below) reproduces the exact upstream
    activation size real ResNet/VGG/MobileNet/YOLO stages use: unchanged at
    stride 1, halved (ResNet-shortcut-consistent) at stride 2 -- checked by
    hand against all 30 instances in LAYER_SHAPE_SUITE. flops_ref (the
    spec's algorithm-invariant cost, see _cost_convolution) is computed from
    Hout/Wout directly and never depends on Hin/Win, so this choice affects
    only what gets multiplied, never what gets reported as the work count.
    """
    return (out_dim - 1) * stride + k - 2 * pad


@dataclass
class ConvWorkload:
    """
    Forward 2D conv (NCHW): Y = conv2d(X, W, stride, pad, groups), per
    benchspecs/convolution/spec.yaml's `operation` field. Padding convention
    owned by this module (the spec doesn't give one): pad = (K-1)//2 per
    axis, applied independently to H and W -- see _infer_in_dim for why this
    is a safe, documented choice rather than a guess.
    """

    name: str
    kernel: str = "convolution"
    variant: str = ""
    Cin: int = 0
    Cout: int = 0
    Hout: int = 0
    Wout: int = 0
    Kh: int = 3
    Kw: int = 3
    stride: int = 1
    groups: int = 1
    N: int = 1
    precision: str = "fp32"
    seed: int = 42
    source: str = "layer_shape_suite (benchspecs/convolution/spec.yaml)"

    pad_h: int = field(init=False)
    pad_w: int = field(init=False)
    Hin: int = field(init=False)
    Win: int = field(init=False)

    def __post_init__(self):
        if self.Cin % self.groups != 0 or self.Cout % self.groups != 0:
            raise ValueError(
                f"conv {self.name!r}: Cin={self.Cin}/Cout={self.Cout} not "
                f"both divisible by groups={self.groups}")
        self.pad_h = (self.Kh - 1) // 2
        self.pad_w = (self.Kw - 1) // 2
        self.Hin = _infer_in_dim(self.Hout, self.Kh, self.stride, self.pad_h)
        self.Win = _infer_in_dim(self.Wout, self.Kw, self.stride, self.pad_w)

    @property
    def cin_per_group(self) -> int:
        return self.Cin // self.groups

    @property
    def cout_per_group(self) -> int:
        return self.Cout // self.groups

    def describe(self) -> dict:
        if self.groups == 1:
            g_note = "standard (groups=1)"
        elif self.groups == self.Cin == self.Cout:
            g_note = "depthwise (groups==Cin==Cout)"
        else:
            g_note = "grouped"
        return {
            "name": self.name, "source": self.source, "kernel": self.kernel,
            "spec_variant": self.variant,
            "N": self.N, "Cin": self.Cin, "Cout": self.Cout,
            "Hin": self.Hin, "Win": self.Win, "Hout": self.Hout, "Wout": self.Wout,
            "Kh": self.Kh, "Kw": self.Kw, "stride": self.stride,
            "pad_h": self.pad_h, "pad_w": self.pad_w, "groups": self.groups,
            "groups_note": g_note,
            "precision": self.precision, "seed": self.seed,
            "padding_convention": (
                "pad=(K-1)//2 per axis; Hin/Win derived from Hout/stride/pad "
                "by inverting the direct-conv output formula with zero "
                "floor-loss -- see _infer_in_dim. flops_ref (the primary "
                "metric) is computed from Hout/Wout directly and does not "
                "depend on this choice."),
        }


def _make_conv_operands(w: ConvWorkload, params: dict, N: int, dtype):
    """U(-1,1) activations and weights. Same seed -> same operands for every
    impl AND the reference (sparse.py's dense-operand generation convention);
    `N` is read from params so a caller can sweep batch independently of the
    workload's own default."""
    seed = params.get("seed", w.seed)
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(N, w.Cin, w.Hin, w.Win)).astype(dtype)
    W = rng.uniform(-1.0, 1.0, size=(w.Cout, w.cin_per_group, w.Kh, w.Kw)).astype(dtype)
    return X, W


def _pad_input(X: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    if pad_h == 0 and pad_w == 0:
        return np.ascontiguousarray(X)
    return np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")


# ResNet-18-style (ASPLOS_artifact TileLoopGenerator/resnet.txt), YOLO/Darknet
# (yolo.txt, subset), MobileNet-depthwise (deepwise.txt), VGG16 (Odyssey
# conv_all.json) -- transcribed verbatim from benchspecs/convolution/spec.yaml
# layer_shape_suite.instances (30 total).
_CONV_LAYER_SHAPES = [
    dict(name="resnet18_1", Cout=64, Hout=56, Wout=56, Cin=64, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="resnet18_2", Cout=64, Hout=56, Wout=56, Cin=64, Kh=1, Kw=1, stride=1, groups=1),
    dict(name="resnet18_3", Cout=128, Hout=28, Wout=28, Cin=64, Kh=3, Kw=3, stride=2, groups=1),
    dict(name="resnet18_4", Cout=128, Hout=28, Wout=28, Cin=64, Kh=1, Kw=1, stride=2, groups=1),
    dict(name="resnet18_5", Cout=128, Hout=28, Wout=28, Cin=128, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="resnet18_6", Cout=256, Hout=14, Wout=14, Cin=128, Kh=3, Kw=3, stride=2, groups=1),
    dict(name="resnet18_7", Cout=256, Hout=14, Wout=14, Cin=128, Kh=1, Kw=1, stride=2, groups=1),
    dict(name="resnet18_8", Cout=256, Hout=14, Wout=14, Cin=256, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="resnet18_9", Cout=512, Hout=7, Wout=7, Cin=256, Kh=3, Kw=3, stride=2, groups=1),
    dict(name="resnet18_10", Cout=512, Hout=7, Wout=7, Cin=256, Kh=1, Kw=1, stride=2, groups=1),
    dict(name="resnet18_11", Cout=512, Hout=7, Wout=7, Cin=512, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="yolo_1", Cout=32, Hout=544, Wout=544, Cin=3, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="yolo_3", Cout=128, Hout=136, Wout=136, Cin=64, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="yolo_5", Cout=256, Hout=68, Wout=68, Cin=128, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="yolo_7", Cout=512, Hout=34, Wout=34, Cin=256, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="yolo_9", Cout=1024, Hout=17, Wout=17, Cin=512, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="yolo_11_head_stress", Cout=28272, Hout=17, Wout=17, Cin=1024, Kh=1, Kw=1, stride=1, groups=1),
    dict(name="mbnet_dw_1", Cout=32, Hout=112, Wout=112, Cin=32, Kh=3, Kw=3, stride=1, groups=32),
    dict(name="mbnet_dw_2", Cout=64, Hout=56, Wout=56, Cin=64, Kh=3, Kw=3, stride=2, groups=64),
    dict(name="mbnet_dw_3", Cout=128, Hout=56, Wout=56, Cin=128, Kh=3, Kw=3, stride=1, groups=128),
    dict(name="mbnet_dw_4", Cout=128, Hout=28, Wout=28, Cin=128, Kh=3, Kw=3, stride=2, groups=128),
    dict(name="mbnet_dw_5", Cout=256, Hout=28, Wout=28, Cin=256, Kh=3, Kw=3, stride=1, groups=256),
    dict(name="mbnet_dw_6", Cout=256, Hout=14, Wout=14, Cin=256, Kh=3, Kw=3, stride=2, groups=256),
    dict(name="mbnet_dw_7", Cout=512, Hout=14, Wout=14, Cin=512, Kh=3, Kw=3, stride=1, groups=512),
    dict(name="mbnet_dw_8", Cout=512, Hout=7, Wout=7, Cin=512, Kh=3, Kw=3, stride=2, groups=512),
    dict(name="mbnet_dw_9", Cout=1024, Hout=7, Wout=7, Cin=1024, Kh=3, Kw=3, stride=1, groups=1024),
    dict(name="vgg16_1", Cout=128, Hout=112, Wout=112, Cin=64, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="vgg16_2", Cout=512, Hout=28, Wout=28, Cin=256, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="vgg16_3", Cout=64, Hout=224, Wout=224, Cin=64, Kh=3, Kw=3, stride=1, groups=1),
    dict(name="vgg16_4", Cout=512, Hout=14, Wout=14, Cin=512, Kh=3, Kw=3, stride=1, groups=1),
]


def _build_conv_registry() -> dict[str, ConvWorkload]:
    return {kw["name"]: ConvWorkload(N=1, variant="conv-dense-kernel-fp32", **kw)
            for kw in _CONV_LAYER_SHAPES}


def _smoke_convolution() -> list[ConvWorkload]:
    return [
        ConvWorkload(name="smoke-conv-3x3", variant="smoke",
                     N=1, Cin=4, Cout=8, Hout=12, Wout=12, Kh=3, Kw=3, stride=1, groups=1),
        ConvWorkload(name="smoke-conv-1x1", variant="smoke",
                     N=1, Cin=8, Cout=8, Hout=8, Wout=8, Kh=1, Kw=1, stride=1, groups=1),
        ConvWorkload(name="smoke-conv-depthwise-s2", variant="smoke",
                     N=1, Cin=6, Cout=6, Hout=10, Wout=10, Kh=3, Kw=3, stride=2, groups=6),
    ]


# ----------------------------------------------------------------- cost rule
def _cost_convolution(w: ConvWorkload, params: dict):
    """
    flops_ref = 2*N*Cout*(Cin/groups)*Kh*Kw*Hout*Wout -- spec.yaml's
    ALGORITHM-INVARIANT formula, computed from the DIRECT-convolution op
    count regardless of which algorithm class (direct/im2col/Winograd/FFT)
    actually ran, so a reduced-op algorithm's GFLOP/s legitimately reads
    above its own executed-op rate (spec's explicit, intended behavior --
    see benchspecs/convolution/spec.yaml's `operation` field and
    conv-dense-kernel-fp32's metric.secondary for the "algorithm-effective"
    number this module does NOT compute: neither CPU impl here is
    Winograd/FFT, so their algorithm-effective count equals flops_ref
    exactly, and there is nothing to disclose beyond this primary number).
    """
    N = int(params.get("N", w.N))
    flops = 2 * N * w.Cout * w.cin_per_group * w.Kh * w.Kw * w.Hout * w.Wout
    itemsize = ITEMSIZE[params.get("precision", w.precision)]
    x_bytes = N * w.Cin * w.Hin * w.Win * itemsize
    w_bytes = w.Cout * w.cin_per_group * w.Kh * w.Kw * itemsize
    y_bytes = N * w.Cout * w.Hout * w.Wout * itemsize
    return int(flops), int(x_bytes + w_bytes + y_bytes)


workload.register_cost("convolution", _cost_convolution, "GFLOP/s")


# ----------------------------------------------------------------- reference
def _direct_conv_fp64(X: np.ndarray, W: np.ndarray, w: ConvWorkload) -> np.ndarray:
    """Explicit loop over the Kh*Kw taps (<=9 for every LAYER_SHAPE_SUITE
    instance), each tap fully vectorized over N/channels/spatial position via
    a strided slice + einsum -- the same "small bounded offset loop, large
    vectorized body" idiom stencil.py's `_sweep` uses for its own independent
    fp64 reference."""
    N = X.shape[0]
    Xp = _pad_input(X, w.pad_h, w.pad_w)
    groups, cin_pg, cout_pg = w.groups, w.cin_per_group, w.cout_per_group
    s = w.stride
    out = np.zeros((N, w.Cout, w.Hout, w.Wout), dtype=np.float64)
    for g in range(groups):
        Xg = Xp[:, g * cin_pg:(g + 1) * cin_pg]
        Wg = W[g * cout_pg:(g + 1) * cout_pg]
        acc = np.zeros((N, cout_pg, w.Hout, w.Wout), dtype=np.float64)
        for kh in range(w.Kh):
            for kw in range(w.Kw):
                patch = Xg[:, :, kh:kh + s * w.Hout:s, kw:kw + s * w.Wout:s]
                tap = Wg[:, :, kh, kw]
                acc += np.einsum('ncij,oc->noij', patch, tap)
        out[:, g * cout_pg:(g + 1) * cout_pg] = acc
    return out


def reference_conv(w: ConvWorkload, params: dict):
    """
    fp64 direct convolution, independently coded from both CPU impls below
    (no im2col matrix, no scipy call). Operands are generated at the RUN's
    own precision first, then widened to fp64 (see module docstring) so the
    gate measures the kernel's arithmetic, not operand-rounding, error.

    `scale` = the identical convolution structure applied to |X| and |W| --
    the direct generalization of sparse.py's |A|@|B| cancellation-robust
    denominator (and stencil.py's T-sweep |weights|-on-|field| one) to a
    channel*Kh*Kw reduction.
    """
    run_dtype = _dtype(params.get("precision", w.precision))
    N = int(params.get("N", w.N))
    X, W = _make_conv_operands(w, params, N, run_dtype)
    X64, W64 = X.astype(np.float64), W.astype(np.float64)
    out = _direct_conv_fp64(X64, W64, w)
    scale = _direct_conv_fp64(np.abs(X64), np.abs(W64), w)
    return out, scale


# --------------------------------------------------------------------- impls
class NumpyIm2colConv:
    """
    Implicit-GEMM ('im2col') algorithm class (see spec.yaml's mandatory
    algorithm_class disclosure: direct | im2col-implicit-gemm | winograd |
    fft -- this is im2col-implicit-gemm). prepare() allocates every
    persistent buffer ONCE: the padded input, the per-group im2col matrix,
    the reshaped weight matrix, and the output tensor (the DOMAIN_GUIDE's
    "im2col buffer allocation... goes in prepare()" instruction, taken
    literally). run() only (a) gathers each of the Kh*Kw taps into the
    preallocated im2col buffer via a strided-slice assignment -- Kh*Kw*groups
    slice-writes, never a Python loop over output positions -- and (b) calls
    BLAS via `@` into the group's output slice; nothing here is reallocated
    per call (the GEMM's own output temporary is the one exception, same as
    dense.py's NumpyGemm/ScipyCholesky -- numpy's `@` exposes no `out=`).
    """

    name = "numpy-im2col-conv"
    platform = "cpu"
    algorithm_class = "im2col-implicit-gemm"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = _dtype(precision)

    def prepare(self, workload_: ConvWorkload, params: dict):
        w = workload_
        N = int(params.get("N", w.N))
        X, W = _make_conv_operands(w, params, N, self.dtype)
        Xp = _pad_input(X, w.pad_h, w.pad_w)
        groups, cin_pg, cout_pg = w.groups, w.cin_per_group, w.cout_per_group
        patch = cin_pg * w.Kh * w.Kw
        rows = N * w.Hout * w.Wout
        cols = np.empty((groups, rows, patch), dtype=self.dtype)
        Wr = W.reshape(groups, cout_pg, patch)
        out = np.empty((N, w.Cout, w.Hout, w.Wout), dtype=self.dtype)
        return {"Xp": Xp, "cols": cols, "Wr": Wr, "out": out, "w": w,
                "N": N, "groups": groups, "cin_pg": cin_pg, "cout_pg": cout_pg}

    def run(self, h):
        w, Xp, cols, Wr, out = h["w"], h["Xp"], h["cols"], h["Wr"], h["out"]
        N, groups, cin_pg, cout_pg = h["N"], h["groups"], h["cin_pg"], h["cout_pg"]
        s = w.stride
        for g in range(groups):
            Xg = Xp[:, g * cin_pg:(g + 1) * cin_pg]
            col = cols[g].reshape(N, w.Hout, w.Wout, cin_pg, w.Kh, w.Kw)
            for kh in range(w.Kh):
                for kw in range(w.Kw):
                    tap = Xg[:, :, kh:kh + s * w.Hout:s, kw:kw + s * w.Wout:s]
                    col[:, :, :, :, kh, kw] = tap.transpose(0, 2, 3, 1)
            og = cols[g] @ Wr[g].T
            out[:, g * cout_pg:(g + 1) * cout_pg] = \
                og.reshape(N, w.Hout, w.Wout, cout_pg).transpose(0, 3, 1, 2)
        return out

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


class ScipyDirectConv:
    """
    'direct' algorithm class: computes the convolution as an explicit sum of
    2D sliding-window correlations via scipy.ndimage.correlate, one call per
    (batch, group, output-channel, input-channel-within-group) -- deliberately
    NOT an im2col/GEMM formulation, so it exercises a genuinely different
    code path from NumpyIm2colConv above (mirrors stencil.py's NumpyStencil-
    vs-ScipyConvolveStencil contrast, and reuses stencil.py's own
    scipy.ndimage entry point).

    scipy.signal.correlate2d would be the more obvious call for a 'direct'
    2D correlation, but this project's verification venv has a broken
    scipy.signal (`ImportError: cannot import name '_sigtools'` -- a missing
    compiled extension, confirmed independent of this module: `import
    scipy.signal` fails standalone). scipy.ndimage.correlate (already relied
    on by stencil.py's ScipyConvolveStencil) works fine and computes the
    identical valid-mode cross-correlation once cropped to the right offset:
    ndimage.correlate returns a SAME-SIZE (not reduced) output with the
    kernel's default `origin` centering it on each pixel, so
    `full[Kh//2 : Kh//2+valid_H, Kw//2 : Kw//2+valid_W]` equals
    scipy.signal.correlate2d(..., mode='valid') exactly (verified
    numerically to float64 rounding, ~1e-16, against a manual triple loop
    for several shapes before relying on it here) -- and boundary mode never
    matters for that cropped region since every tap it reads falls strictly
    inside the input, never touching the boundary-extended edge.

    Padding the input and allocating the output buffer are hoisted into
    prepare(); only the per-channel correlate calls and their accumulation
    happen in run().
    """

    name = "scipy-direct-conv"
    platform = "cpu"
    algorithm_class = "direct"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = _dtype(precision)

    def prepare(self, workload_: ConvWorkload, params: dict):
        from scipy import ndimage  # local import: only this impl needs it
        w = workload_
        N = int(params.get("N", w.N))
        X, W = _make_conv_operands(w, params, N, self.dtype)
        Xp = _pad_input(X, w.pad_h, w.pad_w)
        out = np.empty((N, w.Cout, w.Hout, w.Wout), dtype=self.dtype)
        return {"Xp": Xp, "W": W, "out": out, "w": w, "N": N, "ndimage": ndimage}

    def run(self, h):
        w, Xp, W, out = h["w"], h["Xp"], h["W"], h["out"]
        ndimage = h["ndimage"]
        N, s = h["N"], w.stride
        cin_pg, cout_pg = w.cin_per_group, w.cout_per_group
        oy, ox = w.Kh // 2, w.Kw // 2
        vh, vw = (Xp.shape[2] - w.Kh + 1), (Xp.shape[3] - w.Kw + 1)
        for n in range(N):
            for g in range(w.groups):
                for oc_local in range(cout_pg):
                    oc = g * cout_pg + oc_local
                    acc = None
                    for ic_local in range(cin_pg):
                        ic = g * cin_pg + ic_local
                        full = ndimage.correlate(Xp[n, ic], W[oc, ic_local],
                                                 mode="constant", cval=0.0)
                        valid = full[oy:oy + vh, ox:ox + vw]
                        contrib = valid[::s, ::s] if s > 1 else valid
                        acc = contrib if acc is None else acc + contrib
                    out[n, oc] = acc
        return out

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


# ================================================================ attention
def _causal_mask(Sq: int, Sk: int) -> np.ndarray:
    """
    query i (0..Sq-1) attends to key j (0..Sk-1) iff j <= i + (Sk - Sq).
    offset = Sk - Sq generalizes the textbook square (Sq==Sk, prefill)
    triangular mask to KV-cache-continuation shapes: for decode (Sq=1),
    offset=Sk-1 and every j qualifies -- a single new token always sees its
    ENTIRE KV history, so 'causal' masking removes nothing there (see
    _cost_attention's docstring for why decode's FLOP formula carries no
    1/2 factor).
    """
    offset = Sk - Sq
    i = np.arange(Sq)[:, None]
    j = np.arange(Sk)[None, :]
    return j <= (i + offset)


@dataclass
class AttentionWorkload:
    """
    O[B,H,Sq,d] = softmax(Q@K^T/sqrt(d) + mask) @ V, GQA when Hkv<H, per
    benchspecs/attention-kernel/spec.yaml's `operation` field. `variant_kind`
    is informational only ("prefill" | "decode"): the SAME dataclass, cost
    rule, reference and impls handle both, since decode is exactly this
    computation at Sq=1 -- not a different algorithm.
    """

    name: str
    kernel: str = "attention-kernel"
    variant: str = ""
    variant_kind: str = "prefill"      # "prefill" | "decode"
    B: int = 1
    H: int = 1
    Hkv: int = 1
    d: int = 64
    Sq: int = 1
    Sk: int = 1
    mask: str = "causal"               # "causal" | "bidirectional"
    precision: str = "fp32"
    seed: int = 42
    source: str = "attention-kernel suite (benchspecs/attention-kernel/spec.yaml)"

    def __post_init__(self):
        if self.H % self.Hkv != 0:
            raise ValueError(
                f"attention {self.name!r}: H={self.H} not a multiple of "
                f"Hkv={self.Hkv} (GQA needs an integer group size)")
        if self.mask not in ("causal", "bidirectional"):
            raise ValueError(
                f"attention {self.name!r}: mask must be 'causal' or "
                f"'bidirectional', got {self.mask!r}")

    @property
    def group_size(self) -> int:
        return self.H // self.Hkv

    def describe(self) -> dict:
        return {
            "name": self.name, "source": self.source, "kernel": self.kernel,
            "spec_variant": self.variant, "variant_kind": self.variant_kind,
            "B": self.B, "H": self.H, "Hkv": self.Hkv, "group_size": self.group_size,
            "d": self.d, "Sq": self.Sq, "Sk": self.Sk, "mask": self.mask,
            "precision": self.precision, "seed": self.seed,
            "kv_layout_note": (
                "spec's decode variant uses paged KV storage (layout: paged, "
                "block_size=16); a single contiguous (B,Hkv,Sk,d) KV buffer "
                "is used here instead -- a memory-management detail that "
                "does not change the math this reference/impls compute, "
                "surfaced here per DOMAIN_GUIDE's non-conformance disclosure "
                "rule" if self.variant_kind == "decode" else ""),
        }


def _qkv(w: AttentionWorkload, params: dict, dtype):
    """
    N(0,1) Q,K,V per spec's dense_operand field (the FlashAttention-family
    convention the spec explicitly allows alongside U(-1,1); N(0,1) is used
    here and documented, per this module's brief). Same seed -> same
    operands for every impl AND the reference."""
    seed = params.get("seed", w.seed)
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((w.B, w.H, w.Sq, w.d)).astype(dtype)
    K = rng.standard_normal((w.B, w.Hkv, w.Sk, w.d)).astype(dtype)
    V = rng.standard_normal((w.B, w.Hkv, w.Sk, w.d)).astype(dtype)
    return Q, K, V


# Prefill: attn-prefill-kernel-fp16bf16's recommended_subset, verbatim.
_PREFILL_SHAPES = [
    dict(name="bert-base-short", B=32, H=12, Hkv=12, d=64, S=128, mask="bidirectional"),
    dict(name="bert-base-long", B=8, H=12, Hkv=12, d=64, S=512, mask="bidirectional"),
    dict(name="bert-large", B=16, H=16, Hkv=16, d=64, S=384, mask="bidirectional"),
    dict(name="vit-large", B=32, H=16, Hkv=16, d=64, S=577, mask="bidirectional"),
    dict(name="gpt2-small", B=8, H=12, Hkv=12, d=64, S=1024, mask="causal"),
    dict(name="llama2-7b-mha", B=8, H=32, Hkv=32, d=128, S=2048, mask="causal"),
    dict(name="llama2-13b", B=4, H=40, Hkv=40, d=128, S=4096, mask="causal"),
    dict(name="llama3-8b-gqa", B=8, H=32, Hkv=8, d=128, S=4096, mask="causal"),
    dict(name="mistral-7b-gqa", B=8, H=32, Hkv=8, d=128, S=8192, mask="causal"),
    dict(name="qwen3-8b-gqa", B=8, H=32, Hkv=8, d=128, S=4096, mask="causal"),
    dict(name="deepseek-v2-lite-mla-approx", B=8, H=16, Hkv=16, d=128, S=4096, mask="causal"),
    dict(name="long-context-stress", B=1, H=32, Hkv=8, d=128, S=32768, mask="causal"),
]

# Decode: attn-decode-kv-cache-kernel's recommended_subset, the fixed-KV-
# length numeric entries only -- PAT's own tree-structured prefix-sharing
# cases (prefix-tree-small/-realistic) use a variable/correlated-L shape
# (B_branches/L_levels, branch_factor, prefix_ratio_pct) this dataclass
# doesn't model and are out of scope here (see AttentionWorkload docstring's
# single-shape design).
_DECODE_SHAPES = [
    dict(name="pat-mha-a100-sat", B=1134, H=32, Hkv=32, d=128, L=1024),
    dict(name="pat-gqa8-a100-sat", B=1134, H=64, Hkv=8, d=128, L=1024),
    dict(name="pat-gqa4-a100-sat", B=1134, H=32, Hkv=8, d=128, L=1024),
    dict(name="pat-gqa2-a100-sat", B=1134, H=16, Hkv=8, d=128, L=1024),
    dict(name="mid-batch-medium-ctx", B=256, H=32, Hkv=8, d=128, L=4096),
    dict(name="small-batch-long-ctx", B=64, H=32, Hkv=8, d=128, L=16384),
    dict(name="hexcute-decode-example", B=16, H=16, Hkv=16, d=128, L=1024),
]


def _build_attention_registry() -> dict[str, AttentionWorkload]:
    reg: dict[str, AttentionWorkload] = {}
    for kw in _PREFILL_SHAPES:
        kw = dict(kw)
        name, S, mask = kw.pop("name"), kw.pop("S"), kw.pop("mask")
        reg[name] = AttentionWorkload(
            name=name, variant="attn-prefill-kernel-fp16bf16", variant_kind="prefill",
            Sq=S, Sk=S, mask=mask, **kw)
    for kw in _DECODE_SHAPES:
        kw = dict(kw)
        name, L = kw.pop("name"), kw.pop("L")
        reg[name] = AttentionWorkload(
            name=name, variant="attn-decode-kv-cache-kernel", variant_kind="decode",
            Sq=1, Sk=L, mask="causal", **kw)
    return reg


def _smoke_attention() -> list[AttentionWorkload]:
    return [
        AttentionWorkload(name="smoke-attn-prefill-causal", variant="smoke",
                          variant_kind="prefill", B=2, H=4, Hkv=4, d=16,
                          Sq=24, Sk=24, mask="causal"),
        AttentionWorkload(name="smoke-attn-prefill-bidir", variant="smoke",
                          variant_kind="prefill", B=2, H=4, Hkv=4, d=16,
                          Sq=20, Sk=20, mask="bidirectional"),
        AttentionWorkload(name="smoke-attn-prefill-gqa", variant="smoke",
                          variant_kind="prefill", B=2, H=8, Hkv=2, d=16,
                          Sq=24, Sk=24, mask="causal"),
        AttentionWorkload(name="smoke-attn-decode-gqa", variant="smoke",
                          variant_kind="decode", B=3, H=8, Hkv=2, d=16,
                          Sq=1, Sk=48, mask="causal"),
    ]


# ----------------------------------------------------------------- cost rule
def _cost_attention(w: AttentionWorkload, params: dict):
    """
    Two literal formulas from benchspecs/attention-kernel/spec.yaml, picked
    by shape (a documented convention, not an inference):

    * prefill-shaped (Sq==Sk==S>1, full self-attention): metric.primary's own
      formula, `flops = (mask==bidirectional ? 4 : 2) * B*H*S^2*d` -- "2
      GEMMs x 2 flops/MAC, halved under causal masking since only the lower
      triangle is computed" (spec text, verbatim). This is the asymptotic
      S^2/2 approximation the spec itself states, not an exact
      triangular-entry count (Sq*(Sq+1)/2) -- kept literal per this track's
      instructions rather than "corrected" to a more precise count the spec
      doesn't ask for.
    * decode-shaped (Sq<=1 or Sq!=Sk): the decode variant's own
      metric.secondary formula, `flops = 4*B*H*Sq*Sk*d` (Sk plays the role of
      L), with NO causal factor -- not an oversight: _causal_mask's docstring
      shows a decode query's causal window always covers the FULL KV history
      (offset=Sk-Sq=Sk-1 when Sq=1), so causal masking removes zero work.
      Applying the prefill formula's 1/2 factor here would silently invent a
      speedup no causal-masked decode kernel actually gets.

    Unit: this module registers "GFLOP/s", not the spec's own "TFLOP/s",
    because harness/metrics.py's `throughput()` is hardwired to a /1e9
    division for every kernel in this codebase -- every OTHER registered
    unit here is a "G<something>/s" label for exactly that reason (GFLOP/s,
    GCUP/s, GTEPS, ... across every domain module). Labeling this "TFLOP/s"
    would keep the /1e9-scaled NUMBER but change what the label claims about
    it by 1000x, which is worse than not matching the spec's preferred unit
    name; divide the reported `throughput`/`gflops` figure by 1000 for the
    spec's TFLOP/s convention. The decode variant's TRUE primary metric
    (achieved memory bandwidth) is unaffected by this choice -- it is always
    reported separately as `gbytes_per_s_lower_bound`, regardless of which
    unit is registered here for the flops-based figure.
    """
    causal = w.mask == "causal"
    if w.Sq == w.Sk and w.Sq > 1:
        flops = (2 if causal else 4) * w.B * w.H * w.Sq * w.Sk * w.d
    else:
        flops = 4 * w.B * w.H * w.Sq * w.Sk * w.d
    itemsize = ITEMSIZE[params.get("precision", w.precision)]
    q_bytes = w.B * w.H * w.Sq * w.d * itemsize
    k_bytes = w.B * w.Hkv * w.Sk * w.d * itemsize
    v_bytes = w.B * w.Hkv * w.Sk * w.d * itemsize
    o_bytes = w.B * w.H * w.Sq * w.d * itemsize
    return int(flops), int(q_bytes + k_bytes + v_bytes + o_bytes)


workload.register_cost("attention-kernel", _cost_attention, "GFLOP/s")


# ----------------------------------------------------------------- reference
def reference_attention(w: AttentionWorkload, params: dict):
    """
    fp64 reference: numerically-stable materialized-(Sq,Sk) softmax
    attention (max-subtraction before exp), independent of both CPU impls
    below (no online-softmax bookkeeping shared with NumpyFlashAttention, no
    code shared with NumpyAttention beyond the same textbook formula every
    correct attention implementation necessarily shares). Operands are
    generated at the RUN's own precision first, then widened to fp64 (see
    module docstring), same discipline as reference_conv.

    `scale` = P_ref @ |V|, i.e. the SAME convex combination the kernel
    computes, applied to |V| instead of V. Softmax weights P are already
    elementwise non-negative and each row sums to exactly 1 -- there is no
    signed cancellation in the P@V reduction itself (unlike a general
    matmul), so this is an *exact* elementwise upper bound on how much a
    V-precision error can move the output, not merely the usual |A|@|B|
    heuristic bound; it plays the same "componentwise magnitude of the
    computation" role DOMAIN_GUIDE's scale contract asks for (P takes the
    role of "A", already non-negative here, and V the role of "B"). Errors
    introduced upstream (in QK^T / the softmax itself) redistribute P's
    weight among |V|-sized rows, so they also show up in O at roughly this
    same scale -- this single quantity is the right cancellation-robust
    denominator for O end-to-end, not only for the final matmul.

    Per this module's brief: the softmax WEIGHTS P are themselves exactly in
    [0,1] (row-stochastic); the sensible scale for the KERNEL OUTPUT O (a
    convex combination of V, which is not itself bounded to [0,1] since
    V~N(0,1) is unbounded) is this P-weighted |V| quantity, not a bare
    [0,1] constant.
    """
    run_dtype = _dtype(params.get("precision", w.precision))
    Q, K, V = _qkv(w, params, run_dtype)
    Q, K, V = Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64)
    scale_qk = 1.0 / np.sqrt(w.d)
    g = w.group_size
    Kh = np.repeat(K, g, axis=1)
    Vh = np.repeat(V, g, axis=1)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, Kh) * scale_qk
    if w.mask == "causal":
        # inline, independent mask construction (deliberately NOT _causal_mask:
        # the audit proved a mutation of the shared helper was invisible to the
        # gate when reference and impl both called it)
        offs = w.Sk - w.Sq
        keep = (np.arange(w.Sk)[None, :] <= (np.arange(w.Sq)[:, None] + offs))
        scores = np.where(keep[None, None, :, :], scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=-1, keepdims=True)
    out = np.einsum('bhqk,bhkd->bhqd', p, Vh)
    scale = np.einsum('bhqk,bhkd->bhqd', p, np.abs(Vh))
    return out, scale


# --------------------------------------------------------------------- impls
class NumpyAttention:
    """
    Naive attention: materializes the full (B,H,Sq,Sk) score/softmax matrix.
    This is exactly the quadratic-memory baseline every flash-attention paper
    contrasts itself against -- see NumpyFlashAttention below for the
    tiled/online-softmax contrast this track exists to measure.

    prepare() hoists everything that doesn't depend on the softmax itself:
    Q,K,V (GQA-expanded to H heads once) and the static causal mask (per the
    DOMAIN_GUIDE's "mask construction... goes in prepare()" instruction).
    run() performs the QK^T -> mask -> softmax -> @V pipeline; the
    (B,H,Sq,Sk) score-matrix allocation there IS the naive algorithm's
    per-call cost being measured, not something to hide by hoisting it away
    (hoisting it would silently turn this into the very kernel it exists to
    contrast against).
    """

    name = "numpy-attention"
    platform = "cpu"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = _dtype(precision)

    def prepare(self, workload_: AttentionWorkload, params: dict):
        w = workload_
        Q, K, V = _qkv(w, params, self.dtype)
        g = w.group_size
        Kh = np.repeat(K, g, axis=1)
        Vh = np.repeat(V, g, axis=1)
        mask = _causal_mask(w.Sq, w.Sk) if w.mask == "causal" else None
        return {"Q": Q, "Kh": Kh, "Vh": Vh, "mask": mask, "scale": 1.0 / np.sqrt(w.d)}

    def run(self, h):
        Q, Kh, Vh, mask, scale = h["Q"], h["Kh"], h["Vh"], h["mask"], h["scale"]
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Kh) * scale
        if mask is not None:
            scores = np.where(mask[None, None, :, :], scores, -np.inf)
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p = p / p.sum(axis=-1, keepdims=True)
        return np.einsum('bhqk,bhkd->bhqd', p, Vh)

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


class NumpyFlashAttention:
    """
    Tiled / online-softmax attention (FlashAttention's core algorithm):
    processes K,V in fixed-size blocks along Sk, maintaining a running
    (unnormalized) output accumulator, running row-max, and running softmax
    denominator per query row. The full (Sq,Sk) score/probability matrix is
    NEVER materialized -- only one (Sq, block) tile exists at a time. This is
    the contrast the attention-kernel track exists to measure (see
    NumpyAttention's docstring).

    prepare() allocates every buffer that persists across the whole call --
    Q,K,V (GQA-expanded), the running accumulators (acc/m/l), and the static
    per-row query index used by every block's mask test -- once. run() loops
    over Sk in BLOCK-sized chunks; each iteration's tiny (Sq,block) causal
    mask is legitimately computed there, NOT hoisted: precomputing every
    block's mask up front would total the same O(Sq*Sk) memory the whole
    algorithm exists to avoid, defeating the "never materializes (Sq,Sk)"
    property this impl is measured against (the same "per-call temporaries
    for a bounded per-tap loop are fine, only the OUTER persistent buffers
    must be hoisted" reasoning stencil.py's `_sweep` docstring states
    explicitly).
    """

    name = "numpy-flash-attention"
    platform = "cpu"
    BLOCK = 64  # Sk tile width -- bounds this impl's working memory to O(Sq*BLOCK)

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = _dtype(precision)

    def prepare(self, workload_: AttentionWorkload, params: dict):
        w = workload_
        Q, K, V = _qkv(w, params, self.dtype)
        g = w.group_size
        Kh = np.repeat(K, g, axis=1)
        Vh = np.repeat(V, g, axis=1)
        B, H, Sq, d = Q.shape
        Sk = Kh.shape[2]
        return {
            "Q": Q, "Kh": Kh, "Vh": Vh, "Sq": Sq, "Sk": Sk,
            "scale": 1.0 / np.sqrt(d), "causal": w.mask == "causal",
            "offset": Sk - Sq, "i_idx": np.arange(Sq)[:, None],
            "acc": np.empty((B, H, Sq, d), dtype=np.float64),
            "m": np.empty((B, H, Sq), dtype=np.float64),
            "l": np.empty((B, H, Sq), dtype=np.float64),
        }

    def run(self, h):
        Q, Kh, Vh = h["Q"], h["Kh"], h["Vh"]
        acc, m, l = h["acc"], h["m"], h["l"]
        acc.fill(0.0)
        m.fill(-np.inf)
        l.fill(0.0)
        scale, causal, offset, i_idx = h["scale"], h["causal"], h["offset"], h["i_idx"]
        Sk, block = h["Sk"], self.BLOCK
        for k0 in range(0, Sk, block):
            k1 = min(k0 + block, Sk)
            Kb = Kh[:, :, k0:k1, :]
            Vb = Vh[:, :, k0:k1, :]
            s = np.einsum('bhqd,bhkd->bhqk', Q, Kb).astype(np.float64) * scale
            if causal:
                j = np.arange(k0, k1)[None, :]
                valid = j <= (i_idx + offset)
                s = np.where(valid[None, None, :, :], s, -np.inf)
            m_new = np.maximum(m, s.max(axis=-1))
            alpha = np.where(np.isfinite(m), np.exp(m - m_new), 0.0)
            p = np.where(np.isfinite(s), np.exp(s - m_new[..., None]), 0.0)
            l[:] = l * alpha + p.sum(axis=-1)
            acc[:] = acc * alpha[..., None] + np.einsum('bhqk,bhkd->bhqd', p, Vb)
            m[:] = m_new
        out = acc / l[..., None]
        return out.astype(Q.dtype)

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


# ======================================================= sparse-attention
# Mask-family builders. Each returns a bool (S,S) array, True == "query i is
# allowed to attend to key j". These are WORKLOAD-DATA generators (like
# `_qkv`/`_make_conv_operands` above) -- called exactly once per workload
# instance (see SparseAttentionWorkload.mask below) to build the single fixed
# mask array the reference and every impl consume. Approximate density is
# fine and expected: benchspecs/sparse-attention-kernel/spec.yaml's own
# `measured_sparsity_reported` clause says no named pattern lands exactly on
# its nominal Sf target (BigBird's random component and model-defined masks
# don't either), so this module reports MEASURED density rather than chasing
# an exact target.
def _sliding_window_mask(S: int, window: int) -> np.ndarray:
    i = np.arange(S)[:, None]
    j = np.arange(S)[None, :]
    half = max(window // 2, 0)
    return np.abs(i - j) <= half


def _longformer_mask(S: int, band: int, global_width: int) -> np.ndarray:
    i = np.arange(S)[:, None]
    j = np.arange(S)[None, :]
    local = np.abs(i - j) <= band
    glob = (i < global_width) | (j < global_width)
    return local | glob


def _bigbird_mask(S: int, band: int, global_width: int, fill_rate: float,
                  rng: np.random.Generator) -> np.ndarray:
    base = _longformer_mask(S, band, global_width)
    extra = rng.random((S, S)) < fill_rate
    return base | extra


def _dilated1d_mask(S: int, window: int, dilation: int) -> np.ndarray:
    i = np.arange(S)[:, None]
    j = np.arange(S)[None, :]
    delta = np.abs(i - j)
    dil = max(dilation, 1)
    return (delta <= window) & (delta % dil == 0)


def _block_local_mask(S: int, block: int, n_global_blocks: int) -> np.ndarray:
    """
    Block-diagonal ('regional') sparsity: query i and key j attend iff they
    fall in the same size-`block` block, plus the first `n_global_blocks`
    blocks attend/are-attended-to globally (a CLS-token-style global block).
    The closest synthetic proxy this module can honestly build for
    RegionViT's fixed regional block-sparse pattern (see vit-sparse's own
    artifact STATUS.md under bench/artifacts/sparse-attention-kernel/ for how
    that artifact's REAL ViT-derived masks are actually gated) -- this module
    does not have a trained ViT checkpoint to derive DynamicViT/RegionViT's
    true masks from, so it does not claim to.
    """
    blk = max(block, 1)
    bid = np.arange(S) // blk
    same_block = bid[:, None] == bid[None, :]
    glob = (bid[:, None] < n_global_blocks) | (bid[None, :] < n_global_blocks)
    return same_block | glob


def _coo_arbitrary_mask(S: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Bernoulli(density) i.i.d. mask -- TomczakK25's own Sf-swept
    microbenchmark ('coo_arbitrary' in the spec's recommended_subset), the
    one pattern in this track with no structural constraint at all."""
    return rng.random((S, S)) < density


def _causal_sparse_mask(S: int) -> np.ndarray:
    i = np.arange(S)[:, None]
    j = np.arange(S)[None, :]
    return j <= i


_MASK_BUILDERS = {
    "causal": lambda S, rng, **kw: _causal_sparse_mask(S),
    "sliding_window": lambda S, rng, **kw: _sliding_window_mask(
        S, kw.get("window", max(S // 10, 2))),
    "longformer_local_global": lambda S, rng, **kw: _longformer_mask(
        S, kw.get("band", max(S // 32, 1)), kw.get("global_width", max(S // 32, 1))),
    "bigbird_local_global_random": lambda S, rng, **kw: _bigbird_mask(
        S, kw.get("band", max(S // 32, 1)), kw.get("global_width", max(S // 32, 1)),
        kw.get("fill_rate", 0.1), rng),
    "dilated_1d": lambda S, rng, **kw: _dilated1d_mask(
        S, kw.get("window", max(S // 32, 1)), kw.get("dilation", 2)),
    "block_local": lambda S, rng, **kw: _block_local_mask(
        S, kw.get("block", max(S // 16, 2)), kw.get("n_global_blocks", 1)),
    "coo_arbitrary": lambda S, rng, **kw: _coo_arbitrary_mask(
        S, kw.get("density", 0.05), rng),
}


def _build_sparse_mask(pattern: str, S: int, seed: int, **pattern_params) -> np.ndarray:
    """
    THE single authoritative mask-construction function for a
    SparseAttentionWorkload. This builds WORKLOAD DATA (a fixed sparsity
    pattern, analogous to a real graph's adjacency matrix in the sddmm/spmm
    tracks, or this module's own `_qkv`/`_make_conv_operands` operand
    generators) -- it is correct, and required, for this to be shared across
    the reference and every impl (every impl must see the SAME mask, per the
    spec's own "every impl consumes the same mask" requirement). What must
    NOT be shared (and is not -- see reference_sparse_attention and
    NumpySparseAttention below) is the mask APPLICATION logic: the actual
    `np.where(mask, scores, -inf)` step inside each softmax pipeline. Per
    DOMAIN_GUIDE.md's reference-independence audit ruling, a bug baked into a
    shared "_apply_mask" helper would silently corrupt both sides identically
    and the gate would still read err=0 -- exactly the failure mode that
    ruling exists to prevent. Mask CONSTRUCTION sharing carries no such risk:
    the gate's job is "does this impl correctly compute masked-softmax
    attention GIVEN this fixed mask", not "did this impl independently
    re-derive the same sparsity pattern from pattern_params" (the latter
    isn't even well-posed for BigBird's random component or a real ViT's
    learned mask).

    Every row is forced to have at least one True entry (the diagonal is
    always set True) so no row's softmax denominator can be identically
    zero. An all-masked row produces NaN (0/0), which is an ILL-POSED INPUT,
    not a kernel bug -- forcing self-attention removes that possibility for
    every pattern here, matching how every real causal/local/global/regional
    mask family surveyed already includes the diagonal by construction. This
    also protects the correctness-gate mutation test (see ml.py's own test
    notes / DOMAIN_GUIDE.md): a impl that gets mask POLARITY backwards
    (attends where it shouldn't, or vice-versa) must produce a large,
    catchable numeric error, not a silent NaN-vs-NaN non-comparison -- see
    harness.check_correctness, whose NaN handling (`val <= tolerance` is
    False for NaN `val`) already fails a NaN gate, but a well-posed nonzero
    error is the stronger, more informative failure this module aims for.
    """
    rng = np.random.default_rng(seed)
    fn = _MASK_BUILDERS.get(pattern)
    if fn is None:
        raise ValueError(
            f"unknown sparse-attention-kernel mask pattern {pattern!r}; "
            f"have {sorted(_MASK_BUILDERS)}")
    mask = np.asarray(fn(S, rng, **pattern_params), dtype=bool)
    np.fill_diagonal(mask, True)
    return mask


@dataclass
class SparseAttentionWorkload:
    """
    O[B,H,S,d] = softmax(Q@K^T/sqrt(d) + mask) @ V, mask in {0,-inf}^{SxS},
    per benchspecs/sparse-attention-kernel/spec.yaml's `operation` field
    (structured-mask variant: sparse-attn-structured-mask-kernel-fp16 --
    see this module's own top docstring for why the spec's OTHER variant,
    the graph-adjacency one, is not implemented here).

    The mask is workload DATA, built ONCE (deterministically, seeded) by
    `_build_sparse_mask` and LAZILY cached on `self.mask` (a `@property`,
    not built in `__post_init__`, so constructing a whole registry of
    workloads -- including S=16384 shapes -- stays cheap; the O(S^2) mask
    array is only materialized when something actually reads `.mask`).
    `describe()` reports a sha256 checksum of the packed-bit mask array (so
    two runs can be confirmed to have consumed a bit-identical mask even if
    written with differently-formatted pattern_params) and the MEASURED
    density nnz/S^2 (the spec's own `measured_sparsity_reported` mandate).
    """

    name: str
    kernel: str = "sparse-attention-kernel"
    variant: str = "sparse-attn-structured-mask-kernel-fp16"
    B: int = 1
    H: int = 1
    d: int = 64
    S: int = 1024
    pattern: str = "sliding_window"
    pattern_params: dict = field(default_factory=dict)
    sf_target: float | None = None     # spec's own nominal Sf; informational only
    precision: str = "fp16"
    seed: int = 42
    source: str = ("sparse-attention-kernel structured-mask suite "
                   "(benchspecs/sparse-attention-kernel/spec.yaml)")

    _mask_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.pattern not in _MASK_BUILDERS:
            raise ValueError(
                f"sparse-attention workload {self.name!r}: unknown pattern "
                f"{self.pattern!r}; have {sorted(_MASK_BUILDERS)}")

    @property
    def mask(self) -> np.ndarray:
        if self._mask_cache is None:
            self._mask_cache = _build_sparse_mask(
                self.pattern, self.S, self.seed, **self.pattern_params)
        return self._mask_cache

    @property
    def mask_nnz(self) -> int:
        return int(self.mask.sum())

    @property
    def measured_density(self) -> float:
        return self.mask_nnz / float(self.S * self.S)

    @property
    def mask_checksum(self) -> str:
        import hashlib
        return hashlib.sha256(np.packbits(self.mask).tobytes()).hexdigest()

    def describe(self) -> dict:
        return {
            "name": self.name, "source": self.source, "kernel": self.kernel,
            "spec_variant": self.variant,
            "B": self.B, "H": self.H, "d": self.d, "S": self.S,
            "pattern": self.pattern, "pattern_params": self.pattern_params,
            "sf_target": self.sf_target,
            "mask_nnz": self.mask_nnz, "measured_density": self.measured_density,
            "mask_checksum_sha256": self.mask_checksum,
            "precision": self.precision, "seed": self.seed,
            "mask_provenance": (
                "deterministic, seeded (np.random.default_rng(seed) inside "
                "_build_sparse_mask), built once and cached; shared as fixed "
                "workload data by the reference and every impl -- see "
                "SparseAttentionWorkload's and _build_sparse_mask's "
                "docstrings for why sharing the mask ARRAY (not the mask "
                "APPLICATION code) is correct here."),
        }


def _qkv_sparse(w: SparseAttentionWorkload, params: dict, dtype):
    """
    N(0,1) Q,K,V for a SparseAttentionWorkload. Written as its own function
    (not reusing `_qkv` above) since this track's shape has no GQA
    distinction (Hkv==H always here -- not spec-mandated either way, kept
    simple since none of the 3 surveyed structured-mask papers use GQA) and
    a single shared S (Sq==Sk, matching the spec's square Q@K^T). Same seed
    -> same operands for every impl AND the reference, same discipline as
    every other operand generator in this module.
    """
    seed = params.get("seed", w.seed)
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((w.B, w.H, w.S, w.d)).astype(dtype)
    K = rng.standard_normal((w.B, w.H, w.S, w.d)).astype(dtype)
    V = rng.standard_normal((w.B, w.H, w.S, w.d)).astype(dtype)
    return Q, K, V


# Structured-mask variant's own recommended_subset (benchspecs/
# sparse-attention-kernel/spec.yaml), transcribed with concrete
# pattern_params chosen to land near each entry's nominal Sf (approximate by
# design -- see _build_sparse_mask's docstring). The 3 ViT-model-native
# entries (dynamicvit-vit-l, regionvit-vit-l, visionlongformer-hires) are
# NOT transcribed here: those are properties of trained models this
# synthetic CPU-verification module has no checkpoint to derive from; see
# bench/artifacts/sparse-attention-kernel/vit-sparse/STATUS.md for how that
# artifact's real ViT masks are gated instead. `block-local-s1024-blk64` is
# an addition (not in the spec's own table) -- the closest synthetic proxy
# for RegionViT's regional block-sparse pattern, used by vit-sparse's gate.
_SPARSE_ATTN_SHAPES = [
    dict(name="causal-s1024", S=1024, pattern="causal", pattern_params={},
        B=8, H=12, d=64, sf_target=None),
    dict(name="sliding-s1024-sf01", S=1024, pattern="sliding_window",
        pattern_params={"window": 102}, B=8, H=12, d=64, sf_target=0.1),
    dict(name="sliding-s4096-sf01", S=4096, pattern="sliding_window",
        pattern_params={"window": 410}, B=4, H=32, d=128, sf_target=0.1),
    dict(name="sliding-s16384-sf001", S=16384, pattern="sliding_window",
        pattern_params={"window": 164}, B=1, H=32, d=128, sf_target=0.01),
    dict(name="longformer-s1024-sf01", S=1024, pattern="longformer_local_global",
        pattern_params={"band": 32, "global_width": 32}, B=8, H=12, d=64, sf_target=0.1),
    dict(name="longformer-s4096-sf005", S=4096, pattern="longformer_local_global",
        pattern_params={"band": 64, "global_width": 64}, B=2, H=32, d=128, sf_target=0.05),
    dict(name="bigbird-s1024-sf01", S=1024, pattern="bigbird_local_global_random",
        pattern_params={"band": 32, "global_width": 32, "fill_rate": 0.1},
        B=8, H=12, d=64, sf_target=0.1),
    dict(name="dilated-s4096-sf005", S=4096, pattern="dilated_1d",
        pattern_params={"window": 204, "dilation": 2}, B=4, H=32, d=128, sf_target=0.05),
    dict(name="sf-sweep-microbench-sf0001", S=8192, pattern="coo_arbitrary",
        pattern_params={"density": 0.0001}, B=1, H=1, d=64, sf_target=0.0001),
    dict(name="sf-sweep-microbench-sf001", S=8192, pattern="coo_arbitrary",
        pattern_params={"density": 0.001}, B=1, H=1, d=64, sf_target=0.001),
    dict(name="sf-sweep-microbench-sf01", S=8192, pattern="coo_arbitrary",
        pattern_params={"density": 0.01}, B=1, H=1, d=64, sf_target=0.01),
    dict(name="sf-sweep-microbench-sf1", S=8192, pattern="coo_arbitrary",
        pattern_params={"density": 1.0}, B=1, H=1, d=64, sf_target=1.0),
    dict(name="block-local-s1024-blk64", S=1024, pattern="block_local",
        pattern_params={"block": 64, "n_global_blocks": 1}, B=8, H=12, d=64, sf_target=None),
]


def _build_sparse_attn_registry() -> dict[str, SparseAttentionWorkload]:
    reg: dict[str, SparseAttentionWorkload] = {}
    for kw in _SPARSE_ATTN_SHAPES:
        kw = dict(kw)
        name = kw.pop("name")
        reg[name] = SparseAttentionWorkload(
            name=name, variant="sparse-attn-structured-mask-kernel-fp16", **kw)
    return reg


def _smoke_sparse_attention() -> list[SparseAttentionWorkload]:
    return [
        SparseAttentionWorkload(name="smoke-sparse-attn-causal", variant="smoke",
                                B=2, H=2, d=16, S=32, pattern="causal", pattern_params={}),
        SparseAttentionWorkload(name="smoke-sparse-attn-sliding", variant="smoke",
                                B=2, H=2, d=16, S=32, pattern="sliding_window",
                                pattern_params={"window": 6}, sf_target=0.2),
        SparseAttentionWorkload(name="smoke-sparse-attn-longformer", variant="smoke",
                                B=2, H=2, d=16, S=32, pattern="longformer_local_global",
                                pattern_params={"band": 2, "global_width": 2}, sf_target=0.2),
        SparseAttentionWorkload(name="smoke-sparse-attn-bigbird", variant="smoke",
                                B=2, H=2, d=16, S=32, pattern="bigbird_local_global_random",
                                pattern_params={"band": 2, "global_width": 2, "fill_rate": 0.1},
                                sf_target=0.25),
        SparseAttentionWorkload(name="smoke-sparse-attn-dilated", variant="smoke",
                                B=2, H=2, d=16, S=32, pattern="dilated_1d",
                                pattern_params={"window": 8, "dilation": 2}, sf_target=0.2),
        SparseAttentionWorkload(name="smoke-sparse-attn-block-local", variant="smoke",
                                B=2, H=2, d=16, S=32, pattern="block_local",
                                pattern_params={"block": 8, "n_global_blocks": 1}, sf_target=None),
        SparseAttentionWorkload(name="smoke-sparse-attn-coo", variant="smoke",
                                B=2, H=2, d=16, S=32, pattern="coo_arbitrary",
                                pattern_params={"density": 0.15}, sf_target=0.15),
    ]


# ----------------------------------------------------------------- cost rule
def _cost_sparse_attention(w: SparseAttentionWorkload, params: dict):
    """
    benchspecs/sparse-attention-kernel/spec.yaml's structured-mask variant,
    metric.primary, LITERAL: "effective TFLOP/s: flops = 2 * 2 * B * H *
    nnz(mask) * d (QK^T + PV, 2 flops/MAC, nnz = actual number of unmasked
    (query,key) pairs -- NOT S^2)". `nnz` is read from the workload's own
    cached `.mask_nnz` (the SAME mask array the reference/impls consume),
    not recomputed independently. Registered as "GFLOP/s" for the identical
    /1e9-hardwired-throughput reason _cost_attention documents above --
    divide the reported number by 1000 for the spec's own TFLOP/s label.

    The spec ALSO mandates a secondary dense-equivalent number (same wall
    time, S^2-based formula) reported side by side, specifically so a
    speedup claim states which FLOP convention it uses. This harness's
    cost-rule contract (workload.register_cost) returns exactly one
    (flops, bytes) pair per kernel -- there is no second slot for a
    simultaneously-computed alternative numerator -- so this module reports
    the primary (effective) number only, honestly, rather than silently
    dropping one of the two spec-mandated numbers into a metrics field
    nobody asked for. `describe()`'s `measured_density` lets a reader
    reconstruct the dense-equivalent figure by hand:
    dense_gflops = effective_gflops / measured_density.
    """
    nnz = w.mask_nnz
    flops = 2 * 2 * w.B * w.H * nnz * w.d
    itemsize = ITEMSIZE[params.get("precision", w.precision)]
    q_bytes = w.B * w.H * w.S * w.d * itemsize
    k_bytes = q_bytes
    v_bytes = q_bytes
    o_bytes = q_bytes
    mask_bytes = w.S * w.S  # 1 byte/entry lower bound (bool storage); conservative
    return int(flops), int(q_bytes + k_bytes + v_bytes + o_bytes + mask_bytes)


workload.register_cost("sparse-attention-kernel", _cost_sparse_attention, "GFLOP/s")


# ----------------------------------------------------------------- reference
def reference_sparse_attention(w: SparseAttentionWorkload, params: dict):
    """
    fp64 masked-softmax attention. Reads the workload's fixed `.mask` array
    (shared workload DATA, see SparseAttentionWorkload's docstring) but
    applies it via its OWN inline `np.where` call here -- a separate code
    object from NumpySparseAttention.run()'s masking step below, matching
    reference_attention's own "inline, independent mask construction
    (deliberately NOT _causal_mask...)" precedent above, extended to this
    track's data-driven (not shape-derived) masks: the ARRAY is shared (it
    must be, for the two sides to be comparable at all), the APPLICATION
    code is not.

    scale = P@|V|, the identical convention reference_attention's own
    docstring derives in full (P is row-stochastic and already
    non-negative, so this is an exact elementwise upper bound on how a
    V-precision error moves O, not the usual |A|@|B| heuristic bound) --
    independently re-typed here per this track's instruction to mirror
    without sharing reference_attention's code object.
    """
    run_dtype = _dtype(params.get("precision", w.precision))
    Q, K, V = _qkv_sparse(w, params, run_dtype)
    Q, K, V = Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64)
    scale_qk = 1.0 / np.sqrt(w.d)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) * scale_qk
    # inline, independent mask application -- deliberately NOT shared with
    # NumpySparseAttention.run() (see this function's docstring)
    keep = w.mask
    scores = np.where(keep[None, None, :, :], scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=-1, keepdims=True)
    out = np.einsum('bhqk,bhkd->bhqd', p, V)
    scale = np.einsum('bhqk,bhkd->bhqd', p, np.abs(V))
    return out, scale


# --------------------------------------------------------------------- impls
class NumpySparseAttention:
    """
    The ONE CPU/numpy masked-attention impl this track ships (per this
    track's brief: "one numpy/scipy masked-attention impl", not a
    naive/flash pair like attention-kernel's two CPU impls above -- this
    track's interesting contrast lives in the mask's structure and the
    paper artifacts' native sparse formats under bench/artifacts/
    sparse-attention-kernel/, not in a CPU tiling algorithm). Materializes
    the full (B,H,S,S) score matrix like NumpyAttention above; masking is
    applied via THIS CLASS's own `np.where` call in run() -- a separate code
    object from reference_sparse_attention's (see that function's
    docstring) even though the two lines look textually similar: each is
    independently typed, neither calls the other, and neither calls a
    shared "_apply_mask" helper.

    prepare() hoists Q,K,V generation and reads the workload's fixed mask
    array (no re-derivation -- both are input data, per DOMAIN_GUIDE's
    "everything that can be hoisted out of run() must be"). run() performs
    the QK^T -> mask -> softmax -> @V pipeline; the (B,H,S,S) score-matrix
    allocation there is this (dense, unoptimized) algorithm's genuine
    per-call cost, not hidden by hoisting -- exactly NumpyAttention's own
    precedent above.
    """

    name = "numpy-masked-attention"
    platform = "cpu"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = _dtype(precision)

    def prepare(self, workload_: SparseAttentionWorkload, params: dict):
        w = workload_
        Q, K, V = _qkv_sparse(w, params, self.dtype)
        return {"Q": Q, "K": K, "V": V, "mask": w.mask, "scale": 1.0 / np.sqrt(w.d)}

    def run(self, h):
        Q, K, V, mask, scale = h["Q"], h["K"], h["V"], h["mask"], h["scale"]
        scores = np.einsum('bhqd,bhkd->bhqk', Q, K) * scale
        scores = np.where(mask[None, None, :, :], scores, -np.inf)
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p = p / p.sum(axis=-1, keepdims=True)
        return np.einsum('bhqk,bhkd->bhqd', p, V)

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


# ============================================================ quantized-gemm
_W_OFFSET = 1_000_003  # weight matrix draws from seed+offset -- matches
# dense.py's own _B_OFFSET convention, keeping the activation and
# pre-quantization weight RNG streams independent of each other.


@dataclass
class QuantGemmWorkload:
    """
    LLM-inference-shaped quantized GEMM: C[M,N] = A[M,K] (fp16/bf16
    activation) @ dequantize(W[K,N]), per benchspecs/quantized-gemm/
    spec.yaml's `operation` field -- weight-only quantization (W4A16 the
    spec's dominant/headline format), fp16 activation, per-group symmetric
    scaling (group=128 columns of K, the spec's default).

    `.quantize()` is a PURE function of (seed, K, N, bits, group_size): every
    caller that calls it -- reference_qgemm, this module's own CPU impl,
    torch-dequant-then-cublas-gemm in gpu_cuda.py, and any paper artifact's
    own prepare() -- gets BIT-IDENTICAL codes/scale. That is this
    integration's KEY FAIRNESS POINT (per the task brief): quantization
    error is not part of the correctness gate, only each implementation's
    OWN kernel arithmetic on top of the shared quantized values is. What
    legitimately varies between the reference and any implementation is only
    the DEQUANTIZE-then-multiply ARITHMETIC each writes independently on top
    of these shared values (DOMAIN_GUIDE's audit ruling: same formula, never
    the same code object -- see reference_qgemm and NumpyDequantGemm, which
    each retype `codes.astype(...) * scale[group_idx]` as a fresh
    expression, never calling one shared "_dequantize" helper).

    `.quantize()` returns the "packed quantized weights" this integration's
    task brief asks the workload to hold: `codes` (int8, one element per
    logical weight -- this domain's own canonical CPU-side packed
    representation at a granularity numpy can express; a REAL sub-8-bit
    hardware pack format, e.g. two INT4 codes per byte or a vendor's own
    bit-interleaved tile layout, is each ARTIFACT's OWN prepare()-time
    responsibility, built FROM these codes per ARTIFACT_GUIDE's "convert
    from our quantized values where possible" rule -- see
    bench/artifacts/quantized-gemm/*/adapter.py) and `scale` (fp64,
    per-group-per-column). `.dequantize_fp64()` is the "exact dequantized
    fp64 weights" the brief also asks for: codes*scale at full fp64, the
    canonical ground truth (no additional storage-precision rounding, unlike
    what a real kernel's own fp16-stored scale would introduce -- THAT
    additional rounding is exactly the kernel arithmetic error this track's
    gate exists to catch, not something this domain's ground truth should
    pre-bake in).

    `describe()` does NOT embed the K*N-element codes/scale arrays --
    several of this track's recommended_subset shapes are 74240x14848 (~1.1
    GB of int8 codes alone); embedding that in every JSON result record
    would be absurd. Instead it records (seed, bits, group_size, M, K, N):
    by `.quantize()`'s own determinism, that tuple reproduces the packed
    weights and their exact fp64 dequantization bit-for-bit, which is what
    DOMAIN_GUIDE's Workload contract actually asks for ("identify the input
    completely enough to reproduce it... a checksum where cheap" -- a
    deterministic generator's seed already IS that reproducibility
    guarantee, cheaper than hashing a multi-hundred-MB array on every
    describe() call).
    """

    name: str
    kernel: str = "quantized-gemm"
    variant: str = ""
    variant_kind: str = "decode"     # "decode" | "batched" -- informational
    # only: the SAME dataclass/cost/reference/impl code handles both, exactly
    # AttentionWorkload's prefill/decode convention above.
    model: str = ""
    layer: str = ""
    M: int = 16
    K: int = 4096
    N: int = 4096
    bits: int = 4          # headline format: INT4, the spec's "dominant" weight_formats entry
    group_size: int = 128  # spec's default group granularity (columns of K)
    precision: str = "fp16"    # activation dtype; the weight is quantized regardless of this field
    seed: int = 42
    source: str = ("quantized-gemm suite (benchspecs/quantized-gemm/"
                   "spec.yaml recommended_subset, union-of-model-families)")

    def __post_init__(self):
        if self.bits < 2:
            raise ValueError(
                f"quantized-gemm {self.name!r}: bits={self.bits} must be >=2 "
                "(a 1-bit symmetric code range is degenerate: qmax=0)")

    @property
    def group_size_eff(self) -> int:
        """Effective group size: falls back to a single group spanning all
        of K (per-column quantization, the spec's group=-1 convention) when
        the configured group_size doesn't evenly divide K -- documented
        fallback, not silent corruption; every recommended_subset shape's
        K is either a multiple of 128 or intentionally routed through this
        fallback (e.g. K=6144 with group=128 -> 48 groups exactly; a
        smoke shape with K=96, group=128 -> falls back to one group of 96)."""
        gs = self.group_size
        if gs <= 0 or gs >= self.K or self.K % gs != 0:
            return self.K
        return gs

    @property
    def n_groups(self) -> int:
        return self.K // self.group_size_eff

    def quantize(self):
        """
        Deterministic, PURE function of (seed, K, N, bits, group_size) -- see
        class docstring. Symmetric per-group affine quantization:
        `scale[g,n] = max(|W[group g, n]|) / qmax`, `qmax = 2^(bits-1) - 1`,
        `codes = clip(round(W / scale), -qmax, qmax)`. `W` itself is drawn
        N(0,1) at fp64 (this module's `_qkv`'s own convention for a weight-
        like operand, applied here to the pre-quantization float weight).

        Returns `(codes: int8 (K,N), scale: fp64 (n_groups,N), group_idx: int
        (K,)` mapping each row to its group -- `scale[group_idx]` broadcasts
        directly to (K,N)).
        """
        rng = np.random.default_rng(self.seed + _W_OFFSET)
        Wtrue = rng.standard_normal((self.K, self.N))
        gs, n_groups = self.group_size_eff, self.n_groups
        group_idx = np.repeat(np.arange(n_groups), gs)
        qmax = float((1 << (self.bits - 1)) - 1)
        amax = np.abs(Wtrue.reshape(n_groups, gs, self.N)).max(axis=1)
        scale = np.maximum(amax, 1e-12) / qmax
        codes = np.clip(np.round(Wtrue / scale[group_idx]), -qmax, qmax).astype(np.int8)
        return codes, scale, group_idx

    def dequantize_fp64(self) -> np.ndarray:
        """Convenience wrapper -- codes*scale at full fp64 (the "exact
        dequantized fp64 weights" the task brief asks the workload to hold).
        NOT called by reference_qgemm itself, which inlines this same
        formula as an independent expression -- see that function and the
        class docstring's audit-ruling note."""
        codes, scale, group_idx = self.quantize()
        return codes.astype(np.float64) * scale[group_idx]

    def describe(self) -> dict:
        qmax = (1 << (self.bits - 1)) - 1
        return {
            "name": self.name, "source": self.source, "kernel": self.kernel,
            "spec_variant": self.variant, "variant_kind": self.variant_kind,
            "model": self.model, "layer": self.layer,
            "M": self.M, "K": self.K, "N": self.N,
            "bits": self.bits, "group_size": self.group_size,
            "group_size_eff": self.group_size_eff, "n_groups": self.n_groups,
            "precision": self.precision, "seed": self.seed,
            "quantization_scheme": (
                f"symmetric per-group affine, INT{self.bits}, codes in "
                f"[-{qmax}, {qmax}], reproducible bit-for-bit from (seed, K, "
                "N, bits, group_size) via .quantize() -- see class docstring "
                "for why the packed codes/scale arrays are not embedded "
                "here directly."),
        }


def _make_activation(w: QuantGemmWorkload, params: dict, dtype) -> np.ndarray:
    """N(0,1) activation A[M,K], per spec's dense_operand field
    ("activations initialized N(0,1) (matches MARLIN/Tilus test-harness
    convention)"). Same seed -> same A for every impl AND the reference,
    same discipline as this module's own `_qkv`/`_make_conv_operands`."""
    seed = params.get("seed", w.seed)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((w.M, w.K)).astype(dtype)


# recommended_subset, transcribed verbatim from benchspecs/quantized-gemm/
# spec.yaml's qgemm-w4a16-decode-kernel variant (qgemm-w4a16-batched-kernel
# reuses "same recommended_subset" -- also verbatim, per the spec's own
# text) -- the union-of-4-model-families shape suite built specifically to
# close the "no shared GEMM shape suite" gap the survey documents.
_QGEMM_SHAPES = [
    dict(model="Llama-2-7B", layer="qkv_proj", K=4096, N=12288),
    dict(model="Llama-2-7B", layer="o_proj", K=4096, N=4096),
    dict(model="Llama-2-7B", layer="gate_up_proj", K=4096, N=22016),
    dict(model="Llama-2-7B", layer="down_proj", K=11008, N=4096),
    dict(model="Llama-2-13B", layer="qkv_proj", K=5120, N=15360),
    dict(model="Llama-2-13B", layer="o_proj", K=5120, N=5120),
    dict(model="Llama-2-13B", layer="gate_up_proj", K=5120, N=27648),
    dict(model="Llama-2-13B", layer="down_proj", K=13824, N=5120),
    dict(model="Llama-2-70B", layer="qkv_proj_gqa", K=8192, N=10240),
    dict(model="Llama-2-70B", layer="o_proj", K=8192, N=8192),
    dict(model="Llama-2-70B", layer="gate_up_proj", K=8192, N=57344),
    dict(model="Llama-2-70B", layer="down_proj", K=28672, N=8192),
    dict(model="Llama-3-8B", layer="qkv_proj_gqa", K=4096, N=6144),
    dict(model="Llama-3-8B", layer="gate_up_proj", K=4096, N=28672),
    dict(model="Llama-3-8B", layer="lm_head", K=4096, N=128256),
    dict(model="Mistral-7B", layer="qkv_proj_gqa", K=4096, N=6144),
    dict(model="Mistral-7B", layer="gate_up_proj", K=4096, N=28672),
    dict(model="Falcon-180B", layer="fused_attn_fc", K=14848, N=75264),
    dict(model="Falcon-180B", layer="fused_fc2", K=74240, N=14848),
    dict(model="GPT-NeoX-20B", layer="attn_qkv", K=6144, N=6144),
    dict(model="GPT-NeoX-20B", layer="mlp_up", K=6144, N=24576),
    dict(model="GPT-NeoX-20B", layer="mlp_down", K=24576, N=6144),
    dict(model="GPT-NeoX-20B", layer="lm_head", K=6144, N=50432),
]

# dense_operand M sweeps, verbatim: decode "M in {1,2,4,8,16,32}", batched
# "M in {64, 128, 256, 512, 1024, 2048, 4096}".
_QGEMM_DECODE_M = [1, 2, 4, 8, 16, 32]
_QGEMM_BATCHED_M = [64, 128, 256, 512, 1024, 2048, 4096]


def _qgemm_slug(model: str) -> str:
    return model.lower().replace(".", "").replace("_", "-")


def _build_qgemm_registry() -> dict[str, QuantGemmWorkload]:
    reg: dict[str, QuantGemmWorkload] = {}
    for shp in _QGEMM_SHAPES:
        model, layer, K, N = shp["model"], shp["layer"], shp["K"], shp["N"]
        mslug = _qgemm_slug(model)
        for M in _QGEMM_DECODE_M:
            name = f"qgemm-{mslug}-{layer}-decode-m{M}"
            reg[name] = QuantGemmWorkload(
                name=name, variant="qgemm-w4a16-decode-kernel", variant_kind="decode",
                model=model, layer=layer, M=M, K=K, N=N,
                bits=4, group_size=128, precision="fp16")
        for M in _QGEMM_BATCHED_M:
            name = f"qgemm-{mslug}-{layer}-batched-m{M}"
            reg[name] = QuantGemmWorkload(
                name=name, variant="qgemm-w4a16-batched-kernel", variant_kind="batched",
                model=model, layer=layer, M=M, K=K, N=N,
                bits=4, group_size=128, precision="fp16")
    return reg


def _smoke_qgemm() -> list[QuantGemmWorkload]:
    return [
        QuantGemmWorkload(name="smoke-qgemm-w4-g128-decode", variant="smoke",
                          variant_kind="decode", model="smoke", layer="smoke",
                          M=8, K=256, N=384, bits=4, group_size=128, precision="fp16"),
        QuantGemmWorkload(name="smoke-qgemm-w4-percol-decode", variant="smoke",
                          variant_kind="decode", model="smoke", layer="smoke",
                          M=4, K=96, N=128, bits=4, group_size=-1, precision="fp16"),
        QuantGemmWorkload(name="smoke-qgemm-w3-g64-batched", variant="smoke",
                          variant_kind="batched", model="smoke", layer="smoke",
                          M=64, K=192, N=256, bits=3, group_size=64, precision="fp16"),
    ]


# ----------------------------------------------------------------- cost rule
def _cost_qgemm(w: QuantGemmWorkload, params: dict):
    """
    flops = 2*M*N*K, the LOGICAL dense-GEMM flop count regardless of
    bit-width -- spec's metric.secondary literally ("TFLOP/s (2*M*N*K/
    time)"), the same "algorithm-invariant" convention _cost_convolution
    documents for Winograd/FFT conv: a b-bit fused-dequant kernel moves LESS
    DATA than a dense fp16 GEMM (that is the whole point) but performs the
    same logical multiply-add count, so counting the LOGICAL GEMM's flops
    (not some smaller "quantized-flop" count) is what makes a quantized
    kernel's GFLOP/s number a meaningful multiplier of the fp16 dense
    baseline it is a speedup over (spec's metric.primary).

    bytes = compulsory-traffic lower bound, each datum once: the PACKED
    weight (`K*N*bits/8`, NOT the dequantized K*N*itemsize -- reading less
    weight traffic than a dense GEMM is the memory-bound regime's entire
    reason to exist, so crediting the packed size here is not optional) +
    the scale array (n_groups*N, assumed fp16-stored, the spec's per-group
    convention) + the activation + the output, each at the run's activation
    itemsize. This diverges from MARLIN's own bench.py GB/s formula
    (`2*A+4*B+2*C+2*s numel`, counting B in 32-bit-packed-word units
    specific to its own bit-interleaved INT4 storage layout) -- this module
    uses the general packed-bit-count instead, since it must serve every
    bit-width this spec allows (INT3/INT4/FP5/FP6/INT8), not just MARLIN's
    own INT4 packing.
    """
    itemsize = ITEMSIZE[params.get("precision", w.precision)]
    flops = 2 * w.M * w.N * w.K
    w_bytes = (w.K * w.N * w.bits + 7) // 8       # packed weight, ceil to whole bytes
    scale_bytes = w.n_groups * w.N * 2            # fp16-stored scale, spec's convention
    a_bytes = w.M * w.K * itemsize
    c_bytes = w.M * w.N * itemsize
    return int(flops), int(w_bytes + scale_bytes + a_bytes + c_bytes)


workload.register_cost("quantized-gemm", _cost_qgemm, "GFLOP/s")


# ----------------------------------------------------------------- reference
def reference_qgemm(w: QuantGemmWorkload, params: dict):
    """
    fp64 `A @ dequantize(W)`. `dequantize` here is a FRESH inline expression
    (`codes.astype(np.float64) * scale[group_idx]`) -- never a call to
    `w.dequantize_fp64()` or any impl's own dequant step -- per
    DOMAIN_GUIDE's audit ruling (same formula retyped is fine, same code
    object is not). `w.quantize()` itself IS shared with every impl -- see
    QuantGemmWorkload's docstring: that sharing is this track's own fairness
    requirement (quantization error must not be part of the gate; only each
    impl's own post-quantization kernel arithmetic should be), not a
    reference-independence violation, since `.quantize()` only produces the
    INPUT both sides hold constant, exactly like this module's `_qkv`/
    `_make_conv_operands` operand generators shared between reference and
    impl elsewhere in this file.

    Activation is generated at the run's own precision first, then widened
    to fp64 (this module's established discipline -- see reference_conv/
    reference_attention), so the gate measures kernel arithmetic error, not
    activation operand-rounding error. The weight side has no equivalent
    "run precision" rounding step here: `.quantize()`'s codes/scale ARE
    already the exact quantized representation every impl consumes, so
    there is nothing further to round before dequantizing at fp64.

    scale (for harness.check_correctness's cancellation-robust denominator)
    = `|A| @ |dequantize(W)|`, `dense.py.reference_gemm`'s own `|A|@|B|`
    convention applied to the dequantized weight -- the componentwise
    backward-error bound for the PRODUCT, independent of how W's
    quantization error was introduced (that error is already baked
    identically into `dequantize(W)` on BOTH sides of the gate, since every
    impl consumes the identical quantized values -- see class docstring).

    GENUINELY-REQUIRED SMALL FIX (ARTIFACT_GUIDE.md's "do not change the
    domain module unless required, record it"; identical rule and pattern
    to `dense.py::reference_gemv`'s "dequantized_A_override" -- recorded
    here and in `bench/artifacts/quantized-gemm/fp6llm/STATUS.md`): one
    extra OPTIONAL params key, "dequantized_W_override" -- a plain (K,N)
    fp64 numpy array a caller may supply to be used VERBATIM as the
    ground-truth dequantized weight, bypassing `w.quantize()`'s own
    INTEGER codes/scale entirely. This exists for artifacts whose weight
    format is not an integer code at all: FP6-LLM/Quant-LLM's kernel
    operates on FP6 (e3m2) / FP5 (e2m2) floating MINIFLOAT weights with a
    single per-output-channel (per-N-column) scale -- a structurally
    different arithmetic regime from `QuantGemmWorkload`'s symmetric
    per-group INTEGER scheme (non-uniform mantissa spacing, not a
    uniform step size), so `w.quantize()` cannot represent it and must
    not be bent to try (this integration's explicit brief). The override
    array must be computed by the CALLER, independently of this function
    and of any code this function calls (this function performs no
    quantization of its own on the override path -- it is handed
    finished VALUES, not a formula to run), preserving the reference-
    independence rule (DOMAIN_GUIDE.md "Reference independence" ruling:
    a reference may never call the same function/helper as any
    implementation it gates -- here there is no shared function at all,
    only shared DATA the caller already computed to feed its own kernel,
    exactly `dense.py`'s precedent). Absent (every EXISTING caller,
    including `w.quantize()`-based ones): behavior is BIT-FOR-BIT
    unchanged. Popped from `params` once consumed (below), not left in
    place: a full (K,N) array is neither small nor JSON-serializable
    (`runner.py`'s `json.dump` would otherwise raise on every run that
    sets it) and this is one-shot plumbing from an adapter's `prepare()`
    to this function, called exactly once per `run_variant` (the
    correctness gate, before the warmup/timed-rep loop), so popping here
    cannot affect anything later in the same run.
    """
    dtype = _dtype(params.get("precision", w.precision))
    A = _make_activation(w, params, dtype).astype(np.float64)
    W_override = params.pop("dequantized_W_override", None)
    if W_override is not None:
        W_dequant = np.asarray(W_override, dtype=np.float64)
        if W_dequant.shape != (w.K, w.N):
            raise ValueError(
                f"reference_qgemm: dequantized_W_override shape "
                f"{W_dequant.shape} != workload shape {(w.K, w.N)}")
    else:
        codes, scale, group_idx = w.quantize()
        W_dequant = codes.astype(np.float64) * scale[group_idx]   # independent inline dequantize
    C = A @ W_dequant
    scale_out = np.abs(A) @ np.abs(W_dequant)
    return C, scale_out


# --------------------------------------------------------------------- impls
class NumpyDequantGemm:
    """
    Self-certifying CPU baseline (DOMAIN_GUIDE point 2: shares its
    'dequantize then matmul' FORMULA with reference_qgemm, so passing this
    gate does not independently verify that approach -- it exists to verify
    the harness plumbing / as a dtype-cast sanity check, not as this track's
    real subject; the real subjects are the paper artifacts under
    bench/artifacts/quantized-gemm/). Consumes the SAME shared
    `w.quantize()` codes/scale every other impl must (the fairness point,
    see QuantGemmWorkload's docstring), then dequantizes to the run's
    activation dtype (storage precision -- what a real fused-dequant kernel
    actually materializes, unlike the reference's exact-fp64 dequantize)
    INLINE, a separately-typed expression from reference_qgemm's. The matmul
    itself accumulates in fp32 -- fp16-storage/fp32-accumulate is the
    arithmetic every surveyed fused-dequant kernel in this track (MARLIN,
    Tilus, FP6-LLM, QFactory) actually implements on tensor cores, so this
    baseline's error profile approximates a real kernel's rather than a
    dense-fp64 rerun's (empirically ~1e-4 max_scaled_err across this
    module's shape range, comfortably inside the spec's parsed 1e-3 gate).

    prepare() draws A and dequantizes W ONCE; run() is exactly the timed
    matmul call.
    """

    name = "numpy-dequant-gemm"
    platform = "cpu"

    def __init__(self, precision: str = "fp16"):
        self.precision = precision
        self.dtype = _dtype(precision)

    def prepare(self, workload_: QuantGemmWorkload, params: dict):
        w = workload_
        A = _make_activation(w, params, self.dtype)
        codes, scale, group_idx = w.quantize()
        # storage-precision inline dequantize -- separately typed from
        # reference_qgemm's fp64 one, per the class docstring/audit ruling.
        W_dequant = codes.astype(self.dtype) * scale[group_idx].astype(self.dtype)
        return {"A": A.astype(np.float32), "W": W_dequant.astype(np.float32)}

    def run(self, h):
        return h["A"] @ h["W"]

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


# =============================================================== registry
_REGISTRY: dict[str, object] = {**_build_conv_registry(), **_build_attention_registry(),
                                **_build_sparse_attn_registry(), **_build_qgemm_registry()}


def load_workload(name: str):
    try:
        return _REGISTRY[name]
    except KeyError:
        raise LookupError(
            f"{name!r} is not a known ml workload; {len(_REGISTRY)} available "
            f"(conv layer shapes, attention-kernel prefill/decode shapes, "
            f"sparse-attention-kernel structured-mask shapes, and "
            f"quantized-gemm decode/batched shapes), "
            f"e.g. {sorted(_REGISTRY)[:6]} ...") from None


_SMOKE_BY_KERNEL = {
    "convolution": _smoke_convolution,
    "attention-kernel": _smoke_attention,
    "sparse-attention-kernel": _smoke_sparse_attention,
    "quantized-gemm": _smoke_qgemm,
}


def smoke_workloads(kernel: str | None = None):
    """
    Tiny, synthetic shapes that run in seconds. NOT spec-conforming (see
    DOMAIN_GUIDE.md). convolution, attention-kernel, sparse-attention-kernel
    and quantized-gemm have mutually incompatible workload types
    (ConvWorkload vs AttentionWorkload vs SparseAttentionWorkload vs
    QuantGemmWorkload), so this function is kernel-aware -- the same
    `smoke_workloads(kernel=...)` convention dense.py introduced (see its
    docstring point 4 / its own smoke_workloads for the precedent, and
    runner.py's small compatibility shim that calls it with `kernel=` when
    the domain's function accepts that parameter and falls back to a bare
    call otherwise).
    """
    fn = _SMOKE_BY_KERNEL.get(kernel, _smoke_convolution)
    return fn()


# =============================================================== registries
REFERENCES = {
    "convolution": reference_conv,
    "attention-kernel": reference_attention,
    "sparse-attention-kernel": reference_sparse_attention,
    "quantized-gemm": reference_qgemm,
}

CORRECTNESS_MODE = {
    "convolution": "max_scaled_err",
    # spec text literally says "max abs error < 1e-2"; this module applies
    # the DOMAIN_GUIDE's stronger cancellation-robust scaled gate at the
    # SAME numeric tolerance instead -- see reference_attention's docstring
    # for what the scaled denominator is here, and harness.check_correctness's
    # own docstring for why bare "max relative/abs error" text is
    # underspecified without a stated denominator in general.
    "attention-kernel": "max_scaled_err",
    # sparse-attention-kernel spec text: "max abs error < 1e-2 vs. a
    # fp32-accumulate dense masked-softmax reference" -- same scaled-gate
    # treatment, same reasoning, at the spec's own 1e-2 bound (parsed by
    # spec.py directly from the "< 1e-2" text, not hardcoded here).
    "sparse-attention-kernel": "max_scaled_err",
    # qgemm-w4a16-decode-kernel spec text: "Gate: mean(|C - C_ref|) /
    # mean(|C_ref|) < 1e-3" -- a MEAN-based gate; this module applies the
    # DOMAIN_GUIDE's stronger MAX-based cancellation-robust scaled gate at
    # the SAME parsed number (spec.py extracts tolerance=0.001 from that
    # text directly) rather than the spec's own weaker mean statistic --
    # same precedent as convolution/attention-kernel above. Empirically this
    # module's own numpy-dequant-gemm baseline lands ~1e-4, comfortably
    # inside 1e-3 even under the stronger gate (see NumpyDequantGemm's
    # docstring). qgemm-w4a16-batched-kernel's own correctness text
    # ("identical gate as qgemm-w4a16-decode-kernel (1e-3 relative L1)")
    # does NOT parse via spec.py's cross-reference regex (it says
    # "identical... as", not "same... as" -- a real spec-prose gap, not
    # something this module papers over, matching dense.py's own
    # blas-level1-2 precedent for an analogous unparsed cross-reference);
    # verify/smoke against qgemm-w4a16-decode-kernel, whose tolerance parses
    # cleanly.
    "quantized-gemm": "max_scaled_err",
}

DEFAULT_PRECISION = {"convolution": "fp32", "attention-kernel": "fp32",
                     "sparse-attention-kernel": "fp32", "quantized-gemm": "fp16"}

CPU_IMPLS = {
    "convolution": {
        "numpy-im2col-conv": NumpyIm2colConv,
        "scipy-direct-conv": ScipyDirectConv,
    },
    "attention-kernel": {
        "numpy-attention": NumpyAttention,
        "numpy-flash-attention": NumpyFlashAttention,
    },
    "sparse-attention-kernel": {
        "numpy-masked-attention": NumpySparseAttention,
    },
    "quantized-gemm": {
        "numpy-dequant-gemm": NumpyDequantGemm,
    },
}


def cuda_impls():
    """Torch-based GPU library baselines, wired but never run from here --
    the login node's GPU is shared (see DOMAIN_GUIDE.md and dense.py's/
    sparse.py's identical convention)."""
    from ..impls import gpu_cuda as g
    return {
        "convolution": {"cudnn-conv2d (torch)": g.TorchConv2d},
        "attention-kernel": {"torch-sdpa (flash/mem-efficient backend)": g.TorchAttention},
        "sparse-attention-kernel": {
            "torch-sdpa-masked (dense fallback, torch bool attn_mask)": g.TorchSparseAttention},
        "quantized-gemm": {
            "torch-dequant-then-cublas-gemm": g.TorchDequantGemm},
    }
