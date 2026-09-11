"""
Tetris (PPoPP'24) sparse-conv adapter for the convolution track.

Paper: "Tetris: Accelerating Sparse Convolution by Exploiting Memory Reuse
on GPU" (PPoPP'24). `PAPER_KEY = "conf/ppopp/LiuZYLQ24"`.
Artifact: https://github.com/XG-zheng/Tetris-artifact-evalution

Wraps Tetris's OWN SparseFilter (SPF packer) + SparseConv2d GPU kernel
(source/Tetris/spconv2d_utils.h / spconv2d_kernel.cu) through a thin
C-linkage wrapper (csrc/tetris_wrap.cu, built by build.sh into
csrc/build/libtetris.so) -- NOT the artifact's own benchmark driver
(evalution/conv2d_timing.py / fig6.sh), which subprocess-invokes a CLI
binary (spconv2d_test.o) that has an internal WARM=30/REPEAT=100 loop and
prints one aggregate mean (ARTIFACT_GUIDE.md rule 1). csrc/tetris_wrap.cu
calls Tetris's SparseFilter constructor and SparseConv2d<...> host launcher
directly -- both real, reusable entry points from the artifact's own code,
never modified (see that file's header comment).

Kernel config constraint: the wrapper calls ONE fixed instantiation,
SparseConv2d<float,int,int, K=3, S=1, TILE_IC=128, TILE_H=2, TILE_W=4> --
the exact tile config source/Tetris/spf.cc's own "ablation" driver uses, and
one of ~100 tile configs source/Tetris/spconv2d_kernel.cu pre-instantiates.
This adapter therefore only supports Kh=Kw=3, stride=1, groups=1 conv
shapes (prepare() raises NotImplementedError otherwise) -- reimplementing
spconv2d.cc's full tile-config autotuning-dispatch table was out of this
integration's budget; see csrc/tetris_wrap.cu's header comment.

Pruning granularity -- UNSTRUCTURED (global elementwise) magnitude pruning,
NOT "structured out-channel pruning" as benchspecs/convolution/spec.yaml's
conv-sparse-pruned-kernel-fp32 text states. Verified directly against
Tetris's own shipped checkpoint (source/evalution/conv_weight_data/vgg19/
vgg19-92-acc-71.7/module8_1_module3_0_conv2d_0_..._IC_256_OC_256_KS_3...):
loading that file's 256x256x3x3 filter shows exactly 90.00% of ALL individual
weight elements are zero, and 0 of 256 output channels are fully zero -- i.e.
Tetris's own "pruned" weights are fine-grained/elementwise sparse. This
matches what SparseFilter's packer actually operates on: it walks
(out_channel, in_channel, kh, kw) and packs whichever individual taps happen
to be nonzero (spconv2d_utils.h), with no code path anywhere requiring a
whole output channel to be all-zero. Per this task's own instructions ("or
whatever granularity Tetris's own SPF format actually requires; read their
code/paper for the real constraint rather than guessing"), this adapter
follows Tetris's OWN convention (global unstructured magnitude pruning at
90% -- both a spec-documented level AND the exact level measured in the
artifact's own checkpoint) rather than the spec text's "structured
out-channel" description. Documented here and in STATUS.md, not silently
substituted.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "convolution"
IMPL_NAME = "tetris-sparse-conv"
PAPER_KEY = "conf/ppopp/LiuZYLQ24"
PRECISIONS = ["fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "csrc", "build", "libtetris.so")

_SPARSITY = 0.90  # global elementwise magnitude sparsity -- see module docstring

_lib = None


def _load_lib():
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
        lib = ctypes.CDLL(_LIB_PATH)
        p = ctypes.c_void_p
        lib.tetris_pack_filter.argtypes = [p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.tetris_pack_filter.restype = p
        lib.tetris_free_filter.argtypes = [p]
        lib.tetris_free_filter.restype = None
        lib.tetris_conv_forward.argtypes = [
            p, p, p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, p,  # cuda_stream
        ]
        lib.tetris_conv_forward.restype = None
        _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "csrc/build/libtetris.so missing -- run build.sh"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        _load_lib()
    except Exception as e:
        return False, f"libtetris.so failed to load: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return TetrisSparseConv(precision)


def _prune_unstructured_magnitude(W: np.ndarray, sparsity: float) -> np.ndarray:
    """Zero the smallest-|W| `sparsity` fraction of ALL elements (global,
    fine-grained) -- see module docstring for why this granularity, not
    per-output-channel structured pruning."""
    flat = W.ravel()
    n = flat.size
    k = int(round(sparsity * n))
    if k <= 0:
        return W.copy()
    if k >= n:
        return np.zeros_like(W)
    idx = np.argpartition(np.abs(flat), k)[:k]  # k smallest-magnitude indices
    out = flat.copy()
    out[idx] = 0.0
    return out.reshape(W.shape)


class TetrisSparseConv:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp32 (Tetris's SPF kernel is a "
                f"float-only instantiation); requested {precision}")
        self.precision = precision
        self._lib = _load_lib()

    def prepare(self, workload_, params: dict):
        import torch
        w = workload_
        if not (w.Kh == 3 and w.Kw == 3 and w.stride == 1 and w.groups == 1):
            raise NotImplementedError(
                f"{IMPL_NAME}'s wrapper only calls Tetris's K=3/stride=1/"
                f"groups=1 SparseConv2d instantiation (see csrc/tetris_wrap.cu); "
                f"workload {w.name!r} is Kh={w.Kh},Kw={w.Kw},stride={w.stride},"
                f"groups={w.groups}")

        N = int(params.get("N", w.N))
        seed = params.get("seed", w.seed)
        # EXACT numpy recipe kernelbench.domains.ml._make_conv_operands uses --
        # seed, then X, then W -- so this draws the identical operands the
        # standard conv-dense-kernel-fp32 gate/reference draw for this
        # workload/seed (see the task brief / module docstring point 3).
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1.0, 1.0, size=(N, w.Cin, w.Hin, w.Win)).astype(np.float32)
        W = rng.uniform(-1.0, 1.0, size=(w.Cout, w.cin_per_group, w.Kh, w.Kw)).astype(np.float32)

        # --- Tetris's own preprocessing, timed as THIS run's preprocessing,
        # never inside run(): (a) unstructured magnitude pruning to build the
        # sparse weight this run actually multiplies (see module docstring for
        # why unstructured, not structured-out-channel); (b) SparseFilter's
        # SPF packing of that pruned weight (CPU-side format construction +
        # H2D upload of the packed offsets/position/values/oc_permutation
        # arrays) -- the artifact's own preprocessing step, per
        # ARTIFACT_GUIDE.md rule 2.
        W_pruned = _prune_unstructured_magnitude(W, _SPARSITY)

        h_filter = np.ascontiguousarray(W_pruned, dtype=np.float32)
        filter_ptr = h_filter.ctypes.data_as(ctypes.c_void_p)
        handle = self._lib.tetris_pack_filter(
            filter_ptr, ctypes.c_int(w.Cin), ctypes.c_int(w.Cout), ctypes.c_int(w.Kh))

        # NHWC input layout, the layout SparseConv2dKernel indexes with
        # (source/Tetris/spconv2d_kernel.cu's input(n,h,w,c) macro) -- built
        # once here, not per-call; a data-layout transpose, not "kernel code".
        X_nhwc = np.ascontiguousarray(X.transpose(0, 2, 3, 1), dtype=np.float32)
        d_input = torch.as_tensor(X_nhwc, device="cuda")
        out_h = (w.Hin + 2 * w.pad_h - w.Kh) // w.stride + 1
        out_w = (w.Win + 2 * w.pad_w - w.Kw) // w.stride + 1
        assert out_h == w.Hout and out_w == w.Wout, \
            f"Tetris out shape {out_h}x{out_w} != workload Hout/Wout {w.Hout}x{w.Wout}"
        d_output = torch.zeros((N, out_h, out_w, w.Cout), dtype=torch.float32, device="cuda")

        return {
            "handle": handle, "d_input": d_input, "d_output": d_output,
            "N": N, "img_h": w.Hin, "img_w": w.Win,
            "Cin": w.Cin, "Cout": w.Cout, "K": w.Kh,
            "stride": w.stride, "padding": w.pad_h,
            # kept for the standalone pruned-weight-matched correctness check
            # (see STATUS.md) -- NOT read by run()/to_host().
            "W_pruned": W_pruned, "X": X,
        }

    def run(self, h):
        self._lib.tetris_conv_forward(
            h["handle"],
            ctypes.c_void_p(h["d_input"].data_ptr()),
            ctypes.c_void_p(h["d_output"].data_ptr()),
            ctypes.c_int(h["N"]), ctypes.c_int(h["img_h"]), ctypes.c_int(h["img_w"]),
            ctypes.c_int(h["Cin"]), ctypes.c_int(h["Cout"]), ctypes.c_int(h["K"]),
            ctypes.c_int(h["stride"]), ctypes.c_int(h["padding"]),
            ctypes.c_void_p(0),  # default stream
        )
        return h["d_output"]

    def to_host(self, out):
        import torch
        # NHWC (Tetris's native output layout) -> NCHW (harness convention,
        # matching kernelbench.domains.ml.reference_conv's (N,Cout,Hout,Wout)).
        out_nchw = out.detach().to("cpu", dtype=torch.float64).permute(0, 3, 1, 2)
        return np.ascontiguousarray(out_nchw.numpy())

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        self._lib.tetris_free_filter(h["handle"])
        h.clear()
        torch.cuda.empty_cache()
