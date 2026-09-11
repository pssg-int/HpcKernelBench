"""
Adapter for YACCLAB's BUF (Block-based Union-Find, Allegretti/Bolelli/Grana,
journals/tpds/AllegrettiBG20 -- "Optimized Block-Based Algorithms to Label
Connected Components on GPUs", TPDS 2020) GPU image connected-component
LABELING kernel, 2D 8-connectivity.

Regime note (read this before comparing to the other 3 connected-components
implementations in this benchmark): this is a DIFFERENT problem from
WCC/SCC on a sparse graph -- a regular binary PIXEL GRID, not a general
adjacency matrix. `benchspecs/connected-components/spec.yaml` scopes this
explicitly as its own `cc-image-ccl-2d3d` variant, distinct from
`cc-bcc-kernel-inmemory`/`cc-scc-kernel` (its own notes_on_fairness: "a
related-but-distinct primitive ... not ... a fourth data point on the same
underlying connectivity-kernel curve"). This adapter's `KERNEL` is still
`connected-components` (the track this paper's own centrality/regime entry
was evaluated against) but only ever accepts `Image2D` workloads --
`kernelbench.domains.graph.Image2D`, added specifically to enable this
integration (small, additive extension to that domain module; see its own
"image-CCL (additive, small)" section) -- any `Graph` workload raises
`NotImplementedError` from `prepare()`.

## Why a shim, not real OpenCV (ARTIFACT_GUIDE's explicit fallback)

YACCLAB's own CMakeLists.txt requires OpenCV (`core;imgcodecs;imgproc`, plus
the CUDA-only `cudafeatures2d` contrib module for the GPU algorithms this
adapter wraps). Checked: `pip install opencv-python-headless
--only-binary=:all:` DOES succeed (a prebuilt wheel exists), but that wheel
is a CPU-only OpenCV build with no CUDA support at all, let alone the
CUDA-enabled `cudafeatures2d` contrib module the kernel file itself
`#include`s -- satisfying that C++-level header dependency would require
building OpenCV+contrib from source with CUDA enabled, explicitly out of
scope ("YACCLAB needs OpenCV: only if importable/installable
artifact-locally without building OpenCV from source; otherwise wrap the
CUDA kernels directly with a thin shim").

So: `../shim/{opencv2/cudafeatures2d.hpp,labeling_algorithms.h,register.h}`
(this directory's sibling) provide minimal stand-ins for exactly the
identifiers `source/cuda/src/labeling_allegretti_2019_BUF.cu` itself
references (a `cv::cuda::PtrStepSz<T>`/`GpuMat`-equivalent, and a
YACCLAB-benchmark-harness-equivalent `GpuLabeling2D` base class) -- NOT
real OpenCV, but a "thin shim" wrapping the CUDA kernels directly, per
ARTIFACT_GUIDE's own suggested fallback. `bridge_yacclab_buf.cu`
`#include`s the real, unmodified `labeling_allegretti_2019_BUF.cu`
VERBATIM (`git -C source diff` is empty) -- the 4 kernels
(InitLabeling/Merge/Compression/FinalLabeling) and the `BUF` class
orchestrating them are 100% upstream code; see that file's and the shim
headers' own docstrings for the full accounting, and STATUS.md for a real
correctness bug this shim work surfaced (a missing device-memory pitch
requirement) and how it was fixed.

Direction: this is a LABELING kernel (input image -> output label image),
registered as "-image-ccl" (this domain has no "-compress"/"-decompress"
direction convention -- that is specific to the compression domain).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "connected-components"
IMPL_NAME = "yacclab-buf-image-ccl"
PAPER_KEY = "journals/tpds/AllegrettiBG20"
PRECISIONS = ["int64"]  # matches this domain's DEFAULT_PRECISION for
                        # connected-components; labels are int32 on the
                        # device, upcast to int64/float64 for the gate

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "libyacclab_buf.so")


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    p = ctypes.c_void_p
    lib.yacclab_buf_prepare.argtypes = [ctypes.POINTER(ctypes.c_uint8),
                                        ctypes.c_int, ctypes.c_int]
    lib.yacclab_buf_prepare.restype = p
    lib.yacclab_buf_run.argtypes = [p]
    lib.yacclab_buf_run.restype = None
    lib.yacclab_buf_fetch_labels.argtypes = [p, ctypes.POINTER(ctypes.c_int32)]
    lib.yacclab_buf_fetch_labels.restype = None
    lib.yacclab_buf_free.argtypes = [p]
    lib.yacclab_buf_free.restype = None
    return lib


class YacclabBufImageCcl:
    name = "yacclab-buf-image-ccl"
    platform = "cuda"

    def __init__(self, precision: str = "int64"):
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        # Local import: kernelbench.domains.graph is what OWNS the Image2D
        # workload type this adapter accepts (added specifically to enable
        # this integration -- see that module's "image-CCL" section).
        from kernelbench.domains.graph import Image2D
        if not isinstance(w, Image2D):
            raise NotImplementedError(
                f"{getattr(w, 'name', w)}: yacclab-buf-image-ccl implements binary-image "
                "connected-component LABELING (8-connectivity), not sparse-graph "
                "WCC/SCC -- pass a workload from the cc-image-ccl-2d3d variant "
                "(kernelbench.domains.graph.Image2D), not a Graph")

        img = np.ascontiguousarray(w.image, dtype=np.uint8)
        rows, cols = img.shape
        self._shape = (rows, cols)
        self._foreground_mask = img.astype(bool).ravel()
        params["algorithm_family"] = "block-based Union-Find (BUF)"
        params["connectivity_rule"] = "8-connectivity (2D, fixed per spec)"

        handle = self.lib.yacclab_buf_prepare(
            img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), rows, cols)
        if not handle:
            raise RuntimeError("yacclab_buf_prepare returned NULL")
        return {"handle": handle, "params": params}

    def run(self, h):
        # exactly one call to BUF::PerformLabeling() (upstream, unmodified)
        # -- the only call the harness's CudaEventTimer brackets.
        self.lib.yacclab_buf_run(h["handle"])
        return h

    def to_host(self, out) -> np.ndarray:
        from kernelbench.domains.graph import _canonical_labels_image
        rows, cols = self._shape
        buf = np.empty((rows, cols), dtype=np.int32)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self.lib.yacclab_buf_fetch_labels(out["handle"], ptr)
        # Canonicalize exactly like the reference (reference_image_ccl /
        # _canonical_labels_image in kernelbench.domains.graph) -- a
        # correct labeling may number components however it likes; only
        # the partition (and which pixels are background) must match.
        canon = _canonical_labels_image(buf.ravel().astype(np.int64), self._foreground_mask)
        return canon.reshape(rows, cols).astype(np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h) -> None:
        self.lib.yacclab_buf_free(h["handle"])
        h.clear()


def create(precision: str):
    return YacclabBufImageCcl(precision)
