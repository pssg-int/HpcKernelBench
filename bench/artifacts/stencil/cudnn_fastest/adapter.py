"""
FlashFFTStencil's (PPoPP'25) cuDNN baseline: the same cuDNN convolution
stencil as ../cudnn (same bridge.so), but with the forward algorithm chosen
the way FlashFFTStencil's benchmarks/cudnn/cudnn-test.cpp chooses it -- time
every algorithm cuDNN accepts and keep the fastest -- instead of the
IMPLICIT_PRECOMP_GEMM that ConvStencil's programs hard-code. The search runs
once in prepare() (untimed; it shows up in the preprocessing time) and the
chosen algorithm is recorded in the result's params["cudnn_algo"].

Build: ../cudnn/build.sh (this directory has no build of its own).
"""
from __future__ import annotations

import importlib.util
import os

_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "cudnn", "adapter.py")
_spec = importlib.util.spec_from_file_location("stencil_cudnn_adapter_base", _BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

KERNEL = "stencil"
IMPL_NAME = "cudnn-stencil-fastest"
PAPER_KEY = "conf/ppopp/HanLCBZYCZCY25"   # ships benchmarks/cudnn/cudnn-test.cpp
ROLE = "baseline"
PRECISIONS = _base.PRECISIONS


def available() -> tuple[bool, str]:
    return _base.available()


class CudnnStencilFastest(_base.CudnnStencil):
    name = IMPL_NAME
    ALGO_MODE = 1


def create(precision: str):
    return CudnnStencilFastest(precision)
