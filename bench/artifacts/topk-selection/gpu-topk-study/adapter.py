"""
gpu_topK_benchmark / GridSelect adapter for the topk-selection track.

Paper: "Parallel Top-K Algorithms on GPU: A Comprehensive Study and New
Methods" (AIR Top-K / GridSelect, SC'23; `conf/sc/ZhangNLW23` in
output/included.json).
Artifact: https://github.com/ZhangJingrong/gpu_topK_benchmark

This is a comparative-study suite shipping NINE top-k algorithms behind one
`Factory<T,idxT>` (source/benchmark/factory.h): 7 re-collected baselines
(cub, faiss_block/warp, 3 SampleSelect variants, 2 DrTopK variants) plus the
paper's own TWO new methods -- `raft_radix_11bits_extra_pass` (labeled
"AIR Top-K" in the paper's own plotting scripts) and `grid_select` (the
paper's "second method"). This adapter wraps **GridSelect**
(`source/include/grid_select.h`), chosen over AIR Top-K/raft_radix for a
concrete, disclosed reason: AIR Top-K needs the full RAFT + RMM template
libraries downloaded and patched (`third_party/download.sh` pulls
~3 separate RAPIDS repos and applies `raft.patch`) -- a heavy, fragile
build for a login-node integration pass. GridSelect's CUDA implementation,
by contrast, ships as a PREBUILT shared library already checked into this
repo (`source/third_party/libgridselect.so`, symbol
`nv::grid_select(void*, size_t&, const float*, int, int, int, float*,
int*, bool, CUstream_st*)` confirmed via `nm -D`) -- zero third-party
downloads needed, matching this task's "2 impls fine if cheap" allowance
in the direction of "1 impl is fine when the 2nd is not cheap." Only
GridSelect is wrapped (not a 2nd algorithm) -- see STATUS.md.

topk_shim.cu (this directory, NOT part of the artifact) exposes the same
two-call query/allocate/run boundary the artifact's own
`benchmark.cu::run_algo()` uses for every algorithm in its Factory (query
`buf_size` once with `buf=nullptr`, allocate once, then reuse for every
call) -- `gridselect_prepare()` does the query+allocate (this project's
`prepare()`, timed once as preprocessing), `gridselect_run()` is the
kernel-only call (this project's `run()`).

Capacity limit (found while building this adapter, not documented anywhere
in the artifact's own README/paper text -- see STATUS.md for the full
derivation): `nv::grid_select`'s prebuilt `.so` only produces a correct
result for **k <= 2048**. Verified directly: bit-exact against a numpy
full-sort reference across n in {2^15 .. 2^26} for every k <= 2048 tested;
at k=2049 the kernel silently returns a mostly-zero-filled result (only 32
of the requested k slots are ever populated -- 32 is suspiciously exactly
`WarpSize`, and `nm -D` on the artifact's `.so` shows `block_kernel<...>`
template instantiations only up to capacity 2048) instead of raising an
error. This adapter therefore REFUSES (raises `RuntimeError` in
`prepare()`, matching `bench/artifacts/spmv/diaq/adapter.py`'s
`MAX_DENSE_ELEMENTS` precedent) any request with k > 2048, rather than
silently returning wrong values that would only be caught downstream by
the correctness gate.

Data-generation divergence (disclosed, matches the existing CPU baseline's
own precedent -- kernelbench.domains.primitives.NumpyTopK's docstring):
AIR Top-K's own benchmark regenerates fresh random input EVERY timed
iteration (`TestData` re-drawn per call) specifically to avoid caching
artifacts and to let its correctness check (checked only on the last of
100 iterations in the artifact's own harness) exercise different data. This
project's harness contract calls `prepare()` ONCE then `run()` repeatedly
on the SAME handle -- there is no per-call regeneration hook. This adapter
therefore uploads ONE fixed array (from the domain's `Primitive.array`) in
prepare() and reuses it for every run() call, exactly matching
`NumpyTopK`'s own already-established behavior for the CPU baseline of
this same kernel (not a new divergence introduced here).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "topk-selection"
IMPL_NAME = "gridselect-topk"
PAPER_KEY = "conf/sc/ZhangNLW23"
PRECISIONS = ["fp32"]  # nv::grid_select is instantiated for float/int only

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "topk_shim.so")

# Empirically confirmed capacity ceiling of this artifact's prebuilt
# libgridselect.so -- see module docstring. k <= this value verified
# bit-exact vs. a numpy full-sort reference across n in {2^15..2^26}.
GRIDSELECT_MAX_K = 2048


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        try:
            import torch
        except Exception as e:
            return False, f"torch import failed: {type(e).__name__}: {e}"
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    lib.gridselect_prepare.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.gridselect_prepare.restype = ctypes.c_void_p
    lib.gridselect_run.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
    ]
    lib.gridselect_run.restype = None
    lib.gridselect_free.argtypes = [ctypes.c_void_p]
    lib.gridselect_free.restype = None
    return lib


class GridSelectTopK:
    name = IMPL_NAME
    platform = "cuda"
    batch_size = 1  # domain's Primitive/NumpyTopK model no batch dimension either

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp32 (artifact's Factory<float,int> "
                f"instantiation); requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        import torch

        k = int(w.k)
        n = int(w.array.size)
        if k > GRIDSELECT_MAX_K:
            raise RuntimeError(
                f"{IMPL_NAME}: k={k} exceeds this artifact's empirically-confirmed "
                f"capacity ceiling (k<=  {GRIDSELECT_MAX_K}; the prebuilt "
                f"libgridselect.so silently returns a mostly-zero-filled, wrong "
                f"result above this k rather than erroring -- see adapter.py/"
                f"STATUS.md). Refusing rather than letting the correctness gate "
                f"catch a known-bad configuration.")
        params["k"] = k
        params["k_fraction"] = w.k_fraction
        params["distribution"] = w.distribution
        params["batch_size"] = self.batch_size
        params["tie_rule"] = (
            "same as numpy-topk: ties at the k-th largest value resolved "
            "arbitrarily by index; only the sorted VALUE multiset is gated")

        # H2D of the fixed input array -- the only real "preprocessing" this
        # kernel class has (per the spec: "no preprocessing/format-conversion
        # step exists for this kernel class"), timed once per contract.
        d_in = torch.as_tensor(w.array, dtype=torch.float32, device="cuda").contiguous()
        d_out = torch.zeros(k, dtype=torch.float32, device="cuda")
        d_out_idx = torch.zeros(k, dtype=torch.int32, device="cuda")

        handle = self.lib.gridselect_prepare(self.batch_size, n, k)
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: gridselect_prepare failed")

        return {"handle": handle, "d_in": d_in, "d_out": d_out, "d_out_idx": d_out_idx,
               "n": n, "k": k, "torch": torch}

    def run(self, h):
        d_in_ptr = ctypes.cast(h["d_in"].data_ptr(), ctypes.POINTER(ctypes.c_float))
        d_out_ptr = ctypes.cast(h["d_out"].data_ptr(), ctypes.POINTER(ctypes.c_float))
        d_out_idx_ptr = ctypes.cast(h["d_out_idx"].data_ptr(), ctypes.POINTER(ctypes.c_int))
        self.lib.gridselect_run(
            h["handle"], d_in_ptr, self.batch_size, h["n"], h["k"],
            d_out_ptr, d_out_idx_ptr)
        return h

    def to_host(self, out):
        # Structural index-validity check (bounds/uniqueness/value-mapping),
        # matching the artifact's OWN check_result() (test_util.h) and this
        # track's CPU baseline (NumpyTopK) -- harness.check_correctness only
        # ever sees the numeric VALUE array, never indices, so validity is
        # enforced here instead, per NumpyTopK's own documented precedent.
        n, k = out["n"], out["k"]
        vals = out["d_out"].detach().to("cpu", dtype=out["torch"].float64).numpy()
        idx = out["d_out_idx"].detach().to("cpu").numpy().astype(np.int64)
        if idx.size != k or np.unique(idx).size != k or not np.all((idx >= 0) & (idx < n)):
            raise AssertionError(
                f"{IMPL_NAME}: invalid index set (bounds/uniqueness violated)")
        src = out["d_in"].detach().to("cpu", dtype=out["torch"].float64).numpy()
        if not np.array_equal(src[idx], vals):
            raise AssertionError(f"{IMPL_NAME}: returned value does not match in[idx]")
        return np.sort(vals)[::-1]

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.gridselect_free(h["handle"])
        h.clear()


def create(precision: str):
    return GridSelectTopK(precision)
