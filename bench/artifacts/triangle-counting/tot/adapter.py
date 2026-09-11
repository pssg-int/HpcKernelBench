"""
ToT (Triangle Counting on Tensor Cores) adapter for the triangle-counting
track.

Paper: "ToT: Triangle Counting on Tensor Cores" (TPDS'25, extends PPoPP'25;
`journals/tpds/ChenY25` in output/included.json).
Artifact: https://github.com/yuang-chen/ToT-TPDS25

ToT's only entry point is a CLI driver (source/apps/tot.cu) that reads a
Matrix-Market file, converts it through its own pipeline, and counts. Per
ARTIFACT_GUIDE.md rule 1 ("wrap the kernel, not the paper's benchmark
script"), this adapter does NOT shell out to that driver. Instead
tot_shim.cu (this directory, NOT part of the artifact -- same role as
bench/artifacts/spmv/sspmv/csr_shim.cpp) instantiates ToT's own header-only
template functions (tot::convert_csr_to_coo, tot::convert_undirected,
tot::extract_upper_triangular, tot::convert_coo2bmp,
tot::count_triangles_on_tensors) with concrete types and exposes two
extern "C" entry points, mirroring exactly the driver's own pipeline split:

  tot_prepare() -- CSR -> COO -> symmetrize (convert_undirected) ->
                   orientation (extract_upper_triangular) -> bitmap format
                   (convert_coo2bmp). ALL of this is preprocessing under
                   the spec (tc-gpu-kernel-exact: "orientation ... NOT
                   timed", "algorithm-native reordering ... NOT timed,
                   reported once") -- done ONCE in prepare().
  tot_count()   -- tot::count_triangles_on_tensors(bmp, bmp, bmp) ONLY:
                   the tensor-core (fp16 WMMA) masked-SpGEMM kernel, ToT's
                   own "[Counting Triangles]" timed region. This IS the
                   kernel-only call run() makes.

Orientation caveat: ToT's extract_upper_triangular orients by a FIXED
(source < target, i.e. original vertex-id) total order, NOT this project's
domain-level degree-ascending order (kernelbench.domains.graph's
_orient_by_degree). Both are valid total orders for the forward triangle-
counting algorithm -- the resulting triangle COUNT is identical either way,
only the orientation convention differs -- so this does not affect
correctness, only which edges are internally materialized. Since
extract_upper_triangular was applied, the raw count returned by
tot_count() is already exact (no /6 division needed -- matches tot.cu's own
`count = config.extract ? count : count / 6` branch with extract=1).

Precision caveat (the artifact's own, self-documented, unresolved gap):
count_triangles_on_tensors accumulates via fp16 (`half`) Tensor Core WMMA.
The README warns this "may introduce precision errors during count
summation, particularly with large graphs" and that `-e 1` (== our
extract_upper_triangular step, always applied here) "may mitigate ... in
some cases" -- not a guarantee. Per the triangle-counting spec's
correctness text and notes_on_fairness, this adapter makes NO exception:
the exact-match gate is never loosened, and if it fails on a given graph
that (implementation, graph) pair is simply INVALID, not silently reported.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "triangle-counting"
IMPL_NAME = "tot-tensorcore-tc"
PAPER_KEY = "journals/tpds/ChenY25"
PRECISIONS = ["int64"]  # output is an exact integer count; kernel internals use fp16 MMA

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "tot_shim.so")


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
    lib.tot_prepare.argtypes = [
        ctypes.c_int, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.tot_prepare.restype = ctypes.c_void_p
    lib.tot_count.argtypes = [ctypes.c_void_p]
    lib.tot_count.restype = ctypes.c_longlong
    lib.tot_free.argtypes = [ctypes.c_void_p]
    lib.tot_free.restype = None
    return lib


class TotTriangleCount:
    name = IMPL_NAME
    platform = "cuda"
    orientation = "source-id < target-id (ToT's own extract_upper_triangular; fixed order, not degree-based)"

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer count; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, graph, params: dict):
        # graph.csr is already a 0/1-pattern, self-loop-free CSR (see
        # kernelbench.domains.graph._clean_graph); may still be directed --
        # tot_prepare's own convert_undirected() symmetrizes unconditionally,
        # matching tot.cu's driver, which never skips that step either.
        A = graph.csr
        n = A.shape[0]
        row_ptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        col_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
        nnz = int(A.nnz)

        params["orientation"] = self.orientation
        params["symmetrized_by_artifact"] = True  # tot_prepare always calls convert_undirected
        params["precision_caveat"] = (
            "kernel accumulates in fp16 (Tensor Core WMMA); artifact README "
            "self-documents a precision-error risk on large graphs -- exact-"
            "match gate is NOT loosened for this (see adapter.py docstring)")

        handle = self.lib.tot_prepare(
            n, nnz,
            row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: tot_prepare failed (see stderr)")
        # keep the numpy buffers alive for the lifetime of the handle is not
        # required -- tot_prepare copies host arrays into device_vectors
        # synchronously before returning.
        return {"handle": handle}

    def run(self, h):
        return self.lib.tot_count(h["handle"])

    def to_host(self, out):
        return np.array(int(out), dtype=np.int64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.tot_free(h["handle"])
        h.clear()


def create(precision: str):
    return TotTriangleCount(precision)
