"""
TC-Compare / GroupTC adapter for the triangle-counting track.

Paper: "A Comparative Study of Intersection-Based Triangle Counting
Algorithms on GPUs" (IPDPS'24; `conf/ipps/Li0PTZ24` in output/included.json).
Artifact: https://github.com/Jangbao/TC-Compare

TC-Compare ships NINE GPU triangle-counting kernels (8 re-collected prior
algorithms + the paper's own new one, GroupTC) behind one shared CLI/timing
harness (README, tc.cu of each `approach/*`). Per the task's guidance for
comparative-study artifacts, ONE representative kernel is wrapped here:
**GroupTC** (`approach/GroupTC/tc.cu`) -- the paper's own proposed method
(not a re-collected baseline), and per the survey the only one of the 9
whose format-conversion boundary (kernel-only, H2D done once outside the
100-iteration loop) is code-verified in detail.

tcc_shim.cu (this directory, NOT part of the artifact) #include's
`source/approach/GroupTC/tc.cu` VERBATIM (zero patches) and exposes
`tcc_count()`, which reproduces exactly one iteration of tc.cu's own
`gpu_run()` body (`cudaMemset` -> `grouptc<<<>>>` -> `cudaDeviceSynchronize`
-> `thrust::reduce`, tc.cu:159-171) -- i.e. the SAME kernel-only call the
artifact's own 100-iteration loop repeats, just returned as a value instead
of accumulated into a printf'd mean.

Preprocessing (prepare(), NOT timed as kernel time): TC-Compare's own
pipeline is SNAP2CSR (orient by first-appearance vertex order, dedup, drop
self-loops) -> CSR2RidDCSR (relabel vertices so new_id ascends with degree,
then re-orient low-new-id -> high-new-id). The ID RELABELING is NOT merely
a locality optimization here (an earlier version of this adapter assumed
that and undercounted by exactly 2x -- see STATUS.md's "Finding" section):
`grouptc`'s wedge search (tc.cu's `tb_start = i + tid + 1;
tb_len = beg_pos[src + 1] - tb_start;`) finds "src's neighbors with larger
rank than dst" not via a real search but by reading the NEXT array position
after the current edge -- which only yields the correct set when each row's
column entries are sorted by RANK, i.e. when the CSR's column *index*
ordering coincides with the degree-rank ordering. Sorting a row by raw
vertex ID (this project's default `_orient_by_degree` leaves original IDs
in place) is NOT the same ordering as sorting by rank unless vertex IDs are
first relabeled to equal rank -- which is exactly what CSR2RidDCSR.cpp does
and what this adapter now replicates directly (rank computed the same way
as kernelbench.domains.graph._orient_by_degree: ascending degree, ties
broken by original vertex id; then vertices relabeled new_id := rank, and
the DAG re-oriented new_id(u) < new_id(v)) rather than shelling out to
GroupTC's file-based preprocessing binaries (which read/write
begin.bin/source.bin/adjacent.bin, not a scipy CSR object). See STATUS.md
for the full derivation.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "triangle-counting"
IMPL_NAME = "tc-compare-grouptc"
PAPER_KEY = "conf/ipps/Li0PTZ24"
PRECISIONS = ["int64"]  # exact integer triangle count; kernel itself is integer-only (no MMA/fp)

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "tcc_shim.so")


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
    lib.tcc_prepare.argtypes = [
        ctypes.c_int, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_longlong),
    ]
    lib.tcc_prepare.restype = ctypes.c_void_p
    lib.tcc_count.argtypes = [ctypes.c_void_p]
    lib.tcc_count.restype = ctypes.c_longlong
    lib.tcc_free.argtypes = [ctypes.c_void_p]
    lib.tcc_free.restype = None
    return lib


def _rank_by_degree(A):
    """Same degree-rank kernelbench.domains.graph._orient_by_degree computes
    internally (ascending degree, ties broken by vertex id) -- duplicated
    here (not imported) because this adapter needs the raw `rank` array to
    RELABEL vertices, not just the domain's already-oriented-but-not-
    relabeled L matrix. See module docstring."""
    n = A.shape[0]
    deg = np.diff(A.indptr)
    order = np.lexsort((np.arange(n), deg))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return rank


class TcCompareGroupTC:
    name = IMPL_NAME
    platform = "cuda"
    orientation = ("degree-ascending rank; vertices relabeled new_id := rank "
                   "(matches GroupTC's own CSR2RidDCSR -- required for kernel "
                   "correctness, see adapter.py docstring)")

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer count; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, graph, params: dict):
        import scipy.sparse as sp
        from kernelbench.domains.graph import _clean_graph, _is_symmetric, _binary_csr

        A = _clean_graph(graph.csr)
        if not _is_symmetric(A):
            A = _binary_csr(A.maximum(A.T).tocsr())
            params["symmetrized_in_prepare"] = True
        else:
            params["symmetrized_in_prepare"] = False

        n = A.shape[0]
        rank = _rank_by_degree(A)

        # Relabel vertices new_id := rank, THEN orient new_id(u) < new_id(v)
        # -- see module docstring for why relabeling (not just orienting) is
        # required for grouptc's wedge-search correctness.
        coo = A.tocoo()
        ru = rank[coo.row]
        rv = rank[coo.col]
        keep = ru < rv
        L = sp.csr_matrix((np.ones(int(keep.sum())), (ru[keep], rv[keep])), shape=(n, n))
        L.sum_duplicates()
        L.sort_indices()  # ascending column (= rank) order within each row -- required
        nnz = int(L.nnz)
        indptr = L.indptr.astype(np.int64)
        col = L.indices.astype(np.int32)
        # source_list[e] = row id of edge e -- redundant expansion of indptr,
        # matching tc.cu's graph-file layout (source.bin) exactly.
        row = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))

        params["orientation"] = self.orientation
        params["oriented_edges"] = nnz

        handle = self.lib.tcc_prepare(
            n, nnz,
            row.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            col.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        )
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: tcc_prepare failed (see stderr)")
        return {"handle": handle}

    def run(self, h):
        return self.lib.tcc_count(h["handle"])

    def to_host(self, out):
        return np.array(int(out), dtype=np.int64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.tcc_free(h["handle"])
        h.clear()


def create(precision: str):
    return TcCompareGroupTC(precision)
