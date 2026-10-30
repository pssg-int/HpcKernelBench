"""
Gunrock BFS -- a library BASELINE for the bfs track (ROLE = "baseline").

BLEST (ICS'26) reports 4.64x average speedup over Gunrock, so Gunrock is
BLEST's GPU-library comparator here. This is Gunrock "essentials" v2.1.0
(source.provenance), the CUDA-only release contemporary with BLEST's
comparison, called in-process through gunrock_shim.cu (this directory, not
part of the artifact) -- see its header for the exact API split.

Protocol (same as blest/efg/scipy-bfs, so rows are comparable):
  - graph.csr (0/1 pattern, no self loops, sorted) uploaded once in
    prepare() as a Gunrock device CSR; not timed.
  - the SAME 64 fixed-seed roots from the giant component
    (kernelbench.domains.graph._bfs_roots); each run() is one BFS from the
    next root.
  - Timed with CUDA events around run() (CudaEventTimer). Gunrock's own
    enactor timer (the number its example driver prints) is also recorded
    per call in params["gunrock_internal_ms"]; bfs::run allocates nothing
    large, but it does fill the distance array before the loop, which the
    harness timer includes and Gunrock's internal timer does not.
  - Output: Gunrock's distance array = BFS levels, gated exactly against
    the reference (unreached = INT_MAX -> -1).

Direction optimization: none. Gunrock v2.1.0's BFS is a push-only
(top-down) advance with block-mapped load balancing every level
(include/gunrock/algorithms/bfs.hxx).

Limit: vertex_t = edge_t = int32 (as in Gunrock's own BFS example), so
graphs with nnz >= 2^31 raise NotImplementedError (reported as unsupported).
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

from kernelbench.domains.graph import _bfs_roots

KERNEL = "bfs"
IMPL_NAME = "gunrock-bfs"
PAPER_KEY = "conf/ics/ElbekK26"   # BLEST: "4.64x over Gunrock"
ROLE = "baseline"
PRECISIONS = ["int64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "gunrock_shim.so")
INT32_MAX = 2147483647


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
    ip = ctypes.POINTER(ctypes.c_int)
    lib.gr_prepare.argtypes = [ctypes.c_int, ctypes.c_longlong, ip, ip, ctypes.c_int]
    lib.gr_prepare.restype = ctypes.c_void_p
    lib.gr_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.gr_run.restype = ctypes.c_int
    lib.gr_copy_levels.argtypes = [ctypes.c_void_p, ip]
    lib.gr_copy_levels.restype = ctypes.c_int
    lib.gr_free.argtypes = [ctypes.c_void_p]
    lib.gr_free.restype = None
    return lib


class GunrockBFS:
    name = IMPL_NAME
    platform = "cuda"
    direction_optimized = False  # push-only advance, see module docstring

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer level array; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, graph, params: dict):
        A = graph.csr
        n = A.shape[0]
        nnz = int(A.nnz)
        if nnz >= 2**31 or n >= 2**31:
            raise NotImplementedError(
                f"{IMPL_NAME}: built with int32 vertex/edge ids; graph has nnz={nnz}")
        row_ptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        col_idx = np.ascontiguousarray(A.indices, dtype=np.int32)

        seed = int(params.get("bfs_root_seed", 0))
        k = int(params.get("num_roots", 64))
        roots, comp_edges, comp_verts = _bfs_roots(A, graph.directed, seed, k)
        params["bfs_root_seed"] = seed
        params["num_roots"] = int(len(roots))
        params["giant_component_vertices"] = comp_verts
        params["giant_component_edges"] = comp_edges
        params["direction_optimized"] = self.direction_optimized
        params["gunrock_version"] = "essentials v2.1.0, block-mapped advance"
        params["gunrock_internal_ms"] = []

        ip = ctypes.POINTER(ctypes.c_int)
        handle = self.lib.gr_prepare(n, nnz, row_ptr.ctypes.data_as(ip),
                                     col_idx.ctypes.data_as(ip), int(bool(graph.directed)))
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: gr_prepare failed (see stderr)")
        return {"handle": handle, "n": n, "roots": roots, "cursor": 0, "params": params}

    def run(self, h):
        src = int(h["roots"][h["cursor"] % len(h["roots"])])
        h["cursor"] += 1
        ms = ctypes.c_float(0.0)
        rc = self.lib.gr_run(h["handle"], src, ctypes.byref(ms))
        if rc != 0:
            raise RuntimeError(f"{IMPL_NAME}: gr_run failed for source={src}")
        h["params"]["gunrock_internal_ms"].append(float(ms.value))
        return h

    def to_host(self, h):
        out = np.empty(h["n"], dtype=np.int32)
        if self.lib.gr_copy_levels(h["handle"], out.ctypes.data_as(ctypes.POINTER(ctypes.c_int))) != 0:
            raise RuntimeError(f"{IMPL_NAME}: gr_copy_levels failed")
        level = out.astype(np.int64)
        level[out == INT32_MAX] = -1
        return level

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.gr_free(h["handle"])
        h.clear()


def create(precision: str):
    return GunrockBFS(precision)
