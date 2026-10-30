"""
GAP Benchmark Suite BFS -- the CPU BASELINE for the bfs track
(ROLE = "baseline").

BLEST (ICS'26) reports 13.25x average speedup over GAP, the standard CPU
direction-optimizing BFS (Beamer et al.). This wraps GAP v1.5's own DOBFS
(source/src/bfs.cc, unmodified) through gap_shim.cc, in-process via ctypes.

Protocol (same roots/gate as blest/efg/gunrock/cugraph):
  - prepare(): graph.csr -> GAP CSRGraph (plus the transpose as GAP's
    in-edges when the graph is directed, which GAP's bottom-up step needs).
    Not timed.
  - run(): one DOBFS(g, root) from the next of the 64 fixed-seed roots.
    Timed with the harness wall clock (Timer); GAP's own Timer around the
    same call is recorded per call in params["gap_internal_ms"] (GAP's
    "Trial Time" -- includes InitParent, as GAP's own benchmark does).
  - Output: GAP's parent tree, turned into BFS levels in to_host() (untimed)
    and gated exactly against the reference.

Direction optimization: YES, GAP's hybrid with alpha=15, beta=18 (its
defaults; KB_GAP_ALPHA / KB_GAP_BETA override). Disclosed in params, as the
bfs spec requires.

Threads: OpenMP, OMP_NUM_THREADS (default: all cores the job gives it).
bench/gpu_run.sh asks SLURM for only 4 CPU cores, which would make GAP look
far slower than in BLEST's paper (2x 22-core Xeon). Run GAP on a whole CPU
node; the thread count used is recorded in params["omp_threads"].
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

from kernelbench.domains.graph import _bfs_roots
from kernelbench.harness import Timer

KERNEL = "bfs"
IMPL_NAME = "gap-dobfs"
PAPER_KEY = "conf/ics/ElbekK26"   # BLEST: "13.25x over GAP"
ROLE = "baseline"
PRECISIONS = ["int64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "gap_shim.so")


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    lp = ctypes.POINTER(ctypes.c_int64)
    ip = ctypes.POINTER(ctypes.c_int32)
    lib.gap_prepare.argtypes = [ctypes.c_int64, lp, ip, lp, ip, ctypes.c_int]
    lib.gap_prepare.restype = ctypes.c_void_p
    lib.gap_run.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_double)]
    lib.gap_run.restype = ctypes.c_int
    lib.gap_copy_parent.argtypes = [ctypes.c_void_p, ip]
    lib.gap_copy_parent.restype = ctypes.c_int
    lib.gap_num_threads.argtypes = []
    lib.gap_num_threads.restype = ctypes.c_int
    lib.gap_free.argtypes = [ctypes.c_void_p]
    lib.gap_free.restype = None
    return lib


def parent_to_level(parent: np.ndarray) -> np.ndarray:
    """BFS levels from a parent tree (parent[root] = root, -1 = unreached),
    by pointer jumping: after each round, dist[v] is the distance from v to
    anc[v]; stops when every ancestor is a root."""
    n = parent.shape[0]
    reached = parent >= 0
    anc = np.where(reached, parent, np.arange(n)).astype(np.int64)
    dist = (reached & (anc != np.arange(n))).astype(np.int64)
    while True:
        nxt = anc[anc]
        if np.array_equal(nxt, anc):
            break
        dist = dist + dist[anc]
        anc = nxt
    return np.where(reached, dist, -1)


class GapDOBFS:
    name = IMPL_NAME
    platform = "cpu"
    direction_optimized = True

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer level array; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()
        self.alpha = int(os.environ.get("KB_GAP_ALPHA", 15))
        self.beta = int(os.environ.get("KB_GAP_BETA", 18))

    def prepare(self, graph, params: dict):
        A = graph.csr
        n = A.shape[0]
        if n >= 2**31:
            raise NotImplementedError(f"{IMPL_NAME}: GAP NodeID is int32; graph has {n} vertices")
        out_off = np.ascontiguousarray(A.indptr, dtype=np.int64)
        out_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
        if graph.directed:
            T = A.T.tocsr()
            T.sort_indices()
            in_off = np.ascontiguousarray(T.indptr, dtype=np.int64)
            in_idx = np.ascontiguousarray(T.indices, dtype=np.int32)
        else:
            in_off, in_idx = out_off, out_idx

        seed = int(params.get("bfs_root_seed", 0))
        k = int(params.get("num_roots", 64))
        roots, comp_edges, comp_verts = _bfs_roots(A, graph.directed, seed, k)
        params["bfs_root_seed"] = seed
        params["num_roots"] = int(len(roots))
        params["giant_component_vertices"] = comp_verts
        params["giant_component_edges"] = comp_edges
        params["direction_optimized"] = self.direction_optimized
        params["direction_switch"] = {"alpha": self.alpha, "beta": self.beta}
        params["omp_threads"] = int(self.lib.gap_num_threads())
        params["gap_internal_ms"] = []

        lp = ctypes.POINTER(ctypes.c_int64)
        ip = ctypes.POINTER(ctypes.c_int32)
        handle = self.lib.gap_prepare(n, out_off.ctypes.data_as(lp), out_idx.ctypes.data_as(ip),
                                      in_off.ctypes.data_as(lp), in_idx.ctypes.data_as(ip),
                                      int(bool(graph.directed)))
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: gap_prepare failed (see stderr)")
        return {"handle": handle, "n": n, "roots": roots, "cursor": 0, "params": params}

    def run(self, h):
        src = int(h["roots"][h["cursor"] % len(h["roots"])])
        h["cursor"] += 1
        ms = ctypes.c_double(0.0)
        rc = self.lib.gap_run(h["handle"], src, self.alpha, self.beta, ctypes.byref(ms))
        if rc != 0:
            raise RuntimeError(f"{IMPL_NAME}: gap_run failed for source={src}")
        h["params"]["gap_internal_ms"].append(float(ms.value))
        return h

    def to_host(self, h):
        parent = np.empty(h["n"], dtype=np.int32)
        if self.lib.gap_copy_parent(h["handle"], parent.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))) != 0:
            raise RuntimeError(f"{IMPL_NAME}: gap_copy_parent failed")
        return parent_to_level(parent.astype(np.int64))

    def timer(self):
        return Timer()

    def free(self, h):
        if h.get("handle"):
            self.lib.gap_free(h["handle"])
        h.clear()


def create(precision: str):
    return GapDOBFS(precision)
