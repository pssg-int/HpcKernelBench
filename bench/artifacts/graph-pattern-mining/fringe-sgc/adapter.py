"""
Fringe-SGC adapter for the graph-pattern-mining track.

Paper: "Fringe-SGC: Counting Subgraphs with Fringe Vertices" (SC'25;
`conf/sc/BradleyAB25` in output/included.json).
Artifact: https://github.com/burtscher/Fringe-SGC

Headline pattern wrapped: exact K4 (4-clique) COUNTING -- k=4 ONLY (see
fc_shim.cu's file header for the full derivation). Fringe-SGC's paper
premise is counting <=3-vertex "core" motifs (vertex/edge/triangle/wedge)
with additional fringe (typically degree-1) vertices attached; K4 does not
look like an obvious fit at first (survey.md only lists those 4 core
shapes). But its own preprocessor (`fringePreprocess`, unmodified, run once
by build.sh) is a general core/fringe peeling algorithm, not hardcoded to 4
shapes: fed a K4 edge list, it reduces K4 to "triangle core + 1 fringe
vertex anchored to all 3 core corners" (verified by literally running the
artifact's own binary -- see build.sh's k4.mo generation and its logged
output: "3 core nodes", "anchor sets: A B C: 1"), and
src/fringeCount.cu's `triangleCore()` has a DEDICATED, specialized code path
for exactly this anchor pattern that dispatches to a kernel literally named
`clique`. K5/K6/K7 do NOT reduce this way (the greedy peel does not
terminate at a <=6-node core+fringe shape the way K4's does), so this
adapter raises NotImplementedError for any `clique_k` other than 4 --
GraphFold/GLumin cover k=4..7 in this track; Fringe-SGC covers k=4 only,
which is itself a real, spec-relevant finding: the paper's own framing
("current SGC approaches can only handle very small patterns") is borne out
here -- its own architecture cannot reach K5+ at all, not even as a
mis-formed/slow case, but as a shape its core/fringe decomposition provably
cannot express.

fringeCount.cu's own CliqueSolver-equivalent, `occurrences()`
(src/fringeCount.cu), and its `main()`'s own H2D-then-CPUTimer split are
wrapped per ARTIFACT_GUIDE.md rule 1 by fc_shim.cu (this directory, NOT part
of the artifact) -- see that file's header comment for the full boundary
description:

  fc_prepare() -- ECLgraph construction directly from our own CSR (no file
                  I/O), H2D copy, GPU worklist allocation, and loading the
                  K4 MatchingOrder produced once by build.sh's
                  fringePreprocess step -- matches fringeCount's own
                  pre-CPUTimer.start() scope.
  fc_count()   -- calls `occurrences(mo, d_g, d_wl, SMs, mTpSM)` ONLY (the
                  artifact's own unmodified dispatcher) -- its own
                  CPUTimer-timed region. Explicitly resets 4 `static
                  __device__` accumulator/worklist globals before each call
                  (documented in fc_shim.cu; the artifact's own code already
                  uses this exact reset idiom at another call site --
                  needed here because our harness reuses one handle across
                  warmup+reps, unlike the artifact's single-shot CLI).

Zero lines of source/ were modified.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "graph-pattern-mining"
IMPL_NAME = "fringe-sgc-triangle-core-k4clique"
PAPER_KEY = "conf/sc/BradleyAB25"
PRECISIONS = ["int64"]  # output is an exact integer count

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "fc_shim.so")
_MO_PATH = os.path.join(_HERE, "k4.mo")


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        if not os.path.exists(_MO_PATH):
            return False, f"not built: {_MO_PATH} missing (run build.sh)"
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
    lib.fc_prepare.argtypes = [
        ctypes.c_int, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
    ]
    lib.fc_prepare.restype = ctypes.c_void_p
    lib.fc_count.argtypes = [ctypes.c_void_p]
    lib.fc_count.restype = ctypes.c_ulonglong
    lib.fc_free.argtypes = [ctypes.c_void_p]
    lib.fc_free.restype = None
    return lib


class FringeSGCK4Clique:
    name = IMPL_NAME
    platform = "cuda"
    orientation = "none -- Fringe-SGC's core-search runs on the undirected CSR directly, no DAG orientation"

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer count; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()
        with open(_MO_PATH, "rb") as f:
            self._mo_bytes = f.read()

    def prepare(self, graph, params: dict):
        # graph.csr is already a 0/1-pattern, self-loop-free CSR (see
        # kernelbench.domains.graph._clean_graph). fringeCount.cu's own file
        # header states its input requirements verbatim: undirected (every
        # edge appears twice), sorted adjacency lists, no self-loops, no
        # multi-edges -- our cleaned+symmetrized+sorted CSR satisfies all of
        # these; symmetrize here if the input wasn't already, same
        # convention every other artifact adapter in this project uses.
        A = graph.csr
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"{IMPL_NAME}: graph must be square, got {A.shape}")
        AT = A.T.tocsr()
        sym = (A != AT).nnz == 0
        if not sym:
            before = int(A.nnz)
            A = A.maximum(AT).tocsr()
            A.sum_duplicates()
            A.data[:] = 1.0
            A.sort_indices()
            params["symmetrized_in_prepare"] = True
            params["directed_edges_before_symmetrize"] = before
        else:
            params["symmetrized_in_prepare"] = False
        A.sort_indices()  # fringeCount.cu requires sorted adjacency lists

        k = int(params.get("clique_k", 4))
        if k != 4:
            raise NotImplementedError(
                f"{IMPL_NAME}: Fringe-SGC's core/fringe decomposition only reduces K4 to a "
                f"<=6-node core+fringe motif (triangle core + 1 all-anchored fringe); "
                f"K5/K6/K7 do not reduce this way -- requested clique_k={k}. "
                f"See adapter.py module docstring / fc_shim.cu header.")

        n = A.shape[0]
        row_ptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        col_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
        nnz = int(A.nnz)

        params["clique_k"] = k
        params["orientation"] = self.orientation
        params["pattern_reduction"] = (
            "K4 == triangle core + 1 fringe vertex anchored to all 3 core "
            "vertices (Fringe-SGC's own fringePreprocess reduction, run "
            "once at build time -- see build.sh/k4.mo); dispatches to "
            "triangleCore()'s specialized `clique` kernel")

        mo_buf = (ctypes.c_ubyte * len(self._mo_bytes)).from_buffer_copy(self._mo_bytes)
        handle = self.lib.fc_prepare(
            n, nnz,
            row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            mo_buf, len(self._mo_bytes),
        )
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: fc_prepare failed (see stderr)")
        return {"handle": handle}

    def run(self, h):
        return self.lib.fc_count(h["handle"])

    def to_host(self, out):
        return np.array(int(out), dtype=np.int64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.fc_free(h["handle"])
        h.clear()


def create(precision: str):
    return FringeSGCK4Clique(precision)
