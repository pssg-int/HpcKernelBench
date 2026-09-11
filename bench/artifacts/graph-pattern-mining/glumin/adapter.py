"""
GLumin adapter for the graph-pattern-mining track.

Paper: "GLumin: Fast Connectivity Check Based on LUTs For Efficient Graph
Pattern Mining" (PPoPP'25; `conf/ppopp/CaoMLT25` in output/included.json).
Artifact: https://github.com/AnySparse/GLUMIN

Headline pattern wrapped: exact K4 (4-clique) COUNTING via GLumin's
LUT-accelerated CliqueSolver (src/clique/clique_GM_LUT.cu -- the "+LUT"
variant of G2Miner's CliqueSolver, which IS GLumin's actual paper
contribution per survey.md: "GLumin's contribution is a drop-in
connectivity-check accelerator" layered on G2Miner/GraphFold/AutoMine, as
opposed to the plain non-LUT `clique_GM.cu` G2Miner baseline also shipped in
this repo). Matches bench/kernelbench/domains/graph.py's graph-pattern-mining
headline (`clique_k`, default 4). CliqueSolver supports k=4..7
(clique4/5/6/7_warp_*_subgraph kernels); this adapter exposes whatever
`params['clique_k']` the harness passes, defaulting to 4.

GLumin's own CliqueSolver() (src/clique/clique_GM_LUT.cu) is monolithic: it
builds `GraphGPU gg(g); gg.init_edgelist(g);` (H2D) AND allocates its whole
kernel workspace (frontier_list, BinaryEncode LUT buffer, d_total) INSIDE the
same call that launches the counting kernel, with no public sub-boundary.
Per ARTIFACT_GUIDE.md rule 1 ("wrap at the finest boundary available"),
gl_shim.cu (this directory, NOT part of the artifact) reproduces
CliqueSolver's own setup code verbatim, split across two functions instead
of one -- see gl_shim.cu's file header for the full explanation:

  gl_prepare() -- Graph construction (via the artifact's own public
                  allocateFrom/fixEndEdge/constructEdge API -- the same
                  pattern the artifact's own src/common/graph_partition.cc
                  uses), orientation() (DAG), GraphGPU/init_edgelist (H2D),
                  and ALL of CliqueSolver's own pre-kernel workspace
                  allocation (frontier_list, BinaryEncode LUT, launch-config
                  arithmetic) -- exactly CliqueSolver's own untimed region.
  gl_count()   -- launches ONLY the cliqueK_warp_*_subgraph kernel (the
                  artifact's own unmodified __global__ function) + D2H
                  readback -- CliqueSolver's own timed region. Explicitly
                  resets the accumulator before each call (documented in
                  gl_shim.cu; needed because our harness reuses one handle
                  across warmup+reps, unlike the artifact's single-shot CLI).

Zero lines of source/ were modified.

Orientation caveat: GLumin's own Graph::orientation() (src/common/graph.cc)
orients low-out-degree -> high-out-degree using each vertex's CURRENT
(undirected) degree, ties broken by vertex id ascending -- textually the
same "degree-ascending, ties broken by vertex id" rule this project's own
`_orient_by_degree` uses. The exact COUNT is identical regardless of which
valid total order is used, so this does not affect correctness.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "graph-pattern-mining"
IMPL_NAME = "glumin-cliquesolver-lut-kclique"
PAPER_KEY = "conf/ppopp/CaoMLT25"
PRECISIONS = ["int64"]  # output is an exact integer count

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "gl_shim.so")


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
    lib.gl_prepare.argtypes = [
        ctypes.c_int, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.gl_prepare.restype = ctypes.c_void_p
    lib.gl_count.argtypes = [ctypes.c_void_p]
    lib.gl_count.restype = ctypes.c_ulonglong
    lib.gl_free.argtypes = [ctypes.c_void_p]
    lib.gl_free.restype = None
    return lib


class GLuminKCliqueCount:
    name = IMPL_NAME
    platform = "cuda"
    orientation = "degree-ascending, ties broken by vertex id (GLumin's own Graph::orientation())"

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer count; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, graph, params: dict):
        # graph.csr is already a 0/1-pattern, self-loop-free CSR (see
        # kernelbench.domains.graph._clean_graph). GLumin's CliqueSolver
        # requires a DAG built from an UNDIRECTED input (its own
        # constructor asserts `!directed` before calling orientation()), so
        # symmetrize here if the input wasn't already -- same convention
        # every other artifact adapter in this project uses.
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

        k = int(params.get("clique_k", 4))
        if k < 4 or k > 7:
            raise NotImplementedError(
                f"{IMPL_NAME}: GLumin's CliqueSolver only implements k=4..7; requested clique_k={k}")

        n = A.shape[0]
        row_ptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        col_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
        nnz = int(A.nnz)

        params["clique_k"] = k
        params["orientation"] = self.orientation
        params["preprocessing_includes"] = (
            "Graph construction (host CSR->allocateFrom/fixEndEdge/"
            "constructEdge), Graph::orientation() (DAG), GraphGPU+"
            "init_edgelist (H2D), and CliqueSolver's own kernel-workspace "
            "allocation (frontier_list, BinaryEncode LUT buffer) -- matches "
            "CliqueSolver()'s own pre-Timer.Start() scope")

        handle = self.lib.gl_prepare(
            n, nnz,
            row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            k,
        )
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: gl_prepare failed (see stderr)")
        return {"handle": handle}

    def run(self, h):
        return self.lib.gl_count(h["handle"])

    def to_host(self, out):
        return np.array(int(out), dtype=np.int64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.gl_free(h["handle"])
        h.clear()


def create(precision: str):
    return GLuminKCliqueCount(precision)
