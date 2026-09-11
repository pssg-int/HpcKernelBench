"""
GraphFold adapter for the graph-pattern-mining track.

Paper: "Exploiting Fine-Grained Redundancy in Set-Centric Graph Pattern
Mining" (PPoPP'24; `conf/ppopp/LinMSZXT24` in output/included.json).
Artifact: https://github.com/GPM-lib/GraphFold

Headline pattern wrapped: exact K4 (4-clique) COUNTING via GraphFold's own
CFSolver (its `CF4` kernel family, `src/CFSolver.cuh` / `src/gpu_kernels/
clique_kernels.cuh`) -- matches bench/kernelbench/domains/graph.py's
graph-pattern-mining headline (`clique_k`, default 4; see that module's
docstring deviation (4)). CFSolver supports k=4..7 (CF4-CF7); this adapter
exposes whatever `params['clique_k']` the harness passes, defaulting to 4.

GraphFold's own CLI entry points (test/test_cliquefinding.cu) always read a
Matrix-Market FILE via its own `Loader` then call `Engine::RunCF()`, which
bundles orientation + H2D copy + the CFSolver kernel into ONE call. Per
ARTIFACT_GUIDE.md rule 1 ("wrap the kernel, not the paper's benchmark
script"), gf_shim.cu (this directory, NOT part of the artifact -- same role
as bench/artifacts/triangle-counting/tot/tot_shim.cu) instead:

  gf_prepare() -- builds project_GraphFold::Graph<int,int> DIRECTLY from our
                  own CSR arrays (bypassing GraphFold's file-based Loader --
                  its Loader::Build/_load_mtx just turns (v0,v1) lines into
                  the exact same std::vector<std::pair<VID,VID>> that
                  Graph::Init() consumes, so handing Init() our own CSR's
                  (row,col) pairs directly is equivalent, minus the
                  redundant file round-trip), then calls hg->orientation()
                  (DAG construction) and hg->copyToDevice() (H2D) -- EXACTLY
                  Engine::RunCF()'s own preprocessing steps
                  (src/Engine.h: "orientation here"), done ONCE in prepare().
  gf_count()   -- calls CFSolver(*hg, k, result, 1, e_centric) ONLY: the
                  k-clique CUDA kernel itself (clique4/5/6/7_graphfold),
                  GraphFold's own "CFk matching time" timed region
                  (src/CFSolver.cuh). This is the kernel-only call run()
                  makes.

Zero lines of source/ were modified; a header-shim (compat_include/, see
build.sh) works around a CTK-12.9-vs-vendored-Thrust incompatibility, not a
kernel change.

Orientation caveat: GraphFold's own `orientation()` (src/graph/graph.h)
orients low-out-degree -> high-out-degree using EACH vertex's degree in the
*current* (undirected, doubled) edge list, tie-broken by vertex id ascending
-- textually the same "degree-ascending, ties broken by vertex id" rule this
project's own `_orient_by_degree` (kernelbench/domains/graph.py) uses. Both
are valid total orders for k-clique counting; the exact COUNT is identical
either way (only which edges are internally materialized differs), so this
does not affect correctness.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "graph-pattern-mining"
IMPL_NAME = "graphfold-cfsolver-kclique"
PAPER_KEY = "conf/ppopp/LinMSZXT24"
PRECISIONS = ["int64"]  # output is an exact integer count

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "gf_shim.so")


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
    lib.gf_prepare.argtypes = [
        ctypes.c_int, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.gf_prepare.restype = ctypes.c_void_p
    lib.gf_count.argtypes = [ctypes.c_void_p]
    lib.gf_count.restype = ctypes.c_ulonglong
    lib.gf_free.argtypes = [ctypes.c_void_p]
    lib.gf_free.restype = None
    return lib


class GraphFoldKCliqueCount:
    name = IMPL_NAME
    platform = "cuda"
    orientation = "degree-ascending, ties broken by vertex id (GraphFold's own Graph::orientation())"

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer count; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, graph, params: dict):
        # graph.csr is already a 0/1-pattern, self-loop-free CSR (see
        # kernelbench.domains.graph._clean_graph); GraphFold's own Init()
        # expects BOTH edge directions present (its .mtx graphs ship that
        # way already; _load_mtx does not symmetrize). Symmetrize here if
        # the input wasn't already, matching every other artifact adapter's
        # convention in this project (e.g. triangle-counting/tot).
        A = graph.csr
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"{IMPL_NAME}: graph must be square, got {A.shape}")
        from scipy import sparse as sp
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
                f"{IMPL_NAME}: GraphFold's CFSolver only implements CF4..CF7; requested clique_k={k}")

        n = A.shape[0]
        row_ptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        col_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
        nnz = int(A.nnz)

        params["clique_k"] = k
        params["orientation"] = self.orientation
        params["preprocessing_includes"] = (
            "Graph::Init (host CSR->edge-list build), Graph::orientation() "
            "(DAG construction), Graph::copyToDevice() (H2D) -- matches "
            "Engine::RunCF()'s own preprocessing scope")

        handle = self.lib.gf_prepare(
            n, nnz,
            row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            k,
        )
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: gf_prepare failed (see stderr)")
        return {"handle": handle}

    def run(self, h):
        return self.lib.gf_count(h["handle"])

    def to_host(self, out):
        return np.array(int(out), dtype=np.int64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.gf_free(h["handle"])
        h.clear()


def create(precision: str):
    return GraphFoldKCliqueCount(precision)
