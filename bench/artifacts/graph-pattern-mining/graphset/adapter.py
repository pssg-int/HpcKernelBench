"""
GraphSet adapter for the graph-pattern-mining track.

Paper: "GraphSet: High Performance Graph Mining through Equivalent Set
Transformations" (SC'23; `conf/sc/ShiZWCZHYC23` in output/included.json).
Artifact: https://github.com/sth1997/GraphSet

Headline pattern wrapped: exact k-clique COUNTING via GraphSet's own
dedicated clique driver (`source/gpu/gpu_clique.cu`'s `main()`): a complete
graph K_k is built as a `Pattern` (`for i: for j>i: add_edge(i,j)`),
compiled via `Schedule_IEP` -- the equivalent-set-transformation schedule
search that IS GraphSet's paper contribution -- and executed by the general
`gpu_pattern_matching` kernel every GraphSet application shares, specialized
only by the compiled schedule it is handed. Matches
bench/kernelbench/domains/graph.py's graph-pattern-mining headline
(`params['clique_k']`, default 4).

`pattern_matching_init()` (gpu_clique.cu) is monolithic: it builds the
device Graph representation AND the device GPUSchedule AND launches the
kernel AND reads back the count, with no public sub-boundary. Per
ARTIFACT_GUIDE.md rule 1, `gs_shim.cu` (this directory, NOT part of the
artifact) reproduces that function's own body VERBATIM, split into:

  gs_prepare() -- Graph construction directly from our own CSR (no file
                  I/O, bypassing DataLoader's binary .g format entirely),
                  the artifact's own `reduce_edges_for_clique()` (==
                  `erase_edge()`, source/src/graph.cpp, unmodified --
                  orients every undirected edge into a DAG total-ordered by
                  vertex id), the K_k Pattern + Schedule_IEP compilation,
                  and ALL of `pattern_matching_init()`'s own H2D setup +
                  GPUSchedule device-struct construction -- exactly its own
                  pre-launch scope.
  gs_run()      -- launches ONLY `gpu_pattern_matching` (the artifact's own
                  unmodified `__global__` kernel, defined in
                  source/gpu/gpu_clique.cu and declared `extern` in
                  gs_shim.cu; the two translation units are linked with
                  `-rdc=true`, see build.sh) + reads back the count.
                  Explicitly resets `dev_sum`/`dev_cur_edge` before every
                  call (documented in gs_shim.cu; needed because this
                  harness reuses one handle across warmup+reps, unlike the
                  artifact's own single-shot CLI).

One tracked-file patch (recorded in `source.patch`, see build.sh's comment
for the full rationale): `source/gpu/component/utils.cuh`'s
`dev_alloc_and_copy`/`lower_bound` were marked `static` -- neither was
`inline`/`static` in the original header, which is harmless for the
artifact's own single-.cu-file builds but becomes an ODR "multiple
definition" error once gs_shim.cu (a SECOND .cu file that also includes
this header, needed to launch gpu_clique.cu's kernel from outside it) is
linked against gpu_clique.cu with `-rdc=true`. Pure linkage annotation, zero
arithmetic/behavior change.

Zero lines of `source/` were modified beyond that one linkage fix.

Orientation caveat: GraphSet's own `erase_edge()` orients every undirected
edge {u,v} into the single directed entry max(u,v) -> min(u,v) (a DAG
total-ordered by plain vertex id, not degree) -- a different, but equally
valid, total order from this project's own `_orient_by_degree` convention;
the exact COUNT is identical regardless of which valid total order is used.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "graph-pattern-mining"
IMPL_NAME = "graphset-schedule-iep-kclique"
PAPER_KEY = "conf/sc/ShiZWCZHYC23"
PRECISIONS = ["int64"]  # output is an exact integer count

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "gs_shim.so")


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
    lib.gs_prepare.argtypes = [
        ctypes.c_int32, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
    ]
    lib.gs_prepare.restype = ctypes.c_void_p
    lib.gs_run.argtypes = [ctypes.c_void_p]
    lib.gs_run.restype = ctypes.c_ulonglong
    lib.gs_free.argtypes = [ctypes.c_void_p]
    lib.gs_free.restype = None
    return lib


class GraphSetKCliqueCount:
    name = IMPL_NAME
    platform = "cuda"
    orientation = "vertex-id order, max->min (GraphSet's own erase_edge()/reduce_edges_for_clique())"

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer count; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, graph, params: dict):
        # graph.csr is already a 0/1-pattern, self-loop-free CSR (see
        # kernelbench.domains.graph._clean_graph). GraphSet's
        # reduce_edges_for_clique() requires a full symmetric (both-
        # directions-present) adjacency to correctly reduce to one directed
        # entry per undirected edge -- symmetrize if the input wasn't
        # already, same convention every other artifact adapter in this
        # project uses.
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
        if k < 3:
            raise NotImplementedError(
                f"{IMPL_NAME}: clique_k must be >= 3 (Pattern/Schedule_IEP need at "
                f"least a triangle to compile a non-trivial schedule); requested clique_k={k}")

        n = A.shape[0]
        vertex_ptr = np.ascontiguousarray(A.indptr, dtype=np.int64)
        edge_idx = np.ascontiguousarray(A.indices, dtype=np.int32)
        e_cnt = int(A.nnz)

        params["clique_k"] = k
        params["orientation"] = self.orientation
        params["preprocessing_includes"] = (
            "Graph construction from our CSR, VertexSet::max_intersection_size "
            "derivation, reduce_edges_for_clique() (DAG orientation), K_k "
            "Pattern + Schedule_IEP compilation (equivalent-set-transformation "
            "schedule search), and pattern_matching_init()'s own H2D setup + "
            "GPUSchedule device-struct construction -- matches "
            "pattern_matching_init()'s own pre-launch scope")

        handle = self.lib.gs_prepare(
            n, e_cnt,
            vertex_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            edge_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            k,
        )
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: gs_prepare failed (see stderr)")
        return {"handle": handle}

    def run(self, h):
        return self.lib.gs_run(h["handle"])

    def to_host(self, out):
        return np.array(int(out), dtype=np.int64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.gs_free(h["handle"])
        h.clear()


def create(precision: str):
    return GraphSetKCliqueCount(precision)
