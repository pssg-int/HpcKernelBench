"""
CLOVER (spatio-graph-based exact GPU kNN) adapter for the ann-search track,
exact-spatial-knn-kernel variant.

Paper: "CLOVER: A GPU-native, Spatio-graph-based Approach to Exact kNN"
(ICS'25). `PAPER_KEY = conf/ics/KamelYC25`.
Artifact: https://github.com/ampslab/clover-knn

CLOVER's "hubs" method (source/include/bitonic-hubs.cuh, wrapped by this
directory's clover_shim.cu -- NOT part of the artifact, see its own module
docstring) is CLOVER's own paper contribution: a spatio-graph index (a fixed
set of H=2048 random "hub" points, a per-point assignment to its nearest
hub, and a sorted hub-to-hub lower-bound distance matrix) that prunes which
hubs' point buckets a query needs to scan.

STRUCTURAL FINDING (see clover_shim.cu's header comment for the full
derivation): CLOVER's own `bitonic_hubs::Query` kernel never dereferences
its `Qps` (query-point-index) argument, and its launch grid is sized by the
TOTAL point count `n`, not by the caller's query count `q`. In other words,
`bitonic_hubs::C_and_Q(n, data, q, queries, k, ...)` accepts `q`/`queries`
but its actual GPU kernel ALWAYS computes k-NN for every one of the n base
points against itself ("all-points-as-queries", exactly the convention
CLOVER's own paper evaluates under -- survey.md's own finding, now confirmed
at the kernel-argument level). There is no working code path in this
artifact for a caller-chosen, held-out query subset.

Consequence for this adapter: `run()` always asks CLOVER to self-query the
WHOLE base set (n neighbor-sets, each of size k), then slices out the rows
this workload's actual query set (`AnnWorkload.Q`) corresponds to. Row
recovery works because every ann-search exact-spatial workload in this repo
constructs Q as an EXACT copy of certain rows of X (`Q = X[q_idx].copy()`,
see kernelbench/domains/annsearch.py's smoke/uniform/bunny loaders) -- so
each query row can be matched back to its source index in X by exact byte
equality (done once in prepare(), via a dict keyed on `.tobytes()`).

DISCLOSED CONTAMINATION (ARTIFACT_GUIDE.md rule 1): because CLOVER always
does n self-queries per call regardless of how many queries the workload
nominally asks for, a QPS figure computed as (this workload's n_queries) /
(one call's wall time) UNDERSTATES the work actually performed whenever
n_queries < n_base (CLOVER did MORE search work than the numerator credits
it for), which makes its reported QPS read artificially LOW, not high, for
a held-out-query workload -- the adapter does not attempt to correct this;
it is disclosed here and in STATUS.md. QPS numbers from this adapter are
only apples-to-apples against another impl's QPS when n_queries == n_base
(the all-points-as-queries workloads this repo's loader already produces
for 'uniform-1k'/'uniform-100k'/'uniform-1m' would need n_queries raised to
n_base to remove this gap entirely; the smoke/bunny loaders use a 500-query
held-out subset and are therefore NOT a fair QPS comparison point for this
adapter, only a correctness gate point).

Index build (hub selection, per-hub distance computation, bucket sort, and
the sorted lower-bound matrix -- CLOVER's own spatio-graph index) is done
entirely in prepare(), timed once as preprocessing per DOMAIN_GUIDE; run()
launches only the Query<<<>>> search kernel.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "ann-search"
IMPL_NAME = "clover-hubs-knn"
PAPER_KEY = "conf/ics/KamelYC25"
PRECISIONS = ["fp32"]  # CLOVER's own point/distance dtype throughout

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "clover_shim.so")

EXACT_VARIANT = "exact-spatial-knn-kernel"
RECALL_VARIANT = "ann-highdim-recall-qps-pareto"


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
    lib.clover_prepare.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_float)]
    lib.clover_prepare.restype = ctypes.c_void_p
    lib.clover_query.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
    lib.clover_query.restype = None
    lib.clover_free.argtypes = [ctypes.c_void_p]
    lib.clover_free.restype = None
    return lib


class CloverKnn:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(f"{IMPL_NAME} only supports fp32; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        if w.dim != 3:
            raise NotImplementedError(
                f"{IMPL_NAME}: CLOVER's 'hubs' method hardcodes D=3 "
                f"(source/include/spatial.cuh); workload has D={w.dim}")
        X = np.ascontiguousarray(w.X, dtype=np.float32)
        n = X.shape[0]

        # Recover, for each query row, its source index in X -- see module
        # docstring: every exact-spatial workload this repo builds sets
        # Q = X[q_idx].copy(), so this is an exact byte-equality lookup, not
        # a nearest-neighbor search of its own.
        row_to_idx = {X[i].tobytes(): i for i in range(n)}
        try:
            q_idx = np.array([row_to_idx[q.astype(np.float32).tobytes()] for q in w.Q],
                             dtype=np.int64)
        except KeyError as e:
            raise NotImplementedError(
                f"{IMPL_NAME}: workload query rows are not exact copies of base-set "
                "rows (CLOVER's Query kernel only self-queries the base set -- see "
                "adapter.py module docstring); cannot map queries to base indices"
            ) from e

        params["index_build"] = (
            f"CLOVER spatio-graph index: H=2048 hubs, per-point hub assignment, "
            f"sorted H x H lower-bound matrix (n={n})")
        params["clover_self_query_note"] = (
            "CLOVER's Query kernel always computes k-NN for all n base points "
            "against themselves; this adapter slices the n*k result to this "
            "workload's actual query rows -- see adapter.py module docstring "
            "for the QPS-fairness caveat when n_queries < n_base")

        handle = self.lib.clover_prepare(
            ctypes.c_uint(n), X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: clover_prepare failed (see stderr)")

        # Stashed on self (not just in the handle dict) because to_host()
        # only receives run()'s return value, per the Implementation
        # contract -- same pattern every to_host-canonicalizing impl in
        # annsearch.py uses (self._X/self._Q/self._variant/self._gt_idx).
        self._X = X
        self._Q = np.ascontiguousarray(w.Q, dtype=np.float32)
        self._variant = w.variant
        self._gt_idx = w.gt_idx

        return {"handle": handle, "n": n, "k": w.k, "q_idx": q_idx,
                "out": np.empty(n * w.k, dtype=np.uint32)}

    def run(self, h):
        out = h["out"]
        self.lib.clover_query(h["handle"], ctypes.c_uint(h["k"]),
                              out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)))
        all_knn = out.reshape(h["n"], h["k"]).astype(np.int64)
        return all_knn[h["q_idx"]]  # (n_queries, k), this workload's rows only

    def to_host(self, out):
        # Shared cross-impl canonicalization/recall helpers from the domain
        # module -- see annsearch.py's module docstring for why sharing
        # these across IMPLEMENTATIONS (never with the reference) is fine.
        from kernelbench.domains import annsearch as A
        if self._variant == RECALL_VARIANT:
            recall = A._recall_at_k(out, self._gt_idx)
            return np.array([1.0 - recall], dtype=np.float64)
        return A._canonical_topk_distances(self._X, self._Q, out)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.clover_free(h["handle"])
        h.clear()


def create(precision: str):
    return CloverKnn(precision)
