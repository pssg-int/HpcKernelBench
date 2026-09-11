"""
STMatch adapter for the graph-pattern-mining track.

Paper: "STMatch: Accelerating Graph Pattern Matching on GPU with
Stack-Based Loop Optimizations" (SC'22; `conf/sc/WeiJ22` in
output/included.json).
Artifact: https://github.com/HPC-Research-Lab/STMatch

**Status: BUILT, gate FAILS -- see STATUS.md.** Wired into the registry per
ARTIFACT_GUIDE.md rule 4 ("failures are results; never loosen a gate"): the
mechanical wrapping below (Graph/JobQueue construction, kernel launch,
repeated-call reset) is verified independently correct (job-queue length
matches an independent Python re-implementation of the artifact's own
filter, exactly), but STMatch's own reported k-clique count does not match
ground truth on a brute-force-verified small controlled graph under either
tested optimization config, and a real off-by-one (heap out-of-bounds read)
was found in the artifact's own `PatternPreprocessor::get_labels()`
(`source/src/pattern.h`). Full account in STATUS.md.

Headline pattern wrapped: exact k-clique COUNTING via STMatch's general
stack-based pattern-matching kernel (`_parallel_match`,
`source/src/gpu_match.cu`), driven by a complete-graph K_k `Pattern` written
as STMatch's own `v`/`e` text pattern format. Matches bench/kernelbench/
domains/graph.py's graph-pattern-mining headline (`params['clique_k']`,
default 4).

`main()` (`cu_test.cu`) is monolithic exactly like every other artifact in
this track: it builds the device Graph/Pattern/JobQueue/CallStack
representations AND launches `_parallel_match` AND reads back the per-warp
counts, with no public sub-boundary. Per ARTIFACT_GUIDE.md rule 1,
`st_shim.cu` (this directory, NOT part of the artifact) reproduces that
function's own body VERBATIM, split into:

  st_prepare() -- Graph construction directly from our own CSR (no file
                  I/O, bypassing GraphPreprocessor's binary format
                  entirely), the artifact's own `PatternPreprocessor`
                  (reads the small auto-generated K_k pattern text file
                  this adapter writes -- STMatch's own schedule/set-
                  operation compiler, pattern-only preprocessing) and
                  `JobQueuePreprocessor` (builds the initial 2-vertex job
                  list, unmodified), plus ALL of `main()`'s own device
                  setup (H2D copies, CallStack/slot_storage/idle_warps/
                  global_mutex allocation+init) -- exactly its own
                  pre-launch scope.
  st_run()      -- resets every piece of per-call mutable device state
                  (needed because this harness reuses one handle across
                  warmup+reps, unlike the artifact's own single-shot CLI),
                  launches `_parallel_match` (the artifact's own
                  unmodified `__global__` kernel, compiled together with
                  this file via `-rdc=true` -- the SAME separate-
                  compilation scheme the artifact's own Makefile already
                  uses for this file), and reads back the per-warp counts.

One config-selection edit (recorded in `source.patch`; the artifact's own
Makefile automates the identical choice via a `sed -i` line-replace, see
build.sh's comment) and one register-count build flag (`-maxrregcount=64`,
needed because this kernel's BLOCK_DIM=1024 launch exceeds sm_80's
65536-register/SM budget under nvcc 12.9's optimizer, whereas the artifact's
own older-toolkit/different-arch build apparently fit). Zero lines of
kernel logic were modified.
"""

from __future__ import annotations

import ctypes
import os
import tempfile

import numpy as np

KERNEL = "graph-pattern-mining"
IMPL_NAME = "stmatch-stack-kclique"
PAPER_KEY = "conf/sc/WeiJ22"
PRECISIONS = ["int64"]  # output is an exact integer count

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "st_shim.so")


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
    lib.st_prepare.argtypes = [
        ctypes.c_int32, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p,
    ]
    lib.st_prepare.restype = ctypes.c_void_p
    lib.st_run.argtypes = [ctypes.c_void_p]
    lib.st_run.restype = ctypes.c_ulonglong
    lib.st_free.argtypes = [ctypes.c_void_p]
    lib.st_free.restype = None
    return lib


def _write_kclique_pattern_file(k: int) -> str:
    """
    STMatch's own `v`/`e` text pattern format (source/src/pattern.h's
    `PatternPreprocessor::readfile`): a complete graph K_k -- every pair of
    the k vertices connected, matching count(G, K_k)'s definition exactly.
    Written to a small temp file (a handful of bytes for k<=7); this is
    NOT the workload graph, so it doesn't violate the "no file I/O for the
    real input" boundary -- same posture as fringe-sgc's build-time-
    generated `k4.mo` MatchingOrder file in this same track.
    """
    fd, path = tempfile.mkstemp(prefix=f"stmatch_k{k}clique_", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        for i in range(k):
            f.write(f"v {i} 0\n")
        for i in range(k):
            for j in range(i + 1, k):
                f.write(f"e {i} {j} 0\n")
    return path


class STMatchKCliqueCount:
    name = IMPL_NAME
    platform = "cuda"
    orientation = "none -- STMatch's stack-based search runs on the undirected CSR directly, no DAG orientation"

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer count; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()
        self._pattern_path = None

    def prepare(self, graph, params: dict):
        # graph.csr is already a 0/1-pattern, self-loop-free CSR (see
        # kernelbench.domains.graph._clean_graph). STMatch's job-queue
        # builder assumes a full symmetric (both-directions-present)
        # adjacency -- symmetrize if the input wasn't already, same
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

        k = int(params.get("clique_k", 4))
        # source/src/config.h's PAT_SIZE=7 is a compile-time constant (the
        # artifact's own Pattern/CallStack arrays are sized off it);
        # job_queue.h's own JobQueuePreprocessor additionally indexes
        # `p.pat.degree[2]`, requiring nnodes>=3.
        if k < 3 or k > 7:
            raise NotImplementedError(
                f"{IMPL_NAME}: clique_k must be in [3,7] (source/src/config.h's "
                f"PAT_SIZE=7 compile-time bound); requested clique_k={k}")

        n = A.shape[0]
        rowptr = np.ascontiguousarray(A.indptr, dtype=np.int64)
        colidx = np.ascontiguousarray(A.indices, dtype=np.int32)
        e_cnt = int(A.nnz)

        if self._pattern_path and os.path.exists(self._pattern_path):
            os.remove(self._pattern_path)
        self._pattern_path = _write_kclique_pattern_file(k)

        params["clique_k"] = k
        params["orientation"] = self.orientation
        params["preprocessing_includes"] = (
            "Graph construction from our CSR, PatternPreprocessor (K_k pattern "
            "text file -> matching-order/partial-order/set-ops schedule "
            "compilation), JobQueuePreprocessor (initial 2-vertex job list), "
            "and main()'s own H2D setup + CallStack/slot_storage/idle_warps/"
            "global_mutex allocation -- matches main()'s own pre-launch scope")

        handle = self.lib.st_prepare(
            n, e_cnt,
            rowptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            colidx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._pattern_path.encode(),
        )
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: st_prepare failed (see stderr)")
        return {"handle": handle}

    def run(self, h):
        return self.lib.st_run(h["handle"])

    def to_host(self, out):
        return np.array(int(out), dtype=np.int64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.st_free(h["handle"])
        h.clear()
        if self._pattern_path and os.path.exists(self._pattern_path):
            os.remove(self._pattern_path)
            self._pattern_path = None


def create(precision: str):
    return STMatchKCliqueCount(precision)
