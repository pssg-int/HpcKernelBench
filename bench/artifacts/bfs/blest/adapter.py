"""
BLEST (Blazingly Efficient BFS using Tensor Cores) adapter for the bfs track.

Paper: "BLEST: Blazingly Efficient BFS using Tensor Cores" (ICS'26).
`PAPER_KEY = conf/ics/ElbekK26` (matched by title + artifact_url in
../../../output/included.json).
Artifact: https://github.com/delbek/blest

BLEST's only entry point is a CLI driver (source/main.cu -> Benchmark::main)
that downloads a named SuiteSparse matrix via libcurl, builds its own CSC ->
reorder -> BVSS (bit-sliced tensor-core-friendly adjacency) pipeline, then
dispatches to one of 5 kernels (BFS/MBFS/Closeness/CC/WCC) by CLI flag. Per
ARTIFACT_GUIDE.md rule 1 ("wrap the kernel, not the paper's benchmark
script"), this adapter does NOT shell out to that driver -- blest_shim.cu
(this directory, NOT part of the artifact -- same role as
bench/artifacts/triangle-counting/tot/tot_shim.cu) links directly against
BLEST's own header-heavy classes (CSC, BVSS, BFSKernel) and exposes 3
extern "C" entry points that split EXACTLY along Benchmark::run()'s own
phase boundaries for the "BFS" kernel:

  blest_prepare() -- CSC construction (from an in-memory CSR, bypassing
                      BLEST's file-based/libcurl-based loader entirely) ->
                      csc->reorder(sliceSize=8) (Jaccard-window or RCM
                      vertex reordering, BLEST's own choice) -> BVSS
                      construction (the bit-sliced, tensor-core-MMA-ready
                      adjacency format that IS this paper's data-structure
                      contribution). ALL of this is preprocessing under the
                      bfs spec ("format conversion, reordering, bitmap
                      construction ... strictly separated from per-search
                      kernel time") -- done ONCE in prepare().
  blest_run()     -- BFSKernel::singleSourceRun(source, switching=false)
                      ONLY: launches BLEST's single cooperative-groups
                      kernel (BVSSBFSKernels::BVSSBFS8Enhanced...) that
                      loops over BFS levels internally via grid.sync(),
                      decoding the BVSS bitmap with binary Tensor Core MMA
                      (`mma.sync...b1.b1...and.popc`) each level. THIS is
                      BLEST's own kernel-only timed region (its internal
                      `start = omp_get_wtime()` sits right after the
                      per-call adjacency re-upload, right before the launch
                      -- see the contamination caveat below).

Reordering caveat (documented, not a bug): BLEST's CSC::isSocialNetwork()
classifier (used to choose Jaccard-window vs. RCM reordering, and
FULL_PADDING vs. not) is normally computed by a PRIVATE power-law-tail
degree-distribution test reachable only from the file-based constructor this
adapter bypasses. blest_shim.cu reproduces just the cheap half of that
classifier's own OR-condition (average degree > BLEST's own SOCIAL_THRESHOLD
constant, 18) as a documented simplification. This choice only steers WHICH
valid reordering/kernel-variant path runs -- it cannot make the produced BFS
levels wrong, since every reorder()/kernel branch computes a correct
traversal of the same graph.

Timing-boundary caveat (documented, not a bug -- ARTIFACT_GUIDE.md rule 1's
explicit allowance for "wrap at the finest boundary available and document
the contamination"): BLEST's public per-source API is `singleSourceRun`
(re-uploads the already-reordered adjacency to the GPU on EVERY call, before
its own internal timer starts) or `multiSourceRun` (uploads ONCE, then loops
internally over a whole list of sources it is handed up front). This
harness's protocol calls `run(handle)` once per timed repetition with no
arguments, cycling through the 64-root list one root per call (see
kernelbench.domains.graph's ScipyBFS docstring for the general pattern) --
that shape only matches `singleSourceRun`, not `multiSourceRun` (which would
require handing all 64 roots to ONE `run()` call, violating the harness's
one-search-per-timed-call contract). This harness's CudaEventTimer therefore
measures singleSourceRun's per-call re-upload overhead as well as the kernel
launch, so per-search GTEPS/ms numbers from this adapter will read WORSE
than BLEST's own paper-reported numbers (which time only the launch+sync
span via BLEST's internal omp_get_wtime() pair). This is real, disclosed
contamination inherent to wrapping this artifact's public API at the finest
available per-search boundary -- not a defect in the traversal itself.

Direction-optimization disclosure (spec-mandated, see survey.md's
"Divergences" #4): no push/top-down vs. pull/bottom-up branch was found
anywhere in source/BFS/BVSSBFSKernels.cuh -- BLEST's BFS is a single unified
kernel per level (frontier-driven expansion over the BVSS bit-sliced
format), not a classic direction-optimizing hybrid. `direction_optimized =
False` below reflects that reading of the source, consistent with
survey.md's own finding that BLEST's disclosure is ambiguous in the paper
text itself.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

# Root selection is a SPEC/protocol concern, not an artifact concern: the bfs
# spec requires "same 64 fixed-seed random roots reused across all
# implementations under test" (spec.yaml). Reusing the domain's own
# `_bfs_roots` helper (the SAME one ScipyBFS.prepare() and reference_bfs()
# call) is how every impl -- CPU or GPU artifact -- ends up cycling through
# the IDENTICAL root list; this is workload/protocol setup shared
# symmetrically by design, not the kind of reference/implementation code
# sharing DOMAIN_GUIDE.md's audit ruling forbids (that ruling is about a
# correctness-gate reference sharing algorithmic code with the
# implementation it gates -- root selection is neither).
from kernelbench.domains.graph import _bfs_roots

KERNEL = "bfs"
IMPL_NAME = "blest-tensorcore-bfs"
PAPER_KEY = "conf/ics/ElbekK26"
PRECISIONS = ["int64"]  # output is an exact integer level/distance array

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "blest_shim.so")


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
    lib.blest_prepare.argtypes = [
        ctypes.c_uint, ctypes.c_ulonglong,
        ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
    ]
    lib.blest_prepare.restype = ctypes.c_void_p
    lib.blest_run.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
    lib.blest_run.restype = ctypes.c_int
    lib.blest_free.argtypes = [ctypes.c_void_p]
    lib.blest_free.restype = None
    return lib


UNSIGNED_MAX = 4294967295  # BLEST's own UNSIGNED_MAX sentinel (Common.cuh)


class BlestBFS:
    name = IMPL_NAME
    platform = "cuda"
    direction_optimized = False  # see adapter.py module docstring

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer level array; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, graph, params: dict):
        # graph.csr is already a 0/1-pattern, self-loop-free, sorted-index
        # CSR (kernelbench.domains.graph._clean_graph) -- exactly what
        # BVSS::constructFromCSCMatrix requires.
        A = graph.csr
        n = A.shape[0]
        row_ptr = np.ascontiguousarray(A.indptr, dtype=np.uint32)
        col_idx = np.ascontiguousarray(A.indices, dtype=np.uint32)
        nnz = int(A.nnz)

        # SAME root list, SAME cycling convention as ScipyBFS/reference_bfs
        # (see the module-level import comment above) -- root #0 is what the
        # harness's single-call correctness gate actually exercises.
        seed = int(params.get("bfs_root_seed", 0))
        k = int(params.get("num_roots", 64))
        roots, comp_edges, comp_verts = _bfs_roots(A, graph.directed, seed, k)
        params["bfs_root_seed"] = seed
        params["num_roots"] = int(len(roots))
        params["giant_component_vertices"] = comp_verts
        params["giant_component_edges"] = comp_edges
        params["direction_optimized"] = self.direction_optimized
        params["reordering"] = ("Jaccard-window (social-graph heuristic) or RCM "
                                "(BLEST's own CSC::reorder(), see adapter.py docstring "
                                "for the isSocialNetwork() simplification)")
        params["timing_boundary_caveat"] = (
            "singleSourceRun() re-uploads the reordered adjacency to the GPU on every "
            "call before its own internal timer starts; this harness's CudaEventTimer "
            "wraps the whole call, so measured per-search time is larger than BLEST's "
            "own paper-reported (launch+sync-only) number -- see adapter.py docstring")

        n_c = ctypes.c_uint(n)
        nnz_c = ctypes.c_ulonglong(nnz)
        handle = self.lib.blest_prepare(
            n_c, nnz_c,
            row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
            col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        )
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: blest_prepare failed (see stderr)")
        return {"handle": handle, "n": n, "out": np.empty(n, dtype=np.uint32),
                "roots": roots, "cursor": 0}

    def run(self, h):
        src = int(h["roots"][h["cursor"] % len(h["roots"])])
        h["cursor"] += 1
        out = h["out"]
        rc = self.lib.blest_run(h["handle"], ctypes.c_uint(src),
                                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)))
        if rc != 0:
            raise RuntimeError(f"{IMPL_NAME}: blest_run failed for source={src}")
        return out

    def to_host(self, out):
        level = out.astype(np.int64)
        level[out == UNSIGNED_MAX] = -1
        return level

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.blest_free(h["handle"])
        h.clear()


def create(precision: str):
    return BlestBFS(precision)
