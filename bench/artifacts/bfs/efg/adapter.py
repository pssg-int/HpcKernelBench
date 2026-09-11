"""
efg (Elias-Fano-compressed-Graph BFS) adapter for the bfs track.

Paper: "Traversing Large Compressed Graphs on GPUs" (IPDPS'23).
`PAPER_KEY = conf/ipps/GeraK23` (matched by title + artifact_url in
../../../output/included.json).
Artifact: https://github.com/pgera/efg

efg's only entry point is a CLI driver (source/src/main.cu) that reads a
CSR graph from two binary files (vertex.CSR / edge.CSR) in a directory,
builds an Elias-Fano-compressed representation on the CPU (folly's
EliasFanoEncoder), uploads it to the GPU, then runs `num_traversals`
BFS searches from random/fixed roots. Per ARTIFACT_GUIDE.md rule 1 ("wrap
the kernel, not the paper's benchmark script"), this adapter does NOT shell
out to that driver -- efg_shim.cu (this directory, NOT part of the
artifact) links directly against efg's own header-heavy classes (CSR,
EFLayout, EFGraph via CUEFGraph, BFS) and exposes extern "C" entry points
that split exactly along main.cu's own phase boundaries (main.cu lines
68-128):

  efg_prepare() -- CSR(dir) [load] -> EFLayout<0,512>(csr) [compute EF byte
                   sizes] -> CUEFGraph<0,512>(ef_layout, alloc_mode) [THE
                   COMPRESSION STEP: EFGraph's constructor Elias-Fano-
                   encodes every vertex's neighbor list via folly's real,
                   unmodified EliasFanoEncoder (source/src/ef_graph.h
                   encode_nbr_list), then uploads the compressed
                   representation to device memory] -> BFS<0,512>(cu_ef_
                   graph, sort_frontier) [allocate device scratch buffers].
                   ALL of this is preprocessing under the bfs spec ("format
                   conversion ... strictly separated from per-search kernel
                   time") -- efg's own README/paper already reports it
                   separately too (CSR size / EF size / compression ratio
                   printed once, matching this project's own
                   preprocessing_reported convention). Represents a graph
                   as, per vertex: a forward-pointer skip table (every
                   kForwardQuantum=512th cumulative gap count, for fast
                   random access into the bitstream) + Elias-Fano upper
                   bits (unary-coded gaps between sorted neighbor ids) +
                   lower bits (fixed-width remainder bits) -- the classical
                   Elias-Fano monotone-sequence encoding applied to each
                   vertex's sorted adjacency list.
  efg_run()     -- BFS<0,512>::traverse(source) ONLY: the GPU top-down
                   frontier-expansion kernel (source/src/bfs_kernels.cuh),
                   which decodes the Elias-Fano bitstream ENTIRELY ON-
                   DEVICE with a hand-rolled bit-select routine (NOT
                   folly -- folly is CPU-only in this artifact, used only
                   during the encode step above; confirmed by grep, no
                   folly:: symbol appears in any __device__/__global__
                   function). This IS efg's own timed region
                   (BFS::traverse_impl wraps exactly this in a cudaEvent
                   pair, matching `bfs.get_last_elapsed_time()` in main.cu).
                   Returns the LEVEL/distance array directly -- efg was
                   ALREADY level-array-native (its own "distances" output),
                   unlike blest/bit-graphblas.

CSR-loading caveat (documented, not a bug): efg's CSR class only has a
file-based constructor (source/src/csr.cpp) -- there is no in-memory
constructor in the artifact, and this adapter does not add one (that would
mean patching a source file, ARTIFACT_GUIDE.md rule 3). Instead prepare()
below writes the graph's CSR to a FRESH temp directory as vertex.CSR /
edge.CSR (uint64, exactly the format source/README.md documents) and hands
efg_prepare() that directory -- this IS still "the artifact's own format
conversion" inside prepare() (ARTIFACT_GUIDE.md rule 2), just routed
through a temp directory because that's the only loading path CSR offers.
The temp directory is created/destroyed per prepare()/free() pair and never
reused across graphs.

Dependency caveat (see STATUS.md for full detail): efg's own build requires
Facebook folly (built+installed), boost, glog, double-conversion, fmt, and
openssl -- none available on this machine and none installable without
sudo/system packages (ARTIFACT_GUIDE.md: "No sudo; no system package
installs"). Tracing the actual header-level dependency graph of the ONE
folly feature efg uses (folly::compression::EliasFanoEncoder, plus its
Select64/Instructions helpers) showed it resolves to a bounded, genuinely
header-only-compatible closure of ~30 upstream folly headers (bit
manipulation, portability macros, hashing) plus ONE further external
dependency: glog's CHECK/DCHECK/VLOG macros (used only for internal
invariant assertions inside folly's own EliasFanoCoding.h, never in efg's
own code, and never touching the actual encode/decode arithmetic). Building
glog itself needs gflags -- avoided by vendoring the real, unmodified
upstream folly headers (folly_vendor/folly/, pinned to folly's
v2023.05.22.00 tag, matching efg's own README's validated version) plus a
minimal, from-scratch, drop-in-compatible glog/logging.h macro shim
(folly_vendor/glog/logging.h) that reproduces glog's OBSERVABLE macro
semantics exactly. See folly_vendor/glog/logging.h's own docstring and
STATUS.md for the full provenance and scope of what was vendored/written.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import tempfile

import numpy as np

from kernelbench.domains.graph import _bfs_roots

KERNEL = "bfs"
IMPL_NAME = "efg-elias-fano-bfs"
PAPER_KEY = "conf/ipps/GeraK23"
PRECISIONS = ["int64"]  # output is an exact integer level/distance array

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "efg_shim.so")

# main.cu's own compile-time defaults (BFS<>/EFLayout<> are templated on
# these, so they cannot be adapter-level runtime parameters -- see
# efg_shim.cu's file docstring).
K_SKIP_QUANTUM = 0
K_FORWARD_QUANTUM = 512


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
    lib.efg_prepare.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.efg_prepare.restype = ctypes.c_void_p
    for fn in ("efg_num_vertices", "efg_csr_storage_bytes",
              "efg_csr_optimal_bytes", "efg_ef_storage_bytes"):
        getattr(lib, fn).argtypes = [ctypes.c_void_p]
        getattr(lib, fn).restype = ctypes.c_ulonglong
    lib.efg_run.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong,
                            ctypes.POINTER(ctypes.c_ulonglong)]
    lib.efg_run.restype = ctypes.c_int
    lib.efg_free.argtypes = [ctypes.c_void_p]
    lib.efg_free.restype = None
    return lib


UINT64_MAX = np.iinfo(np.uint64).max  # efg's own sentinel (std::numeric_limits<size_t>::max())


class EfgBFS:
    name = IMPL_NAME
    platform = "cuda"
    direction_optimized = False  # pure top-down frontier expansion; no
                                 # push/pull branch anywhere in bfs.cuh/
                                 # bfs_kernels.cuh (confirmed by reading)

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer level array; requested {precision}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, graph, params: dict):
        A = graph.csr
        n = A.shape[0]

        # efg's CSR format: vertex.CSR = (n+1) uint64 row offsets,
        # edge.CSR = nnz uint64 column indices -- see source/README.md
        # "Input Graph Format". This IS the artifact's own format
        # conversion (writing our CSR into ITS on-disk representation),
        # done once here in prepare() (ARTIFACT_GUIDE.md rule 2); see
        # adapter.py module docstring for why a temp directory is used
        # instead of an in-memory array.
        vlist = np.ascontiguousarray(A.indptr, dtype=np.uint64)
        elist = np.ascontiguousarray(A.indices, dtype=np.uint64)
        tmpdir = tempfile.mkdtemp(prefix="efg_csr_")
        vlist.tofile(os.path.join(tmpdir, "vertex.CSR"))
        elist.tofile(os.path.join(tmpdir, "edge.CSR"))

        # SAME root list, SAME cycling convention as ScipyBFS/reference_bfs
        # (spec.yaml requires "same 64 fixed-seed random roots reused
        # across all implementations under test" -- explicitly REPLACES
        # efg's own 100-root convention, see benchspecs/bfs/spec.yaml's
        # root_selection text and notes_on_fairness).
        seed = int(params.get("bfs_root_seed", 0))
        k = int(params.get("num_roots", 64))
        roots, comp_edges, comp_verts = _bfs_roots(A, graph.directed, seed, k)
        params["bfs_root_seed"] = seed
        params["num_roots"] = int(len(roots))
        params["giant_component_vertices"] = comp_verts
        params["giant_component_edges"] = comp_edges
        params["direction_optimized"] = self.direction_optimized
        params["ef_quantums"] = {"kSkipQuantum": K_SKIP_QUANTUM,
                                 "kForwardQuantum": K_FORWARD_QUANTUM}

        handle = self.lib.efg_prepare(tmpdir.encode("utf-8"), 0, 1)  # use_uvm=0, sort_frontier=1 (main.cu defaults)
        if not handle:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(f"{IMPL_NAME}: efg_prepare failed (see stderr)")

        csr_bytes = self.lib.efg_csr_storage_bytes(handle)
        csr_optimal_bytes = self.lib.efg_csr_optimal_bytes(handle)
        ef_bytes = self.lib.efg_ef_storage_bytes(handle)
        params["csr_storage_bytes"] = int(csr_bytes)
        params["csr_optimal_bytes"] = int(csr_optimal_bytes)
        params["ef_storage_bytes"] = int(ef_bytes)
        params["compression_ratio_vs_optimal_csr"] = (
            float(csr_optimal_bytes) / float(ef_bytes) if ef_bytes else None)

        return {"handle": handle, "n": n, "tmpdir": tmpdir,
                "out": np.empty(n, dtype=np.uint64), "roots": roots, "cursor": 0}

    def run(self, h):
        src = int(h["roots"][h["cursor"] % len(h["roots"])])
        h["cursor"] += 1
        out = h["out"]
        rc = self.lib.efg_run(h["handle"], ctypes.c_ulonglong(src),
                              out.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)))
        if rc != 0:
            raise RuntimeError(f"{IMPL_NAME}: efg_run failed for source={src}")
        return out

    def to_host(self, out):
        level = out.astype(np.int64)
        level[out == UINT64_MAX] = -1
        return level

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.efg_free(h["handle"])
        if h.get("tmpdir"):
            shutil.rmtree(h["tmpdir"], ignore_errors=True)
        h.clear()


def create(precision: str):
    return EfgBFS(precision)
