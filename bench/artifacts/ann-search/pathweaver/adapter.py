"""
PathWeaver (multi-GPU graph-based ANNS) adapter for the ann-search track,
ann-highdim-recall-qps-pareto variant -- SEARCH KERNEL ONLY, single GPU.

Paper: "PathWeaver: A High-Throughput Multi-GPU System for Graph-Based
Approximate Nearest Neighbor Search" (USENIX ATC'25).
`PAPER_KEY = conf/usenix/KimPNHKLL25`.
Artifact: https://github.com/AIS-SNU/PathWeaver

SCOPE (why this wraps only search(), never the paper's build/multi-GPU path)
-----------------------------------------------------------------------------
PathWeaver's headline contribution is a MULTI-GPU pipeline (pipelining-based
path extension across GPUs via NVLink, per the paper's abstract); its own
README requires 4 GPUs + NVLink + CUDA 12.1 + a multi-hundred-GB dataset
download. That is out of this integration's single-GPU, login-node, no-
large-download scope. However, the artifact also ships a genuine single-GPU
driver, `source/pathweaver/single_pathweaver_one.py`, which calls exactly
ONE exported function, `pathweaver.search(...)` (built from
`csrc/pathweaver.cu`, PYBIND11-exposed as `search`, pathweaver.cu:2200-2201)
-- this IS the paper's actual GPU search kernel (a CAGRA-family single-CTA
greedy graph-traversal kernel: `search_kernel<VECTOR_DIM, INTERNAL_TOPK>`,
pathweaver.cu:1310-1636), independent of how many GPUs orchestrate calls to
it. This adapter wraps exactly that kernel, called once per run() over the
full query batch, per ARTIFACT_GUIDE rule 1.

The GRAPH is not built by this kernel. In PathWeaver's own pipeline the
graph comes from RAFT/cuVS's CAGRA index-construction algorithm
(`csrc/cagra_graph/`, a `rapids-cmake` CPM fetch of RAPIDS RAFT -- an hours-
long, network-heavy build with its own optimization objective distinct from
"be a valid input to search()"). Building CAGRA's graph constructor is
explicitly OUT OF SCOPE here (network/build-time budget; see build.sh and
STATUS.md). Instead, `prepare()` below builds its own, independently-
computed EXACT k-NN graph over the workload's base set, of the SAME fixed
degree PathWeaver's search kernel requires (see GRAPH_DEGREE below) --
timed once as preprocessing, per DOMAIN_GUIDE. This means what this adapter
measures is PathWeaver's SEARCH KERNEL's correctness/recall behavior on a
graph of matching shape, NOT PathWeaver's (or CAGRA's) own graph-quality
contribution -- an exact k-NN graph is not a CAGRA-optimized navigable
graph, and recall achieved here says nothing about how PathWeaver performs
on its own paper's graphs. This substitution and its fairness implications
are recorded in STATUS.md; it is the same category of documented
substitution ARTIFACT_GUIDE rule 1 asks for when an artifact's full
pipeline cannot be reproduced end-to-end.

HARD KERNEL CONSTRAINTS DISCOVERED (compile-time, not runtime-tunable)
-----------------------------------------------------------------------------
1. VECTOR_DIM is a C++ TEMPLATE parameter. `search()`'s host-side dispatch
   (pathweaver.cu:1732-2158, a `switch(VECTOR_DIM)`) only instantiates
   `search_kernel<D, *>` for D in {96, 100, 128, 256, 768, 960} (the
   dimensions of the ann-benchmarks datasets PathWeaver's own paper
   evaluates: sift-128, gist-960, glove-100, deep-256, glove-96 [sic,
   comment says "glove-100-inner" at D=100 and an unlabeled D=96 case],
   nytimes-768). Any other VECTOR_DIM value falls through to `default:`,
   which calls `assert(false)` ON THE HOST -- confirmed experimentally: this
   aborts the whole Python process, not a catchable exception. This
   module's smoke workloads use D=32/64 (kernelbench/domains/annsearch.py's
   `_make_recall_smoke`), neither of which is wired. `_padded_dim()` below
   therefore zero-pads every base/query vector up to the smallest wired D
   that is >= the workload's own D. Zero-padding is DISTANCE-PRESERVING:
   for Euclidean L2, appending identical zero coordinates to every vector
   being compared contributes exactly 0 to every pairwise squared
   difference, so recall computed on the padded representation is
   IDENTICAL, not merely close, to recall on the original D-dim vectors --
   confirmed by inspecting the distance kernel itself
   (`compute_similarity_vector_load_full`, pathweaver.cu:510-567: a plain
   sum of `(query[k]-data[k])^2` over `VECTOR_DIM` coordinates, no
   normalization or dimension-dependent scaling that padding could disturb).
   Padding is done in this adapter (prepare()), never in the domain module
   or the reference -- the workload's true dimensionality is unchanged;
   only this implementation's on-device representation is.
2. INTERNAL_TOPK is ALSO a template parameter, wired for {64, 128} only
   (same switch). This adapter always uses 64 (the smaller of the two,
   valid whenever the workload's k <= 64, true for every workload this repo
   builds -- see the explicit check in prepare()).
3. CANDIDATE_BUFFER_SIZE (= SEARCH_WIDTH * GRAPH_DEGREE at RUNTIME) is
   asserted to equal EXACTLY 64 by three separate device functions
   (`candidate_by_bitonic_sort`, `candidate_by_bitonic_sort_inverse`,
   `topk_by_bitonic_sort` -- pathweaver.cu:255-258, 305-308, 358-361: `if
   (CANDIDATE_BUFFER_SIZE != 64) { printf(...); assert(false); }`, a
   DEVICE-side assert hit on every iteration of the main search loop). This
   is a hard-coded property of the kernel's warp-level bitonic sort network
   (32 lanes x 2 elements/lane), not a tunable "config" despite GRAPH_DEGREE
   nominally being a runtime `configs[]` entry -- discovered experimentally
   (an initial attempt with GRAPH_DEGREE=32 produced this exact device
   printf on every launch). With SEARCH_WIDTH=1 (the artifact's own
   driver's fixed value, `single_pathweaver_one.py:71`), GRAPH_DEGREE MUST
   be exactly 64. This adapter's independently-built k-NN graph therefore
   uses degree 64 -- conveniently, also PathWeaver's own paper's CAGRA
   `build_graph_degree` for sift-128 (survey.md's own reading of
   `single_pathweaver_all.sh`), so the substitution graph's DEGREE matches
   the paper's own convention even though its CONSTRUCTION algorithm does
   not (see the SCOPE section above).
4. TEAM_SIZE must be exactly 8 (`search_kernel`'s own first check,
   pathweaver.cu:1344-1348, host-visible `assert(false)` inside the kernel
   otherwise) -- always passed as 8, matching the driver.

SIMPLEST-VALID SINGLE-PASS CONFIGURATION (no ghost stage, no sign-bit prune)
-----------------------------------------------------------------------------
Per this integration's own brief: this adapter does NOT reproduce
PathWeaver's two-phase "ghost search" (a fast pre-search over a subsampled
graph used only to pick a good entry point for the real search -- itself
just TWO MORE calls to the same `search()` kernel this adapter already
wraps, so skipping it changes seeding quality, not which kernel is under
test) or its SIGN_BIT_PRUNE direction-guided pruning feature (a search-
quality optimization requiring precomputed per-edge sign bits, built by the
separate `cpu_generate_sign_bit` extension this build.sh deliberately does
not build). Instead `run()` issues ONE `search()` call per query batch,
configured exactly like the driver's own "ghost pass"
(`single_pathweaver_one.py:140-166`): PRUNE_CONFIG=NO_PRUNE,
THRESHOLD_CONFIG=NO_THRESHOLD, an empty (0,0) sign_bit tensor,
SEED_CONFIG=USE_SEED with a TRIVIAL seed (all-zero `top10`/
`initial_starting_point` + an IDENTITY `seed_map`) -- which, per
`compute_distance_to_maped_top10_nodes` (pathweaver.cu:766-850), simply
means every query's greedy graph walk starts from node 0's neighbor list
(`top1 = top10_ptr[query_id] == 0`, `parent_id = seed_map_ptr[0] == 0`), a
fixed deterministic entry point -- the same category of simplification
CAGRA/HNSW-style search commonly uses (a single fixed "navigation node").
HASH_TABLE_CONFIG=USE_HASH_TABLE (visited-node dedup, a pure efficiency
optimization with no effect on which candidates are found, matching the
driver's own default). Empirically (see STATUS.md), this single-pass
configuration with MAX_ITER=48 clears this module's recall@10 >= 0.9 floor
on both smoke instances.

CONFIG SLOT LAYOUT (configs[] tensor, matching pathweaver.cu:1680-1712 and
single_pathweaver_one.py:140-165's own array literal exactly):
  0 NUM_QUERIES  1 TOPK(=k)      2 SEARCH_WIDTH   3 MAX_ITER
  4 MIN_ITER     5 INTERNAL_TOPK 6 VECTOR_DIM      7 DATASET_SIZE
  8 TEAM_SIZE    9 GRAPH_DEGREE  10 NUM_DIST       11 BLOCK_SIZE
  12 BITLEN      13 SMALL_HASH_RESET_INTERVAL      14 SHARED_MEM_SIZE
  15 PRUNE_CONFIG            16 ITERATION_DIRECTION_RATIO (unused, NO_PRUNE)
  17 PRUNE_RATIO (unused, NO_PRUNE)  18 THRESHOLD_CONFIG  19 FULL_COMPUTE_RATIO
  20 SEED_CONFIG  21 HASH_TABLE_CONFIG
(index 22, SEED_TOPK_SIZE, intentionally omitted -- pathweaver.cu:1707-1712
 defaults it to 1 via a caught out-of-range exception, exactly matching
 `top10_ptr[query_id * 1]`'s indexing this adapter relies on.)

Index build (the independent exact k-NN graph) is entirely in prepare(),
timed once as preprocessing per DOMAIN_GUIDE; run() launches only search().
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

KERNEL = "ann-search"
IMPL_NAME = "pathweaver-search"
PAPER_KEY = "conf/usenix/KimPNHKLL25"
PRECISIONS = ["fp32"]  # PathWeaver's own DATA_T/DISTANCE_T throughout (pathweaver.cu:36,38)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_HERE, "build")
_SO_PATH = os.path.join(_BUILD_DIR, "pathweaver.so")

RECALL_VARIANT = "ann-highdim-recall-qps-pareto"
EXACT_VARIANT = "exact-spatial-knn-kernel"

# Compile-time VECTOR_DIM instantiations wired in search()'s switch
# (pathweaver.cu:1732-2158). Sorted ascending so _padded_dim() can pick the
# smallest one that fits.
_WIRED_DIMS = [96, 100, 128, 256, 768, 960]
_WIRED_INTERNAL_TOPK = (64, 128)

# Hard kernel constants -- see module docstring's "HARD KERNEL CONSTRAINTS".
TEAM_SIZE = 8
SEARCH_WIDTH = 1
GRAPH_DEGREE = 64            # CANDIDATE_BUFFER_SIZE = SEARCH_WIDTH*GRAPH_DEGREE MUST be 64
INTERNAL_TOPK = 64           # one of the two compile-time-wired INTERNAL_TOPK values
BLOCK_SIZE = 32
BITLEN = 10                  # hash table = 2^10 = 1024 slots; ample for smoke-sized N
HASH_RESET_INTERVAL = 16
MIN_ITER = 0
NUM_DIST = 1
# Tuned empirically (see STATUS.md) against ONLY this integration's smoke
# workloads (N=3000, D=32/64, Nq=64, k=10) to comfortably clear the recall
# variant's 0.9 floor with margin. NOT tuned against, or validated on, any
# larger/real dataset -- a real run would need its own sweep, out of scope
# for a login-node functional gate check.
MAX_ITER = 48

NO_PRUNE = 0
NO_THRESHOLD = 0
USE_SEED = 1
USE_HASH_TABLE = 1


def available() -> tuple[bool, str]:
    """Cheap and never-raising, per ARTIFACT_GUIDE rule 8: file check + a
    plain (non-compiling) import probe. Never calls torch.utils.cpp_
    extension.load() -- compilation belongs to build.sh only."""
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


def _load_ext():
    """
    Import torch BEFORE `pathweaver`: the .so (built via a raw torch.utils.
    cpp_extension.load() call, not a setup.py/CUDAExtension wheel) links
    against -lc10/-lc10_cuda/-ltorch* by SONAME only, with no baked-in
    rpath to torch/lib -- confirmed by build.sh's own verification step,
    which failed with "ImportError: libc10.so: cannot open shared object
    file" when `pathweaver` was imported in a process that had not already
    loaded torch's libraries. The harness always imports torch before any
    CUDA adapter, so this ordering is naturally satisfied in normal use;
    enforced here explicitly for standalone correctness.
    """
    if _BUILD_DIR not in sys.path:
        sys.path.insert(0, _BUILD_DIR)
    import torch  # noqa: F401 -- import order matters, see docstring above
    import pathweaver as _pw
    return _pw


def _padded_dim(d: int) -> int:
    """
    Smallest compile-time-wired VECTOR_DIM >= d. Zero-padding to this width
    is distance-preserving for L2 -- see module docstring point 1.
    """
    for w in _WIRED_DIMS:
        if d <= w:
            return w
    raise NotImplementedError(
        f"{IMPL_NAME}: workload dim={d} exceeds every compile-time VECTOR_DIM "
        f"pathweaver.cu's search() dispatches to (wired: {_WIRED_DIMS}, "
        "pathweaver.cu:1732-2158's switch(VECTOR_DIM)); any unwired value "
        "hits that switch's `default:` branch, which calls assert(false) on "
        "the HOST and aborts the process (confirmed experimentally, not "
        "catchable as a Python exception). No padding target is large "
        "enough for this workload; raising here instead of attempting the "
        "call.")


def _shared_mem_size(vector_dim: int, internal_topk: int, graph_degree: int,
                     search_width: int, bitlen: int) -> int:
    """
    Reimplementation of source/pathweaver/util.py's calculate_shared_mem_
    size (NOT imported: util.py does `import pymetis` at module scope, an
    unrelated graph-partitioning dependency for PathWeaver's multi-GPU
    sharding this single-GPU integration never exercises and does not want
    as a hard adapter dependency). Formula copied verbatim from util.py's
    own arithmetic, matching search_kernel's own shared-memory layout
    (pathweaver.cu:1370-1388: query buffer + result indices/distances
    buffers + visited-hash table + parent-list buffer + 2 scalar flags).
    """
    result_buffer_size = internal_topk + search_width * graph_degree
    hash_table_size = 1 << bitlen
    return (hash_table_size * 4 + vector_dim * 4 + 2 * result_buffer_size * 4
            + search_width * 4 + 4 + 4)


def _build_knn_graph(X, degree: int, chunk: int = 1024):
    """
    Independent exact base-vs-base k-NN graph construction (preprocessing,
    timed as part of prepare()): for every point in the (padded) base set,
    its `degree` nearest OTHER base points, via a chunked Gram-matrix L2
    computation on the GPU (torch, fp32). Chunked over ROWS (not columns)
    to bound peak memory to chunk x N regardless of N, the same style
    kernelbench.domains.annsearch._bruteforce_ground_truth uses for its own
    (query-vs-base) ground truth -- this function is a SEPARATE, base-vs-
    base computation over X alone, never touching the workload's query set
    Q or its cached gt_idx/gt_dist answer key.

    This is NOT CAGRA's own graph-construction/optimization algorithm
    (RAFT/cuVS out of scope -- see module docstring's SCOPE section); it is
    a plain exact-kNN substitute of the SAME fixed degree (64) PathWeaver's
    kernel requires. Returns an [N, degree] int64 tensor on the same device
    as X.
    """
    import torch
    n = X.shape[0]
    if degree >= n:
        raise NotImplementedError(
            f"{IMPL_NAME}: n_base={n} <= GRAPH_DEGREE={degree}; GRAPH_DEGREE "
            "is a hard kernel constant (CANDIDATE_BUFFER_SIZE must be "
            "exactly 64, see module docstring point 3), not reducible "
            "per-workload")
    graph = torch.empty((n, degree), dtype=torch.int64, device=X.device)
    x2 = (X * X).sum(dim=1)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        blk = X[s:e]
        d2 = x2[s:e, None] - 2.0 * (blk @ X.T) + x2[None, :]
        rows = torch.arange(e - s, device=X.device)
        d2[rows, rows + s] = float("inf")  # exclude self
        graph[s:e] = torch.topk(d2, degree, dim=1, largest=False).indices
    return graph


class PathweaverSearch:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(f"{IMPL_NAME} only supports fp32; requested {precision}")
        self.precision = precision
        self._pw = _load_ext()

    def prepare(self, w, params: dict):
        import torch

        if w.variant != RECALL_VARIANT:
            raise NotImplementedError(
                f"{IMPL_NAME}: only wired for {RECALL_VARIANT!r} -- PathWeaver's "
                "search kernel targets high-dimensional ANN over dense vectors; "
                f"{EXACT_VARIANT!r}'s low-dimensional (D=3) exact-match workloads "
                "are out of scope for this adapter (see STATUS.md)")
        if w.k > INTERNAL_TOPK:
            raise NotImplementedError(
                f"{IMPL_NAME}: workload k={w.k} > INTERNAL_TOPK={INTERNAL_TOPK} "
                f"(the smaller of the two compile-time-wired values "
                f"{_WIRED_INTERNAL_TOPK}); the internal beam width must be >= "
                "the requested output k")

        device = torch.device("cuda:0")
        dpad = _padded_dim(w.dim)

        t0 = time.perf_counter()

        X = np.ascontiguousarray(w.X, dtype=np.float32)
        Xpad = np.zeros((w.n_base, dpad), dtype=np.float32)
        Xpad[:, :w.dim] = X
        Q = np.ascontiguousarray(w.Q, dtype=np.float32)
        Qpad = np.zeros((w.n_queries, dpad), dtype=np.float32)
        Qpad[:, :w.dim] = Q

        dataset_t = torch.from_numpy(Xpad).to(device)
        queries_t = torch.from_numpy(Qpad).to(device)

        graph_idx = _build_knn_graph(dataset_t, GRAPH_DEGREE)   # [n_base, 64] int64
        graph_t = graph_idx.to(torch.uint32).contiguous()

        # SIMPLEST-VALID single-pass seed: all-zero starting point + identity
        # map -- see module docstring. Built via numpy then torch.from_numpy
        # (torch.arange/torch.zeros do not support dtype=uint32 on CUDA in
        # this torch build; matches source/pathweaver/util.py's own
        # convert_to_torch(np.zeros(...,dtype=np.uint32)) pattern).
        top10_t = torch.from_numpy(np.zeros(w.n_queries, dtype=np.uint32)).to(device)
        seed_map_t = torch.from_numpy(np.arange(w.n_base, dtype=np.uint32)).to(device)
        sign_bit_t = torch.zeros((0, 0), dtype=torch.uint32, device=device)

        shared_mem = _shared_mem_size(dpad, INTERNAL_TOPK, GRAPH_DEGREE, SEARCH_WIDTH, BITLEN)
        configs = np.array([
            w.n_queries, w.k, SEARCH_WIDTH, MAX_ITER, MIN_ITER, INTERNAL_TOPK,
            dpad, w.n_base, TEAM_SIZE, GRAPH_DEGREE, NUM_DIST, BLOCK_SIZE,
            BITLEN, HASH_RESET_INTERVAL, shared_mem,
            NO_PRUNE, 1.0, 0.0,
            NO_THRESHOLD, 0,
            USE_SEED,
            USE_HASH_TABLE,
        ], dtype=np.float64)  # configs[i].item<int/float>() casts regardless of stored dtype
        configs_t = torch.from_numpy(configs).to(device)
        results_distances = torch.zeros((w.n_queries, w.k), dtype=torch.float32, device=device)

        preprocessing_s = time.perf_counter() - t0
        params["index_build"] = (
            f"independent exact base-vs-base k-NN graph, degree={GRAPH_DEGREE} "
            f"(chunked GPU Gram-matrix L2, fp32, {preprocessing_s:.3f}s); "
            f"D padded {w.dim}->{dpad} via zero-columns (distance-preserving "
            "for L2, see adapter.py module docstring); NOT CAGRA/RAFT graph "
            "construction (out of scope -- see STATUS.md)")
        params["vector_dim_padded"] = dpad
        params["max_iter"] = MAX_ITER
        params["graph_degree"] = GRAPH_DEGREE
        params["seed_config"] = "trivial (zero starting point + identity map, no ghost stage)"

        self._gt_idx = w.gt_idx
        self._variant = w.variant
        self._n_base = w.n_base

        return {
            "graph": graph_t, "dataset": dataset_t, "queries": queries_t,
            "top10": top10_t, "seed_map": seed_map_t, "sign_bit": sign_bit_t,
            "configs": configs_t, "results_distances": results_distances,
        }

    def run(self, h):
        return self._pw.search(
            h["graph"], h["dataset"], h["queries"], h["top10"],
            h["seed_map"], h["sign_bit"], h["configs"], h["results_distances"])

    def to_host(self, out):
        idx = out.to("cpu").numpy().astype(np.int64)
        # Shared cross-impl recall helper from the domain module -- see
        # annsearch.py's module docstring for why sharing this across
        # IMPLEMENTATIONS (never with the reference) is fine.
        from kernelbench.domains import annsearch as A
        recall = A._recall_at_k(idx, self._gt_idx)
        return np.array([1.0 - recall], dtype=np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        h.clear()


def create(precision: str):
    return PathweaverSearch(precision)
