"""
Graph kernels: bfs, pagerank, triangle-counting, connected-components,
graph-pattern-mining (+ planned: sssp).

Graphs ARE sparse matrices (adjacency, CSR), so workload acquisition/caching
is a thin wrapper around `matrices.py` — `Graph` below owns a `matrices.Matrix`
and adds the graph-specific metadata (directedness, whether it was
symmetrized) the sparse-linear-algebra domain has no reason to track.

  WORKLOADS   — Graph wraps matrices.Matrix; loading/caching is inherited
  COST        — GTEPS (edges/s) for bfs/triangle-counting/connected-components,
                GEPS (edges/s, benchspecs/graph-pattern-mining's own unit
                name) for graph-pattern-mining, GFLOP/s (2*nnz*iterations)
                for pagerank, per the specs
  REFERENCE   — independent (non-scipy, where practical) CPU implementations
                used only for the correctness gate, never timed
  IMPLS       — scipy/numpy/pure-Python CPU implementations; no CUDA impl
                shipped by this module (see cuda_impls() below for why) —
                paper-artifact GPU adapters for graph-pattern-mining live
                under bench/artifacts/graph-pattern-mining/<shortname>/

Five deliberate, documented deviations from a byte-for-byte reading of the
benchspecs (each is called out again at its point of use):

1. BFS's spec times 64 *independent* per-root searches and validates each one
   individually. This harness's `run_variant` calls `impl.run(handle)` with no
   per-call arguments and gates correctness on only the FIRST call's output.
   ScipyBFS honors "64 fixed-seed roots, one search per rep" by cycling
   through a precomputed root list inside `run()`, but the harness's
   correctness gate therefore only certifies root #0 — the other 63 searches
   run and are timed, but aren't independently re-validated. This is a
   harness-level limitation (see harness.py's fixed prepare/gate/warmup/reps
   protocol), not something a graph-specific hack should paper over, so it is
   documented here instead of silently "fixed" by e.g. re-validating all 64
   roots inside prepare() (which would inflate preprocessing_ms by ~64x on
   real graphs and defeat the point of separating prep from search time).
2. PageRank's spec correctness gate is an L1-distance bound; harness.py only
   implements max-based error modes. `max_scaled_err` against a uniform
   1/N-magnitude scale is used as a conservative proxy (max error is a
   stricter bound than mean/L1 error over the same tolerance).
3. connected-components implements plain (weak, for undirected input) /
   strong (for directed input) connectivity via
   `scipy.sparse.csgraph.connected_components`, matching the connectivity
   primitive that appears at the base of both connected-components benchspec
   sub-problems (cc-scc-kernel's own operation IS this: `scc_id[V] =
   SCC(directed adjacency)`). It does not implement full biconnected
   components (cc-bcc-*) or image-CCL (cc-image-ccl-2d3d) — those are
   different problems (cut-vertex structure; regular pixel grids) explicitly
   out of scope for a first "connected-components" kernel per the task that
   produced this module.
4. graph-pattern-mining's spec (`benchspecs/graph-pattern-mining/spec.yaml`)
   defines a whole `pattern_set` (the k=3..7 clique ladder, the 6-graphlet
   4-motif census, and five named 5-7-vertex shapes) across three variants.
   This first "graph-pattern-mining" kernel implements only the spec's own
   cross-verified common denominator — exact k-clique COUNTING, k
   parameterized via `params['clique_k']` (default 4, i.e. K4) — since it is
   the one pattern family explicitly run by name across multiple in-scope
   surveyed artifacts (GLumin's `CliqueSolver` k=4,5,6,7; GraphFold's
   CF4-CF7; VSGM's `-kc 3..7`), per spec.yaml's own "Pattern-set naming is
   inconsistent" divergence note (survey.md: only the clique ladder is
   cross-verified; everything else is an unlabeled `Pn.g`). K3 (triangle) is
   already its own separate kernel in this module; k=3 remains a valid,
   degenerate call into this implementation (count(G, K3) == triangle
   count) but the default here is k=4, the smallest rung GLumin/GraphFold
   actually name a dedicated solver for. The four_motif_census and
   named_5v_plus pattern families are NOT implemented by this kernel —
   explicitly out of scope for now, same posture as connected-components'
   partial scope in (3) above.
5. BFS's correctness gate compares LEVEL/DISTANCE arrays, not parent arrays
   -- changed 2026-08-08 while integrating GPU bfs artifacts (blest, efg,
   bit-graphblas; see bench/artifacts/bfs/). The bfs benchspec's own
   `operation` text explicitly sanctions either representation ("output is a
   parent/predecessor array (or level array) over all V vertices"), but a
   PARENT array is only a meaningful EXACT-match target between two
   implementations that break same-level ties identically -- BFS distances
   are unique per (graph, source), but which same-level neighbor becomes the
   recorded parent is not, and a massively parallel / direction-optimizing
   GPU BFS essentially never reproduces a single-threaded reference's
   tie-breaking order. The previous version of this module compared parent
   arrays and only "worked" because ScipyBFS and the pure-Python reference
   both happen to be sequential top-down traversals over the same sorted
   CSR, which resolve ties identically by construction -- not a property any
   other implementation shares. `_bfs_tree()` below still builds a parent
   tree (kept since it costs nothing extra and some callers may want it),
   but `reference_bfs()` now returns the LEVEL array computed by that same
   traversal, and `ScipyBFS` was switched from `breadth_first_order`
   (predecessors) to `dijkstra(..., unweighted=True, return_predecessors=
   False)` (distances = BFS levels, since every edge already carries unit
   weight post-`_clean_graph`), so the CPU impl's own gate is exercised on
   the same representation as every other impl, not silently loosened.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

from .. import matrices, workload
from ..harness import Timer

KERNELS = ["bfs", "pagerank", "triangle-counting", "connected-components",
           "graph-pattern-mining"]
# owned by this domain but not yet implemented; kept honest for `--list-kernels`
PLANNED = ["sssp"]


# ============================================================== workloads
@dataclass
class Graph:
    """
    A graph is an adjacency matrix plus the metadata a sparse Matrix alone
    doesn't carry: directedness and whether/how it was symmetrized. Values
    are always dropped to a 0/1 pattern at construction time (every kernel
    in this module is unweighted/weight-agnostic per its spec's `operation`).
    """

    name: str
    matrix: matrices.Matrix
    directed: bool
    symmetrized: bool = False
    symmetrize_note: str = ""

    @property
    def csr(self) -> sp.csr_matrix:
        return self.matrix.csr

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def nnz(self) -> int:
        return self.matrix.nnz

    def describe(self) -> dict:
        d = self.matrix.describe()
        d.update({
            "vertices": int(self.shape[0]),
            "edges": int(self.nnz),
            "directed": self.directed,
            "symmetrized": self.symmetrized,
            "symmetrize_note": self.symmetrize_note,
        })
        return d


def _binary_csr(A: sp.csr_matrix) -> sp.csr_matrix:
    """Collapse duplicate entries, then drop all values to a 0/1 pattern."""
    B = A.tocsr(copy=True)
    B.sum_duplicates()
    B.data[:] = 1.0
    B.eliminate_zeros()
    B.sort_indices()
    return B


def _remove_self_loops(A: sp.csr_matrix) -> sp.csr_matrix:
    A = A.tocoo()
    keep = A.row != A.col
    out = sp.coo_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=A.shape).tocsr()
    out.sort_indices()
    return out


def _is_symmetric(A: sp.csr_matrix) -> bool:
    AT = A.T.tocsr()
    diff = A != AT
    return diff.nnz == 0


def _clean_graph(csr: sp.csr_matrix) -> sp.csr_matrix:
    return _remove_self_loops(_binary_csr(csr))


# -------------------------------------------- image-CCL (additive, small)
# benchspecs/connected-components/spec.yaml's `cc-image-ccl-2d3d` variant is
# a REGULAR-GRID sub-problem (binary-image 2D/3D connected-component
# LABELING, 8-connectivity fixed) explicitly scoped as "a related-but-
# distinct primitive" from this module's other 3 (sparse-adjacency) variants
# -- not a fourth data point on the same "connectivity kernel" curve (see
# the spec's own notes_on_fairness). Module docstring deviation (3) above
# says this first pass deliberately left it out; this addition is the
# small, additive exception the YACCLAB (journals/tpds/AllegrettiBG20)
# integration task explicitly authorized ("only add [a workload] if the
# spec defines it and the change is small and additive") -- 2D only (no 3D/
# 26-connectivity), synthetic images only (no real YACCLAB dataset
# download, same "no real corpus download" posture this module and
# compression.py already take), reusing the EXISTING independent
# _union_find_components helper (a different algorithm from any block-based
# GPU labeling implementation this gates) rather than introducing a new
# reference algorithm from scratch.
IMAGE_SEED = 20260806  # matches compression.py's synthetic-data seed convention


@dataclass
class Image2D:
    """A binary foreground(1)/background(0) 2D image -- the workload for
    the image-CCL sub-variant only (NOT interchangeable with `Graph`;
    connected-components' other 3 variants never see this type)."""

    name: str
    image: np.ndarray   # uint8, values in {0,1}, 2D (rows, cols)
    seed: int

    def describe(self) -> dict:
        return {
            "name": self.name,
            "source": "synthetic",
            "synthetic": True,
            "regime": "binary-image-2d-8connectivity",
            "shape": list(self.image.shape),
            "seed": self.seed,
            "foreground_fraction": float(self.image.mean()),
        }


def _smoke_binary_image(seed: int, shape: tuple[int, int], threshold: float) -> np.ndarray:
    """A small correlated (blob-like, not iid-noise) binary image. iid
    Bernoulli noise at any nontrivial density under 8-connectivity sits well
    past the 2D percolation threshold and tends toward one giant component
    -- a poor smoke test for a LABELING algorithm (one blob exercises far
    less of the block-merging logic than several small ones). A blurred
    random field thresholded to a target foreground fraction gives
    multi-blob images instead, independent of any other module (a small
    inline box blur, not a shared helper)."""
    rng = np.random.default_rng(seed)
    field = rng.standard_normal(shape)
    kernel = np.ones(5) / 5.0
    for axis in range(field.ndim):
        field = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis, field)
    return (field > threshold).astype(np.uint8)


def _image_smoke_workloads() -> list[Image2D]:
    return [
        Image2D("smoke-image-blobs-sparse", _smoke_binary_image(IMAGE_SEED, (48, 48), 0.60), IMAGE_SEED),
        Image2D("smoke-image-blobs-medium", _smoke_binary_image(IMAGE_SEED + 1, (40, 64), 0.10), IMAGE_SEED + 1),
        Image2D("smoke-image-blobs-dense", _smoke_binary_image(IMAGE_SEED + 2, (56, 40), -0.30), IMAGE_SEED + 2),
    ]


_IMAGE_DATASET_NAME_HINTS = (
    "3dpes", "fingerprint", "hamlet", "medical", "mirflickr", "tobacco800", "xdocs")


def _load_image_workload(name: str) -> Image2D:
    """Synthetic surrogate for one of cc-image-ccl-2d3d's named YACCLAB
    datasets -- same "generate a representative stand-in, always disclosed
    `synthetic: True`" posture as compression.py's load_workload(), since
    downloading YACCLAB's actual image corpus is out of scope here."""
    text = name.lower()
    seed = IMAGE_SEED + (abs(hash(text)) % 1000)
    return Image2D(f"synthetic-surrogate::{name}",
                   _smoke_binary_image(seed, (256, 256), 0.15), seed)


def load_workload(name: str) -> Graph:
    """
    Fetch a named real graph from the SuiteSparse collection (many of the
    specs' recommended_subset names — SNAP graphs, LAW webgraphs, DIMACS10
    road networks — are mirrored there under the group the graph belongs to).
    Directedness is *detected* (pattern symmetry), never assumed; this module
    deliberately does NOT force-symmetrize a directed input here, since two
    of the four kernels (bfs, connected-components/SCC) must see the graph
    exactly as distributed — symmetrizing centrally would make an SCC
    computation silently degenerate into WCC, which the connected-components
    spec explicitly forbids.

    Exception: a name matching one of cc-image-ccl-2d3d's named image
    datasets returns an `Image2D` synthetic surrogate instead (see above) --
    the only workload type this domain returns that is NOT a `Graph`.
    """
    if any(h in name.lower() for h in _IMAGE_DATASET_NAME_HINTS):
        return _load_image_workload(name)
    m = matrices.load_matrix(name)
    A = _clean_graph(m.csr)
    directed = not _is_symmetric(A)
    m2 = matrices.Matrix(name=m.name, csr=A, source=m.source, group=m.group,
                         load_seconds=m.load_seconds)
    note = "" if directed else "input already symmetric; no forcing applied"
    return Graph(name=name, matrix=m2, directed=directed, symmetrized=False,
                 symmetrize_note=note)


SMOKE = [
    ("graph-smoke-uniform", dict(rows=1500, cols=1500, nnz_per_row=8, pattern="uniform")),
    ("graph-smoke-powerlaw", dict(rows=1500, cols=1500, nnz_per_row=6, pattern="powerlaw")),
]


def smoke_workloads(kernel: str | None = None, variant: str | None = None) -> list:
    """
    Small synthetic graphs, symmetrized so every kernel in this module (some
    of which need undirected input) can share the same smoke set. NOT
    spec-conforming — real runs use load_workload() on the spec's named
    graphs.

    `kernel`/`variant` are OPTIONAL and purely additive (runner.py only
    passes them if this signature accepts them, per its own dispatch
    comment) -- every OTHER kernel's behavior here is byte-for-byte
    unchanged. The one case that branches: the image-CCL sub-variant
    (`cc-image-ccl-2d3d`) needs `Image2D` workloads, not `Graph`s -- a
    completely different regime (see the Image2D block above), so smoking
    it returns ONLY image workloads rather than mixing in graphs no
    image-CCL implementation could ever accept.
    """
    if variant == "cc-image-ccl-2d3d":
        return _image_smoke_workloads()
    out = []
    for name, kw in SMOKE:
        m = matrices.synthetic(name, **kw)
        A = _clean_graph(m.csr)
        S = A.maximum(A.T).tocsr()  # union of forward+reverse edges
        S = _binary_csr(S)
        m2 = matrices.Matrix(name=name, csr=S, source="synthetic", group=m.group,
                             load_seconds=m.load_seconds)
        out.append(Graph(name=name, matrix=m2, directed=False, symmetrized=True,
                         symmetrize_note="union of forward+reverse edges (smoke set)"))
    return out


# ==================================================================== bfs
def _largest_weak_component(A: sp.csr_matrix, directed: bool):
    """Vertex indices of the largest (weakly, for directed input) connected
    component, plus the sum of out-degrees within it (a proxy for the number
    of tree edges a BFS rooted there can traverse)."""
    n_comp, labels = csgraph.connected_components(A, directed=directed, connection="weak")
    sizes = np.bincount(labels, minlength=n_comp)
    giant = int(np.argmax(sizes))
    idx = np.flatnonzero(labels == giant)
    deg = np.diff(A.indptr)
    edges = int(deg[idx].sum())
    return idx, edges


def _bfs_roots(A: sp.csr_matrix, directed: bool, seed: int, k: int):
    idx, edges = _largest_weak_component(A, directed)
    rng = np.random.default_rng(seed)
    k_eff = min(k, len(idx))
    roots = rng.choice(idx, size=k_eff, replace=False)
    return roots, edges, int(len(idx))


def _bfs_tree(A: sp.csr_matrix, source: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Independent (pure-Python, non-scipy) level-synchronous BFS: the
    correctness reference underlying both ScipyBFS (module docstring
    deviation 5) and every bfs artifact adapter's gate. Always walks CSR
    rows as out-edges; correct for directed graphs (row = out-neighbors) and
    for undirected ones (rows are symmetric by construction, see
    _clean_graph / load_workload's symmetry detection), so no direction
    branch is needed.

    Returns (parent, level). Sentinel -1 for unreached vertices in BOTH
    arrays; parent[source] == -1 too (no parent), but level[source] == 0
    (it IS reached, at distance zero) -- these are deliberately different
    sentinel conventions for the two arrays, not a bug: "no parent" and
    "distance zero" are both true of the source simultaneously. `level` is
    the array actually used for the exact-match correctness gate (deviation
    5); `parent` is kept because it costs nothing extra to compute
    alongside and some callers may still want a concrete tree.
    """
    n = A.shape[0]
    indptr, indices = A.indptr, A.indices
    parent = np.full(n, -1, dtype=np.int64)
    level = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    visited[source] = True
    level[source] = 0
    frontier = deque([source])
    while frontier:
        u = frontier.popleft()
        for j in range(indptr[u], indptr[u + 1]):
            v = int(indices[j])
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                level[v] = level[u] + 1
                frontier.append(v)
    return parent, level


class ScipyBFS:
    """
    scipy.sparse.csgraph.dijkstra(unweighted=True): a pure top-down,
    level-synchronous BFS (scipy dispatches unweighted-graph dijkstra to a
    plain breadth-first traversal internally; this is not Dijkstra's
    algorithm in any way that matters here). NOT direction-optimizing (no
    bottom-up/pull phase) — disclosed via `direction_optimized` below, per
    the spec's mandatory disclosure requirement.

    Returns the LEVEL/distance array directly (module docstring deviation
    5), not a predecessor array — this is what changed from the original
    `breadth_first_order`-based version and why: distances are what an
    exact cross-implementation gate can actually require, since they are
    unique per (graph, source) regardless of traversal/tie-break order,
    unlike a specific parent tree.

    Root selection (64 fixed-seed roots from the largest weakly-connected
    component) happens once in prepare(), never in run(); each run() call
    performs exactly one BFS from the next root in that fixed cyclic list, so
    a `reps=64` timing loop naturally reproduces "one measurement per search,
    same 64 roots reused" — see the module docstring for the one place this
    diverges from the spec (only the first call is correctness-gated).
    """

    name = "scipy-bfs"
    platform = "cpu"
    direction_optimized = False

    def __init__(self, precision: str = "int64"):
        self.precision = precision

    def prepare(self, graph: Graph, params: dict):
        A = graph.csr
        seed = int(params.get("bfs_root_seed", 0))
        k = int(params.get("num_roots", 64))
        roots, comp_edges, comp_verts = _bfs_roots(A, graph.directed, seed, k)
        # required disclosures: land in `params`, which the harness copies
        # verbatim into the result record.
        params["direction_optimized"] = self.direction_optimized
        params["bfs_root_seed"] = seed
        params["num_roots"] = int(len(roots))
        params["giant_component_vertices"] = comp_verts
        params["giant_component_edges"] = comp_edges
        params["gteps_definition"] = (
            "all edges of the giant (searched) component / search_time_s / 1e9 "
            "— the Graph500/GAP input-edge convention, matching _cost_bfs; "
            "NOT the (much smaller) BFS-tree edge count")
        return {"A": A, "directed": graph.directed, "roots": roots, "cursor": 0}

    def run(self, h):
        src = int(h["roots"][h["cursor"] % len(h["roots"])])
        h["cursor"] += 1
        dist = csgraph.dijkstra(h["A"], directed=h["directed"], indices=src,
                                unweighted=True, return_predecessors=False)
        level = np.where(np.isinf(dist), -1, dist).astype(np.int64)
        return level

    def to_host(self, out):
        return np.asarray(out, dtype=np.int64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


def reference_bfs(graph: Graph, params: dict):
    """Reconstructs the SAME root list ScipyBFS.prepare() built (same seed,
    same k, same component search) and runs the independent pure-Python BFS
    from root #0 — the root the harness's single-call correctness gate
    actually exercises. Returns the LEVEL array (module docstring
    deviation 5); the parent tree _bfs_tree also computes is discarded here
    since the gate compares levels, not parents."""
    A = graph.csr
    seed = int(params.get("bfs_root_seed", 0))
    k = int(params.get("num_roots", 64))
    roots, _, _ = _bfs_roots(A, graph.directed, seed, k)
    _parent, level = _bfs_tree(A, int(roots[0]))
    return level


def _cost_bfs(g: Graph, params: dict):
    edges = int(params.get("giant_component_edges") or g.nnz)
    ib = 4
    n = g.shape[0]
    byts = edges * ib + (n + 1) * ib  # CSR structure touched by one search, lower bound
    return edges, byts


workload.register_cost("bfs", _cost_bfs, "GTEPS")


# ============================================================= pagerank
class NumpyPageRank:
    """
    Push-style power iteration: A^T @ (r / outdeg), CSR source-scatter via a
    one-shot transpose done in prepare(). NOT direction-optimized (no
    pull/CSC path) — disclosed via `direction` below.

    Runs a FIXED k-iteration budget entirely inside one run() call, matching
    pagerank-fixed-iter-kernel-f32's timing_scope ("kernel only: the
    k-iteration loop"); `iterations` is an explicit, recorded param
    (default 20, the spec's own fixed budget) rather than a hidden constant.
    """

    name = "numpy-pagerank"
    platform = "cpu"
    direction = "push"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = np.float32 if precision == "fp32" else np.float64

    def prepare(self, graph: Graph, params: dict):
        A = graph.csr.astype(self.dtype)
        n = A.shape[0]
        alpha = float(params.get("damping", 0.85))
        k = int(params.get("iterations", 20))
        outdeg = np.diff(A.indptr).astype(self.dtype)
        inv_outdeg = np.zeros(n, dtype=self.dtype)
        nz = outdeg > 0
        inv_outdeg[nz] = 1.0 / outdeg[nz]
        dangling = ~nz
        AT = A.T.tocsr()  # one-shot transpose: preprocessing, not per-iteration cost
        r0 = np.full(n, 1.0 / n, dtype=self.dtype)
        params["damping"] = alpha
        params["iterations"] = k
        params["direction"] = self.direction
        params["dangling_node_handling"] = "redistribute uniformly over all V"
        params["dangling_count"] = int(dangling.sum())
        return dict(AT=AT, inv_outdeg=inv_outdeg, dangling=dangling,
                    alpha=alpha, k=k, n=n, r0=r0)

    def run(self, h):
        AT, inv_outdeg, dangling = h["AT"], h["inv_outdeg"], h["dangling"]
        alpha, k, n = h["alpha"], h["k"], h["n"]
        r = h["r0"].copy()
        base = (1.0 - alpha) / n
        for _ in range(k):
            dangling_mass = float(r[dangling].sum())
            r = (base + alpha * (AT @ (r * inv_outdeg)) + alpha * dangling_mass / n)
            r = r.astype(h["r0"].dtype, copy=False)
        return r

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


def reference_pagerank(graph: Graph, params: dict):
    """fp64 power iteration, same alpha/k/dangling-handling, computed
    independently (own vectorized path, not a call into NumpyPageRank)."""
    A = graph.csr.astype(np.float64)
    n = A.shape[0]
    alpha = float(params.get("damping", 0.85))
    k = int(params.get("iterations", 20))
    outdeg = np.diff(A.indptr).astype(np.float64)
    inv_outdeg = np.zeros(n)
    nz = outdeg > 0
    inv_outdeg[nz] = 1.0 / outdeg[nz]
    dangling = ~nz
    AT = A.T.tocsr()
    r = np.full(n, 1.0 / n)
    base = (1.0 - alpha) / n
    for _ in range(k):
        dangling_mass = r[dangling].sum()
        r = base + alpha * (AT @ (r * inv_outdeg)) + alpha * dangling_mass / n
    # see module docstring deviation (2): harness has no L1-distance gate, so
    # `scale` here is the uniform 1/N magnitude used by max_scaled_err as a
    # conservative proxy for "relative to the rank vector summing to 1.0".
    scale = np.full(n, 1.0 / n)
    return r, scale


def _cost_pagerank(g: Graph, params: dict):
    vb = 4 if params.get("precision", "fp32") == "fp32" else 8
    ib = 4
    n = g.shape[0]
    k = int(params.get("iterations", 20))
    flops = 2 * g.nnz * k
    byts = k * (g.nnz * (vb + ib) + (n + 1) * ib + n * vb + n * vb)
    return int(flops), int(byts)


workload.register_cost("pagerank", _cost_pagerank, "GFLOP/s")


# ==================================================== triangle-counting
def _orient_by_degree(A: sp.csr_matrix):
    """Orient every undirected edge low-rank -> high-rank, rank = ascending
    (degree, vertex id). This is preprocessing (never timed as part of
    run()); disclosed via params['orientation'] in prepare()."""
    n = A.shape[0]
    deg = np.diff(A.indptr)
    order = np.lexsort((np.arange(n), deg))  # primary key = deg, tiebreak = id
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    coo = A.tocoo()
    mask = rank[coo.row] < rank[coo.col]
    L = sp.csr_matrix((np.ones(int(mask.sum())), (coo.row[mask], coo.col[mask])), shape=A.shape)
    L.sum_duplicates()
    L.sort_indices()
    return L, rank


class SpgemmTriangleCount:
    """
    Forward algorithm (Cohen 2009 / Schank-Wagner) via masked SpGEMM: with L
    the degree-ascending oriented DAG, T = sum((L @ L.T).multiply(L)). For
    an oriented edge (u,v), (L @ L.T)[u,v] counts vertices w with u->w AND
    v->w both in L — i.e. triangles {u,v,w} with u<v<w under the chosen
    order — so every triangle is counted EXACTLY ONCE and the divisor is 1
    (never 3 or 6, unlike some other orientation conventions the spec
    mentions). Orientation (and the symmetrize/dedup/self-loop-removal a
    directed or non-simple input needs first) is entirely inside prepare(),
    per the spec's explicit requirement that orientation never be folded
    into kernel time.
    """

    name = "spgemm-triangle-count"
    platform = "cpu"
    orientation = "degree-ascending, ties broken by vertex id"

    def __init__(self, precision: str = "int64"):
        self.precision = precision

    def prepare(self, graph: Graph, params: dict):
        A = _clean_graph(graph.csr)
        if not _is_symmetric(A):
            before = int(A.nnz)
            A = _binary_csr(A.maximum(A.T).tocsr())
            params["symmetrized_in_prepare"] = True
            params["directed_edges_before_symmetrize"] = before
        else:
            params["symmetrized_in_prepare"] = False
        L, _rank = _orient_by_degree(A)
        params["orientation"] = self.orientation
        params["oriented_edges"] = int(L.nnz)
        return {"L": L}

    def run(self, h):
        L = h["L"]
        C = (L @ L.T).multiply(L)
        return int(round(C.sum()))

    def to_host(self, out):
        return np.array(out, dtype=np.int64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


def reference_triangle_count(graph: Graph, params: dict):
    """
    Merge/set-intersection sorted-neighbor counting — the spec's own named
    reference algorithm (tc-gpu-kernel-exact's correctness text) — over the
    SAME degree-ascending orientation, implemented independently (Python
    sets, not scipy sparse matmul) so a bug in scipy's sparse product isn't
    silently replicated in both sides of the gate.
    """
    A = _clean_graph(graph.csr)
    if not _is_symmetric(A):
        A = _binary_csr(A.maximum(A.T).tocsr())
    n = A.shape[0]
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr)
    order = np.lexsort((np.arange(n), deg))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    out_sets = [set() for _ in range(n)]
    for u in range(n):
        for j in range(indptr[u], indptr[u + 1]):
            v = int(indices[j])
            if rank[v] > rank[u]:
                out_sets[u].add(v)
    count = 0
    for u in range(n):
        for v in out_sets[u]:
            count += len(out_sets[u] & out_sets[v])
    return np.array(count, dtype=np.int64)


def _cost_triangle_counting(g: Graph, params: dict):
    # |E| per the spec's own metric definition (GTEPS = |E| / time / 1e9);
    # falls back to nnz/2 (undirected pattern stores both directions) when
    # prepare() hasn't run yet.
    edges = int(params.get("oriented_edges") or (g.nnz // 2 if not g.directed else g.nnz))
    ib = 4
    n = g.shape[0]
    byts = edges * ib + (n + 1) * ib
    return edges, byts


workload.register_cost("triangle-counting", _cost_triangle_counting, "GTEPS")


# ========================================== graph-pattern-mining (k-clique)
# Headline pattern: exact k-clique COUNTING, per benchspecs/graph-pattern-
# mining/spec.yaml's `pattern_set.clique_ladder` — see module docstring
# deviation (4) for why this is the pattern chosen out of the spec's larger
# pattern_set. `count(G, K_k)` = number of distinct k-vertex subsets of G
# that are pairwise fully connected (spec's `operation`, specialized to
# P = K_k). k is a workload parameter, not a fixed constant: real GPM
# artifacts (GLumin's CliqueSolver, GraphFold's CFk) parameterize it the
# same way.
DEFAULT_CLIQUE_K = 4  # smallest k GLumin/GraphFold ship a dedicated k-clique solver for


def _dag_adjacency_sets(L: sp.csr_matrix) -> list:
    """Python-set adjacency list of a DAG-oriented CSR: adj[u] = {v : (u,v) in L}."""
    indptr, indices = L.indptr, L.indices
    return [set(indices[indptr[u]:indptr[u + 1]].tolist()) for u in range(L.shape[0])]


def _count_k_cliques_sets(adj: list, k: int) -> int:
    """
    Exact k-clique count over a degree-oriented DAG's successor sets.

    For an oriented total order (here: degree-ascending, see
    _orient_by_degree), every k-clique {v_1<...<v_k} appears as EXACTLY ONE
    chain v_1 -> ... -> v_k in the DAG, so recursively intersecting successor
    sets and counting leaves at depth k enumerates each k-clique once (this
    is the standard DAG-orientation formulation behind the forward triangle
    algorithm above, generalized from 2 hops to k-1 — the same formulation
    GraphFold's CFk / GLumin's CliqueSolver implement on the GPU, described
    e.g. in Danisch/Chan/Sozio's "kClist" k-clique listing algorithm).
    """
    def extend(candidates, remaining: int) -> int:
        if remaining == 0:
            return 1
        total = 0
        for v in candidates:
            total += extend(candidates & adj[v], remaining - 1)
        return total

    total = 0
    for u in range(len(adj)):
        total += extend(adj[u], k - 1)
    return total


class DagKCliqueCount:
    """
    CPU exact k-clique counter: reuses this module's own degree-ascending
    DAG orientation (_orient_by_degree, the SAME helper SpgemmTriangleCount
    above uses — sharing it across two IMPLEMENTATIONS is fine; only a
    reference may not share code with an implementation it gates, see
    DOMAIN_GUIDE.md's audit ruling) followed by a recursive Python-set
    candidate-intersection search (_count_k_cliques_sets above).

    Preprocessing (cleanup, symmetrization if needed, orientation, building
    the adjacency-set list) is entirely inside prepare(); run() only walks
    the already-built DAG, matching every surveyed artifact's own
    kernel/preprocessing split (GraphFold's TCSolver/CFSolver time only the
    post-orientation search; see survey.md).
    """

    name = "dag-kclique-count"
    platform = "cpu"
    orientation = "degree-ascending, ties broken by vertex id"

    def __init__(self, precision: str = "int64"):
        self.precision = precision

    def prepare(self, graph: Graph, params: dict):
        A = _clean_graph(graph.csr)
        if not _is_symmetric(A):
            before = int(A.nnz)
            A = _binary_csr(A.maximum(A.T).tocsr())
            params["symmetrized_in_prepare"] = True
            params["directed_edges_before_symmetrize"] = before
        else:
            params["symmetrized_in_prepare"] = False
        L, _rank = _orient_by_degree(A)
        k = int(params.get("clique_k", DEFAULT_CLIQUE_K))
        if k < 1:
            raise ValueError(f"clique_k must be >= 1, got {k}")
        params["clique_k"] = k
        params["orientation"] = self.orientation
        params["oriented_edges"] = int(L.nnz)
        adj = _dag_adjacency_sets(L)
        return {"adj": adj, "k": k}

    def run(self, h):
        return _count_k_cliques_sets(h["adj"], h["k"])

    def to_host(self, out):
        return np.array(int(out), dtype=np.int64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


def reference_k_clique_count(graph: Graph, params: dict):
    """
    Independent exact k-clique reference for the correctness gate.

    Per DOMAIN_GUIDE.md's audit ruling ("a reference may NEVER call the same
    function/helper as any implementation it gates ... same formula re-typed
    inline is acceptable; same code object is not"), this function:
      * re-derives the degree-ascending orientation with its OWN inline
        lexsort (duplicated from _orient_by_degree's formula, NOT a call to
        that shared helper — the same pattern reference_triangle_count above
        already uses), and
      * counts cliques with a Python big-int BITMASK representation
        (successor sets as integers, popcount/lowest-set-bit iteration)
        instead of DagKCliqueCount's Python `set` objects — a genuinely
        different data representation and code path, not just a renamed
        copy of the same function.
    Both sides implement the same textbook DAG-orientation + candidate-
    intersection formula (that repetition is explicitly permitted), but share
    no code object and no data structure.
    """
    A = _clean_graph(graph.csr)
    if not _is_symmetric(A):
        A = _binary_csr(A.maximum(A.T).tocsr())
    n = A.shape[0]
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr)
    order = np.lexsort((np.arange(n), deg))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)

    # successor bitmasks indexed BY RANK (not original vertex id), so bit i
    # means "rank i is a DAG successor" — matches the invariant that every
    # bit in succ_bits[r] is > r, needed for the intersection below to stay
    # correctly restricted to higher rank without extra masking.
    succ_bits = [0] * n
    for u in range(n):
        bits = 0
        for j in range(indptr[u], indptr[u + 1]):
            v = int(indices[j])
            if rank[v] > rank[u]:
                bits |= (1 << int(rank[v]))
        succ_bits[int(rank[u])] = bits

    k = int(params.get("clique_k", DEFAULT_CLIQUE_K))
    if k < 1:
        raise ValueError(f"clique_k must be >= 1, got {k}")

    def extend(bits: int, remaining: int) -> int:
        if remaining == 0:
            return 1
        total = 0
        b = bits
        while b:
            lsb = b & (-b)
            v = lsb.bit_length() - 1
            b ^= lsb
            total += extend(bits & succ_bits[v], remaining - 1)
        return total

    total = 0
    for u in range(n):
        total += extend(succ_bits[u], k - 1)
    return np.array(total, dtype=np.int64)


def _cost_graph_pattern_mining(g: Graph, params: dict):
    # spec.yaml's metric is edges/time (GEPS, "follows Fringe-SGC's edges/s
    # convention"), NOT a combinatorial clique-search work count — followed
    # literally here per DOMAIN_GUIDE.md ("work_count must follow the spec's
    # metric field literally"), same convention _cost_triangle_counting uses.
    edges = int(params.get("oriented_edges") or (g.nnz // 2 if not g.directed else g.nnz))
    ib = 4
    n = g.shape[0]
    byts = edges * ib + (n + 1) * ib
    return edges, byts


workload.register_cost("graph-pattern-mining", _cost_graph_pattern_mining, "GEPS")


# ================================================= connected-components
def _canonical_labels(labels: np.ndarray) -> np.ndarray:
    """
    Relabel a partition by order of first appearance, so two labelings of
    the SAME partition (differing only in which integer names which
    component) compare equal. A correct implementation may number components
    however it likes; only the grouping of vertices is required to match.
    """
    labels = np.asarray(labels)
    _uniq, first_idx, inv = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(first_idx)
    rank_of_unique = np.empty_like(order)
    rank_of_unique[order] = np.arange(len(order))
    return rank_of_unique[inv].astype(np.int64)


def _union_find_components(A: sp.csr_matrix) -> np.ndarray:
    """Independent (non-scipy) weakly-connected-components reference."""
    n = A.shape[0]
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    coo = A.tocoo()
    for u, v in zip(coo.row.tolist(), coo.col.tolist()):
        if u != v:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    labels = np.array([find(i) for i in range(n)])
    return labels


def _kosaraju_scc(A: sp.csr_matrix) -> np.ndarray:
    """Independent (non-scipy) strongly-connected-components reference:
    classic two-pass Kosaraju, iterative to avoid recursion limits — a
    genuinely different algorithm family from scipy's Tarjan implementation."""
    n = A.shape[0]
    indptr, indices = A.indptr, A.indices
    visited = np.zeros(n, dtype=bool)
    order = []
    for s in range(n):
        if visited[s]:
            continue
        stack = [(s, indptr[s])]
        visited[s] = True
        while stack:
            u, it = stack.pop()
            if it < indptr[u + 1]:
                stack.append((u, it + 1))
                v = int(indices[it])
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, indptr[v]))
            else:
                order.append(u)
    AT = A.T.tocsr()
    labels = np.full(n, -1, dtype=np.int64)
    comp = 0
    for u in reversed(order):
        if labels[u] != -1:
            continue
        stack = [u]
        labels[u] = comp
        while stack:
            x = stack.pop()
            lo, hi = AT.indptr[x], AT.indptr[x + 1]
            for j in range(lo, hi):
                y = int(AT.indices[j])
                if labels[y] == -1:
                    labels[y] = comp
                    stack.append(y)
        comp += 1
    return labels


class ScipyCC:
    """
    scipy.sparse.csgraph.connected_components. For directed input this
    computes STRONGLY connected components (Tarjan's algorithm) and never
    symmetrizes first — symmetrizing a directed graph would silently turn
    the SCC problem into WCC, which cc-scc-kernel explicitly forbids. For
    undirected (already-symmetric) input it computes plain connectivity.
    """

    name = "scipy-cc"
    platform = "cpu"
    algorithm_family = ("scipy connected_components: Tarjan's algorithm for "
                        "strong connectivity, BFS-based union for weak/undirected")

    def __init__(self, precision: str = "int64"):
        self.precision = precision

    def prepare(self, graph, params: dict):
        if isinstance(graph, Image2D):
            raise NotImplementedError(
                "scipy-cc implements sparse-graph WCC/SCC only, not binary-image "
                "connected-component LABELING (cc-image-ccl-2d3d is a distinct "
                "regime -- see this module's Image2D block / docstring deviation 3)")
        connection = params.get("connectivity_mode") or ("strong" if graph.directed else "weak")
        params["connectivity_mode"] = connection
        params["algorithm_family"] = self.algorithm_family
        return {"A": graph.csr, "directed": graph.directed, "connection": connection}

    def run(self, h):
        _n, labels = csgraph.connected_components(
            h["A"], directed=h["directed"], connection=h["connection"])
        return labels

    def to_host(self, out):
        return _canonical_labels(out)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


def _canonical_labels_image(labels: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    """Like _canonical_labels, but background (foreground_mask == False)
    stays fixed at 0 -- a real semantic value here ("no component"), not
    just "whichever label a union-find pass happened to assign first" the
    way it is for the pure-graph WCC/SCC case _canonical_labels serves.
    Only the foreground labels are canonicalized (by first appearance),
    into 1..n_components."""
    labels = np.asarray(labels)
    out = np.zeros(labels.shape, dtype=np.int64)
    if foreground_mask.any():
        out[foreground_mask] = _canonical_labels(labels[foreground_mask]) + 1
    return out


def reference_image_ccl(img2d: "Image2D", params: dict):
    """Independent (non-scipy, non-CUDA) 8-connectivity image-CCL
    reference: builds an EXPLICIT adjacency matrix among foreground pixels
    (4 of the 8 neighbor offsets suffice -- the other 4 are the same edges
    seen from the opposite pixel) and reuses _union_find_components, the
    SAME independent union-find helper already used for the graph WCC
    reference -- a different algorithm family entirely from the block-based
    GPU labeling kernels (BUF/BKE) this gates, satisfying DOMAIN_GUIDE's
    reference-independence rule."""
    img = img2d.image
    h, w = img.shape
    fg = img.astype(bool)
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)

    rows_list, cols_list = [], []
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
        ys, xs = np.nonzero(fg)
        ny, nx = ys + dy, xs + dx
        in_bounds = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        ys, xs, ny, nx = ys[in_bounds], xs[in_bounds], ny[in_bounds], nx[in_bounds]
        neighbor_fg = fg[ny, nx]
        rows_list.append(idx[ys[neighbor_fg], xs[neighbor_fg]])
        cols_list.append(idx[ny[neighbor_fg], nx[neighbor_fg]])

    n = h * w
    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        A = sp.coo_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                          shape=(n, n)).tocsr()
    else:
        A = sp.csr_matrix((n, n))

    raw = _union_find_components(A)
    canon = _canonical_labels_image(raw, fg.ravel())
    return canon.reshape(h, w).astype(np.float64)


def reference_cc(graph, params: dict):
    if isinstance(graph, Image2D):
        return reference_image_ccl(graph, params)
    connection = params.get("connectivity_mode") or ("strong" if graph.directed else "weak")
    labels = _kosaraju_scc(graph.csr) if connection == "strong" else _union_find_components(graph.csr)
    return _canonical_labels(labels)


def _cost_connected_components(g, params: dict):
    if isinstance(g, Image2D):
        # Disclosed simplification: this kernel registers ONE cost unit
        # ("GTEPS") shared by all 4 connected-components variants, but
        # cc-image-ccl-2d3d's own spec metric is `labeling_time_ms` (a
        # per-image latency, not a throughput rate) -- wiring a genuinely
        # per-variant throughput UNIT through workload.register_cost would
        # be a larger, non-additive harness change, out of scope for this
        # integration. The real per-run `times_ms` is still recorded
        # exactly regardless of this "GTEPS"-labeled throughput number;
        # `work` here is pixel count (read 1 img byte + write 1 int32
        # label per pixel is the compulsory-traffic lower bound), NOT an
        # edge count -- read this number as "Gpixels/s", not literal GTEPS.
        n_pixels = int(g.image.size)
        ib = 4
        byts = n_pixels * 1 + n_pixels * ib
        return n_pixels, byts
    ib = 4
    n = g.shape[0]
    byts = g.nnz * ib + (n + 1) * ib
    return int(g.nnz), int(byts)  # work = edges, per the task's cost rule for this kernel


workload.register_cost("connected-components", _cost_connected_components, "GTEPS")


# =============================================================== registry
REFERENCES = {
    "bfs": reference_bfs,
    "pagerank": reference_pagerank,
    "triangle-counting": reference_triangle_count,
    "connected-components": reference_cc,
    "graph-pattern-mining": reference_k_clique_count,
}

CORRECTNESS_MODE = {
    "bfs": "exact",
    "pagerank": "max_scaled_err",
    "triangle-counting": "exact",
    "connected-components": "exact",
    "graph-pattern-mining": "exact",
}

DEFAULT_PRECISION = {
    "bfs": "int64",
    "pagerank": "fp32",
    "triangle-counting": "int64",
    "connected-components": "int64",
    "graph-pattern-mining": "int64",
}

CPU_IMPLS = {
    "bfs": {"scipy-bfs": ScipyBFS},
    "pagerank": {"numpy-pagerank": NumpyPageRank},
    "triangle-counting": {"spgemm-triangle-count": SpgemmTriangleCount},
    "connected-components": {"scipy-cc": ScipyCC},
    "graph-pattern-mining": {"dag-kclique-count": DagKCliqueCount},
}


def cuda_impls():
    """
    No CUDA implementation shipped BY THIS MODULE: a reasonable GPU graph
    kernel needs either a real device-resident CSR BFS/SpGEMM/union-find
    (cuGraph, Gunrock, or a hand-written kernel none of this repo's csrc/
    currently has for graphs) or a heavyweight optional dependency this repo
    doesn't vendor. Returning {} here is the documented, honest option
    DOMAIN_GUIDE.md allows rather than shipping a fake/mislabeled GPU path.

    graph-pattern-mining's GPU implementations instead come from paper
    artifacts under bench/artifacts/graph-pattern-mining/<shortname>/,
    discovered separately by kernelbench.artifact_registry (see
    ARTIFACT_GUIDE.md) — runner.py merges those in alongside this dict's
    (empty) cuda_impls() and CPU_IMPLS, so `--impl <artifact-impl-name>`
    works without this function needing to know about them.
    """
    return {}
