"""
ann-search: nearest-neighbor search over a fixed vector/point database.

benchspecs/ann-search/spec.yaml splits into two variants this module treats
completely differently -- fundamentally different data, different notions of
correctness, and (per survey.md's own "Divergences" section) not sharing a
benchmark, input suite or accuracy notion at all:

  * ann-highdim-recall-qps-pareto -- high-dimensional (96-128d) APPROXIMATE
    search over the SIFT/DEEP/SPACEV billion-scale family. Correctness is
    recall@k (a RATIO against exact ground-truth top-k), never a bare QPS
    number (spec's own metric.primary: "a bare QPS or speedup number without
    an accompanying recall value is NOT an admissible submission").
  * exact-spatial-knn-kernel -- low-dimensional (2D/3D) EXACT (or tightly
    bounded-approximate, RTNN-style) k-NN over point-cloud/geometric data.
    Correctness is exact neighbor-set match (recall@k is meaningless here:
    search is exact by construction, so recall==1.0 trivially).

Both variants share ONE ownership entry (`ann-search` -> this module, per
domains/__init__.py's OWNER map -- there is only one benchspec file, one
`kernel:` name) and ONE ground-truth-establishing idea: brute-force fp64
k-NN, computed ONCE at workload-build time and cached on the workload
(`AnnWorkload.gt_idx` / `.gt_dist` -- see _bruteforce_ground_truth). That
single build-time computation IS the independent reference for BOTH
variants; what differs is how each variant's impls are graded against it
(recall ratio vs. exact-multiset distance match), which lives in this
module's shared to_host() canonicalization helpers, not in the reference
itself. This mirrors ml.py's precedent for a kernel whose two variants need
different treatment (sparse-attention-kernel implements one variant here and
points at a different track for the other) and graph.py's precedent for
workload-carried, variant-specific gate parameters.

  WORKLOADS -- AnnWorkload: base set X[N,D], query set Q[Nq,D], k, a fixed
               distance metric (L2, the only metric attested anywhere in
               this track's 7-paper survey), plus the cached ground truth.
               smoke_workloads(): small seeded synthetic Gaussian (recall
               variant) / uniform-cube (exact variant) sets, NOT spec-
               conforming -- returns instances of BOTH variants together
               (see smoke_workloads' own docstring for why that is correct,
               not sloppy). load_workload(name): "sift1m" (real texmex
               SIFT1M corpus, direct-URL, cached under bench/annsearch-data/,
               BOUNDED-subsampled -- see _load_sift1m_subset), "stanford-
               bunny" (real Stanford 3D Scanning Repository mesh, same
               caching convention) and "uniform-1k"/"uniform-100k"/
               "uniform-1m" (synthetic, CLOVER's own named sizes) are wired;
               every other recommended_subset name (SIFT100M/1B, DEEP*,
               SPACEV1B, KITTI, Millennium N-body, CLOVER's mesh sweep)
               raises a clear, documented NotImplementedError rather than
               attempting a multi-GB-to-multi-TB or access-gated download --
               see load_workload's docstring for exactly which names are
               wired and why the rest are not.
  COST      -- QPS (queries/second), literally the spec's own primary
               throughput unit for both variants. See _cost_ann_search's
               docstring for the /1e9-hardwire workaround this needs (same
               shape as ml.py's GFLOP/s-vs-TFLOP/s note for attention-kernel).
  REFERENCE -- reference_ann() returns the cached ground truth (exact
               variant) or a constant zero-gap target (recall variant, see
               below); ALWAYS independent of every impl (built once at
               workload-load time by _bruteforce_ground_truth, a function no
               impl in this module calls).
  IMPLS     -- CPU_IMPLS: ONE independent numpy brute-force exact-kNN impl
               (NumpyBruteForceKNN) -- a genuinely different numerical route
               (direct pairwise subtraction, batched over the BASE set) from
               the ground-truth builder's Gram-matrix/matmul expansion
               (batched over the QUERY set), per DOMAIN_GUIDE's reference-
               independence audit ruling (same formula, different code
               object -- see both docstrings). It is a legitimate competitor
               on BOTH variants: an exact method trivially achieves
               recall==1.0 on the recall variant, and is the natural CPU
               baseline for the exact variant. GPU competitors are the paper
               artifacts under bench/artifacts/ann-search/<shortname>/
               (clover, ...), discovered separately by
               kernelbench.artifact_registry, same as every other domain.

Correctness-gate design (reuses harness.py's EXISTING "max_abs_err" mode --
no harness.py change needed; see harness.check_correctness):

  * exact-spatial-knn-kernel: to_host() recomputes fp64 L2 distances for
    whatever indices the impl actually returned (against the workload's own
    X/Q -- a computation shared across every impl's to_host, never shared
    with the reference; see DOMAIN_GUIDE's "sharing across IMPLEMENTATIONS
    is fine" allowance, same pattern graph.py's DagKCliqueCount/
    reference_k_clique_count pair uses), sorted ascending per query. This
    canonicalizes away tied-distance permutation (the task's own "compare
    distance multisets... with tie tolerance" framing) automatically: a
    wrong neighbor set produces recomputed distances that do NOT match the
    ground truth's sorted distances, while a right one (regardless of which
    same-distance point among ties was chosen) produces the identical
    sorted array up to floating-point rounding. `max_abs_err` against
    `AnnWorkload.gt_dist` with a small fp-equality tolerance
    (`correctness_tolerance`, see AnnWorkload) is exactly this check.
  * ann-highdim-recall-qps-pareto: to_host() computes recall@k (the
    fraction of returned indices that appear in the ground-truth top-k,
    averaged over queries -- the standard ann-benchmarks/PathWeaver
    convention; index-set intersection is safe here because tied distances
    are measure-zero for continuous high-dimensional data, unlike the
    exact-spatial variant's low-dimensional synthetic/grid data) and
    returns `[1.0 - recall]` (a 1-element array). reference_ann() returns
    `[0.0]` (the "perfect recall" target) and AnnWorkload.correctness_
    tolerance carries `1.0 - recall_floor` -- so `max_abs_err`'s existing
    "value <= tolerance" check becomes exactly "recall >= floor" without any
    new harness mode. This mirrors graph.py's PageRank deviation (reusing
    max_scaled_err for an L1-shaped spec gate) and compression.py's
    tolerance_override mechanism (a workload-carried, per-run numeric bound)
    -- both established precedents for adapting an existing gate shape
    rather than inventing a new CORRECTNESS_MODE string.

Batch-vs-single-query (spec-mandated dual report, both variants): NOT
implemented. Every impl's run() processes the FULL query set Q in one call
(the "large saturating batch" mode), giving genuine batch QPS. The spec's
OTHER mandatory mode -- held-out single-query latency with a p50/p95/p99
distribution -- would need a per-query timed call structure this first
integration pass does not build (a materially bigger undertaking than BFS's
documented per-root cycling: this track wants percentiles over a MUCH larger
held-out query count than this harness's reps floor gives a meaningful
distribution over). Disclosed here rather than silently reported as if both
modes were covered, same posture as ml.py's qgemm-model-accuracy scope note.

Index-build/preprocessing (spec-mandated, both variants): handled correctly
by construction -- prepare() is where every impl (CPU or GPU artifact) does
its index/spatial-structure construction, timed once as preprocessing_ms and
reported separately from search time, exactly per DOMAIN_GUIDE and every
other domain module in this repo. NumpyBruteForceKNN's own prepare() does
only dtype/layout setup (a brute-force method has no index to build).
"""

from __future__ import annotations

import os
import tarfile
import time
import urllib.request
from dataclasses import dataclass

import numpy as np

from .. import workload
from ..harness import Timer

KERNELS = ["ann-search"]
PLANNED: list[str] = []

EXACT_VARIANT = "exact-spatial-knn-kernel"
RECALL_VARIANT = "ann-highdim-recall-qps-pareto"

DATA_DIR = os.environ.get(
    "KERNELBENCH_ANNSEARCH_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "annsearch-data"),
)
DATA_DIR = os.path.normpath(DATA_DIR)


# ============================================================== workload
@dataclass
class AnnWorkload:
    """
    A fixed base set + query set + k, plus the cached ground truth every
    impl is gated against. `correctness_tolerance` is the resolved numeric
    bound the runner's `tolerance_override` hook picks up automatically
    (`getattr(m, "correctness_tolerance", None)` in runner.py -- the same
    mechanism compression.py's Field uses for lossy-compression's eb), so no
    harness.py change is needed to feed a workload-derived bound into the
    gate.
    """

    name: str
    variant: str                    # EXACT_VARIANT | RECALL_VARIANT
    X: np.ndarray                   # [N, D] fp32 base vectors
    Q: np.ndarray                   # [Nq, D] fp32 query vectors
    k: int
    gt_idx: np.ndarray              # [Nq, k] int64, ground truth, ascending (dist, idx)
    gt_dist: np.ndarray             # [Nq, k] float64, ascending sorted distances
    recall_floor: float | None      # RECALL_VARIANT only
    correctness_tolerance: float    # resolved gate bound (see class docstring)
    synthetic: bool
    seed: int
    source: str
    metric: str = "L2"
    ground_truth_build_seconds: float = 0.0

    @property
    def n_base(self) -> int:
        return self.X.shape[0]

    @property
    def n_queries(self) -> int:
        return self.Q.shape[0]

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    def describe(self) -> dict:
        d = {
            "name": self.name, "kernel": "ann-search", "spec_variant": self.variant,
            "source": self.source, "synthetic": self.synthetic, "seed": self.seed,
            "metric": self.metric, "n_base": int(self.n_base),
            "n_queries": int(self.n_queries), "dim": int(self.dim), "k": int(self.k),
            "batch_mode": (
                "saturating-batch (full query set per run() call); single-query "
                "p50/p95/p99 latency mode NOT implemented -- see module docstring"),
            "ground_truth": (
                f"brute-force fp64, computed once at workload build "
                f"({self.ground_truth_build_seconds:.3f}s)"),
        }
        if self.variant == RECALL_VARIANT:
            d["recall_floor"] = self.recall_floor
            d["correctness_gate"] = (
                f"recall@{self.k} >= {self.recall_floor} (encoded as "
                f"max_abs_err(1-recall, 0) <= {self.correctness_tolerance:g}; "
                "see module docstring)")
        else:
            d["correctness_gate"] = (
                f"exact top-{self.k} distance-multiset match, "
                f"fp tolerance {self.correctness_tolerance:g}")
        return d


# ======================================================== ground truth
def _bruteforce_ground_truth(X: np.ndarray, Q: np.ndarray, k: int,
                             query_block: int = 128,
                             base_block: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """
    Independent, workload-BUILD-TIME-ONLY brute-force exact k-NN (fp64,
    Euclidean/L2) via the Gram-matrix expansion
    `||x-q||^2 = ||x||^2 - 2 x.q + ||q||^2` computed with matmul, double-
    chunked (over both queries AND base points) to bound memory regardless
    of N -- real named datasets (sift1m) can have N in the hundreds of
    thousands. This is the SOLE source of ground truth for both variants
    (recall@k's answer key, and the exact variant's target distances) --
    called exactly ONCE per workload, never called by any Implementation in
    CPU_IMPLS or by reference_ann() at RUN time (reference_ann only reads
    the arrays this function already produced). NumpyBruteForceKNN below
    computes the same textbook k-NN with a genuinely different numerical
    route (direct pairwise subtraction, no Gram trick, batched over the
    BASE axis instead of the query axis) so the two never share a code
    object, per DOMAIN_GUIDE's reference-independence audit ruling.

    Returns (idx, dist): both [Nq, k], sorted ascending by (distance, index)
    -- the tie-break convention this module treats as canonical.
    """
    Xf = np.ascontiguousarray(X, dtype=np.float64)
    Qf = np.ascontiguousarray(Q, dtype=np.float64)
    n, nq = Xf.shape[0], Qf.shape[0]
    if k > n:
        raise ValueError(f"ann-search ground truth: k={k} > n_base={n}")
    idx_out = np.empty((nq, k), dtype=np.int64)
    dist_out = np.empty((nq, k), dtype=np.float64)
    for qs in range(0, nq, query_block):
        qb = Qf[qs:qs + query_block]
        q2 = np.sum(qb * qb, axis=1)
        best_d2 = None
        best_idx = None
        for bs in range(0, n, base_block):
            xb = Xf[bs:bs + base_block]
            x2 = np.sum(xb * xb, axis=1)
            d2 = q2[:, None] - 2.0 * (qb @ xb.T) + x2[None, :]
            np.maximum(d2, 0.0, out=d2)  # clamp fp round-off near zero
            idxb = np.broadcast_to(np.arange(bs, bs + xb.shape[0]), d2.shape)
            if best_d2 is None:
                cat_d2, cat_idx = d2, idxb
            else:
                cat_d2 = np.concatenate([best_d2, d2], axis=1)
                cat_idx = np.concatenate([best_idx, idxb], axis=1)
            is_last = bs + base_block >= n
            if cat_d2.shape[1] > 4 * k or is_last:
                kth = min(k, cat_d2.shape[1]) - 1
                keep = np.argpartition(cat_d2, kth=kth, axis=1)[:, :k]
                best_d2 = np.take_along_axis(cat_d2, keep, axis=1)
                best_idx = np.take_along_axis(cat_idx, keep, axis=1)
            else:
                best_d2, best_idx = cat_d2, cat_idx
        for r in range(qb.shape[0]):
            order = np.lexsort((best_idx[r], best_d2[r]))  # primary: dist, tie: index
            idx_out[qs + r] = best_idx[r][order]
            dist_out[qs + r] = np.sqrt(best_d2[r][order])
    return idx_out, dist_out


def _direct_pairwise_knn(X: np.ndarray, Q: np.ndarray, k: int, base_block: int = 2000) -> np.ndarray:
    """
    Direct (non-Gram) pairwise squared-L2 via broadcasted subtraction,
    batched over the BASE set (the opposite chunking axis from
    _bruteforce_ground_truth's query-batched Gram-matrix expansion), no
    matmul anywhere, running top-k refresh via concatenation. A genuinely
    different numerical route to the same textbook distance -- see module
    docstring. This IS NumpyBruteForceKNN's timed run() body: deliberately
    not the fastest possible brute force (that would be the Gram trick
    above), because it must be a DIFFERENT implementation from the ground-
    truth builder, not a copy of it.
    """
    n, nq = X.shape[0], Q.shape[0]
    if k > n:
        raise ValueError(f"ann-search: k={k} > n_base={n}")
    cur_idx = np.empty((nq, 0), dtype=np.int64)
    cur_d2 = np.empty((nq, 0), dtype=np.float64)
    for s in range(0, n, base_block):
        xb = X[s:s + base_block]
        diff = Q[:, None, :] - xb[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)                    # [Nq, c]
        idxb = np.broadcast_to(np.arange(s, s + xb.shape[0]), d2.shape)
        cur_d2 = np.concatenate([cur_d2, d2], axis=1)
        cur_idx = np.concatenate([cur_idx, idxb], axis=1)
        is_last = s + base_block >= n
        if cur_d2.shape[1] > 4 * k or is_last:
            kth = min(k, cur_d2.shape[1]) - 1
            keep = np.argpartition(cur_d2, kth=kth, axis=1)[:, :k]
            cur_d2 = np.take_along_axis(cur_d2, keep, axis=1)
            cur_idx = np.take_along_axis(cur_idx, keep, axis=1)
    order = np.argsort(cur_d2, axis=1)
    return np.take_along_axis(cur_idx, order, axis=1)


def _canonical_topk_distances(X: np.ndarray, Q: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Recompute fp64 L2 distances for the RETURNED indices (whatever an
    impl's run() produced) against the actual query vectors, sort each row
    ascending. Shared across every impl's to_host in this module -- fine
    per DOMAIN_GUIDE's audit ruling ("sharing it across two IMPLEMENTATIONS
    is fine; only a reference may not share code with an implementation it
    gates") -- never called by reference_ann or by _bruteforce_ground_truth.
    A wrong neighbor set produces a sorted array that does not match the
    ground truth; a correct one does, regardless of internal tie-break
    order -- this IS the "compare distance multisets... with tie tolerance"
    check the task calls for, implemented via canonicalization rather than
    a bespoke set-comparison harness mode.
    """
    idx = np.asarray(idx, dtype=np.int64)
    pts = X[idx]                                    # [Nq, k, D]
    diff = pts - Q[:, None, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))        # [Nq, k]
    d.sort(axis=1)
    return d


def _recall_at_k(idx: np.ndarray, gt_idx: np.ndarray) -> float:
    """
    recall@k = mean over queries of |returned ∩ ground_truth| / k -- the
    ann-benchmarks/PathWeaver convention (index-set intersection; safe here
    since tied distances are measure-zero for continuous high-D data,
    unlike the exact-spatial variant's low-D synthetic/grid data). Shared
    helper, same cross-impl sharing allowance as _canonical_topk_distances.
    """
    idx = np.asarray(idx, dtype=np.int64)
    gt_idx = np.asarray(gt_idx, dtype=np.int64)
    k = gt_idx.shape[1]
    total = 0
    for i in range(idx.shape[0]):
        total += len(set(idx[i, :k].tolist()) & set(gt_idx[i, :k].tolist()))
    return total / (idx.shape[0] * k)


# ----------------------------------------------------------------- cost rule
def _cost_ann_search(w: AnnWorkload, params: dict) -> tuple[int, int]:
    """
    QPS = queries/second, the spec's own primary throughput unit for BOTH
    variants (metric.primary/secondary, literally). harness/metrics.py's
    throughput() is hardwired to divide by 1e9 for every kernel in this
    codebase (see ml.py's _cost_attention docstring for the identical issue
    with GFLOP/s); the workaround here is the same shape: work_count is the
    per-call query count SCALED BY 1e9, so the reported `throughput` field
    equals literal queries/second (not a giga-scaled figure), and "QPS" is
    registered as the unit name honestly, matching what the field actually
    contains after this cancellation.

    `bytes`: one query row read per call (compulsory lower bound; the base
    set/index is resident across calls, not re-read per query, per every
    other domain's own "each datum counted once" convention).
    """
    n_per_call = w.n_queries  # saturating-batch mode; see module docstring
    work = int(n_per_call) * 1_000_000_000
    itemsize = 4  # fp32 coordinates, per spec's dtype convention
    byts = int(n_per_call) * int(w.dim) * itemsize
    return work, byts


workload.register_cost("ann-search", _cost_ann_search, "QPS")


# ----------------------------------------------------------------- reference
def reference_ann(w: AnnWorkload, params: dict):
    """
    Independent of every impl in CPU_IMPLS: reads the workload's cached
    ground truth (built once by _bruteforce_ground_truth, a function no
    impl calls) rather than recomputing anything at run time. See module
    docstring for why each variant's "reference array" takes the shape it
    does (gt_dist directly for the exact variant; a constant zero-gap
    target for the recall variant, whose real answer key -- gt_idx -- is
    consumed inside each impl's to_host(), not here).
    """
    if w.variant == EXACT_VARIANT:
        return w.gt_dist, None
    return np.zeros(1, dtype=np.float64), None


REFERENCES = {"ann-search": reference_ann}
REFERENCE_NAME = {
    "ann-search": ("brute-force fp64 ground truth (workload-build-time, "
                   "independent of every impl -- see _bruteforce_ground_truth)"),
}
CORRECTNESS_MODE = {"ann-search": "max_abs_err"}
DEFAULT_PRECISION = {"ann-search": "fp32"}
DIM_KEY: dict[str, str] = {}


# --------------------------------------------------------------------- impls
class NumpyBruteForceKNN:
    """
    Exact brute-force kNN via direct pairwise squared-L2 (broadcasted
    subtraction, batched over the base set) -- see _direct_pairwise_knn's
    docstring for why this is a genuinely different numerical route from
    _bruteforce_ground_truth's Gram-matrix/matmul expansion (batched over
    the query set), never sharing a code object with it, per DOMAIN_GUIDE's
    reference-independence audit ruling: this impl IS gated against the
    ground truth that function produces, so it must not share its code.

    Competes on BOTH variants: it is a legitimate (if unoptimized)
    implementation of the high-dim recall variant too (recall==1.0 by
    construction, an exact method), and the natural CPU baseline for the
    exact-spatial variant. Builds no index -- prepare() only casts to fp64
    and stashes buffers, honestly near-zero preprocessing time.
    """

    name = "numpy-bruteforce-knn"
    platform = "cpu"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision

    def prepare(self, w: AnnWorkload, params: dict):
        self._X = np.ascontiguousarray(w.X, dtype=np.float64)
        self._Q = np.ascontiguousarray(w.Q, dtype=np.float64)
        self._variant = w.variant
        self._gt_idx = w.gt_idx
        self._k = w.k
        self._params = params
        params["index_build"] = "none (brute-force; no spatial structure built)"
        return {"X": self._X, "Q": self._Q, "k": w.k}

    def run(self, h):
        return _direct_pairwise_knn(h["X"], h["Q"], h["k"])   # [Nq, k] indices

    def to_host(self, out):
        idx = out
        if self._variant == RECALL_VARIANT:
            recall = _recall_at_k(idx, self._gt_idx)
            self._params["achieved_recall_at_k"] = recall
            return np.array([1.0 - recall], dtype=np.float64)
        d = _canonical_topk_distances(self._X, self._Q, idx)
        gap = float(np.max(np.abs(d - _canonical_topk_distances(
            self._X, self._Q, self._gt_idx[:d.shape[0]])))) if d.shape[0] else 0.0
        self._params["max_distance_gap_vs_ground_truth"] = gap
        return d

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


CPU_IMPLS = {"ann-search": {"numpy-bruteforce-knn": NumpyBruteForceKNN}}


def cuda_impls():
    """
    No CUDA implementation shipped BY THIS MODULE -- same documented,
    honest choice graph.py's cuda_impls() makes. GPU competitors for
    ann-search are the paper artifacts under bench/artifacts/ann-search/
    <shortname>/ (clover, ...), discovered separately by
    kernelbench.artifact_registry (see ARTIFACT_GUIDE.md) -- runner.py
    merges those in alongside this dict's (empty) cuda_impls() and
    CPU_IMPLS, so `--impl <artifact-impl-name>` works without this function
    needing to know about them.
    """
    return {}


# ============================================================== smoke set
SMOKE_SEED = 20260808


def _make_recall_smoke(name: str, n: int = 3000, d: int = 32, nq: int = 64,
                       k: int = 10, seed: int = SMOKE_SEED) -> AnnWorkload:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Q = rng.standard_normal((nq, d)).astype(np.float32)
    t0 = time.perf_counter()
    gt_idx, gt_dist = _bruteforce_ground_truth(X, Q, k)
    dt = time.perf_counter() - t0
    recall_floor = 0.9
    return AnnWorkload(name=name, variant=RECALL_VARIANT, X=X, Q=Q, k=k,
                       gt_idx=gt_idx, gt_dist=gt_dist, recall_floor=recall_floor,
                       correctness_tolerance=1.0 - recall_floor, synthetic=True,
                       seed=seed, source="synthetic Gaussian N(0,1) (smoke, NOT spec-conforming)",
                       ground_truth_build_seconds=dt)


def _make_exact_smoke(name: str, n: int = 1500, d: int = 3, nq: int = 64,
                      k: int = 10, seed: int = SMOKE_SEED) -> AnnWorkload:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5.0, 5.0, size=(n, d)).astype(np.float32)
    q_idx = rng.choice(n, size=nq, replace=False)
    Q = X[q_idx].copy()  # "all-points-as-queries" convention (CLOVER/RTNN's own)
    t0 = time.perf_counter()
    gt_idx, gt_dist = _bruteforce_ground_truth(X, Q, k)
    dt = time.perf_counter() - t0
    tol = max(1e-4, float(np.median(gt_dist)) * 1e-5)
    return AnnWorkload(name=name, variant=EXACT_VARIANT, X=X, Q=Q, k=k,
                       gt_idx=gt_idx, gt_dist=gt_dist, recall_floor=None,
                       correctness_tolerance=tol, synthetic=True, seed=seed,
                       source="synthetic uniform U(-5,5)^3, CLOVER-style (smoke, NOT spec-conforming)",
                       ground_truth_build_seconds=dt)


def smoke_workloads(variant: str | None = None) -> list[AnnWorkload]:
    """
    Small, synthetic, runs anywhere in seconds. NOT spec-conforming.

    With `variant=None` (the default -- every direct caller in this module's
    own test code, and every OTHER domain's zero-arg smoke_workloads()
    convention, gets this), returns instances of BOTH variants together.
    This is deliberate, not an oversight: under `--smoke`, runner.py always
    overrides warmup/reps to fixed synthetic-protocol defaults (5/20)
    REGARDLESS of which `--variant` flag was passed (see runner.py's
    `warmup = ... if ... else (5 if args.smoke else None)`), so the CLI-
    selected variant's own protocol constants are moot in the first place.
    Correctness gating is decided entirely by each AnnWorkload's OWN
    `.variant` field (consumed by reference_ann() and by every impl's
    to_host()), which this function sets correctly per instance -- so
    running `--smoke` under either variant id, for the CPU brute-force impl
    (which accepts any D), exercises both gate types correctly.

    Some GPU artifact adapters (e.g. clover-hubs-knn, hardcoded to D=3) can
    only accept ONE variant's workload shape and would crash on the other.
    runner.py inspects this function's signature (same convention as
    dense.py's kernel-aware smoke_workloads(kernel=...)) and passes
    `variant=<the CLI's --variant>` when present, so `--variant
    exact-spatial-knn-kernel --impl clover-hubs-knn --smoke` only receives
    D=3 workloads.
    """
    recall = [
        _make_recall_smoke("smoke-recall-gaussian-32d"),
        _make_recall_smoke("smoke-recall-gaussian-64d", d=64, seed=SMOKE_SEED + 1),
    ]
    exact = [
        _make_exact_smoke("smoke-exact-uniform3d-1500"),
        _make_exact_smoke("smoke-exact-uniform3d-3000", n=3000, seed=SMOKE_SEED + 2),
    ]
    if variant == RECALL_VARIANT:
        return recall
    if variant == EXACT_VARIANT:
        return exact
    return recall + exact


# ============================================================ real datasets
_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _read_fvecs(path: str) -> np.ndarray:
    """texmex .fvecs format: each vector is [int32 dim][dim x float32]."""
    with open(path, "rb") as f:
        raw = f.read()
    a = np.frombuffer(raw, dtype=np.int32)
    dim = int(a[0])
    row_words = dim + 1
    n = a.size // row_words
    a = a[:n * row_words].reshape(n, row_words)
    return a[:, 1:].copy().view(np.float32)


def _read_ivecs(path: str) -> np.ndarray:
    """texmex .ivecs format: each vector is [int32 dim][dim x int32]."""
    with open(path, "rb") as f:
        raw = f.read()
    a = np.frombuffer(raw, dtype=np.int32)
    dim = int(a[0])
    row_words = dim + 1
    n = a.size // row_words
    a = a[:n * row_words].reshape(n, row_words)
    return a[:, 1:].copy()


def _read_ply_vertices(path: str) -> np.ndarray:
    """
    Minimal PLY vertex reader: parses the header for the `vertex` element's
    property list (any order, any of the standard scalar types), then reads
    exactly that many vertex records in ascii/binary_little_endian/
    binary_big_endian format and returns the x,y,z columns. Face data (if
    any) is never read -- vertex data always precedes it in these files, and
    ann-search only needs point positions.
    """
    with open(path, "rb") as f:
        fmt = None
        n_vertices = 0
        props: list[tuple[str, str]] = []
        in_vertex_element = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: PLY header never terminated (no end_header)")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("format"):
                fmt = text.split()[1]
            elif text.startswith("element"):
                parts = text.split()
                in_vertex_element = (parts[1] == "vertex")
                if in_vertex_element:
                    n_vertices = int(parts[2])
            elif text.startswith("property") and in_vertex_element:
                parts = text.split()
                props.append((parts[2], parts[1]))  # (name, type)
            elif text.startswith("end_header"):
                break
        names = [p[0] for p in props]
        xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
        if fmt == "ascii":
            data = np.empty((n_vertices, 3), dtype=np.float64)
            for i in range(n_vertices):
                vals = f.readline().split()
                data[i, 0] = float(vals[xi])
                data[i, 1] = float(vals[yi])
                data[i, 2] = float(vals[zi])
            return data.astype(np.float32)
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(n, endian + _PLY_TYPES[t]) for n, t in props])
        raw = np.fromfile(f, dtype=dtype, count=n_vertices)
        xyz = np.stack([raw["x"].astype(np.float32), raw["y"].astype(np.float32),
                        raw["z"].astype(np.float32)], axis=1)
        return xyz


SIFT1M_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT1M_SUBSET_N = 200_000     # bounded subsample of the real 1,000,000 base set
SIFT1M_SUBSET_NQ = 1_000      # bounded subsample of the real 10,000 queries
SIFT1M_RECALL_FLOOR = 0.8     # DRIM-ANN's own explicit recall@10 floor (survey.md)

STANFORD_BUNNY_URL = "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"

_UNIFORM_SIZES = {"uniform-1k": 1000, "uniform-100k": 100_000, "uniform-1m": 1_000_000}


def _fetch_sift1m() -> tuple[np.ndarray, np.ndarray]:
    """
    Bounded, direct-URL loader for the real SIFT1M texmex corpus (the
    "sift1m" entry in ann-highdim-recall-qps-pareto's recommended_subset):
    1M base vectors + 10K queries, 128-dim, L2 -- ~168MB compressed. Cached
    under DATA_DIR; downloaded once via urllib.request (no library auto-
    discovery, per this integration's brief). The corpus's own shipped
    sift_groundtruth.ivecs is intentionally NOT used for gating -- this
    module computes its own independent ground truth (see load_workload).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    tgz = os.path.join(DATA_DIR, "sift.tar.gz")
    base_path = os.path.join(DATA_DIR, "sift", "sift_base.fvecs")
    query_path = os.path.join(DATA_DIR, "sift", "sift_query.fvecs")
    if not (os.path.exists(base_path) and os.path.exists(query_path)):
        if not os.path.exists(tgz):
            urllib.request.urlretrieve(SIFT1M_URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(DATA_DIR)  # noqa: S202 -- trusted, fixed, direct-URL source
    return _read_fvecs(base_path), _read_fvecs(query_path)


def _load_sift1m_subset(name: str) -> AnnWorkload:
    base, query = _fetch_sift1m()
    n_sub = min(SIFT1M_SUBSET_N, base.shape[0])
    nq_sub = min(SIFT1M_SUBSET_NQ, query.shape[0])
    X = np.ascontiguousarray(base[:n_sub])
    Q = np.ascontiguousarray(query[:nq_sub])
    k = 10  # recall@10, the convention shared by DRIM-ANN/UpANNS/PathWeaver
    t0 = time.perf_counter()
    gt_idx, gt_dist = _bruteforce_ground_truth(X, Q, k)
    dt = time.perf_counter() - t0
    return AnnWorkload(
        name=name, variant=RECALL_VARIANT, X=X, Q=Q, k=k,
        gt_idx=gt_idx, gt_dist=gt_dist, recall_floor=SIFT1M_RECALL_FLOOR,
        correctness_tolerance=1.0 - SIFT1M_RECALL_FLOOR, synthetic=False, seed=-1,
        source=(f"real texmex SIFT1M corpus ({SIFT1M_URL}), BOUNDED subsample: "
                f"first {n_sub} of 1,000,000 base vectors, first {nq_sub} of "
                "10,000 queries -- full-scale would need a much larger "
                "brute-force ground-truth build than this login-node loader "
                "attempts. Ground truth is OUR OWN brute-force fp64 computation "
                "over the subsample, independent of the corpus's own shipped "
                "sift_groundtruth.ivecs (not used for gating)."),
        ground_truth_build_seconds=dt)


def _fetch_stanford_bunny() -> np.ndarray:
    os.makedirs(DATA_DIR, exist_ok=True)
    tgz = os.path.join(DATA_DIR, "bunny.tar.gz")
    ply = os.path.join(DATA_DIR, "bunny", "reconstruction", "bun_zipper.ply")
    if not os.path.exists(ply):
        if not os.path.exists(tgz):
            urllib.request.urlretrieve(STANFORD_BUNNY_URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(DATA_DIR)  # noqa: S202 -- trusted, fixed, direct-URL source
    return _read_ply_vertices(ply)


def _load_stanford_bunny(name: str) -> AnnWorkload:
    X = _fetch_stanford_bunny()
    n = X.shape[0]
    nq = min(500, n)
    rng = np.random.default_rng(SMOKE_SEED)
    q_idx = rng.choice(n, size=nq, replace=False)
    Q = X[q_idx].copy()
    k = 50  # spec's own stanford-bunny recommended_subset k
    t0 = time.perf_counter()
    gt_idx, gt_dist = _bruteforce_ground_truth(X, Q, k)
    dt = time.perf_counter() - t0
    tol = max(1e-4, float(np.median(gt_dist)) * 1e-5)
    return AnnWorkload(
        name=name, variant=EXACT_VARIANT, X=X, Q=Q, k=k,
        gt_idx=gt_idx, gt_dist=gt_dist, recall_floor=None, correctness_tolerance=tol,
        synthetic=False, seed=SMOKE_SEED,
        source=(f"real Stanford 3D Scanning Repository mesh ({STANFORD_BUNNY_URL}), "
                f"bun_zipper.ply, {n} vertices; {nq} held-out points as queries"),
        ground_truth_build_seconds=dt)


def _load_uniform_named(key: str) -> AnnWorkload:
    n = _UNIFORM_SIZES[key]
    # Bounded held-out query count regardless of N: CLOVER's own convention
    # is all-points-as-queries, which at n=1e6 would need a ~1e6 x 1e6
    # brute-force ground truth -- infeasible on a login node. A documented,
    # bounded subset (500 points) is used for every size instead.
    nq = min(500, n)
    rng = np.random.default_rng(SMOKE_SEED)
    X = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    q_idx = rng.choice(n, size=nq, replace=False)
    Q = X[q_idx].copy()
    k = 30  # CLOVER's own k for its synthetic size sweep
    t0 = time.perf_counter()
    gt_idx, gt_dist = _bruteforce_ground_truth(X, Q, k)
    dt = time.perf_counter() - t0
    tol = max(1e-4, float(np.median(gt_dist)) * 1e-5)
    return AnnWorkload(
        name=key, variant=EXACT_VARIANT, X=X, Q=Q, k=k,
        gt_idx=gt_idx, gt_dist=gt_dist, recall_floor=None, correctness_tolerance=tol,
        synthetic=True, seed=SMOKE_SEED,
        source=(f"synthetic, U(-5,5)^3, CLOVER's own size sweep (recommended_"
                f"subset {key!r}); {nq} held-out queries (bounded -- see loader "
                "docstring for why all-points-as-queries is not used at this N)"),
        ground_truth_build_seconds=dt)


def load_workload(name: str) -> AnnWorkload:
    """
    Fetch/build a named workload from the spec's recommended_subset. Wired:
    'sift1m' (real, BOUNDED-subsampled), 'stanford-bunny' (real),
    'uniform-1k'/'uniform-100k'/'uniform-1m' (synthetic, CLOVER's own named
    sizes). Every other recommended_subset name raises NotImplementedError
    with the specific reason (see the message below) rather than silently
    falling back to something not actually the named dataset.
    """
    key = name.lower().strip()
    if key in _UNIFORM_SIZES:
        return _load_uniform_named(key)
    if "sift1m" in key or key == "sift1m":
        return _load_sift1m_subset(name)
    if "bunny" in key:
        return _load_stanford_bunny(name)
    raise NotImplementedError(
        f"ann-search: {name!r} is in the spec's recommended_subset but has no "
        "bounded direct-URL loader in this integration pass. Wired real "
        "datasets: 'sift1m' (texmex corpus, subsampled) and 'stanford-bunny' "
        "(Stanford 3D Scanning Repository); wired synthetic: 'uniform-1k' / "
        "'uniform-100k' / 'uniform-1m' (CLOVER's own size sweep). NOT wired, "
        "with reasons: sift100m/sift1b/deep100m/deep1b/spacev1b (multi-GB-to-"
        "multi-TB, out of scope for a login-node loader); stanford-asian-"
        "dragon (large binary PLY, not fetched in this pass); kitti-frame "
        "(KITTI requires registration, no anonymous direct URL); millennium-"
        "nbody (Millennium Simulation data release requires registration); "
        "clover-mesh (CLOVER's own paper never names its mesh files -- "
        "survey.md's own open_questions flags this as unrecoverable from the "
        "source read). Use smoke_workloads() for a working, fast smoke set.")
