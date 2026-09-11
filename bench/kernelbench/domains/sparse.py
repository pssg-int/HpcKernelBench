"""
Sparse linear algebra: spmv, spmm, sddmm, spgemm, gnn-aggregation, sptrsv
(+ spmspv planned).

This is the reference domain module; other domains follow its shape:

  WORKLOADS   — how to build/fetch inputs, and a smoke set that runs anywhere
  COST        — flop/byte rules registered with the workload registry
  REFERENCE   — an fp64 reference returning (result, scale) where `scale` is the
                magnitude of the computation, used by the cancellation-robust gate
  IMPLS       — CPU and CUDA implementations

## gnn-aggregation (benchspecs/gnn-aggregation/spec.yaml)

H' = A_hat @ H, A_hat = D^-1/2 (A + I) D^-1/2 — the Kipf & Welling (ICLR'17)
GCN symmetric normalization. This is the normalization named explicitly by
gnn-agg-layer-forward-real-features and gnn-agg-epoch-training-e2e ("2-layer
GCN, D^-1/2 A D^-1/2 normalization") — 3 of the spec's 4 variants center on a
GCN model, so this is the track's headline operation. gnn-agg-kernel-f32
alone additionally wants the UNnormalized A@X reported side by side
("operand_variant: both plain A@X and normalized ... reported"); this domain
ships exactly one CPU_IMPLS entry per DOMAIN_GUIDE.md's contract, so only the
normalized variant is implemented here — plain A@X is already covered
verbatim by the spmm kernel above (same class of computation, unnormalized).
Self-loops are added unconditionally via sparse A + I (scipy performs a real
sparse add, merging any pre-existing self-loop entry rather than duplicating
it), which matches "self-loops added if not already present" for this
track's (loop-free) SNAP/citation/protein graphs and is the literal Kipf &
Welling formula. Workloads reuse the plain adjacency Matrix type (same as
spmv/spmm/sddmm, unnormalized); normalization happens in prepare()/
reference() as preprocessing, not baked into load_workload(), so
smoke_workloads() below (shared, zero-arg, synthetic) needs no
gnn-aggregation-specific branch.

## sptrsv (benchspecs/sptrsv/spec.yaml)

Solve L x = b for x, L sparse lower-triangular n x n (unit diagonal, per the
spec's canonical `triangular_factor_derivation`), b/x dense, nrhs=1. Same
"reuse the plain Matrix workload, derive the real input inside prepare()/
reference()" shape as gnn-aggregation above: `load_workload`/
`smoke_workloads` hand back an ordinary SuiteSparse/synthetic matrix A, and
`kernelbench.impls.cpu_ref.unit_lower_triangular` (shared by the one
CPU_IMPLS entry and the reference -- see that function's docstring for why
sharing a WORKLOAD-DERIVATION helper does not violate the reference-
independence ruling) turns it into L = strict-lower(A) + unit diagonal, the
majority (4/5) structural-proxy practice this track's own papers use, fixing
survey.md's Divergence #1 with ONE canonical method rather than the field's
three different ad hoc ones. This is the "triangular-part-of-arbitrary-
matrix" variant spec.yaml sanctions for the 3 single-device variants
(sptrsv-solve-kernel / sptrsv-analysis-phase / sptrsv-e2e-amortized); the
4th variant, sptrsv-distributed-scaling, wants genuine LU-factor input with
real fill-in instead and is not implemented by this module.

Cost model follows sptrsv-solve-kernel's `metric.secondary` field literally:
GFLOP/s = 2*nnz(L)/time. nnz(L) (NOT nnz(A)) is what counts, stashed into
`params` by ScipyGnnAggregation.prepare()'s established pattern.

Analysis/solve split: this track's spec gives the analysis (level-set/DAG/
schedule construction, or here, the trivial L-extraction) its own first-
class timing as `preprocessing_ms` -- which harness.run_variant() already
computes generically (impl.prepare(), timed once, reported separately from
the timed run() reps) for every kernel in this codebase. No sptrsv-specific
harness change was needed for sptrsv-analysis-phase or sptrsv-e2e-amortized:
the latter's "amortized (analysis + k*solve)/k" number is exactly
harness.py's existing `throughput_amortized_over_reps` metric (divisor is
this run's rep count, not necessarily the spec's k -- already flagged by
that field's own name).
"""

from __future__ import annotations

import dataclasses
import os
import urllib.request

import numpy as np
import scipy.sparse as sp

from .. import matrices, workload
from ..impls import cpu_ref

KERNELS = ["spmv", "spmm", "sddmm", "spgemm", "gnn-aggregation", "sptrsv"]
# declared but not yet implemented here; listed so `--list` can say so honestly
PLANNED = ["spmspv"]

ITEMSIZE = {"fp64": 8, "fp32": 4, "fp16": 2, "tf32": 4, "int8": 1}


# ------------------------------------------------------------------ workloads
SMOKE = [
    ("smoke-uniform", dict(rows=4000, cols=4000, nnz_per_row=24, pattern="uniform")),
    ("smoke-banded", dict(rows=4000, cols=4000, nnz_per_row=24, pattern="banded")),
    ("smoke-powerlaw", dict(rows=4000, cols=4000, nnz_per_row=12, pattern="powerlaw")),
]


def smoke_workloads(kernel: str | None = None):
    """
    Shared, zero-arg-compatible smoke set for every kernel in this module
    EXCEPT sptrsv, which needs one pattern excluded -- see below. Optional
    `kernel` param opts into the same per-kernel branch runner.py already
    supports (`inspect.signature(...).parameters` check), without breaking
    the original zero-arg contract every other caller/kernel here relies on.

    sptrsv finding: SMOKE's "smoke-banded" pattern (entries clustered within
    a narrow +/-12-column window of the diagonal, U(-1,1) magnitude) is a
    fine stress matrix for spmv/spmm/sddmm/spgemm/gnn-aggregation (numeric
    conditioning is irrelevant to those kernels), but once run through
    sptrsv's canonical structural-proxy derivation
    (cpu_ref.unit_lower_triangular: strict-lower(A) + unit diagonal, values
    AS-IS) it produces a genuinely, catastrophically ill-conditioned
    triangular system: tight coupling between each row and its immediate
    predecessors compounds forward substitution's growth multiplicatively
    over thousands of rows and overflows float64 (confirmed:
    max_scaled_err > 1e180, not a subtle precision issue -- an actually
    diverging solution). This is a real numerical property of the "raw
    triangular part of an arbitrary matrix" majority practice this track's
    own spec.yaml notes_on_fairness already flags as a structural PROXY, not
    a genuine factor -- not something to paper over by loosening the gate.
    "smoke-uniform"/"smoke-powerlaw" spread each row's nonzeros across the
    FULL column range rather than clustering them near the diagonal, which
    stays numerically bounded (confirmed: both pass at max_scaled_err<1e-13
    against the same 1e-9 gate), so sptrsv's smoke set is the same SMOKE
    list minus "banded" -- still synthetic/NOT spec-conforming, just not
    handed a fixture this kernel's own derivation makes unstable by
    construction.
    """
    entries = SMOKE
    if kernel == "sptrsv":
        entries = [(n, kw) for n, kw in SMOKE if kw["pattern"] != "banded"]
    mats = [matrices.synthetic(n, **kw) for n, kw in entries]
    if kernel == "gnn-aggregation":
        # GCN adjacency weights are nonnegative by construction (a real graph
        # is either unweighted (0/1) or carries nonnegative edge weights);
        # matrices.synthetic()'s SIGNED U(-1,1) weights are fine for
        # spmv/spmm/sddmm/spgemm (sign is irrelevant to those kernels) but
        # break this kernel's D^-1/2(A+I)D^-1/2 normalization structurally:
        # a meaningful fraction of rows sum to a near-zero-magnitude SIGNED
        # degree, so `deg_row**-0.5` is either undefined (deg<=0 -- masked to
        # 0 by ScipyGnnAggregation/reference_gnn_aggregation, per this
        # module's own docstring) or explodes for a tiny positive deg. On
        # the masked-to-0 rows specifically: the CPU reference produces a
        # structurally EXACT-zero row of A_hat, so its gate scale
        # `|A_hat|@|X|` is exactly 0 there too, floored to 1e-300 by
        # harness.check_correctness -- any artifact whose own kernel algebra
        # does not preserve exact-zero the same way (observed for real:
        # Strassen-decomposed SpMM residues at ~1e-8, see
        # artifacts/gnn-aggregation/stragcn/STATUS.md) divides a tiny but
        # nonzero numerator by that ~0 floor and reports max_scaled_err in
        # the 1e290+ range, independent of the artifact's actual numerical
        # accuracy. Taking the absolute value of the smoke weights keeps
        # every other structural property (sparsity pattern, row-length
        # distribution) identical while making every row's degree sum
        # well-defined and strictly positive, matching every real graph in
        # gnn-aggregation's recommended_subset (SNAP/citation/protein
        # adjacency, always nonnegative).
        for m in mats:
            m.csr.data[:] = np.abs(m.csr.data)
    return mats


# gnn-aggregation's recommended_subset mixes SuiteSparse-collection names
# (most of the 24) with a few classic Planetoid citation graphs (cora,
# citeseer, pubmed) that SuiteSparse does not host. Both are handled by the
# single load_workload() below, per this module's shared, kernel-agnostic
# load_workload(name) contract (the runner calls it without knowing which
# kernel a matrix name belongs to).
_PLANETOID_NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
_GRAPH_CACHE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "graphs"))


def _resolve_suitesparse_case(name: str) -> str | None:
    """
    matrices.py's SuiteSparse lookup is case-sensitive by design (it never
    silently coerces an ambiguous name). gnn-aggregation's spec text doesn't
    always match the collection's own casing byte-for-byte (e.g.
    'com-amazon' in the spec vs 'com-Amazon' in SuiteSparse) -- this
    resolves a case-insensitive match against the same cached ssstats.csv
    index matrices.py uses, and returns the CANONICAL name matrices.py needs
    (its download URL and the .mtx member inside the tarball are both
    case-sensitive), or None if the name isn't in the index at all.
    """
    os.makedirs(matrices.CACHE, exist_ok=True)
    idx = os.path.join(matrices.CACHE, "ssstats.csv")
    if not os.path.exists(idx):
        urllib.request.urlretrieve(matrices.SS_INDEX_URL, idx)
    low = name.lower()
    with open(idx) as f:
        for line in f.read().splitlines()[2:]:
            parts = line.split(",")
            if len(parts) > 2 and parts[1].lower() == low:
                return parts[1]
    return None


def _load_planetoid(key: str) -> matrices.Matrix:
    """
    Bounded loader for the three classic Planetoid citation graphs named in
    gnn-agg-kernel-f32's recommended_subset (cora/citeseer/pubmed) -- NOT a
    general dataset-discovery mechanism; only these three names resolve
    here, nothing else falls through to torch_geometric. Uses
    torch_geometric.datasets.Planetoid, which is already present in this
    machine's venv (no new heavyweight dependency introduced), cached under
    bench/graphs/planetoid/ per this task's caching requirement. The
    adjacency is rebuilt from edge_index via scipy.sparse here, independent
    of anything torch_geometric itself does internally for message passing.
    """
    import time as _time
    t0 = _time.perf_counter()
    from torch_geometric.datasets import Planetoid
    root = os.path.join(_GRAPH_CACHE, "planetoid")
    ds = Planetoid(root=root, name=_PLANETOID_NAMES[key])
    data = ds[0]
    n = int(data.num_nodes)
    ei = data.edge_index.numpy()
    A = sp.coo_matrix((np.ones(ei.shape[1], dtype=np.float64), (ei[0], ei[1])),
                      shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0
    A.sort_indices()
    return matrices.Matrix(name=key, csr=A, source="planetoid",
                           group="torch_geometric",
                           load_seconds=_time.perf_counter() - t0)


def load_workload(name: str):
    canon = _resolve_suitesparse_case(name)
    if canon is not None:
        return matrices.load_matrix(canon)
    if name.lower() in _PLANETOID_NAMES:
        return _load_planetoid(name.lower())
    raise LookupError(
        f"{name!r} not found in the cached SuiteSparse index and is not one "
        f"of the Planetoid names this loader covers "
        f"({sorted(_PLANETOID_NAMES)}). gnn-agg-kernel-f32's full "
        "recommended_subset also includes GNNAdvisor-lineage/TUDataset names "
        "(ppi, artist, soc-BlogCatalog, OVCAR-8H, YeastH, DD, PROTEINS_full, "
        "COLLAB) hosted only via Zenodo/Google-Drive mirrors that this "
        "bounded loader deliberately does not fetch (see DOMAIN_GUIDE.md: "
        "'download only the named datasets, no dataset-library "
        "auto-discovery').")


def variant_transform(kernel: str, variant_id: str, workload):
    """
    Domain hook: runner.py calls this on every workload (smoke or loaded via
    load_workload) right after it is built, before the reference or any
    implementation ever sees it (`xf = getattr(domain, "variant_transform",
    None); mats = [xf(args.kernel, variant.id, m) for m in mats] if xf else
    mats`). Returning the workload UNCHANGED is the default for every
    (kernel, variant) pair not named below -- this hook is purely additive
    and every other variant in this codebase is unaffected.

    spmm-binary-adjacency-kernel (benchspecs/spmm/spec.yaml) requires that
    EVERY implementation gated under it, INCLUDING the fp64 CSR reference,
    consume the SAME binarized matrix (structure/pattern unchanged, every
    stored value forced to 1.0) -- see that variant's `inputs.selection`
    field. `cpu_ref.reference_spmm` and every CPU/CUDA/artifact
    implementation in this codebase read `workload.csr` directly (none keeps
    a separate private copy of the sparse values), so binarizing
    `workload.csr` once, here, is sufficient to binarize the input for the
    entire run -- no reference- or implementation-specific change is needed.
    This is exactly the regime dtcspmm/flashsparse/generalsparse's released
    code is structurally limited to (see their STATUS.md findings, and this
    spec's `bench_finding_2026-09-06` evidence entry): they hardcode the
    sparse operand's values to 1.0 internally, so gating them here (rather
    than under the general weighted variants) tests them on the input class
    their kernel actually computes.

    Returns a COPY: `workload.csr.copy()` (a real, independent scipy CSR
    copy: separate data/indices/indptr arrays) plus `dataclasses.replace` for
    the rest of the Matrix fields, so the process-wide matrix cache
    (`matrices.load_matrix`/`_load_planetoid` results, reused across --dims
    sweeps and by other variants/impls in the same run) is never mutated in
    place. `name` is suffixed `+bin` so result records/logs distinguish a
    binarized run from the same-named matrix's general-variant run; the
    extra `binary_pattern = True` attribute lets an adapter that wants to
    know explicitly (rather than infer it from `np.all(csr.data == 1)`)
    check for it.
    """
    if kernel != "spmm" or variant_id != "spmm-binary-adjacency-kernel":
        return workload
    csr = workload.csr.copy()
    csr.data[:] = 1  # assigns into the existing array -- dtype is preserved, not reset to a default
    binarized = dataclasses.replace(workload, csr=csr, name=f"{workload.name}+bin")
    binarized.binary_pattern = True
    return binarized


# ----------------------------------------------------------------- cost rules
def _dim(params: dict) -> int:
    return int(params.get("N") or params.get("K") or 1)


def _cost_spmv(m, params):
    vb = ITEMSIZE[params.get("precision", "fp64")]
    ib = 4
    rows, cols = m.shape
    nnz = params.get("unfolded_nnz") or m.nnz
    flops = 2 * nnz
    byts = m.nnz * (vb + ib) + (rows + 1) * ib + cols * vb + rows * vb
    return flops, int(byts)


def _cost_spmm(m, params):
    vb = ITEMSIZE[params.get("precision", "fp32")]
    ib = 4
    rows, cols = m.shape
    n = _dim(params)
    flops = 2 * m.nnz * n
    byts = m.nnz * (vb + ib) + (rows + 1) * ib + cols * n * vb + rows * n * vb
    return flops, int(byts)


def _cost_sddmm(m, params):
    vb = ITEMSIZE[params.get("precision", "fp32")]
    ib = 4
    rows, cols = m.shape
    k = _dim(params)
    flops = 2 * m.nnz * k
    byts = m.nnz * ib + (rows + 1) * ib + rows * k * vb + cols * k * vb + m.nnz * vb
    return flops, int(byts)


def _cost_spgemm(m, params):
    """
    C = A @ A (self-product, per this track's `operation` field).

    flops_naive = sum, over every nonzero A[i,k], of nnz(row k of A) -- the
    naive/symbolic intermediate-product count BEFORE duplicate-index
    compression (Gustavson-style row-wise SpGEMM cost), matching the
    Ocean/TileSpGEMM convention benchspecs/spgemm/survey.md documents
    repeatedly ("FLOPs = 2x intermediate products", "twice the number of
    intermediate products") and every `evidence` citation in spec.yaml
    restates.

    NOTE on spec.yaml wording: `metric.primary`'s parenthetical literally
    reads "sum over i,k of nnz(A row i)*nnz(B row k) for each A[i,k]!=0" --
    taken completely literally that adds an extra nnz(A row i) factor,
    producing a formula that matches NEITHER Ocean's nor TileSpGEMM's own
    published definition NEITHER survey.md's own repeated restatement of
    "2x intermediate products" NOR any `evidence` line in the same spec.yaml
    file. Treated as spec prose imprecision (a stray "nnz(A row i)*" left
    over from paraphrasing), not a deliberate literal instruction to diverge
    from the rest of the same document -- implementing the literal reading
    would make this track's numbers incomparable with the very papers the
    spec is built from. Flagged in STATUS.md / the integration report so a
    human can amend spec.yaml if the imprecision was unintentional.

    Byte count follows spmm's A/B/C accounting (`_cost_spmm` above): read A
    once (values+indices+rowptr), write the TRUE C once (values+indices+
    rowptr) -- nnz(C) is computed exactly here (a real, untimed CPU SpGEMM;
    this function runs once per run_variant call, after all timed reps, so
    it never pollutes any measured metric) rather than upper-bounded by
    flops_naive, since `bytes` is documented as a compulsory-traffic LOWER
    bound, not a symbolic estimate.
    """
    A = m.csr
    row_nnz = np.diff(A.indptr).astype(np.int64)
    naive = int(row_nnz[A.indices].sum())
    flops = 2 * naive

    vb = ITEMSIZE.get(params.get("precision", "fp64"), 8)
    ib = 4
    rows, _cols = A.shape
    C = (A.astype(np.float64) @ A.astype(np.float64)).tocsr()
    C.sum_duplicates()
    C.eliminate_zeros()
    byts = (A.nnz * (vb + ib) + (rows + 1) * ib +
            C.nnz * (vb + ib) + (rows + 1) * ib)
    return flops, int(byts)


def _cost_gnn_agg(m, params):
    """
    Per gnn-agg-kernel-f32's metric field literally: 'GFLOP/s (2*nnz*N
    flops)'. nnz here is A_hat's nnz (post self-loop add via A+I), NOT the
    raw input adjacency's nnz -- cpu_ref.ScipyGnnAggregation.prepare()
    stashes the real count into params['gnn_agg_nnz'] once normalization has
    run; before that (or for an impl that doesn't stash it) this falls back
    to nnz(A) + rows, the exact count for a loop-free input (this track's
    SNAP/citation graphs), matching the A+I self-loop convention documented
    in this module's docstring.
    """
    vb = ITEMSIZE[params.get("precision", "fp32")]
    ib = 4
    rows, cols = m.shape
    n = _dim(params)
    nnz = params.get("gnn_agg_nnz")
    if nnz is None:
        nnz = m.nnz + rows
    flops = 2 * nnz * n
    byts = nnz * (vb + ib) + (rows + 1) * ib + cols * n * vb + rows * n * vb
    return flops, int(byts)


def _cost_sptrsv(m, params):
    """
    Per sptrsv-solve-kernel's `metric.secondary` field literally:
    'GFLOP/s = 2*nnz(L)/(time_ns)'. nnz(L) is the CANONICAL derived
    triangular factor's nnz (unit-lower-triangular-of-A, see
    cpu_ref.unit_lower_triangular), NOT nnz(A) -- ScipySpTRSV.prepare() and
    reference_sptrsv() both stash the real count into params['sptrsv_L_nnz']
    once the derivation has run (same stash-into-params pattern as
    _cost_gnn_agg above); before that (or for an impl -- e.g. a paper
    artifact adapter -- that doesn't stash it) this recomputes it directly
    via the same canonical derivation, cheap even on the recommended_subset's
    largest matrices (one sparse add + one sort).

    `metric.primary` is wall time (ms), not GFLOP/s -- registered here as
    "GFLOP/s" anyway (spec's own SECONDARY metric label) purely for the
    secondary-metric field every other kernel in this file also reports
    alongside `stats_ms`; the primary number lives in `stats_ms`/`times_ms`
    regardless of this registration, per harness.run_variant.

    Bytes: read L once (values+indices+rowptr) + read b + write x (both
    dense, length rows) -- the compulsory-traffic lower bound, same
    accounting style as `_cost_spmv` above.
    """
    vb = ITEMSIZE[params.get("precision", "fp64")]
    ib = 4
    rows, _cols = m.shape
    nnz_l = params.get("sptrsv_L_nnz")
    if nnz_l is None:
        from ..impls import cpu_ref
        nnz_l = int(cpu_ref.unit_lower_triangular(m.csr).nnz)
    flops = 2 * nnz_l
    byts = nnz_l * (vb + ib) + (rows + 1) * ib + rows * vb + rows * vb
    return flops, int(byts)


workload.register_cost("spmv", _cost_spmv, "GFLOP/s")
workload.register_cost("spmm", _cost_spmm, "GFLOP/s")
workload.register_cost("sddmm", _cost_sddmm, "GFLOP/s")
workload.register_cost("spgemm", _cost_spgemm, "GFLOP/s")
workload.register_cost("gnn-aggregation", _cost_gnn_agg, "GFLOP/s")
workload.register_cost("sptrsv", _cost_sptrsv, "GFLOP/s")


# ----------------------------------------------------------------- references
REFERENCES = {
    "spmv": cpu_ref.reference_spmv,
    "spmm": cpu_ref.reference_spmm,
    "sddmm": cpu_ref.reference_sddmm,
    "spgemm": cpu_ref.reference_spgemm,
    "gnn-aggregation": cpu_ref.reference_gnn_aggregation,
    "sptrsv": cpu_ref.reference_sptrsv,
}

# gate mode per kernel: scale-aware everywhere (see harness.check_correctness)
CORRECTNESS_MODE = {k: "max_scaled_err" for k in KERNELS}

DEFAULT_PRECISION = {"spmv": "fp64", "spmm": "fp32", "sddmm": "fp32", "spgemm": "fp64",
                     "gnn-aggregation": "fp32", "sptrsv": "fp64"}

# reference provenance disclosure (runner.py falls back to "scipy fp64 CSR"
# for every kernel not listed here, accurate for spmv/spmm/sddmm/spgemm/
# gnn-aggregation, whose references genuinely are scipy-computed). sptrsv's
# reference is deliberately NOT scipy-backed (cpu_ref._forward_substitute,
# an independent pure-Python/NumPy sweep) -- see reference_sptrsv's
# docstring -- so it needs its own honest label here.
REFERENCE_NAME = {"sptrsv": "independent fp64 forward-substitution (non-scipy CSR row-sweep)"}


# --------------------------------------------------------------------- impls
CPU_IMPLS = {
    "spmv": {"scipy-csr-spmv": cpu_ref.ScipySpMV,
             "naive-csr-spmv": cpu_ref.NaiveSpMV},
    "spmm": {"scipy-csr-spmm": cpu_ref.ScipySpMM},
    "sddmm": {"scipy-csr-sddmm": cpu_ref.ScipySDDMM},
    "spgemm": {"scipy-csr-spgemm": cpu_ref.ScipySpGEMM},
    "gnn-aggregation": {"scipy-csr-gnn-agg-gcnnorm": cpu_ref.ScipyGnnAggregation},
    "sptrsv": {"scipy-csr-sptrsv": cpu_ref.ScipySpTRSV},
}


def cuda_impls():
    from ..impls import gpu_cuda as g
    return {
        "spmv": {"cusparse-csr-spmv": g.TorchSpMV,
                 "custom-warp-csr-spmv": g.CustomSpMV},
        "spmm": {"cusparse-csr-spmm": g.TorchSpMM,
                 "custom-warp-csr-spmm": g.CustomSpMM},
        "sddmm": {"torch-sampled-addmm-sddmm": g.TorchSDDMM,
                  "custom-warp-csr-sddmm": g.CustomSDDMM},
        "spgemm": {"cusparse-csr-spgemm": g.TorchSpGEMM},
        # gnn-aggregation: no built-in CUDA baseline shipped by this task's
        # Part A (out of the requested scope: CPU_IMPLS + workloads + cost +
        # reference); artifact adapters under bench/artifacts/gnn-aggregation/
        # supply GPU implementations instead (see artifact_registry.py).
        # sptrsv: likewise no built-in CUDA baseline (no cuSPARSE-via-torch
        # triangular-solve wrapper shipped here); artifact adapters under
        # bench/artifacts/sptrsv/ supply the GPU implementations for this
        # kernel entirely.
    }


# which params key carries the swept dense dimension (None => no sweep)
DIM_KEY = {"spmm": "N", "sddmm": "K", "gnn-aggregation": "N"}
