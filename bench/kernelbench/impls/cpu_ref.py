"""
CPU implementations: the reference (scipy/MKL-backed) and a naive baseline.

These are what makes a result meaningful even before any GPU work exists — every
GPU number is checked against the fp64 CPU answer, and the CPU variants of the
specs (e.g. spmm-cpu-kernel-f32) are measured here directly.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..harness import Timer


def _dense_operand(rows: int, dim: int, seed: int, dtype) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(dtype)


# --------------------------------------------------------------------- SpMV
class ScipySpMV:
    name = "scipy-csr-spmv"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        self.precision = precision
        self.dtype = np.float64 if precision == "fp64" else np.float32

    def prepare(self, matrix, params):
        A = matrix.csr.astype(self.dtype)
        A.sort_indices()
        rng = np.random.default_rng(params.get("seed", 42))
        x = rng.uniform(-1.0, 1.0, size=A.shape[1]).astype(self.dtype)
        return {"A": A, "x": x}

    def run(self, h):
        return h["A"] @ h["x"]

    def to_host(self, out):
        return np.asarray(out)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


class NaiveSpMV(ScipySpMV):
    """Deliberately unoptimized CSR SpMV — the floor of the leaderboard."""

    name = "naive-csr-spmv"

    def run(self, h):
        A, x = h["A"], h["x"]
        y = np.zeros(A.shape[0], dtype=self.dtype)
        indptr, indices, data = A.indptr, A.indices, A.data
        for i in range(A.shape[0]):
            lo, hi = indptr[i], indptr[i + 1]
            y[i] = np.dot(data[lo:hi], x[indices[lo:hi]])
        return y


# --------------------------------------------------------------------- SpMM
class ScipySpMM:
    name = "scipy-csr-spmm"
    platform = "cpu"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = np.float32 if precision == "fp32" else np.float64

    def prepare(self, matrix, params):
        A = matrix.csr.astype(self.dtype)
        A.sort_indices()
        N = int(params["N"])
        B = _dense_operand(A.shape[1], N, params.get("seed", 42), self.dtype)
        return {"A": A, "B": B}

    def run(self, h):
        return h["A"] @ h["B"]

    def to_host(self, out):
        return np.asarray(out)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


# -------------------------------------------------------------------- SDDMM
class ScipySDDMM:
    """
    P[i,j] = S[i,j] * dot(A[i,:], B[j,:]) for (i,j) in nnz(S).
    Computed row-blockwise so it never materializes the dense M x N product.
    """

    name = "scipy-csr-sddmm"
    platform = "cpu"

    def __init__(self, precision: str = "fp32", block_rows: int = 256):
        self.precision = precision
        self.dtype = np.float32 if precision == "fp32" else np.float64
        self.block_rows = block_rows

    def prepare(self, matrix, params):
        S = matrix.csr.astype(self.dtype)
        S.sort_indices()
        K = int(params["K"])
        M, N = S.shape
        seed = params.get("seed", 42)
        A = _dense_operand(M, K, seed, self.dtype)
        B = _dense_operand(N, K, seed + 1, self.dtype)
        return {"S": S, "A": A, "B": B, "out": np.empty_like(S.data)}

    def run(self, h):
        S, A, B, out = h["S"], h["A"], h["B"], h["out"]
        indptr, indices, data = S.indptr, S.indices, S.data
        for lo_row in range(0, S.shape[0], self.block_rows):
            hi_row = min(lo_row + self.block_rows, S.shape[0])
            lo, hi = indptr[lo_row], indptr[hi_row]
            if hi == lo:
                continue
            cols = indices[lo:hi]
            rows = np.repeat(np.arange(lo_row, hi_row),
                             np.diff(indptr[lo_row:hi_row + 1]))
            out[lo:hi] = np.einsum("ij,ij->i", A[rows], B[cols]) * data[lo:hi]
        return out

    def to_host(self, out):
        return np.asarray(out)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


# ------------------------------------------------------------- references
# Each reference returns (result, scale). `scale` is the componentwise magnitude
# of the computation — |A|*|B| for a product — which is the denominator a
# cancellation-robust error gate needs. See harness.check_correctness.

def reference_spmv(matrix, params):
    A = matrix.csr.astype(np.float64)
    dtype = np.float32 if params.get("precision", "fp64") == "fp32" else np.float64
    rng = np.random.default_rng(params.get("seed", 42))
    x = rng.uniform(-1.0, 1.0, size=A.shape[1]).astype(dtype).astype(np.float64)
    absA = abs(A)
    return A @ x, absA @ np.abs(x)


def reference_spmm(matrix, params):
    A = matrix.csr.astype(np.float64)
    N = int(params["N"])
    dtype = np.float32 if params.get("precision", "fp32") == "fp32" else np.float64
    B = _dense_operand(A.shape[1], N, params.get("seed", 42), dtype).astype(np.float64)
    return A @ B, abs(A) @ np.abs(B)


def _canonical_pattern(A: sp.csr_matrix) -> sp.csr_matrix:
    """
    Structural nonzero pattern of |A| @ |A| -- the FIXED comparison index set
    for the spgemm correctness gate. Deterministic given A alone (same
    sparse-matmul call on the same inputs produces the same indices in the
    same sorted order every time), so both `reference_spgemm` and every
    impl's `to_host()` can independently recompute this exact index grid and
    land their values on it, without either side needing to see the other's
    raw output pattern.

    Why |A| @ |A| specifically: SpGEMM's true output pattern (nonzero
    positions of A @ A) is implementation-defined in what gets explicitly
    stored (an impl may drop an exact numerical cancellation the reference
    keeps, or vice versa) -- "output pattern order is impl-defined" per
    DOMAIN_GUIDE.md. |A| @ |A| uses all-nonnegative operands, so no
    cancellation is mathematically possible there: every position that could
    POSSIBLY be nonzero in A @ A (i.e. has at least one connecting k with
    A[i,k]!=0 and A[k,j]!=0) is guaranteed nonzero in |A| @ |A|. That makes
    it a safe structural superset -- the canonical "union pattern" the
    correctness rule calls for -- of any correct implementation's true
    output support.
    """
    Aabs = A.tocsr(copy=True)
    Aabs.data = np.abs(Aabs.data)
    Aabs.sort_indices()
    S = (Aabs @ Aabs).tocsr()
    S.sum_duplicates()
    S.eliminate_zeros()
    S.sort_indices()
    return S


def _reindex_to_pattern(mat, pattern: sp.csr_matrix) -> np.ndarray:
    """
    Extract `mat`'s values at `pattern`'s exact (row, col) positions, in
    pattern's row-major sorted order; 0.0 at positions `mat` does not store.

    Raises if `mat` has a nonzero position `pattern` does not cover: since
    `pattern` (|A|@|A|'s structural pattern) is a guaranteed mathematical
    superset of SpGEMM's true output support, an implementation reporting a
    value outside it is a real indexing bug, not a formatting mismatch that
    should be silently absorbed by the gate.
    """
    mat = mat.tocsr(copy=True)
    mat.sum_duplicates()
    mat.eliminate_zeros()
    mat.sort_indices()
    out = np.zeros(pattern.nnz, dtype=np.float64)
    p_indptr, p_indices = pattern.indptr, pattern.indices
    m_indptr, m_indices, m_data = mat.indptr, mat.indices, mat.data.astype(np.float64)
    for i in range(pattern.shape[0]):
        m_lo, m_hi = m_indptr[i], m_indptr[i + 1]
        if m_hi == m_lo:
            continue
        m_cols = m_indices[m_lo:m_hi]
        p_lo, p_hi = p_indptr[i], p_indptr[i + 1]
        p_cols = p_indices[p_lo:p_hi]
        idx = np.searchsorted(p_cols, m_cols)
        in_range = idx < len(p_cols)
        matched = np.zeros(len(m_cols), dtype=bool)
        matched[in_range] = p_cols[idx[in_range]] == m_cols[in_range]
        if not matched.all():
            bad = m_cols[~matched]
            raise ValueError(
                f"spgemm gate: row {i} has {len(bad)} nonzero(s) at column(s) "
                f"{bad[:5].tolist()}{'...' if len(bad) > 5 else ''} outside "
                "|A|@|A|'s structural pattern -- a real indexing bug, not a "
                "formatting mismatch (pattern is a guaranteed superset of "
                "any correct SpGEMM implementation's output support)")
        out[p_lo + idx[matched]] = m_data[m_lo:m_hi][matched]
    return out


# ------------------------------------------------------------------ SpGEMM
class ScipySpGEMM:
    """
    C = A @ A via scipy's sparse CSR-CSR product.

    SELF-CERTIFYING, DOCUMENTED (DOMAIN_GUIDE.md's audit ruling): this impl
    and `reference_spgemm` both call the same scipy/SciPy-sparse backend, so
    the correctness gate is bit-identical-by-construction for THIS impl and
    verifies nothing about it. It exists as the CPU floor every other impl
    (GPU library baselines, paper artifacts) is measured against for
    THROUGHPUT; the gate's real job is certifying those independent
    implementations, not this one.
    """

    name = "scipy-csr-spgemm"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        self.precision = precision
        self.dtype = np.float64 if precision == "fp64" else np.float32

    def prepare(self, matrix, params):
        A = matrix.csr.astype(self.dtype)
        A.sort_indices()
        self._matrix = matrix
        return {"A": A}

    def run(self, h):
        return h["A"] @ h["A"]

    def to_host(self, out):
        A64 = self._matrix.csr.astype(np.float64)
        A64.sort_indices()
        pattern = _canonical_pattern(A64)
        return _reindex_to_pattern(out, pattern)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


def reference_spgemm(matrix, params):
    """
    fp64 SpGEMM reference: C = A @ A via scipy's sparse CSR-CSR product.

    INDEPENDENCE NOTE (DOMAIN_GUIDE.md "Reference independence" ruling):
    `ScipySpGEMM` (CPU_IMPLS["spgemm"]'s only CPU impl) ALSO calls scipy's
    sparse '@'. A library-backed CPU impl is bit-identical to a
    library-backed reference by construction, so this reference gate
    verifies NOTHING for that impl -- documented here rather than implied
    otherwise. The gate's actual subjects are the independent
    implementations (paper artifacts: ocean/popcorn/amgt -- their own CUDA
    kernels, sharing no code with scipy), which is what this gate exists for.

    Comparison representation: returns flat arrays over the FIXED structural
    pattern of |A|@|A| (see `_canonical_pattern`), not a dense M x N matrix
    -- avoids O(M*N) memory for the multi-million-row matrices in the
    spec's recommended_subset, and correctly handles "SpGEMM output pattern
    order is impl-defined" / "if output patterns can legitimately differ...
    compare on the union pattern with the same scale" per DOMAIN_GUIDE.md:
    |A|@|A| structurally dominates A@A's pattern (no cancellation possible
    with nonnegative operands), so it is a safe canonical union pattern.
    `scale` is |A|@|A|'s own values at that same pattern -- the componentwise
    backward-error denominator DOMAIN_GUIDE.md's reference contract asks for.
    """
    A = matrix.csr.astype(np.float64)
    A.sort_indices()
    C = (A @ A).tocsr()
    pattern = _canonical_pattern(A)
    c_vals = _reindex_to_pattern(C, pattern)
    return c_vals, pattern.data.astype(np.float64)


def reference_sddmm(matrix, params):
    S = matrix.csr.astype(np.float64)
    K = int(params["K"])
    M, N = S.shape
    seed = params.get("seed", 42)
    dtype = np.float32 if params.get("precision", "fp32") == "fp32" else np.float64
    A = _dense_operand(M, K, seed, dtype).astype(np.float64)
    B = _dense_operand(N, K, seed + 1, dtype).astype(np.float64)
    out = np.empty_like(S.data)
    scale = np.empty_like(S.data)
    indptr, indices, data = S.indptr, S.indices, S.data
    absA, absB = np.abs(A), np.abs(B)
    for i in range(M):
        lo, hi = indptr[i], indptr[i + 1]
        if hi > lo:
            cols = indices[lo:hi]
            out[lo:hi] = (B[cols] @ A[i]) * data[lo:hi]
            scale[lo:hi] = (absB[cols] @ absA[i]) * np.abs(data[lo:hi])
    return out, scale


# ------------------------------------------------------------ GNN aggregation
class ScipyGnnAggregation:
    """
    H' = A_hat @ H, A_hat = D^-1/2 (A + I) D^-1/2 -- the Kipf & Welling
    (ICLR'17) GCN symmetric normalization, the model explicitly named by two
    of gnn-aggregation's four spec variants (gnn-agg-layer-forward-real-
    features and gnn-agg-epoch-training-e2e both specify "2-layer GCN,
    D^-1/2 A D^-1/2 normalization"); see kernelbench/domains/sparse.py's
    module docstring for why this is the ONE CPU_IMPLS entry this domain
    ships for the kernel (DOMAIN_GUIDE.md's one-impl-per-kernel contract).

    Self-loops: A + I unconditionally. scipy performs a real sparse add
    (merges any pre-existing self-loop entry into the +1 rather than
    duplicating it), matching "self-loops added if not already present" for
    this track's (loop-free) SNAP/citation/protein recommended graphs, and
    matching the literal Kipf & Welling formula.

    Row/column degree are computed separately (not assumed equal) so the
    formula stays well-defined on the directed graphs the recommended_subset
    also contains; it reduces to the textbook symmetric normalization when A
    is symmetric.

    REFERENCE INDEPENDENCE (DOMAIN_GUIDE.md's audit ruling): the D^-1/2(A+I)
    D^-1/2 normalization below is written inline here, and AGAIN separately
    inline inside reference_gnn_aggregation() further down this file -- same
    formula, two distinct code objects, neither calls the other, per "same
    formula re-typed inline is acceptable; same code object is not."
    """

    name = "scipy-csr-gnn-agg-gcnnorm"
    platform = "cpu"
    normalization = "gcn-sym: D^-1/2 (A+I) D^-1/2"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = np.float32 if precision == "fp32" else np.float64

    def prepare(self, matrix, params):
        # normalize in fp64 (more faithful to what a real implementation's
        # one-shot preprocessing does), cast to the requested dtype once at
        # the end -- this whole block is preprocessing (timed once, as
        # preprocessing_ms, never inside run()), matching the "format
        # conversion goes in prepare()" split every other impl in this file
        # follows.
        A = matrix.csr.astype(np.float64)
        n = A.shape[0]
        A_tilde = (A + sp.identity(n, format="csr", dtype=np.float64)).tocsr()
        A_tilde.sort_indices()
        deg_row = np.asarray(A_tilde.sum(axis=1)).ravel()
        deg_col = np.asarray(A_tilde.sum(axis=0)).ravel()
        dinv_row = np.zeros_like(deg_row)
        dinv_col = np.zeros_like(deg_col)
        nzr, nzc = deg_row > 0, deg_col > 0
        dinv_row[nzr] = deg_row[nzr] ** -0.5
        dinv_col[nzc] = deg_col[nzc] ** -0.5
        Dr = sp.diags(dinv_row)
        Dc = sp.diags(dinv_col)
        A_hat = (Dr @ A_tilde @ Dc).tocsr().astype(self.dtype)
        A_hat.sort_indices()
        # disclosure for the cost model (kernelbench/domains/sparse.py's
        # _cost_gnn_agg reads this back) and for the result record's
        # provenance -- same pattern as graph.py's prepare()-writes-to-params.
        params["gnn_agg_nnz"] = int(A_hat.nnz)
        params["gnn_agg_normalization"] = self.normalization
        F = int(params["N"])
        X = _dense_operand(A_hat.shape[1], F, params.get("seed", 42), self.dtype)
        return {"A": A_hat, "X": X}

    def run(self, h):
        return h["A"] @ h["X"]

    def to_host(self, out):
        return np.asarray(out)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


def reference_gnn_aggregation(matrix, params):
    """
    Independent fp64 reference for gnn-aggregation.

    Recomputes A_hat = D^-1/2 (A+I) D^-1/2 from scratch, in its own code (NOT
    a call into ScipyGnnAggregation.prepare() or any shared normalization
    helper -- see the reference-independence audit ruling in DOMAIN_GUIDE.md
    and the note on ScipyGnnAggregation above), then returns
    (A_hat @ X, |A_hat| @ |X|) -- the componentwise magnitude scale a
    cancellation-robust gate needs (harness.check_correctness), per the same
    (result, scale) pattern as reference_spmm above.
    """
    A = matrix.csr.astype(np.float64)
    n = A.shape[0]
    A_tilde = (A + sp.identity(n, format="csr", dtype=np.float64)).tocsr()
    row_deg = np.asarray(A_tilde.sum(axis=1)).ravel()
    col_deg = np.asarray(A_tilde.sum(axis=0)).ravel()
    # degree can be non-positive on the SIGNED synthetic smoke matrices
    # (matrices.synthetic draws U(-1,1) weights; real graphs in
    # recommended_subset are non-negative and never hit this branch) --
    # np.errstate silences the resulting invalid-power warning for the
    # entries np.where discards anyway (row_deg**-0.5 is still evaluated
    # eagerly on the whole array before the mask is applied).
    with np.errstate(invalid="ignore"):
        row_scale = np.where(row_deg > 0, row_deg ** -0.5, 0.0)
        col_scale = np.where(col_deg > 0, col_deg ** -0.5, 0.0)
    A_hat = (sp.diags(row_scale) @ A_tilde @ sp.diags(col_scale)).tocsr()

    F = int(params["N"])
    dtype = np.float32 if params.get("precision", "fp32") == "fp32" else np.float64
    X = _dense_operand(A_hat.shape[1], F, params.get("seed", 42), dtype).astype(np.float64)
    return A_hat @ X, abs(A_hat) @ np.abs(X)


# -------------------------------------------------------------------- SpTRSV
def unit_lower_triangular(A: sp.csr_matrix) -> sp.csr_matrix:
    """
    CANONICAL sptrsv workload derivation (benchspecs/sptrsv/spec.yaml's
    sptrsv-solve-kernel `triangular_factor_derivation`, the ruling that fixes
    survey.md Divergence #1 -- 4 of 5 surveyed papers extract a "triangular
    factor" from an arbitrary matrix, but do it THREE different ways):

        L = strict lower-triangular part of A (entries with col < row,
        values taken AS-IS from A) PLUS an explicit unit diagonal
        (L[i,i] = 1 for every i, never A's own diagonal entry).

    This is a STRUCTURAL PROXY for a real LU/Cholesky factor, not a genuine
    numeric factorization -- named because it is the majority (4/5) practice
    in this track and needs no factorization-library dependency (spec.yaml's
    notes_on_fairness). The explicit unit diagonal guarantees L is always
    solvable regardless of A's own diagonal, unlike YuenyeungSpTRSV's own
    artifact (divides by whatever nonzero happens to sit last in the row --
    an unstated, unverified assumption per the survey).

    Always returns fp64 regardless of A's own dtype; callers cast to the
    target precision afterward (same "normalize/derive in fp64, cast once at
    the end" convention as ScipyGnnAggregation.prepare() above).

    Shared verbatim by ScipySpTRSV.prepare() and reference_sptrsv() below --
    this is WORKLOAD DERIVATION (which triangular system is being solved),
    not the sptrsv KERNEL itself, so sharing it does not violate
    DOMAIN_GUIDE.md's reference-independence ruling any more than
    `_dense_operand` or `_canonical_pattern` (SpGEMM) above do: the part
    that is actually gated -- turning (L, b) into x -- stays independent
    (scipy's `spsolve_triangular` for ScipySpTRSV.run() vs.
    `_forward_substitute`'s pure-Python/NumPy sweep for the reference).
    """
    n = A.shape[0]
    strict_lower = sp.tril(A, k=-1, format="csr")
    L = (strict_lower + sp.identity(n, format="csr", dtype=np.float64)).tocsr()
    L.sort_indices()
    return L


def _forward_substitute(L: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
    """
    Independent (non-scipy) row-wise forward substitution for L x = b.

    Exploits the guaranteed layout `unit_lower_triangular` produces: every
    row's stored entries have col <= row, and after `sort_indices()` they
    are column-ascending, so the diagonal entry (col == row) is always the
    LAST entry in its row's slice -- the per-row dot product below excludes
    it explicitly by position rather than searching for it.

    This is the correctness reference underlying ScipySpTRSV's gate
    (DOMAIN_GUIDE.md's reference-independence ruling): ScipySpTRSV.run()
    calls `scipy.sparse.linalg.spsolve_triangular`; this function shares no
    code with that call -- just the same textbook forward-substitution
    formula re-typed independently, the ruling's explicitly-sanctioned case
    ("same formula re-typed inline is acceptable; same code object is not").
    """
    n = L.shape[0]
    indptr, indices, data = L.indptr, L.indices, L.data.astype(np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo, hi = indptr[i], indptr[i + 1]
        if hi > lo:
            cols = indices[lo:hi - 1]      # every off-diagonal entry (col < i)
            vals = data[lo:hi - 1]
            diag = data[hi - 1]            # last entry == the unit diagonal
            x[i] = (b[i] - vals @ x[cols]) / diag
        else:
            x[i] = b[i]                    # defensive: diagonal always present in practice
    return x


class ScipySpTRSV:
    """
    scipy.sparse.linalg.spsolve_triangular -- the CPU floor for sptrsv.
    Documented as self-certifying is NOT needed here the way it is for
    ScipySpGEMM above: the gate's reference (`reference_sptrsv` /
    `_forward_substitute`) is deliberately NOT scipy-backed, so comparing
    this impl against it is a genuine independent check.

    Workload: `unit_lower_triangular(A)` (see that function's docstring) for
    L, and `sptrsv_num_rhs` (default 64) precomputed U(0,1) RHS draws for b.
    benchspecs/sptrsv/spec.yaml's sptrsv-solve-kernel calls for "one fresh
    draw PER TIMED SOLVE CALL, never reused across repeated calls" so a
    synchronization-free/race-prone technique cannot get lucky on a single
    fixed b -- but harness.run_variant()'s prepare()-once/run()-repeated
    split has no per-call-argument hook, so the fixed pool is precomputed
    here (prepare() is exactly where "everything that can be hoisted out of
    run() must be" applies, per DOMAIN_GUIDE.md) and cycled by run() via a
    cursor -- the identical pattern kernelbench/domains/graph.py's ScipyBFS
    uses for its 64 fixed-seed BFS roots (see that class's docstring for the
    matching harness-level caveat this inherits too: run_variant() only
    correctness-gates the FIRST run() call, so draws #1..63 are exercised
    and timed but not independently re-validated -- a harness-wide
    limitation, not sptrsv-specific).
    """

    name = "scipy-csr-sptrsv"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        self.precision = precision
        self.dtype = np.float64 if precision == "fp64" else np.float32

    def prepare(self, matrix, params):
        L64 = unit_lower_triangular(matrix.csr)
        L = L64.astype(self.dtype)
        L.sort_indices()

        num_rhs = int(params.get("sptrsv_num_rhs", 64))
        seed = params.get("seed", 42)
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.0, 1.0, size=(num_rhs, L64.shape[0]))
        # b = L @ x_ref via one reference fp64 serial SpMV per draw -- scipy's
        # CSR matvec, computed once here (untimed preprocessing), never via
        # the technique under test.
        # (L64 @ X.T).T is a transposed VIEW (F-contiguous, not C-contiguous)
        # -- functionally fine for numpy/scipy consumers (this CPU impl,
        # correct strides respected throughout), but the sptrsv GPU artifact
        # adapters under bench/artifacts/sptrsv/ hand the equivalent array's
        # raw pointer to C/CUDA code expecting a flat contiguous row, where
        # skipping this copy is a real bug (see those adapters' STATUS.md).
        # Forced contiguous here too, for the same construction to stay
        # identical/safe across every sptrsv competitor in this codebase.
        B = np.ascontiguousarray((L64 @ X.T).T)  # (num_rhs, n); row i is L64 @ X[i]

        # disclosures for the cost model (_cost_sptrsv reads sptrsv_L_nnz
        # back) and for the result record's provenance -- same
        # stash-into-params pattern as ScipyGnnAggregation.prepare() above.
        params["sptrsv_L_nnz"] = int(L64.nnz)
        params["sptrsv_num_rhs"] = num_rhs
        params["sptrsv_rhs_seed"] = int(seed)
        params["sptrsv_triangular_factor"] = (
            "unit-lower-triangular-of-A (canonical structural proxy; "
            "benchspecs/sptrsv/spec.yaml triangular_factor_derivation)")

        return {"L": L, "X": X, "B": B.astype(self.dtype), "cursor": 0}

    def run(self, h):
        i = h["cursor"] % h["B"].shape[0]
        h["cursor"] += 1
        return spla.spsolve_triangular(
            h["L"], h["B"][i], lower=True, unit_diagonal=True, overwrite_b=False)

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


def reference_sptrsv(matrix, params):
    """
    Independent fp64 reference for sptrsv.

    Rebuilds the SAME canonical L (`unit_lower_triangular` -- workload
    derivation, sharing it is fine, see that function's docstring) and the
    SAME draw #0 of the fixed U(0,1) RHS pool ScipySpTRSV.prepare() built
    (same seed/count, read back from `params` since prepare() already ran
    and stashed them there before the harness calls this), then solves
    L x = b via `_forward_substitute` -- an independent pure-Python/NumPy
    forward substitution sharing no code with ScipySpTRSV.run()'s
    `scipy.sparse.linalg.spsolve_triangular` call.

    Only draw #0 is reconstructed: harness.run_variant() gates correctness
    against exactly one impl.run() call (the first, before warmup even
    starts), which consumes cursor=0 -- see ScipySpTRSV's docstring and
    graph.py's ScipyBFS for the identical harness-wide limitation.

    Since b = L @ x_ref by construction, `_forward_substitute(L, b)` should
    recover x_ref itself up to fp64 rounding -- so this reference also
    matches the spec's own literal correctness wording ("max relative error
    vs. THE x_ref used to generate b") while remaining an actual independent
    computation rather than just replaying the planted vector.

    scale = |L| @ |x_ref| -- the same (|operand| @ |operand|) backward-
    error-denominator convention reference_spmv/reference_spmm use above
    (scale = |A| @ |x| for their product); here the underlying "product" is
    b = L @ x_ref, so the analogous denominator is |L| @ |x_ref| -- the
    simpler of the two scale choices considered (vs. a literal residual-
    based componentwise backward error |L|@|x_hat|+|b|, which would need an
    extra kernel-dependent recompute inside the gate for no accuracy benefit
    at this tolerance).
    """
    A = matrix.csr
    L = unit_lower_triangular(A)
    num_rhs = int(params.get("sptrsv_num_rhs", 64))
    seed = params.get("sptrsv_rhs_seed", params.get("seed", 42))
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(num_rhs, L.shape[0]))
    x_ref0 = X[0]
    b0 = L @ x_ref0
    x_hat0 = _forward_substitute(L, b0)
    scale = abs(L) @ np.abs(x_ref0)
    return x_hat0, scale
