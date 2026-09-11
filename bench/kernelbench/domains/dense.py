"""
Dense linear algebra: gemm, gemv, cholesky, blas-level1-2
(+ batched-gemm, trsm, lu, qr, svd, eigensolver -- PLANNED, not yet here).

This module follows sparse.py's shape (WORKLOADS / COST / REFERENCE / IMPLS),
with one structural difference worth stating up front: sparse workloads are
*matrices loaded from disk*, shared verbatim across spmv/spmm/sddmm; dense
workloads here are *shapes* (M, N, K or n, precision, seed, ...). Nothing is
stored -- an implementation and its reference each regenerate byte-identical
operands from the same seeded RNG, the same discipline
`impls/cpu_ref.py`'s reference_spmm/reference_sddmm already use for their own
dense companions: generate at the RUN's own precision first, THEN widen to
fp64 for the reference computation. Generating directly at fp64 instead would
silently compare "GEMM of full-precision operands" against "GEMM of
precision-rounded operands", conflating operand-rounding error with the
kernel's own arithmetic error -- exactly the trap this project's harness is
built to avoid.

Four design decisions worth documenting precisely (each is non-obvious enough
that a future reader re-deriving the reasoning from scratch would likely get
it wrong):

1. CHOLESKY's correctness gate is NOT an elementwise comparison of the
   computed factor L against an independently computed reference factor.
   Cholesky's L is unique (up to floating point) for an SPD input, but
   comparing L elementwise still measures the wrong thing: two valid
   factorizations of the same A that differ only in rounding/blocking order
   can disagree substantially, pointwise, on individual small entries of L
   (classic cancellation) while both satisfying the standard backward-
   residual bound at essentially machine precision -- and conversely a
   pointwise-close L can still hide a real bug that only shows up in the
   product. So `reference_cholesky` does not return a reference L at all;
   see its docstring for exactly how it repurposes harness.py's generic
   "max_scaled_err" gate (diff = |got - ref|, val = max(diff / scale)) to
   compute the field-standard normalized residual
       ||L_computed @ L_computed.T - A||_F / (||A||_F * n * eps_fp64)
   instead, matching this spec's own stated test verbatim
   (`cholesky-dense-single-node-fp64-kernel`'s correctness field, "< 30").

2. blas-level1-2's own spec (`blas-l1-vector-kernel`) states its tolerance as
   "|result - reference| < flteps, ... flteps = 1e-4 for fp32, 1e-6 for
   fp64" -- but `spec.py`'s tolerance parser is deliberately conservative
   (DOMAIN_GUIDE non-negotiable #1: never hardcode a tolerance in a domain
   module, so a wrong auto-parse must fail closed, not silently accept a
   bogus number). It only accepts a number immediately after a comparison
   operator; here the operator is followed by the *name* `flteps`, whose
   value is given later via "=", not "<". Verified directly:
   `spec.load("blas-level1-2").variant("blas-l1-vector-kernel").tolerance`
   is `None` ("unparsed"). This is a real gap in the spec prose, not
   something this module papers over -- CORRECTNESS_MODE is still set to
   what the spec actually asks for (max_abs_err, fBLAS's own convention),
   and a run against that specific variant id is correctly reported
   uncertifiable until the spec text is revised to name the number
   explicitly. Of blas-level1-2's four variants, only
   `blas-l2-gemv-symv-kernel` carries a tolerance the parser can extract
   (1e-6, matching the vector-output max-relative-error convention used
   elsewhere in this suite); it is the variant used for this module's own
   verification runs of axpy/dot/nrm2, since it is the only place in this
   spec where a machine-checkable numeric bound currently exists. Our fp64
   result vs. fp64 reference error is ~1e-15 regardless of which variant's
   bound is used, so this is a safe substitution for a smoke check, not a
   loosening of the actual gate.

3. Every spec in this track states its tolerance as a table keyed by
   precision (1e-6/1e-3/1e-2 for fp64/fp32/fp16, verbatim in gemm, gemv and,
   implicitly, cholesky's residual bound). `spec.py`'s parser is a single
   regex pass over prose and returns exactly one number: the FIRST one
   following a comparison operator, which for every variant in this module
   happens to be the fp64 entry. Running fp32 by default against a
   fp64-scoped bound risks flagging a numerically-fine fp32 kernel as
   failing (a correct fp32 GEMM's relative error is itself typically
   ~1e-6ish, right at that boundary). DEFAULT_PRECISION is therefore fp64
   for every kernel here -- the one precision for which the parsed number is
   guaranteed to match the spec's own intent. fp32/fp16 remain selectable
   via `--precision`, with this caveat inherited.

4. DIM_KEY: unlike sparse.py's spmm/sddmm (where the *matrix* is fixed and
   the dense companion dimension N/K is a genuinely independent axis chosen
   at run time, decoupled from which matrix was loaded), every dense shape
   here is fully self-contained -- M, N, K (or n) all come from the
   Workload, never from params. Per this track's own instructions, DIM_KEY
   is still registered for "gemm" (-> "K", the contraction dimension) and
   "blas-level1-2" (-> "n", vector length), matching the runner's generic
   `--dims` sweep mechanism. Because our shapes are already fully
   determined, the runner's fallback dims=[128] (when no --dims/spec sweep
   is given) attaches an EXTRA, INERT `params["K"]`/`params["n"]` entry to
   the result record when one isn't explicitly requested; every cost rule
   and implementation in this module reads the actual size from the
   Workload only and ignores params[dim_key], so this never changes what is
   computed -- it is retained purely for audit/CLI-sweep compatibility, and
   readers of a raw result record should treat `matrix.{M,N,K,n}` as
   authoritative over any stray params[dim_key]. gemv and cholesky have no
   entry in DIM_KEY (matches spmv's own omission in sparse.py -- there is no
   free companion dimension to sweep).

Known, disclosed simplifications (protocol axes this module does not
implement, rather than silently mislabeling):
  * GEMM/GEMV's hot/cold/sram-resident `cache_state` axis is not
    implemented; a plain back-to-back Python timing loop with buffers
    reused across reps is closer to "hot" than "cold" and is reported as
    such via the smoke/nonconformance machinery, never mislabeled.
  * Cholesky's tile-size (NB) sweep is not implemented -- LAPACK's potrf
    (via scipy) blocks internally with no user-exposed NB, so there is
    nothing to sweep at this layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .. import workload
from ..harness import Timer

KERNELS = ["gemm", "gemv", "cholesky", "blas-level1-2"]
# owned by this domain but not yet implemented; kept honest for --list
PLANNED = ["batched-gemm", "trsm", "lu", "qr", "svd", "eigensolver"]

ITEMSIZE = {"fp64": 8, "fp32": 4, "fp16": 2, "tf32": 4}
# numpy/scipy have no native tf32 dtype; approximate it with fp32 for the
# CPU reference path (tf32 has no meaning without real tensor-core hardware).
_NP_DTYPE = {"fp64": np.float64, "fp32": np.float32, "fp16": np.float16,
             "tf32": np.float32}

# arbitrary, large, collision-free offset applied to a workload's seed to
# derive the SECOND operand's seed (B, x, or y), so e.g. a batched GEMM's A
# at batch index b never draws the same stream as B at some other batch
# index (max batch swept anywhere in this module is 4096, far below this).
_B_OFFSET = 1_000_003
# fixed AXPY scalar multiplier y = a*x + y; not spec-mandated (no surveyed
# blas-level1-2 paper states a value) -- a nonzero constant so the a=0/a=1
# degenerate cases aren't accidentally what gets measured.
_AXPY_ALPHA = 2.0


def _dtype(precision: str):
    return _NP_DTYPE.get(precision, np.float64)


# ------------------------------------------------------------------ workload
@dataclass
class DenseShape:
    """
    A dense workload is a SHAPE, not a file: enough (M, N, K/n, precision,
    seed) to let any implementation or reference regenerate byte-identical
    operands. `shape_class` records the taxonomy this track's specs use --
    "square", "tall-skinny" and "short-fat" (gemv distinguishes M>>K from
    K>>M explicitly), "small-irregular" (gemm's cubic/irregular/LLM-shaped
    sweep), "batched" (independent same-shape problems), and "vector"
    (blas-level1-2, which has no 2D shape at all).
    """

    name: str
    kernel: str = ""          # which KERNELS entry this shape belongs to
    variant: str = ""         # spec variant id this shape is drawn from
    shape_class: str = ""
    M: int = 0
    N: int = 0
    K: int = 0
    n: int = 0                # generic order/length: cholesky's N, blas-1's vector length
    batch: int = 1
    precision: str = "fp64"
    seed: int = 42
    op: str = ""              # blas-level1-2 dispatch tag: axpy | dot | nrm2
    layout: str = ""          # gemm/gemv operand layout disclosure, e.g. "NN"

    def describe(self) -> dict:
        d = {
            "name": self.name,
            "source": "synthetic-shape",   # dense workloads are shapes, never files
            "kernel": self.kernel,
            "spec_variant": self.variant,
            "shape_class": self.shape_class,
            "precision": self.precision,
            "seed": self.seed,
        }
        if self.M:
            d["M"] = self.M
        if self.N:
            d["N"] = self.N
        if self.K:
            d["K"] = self.K
        if self.n:
            d["n"] = self.n
        if self.batch != 1:
            d["batch"] = self.batch
        if self.op:
            d["op"] = self.op
        if self.layout:
            d["layout"] = self.layout
        return d


def _rng_operand(rows: int, cols: int, seed: int, dtype) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(dtype)


def _rng_vector(n: int, seed: int, dtype) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=n).astype(dtype)


def _spd_matrix(n: int, seed: int, dtype) -> np.ndarray:
    """SPD matrix A = M @ M.T + n*I from a random M (U(-1,1) entries), per
    this module's brief -- mathematically equivalent to the R^T*R construction
    the cholesky spec itself describes (M@M.T is SPD for any full-rank M;
    adding n*I keeps it strongly positive-definite at every swept size)."""
    M = _rng_operand(n, n, seed, dtype)
    return M @ M.T + n * np.eye(n, dtype=dtype)


# --------------------------------------------------------- gemm shape lists
# Verbatim from benchspecs/gemm/spec.yaml's recommended_subset prose (that
# field is free text, not a YAML list, so Variant.recommended_subset() can't
# extract it automatically -- these lists are this module's own transcription).

def _gemm_square_shapes():
    sizes = [128, 256, 512, 1024, 1536, 2048, 2049, 2560, 3072, 3584,
              4096, 4097, 4608, 5120, 5632, 6144]
    return [DenseShape(name=f"gemm-square-{s}", kernel="gemm",
                        variant="gemm-square-kernel", shape_class="square",
                        M=s, N=s, K=s, precision="fp64", seed=42, layout="NN")
            for s in sizes]


def _gemm_cubic_shapes():
    sizes = [8, 16, 24, 32, 64, 96, 120, 128]
    return [DenseShape(name=f"gemm-cubic-{s}", kernel="gemm",
                        variant="gemm-small-irregular-kernel",
                        shape_class="small-irregular",
                        M=s, N=s, K=s, precision="fp64", seed=42, layout="NN")
            for s in sizes]


def _gemm_irregular_shapes():
    mn = [64, 128, 256, 384, 490]
    ks = [256, 1024]
    return [DenseShape(name=f"gemm-irregular-{m}x{m}x{k}", kernel="gemm",
                        variant="gemm-small-irregular-kernel",
                        shape_class="small-irregular",
                        M=m, N=m, K=k, precision="fp64", seed=42, layout="NN")
            for k in ks for m in mn]


def _gemm_llm_shapes():
    # MPGEMM's DeepSeek/LLaMA linear-layer shapes, (M, N, K), verbatim.
    shapes = [(64, 2112, 7168), (64, 24576, 1536), (64, 32768, 512),
              (64, 7168, 16384), (64, 4096, 7168), (64, 7168, 2048)]
    return [DenseShape(name=f"gemm-llm-{m}x{n}x{k}", kernel="gemm",
                        variant="gemm-small-irregular-kernel",
                        shape_class="small-irregular",
                        M=m, N=n, K=k, precision="fp64", seed=42, layout="NN")
            for (m, n, k) in shapes]


def _gemm_batched_shapes():
    # CL-DB-GEMM's GEMMBATCHED sweep, reproduced verbatim: fp64 batched GEMM
    # has no evidence in this track's survey (spec's own open_questions), so
    # these use fp32, matching the spec's dense_operand for this variant.
    out = []
    for batch in (8, 64):
        for s in range(32, 641, 32):
            out.append(DenseShape(
                name=f"gemm-batched-b{batch}-{s}", kernel="gemm",
                variant="gemm-batched-kernel", shape_class="batched",
                M=s, N=s, K=s, batch=batch, precision="fp32", seed=42, layout="NN"))
    for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        out.append(DenseShape(
            name=f"gemm-batched-m128-b{batch}", kernel="gemm",
            variant="gemm-batched-kernel", shape_class="batched",
            M=128, N=128, K=128, batch=batch, precision="fp32", seed=42, layout="NN"))
    return out


# --------------------------------------------------------- gemv shape lists
def _gemv_shapes():
    # gemv-dense-kernel's recommended_subset, verbatim; GEMV is modeled
    # structurally as GEMM-with-N=1 throughout this module (N=1 is recorded
    # explicitly rather than left implicit).
    out = []
    for (m, k) in [(16384, 16), (65536, 16), (262144, 16), (1048576, 16)]:
        out.append(DenseShape(name=f"gemv-tallskinny-{m}x{k}", kernel="gemv",
                               variant="gemv-dense-kernel", shape_class="tall-skinny",
                               M=m, N=1, K=k, precision="fp64", seed=42))
    for (m, k) in [(128, 32768), (128, 131072)]:
        out.append(DenseShape(name=f"gemv-shortfat-{m}x{k}", kernel="gemv",
                               variant="gemv-dense-kernel", shape_class="short-fat",
                               M=m, N=1, K=k, precision="fp64", seed=42))
    for s in (1024, 2048, 4096, 8192, 16384):
        out.append(DenseShape(name=f"gemv-square-{s}", kernel="gemv",
                               variant="gemv-dense-kernel", shape_class="square",
                               M=s, N=1, K=s, precision="fp64", seed=42))
    return out


# ----------------------------------------------------- cholesky shape lists
def _cholesky_shapes():
    # cholesky-dense-single-node-fp64-kernel's recommended_subset, verbatim.
    ns = [2048, 4096, 8192, 16384, 22464]
    return [DenseShape(name=f"cholesky-n{n}", kernel="cholesky",
                        variant="cholesky-dense-single-node-fp64-kernel",
                        shape_class="square", M=n, K=n, n=n,
                        precision="fp64", seed=42)
            for n in ns]


# ----------------------------------------------------- blas-1 shape lists
def _blas1_shapes():
    # blas-l1-vector-kernel's recommended_subset vector-length sweep,
    # verbatim, crossed with the three ops this module dispatches (axpy,
    # dot, nrm2 -- scal and the level-2 gemv/symv pair from this same spec
    # are out of scope for "numpy-blas12" per this module's brief).
    ns = [2048, 8192, 65536, 1048576, 8388608, 67108864]
    out = []
    for op in ("axpy", "dot", "nrm2"):
        for n in ns:
            out.append(DenseShape(name=f"blas-{op}-n{n}", kernel="blas-level1-2",
                                   variant="blas-l1-vector-kernel",
                                   shape_class="vector", n=n, op=op,
                                   precision="fp64", seed=42))
    return out


def _build_registry() -> dict:
    shapes = (
        _gemm_square_shapes() + _gemm_cubic_shapes() + _gemm_irregular_shapes()
        + _gemm_llm_shapes() + _gemm_batched_shapes()
        + _gemv_shapes()
        + _cholesky_shapes()
        + _blas1_shapes()
    )
    reg: dict[str, DenseShape] = {}
    for s in shapes:
        assert s.name not in reg, f"duplicate dense workload name {s.name!r}"
        reg[s.name] = s
    return reg


_REGISTRY = _build_registry()


def load_workload(name: str) -> DenseShape:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise LookupError(
            f"{name!r} is not a known dense workload; {len(_REGISTRY)} "
            f"available, e.g. {sorted(_REGISTRY)[:6]} ...") from None


# --------------------------------------------------------------- smoke set
def _smoke_gemm():
    return [
        DenseShape(name="smoke-gemm-square-256", kernel="gemm", variant="smoke",
                   shape_class="square", M=256, N=256, K=256,
                   precision="fp64", seed=42, layout="NN"),
        DenseShape(name="smoke-gemm-irregular-384x256x512", kernel="gemm",
                   variant="smoke", shape_class="small-irregular",
                   M=384, N=256, K=512, precision="fp64", seed=42, layout="NN"),
        DenseShape(name="smoke-gemm-batched-b4-64", kernel="gemm", variant="smoke",
                   shape_class="batched", M=64, N=64, K=64, batch=4,
                   precision="fp32", seed=42, layout="NN"),
    ]


def _smoke_gemv():
    return [
        DenseShape(name="smoke-gemv-square-512", kernel="gemv", variant="smoke",
                   shape_class="square", M=512, N=1, K=512, precision="fp64", seed=42),
        DenseShape(name="smoke-gemv-tallskinny-4096x32", kernel="gemv", variant="smoke",
                   shape_class="tall-skinny", M=4096, N=1, K=32, precision="fp64", seed=42),
        DenseShape(name="smoke-gemv-shortfat-32x4096", kernel="gemv", variant="smoke",
                   shape_class="short-fat", M=32, N=1, K=4096, precision="fp64", seed=42),
    ]


def _smoke_cholesky():
    return [
        DenseShape(name="smoke-cholesky-n256", kernel="cholesky", variant="smoke",
                   shape_class="square", M=256, K=256, n=256, precision="fp64", seed=42),
        DenseShape(name="smoke-cholesky-n512", kernel="cholesky", variant="smoke",
                   shape_class="square", M=512, K=512, n=512, precision="fp64", seed=42),
    ]


def _smoke_blas12():
    return [
        DenseShape(name=f"smoke-blas-{op}-n4096", kernel="blas-level1-2", variant="smoke",
                   shape_class="vector", n=4096, op=op, precision="fp64", seed=42)
        for op in ("axpy", "dot", "nrm2")
    ]


_SMOKE_BY_KERNEL = {
    "gemm": _smoke_gemm, "gemv": _smoke_gemv,
    "cholesky": _smoke_cholesky, "blas-level1-2": _smoke_blas12,
}


def smoke_workloads(kernel: str | None = None):
    """
    Tiny, synthetic shapes that run in seconds. NOT spec-conforming (see
    DOMAIN_GUIDE.md). Unlike sparse.py's three kernels -- which all consume
    the SAME matrix type and so share one smoke list -- gemm/gemv/cholesky/
    blas-level1-2 have mutually incompatible shapes (a Cholesky needs a
    square SPD n x n input; a GEMM needs M,N,K; a BLAS-1 op needs a vector
    length + op tag), so this function is kernel-aware. `kernel` defaults to
    None only for callers that don't know their own kernel context; the
    runner always passes it explicitly (see the small `runner.py`
    compatibility shim next to `domain.smoke_workloads` -- added because
    sparse.py's original zero-argument contract can't express this and this
    is the first domain module for which it matters).
    """
    fn = _SMOKE_BY_KERNEL.get(kernel, _smoke_gemm)
    return fn()


# ----------------------------------------------------------------- cost rules
def _cost_gemm(w: DenseShape, params: dict):
    vb = ITEMSIZE[params.get("precision", w.precision)]
    flops = 2 * w.M * w.N * w.K * w.batch
    # compulsory-traffic lower bound: read A, read B, write C, each once.
    byts = w.batch * (w.M * w.K + w.K * w.N + w.M * w.N) * vb
    return int(flops), int(byts)


def _cost_gemv(w: DenseShape, params: dict):
    vb = ITEMSIZE[params.get("precision", w.precision)]
    # matches gemv spec's own secondary GFLOP/s formula (2*M*K); GEMV is
    # bandwidth-bound (see module docstring), so `byts` below -- reported by
    # the harness as gbytes_per_s_lower_bound -- is the metric that actually
    # matters, per gemv-dense-kernel's own primary=GB/s metric.
    flops = 2 * w.M * w.K
    byts = w.M * w.K * vb + w.K * vb + w.M * vb   # spec's own GB/s formula
    return int(flops), int(byts)


def _cost_cholesky(w: DenseShape, params: dict):
    vb = ITEMSIZE[params.get("precision", w.precision)]
    n = w.n
    flops = (n ** 3) // 3   # standard dense-Cholesky FLOP count, per spec
    # compulsory-traffic lower bound: read A's lower triangle once, write
    # L's lower triangle once. A real blocked factorization moves far more
    # (O(n^3) working-set reuse); this is explicitly a LOWER bound, per
    # workload.register_cost's own contract.
    tri = n * (n + 1) // 2
    byts = 2 * tri * vb
    return int(flops), int(byts)


_BLAS1_OPS = ("axpy", "dot", "nrm2")
_BLAS1_FLOPS_PER_ELEM = {"axpy": 2, "dot": 2, "nrm2": 2}   # spec's own GOps/s formula
_BLAS1_BYTES_PER_ELEM = {"axpy": 3, "dot": 2, "nrm2": 1}   # spec's own GB/s formula


def _resolve_op(w: DenseShape, params: dict) -> str:
    """Primarily params["op"] (per this module's brief), falling back to the
    workload's own `.op` tag -- the path actually exercised through
    runner.py today, since nothing on that CLI currently sets params["op"]
    (see module docstring point 4 for the analogous DIM_KEY discussion)."""
    op = params.get("op") or getattr(w, "op", "")
    if op not in _BLAS1_OPS:
        raise ValueError(f"blas-level1-2: unknown/unset op {op!r}; have {_BLAS1_OPS}")
    return op


def _cost_blas12(w: DenseShape, params: dict):
    vb = ITEMSIZE[params.get("precision", w.precision)]
    op = _resolve_op(w, params)
    n = w.n
    flops = _BLAS1_FLOPS_PER_ELEM[op] * n
    byts = _BLAS1_BYTES_PER_ELEM[op] * n * vb
    return int(flops), int(byts)


workload.register_cost("gemm", _cost_gemm, "GFLOP/s")
# Specs mandate GB/s as the PRIMARY metric for these bandwidth-bound kernels
# (audit finding: registering GFLOP/s put the wrong number in the primary slot
# and made report.py's leaderboard rank by the wrong field).
def _cost_gemv_bw(w, params):
    fl, by = _cost_gemv(w, params)
    return by, by      # primary throughput = bytes/s; byte bound unchanged


workload.register_cost("gemv", _cost_gemv_bw, "GB/s")
workload.register_cost("cholesky", _cost_cholesky, "GFLOP/s")
def _cost_blas12_bw(w, params):
    fl, by = _cost_blas12(w, params)
    return by, by


workload.register_cost("blas-level1-2", _cost_blas12_bw, "GB/s")


def quantize_dequantize_groupwise(A: np.ndarray, bits: int, group_size: int):
    """
    Symmetric, round-to-nearest, per-(row,group) quantize-then-dequantize of
    a dense fp64 operand. Groups partition the COLUMN (K) axis; group_size
    must evenly divide A.shape[1].

    PUBLIC (no leading underscore) and imported directly by
    bench/artifacts/gemv/marlin/adapter.py -- unlike this module's other
    private RNG helpers (_rng_operand etc., which every artifact adapter
    replicates verbatim per this project's own cross-module convention),
    this one is genuinely NEW, purpose-built shared infrastructure: it only
    exists so a quantized-weight GEMV impl and reference_gemv's own
    quantized branch below can derive BIT-IDENTICAL codes/scale from the
    same (A, bits, group_size) inputs, which requires calling the exact
    same function, not two independently-written ones that could drift.

    Returns (codes: int32 (rows, n_groups, group_size) in [-qmax, qmax],
    scale: fp64 (rows, n_groups), dequantized: fp64 (rows, K) with
    dequantized == codes*scale broadcast per group). A caller that only
    needs the reference value uses `dequantized`; a caller that needs to
    feed an artifact's own packing API (e.g. marlin's `Layer.pack`, which
    itself re-derives `round(dequantized_weight / scale)`) uses `codes`/
    `scale` directly -- see gemv/marlin/STATUS.md for why this dual-use
    matters (marlin's pack() takes a DEQUANTIZED fp16 weight + scale, not
    raw codes, and internally requantizes; feeding it `dequantized` here
    is therefore how "shared quantized codes" reach marlin's own API).
    """
    rows, K = A.shape
    if K % group_size != 0:
        raise ValueError(f"group_size={group_size} does not divide K={K}")
    n_groups = K // group_size
    qmax = (1 << (bits - 1)) - 1
    Ag = A.reshape(rows, n_groups, group_size)
    amax = np.abs(Ag).max(axis=2)                       # (rows, n_groups)
    scale = np.where(amax == 0, 1.0, amax / qmax)
    codes = np.clip(np.round(Ag / scale[:, :, None]), -qmax, qmax)
    dequant = (codes * scale[:, :, None]).reshape(rows, K)
    return codes.astype(np.int32), scale, dequant


# ----------------------------------------------------------------- references
def reference_gemm(w: DenseShape, params: dict):
    """fp64 ground truth; scale = |A|@|B|, the componentwise backward-error
    denominator max_scaled_err needs (see harness.check_correctness).
    Batched (w.batch > 1) stacks per-batch-element operands; numpy's `@`
    on 3D arrays performs batched matmul directly, matching NumpyGemm."""
    seed = params.get("seed", w.seed)
    dtype = _dtype(params.get("precision", w.precision))
    if w.batch > 1:
        A = np.stack([_rng_operand(w.M, w.K, seed + b, dtype)
                      for b in range(w.batch)]).astype(np.float64)
        B = np.stack([_rng_operand(w.K, w.N, seed + _B_OFFSET + b, dtype)
                      for b in range(w.batch)]).astype(np.float64)
    else:
        A = _rng_operand(w.M, w.K, seed, dtype).astype(np.float64)
        B = _rng_operand(w.K, w.N, seed + _B_OFFSET, dtype).astype(np.float64)
    C = A @ B
    scale = np.abs(A) @ np.abs(B)
    return C, scale


def reference_gemv(w: DenseShape, params: dict):
    """
    fp64 ground truth; scale = |A|@|x|.

    GENUINELY-REQUIRED SMALL FIX (ARTIFACT_GUIDE.md's "do not change the
    domain module unless required, record it" -- recorded here and in
    bench/artifacts/gemv/marlin/STATUS.md): opt-in quantized-weight
    support via two extra, OPTIONAL params keys, "quant_bits" and
    "quant_group_size". When "quant_bits" is absent (every EXISTING
    caller of gemv-dense-kernel), this function is BIT-FOR-BIT unchanged
    from before this fix -- A is the plain continuous dense operand.

    When a quantized-weight GEMV impl (e.g. marlin, whose kernel
    arithmetic is fundamentally int4-weight x fp16-activation, never
    full-precision GEMV) sets params["quant_bits"] during its own
    prepare() -- called BEFORE this reference, see harness.run_variant,
    so the mutation is visible here -- A is symmetric per-(row,group)
    quantized then DEQUANTIZED (via quantize_dequantize_groupwise above)
    before being used as ground truth. This compares the impl against the
    SAME dequantized values it is required to consume, isolating the
    kernel's own fp16-accumulation error from quantization error -- this
    project's shared-quantized-value fairness rule (see ARTIFACT_GUIDE.md
    and domains/ml.py's QuantGemmWorkload docstring for the identical rule
    applied to the quantized-gemm track). This is NOT a looser gate: an
    impl that computed the wrong output from the SAME dequantized weights
    still fails. Without this fix, gating a quantized-weight kernel
    against the plain continuous-A reference below conflates quantization
    noise with kernel error -- verified empirically before writing this
    fix: INT4-quantizing this domain's own U(-1,1) gemv operands and
    comparing against the continuous-A reference produces 0.15%-6.6%
    max_scaled_err from quantization noise ALONE (shape-dependent), which
    would swamp gemv-quantized-weight-kernel's 1e-3 parsed tolerance
    regardless of kernel correctness -- see marlin's STATUS.md for the
    numeric check.

    SECOND GENUINELY-REQUIRED SMALL FIX (same rule, recorded here and in
    bench/artifacts/gemv/packkv/STATUS.md): a THIRD optional params key,
    "dequantized_A_override" -- a plain (M,K) fp64 numpy array a caller may
    supply to be used VERBATIM as the ground-truth A, bypassing both the
    RNG generation above AND the quant_bits/quant_group_size path. This
    exists for artifacts whose own quantization scheme is not expressible
    via `quantize_dequantize_groupwise` (that helper is SYMMETRIC,
    zero-point-free, and groups along COLUMNS; PackKV's own K-cache
    quantizer is an ASYMMETRIC affine scheme with a zero-point, grouped
    along ROWS -- a different quantization FAMILY, not just a different
    axis, so reusing the helper would silently misrepresent what PackKV's
    kernel actually consumes). The override array must be computed by the
    CALLER, independently of this function and of any code this function
    calls (this function performs no quantization of its own on the
    override path -- it is handed finished VALUES, not a formula to run),
    preserving the reference-independence rule (DOMAIN_GUIDE.md "Reference
    independence" ruling: a reference may never call the same function/
    helper as any implementation it gates -- here there is no shared
    function at all, only shared DATA the caller already computed to feed
    its own kernel). Absent (every existing caller, including every
    quant_bits caller above): behavior is bit-for-bit unchanged. `x` is
    ALWAYS generated by this function's own seeded RNG below, override or
    not -- only `A` is ever substituted.

    The override is POPPED from `params` once consumed (below), rather than
    left in place like `quant_bits`/`quant_group_size` above: those are
    small, JSON-serializable metadata worth keeping in the result record
    (`harness.RunResult.to_dict` serializes `params` verbatim); a full
    (M,K) array is neither small nor JSON-serializable (`runner.py`'s
    `json.dump` would otherwise raise `TypeError: Object of type ndarray is
    not JSON serializable` on every run that sets it) and was never meant
    to be a permanent record field, only one-shot plumbing from an
    adapter's `prepare()` to this function. `reference_gemv` runs exactly
    once per `run_variant` call (the correctness gate, before the
    warmup/timed-rep loop -- see `harness.run_variant`), so popping here
    cannot affect anything later in the same run.
    """
    seed = params.get("seed", w.seed)
    dtype = _dtype(params.get("precision", w.precision))
    A_override = params.pop("dequantized_A_override", None)
    if A_override is not None:
        A = np.asarray(A_override, dtype=np.float64)
        if A.shape != (w.M, w.K):
            raise ValueError(
                f"reference_gemv: dequantized_A_override shape {A.shape} "
                f"!= workload shape {(w.M, w.K)}")
    else:
        A = _rng_operand(w.M, w.K, seed, dtype).astype(np.float64)
        bits = params.get("quant_bits")
        if bits:
            group_size = params.get("quant_group_size", w.K)
            _, _, A = quantize_dequantize_groupwise(A, bits, group_size)
    x = _rng_vector(w.K, seed + _B_OFFSET, dtype).astype(np.float64)
    y = A @ x
    scale = np.abs(A) @ np.abs(x)
    return y, scale


def reference_cholesky(w: DenseShape, params: dict):
    """
    Does NOT return an independently-computed reference factor -- see the
    module docstring's point 1 for why an elementwise L-vs-L_ref comparison
    is the wrong gate for Cholesky. Instead this pairs with
    ScipyCholesky.to_host (which returns a ONE-ELEMENT array holding the
    computed backward residual ||L_computed @ L_computed.T - A||_F, not the
    raw factor) to turn harness.py's generic max_scaled_err gate

        val = max(|got - ref| / scale)

    into exactly the LAPACK/ScaLAPACK-testsuite normalized residual test
    this spec's correctness field states ("< 30"):

        got   = ||L_computed @ L_computed.T - A||_F        (1 element)
        ref   = 0                                            (a perfect
                factorization reproduces A exactly)
        scale = ||A||_F * n * eps_fp64                       (the spec's own
                normalization)
        =>  val = ||L L^T - A||_F / (||A||_F * n * eps_fp64)

    A matches ScipyCholesky.prepare's SPD construction exactly (same seed,
    same precision-then-widen recipe) so `got`'s A and `ref`'s A are the
    same matrix.

    Optional hook -- params["external_A_fro_norm"] (added for the hicma-x
    paper-artifact integration, ARTIFACT_GUIDE.md's TLR/mixed-precision gate
    note): some artifacts (e.g. hicma-x's testing_potrf_tlr) generate their
    OWN SPD operand internally (a STARS-H tile low-rank covariance matrix)
    with no way to substitute this module's `_spd_matrix` -- there is no
    library entry point that accepts an externally supplied dense operand.
    The normalized-residual FORMULA is still exactly this spec's own
    (||L L^T - A||_F / (||A||_F * n * eps_fp64) < 30); only the operand A
    differs. When an implementation's prepare()/run() sets this key to the
    Frobenius norm of ITS OWN A (read back from the artifact's own printed
    ||A||_F, never invented), this function uses that value for `scale`
    instead of computing ||_spd_matrix(...)||_F -- keeping the identical
    formula and threshold, just with the honestly-disclosed operand
    substitution recorded by the calling adapter. Every other caller (every
    impl in CPU_IMPLS/cuda_impls here) never sets this key, so this is a
    no-op for them -- default behavior is completely unchanged.
    """
    external_norm = params.get("external_A_fro_norm")
    if external_norm is not None:
        denom = float(external_norm) * w.n * np.finfo(np.float64).eps
        return np.zeros(1), np.array([denom])
    seed = params.get("seed", w.seed)
    dtype = _dtype(params.get("precision", w.precision))
    A = _spd_matrix(w.n, seed, dtype).astype(np.float64)
    denom = np.linalg.norm(A, "fro") * w.n * np.finfo(np.float64).eps
    return np.zeros(1), np.array([denom])


def reference_blas12(w: DenseShape, params: dict):
    """
    fp64 ground truth for axpy/dot/nrm2, generated with the same seeded-RNG
    recipe (target precision first, widened to fp64 after) as
    NumpyBlas12.prepare. CORRECTNESS_MODE["blas-level1-2"] is "max_abs_err"
    (fBLAS's own hardcoded absolute tolerance -- see module docstring point
    2), so `scale` is unused here and always None.
    """
    op = _resolve_op(w, params)
    seed = params.get("seed", w.seed)
    dtype = _dtype(params.get("precision", w.precision))
    x = _rng_vector(w.n, seed, dtype).astype(np.float64)
    if op == "dot":
        y = _rng_vector(w.n, seed + _B_OFFSET, dtype).astype(np.float64)
        return np.array([np.dot(x, y)]), None
    if op == "axpy":
        y = _rng_vector(w.n, seed + _B_OFFSET, dtype).astype(np.float64)
        return _AXPY_ALPHA * x + y, None
    # nrm2
    return np.array([np.linalg.norm(x)]), None


REFERENCES = {
    "gemm": reference_gemm,
    "gemv": reference_gemv,
    "cholesky": reference_cholesky,
    "blas-level1-2": reference_blas12,
}

CORRECTNESS_MODE = {
    "gemm": "max_scaled_err",
    "gemv": "max_scaled_err",
    "cholesky": "max_scaled_err",     # engineered residual test -- see reference_cholesky
    "blas-level1-2": "max_abs_err",   # fBLAS's own absolute-tolerance convention
}

# fp64 everywhere -- see module docstring point 3 for why.
DEFAULT_PRECISION = {"gemm": "fp64", "gemv": "fp64", "cholesky": "fp64",
                     "blas-level1-2": "fp64"}


# --------------------------------------------------------------------- impls
class NumpyGemm:
    """C = A @ B via numpy (vendor BLAS underneath). Operand generation is
    hoisted into prepare(); run() is exactly the timed matmul call. Handles
    both plain (2D) and batched (3D, batch-leading) shapes uniformly, since
    numpy's `@` performs batched matmul directly on stacked 3D arrays."""

    name = "numpy-gemm"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        self.precision = precision
        self.dtype = _dtype(precision)

    def prepare(self, workload, params):
        seed = params.get("seed", workload.seed)
        w = workload
        if w.batch > 1:
            A = np.stack([_rng_operand(w.M, w.K, seed + b, self.dtype)
                          for b in range(w.batch)])
            B = np.stack([_rng_operand(w.K, w.N, seed + _B_OFFSET + b, self.dtype)
                          for b in range(w.batch)])
        else:
            A = _rng_operand(w.M, w.K, seed, self.dtype)
            B = _rng_operand(w.K, w.N, seed + _B_OFFSET, self.dtype)
        return {"A": A, "B": B}

    def run(self, h):
        return h["A"] @ h["B"]

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


class NumpyGemv:
    """y = A @ x via numpy. GEMV modeled as GEMM-with-N=1 (see module
    docstring); operand generation hoisted into prepare()."""

    name = "numpy-gemv"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        self.precision = precision
        self.dtype = _dtype(precision)

    def prepare(self, workload, params):
        seed = params.get("seed", workload.seed)
        A = _rng_operand(workload.M, workload.K, seed, self.dtype)
        x = _rng_vector(workload.K, seed + _B_OFFSET, self.dtype)
        return {"A": A, "x": x}

    def run(self, h):
        return h["A"] @ h["x"]

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


class ScipyCholesky:
    """L = cholesky(A, lower=True) via scipy (LAPACK potrf underneath). SPD
    matrix generation is hoisted into prepare(); run() is exactly the timed
    factorization call. scipy.linalg.cholesky allocates its output fresh
    each call (no in-place/out= variant is exposed), so that one allocation
    cannot be hoisted any further than shown."""

    name = "scipy-cholesky"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        self.precision = precision
        self.dtype = _dtype(precision)

    def prepare(self, workload, params):
        seed = params.get("seed", workload.seed)
        A = _spd_matrix(workload.n, seed, self.dtype)
        self._A = A   # kept for the residual computed in to_host(), see below
        return {"A": A}

    def run(self, h):
        return sla.cholesky(h["A"], lower=True)

    def to_host(self, out):
        """Returns a ONE-ELEMENT array holding the backward residual
        ||L @ L.T - A||_F, NOT the raw factor -- see reference_cholesky's
        docstring for exactly how this pairs with it to reproduce the
        LAPACK-testsuite normalized-residual test."""
        L = np.asarray(out, dtype=np.float64)
        resid = np.linalg.norm(L @ L.T - self._A, "fro")
        return np.array([resid])

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


class NumpyBlas12:
    """
    Dispatch across the three level-1 vector ops this module implements
    (axpy, dot, nrm2), selected via params["op"] with a fallback to the
    workload's own `.op` tag (see _resolve_op). scal and the level-2
    gemv/symv pair this same spec also defines are out of scope for
    "numpy-blas12" per this module's brief -- use "gemv" (a different
    KERNELS entry) for matrix-vector coverage.
    """

    name = "numpy-blas12"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        self.precision = precision
        self.dtype = _dtype(precision)

    def prepare(self, workload, params):
        op = _resolve_op(workload, params)
        seed = params.get("seed", workload.seed)
        n = workload.n
        h = {"op": op, "x": _rng_vector(n, seed, self.dtype)}
        if op in ("axpy", "dot"):
            h["y"] = _rng_vector(n, seed + _B_OFFSET, self.dtype)
        if op == "axpy":
            h["a"] = self.dtype(_AXPY_ALPHA)
        return h

    def run(self, h):
        op = h["op"]
        if op == "dot":
            return np.dot(h["x"], h["y"])
        if op == "axpy":
            return h["a"] * h["x"] + h["y"]
        if op == "nrm2":
            return np.linalg.norm(h["x"])
        raise ValueError(f"unknown op {op!r}")

    def to_host(self, out):
        return np.atleast_1d(np.asarray(out, dtype=np.float64))

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


CPU_IMPLS = {
    "gemm": {"numpy-gemm": NumpyGemm},
    "gemv": {"numpy-gemv": NumpyGemv},
    "cholesky": {"scipy-cholesky": ScipyCholesky},
    "blas-level1-2": {"numpy-blas12": NumpyBlas12},
}


def cuda_impls():
    """Torch-based GPU baselines, wired but never run from here -- the login
    node's GPU is shared (see csrc/ conventions and DOMAIN_GUIDE.md). Timed
    with CudaEventTimer, one event pair per iteration, same as sparse.py's
    CUDA impls."""
    from ..impls import gpu_cuda as g
    return {
        "gemm": {"torch-matmul": g.TorchGemm},
        "gemv": {"torch-mv": g.TorchGemv},
        "cholesky": {"torch-cholesky": g.TorchCholesky},
    }


# which params key carries the swept dimension (None => no sweep). See
# module docstring point 4: for this domain the value is inert/audit-only,
# never read by any cost rule or impl above.
DIM_KEY = {"gemm": "K", "blas-level1-2": "n"}
