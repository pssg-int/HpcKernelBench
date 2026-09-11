"""
Tensor algebra: MTTKRP (the CP-ALS/HOOI workhorse) and general tensor
contraction / einsum evaluation.

Follows the shape of sparse.py (the reference domain module) with one
structural difference worth flagging up front: mttkrp and tensor-contraction
do NOT share a workload representation (a sparse N-mode COO tensor vs. a pair
of dense operands for a binary einsum step), unlike every other multi-kernel
domain in this codebase (sparse.py's 3 kernels all consume a Matrix, graph.py's
4 consume a Graph, ...). `smoke_workloads(kernel=...)` is therefore
kernel-parameterized -- see its docstring and the corresponding change in
runner.py (inspect.signature dispatch, backward compatible with every other
domain's zero-arg smoke_workloads()).

  WORKLOADS   — SparseTensorWorkload (COO, synthetic-by-default; reads a
                real FROSTT .tns file when present, never auto-downloads one)
                and ContractionWorkload (a pair of dense tensors plus the
                einsum equation relating them to the output)
  COST        — mttkrp: op-literal per-nonzero-per-rank flop count, disclosed
                against spec.yaml's own differing convention (see _cost_mttkrp)
                tensor-contraction: 2*prod(free)*prod(contracted), the spec's
                own literal formula
  REFERENCE   — fp64 evaluations returning (result, scale), scale = the same
                computation run on |inputs| (the cancellation-robust
                denominator, as in sparse.py's |A|*|B|)
  IMPLS       — numpy-mttkrp (vectorized COO with np.add.at over the mode
                index); numpy-einsum-contraction and numpy-tensordot-
                contraction, whose contrast is the point: einsum fuses
                permutation into one opaque call, tensordot exposes it as
                separate, individually timeable/skippable transpose steps
                (see NumpyTensordotContraction's docstring and the
                `transpose_in_timing` param)
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .. import workload
from ..harness import Timer

KERNELS = ["mttkrp", "tensor-contraction"]
# declared but not yet implemented here; listed so `--list` can say so honestly
PLANNED = ["sparse-tensor-contraction", "tucker", "tensor-network"]

ITEMSIZE = {"fp64": 8, "fp32": 4, "fp16": 2, "tf32": 4}

TENSOR_CACHE = os.environ.get(
    "KERNELBENCH_TENSOR_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tensors"),
)
TENSOR_CACHE = os.path.normpath(TENSOR_CACHE)

SEED_DEFAULT = 20260806
DEFAULT_RANK = 32   # middle of the spec's R in {16, 32, 64} sweep; BLCO's own default


# ==================================================================== mttkrp
@dataclass
class SparseTensorWorkload:
    """
    An N-mode sparse tensor in COO form: `indices` is (nnz, N) int64 (one row
    per nonzero, one column per mode), `values` is (nnz,) float64. Duplicate
    coordinate rows are legal (accumulate, same as a real tensor's coincident
    entries) -- the MTTKRP kernel below handles them correctly via np.add.at,
    so the synthetic generator does not bother deduplicating.
    """

    name: str
    shape: tuple[int, ...]
    indices: np.ndarray
    values: np.ndarray
    source: str = "synthetic"        # "synthetic" | "frostt"
    seed: int = SEED_DEFAULT
    load_seconds: float = 0.0

    @property
    def order(self) -> int:
        return len(self.shape)

    @property
    def nnz(self) -> int:
        return int(self.values.shape[0])

    def describe(self) -> dict:
        total = 1
        for s in self.shape:
            total *= s
        return {
            "name": self.name,
            "source": self.source,
            "order": self.order,
            "shape": list(self.shape),
            "nnz": self.nnz,
            "density": (self.nnz / total) if total else 0.0,
            "value_checksum": hashlib.sha256(
                np.ascontiguousarray(self.values).tobytes()).hexdigest()[:16],
            "seed": self.seed,
            "load_seconds": round(self.load_seconds, 4),
        }


def synthetic_tensor(name: str, shape: tuple[int, ...], nnz: int,
                     seed: int = SEED_DEFAULT) -> SparseTensorWorkload:
    """
    Fixed-seed synthetic sparse COO tensor -- NOT spec-conforming (the spec's
    suite is the real FROSTT corpus), used for the login-node smoke test and
    as a no-download stand-in wherever a real tensor is unavailable. Labelled
    source="synthetic" so a result record can never be mistaken for a
    spec-conforming FROSTT run.
    """
    rng = np.random.default_rng(seed)
    indices = np.stack([rng.integers(0, s, size=nnz) for s in shape], axis=1).astype(np.int64)
    values = rng.uniform(-1.0, 1.0, size=nnz)
    return SparseTensorWorkload(name=name, shape=tuple(int(s) for s in shape),
                                indices=indices, values=values,
                                source="synthetic", seed=seed)


# order-3 (the mode-0 example spec.yaml/this module's docstrings work through)
# plus one order-4 case to exercise the general-N path the real FROSTT tensors
# (nips, uber, chicago-crime, ... are all order 3-4) actually need.
SMOKE_MTTKRP = [
    ("smoke-3d-small", dict(shape=(60, 50, 40), nnz=2000)),
    ("smoke-3d-tiny", dict(shape=(30, 25, 20), nnz=400)),
    ("smoke-4d-small", dict(shape=(20, 18, 16, 14), nnz=1200)),
]

# spec.yaml's mttkrp-general-kernel-fp64 recommended_subset, verbatim shape/nnz
# for the error message below (informational only -- this module never fetches
# these). Real download: http://frostt.io/tensors/<name>/ (10s of GB for the
# largest, e.g. nell-1's 143.6M nonzeros).
FROSTT_SHAPES = {
    "nips": ((2500, 2900, 14000, 17), 3_100_000),
    "uber": ((183, 24, 1140, 1717), 3_300_000),
    "chicago-crime": ((6200, 24, 77, 32), 5_300_000),
    "vast-2015-mc1": ((165_400, 11_400, 2), 26_000_000),
    "darpa": ((22_500, 22_500, 23_800_000), 28_400_000),
    "enron": ((6000, 5700, 244_300, 1200), 54_200_000),
    "nell-2": ((12_100, 9200, 28_800), 76_900_000),
    "flickr-4d": ((319_700, 28_200_000, 1_600_000, 731), 112_900_000),
    "delicious-4d": ((532_900, 17_300_000, 2_500_000, 1400), 140_100_000),
    "nell-1": ((2_900_000, 2_100_000, 25_500_000), 143_600_000),
}

# mttkrp-symmetric-kernel-fp64's synthetic Erdos-Renyi generator (SySTeC
# protocol) and SymProp's 5 real hypergraph-derived datasets need a different
# generator/loader than the general (asymmetric) path below; neither is
# implemented here (see load_workload's NotImplementedError for these names).
_SYMMETRIC_HYPERGRAPH_NAMES = {
    "contact-school", "trivago-clicks", "walmart-trips",
    "stackoverflow", "amazon-reviews",
}


def _frostt_path(slug: str) -> str:
    return os.path.join(TENSOR_CACHE, f"{slug}.tns")


def _load_frostt_tns(path: str) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """
    Parse a FROSTT .tns file: one nonzero per line, `N` whitespace-separated
    1-indexed mode coordinates followed by the value. No header row (per
    FROSTT's own format description) -- shape is inferred as (max index per
    mode + 1), which is exactly right for a full-rank real tensor and is the
    same convention every FROSTT-consuming tool (SPLATT, BLCO, ...) uses.
    """
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    order = raw.shape[1] - 1
    indices = raw[:, :order].astype(np.int64) - 1
    values = raw[:, order].astype(np.float64)
    shape = tuple(int(indices[:, m].max()) + 1 for m in range(order))
    return indices, values, shape


def _factor_matrix(rows: int, R: int, seed: int, dtype) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, R)).astype(dtype)


def _make_factors(shape: tuple[int, ...], R: int, seed: int, dtype) -> list[np.ndarray]:
    """One factor matrix per mode, all generated up front (MTTKRP for mode n
    uses all EXCEPT factors[n]); seeded per-mode (seed+m) so changing R or the
    tensor shape never perturbs an unrelated mode's factor values."""
    return [_factor_matrix(shape[m], R, seed + m, dtype) for m in range(len(shape))]


def _mttkrp_coo(shape: tuple[int, ...], indices: np.ndarray, values: np.ndarray,
                factors: list[np.ndarray], mode: int, R: int) -> np.ndarray:
    """
    M[i_mode, :] = sum over nnz sharing that i_mode of
                   X_val * prod_{m != mode} factor[m][i_m, :]

    Vectorized over nnz: build the per-nonzero Khatri-Rao row product by
    elementwise-multiplying the gathered factor rows together (numpy fancy
    indexing, no Python loop over nnz), then scatter-accumulate into the
    output via np.add.at, which -- unlike `out[idx] += kr` -- correctly
    accumulates repeated indices.
    """
    order = len(shape)
    kr = np.empty((values.shape[0], R), dtype=values.dtype)
    kr[:] = values[:, None]
    for m in range(order):
        if m == mode:
            continue
        kr *= factors[m][indices[:, m], :]
    out = np.zeros((shape[mode], R), dtype=values.dtype)
    np.add.at(out, indices[:, mode], kr)
    return out


def _cost_mttkrp(w: SparseTensorWorkload, params: dict) -> tuple[int, int]:
    """
    Per-mode flop count (this domain module's chosen convention, disclosed):
    for each nonzero, combining the tensor value with the (order-1) other
    modes' factor rows into the Khatri-Rao row takes (order-1) elementwise
    multiplies of R-vectors, then 1 more R-vector add to scatter-accumulate
    into the output -- `order` R-wide ops per nonzero, i.e. order*nnz*R total.
    For order=3 (the mode-0 case this track's specs and this module work
    through) that is the literal "2 multiplies + 1 add per element per rank"
    = 3*nnz*R.

    spec.yaml's own metric.primary field states a DIFFERENT convention --
    "2*R*nnz(X)" per mode -- and its open_questions section admits neither
    was checked against any surveyed paper's internal flop counter ("whether
    the final scale-and-accumulate per nonzero is counted as 2R or R flops").
    This module uses the op-literal order*nnz*R count rather than silently
    picking a number that disagrees with what the reference implementation
    actually computes.
    """
    R = int(params.get("R", DEFAULT_RANK))
    params["R"] = R                                    # echo the resolved value
    mode = int(params.get("mode", 0)) % w.order
    params["mode"] = mode                               # echo the wrapped value
    itemsize = ITEMSIZE[params.get("precision", "fp64")]
    flops = w.order * w.nnz * R
    idx_bytes = w.nnz * w.order * 4
    val_bytes = w.nnz * itemsize
    factor_bytes = sum(w.shape[m] * R * itemsize for m in range(w.order) if m != mode)
    out_bytes = w.shape[mode] * R * itemsize
    return int(flops), int(idx_bytes + val_bytes + factor_bytes + out_bytes)


workload.register_cost("mttkrp", _cost_mttkrp, "GFLOP/s")


def _mttkrp_grouped_reference(shape, indices, values, factors, mode, R):
    """
    INDEPENDENT reference formulation: sort nonzeros by their output-row index
    and reduce each contiguous group with an explicit per-row einsum. Different
    traversal order, different gather pattern, different accumulation primitive
    (np.einsum per group vs one global np.add.at scatter) from _mttkrp_coo.

    This exists because the deep audit proved the previous reference — which
    called _mttkrp_coo itself — was vacuous: a wrong-scatter-axis mutation left
    the gate at err=0.0. Sharing zero compute code with the impl is the point.
    """
    order = len(shape)
    perm = np.argsort(indices[:, mode], kind="stable")
    sidx = indices[perm]
    svals = values[perm]
    rows = sidx[:, mode]
    out = np.zeros((shape[mode], R), dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(rows)) + 1
    for lo, hi in zip(np.concatenate(([0], boundaries)),
                      np.concatenate((boundaries, [len(rows)]))):
        block = np.full((hi - lo, R), 1.0, dtype=np.float64)
        for m in range(order):
            if m == mode:
                continue
            block = block * factors[m][sidx[lo:hi, m], :]
        out[rows[lo]] = np.einsum("n,nr->r", svals[lo:hi], block)
    return out


def reference_mttkrp(w: SparseTensorWorkload, params: dict):
    """fp64 reference via an INDEPENDENT grouped/einsum formulation (see
    _mttkrp_grouped_reference); `scale` is the same independent computation on
    |values| and |factors| -- the cancellation-robust denominator."""
    R = int(params.get("R", DEFAULT_RANK))
    mode = int(params.get("mode", 0)) % w.order
    seed = params.get("seed", 42)
    factors = _make_factors(w.shape, R, seed, np.float64)
    values = w.values.astype(np.float64)
    out = _mttkrp_grouped_reference(w.shape, w.indices, values, factors, mode, R)
    abs_factors = [np.abs(f) for f in factors]
    scale = _mttkrp_grouped_reference(w.shape, w.indices, np.abs(values),
                                      abs_factors, mode, R)
    return out, scale


class NumpyMTTKRP:
    """Vectorized COO MTTKRP: np.add.at scatter-accumulate of Khatri-Rao row
    products (see _mttkrp_coo). Factor-matrix generation and the mode/rank
    resolution are hoisted into prepare(); run() is the timed kernel only."""

    name = "numpy-mttkrp"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        self.precision = precision
        self.dtype = np.float64 if precision == "fp64" else np.float32

    def prepare(self, w: SparseTensorWorkload, params: dict):
        R = int(params.get("R", DEFAULT_RANK))
        params["R"] = R
        # runner.py's generic --dims fallback (variant.dense_dims() or [128])
        # knows nothing about `mode < order`; wrap rather than crash on a bare
        # `--smoke`/`--list` run with no explicit --dims, and echo the actual
        # mode used back into params so the record never hides the wrap.
        mode = int(params.get("mode", 0)) % w.order
        params["mode"] = mode
        factors = _make_factors(w.shape, R, params.get("seed", 42), self.dtype)
        values = w.values.astype(self.dtype)
        return {"shape": w.shape, "indices": w.indices, "values": values,
                "factors": factors, "mode": mode, "R": R}

    def run(self, h):
        return _mttkrp_coo(h["shape"], h["indices"], h["values"], h["factors"],
                           h["mode"], h["R"])

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


# ========================================================= tensor-contraction
_EQ_RE = re.compile(r"^\s*([a-zA-Z]+(?:\s*,\s*[a-zA-Z]+)*)\s*->\s*([a-zA-Z]+)\s*$")


def _parse_equation(equation: str) -> tuple[list[str], str]:
    m = _EQ_RE.match(equation)
    if not m:
        raise ValueError(f"tensor-contraction: cannot parse einsum equation "
                         f"{equation!r}; expected 'lhs1,lhs2->out' (letters only)")
    inputs = [s.strip() for s in m.group(1).split(",")]
    return inputs, m.group(2)


def _first_appearance_letters(equation: str) -> list[str]:
    seen: list[str] = []
    for ch in equation:
        if ch.isalpha() and ch not in seen:
            seen.append(ch)
    return seen


@dataclass
class ContractionWorkload:
    """
    A single binary contraction step: two dense input tensors related to a
    dense output by an einsum equation with exactly 2 input operands (the
    scope this domain module's numpy-einsum/numpy-tensordot implementations
    cover -- see load_workload's docstring for what is explicitly out of
    scope: multi-operand contraction trees and batched/shared free indices).
    """

    name: str
    equation: str
    operand_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    precision: str = "fp32"
    seed: int = SEED_DEFAULT
    source: str = "synthetic"        # "synthetic" | "tccg"
    input_specs: list[str] = field(init=False, repr=False)
    output_spec: str = field(init=False, repr=False)

    def __post_init__(self):
        self.input_specs, self.output_spec = _parse_equation(self.equation)
        if len(self.input_specs) != 2:
            raise ValueError(
                f"tensor-contraction workload {self.name!r}: only 2-operand "
                f"(binary) contractions are supported -- equation "
                f"{self.equation!r} has {len(self.input_specs)} inputs; "
                "multi-operand contraction trees (the spec's SYN/TT/FCTN/TW/"
                "GETD/TRN/MERA/TNLM named application trees) are out of scope "
                "for numpy-tensordot-contraction's pairwise transpose+GEMM "
                "formulation")
        for spec_, shape in zip(self.input_specs, self.operand_shapes):
            if len(spec_) != len(shape):
                raise ValueError(
                    f"tensor-contraction workload {self.name!r}: operand "
                    f"{spec_!r} has {len(spec_)} indices but shape {shape} "
                    f"has {len(shape)} dims")
        sizes = self.letter_sizes()          # also validates consistent sizes
        a_letters = set(self.input_specs[0])
        b_letters = set(self.input_specs[1])
        out_letters = set(self.output_spec)
        batch = (a_letters & b_letters) & out_letters
        if batch:
            raise NotImplementedError(
                f"tensor-contraction workload {self.name!r}: index(es) "
                f"{sorted(batch)} appear in both operands AND the output (a "
                "batched/shared free dimension) -- numpy-tensordot-contraction "
                "does a plain 2D GEMM reshape, not a batched matmul; none of "
                "this kernel's recommended TCCG cases need one")
        lone = (a_letters - b_letters - out_letters) | (b_letters - a_letters - out_letters)
        if lone:
            raise NotImplementedError(
                f"tensor-contraction workload {self.name!r}: index(es) "
                f"{sorted(lone)} appear in exactly one operand and not in the "
                "output (a single-operand reduction) -- np.einsum handles this "
                "natively but numpy-tensordot-contraction's explicit "
                "transpose+GEMM assumes every contracted index is shared by "
                "both operands")
        if any(l not in sizes for l in self.output_spec):
            raise ValueError(f"tensor-contraction workload {self.name!r}: output "
                             f"index not present in either operand")

    def letter_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for spec_, shape in zip(self.input_specs, self.operand_shapes):
            for letter, size in zip(spec_, shape):
                if letter in sizes and sizes[letter] != size:
                    raise ValueError(
                        f"tensor-contraction workload {self.name!r}: index "
                        f"{letter!r} has inconsistent sizes {sizes[letter]} vs "
                        f"{size} across operands")
                sizes[letter] = size
        return sizes

    @property
    def contracted_letters(self) -> list[str]:
        a, b = set(self.input_specs[0]), set(self.input_specs[1])
        return sorted((a & b) - set(self.output_spec))

    @property
    def free_letters(self) -> str:
        return self.output_spec

    def operands(self, dtype) -> tuple[np.ndarray, np.ndarray]:
        """The workload's own dense input tensors ARE the thing generated
        here (there is no separate 'dense operand' concept for this kernel,
        per spec.yaml's dense_operand: n/a) -- seeded from self.seed, the
        same convention stencil.py's StencilWorkload.initial_field() uses."""
        rngA = np.random.default_rng(self.seed)
        rngB = np.random.default_rng(self.seed + 1)
        A = rngA.uniform(-1.0, 1.0, size=self.operand_shapes[0]).astype(dtype)
        B = rngB.uniform(-1.0, 1.0, size=self.operand_shapes[1]).astype(dtype)
        return A, B

    def describe(self) -> dict:
        sizes = self.letter_sizes()
        return {
            "name": self.name,
            "source": self.source,
            "equation": self.equation,
            "operand_shapes": [list(s) for s in self.operand_shapes],
            "output_shape": [sizes[l] for l in self.output_spec],
            "contracted_letters": self.contracted_letters,
            "free_letters": list(self.free_letters),
            "seed": self.seed,
        }


SMOKE_CONTRACTION = [
    # trivial: A already GEMM-ready (contracted axis trailing A, leading B),
    # output order already matches tensordot's natural free-A-then-free-B
    # order -- the "transpose is a no-op" baseline.
    ("smoke-matmul", "ij,jk->ik", ((40, 32), (32, 36))),
    # spec.yaml's own 2nd TCCG recommended_subset case (dbea,ec->abcd),
    # scaled down: A's leftover order (d,b,a) followed by B's (c) is "dbac",
    # which does NOT match the target output "abcd" -- a real permutation is
    # required, not a no-op view. This is the case that makes the einsum-vs-
    # tensordot transpose-cost contrast visible.
    ("smoke-permute", "dbea,ec->abcd", ((6, 5, 4, 7), (4, 8))),
    # spec.yaml's 1st TCCG case (efbad,cf->abcde), scaled down: 5 free + 1
    # contracted index, exercising a wider free-index set than the other two.
    ("smoke-multi-index", "efbad,cf->abcde", ((6, 5, 4, 3, 5), (7, 5))),
]

# spec.yaml's tccg-tree-kernel-fp32 recommended_subset also lists these bare
# (no "->", so they don't hit the TCCG-line branch below) identifiers: the
# multi-operand named application trees, plus kge-batched-fused-kernel's 4
# named knowledge graphs -- neither is a parseable 2-operand equation, so
# load_workload routes them to the same "out of scope" message _load_tccg_case
# gives for an unparseable line, rather than letting them fall through to the
# FROSTT/mttkrp branch below (which would misreport them as a missing FROSTT
# tensor -- these names have nothing to do with mttkrp).
_CONTRACTION_TREE_NAMES = {"syn", "tt", "fctn", "tw", "getd", "trn", "mera", "tnlm"}
_KGE_GRAPH_NAMES = {"fb15k", "fb15k-237", "biokg", "wn18"}

# spec.yaml's tccg-tree-kernel-fp32 recommended_subset lines that are
# literally parseable as "equation  dims  (path)" (the 5 binary-contraction
# TCCG cases). The bare-named application trees (SYN, TT, MERA, TNLM, ...)
# and the other 3 variants' shape families (kge/kron-matmul/quantum-circuit)
# are NOT covered by load_workload -- see its NotImplementedError message.
_TCCG_LINE_RE = re.compile(
    r"^\s*([a-zA-Z]+(?:\s*,\s*[a-zA-Z]+)*\s*->\s*[a-zA-Z]+)\s+"
    r"([0-9]+(?:\s*,\s*[0-9]+)*)\s*(?:\(([^)]*)\))?\s*$"
)


def _contraction_out_of_scope(name: str) -> NotImplementedError:
    return NotImplementedError(
        f"tensor-contraction: {name!r} is not a parseable 'equation  "
        "dims  (path)' TCCG line. This domain module's numpy-einsum/"
        "numpy-tensordot implementations handle pairwise (2-operand) "
        "contractions only; the spec's named multi-operand application "
        "trees (SYN, TT, FCTN, TW, GETD, TRN, MERA, TNLM) and the kge-"
        "batched/kron-matmul/quantum-circuit variants' specialized shape "
        "families are out of scope here -- use smoke_workloads() or "
        "construct a ContractionWorkload directly for a 2-operand "
        "stand-in instead")


def _load_tccg_case(name: str) -> ContractionWorkload:
    m = _TCCG_LINE_RE.match(name)
    if not m:
        raise _contraction_out_of_scope(name)
    equation = re.sub(r"\s+", "", m.group(1))
    dims = [int(x) for x in re.findall(r"\d+", m.group(2))]
    letters = _first_appearance_letters(equation)
    if len(letters) != len(dims):
        raise ValueError(f"tensor-contraction: {name!r} has {len(letters)} "
                         f"distinct index letters but {len(dims)} dims given")
    sizes = dict(zip(letters, dims))
    input_specs, _ = _parse_equation(equation)
    operand_shapes = tuple(tuple(sizes[l] for l in spec_) for spec_ in input_specs)
    return ContractionWorkload(name=equation, equation=equation,
                               operand_shapes=operand_shapes, source="tccg")


def _cost_tensor_contraction(w: ContractionWorkload, params: dict) -> tuple[int, int]:
    """spec.yaml metric.primary, literally: 2 * product of all contracted-
    and-free dimension sizes for one binary contraction step."""
    sizes = w.letter_sizes()
    itemsize = ITEMSIZE[params.get("precision", w.precision)]
    free_size = 1
    for l in w.free_letters:
        free_size *= sizes[l]
    contracted_size = 1
    for l in w.contracted_letters:
        contracted_size *= sizes[l]
    flops = 2 * free_size * contracted_size
    a_elems = 1
    for s in w.operand_shapes[0]:
        a_elems *= s
    b_elems = 1
    for s in w.operand_shapes[1]:
        b_elems *= s
    byts = (a_elems + b_elems + free_size) * itemsize
    return int(flops), int(byts)


workload.register_cost("tensor-contraction", _cost_tensor_contraction, "GFLOP/s")


def reference_tensor_contraction(w: ContractionWorkload, params: dict):
    """fp64 np.einsum ground truth; `scale` is the same contraction evaluated
    on |A|,|B| -- the cancellation-robust denominator (harness.
    check_correctness), the tensor-contraction analog of sparse.py's |A|@|B|."""
    A, B = w.operands(dtype=np.float64)
    out = np.einsum(w.equation, A, B)
    scale = np.einsum(w.equation, np.abs(A), np.abs(B))
    return out, scale


class NumpyEinsumContraction:
    """np.einsum with a pre-searched contraction path -- the "what you get
    for free" baseline. Path search (np.einsum_path, our stand-in for
    CoTenGra/opt_einsum) is hoisted into prepare() per spec.yaml's
    preprocessing_reported rule ("one-time contraction-path search ...
    reported separately, once, never amortized"); run() only re-executes the
    already-planned contraction. Index permutation is fused inside einsum's
    own optimize path -- opaque from the caller's side, which is exactly
    the contrast with NumpyTensordotContraction below."""

    name = "numpy-einsum-contraction"
    platform = "cpu"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = np.float32 if precision == "fp32" else np.float64

    def prepare(self, w: ContractionWorkload, params: dict):
        A, B = w.operands(dtype=self.dtype)
        path, _ = np.einsum_path(w.equation, A, B, optimize="optimal")
        return {"equation": w.equation, "A": A, "B": B, "path": path}

    def run(self, h):
        return np.einsum(h["equation"], h["A"], h["B"], optimize=h["path"])

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


class NumpyTensordotContraction:
    """
    Explicit transpose + GEMM: permute A to (free axes..., contracted
    axes...) and B to (contracted axes..., free axes...), reshape both to
    2D, one BLAS matmul, reshape back, then permute to the equation's own
    output axis order if that differs from tensordot's natural free-A-then-
    free-B layout (see smoke-permute above for a case where it does).

    Whether that permutation work counts as part of the timed kernel is a
    literal spec.yaml notes_on_fairness point (all 4 tensor-contraction
    papers agree it must stay in-kernel) -- exposed here as the explicit
    `transpose_in_timing` param (default True, spec-conforming: prepare()
    stores the raw operands and every permute happens inside run()).
    Setting it False hoists BOTH input permutes into prepare() instead, so
    run() is pure GEMM; the corresponding output permute is then applied
    once in to_host() (called once for the correctness gate, never inside
    the timed reps loop) so the gate still passes on a "transpose excluded"
    number. Running the same workload/impl once with each setting is how
    this module reports both, per the task's "report both where feasible".

    This explicit, steppable pipeline is what numpy-einsum-contraction's
    single opaque np.einsum(..., optimize=path) call does NOT expose --
    that opacity is the point of the contrast.
    """

    name = "numpy-tensordot-contraction"
    platform = "cpu"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = np.float32 if precision == "fp32" else np.float64
        self._skip_output_transpose = False
        self._out_perm: list[int] = []

    def prepare(self, w: ContractionWorkload, params: dict):
        A, B = w.operands(dtype=self.dtype)
        in_a, in_b = w.input_specs
        contracted = w.contracted_letters
        free_a = [l for l in in_a if l not in contracted]
        free_b = [l for l in in_b if l not in contracted]
        axes_a = [in_a.index(l) for l in free_a] + [in_a.index(l) for l in contracted]
        axes_b = [in_b.index(l) for l in contracted] + [in_b.index(l) for l in free_b]
        m = 1
        for l in free_a:
            m *= A.shape[in_a.index(l)]
        n = 1
        for l in free_b:
            n *= B.shape[in_b.index(l)]
        k = 1
        for l in contracted:
            k *= A.shape[in_a.index(l)]
        natural_order = free_a + free_b
        natural_shape = tuple(A.shape[in_a.index(l)] for l in free_a) + \
            tuple(B.shape[in_b.index(l)] for l in free_b)
        out_perm = [natural_order.index(l) for l in w.output_spec]

        transpose_in_timing = bool(params.get("transpose_in_timing", True))
        params["transpose_in_timing"] = transpose_in_timing   # echo resolved value
        self._skip_output_transpose = not transpose_in_timing
        self._out_perm = out_perm

        h = {"axes_a": axes_a, "axes_b": axes_b, "m": m, "n": n, "k": k,
             "natural_shape": natural_shape, "out_perm": out_perm,
             "transpose_in_timing": transpose_in_timing}
        if transpose_in_timing:
            h["A"], h["B"] = A, B
        else:
            # hoist BOTH input transposes out of run(): store already-GEMM-
            # ready 2D operands so run() is pure matmul.
            h["A2d"] = np.transpose(A, axes_a).reshape(m, k)
            h["B2d"] = np.transpose(B, axes_b).reshape(k, n)
        return h

    def run(self, h):
        if h["transpose_in_timing"]:
            A2d = np.transpose(h["A"], h["axes_a"]).reshape(h["m"], h["k"])
            B2d = np.transpose(h["B"], h["axes_b"]).reshape(h["k"], h["n"])
            out = (A2d @ B2d).reshape(h["natural_shape"])
            return np.transpose(out, h["out_perm"])
        return (h["A2d"] @ h["B2d"]).reshape(h["natural_shape"])

    def to_host(self, out):
        out = np.asarray(out, dtype=np.float64)
        if self._skip_output_transpose:
            out = np.transpose(out, self._out_perm)
        return out

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


# ==================================================================== shared
def smoke_workloads(kernel: str | None = None):
    """
    `kernel` selects which of this domain's two structurally different input
    types to build. Unlike every other domain module (which shares one
    workload type across all its kernels and so needs no such parameter),
    tensor.py's mttkrp (sparse COO tensor) and tensor-contraction (dense
    operand pair) cannot share one -- runner.py detects this parameter via
    inspect.signature and passes args.kernel automatically; a direct caller
    that omits it gets both smoke sets concatenated.
    """
    mttkrp_smoke = [synthetic_tensor(n, **kw) for n, kw in SMOKE_MTTKRP]
    contraction_smoke = [
        ContractionWorkload(name=n, equation=eq, operand_shapes=shapes)
        for n, eq, shapes in SMOKE_CONTRACTION
    ]
    if kernel == "mttkrp":
        return mttkrp_smoke
    if kernel == "tensor-contraction":
        return contraction_smoke
    return mttkrp_smoke + contraction_smoke


def load_workload(name: str):
    """
    Dispatches on the shape of `name`:
      * contains "->"          -> a TCCG "equation  dims  (path)" line (see
                                   _load_tccg_case)
      * a bare TCCG named application tree (SYN, TT, FCTN, TW, GETD, TRN,
        MERA, TNLM) or one of the 4 kge-batched-fused-kernel graph names
                                -> NotImplementedError (multi-operand trees /
                                   a different shape family, out of scope for
                                   the pairwise numpy-einsum/tensordot impls)
      * "synthetic-*" or one of the 5 named SymProp hypergraph datasets
                                -> NotImplementedError (symmetric-kernel
                                   variant's generator/loader not built here)
      * otherwise               -> a FROSTT tensor name (first whitespace-
                                   separated token, so spec.yaml's
                                   recommended_subset entries like "nips
                                   (2.5K x 2.9K x 14K x 17, 3.1M nnz)" work
                                   verbatim); reads TENSOR_CACHE/<name>.tns if
                                   present, otherwise raises with a download
                                   pointer -- NEVER auto-downloads (several
                                   FROSTT tensors are 10s of GB uncompressed)
    """
    if "->" in name:
        return _load_tccg_case(name)
    slug = name.split()[0] if name.split() else name
    if slug.lower() in _CONTRACTION_TREE_NAMES or slug.lower() in _KGE_GRAPH_NAMES:
        raise _contraction_out_of_scope(name)
    if slug.startswith("synthetic-") or slug in _SYMMETRIC_HYPERGRAPH_NAMES:
        raise NotImplementedError(
            f"mttkrp: {slug!r} belongs to the symmetric-kernel variant's "
            "Erdos-Renyi generator (SySTeC protocol) or SymProp's real "
            "hypergraph-derived corpus; neither generator/loader is "
            "implemented in this module. Use smoke_workloads() or "
            "synthetic_tensor(...) for a general (asymmetric) stand-in.")
    path = _frostt_path(slug)
    if not os.path.exists(path):
        hint = FROSTT_SHAPES.get(slug)
        hint_txt = f" (expected shape {hint[0]}, ~{hint[1]:,} nnz per FROSTT/BLCO)" \
            if hint else ""
        raise LookupError(
            f"mttkrp: FROSTT tensor {slug!r} not found at {path!r}{hint_txt}. "
            "This loader does NOT auto-download FROSTT tensors -- several "
            "exceed 10s of GB uncompressed (nell-1 alone has 143.6M "
            f"nonzeros). Download the .tns file from "
            f"http://frostt.io/tensors/{slug}/ (or browse http://frostt.io/) "
            "and place it at the path above, or point "
            "KERNELBENCH_TENSOR_CACHE at a directory that already has it. "
            "For a no-download stand-in, use smoke_workloads() or "
            "synthetic_tensor(...).")
    t0 = time.perf_counter()
    indices, values, shape = _load_frostt_tns(path)
    return SparseTensorWorkload(name=slug, shape=shape, indices=indices, values=values,
                                source="frostt", load_seconds=time.perf_counter() - t0)


REFERENCES = {
    "mttkrp": reference_mttkrp,
    "tensor-contraction": reference_tensor_contraction,
}

# gate mode: scale-aware everywhere, matching sparse.py's rationale in
# harness.check_correctness (both kernels are sums of signed products, so a
# pointwise-relative gate is not cancellation-robust)
CORRECTNESS_MODE = {k: "max_scaled_err" for k in KERNELS}

DEFAULT_PRECISION = {"mttkrp": "fp64", "tensor-contraction": "fp32"}

REFERENCE_NAME = {
    "mttkrp": "numpy fp64 COO MTTKRP (np.add.at over Khatri-Rao row products)",
    "tensor-contraction": "numpy fp64 einsum",
}

CPU_IMPLS = {
    "mttkrp": {"numpy-mttkrp": NumpyMTTKRP},
    "tensor-contraction": {
        "numpy-einsum-contraction": NumpyEinsumContraction,
        "numpy-tensordot-contraction": NumpyTensordotContraction,
    },
}


def cuda_impls():
    from ..impls import gpu_cuda as g
    return {
        "mttkrp": {"torch-mttkrp": g.TorchMTTKRP},
        "tensor-contraction": {
            "torch-einsum-contraction": g.TorchEinsumContraction,
            "torch-tensordot-contraction": g.TorchTensordotContraction,
        },
    }


# which params key carries the swept dimension (None => no sweep). mttkrp
# sweeps `mode` (the spec's own fairness point: several papers report mode-0
# only); tensor-contraction's shape comes entirely from the workload itself,
# so it has no DIM_KEY entry (runner.py then uses dims=[1], no extra param).
DIM_KEY = {"mttkrp": "mode"}
