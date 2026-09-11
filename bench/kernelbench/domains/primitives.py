"""
Parallel primitives: scan-reduction, topk-selection, hash-table
(+ planned: sort, set-intersection).

These are the textbook BANDWIDTH-bound building blocks -- every kernel here
does O(1) work per element touched, so the meaningful metric is GB/s (bytes
moved per second) or its op-counted cousin Mops/s, never GFLOP/s. Every cost
rule below registers a byte- or op-count unit accordingly (see DOMAIN_GUIDE.md
"work_count must follow the spec's metric field literally" -- here the metric
field literally IS a bandwidth/op-rate formula, not a flop formula).

Five deliberate, documented deviations from a byte-for-byte reading of the
benchspecs (each called out again at its point of use, following graph.py's
precedent for this kind of disclosure):

1. ONE shared `Primitive` workload class serves all three kernels, even
   though their natural inputs differ (a dense array; a dense array + k; a
   key/value population). This is forced by the runner's contract: unlike a
   per-kernel signature, `runner.py` calls `domain.smoke_workloads()` ONCE
   and iterates the SAME list of objects against whichever kernel's impls
   were selected on the command line. `sparse.py`/`graph.py`/`compression.py`
   get away with a single natural type because every kernel THEY own already
   consumes the same sparse matrix / graph / field. This module's three
   kernels do not share a natural type, so `Primitive` below is built to
   always carry a consistent, seed-derived superset covering all three --
   `array`/`distribution` for scan-reduction and topk-selection (which
   genuinely do share the "dense array" shape), plus `keys`/`values`/
   `capacity`/... for hash-table. Each kernel's `prepare()` reads only the
   subset it needs; nothing is fabricated per-kernel, everything derives from
   the same seed family.
2. scan-reduction: the spec (see benchspecs/scan-reduction/spec.yaml's
   open_questions) is explicit that NO surveyed paper implements a literal
   Blelloch/Hillis-Steele all-prefixes scan despite the track's own name.
   This module implements BOTH the spec-grounded flat reduction
   (`numpy-reduction`, matches scan-reduction-flat-kernel) AND a literal
   inclusive/exclusive scan (`numpy-scan`, via `np.cumsum`) to close that gap,
   recording which ran (`params["mode"]`) and, for scan, which variant
   (`params["scan_type"]`) in every result record. The segmented-histogram
   and fused-cascaded variants are NOT implemented here (out of scope for a
   first CPU/numpy pass; a real GPU atomic-histogram or fused-softmax-GEMM
   kernel needs device code this module doesn't have).
3. topk-selection: the spec's three distributions for topk-array-kernel-exact
   are Uniform, Normal, and an adversarial "Unfriendly" bit-pattern-clustered
   distribution. This module implements Uniform and the adversarial case
   verbatim (`_adversarial_clustered` masks all but the low 12 bits of each
   float's IEEE-754 pattern to a common prefix, exactly as specified), but
   substitutes a right-skewed Gamma(shape=2) distribution ("skewed") for the
   spec's Normal slot, matching this task's explicit request for a "skewed"
   third distribution over a symmetric one -- documented here rather than
   silently presented as the spec's own Normal case.
4. hash-table: all three spec variants pull real corpora (SNAP/DIMACS
   relations, genomic k-mer streams, YCSB traces) this stdlib/numpy-only
   module does not download (same "synthetic surrogate, not the real
   multi-GB corpus" choice compression.py already made, disclosed the same
   way: every `Primitive.describe()` always says `synthetic: True`). More
   substantially, this module does not implement dynamic resizing or delete
   -- the op-mix trace's "write" ops are therefore upserts of
   ALREADY-populated keys, never new keys that would grow the table past its
   pre-sized capacity. This keeps a load-factor sweep point well-defined and
   every repeated `run()` call idempotent (see `Primitive`/impl docstrings),
   at the cost of not exercising Clevel-Hashing's own resize-triggered
   max-load-factor comparison (hash-table-crud-mixed-ops's own headline
   number). `key_distribution` ('uniform'|'zipf') governs the ACCESS pattern
   over an already-distinct key pool (which key index a trace op touches),
   matching the YCSB-Zipfian convention (skewed ACCESS to a fixed key set),
   not skewed key VALUES -- the key set itself is always drawn uniformly at
   random from a keyspace far larger than num_keys, so it is distinct by
   construction.
5. hash-table's registered unit is "Mops/s", but harness.metrics.throughput
   always computes `work_count / seconds / 1e9` -- there is no separate
   mega-scale code path in harness.py. `_cost_hash` therefore returns
   `work_count = ops * 1000`, which makes the harness's fixed /1e9 division
   land on `ops / seconds / 1e6` = Mops/s exactly. This is a disclosed
   scaling correction for harness.py's fixed Giga-scale formula, not a
   gaming trick -- see `_cost_hash`'s own docstring.

Correctness gates, one per kernel, none the harness's plain default:

  * scan-reduction: `max_scaled_err` with `scale = sum(|x|)` (flat) or a
    running `cumsum(|x|)` (scan) -- the standard backward-error bound for
    summation. Floating-point addition is not associative, and every
    implementation here (numpy's own pairwise summation for the flat case,
    a left-to-right cumsum for the scan case) reassociates relative to a
    naive reference; a bit-exact gate would fail a CORRECT reduction, which
    is exactly the trap DOMAIN_GUIDE.md and harness.check_correctness's own
    docstring warn about. See `reference_scan_reduction`.
  * topk-selection: `exact`, but on the SORTED-DESCENDING VALUE array, not
    the raw (value, index) output -- see `NumpyTopK`'s docstring for the
    tie-handling rule this is built around.
  * hash-table: `exact` integer match -- "the table must return exactly the
    inserted mapping" has no meaningful tolerance, only right or wrong.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import numpy as np

from .. import workload
from ..harness import Timer

KERNELS = ["scan-reduction", "topk-selection", "hash-table"]
# owned by this domain but not yet implemented; kept honest for --list-kernels
PLANNED = ["sort", "set-intersection"]

ITEMSIZE = {"fp64": 8, "fp32": 4, "fp16": 2, "int64": 8, "int32": 4}

SEED = 20260806  # matches sparse.py/compression.py's synthetic-data seed convention

# spec's own array-size breakpoints (scan-reduction-flat-kernel / topk-array-
# kernel-exact): cache-resident through HBM-resident, at fp32. Used by
# load_workload() when a name references one explicitly; NOT exercised by
# smoke_workloads() (2^30 elements is ~4-8GB, far outside "runs anywhere in
# seconds" -- see DOMAIN_GUIDE.md).
FULL_SIZE_SWEEP = {"2^18": 1 << 18, "2^20": 1 << 20, "2^24": 1 << 24,
                    "2^27": 1 << 27, "2^30": 1 << 30}


# ============================================================== workload
@dataclass
class Primitive:
    """
    Unified synthetic input for scan-reduction, topk-selection and
    hash-table -- see the module docstring (deviation 1) for why one class
    serves three structurally different kernels. Every field is derived from
    the same `seed`, so two `Primitive`s built with the same seed are
    reproducible end to end, matching this project's synthetic-data
    convention elsewhere (sparse.py's `matrices.synthetic`, compression.py's
    `Field`).
    """

    name: str
    seed: int
    # --- scan-reduction / topk-selection: a dense array ---
    array: np.ndarray
    distribution: str          # 'uniform' | 'skewed' | 'adversarial-clustered'
    # --- topk-selection only ---
    k: int
    k_fraction: float
    # --- hash-table only ---
    num_keys: int
    keys: np.ndarray                  # distinct int64 keys, the "build" population
    values: np.ndarray                # matching int64 values
    target_load_factor: float
    capacity: int                     # power-of-2 slot count
    key_distribution: str             # 'uniform' | 'zipf' -- ACCESS pattern, see module docstring
    insert_ratio: float               # fraction of trace ops that are writes (upserts)
    trace_pool_idx: np.ndarray        # int, indices into `keys`/`values` for each trace op
    trace_is_write: np.ndarray        # bool, per trace op
    trace_write_value: np.ndarray     # int64, the value a write op stores

    def describe(self) -> dict:
        return {
            "name": self.name,
            "source": "synthetic",
            "synthetic": True,   # never mistaken for a real corpus pull (see module docstring)
            "seed": self.seed,
            "array": {
                "n": int(self.array.size),
                "dtype": str(self.array.dtype),
                "bytes": int(self.array.nbytes),
                "distribution": self.distribution,
                "size_regime": _size_regime(int(self.array.size), int(self.array.itemsize)),
            },
            "topk": {
                "k": self.k,
                "k_fraction": self.k_fraction,
            },
            "hashtable": {
                "num_keys": self.num_keys,
                "capacity": self.capacity,
                "target_load_factor": self.target_load_factor,
                "load_factor_actual": self.num_keys / self.capacity,
                "key_distribution": self.key_distribution,
                "insert_ratio": self.insert_ratio,
                "trace_len": int(self.trace_pool_idx.size),
                "key_bytes": 8,
                "value_bytes": 8,
            },
        }


def _size_regime(n: int, itemsize: int) -> str:
    """
    A coarse CPU-cache-size label standing in for the spec's GPU L2/HBM
    framing (this module has no GPU implementation -- see cuda_impls()).
    Boundaries are typical LLC sizes, not measured on the actual machine.
    """
    nbytes = n * itemsize
    if nbytes <= 256 * 1024:
        return "L1/L2-resident (<=256KB)"
    if nbytes <= 8 * 1024 * 1024:
        return "L3-resident (<=8MB)"
    if nbytes <= 512 * 1024 * 1024:
        return "exceeds typical LLC, sub-HBM-scale (<=512MB)"
    return "HBM-scale (>512MB)"


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _gen_array(n: int, distribution: str, seed: int, dtype=np.float32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        return rng.random(n, dtype=np.float32).astype(dtype, copy=False)
    if distribution == "skewed":
        # right-skewed Gamma(shape=2): a stand-in for the literature's skewed
        # score distributions (see module docstring, deviation 3)
        return rng.standard_gamma(2.0, size=n).astype(dtype)
    if distribution == "adversarial-clustered":
        return _adversarial_clustered(n, seed, dtype)
    raise ValueError(f"unknown distribution {distribution!r}")


def _adversarial_clustered(n: int, seed: int, dtype) -> np.ndarray:
    """
    AIR Top-K/GridSelect's own "Unfriendly" distribution: fix all but the
    low 12 bits of a common fp32 bit pattern, vary only the low 12 bits per
    element. This masks the entropy a radix-select pass normally prunes on,
    and is the only distribution this track's literature documents as
    changing which top-k algorithm wins -- REQUIRED, not decorative.
    """
    rng = np.random.default_rng(seed)
    base_f32 = np.array([rng.random()], dtype=np.float32)
    base_bits = base_f32.view(np.uint32)[0]
    low_bits = rng.integers(0, 1 << 12, size=n, dtype=np.uint32)
    bits = np.uint32(base_bits & np.uint32(0xFFFFF000)) | low_bits
    return bits.view(np.float32).astype(dtype, copy=False)


def _gen_keys(num_keys: int, seed: int) -> np.ndarray:
    """
    Distinct non-negative int64 keys (top bit cleared, so -1 is always a
    safe EMPTY sentinel that can never collide with a real key). The
    keyspace (2^62) is astronomically larger than any num_keys this module
    exercises, so a couple of oversample-and-dedupe rounds always converge.
    """
    rng = np.random.default_rng(seed)
    keys: set[int] = set()
    while len(keys) < num_keys:
        need = num_keys - len(keys)
        cand = rng.integers(0, 1 << 62, size=need * 2 + 16, dtype=np.int64)
        keys.update(cand.tolist())
    arr = np.array(sorted(keys)[:num_keys], dtype=np.int64)
    rng.shuffle(arr)
    return arr


def _derive_values(keys: np.ndarray) -> np.ndarray:
    """Deterministic value-per-key (Knuth multiplicative hash of the key,
    masked to 31 bits): cheap, reproducible, and exactly representable in
    float64 for the correctness gate."""
    return ((keys.astype(np.uint64) * np.uint64(2654435761)) & np.uint64(0x7FFFFFFF)).astype(np.int64)


def _gen_trace_idx(num_keys: int, trace_len: int, key_distribution: str, seed: int) -> np.ndarray:
    """Which key-pool index each trace op touches -- see module docstring
    (deviation 4) for why this, not the key VALUES, is what 'uniform vs
    Zipf' governs here."""
    rng = np.random.default_rng(seed)
    if key_distribution == "uniform":
        return rng.integers(0, num_keys, size=trace_len)
    if key_distribution == "zipf":
        raw = rng.zipf(a=1.5, size=trace_len).astype(np.int64) - 1
        return raw % num_keys
    raise ValueError(f"unknown key_distribution {key_distribution!r}")


def _make_primitive(name: str, seed: int, n: int, distribution: str, k_fraction: float,
                     num_keys: int, target_load_factor: float, key_distribution: str,
                     insert_ratio: float, trace_len: int) -> Primitive:
    array = _gen_array(n, distribution, seed, dtype=np.float32)
    k = max(1, min(n - 1, int(round(n * k_fraction))))
    keys = _gen_keys(num_keys, seed + 1000)
    values = _derive_values(keys)
    capacity = _next_pow2(int(np.ceil(num_keys / target_load_factor)))
    idx = _gen_trace_idx(num_keys, trace_len, key_distribution, seed + 2000)
    is_write = np.random.default_rng(seed + 3000).random(trace_len) < insert_ratio
    # write value = f(key, trace position): deterministic, reproducible, and
    # shared verbatim by every impl's run() and by reference_hash() below --
    # this is a WORKLOAD input (like an SpMM's dense operand B), not
    # something each impl should independently reinvent.
    write_val = ((keys[idx].astype(np.uint64) * np.uint64(2654435761)
                  + np.arange(trace_len, dtype=np.uint64)) & np.uint64(0x7FFFFFFF)).astype(np.int64)
    return Primitive(
        name=name, seed=seed, array=array, distribution=distribution,
        k=k, k_fraction=k_fraction, num_keys=num_keys, keys=keys, values=values,
        target_load_factor=target_load_factor, capacity=capacity,
        key_distribution=key_distribution, insert_ratio=insert_ratio,
        trace_pool_idx=idx, trace_is_write=is_write, trace_write_value=write_val,
    )


# spec's own topk k_fraction sweep points: k/n in {2^-12, 2^-8, 2^-4, 2^-2}
SMOKE = [
    dict(name="prim-smoke-uniform", n=4096, distribution="uniform",
         k_fraction=2 ** -4, num_keys=2000, target_load_factor=0.5,
         key_distribution="uniform", insert_ratio=0.2, trace_len=1500, seed=SEED),
    dict(name="prim-smoke-skewed", n=1 << 20, distribution="skewed",
         k_fraction=2 ** -8, num_keys=3000, target_load_factor=0.75,
         key_distribution="zipf", insert_ratio=0.5, trace_len=2000, seed=SEED + 1),
    dict(name="prim-smoke-adversarial", n=1 << 22, distribution="adversarial-clustered",
         k_fraction=2 ** -8, num_keys=2500, target_load_factor=0.4,
         key_distribution="zipf", insert_ratio=0.0, trace_len=1800, seed=SEED + 2),
]


def smoke_workloads() -> list[Primitive]:
    """Small, synthetic, runs anywhere in seconds. NOT spec-conforming --
    see Primitive.describe() (`synthetic: True` always) and FULL_SIZE_SWEEP
    for the real spec breakpoints load_workload() can reach."""
    return [_make_primitive(**cfg) for cfg in SMOKE]


# ------------------------------------------------------------- name parsing
_INT_RE = re.compile(r"(\d{3,})")


def _parse_n(text: str, default: int) -> int:
    m = re.search(r"(?<!-)2\^(\d+)", text)
    if m:
        return 1 << int(m.group(1))
    m = _INT_RE.search(text)
    return int(m.group(1)) if m else default


def _parse_distribution(text: str, default: str) -> str:
    if any(k in text for k in ("adversarial", "unfriendly", "clustered")):
        return "adversarial-clustered"
    if any(k in text for k in ("skew", "gamma", "normal", "gauss")):
        # 'normal' maps to 'skewed' too -- this module substitutes a skewed
        # distribution for the spec's Normal slot everywhere (deviation 3)
        return "skewed"
    return "uniform" if "uniform" in text else default


def _parse_k_fraction(text: str, default: float) -> float:
    m = re.search(r"2\^-(\d+)", text)
    return 2.0 ** (-int(m.group(1))) if m else default


def _parse_key_distribution(text: str, default: str) -> str:
    if "zipf" in text:
        return "zipf"
    if "uniform" in text:
        return "uniform"
    return default


def load_workload(name: str) -> Primitive:
    """
    Build a synthetic surrogate from hints in `name` -- e.g. a name
    containing "2^24" or "adversarial" steers the array side; "zipf"
    steers the hash-table access pattern; a bare "0.NN" is read as a target
    load factor. None of the three kernels' real corpora (SuiteSparse-hosted
    relations, genomic reads, YCSB traces) are downloaded here -- see the
    module docstring (deviation 4) and compression.py's identical precedent.
    Always reports `synthetic: True` via Primitive.describe().
    """
    text = name.lower()
    n = _parse_n(text, default=1 << 20)
    distribution = _parse_distribution(text, default="uniform")
    k_fraction = _parse_k_fraction(text, default=2 ** -8)
    key_distribution = _parse_key_distribution(text, default="zipf")
    lf_m = re.search(r"(?:lf|load.?factor)[=_]?(0?\.\d+)", text)
    target_load_factor = float(lf_m.group(1)) if lf_m else 0.5
    # pure-Python open-addressing build (see NumpyOpenAddressingHash): keep
    # num_keys tractable even when `n` (the array side) is large
    num_keys = min(n, 20000)
    if any(k in text for k in ("crud", "mixed", "write")):
        insert_ratio = 0.5
    elif any(k in text for k in ("probe", "lookup", "read")):
        insert_ratio = 0.0
    else:
        insert_ratio = 0.2
    return _make_primitive(
        name=f"named::{name}", seed=SEED, n=n, distribution=distribution,
        k_fraction=k_fraction, num_keys=num_keys, target_load_factor=target_load_factor,
        key_distribution=key_distribution, insert_ratio=insert_ratio,
        trace_len=min(num_keys, 5000))


# =========================================================== scan-reduction
def _cost_scan(w: Primitive, params: dict) -> tuple[int, int]:
    """
    Literal per-variant formula: flat reduction reads n once, writes 1
    scalar (scan-reduction-flat-kernel's own GB/s formula); a full
    inclusive/exclusive scan reads n once and writes n outputs. Both
    directions are, by the spec's own definition, exactly the compulsory
    byte lower bound too -- so work_count == byte_count here is not
    redundant, it is what "the primary metric literally is bytes/time"
    means for a bandwidth-bound kernel (see module docstring).
    """
    itemsize = ITEMSIZE.get(params.get("precision", "fp32"), 4)
    n = int(w.array.size)
    mode = params.get("mode", "reduction")
    b = n * itemsize + (n * itemsize if mode == "scan" else itemsize)
    return b, b


workload.register_cost("scan-reduction", _cost_scan, "GB/s")


def reference_scan_reduction(w: Primitive, params: dict):
    """
    fp64 independent reference. `scale` is the standard backward-error bound
    for summation: `sum(|x|)` for the flat case, a running `cumsum(|x|)` for
    the scan case (the natural elementwise generalization -- each prefix's
    error bound grows with the magnitude summed into IT, not the whole
    array). See harness.check_correctness's own docstring for why this, not
    a bit-exact gate, is the correct check for a reassociated reduction.
    """
    x = w.array.astype(np.float64)
    mode = params.get("mode", "reduction")
    if mode == "scan":
        scan_type = params.get("scan_type", "inclusive")
        absx = np.abs(x)
        if scan_type == "exclusive":
            ref = np.concatenate(([0.0], np.cumsum(x)[:-1]))
            scale = np.concatenate(([1.0], np.cumsum(absx)[:-1]))
        else:
            ref = np.cumsum(x)
            scale = np.cumsum(absx)
        return ref, np.maximum(scale, 1e-300)
    ref = np.array([np.sum(x)])
    scale = np.array([np.sum(np.abs(x))])
    return ref, np.maximum(scale, 1e-300)


class NumpyReduction:
    """
    Flat sum-reduction (scan-reduction-flat-kernel's own shape): read the
    array once, write one scalar. Accumulates in the array's OWN dtype
    (fp32 by default), so the reduction genuinely exercises fp32 rounding,
    not an inflated fp64-accumulate shortcut -- `np.sum`'s pairwise
    (cascade) summation reassociates relative to a naive left-to-right
    reference, which is exactly what the max_scaled_err gate (see
    reference_scan_reduction) is built to tolerate rather than penalize.
    """

    name = "numpy-reduction"
    platform = "cpu"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = np.float32 if precision == "fp32" else np.float64

    def prepare(self, w: Primitive, params: dict):
        params["mode"] = "reduction"
        params["n"] = int(w.array.size)
        params["distribution"] = w.distribution
        return {"data": w.array.astype(self.dtype, copy=False)}

    def run(self, h):
        return np.array([np.sum(h["data"])])

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


class NumpyScan:
    """
    Inclusive/exclusive all-prefixes scan via `np.cumsum` -- the literal
    "scan" this track is named after, which no surveyed paper actually
    implements (see module docstring, deviation 2). `scan_type` is fixed at
    construction and always recorded in `params`, so which mode ran is
    never ambiguous in the result record.
    """

    name = "numpy-scan"
    platform = "cpu"

    def __init__(self, precision: str = "fp32", scan_type: str = "inclusive"):
        self.precision = precision
        self.dtype = np.float32 if precision == "fp32" else np.float64
        self.scan_type = scan_type

    def prepare(self, w: Primitive, params: dict):
        params["mode"] = "scan"
        params["scan_type"] = self.scan_type
        params["n"] = int(w.array.size)
        params["distribution"] = w.distribution
        return {"data": w.array.astype(self.dtype, copy=False)}

    def run(self, h):
        out = np.cumsum(h["data"])
        if self.scan_type == "exclusive":
            out = np.concatenate(([np.array(0, dtype=out.dtype)], out[:-1]))
        return out

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


# ============================================================ topk-selection
def _cost_topk(w: Primitive, params: dict) -> tuple[int, int]:
    """Read n candidates once, write k (value, index) pairs -- the
    compulsory byte lower bound, used directly as the GB/s numerator."""
    itemsize = ITEMSIZE.get(params.get("precision", "fp32"), 4)
    ib = 4
    n = int(w.array.size)
    k = int(params.get("k", w.k))
    b = n * itemsize + k * (itemsize + ib)
    return b, b


workload.register_cost("topk-selection", _cost_topk, "GB/s")


def reference_topk(w: Primitive, params: dict):
    """fp64 exact ground truth: full sort, top k, descending. See
    NumpyTopK's docstring for why comparing SORTED VALUES (not raw
    (value, index) pairs) is the correct, tie-order-independent gate."""
    data = w.array.astype(np.float64)
    k = int(params.get("k", w.k))
    ref_vals = np.sort(data)[::-1][:k]
    return ref_vals, None


class NumpyTopK:
    """
    Exact top-k via `np.argpartition` (expected O(n) selection, not a full
    O(n log n) sort) -- topk-array-kernel-exact's own benchmark shape.

    Tie-handling rule (spec's own correctness text, adopted verbatim):
    "sorted top-k value multiset must exactly match ... no duplicate
    indices [...] every returned index must be < n and must map back to its
    claimed value". Ties at the k-th largest value change WHICH index holds
    a given repeated value, never the resulting value multiset -- so
    to_host()/reference_topk() compare sorted-descending VALUE arrays only,
    which is exact and tie-order-independent by construction. Index
    validity (bounds, uniqueness, value-mapping) is enforced structurally
    inside run() via an assertion, since harness.check_correctness never
    sees indices, only the numeric array it gates on.
    """

    name = "numpy-topk"
    platform = "cpu"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.dtype = np.float32 if precision == "fp32" else np.float64

    def prepare(self, w: Primitive, params: dict):
        params["k"] = w.k
        params["k_fraction"] = w.k_fraction
        params["distribution"] = w.distribution
        params["tie_rule"] = (
            "ties at the k-th largest value are resolved arbitrarily by "
            "index; only the resulting sorted VALUE multiset is gated for "
            "correctness (matches the spec's correctness text verbatim)")
        return {"data": w.array.astype(self.dtype, copy=False), "k": w.k, "n": int(w.array.size)}

    def run(self, h):
        data, k, n = h["data"], h["k"], h["n"]
        idx = np.argpartition(-data, k - 1)[:k]
        if idx.size != k or np.unique(idx).size != k or not np.all((idx >= 0) & (idx < n)):
            raise AssertionError(
                "numpy-topk: invalid index set (bounds/uniqueness violated)")
        return data[idx]

    def to_host(self, out):
        return np.sort(np.asarray(out, dtype=np.float64))[::-1]

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


# =============================================================== hash-table
EMPTY = np.int64(-1)


def _hash_mult(k: int, capacity: int) -> int:
    """Knuth multiplicative hash, masked to the (power-of-two) capacity --
    disclosed explicitly per the spec's 'disclose hash function' requirement;
    identical across every impl below."""
    return ((k * 2654435761) & 0xFFFFFFFFFFFFFFFF) & (capacity - 1)


def _oa_insert(tk: np.ndarray, tv: np.ndarray, k: int, v: int, capacity: int) -> None:
    idx = _hash_mult(k, capacity)
    for _ in range(capacity):
        cur = tk[idx]
        if cur == EMPTY or cur == k:
            tk[idx] = k
            tv[idx] = v
            return
        idx = (idx + 1) & (capacity - 1)
    raise RuntimeError("numpy-open-addressing-hash: table full (load factor > 1.0)")


def _oa_lookup(tk: np.ndarray, tv: np.ndarray, k: int, capacity: int) -> int:
    idx = _hash_mult(k, capacity)
    for _ in range(capacity):
        cur = tk[idx]
        if cur == k:
            return int(tv[idx])
        if cur == EMPTY:
            return -1
        idx = (idx + 1) & (capacity - 1)
    return -1


def _cost_hash(w: Primitive, params: dict) -> tuple[int, int]:
    """
    work_count = ops * 1000, NOT ops -- see module docstring (deviation 5):
    harness.metrics.throughput always divides by seconds*1e9, so this is
    the exact compensation needed for the registered unit ("Mops/s") to be
    numerically correct rather than 1000x understated. `ops` is the number
    of trace operations one run() call performs (build/insert is
    preprocessing, excluded here and reported separately via
    params['table_bytes']/['memory_overhead'] -- see the impls below).
    """
    ops = int(w.trace_pool_idx.size)
    key_bytes, value_bytes = 8, 8
    byts = ops * (key_bytes + value_bytes)
    work = ops * 1000
    return work, byts


workload.register_cost("hash-table", _cost_hash, "Mops/s")


def reference_hash(w: Primitive, params: dict):
    """
    Independent ground truth: a fresh Python dict (built and replayed here,
    never sharing code with either impl below -- for python-dict-hash this
    is not a fully independent algorithm family, since it too is a Python
    dict; see PythonDictHash's docstring for that one disclosed exception).
    Returns, for every trace op, the value observed immediately after that
    op (just-written value for a write; found value, or -1, for a lookup) --
    exact equality against an impl's own same-shaped output validates every
    insert AND every lookup in one pass, including read-after-write
    ordering within the trace.
    """
    table: dict[int, int] = {}
    for key, value in zip(w.keys.tolist(), w.values.tolist()):
        table[key] = value
    trace_keys = w.keys[w.trace_pool_idx]
    out = np.empty(w.trace_pool_idx.size, dtype=np.int64)
    for i in range(w.trace_pool_idx.size):
        k = int(trace_keys[i])
        if w.trace_is_write[i]:
            v = int(w.trace_write_value[i])
            table[k] = v
            out[i] = v
        else:
            out[i] = table.get(k, -1)
    return out.astype(np.float64), None


class NumpyOpenAddressingHash:
    """
    A REAL linear-probing open-addressing table over plain numpy int64
    arrays (not a Python dict wrapping an opaque table) -- so the
    load-factor sweep is actually meaningful; CPython dict's own resize
    policy/load factor is an implementation detail, not user-controllable
    (see PythonDictHash for the naive baseline that has this limitation).

    BUILD (bulk insert of the base num_keys population) is entirely inside
    prepare(), timed once as preprocessing and excluded from the timed
    op-mix loop, matching every hash-table spec variant's
    timing_scope/preprocessing_reported split. This module does not
    implement dynamic resizing or delete (see module docstring, deviation
    4) -- op-mix "write" ops are upserts of already-populated keys, never
    new keys that would grow past the pre-sized capacity, which is what
    keeps repeated run() calls (correctness check + warmup + timed reps)
    idempotent and the fixed load-factor point well-defined throughout.
    """

    name = "numpy-open-addressing-hash"
    platform = "cpu"

    def __init__(self, precision: str = "int64"):
        self.precision = precision

    def prepare(self, w: Primitive, params: dict):
        capacity = w.capacity
        tk = np.full(capacity, EMPTY, dtype=np.int64)
        tv = np.zeros(capacity, dtype=np.int64)
        for k, v in zip(w.keys.tolist(), w.values.tolist()):
            _oa_insert(tk, tv, k, v, capacity)
        key_bytes, value_bytes = 8, 8
        table_bytes = capacity * (key_bytes + value_bytes)
        params["load_factor_target"] = w.target_load_factor
        params["capacity"] = capacity
        params["load_factor_actual"] = w.num_keys / capacity
        params["insert_ratio"] = w.insert_ratio
        params["key_distribution"] = w.key_distribution
        params["hash_function"] = "Knuth multiplicative (k*2654435761 mod capacity), linear probing"
        params["memory_overhead"] = table_bytes / (w.num_keys * (key_bytes + value_bytes)) - 1
        params["table_bytes"] = table_bytes
        return dict(tk=tk, tv=tv, capacity=capacity,
                    trace_keys=w.keys[w.trace_pool_idx],
                    trace_is_write=w.trace_is_write,
                    trace_write_value=w.trace_write_value)

    def run(self, h):
        tk, tv, capacity = h["tk"], h["tv"], h["capacity"]
        keys, is_write, wval = h["trace_keys"], h["trace_is_write"], h["trace_write_value"]
        out = np.empty(keys.shape[0], dtype=np.int64)
        for i in range(keys.shape[0]):
            k = int(keys[i])
            if is_write[i]:
                v = int(wval[i])
                _oa_insert(tk, tv, k, v, capacity)
                out[i] = v
            else:
                out[i] = _oa_lookup(tk, tv, k, capacity)
        return out

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


class PythonDictHash:
    """
    CPython dict baseline: fast, but its internal table layout, load
    factor and resize policy are opaque and not independently
    controllable -- included as a naive/reference-speed comparison point,
    NOT as the implementation the load-factor sweep is meant to
    characterize (that is numpy-open-addressing-hash's job). `capacity`/
    `load_factor_*` below are reported against the SAME nominal sweep
    target as the open-addressing impl for comparability; `memory_overhead`
    uses `sys.getsizeof`, an approximation disclosed via
    `memory_overhead_note`. Note this baseline is not a fully independent
    check of reference_hash() above, which is also dict-based -- see that
    function's docstring.
    """

    name = "python-dict-hash"
    platform = "cpu"

    def __init__(self, precision: str = "int64"):
        self.precision = precision

    def prepare(self, w: Primitive, params: dict):
        table: dict[int, int] = {}
        for k, v in zip(w.keys.tolist(), w.values.tolist()):
            table[k] = v
        key_bytes, value_bytes = 8, 8
        approx_bytes = sys.getsizeof(table)
        params["load_factor_target"] = w.target_load_factor
        params["capacity"] = w.capacity
        params["load_factor_actual"] = w.num_keys / w.capacity
        params["insert_ratio"] = w.insert_ratio
        params["key_distribution"] = w.key_distribution
        params["hash_function"] = "CPython builtin dict (internal SipHash + open addressing, opaque)"
        params["memory_overhead"] = approx_bytes / (w.num_keys * (key_bytes + value_bytes)) - 1
        params["memory_overhead_note"] = (
            "sys.getsizeof(dict) only; CPython dict internals are opaque, "
            "unlike numpy-open-addressing-hash's exact table_bytes")
        params["table_bytes"] = approx_bytes
        return dict(table=table, trace_keys=w.keys[w.trace_pool_idx],
                    trace_is_write=w.trace_is_write, trace_write_value=w.trace_write_value)

    def run(self, h):
        table = h["table"]
        keys, is_write, wval = h["trace_keys"], h["trace_is_write"], h["trace_write_value"]
        out = np.empty(keys.shape[0], dtype=np.int64)
        for i in range(keys.shape[0]):
            k = int(keys[i])
            if is_write[i]:
                v = int(wval[i])
                table[k] = v
                out[i] = v
            else:
                out[i] = table.get(k, -1)
        return out

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


# =============================================================== registry
REFERENCES = {
    "scan-reduction": reference_scan_reduction,
    "topk-selection": reference_topk,
    "hash-table": reference_hash,
}

CORRECTNESS_MODE = {
    "scan-reduction": "max_scaled_err",
    "topk-selection": "exact",
    "hash-table": "exact",
}

DEFAULT_PRECISION = {
    "scan-reduction": "fp32",
    "topk-selection": "fp32",
    "hash-table": "int64",
}

REFERENCE_NAME = {
    "scan-reduction": "independent fp64 CPU sum/cumsum (reassociation-tolerant gate; see module docstring)",
    "topk-selection": "independent fp64 CPU full-sort ground truth (sorted value multiset)",
    "hash-table": "independent Python dict replay of the identical op trace",
}

CPU_IMPLS = {
    "scan-reduction": {"numpy-reduction": NumpyReduction, "numpy-scan": NumpyScan},
    "topk-selection": {"numpy-topk": NumpyTopK},
    "hash-table": {"numpy-open-addressing-hash": NumpyOpenAddressingHash,
                   "python-dict-hash": PythonDictHash},
}


def cuda_impls():
    """
    No GPU implementation shipped: all three kernels here are CPU/numpy/
    stdlib-only per this module's scope (see DOMAIN_GUIDE.md's "cuda_impls()
    may return {} with a comment" allowance). A real GPU pass would need
    device-resident radix-select/warp-histogram/atomic-hash code this repo's
    csrc/ does not currently have for these three kernels.
    """
    return {}
