"""
Spectral transforms: fft, ntt.

Two kernels that share almost nothing structurally (dense complex/real arrays
vs. RNS-limb integer polynomials) but share the family trait that gives this
domain its name: the throughput number is preceded by a one-shot setup step
that specs across this whole project keep calling out as its own "amortized
preprocessing" phase --

  fft: plan creation (a cuFFT/FFTW plan object, or TurboFFT/TurboFNO's
       build-time kernel-codegen+compile step) -- see benchspecs/fft/spec.yaml
       `fft-plan-creation` variant. Both CPU backends here approximate this
       with the first-call pocketfft-plan-cache warm-up, done in prepare()
       and timed once, exactly as the harness's preprocessing split expects.
  ntt: root-of-unity twiddle-table precomputation -- see
       benchspecs/ntt/spec.yaml `ntt-kernel-isolated` protocol.timing_scope
       ("if an implementation instead precomputes/loads a twiddle table,
       that load is reported separately, once, analogous to preprocessing").

The two kernels also share nothing on the CORRECTNESS side, and that split is
the more important one to get right:
  fft: floating point, gated with `max_scaled_err` like every other domain in
       this project (see harness.check_correctness's docstring).
  ntt: EXACT modular integer arithmetic. The spec is emphatic that an epsilon
       gate here would hide a real NTT bug inside the FHE noise budget it is
       normally embedded in (benchspecs/ntt/spec.yaml notes_on_fairness) --
       CORRECTNESS_MODE="exact", no tolerance, see NumpyNTT's docstring for
       exactly how the spec's two required checks (round-trip identity AND
       an independent-method cross-check) are both realized.

WORKLOADS   — FFTWorkload (dense array shape/kind/direction/layout) and
              NTTWorkload (degree, RNS moduli+roots); see load_workload()
COST        — flop/byte rules registered with the workload registry
REFERENCE   — fp64 numpy.fft for fft; the original input (round trip) plus a
              small-N independent schoolbook cross-check for ntt
IMPLS       — CPU (numpy/scipy) and CUDA (torch.fft, wired but not executed
              here -- login node, shared with other users) implementations
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .. import workload
from ..harness import Timer

KERNELS = ["fft", "ntt"]
PLANNED: list[str] = []


# =========================================================================
# shared small helpers
# =========================================================================
def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# =========================================================================
# FFT
# =========================================================================
@dataclass
class FFTWorkload:
    """
    A dense-array FFT problem: [batch, *dims] complex (C2C) or real (R2C
    forward / C2R inverse). Structural fields only (shape/kind/direction/
    layout) -- the actual operand is regenerated deterministically from
    `seed` by both the implementation under test and the reference (the same
    convention sparse.py's dense operands use), so describe() stays a small,
    reproducible record rather than embedding the array itself.
    """

    name: str
    dims: tuple[int, ...]          # spatial size per axis; len in {1, 2, 3}
    batch: int
    real_input: bool               # True: R2C forward / C2R inverse. False: C2C.
    direction: str                 # "forward" | "inverse" -- what run() times
    layout: str                    # "out-of-place" | "in-place"
    seed: int = 42

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def n_total(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    def describe(self) -> dict:
        return {
            "name": self.name, "dims": list(self.dims), "ndim": self.ndim,
            "n_total": self.n_total, "batch": self.batch,
            "kind": "r2c" if self.real_input else "c2c",
            "direction": self.direction, "layout": self.layout,
            "seed": self.seed,
            "mixed_radix": not _is_power_of_two(self.n_total),
        }


def _realify(arr) -> np.ndarray:
    """
    harness.check_correctness casts both `out` and `ref` to float64 before
    comparing; a naive complex->float64 cast silently discards the imaginary
    part (numpy raises/warns and truncates). Stack real/imag as a trailing
    axis instead so no information is lost through that cast. Used by every
    FFT impl's to_host() and by reference_fft(), so both sides go through the
    identical encoding and the gate sees the full complex error.
    """
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return np.stack([arr.real, arr.imag], axis=-1).astype(np.float64)
    return arr.astype(np.float64)


def _fft_operand(workload_: FFTWorkload, dtype_real, dtype_complex, seed: int, rfftn):
    """
    Build the array `run()` transforms, matching the workload's kind/
    direction: real U(-1,1) for a forward transform (matches
    TurboFFT/FlashFFTStencil's own random-fill convention, per
    benchspecs/fft/spec.yaml's `dense_operand` field -- not a cherry-picked
    "nice" input); for an inverse transform, C2C uses arbitrary random
    complex data (ifft is defined for any complex input, no symmetry needed)
    and R2C's C2R inverse needs a genuinely Hermitian-packed input, built by
    forward-transforming a fresh real signal with the SAME rfftn used by the
    caller (so numpy/scipy/torch each build a self-consistent operand with
    their own backend). `rfftn(x, axes)` is the backend-specific R2C forward
    call used only for that construction.
    """
    rng = np.random.default_rng(seed)
    shape = (workload_.batch,) + tuple(workload_.dims)
    axes = tuple(range(-workload_.ndim, 0))
    if workload_.real_input and workload_.direction == "forward":
        return rng.uniform(-1.0, 1.0, size=shape).astype(dtype_real)
    if workload_.real_input:  # inverse (C2R)
        x = rng.uniform(-1.0, 1.0, size=shape).astype(dtype_real)
        return rfftn(x, axes).astype(dtype_complex)
    re = rng.uniform(-1.0, 1.0, size=shape)
    im = rng.uniform(-1.0, 1.0, size=shape)
    return (re + 1j * im).astype(dtype_complex)


class NumpyFFT:
    """
    numpy.fft backend (pocketfft since numpy>=1.17). pocketfft keeps an
    internal per-shape setup cache; the first transform of a new shape pays
    that cost. prepare() performs one throwaway transform to warm this cache
    -- this module's realization of the "plan creation" the spec's
    `fft-plan-creation` variant asks to be measured once and excluded from
    run()'s per-call time -- and the harness already times prepare()
    separately (RunResult.preprocessing_ms).

    Layout: numpy.fft never exposes true in-place execution (every call
    allocates a fresh array). For layout="in-place" AND a shape/dtype-
    preserving transform (C2C only -- R2C/C2R changes both shape and dtype
    between real and complex, so there is nothing to copy back into), this
    impl copies the result back into the input buffer so at least the *data
    movement* the in-place byte-cost convention assumes is honest; the
    *allocation* saving a real in-place FFT gives is NOT realized here. Use
    scipy-fft's overwrite_x for a real in-place path, including for R2C/C2R.
    """

    name = "numpy-fft"
    platform = "cpu"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.real_dtype = np.float32 if precision == "fp32" else np.float64
        self.complex_dtype = np.complex64 if precision == "fp32" else np.complex128

    def _transform(self, w: FFTWorkload, x: np.ndarray, axes: tuple):
        if w.real_input:
            if w.direction == "forward":
                return np.fft.rfftn(x, axes=axes)
            return np.fft.irfftn(x, s=w.dims, axes=axes)
        return np.fft.fftn(x, axes=axes) if w.direction == "forward" \
            else np.fft.ifftn(x, axes=axes)

    def prepare(self, w: FFTWorkload, params: dict):
        axes = tuple(range(-w.ndim, 0))
        x = _fft_operand(w, self.real_dtype, self.complex_dtype,
                          params.get("seed", w.seed),
                          lambda a, ax: np.fft.rfftn(a, axes=ax))
        _ = self._transform(w, x, axes)   # warm pocketfft's plan/twiddle cache
        return {"x": x, "axes": axes, "workload": w}

    def run(self, h: dict):
        w = h["workload"]
        out = self._transform(w, h["x"], h["axes"])
        if w.layout == "in-place" and out.shape == h["x"].shape and out.dtype == h["x"].dtype:
            h["x"][...] = out
            out = h["x"]
        return out

    def to_host(self, out):
        return _realify(out)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


class ScipyFFT:
    """
    scipy.fft backend. Like numpy.fft, its pocketfft implementation caches
    per-shape setup internally; prepare() warms it the same way, for the
    same "plan creation" reason (see NumpyFFT's docstring).

    Unlike numpy.fft, scipy.fft exposes a genuine in-place-capable path:
    `overwrite_x=True` lets pocketfft reuse the input buffer as scratch (and,
    for C2C, as output) instead of allocating fresh memory -- used here
    whenever workload.layout == "in-place", for BOTH C2C and R2C/C2R (scipy
    does not require the output to alias the input to honor overwrite_x, so
    this works uniformly where NumpyFFT's copy-back approximation cannot).
    `workers` exposes scipy.fft's intra-transform threading; kept at 1 here
    so CPU-vs-CPU comparisons across implementations stay apples-to-apples
    on a shared login node.
    """

    name = "scipy-fft"
    platform = "cpu"

    def __init__(self, precision: str = "fp32", workers: int = 1):
        import scipy.fft
        self._sfft = scipy.fft
        self.precision = precision
        self.workers = workers
        self.real_dtype = np.float32 if precision == "fp32" else np.float64
        self.complex_dtype = np.complex64 if precision == "fp32" else np.complex128

    def _transform(self, w: FFTWorkload, x: np.ndarray, axes: tuple, overwrite: bool):
        sf = self._sfft
        if w.real_input:
            if w.direction == "forward":
                return sf.rfftn(x, axes=axes, workers=self.workers, overwrite_x=overwrite)
            return sf.irfftn(x, s=w.dims, axes=axes, workers=self.workers, overwrite_x=overwrite)
        fn = sf.fftn if w.direction == "forward" else sf.ifftn
        return fn(x, axes=axes, workers=self.workers, overwrite_x=overwrite)

    def prepare(self, w: FFTWorkload, params: dict):
        axes = tuple(range(-w.ndim, 0))
        x = _fft_operand(w, self.real_dtype, self.complex_dtype,
                          params.get("seed", w.seed),
                          lambda a, ax: self._sfft.rfftn(a, axes=ax))
        # warm the plan cache on a throwaway copy so overwrite_x in run()
        # (which may consume the buffer) still gets a pristine operand
        _ = self._transform(w, x.copy(), axes, overwrite=False)
        return {"x": x, "axes": axes, "workload": w}

    def run(self, h: dict):
        w = h["workload"]
        # NOTE: with overwrite_x=True the input buffer may be mutated, so
        # repeated warmup/timed calls each transform whatever the PREVIOUS
        # call left behind, not the original operand. FFT cost is
        # data-independent (shape/dtype only), so this does not bias timing;
        # only the FIRST call (used for the correctness gate, before any
        # warmup) is guaranteed to see the pristine, seed-generated operand.
        return self._transform(w, h["x"], h["axes"], overwrite=(w.layout == "in-place"))

    def to_host(self, out):
        return _realify(out)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


def reference_fft(w: FFTWorkload, params: dict):
    """
    fp64 reference via numpy.fft (complex128 / float64), independent of
    whatever precision the implementation under test runs at -- same operand
    construction as the implementations, regenerated from the same seed
    (matching sparse.py's cpu_ref reference-regeneration convention rather
    than storing the array on the workload).

    `scale`: harness.check_correctness's max_scaled_err gate needs a
    magnitude denominator. FFT has no natural per-output-element
    "componentwise product" scale the way SpMM's |A|@|B| does -- each output
    bin is a linear combination of EVERY input sample, not a local product --
    so this returns a single GLOBAL scale instead: sqrt(N_total) * max(|x|).
    This is the standard rough bound for FFT rounding-error growth (Higham,
    "Accuracy and Stability of Numerical Algorithms": the computed and exact
    transforms differ by O(log2(N) * eps * ||x||)); sqrt(N)*max(|x|) is a
    convenient, slightly loose but always-safe stand-in for ||x|| that avoids
    committing to which norm a given paper implicitly assumes.
    """
    rng = np.random.default_rng(params.get("seed", w.seed))
    shape = (w.batch,) + tuple(w.dims)
    axes = tuple(range(-w.ndim, 0))
    if w.real_input and w.direction == "forward":
        x = rng.uniform(-1.0, 1.0, size=shape).astype(np.float64)
        ref = np.fft.rfftn(x, axes=axes)
        mag_src = x
    elif w.real_input:  # inverse (C2R)
        xr = rng.uniform(-1.0, 1.0, size=shape).astype(np.float64)
        freq = np.fft.rfftn(xr, axes=axes)
        ref = np.fft.irfftn(freq, s=w.dims, axes=axes)
        mag_src = freq
    else:
        re = rng.uniform(-1.0, 1.0, size=shape)
        im = rng.uniform(-1.0, 1.0, size=shape)
        x = (re + 1j * im).astype(np.complex128)
        ref = np.fft.fftn(x, axes=axes) if w.direction == "forward" else np.fft.ifftn(x, axes=axes)
        mag_src = x
    scale = math.sqrt(w.n_total) * float(np.abs(mag_src).max())
    return _realify(ref), scale


def _itemsize_complex(precision: str) -> int:
    return 8 if precision == "fp32" else 16   # complex64 vs complex128, bytes/element


def _cost_fft(w: FFTWorkload, params: dict):
    """
    FLOP convention: 5*N_total*log2(N_total)*batch -- the standard radix-2
    Cooley-Tukey operation count, literally TurboFFT's own formula
    (benchspecs/fft/spec.yaml, `fft-1d-batched-kernel.metric.primary`), with
    N_total generalized to product(dims) for the 2D/3D case per
    `fft-ndgrid-e2e.metric.secondary`'s own "N_total = product of grid dims"
    wording. THIS IS A CONVENTION, not a real operation count for every N:
    the spec itself says this formula "does not cleanly apply" to
    mixed-radix N and asks those points be reported as ms-only, N/A for
    GFLOP/s. This cost-rule contract (workload.register_cost) always returns
    a single flop count -- there is no per-point "N/A" channel in
    harness.run_variant -- so the convention is applied uniformly here and
    flagged instead via FFTWorkload.describe()["mixed_radix"], so a reader of
    the result record can tell which GFLOP/s numbers are the "real" ones.

    Byte convention: 2*N_total*batch*itemsize(precision) -- the spec's own
    "GB/s effective bandwidth" formula, halved for the in-place layout per
    the same spec passage (fft-1d-batched-kernel.metric.secondary).
    """
    n = w.n_total
    flops = int(round(5 * n * math.log2(n)) * w.batch) if n > 1 else 0
    itemsize = _itemsize_complex(params.get("precision", "fp32"))
    byts = n * w.batch * itemsize * (1 if w.layout == "in-place" else 2)
    return flops, int(byts)


workload.register_cost("fft", _cost_fft, "GFLOP/s")


# ---- fft smoke / load ----------------------------------------------------
_FFT_SMOKE = [
    FFTWorkload("smoke-1d-pow2-c2c-fwd-oop", dims=(1024,), batch=4,
                real_input=False, direction="forward", layout="out-of-place"),
    # 56 = 2^3 * 7, FlashFFTStencil's own PFA tile size -- the concrete
    # mixed-radix point this track's spec pulls from a surveyed paper
    # (benchspecs/fft/spec.yaml fft-1d-batched-kernel.inputs.suite)
    FFTWorkload("smoke-1d-mixedradix56-c2c-fwd-oop", dims=(56,), batch=8,
                real_input=False, direction="forward", layout="out-of-place"),
    FFTWorkload("smoke-1d-pow2-c2c-fwd-inplace", dims=(512,), batch=1,
                real_input=False, direction="forward", layout="in-place"),
    FFTWorkload("smoke-1d-pow2-c2c-inverse-oop", dims=(1024,), batch=1,
                real_input=False, direction="inverse", layout="out-of-place"),
    FFTWorkload("smoke-1d-r2c-fwd-oop", dims=(1024,), batch=2,
                real_input=True, direction="forward", layout="out-of-place"),
    FFTWorkload("smoke-1d-r2c-inverse-oop", dims=(1024,), batch=1,
                real_input=True, direction="inverse", layout="out-of-place"),
    FFTWorkload("smoke-2d-c2c-fwd-oop", dims=(64, 64), batch=2,
                real_input=False, direction="forward", layout="out-of-place"),
    FFTWorkload("smoke-3d-c2c-fwd-oop", dims=(16, 16, 16), batch=1,
                real_input=False, direction="forward", layout="out-of-place"),
]

_GRID_RE = re.compile(r"^(\d+)\^(\d)$")
_EXPLICIT_DIMS_RE = re.compile(r"^\d+(x\d+)+$", re.I)


def _load_fft_named(name) -> FFTWorkload:
    """
    Best-effort parse of the spec's own recommended_subset entry formats
    (benchspecs/fft/spec.yaml): a bare N for fft-1d-batched-kernel (int, e.g.
    512, 1024, 56, 3071), "BASE^NDIM" for the cubic/square grid entries in
    fft-ndgrid-e2e (e.g. "256^3", "4096^2"), or an explicit "D1xD2x..." shape
    (e.g. "1024x768x768", CLAIRE's CLARITY dataset). Anything else raises
    rather than guessing.
    """
    if isinstance(name, int) or (isinstance(name, str) and name.strip().isdigit()):
        n = int(name)
        return FFTWorkload(f"fft-1d-N{n}", dims=(n,), batch=1,
                            real_input=False, direction="forward", layout="out-of-place")
    text = str(name).strip()
    m = _GRID_RE.match(text)
    if m:
        n, ndim = int(m.group(1)), int(m.group(2))
        return FFTWorkload(f"fft-grid-{text}", dims=(n,) * ndim, batch=1,
                            real_input=True, direction="forward", layout="out-of-place")
    if _EXPLICIT_DIMS_RE.match(text):
        dims = tuple(int(p) for p in text.split("x"))
        return FFTWorkload(f"fft-grid-{text}", dims=dims, batch=1,
                            real_input=True, direction="forward", layout="out-of-place")
    raise ValueError(
        f"cannot parse fft workload name {text!r}; supported forms: an integer "
        "N (1D C2C forward), 'BASE^NDIM' (e.g. '256^3'), or 'D1xD2x...' (e.g. "
        "'1024x768x768'). See benchspecs/fft/spec.yaml recommended_subset "
        "entries this is meant to cover, or call load_workload with a name "
        "not drawn from the spec text.")


# =========================================================================
# NTT
# =========================================================================
_MR_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin. The fixed witness set (2..37) is a known
    deterministic certificate for all n < 3.3*10^24 (Pomerance/Jaeschke),
    comfortably covering every modulus bit-width this module searches
    (b <= 31, see NumpyNTT's scope note)."""
    if n < 2:
        return False
    for p in _MR_WITNESSES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_WITNESSES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _prime_factors(n: int) -> list[int]:
    """Trial-division factorization. q-1 is at most ~2^31 here, so
    sqrt(q-1) < 2^16 -- trial division is milliseconds, no need for
    anything cleverer."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _find_generator(q: int) -> int:
    """Smallest g that generates Z_q^* (q prime): g^((q-1)/p) != 1 mod q for
    every prime factor p of q-1. About half of all residues are generators
    for a typical prime, so this converges in a handful of tries."""
    factors = _prime_factors(q - 1)
    g = 2
    while True:
        if all(pow(g, (q - 1) // p, q) != 1 for p in factors):
            return g
        g += 1


def _ntt_prime_and_root(degree: int, start_at: int) -> tuple[int, int]:
    """
    Deterministic downward search from `start_at` for the largest
    NTT-friendly prime q <= start_at with q = 1 (mod degree) -- guaranteeing
    a primitive `degree`-th root of unity mod q exists -- and returns
    (q, root) with root = generator**((q-1)/degree) mod q, which has order
    exactly `degree` (see module notes below _find_generator).
    """
    step = degree
    cand = start_at - ((start_at - 1) % step)
    while cand > step:
        if _is_prime(cand):
            g = _find_generator(cand)
            root = pow(g, (cand - 1) // degree, cand)
            return cand, root
        cand -= step
    raise RuntimeError(f"no NTT-friendly prime found near {start_at} for degree={degree}")


def _powers(base: int, n: int, q: int) -> np.ndarray:
    """base**0 .. base**(n-1) mod q, as an int64 array. Plain Python-int loop
    (O(n), not O(n^2)) -- cheap even at n=2**17."""
    out = np.empty(n, dtype=np.int64)
    val = 1
    for i in range(n):
        out[i] = val
        val = (val * base) % q
    return out


class _TwiddleTable:
    """
    Precomputed root-of-unity power tables for one (N, q, root) NTT
    configuration -- this kernel's analogue of an FFT "plan". Built once per
    limb in prepare(), reused by every timed run() call; see
    benchspecs/ntt/spec.yaml `ntt-kernel-isolated.protocol.timing_scope`.
    """

    __slots__ = ("q", "fwd", "inv", "n_inv")

    def __init__(self, n: int, q: int, root: int):
        self.q = q
        root_inv = pow(root, q - 2, q)   # q prime => Fermat's-little-theorem inverse
        self.fwd = _powers(root, n, q)
        self.inv = _powers(root_inv, n, q)
        self.n_inv = pow(n, q - 2, q)


def _bit_reverse_perm(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    idx = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def _ntt_transform(a: np.ndarray, twiddles: np.ndarray, q: int) -> np.ndarray:
    """
    One O(N log N) iterative Cooley-Tukey NTT pass (decimation-in-time,
    bit-reversed input), evaluating out[i] = sum_j a[j] * root**(i*j) mod q
    via log2(N) fully-vectorized radix-2 butterfly stages -- the textbook
    iterative-FFT structure (CLRS), transcribed to modular arithmetic.
    `twiddles[k]` must equal `root**k mod q` for whichever root this pass
    should use (`_TwiddleTable.fwd` or `.inv`); calling this with `.inv`
    computes the UNSCALED inverse transform, the N^-1 scaling is applied by
    the caller (`_ntt_round_trip`), matching benchspecs/ntt/spec.yaml's
    literal INTT definition.
    """
    n = a.shape[0]
    rev = _bit_reverse_perm(n)
    a = a[rev].astype(np.int64)
    length = 2
    while length <= n:
        half = length // 2
        step = n // length
        w = twiddles[0:step * half:step]
        blocks = a.reshape(-1, length)
        u = blocks[:, :half]
        v = (blocks[:, half:] * w) % q
        top = (u + v) % q
        bot = (u - v) % q
        a = np.concatenate([top, bot], axis=1).reshape(-1)
        length *= 2
    return a


def _ntt_round_trip(x: np.ndarray, table: _TwiddleTable) -> np.ndarray:
    y = _ntt_transform(x, table.fwd, table.q)
    y = _ntt_transform(y, table.inv, table.q)
    return (y * table.n_inv) % table.q


def _schoolbook_ntt(a: np.ndarray, q: int, root: int) -> np.ndarray:
    """
    O(N^2) reference DFT mod q: out[i] = sum_j a[j] * root**(i*j) mod q,
    using Python's arbitrary-precision ints (NOT numpy int64) so summing up
    to N terms of size < q^2 never overflows regardless of q's bit width.
    This is the "independent method" benchspecs/ntt/spec.yaml's correctness
    section requires the iterative NTT be cross-checked against; only used
    for small N (see NumpyNTT._SCHOOLBOOK_MAX_N), never for the real N=2**16
    workload, where O(N^2) is intractable.
    """
    n = len(a)
    av = [int(v) % q for v in a]
    w = [1] * n
    for k in range(1, n):
        w[k] = (w[k - 1] * root) % q
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        step = i % n
        acc = 0
        idx = 0
        for j in range(n):
            acc += av[j] * w[idx]
            idx += step
            if idx >= n:
                idx -= n
        out[i] = acc % q
    return out


@dataclass
class NTTWorkload:
    """
    An RNS-batched NTT problem: `limbs` independent length-`degree` residue
    vectors, one per RNS prime `moduli[i]` (each a primitive `degree`-th root
    `roots[i]` mod `moduli[i]`). Structural fields only, like FFTWorkload --
    the random input data is regenerated deterministically from `seed` by
    both the implementation and the reference.
    """

    name: str
    degree: int
    limbs: int
    moduli: list = field(default_factory=list)
    roots: list = field(default_factory=list)
    bit_width: int = 0
    seed: int = 42

    def describe(self) -> dict:
        return {
            "name": self.name, "degree": self.degree, "limbs": self.limbs,
            "bit_width": self.bit_width, "moduli": list(self.moduli),
            "seed": self.seed,
        }


def _make_ntt_workload(name: str, degree: int, bit_width: int, limbs: int,
                        seed: int = 42) -> NTTWorkload:
    """
    Deterministically find `limbs` distinct NTT-friendly primes below
    2**bit_width (each = 1 mod degree, walking downward by `degree` per
    benchspecs/ntt/spec.yaml's "q_i = 1 mod 2N" convention -- see
    _ntt_prime_and_root) together with a primitive `degree`-th root of unity
    for each, so re-running this with the same arguments reproduces the same
    problem instance (no RNG involved in the modulus/root search; `seed`
    only drives the random INPUT data, generated later in prepare()/
    reference_ntt()).
    """
    moduli, roots = [], []
    hint = (1 << bit_width) - 1
    for _ in range(limbs):
        q, root = _ntt_prime_and_root(degree, hint)
        moduli.append(q)
        roots.append(root)
        hint = q - degree
    return NTTWorkload(name=name, degree=degree, limbs=limbs, moduli=moduli,
                        roots=roots, bit_width=bit_width, seed=seed)


class NumpyNTT:
    """
    Iterative Cooley-Tukey NTT/INTT over Z_q, one limb (RNS prime) at a time,
    twiddle table precomputed once per limb in prepare() (see
    `_TwiddleTable`). run() times the full round trip INTT(NTT(x)) for every
    limb in one call, matching the ntt-kernel-isolated variant's primary
    claim ("a STANDALONE forward+inverse NTT pair").

    SCOPE: int64 arithmetic requires q**2 to fit comfortably in a signed
    64-bit accumulator, so this implementation supports modulus bit-widths up
    to 31 -- exactly benchspecs/ntt/spec.yaml's own concrete parameter set
    (Cheddar's 31-bit SMR primes, log_degree=16). The spec's 62/64-bit
    baseline-library comparison point is out of scope here (would need
    Python-int or int128 arithmetic); this module only implements the
    "numpy-ntt" CPU reference the task asks for.

    CORRECTNESS -- checked TWICE, for the two things
    benchspecs/ntt/spec.yaml's correctness section is emphatic about ("EXACT
    modular equality... AND NTT_i(x) matches a reference NTT computed via an
    independent method... for at least one N per bit-width class"):

      1. THE HARNESS GATE (harness.check_correctness, CORRECTNESS_MODE=
         "exact"): run()'s round-trip output vs. reference_ntt()'s return
         value, which is simply the original x -- i.e. INTT(NTT(x)) == x mod
         q_i for every limb. This is O(N log N), affordable at ANY N
         (including the real N=2**16 workload), so it runs on every
         --smoke and every real invocation.
      2. AN EXTRA cross-check against the O(N^2) schoolbook reference
         (`_schoolbook_ntt`), performed HERE in prepare() -- excluded from
         timing -- but ONLY when N <= _SCHOOLBOOK_MAX_N, since O(N^2) is
         intractable at N=2**16. This is the stronger, more diagnostic check
         (it would catch a forward/inverse bug pair that happens to cancel
         and pass check #1 anyway) but the spec itself only asks for it "for
         at least one N per bit-width class", not at every N -- exactly
         this module's smoke workloads (small N), not its real ones. When N
         is too large, this step is skipped and a note is attached to the
         handle (surfaced by the caller if it inspects it) rather than
         silently doing nothing.
    """

    name = "numpy-ntt"
    platform = "cpu"
    _SCHOOLBOOK_MAX_N = 2048

    def __init__(self, precision: str = "int64"):
        self.precision = precision   # NTT is exact integer arithmetic; no fp precision axis

    def prepare(self, w: NTTWorkload, params: dict):
        seed = params.get("seed", w.seed)
        tables: list[_TwiddleTable] = []
        x = np.empty((w.limbs, w.degree), dtype=np.int64)
        notes = []
        for l in range(w.limbs):
            q, root = w.moduli[l], w.roots[l]
            table = _TwiddleTable(w.degree, q, root)
            tables.append(table)
            rng = np.random.default_rng(seed + l)
            x[l] = rng.integers(0, q, size=w.degree, dtype=np.int64)
            if w.degree <= self._SCHOOLBOOK_MAX_N:
                fwd = _ntt_transform(x[l], table.fwd, q)
                ref = _schoolbook_ntt(x[l], q, root)
                if not np.array_equal(fwd, ref):
                    raise AssertionError(
                        f"{self.name}: forward NTT disagrees with the independent "
                        f"schoolbook O(N^2) reference for limb {l} (N={w.degree}, "
                        f"q={q}) -- see benchspecs/ntt/spec.yaml correctness "
                        "requirement #2")
            else:
                notes.append(
                    f"limb {l}: N={w.degree} > {self._SCHOOLBOOK_MAX_N}, schoolbook "
                    "O(N^2) cross-check skipped (intractable at this size); only the "
                    "INTT(NTT(x))==x round-trip gate is checked for this limb")
        return {"tables": tables, "x": x, "notes": notes}

    def run(self, h: dict):
        tables, x = h["tables"], h["x"]
        y = np.empty_like(x)
        for l in range(x.shape[0]):
            y[l] = _ntt_round_trip(x[l], tables[l])
        return y

    def to_host(self, out):
        return np.asarray(out)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


def reference_ntt(w: NTTWorkload, params: dict):
    """
    The round-trip reference IS the original input: INTT(NTT(x)) must equal
    x mod q_i exactly for every limb (harness CORRECTNESS_MODE="exact" does
    a bit-exact np.array_equal). Regenerated from the same seed NumpyNTT.
    prepare() uses (per-limb offset `seed + l`, matching that method exactly)
    rather than read off the implementation's handle, so an implementation
    bug that also corrupts its OWN copy of x cannot silently pass. See
    NumpyNTT's docstring for where the spec's second, independent-method
    correctness requirement is additionally checked.
    """
    seed = params.get("seed", w.seed)
    x = np.empty((w.limbs, w.degree), dtype=np.int64)
    for l in range(w.limbs):
        rng = np.random.default_rng(seed + l)
        x[l] = rng.integers(0, w.moduli[l], size=w.degree, dtype=np.int64)
    return x, None


def _cost_ntt(w: NTTWorkload, params: dict):
    """
    The spec's own primary metric is GNTT-elements/s = N*L/time
    (benchspecs/ntt/spec.yaml `ntt-kernel-isolated.metric.primary`) --
    element-rate, and explicitly never GFLOP/s (NTT is exact integer
    arithmetic, not floating-point work). This module registers that same
    element-rate convention under the coarser label the task asks for
    (unit="NTT/s" rather than "GNTT-elements/s"): work_count = 2*N*L,
    doubled from the spec's own N*L because run() times the FULL round trip
    (one NTT call + one INTT call per limb, each touching N elements).

    NOTE ON SCALE: harness.run_variant's throughput field is always computed
    as work_count/seconds/1e9 (metrics.throughput -- shared by every domain,
    written assuming "giga-something" quantities like GFLOP/s or stencil's
    GCUP/s). At the real N=2**16-scale workloads this module's
    `ntt-cheddar-*` named configs use, 2*N*L is itself already in the
    millions-to-hundred-millions range, so this comes out at a sensible
    O(1-10) scale, same as every other domain's numbers. At the tiny N=1024
    smoke sizes it does not (the printed figure rounds to ~0.00) -- that is
    a real property of how small the smoke problem is, not a units bug;
    compare fft's own smoke GFLOP/s numbers, which are similarly small
    because 5*N*log2(N)*batch is small at smoke scale.
    """
    work_count = 2 * w.degree * w.limbs
    itemsize = 8                # int64 residues
    byts = w.degree * w.limbs * itemsize * 2   # read x once, write y once (lower bound)
    return work_count, int(byts)


workload.register_cost("ntt", _cost_ntt, "NTT/s")


# ---- ntt smoke / load -----------------------------------------------------
_NTT_SMOKE = [
    dict(name="smoke-ntt-small", degree=1024, bit_width=17, limbs=2),
    dict(name="smoke-ntt-batched", degree=1024, bit_width=20, limbs=4),
]

# benchspecs/ntt/spec.yaml has no `recommended_subset` list (unlike fft) --
# ntt-kernel-isolated "does not exist as a ready-to-run benchmark" in the
# source artifact, so there is no paper-sourced name list to draw from. These
# are Cheddar's own literal parameter points instead (spec.yaml batch_limbs;
# survey.md's bootparam_40.json: log_degree=16, 31-bit SMR primes, default
# 48-limb set decomposed there as 43 main + 4 terminal + 12 auxiliary primes
# -- this module uses the spec's own "L in {12,24,48}" batch_limbs axis
# directly rather than the 43-main-primes sub-count, since batch_limbs is the
# field the spec's `ntt-kernel-isolated.inputs` actually names).
_NTT_NAMED = {
    "ntt-cheddar-default": dict(degree=2 ** 16, bit_width=31, limbs=48),
    "ntt-cheddar-reduced": dict(degree=2 ** 16, bit_width=31, limbs=24),
    "ntt-cheddar-min": dict(degree=2 ** 16, bit_width=31, limbs=12),
}


def _load_ntt_named(name: str) -> NTTWorkload:
    if name in _NTT_NAMED:
        return _make_ntt_workload(name, **_NTT_NAMED[name])
    raise ValueError(
        f"unknown ntt workload {name!r}; known: {sorted(_NTT_NAMED)} (see the "
        "comment above _NTT_NAMED -- benchspecs/ntt/spec.yaml has no "
        "recommended_subset to draw further names from)")


# =========================================================================
# domain-module contract: KERNELS / PLANNED / smoke / load / references /
# correctness / precision / impls
# =========================================================================
def smoke_workloads(kernel: str | None = None):
    """
    Small, synthetic, runs anywhere in seconds. NOT spec-conforming.

    `kernel` selects which of this domain's two, structurally incompatible
    workload types to build (FFTWorkload vs. NTTWorkload) -- an extension of
    DOMAIN_GUIDE.md's zero-arg contract, needed the moment a domain owns more
    than one kernel whose workloads are not interchangeable objects (sparse.py
    gets away without this because spmv/spmm/sddmm all share one Matrix
    type). Defaults to the fft workloads when `kernel` is omitted so the
    zero-arg call sparse.py-style callers might still make does not crash.
    """
    if kernel == "ntt":
        return [_make_ntt_workload(**cfg) for cfg in _NTT_SMOKE]
    return list(_FFT_SMOKE)


def load_workload(name, kernel: str | None = None):
    """
    Build/fetch a real workload by the name the spec's recommended_subset
    uses (fft), or one of this module's own named Cheddar-parameter configs
    (ntt -- see _NTT_NAMED, benchspecs/ntt/spec.yaml has no
    recommended_subset of its own).

    `kernel` is honored when given, but runner.py's non-smoke matrix-loading
    path calls this with a bare name and no kernel kwarg (unlike
    smoke_workloads, which it inspects for a `kernel` parameter -- see
    runner.py's comment at the smoke_workloads call site), so the primary
    dispatch is NAME-based: a name found in `_NTT_NAMED` is unambiguously an
    ntt workload; anything else is parsed as an fft size/grid spec.
    """
    if kernel == "ntt" or (kernel is None and str(name) in _NTT_NAMED):
        return _load_ntt_named(str(name))
    return _load_fft_named(name)


REFERENCES = {"fft": reference_fft, "ntt": reference_ntt}

# gate mode per kernel (see harness.check_correctness and NumpyNTT's
# docstring for exactly why ntt is "exact" and not scaled/relative)
CORRECTNESS_MODE = {"fft": "max_scaled_err", "ntt": "exact"}

DEFAULT_PRECISION = {"fft": "fp32", "ntt": "int64"}

CPU_IMPLS = {
    "fft": {"numpy-fft": NumpyFFT, "scipy-fft": ScipyFFT},
    "ntt": {"numpy-ntt": NumpyNTT},
}


def cuda_impls():
    """
    torch.fft-based GPU baseline for fft (cuFFT-backed), timed with
    CudaEventTimer per the specs. NOT executed from this module or by the
    task that built it -- this login node is shared, and DOMAIN_GUIDE.md
    is explicit that GPU timing must not run here; `make -C csrc check`-style
    compile-only verification is the applicable bar, and this file has no
    .cu code to compile (torch.fft needs no custom kernel). No CUDA impl is
    offered for ntt: torch has no native NTT primitive to wrap, and writing
    a custom one is out of this module's scope (see NumpyNTT's docstring).
    """
    from ..impls import gpu_cuda as g
    return {"fft": {"torch-fft": g.TorchFFT}}
