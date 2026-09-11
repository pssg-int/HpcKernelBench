"""
Compression: error-bounded lossy compression and bit-exact lossless
compression of scientific floating-point fields.

This domain's headline metric is a PAIR, not a scalar: both specs are explicit
that compression ratio and throughput must be reported jointly, never one
alone (lossless-compression/spec.yaml summary: "reporting compression ratio
and compress/decompress throughput (GB/s) jointly, never one alone"). The cost
rule below drives the scalar `throughput` the harness times (GB/s, uncompressed
bytes / seconds); the achieved compression ratio is not a "cost" in the
harness's flop/byte sense (it depends on what the compressor produced, not on
the workload alone), so it rides along on the `params` dict each impl mutates
during prepare()/free() -- the harness echoes `params` straight into the
result record (see harness.RunResult.to_dict), which is what lands it next to
the throughput number rather than requiring a second metric channel.

Both specs are equally explicit that compress and decompress are separate
measurements ("never averaged together" / "reported jointly ... never one
alone" -- meaning never collapsed into a single number, not that one may be
skipped). Each implementation below is therefore direction-fixed: a
"-compress" and a "-decompress" registration share one codec class, and
`prepare()` stamps `params["direction"]` so the direction is explicit in every
result record, never inferred from the implementation name alone.

Throughput numerator convention (both specs, cross-checked against multiple
papers' own worked examples in their evidence blocks -- lsCOMP for lossy,
MANS/GPULZ/the Huffman-decoder paper for lossless): ALWAYS original
(uncompressed) byte count, for both compress and decompress. See _cost_bytes.

Correctness gates differ sharply by kernel and neither is the generic
"max-scaled-error-vs-reference" gate `harness.check_correctness` defaults to:

  * lossless-compression: CORRECTNESS_MODE = "exact" -- bit-exact roundtrip,
    non-negotiable per spec ("correctness requires D' == D bit-for-bit for
    every byte, not a statistical or average tolerance").
  * lossy-compression: CORRECTNESS_MODE = "max_abs_err" -- but the bound is
    the error bound eb itself ("max_i |D_i - D'_i| <= eb"), and eb is a
    workload/param, not a number the spec text carries literally (its
    correctness sentence reads "<= eb" symbolically, so spec.py's tolerance
    parser correctly leaves `variant.tolerance` unparsed rather than
    inventing a number -- see spec._tolerance's docstring). Each lossy
    workload therefore carries its own resolved absolute bound as
    `correctness_tolerance`; the runner picks it up and feeds it to
    harness.run_variant as `tolerance_override`, which exists in harness.py
    for exactly this class of kernel (see its docstring there).

Workloads are synthetic scientific-data-like fields, NOT downloaded SDRBench
data (that corpus is 10s of GB and out of scope for a CPU-stdlib benchmark
module). Three representative field kinds cover the shapes SDRBench's own
named fields fall into: a smooth low-frequency field (cosmology/climate
density-style), a turbulent/noisy field (combustion/turbulence-style), and a
multiscale field mixing both (Miranda/S3D-style). Every workload's describe()
sets `synthetic: True` so a run built from one is never mistaken for a
spec-conforming pull of the real corpus -- the runner already treats
non-conforming inputs specially; this module just has to say so honestly.
"""

from __future__ import annotations

import lzma
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .. import workload
from ..harness import Timer

KERNELS = ["lossy-compression", "lossless-compression"]
PLANNED: list[str] = []

SEED = 20260806                  # matches sparse.py's synthetic-data seed convention
SMOKE_SHAPE = (24, 24, 24)       # ~13.8k elements: seconds, not minutes, on a login node
FULL_SHAPE = (64, 64, 64)        # still login-node-friendly; a stand-in for a real 3D field
DEFAULT_REL_EB = 1e-3            # mid-point of the spec's rel sweep {1e-2, 1e-3, 1e-4}


# ------------------------------------------------------------------ workload
@dataclass
class Field:
    """
    A scientific-data-like float field -- the workload for both compression
    kernels. `error_bound_*` is only meaningful for lossy-compression (a
    lossless codec ignores it); `correctness_tolerance` is the resolved
    ABSOLUTE bound (already converted out of rel-mode's fraction-of-range
    form) that the harness's correctness gate is checked against -- see the
    module docstring for why this lives on the workload rather than the spec.
    """

    name: str
    data: np.ndarray                          # flat or n-D, row-major (C order)
    kind: str                                  # 'smooth' | 'turbulent' | 'multiscale'
    seed: int
    error_bound_mode: str | None = None        # 'abs' | 'rel'
    error_bound_value: float | None = None     # the swept fraction (rel) or value (abs)
    correctness_tolerance: float | None = None  # resolved absolute eb

    def describe(self) -> dict:
        d = self.data
        desc = {
            "name": self.name,
            "source": "synthetic",
            "synthetic": True,      # never mistaken for a real SDRBench pull
            "kind": self.kind,
            "shape": list(d.shape),
            "n_elements": int(d.size),
            "dtype": str(d.dtype),
            "seed": self.seed,
            "bytes_uncompressed": int(d.nbytes),
            "value_stats": {
                "min": float(d.min()), "max": float(d.max()),
                "mean": float(d.mean()), "std": float(d.std()),
            },
        }
        if self.error_bound_mode is not None:
            desc["error_bound_mode"] = self.error_bound_mode
            desc["error_bound_value"] = self.error_bound_value
            desc["error_bound_abs"] = self.correctness_tolerance
        return desc


def _smooth_field(shape, seed: int, dtype=np.float32) -> np.ndarray:
    """Sum of a few Gaussian bumps: the coarse-scale shape of SDRBench's
    cosmology/climate density fields (HACC, NYX, CESM-ATM)."""
    rng = np.random.default_rng(seed)
    axes = [np.linspace(-1.0, 1.0, n) for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    field = np.zeros(shape, dtype=np.float64)
    for _ in range(4):
        center = rng.uniform(-0.6, 0.6, size=len(shape))
        sigma = rng.uniform(0.2, 0.5)
        amp = rng.uniform(0.5, 2.0)
        r2 = sum((g - c) ** 2 for g, c in zip(grids, center))
        field += amp * np.exp(-r2 / (2 * sigma ** 2))
    return field.astype(dtype)


def _turbulent_field(shape, seed: int, dtype=np.float32) -> np.ndarray:
    """Box-filtered white noise: high-entropy content in the style of
    combustion/turbulence fields (S3D, Miranda) -- stdlib/numpy only, no FFT
    dependency needed for a "noisy" stand-in."""
    rng = np.random.default_rng(seed)
    field = rng.standard_normal(shape).astype(np.float64)
    kernel = np.ones(3) / 3.0
    for axis in range(field.ndim):
        field = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis, field)
    return field.astype(dtype)


def _multiscale_field(shape, seed: int, dtype=np.float32) -> np.ndarray:
    """Smooth base + a 10%-amplitude turbulent perturbation: the mixed
    macro-structure-plus-fine-fluctuation shape real multiscale SDRBench
    fields (Miranda/S3D) have."""
    base = _smooth_field(shape, seed, np.float64)
    fine = _turbulent_field(shape, seed + 1, np.float64)
    return (base + 0.1 * fine).astype(dtype)


_GENERATORS = {
    "smooth": _smooth_field,
    "turbulent": _turbulent_field,
    "multiscale": _multiscale_field,
}


def _resolve_eb(data: np.ndarray, mode: str, value: float) -> float:
    """Turn a rel/abs error-bound spec into the absolute bound the gate uses."""
    if mode == "abs":
        return float(value)
    rng = float(data.max()) - float(data.min())
    return float(value) * rng if rng > 0 else float(value)


def _make_field(kind: str, shape, seed: int, name: str,
                 eb_mode: str = "rel", eb_value: float = DEFAULT_REL_EB) -> Field:
    data = _GENERATORS[kind](shape, seed)
    tol = _resolve_eb(data, eb_mode, eb_value)
    return Field(name=name, data=data, kind=kind, seed=seed,
                 error_bound_mode=eb_mode, error_bound_value=eb_value,
                 correctness_tolerance=tol)


def smoke_workloads() -> list[Field]:
    """
    Small, synthetic, runs anywhere in seconds. NOT spec-conforming (see
    Field.describe -- `synthetic: True` always). Every field carries a
    default rel error bound too: lossless-compression ignores it (mode is
    exact), lossy-compression's smoke gate uses it.
    """
    return [
        _make_field("smooth", SMOKE_SHAPE, SEED, "smoke-smooth-3d"),
        _make_field("turbulent", SMOKE_SHAPE, SEED + 1, "smoke-turbulent-3d"),
        _make_field("multiscale", SMOKE_SHAPE, SEED + 2, "smoke-multiscale-3d"),
    ]


def load_workload(name: str) -> Field:
    """
    Build a synthetic surrogate for a named workload -- typically one of the
    strings a spec variant's `recommended_subset` uses (e.g. "NYX/baryon_
    density (cosmology, 3D 512^3, fp32/fp64)"). Real SDRBench downloads are a
    multi-GB dependency this stdlib-only module deliberately avoids; instead
    this parses whatever dimensionality/dtype/field-kind hints are present in
    the name and generates a representative field at FULL_SHAPE, always
    reporting `synthetic: True` (see Field.describe) so it can never be
    mistaken for a real pull of the named corpus.
    """
    text = name.lower()
    dtype = np.float64 if "fp64" in text else np.float32
    kind = "multiscale"
    if any(k in text for k in ("hacc", "nyx", "cesm", "letkf")):
        kind = "smooth"
    elif any(k in text for k in ("s3d", "miranda", "isabel", "exaalt")):
        kind = "turbulent"
    field = _make_field(kind, FULL_SHAPE, SEED, f"synthetic-surrogate::{name}")
    if dtype != field.data.dtype:
        data = field.data.astype(dtype)
        tol = _resolve_eb(data, field.error_bound_mode, field.error_bound_value)
        field = Field(field.name, data, field.kind, field.seed,
                      field.error_bound_mode, field.error_bound_value, tol)
    return field


# ----------------------------------------------------------------- cost rule
def _cost_bytes(w: Field, params: dict) -> tuple[int, int]:
    """
    work = ORIGINAL (uncompressed) byte count -- both specs are explicit this
    is the throughput numerator for BOTH directions (lossy spec: "numerator is
    ALWAYS original (uncompressed) byte count, confirmed community convention
    (cross-checked from lsCOMP's worked example)"; lossless spec: "numerator
    is ALWAYS original (uncompressed) byte count -- confirmed community
    convention (cross-checked from MANS's, GPULZ's, the Huffman-decoder's, and
    lsCOMP's own throughput formulas, all consistent)"). Registered with unit
    "GB/s", so metrics.throughput's generic `count/seconds/1e9` division reads
    as GB/s directly.

    `bytes` is the compulsory-traffic lower bound: one read of the input side
    plus one write of the output side, for whichever direction ran. The
    achieved compressed size isn't known until an implementation has actually
    compressed something, so before that this falls back to 2x the
    uncompressed size (the tightest bound available with no compressor run
    yet); once an impl has reported `achieved_compressed_bytes` into `params`
    (see the codec classes below), that measured figure is used instead.
    """
    orig = int(w.data.nbytes)
    compressed = int(params.get("achieved_compressed_bytes", orig))
    return orig, orig + compressed


workload.register_cost("lossy-compression", _cost_bytes, "GB/s")
workload.register_cost("lossless-compression", _cost_bytes, "GB/s")


# ----------------------------------------------------------------- reference
def reference_compression(w: Field, params: dict):
    """The reference IS the original data -- both kernels' correctness gate
    is against the input field itself, not a derived quantity."""
    return w.data.astype(np.float64), None


REFERENCES = {
    "lossy-compression": reference_compression,
    "lossless-compression": reference_compression,
}

REFERENCE_NAME = {
    "lossy-compression": "original synthetic field (fp64-cast)",
    "lossless-compression": "original synthetic field (bit-exact)",
}

# lossless: hard bit-exact roundtrip, no tolerance (see module docstring).
# lossy: max|x-x'| <= eb literally -- "max_abs_err" computes exactly that;
#   the numeric eb comes from Field.correctness_tolerance via the runner's
#   tolerance_override hook, not from spec-parsed text (see harness.py).
CORRECTNESS_MODE = {
    "lossy-compression": "max_abs_err",
    "lossless-compression": "exact",
}

DEFAULT_PRECISION = {"lossy-compression": "fp32", "lossless-compression": "fp32"}

# no dense-operand dimension to sweep for either kernel
DIM_KEY: dict[str, str] = {}


# --------------------------------------------------------------------- impls
class _LosslessCodec:
    """
    Shared plumbing for the two stdlib lossless codecs. Operates directly on
    the raw bytes of the workload's float array. `direction` is fixed per
    registered subclass so it is always explicit, never inferred; `prepare()`
    stamps it into `params` so it lands in the result record.

    Preprocessing discipline: for "compress", the timed call IS the codec's
    compress(); for "decompress" the compressed input has to exist before
    decompress() can be timed, so producing it is a one-shot prepare() cost,
    excluded from the timed region, exactly like the format-conversion
    preprocessing sparse.py's impls hoist out of run().
    """

    platform = "cpu"
    direction = "compress"        # overridden per concrete registration below

    def __init__(self, precision: str = "fp32"):
        self.precision = precision

    # subclasses fix the actual codec
    @staticmethod
    def _compress(raw: bytes) -> bytes:
        raise NotImplementedError

    @staticmethod
    def _decompress(blob: bytes) -> bytes:
        raise NotImplementedError

    def prepare(self, w: Field, params: dict):
        self._dtype = w.data.dtype
        self._shape = w.data.shape
        params["direction"] = self.direction
        raw = np.ascontiguousarray(w.data).tobytes()
        h = {"raw": raw, "params": params}
        if self.direction == "decompress":
            compressed = self._compress(raw)          # one-shot, untimed
            h["compressed"] = compressed
            self._record(params, len(raw), len(compressed))
        return h

    def run(self, h):
        if self.direction == "compress":
            out = self._compress(h["raw"])
            h["last_out"] = out                        # for free()'s bookkeeping
            return out
        return self._decompress(h["compressed"])

    def to_host(self, out) -> np.ndarray:
        # a "compress" call's correctness is judged by round-tripping its own
        # output -- it is only correct if what it produced decompresses back
        # to the original bytes
        raw = out if self.direction == "decompress" else self._decompress(out)
        arr = np.frombuffer(raw, dtype=self._dtype).reshape(self._shape)
        return arr.astype(np.float64)

    def timer(self) -> Timer:
        return Timer()

    def free(self, h) -> None:
        # ratio bookkeeping happens here (after all timed reps, before the
        # harness copies `params` for the cost model) so it never inflates
        # the timed region for the "compress" direction
        if self.direction == "compress" and "last_out" in h:
            self._record(h["params"], len(h["raw"]), len(h["last_out"]))
        h.clear()

    @staticmethod
    def _record(params: dict, orig_bytes: int, compressed_bytes: int) -> None:
        params["achieved_compressed_bytes"] = compressed_bytes
        params["compression_ratio"] = (
            orig_bytes / compressed_bytes if compressed_bytes else float("inf"))


class _ZlibCodec(_LosslessCodec):
    @staticmethod
    def _compress(raw: bytes) -> bytes:
        return zlib.compress(raw, level=6)   # zlib's own default level

    @staticmethod
    def _decompress(blob: bytes) -> bytes:
        return zlib.decompress(blob)


class _LzmaCodec(_LosslessCodec):
    @staticmethod
    def _compress(raw: bytes) -> bytes:
        return lzma.compress(raw, preset=6)  # lzma's own default preset

    @staticmethod
    def _decompress(blob: bytes) -> bytes:
        return lzma.decompress(blob)


class ZlibCompress(_ZlibCodec):
    name = "zlib-lossless-compress"
    direction = "compress"


class ZlibDecompress(_ZlibCodec):
    name = "zlib-lossless-decompress"
    direction = "decompress"


class LzmaCompress(_LzmaCodec):
    name = "lzma-lossless-compress"
    direction = "compress"


class LzmaDecompress(_LzmaCodec):
    name = "lzma-lossless-decompress"
    direction = "decompress"


# ------------------------------------------------------------- lossy codec
# Uniform quantization to the absolute error bound (bin width = 2*eb, so the
# nearest-bin reconstruction error is always <= eb by construction) followed
# by zlib of the quantized integers. This two-stage design -- linear-scaling
# quantizer + general-purpose entropy/back-end coder -- is a legitimate
# baseline in the SZ family's own lineage (SZ itself is curve-fit-predictor +
# quantizer + Huffman/back-end coder; this is the same shape minus the
# predictor), not a strawman.
_HEADER = struct.Struct("<dd")     # (x_min, step) prepended to the zlib payload
_QDTYPE = np.int32


def _quantize(data: np.ndarray, eb_abs: float):
    x_min = float(data.min())
    step = max(2.0 * eb_abs, np.finfo(np.float64).tiny)
    q = np.round((data.astype(np.float64) - x_min) / step).astype(_QDTYPE)
    return q, x_min, step


def _dequantize(q: np.ndarray, x_min: float, step: float, dtype) -> np.ndarray:
    return (x_min + q.astype(np.float64) * step).astype(dtype)


def _compress_lossy(data: np.ndarray, eb_abs: float) -> bytes:
    q, x_min, step = _quantize(data, eb_abs)
    payload = zlib.compress(np.ascontiguousarray(q).tobytes(), level=6)
    return _HEADER.pack(x_min, step) + payload


def _decompress_lossy(blob: bytes, shape, dtype) -> np.ndarray:
    x_min, step = _HEADER.unpack(blob[:_HEADER.size])
    q = np.frombuffer(zlib.decompress(blob[_HEADER.size:]), dtype=_QDTYPE).reshape(shape)
    return _dequantize(q, x_min, step, dtype)


class _QuantizeZlibCodec:
    """quantize-to-eb + zlib. `direction` fixed per concrete registration."""

    platform = "cpu"
    direction = "compress"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision

    def prepare(self, w: Field, params: dict):
        eb = w.correctness_tolerance
        if eb is None:
            raise ValueError(
                f"{w.name}: no error bound attached (correctness_tolerance is "
                "None) -- quantize-zlib-lossy needs one to size its bins")
        self._dtype = w.data.dtype
        self._shape = w.data.shape
        self._eb = eb
        params["direction"] = self.direction
        params["error_bound_mode"] = w.error_bound_mode
        params["error_bound_value"] = w.error_bound_value
        params["error_bound_abs"] = eb
        h = {"data": w.data, "params": params}
        if self.direction == "decompress":
            blob = _compress_lossy(w.data, eb)         # one-shot, untimed
            h["blob"] = blob
        return h

    def run(self, h):
        if self.direction == "compress":
            out = _compress_lossy(h["data"], self._eb)
            h["last_out"] = out
            return out
        return _decompress_lossy(h["blob"], self._shape, self._dtype)

    def to_host(self, out) -> np.ndarray:
        arr = out if self.direction == "decompress" else \
            _decompress_lossy(out, self._shape, self._dtype)
        return arr.astype(np.float64)

    def timer(self) -> Timer:
        return Timer()

    def free(self, h) -> None:
        # report ratio, achieved max error and PSNR once, after timing, from
        # whichever compressed blob this run actually produced/used
        blob = h.get("last_out") if self.direction == "compress" else h.get("blob")
        if blob is not None:
            orig = h["data"].astype(np.float64)
            recon = _decompress_lossy(blob, self._shape, self._dtype).astype(np.float64)
            diff = recon - orig
            mse = float(np.mean(diff * diff))
            value_range = float(orig.max() - orig.min())
            psnr = float("inf") if mse == 0.0 else \
                20.0 * np.log10(value_range) - 10.0 * np.log10(mse)
            params = h["params"]
            params["achieved_compressed_bytes"] = len(blob)
            params["compression_ratio"] = orig.nbytes / len(blob) if len(blob) else float("inf")
            params["achieved_max_abs_error"] = float(np.max(np.abs(diff)))
            params["psnr_db"] = psnr
        h.clear()


class QuantizeZlibCompress(_QuantizeZlibCodec):
    name = "quantize-zlib-lossy-compress"
    direction = "compress"


class QuantizeZlibDecompress(_QuantizeZlibCodec):
    name = "quantize-zlib-lossy-decompress"
    direction = "decompress"


CPU_IMPLS = {
    "lossless-compression": {
        "zlib-lossless-compress": ZlibCompress,
        "zlib-lossless-decompress": ZlibDecompress,
        "lzma-lossless-compress": LzmaCompress,
        "lzma-lossless-decompress": LzmaDecompress,
    },
    "lossy-compression": {
        "quantize-zlib-lossy-compress": QuantizeZlibCompress,
        "quantize-zlib-lossy-decompress": QuantizeZlibDecompress,
    },
}


def cuda_impls():
    # no GPU implementations yet; the two GPU-dual-scope variants in both
    # specs are PLANNED, not implemented here
    return {}
