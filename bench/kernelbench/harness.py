"""
The measurement harness.

Order of operations is fixed by the specs and is not negotiable per-run:

  1. prepare()      — one-shot format conversion / packing. Timed ONCE, reported
                      separately, never folded into per-call time.
  2. correctness    — compared against an independent reference. Runs BEFORE any
                      timing; a failing gate makes the run invalid and no
                      throughput number is emitted.
  3. warmup         — protocol.warmup iterations, discarded.
  4. measure        — protocol.reps iterations, ONE timestamp pair PER ITERATION
                      (not one pair around the whole loop, which is what most
                      surveyed artifacts do and which cannot expose variance).
  5. statistic      — median primary, with min/max/stdev always reported.

Every constant used is copied into the result record together with where it came
from, so a result can be audited against its spec without rerunning it.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol as TypingProtocol

import numpy as np

from . import metrics, workload
from .spec import Variant


class Implementation(TypingProtocol):
    """What a kernel implementation must provide to be benchmarkable."""

    name: str
    platform: str          # 'cpu' | 'cuda'
    precision: str         # 'fp64' | 'fp32' | ...

    def prepare(self, matrix, params: dict) -> Any: ...
    def run(self, handle: Any) -> Any: ...
    def to_host(self, out: Any) -> np.ndarray: ...
    def timer(self) -> "Timer": ...
    def free(self, handle: Any) -> None: ...


class Timer:
    """Per-iteration timer. Subclassed for device-side timing."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False


@dataclass
class CorrectnessResult:
    passed: bool
    metric: str                 # 'max_scaled_err' | 'max_rel_err' | 'max_abs_err' | 'exact'
    value: float
    tolerance: float | None
    reference: str
    note: str = ""
    # secondary views, always recorded so the gate's choice is auditable
    max_pointwise_rel_err: float | None = None
    max_abs_err: float | None = None
    l2_rel_err: float | None = None


@dataclass
class RunResult:
    valid: bool
    kernel: str
    variant: str
    implementation: str
    platform: str
    precision: str
    matrix: dict
    params: dict
    protocol_used: dict
    preprocessing_ms: float
    correctness: CorrectnessResult
    times_ms: list[float] = field(default_factory=list)
    stats_ms: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "valid": self.valid,
            "kernel": self.kernel,
            "variant": self.variant,
            "implementation": self.implementation,
            "platform": self.platform,
            "precision": self.precision,
            "matrix": self.matrix,
            "params": self.params,
            "protocol_used": self.protocol_used,
            "preprocessing_ms": self.preprocessing_ms,
            "correctness": vars(self.correctness),
            "stats_ms": self.stats_ms,
            "metrics": self.metrics,
            "warnings": self.warnings,
        }
        # raw per-iteration samples kept so variance claims stay auditable
        d["times_ms"] = [round(t, 6) for t in self.times_ms]
        return d


def _stats(times_ms: list[float]) -> dict:
    s = sorted(times_ms)
    return {
        "median": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "mean": statistics.fmean(s),
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
        "p10": s[int(0.10 * (len(s) - 1))],
        "p90": s[int(0.90 * (len(s) - 1))],
        "n": len(s),
    }


def check_correctness(out: np.ndarray, ref: np.ndarray, tolerance: float | None,
                      reference_name: str, mode: str = "max_scaled_err",
                      scale: np.ndarray | None = None) -> CorrectnessResult:
    """
    Compare against the fp64 reference.

    The default gate is `max_scaled_err`: |out - ref| divided by the *magnitude
    scale of the computation* (|A|·|B| for a product), not by |ref|. This matters
    and is not pedantry. An SpMM output element is a sum of tens of signed
    products; cancellation routinely leaves a true value near zero, and then the
    pointwise relative error |out-ref|/|ref| explodes to O(1) no matter how good
    the kernel is. Every surveyed spec says "max relative error < tol" without
    saying relative to what — measured naively, a correct fp32 SpMM fails a 1e-6
    bound on ordinary random input.

    |A|·|B| is the standard componentwise backward-error denominator (the
    computed product satisfies |C_hat - C| <= n*u*(|A|*|B|) elementwise), so a
    kernel passing this gate is backward stable at the stated precision, which is
    the property the gate is actually trying to express.

    The pointwise-relative, absolute and L2-relative views are computed too and
    always recorded, so nothing is hidden by the choice.
    """
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if out.shape != ref.shape:
        return CorrectnessResult(False, "shape", float("nan"), tolerance,
                                 reference_name, f"shape {out.shape} != {ref.shape}")
    if mode == "exact":
        eq = bool(np.array_equal(out, ref))
        return CorrectnessResult(eq, "exact", 0.0 if eq else 1.0, None, reference_name)

    diff = np.abs(out - ref)
    max_abs = float(diff.max()) if diff.size else 0.0
    pointwise = float((diff / np.maximum(np.abs(ref), 1e-300)).max()) if diff.size else 0.0
    ref_norm = float(np.linalg.norm(ref.ravel()))
    l2_rel = float(np.linalg.norm((out - ref).ravel()) / ref_norm) if ref_norm > 0 else 0.0

    if mode == "max_abs_err":
        val = max_abs
    elif mode == "max_rel_err":
        val = pointwise
    else:  # max_scaled_err
        if scale is None:
            # fall back to a global scale rather than a per-element one
            denom = float(np.maximum(np.abs(ref).max(), 1e-300))
            val = float(max_abs / denom)
        else:
            s = np.asarray(scale, dtype=np.float64)
            val = float((diff / np.maximum(s, 1e-300)).max()) if diff.size else 0.0

    # float()/bool() are not cosmetic: numpy scalars leak into the result
    # record and json.dump refuses them (hit for real by the solvers domain).
    val = float(val)
    passed = bool(tolerance is not None and val <= tolerance)
    note = "" if tolerance is not None else \
        "no tolerance parsed from spec; gate cannot pass — run is uncertifiable"
    return CorrectnessResult(passed, mode, val, tolerance, reference_name, note,
                             max_pointwise_rel_err=pointwise, max_abs_err=max_abs,
                             l2_rel_err=l2_rel)


def run_variant(impl: Implementation, matrix, variant: Variant, params: dict,
                reference: Callable[[Any, dict], np.ndarray],
                *, correctness_mode: str = "max_rel_err",
                reference_name: str = "scipy fp64",
                warmup_override: int | None = None,
                reps_override: int | None = None,
                tolerance_override: float | None = None) -> RunResult:
    """
    Execute one (implementation, matrix, variant, params) point.

    `tolerance_override`, unlike `warmup_override`/`reps_override`, is not a
    protocol deviation and does not get flagged as one. Some kernels' spec
    text states their correctness bound symbolically (e.g. lossy-compression's
    "|D_i - D'_i| <= eb", where eb is an error-bound value chosen per
    workload/run, not a literal number in the spec) so `spec.py` correctly
    leaves `variant.tolerance` unparsed (None) rather than inventing a number.
    The domain module then supplies the real, workload-derived bound here;
    when omitted, behavior is unchanged (`variant.tolerance` is used, as
    before).
    """
    proto = variant.protocol
    warnings: list[str] = []

    warmup = warmup_override if warmup_override is not None else proto.warmup
    reps = reps_override if reps_override is not None else proto.reps
    if warmup is None:
        warmup = 10
        warnings.append("spec warmup unparsable; harness default 10 used")
    if reps is None:
        reps = 100
        warnings.append("spec reps unparsable; harness default 100 used")
    if warmup_override is not None or reps_override is not None:
        warnings.append(
            f"protocol overridden for this run (warmup={warmup}, reps={reps}); "
            "result is NOT spec-conforming and must not be published as such")

    # ---- 1. preprocessing, timed once, excluded from per-call time -----------
    t0 = time.perf_counter()
    handle = impl.prepare(matrix, params)
    preprocessing_ms = (time.perf_counter() - t0) * 1e3

    try:
        # ---- 2. correctness gate BEFORE timing ------------------------------
        out = impl.run(handle)
        got = impl.to_host(out)
        ref_out = reference(matrix, params)
        ref, scale = ref_out if isinstance(ref_out, tuple) else (ref_out, None)
        tol = (variant.tolerance_for(params.get("precision")) if hasattr(variant, "tolerance_for")
               else variant.tolerance) if tolerance_override is None else tolerance_override
        corr = check_correctness(got, ref, tol, reference_name,
                                 correctness_mode, scale=scale)
        if not corr.passed:
            return RunResult(
                valid=False, kernel=variant.kernel, variant=variant.id,
                implementation=impl.name, platform=impl.platform,
                precision=impl.precision, matrix=matrix.describe(), params=params,
                protocol_used={"warmup": warmup, "reps": reps,
                               "statistic": proto.statistic,
                               "timer": proto.timer,
                               "timing_scope": proto.timing_scope},
                preprocessing_ms=preprocessing_ms, correctness=corr,
                warnings=warnings + ["correctness gate failed; no timing reported"],
            )

        # ---- 3. warmup, discarded -------------------------------------------
        for _ in range(warmup):
            impl.run(handle)

        # ---- 4. measured reps, one timer pair per iteration ------------------
        times_ms: list[float] = []
        for _ in range(reps):
            with impl.timer() as t:
                impl.run(handle)
            times_ms.append(t.seconds * 1e3)

    finally:
        impl.free(handle)

    stats = _stats(times_ms)
    stat_used = proto.statistic if proto.statistic in stats else "median"
    if stat_used != proto.statistic:
        warnings.append(
            f"spec statistic {proto.statistic!r} not computable; median used")
    sec = stats[stat_used] / 1e3

    # cost model comes from the domain module that owns this kernel
    params_for_cost = dict(params)
    params_for_cost.setdefault("precision", impl.precision)
    fl, by = workload.cost(variant.kernel, matrix, params_for_cost)
    met = {
        "throughput": metrics.throughput(fl, sec),
        "throughput_unit": workload.unit(variant.kernel),
        "gbytes_per_s_lower_bound": metrics.bandwidth(by, sec),
        "work_count": fl,
        "byte_count_compulsory": by,
        "seconds_used_for_metric": sec,
        "statistic_used": stat_used,
    }
    # kept under its historical name so existing readers/reports keep working
    met["gflops"] = met["throughput"]
    # Amortized figure whenever preprocessing is real; makes the tradeoff
    # visible. NOTE: the divisor is this run's rep count, NOT any spec-defined
    # amortization k — the field names say so to prevent misquoting.
    if preprocessing_ms > 0:
        amortized_sec = sec + (preprocessing_ms / 1e3) / reps
        met["throughput_amortized_over_reps"] = metrics.throughput(fl, amortized_sec)
        met["amortization_divisor_is_reps_not_spec_k"] = reps

    if stats["stdev"] > 0.25 * stats["median"]:
        warnings.append(
            f"high variance: stdev {stats['stdev']:.4f} ms is >25% of median "
            f"{stats['median']:.4f} ms — check for a shared/contended device")

    return RunResult(
        valid=True, kernel=variant.kernel, variant=variant.id,
        implementation=impl.name, platform=impl.platform, precision=impl.precision,
        matrix=matrix.describe(), params=params,
        protocol_used={"warmup": warmup, "reps": reps, "statistic": proto.statistic,
                       "timer": proto.timer, "timing_scope": proto.timing_scope,
                       "constants_provenance": proto.provenance},
        preprocessing_ms=preprocessing_ms, correctness=corr,
        times_ms=times_ms, stats_ms=stats, metrics=met, warnings=warnings,
    )
