"""
Load a benchspec and turn it into an *executable* protocol.

The specs in benchspecs/<kernel>/spec.yaml were written to be read by humans, so
fields carry prose alongside their values ("warmup: 10", but also
"reps: 'k=100 post-conversion SpMM calls; report both ...'"). This module
extracts the machine-actionable part and — importantly — records what it could
NOT extract, so a run never silently invents a protocol constant.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BENCHSPECS = os.path.normpath(os.path.join(HERE, "..", "..", "benchspecs"))

_INT_RE = re.compile(r"(\d+)")
_TOL_RE = re.compile(r"([0-9.]+e-?\d+|[0-9]*\.?[0-9]+)")


def _first_int(value: Any, default: int | None = None) -> tuple[int | None, str]:
    """Return (int, provenance). provenance is 'literal', 'parsed' or 'default'."""
    if isinstance(value, bool):
        return default, "default"
    if isinstance(value, int):
        return value, "literal"
    if isinstance(value, str):
        # a sweep expression ("k in {2, 4, 8}; report each k separately") is not
        # a single protocol constant; grabbing its first number silently turns a
        # mandated sweep into reps=2 (hit for real by spmv-sequence-krylov)
        if re.search(r"\bin\s*\{[^}]*,", value):
            return default, "sweep-not-scalar"
        m = _INT_RE.search(value)
        if m:
            return int(m.group(1)), "parsed"
    return default, "default"


_SAME_AS_RE = re.compile(r"same\b[^.]*?\bas\s+([a-z0-9][a-z0-9\-]*[a-z0-9])", re.I)
# "rtol = 1e-6", "flteps = 1e-4 for fp32", "eps = 1e-9"
_ASSIGN_RE = re.compile(
    r"\b(?:tol|rtol|atol|eps|flteps|tolerance|threshold)\w*\s*(?:=|of|is)\s*"
    r"([0-9]*\.?[0-9]+[eE][-+]?\d+|[0-9]*\.?[0-9]+)", re.I)

# Gates that are legitimately non-numeric. These are NOT parse failures: the
# kernel's answer is exact (counts, labels, paths, bit-exact roundtrips) or the
# gate is a structural/property check. A domain module supplies the comparison;
# the spec supplies the requirement.
_STRUCTURAL_MARKERS = (
    "exact match", "bit-exact", "exactly equal", "exact maximum", "exact top-k",
    "validation_level", "must be a valid", "must be non-negative",
    "cost-optimal", "bijection", "partition", "no tolerance", "integer",
    "recall@", "ground-truth", "invariant", "verified against", "must equal",
)


def _classify_gate(text: str) -> str:
    t = text.lower()
    if any(m in t for m in _STRUCTURAL_MARKERS):
        return "structural"
    return "numeric"


_PREC_ALIAS = {"double": "fp64", "single": "fp32", "half": "fp16"}
_PER_PREC_RE = re.compile(
    r"[<≤]=?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*(?:for|at|@|\()?\s*"
    r"(fp64|fp32|fp16|bf16|tf32|double|single|half)", re.I)
_PER_PREC_FOR_RE = re.compile(
    r"(?<![\w.])([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s+for\s+(fp64|fp32|fp16|bf16|tf32|double|single|half)\b", re.I)
_PER_PREC_SLASH_RE = re.compile(
    r"([0-9.]+(?:[eE]-?\d+)?)\s*/\s*([0-9.]+(?:[eE]-?\d+)?)\s*/\s*([0-9.]+(?:[eE]-?\d+)?)"
    r"\s*for\s*(fp\d+)\s*/\s*(fp\d+)\s*/\s*(fp\d+)", re.I)


def _tolerance_by_precision(text: Any) -> dict[str, float]:
    """Per-precision bounds when a correctness sentence states them, e.g.
    "< 1e-6 for fp64 kernels, < 1e-3 for fp32, < 1e-2 for fp16" (gemm, gemv,
    46 variants overall). Empty when the sentence names a single bound. The
    harness prefers the entry matching the run's precision over the generic
    `tolerance` (which stays the first bound parsed, for backward compatibility)
    -- this is the spec's own stated bound, not a loosening: without it an fp32
    or fp16 kernel was gated against the fp64 number."""
    if not isinstance(text, str):
        return {}
    out: dict[str, float] = {}
    for num, prec in _PER_PREC_RE.findall(text):
        prec = _PREC_ALIAS.get(prec.lower(), prec.lower())
        out.setdefault(prec, float(num))
    # "flteps = 1e-4 for fp32, 1e-6 for fp64" / "1e-3 for fp32-base runs":
    # a bound immediately followed by "for <precision>" without an operator.
    for num, prec in _PER_PREC_FOR_RE.findall(text):
        prec = _PREC_ALIAS.get(prec.lower(), prec.lower())
        out.setdefault(prec, float(num))
    m = _PER_PREC_SLASH_RE.search(text)
    if m:
        for num, prec in zip(m.groups()[:3], m.groups()[3:]):
            out.setdefault(prec.lower(), float(num))
    return out if len(out) >= 2 else {}


def _tolerance(text: Any) -> tuple[float | None, str]:
    """
    Pull the numeric tolerance out of a correctness sentence.

    Deliberately conservative: a number is only accepted when it follows a
    comparison operator. Grabbing "the first number in the sentence" is how a
    gate of 32.0 gets silently derived from "same tolerance as
    spmm-gpu-kernel-f32" — a threshold that would pass literally any output.
    When nothing is confidently parseable we return None, and the harness then
    refuses to certify the run rather than inventing a bound.
    """
    if isinstance(text, (int, float)):
        return float(text), "literal"
    if isinstance(text, dict):
        text = " ".join(str(v) for v in text.values())
    if not isinstance(text, str):
        return None, "unparsed"
    # 1. an explicit numeric bound after a comparison operator — the clearest form
    m = re.search(r"[<≤]=?\s*([0-9]*\.?[0-9]+[eE][-+]?\d+|[0-9]*\.?[0-9]+)", text)
    if m:
        return float(m.group(1)), "parsed"
    # 2. a named bound assigned a value ("rtol = 1e-6", "flteps = 1e-4 for fp32").
    #    The symbol appears after the operator, so form 1 misses it.
    m = _ASSIGN_RE.search(text)
    if m:
        return float(m.group(1)), "parsed-assignment"
    # 3. a cross-reference to another variant: resolved by Spec, not here. Only
    #    accept targets that look like variant ids (hyphenated), else "same
    #    ... as the reference" yields a phantom target called "the".
    m = _SAME_AS_RE.search(text)
    if m and "-" in m.group(1):
        return None, f"same-as:{m.group(1)}"
    # 4. gates that are structural/exact by nature carry no number by design
    if _classify_gate(text) == "structural":
        return None, "structural"
    return None, "unparsed"


@dataclass
class Protocol:
    """The part of a variant a runner must obey."""

    warmup: int | None = None
    reps: int | None = None
    statistic: str = "median"
    timer: str = ""
    timing_scope: str = ""
    preprocessing_reported: str = ""
    precision: str = ""
    provenance: dict = field(default_factory=dict)
    unparsed: dict = field(default_factory=dict)


@dataclass
class Variant:
    kernel: str
    id: str
    claim: str
    protocol: Protocol
    inputs: dict
    metric: dict
    correctness_text: str
    tolerance: float | None
    tolerance_provenance: str
    gate_kind: str = "numeric"   # 'numeric' | 'structural'
    tolerance_by_precision: dict = field(default_factory=dict)  # {"fp32": 1e-3, ...}

    def tolerance_for(self, precision: str | None) -> float | None:
        """The bound to gate a run at `precision`: the spec's per-precision
        bound when it states one, else the generic parsed tolerance."""
        if precision and self.tolerance_by_precision:
            t = self.tolerance_by_precision.get(precision.lower())
            if t is not None:
                return t
        return self.tolerance

    def recommended_subset(self) -> list[str]:
        rs = self.inputs.get("recommended_subset")
        if isinstance(rs, list):
            return rs
        return []

    def dense_dims(self, key_names=("N", "K")) -> list[int]:
        """Extract the dense-operand dimension sweep, e.g. 'N in {32, 128, 256}'."""
        text = str(self.inputs.get("dense_operand", "")) + " " + \
               str(self.inputs.get("vector_operand", ""))
        for k in key_names:
            m = re.search(rf"\b{k}\b[^{{]*\{{([^}}]*)\}}", text)
            if m:
                vals = [int(x) for x in re.findall(r"\d+", m.group(1))]
                if vals:
                    return sorted(set(vals))
        return []

    def as_dict(self) -> dict:
        d = asdict(self)
        d["protocol"] = asdict(self.protocol) if not isinstance(self.protocol, dict) else self.protocol
        return d


class Spec:
    def __init__(self, kernel: str, path: str | None = None):
        self.kernel = kernel
        self.path = path or os.path.join(BENCHSPECS, kernel, "spec.yaml")
        with open(self.path) as f:
            self.raw = yaml.safe_load(f)
        self.summary = str(self.raw.get("summary", "")).strip()
        self.operation = str(self.raw.get("operation", "")).strip()
        self.notes_on_fairness = self.raw.get("notes_on_fairness") or []
        self.open_questions = self.raw.get("open_questions") or []
        self.evidence = self.raw.get("evidence") or {}
        self._variants = {v["id"]: self._parse_variant(v) for v in self.raw["variants"]}
        self._resolve_protocol_references()
        self._resolve_cross_references()

    def _resolve_protocol_references(self) -> None:
        """Copy a referenced variant's protocol for variants that only cite one."""
        for var in self._variants.values():
            ref = var.protocol.provenance.get("protocol", "")
            if not ref.startswith("same-as:"):
                continue
            target = ref.split(":", 1)[1]
            match = self._variants.get(target) or next(
                (v for k, v in self._variants.items() if k.startswith(target)), None)
            if match is not None and match is not var:
                src = match.protocol
                var.protocol = Protocol(
                    warmup=src.warmup, reps=src.reps, statistic=src.statistic,
                    timer=src.timer, timing_scope=src.timing_scope,
                    preprocessing_reported=src.preprocessing_reported,
                    precision=src.precision,
                    provenance={**src.provenance,
                                "protocol": f"inherited from {match.id}"},
                    unparsed=dict(src.unparsed))
            else:
                var.protocol.unparsed["protocol"] = \
                    f"unresolved protocol reference to {target!r}"

    def _resolve_cross_references(self) -> None:
        """
        Specs routinely say "same tolerance as <other-variant>". Follow the
        reference (one hop, no cycles) so the gate is real rather than absent.
        """
        for var in self._variants.values():
            prov = var.tolerance_provenance
            if not prov.startswith("same-as:"):
                continue
            target = prov.split(":", 1)[1]
            # the reference may be a bare id or a prefix of one
            match = self._variants.get(target) or next(
                (v for k, v in self._variants.items()
                 if k.startswith(target) or target.startswith(k)), None)
            if match is not None and match is not var and match.tolerance is not None:
                var.tolerance = match.tolerance
                var.tolerance_provenance = f"inherited from {match.id}"
            elif match is not None and match.gate_kind == "structural":
                var.gate_kind = "structural"
                var.tolerance_provenance = f"structural, inherited from {match.id}"
            else:
                var.tolerance_provenance = f"unresolved reference to {target!r}"

    def _parse_variant(self, v: dict) -> Variant:
        p = v.get("protocol") or {}
        unparsed, prov = {}, {}
        # a variant may express its whole protocol as a cross-reference in prose
        # ("same timing protocol as <other-variant>") instead of a mapping.
        # Record the target; Spec resolves it once all variants are parsed.
        protocol_ref = None
        if isinstance(p, str):
            m = _SAME_AS_RE.search(p)
            protocol_ref = m.group(1) if m and "-" in m.group(1) else "unresolved"
            p = {}

        warmup, prov["warmup"] = _first_int(p.get("warmup"))
        if warmup is None:
            unparsed["warmup"] = p.get("warmup")
        reps, prov["reps"] = _first_int(p.get("reps"))
        if reps is None:
            unparsed["reps"] = p.get("reps")

        stat = str(p.get("statistic", "median"))
        stat_norm = "median" if "median" in stat.lower() else \
                    ("mean" if "mean" in stat.lower() else stat.lower().split()[0])

        proto = Protocol(
            warmup=warmup,
            reps=reps,
            statistic=stat_norm,
            timer=str(p.get("timer", "")),
            timing_scope=str(p.get("timing_scope", "")),
            preprocessing_reported=str(p.get("preprocessing_reported", "")),
            precision=str(p.get("precision", "")),
            provenance=prov,
            unparsed=unparsed,
        )
        if protocol_ref is not None:
            proto.provenance["protocol"] = f"same-as:{protocol_ref}"
        tol, tol_prov = _tolerance(v.get("correctness"))
        gate_kind = "structural" if tol_prov == "structural" else "numeric"
        return Variant(
            kernel=self.kernel,
            id=v["id"],
            claim=str(v.get("claim", "")).strip(),
            protocol=proto,
            inputs=v.get("inputs") or {},
            metric=v.get("metric") or {},
            correctness_text=str(v.get("correctness", "")).strip(),
            tolerance=tol,
            tolerance_provenance=tol_prov,
            tolerance_by_precision=_tolerance_by_precision(v.get('correctness')),
            gate_kind=gate_kind,
        )

    @property
    def variant_ids(self) -> list[str]:
        return list(self._variants)

    def variant(self, vid: str) -> Variant:
        if vid not in self._variants:
            raise KeyError(f"{self.kernel}: no variant {vid!r}; have {self.variant_ids}")
        return self._variants[vid]


def load(kernel: str) -> Spec:
    return Spec(kernel)
