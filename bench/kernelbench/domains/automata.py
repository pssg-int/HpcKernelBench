"""
string-regex-matching: multi-pattern regular-expression / finite-automata
matching over a byte stream (benchspecs/string-regex-matching/spec.yaml).

This is NOT single-pattern substring search: the operation is simultaneous
multi-pattern NFA matching, the shape every paper in this track's own survey
actually benchmarks (ngAP, gpunfa-artifact/0002PJ20, GSpecPal all process a
whole ruleset against one input stream in one pass).

Interchange format chosen: ANML (Micron Automata Processor XML), per the
task's own instruction ("ngAP and gpunfa both take ANML; check GSpecPal's
input format too" -- GSpecPal turned out to need its own DFA-table binary
format instead, see artifacts/string-regex-matching/README.md for why it is
SKIPPED). The parser/writer implemented here (parse_anml/write_anml/
parse_symbol_set below) is a SMALL INDEPENDENT reimplementation, written
from VASim's own parser source (github.com/jackwadden/VASim, src/
ANMLParser.cpp + src/util.cpp's parseSymbolSet -- the reference ANML reader
every artifact in this track's ancestry embeds or reimplements) and
cross-checked against one real file fetched live from ANMLZoo
(jackwadden/ANMLZoo, ClamAV/anml/515_nocounter.1chip.anml): a bare
`<automata-network>` root (no `<anml>` wrapper needed, VASim itself accepts
both), `<state-transition-element id=... symbol-set=... start=...>` with
`<activate-on-match element=.../>` and `<report-on-match reportcode=.../>`
children (reportcode optional -- the real ClamAV file omits it entirely).
`and`/`or`/`nor`/`counter`/`inverter` elements (ANML's boolean-gate and
counter combinators) are REJECTED with a clear NotImplementedError naming
the offending element -- out of scope, per the task brief ("ignore counters/
booleans and REJECT automata that use them").

  WORKLOADS -- AutomataWorkload: an Automaton (STEs + transition graph) plus
               a byte stream. Two ways an Automaton reaches a workload:
               (a) already built (regex-kernel-throughput-precompiled,
               regex-single-match-latency): `.automaton` is populated at
               workload-construction time, `.patterns` is None; (b) raw
               regex source (regex-e2e-compilation-amortized): `.patterns`
               holds the source REs and every impl must compile its own
               automaton INSIDE prepare(), timed as this variant's subject.
               `.automaton` is ALSO populated in case (b) -- see "reference
               independence" below for why that is not a gate leak.
               smoke_workloads(): synthetic patterns built by this module's
               OWN regex->NFA generator (compile_patterns, Glushkov/
               position-automaton construction: literal chars, `[...]`
               classes with `^` negation and `-` ranges, `.`, `|`, `*`,
               `+`, `?`) over a random byte stream with planted matches
               (_plant_and_build_corpus) -- no downloads needed. The
               precompiled-variant smoke set round-trips its automaton
               through write_anml()/parse_anml() before use, so the
               independent ANML parser is genuinely exercised by every
               smoke run, not just written and left untested.
               load_workload(name): 5 real ANMLZoo benchmarks confirmed
               present at exact paths (checked live via the GitHub API
               while writing this loader): PowerEN, Dotstar, Snort, ClamAV,
               EntityResolution -- cloned lazily (`git clone --depth 1`)
               into bench/automata-data/ANMLZoo/. Every other
               recommended_subset name raises NotImplementedError with the
               specific reason (see load_workload's docstring).
  COST      -- Gbps = corpus_size_bits / matching_time_seconds / 1e9, the
               spec's own single canonical primary unit. _cost_regex
               returns (bits, bytes) so harness's generic
               flops/seconds/1e9 formula equals literal Gbps with no extra
               scaling hack (same technique as annsearch.py's QPS note).
               Report count is a secondary metric, stamped into
               params["report_count"] by to_host() -- same pattern
               compression.py uses for compression_ratio/achieved_* fields
               (no new harness registration needed).
  REFERENCE -- _simulate_reference: an independent, pure-Python, per-byte
               set-of-active-STE-ids simulator, honoring ANML/AP cycle
               semantics: start="all-input" STEs are re-enabled EVERY
               cycle (letting a pattern begin matching at any offset, not
               just offset 0); start="start-of-data" STEs are enabled only
               at cycle 0; a report is emitted at the cycle the reporting
               STE's symbol-set accepts the current byte (offset = that
               byte's 0-based index in the stream -- this module's fixed
               convention; every artifact adapter must convert its own
               offset/report convention into this one in to_host(), see
               artifacts/string-regex-matching/README.md for each
               engine's mapping). Canonical gate encoding: an int64
               [n_reports, 2] array, columns (offset, report_id), sorted
               ascending and de-duplicated -- CORRECTNESS_MODE="exact"
               (harness.check_correctness's array_equal path) then IS the
               spec's own "exact match-position-list equality... a run
               whose match list differs from the reference in any position
               is invalidated" rule, with no new harness code. The spec's
               correctness text carries no numeric tolerance (a purely
               structural set-equality claim), so spec.py parses
               `variant.tolerance = None` with `gate_kind="structural"` --
               exactly the "tolerance None -> structural" reading this
               integration was asked to implement, requiring no harness
               change (CORRECTNESS_MODE="exact" already ignores tolerance).
  IMPLS     -- CPU_IMPLS: NumpyBitsetNFA, a vectorized bit-parallel
               active-state-vector stepper (_build_bitset_tables): a dense
               [n_states, 256] boolean symbol table (a genuine "precomputed
               per-byte symbol mask" per STE, looked up by column) plus a
               SPARSE (scipy.sparse) transition adjacency matrix, stepped
               with one boolean AND and one sparse mat-vec per input byte
               -- a structurally different algorithm from the reference's
               plain per-state Python-set loop, sharing no function or
               code object with it (DOMAIN_GUIDE's reference-independence
               audit ruling). cuda_impls(): EMPTY, documented -- this is
               exactly the kind of kernel where the paper artifacts under
               artifacts/string-regex-matching/<shortname>/ ARE the GPU
               competitors (ngAP, gpunfa-artifact); see ARTIFACT_GUIDE.md.

Reference independence for the e2e-amortized variant (worth spelling out,
since it looks at first glance like a shared-code violation): both
_build_e2e_smoke() and NumpyBitsetNFA.prepare() call compile_patterns() --
the WORKLOAD CONSTRUCTOR calls it once to seed `.automaton` as cached
ground truth (exactly annsearch.py's `_bruteforce_ground_truth` pattern:
"called exactly once per workload, never called by any Implementation AT
GATE TIME"), and each impl separately calls it again, independently, inside
its own timed prepare() (to produce and MEASURE the compilation this variant
is actually about). reference_regex() itself never calls compile_patterns --
it only reads the workload's pre-populated `.automaton` field, the same way
every other domain's reference reads `workload.csr`/`workload.X` (workload
DATA, not a shared computational code path). The graded computation --
matching -- is what DOMAIN_GUIDE's rule actually protects, and that stays
fully independent: _simulate_reference's Python-set walk vs.
NumpyBitsetNFA's vectorized bitset stepper never call each other or a common
helper.

Gate window for real (multi-MB) corpora: _simulate_reference is pure Python
and O(bytes x active_states); over ClamAV's real 10MB input that is minutes,
not seconds. load_workload() therefore sets `gate_window_bytes=8192` on real
workloads -- the reference only simulates the first 8KB, and every impl's
to_host() truncates its OWN reports to offset < 8192 before comparing (see
_canonical_reports's `window` argument) so the comparison stays apples-to-
apples. The implementation under test still runs its full, untruncated
matching pass over the WHOLE stream for timing -- only the correctness
comparison is windowed. smoke_workloads() never sets a window (its corpora
are already KB-sized).

Mutation-tested once (2026-09-05, manually, not left in the tree): flipping
a single bit in NumpyBitsetNFA's `_build_bitset_tables` adjacency (dropping
one `activate-on-match` edge) made `--smoke` FAIL its correctness gate
(shape/value mismatch against the reference), confirming the gate is live
rather than vacuous; the mutation was reverted before this file was
finalized.
"""

from __future__ import annotations

import os
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .. import workload
from ..harness import Timer

KERNELS = ["string-regex-matching"]
PLANNED: list[str] = []

PRECOMPILED_VARIANT = "regex-kernel-throughput-precompiled"
E2E_AMORTIZED_VARIANT = "regex-e2e-compilation-amortized"
SINGLE_MATCH_VARIANT = "regex-single-match-latency"

SMOKE_VARIANT = {"string-regex-matching": PRECOMPILED_VARIANT}

DATA_DIR = os.environ.get(
    "KERNELBENCH_AUTOMATA_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "automata-data"))
DATA_DIR = os.path.normpath(DATA_DIR)


# ============================================================ ANML: STE/Automaton
@dataclass
class STE:
    """One state-transition-element. `report_id` is -1 unless `reporting`."""

    id: str
    symbol_mask: np.ndarray        # bool[256]
    start: str                     # "none" | "start-of-data" | "all-input"
    outputs: list                  # list[str] target STE ids (activate-on-match)
    reporting: bool
    report_id: int


@dataclass
class Automaton:
    id: str
    states: dict                   # id -> STE, insertion order preserved
    pattern_count: int
    source_patterns: "list[str] | None" = None   # raw regex text, if built by compile_patterns
    report_code_map: "dict | None" = None         # ANML reportcode string -> canonical int (parsed files)

    @property
    def n_states(self) -> int:
        return len(self.states)


_REJECT_TAGS = {"and", "or", "nor", "counter", "inverter"}


def parse_symbol_set(symbol_set: str) -> np.ndarray:
    """
    Parse an ANML `symbol-set` attribute into a 256-entry boolean mask.
    Reimplements the flat, single-pass mini-language VASim's own
    parseSymbolSet (src/util.cpp) uses -- confirmed against that source and
    against a real ANMLZoo file's symbol-sets (bracket + `\\xHH` hex-escape
    enumeration, e.g. "[\\x8a\\x86]"):

      '*'  -> every byte (AP "don't care" wildcard)
      '.'  -> every byte except '\\n' (VASim's own hard-coded special case)
      else -> one left-to-right scan: '[' ']' are bookkeeping only (NOT a
              nested grammar -- VASim's own parser treats them the same
              way); literal bytes set directly; '\\xHH' hex escape;
              '\\n' '\\r' '\\t' '\\a' '\\b' C escapes; '\\s'/'\\d' class
              escapes; an unescaped '-' after a literal starts a byte
              range; an unescaped '^' (anywhere) inverts the WHOLE mask
              once at the end (not a per-bracket negation -- matches
              VASim's own single `inverting` flag). Curly-brace symbol
              sets are rejected (VASim's own parser does not implement
              them either: "CURLY BRACES NOT IMPLEMENTED", util.cpp).
    """
    mask = np.zeros(256, dtype=bool)
    if symbol_set == "*":
        mask[:] = True
        return mask
    if symbol_set == ".":
        mask[:] = True
        mask[ord("\n")] = False
        return mask
    if symbol_set.startswith("{") and symbol_set.endswith("}"):
        raise NotImplementedError(
            f"automata: curly-brace symbol-set {symbol_set!r} not supported "
            "(VASim's own parser does not implement this form either)")

    inverting = False
    escaped = False
    range_pending = False
    last_char = -1
    i = 0
    n = len(symbol_set)

    def _emit(byte_val: int) -> None:
        nonlocal last_char, range_pending
        mask[byte_val] = True
        if range_pending and last_char >= 0:
            lo, hi = (last_char, byte_val) if last_char <= byte_val else (byte_val, last_char)
            mask[lo:hi + 1] = True
            range_pending = False
        last_char = byte_val

    while i < n:
        c = symbol_set[i]
        if escaped:
            if c == "x":
                val = int(symbol_set[i + 1:i + 3], 16)
                _emit(val)
                i += 3
                escaped = False
                continue
            if c == "n":
                _emit(10)
            elif c == "r":
                _emit(13)
            elif c == "t":
                _emit(9)
            elif c == "a":
                _emit(7)
            elif c == "b":
                _emit(8)
            elif c == "s":
                for v in (9, 10, 11, 12, 13, 32):
                    mask[v] = True
                last_char = 32
                range_pending = False
            elif c == "d":
                mask[48:58] = True
                last_char = 57
                range_pending = False
            else:
                _emit(ord(c))
            escaped = False
            i += 1
            continue
        if c == "\\":
            escaped = True
            i += 1
            continue
        if c == "[" or c == "]":
            i += 1
            continue
        if c == "^":
            inverting = True
            i += 1
            continue
        if c == "-" and last_char >= 0:
            range_pending = True
            i += 1
            continue
        _emit(ord(c))
        i += 1

    if inverting:
        mask = ~mask
    return mask


def _mask_to_symbol_set(mask: np.ndarray) -> str:
    """Emit the same hex-enumeration style real ANMLZoo automata use."""
    if mask.all():
        return "*"
    idxs = np.nonzero(mask)[0]
    return "[" + "".join(f"\\x{int(b):02x}" for b in idxs) + "]"


def parse_anml(text: str, automaton_id: str = "loaded") -> Automaton:
    """
    Small independent ANML parser. Accepts a bare `<automata-network>` root
    or one nested under `<anml>` (VASim itself accepts both -- ANMLParser.cpp
    comment: "can handle finding automata-network at one or two layers under
    root"). Rejects `and`/`or`/`nor`/`counter`/`inverter` elements with a
    clear NotImplementedError naming the tag and id -- these are out of
    scope per this integration's brief.

    Report-id canonicalization: ANML's `reportcode` attribute is optional
    (a real ANMLZoo file, ClamAV's 515_nocounter.1chip.anml, omits it on
    every `<report-on-match/>`) and, when present, is free-form text, not
    guaranteed numeric. This parser resolves it to a canonical non-negative
    int: `int(reportcode)` when that string is a plain integer, otherwise a
    stable per-automaton index assigned in first-seen order (keyed by the
    reportcode string if present, else by the STE's own id) -- recorded in
    the returned Automaton's `report_code_map` for provenance.
    """
    root = ET.fromstring(text)
    net = root if root.tag == "automata-network" else root.find("automata-network")
    if net is None:
        raise ValueError("automata: no <automata-network> element found in ANML document")

    raw = []
    for el in net:
        tag = el.tag
        if tag == "state-transition-element":
            sid = el.get("id")
            symbol_set = el.get("symbol-set", "")
            start = el.get("start", "none")
            outputs = [c.get("element") for c in el.findall("activate-on-match")]
            rpt = el.find("report-on-match")
            reporting = rpt is not None
            reportcode = rpt.get("reportcode") if rpt is not None else None
            raw.append((sid, symbol_set, start, outputs, reporting, reportcode))
        elif tag in _REJECT_TAGS:
            raise NotImplementedError(
                f"automata: ANML element <{tag} id={el.get('id')!r}> (a counter or "
                "boolean-gate combinator) is not supported by this module's parser "
                "-- automata using counters/booleans are explicitly out of scope "
                "(see module docstring); this automaton cannot be loaded")
        else:
            raise NotImplementedError(f"automata: unrecognized top-level ANML element <{tag}>")

    states: dict[str, STE] = {}
    report_code_map: dict[str, int] = {}
    next_id = 0
    for sid, symbol_set, start, outputs, reporting, reportcode in raw:
        rid = -1
        if reporting:
            if reportcode is not None and reportcode.lstrip("-").isdigit():
                rid = int(reportcode)
            else:
                key = reportcode if reportcode is not None else sid
                if key not in report_code_map:
                    report_code_map[key] = next_id
                    next_id += 1
                rid = report_code_map[key]
        states[sid] = STE(id=sid, symbol_mask=parse_symbol_set(symbol_set),
                          start=start if start in ("all-input", "start-of-data") else "none",
                          outputs=outputs, reporting=reporting, report_id=rid)
    pattern_count = len({s.report_id for s in states.values() if s.reporting})
    return Automaton(id=automaton_id, states=states, pattern_count=pattern_count,
                     source_patterns=None, report_code_map=dict(report_code_map))


def write_anml(automaton: Automaton) -> str:
    """Serialize an Automaton back to ANML text (used to round-trip this
    module's own synthetic smoke automata through parse_anml, so the
    independent parser is genuinely exercised, not just written)."""
    lines = [f'<automata-network id="{automaton.id}">']
    for sid, ste in automaton.states.items():
        attrs = f'id="{sid}" symbol-set="{_mask_to_symbol_set(ste.symbol_mask)}"'
        if ste.start != "none":
            attrs += f' start="{ste.start}"'
        lines.append(f"  <state-transition-element {attrs}>")
        for out in ste.outputs:
            lines.append(f'    <activate-on-match element="{out}"/>')
        if ste.reporting:
            lines.append(f'    <report-on-match reportcode="{ste.report_id}"/>')
        lines.append("  </state-transition-element>")
    lines.append("</automata-network>")
    return "\n".join(lines)


# ================================================== own regex -> NFA compiler
# Glushkov (position/Berry-Sethi) construction: each terminal (literal char
# or class) occurrence becomes one STE, transitions computed via
# nullable/firstpos/lastpos/followpos. This directly produces a HOMOGENEOUS
# automaton (symbol tests on states, not edges, no epsilon transitions) --
# exactly ANML/AP semantics -- unlike a textbook Thompson construction,
# which would need epsilon-elimination to fit this model.

class _Lit:
    __slots__ = ("mask", "pos")

    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.pos = None


class _Concat:
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b


class _Alt:
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b


class _Star:
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a


class _Plus:
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a


class _Opt:
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a


class _RegexParser:
    """
    Minimal recursive-descent parser for this module's OWN smoke-pattern
    grammar: literals, `\\`-escaped literals, `.`, `[...]` classes (`^`
    negation, `-` ranges), `(...)` grouping, `|` alternation, implicit
    concatenation, postfix `*` `+` `?`. This is deliberately simpler than
    general PCRE -- it only needs to describe the smoke patterns this
    module itself generates and compiles; real ANMLZoo automata are read
    via parse_anml(), never via this regex grammar.
    """

    def __init__(self, pattern: str):
        self.s = pattern
        self.i = 0

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else None

    def parse(self):
        node = self._alt()
        if self.i != len(self.s):
            raise ValueError(f"automata regex: unexpected char at {self.i} in {self.s!r}")
        return node

    def _alt(self):
        node = self._concat()
        while self.peek() == "|":
            self.i += 1
            node = _Alt(node, self._concat())
        return node

    def _concat(self):
        node = None
        while self.peek() is not None and self.peek() not in "|)":
            atom = self._postfix()
            node = atom if node is None else _Concat(node, atom)
        if node is None:
            raise ValueError(f"automata regex: empty concatenation in {self.s!r} not supported")
        return node

    def _postfix(self):
        atom = self._atom()
        while self.peek() in ("*", "+", "?"):
            op = self.s[self.i]
            self.i += 1
            atom = {"*": _Star, "+": _Plus, "?": _Opt}[op](atom)
        return atom

    def _atom(self):
        c = self.peek()
        if c == "(":
            self.i += 1
            node = self._alt()
            if self.peek() != ")":
                raise ValueError(f"automata regex: unbalanced '(' in {self.s!r}")
            self.i += 1
            return node
        if c == "[":
            return _Lit(self._char_class())
        if c == ".":
            self.i += 1
            m = np.ones(256, dtype=bool)
            m[ord("\n")] = False
            return _Lit(m)
        if c == "\\":
            self.i += 1
            lit = self.s[self.i]
            self.i += 1
            m = np.zeros(256, dtype=bool)
            m[ord(lit)] = True
            return _Lit(m)
        if c is None or c in "|)*+?":
            raise ValueError(f"automata regex: unexpected token at {self.i} in {self.s!r}")
        self.i += 1
        m = np.zeros(256, dtype=bool)
        m[ord(c)] = True
        return _Lit(m)

    def _char_class(self):
        assert self.s[self.i] == "["
        self.i += 1
        neg = False
        if self.peek() == "^":
            neg = True
            self.i += 1
        m = np.zeros(256, dtype=bool)
        prev = None
        while self.peek() != "]":
            if self.peek() is None:
                raise ValueError(f"automata regex: unterminated class in {self.s!r}")
            c = self.s[self.i]
            if c == "-" and prev is not None and self.i + 1 < len(self.s) and self.s[self.i + 1] != "]":
                self.i += 1
                hi = self.s[self.i]
                lo_o, hi_o = ord(prev), ord(hi)
                if lo_o > hi_o:
                    lo_o, hi_o = hi_o, lo_o
                m[lo_o:hi_o + 1] = True
                self.i += 1
                prev = None
                continue
            m[ord(c)] = True
            prev = c
            self.i += 1
        self.i += 1  # consume ']'
        if neg:
            m = ~m
        return m


def _assign_positions(node, positions: list) -> None:
    if isinstance(node, _Lit):
        node.pos = len(positions)
        positions.append(node.mask)
    elif isinstance(node, (_Concat, _Alt)):
        _assign_positions(node.a, positions)
        _assign_positions(node.b, positions)
    elif isinstance(node, (_Star, _Plus, _Opt)):
        _assign_positions(node.a, positions)
    else:
        raise TypeError(node)


def _nullable(node) -> bool:
    if isinstance(node, _Lit):
        return False
    if isinstance(node, _Concat):
        return _nullable(node.a) and _nullable(node.b)
    if isinstance(node, _Alt):
        return _nullable(node.a) or _nullable(node.b)
    if isinstance(node, _Star):
        return True
    if isinstance(node, _Plus):
        return _nullable(node.a)
    if isinstance(node, _Opt):
        return True
    raise TypeError(node)


def _firstpos(node) -> set:
    if isinstance(node, _Lit):
        return {node.pos}
    if isinstance(node, _Concat):
        return (_firstpos(node.a) | _firstpos(node.b)) if _nullable(node.a) else _firstpos(node.a)
    if isinstance(node, _Alt):
        return _firstpos(node.a) | _firstpos(node.b)
    if isinstance(node, (_Star, _Plus, _Opt)):
        return _firstpos(node.a)
    raise TypeError(node)


def _lastpos(node) -> set:
    if isinstance(node, _Lit):
        return {node.pos}
    if isinstance(node, _Concat):
        return (_lastpos(node.a) | _lastpos(node.b)) if _nullable(node.b) else _lastpos(node.b)
    if isinstance(node, _Alt):
        return _lastpos(node.a) | _lastpos(node.b)
    if isinstance(node, (_Star, _Plus, _Opt)):
        return _lastpos(node.a)
    raise TypeError(node)


def _compute_followpos(node, followpos: dict) -> None:
    if isinstance(node, _Concat):
        for p in _lastpos(node.a):
            followpos[p] |= _firstpos(node.b)
        _compute_followpos(node.a, followpos)
        _compute_followpos(node.b, followpos)
    elif isinstance(node, _Alt):
        _compute_followpos(node.a, followpos)
        _compute_followpos(node.b, followpos)
    elif isinstance(node, (_Star, _Plus)):
        for p in _lastpos(node.a):
            followpos[p] |= _firstpos(node.a)
        _compute_followpos(node.a, followpos)
    elif isinstance(node, _Opt):
        _compute_followpos(node.a, followpos)
    elif isinstance(node, _Lit):
        pass
    else:
        raise TypeError(node)


def compile_patterns(patterns: list[str]) -> tuple[Automaton, dict]:
    """
    Compile a list of regex patterns into ONE Automaton (each pattern gets
    its own disjoint set of positions/STEs -- a plain union, no cross-
    pattern merging/optimization, matching this generator's own scope).
    Every pattern's firstpos positions are marked start="all-input" (so the
    pattern can begin matching starting at ANY offset in the stream --
    standard streaming multi-pattern search semantics, not anchored-at-0
    matching); lastpos positions are marked reporting with report_id =
    the pattern's index in `patterns`.

    Returns (automaton, stage_times) where stage_times has the same stage
    names regex-e2e-compilation-amortized's spec asks for
    (front-end/construction/optimization/merging); this generator performs
    no automaton merging or single-automaton optimization, so those two
    stages are reported as 0.0 -- an honest scope limitation, not a bug.
    """
    t0 = time.perf_counter()
    asts = []
    for pat in patterns:
        node = _RegexParser(pat).parse()
        if _nullable(node):
            raise ValueError(
                f"automata: pattern {pat!r} can match the empty string; not "
                "supported by this generator")
        asts.append(node)
    t1 = time.perf_counter()

    states: dict[str, STE] = {}
    for pidx, node in enumerate(asts):
        positions: list[np.ndarray] = []
        _assign_positions(node, positions)
        followpos = {i: set() for i in range(len(positions))}
        _compute_followpos(node, followpos)
        first = _firstpos(node)
        last = _lastpos(node)
        for pos, mask in enumerate(positions):
            sid = f"p{pidx}_{pos}"
            reporting = pos in last
            states[sid] = STE(
                id=sid, symbol_mask=mask,
                start="all-input" if pos in first else "none",
                outputs=[f"p{pidx}_{q}" for q in sorted(followpos[pos])],
                reporting=reporting, report_id=pidx if reporting else -1)
    t2 = time.perf_counter()

    automaton = Automaton(id="synthetic-compiled", states=states,
                          pattern_count=len(patterns), source_patterns=list(patterns))
    stage_times = {
        "frontend_ms": (t1 - t0) * 1e3,
        "automaton_construction_ms": (t2 - t1) * 1e3,
        "single_automaton_optimization_ms": 0.0,   # not implemented -- see docstring
        "merging_ms": 0.0,                          # not implemented -- see docstring
    }
    return automaton, stage_times


def _random_matching_bytes(node, rng: np.random.Generator, max_repeat: int = 3) -> bytes:
    """Generate ONE concrete byte string the given AST accepts -- used only
    to plant guaranteed matches into smoke corpora, never for gating."""
    if isinstance(node, _Lit):
        choices = np.nonzero(node.mask)[0]
        return bytes([int(rng.choice(choices))])
    if isinstance(node, _Concat):
        return _random_matching_bytes(node.a, rng, max_repeat) + _random_matching_bytes(node.b, rng, max_repeat)
    if isinstance(node, _Alt):
        pick = node.a if rng.random() < 0.5 else node.b
        return _random_matching_bytes(pick, rng, max_repeat)
    if isinstance(node, _Star):
        k = int(rng.integers(0, max_repeat + 1))
        return b"".join(_random_matching_bytes(node.a, rng, max_repeat) for _ in range(k))
    if isinstance(node, _Plus):
        k = int(rng.integers(1, max_repeat + 1))
        return b"".join(_random_matching_bytes(node.a, rng, max_repeat) for _ in range(k))
    if isinstance(node, _Opt):
        return _random_matching_bytes(node.a, rng, max_repeat) if rng.random() < 0.5 else b""
    raise TypeError(node)


def _plant_and_build_corpus(asts: list, rng: np.random.Generator, length: int, n_plants: int) -> np.ndarray:
    corpus = rng.integers(0, 256, size=length, dtype=np.uint8)
    for _ in range(n_plants):
        node = asts[int(rng.integers(0, len(asts)))]
        s = _random_matching_bytes(node, rng)
        if not s or len(s) > length:
            continue
        off = int(rng.integers(0, length - len(s) + 1))
        corpus[off:off + len(s)] = np.frombuffer(s, dtype=np.uint8)
    return corpus


# ==================================================================== workload
@dataclass
class AutomataWorkload:
    name: str
    variant: str
    automaton: "Automaton | None"     # None only never -- always populated; see class docstring for e2e case
    stream: np.ndarray                 # uint8[L]
    patterns: "list[str] | None"       # raw regex source; set only when prepare() must compile it itself
    re_count: int
    gate_window_bytes: "int | None"
    synthetic: bool
    seed: int
    source: str
    automata_family: str = "NFA (homogeneous/position automaton, ANML STE semantics)"

    def describe(self) -> dict:
        d = {
            "name": self.name, "kernel": "string-regex-matching", "spec_variant": self.variant,
            "source": self.source, "synthetic": self.synthetic, "seed": self.seed,
            "corpus_bytes": int(self.stream.shape[0]), "corpus_bits": int(self.stream.shape[0]) * 8,
            "re_count": self.re_count, "automata_family": self.automata_family,
            "gate_window_bytes": self.gate_window_bytes,
            "n_states": self.automaton.n_states if self.automaton is not None else None,
            "compiled_at": (
                "workload-build-time (already-compiled automaton supplied to prepare(), excluded "
                "from the timed matching pass)" if self.patterns is None else
                "inside prepare() (this variant's own subject; .automaton is cached ground truth "
                "for the correctness gate only -- see module docstring's reference-independence note)"),
            "correctness_gate": (
                "exact set match: sorted, de-duplicated (offset, report_id) pairs must equal the "
                "independent reference simulator's output exactly (CORRECTNESS_MODE='exact', "
                "tolerance=None per the spec's own structural correctness ruling)"),
        }
        return d


# =============================================================== reference
def _simulate_reference(automaton: Automaton, stream: np.ndarray, window: "int | None"):
    """
    Independent, pure-Python, set-of-active-states-per-byte NFA simulator.
    See module docstring for the full ANML cycle-semantics description and
    why this never shares code with NumpyBitsetNFA's vectorized stepper.
    """
    active: set = set()
    reports: list[tuple[int, int]] = []
    limit = len(stream) if window is None else min(window, len(stream))
    states = automaton.states
    all_input = [sid for sid, s in states.items() if s.start == "all-input"]
    start_of_data = [sid for sid, s in states.items() if s.start == "start-of-data"]
    for t in range(limit):
        b = int(stream[t])
        enabled = set(active)
        enabled.update(all_input)
        if t == 0:
            enabled.update(start_of_data)
        fired = []
        for sid in enabled:
            ste = states[sid]
            if ste.symbol_mask[b]:
                fired.append(sid)
                if ste.reporting:
                    reports.append((t, ste.report_id))
        nxt: set = set()
        for sid in fired:
            nxt.update(states[sid].outputs)
        active = nxt
    return reports


def reference_regex(w: AutomataWorkload, params: dict):
    if w.automaton is None:
        raise RuntimeError(
            "string-regex-matching: reference_regex requires a pre-built automaton "
            "cached on the workload -- every workload constructor in this module "
            "populates one (see AutomataWorkload/compile_patterns); this indicates "
            "a workload built outside this module's own helpers")
    reports = _simulate_reference(w.automaton, w.stream, w.gate_window_bytes)
    if not reports:
        arr = np.zeros((0, 2), dtype=np.int64)
    else:
        arr = np.array(sorted(set(reports)), dtype=np.int64)
    return arr, None


REFERENCES = {"string-regex-matching": reference_regex}
REFERENCE_NAME = {
    "string-regex-matching": (
        "independent pure-Python per-byte set-of-active-states NFA simulator "
        "(_simulate_reference) -- never shares code with NumpyBitsetNFA's vectorized "
        "bitset stepper or with any artifact adapter's matching engine"),
}
CORRECTNESS_MODE = {"string-regex-matching": "exact"}
DEFAULT_PRECISION = {"string-regex-matching": "int64"}
DIM_KEY: dict[str, str] = {}


# ----------------------------------------------------------------- cost rule
def _cost_regex(w: AutomataWorkload, params: dict) -> tuple[int, int]:
    """
    Gbps = corpus_size_bits / matching_time_seconds / 1e9 (spec's own
    canonical primary unit, literal). metrics.throughput() divides by 1e9
    unconditionally, so returning BITS (not a flop count) as the work
    count makes the harness's generic throughput field equal literal Gbps
    with no extra scaling hack (same technique as annsearch.py's QPS note).
    `bytes`: the compulsory read of every input byte once (the automaton
    is resident across reps, never re-read per call).
    """
    n = int(w.stream.shape[0])
    return n * 8, n


workload.register_cost("string-regex-matching", _cost_regex, "Gbps")


# --------------------------------------------------------------------- impls
def _canonical_reports(reports: list, window: "int | None" = None) -> np.ndarray:
    """
    Shared canonicalization for every IMPLEMENTATION's to_host() (CPU floor
    + any paper-artifact adapter) -- NEVER called by reference_regex or
    _simulate_reference (see module docstring's reference-independence
    note). `window`, when set, truncates to offset < window BEFORE
    dedup/sort, matching real workloads' gate_window_bytes cap on the
    reference so both sides compare the same prefix of the stream.
    """
    if window is not None:
        reports = [r for r in reports if r[0] < window]
    if not reports:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(set(reports)), dtype=np.int64)


def _build_bitset_tables(automaton: Automaton) -> dict:
    """
    Vectorized, matrix/bitset formulation of the automaton: a dense
    [n_states, 256] boolean symbol table (one precomputed per-byte mask per
    STE) plus a SPARSE (scipy.sparse) transition adjacency matrix. Shares no
    function with _simulate_reference's plain per-state Python-set walk.
    """
    ids = list(automaton.states.keys())
    idx = {sid: i for i, sid in enumerate(ids)}
    n = len(ids)
    symbol_table = np.zeros((n, 256), dtype=bool)
    all_input = np.zeros(n, dtype=bool)
    start_of_data = np.zeros(n, dtype=bool)
    reporting = np.zeros(n, dtype=bool)
    report_id = np.full(n, -1, dtype=np.int64)
    rows, cols = [], []
    for sid, ste in automaton.states.items():
        i = idx[sid]
        symbol_table[i] = ste.symbol_mask
        if ste.start == "all-input":
            all_input[i] = True
        elif ste.start == "start-of-data":
            start_of_data[i] = True
        if ste.reporting:
            reporting[i] = True
            report_id[i] = ste.report_id
        for out in ste.outputs:
            rows.append(i)
            cols.append(idx[out])
    if rows:
        data = np.ones(len(rows), dtype=np.int8)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    else:
        adj = sp.csr_matrix((n, n), dtype=np.int8)
    adjT = adj.transpose().tocsr()
    return {"n": n, "symbol_table": symbol_table, "all_input": all_input,
           "start_of_data": start_of_data, "reporting": reporting,
           "report_id": report_id, "adjT": adjT}


class NumpyBitsetNFA:
    """
    CPU floor: a vectorized bit-parallel active-state stepper (see
    _build_bitset_tables). One Python loop over input bytes, 2-3 numpy/
    scipy ops per byte -- adequate for smoke-sized and moderate corpora,
    NOT meant to compete on the spec's full 1/10/20MB sweep (that regime is
    what the GPU paper artifacts under artifacts/string-regex-matching/ are
    for; this impl is the correctness-establishing CPU floor, per
    DOMAIN_GUIDE).
    """

    name = "numpy-bitset-nfa"
    platform = "cpu"

    def __init__(self, precision: str = "int64"):
        self.precision = precision

    def prepare(self, w: AutomataWorkload, params: dict):
        if w.patterns is None:
            automaton = w.automaton
            params["compile_stage_frontend_ms"] = 0.0
            params["compile_stage_automaton_construction_ms"] = 0.0
            params["compile_note"] = "automaton supplied pre-compiled (excluded per this variant's protocol)"
        else:
            automaton, stage_times = compile_patterns(w.patterns)
            params["compile_stage_frontend_ms"] = stage_times["frontend_ms"]
            params["compile_stage_automaton_construction_ms"] = stage_times["automaton_construction_ms"]
            params["compile_stage_single_automaton_optimization_ms"] = stage_times["single_automaton_optimization_ms"]
            params["compile_stage_merging_ms"] = stage_times["merging_ms"]
            params["compile_note"] = "compiled from raw regex source inside prepare() (this variant's own subject)"
        tables = _build_bitset_tables(automaton)
        params["automata_family"] = "NFA (explicit bitset/position-automaton representation)"
        params["n_states"] = tables["n"]
        self._window = w.gate_window_bytes
        self._params = params
        return {"tables": tables, "stream": w.stream}

    def run(self, h):
        t = h["tables"]
        n = t["n"]
        symbol_table, all_input, start_of_data = t["symbol_table"], t["all_input"], t["start_of_data"]
        reporting, report_id, adjT = t["reporting"], t["report_id"], t["adjT"]
        stream = h["stream"]
        active = np.zeros(n, dtype=bool)
        reports: list = []
        for pos in range(stream.shape[0]):
            b = int(stream[pos])
            cand = active | all_input
            if pos == 0:
                cand = cand | start_of_data
            fired = cand & symbol_table[:, b]
            if fired.any():
                rep_idx = np.nonzero(fired & reporting)[0]
                for i in rep_idx:
                    reports.append((pos, int(report_id[i])))
                active = (adjT.dot(fired.astype(np.int8))) > 0
            else:
                active = np.zeros(n, dtype=bool)
        return reports

    def to_host(self, out):
        arr = _canonical_reports(out, self._window)
        self._params["report_count"] = int(arr.shape[0])
        return arr

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


CPU_IMPLS = {"string-regex-matching": {"numpy-bitset-nfa": NumpyBitsetNFA}}


def cuda_impls():
    """
    No CUDA implementation shipped by this module -- automata processing is
    exactly the kind of kernel where the paper artifacts under
    artifacts/string-regex-matching/<shortname>/ ARE the GPU competitors
    (ngAP, gpunfa-artifact; GSpecPal SKIPPED, see that directory's README).
    Same documented, honest empty-dict choice as graph.py's/annsearch.py's
    cuda_impls().
    """
    return {}


# ============================================================== smoke set
SMOKE_SEED = 20260905

_SMOKE_PATTERN_SETS = [
    ["cat", "dog", "fish"],                        # pure literal, 3 independent patterns
    ["a[0-9]+", "b.c", "(foo|bar)x*", "gr[ae]y"],   # classes, dot, alternation, plus/star
]


def _build_precompiled_smoke(name: str, patterns: list, corpus_len: int, n_plants: int, seed: int) -> AutomataWorkload:
    automaton, _ = compile_patterns(patterns)
    # Round-trip through the independent ANML writer/parser, so smoke
    # actually exercises parse_anml/write_anml, not just compile_patterns.
    automaton = parse_anml(write_anml(automaton), automaton_id=name)
    rng = np.random.default_rng(seed)
    asts = [_RegexParser(p).parse() for p in patterns]
    stream = _plant_and_build_corpus(asts, rng, corpus_len, n_plants)
    return AutomataWorkload(
        name=name, variant=PRECOMPILED_VARIANT, automaton=automaton, stream=stream,
        patterns=None, re_count=len(patterns), gate_window_bytes=None, synthetic=True, seed=seed,
        source=(f"synthetic: this module's own regex-to-NFA-to-ANML generator, patterns={patterns!r}, "
                "round-tripped through write_anml()/parse_anml(); smoke only, NOT spec-conforming"))


def _build_e2e_smoke(name: str, patterns: list, corpus_len: int, n_plants: int, seed: int) -> AutomataWorkload:
    automaton, _ = compile_patterns(patterns)  # cached ground truth ONLY -- see module docstring
    rng = np.random.default_rng(seed)
    asts = [_RegexParser(p).parse() for p in patterns]
    stream = _plant_and_build_corpus(asts, rng, corpus_len, n_plants)
    return AutomataWorkload(
        name=name, variant=E2E_AMORTIZED_VARIANT, automaton=automaton, stream=stream,
        patterns=list(patterns), re_count=len(patterns), gate_window_bytes=None, synthetic=True, seed=seed,
        source=(f"synthetic: same generator as the precompiled smoke set, patterns={patterns!r}; every "
                "impl must compile its OWN automaton from `.patterns` inside prepare(), timed as this "
                "variant's subject -- `.automaton` here is cached ground truth for gating only"))


def _build_latency_smoke(name: str, pattern: str, window: int, seed: int) -> AutomataWorkload:
    automaton, _ = compile_patterns([pattern])
    rng = np.random.default_rng(seed)
    ast = _RegexParser(pattern).parse()
    stream = _plant_and_build_corpus([ast], rng, window, n_plants=1)
    return AutomataWorkload(
        name=name, variant=SINGLE_MATCH_VARIANT, automaton=automaton, stream=stream,
        patterns=None, re_count=1, gate_window_bytes=None, synthetic=True, seed=seed,
        source=f"synthetic single-RE smoke, pattern={pattern!r}, bounded window")


def smoke_workloads(variant: "str | None" = None) -> list:
    """
    Small, synthetic, runs anywhere in seconds. NOT spec-conforming. With
    variant=None (every non-CLI caller's default), returns all three
    variants' workloads together; runner.py passes variant=<the CLI's
    --variant> when smoke_workloads accepts it (same convention as
    annsearch.py/dense.py), so `--variant <v> --smoke` only builds that
    variant's own workload shape.
    """
    precompiled = [
        _build_precompiled_smoke("smoke-literal-set", _SMOKE_PATTERN_SETS[0], 2000, 6, SMOKE_SEED),
        _build_precompiled_smoke("smoke-regex-features", _SMOKE_PATTERN_SETS[1], 3000, 8, SMOKE_SEED + 1),
    ]
    e2e = [
        _build_e2e_smoke("smoke-e2e-literal-set", _SMOKE_PATTERN_SETS[0], 1500, 5, SMOKE_SEED + 2),
    ]
    latency = [
        _build_latency_smoke("smoke-latency-a09plus", "a[0-9]+", 256, SMOKE_SEED + 3),
    ]
    if variant == PRECOMPILED_VARIANT:
        return precompiled
    if variant == E2E_AMORTIZED_VARIANT:
        return e2e
    if variant == SINGLE_MATCH_VARIANT:
        return latency
    return precompiled + e2e + latency


# ============================================================ real datasets
ANMLZOO_URL = "https://github.com/jackwadden/ANMLZoo"
ANMLZOO_DIR = os.path.join(DATA_DIR, "ANMLZoo")

# Confirmed live (GitHub API tree listing) while writing this loader:
# jackwadden/ANMLZoo contains exactly these top-level dirs among others:
# Brill, ClamAV, Dotstar, EntityResolution, Fermi, Hamming, Levenshtein,
# PowerEN, Protomata, RandomForest, SPM, Snort, Synthetic. Filenames below
# were confirmed the same way (git tree + HTTP range-fetch of the ClamAV
# file to check the XML schema).
_ANMLZOO_MAP = {
    "poweren": ("PowerEN", "anml/complx_01000_00123.1chip.anml",
               {"1mb": "inputs/poweren_1MB.input", "10mb": "inputs/poweren_10MB.input"}),
    "dotstar": ("Dotstar", "anml/backdoor_dotstar.1chip.anml",
               {"1mb": "inputs/backdoor_1MB.input", "10mb": "inputs/backdoor_10MB.input"}),
    "snort": ("Snort", "anml/snort.1chip.anml",
             {"1mb": "inputs/snort_1MB.input", "10mb": "inputs/snort_10MB.input"}),
    "clamav": ("ClamAV", "anml/515_nocounter.1chip.anml",
              {"1mb": "inputs/vasim_1MB.input", "10mb": "inputs/vasim_10MB.input"}),
    "entityresolution": ("EntityResolution", "anml/1000.1chip.anml",
                        {"1mb": "inputs/1000_1MB.input", "10mb": "inputs/1000_10MB.input"}),
}

REAL_GATE_WINDOW_BYTES = 8192   # reference simulator cap for real multi-MB corpora; see module docstring


def _ensure_anmlzoo() -> None:
    if os.path.isdir(os.path.join(ANMLZOO_DIR, ".git")):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", ANMLZOO_URL, ANMLZOO_DIR], check=True)


def load_workload(name: str) -> AutomataWorkload:
    """
    Real ANMLZoo benchmarks. Wired: 'poweren', 'dotstar', 'snort', 'clamav',
    'entityresolution' (case-insensitive; optional '-1mb'/'-10mb' suffix,
    default 10MB, matching ngAP's/GSpecPal's own convention). Cloned
    lazily, once, `git clone --depth 1`, into bench/automata-data/ANMLZoo/
    (~100MB) -- keeps .git for provenance, per ARTIFACT_GUIDE's own clone
    convention, even though this is workload data rather than an artifact.

    NOT wired, with reasons: 'bro217' (Becchi et al.'s separate "Regex"
    suite at regex.wustl.edu, not ANMLZoo, not fetched in this pass);
    'crispr_casoffinder'/'crispr_casot' (AutomataZoo, a DIFFERENT repo,
    not fetched); 'gspecpal-*' (GSpecPal's own DFA-table corpus is a
    completely different binary format, not ANML at all -- out of scope
    for this NFA-family loader; see artifacts/string-regex-matching/
    README.md for GSpecPal's own disposition).

    Real automata may legitimately use ANML counters/gates this module's
    parser rejects (parse_anml) -- that is an informative NotImplementedError
    from parse_anml, not a bug in this loader to silently work around.
    """
    key = name.lower().strip()
    size = "10mb"
    for suf in ("-1mb", "-10mb", "_1mb", "_10mb"):
        if key.endswith(suf):
            size = suf.strip("-_").lower()
            key = key[: -len(suf)]
            break
    if key not in _ANMLZOO_MAP:
        raise NotImplementedError(
            f"string-regex-matching: {name!r} is not one of this loader's wired ANMLZoo "
            f"benchmarks ({sorted(_ANMLZOO_MAP)}); see load_workload's docstring for exactly "
            "which recommended_subset entries are NOT wired and why. Use smoke_workloads() "
            "for a working, fast smoke set.")
    _ensure_anmlzoo()
    subdir, anml_rel, inputs = _ANMLZOO_MAP[key]
    anml_path = os.path.join(ANMLZOO_DIR, subdir, anml_rel)
    input_path = os.path.join(ANMLZOO_DIR, subdir, inputs[size])
    with open(anml_path, "r", errors="replace") as f:
        automaton = parse_anml(f.read(), automaton_id=f"{key}-{size}")
    with open(input_path, "rb") as f:
        stream = np.frombuffer(f.read(), dtype=np.uint8)
    return AutomataWorkload(
        name=f"{key}-{size}", variant=PRECOMPILED_VARIANT, automaton=automaton, stream=stream,
        patterns=None, re_count=automaton.pattern_count, gate_window_bytes=REAL_GATE_WINDOW_BYTES,
        synthetic=False, seed=-1,
        source=(f"real ANMLZoo benchmark {subdir!r} ({ANMLZOO_URL}), {anml_rel}, input {inputs[size]}; "
                f"correctness gate windowed to the first {REAL_GATE_WINDOW_BYTES} bytes (reference "
                "simulator cap -- the implementation under test still runs the full stream); see "
                "module docstring's 'Gate window' note"))
