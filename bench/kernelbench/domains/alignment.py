"""
sequence-alignment: pairwise dynamic-programming DNA/protein alignment
(Smith-Waterman local / Needleman-Wunsch global, exact-optimal or
X-drop-heuristic).

benchspecs/sequence-alignment/spec.yaml has FOUR variants; this module
implements the two pure-kernel ones and deliberately does NOT implement the
two end-to-end pipeline ones:

  * seqalign-exact-pairwise-kernel  -- IMPLEMENTED. Exact optimal pairwise
    alignment (bit-exact score vs. an independent O(n*m) reference), local
    (Smith-Waterman) or global (Needleman-Wunsch), linear or affine gap.
    Primary metric: GCUPS_nominal = n*m/time over the FULL (unpruned) matrix.
  * seqalign-xdrop-heuristic-kernel -- IMPLEMENTED. X-drop banded local
    heuristic; gated against an independent X-drop reference (NOT the
    unbounded exact optimum -- X-drop is a deliberate approximation, per the
    spec's own correctness text). Primary metric: nominal-window GCUPS over
    a (2X+1)-wide band.
  * seqalign-e2e-short-read-mapping     -- PLANNED-in-module, NOT implemented.
  * seqalign-e2e-whole-genome-alignment -- PLANNED-in-module, NOT implemented.
    Both e2e variants wrap a pairwise-alignment kernel inside a much larger
    seed/index -> filter -> extend pipeline (BWA-MEM-class read mapping;
    LASTZ-class whole-genome alignment) with their own index-build phase,
    bit-exact-vs-BWA-MEM or superset-vs-LASTZ correctness bars, and
    human/genome-scale real datasets -- a materially different, much bigger
    undertaking than the DP kernel this module implements (no index
    structure, no seed/filter stage, no CIGAR/SAM output exists here at
    all). `smoke_workloads(variant=...)` and `load_workload(name)` both
    raise a clear `NotImplementedError` naming these two variant ids rather
    than silently returning something unrelated. KERNELS/PLANNED are a
    per-KERNEL, not per-VARIANT, contract (DOMAIN_GUIDE.md); this is the
    same situation ann-search's two-variants-one-kernel module is in, just
    with "not implemented" instead of "implemented differently."

Scoring model (read literally from spec.yaml's `scoring` fields):
  * dna_linear  : match=+2, mismatch=-1, gap=-1            (AnySeq's disclosed
                  scheme -- the only fully-disclosed DNA scheme in the track)
  * dna_affine  : match=+2, mismatch=-1, gap_open=-2, gap_extend=-1
  * protein     : BLOSUM62 24x24 (here: the 20-standard-amino-acid 20x20
                  core; ambiguity codes B/Z/X/* are never emitted by the
                  synthetic generator, matching futhark-mem-sc22's own
                  generator per the spec's open_questions), linear gap
                  penalty = 10 (Rodinia/futhark-mem-sc22 convention: a gap
                  costs -10 regardless of run length, i.e. gap_open==
                  gap_extend==-10 in this module's unified affine
                  representation, which correctly degenerates to linear gap
                  when the two are equal -- see _numpy_wavefront_affine).
  Gap-run cost convention (Gotoh): a run of L gap characters costs
  `gap_open + (L-1)*gap_extend` (open pays for the first character, extend
  for each additional one); linear gap is the gap_open==gap_extend
  specialization, so ONE recurrence (with E/F matrices) implements both.
  alignment_type: "local" (Smith-Waterman, H floored at 0) or "global"
  (Needleman-Wunsch, no floor) per workload; X-drop is neither -- see below.

Correctness gate (spec: `tolerance` parses to None for BOTH implemented
variants -- both correctness texts hit spec.py's structural markers,
"bit-exact"/"exact match" -- so this is a STRUCTURAL gate, not a numeric
tolerance): CORRECTNESS_MODE = "exact" for this kernel. The canonical result
gated is the ARRAY OF PER-PAIR OPTIMAL SCORES (float64, exact integers for
this track's integer scoring schemes) -- never the alignment itself, which
is non-unique (many optimal tracebacks can share one optimal score, and the
spec's own correctness text is a SCORE match, not a traceback match).
`harness.check_correctness(mode="exact")` does `np.array_equal`, which is
exact for these integer-valued float64 arrays (no rounding within the
representable range at these sequence lengths).

Reference independence (DOMAIN_GUIDE.md's audit ruling): the reference
(`_reference_affine_score` / `_reference_xdrop_score`) is a plain row-major
(exact variant) or antidiagonal-batch (X-drop variant) Python loop over
dict/list state -- NEVER calling, importing, or sharing a code object with
`NumpyWavefrontAlignment`'s vectorized antidiagonal NumPy implementation
below (`_numpy_wavefront_affine` / `_numpy_wavefront_xdrop`). The two
implementations share the mathematical FORMULA (same Gotoh recurrence /
same X-drop rule, necessarily, or the gate could never pass for a correct
CPU implementation) but not the code, exactly the DOMAIN_GUIDE-sanctioned
distinction ("same formula re-typed inline is acceptable; same code object
is not"). A mutation test (flip the mismatch-vs-match branch in the
vectorized implementation) was run to confirm the gate actually rejects a
broken implementation rather than passing vacuously -- see this module's
integration report for the result.

X-drop semantics -- a DELIBERATE CHOICE, because the spec leaves exact X-drop
banding implementation-defined (its own open_questions admit LOGAN's exact
scoring/banding rule was not recoverable from accessible sources). This
module treats X-drop as a SEEDED banded EXTENSION from the sequence origin
(0,0) -- NOT a free-restart local search like Smith-Waterman -- using the
floor-free recurrence H[i,j] = max(H[i-1,j-1]+sigma, H[i-1,j]+gap,
H[i,j-1]+gap) (linear gap only; the spec's own X-drop scoring is linear-gap
DNA, and X-drop's literature definition has no affine variant here), where a
cell is PRUNED (treated unreachable) once its value falls more than X below
the best score seen on any EARLIER antidiagonal. Antidiagonals are processed
as a BATCH: the running-best used to prune antidiagonal d is fixed BEFORE d
is computed and only updated once d is entirely finished -- this specific
granularity (not a cell-by-cell running max within one antidiagonal) is what
makes the scalar reference and the vectorized implementation deterministic
and mutually agreeing regardless of which one processes cells "first" within
an antidiagonal. This is the standard Zhang/Miller X-drop definition used by
BLAST-family tools, applied here to a full pairwise input (no separate
seeding stage exists in this benchmark) rather than to a real seed hit.

Cost model (GCUPS, per spec's own literal per-variant convention):
  * exact:  n*m cells per pair (FULL unpruned matrix), summed over the batch
            -- "GCUPS_nominal", so pruning methods (paper artifacts) stay
            comparable to unpruned ones on the SAME denominator.
  * x-drop: (2X+1) * min(n,m) cells per pair -- "nominal-window GCUPS",
            assuming the full band is traversed, per spec.yaml literally.
  Both registered under one unit, "GCUPS" (this track's literal metric
  name), following stencil.py's GCUP/s and graph.py's GTEPS precedent for
  non-FLOP units.

CPU_IMPLS: NumpyWavefrontAlignment, a genuinely vectorized ANTIDIAGONAL
("wavefront") NumPy implementation -- every cell in one antidiagonal depends
only on the PRIOR one or two antidiagonals (never on another cell of the
same antidiagonal), so a whole antidiagonal's H/E/F values are computed as
one batch of array ops instead of a per-cell Python loop. This is the CPU
floor for both variants; it is deliberately NOT the fastest possible CPU
implementation (that would defeat the reference-independence point).

cuda_impls(): returns {} -- NO GPU implementation is shipped by this module
(same documented, honest choice sparse.py makes for sptrsv and graph.py
makes for every graph kernel: "a reasonable GPU implementation needs either
a real device-resident kernel this repo doesn't have, or a heavyweight
dependency it doesn't vendor"). This choice is especially well-motivated
here: the two variants' natural GPU competitors are literally the paper
artifacts this track integrates in Part B -- MASA-CUDAlign
(bench/artifacts/sequence-alignment/masa-cudalign/) for
seqalign-exact-pairwise-kernel, LOGAN (bench/artifacts/sequence-alignment/
logan/) for seqalign-xdrop-heuristic-kernel -- so a from-scratch GPU
baseline here would be strictly redundant with what this same integration
pass already wires up.

Workloads: `AlignmentWorkload` holds a BATCH of pairs (one or more (seq1,
seq2) strings sharing one scoring scheme / alignment_type / x_drop), matching
the spec's own "pairs (or batches of pairs)" framing and its 12.5M-pair
short-read-batch convention (a throughput number over many pairs in one
timed call, not a per-pair call). `smoke_workloads()`: small seeded random
DNA/protein pairs PLUS, for every batch, at least one pair built by
mutating a copy of the other sequence (point substitutions + small indels)
so the optimal alignment is non-trivial (not all-mismatch) -- see
`_mutate_seq`. `load_workload(name)`: wires (a) three real NCBI accession
pairs from the spec's own dna_long_small_medium recommended_subset, BOUNDED
to a small prefix of each accession (via NCBI eutils' own seq_start/seq_stop
range parameters -- this never downloads a full multi-Mbp/Gbp chromosome,
literally, not just "downloads it but only uses part") because this
module's own O(n*m) reference/CPU implementation could not run full-scale
exact DP on a login node in any amount of time (the spec's own scale is up
to 50 Mbp/side, i.e. up to ~10^15 cells) -- full-scale timing on these
accessions is exactly what the MASA-CUDAlign artifact adapter is for; (b)
futhark-mem-sc22's own synthetic protein-tier sizes (8192/16384/32768,
generated locally, no download). Everything else (the short-read Mason
batch, LOGAN's E. coli/C. elegans real tier, the large chromosome-scale
optional tier, and both e2e variants' own datasets) raises a clear,
itemized `NotImplementedError`, same honest-refusal pattern as
annsearch.py's `load_workload`.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np

from .. import workload
from ..harness import Timer

KERNELS = ["sequence-alignment"]
# The KERNEL is implemented; two of its four spec VARIANTS are not (see
# module docstring). KERNELS/PLANNED is a per-kernel contract
# (DOMAIN_GUIDE.md), so PLANNED stays empty here -- the two unsupported
# variant ids are surfaced instead via _UNSUPPORTED_VARIANTS below, raised
# with a specific reason wherever a caller asks for one by id.
PLANNED: list[str] = []

EXACT_VARIANT = "seqalign-exact-pairwise-kernel"
XDROP_VARIANT = "seqalign-xdrop-heuristic-kernel"
E2E_SHORT_READ_VARIANT = "seqalign-e2e-short-read-mapping"
E2E_WGA_VARIANT = "seqalign-e2e-whole-genome-alignment"

_UNSUPPORTED_VARIANTS = {
    E2E_SHORT_READ_VARIANT: (
        "seqalign-e2e-short-read-mapping is an end-to-end short-read mapping "
        "pipeline (one-shot index build + BWA-MEM-class seed/chain/extend "
        "over a real human-scale WGS read set, gated bit-exact vs. BWA-MEM's "
        "own SAM/BAM output) -- not implemented by this module, which only "
        "implements the pairwise DP kernel. See module docstring."),
    E2E_WGA_VARIANT: (
        "seqalign-e2e-whole-genome-alignment is an end-to-end whole-genome "
        "seed->GPU-filter->LASTZ-extend pipeline over real UCSC genome "
        "assemblies, gated as a superset of LASTZ's own alignment set -- not "
        "implemented by this module, which only implements the pairwise DP "
        "kernel. See module docstring."),
}

DATA_DIR = os.environ.get(
    "KERNELBENCH_SEQALIGN_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "seqalign-data"),
)
DATA_DIR = os.path.normpath(DATA_DIR)

NEG_INF = -1e18  # finite sentinel: avoids NaN from inf-inf in threshold math,
                 # large enough to never be chosen by max() at these scales.


# ============================================================== scoring
@dataclass
class Scoring:
    """
    A substitution matrix + gap model. DNA schemes build a trivial KxK
    match/mismatch matrix (K=len(alphabet)); the protein scheme carries the
    real BLOSUM62 table. Affine gap: a run of L gap characters costs
    `gap_open + (L-1)*gap_extend`; linear gap is the gap_open==gap_extend
    case of the exact same formula (cost = L*gap_open), so one Gotoh
    recurrence (E/F matrices) implements both without a separate code path.
    """

    name: str
    alphabet: str
    substitution: np.ndarray  # [K, K] float64
    gap_open: float
    gap_extend: float

    def __post_init__(self) -> None:
        self._idx = {c: i for i, c in enumerate(self.alphabet)}

    def encode(self, seq: str) -> np.ndarray:
        return np.array([self._idx[c] for c in seq], dtype=np.int64)

    def sigma(self, a: str, b: str) -> float:
        return float(self.substitution[self._idx[a], self._idx[b]])


def _dna_scoring(name: str, match: float, mismatch: float,
                 gap_open: float, gap_extend: float, alphabet: str = "ACGT") -> Scoring:
    k = len(alphabet)
    sub = np.full((k, k), mismatch, dtype=np.float64)
    np.fill_diagonal(sub, match)
    return Scoring(name=name, alphabet=alphabet, substitution=sub,
                   gap_open=gap_open, gap_extend=gap_extend)


DNA_LINEAR = _dna_scoring("dna_linear", 2.0, -1.0, -1.0, -1.0)
DNA_AFFINE = _dna_scoring("dna_affine", 2.0, -1.0, -2.0, -1.0)

# 20 standard amino acids, standard BLOSUM ordering. futhark-mem-sc22's own
# synthetic generator is assumed (per the spec's own open_questions) to only
# ever emit these 20 symbols -- ambiguity codes (B/Z/X/*) are not modeled.
PROTEIN_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62 = np.array([
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
], dtype=np.float64)
PROTEIN_BLOSUM62 = Scoring(name="protein_blosum62", alphabet=PROTEIN_ALPHABET,
                           substitution=_BLOSUM62, gap_open=-10.0, gap_extend=-10.0)


# ============================================================== workload
@dataclass
class AlignmentWorkload:
    name: str
    variant: str                # EXACT_VARIANT | XDROP_VARIANT
    seqs1: list[str]
    seqs2: list[str]
    alignment_type: str         # "local" | "global" (X-drop is neither -- see docstring)
    scoring: Scoring
    x_drop: int | None          # XDROP_VARIANT only
    synthetic: bool
    seed: int
    source: str

    def describe(self) -> dict:
        d = {
            "name": self.name, "kernel": "sequence-alignment",
            "spec_variant": self.variant, "n_pairs": len(self.seqs1),
            "len1": [len(s) for s in self.seqs1],
            "len2": [len(s) for s in self.seqs2],
            "alignment_type": self.alignment_type,
            "scoring": self.scoring.name,
            "gap_open": self.scoring.gap_open, "gap_extend": self.scoring.gap_extend,
            "synthetic": self.synthetic, "seed": self.seed, "source": self.source,
        }
        if self.variant == XDROP_VARIANT:
            d["x_drop"] = self.x_drop
        return d


# ----------------------------------------------------------------- cost rule
def _cost_alignment(w: AlignmentWorkload, params: dict) -> tuple[int, int]:
    """
    GCUPS per spec.yaml's own literal, per-variant convention (see module
    docstring): exact -> n*m per pair (full unpruned matrix); x-drop ->
    (2X+1)*min(n,m) per pair ("nominal-window", band length taken as the
    number of antidiagonal steps before the shorter sequence is exhausted).
    Both summed over the workload's batch. `bytes`: read each sequence once
    (1 byte/base or /residue) + one small score output per pair -- the
    compulsory-traffic lower bound, never a measured figure.
    """
    total_cells = 0
    total_bytes = 0
    for s1, s2 in zip(w.seqs1, w.seqs2):
        n, m = len(s1), len(s2)
        if w.variant == XDROP_VARIANT:
            x = w.x_drop or 0
            total_cells += (2 * x + 1) * min(n, m)
        else:
            total_cells += n * m
        total_bytes += n + m + 8
    return int(total_cells), int(total_bytes)


workload.register_cost("sequence-alignment", _cost_alignment, "GCUPS")


# ----------------------------------------------------------------- reference
def _reference_affine_score(seq1: str, seq2: str, scoring: Scoring, mode: str) -> float:
    """
    Independent reference for the EXACT variant: Gotoh's algorithm (H/E/F),
    plain row-major Python double loop over explicit list state -- NEVER
    calling or sharing a code object with NumpyWavefrontAlignment's
    antidiagonal-vectorized implementation below (DOMAIN_GUIDE.md's
    reference-independence audit ruling). `mode`: "local" (Smith-Waterman,
    H floored at 0 everywhere, including the boundary row/col, which is set
    to exactly 0) or "global" (Needleman-Wunsch, boundary row/col set to the
    explicit gap-run cost formula, no floor anywhere).
    """
    n, m = len(seq1), len(seq2)
    go, ge = scoring.gap_open, scoring.gap_extend
    H = [[0.0] * (m + 1) for _ in range(n + 1)]
    E = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    F = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    if mode == "global":
        for j in range(1, m + 1):
            H[0][j] = go + (j - 1) * ge
        for i in range(1, n + 1):
            H[i][0] = go + (i - 1) * ge
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            e = max(H[i][j - 1] + go, E[i][j - 1] + ge)
            f = max(H[i - 1][j] + go, F[i - 1][j] + ge)
            d = H[i - 1][j - 1] + scoring.sigma(seq1[i - 1], seq2[j - 1])
            raw = max(d, e, f)
            h = max(raw, 0.0) if mode == "local" else raw
            H[i][j] = h
            E[i][j] = e
            F[i][j] = f
            if mode == "local" and h > best:
                best = h
    return best if mode == "local" else H[n][m]


def _reference_xdrop_score(seq1: str, seq2: str, scoring: Scoring, x_drop: int) -> float:
    """
    Independent reference for the X-DROP variant: the seeded, floor-free
    banded extension described in the module docstring, processed
    ANTIDIAGONAL BY ANTIDIAGONAL with dict-based state (deliberately NOT
    numpy -- different code from _numpy_wavefront_xdrop below). `best` is
    updated only after an entire antidiagonal is finished, using the RAW
    (pre-threshold) values computed for that antidiagonal -- see module
    docstring for why this exact granularity is required for the reference
    and the vectorized implementation to ever agree.
    """
    n, m = len(seq1), len(seq2)
    gap = scoring.gap_open  # linear only (gap_open == gap_extend), per spec
    H: dict[tuple[int, int], float] = {(0, 0): 0.0}
    best = 0.0
    for d in range(1, n + m + 1):
        cur: dict[tuple[int, int], float] = {}
        for i in range(max(0, d - m), min(n, d) + 1):
            j = d - i
            diag = (H.get((i - 1, j - 1), NEG_INF) + scoring.sigma(seq1[i - 1], seq2[j - 1])) \
                if (i > 0 and j > 0) else NEG_INF
            up = (H.get((i - 1, j), NEG_INF) + gap) if i > 0 else NEG_INF
            left = (H.get((i, j - 1), NEG_INF) + gap) if j > 0 else NEG_INF
            cur[(i, j)] = max(diag, up, left)
        if not cur:
            continue
        threshold = best - x_drop
        for key, val in cur.items():
            H[key] = val if val >= threshold else NEG_INF
        best = max(best, max(cur.values()))
    return best


def reference_alignment(w: AlignmentWorkload, params: dict):
    scores = np.empty(len(w.seqs1), dtype=np.float64)
    for k, (s1, s2) in enumerate(zip(w.seqs1, w.seqs2)):
        if w.variant == XDROP_VARIANT:
            scores[k] = _reference_xdrop_score(s1, s2, w.scoring, w.x_drop)
        else:
            scores[k] = _reference_affine_score(s1, s2, w.scoring, w.alignment_type)
    return scores


REFERENCES = {"sequence-alignment": reference_alignment}
REFERENCE_NAME = {
    "sequence-alignment": (
        "independent scalar Python DP: row-major Gotoh H/E/F for the exact "
        "variant, antidiagonal-batched seeded X-drop extension for the "
        "x-drop variant -- see module docstring"),
}
CORRECTNESS_MODE = {"sequence-alignment": "exact"}
DEFAULT_PRECISION = {"sequence-alignment": "int64"}
DIM_KEY: dict[str, str] = {}


# ------------------------------------------------------- vectorized (CPU/GPU)
def _numpy_wavefront_affine(idx1: np.ndarray, idx2: np.ndarray, substitution: np.ndarray,
                            go: float, ge: float, mode: str) -> float:
    """
    Vectorized antidiagonal Gotoh DP. Every cell (i,j) depends only on cells
    of antidiagonal d-1 or d-2 (d=i+j) -- never another cell of its own
    antidiagonal -- so one antidiagonal's H/E/F values are computed as a
    single batch of NumPy array ops. Padded (n+2)x(m+2) arrays let every
    real coordinate be read/written uniformly (padded index = real index +
    1); the single origin cell Hp[1,1]=0 is seeded before the loop, and
    EVERY other cell -- including the boundary row/col -- is produced by the
    exact same generic recurrence (no special-cased boundary formulas): for
    "local" mode this recurrence provably reduces the entire boundary to 0
    (gap penalties are always <= 0), and for "global" mode it provably
    reduces to the standard `go + (L-1)*ge` gap-run cost -- both verified
    against _reference_affine_score's explicit boundary formulas while
    building this module. See _reference_affine_score for the independent,
    non-vectorized implementation this is gated against.
    """
    n, m = idx1.shape[0], idx2.shape[0]
    Hp = np.full((n + 2, m + 2), NEG_INF, dtype=np.float64)
    Ep = np.full((n + 2, m + 2), NEG_INF, dtype=np.float64)
    Fp = np.full((n + 2, m + 2), NEG_INF, dtype=np.float64)
    Hp[1, 1] = 0.0
    best = 0.0
    for d in range(1, n + m + 1):
        i_arr = np.arange(max(0, d - m), min(n, d) + 1)
        j_arr = d - i_arr
        pi, pj = i_arr + 1, j_arr + 1
        i_pos, j_pos = i_arr > 0, j_arr > 0
        diag_valid = i_pos & j_pos
        i_safe = np.clip(i_arr - 1, 0, n - 1)
        j_safe = np.clip(j_arr - 1, 0, m - 1)
        sigma = substitution[idx1[i_safe], idx2[j_safe]]
        diagH = np.where(diag_valid, Hp[pi - 1, pj - 1] + sigma, NEG_INF)
        e_new = np.where(j_pos, np.maximum(Hp[pi, pj - 1] + go, Ep[pi, pj - 1] + ge), NEG_INF)
        f_new = np.where(i_pos, np.maximum(Hp[pi - 1, pj] + go, Fp[pi - 1, pj] + ge), NEG_INF)
        raw = np.maximum(diagH, np.maximum(e_new, f_new))
        h_new = np.maximum(raw, 0.0) if mode == "local" else raw
        Hp[pi, pj] = h_new
        Ep[pi, pj] = e_new
        Fp[pi, pj] = f_new
        if mode == "local" and h_new.size:
            best = max(best, float(h_new.max()))
    return best if mode == "local" else float(Hp[n + 1, m + 1])


def _numpy_wavefront_xdrop(idx1: np.ndarray, idx2: np.ndarray, substitution: np.ndarray,
                           gap: float, x_drop: int) -> float:
    """
    Vectorized antidiagonal X-drop extension -- same seeded, floor-free
    recurrence and same "threshold fixed before the antidiagonal, best
    updated after" granularity as _reference_xdrop_score, just computed as
    a batch of array ops per antidiagonal instead of a Python dict loop.
    """
    n, m = idx1.shape[0], idx2.shape[0]
    Hp = np.full((n + 2, m + 2), NEG_INF, dtype=np.float64)
    Hp[1, 1] = 0.0
    best = 0.0
    for d in range(1, n + m + 1):
        i_arr = np.arange(max(0, d - m), min(n, d) + 1)
        j_arr = d - i_arr
        pi, pj = i_arr + 1, j_arr + 1
        i_pos, j_pos = i_arr > 0, j_arr > 0
        diag_valid = i_pos & j_pos
        i_safe = np.clip(i_arr - 1, 0, n - 1)
        j_safe = np.clip(j_arr - 1, 0, m - 1)
        sigma = substitution[idx1[i_safe], idx2[j_safe]]
        diagH = np.where(diag_valid, Hp[pi - 1, pj - 1] + sigma, NEG_INF)
        upH = np.where(i_pos, Hp[pi - 1, pj] + gap, NEG_INF)
        leftH = np.where(j_pos, Hp[pi, pj - 1] + gap, NEG_INF)
        raw = np.maximum(diagH, np.maximum(upH, leftH))
        threshold = best - x_drop
        Hp[pi, pj] = np.where(raw >= threshold, raw, NEG_INF)
        if raw.size:
            best = max(best, float(raw.max()))
    return best


class NumpyWavefrontAlignment:
    """
    The CPU floor for both variants -- see module docstring for the
    vectorization strategy and the reference-independence argument.
    """

    name = "numpy-wavefront-align"
    platform = "cpu"

    def __init__(self, precision: str = "int64"):
        self.precision = precision

    def prepare(self, w: AlignmentWorkload, params: dict):
        self._w = w
        params["alignment_type"] = w.alignment_type
        params["scoring"] = w.scoring.name
        if w.variant == XDROP_VARIANT:
            params["x_drop"] = w.x_drop
        return [(w.scoring.encode(s1), w.scoring.encode(s2))
                for s1, s2 in zip(w.seqs1, w.seqs2)]

    def run(self, h):
        w = self._w
        scores = np.empty(len(h), dtype=np.float64)
        for k, (idx1, idx2) in enumerate(h):
            if w.variant == XDROP_VARIANT:
                scores[k] = _numpy_wavefront_xdrop(
                    idx1, idx2, w.scoring.substitution, w.scoring.gap_open, w.x_drop)
            else:
                scores[k] = _numpy_wavefront_affine(
                    idx1, idx2, w.scoring.substitution,
                    w.scoring.gap_open, w.scoring.gap_extend, w.alignment_type)
        return scores

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        return Timer()

    def free(self, h):
        h.clear()


CPU_IMPLS = {"sequence-alignment": {"numpy-wavefront-align": NumpyWavefrontAlignment}}


def cuda_impls():
    """
    No CUDA implementation shipped BY THIS MODULE -- see module docstring
    for the (especially well-motivated, here) reasoning: the paper artifacts
    integrated in Part B of this track's own integration pass (MASA-CUDAlign,
    LOGAN) are literally the GPU competitors for these two variants, so a
    from-scratch GPU baseline would be redundant with work this same pass
    already does. Same documented, honest choice as sparse.py's sptrsv and
    every graph.py kernel.
    """
    return {}


# ============================================================== smoke set
SMOKE_SEED = 20260905


def _random_seq(rng: np.random.Generator, n: int, alphabet: str) -> str:
    return "".join(rng.choice(list(alphabet), size=n))


def _mutate_seq(rng: np.random.Generator, seq: str, sub_rate: float = 0.12,
                indel_rate: float = 0.04, alphabet: str = "ACGT") -> str:
    """
    Point substitutions + small insertions/deletions from a common-ancestor
    sequence, so the two sequences in a pair share genuine local similarity
    and the optimal alignment is non-trivial (not all-mismatch) -- the
    "pair with planted similarity" the task brief asks for.
    """
    alpha_list = list(alphabet)
    out: list[str] = []
    for c in seq:
        r = rng.random()
        if r < indel_rate / 2:
            continue  # deletion: drop this character
        if r < indel_rate:
            out.append(str(rng.choice(alpha_list)))  # insertion before this character
            out.append(c)
        elif r < indel_rate + sub_rate:
            choices = [a for a in alpha_list if a != c]
            out.append(str(rng.choice(choices)))
        else:
            out.append(c)
    return "".join(out)


def _make_exact_smoke() -> list[AlignmentWorkload]:
    rng = np.random.default_rng(SMOKE_SEED)
    out = []

    s1a, s2a = _random_seq(rng, 80, "ACGT"), _random_seq(rng, 85, "ACGT")
    s1b = _random_seq(rng, 90, "ACGT")
    s2b = _mutate_seq(rng, s1b)
    out.append(AlignmentWorkload(
        name="smoke-dna-local-linear", variant=EXACT_VARIANT,
        seqs1=[s1a, s1b], seqs2=[s2a, s2b], alignment_type="local",
        scoring=DNA_LINEAR, x_drop=None, synthetic=True, seed=SMOKE_SEED,
        source="synthetic seeded random DNA (smoke, NOT spec-conforming): "
               "pair 0 fully random (near-zero optimal score expected), "
               "pair 1 planted similarity (point mutations + small indels "
               "from a common ancestor, non-trivial optimal alignment)"))

    s1c = _random_seq(rng, 60, "ACGT")
    s2c = _mutate_seq(rng, s1c, sub_rate=0.15, indel_rate=0.0)  # no indels: clean global-alignment case
    out.append(AlignmentWorkload(
        name="smoke-dna-global-linear", variant=EXACT_VARIANT,
        seqs1=[s1c], seqs2=[s2c], alignment_type="global",
        scoring=DNA_LINEAR, x_drop=None, synthetic=True, seed=SMOKE_SEED + 1,
        source="synthetic seeded random DNA, point-mutated pair (smoke, NOT "
               "spec-conforming); global (Needleman-Wunsch) alignment"))

    s1d = _random_seq(rng, 70, "ACGT")
    s2d = _mutate_seq(rng, s1d, sub_rate=0.10, indel_rate=0.08)
    out.append(AlignmentWorkload(
        name="smoke-dna-local-affine", variant=EXACT_VARIANT,
        seqs1=[s1d], seqs2=[s2d], alignment_type="local",
        scoring=DNA_AFFINE, x_drop=None, synthetic=True, seed=SMOKE_SEED + 2,
        source="synthetic seeded random DNA with indels (smoke, NOT "
               "spec-conforming); exercises the affine gap_open/gap_extend "
               "path (Gotoh E/F recurrence, gap_open != gap_extend)"))

    s1e = _random_seq(rng, 40, PROTEIN_ALPHABET)
    s2e = _mutate_seq(rng, s1e, sub_rate=0.15, indel_rate=0.05, alphabet=PROTEIN_ALPHABET)
    out.append(AlignmentWorkload(
        name="smoke-protein-local-blosum62", variant=EXACT_VARIANT,
        seqs1=[s1e], seqs2=[s2e], alignment_type="local",
        scoring=PROTEIN_BLOSUM62, x_drop=None, synthetic=True, seed=SMOKE_SEED + 3,
        source="synthetic seeded random protein (smoke, NOT spec-conforming); "
               "futhark-mem-sc22's own protein-tier scoring "
               "(BLOSUM62 + linear gap=10)"))
    return out


def _make_xdrop_smoke() -> list[AlignmentWorkload]:
    rng = np.random.default_rng(SMOKE_SEED + 10)
    out = []
    s1 = _random_seq(rng, 120, "ACGT")
    s2 = _mutate_seq(rng, s1, sub_rate=0.15, indel_rate=0.02)
    out.append(AlignmentWorkload(
        name="smoke-xdrop-X50", variant=XDROP_VARIANT,
        seqs1=[s1], seqs2=[s2], alignment_type="local",
        scoring=DNA_LINEAR, x_drop=50, synthetic=True, seed=SMOKE_SEED + 10,
        source="synthetic seeded ~15%-error-rate long-read pair (LOGAN's own "
               "methodology, tiny smoke scale), X=50"))

    s3 = _random_seq(rng, 100, "ACGT")
    s4 = _random_seq(rng, 100, "ACGT")
    out.append(AlignmentWorkload(
        name="smoke-xdrop-X100", variant=XDROP_VARIANT,
        seqs1=[s3], seqs2=[s4], alignment_type="local",
        scoring=DNA_LINEAR, x_drop=100, synthetic=True, seed=SMOKE_SEED + 11,
        source="synthetic seeded fully-random DNA pair (near-zero expected "
               "score), X=100 -- LOGAN's own single REQUIRED reported point"))
    return out


def smoke_workloads(variant: str | None = None) -> list[AlignmentWorkload]:
    """
    Small, synthetic, runs anywhere in seconds. NOT spec-conforming.
    `variant=None` (every zero-arg caller, matching every other domain's
    convention) returns both variants' workloads together -- runner.py
    always overrides warmup/reps to fixed smoke defaults regardless of
    --variant (see annsearch.py's identical note), and each AlignmentWorkload
    carries its own `.variant`, so gating is always correct regardless of
    which variant id was passed on the command line.
    """
    if variant in _UNSUPPORTED_VARIANTS:
        raise NotImplementedError(_UNSUPPORTED_VARIANTS[variant])
    exact = _make_exact_smoke()
    xdrop = _make_xdrop_smoke()
    if variant == EXACT_VARIANT:
        return exact
    if variant == XDROP_VARIANT:
        return xdrop
    return exact + xdrop


# ============================================================ real datasets
_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_BOUNDED_PREFIX_BP = 1500  # see module docstring: bounded so THIS module's
                           # own O(n*m) reference/CPU impl stays tractable.

_NCBI_ACCESSIONS = {
    "nc_000962.3": ("NC_000962.3", "M. tuberculosis H37Rv"),
    "nc_000913.3": ("NC_000913.3", "E. coli K12 MG1655"),
    "nt_033779.4": ("NT_033779.4", "D. melanogaster chr2L"),
    "ba000046.3": ("BA000046.3", "P. troglodytes chr22"),
    "nc_019481.1": ("NC_019481.1", "O. aries breed Texel chr24"),
    "nc_019478.1": ("NC_019478.1", "O. aries breed Texel chr21"),
}
# Spec's own "3 pairs of roughly similar length" pairing (survey.md's AnySeq
# entry): smallest-with-smallest, ..., largest-with-largest.
_REAL_PAIRS = [
    ("mtb-vs-ecoli", "nc_000962.3", "nc_000913.3"),
    ("fly-vs-chimp", "nt_033779.4", "ba000046.3"),
    ("sheep-chr24-vs-chr21", "nc_019481.1", "nc_019478.1"),
]
_PROTEIN_SIZES = {"protein-8192": 8192, "protein-16384": 16384, "protein-32768": 32768}


def _fetch_ncbi_prefix(accession: str, n_bp: int) -> str:
    """
    BOUNDED direct download: NCBI eutils' own seq_start/seq_stop range
    parameters mean only the first `n_bp` bases are ever transferred --
    this never pulls the full chromosome over the wire, not merely "uses
    only part of what was downloaded." Cached under DATA_DIR.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"{accession}_{n_bp}bp.fasta")
    if not os.path.exists(cache):
        url = (f"{_EUTILS}?db=nuccore&id={accession}&rettype=fasta&retmode=text"
               f"&seq_start=1&seq_stop={n_bp}")
        urllib.request.urlretrieve(url, cache)
    with open(cache) as f:
        lines = f.read().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">")).upper()


def _load_real_pair(pair_name: str) -> AlignmentWorkload:
    _, k1, k2 = next(p for p in _REAL_PAIRS if p[0] == pair_name)
    acc1, label1 = _NCBI_ACCESSIONS[k1]
    acc2, label2 = _NCBI_ACCESSIONS[k2]
    seq1 = _fetch_ncbi_prefix(acc1, _BOUNDED_PREFIX_BP)
    seq2 = _fetch_ncbi_prefix(acc2, _BOUNDED_PREFIX_BP)
    return AlignmentWorkload(
        name=pair_name, variant=EXACT_VARIANT, seqs1=[seq1], seqs2=[seq2],
        alignment_type="global", scoring=DNA_LINEAR, x_drop=None,
        synthetic=False, seed=-1,
        source=(f"real NCBI accessions {acc1} ({label1}) x {acc2} ({label2}), "
                f"BOUNDED to the first {_BOUNDED_PREFIX_BP} bp of each via "
                "NCBI eutils efetch seq_start/seq_stop (never transfers the "
                "full multi-Mbp chromosome) -- spec's own dna_long_small_"
                "medium recommended_subset pairing, dna_linear scoring "
                "(AnySeq's disclosed scheme). Bounded because this module's "
                "own O(n*m) reference/CPU implementation cannot run "
                "full-chromosome exact DP on a login node at all (spec scale "
                "is up to 50 Mbp/side, ~10^15 cells); full-scale timing on "
                "these accessions is what the MASA-CUDAlign artifact adapter "
                "is for (bench/artifacts/sequence-alignment/masa-cudalign/)."))


def _load_protein_synthetic(key: str) -> AlignmentWorkload:
    n = _PROTEIN_SIZES[key]
    rng = np.random.default_rng(SMOKE_SEED + n)
    s1 = _random_seq(rng, n, PROTEIN_ALPHABET)
    s2 = _random_seq(rng, n, PROTEIN_ALPHABET)
    return AlignmentWorkload(
        name=key, variant=EXACT_VARIANT, seqs1=[s1], seqs2=[s2],
        alignment_type="global", scoring=PROTEIN_BLOSUM62, x_drop=None,
        synthetic=True, seed=SMOKE_SEED + n,
        source=(f"synthetic random protein sequences, length {n} "
                "(futhark-mem-sc22's own row_length convention, "
                "benchmarks/nw/futhark/nw.fut's compiled-script header); "
                "BLOSUM62 + linear gap=10, global (Needleman-Wunsch), "
                "matching the Rodinia `nw` reference this paper benchmarks "
                f"against. WARNING: this module's O(n*m) reference/CPU "
                f"implementation at n={n} is {n * n:,} cells -- runs but is "
                "slow (minutes) on a login node; NOT exercised by --smoke."))


def load_workload(name: str) -> AlignmentWorkload:
    key = name.lower().strip()
    if key in _PROTEIN_SIZES:
        return _load_protein_synthetic(key)
    if key in {p[0] for p in _REAL_PAIRS}:
        return _load_real_pair(key)
    if key in _UNSUPPORTED_VARIANTS:
        raise NotImplementedError(_UNSUPPORTED_VARIANTS[key])
    raise NotImplementedError(
        f"sequence-alignment: {name!r} is not a wired workload. Wired real: "
        f"{[p[0] for p in _REAL_PAIRS]} (bounded {_BOUNDED_PREFIX_BP}bp-per-"
        "side NCBI prefixes of the spec's dna_long_small_medium pairing). "
        f"Wired synthetic: {sorted(_PROTEIN_SIZES)} (futhark-mem-sc22's own "
        "protein tier, generated locally, no download). NOT wired, with "
        "reasons: dna_long_large_optional (human chr1 x chr21, 249/228 Mbp "
        "-- multi-GB-scale, explicitly out of scope for a login-node "
        "loader); dna_short_read_batch (12.5M Mason-simulated 150bp reads -- "
        "Mason is not installed/available in this environment, and the "
        "exact Mason version/seed is unrecoverable per the spec's own "
        "open_questions); LOGAN's E. coli/C. elegans real-data tier (exact "
        "accession/download source not confirmed, per the spec's own "
        "open_questions); SegAlign's UCSC genome assemblies and both e2e "
        "variants' own datasets (both e2e variants are PLANNED-in-module, "
        "not implemented at all -- see module docstring). Use "
        "smoke_workloads() for a working, fast smoke set.")
