"""
LOGAN adapter for the sequence-alignment track's seqalign-xdrop-heuristic-kernel
variant.

Paper: "LOGAN: High-Performance GPU-Based X-Drop Long-Read Alignment" (IPDPS'20).
PAPER_KEY = "conf/ipps/ZeniGEDSHBOY20". Artifact: https://github.com/albertozeni/LOGAN,
commit 336907643a8404798ceaa33c7f0db05fcce6e030.

What is wrapped
----------------
LOGAN's own public library entry point, `extendSeedL()` (src/logan_functions.cuh) --
the SAME function the upstream `demo` binary calls. This is NOT the shipped
`demo` CLI: reading demo.cu shows it computes scores into a local `int* res`
buffer inside its LOGAN() helper and `free(res)` them at the end of every
batch WITHOUT ever printing or returning a single score -- only one
aggregate wall-clock number reaches stdout. There is therefore no way to
observe a per-pair alignment score through the shipped binary at all.

Per ARTIFACT_GUIDE.md rule 1 ("wrap at the finest boundary available"),
this integration adds `source/src/kernelbench_driver.cu` -- a NEW,
ADDITIVE file (no upstream LOGAN source file is modified; `git -C source
status`/`diff` show only this one untracked addition) that calls
extendSeedL() directly and writes one score per line to an output file.
See that file's header comment for details.

Two significant findings from reading (and empirically confirming against)
LOGAN's own kernel source, both material to how this adapter's gate must be
read -- see STATUS.md for the full evidence and the empirical confirmation:

1. LOGAN's compiled GPU kernel (logan_functions.cu:computeAntidiag) scores
   matches/mismatches/gaps using the compile-time macros `MATCH`/`MISMATCH`/
   `GAP_EXT` (#define'd to 1/-1/-1 in logan_functions.cuh) DIRECTLY --
   the `ScoringSchemeL` struct passed through the public `extendSeedL()` API
   (and threaded down to this driver's own `match`/`mismatch`/`gap_ext`/
   `gap_open` CLI arguments) is read ONLY for a host-side validity check
   (gap_extend_score/gap_open_score must be < 0) and is NEVER passed into
   any kernel launch or read by the DP recurrence. Empirically confirmed:
   requesting match=2 (this benchmark's dna_linear scoring) on a 20-char
   all-match pair returns 20 (=20*1, the hardcoded MATCH), not 40. LOGAN's
   alignment score is therefore ALWAYS computed under a hardcoded
   match=+1/mismatch=-1/gap=-1 scheme, regardless of what this adapter (or
   any caller) requests.
2. The `direction` parameter to `extendSeedL()` is similarly dead code: the
   host function unconditionally launches BOTH the EXTEND_LEFTL and
   EXTEND_RIGHTL kernels on every call (logan_functions.cu, the two
   `extendSeedLGappedXDropOneDirectionGlobal<<<...>>>` launches use the
   hardcoded literals EXTEND_LEFTL/EXTEND_RIGHTL, not the `direction`
   argument at all). LOGAN's real algorithm is therefore always a
   BIDIRECTIONAL seed-anchored extension (extend both ways from a k-mer
   seed placed somewhere inside the sequence), structurally different from
   this benchmark's seqalign-xdrop-heuristic-kernel definition (a
   single-direction, floor-free extension FROM THE SEQUENCE ORIGIN (0,0) --
   see kernelbench/domains/alignment.py's module docstring).

Because of (1) and (2), this adapter's gate is expected to fail, and does
fail, deterministically -- NOT because of an adapter bug, but because
LOGAN's actual compiled algorithm is not the same computation as this
benchmark's x-drop definition and cannot be reconfigured into it without
patching kernel-adjacent source (out of scope per ARTIFACT_GUIDE.md rule 3).
Per rule 4 ("failures are results"): this is reported honestly rather than
papered over, and the reference is NOT loosened or adjusted to match LOGAN's
hardcoded behavior.

prepare() writes one line per pair to a scratch TSV file in LOGAN's own
input format (seqV\\tposV\\tseqH\\tposH\\tstrand, identical to
inputs_demo/example.txt), anchoring a length-1 seed at position 1 on BOTH
sequences (not position 0): extendSeedLGappedXDropOneDirectionGlobal
returns immediately without writing its result slot whenever a seed sits at
the sequence start (`rows==1||cols==1`), leaving that pair's `scoreLeft`
read from memory the kernel never initialized (a real bug in LOGAN's own
code -- `scoreLeft`/`scoreRight` are plain `malloc`'d, never zeroed).
Anchoring 1 base in from each start avoids relying on that undefined
behavior for this adapter's own gate check (empirically it happened to read
back as 0 in this environment, consistent with a fresh zero-filled page --
but that is not a guarantee, so this adapter does not depend on it).
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

KERNEL = "sequence-alignment"
IMPL_NAME = "logan-xdrop"
PAPER_KEY = "conf/ipps/ZeniGEDSHBOY20"
PRECISIONS = ["int32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_BINARY = os.path.join(_HERE, "source", "kernelbench_driver")

XDROP_VARIANT = "seqalign-xdrop-heuristic-kernel"


def available() -> tuple[bool, str]:
    if not os.path.exists(_BINARY):
        return False, f"binary not built: {_BINARY} missing (run build.sh)"
    if not os.access(_BINARY, os.X_OK):
        return False, f"binary present but not executable: {_BINARY}"
    return True, ""


def create(precision: str):
    return LoganXdrop(precision)


class LoganXdrop:
    """
    One run() call = one kernelbench_driver subprocess = one LOGAN
    extendSeedL() call over the whole batch (LOGAN's own batching, all
    pairs in a single library call, matching how the upstream demo uses it).
    See module docstring for the two structural findings (hardcoded
    scoring, always-bidirectional extension) that make this adapter's gate
    fail deterministically and honestly, not due to an adapter bug.
    """

    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "int32"):
        self.precision = precision
        self._workdir = None
        self._input_path = None
        self._output_path = None
        self._n = 0
        self._x_drop = None
        self._match = None
        self._mismatch = None
        self._gap = None
        self._params = None

    def prepare(self, w, params: dict):
        if w.variant != XDROP_VARIANT:
            raise ValueError(
                f"logan-xdrop only implements {XDROP_VARIANT!r}, got {w.variant!r}")
        self._params = params
        self._workdir = tempfile.mkdtemp(prefix="logan_")
        self._input_path = os.path.join(self._workdir, "pairs.tsv")
        self._output_path = os.path.join(self._workdir, "scores.txt")

        sub = w.scoring.substitution
        self._match = int(sub[0, 0])
        self._mismatch = int(sub[0, 1])
        self._gap = int(w.scoring.gap_open)  # linear gap: gap_open == gap_extend
        self._x_drop = int(w.x_drop)
        self._n = len(w.seqs1)

        with open(self._input_path, "w") as f:
            for s1, s2 in zip(w.seqs1, w.seqs2):
                # seqV=s1 (query), seqH=s2 (target); anchor 1 base in from each
                # start (see module docstring) so this adapter never depends
                # on LOGAN's own uninitialized-memory edge case at position 0.
                f.write(f"{s1}\t1\t{s2}\t1\tn\n")

        params["x_drop"] = self._x_drop
        params["ksize"] = 1
        params["logan_requested_scoring"] = {
            "match": self._match, "mismatch": self._mismatch, "gap": self._gap}
        params["logan_note"] = (
            "LOGAN's compiled kernel hardcodes match=+1/mismatch=-1/gap_extend=-1 "
            "regardless of the values requested here -- see adapter.py / STATUS.md. "
            "The correctness gate is expected to fail for that reason (a genuine "
            "artifact limitation, not an adapter bug).")
        return {"input": self._input_path, "output": self._output_path}

    def run(self, h):
        cmd = [_BINARY, h["input"], h["output"], "1", str(self._x_drop),
               str(self._match), str(self._mismatch), str(self._gap), str(self._gap), "1"]
        proc = subprocess.run(cmd, cwd=self._workdir, capture_output=True, text=True, timeout=300)
        self._params["logan_subprocess_returncode"] = proc.returncode
        if proc.returncode != 0 or not os.path.exists(h["output"]):
            raise RuntimeError(
                f"logan kernelbench_driver failed (returncode={proc.returncode}); "
                f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}")
        with open(h["output"]) as f:
            scores = [int(line.strip()) for line in f if line.strip()]
        if len(scores) != self._n:
            raise RuntimeError(
                f"logan kernelbench_driver produced {len(scores)} scores, expected {self._n}")
        return np.array(scores, dtype=np.float64)

    def to_host(self, out):
        return np.asarray(out, dtype=np.float64)

    def timer(self):
        # plain CPU wall-clock: the alignment runs in a separate subprocess
        # with its own CUDA context, same rationale as SPCG's cg-krylov
        # adapter (bench/artifacts/cg-krylov/spcg/adapter.py).
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        import shutil
        if self._workdir and os.path.isdir(self._workdir):
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
