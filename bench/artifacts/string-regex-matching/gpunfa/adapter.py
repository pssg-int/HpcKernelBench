"""
gpunfa-artifact adapter for the string-regex-matching track.

Paper: "Why GPUs are Slow at Executing NFAs and How to Make them Faster"
(ASPLOS'20, conf/asplos/0002PJ20). Artifact: https://github.com/bigwater/gpunfa-artifact

Wraps the `obat` CLI's "obat2" algorithm (the paper's own "NewTran" scheme,
`one_byte_at_a_time::OBAT_baseline_2` in src/obat/one_byte_at_a_time.cu) --
the artifact's own README lists this as one of its three headline schemes
(obat2/obat_MC/hotstart*). `obat` is a MONOLITHIC CLI (ANML load -> NFA
build -> H2D -> kernel launch -> report file write, all in one process),
not a library with a separable prepare/run API, so this adapter wraps at
the finest boundary actually available: one subprocess invocation per
run() call, per ARTIFACT_GUIDE.md rule 1's own allowance for end-to-end-
binary artifacts ("wrap at the finest boundary available and document the
contamination"). See STATUS.md for the preprocessing/timing-boundary note
and provenance.

Report-convention mapping (hand-verified on the artifact's own small_dataset
smoke automaton, apple.anml + inputstream.txt, per ARTIFACT_GUIDE.md's own
"first rule out a convention mismatch on a tiny hand-checkable automaton"):
  gpunfa writes report.txt as one "<offset>\\t<original_ANML_id>" line per
  report (report_formatter::print_to_file, src/commons/report_formatter.cpp):
  `offset` is 0-based, the byte index that satisfies the reporting STE's
  symbol-set (identical convention to this module's own
  _simulate_reference); `original_ANML_id` is the STE's `id` STRING
  attribute (NOT its `reportcode`) -- resolved back to this module's
  canonical int report_id via `automaton.states[id].report_id` (the SAME
  Automaton object this adapter serialized to ANML for gpunfa to read, so
  the id->report_id mapping is exactly what parse_anml() assigned).
  Verified empirically: running `obat -g obat2` on apple.anml (pattern
  "apple", reportcode=2019, STE id "__45__") over the shipped
  inputstream.txt produced report lines "5\\t__45__ / 47\\t__45__ /
  64\\t__45__ / 75\\t__45__", IDENTICAL to this module's own
  _simulate_reference/_build_bitset_tables output on the same automaton
  and stream (offsets 5, 47, 64, 75, report_id 2019) -- zero conversion
  needed beyond the id->report_id lookup above.

A small, artifact-internal alignment pad (observed: 81-byte input ->
"padding_input_stream = 4", independent of any CLI flag) can add a few
extra bytes past the true corpus length; this adapter truncates reports at
offset >= the workload's true stream length before gating, same mechanism
as real-workload gate_window_bytes truncation (see automata.py's
_canonical_reports).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from kernelbench.domains import automata  # noqa: E402
from kernelbench.harness import Timer  # noqa: E402

KERNEL = "string-regex-matching"
IMPL_NAME = "gpunfa-obat2"
PAPER_KEY = "conf/asplos/0002PJ20"
PRECISIONS = ["int64"]

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "build", "bin", "obat")
_LD_PRELOAD = "/usr/lib64/libstdc++.so.6"


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(BIN):
            return False, f"not built: {BIN} missing (run build.sh)"
        try:
            import torch
        except Exception as e:
            return False, f"torch import failed: {type(e).__name__}: {e}"
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


class GpunfaObat2:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "int64"):
        self.precision = precision

    def prepare(self, w, params: dict):
        automaton = w.automaton if w.patterns is None else automata.compile_patterns(w.patterns)[0]
        tmpdir = tempfile.mkdtemp(prefix="gpunfa_")
        anml_path = os.path.join(tmpdir, "automaton.anml")
        input_path = os.path.join(tmpdir, "stream.bin")
        with open(anml_path, "w") as f:
            f.write(automata.write_anml(automaton))
        with open(input_path, "wb") as f:
            f.write(w.stream.tobytes())
        params["automata_family"] = (
            "NFA (gpunfa-artifact's own GPU-NFA scheme; algorithm=obat2, "
            "the paper's 'NewTran' read-privatization technique)")
        params["preprocessing_includes"] = (
            "NOTHING is separately timed here: `obat` is a monolithic CLI that "
            "loads the ANML file, builds its internal NFA representation, does "
            "the H2D copy, launches the matching kernel and writes the report "
            "file all inside ONE process invocation started by run(), so ANML "
            "parse + H2D land inside run()'s wall-clock (contaminating the "
            "per-call time this harness measures). The artifact's OWN internal "
            "cudaEvent_t timer wraps ONLY the matching kernel and is captured "
            "separately into params['artifact_reported_elapsed_ms'] / "
            "params['artifact_reported_throughput_bytes_per_s'] -- use those "
            "for an uncontaminated number; see STATUS.md.")
        self._window = w.gate_window_bytes if w.gate_window_bytes is not None else int(w.stream.shape[0])
        self._params = params
        return {"anml_path": anml_path, "input_path": input_path, "tmpdir": tmpdir,
               "report_path": os.path.join(tmpdir, "report.txt"), "automaton": automaton}

    def run(self, h):
        cmd = [BIN, "-i", h["input_path"], "-a", h["anml_path"], "-g", "obat2",
              "--padding", "1", "--report-filename", h["report_path"]]
        env = dict(os.environ)
        # Only fall back to the hardcoded Perlmutter libstdc++ path if the
        # calling shell did not already export a (correct, machine-specific)
        # LD_PRELOAD itself (e.g. zaratan's env.sh points this at its own
        # conda-forge libstdc++.so.6, newer than /usr/lib64's -- overwriting
        # it unconditionally with the old path breaks any binary linked
        # against a newer libstdc++, "GLIBCXX_3.4.29 not found").
        env.setdefault("LD_PRELOAD", _LD_PRELOAD)
        res = subprocess.run(cmd, capture_output=True, text=True, env=env,
                             cwd=h["tmpdir"], timeout=120)
        if res.returncode != 0:
            raise RuntimeError(
                f"{IMPL_NAME}: obat exited {res.returncode}\n"
                f"STDOUT(tail):{res.stdout[-1500:]}\nSTDERR(tail):{res.stderr[-1500:]}")
        m = re.search(r"throughput\s*=\s*([0-9.eE+-]+)", res.stdout)
        if m:
            self._params["artifact_reported_throughput_bytes_per_s"] = float(m.group(1))
        m2 = re.search(r"Elapsed time\s*:\s*([0-9.eE+-]+)\s*ms", res.stdout)
        if m2:
            self._params["artifact_reported_elapsed_ms"] = float(m2.group(1))

        automaton = h["automaton"]
        reports = []
        if os.path.exists(h["report_path"]):
            with open(h["report_path"]) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    off_s, sid = line.split("\t")
                    ste = automaton.states.get(sid)
                    reports.append((int(off_s), ste.report_id if ste is not None else -1))
        return reports

    def to_host(self, out):
        arr = automata._canonical_reports(out, self._window)
        self._params["report_count"] = int(arr.shape[0])
        return arr

    def timer(self):
        return Timer()

    def free(self, h):
        shutil.rmtree(h["tmpdir"], ignore_errors=True)


def create(precision: str):
    return GpunfaObat2(precision)
