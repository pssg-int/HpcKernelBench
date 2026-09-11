"""
ngAP adapter for the string-regex-matching track.

Paper: "ngAP: Non-blocking Large-scale Automata Processing on GPUs"
(ASPLOS'24, conf/asplos/GeZ024). Artifact: https://github.com/getianao/ngAP

Wraps the `ngap` CLI's `nonblockingallgroups` algorithm -- the paper's own
full headline scheme (non-blocking + Prefetching Always-Active-states +
Prefix Memoization + Work Privatization, per the artifact's own README
algorithm list). `ngap` is, like gpunfa's `obat`, a monolithic CLI (ANML
load -> NFA build -> H2D -> kernel launch -> stdout report -> optional
internal CPU validation, all in one process) -- see STATUS.md for the same
preprocessing/timing-boundary contamination note as the gpunfa adapter.

CLI flags: `ngap` requires a large, specific flag set beyond `-i/-a/-g`
(discovered from the artifact's OWN worked example in its README, section
2 "small dataset" usage) -- omitting several of them (`--result-capacity`,
`--data-buffer-fetch-size`, etc, all left at internal defaults) caused a
SEGFAULT on every automaton tried (both the artifact's own 5-state
apple.anml and this module's own 16-state synthetic smoke automaton),
while the full documented flag set ran cleanly on both. `--group-num` MUST
be set to the automaton's actual connected-component (CC) count -- the
default (10) only self-corrects via a printed warning but the corrected
value still segfaults; passing the true CC count from the start does not.
This adapter computes the CC count itself (scipy.sparse.csgraph) before
invoking the CLI.

Report-convention mapping (empirically decoded, per ARTIFACT_GUIDE's own
"first rule out a convention mismatch on a tiny hand-checkable automaton"):
ngap does NOT write the `--report-filename` file for this algorithm family
(confirmed empirically: the file is never created); results are printed to
stdout as `Result(N): \n0x.., 0x.., ...`. Each token is a 64-bit value
`(node << 32) | offset` (see `getResult()`, src/ngap/kernel_helper.h),
where `node = local_state_id | (blockIdx.x << 22)` (src/ngap/kernel_helper.h)
-- i.e. the high bits encode which CUDA thread block (== which CC "group")
processed the match, and the low 22 bits are the STE's 0-based index WITHIN
that CC, in the same relative document order the STE appears in the ANML
file. Verified on two hand-checkable cases:
  1. apple.anml (5 states, 1 CC): token 0x400000005 decodes to
     node=4, offset=5 -- node=4 is the 5th (0-based index 4) and ONLY CC's
     STE in document order (`__45__`, the sole reporting STE) -- matches
     the artifact's OWN README claim ("ending positions ... state index of
     4") and this module's own _simulate_reference output on the same
     automaton+stream (offsets 5, 47, 64, 75, report_id 2019) exactly.
  2. This module's own 16-state, 4-CC synthetic smoke automaton
     (patterns=["a[0-9]+","b.c","(foo|bar)x*","gr[ae]y"]): token
     0x4000020000012f decodes to node=0x400002=(group 1, local_index 2),
     offset=0x12f=303 -- group 1 (the 2nd CC discovered, 0-based) is
     pattern index 1 ("b.c"), local_index 2 is that CC's 3rd state in
     document order (p1_2, the pattern's only reporting STE, report_id=1)
     -- matches this module's own reference exactly ((303, 1) is in its
     output). All 18 of this automaton's reports decoded and matched this
     way (verified manually while writing this adapter; see STATUS.md).
This adapter replicates ngap's own (group, local_index) numbering via
scipy connected-components (treating the STE graph as undirected, matching
standard CC discovery) with same-document-order tie-breaking, then maps
(group, local_index) -> the corresponding STE's canonical report_id via
the SAME Automaton object serialized to ANML for `ngap` to read.

Known limitation: `ngap` exits with a NONZERO return code (`1`) even on a
fully successful run that prints "FINISHED!" and "Validation PASS!" --
this adapter treats "FINISHED!" appearing in stdout as the success
signal instead of the process return code (documented, not a bug in our
wrapping).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from kernelbench.domains import automata  # noqa: E402
from kernelbench.harness import Timer  # noqa: E402

KERNEL = "string-regex-matching"
IMPL_NAME = "ngap-nonblockingallgroups"
PAPER_KEY = "conf/asplos/GeZ024"
PRECISIONS = ["int64"]

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "build", "bin", "ngap")
_LD_PRELOAD = "/usr/lib64/libstdc++.so.6"
_RESULT_RE = re.compile(r"Result\(\d+\):\s*\n([^\n]*)")


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
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _group_local_index_map(automaton) -> dict:
    """(group_id, local_index) -> STE, replicating ngap's own CC-based
    numbering (see module docstring). group_id/local_index are assigned in
    first-encounter document order, matching how ngap's own NFALoader/
    nfa_utils build their CC list while walking the ANML file top to
    bottom (same lineage codebase as gpunfa's own nfa_utils.cpp)."""
    ids = list(automaton.states.keys())
    idx = {sid: i for i, sid in enumerate(ids)}
    n = len(ids)
    rows, cols = [], []
    for sid, ste in automaton.states.items():
        i = idx[sid]
        for out in ste.outputs:
            j = idx[out]
            rows.append(i)
            cols.append(j)
    if rows:
        data = np.ones(len(rows), dtype=np.int8)
        graph = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    else:
        graph = sp.csr_matrix((n, n), dtype=np.int8)
    n_cc, labels = sp.csgraph.connected_components(graph, directed=False)
    group_of_label: dict = {}
    next_group = 0
    local_counter: dict = {}
    result = {}
    for sid in ids:  # document order
        lab = int(labels[idx[sid]])
        if lab not in group_of_label:
            group_of_label[lab] = next_group
            local_counter[lab] = 0
            next_group += 1
        g = group_of_label[lab]
        local = local_counter[lab]
        local_counter[lab] += 1
        result[(g, local)] = automaton.states[sid]
    return result, n_cc


class NgapNonblockingAllGroups:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "int64"):
        self.precision = precision

    def prepare(self, w, params: dict):
        automaton = w.automaton if w.patterns is None else automata.compile_patterns(w.patterns)[0]
        gl_map, n_cc = _group_local_index_map(automaton)
        tmpdir = tempfile.mkdtemp(prefix="ngap_")
        anml_path = os.path.join(tmpdir, "automaton.anml")
        input_path = os.path.join(tmpdir, "stream.bin")
        with open(anml_path, "w") as f:
            f.write(automata.write_anml(automaton))
        with open(input_path, "wb") as f:
            f.write(w.stream.tobytes())
        params["automata_family"] = (
            "NFA (ngAP's own non-blocking GPU-NFA engine; algorithm="
            "nonblockingallgroups, the paper's full optimization stack)")
        params["preprocessing_includes"] = (
            "NOTHING is separately timed here: `ngap` is a monolithic CLI "
            "(ANML load, NFA/CC build, H2D, kernel launch, stdout report, "
            "internal CPU cross-validation all in one process invocation "
            "started by run()) -- see STATUS.md for the same contamination "
            "note as the gpunfa adapter. The artifact's own device-side "
            "elapsed time / throughput are captured into "
            "params['artifact_reported_elapsed_s'] / "
            "params['artifact_reported_throughput_mb_per_s'].")
        self._window = w.gate_window_bytes if w.gate_window_bytes is not None else int(w.stream.shape[0])
        self._params = params
        return {"anml_path": anml_path, "input_path": input_path, "tmpdir": tmpdir,
               "gl_map": gl_map, "n_cc": n_cc, "corpus_len": int(w.stream.shape[0])}

    def run(self, h):
        n = h["corpus_len"]
        cmd = [
            BIN, "-a", h["anml_path"], "-i", h["input_path"],
            "--app-name=kb", "--algorithm=nonblockingallgroups",
            "--input-start-pos=0", f"--input-len={n}",
            f"--split-entire-inputstream-to-chunk-size={n}",
            f"--group-num={max(1, h['n_cc'])}",
            "--duplicate-input-stream=1", "--unique=false", "--unique-frequency=10",
            "--use-soa=false", "--result-capacity=54619400", "--use-uvm=false",
            "--data-buffer-fetch-size=25600000", "--add-aan-start=256",
            "--add-aas-interval=32", "--active-threshold=10",
            "--precompute-cutoff=-1", "--precompute-depth=3",
            "--compress-prec-table=true", "--report-off=false", "--validation=true",
        ]
        env = dict(os.environ)
        # Only fall back to the hardcoded Perlmutter libstdc++ path if the
        # calling shell did not already export a (correct, machine-specific)
        # LD_PRELOAD itself (e.g. zaratan's env.sh points this at its own
        # conda-forge libstdc++.so.6, newer than /usr/lib64's -- overwriting
        # it unconditionally with the old path breaks any binary linked
        # against a newer libstdc++, "GLIBCXX_3.4.29 not found"; same fix as
        # ../gpunfa/adapter.py, found while reproducing that artifact first).
        env.setdefault("LD_PRELOAD", _LD_PRELOAD)
        res = subprocess.run(cmd, capture_output=True, text=True, env=env,
                             cwd=h["tmpdir"], timeout=120)
        # ngap exits nonzero (1) even on a fully successful run -- see
        # module docstring; "FINISHED!" in stdout is the real success signal.
        if "FINISHED!" not in res.stdout:
            raise RuntimeError(
                f"{IMPL_NAME}: ngap did not report FINISHED (rc={res.returncode})\n"
                f"STDOUT(tail):{res.stdout[-2000:]}\nSTDERR(tail):{res.stderr[-1500:]}")
        m = re.search(r"throughput\s*=\s*([0-9.eE+-]+)\s*MB/s", res.stdout)
        if m:
            self._params["artifact_reported_throughput_mb_per_s"] = float(m.group(1))
        m2 = re.search(r"elapsed time:\s*([0-9.eE+-]+)\s*seconds", res.stdout)
        if m2:
            self._params["artifact_reported_elapsed_s"] = float(m2.group(1))

        m3 = _RESULT_RE.search(res.stdout)
        gl_map = h["gl_map"]
        reports = []
        if m3:
            tokens = [t.strip() for t in m3.group(1).split(",") if t.strip()]
            for tok in tokens:
                val = int(tok, 16)
                offset = val & 0xFFFFFFFF
                node = val >> 32
                local_index = node & 0x3FFFFF   # low 22 bits, see kernel_helper.h getResult()
                group = node >> 22
                ste = gl_map.get((group, local_index))
                reports.append((offset, ste.report_id if ste is not None else -1))
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
    return NgapNonblockingAllGroups(precision)
