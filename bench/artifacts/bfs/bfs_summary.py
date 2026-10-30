"""
BFS paper-vs-baseline tables (BLEST vs Gunrock/GAP, efg vs cuGraph).

Reads the runner's result JSONs and prints Markdown: per graph, the median
per-search time of every implementation and the speedups each paper claims.
Time used per implementation:
  - its own internal timer when the adapter records one
    (gunrock_internal_ms, gap_internal_ms, cugraph_internal_ms): median of
    the last `reps` entries (earlier entries are the gate and warmup calls);
  - otherwise the harness median (blest, efg).
GTEPS = giant-component edges / time, the harness's own definition.

  python bfs_summary.py results/bfs_*.json
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict

INTERNAL = {"gunrock-bfs": "gunrock_internal_ms", "gap-dobfs": "gap_internal_ms",
            "cugraph-bfs": "cugraph_internal_ms"}
COLS = ["blest-tensorcore-bfs", "efg-elias-fano-bfs", "cugraph-bfs", "gunrock-bfs",
        "gap-dobfs", "scipy-bfs"]
SHORT = {"blest-tensorcore-bfs": "BLEST", "efg-elias-fano-bfs": "efg",
         "cugraph-bfs": "cuGraph", "gunrock-bfs": "Gunrock", "gap-dobfs": "GAP",
         "scipy-bfs": "scipy"}
CLAIMS = [  # (paper impl, baseline impl, paper's claimed average speedup)
    ("blest-tensorcore-bfs", "gunrock-bfs", "4.64x"),
    ("blest-tensorcore-bfs", "gap-dobfs", "13.25x"),
    ("efg-elias-fano-bfs", "cugraph-bfs", "0.82x if it fits in GPU memory, 3.8x if not"),
]


def _ms(run):
    p = run.get("params", {})
    key = INTERNAL.get(run["implementation"])
    reps = (run.get("protocol_used") or {}).get("reps") or 0
    if key and p.get(key):
        xs = p[key][-reps:] if reps else p[key]
        return statistics.median(xs), "internal"
    med = (run.get("stats_ms") or {}).get("median")
    return med, "harness"


def load(paths):
    table = defaultdict(dict)   # graph -> impl -> (ms, src, valid, edges, note)
    for f in paths:
        d = json.load(open(f))
        for r in d.get("runs", []):
            g = (r.get("matrix") or {}).get("name")
            ms, src = _ms(r) if r.get("valid") else (None, "")
            edges = r.get("params", {}).get("giant_component_edges")
            table[g][r["implementation"]] = (ms, src, r.get("valid"), edges, "")
        for e in d.get("errors", []):
            table[e.get("workload")][e["impl"]] = (None, "", False, None,
                                                   "error: " + str(e.get("error", ""))[:60])
        for u in d.get("unsupported", []):
            table[u.get("workload")][u["impl"]] = (None, "", False, None,
                                                   "unsupported: " + str(u.get("reason", ""))[:60])
    return table


def _fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}g}"


def main(paths):
    t = load(paths)
    print("## BFS: median ms per search (GTEPS)\n")
    print("| graph | " + " | ".join(SHORT[c] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for g, impls in sorted(t.items()):
        cells = []
        for c in COLS:
            if c not in impls:
                cells.append("not run")
                continue
            ms, src, valid, edges, note = impls[c]
            if note:
                cells.append(note)
            elif not valid:
                cells.append("gate FAIL")
            else:
                gteps = edges / (ms * 1e6) if edges and ms else None
                cells.append(f"{_fmt(ms)} ({_fmt(gteps, 3)})" + ("" if src == "internal" else "*"))
        print(f"| {g} | " + " | ".join(cells) + " |")
    print("\n`*` = harness timer (no internal timer recorded); BLEST's harness time "
          "includes its per-call graph re-upload (see blest/STATUS.md).")

    print("\n## Paper claims vs measured (speedup = baseline ms / paper ms)\n")
    print("| paper | baseline | graph | measured | paper's claim |")
    print("|---|---|---|---|---|")
    for paper, base, claim in CLAIMS:
        sp = []
        for g, impls in sorted(t.items()):
            a, b = impls.get(paper), impls.get(base)
            if a and b and a[0] and b[0] and a[2] and b[2]:
                s = b[0] / a[0]
                sp.append(s)
                print(f"| {SHORT[paper]} | {SHORT[base]} | {g} | {s:.3g}x | {claim} |")
        if sp:
            gm = math.exp(sum(math.log(x) for x in sp) / len(sp))
            print(f"| {SHORT[paper]} | {SHORT[base]} | **geomean** | **{gm:.3g}x** | {claim} |")


if __name__ == "__main__":
    main(sys.argv[1:])
