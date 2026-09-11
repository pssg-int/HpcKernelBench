#!/usr/bin/env python3
"""Rank baseline candidates per kernel track under the revised selection rule.

Rule (user decision, 2026-09-05; replaces "3 newest per track"):
  1. kernel centrality: core > component > tangential  (tangential never selected)
  2. regime match with the track spec's inputs: matches > partial > mismatch
  3. platform: single-NVIDIA-GPU path required (current scope)
  4. recency (year) only as the tiebreak
  up to TOP_N per track.

Reads  output/benchmark_groups.json (with centrality fields merged in),
       bench/artifacts/<k>/<short>/{adapter.py,STATUS.md} (what is integrated)
Writes output/baseline_selection.json and output/baseline_selection.md
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench")
TOP_N = 5
C_RANK = {"core": 0, "component": 1, "tangential": 9}
R_RANK = {"matches": 0, "partial": 1, "mismatch": 2}

groups = json.load(open(os.path.join(HERE, "output/benchmark_groups.json")))


def integrated(k):
    """{paper_key: (short, has_adapter, status_line)} for a track's artifact dirs."""
    out = {}
    kdir = os.path.join(BENCH, "artifacts", k)
    if not os.path.isdir(kdir):
        return out
    for short in sorted(os.listdir(kdir)):
        d = os.path.join(kdir, short)
        ap, st = os.path.join(d, "adapter.py"), os.path.join(d, "STATUS.md")
        key = ""
        if os.path.exists(ap):
            m = re.search(r'^PAPER_KEY\s*=\s*["\']([^"\']+)["\']', open(ap).read(), re.M)
            key = m.group(1) if m else ""
        if not key and os.path.exists(st):
            m = re.search(r"\b((?:conf|journals)/[a-z]+/[A-Za-z0-9]+)", open(st).read())
            key = m.group(1) if m else ""
        line = ""
        if os.path.exists(st):
            for ln in open(st).read().splitlines()[:6]:
                if "STATUS" in ln or "Status" in ln or "Outcome" in ln:
                    line = re.sub(r"[*#]", "", ln).strip()[:120]
                    break
        if key:
            out[key] = (short, os.path.exists(ap), line)
    return out


selection = {}
lines = ["# Baseline selection under the revised rule (core-first)", "",
         "Rule: centrality core > component (tangential excluded); regime matches > partial > mismatch; "
         "single-NVIDIA-GPU path required; year only as tiebreak; up to %d per track. "
         "'integrated' = an artifact dir already exists for that paper." % TOP_N, ""]
for name, papers in sorted(groups.items()):
    k = name.replace("other:", "")
    rated = [p for p in papers if p.get("centrality")]
    if not rated:
        continue
    integ = integrated(k)
    elig = [p for p in rated if p["centrality"] != "tangential" and p.get("gpu_single_card")]
    elig.sort(key=lambda p: (C_RANK[p["centrality"]], R_RANK.get(p.get("regime_match"), 2), -int(p["year"])))
    picks = elig[:TOP_N]
    rows = []
    for p in picks:
        i = integ.get(p["key"])
        rows.append({"key": p["key"], "title": p["title"], "venue": p["venue"], "year": p["year"],
                     "centrality": p["centrality"], "regime_match": p.get("regime_match"),
                     "regime": p.get("regime", ""), "url": p.get("url", ""),
                     "integrated": bool(i), "artifact_dir": i[0] if i else "", "adapter": bool(i and i[1]),
                     "status": i[2] if i else ""})
    # integrated artifacts the rule would NOT have picked at all: not core, or
    # off-regime, or no single-GPU path. (Eligible ones merely outside the top-N
    # cutoff are fine -- listed as "kept".)
    demoted, kept = [], []
    picked_keys = {r["key"] for r in rows}
    for p in rated:
        i = integ.get(p["key"])
        if not (i and i[1]) or p["key"] in picked_keys:
            continue
        rec = {"key": p["key"], "title": p["title"], "artifact_dir": i[0],
               "centrality": p["centrality"], "regime_match": p.get("regime_match"),
               "gpu_single_card": bool(p.get("gpu_single_card"))}
        if p["centrality"] != "core" or p.get("regime_match") == "mismatch" or not p.get("gpu_single_card"):
            demoted.append(rec)
        else:
            kept.append(rec)
    # recommendations: only core picks on a matching/partial regime. A top-N
    # list padded with component papers (a track with fewer than N core papers)
    # must not read as "work left"; component picks are listed but not counted.
    todo = [r for r in rows if not r["integrated"] and r["centrality"] == "core"
            and r["regime_match"] != "mismatch"]
    selection[k] = {"picks": rows, "demoted_integrated": demoted, "kept_integrated": kept,
                    "todo": [r["key"] for r in todo],
                    "n_candidates": len(rated), "n_eligible": len(elig)}
    lines.append(f"## {k}  ({len(rated)} papers, {len(elig)} eligible, {len(todo)} to integrate)")
    for r in rows:
        flag = "integrated" if r["integrated"] else "TODO"
        lines.append(f"- [{flag}] {r['year']} {r['venue']} — {r['title'][:90]}  "
                     f"({r['centrality']}, regime {r['regime_match']})" + (f" → `{r['artifact_dir']}`" if r["integrated"] else ""))
    for d in demoted:
        lines.append(f"- [demoted] `{d['artifact_dir']}` — {d['title'][:80]} ({d['centrality']}, regime {d['regime_match']}"
                     f"{'' if d['gpu_single_card'] else ', no single-GPU path'}): keep as competitor, not a SOTA baseline")
    for d in kept:
        lines.append(f"- [kept] `{d['artifact_dir']}` — {d['title'][:80]} (core, regime {d['regime_match']}; outside the top-{TOP_N} cutoff only)")
    lines.append("")

json.dump(selection, open(os.path.join(HERE, "output/baseline_selection.json"), "w"), ensure_ascii=False, indent=1)
open(os.path.join(HERE, "output/baseline_selection.md"), "w").write("\n".join(lines))
n_todo = sum(len(v["todo"]) for v in selection.values())
n_dem = sum(len(v["demoted_integrated"]) for v in selection.values())
print(f"tracks: {len(selection)}; picks to integrate: {n_todo}; integrated-but-demoted: {n_dem}")
print("per implemented-track TODO counts (top 15):")
for k, v in sorted(selection.items(), key=lambda kv: -len(kv[1]["todo"]))[:15]:
    print(f"  {k:26s} todo={len(v['todo'])} eligible={v['n_eligible']}/{v['n_candidates']}")
