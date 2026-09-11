#!/usr/bin/env python3
"""Merge the agents' kernel-centrality ratings back into the group table.

Reads  data/centrality/chunk_NN.json (written by the rating agents; schema in
       data/centrality/INSTRUCTIONS.md)
Writes output/kernel_centrality.json   {"<group>|<key>": {...rating...}}
       output/benchmark_groups.json    each paper entry gains `centrality`,
                                       `regime_match`, `regime`, `gpu_single_card`,
                                       `centrality_rationale` (additive; other
                                       fields untouched)
Exits non-zero if any (paper, group) pair is missing or has an invalid label.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VALID_C = {"core", "component", "tangential"}
VALID_R = {"matches", "partial", "mismatch"}

groups = json.load(open(os.path.join(HERE, "output/benchmark_groups.json")))
expected = {(name.replace("other:", ""), p["key"]) for name, ps in groups.items() for p in ps}

ratings = {}
problems = []
for path in sorted(glob.glob(os.path.join(HERE, "data/centrality/chunk_*.json"))):
    try:
        rows = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        problems.append(f"{path}: unreadable JSON ({e})")
        continue
    for r in rows:
        g = str(r.get("group", "")).replace("other:", "")
        k = r.get("key", "")
        if (g, k) not in expected:
            problems.append(f"{path}: unexpected pair {g}|{k}")
            continue
        if r.get("centrality") not in VALID_C or r.get("regime_match") not in VALID_R:
            problems.append(f"{path}: invalid labels for {g}|{k}: {r.get('centrality')}/{r.get('regime_match')}")
            continue
        if (g, k) in ratings:
            problems.append(f"{path}: duplicate pair {g}|{k}")
        ratings[(g, k)] = {
            "centrality": r["centrality"], "regime_match": r["regime_match"],
            "regime": str(r.get("regime", ""))[:200],
            "gpu_single_card": bool(r.get("gpu_single_card", False)),
            "rationale": str(r.get("rationale", ""))[:600],
        }

missing = sorted(expected - set(ratings))
print(f"rated {len(ratings)}/{len(expected)} pairs; missing {len(missing)}; problems {len(problems)}")
for m in missing[:20]:
    print("  missing:", m)
for p in problems[:20]:
    print("  problem:", p)
if missing or problems:
    if "--force" not in sys.argv:
        sys.exit(1)

for name, ps in groups.items():
    g = name.replace("other:", "")
    for p in ps:
        r = ratings.get((g, p["key"]))
        if not r:
            continue
        p["centrality"] = r["centrality"]
        p["regime_match"] = r["regime_match"]
        p["regime"] = r["regime"]
        p["gpu_single_card"] = r["gpu_single_card"]
        p["centrality_rationale"] = r["rationale"]

json.dump({f"{g}|{k}": v for (g, k), v in sorted(ratings.items())},
          open(os.path.join(HERE, "output/kernel_centrality.json"), "w"), ensure_ascii=False, indent=1)
json.dump(groups, open(os.path.join(HERE, "output/benchmark_groups.json"), "w"), ensure_ascii=False, indent=1)
from collections import Counter  # noqa: E402
print("centrality:", Counter(v["centrality"] for v in ratings.values()))
print("regime_match:", Counter(v["regime_match"] for v in ratings.values()))
print("gpu_single_card:", Counter(v["gpu_single_card"] for v in ratings.values()))
