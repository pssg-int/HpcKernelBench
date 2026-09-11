#!/usr/bin/env python3
"""Prepare input chunks for the kernel-centrality tagging pass.

Every (paper, kernel-group) pair among the with-artifact papers gets rated by an
agent: is this kernel the paper's headline contribution (core), a component of
a bigger system / a different regime (component), or only touched (tangential)?
Chunks are formed by whole groups so an agent reads each track's spec once.

Inputs : output/benchmark_groups.json, output/included.json, benchspecs/<k>/spec.yaml
Outputs: data/centrality/input_chunk_NN.json (+ INSTRUCTIONS.md written separately)
Merge  : merge_centrality.py (reads data/centrality/chunk_NN.json written by agents)
"""
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
groups = json.load(open(os.path.join(HERE, "output/benchmark_groups.json")))
included = {p["key"]: p for p in json.load(open(os.path.join(HERE, "output/included.json")))}
N_CHUNKS = 6
OUT = os.path.join(HERE, "data/centrality")
os.makedirs(OUT, exist_ok=True)


def spec_digest(slug):
    path = os.path.join(HERE, "benchspecs", slug, "spec.yaml")
    if not os.path.exists(path):
        return {"summary": "(no spec)", "variants": []}
    sp = yaml.safe_load(open(path))
    vs = []
    for v in sp.get("variants", []):
        inp = v.get("inputs") or {}
        if not isinstance(inp, dict):
            inp = {"raw": str(inp)}
        rs = inp.get("recommended_subset")
        vs.append({
            "id": v.get("id"),
            "claim": str(v.get("claim", ""))[:400],
            "inputs": {
                "suite": str(inp.get("suite", ""))[:200],
                "selection": str(inp.get("selection", ""))[:400],
                "recommended_subset": rs[:12] if isinstance(rs, list) else str(rs or "")[:200],
                "dense_operand": str(inp.get("dense_operand", ""))[:200],
            },
        })
    return {"summary": str(sp.get("summary", ""))[:600], "operation": str(sp.get("operation", ""))[:300],
            "variants": vs}


entries = []
for name, papers in groups.items():
    slug = name.replace("other:", "")
    g = {"group": name, "slug": slug, "spec": spec_digest(slug), "papers": []}
    for p in papers:
        full = included.get(p["key"], {})
        g["papers"].append({
            "key": p["key"], "title": p["title"], "venue": p["venue"], "year": p["year"],
            "platform": p.get("platform", []), "approach": p.get("approach", ""),
            "artifact_url": p.get("url", ""), "one_liner": p.get("one_liner", ""),
            "kernels_assigned": full.get("kernels", ""),
            "categories": full.get("categories", ""),
            "abstract": (full.get("abstract") or "")[:1800],
        })
    entries.append(g)

# balance by pair count, largest groups first, greedy into N_CHUNKS bins
entries.sort(key=lambda g: -len(g["papers"]))
bins = [{"chunk": i, "pairs": 0, "groups": []} for i in range(N_CHUNKS)]
for g in entries:
    b = min(bins, key=lambda b: b["pairs"])
    b["groups"].append(g)
    b["pairs"] += len(g["papers"])
for b in bins:
    path = os.path.join(OUT, f"input_chunk_{b['chunk']:02d}.json")
    json.dump({"chunk": b["chunk"], "groups": b["groups"]}, open(path, "w"), ensure_ascii=False, indent=1)
    print(path, "groups:", len(b["groups"]), "pairs:", b["pairs"],
          "slugs:", [g["slug"] for g in b["groups"]])
print("total pairs:", sum(b["pairs"] for b in bins))
