#!/usr/bin/env python3
"""
Merge classified chunk results with corpus metadata.

Reads data/chunks/*.json + data/classified/*.json, validates coverage
(every candidate classified exactly once, keys match), joins with corpus
metadata, writes:
  output/papers.json     — every candidate with full metadata + verdict
  output/included.json   — only is_kernel_opt=true
Prints coverage/consistency diagnostics and per-category counts.
"""

import glob
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

VALID_CATS = {"dense_la", "sparse_la", "stencil_pde", "fft_spectral", "tensor",
              "graph", "primitives", "nbody_md", "unstructured_amr",
              "solver_components", "compression", "ml_kernels"}


def main():
    with open(os.path.join(HERE, "data", "corpus.json")) as f:
        corpus = {p["key"]: p for p in json.load(f)}
    with open(os.path.join(HERE, "data", "abstracts.json")) as f:
        abstracts = json.load(f)

    chunk_keys = set()
    for cf in glob.glob(os.path.join(HERE, "data", "chunks", "chunk_*.json")):
        with open(cf) as f:
            for p in json.load(f):
                chunk_keys.add(p["key"])

    verdicts = {}
    bad = []
    for rf in sorted(glob.glob(os.path.join(HERE, "data", "classified", "chunk_*.json"))):
        with open(rf) as f:
            try:
                arr = json.load(f)
            except json.JSONDecodeError as e:
                bad.append((rf, str(e)))
                continue
        for v in arr:
            k = v.get("key")
            if k not in corpus:
                bad.append((rf, f"unknown key {k}"))
                continue
            if k in verdicts:
                bad.append((rf, f"duplicate key {k}"))
                continue
            cats = [c for c in v.get("categories", []) if c in VALID_CATS]
            v["categories"] = cats
            verdicts[k] = v

    missing = chunk_keys - set(verdicts)
    extra = set(verdicts) - chunk_keys
    print(f"chunks: {len(chunk_keys)} papers; verdicts: {len(verdicts)}; "
          f"missing: {len(missing)}; extra: {len(extra)}; parse errors: {len(bad)}")
    for rf, msg in bad[:20]:
        print(f"  BAD {os.path.basename(rf)}: {msg}")
    if missing:
        print("  sample missing:", sorted(missing)[:10])

    papers = []
    for k, v in verdicts.items():
        p = corpus[k]
        meta = abstracts.get(k, {})
        papers.append({
            **p,
            "abstract": meta.get("abstract", ""),
            "cited_by": meta.get("cited_by", 0),
            "oa_url": meta.get("oa_url", ""),
            "is_kernel_opt": bool(v.get("is_kernel_opt")),
            "confidence": v.get("confidence", "low"),
            "categories": v.get("categories", []),
            "kernels": v.get("kernels", []),
            "platform": v.get("platform", []),
            "approach": v.get("approach"),
            "excluded_reason": v.get("excluded_reason"),
            "one_liner": v.get("one_liner", ""),
        })

    papers.sort(key=lambda p: (p["venue"], p["year"], p["title"]))
    outdir = os.path.join(HERE, "output")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "papers.json"), "w") as f:
        json.dump(papers, f, ensure_ascii=False, indent=1)
    included = [p for p in papers if p["is_kernel_opt"]]
    with open(os.path.join(outdir, "included.json"), "w") as f:
        json.dump(included, f, ensure_ascii=False, indent=1)

    print(f"\nincluded {len(included)} / {len(papers)} classified")
    print("\n=== included per category ===")
    cc = Counter()
    for p in included:
        for c in p["categories"]:
            cc[c] += 1
    for c, n in cc.most_common():
        print(f"  {c:20s} {n}")
    print("\n=== included per venue ===")
    for v, n in Counter(p["venue"] for p in included).most_common():
        print(f"  {v:8s} {n}")
    print("\n=== excluded reasons ===")
    for r, n in Counter(p["excluded_reason"] for p in papers
                        if not p["is_kernel_opt"]).most_common():
        print(f"  {str(r):20s} {n}")
    print("\n=== low/medium confidence included (need verify pass) ===")
    lc = [p for p in included if p["confidence"] != "high"]
    print(f"  {len(lc)} papers")


if __name__ == "__main__":
    main()
