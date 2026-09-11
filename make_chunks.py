#!/usr/bin/env python3
"""
Split candidates.json into chunks for parallel LLM classification.

Each chunk file holds ~CHUNK papers (key, venue, year, title, abstract,
matched_categories). Classification agents read one chunk and write
data/classified/<chunk>.json with one verdict per paper.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 50

def main():
    with open(os.path.join(HERE, "data", "candidates.json")) as f:
        cands = json.load(f)
    outdir = os.path.join(HERE, "data", "chunks")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(HERE, "data", "classified"), exist_ok=True)

    # group by venue/year so each agent sees coherent context
    cands.sort(key=lambda p: (p["venue"], p["year"], p["title"]))
    n = 0
    for i in range(0, len(cands), CHUNK):
        chunk = cands[i:i + CHUNK]
        slim = [{
            "key": p["key"], "venue": p["venue"], "year": p["year"],
            "title": p["title"], "abstract": p.get("abstract", ""),
            "hint_categories": p.get("matched_categories", []),
        } for p in chunk]
        with open(os.path.join(outdir, f"chunk_{i//CHUNK:03d}.json"), "w") as f:
            json.dump(slim, f, ensure_ascii=False, indent=1)
        n += 1
    print(f"{len(cands)} candidates -> {n} chunks of <= {CHUNK} in {outdir}")

if __name__ == "__main__":
    main()
