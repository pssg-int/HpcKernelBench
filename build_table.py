#!/usr/bin/env python3
"""
Build the kernel x paper deliverables from output/included.json:

  output/kernel_table.csv   — one row per (category, paper)
  output/papers_flat.csv    — one row per included paper
  output/summary.md         — counts per category/venue/year, top-cited per category

Run merge_classified.py first.
"""

import csv
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

CAT_LABELS = {
    "dense_la": "Dense linear algebra",
    "sparse_la": "Sparse linear algebra",
    "stencil_pde": "Stencil / structured-grid PDE",
    "fft_spectral": "FFT / spectral",
    "tensor": "Tensor algebra",
    "graph": "Graph kernels",
    "primitives": "Parallel primitives",
    "nbody_md": "N-body / MD",
    "unstructured_amr": "Unstructured mesh / AMR / PIC",
    "solver_components": "Iterative-solver components",
    "compression": "Compression kernels",
    "ml_kernels": "ML kernels (HPC-crossing)",
}


def main():
    with open(os.path.join(OUT, "included.json")) as f:
        papers = json.load(f)

    # one row per (category, paper)
    with open(os.path.join(OUT, "kernel_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "kernels", "venue", "year", "title", "authors",
                    "platform", "approach", "confidence", "cited_by", "doi",
                    "dblp_url", "oa_url", "one_liner",
                    "artifact_status", "artifact_url", "artifact_stars",
                    "artifact_license", "artifact_last_push"])
        for p in sorted(papers, key=lambda x: (x["categories"][0] if x["categories"]
                                               else "zz", -x["year"], x["venue"])):
            for cat in (p["categories"] or ["uncategorized"]):
                w.writerow([
                    CAT_LABELS.get(cat, cat), "; ".join(p["kernels"]),
                    p["venue"], p["year"], p["title"],
                    "; ".join(p["authors"][:6]) + (" et al." if len(p["authors"]) > 6 else ""),
                    "; ".join(p["platform"]), p["approach"] or "", p["confidence"],
                    p["cited_by"], p["doi"], p["dblp_url"], p["oa_url"], p["one_liner"],
                    p.get("artifact_status", "none"), p.get("artifact_url", ""),
                    p.get("artifact_stars", 0), p.get("artifact_license", ""),
                    p.get("artifact_last_push", ""),
                ])

    # one row per paper
    with open(os.path.join(OUT, "papers_flat.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["venue", "year", "title", "categories", "kernels", "platform",
                    "approach", "confidence", "cited_by", "doi", "one_liner",
                    "artifact_status", "artifact_url", "artifact_stars"])
        for p in sorted(papers, key=lambda x: (-x["year"], x["venue"], x["title"])):
            w.writerow([p["venue"], p["year"], p["title"],
                        "; ".join(p["categories"]), "; ".join(p["kernels"]),
                        "; ".join(p["platform"]), p["approach"] or "",
                        p["confidence"], p["cited_by"], p["doi"], p["one_liner"],
                        p.get("artifact_status", "none"),
                        p.get("artifact_url", ""), p.get("artifact_stars", 0)])

    # summary
    by_cat = defaultdict(list)
    for p in papers:
        for c in p["categories"]:
            by_cat[c].append(p)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# HPC kernel-optimization papers, 2020+ \n\n")
        f.write(f"Included papers: **{len(papers)}** (out of 7029 main-track papers "
                "at SC, IPDPS, ICS, HPDC, ASPLOS, PPoPP, CGO, ATC, DAC, TPDS).\n\n")
        f.write("| Category | Papers | Top venues | Most-cited paper |\n|---|---|---|---|\n")
        for c, ps in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            vens = Counter(p["venue"] for p in ps).most_common(3)
            top = max(ps, key=lambda p: p["cited_by"])
            f.write(f"| {CAT_LABELS.get(c, c)} | {len(ps)} | "
                    f"{', '.join(f'{v} ({n})' for v, n in vens)} | "
                    f"{top['title'][:60]} ({top['venue']}'{str(top['year'])[2:]}, "
                    f"{top['cited_by']} cites) |\n")
        f.write("\n## Papers per venue\n\n| Venue | Included | Corpus 2020+ |\n|---|---|---|\n")
        corpus_counts = {"SC": 643, "IPDPS": 738, "ICS": 393, "HPDC": 213,
                         "ASPLOS": 925, "PPoPP": 242, "CGO": 250, "ATC": 440,
                         "DAC": 1703, "TPDS": 1482}
        inc_ven = Counter(p["venue"] for p in papers)
        for v, tot in sorted(corpus_counts.items(), key=lambda kv: -inc_ven.get(kv[0], 0)):
            f.write(f"| {v} | {inc_ven.get(v, 0)} | {tot} |\n")
        f.write("\n## Papers per year\n\n| Year | Included |\n|---|---|\n")
        for y, n in sorted(Counter(p["year"] for p in papers).items()):
            f.write(f"| {y} | {n} |\n")

    print(f"Wrote kernel_table.csv, papers_flat.csv, summary.md for {len(papers)} papers")


if __name__ == "__main__":
    main()
