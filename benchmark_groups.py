#!/usr/bin/env python3
"""
Group artifact-bearing papers into benchmark kernel tracks.

Reads data/normed/nchunk_*.json (paper -> canonical kernel slugs) and
output/included.json; writes:
  output/benchmark_groups.json   — {slug: [paper records]}
  output/benchmark_candidates.html — grouped, self-contained page
  output/benchmark_groups.csv    — one row per (slug, paper)
"""

import glob
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

DOMAIN = {  # slug prefix -> display domain (ordering of sections)
    "sparse LA": ["spmv", "spmm", "sddmm", "spgemm", "sptrsv", "spmspv",
                  "sparse-format-conversion", "sparse-factorization",
                  "sparse-attention-kernel"],
    "dense LA": ["gemm", "batched-gemm", "quantized-gemm", "gemv", "trsm", "lu",
                 "cholesky", "qr", "eigensolver", "svd", "matrix-inversion",
                 "blas-level1-2"],
    "stencil / PDE": ["stencil", "lattice-boltzmann", "fdtd-seismic", "climate-kernel"],
    "spectral": ["fft", "ntt"],
    "tensor": ["mttkrp", "tensor-contraction", "sparse-tensor-contraction",
               "tucker", "tensor-network"],
    "graph": ["bfs", "sssp", "pagerank", "triangle-counting", "graph-pattern-mining",
              "connected-components", "community-detection", "graph-coloring",
              "betweenness", "kcore-kclique", "graph-partitioning",
              "dynamic-graph-kernel", "random-walk"],
    "primitives": ["sort", "scan-reduction", "hash-table", "set-intersection",
                   "topk-selection", "string-regex-matching", "sequence-alignment",
                   "sampling", "random-number"],
    "N-body / MD": ["md-force", "fmm", "nbody", "particle-mesh"],
    "mesh / AMR": ["unstructured-mesh-kernel", "amr-kernel", "particle-in-cell"],
    "solvers": ["cg-krylov", "multigrid", "preconditioner", "tridiagonal-solve",
                "iterative-refinement", "mixed-precision-solver"],
    "compression": ["lossy-compression", "lossless-compression"],
    "ML kernels": ["convolution", "attention-kernel", "gnn-aggregation",
                   "embedding-ops", "moe-kernel", "layernorm-softmax-fused",
                   "transformer-inference-fused", "dnn-operator-fusion", "winograd"],
    "codegen / misc": ["autotuner-multi-kernel", "math-functions",
                       "dp-dynamic-programming", "stream-processing-kernel",
                       "morphology-image-kernel", "cellular-automata"],
}
SLUG_DOMAIN = {s: d for d, slugs in DOMAIN.items() for s in slugs}

# judge-invented slugs that are the same benchmark target
ALIAS = {"other:knn-search": "other:ann-search"}


def main():
    slugs_by_key = {}
    for nf in sorted(glob.glob(os.path.join(HERE, "data", "normed", "nchunk_*.json"))):
        with open(nf) as f:
            for v in json.load(f):
                slugs_by_key[v["key"]] = v.get("slugs", [])

    with open(os.path.join(OUT, "included.json")) as f:
        papers = {p["key"]: p for p in json.load(f) if p["artifact_status"] != "none"}

    missing = set(papers) - set(slugs_by_key)
    if missing:
        print(f"WARN: {len(missing)} artifact papers not normalized:",
              sorted(missing)[:5])

    groups = defaultdict(list)
    for key, p in papers.items():
        for s in slugs_by_key.get(key, []):
            s = ALIAS.get(s, s)
            groups[s].append({
                "key": key, "title": p["title"], "venue": p["venue"],
                "year": p["year"], "platform": p["platform"],
                "approach": p["approach"], "cited_by": p["cited_by"],
                "doi": p["doi"], "url": p["artifact_url"],
                "status": p["artifact_status"], "one_liner": p["one_liner"],
            })
    for s in groups:
        groups[s].sort(key=lambda r: (-r["year"], r["venue"]))

    with open(os.path.join(OUT, "benchmark_groups.json"), "w") as f:
        json.dump(groups, f, ensure_ascii=False, indent=1)

    import csv
    with open(os.path.join(OUT, "benchmark_groups.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "domain", "n_papers_in_group", "venue", "year",
                    "title", "platform", "approach", "artifact_status",
                    "artifact_url", "one_liner"])
        for s, ps in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            for r in ps:
                w.writerow([s, SLUG_DOMAIN.get(s, "other"), len(ps), r["venue"],
                            r["year"], r["title"], "; ".join(r["platform"]),
                            r["approach"] or "", r["status"], r["url"],
                            r["one_liner"]])

    # ---- console summary ----
    sizes = Counter({s: len(ps) for s, ps in groups.items()})
    strong = [(s, n) for s, n in sizes.most_common() if n >= 3]
    pair = [(s, n) for s, n in sizes.most_common() if n == 2]
    single = [(s, n) for s, n in sizes.most_common() if n == 1]
    print(f"\n{len(papers)} papers -> {len(groups)} kernel groups")
    print(f"strong tracks (>=3 papers): {len(strong)}")
    for s, n in strong:
        print(f"  {n:3d}  {s}  [{SLUG_DOMAIN.get(s,'other')}]")
    print(f"pairs (2): {len(pair)} -> {', '.join(s for s,_ in pair)}")
    print(f"singletons: {len(single)}")
    return groups


if __name__ == "__main__":
    main()
