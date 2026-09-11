#!/usr/bin/env python3
"""
High-recall keyword prefilter: match the kernel taxonomy against title+abstract.

This stage only REDUCES the set that goes to LLM classification; it must not
miss kernel papers (recall >> precision). A paper is a candidate if any
category's keyword list matches title or abstract. Word-boundary regexes,
case-insensitive except for short ambiguous acronyms which are case-sensitive.

Output: data/candidates.json  (corpus records + matched_categories + abstract)
        data/rejected.json    (title-only records of non-candidates, for auditing)
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# category -> list of (pattern, case_sensitive)
# Patterns are matched with word boundaries added automatically unless the
# pattern already contains regex syntax.
TAXONOMY = {
    "dense_la": [
        (r"[sdhczb]?gemm", 0), (r"gemv", 0), (r"matrix[- ]matrix multiplication", 0),
        (r"matrix multiplication", 0), (r"matrix[- ]vector multiplication", 0),
        (r"trsm", 0), (r"cholesky", 0), (r"lu factorization", 0),
        (r"lu decomposition", 0), (r"qr factorization", 0), (r"qr decomposition", 0),
        (r"eigensolver", 0), (r"eigenvalue", 0), (r"svd", 0),
        (r"singular value decomposition", 0), (r"blas", 0), (r"lapack", 0),
        (r"linear algebra", 0), (r"matrix inversion", 0), (r"triangular solve", 0),
        (r"tensor core", 0), (r"matrix engine", 0), (r"mixed[- ]precision", 0),
        (r"batched", 0),
    ],
    "sparse_la": [
        (r"spmv", 0), (r"spmm", 0), (r"spgemm", 0), (r"sddmm", 0),
        (r"sptrsv", 0), (r"spmspm", 0), (r"spmspv", 0),
        (r"sparse matrix", 0), (r"sparse matrices", 0),
        (r"sparse tensor", 0), (r"sparse format", 0), (r"sparse storage", 0),
        (r"csr", 1), (r"csc", 1), (r"ell", None),  # ELL handled specially below
        (r"sparse triangular", 0), (r"incomplete factorization", 0),
        (r"sparse direct solver", 0), (r"sparse cholesky", 0), (r"sparse lu", 0),
        (r"sparsity pattern", 0), (r"sparse kernel", 0), (r"sparse linear", 0),
        (r"sparse workload", 0), (r"sparse computation", 0), (r"sparse accelerat", 0),
    ],
    "stencil_pde": [
        (r"stencil", 0), (r"structured grid", 0), (r"finite difference", 0),
        (r"finite volume", 0), (r"finite element", 0), (r"pde", 0),
        (r"partial differential", 0), (r"jacobi", 0), (r"gauss[- ]seidel", 0),
        (r"seismic", 0), (r"wave equation", 0), (r"weather", 0), (r"climate", 0),
        (r"lattice boltzmann", 0), (r"computational fluid", 0), (r"cfd", 0),
        (r"navier[- ]stokes", 0), (r"hydrodynamic", 0), (r"lattice qcd", 0),
    ],
    "fft_spectral": [
        (r"fft", 0), (r"fourier", 0), (r"spectral method", 0),
        (r"convolution theorem", 0), (r"ntt", 1), (r"number theoretic transform", 0),
    ],
    "tensor": [
        (r"tensor contraction", 0), (r"tensor decomposition", 0), (r"mttkrp", 0),
        (r"tucker", 0), (r"cp decomposition", 0), (r"candecomp", 0),
        (r"tensor network", 0), (r"tensor[- ]times", 0), (r"ttm", 1), (r"ttv", 1),
        (r"einsum", 0), (r"tensor algebra", 0), (r"tensor kernel", 0),
        (r"tensor program", 0), (r"tensor compiler", 0),
        (r"tensor computation", 0), (r"tensor operation", 0),
    ],
    "graph": [
        (r"graph processing", 0), (r"graph analytics", 0), (r"graph algorithm", 0),
        (r"bfs", 0), (r"breadth[- ]first", 0), (r"pagerank", 0),
        (r"triangle counting", 0), (r"graph mining", 0), (r"graph pattern", 0),
        (r"connected component", 0), (r"shortest path", 0), (r"sssp", 0),
        (r"betweenness", 0), (r"graph traversal", 0), (r"subgraph", 0),
        (r"community detection", 0), (r"graph partition", 0), (r"minimum spanning", 0),
        (r"k[- ]?core", 0), (r"maximal clique", 0), (r"graph color", 0),
        (r"butterfly counting", 0), (r"motif", 0),
    ],
    "primitives": [
        (r"sort", 0), (r"radix", 0), (r"merge", 0), (r"prefix sum", 0),
        (r"scan primitive", 0), (r"reduction", 0), (r"hash table", 0),
        (r"hash join", 0), (r"histogram", 0), (r"top[- ]k", 0),
        (r"set intersection", 0), (r"string matching", 0), (r"regular expression", 0),
        (r"sequence alignment", 0), (r"smith[- ]waterman", 0), (r"dedupl", 0),
        (r"random number", 0), (r"sampling", 0),
    ],
    "nbody_md": [
        (r"molecular dynamics", 0), (r"n[- ]body", 0), (r"fast multipole", 0),
        (r"fmm", 0), (r"particle simulation", 0), (r"force field", 0),
        (r"barnes[- ]hut", 0), (r"neighbor list", 0), (r"lennard[- ]jones", 0),
        (r"electrostatic", 0), (r"particle[- ]mesh", 0), (r"ewald", 0),
    ],
    "unstructured_amr": [
        (r"unstructured mesh", 0), (r"unstructured grid", 0),
        (r"adaptive mesh", 0), (r"amr", 0), (r"particle[- ]in[- ]cell", 0),
        (r"pic", 1), (r"mesh refinement", 0), (r"octree", 0),
        (r"space[- ]filling curve", 0), (r"morton", 0),
    ],
    "solver_components": [
        (r"krylov", 0), (r"conjugate gradient", 0), (r"gmres", 0),
        (r"multigrid", 0), (r"amg", 0), (r"precondition", 0),
        (r"tridiagonal", 0), (r"pentadiagonal", 0), (r"smoother", 0),
        (r"domain decomposition", 0), (r"iterative solver", 0),
        (r"iterative refinement", 0), (r"linear solver", 0), (r"linear system", 0),
        (r"bicgstab", 0), (r"chebyshev", 0),
    ],
    "compression": [
        (r"lossy compress", 0), (r"lossless compress", 0), (r"error[- ]bounded", 0),
        (r"zfp", 0), (r"sz[23]?", 1), (r"decompress", 0), (r"compression kernel", 0),
        (r"data reduction", 0), (r"quantiz", 0), (r"huffman", 0), (r"lz4", 0),
        (r"bit[- ]?plane", 0), (r"floating[- ]point compress", 0),
    ],
    # HPC-crossing ML kernels only (pure LLM-serving work is filtered by the LLM
    # classifier later; the prefilter keeps them so the classifier can decide).
    "ml_kernels": [
        (r"convolution", 0), (r"winograd", 0), (r"attention", 0),
        (r"transformer", 0), (r"gnn", 0), (r"graph neural", 0),
        (r"deep learning kernel", 0), (r"dnn inference", 0), (r"dnn training", 0),
        (r"embedding", 0), (r"moe", 0), (r"mixture[- ]of[- ]experts", 0),
        (r"low[- ]precision", 0), (r"int8", 0), (r"fp16", 0), (r"bfloat", 0),
        (r"quantized", 0),
    ],
    # cross-cutting flag, not a category by itself: kernel-y language
    "_kernel_language": [
        (r"kernel", 0), (r"code generation", 0), (r"auto[- ]?tun", 0),
        (r"compiler optimization", 0), (r"loop transformation", 0),
        (r"loop tiling", 0), (r"polyhedral", 0), (r"vectoriz", 0),
        (r"simd", 0), (r"gpu implementation", 0), (r"gpu accelerat", 0),
        (r"performance portab", 0), (r"roofline", 0), (r"memory[- ]bound", 0),
        (r"compute[- ]bound", 0), (r"data layout", 0), (r"register", 0),
        (r"shared memory", 0), (r"warp", 0), (r"cuda", 0), (r"rocm", 0),
        (r"sycl", 0), (r"openmp offload", 0), (r"opencl", 0),
        (r"gpus?", 0), (r"many[- ]core", 0), (r"multi[- ]?core", 0),
        (r"high[- ]performance", 0),
        (r"cusparse", 0), (r"cublas", 0), (r"cutlass", 0), (r"cudnn", 0),
        (r"rocblas", 0), (r"hipsparse", 0), (r"rocsparse", 0), (r"onemkl", 0),
        (r"mkl", 0), (r"cub", 1), (r"thrust", 0), (r"kokkos", 0), (r"raja", 0),
        (r"half[- ]precision", 0), (r"double[- ]precision", 0),
        (r"fp64", 0), (r"fp32", 0), (r"fp8", 0), (r"tf32", 0), (r"int4", 0),
        (r"avx[- ]?[0-9]*", 0), (r"sve", 1), (r"neon", 0),
    ],
}


def compile_taxonomy():
    compiled = {}
    for cat, pats in TAXONOMY.items():
        lst = []
        for pat, flags in pats:
            if pat == r"ell":
                # ELL/ELLPACK sparse format: require uppercase or ELLPACK
                lst.append(re.compile(r"\b(ELL(?:PACK)?)\b"))
                continue
            cs = flags == 1
            rx = re.compile(r"\b(?:%s)\b" % pat, 0 if cs else re.IGNORECASE)
            lst.append(rx)
        compiled[cat] = lst
    return compiled


def main():
    with open(os.path.join(HERE, "data", "corpus.json")) as f:
        corpus = json.load(f)
    with open(os.path.join(HERE, "data", "abstracts.json")) as f:
        abstracts = json.load(f)

    compiled = compile_taxonomy()
    candidates, rejected = [], []
    from collections import Counter
    cat_counts = Counter()

    for p in corpus:
        meta = abstracts.get(p["key"], {})
        text = p["title"] + "\n" + meta.get("abstract", "")
        matched = [cat for cat, rxs in compiled.items()
                   if cat != "_kernel_language" and any(r.search(text) for r in rxs)]
        kernel_lang = any(r.search(text) for r in compiled["_kernel_language"])
        if matched or kernel_lang:
            rec = dict(p)
            rec["abstract"] = meta.get("abstract", "")
            rec["cited_by"] = meta.get("cited_by", 0)
            rec["oa_url"] = meta.get("oa_url", "")
            rec["matched_categories"] = matched
            rec["kernel_language"] = kernel_lang
            candidates.append(rec)
            for c in matched:
                cat_counts[c] += 1
        else:
            rejected.append({"key": p["key"], "venue": p["venue"],
                             "year": p["year"], "title": p["title"]})

    with open(os.path.join(HERE, "data", "candidates.json"), "w") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "data", "rejected.json"), "w") as f:
        json.dump(rejected, f, ensure_ascii=False, indent=1)

    print(f"Corpus {len(corpus)} -> candidates {len(candidates)} "
          f"(rejected {len(rejected)})")
    print("\nMatches per category (a paper can match several):")
    for c, n in cat_counts.most_common():
        print(f"  {c:20s} {n}")
    no_abs = sum(1 for c in candidates if not c["abstract"])
    print(f"\nCandidates without abstract (title-only match): {no_abs}")


if __name__ == "__main__":
    main()
