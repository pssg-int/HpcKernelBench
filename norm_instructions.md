# Kernel-name normalization — benchmark grouping

Map each paper's free-text `kernels_raw` to canonical kernel slugs, so papers
implementing the SAME kernel land in the same benchmark group. Use `title`,
`one_liner` and `categories` to disambiguate. A paper may map to 1-3 slugs
(main contributions only — do not enumerate every kernel it merely touches).

## Canonical vocabulary (use these exact slugs)

Sparse LA: `spmv` `spmm` `sddmm` `spgemm` `sptrsv` `spmspv` `sparse-format-conversion`
`sparse-factorization` (LU/Cholesky/QR on sparse) `sparse-attention-kernel`
Dense LA: `gemm` `batched-gemm` `quantized-gemm` (low/mixed-precision incl. tensor-core INT/FP8)
`gemv` `trsm` `lu` `cholesky` `qr` `eigensolver` `svd` `matrix-inversion` `blas-level1-2`
Stencil/PDE: `stencil` `lattice-boltzmann` `fdtd-seismic` (wave/seismic solvers) `climate-kernel`
Spectral: `fft` `ntt` (incl. FHE/PQC polynomial mult)
Tensor: `mttkrp` `tensor-contraction` `tucker` `tensor-network` (incl. quantum-circuit sim)
`sparse-tensor-contraction`
Graph: `bfs` `sssp` `pagerank` `triangle-counting` `graph-pattern-mining` `connected-components`
`community-detection` `graph-coloring` `betweenness` `kcore-kclique` `graph-partitioning`
`dynamic-graph-kernel` `random-walk`
Primitives: `sort` `scan-reduction` `hash-table` `set-intersection` `topk-selection`
`string-regex-matching` `sequence-alignment` `sampling` `random-number`
N-body/MD: `md-force` `fmm` `nbody` `particle-mesh`
Mesh/AMR: `unstructured-mesh-kernel` `amr-kernel` `particle-in-cell`
Solvers: `cg-krylov` `multigrid` `preconditioner` `tridiagonal-solve` `iterative-refinement`
`mixed-precision-solver`
Compression: `lossy-compression` `lossless-compression`
ML: `convolution` `attention-kernel` `gnn-aggregation` (SpMM-like GNN ops) `embedding-ops`
`moe-kernel` `layernorm-softmax-fused` `transformer-inference-fused` `dnn-operator-fusion`
`winograd`
Codegen targets: when the contribution is a compiler/autotuner over MANY kernel
types (Polybench etc.), use `autotuner-multi-kernel` (optionally + the single
dominant kernel slug if one dominates evaluation).
Other: `math-functions` (libm/SLEEF-like) `dp-dynamic-programming` `stream-processing-kernel`
`morphology-image-kernel` `cellular-automata`

If truly nothing fits, invent `other:<short-slug>` (use sparingly).

## Output
Read your input chunk; write a raw JSON array (same order, no fences), one object
per paper: {"key": "<copied>", "slugs": ["spmm", ...]}. 1-3 slugs per paper.
