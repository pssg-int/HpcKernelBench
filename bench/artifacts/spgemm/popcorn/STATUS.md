# Popcorn — STATUS

**Outcome: SKIPPED — no working SpGEMM kernel in the artifact**

Paper: "Popcorn: Accelerating Kernel K-means on GPUs through Sparse Linear
Algebra", PPoPP 2025. `PAPER_KEY = conf/ppopp/BellavitaPMVG25` (matched by
title against `../../output/included.json`'s `artifact_url` field; note the
task brief's "IPDPS25" venue guess does not match included.json's own
record — PPoPP 2025 is what the source of truth says, used here).
Repo: `https://github.com/HicrestLaboratory/Popcorn`
(commit `8066ead7bc4328c6b0616c4b7ff071746c090ae0`, `git clone --depth 50`).

## What the artifact actually is

Popcorn reformulates Kernel K-means as a sequence of **sparse-times-dense**
operations (a one-hot cluster-assignment matrix `V` times dense point/kernel
data) — i.e. genuine SpMM and SpMV, not SpGEMM (sparse-times-sparse). Every
compute kernel in `src/kernels/` that touches cuSPARSE calls
`cusparseSpMM`/`cusparseSpMV`:

```
$ grep -rn 'cusparseSpMM\|cusparseSpMV' src/kernels/*.cu | wc -l
19
```

- `distances.cu`: `compute_distances_spmm`, `compute_distances_popcorn_spmm`,
  `compute_distances_popcorn_spmv` — all `cusparseSpMM`/`cusparseSpMV` on a
  sparse assignment matrix `V` (CSR) times a dense matrix/vector.
- `centroids.cu`: `compute_centroids_spmm` — `cusparseSpMM` again (`V @ P`,
  sparse-times-dense).

## Evidence: the one SpGEMM-named function is a deleted stub

```
$ grep -rniE "spgemm" src/
src/kernels/kernels.cuh:184:void compute_spgemm_distances (cublasHandle_t& handle, cudaDeviceProp *deviceProps,
    const uint32_t d1, const uint32_t n, const uint32_t k,
     DATA_TYPE* d_P,  DATA_TYPE* d_C, DATA_TYPE* d_distances) = delete;
```

`compute_spgemm_distances` is declared with the C++11 `= delete` specifier —
there is no function body anywhere in the repository; calling it is a
compile error by design. It is the one and only place the string "spgemm"
appears in `src/` (case-insensitive), and it is explicitly disabled, not
merely unused. `compute_gemm_distances`/`compute_gemm_distances_fast`
(the dense-times-dense GEMM sibling, immediately above it in the same
header) are real, implemented functions; the SpGEMM variant was evidently
scaffolded and abandoned.

No other cuSPARSE sparse-times-**sparse** API (`cusparseSpGEMM_*`,
`cusparseXcsrgemm*`) appears anywhere in the tree:

```
$ grep -rniE "cusparseSpGEMM|csrgemm" . --include=*.cu --include=*.cuh \
    --include=*.c --include=*.cpp --include=*.h
(zero hits, .git excluded)
```

## Verdict

`popcorn: SKIPPED (no callable SpGEMM boundary exists -- the artifact's
kernel-k-means algorithm is built entirely on cusparseSpMM/cusparseSpMV
against a sparse assignment matrix, i.e. genuinely a different track
(spmm/spmv), not spgemm; the sole spgemm-named function,
compute_spgemm_distances in src/kernels/kernels.cuh:184, is declared
"= delete" with no implementation anywhere in the repository)`

No build attempted (cheap skip per ARTIFACT_GUIDE.md rule 7 / the scope
ruling: a genuinely absent kernel does not warrant spending build budget).
`source/` is kept for provenance per the standard layout; no `build.sh` or
`adapter.py` is provided (nothing to build or wrap).
