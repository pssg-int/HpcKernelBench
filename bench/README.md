# bench/ — the runnable harness

A spec-driven benchmark runner for the tracks in `../benchspecs/`. **25 kernels
across 10 domains** are implemented; adding a track means adding a domain module,
not rewriting the harness (see `DOMAIN_GUIDE.md`).

| domain | kernels implemented | primary unit |
|---|---|---|
| sparse | spmv, spmm, sddmm | GFLOP/s |
| dense | gemm, gemv, cholesky, blas-level1-2 | GFLOP/s |
| stencil | stencil | **GCUP/s** (spec refuses GFLOP/s) |
| graph | bfs, pagerank, triangle-counting, connected-components | GTEPS |
| compression | lossy-compression, lossless-compression | **GB/s + ratio (a pair)** |
| primitives | scan-reduction, topk-selection, hash-table | GB/s, Mops/s |
| solvers | cg-krylov, preconditioner | GFLOP/s |
| tensor | mttkrp, tensor-contraction | GFLOP/s |
| ml | convolution, attention-kernel | GFLOP/s |
| spectral | fft, ntt | GFLOP/s, **NTT/s** (spec refuses GFLOP/s for NTT) |

`./smoke_all.sh` runs every implemented kernel end to end (CPU, synthetic
inputs, seconds each) — **25/25 green**. It is safe on a login node and every
result it produces is marked non-conforming by construction.

The point of this layout: `../benchspecs/<kernel>/spec.yaml` is not documentation
that a script happens to agree with — it is *read at run time* and its protocol
is what executes. A result that deviates from its spec says so, in the record.

```bash
source env.sh            # sets PY/CUDA_HOME/... from KB_* (edit env.sh on a new machine)
$PY check_env.py         # what this machine has / lacks -- see ENVIRONMENT.md

# what does the spec actually demand?
$PY -m kernelbench.runner --kernel spmm --list

# tiny synthetic matrices, reduced protocol — safe anywhere, NOT publishable
$PY -m kernelbench.runner --kernel spmm --variant spmm-cpu-kernel-f32 \
    --impl scipy-csr-spmm --smoke --dims 32,128

# real SuiteSparse matrices (downloaded and cached on first use)
$PY -m kernelbench.runner --kernel spmm --variant spmm-cpu-kernel-f32 \
    --impl scipy-csr-spmm --matrices cant,pdb1HYS --dims 32,128

# read the results
$PY -m kernelbench.report "results/*.json" --html results/report.html
```

## What the harness enforces

Order is fixed and comes straight from the specs:

1. **`prepare()`** — one-shot format conversion / packing, timed **once**,
   reported separately, never folded into per-call time.
2. **correctness gate** — against an independent fp64 reference, **before** any
   timing. A failing gate yields `valid: false` and *no* throughput number. This
   is deliberate: the specs found that GE-SpMM's check macro is commented out by
   default and SMaT's sweep runs `-enable_check=false`, i.e. their own published
   numbers come from unverified runs.
3. **warmup** — `protocol.warmup` iterations, discarded.
4. **measurement** — `protocol.reps` iterations, **one timestamp pair per
   iteration**. RoDe / GE-SpMM / SMaT wrap the whole loop in a single CUDA-event
   pair and report the mean, which cannot expose variance; this harness reports
   median + min/max/stdev/p10/p90 and keeps every raw sample in the record.
5. **environment gate** — login node, missing SLURM allocation, or unlocked GPU
   clocks each mark the document `conforming: false` with the reason attached.

## The correctness gate, and why it is not "relative error"

Building this surfaced a real defect in the specs (now recorded in their
`notes_on_fairness`): **"max relative error < 1e-6" is not a well-defined gate**
for these kernels. An SpMM output element is a sum of tens of signed products, so
cancellation routinely leaves a true value near zero, and `|out−ref|/|ref|` then
explodes regardless of kernel quality. Measured on SuiteSparse/cant, N=128, with
scipy's own fp32 CSR product:

| view | value | verdict against the spec's 1e-6 |
|---|---|---|
| naive pointwise `\|out−ref\|/\|ref\|` | 3.9e-2 | **fails** (spuriously) |
| `\|out−ref\|/(\|A\|·\|B\|)` (this harness) | 4.2e-7 | passes |
| L2 relative | 1.3e-7 | passes |

`|A|·|B|` is the componentwise backward-error denominator, so passing means the
kernel is backward stable at the stated precision — which is what the gate was
trying to say. All four views are recorded in every result.

## Layout

```
kernelbench/
  spec.py       load benchspecs/<k>/spec.yaml -> executable Protocol; records
                which constants were literal / parsed / defaulted, and resolves
                "same tolerance as <other-variant>" cross-references
  matrices.py   SuiteSparse fetch+cache (auto group lookup) + synthetic smoke
                matrices (uniform / banded / powerlaw row-length regimes)
  harness.py    the measurement loop and the correctness gate
  metrics.py    flop/byte accounting, one place, per each spec's metric field
  env.py        machine capture + "this number is not publishable" warnings
  runner.py     CLI
  report.py     text/HTML report + per-(matrix,dim) leaderboard
  impls/
    cpu_ref.py  scipy CSR (reference) and a naive CSR SpMV (leaderboard floor)
    gpu_cuda.py cuSPARSE-via-torch baselines + the custom kernels below
csrc/
  kernels.cu    warp-per-row SpMV/SpMM and warp-per-nonzero SDDMM baselines
  Makefile      `make` -> libkernelbench.so ; `make check` -> compile only
results/        one JSON document per invocation
matrices/       SuiteSparse cache (git-ignored)
```

## Status

- CPU path: **working** for all 25 kernels (`./smoke_all.sh`), plus real
  SuiteSparse matrices for the sparse domain.
- CUDA kernels: **compile clean** (`make check`, sm_80) and are wired into
  `impls/gpu_cuda.py`. No GPU **timing** runs from the login node's shared A100
  — run that from a compute-node allocation. Functional (`--smoke`)
  correctness gates DO run there, and did on 2026-09-06 for every built-in
  CUDA impl, after fixing a genuine harness bug: every one of them drew its
  dense operands with `torch.Generator`+`torch.rand`/`randn` instead of the
  `numpy.random.default_rng(seed)` recipe every reference/CPU impl uses,
  feeding the GPU kernel a different operand than the one being checked
  against (spmm's two built-ins were 0/6 INVALID under both
  `spmm-gpu-kernel-f32` and `spmm-binary-adjacency-kernel` before the fix).
  After the fix: spmv 6/6, spmm 24/24 (both variants), sddmm 12/12, spgemm
  3/3, fft 8/8, sparse-attention-kernel 7/7, quantized-gemm 3/3, mttkrp 3/3
  (also fixing an unrelated pre-existing missing `import torch` in its
  `run()`), tensor-contraction 6/6, gemm 3/3, gemv 3/3 all valid.
  attention-kernel is 3/4 valid — the remaining failure (decode+GQA) is a
  separate, genuine bug in torch's `is_causal` alignment for Sq!=Sk, not RNG,
  documented in `TorchAttention`'s docstring. convolution and stencil's
  torch-conv paths, and cholesky, are blocked by pre-existing environment
  defects (this venv's cuDNN/torch installs are missing shared libraries),
  documented in their classes' docstrings, not code bugs. `./smoke_all.sh`
  (CPU-only) stayed 35/35 green throughout.
- `spec.py` parses **all 90 specs** (256 variants): 143 carry a numeric
  tolerance, 53 declare a structural/exact gate (counts, labels, bit-exact
  roundtrips — legitimately non-numeric), 60 have compound gates a domain module
  must implement, and 22 have an incomplete protocol constant. Nothing is
  silently defaulted; the runner prints provenance under `--list` and the
  harness refuses to certify a run whose gate it could not establish.

## Not yet built

- GPU e2e/preprocessing variants (`spmm-gpu-e2e-preproc-f32`,
  `spmv-e2e-preproc`) need an implementation that owns its own format
  conversion; the harness already times and amortizes it, the impls do not yet
  expose a non-CSR native format.
- Quantized variants (`spmm-gpu-quantized-int`, `sddmm-quantized-dlmc`) need the
  DLMC dataset and integer reference kernels; the exact-match gate is in place.
- `spmv-symmetric-kernel` needs the unfolded-flop rule wired to a symmetric
  storage impl (`metrics.flops(..., unfolded_nnz=...)` already supports it).
