# Classification instructions — HPC kernel-optimization survey

You are classifying HPC papers (2020+, top venues) for a kernel benchmark project.
The goal: find every paper whose **main contribution is an optimized implementation
of one or more computational kernels** — the kind of work a benchmark could re-measure
(GEMM, SpMM, stencil, FFT, sort, ...) and an AI agent could attempt to reproduce
from scratch.

## INCLUDE (is_kernel_opt = true) iff the paper's primary contribution is:
- A faster/better **software implementation** of a computational kernel on real
  hardware: CPU, GPU (NVIDIA/AMD/Intel), many-core, or FPGA-via-HLS.
- A **code generator / compiler / autotuner** whose output is such kernels and whose
  evaluation is kernel performance (mark `approach` = codegen or autotuning).
- A **new sparse/dense format or data layout** evaluated via kernel performance.
- A **numerical-precision technique** (mixed precision, tensor-core exploitation)
  applied to a specific kernel.
- A multi-kernel composition where the kernels themselves are the contribution
  (e.g. fused SpMM+SDDMM, solver built from optimized building blocks).
- **HPC-crossing ML kernels**: tensor-core GEMM/mixed precision, GNN aggregation
  (SpMM-like), scientific convolutions, sparse attention with novel kernel work.

## EXCLUDE (is_kernel_opt = false), set excluded_reason:
- `hardware-accel`: ASIC/architecture proposals evaluated in simulation (most DAC
  accelerator papers). The contribution is hardware, not a runnable kernel.
- `pure-ml-serving`: LLM/DNN serving or training *systems* — KV-cache management,
  request scheduling, parallelism strategy, checkpointing — where no individual
  compute kernel is the contribution.
- `application`: whole-application porting/scaling studies (a QCD app, a weather
  model) where no single kernel implementation is isolated as the contribution.
- `communication`: collectives/network optimization (MPI allreduce etc.) with no
  compute-kernel contribution.
- `runtime-sched`: task scheduling, load balancing, runtime systems, resource mgmt.
- `perf-model`: performance modeling/prediction/analysis tools, profilers.
- `io-storage`: I/O, file systems, checkpointing, storage.
- `other`: anything else (security, verification, quantum-circuit compilation,
  EDA/CAD algorithms, ...). Quantum circuit simulation ON classical HPC hardware
  with kernel-level contribution counts as INCLUDE (category tensor or dense_la).
- `unclear`: cannot tell from title+abstract. Use sparingly; set confidence low.

Judge by the **contribution**, not the evaluation. A GNN training system that merely
*uses* cuSPARSE is excluded; a paper contributing a new SpMM kernel motivated by GNNs
is included. Data compression counts when the (de)compression kernel itself is
optimized for throughput on CPU/GPU (SZ/zfp family), not when compression is merely
applied. FPGA counts only for HLS/software-programmable implementations of standard
computational kernels, not for novel circuit architecture.

## Categories (a paper may have several; use these exact ids)
- `dense_la` — GEMM, batched GEMM, TRSM, LU/Cholesky/QR, eigensolvers, BLAS-like
- `sparse_la` — SpMV, SpMM, SDDMM, SpGEMM, SpTRSV, sparse factorization, formats
- `stencil_pde` — stencils, structured grids, FD/FV/FE kernels, lattice Boltzmann
- `fft_spectral` — FFT, DFT, NTT, spectral transforms
- `tensor` — tensor contraction, MTTKRP, TTM, tensor networks, einsum
- `graph` — BFS, PageRank, triangle counting, graph mining, traversal
- `primitives` — sort, scan, reduction, hash, set ops, string/sequence matching
- `nbody_md` — MD force kernels, N-body, FMM, particle-mesh, Ewald
- `unstructured_amr` — unstructured mesh, AMR, particle-in-cell kernels
- `solver_components` — Krylov, multigrid/AMG, preconditioners, tridiagonal solve
- `compression` — error-bounded lossy / lossless compression kernels
- `ml_kernels` — HPC-crossing ML kernels (see above)

## Output — write a JSON array, one object per input paper, ALL papers, same order:
```json
{
  "key": "<dblp key, copied exactly>",
  "is_kernel_opt": true,
  "confidence": "high|medium|low",
  "categories": ["sparse_la"],
  "kernels": ["SpMM", "SDDMM"],
  "platform": ["nvidia-gpu"],
  "approach": "manual|codegen|autotuning|library|algorithmic",
  "excluded_reason": null,
  "one_liner": "Row-decomposition SpMM/SDDMM kernels beating cuSPARSE by 1.8x"
}
```
- `platform` ids: `nvidia-gpu`, `amd-gpu`, `intel-gpu`, `cpu`, `arm-cpu`, `fpga`,
  `distributed` (multi-node), `sim` (simulator), `other`.
- `kernels`: short canonical names (SpMM, GEMM, 3D stencil, FFT, radix sort, BFS,
  MTTKRP, ...). Empty list if excluded.
- `one_liner`: <= 120 chars, what was optimized and the headline claim. For
  excluded papers, a few words on what it actually is.
- For excluded papers: categories=[], kernels=[], approach can be null.
- Output raw JSON only (no markdown fences) to the specified output file.
