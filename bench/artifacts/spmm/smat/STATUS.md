# smat (SMaT) — spmm

**Status: BUILT — gate FAILS (0/9 smoke err 2.0–5.7, 0/1 on `cant` err ~1.7e4): confirmed B-operand indexing bug in the artifact's own kernel (memcpy_async stride MMA_K instead of K), see findings below**

- Paper: "High Performance Unstructured SpMM Computation Using Tensor Cores"
  (SC'24). `PAPER_KEY = conf/sc/OkanovicKLBVH24`. Artifact:
  https://github.com/spcl/smat, cloned `--depth 1` at
  `057e44a9d08fc9fb4ba3fd6fe5aefe30895f88c1` (see `source.provenance`,
  already recorded from an earlier session in this project).
- Toolchain: `nvcc` 12.9 (`-ccbin` `/opt/cray/pe/gcc-native/14/bin/g++`,
  `-std=c++17`, `-gencode arch=compute_80,code=sm_80`). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.

## The gflags/CMake blocker turned out to be avoidable, not fixed

The prior session's survey (see git history of this file / the original
STATUS.md this replaces) found `source/src/cuda_hgemm`'s CMake build needs
system `gflags` (`find_package(gflags REQUIRED)`), unavailable here (no
sudo, no matching NERSC module). ARTIFACT_GUIDE.md anticipated two build-
system fixes: vendor gflags from source, or stub `DEFINE_*`/`FLAGS_*` via an
include shim. Neither was needed: `gflags` is used **only** by
`source/.../src/main.cu`'s CLI-flag parsing (`DEFINE_uint32(M, ...)`, etc.)
— the artifact's own benchmark-script entry point, which ARTIFACT_GUIDE.md
rule 1 says to wrap around, not link in. The actual preprocessing class
(`SparseMatrix`, `source/.../common/matrix.h`) and the kernel it launches
(`mmaCBTKernel`/`mmaCBTKernelSparse`, `source/.../src/mma/mmaCBT.cu`) are
both **header-only / dependency-free of gflags, OpenMP, and cublas**
(verified by tracing every `#include` reachable from `matrix.h`). `build.sh`
compiles `wrapper.cu` (this directory) + `mmaCBT.cu` directly via `nvcc`,
bypassing SMaT's own CMake build and `main.cu` entirely — no vendored
gflags, no stub, no CMake at all.

## What was wrapped (rule 1)

`SparseMatrix`'s constructor (unmodified): reads the `.mtx` file, builds CSR
on the host, converts CSR -> BCSR (16x16 = `MMA_M x MMA_K` tiles, storing
every tile with >=1 nonzero in dense form, skipping all-zero tiles — the
paper's real "block reordering"/tiling preprocessing), uploads to device.
Wrapped verbatim by `wrapper.cu::smat_prepare`, timed as preprocessing in
`adapter.py::prepare()` (rule 2).

`mmaCBTKernel`/`mmaCBTKernelSparse` (unmodified): a raw `mma.sync.aligned.
m16n8k16` PTX Tensor-Core kernel — the ONLY sparse kernel `main.cu`'s
benchmark actually calls (`tester.evaluateSparse2(mmaCBTKernel, ...)`; the
sibling `mmaB`/`mmaBT`/`mmaNaive` kernels are commented out there). Wrapped
via `wrapper.cu::smat_run`, one call = one kernel launch, exactly matching
`tester.h::evaluateSparse2`'s own call convention.

## IMPLEMENTATION FINDING: a genuine B-operand indexing bug in
## `mmaCBTKernelSparse` for any matrix with K > 16, invisible under the
## artifact's own (degenerate) evaluation methodology — confirmed by direct
## code inspection AND an isolated minimal reproduction, not inferred from
## gate numbers alone

**The bug** (`source/.../src/mma/mmaCBT.cu:51-52`):
```cpp
cooperative_groups::memcpy_async(group, &B_smem[0][0],
                &B[i * MMA_K + warp_col * K], B_size);
```
`B_smem` is `half[MMA_N][MMA_K]` (8x16 = 128 halfs), copied as ONE
CONTIGUOUS 128-element block starting at flat offset `i*MMA_K +
warp_col*K` (`i` = the current column-block index, `warp_col` =
`blockIdx.x * MMA_N`, i.e. the output column-tile's starting column, line
18). For this contiguous copy to land `B_smem[n][k]` on the intended
logical value `B[row = i*MMA_K + k, col = warp_col + n]` (B stored
column-major over the logical (K, N) operand, i.e. `flat = col*K + row`),
every one of the `MMA_N=8` "rows" of `B_smem` (each stride `MMA_K=16` in
the copy) would need to jump by `K` (the REAL number of columns of A, i.e.
rows of B) in the source, but a single `memcpy_async` call has no such
per-row stride — it advances by exactly `MMA_K` between B_smem's rows,
never by `K`. **These coincide only when `K == MMA_K == 16`.** For any
`K > 16` (i.e. essentially every real matrix — `MMA_K=16` is a hardcoded
tile width, not a size any interesting SpMM input actually has), the
second and later column-blocks (`i >= 1`) copy 128 CONSECUTIVE flat
elements starting at `i*MMA_K`, which lands `B_smem[n][k] = B_flat[i*MMA_K +
n*MMA_K + k]` instead of the intended `B_flat[(warp_col+n)*K + i*MMA_K +
k]` — reading the wrong 16-row window of B (row-shifted by `n*MMA_K`
instead of `n*K`) for every column `n >= 1`.

**Confirmed reproducible, isolated from the harness/gate machinery**:
1. A 16x16 fully-dense matrix (`K = MMA_K = 16`, exactly one column block)
   against a hand-labeled B (`B[k,n] = 10k+n`) reproduces `A @ B` **exactly**
   (bit-identical after fp16 rounding) once B is supplied to the kernel in
   the column-major-over-(K,N) layout the indexing arithmetic above implies
   (`np.ascontiguousarray(B.T)`, shape `(N, K)` row-major).
2. A 16x32 fully-dense matrix (`K = 32`, two column blocks) with a
   **uniform** B (all-ones) also reproduces the correct sum (1616 =
   16*1 + 16*100 for a two-value-per-block test) — i.e. the loop that
   accumulates across multiple column-blocks per row (`for (ptr =
   bcsrRowPtrA[blockRow]; ...)`, `mmaCBT.cu:37`) is itself correct.
3. The SAME 16x32 matrix with a **non-uniform** random B (`np.random.
   default_rng(0).uniform(-1,1,(32,8))`) reproduces `A @ B` correctly ONLY
   in the FIRST column-block's contribution; the result is wrong once the
   second block's (`i=1`) B-window is read — exactly the corrupted-offset
   behavior derived above.

**Why the artifact's own reported numbers never caught this**: SMaT's own
dense-`Matrix` class (`source/.../common/matrix.h:35-38`) that builds
`B_for_sparse` has its random-fill line **commented out**:
```cpp
//m_host_ptr[i] = __float2half(uniform(engine));
m_host_ptr[i] = __float2half(1);
```
i.e. the artifact's own B operand is **hardcoded to all-ones**, not random
— a uniform operand for which every possible memory-offset permutation of
B's elements yields the identical value, so the bug above is
mathematically invisible under the artifact's own evaluation. Compounding
this, `README.md`'s own `run_smat.sh` / SC'24 evaluation protocol runs with
correctness checking off by default (already flagged, independently, in
this track's own `benchspecs/spmm/spec.yaml` `notes_on_fairness`: "SMaT's
sweep script runs with `-enable_check=false`: the artifact's OWN reported
numbers come from unverified runs" — this integration's finding gives that
pre-existing suspicion a concrete, reproducible root cause). This adapter's
own dense operand is real, harness-generated `U(-1,1)` data (`np.random.
default_rng` matching `cpu_ref.reference_spmm`'s `_dense_operand`), so the
gate below exercises exactly the input class the artifact's own methodology
never tested.

`adapter.py`'s `prepare()` supplies B in the layout the kernel's indexing
arithmetic implies is intended for `K == MMA_K` (transposed to `(N, K)`
row-major, matching a correct column-major-over-`(K,N)` convention) — the
one demonstrably correct convention for single-column-block inputs — since
there is no other documented convention to prefer; the bug is in the
multi-block indexing itself; no amount of correct B layout on the caller's
side works around it (rule 4: not a gate-loosening choice, there IS no
alternative caller-side fix).

## Gate verification (login node, functional check + real matrix)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-tensorcore-fp16 --impl smat-mmacbt --precision fp16 \
    --smoke --warmup 1 --reps 3
```
Result: **0/9 valid** — `max_scaled_err` 1.98 – 5.66 (all smoke matrices have
`cols=4000 >> MMA_K=16`, i.e. hundreds of column blocks per row, well into
the bug's regime).

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-tensorcore-fp16 --impl smat-mmacbt --precision fp16 \
    --matrices cant --dims 128 --warmup 1 --reps 3
```
Result: **0/1 valid** — `cant` (62451x62451, nnz=4,007,383, `K` far beyond
16): kernel launches and runs cleanly (no crash, no NaN/Inf), `max_scaled_err
~= 1.8e4` at its worst entry (median ~0.34) — again the indexing bug, not a
build or launch failure.

Both reduced-protocol (warmup=1, reps=3, shared login-node GPU;
`conforming: False`), per ARTIFACT_GUIDE.md rule 5.

## Not done

- The full CMake `hgemm` CLI binary (with vendored gflags) was not built —
  turned out unnecessary once wrapped at the `SparseMatrix`+`mmaCBTKernel`
  boundary (see above). If a future task wants to cross-check this
  wrapper's numbers against SMaT's own driver end-to-end, the vendor-gflags
  recipe ARTIFACT_GUIDE.md anticipated (`cmake -Dgflags_DIR=<vendored
  prefix>/lib/cmake/gflags` when invoking `source/src/cuda_hgemm/build.sh`)
  is still the right approach; not pursued since it would only reproduce
  the SAME masked-by-all-ones-B result already explained above.
- The other four kernel variants in `source/.../src/mma/` (`mmaNaiveKernel`,
  `mmaTKernel`, `mmaBKernel`, `mmaBTKernel`) were not wrapped — `main.cu`
  itself only exercises `mmaCBTKernel` (the others are commented out there),
  so `mmaCBTKernel` is the one this integration targets per rule 1. Checked
  (grep, not run) whether they share the same B-indexing bug: they do NOT
  appear to — `mmaB.cu`/`mmaBT.cu`/`mmaT.cu`/`mma_naive.cu` all load B
  per-thread as `B[i*MMA_K + (warp_col + lane_id/2)*K]` (the column index
  `warp_col + lane_id/2` is multiplied by the REAL `K` inline, per thread,
  not via one bulk `memcpy_async` with a fixed `MMA_K` stride across
  `MMA_N` rows), which does not have the same single-stride flaw. This bug
  looks specific to `mmaCBT.cu`'s optimization (a bulk async copy instead of
  per-thread strided loads) introducing the wrong stride for the
  across-column dimension. Not independently run/verified against a
  multi-block test the way `mmaCBTKernel` was — out of scope for this
  session's time budget.
- No sweep across the full `recommended_subset` or N in {256, 512}; not
  exercised.
- `preprocess/` (a separate C++ tool for an alternative row/column
  permutation to further reduce nonzero-block count, per the paper's
  abstract) was not built or exercised — `SparseMatrix::csrToBcsr()`'s
  direct (unpermuted) tiling is what this adapter uses, matching what
  `main.cu`'s own default invocation does.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge,
  `$CXX`/`$KB_CXX`), torch 2.8.0+cu128, Python 3.12.14, arch
  `compute_80,code=sm_80` as recorded.
- Build: OK (only pre-existing `#177-D unused variable` remarks). Build-
  system changes: none (build.sh's existing `${HOST_COMPILER:-${CXX:-...}}`
  fallback already reads this machine's exported `CXX`).
- Gate: `spmm-tensorcore-fp16` (fp16, smoke, warmup=1, reps=3) — 0/9 valid,
  `max_scaled_err` 1.979e+00 (uniform, dim=128) – 5.662e+00 (powerlaw,
  dim=512) — matches the recorded range (1.98-5.66) to 3 significant
  figures. `cant` (N=128) — 0/1 valid, `max_scaled_err = 1.672e+04` vs the
  recorded "~1.8e4" — same order of magnitude, both far beyond `tol=0.01`
  (the recorded figure was itself only an approximate "worst entry" value,
  not a reproducibility target to the last digit).
- Deviation from the recorded ruling: none — the B-operand indexing bug
  (`memcpy_async` stride `MMA_K` instead of `K`) reproduces on this
  machine's toolchain/GPU exactly as documented: same failure pattern, same
  order-of-magnitude errors, both smoke and real-matrix.
- Verdict here: BUILT (gate FAILS, confirmed B-operand indexing bug) — same
  as the recorded ruling.
