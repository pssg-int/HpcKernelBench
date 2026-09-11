# tot (ToT: Triangle Counting on Tensor Cores) — triangle-counting

**Status: BUILT+GATED**

- Paper: "ToT: Triangle Counting on Tensor Cores" (TPDS'25, extends
  PPoPP'25). `PAPER_KEY = journals/tpds/ChenY25` (matched by title +
  artifact_url in `../../../output/included.json`).
- Artifact: https://github.com/yuang-chen/ToT-TPDS25
- Commit cloned: `5f9cde83a528120e8551b69fdd74425358b476f1`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  host compiler `g++` 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`),
  `-arch=sm_80` (A100), C++20, `--extended-lambda
  --expt-relaxed-constexpr -Xcompiler=-fopenmp`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`. This machine's
  `nvcc` is already on `PATH` (no `module load cudatoolkit` needed).

## Track thinness

triangle-counting is one of two "thin tracks" in this integration pass
(2 GPU artifacts total in this track's survey; the other is tc-compare
below). Both are integrated.

## What the artifact actually is

Header-only C++/CUDA/Thrust library (`source/tot/`, `totlib` INTERFACE
CMake target) plus one CLI driver (`source/apps/tot.cu`) that reads a
Matrix-Market graph, converts CSR -> COO -> (optional upper-triangular
extraction) -> a tensor-core-friendly bitmap format, then counts triangles
via `tot::count_triangles_on_tensors` — masked SpGEMM implemented with
`nvcuda::wmma` fp16 Tensor Core MMA instructions (confirmed in
`source/tot/operations/kernel.h`'s `tot_kernel`).

## The shim (rule 1: wrap the kernel, not the driver)

`tot.cu`'s only entry point is the CLI driver, which always does
read+convert+count in one process. Per ARTIFACT_GUIDE.md rule 1 ("wrap at
the finest boundary available"), `tot_shim.cu` (this directory, **NOT**
part of the artifact — same role as `bench/artifacts/spmv/sspmv/csr_shim.cpp`)
instantiates ToT's own header-only template functions with concrete types
(`CsrMatrix<int,float,device_memory>`, `BitmapCOO<int,float,bmp64_t,4,device_memory>`)
and exposes two `extern "C"` entry points that split EXACTLY along tot.cu's
own phase boundaries:

- `tot_prepare(n, nnz, row_ptr, col_idx)` — CSR -> `convert_csr_to_coo` ->
  `convert_undirected` (symmetrize + dedup + drop self-loops) ->
  `extract_upper_triangular` (orientation: source-id < target-id, a fixed
  total order — **not** this project's domain-level degree-ascending
  order; both are valid orientations for the forward triangle-counting
  algorithm, count is identical either way) -> `convert_coo2bmp`
  (algorithm-native bitmap format). ALL of this matches tot.cu's own
  "[Converting to Bitmap]"/orientation preprocessing, and is exactly what
  the spec (`tc-gpu-kernel-exact`) requires be excluded from timed kernel
  time.
- `tot_count(handle)` — `tot::count_triangles_on_tensors(bmp, bmp, bmp)`
  ONLY: the tensor-core kernel itself, tot.cu's own
  "[Counting Triangles]" timed region. Since orientation was applied
  (equivalent to tot.cu's `-e 1`), the raw returned count is already exact
  (matches tot.cu's own `count = config.extract ? count : count / 6`
  branch with `extract=1` — no division needed here).

Zero lines of `source/` were modified. `tot_shim.cu` only `#include
"tot.h"` and calls the library's own public template functions.

## Build

```
./build.sh
```
Compiles `tot_shim.cu` directly with `nvcc -std=c++20 -arch=sm_80
--extended-lambda --expt-relaxed-constexpr -Xcompiler=-fopenmp,-fPIC
-I source/tot -shared -o tot_shim.so`. One harmless artifact-internal
warning (`M_row_pointers` declared but unused, in
`source/tot/operations/triangle_count.h`) — not touched, cosmetic only.
Idempotent (plain `nvcc` recompile, always exit 0 on success).

## Precision caveat (the artifact's own, self-documented, unresolved gap)

`count_triangles_on_tensors` accumulates in `half` (fp16) via Tensor Core
WMMA. The README warns this "may introduce precision errors during count
summation, particularly with large graphs," and that the `-e 1`
extraction (always applied by this adapter) only "mitigate[s] ... in some
cases" — not a guarantee. Per the track spec's correctness text
(`tc-gpu-kernel-exact`) and `notes_on_fairness`, **this adapter does not
loosen the gate for this**: the exact-integer-match gate is applied
unconditionally; if it fails on some (graph) instance that result would be
INVALID, not silently reported. On every graph tested here it passed
exactly (see below).

## Gate verification (login node, functional check only)

Mandated smoke command:
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel triangle-counting --variant tc-gpu-kernel-exact \
    --impl tot-tensorcore-tc --smoke
```
Result: **2/2 runs valid**, `err = 0.00e+00` (exact) on both synthetic
smoke graphs (`graph-smoke-uniform` 1500x1500/23926 edges,
`graph-smoke-powerlaw` 1500x1500/17968 edges).

Additionally cross-checked on a real cached SuiteSparse graph
(`ca-HepPh`, 12008 vertices / 236978 directed entries, symmetrized in
prepare()) against the domain's independent CPU reference
(`kernelbench.domains.graph.reference_triangle_count`, a pure-Python
merge/set-intersection counter, NOT scipy sparse matmul):

```
CPU reference triangle count: 3358499
ToT:                          3358499   MATCH
```

Reduced-protocol numbers only (warmup=5, reps=20 via `--smoke`, shared
login-node GPU) — explicitly non-conforming, not a timing claim, per
ARTIFACT_GUIDE.md rule 5. No timing sweep was run.

## Not done

- No sweep across the spec's `recommended_subset` (as-caida,
  p2p-Gnutella31, ..., com-orkut) — out of scope for the login-node
  budget; a functional/gate check only. `tc-gpu-e2e-preproc` (the
  amortized-preprocessing variant) was not separately exercised, though
  the same `prepare()`/`run()` split would serve it unchanged.
- `apps/tot.cu`'s own `-v 1` CPU verification path (`bfs_tc`) was not
  wired into the adapter — this project's own independent reference
  (`kernelbench.domains.graph.reference_triangle_count`) is used instead,
  which is what `run_variant`'s gate actually checks against.

## Verdict

`tot-tensorcore-tc: BUILT+GATED, exact match on 2 smoke graphs + 1 real
graph (ca-HepPh, 3,358,499 triangles); no artifact bugs found — the only
real hazard is the artifact's own self-documented fp16 precision risk,
which did not manifest on any tested graph.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, NVIDIA A100-SXM4-40GB (sm_80, gres `a100_1g.5gb`
  request landed a full card this run), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14, no cmake involved (`nvcc -std=c++20
  --extended-lambda --expt-relaxed-constexpr -Xcompiler=-fopenmp,-fPIC
  -shared ... -lgomp`); `-arch=sm_80`. Same harmless
  `M_row_pointers declared but never referenced` warning as before, cosmetic
  only.
- Build: OK. Build-system changes: none.
- Gate: `tc-gpu-kernel-exact`/`tot-tensorcore-tc`: PASS, 2/2 runs valid, err
  0.00e+00 (exact) on both smoke graphs (`graph-smoke-uniform`,
  `graph-smoke-powerlaw`) — the fp16 WMMA precision risk did not manifest;
  non-conforming (synthetic smoke, reduced protocol, high run-to-run
  variance on the shared GPU), as expected.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
