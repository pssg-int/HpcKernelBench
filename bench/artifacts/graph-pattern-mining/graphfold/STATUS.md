# graphfold (GraphFold) — graph-pattern-mining

**Status: BUILT+GATED**

- Paper: "Exploiting Fine-Grained Redundancy in Set-Centric Graph Pattern
  Mining" (PPoPP'24, `conf/ppopp/LinMSZXT24` in `../../../output/included.json`,
  matched by artifact_url).
- Artifact: https://github.com/GPM-lib/GraphFold
- Commit cloned: `36c12f774628d3ec07ac42fde7a801a7372946dd`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  host compiler `g++` 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`),
  `-arch=sm_80` (A100), C++14, `--expt-extended-lambda
  --expt-relaxed-constexpr -Xcompiler=-fopenmp,-fPIC`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`. `LD_PRELOAD=/usr/lib64/
  libstdc++.so.6` required at runtime (CXXABI/GLIBCXX version mismatch
  between the system libstdc++ the Python interpreter links and the newer
  one `g++`-14/nvcc produce symbols against).

## Pattern wrapped

Domain headline: exact **K4 (4-clique) COUNTING**
(`params['clique_k']`, default 4; see
`bench/kernelbench/domains/graph.py`'s module docstring deviation (4)).
GraphFold's own `CFSolver` (`src/CFSolver.cuh`) implements exactly this —
`CF4`-`CF7`, dispatched by `k`. This adapter supports the full k=4..7 range
CFSolver ships (`GraphFoldKCliqueCount.prepare()` guards `4 <= k <= 7`), even
though the domain's own default/gated value is k=4.

## What the artifact actually is

Header-only-ish C++/CUDA/Thrust library (`source/src/`, template functions
in `.cuh` headers) plus CLI test drivers (`source/test/test_*.cu`) that each
load a Matrix-Market graph via `Loader`/`Graph::Init`, run one
`Engine::Run*()` method, and print a count. `Engine::RunCF()`
(`src/Engine.h`) is itself a clean two-step call —
`hg.orientation(); hg.copyToDevice(...); CFSolver(hg, k, result, n_dev,
cal_mode);` — the cleanest split of the three artifacts in this track.

## The shim (rule 1: wrap the kernel, not the driver)

`gf_shim.cu` (this directory, **NOT** part of the artifact — same role as
`bench/artifacts/triangle-counting/tot/tot_shim.cu`) builds a
`project_GraphFold::Graph<int,int>` **directly from our own CSR arrays**
(`Graph::Init` takes a plain `std::vector<std::pair<VID,VID>>` edge list —
exactly what `Loader::Build`'s own `_load_mtx` parser produces from a file,
so handing it our CSR's `(row,col)` pairs directly is equivalent, minus the
file round-trip; this exact "build from arrays, not a file" pattern is also
how GraphFold's own multi-GPU partitioner builds sub-`Graph`s internally):

- `gf_prepare(n, nnz, row_ptr, col_idx, k)` — `Graph::Init` (CSR→edge-list
  build) → `hg->orientation()` (DAG construction, degree-ascending,
  ties broken by vertex id — **same rule** as this project's own
  `_orient_by_degree`) → `hg->copyToDevice(...)` (H2D). Exactly
  `Engine::RunCF()`'s own preprocessing scope, done ONCE.
- `gf_count(handle)` — `CFSolver(*hg, k, result, 1, e_centric)` ONLY: the
  `cliqueK_graphfold` CUDA kernel itself, GraphFold's own "CFk matching
  time" timed region (`src/CFSolver.cuh`).
- `gf_free(handle)` — releases the `Graph`.

Zero lines of `source/` were modified.

### Build-compat shim (not a kernel patch)

`source/thirdparty/thrust` vendors an **old** Thrust (needed for
`thrust/system/cuda/experimental/pinned_allocator.h`, which CTK-12.9's
bundled Thrust no longer ships). Mixing that old vendored Thrust with CTK
12.9's bundled CUB fails catastrophically (hundreds of `THRUST_NS_QUALIFIER
undefined` errors deep in `cub/device/dispatch/dispatch_streaming_reduce.cuh`
— verified by trying it first). Instead, `build.sh` uses CTK 12.9's own
(version-matched) Thrust+CUB throughout, and `compat_include/thrust/system/
cuda/experimental/pinned_allocator.h` (this directory) stubs the ONE missing
header — it only needs to make the type name `thrust::cuda::experimental::
pinned_allocator<T>` resolvable at parse time for an **unused**
`Buffer(const thrust::host_vector<T, pinned_allocator<T>>&)` constructor
overload in `src/utils/buffer.h` (this shim's code path only ever builds
`Buffer` from a `thrust::device_vector` or a raw pointer+size — that
overload is never instantiated). This is a build-system/CTK-version
compatibility fix (ARTIFACT_GUIDE.md rule 3's "CUDA-version guards" carve-out),
not a change to any kernel or algorithm.

## Build

```
./build.sh
```
Idempotent. Compiles `gf_shim.cu` with `nvcc -std=c++14 -arch=sm_80
--expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-fopenmp,-fPIC,-w
-DTHRUST_IGNORE_CUB_VERSION_CHECK`-free (not needed with this include-order
fix) `-I compat_include -I source -shared -o gf_shim.so`. Built clean, no
warnings of note (artifact's own `-w` suppression matches its Makefile).

## Gate verification (login node, functional check only)

Mandated smoke command:
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel graph-pattern-mining --variant gpm-kernel-small-f32 \
    --impl graphfold-cfsolver-kclique --smoke
```
Result: **2/2 runs valid**, `err = 0.00e+00` (exact) against
`kernelbench.domains.graph.reference_k_clique_count` (independent bitmask
Python reference) on both synthetic smoke graphs (`graph-smoke-uniform`
1500x1500, `graph-smoke-powerlaw` 1500x1500).

### Real-graph cross-validation

The Python K4 reference is combinatorially infeasible on every cached real
graph tried (ca-HepPh: 3.3M triangles already takes 6.5s just to *count*
triangles; K4 recursion times out well past any reasonable login-node
budget) — this is a genuine, documented limitation of the exact-but-slow
Python reference at real-graph scale, not something silently worked around.
Per the track spec's own sanctioned methodology for this regime
(`spec.yaml`'s `correctness` text: "cross-validate between >=2
independently-implemented systems in the suite" when no single-machine
brute-force reference is feasible), this artifact's K4/K5 counts were
cross-checked against **glumin** (a completely independent implementation)
on `ca-HepPh` (12,008 vertices / 236,978 directed entries, symmetrized in
`prepare()`):

```
K4: graphfold=150281372  glumin=150281372   MATCH
K5: graphfold=6491049885 glumin=6491049885  MATCH
K4: graphfold=150281372  glumin=150281372  fringe-sgc=150281372   ALL MATCH (3-way)
```

Reduced-protocol numbers only (warmup=5, reps=20 via `--smoke`, shared
login-node GPU) — explicitly non-conforming, not a timing claim, per
ARTIFACT_GUIDE.md rule 5. No timing sweep was run.

## Not done

- No sweep across the spec's `recommended_subset` — out of scope for the
  login-node budget; functional/gate check only.
- `gpm-e2e-preproc` (amortized-preprocessing variant) not separately
  exercised, though the same `prepare()`/`run()` split serves it unchanged.
- K6/K7 built and callable but not separately smoke-tested (K4 is gated;
  K5 cross-validated above; K6/K7 use the identical code path with a
  different kernel dispatch, same as K4/K5).

## Verdict

`graphfold-cfsolver-kclique: BUILT+GATED, exact match on 2 smoke graphs
(vs. independent Python reference) + K4/K5 cross-validated on 1 real graph
(ca-HepPh) against glumin (and K4 additionally against fringe-sgc, 3-way
agreement) — no artifact bugs found.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14; `-arch=sm_80` (via toolchain.sh's
  `KB_SM=80` default), no cmake used.
- Build: OK. Build-system changes: none (build.sh's existing
  `${CXX:-...}`/`${CUDA_HOME:-...}` knobs already resolved to the zaratan
  toolchain via `bench/artifacts/toolchain.sh`/`bench/env.sh`; the CTK-12.9
  Thrust/CUB compat_include shim built cleanly against CTK 12.8 too).
- Gate: gpm-kernel-small-f32, int64: PASS, 2/2 runs valid, err 0.00e+00 <= None
  (structural/exact), 0.1677 ms and 0.2288 ms medians (high per-run variance
  flagged by the runner — shared/contended MIG slice, not a correctness
  issue). Same mandated smoke command as recorded (`--impl
  graphfold-cfsolver-kclique --smoke`); "CF4 matching time" log lines confirm
  the CFSolver K4 dispatch.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
