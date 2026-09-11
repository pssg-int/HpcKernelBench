# glumin (GLumin) — graph-pattern-mining

**Status: BUILT+GATED**

- Paper: "GLumin: Fast Connectivity Check Based on LUTs For Efficient Graph
  Pattern Mining" (PPoPP'25, `conf/ppopp/CaoMLT25` in
  `../../../output/included.json`, matched by artifact_url).
- Artifact: https://github.com/AnySparse/GLUMIN
- Commit cloned: `b6f4e8c3fb2fd88c5aa17971eebbbe36241ecd69`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9, host compiler `g++` 14.3.0
  (`/opt/cray/pe/gcc-native/14/bin/g++`), `-arch=sm_80` (A100), C++17,
  `--extended-lambda --expt-relaxed-constexpr -Xcompiler=-fopenmp,-fPIC`,
  `-DUSE_GPU` (matches the artifact's own `common.mk` NVFLAGS). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required at runtime (CXXABI/
  GLIBCXX version mismatch, same as the other two artifacts in this track).

## Pattern wrapped

Domain headline: exact **K4 (4-clique) COUNTING**
(`params['clique_k']`, default 4; see
`bench/kernelbench/domains/graph.py`'s module docstring deviation (4)).
GLumin's own `CliqueSolver` (`src/clique/clique_GM_LUT.cu` — the **+LUT**
variant of G2Miner's `CliqueSolver`, which is GLumin's actual paper
contribution per survey.md: "GLumin's contribution is a drop-in
connectivity-check accelerator" layered on G2Miner/GraphFold/AutoMine, as
opposed to the plain non-LUT `clique_GM.cu` G2Miner baseline the repo also
ships) implements k=4..7 (`clique4/5/6/7_warp_*_subgraph` kernels). This
adapter exposes the full k=4..7 range, guarded in `prepare()`, even though
the domain's own default/gated value is k=4.

## What the artifact actually is

A large CUDA graph-mining framework (`source/include/`,
`source/src/{G2Miner,GraphFold,AutoMine,clique,pattern,common}/`) — NOT
header-only: `Graph`'s own methods (`allocateFrom`, `orientation`,
`init_edgelist`, ...) and `VertexSet`'s static buffer-pool members are
defined out-of-line in `src/common/graph.cc` / `src/common/VertexSet.cc`
(confirmed via `src/common.mk`'s own `OBJS=main.o VertexSet.o graph.o`).
`CliqueSolver` (`src/clique/clique_GM_LUT.cu`) is monolithic: it builds
`GraphGPU gg(g); gg.init_edgelist(g);` (H2D) AND allocates its whole kernel
workspace (frontier list, `BinaryEncode` LUT buffer, `d_total`) INSIDE the
same function call that launches the counting kernel, separated from the
kernel launch only by its own internal (not externally callable) `Timer`.

## The shim (rule 1: wrap at the finest boundary available)

`gl_shim.cu` (this directory, **NOT** part of the artifact) reproduces
CliqueSolver's own setup code **verbatim** (same `GraphGPU`/`BinaryEncode`/
`maximum_residency` calls, same launch-config arithmetic, literally copied
from `clique_GM_LUT.cu`), split across two functions instead of one:

- `gl_prepare(n, nnz, row_ptr, col_idx, k)` — builds a `Graph` **directly
  from our own CSR arrays** via the artifact's own public
  `allocateFrom()`/`fixEndEdge()`/`constructEdge()` API — the exact same
  pattern the artifact's own `src/common/graph_partition.cc` uses to build
  a `Graph` from arrays instead of a file (`Graph g; g.allocateFrom(nv,
  ne);` — using the DEFAULT constructor, not `Graph(nv, ne)` directly, so
  that `nnz` is zero-initialized; `Graph(nv,ne)`'s own constructor leaves
  several members, including `nnz`, uninitialized, which would silently
  break `Graph::init_edgelist()`'s `if (nnz != 0) return;` guard — worked
  around here, not a patch to the artifact). Then `g->orientation()` (DAG,
  same as passing `USE_DAG=true` to the file-based constructor),
  `GraphGPU(g)`/`init_edgelist(g)` (H2D), and ALL of CliqueSolver's own
  pre-kernel workspace allocation — exactly CliqueSolver's own untimed
  region.
- `gl_count(handle)` — launches ONLY the appropriate
  `cliqueK_warp_*_subgraph` kernel (the artifact's own unmodified
  `__global__` function) + D2H readback — CliqueSolver's own timed region.
  **One addition, documented, not a kernel change**: the accumulator
  `d_total` is explicitly reset to 0 before every call, because this
  harness reuses one handle across warmup+reps, whereas the artifact's own
  CLI driver only ever calls `CliqueSolver()` once per process (so it only
  zeroes `d_total` once, before its single call) — without this reset,
  counts accumulate across repetitions.
- `gl_free(handle)` — releases everything.

Zero lines of `source/` were modified; the kernel functions themselves
(`clique4_warp_vertex_subgraph` etc., in `src/clique/gpu_kernels/*.cuh`) are
`#include`d and called verbatim.

## Build

```
./build.sh
```
Idempotent. Compiles `gl_shim.cu` PLUS `source/src/common/graph.cc` and
`source/src/common/VertexSet.cc` (required — see "not header-only" above)
with `nvcc -std=c++17 -arch=sm_80 --extended-lambda --expt-relaxed-constexpr
-Xcompiler=-fopenmp,-fPIC,-w -DUSE_GPU -I source/include -I
source/src/clique/gpu_kernels -shared -o gl_shim.so`. No vendored Thrust/CUB
issue here (GLumin relies on the CTK's own `<cub/cub.cuh>`, already
version-matched with nvcc 12.9 — unlike graphfold/, no compat shim needed).

## Gate verification (login node, functional check only)

Mandated smoke command:
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel graph-pattern-mining --variant gpm-kernel-small-f32 \
    --impl glumin-cliquesolver-lut-kclique --smoke
```
Result: **2/2 runs valid**, `err = 0.00e+00` (exact) against
`kernelbench.domains.graph.reference_k_clique_count` on both synthetic
smoke graphs.

### Real-graph cross-validation

Same combinatorial-infeasibility issue as graphfold's Python-reference check
(see graphfold/STATUS.md) — cross-validated instead against **graphfold**
(and, for K4, additionally against **fringe-sgc**) on `ca-HepPh` (12,008
vertices / 236,978 directed entries):

```
K4: graphfold=150281372  glumin=150281372   MATCH
K5: graphfold=6491049885 glumin=6491049885  MATCH
K4: graphfold=150281372  glumin=150281372  fringe-sgc=150281372   ALL MATCH (3-way)
```

Reduced-protocol numbers only (warmup=5, reps=20 via `--smoke`, shared
login-node GPU) — explicitly non-conforming, not a timing claim, per
ARTIFACT_GUIDE.md rule 5. No timing sweep was run.

## Real artifact bug found

`Graph(vidType nv, eidType ne) { allocateFrom(nv, ne); }` (`include/graph.h`)
does **not** zero-initialize `nnz` (nor `is_directed_`, `is_bipartite`,
`has_reverse`, `n_vert0`, `n_vert1` — only `allocateFrom`'s own
`vertices`/`edges` arrays get set). `Graph::init_edgelist()`
(`src/common/graph.cc`) guards on `if (nnz != 0) return nnz;` to memoize —
with `nnz` uninitialized garbage, this guard can silently short-circuit and
skip building `src_list`/`dst_list` entirely. The artifact's OWN
`src/common/graph_partition.cc` uses this exact `Graph(nv, ne)` constructor
(`subg = new Graph(nv_subg, ne_subg);`) without ever calling
`init_edgelist()` on the result afterward, so this landmine is latent, not
yet triggered, in the artifact's own shipped code paths — but any future
caller (including this adapter, had it used `Graph(nv,ne)` directly instead
of `Graph()` + `allocateFrom()`) combining that constructor with
`init_edgelist()` would hit it. Documented here rather than patched, per
ARTIFACT_GUIDE.md rule 3 (this adapter works around it by using the
zero-initializing default constructor instead).

## Not done

- No sweep across the spec's `recommended_subset` — functional/gate check
  only, login-node budget.
- `gpm-e2e-preproc` variant not separately exercised.
- K6/K7 built and callable but not separately smoke-tested.

## Verdict

`glumin-cliquesolver-lut-kclique: BUILT+GATED, exact match on 2 smoke graphs
+ K4/K5 cross-validated on 1 real graph against graphfold (K4 additionally
3-way against fringe-sgc) — one real (latent, non-triggered by the
artifact's own current code paths) uninitialized-member bug found in the
artifact's `Graph(nv,ne)` constructor, worked around (not patched) in the
adapter.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14; `-arch=sm_80` (via toolchain.sh's
  `KB_SM=80` default), no cmake used.
- Build: OK. Build-system changes: none (build.sh's existing
  `${CXX:-...}`/`${CUDA_HOME:-...}` knobs already resolved to the zaratan
  toolchain via `bench/artifacts/toolchain.sh`/`bench/env.sh`).
- Gate: gpm-kernel-small-f32, int64: PASS, 2/2 runs valid, err 0.00e+00 <= None
  (structural/exact), on graph-smoke-uniform (0.070 ms) and graph-smoke-powerlaw
  (0.115 ms). Same mandated smoke command as recorded (`--impl
  glumin-cliquesolver-lut-kclique --smoke`); "Orientation enabled, using DAG"
  log lines match the recorded DAG-build path.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
