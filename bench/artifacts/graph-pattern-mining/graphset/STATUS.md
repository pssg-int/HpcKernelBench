# graphset (GraphSet) — graph-pattern-mining

**Status: BUILT+GATED**

- Paper: "GraphSet: High Performance Graph Mining through Equivalent Set
  Transformations" (SC'23, `conf/sc/ShiZWCZHYC23` in
  `../../../output/included.json`). Selected as a **core baseline** under
  the revised kernel-centrality rule (`output/kernel_centrality.json` key
  `graph-pattern-mining|conf/sc/ShiZWCZHYC23`: centrality `core`, regime
  `matches` — "equivalent-set-transformation compiler for subgraph pattern
  counting, mico/patents/orkut/livejournal/friendster... the only one with
  a real automated correctness oracle"; see `output/baseline_selection.md`).
- Artifact: https://github.com/sth1997/GraphSet
- Commit cloned: `3bc6e6b9e2e0f61ba799a9c25cab96a9da6b4dc8`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9, host compiler `g++` 14.3.0
  (`/opt/cray/pe/gcc-native/14/bin/g++`), `-arch=sm_80` (A100), C++14,
  `-rdc=true`, `-march=native` (host-only, for `set_operation.cpp`'s SSE/
  SSSE3/POPCNT intrinsics), MPI headers from Cray MPICH
  (`/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3`) — needed transitively by
  `source/src/graph.cpp`'s `#include <mpi.h>` (its own MPI-based
  pattern-matching path, unused by this adapter, but part of the same
  translation unit as `reduce_edges_for_clique`, which we DO need).
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required at runtime.

## Pattern wrapped

Domain headline: exact **K4 (4-clique) COUNTING**
(`params['clique_k']`, default 4; see
`bench/kernelbench/domains/graph.py`'s module docstring deviation (4)).
GraphSet ships a **dedicated clique driver**,
`source/gpu/gpu_clique.cu`'s `main()`: it builds a `Pattern` that IS a
complete graph K_k (`for i: for j>i: add_edge(i,j)`), compiles it via
`Schedule_IEP` — the equivalent-set-transformation schedule search that IS
GraphSet's actual paper contribution (in-exclusion-optimized prefix sharing
across the pattern's automorphisms) — and launches `gpu_pattern_matching`,
the SAME general pattern-matching kernel every GraphSet application shares,
specialized only by the compiled `Schedule_IEP` it is handed. This adapter
exposes `clique_k` in `[3,7]` (`get_third_layer_size`/`job_queue`-style
degree-array accesses need `nnodes>=3`; `PAT_SIZE`-equivalent structures
here are dynamically sized, so there is no hard upper compile-time bound,
but k<=7 matches the range every other adapter in this track exercises).

## What the artifact actually is

`pattern_matching_init()` (`source/gpu/gpu_clique.cu`) is monolithic
exactly like GLumin's `CliqueSolver`/TileSpGEMM's `tilespgemm()`: it builds
the device Graph representation AND the device `GPUSchedule` AND launches
the kernel AND reads back the count, all in one function, with no public
sub-boundary.

## The shim (rule 1: wrap at the finest boundary available)

`gs_shim.cu` (this directory, **NOT** part of the artifact) reproduces
`pattern_matching_init()`'s own body **verbatim**, split into:

- `gs_prepare(n, e_cnt, vertex_ptr, edge_idx, k)` — builds the artifact's
  own `Graph` struct **directly from our CSR** (no file I/O, bypassing
  `DataLoader`'s binary `.g` format entirely; `new[]`-allocated to match
  `Graph::~Graph()`'s `delete[]`), sets `VertexSet::max_intersection_size`
  from the actual max out-degree (a safe, at-least-as-conservative bound
  vs. the artifact's own "second-largest degree" file-header convention,
  which we bypass), calls the artifact's own `reduce_edges_for_clique()`
  (`== erase_edge()`, `source/src/graph.cpp`, unmodified — keeps, for every
  undirected edge `{u,v}`, exactly the directed entry `max(u,v)->min(u,v)`,
  a DAG total-ordered by vertex id), builds the K_k `Pattern` +
  `Schedule_IEP` (pure pattern-only preprocessing), then reproduces
  `pattern_matching_init()`'s own H2D setup (`edge_from` computation,
  `dev_edge`/`dev_edge_from`/`dev_vertex`/`dev_tmp` allocation+copy) and its
  entire `GPUSchedule` device-struct construction (every
  `cudaMallocManaged`+`cudaMemcpy` call) verbatim — exactly its own
  pre-launch scope.
- `gs_run(handle)` — resets `dev_sum`/`dev_cur_edge` to 0 (needed because
  this harness reuses one handle across warmup+reps, unlike the artifact's
  own single-shot CLI, which only zeroes them once — same idiom
  `glumin`/`fringe-sgc`'s adapters in this track already use), launches
  `gpu_pattern_matching` (the artifact's own unmodified `__global__`
  kernel, defined in the unmodified `gpu_clique.cu` and declared `extern`
  in `gs_shim.cu` — the two translation units linked with `-rdc=true`),
  reads back `dev_sum`, divides by `get_in_exclusion_optimize_redundancy()`
  — exactly `pattern_matching_init()`'s own post-launch scope.
- `gs_free(handle)` — releases everything.

Zero lines of `source/`'s kernel logic were modified; `gpu_pattern_matching`,
`GPU_pattern_matching_func`, `GPUVertexSet`, `intersection2`,
`do_intersection`, `dev_sum`, `dev_cur_edge` all remain exclusively in the
unmodified `gpu_clique.cu`.

## Build-system fixes and the one linkage patch

1. **Arch retarget**: `-arch=sm_80` for this machine's A100 (artifact's own
   CMakeLists targets `sm_70` — Volta — as its documented minimum; sm_80 is
   newer and compatible).
2. **`-march=native`** (host-only): `source/src/set_operation.cpp`'s SIMD
   set-intersection routine (SSE4.1/SSSE3/POPCNT intrinsics) fails to
   compile without a target ISA flag on this machine's default `g++`
   target — matches the artifact's own `CMAKE_CXX_FLAGS_RELEASE`
   (`-march=native`).
3. **MPI headers**: `source/src/graph.cpp`/`graphmpi.cpp` `#include <mpi.h>`
   unconditionally; supplied via Cray MPICH's include/lib paths (`-I`/`-L`
   + `-lmpi_gnu`), not called by this adapter's single-GPU clique path but
   needed to compile/link the translation unit that also defines
   `reduce_edges_for_clique`.
4. **One tracked-file patch** (recorded in `source.patch`):
   `source/gpu/component/utils.cuh`'s `dev_alloc_and_copy()` and
   `lower_bound()` were marked `static`. Neither was `inline`/`static` in
   the original header, which is harmless for the artifact's own
   single-`.cu`-file builds (each of its `.exe` targets compiles exactly
   one `.cu` file including this header) but becomes an ODR "multiple
   definition" **nvlink** error the moment a SECOND `.cu` file
   (`gs_shim.cu`, needed to launch `gpu_clique.cu`'s kernel from outside
   it) also includes this header and both are device-linked together with
   `-rdc=true` (`-rdc=true` is itself required for `gs_shim.cu` to
   reference `gpu_clique.cu`'s `__global__`/`__device__` symbols across
   translation units — no way around needing it here). Pure linkage
   annotation, zero arithmetic/behavior change; confirmed by the identical
   build succeeding immediately after this one-line-per-symbol edit.

No other file in `source/` needed a `-D` flag beyond what CMakeLists
already implies (`THRUST_IGNORE_CUB_VERSION_CHECK`, already present at the
top of `gpu_clique.cu` itself).

## Gate verification

### Harness smoke (`--smoke`, synthetic 4000x4000, reduced protocol)
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel graph-pattern-mining --variant gpm-kernel-small-f32 \
    --impl graphset-schedule-iep-kclique --smoke
```
Result: **2/2 runs valid**, `err = 0.00e+00` (exact) against
`kernelbench.domains.graph.reference_k_clique_count` on both synthetic
smoke graphs.

### Real-graph mandated command
```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel graph-pattern-mining --variant gpm-kernel-small-f32 \
    --impl graphset-schedule-iep-kclique --matrices ca-HepPh --warmup 1 --reps 2
```
Result: **1/1 run valid**, `err = 0.00e+00` — exact match against the
harness's own independent reference on `ca-HepPh` (12,008 vertices /
236,978 undirected entries), which cross-validates with the other three
adapters' K4 count of **150,281,372** in this track (see
`../glumin/STATUS.md`: "K4: graphfold=150281372 glumin=150281372
fringe-sgc=150281372 ALL MATCH"; GraphSet's own run, gated against the same
independent reference, agrees).

Reduced-protocol numbers only (warmup=1/5, reps=2/20, shared login-node
GPU) — explicitly non-conforming, not a timing claim, per
ARTIFACT_GUIDE.md rule 5. No timing sweep was run.

## Not done

- No sweep across the spec's `recommended_subset` — functional/gate check
  only, login-node budget.
- K5/K6/K7 not separately tested (the domain's own default/gated value is
  K4; nothing in `gs_shim.cu` restricts k beyond `>=3`, but higher k was
  not exercised here).
- `gpm-e2e-preproc` variant not separately exercised.

## Verdict

`graphset-schedule-iep-kclique: BUILT+GATED, exact match on 2 smoke graphs
+ exact match on ca-HepPh (150281372, cross-validated with
glumin/graphfold/fringe-sgc) — one tracked-file linkage patch (2 symbols
marked `static`, zero behavior change, required for -rdc=true multi-TU
compilation) and 3 build-system fixes (arch, -march=native, MPI headers),
zero kernel-code changes.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14; `-arch=sm_80` (via toolchain.sh's
  `KB_SM=80` default), no cmake used. MPI: OpenMPI 5.0.10 at `$KB_MPI_ROOT`
  (link-only; `KB_MPI_LIBNAME=mpi` on zaratan, vs. Cray MPICH's `mpi_gnu` on
  Perlmutter).
- Build: OK. Build-system changes: `build.sh` hardcoded `-lmpi_gnu`
  (Perlmutter's Cray-MPICH library name) with no override knob; changed to
  `-l"${KB_MPI_LIBNAME:-mpi_gnu}"` (Perlmutter default preserved, zaratan's
  `toolchain.sh`-derived `KB_MPI_LIBNAME=mpi` now honoured) — link-only, no
  kernel-code change. All other flags/paths (`-march=native`, `-arch=sm_80`,
  MPI_INC/MPI_LIB) already resolved through the existing `${VAR:-...}` knobs.
- Gate: gpm-kernel-small-f32 smoke, int64: PASS, 2/2 runs valid, err
  0.00e+00 <= None (structural/exact), 0.193 ms / 0.416 ms medians, on
  graph-smoke-uniform / graph-smoke-powerlaw. Real-graph mandated command
  (`--matrices ca-HepPh --warmup 1 --reps 2`): PASS, 1/1 run valid, err
  0.00e+00, 55.765 ms — same mandated commands as recorded; "Trying to
  reduce edge. the pattern is a clique." / "start erasing edge" log lines
  confirm the `reduce_edges_for_clique` path.
- Deviation from the recorded ruling: none (the MPI library-name fix is a
  build-system portability fix, not a behavioral deviation).
- Verdict here: BUILT+GATED — equals the recorded ruling.
