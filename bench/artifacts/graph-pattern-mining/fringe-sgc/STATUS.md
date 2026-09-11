# fringe-sgc (Fringe-SGC) — graph-pattern-mining

**Status: BUILT+GATED (k=4 only)**

- Paper: "Fringe-SGC: Counting Subgraphs with Fringe Vertices" (SC'25,
  `conf/sc/BradleyAB25` in `../../../output/included.json`, matched by
  artifact_url).
- Artifact: https://github.com/burtscher/Fringe-SGC
- Commit cloned: `5141dc04d24bfc82bb15b31e103e33952ad26fd8`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9, host compiler `g++` 14.3.0
  (`/opt/cray/pe/gcc-native/14/bin/g++`), `-arch=sm_80` (A100; artifact's own
  Makefile defaults to `sm_86`, overridden), C++17,
  `--extended-lambda --expt-relaxed-constexpr -Xcompiler=-fopenmp,-fPIC`.
  Python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required at runtime (same
  CXXABI/GLIBCXX mismatch as the other two artifacts in this track).

## Pattern wrapped: K4 only — and why

Domain headline: exact **K4 (4-clique) COUNTING** (`params['clique_k']`,
default 4). Fringe-SGC's paper premise is counting small "core" motifs
(vertex/edge/triangle/wedge, <=3 vertices) with additional **fringe**
(pendant) vertices attached — survey.md lists exactly those 4 predefined
core shapes, which does not look like an obvious match for a 4-clique
headline task at first glance (a K4 has no degree-1 vertices at all, so
"fringe decomposition" seems inapplicable).

**It turns out to work anyway.** `src/fringePreprocess.cpp`'s
`determineCore()` is a fully general greedy core/fringe peeling algorithm
(core up to `MaxCoreSize=6`, `src/fringes.h`), not hardcoded to 4 shapes —
those 4 are just the paper's worked examples. This was verified by
literally **running the artifact's own unmodified `fringePreprocess`
binary** with a K4 edge list (`build.sh`'s step 1:
`./fringePreprocess 4 0 1 0 2 0 3 1 2 1 3 2 3`):

```
motif: user defined
4 nodes and 6 edges
3 core nodes
...
anchor sets
 A B C: 1
```

I.e. **K4 == triangle core + 1 fringe vertex anchored to all 3 core
vertices**. `src/fringeCount.cu`'s `triangleCore()` has a *dedicated*
specialized branch for exactly this anchor pattern
(`f[iuvw]==1 && f[iuv]+f[iuw]+f[ivw]==0 && >=2 of f[iu],f[iv],f[iw]==0`)
that dispatches to a kernel **literally named `clique`** — this is a real,
intentionally-supported case in the artifact's own source, not a fragile
coincidence exploited by this adapter.

**K5/K6/K7 do not reduce this way.** Fringe-SGC counts occurrences of a
FIXED core+fringe shape; there is no motif file that makes its greedy
core/fringe peel land on "K5" (K5's peel does not terminate at a <=6-node
"small core + simple fringe" decomposition the way K4's does — every K5
vertex has degree 4, so the same greedy peel produces a 4-node core, which
is itself a K4, i.e. the peel never bottoms out at Fringe-SGC's `<=3`-node
"core" categories the way the paper's dedicated kernels expect, and even if
it did, K5 counting is not what this framework was built to do — its own
abstract states its target regime is core<=3 by design). This adapter
therefore raises `NotImplementedError` for any `clique_k != 4`
(`FringeSGCK4Clique.prepare()`), rather than silently attempting an
unsupported reduction. **This is itself a real, spec-relevant finding**: the
paper's own claim ("current SGC approaches can only handle very small
patterns because computational load increases exponentially with pattern
size") is borne out concretely here — Fringe-SGC's own architecture cannot
reach K5+ at all, not as a slow/DNF case, but as a shape its core/fringe
decomposition provably cannot express.

## What the artifact actually is

Two small, separate CUDA/C++ programs (~2700 lines total, no CMake/library
target — just `nvcc`/`g++` compiling each `.cu`/`.cpp` to its own binary):
`fringePreprocess` (host-only; turns a motif edge-list into a serialized
`MatchingOrder` struct file) and `fringeCount` (GPU; loads a data graph +
`MatchingOrder`, dispatches to one of `vertexCore`/`edgeCore`/`wedgeCore`/
`triangleCore`/`generalCore` based on the motif's core size/shape, and
prints a count). `fringeCount.cu`'s core-search functions
(`occurrences`, `triangleCore`, ...) have `static` (internal) linkage.

## The shim (rule 1: wrap at the finest boundary available)

`fc_shim.cu` (this directory, **NOT** part of the artifact) `#include`s
`source/src/fringeCount.cu` directly (the only way to reach its `static`
functions without patching their linkage — its own unused `main()` gets
compiled into the shared library too, harmlessly, via `#define main
fringecount_cli_main_unused`), then wraps `occurrences()`, mirroring
`fringeCount.cu`'s own `main()`: it reads the graph (CPU), copies it to the
GPU (H2D), THEN starts its own `CPUTimer` around `occurrences(...)` only.

- `fc_prepare(n, nnz, row_ptr, col_idx, mo_bytes, mo_len)` — builds an
  `ECLgraph` **directly from our own CSR arrays** (no file I/O — `ECLgraph`
  is just `{nodes, edges, nindex, nlist}`, the same layout as CSR with
  doubled/symmetric edges, which our cleaned+symmetrized workload already
  provides), H2D copy, GPU worklist allocation (`d_wl`) — exactly
  `main()`'s own pre-`CPUTimer.start()` scope — plus loading the K4
  `MatchingOrder` produced once, at build time, by `fringePreprocess`
  (graph-INDEPENDENT pattern preprocessing; see `build.sh`).
- `fc_count(handle)` — calls `occurrences(mo, d_g, d_wl, SMs, mTpSM)` ONLY
  (the artifact's own unmodified dispatcher) — its own `CPUTimer`-timed
  region. **One addition, documented, not a kernel change**: 4 `static
  __device__` globals (`total`, `wlsize`, `wlpos`, `maxdeg`) are explicitly
  reset to 0 before every call. `fringeCount.cu` itself zero-initializes
  these ONCE via `= 0` at file scope and only resets them again inside its
  OWN automorphism-count recursive sub-call (lines ~2235-2239) and inside
  `edgeCore`/`vertexCore` before their own launches — `triangleCore`/
  `generalCore` (the path K4 takes) do **not** reset them, relying on the
  process-lifetime static initializer since the artifact's CLI only ever
  calls `occurrences()` once per process. Verified directly: `init_wl_triangle`
  (`src/fringeCount.cu`) does `atomicAdd(&wlsize, 1)` with no preceding
  reset in `triangleCore`. This shim's reset uses the artifact's own
  4-symbol idiom (copied from its automorphism sub-call site), applied at
  one more call boundary — necessary because this harness reuses one handle
  across warmup+reps, unlike the artifact's single-shot CLI.
- `fc_free(handle)` — releases everything.

Zero lines of `source/` were modified.

## Build

```
./build.sh
```
Idempotent (re-running regenerates `k4.mo` and `fc_shim.so` from scratch;
both steps are deterministic). Step 1 builds+runs the artifact's own
`fringePreprocess` (`g++ -O3 -fopenmp`, matching `source/Makefile`'s own
recipe) once, producing `k4.mo`. Step 2 compiles `fc_shim.cu` (which
`#include`s `source/src/fringeCount.cu`) with `nvcc -std=c++17 -arch=sm_80
--extended-lambda --expt-relaxed-constexpr -Xcompiler=-fopenmp,-fPIC,-w -I
source/src -shared -o fc_shim.so`. No vendored Thrust/CUB dependency issue
(`fringeCount.cu` uses only the CTK's own `<cub/cub.cuh>` /
`<cub/device/device_radix_sort.cuh>`, version-matched with nvcc 12.9).

## Gate verification (login node, functional check only)

Mandated smoke command:
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel graph-pattern-mining --variant gpm-kernel-small-f32 \
    --impl fringe-sgc-triangle-core-k4clique --smoke
```
Result: **2/2 runs valid**, `err = 0.00e+00` (exact) against
`kernelbench.domains.graph.reference_k_clique_count` on both synthetic
smoke graphs (`graph-smoke-uniform`, `graph-smoke-powerlaw`); the repeated
`triangle core` / `automorphisms: 1` log lines confirm the `mo` dispatch and
per-call automorphism computation run correctly across all 25 calls
(5 warmup + 20 reps) per graph, with the explicit-reset fix in place.

### Real-graph cross-validation

Same combinatorial-infeasibility issue as graphfold's Python-reference check
(see graphfold/STATUS.md) — cross-validated instead against **graphfold**
and **glumin** (both independent implementations) on `ca-HepPh` (12,008
vertices / 236,978 directed entries):

```
K4: graphfold=150281372  glumin=150281372  fringe-sgc=150281372   ALL MATCH (3-way)
```

Reduced-protocol numbers only (warmup=5, reps=20 via `--smoke`, shared
login-node GPU) — explicitly non-conforming, not a timing claim, per
ARTIFACT_GUIDE.md rule 5. No timing sweep was run.

## Not done

- K5/K6/K7 genuinely out of scope for this artifact (see above) —
  `NotImplementedError` on request, not a silent skip.
- No sweep across the spec's `recommended_subset` — functional/gate check
  only, login-node budget.
- `gpm-e2e-preproc` variant not separately exercised.

## Verdict

`fringe-sgc-triangle-core-k4clique: BUILT+GATED, exact match on 2 smoke
graphs + K4 cross-validated 3-way on 1 real graph (ca-HepPh) against
graphfold and glumin — k=4 ONLY (K5+ not expressible in this artifact's
core/fringe decomposition, a genuine architectural limit, documented rather
than worked around); one real correctness hazard found (unreset
`static __device__` accumulator globals across repeated calls — see
fc_shim.cu), fixed in the shim, not the artifact.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14; `-arch=sm_80` (via toolchain.sh's
  `KB_SM=80` default), no cmake used.
- Build: OK. Build-system changes: none (build.sh's existing
  `${CXX:-...}`/`${CUDA_HOME:-...}` knobs already resolved to the zaratan
  toolchain via `bench/artifacts/toolchain.sh`/`bench/env.sh`).
- Gate: gpm-kernel-small-f32, int64: PASS, 2/2 runs valid, err 0.00e+00 <= None
  (structural/exact), on graph-smoke-uniform (0.265 ms) and graph-smoke-powerlaw
  (0.477 ms). Same mandated smoke command as recorded (`--impl
  fringe-sgc-triangle-core-k4clique --smoke`); the "triangle core" /
  "automorphisms: 1" log lines repeat once per call (25 calls/graph), matching
  the recorded k4-anchor dispatch.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED (k=4 only) — equals the recorded ruling.
