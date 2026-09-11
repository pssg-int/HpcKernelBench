# blest (BLEST: Blazingly Efficient BFS using Tensor Cores) — bfs

**Status: BUILT+GATED**

- Paper: "BLEST: Blazingly Efficient BFS using Tensor Cores" (ICS'26).
  `PAPER_KEY = conf/ics/ElbekK26` (matched by title + artifact_url in
  `../../../output/included.json`).
- Artifact: https://github.com/delbek/blest
- Commit cloned: `7063f0dcc9f3842b08efc5d20f5a330b8ee0f795`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  host compiler `g++` 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`),
  `-arch=sm_80` (A100), C++20, `-Xcompiler=-fopenmp,-fPIC`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`. `nvcc` is already on
  `PATH` on this machine (no `module load cudatoolkit` needed). Torch
  extension ABI needs `LD_PRELOAD=/usr/lib64/libstdc++.so.6` at run time
  (CXXABI/GLIBCXX mismatch between the system libstdc++ and the one torch
  links, same fix used by every other artifact in this repo).

## Toolchain version note (not a real blocker)

The README states a **hard requirement of CUDA >= 13.0**; this machine has
12.9. The only CUDA-13-sounding feature the BFS path (`source/BFS/*.cuh`)
actually uses is an inline `mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.
and.popc` PTX instruction (binary/1-bit Tensor Core MMA) — that instruction
has been valid since sm_75 / CUDA 10.2, and empirically assembled and ran
correctly under 12.9 for sm_80 (see gate results below). The README's stated
minimum is evidently the authors' own dev/test toolchain, not a hard
technical floor for this code path.

## What the artifact actually is

C++20/CUDA library (header-heavy: `source/DataStructures/{CSC,BVSS,
BitMatrix}.cuh`, `source/BFS/{BFSKernel,BVSSBFSKernels}.cuh`) plus one CLI
driver (`source/main.cu` -> `Benchmark::main`) that downloads a named
SuiteSparse matrix via libcurl, builds BLEST's own CSC -> reorder -> BVSS
(bit-sliced, tensor-core-MMA-ready adjacency) pipeline, then dispatches to
one of 5 kernels (BFS/MBFS/Closeness/CC/WCC) by CLI flag. The BFS kernel
itself (`BVSSBFSKernels::BVSSBFS8Enhanced...`, `source/BFS/
BVSSBFSKernels.cuh`) is a single cooperative-groups kernel that loops over
BFS levels internally via `grid.sync()`, expanding the frontier each level
by decoding the BVSS bit-sliced adjacency with binary Tensor Core MMA
(`mma.sync...b1.b1...and.popc`) — this bit-matrix-multiply-as-frontier-
expansion IS the paper's core contribution.

## The shim (rule 1: wrap the kernel, not the driver)

`main.cu`'s only entry point is the CLI driver, which does
download+CSC+reorder+BVSS+kernel-dispatch in one process, and needs libcurl
(for `SuiteSparseMatrixDownloader.hpp`) purely to fetch matrices from the
SuiteSparse website. Per ARTIFACT_GUIDE.md rule 1, `blest_shim.cu` (this
directory, **NOT** part of the artifact) links directly against BLEST's own
`CSC`, `BVSS`, `BFSKernel` classes (via `#include "BFS/BFSKernel.cuh"`,
which pulls in `BVSS.cuh`/`CSC.cuh`/`Common.cuh` transitively) and exposes 3
`extern "C"` entry points that split EXACTLY along `Benchmark::run()`'s own
phase boundaries for the "BFS" kernel — **`Benchmark.cuh`/`main.cu` are
never included**, so libcurl/MPI are never needed either:

- `blest_prepare(n, nnz, row_ptr, col_idx)` — builds a `CSC` object directly
  from an in-memory CSR (bypassing BLEST's file-based/libcurl-based loader
  entirely: `CSC`'s fields are populated via its own public reference-
  returning getters, `getN()`/`getNNZ()`/`getColPtrs()`/`getRows()`, no
  `source/` code touched) -> `csc->reorder(sliceSize=8)` (Jaccard-window or
  RCM vertex reordering — BLEST's own choice, made by its own
  `CSC::reorder()`) -> `BVSS::constructFromCSCMatrix` (the bit-sliced
  format construction). ALL of this is preprocessing under the bfs spec.
- `blest_run(handle, source, out_levels)` — `BFSKernel::singleSourceRun
  (source, switching=false)` ONLY: the kernel-only call, with the source
  vertex and the returned level array translated through
  `inversePermutation` both ways exactly as `Benchmark::run()`'s "BFS"
  branch does (since BLEST reorders vertices internally for locality).

Zero lines of `source/` were modified.

## Reordering-classifier simplification (documented, not a correctness issue)

`CSC::isSocialNetwork()` (chooses Jaccard-window vs. RCM reordering, and
`FULL_PADDING` vs. not) is normally computed by a **private**
power-law-tail degree-distribution test (`CSC::socialNetworkHelper()`)
reachable only from the file-reading constructor this shim bypasses. The
shim reproduces just the cheap half of that classifier's own
OR-condition (average degree > BLEST's own `SOCIAL_THRESHOLD` constant, 18)
and sets `csc->isSocialNetwork()` directly (a public, reference-returning
setter). This choice only steers WHICH valid reordering/kernel-variant path
runs — every branch of `reorder()` and every `BVSSBFSKernels::` variant
produces a correct BFS traversal of the same underlying graph, so this
cannot silently produce wrong levels. Confirmed empirically: the two smoke
graphs below exercise BOTH branches (RCM for the denser/uniform graph, since
avg degree 8 was near/above the boundary in one case and JaccardWithWindows
for the sparser powerlaw graph — see gate output), and both passed exactly.

## Timing-boundary caveat (real, disclosed contamination — not a bug)

BLEST's public per-source API is `singleSourceRun` (re-uploads the
already-reordered adjacency to the GPU on **every** call, before its own
internal `omp_get_wtime()` pair starts) or `multiSourceRun` (uploads once,
then loops internally over a whole list of sources handed to it up front).
This harness's protocol calls `run(handle)` once per timed repetition with
no arguments, cycling through the 64-root list one root per call — that
shape only matches `singleSourceRun`. This harness's `CudaEventTimer`
therefore measures `singleSourceRun`'s per-call re-upload overhead as well
as the kernel launch, so a full timed run's per-search ms/GTEPS numbers from
this adapter will read **worse** than BLEST's own paper-reported numbers
(which time only the launch+sync span). This is real, disclosed
contamination inherent to wrapping this artifact's public API at the finest
available per-search boundary (ARTIFACT_GUIDE.md rule 1's explicit
allowance), not a defect in the traversal itself — and it does not affect
correctness at all, only absolute timing.

## Direction-optimization disclosure (spec-mandated)

No push/top-down vs. pull/bottom-up branch was found anywhere in
`source/BFS/BVSSBFSKernels.cuh` (read in full) — BLEST's BFS is a single
unified per-level kernel (frontier-driven expansion over the BVSS bitmap),
not a classic direction-optimizing hybrid. `direction_optimized = False`,
consistent with survey.md's own finding that BLEST's paper text is
ambiguous on this point ("pull-based ... bitmap-oriented" without
confirming a push phase) — this adapter's reading of the actual kernel
source resolves that ambiguity as: no hybrid switch exists in the code.

## bfs correctness gate: LEVEL array (domain fix — see below)

A **genuinely required small fix** was made to
`bench/kernelbench/domains/graph.py` (the bfs domain, not this artifact):
the correctness gate was switched from comparing PARENT arrays to comparing
LEVEL/DISTANCE arrays. `benchspecs/bfs/spec.yaml`'s own `operation` text
explicitly sanctions either representation ("output is a parent/predecessor
array (or level array)"), but a parent array is only a meaningful
exact-match target between two implementations that break same-level ties
**identically** — BFS distances are unique per (graph, source), but which
same-level neighbor becomes the recorded parent is not, and a massively
parallel / direction-optimizing GPU BFS essentially never reproduces a
single-threaded reference's tie-breaking order. The previous version only
"worked" because `ScipyBFS` (`breadth_first_order`) and the pure-Python
reference both happen to be sequential top-down traversals over the same
sorted CSR. `reference_bfs()` now returns the level array (computed
alongside the parent tree by the same independent pure-Python BFS, which
still exists and is unchanged in its traversal logic); `ScipyBFS` was
switched from `breadth_first_order` to
`dijkstra(..., unweighted=True, return_predecessors=False)` (distances =
BFS levels for an already-unit-weight graph) so its own gate exercises the
same representation. **The CPU smoke test (`scipy-bfs`) was re-run after
this change and still passes exactly** (see
`bench/kernelbench/domains/graph.py`'s module docstring, deviation 5, for
the full writeup). This is the fix that made a meaningful gate possible for
blest/efg (both of which are natively level-array-producing GPU kernels,
like virtually every real-world parallel BFS implementation) — without it,
every GPU bfs artifact in this track would fail the gate by construction,
regardless of correctness.

## Gate verification (login node, functional check only)

Mandated smoke command:
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel bfs --variant bfs-kernel-real-graphs \
    --impl blest-tensorcore-bfs --smoke
```
Result: **2/2 runs valid**, `err = 0.00e+00` (exact level-array match) on
both synthetic smoke graphs:
- `graph-smoke-uniform` (1500 v / 23926 directed entries) — RCM reordering
  path exercised.
- `graph-smoke-powerlaw` (1500 v / 17968 directed entries) — Jaccard-window
  reordering path exercised.

Additionally cross-checked on a real cached SuiteSparse graph (`ca-HepPh`,
12008 vertices / 236978 directed entries) against
`kernelbench.domains.graph.reference_bfs` (independent pure-Python level-
synchronous BFS, root #0 of the fixed-seed 64-root list from the giant
weakly-connected component):
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel bfs --variant bfs-kernel-real-graphs \
    --impl blest-tensorcore-bfs,scipy-bfs --matrices ca-HepPh \
    --warmup 1 --reps 3
```
Result: **2/2 runs valid**, `err = 0.00e+00` for BOTH `blest-tensorcore-bfs`
and `scipy-bfs` (Jaccard-window reordering path exercised on this graph
too).

Reduced-protocol numbers only (warmup=5, reps=20 via `--smoke`, or explicit
`--warmup 1 --reps 3` for the real-graph check; shared login-node GPU) —
explicitly non-conforming, not a timing claim, per ARTIFACT_GUIDE.md rule 5.
No timing sweep was run.

## Not done

- No sweep across the spec's `recommended_subset` real graphs (com-
  LiveJournal, com-Orkut, GAP-*, uk-2005, ...) beyond the one cached
  cross-check (`ca-HepPh`) — out of scope for the login-node budget.
- `MBFS`/`Closeness`/`CC`/`WCC` kernels (also shipped by this artifact) were
  not wrapped — out of scope for the bfs track (CC/WCC belong to the
  connected-components track, which already has its own adapters).
- `bfs-e2e-preprocessing` and `bfs-reordering-cost` variants were not
  separately exercised, though the same `prepare()`/`run()` split serves
  them unchanged (preprocessing time is already implicitly separated by the
  harness's own timing split).

## Artifact bugs found

None. The kernel produced exact-match BFS levels on every graph and every
internal code-path branch (RCM/Jaccard reordering, FULL_PADDING/not)
exercised during this integration.

## Verdict

`blest-tensorcore-bfs: BUILT+GATED, exact level-array match on 2 smoke
graphs (both reordering branches) + 1 real graph (ca-HepPh); pure
frontier-driven kernel (no push/pull hybrid found in source); disclosed
timing-boundary caveat (singleSourceRun's per-call adjacency re-upload is
included in this harness's measured time, unlike BLEST's own narrower
internal timer) means future timed runs will read slower than the paper's
own numbers even though the kernel itself is unmodified and correct.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge
  gcc 13.4.0-20), Python 3.12.14; no cmake. `-arch=sm_80`, C++20,
  `-Xcompiler=-fopenmp,-fPIC` (unchanged).
- Build: OK. Build-system changes: none (`build.sh`'s existing
  `${CUDA_HOME:-...}`/`${CXX:-...}` fallbacks are already satisfied via
  `toolchain.sh`'s zaratan-side exports).
- Gate: `bfs-kernel-real-graphs`, int64: PASS both —
  `graph-smoke-uniform` (RCM reordering path) err 0.00e+00;
  `graph-smoke-powerlaw` (JaccardWithWindows reordering path) err 0.00e+00.
  2/2 runs valid.
- Deviation from the recorded ruling: none in the gate itself. Scope note:
  the additional `ca-HepPh` real-graph cross-check (against `scipy-bfs`,
  recorded above) was NOT repeated here — this login node was under heavy
  multi-session process contention (RLIMIT_NPROC=256 is shared across the
  whole user session, several concurrent artifact builds/gates were running
  at once) and the SuiteSparse matrix fetch (`urllib` to sparse.tamu.edu)
  stalled; abandoned per the bounded-effort rule rather than retried
  indefinitely. This does not touch the smoke-gate result the ruling below
  is based on.
- Verdict here: BUILT+GATED — same as recorded ruling.
