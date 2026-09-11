# clover (CLOVER: GPU-native, Spatio-graph-based Exact kNN) — ann-search

**Status: BUILT+GATED**

- Paper: "CLOVER: A GPU-native, Spatio-graph-based Approach to Exact kNN"
  (ICS'25). `PAPER_KEY = conf/ics/KamelYC25`.
- Artifact: https://github.com/ampslab/clover-knn
- Commit cloned: `03cfaa2ef3ac1403a3de60b43c0d92ca31761571`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  host compiler `g++` 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`),
  `-arch=sm_80` (A100), C++20. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required at run time (CXXABI/
  GLIBCXX mismatch between the system libstdc++ and the one torch links,
  same fix used by every other CUDA artifact in this repo).

## Toolchain deviation (build-system fix, not a functional change)

`source/CMakeLists.txt` hardcodes `-arch=sm_89` (Ada Lovelace). This
machine's GPU is an A100 (sm_80). `clover_shim.cu`'s own `build.sh` compiles
directly with `nvcc -arch=sm_80` instead — allowed per ARTIFACT_GUIDE.md
rule 3 (arch-flag overrides are a build-system fix). CLOVER's "hubs" kernel
uses no sm_89-specific feature (no `wgmma`/TMA — plain global/shared-memory
CUDA C++ with `__shfl_up_sync`), so this is a straightforward retarget.
`-DLINK_FAISS` (the CMake-only FAISS baseline path) is not used — this
integration wraps CLOVER's own "hubs" method only, which needs no FAISS.

## What the artifact actually is

Header-only CUDA library (`include/bitonic-based.cuh`, `include/warp-
wise.cuh`, `include/bitonic-hubs.cuh`, ...) implementing several exact-kNN
algorithms in one binary, selected by CLI flag: `bitonic` (data-parallel
batch insertion), `warpwise` (warp-ballot insertion sort), **`hubs`/
`hubs_ws`** (CLOVER's own spatio-graph-with-hubs-and-lower-bounds method —
the paper's actual contribution, wrapped here), `faiss*` (FAISS baselines,
requires `-DLINK_FAISS`, not wired), `treelogy_kdtree` (a GPU kd-tree
baseline). `src/linear-scans.cu` is a CLI driver (`main()` + `dispatch_knn`)
that loads a dataset from disk and calls one of these; this integration does
not use it (per ARTIFACT_GUIDE rule 1, wraps the kernel, not the driver).

## The shim (rule 1: wrap the kernel, not the driver)

`clover_shim.cu` (this directory, **NOT** part of the artifact) `#include`s
`include/bitonic-hubs.cuh` directly and splits `bitonic_hubs::C_and_Q`
(CLOVER's own host entry point for the "hubs" method) into two functions
along its own natural phase boundary — verbatim code, copied out of that one
function, zero lines of `source/` modified:

- `clover_prepare(n, data)` — hub selection (`Randomly_Select_Hubs`) →
  per-hub distance computation (`Calculate_Distances`) → lower-bound matrix
  construction (`Construct_D`) → bucket sort by assigned hub (`BucketSort`)
  → sorted lower-bound matrix (`fused_transform_sort_D`). This **is**
  CLOVER's own spatio-graph index (H=2048 hubs, per-point hub assignment,
  sorted H×H lower-bound matrix) — timed once as preprocessing, per the
  exact-spatial-knn-kernel spec's mandatory index-build/search split.
- `clover_query(handle, k, out)` — `bitonic_hubs::Query<ROUNDS>` only, the
  search kernel, dispatched by `ROUNDS = ceil(k/32)` exactly as `C_and_Q`'s
  own switch statement does.
- `clover_free(handle)` — releases every device buffer `prepare()` allocated.

## Structural finding (real, worth recording)

`bitonic_hubs::Query`'s first parameter, `Qps` (documented by `C_and_Q`'s
own signature as the caller-supplied query-point-index array), is **never
dereferenced anywhere in the kernel body** — confirmed by grepping
`include/bitonic-hubs.cuh` for every occurrence of the identifier `Qps`
(it appears exactly once: in the parameter list). The kernel instead derives
which point is being queried purely from
`arr_idx[blockIdx.x * queries_per_block + threadIdx.y]`, and its launch grid
is sized by `n` (`Points_num`), never by the caller's `q`. **CLOVER's own
public "hubs" method therefore always computes k-NN for every one of the n
base points against itself** ("all-points-as-queries" — survey.md's own
reading of the paper's evaluation convention, now confirmed at the
kernel-argument level: `q`/`queries` are accepted parameters that do
nothing). There is no working code path in this artifact for a caller-
chosen, held-out query subset. `clover_shim.cu`'s header comment documents
this in full; `adapter.py`'s module docstring explains the consequence.

## Adapter behavior and disclosed contamination

Because CLOVER always self-queries the whole base set, `CloverKnn.run()`
asks it for all n neighbor-sets, then slices out the rows corresponding to
this workload's actual query set. Row recovery works because every
`ann-search` exact-spatial workload this repo's domain module builds sets
`Q = X[q_idx].copy()` (an exact copy of certain base rows) — `prepare()`
recovers `q_idx` by an exact byte-equality lookup (`X[i].tobytes()`), not a
nearest-neighbor search.

**QPS caveat, disclosed per ARTIFACT_GUIDE rule 1**: since CLOVER does `n`
searches per call regardless of the workload's nominal query count, a QPS
figure computed as `n_queries / wall_time` **understates** CLOVER's true
per-search rate whenever `n_queries < n_base` (it did more work than the
numerator credits). This adapter does not correct for it. QPS numbers from
this adapter are apples-to-apples against another impl's QPS only when
`n_queries == n_base`; the smoke/bunny/uniform-* loaders currently use a
held-out query subset and are therefore a correctness-gate point for this
adapter, not (yet) a fair QPS comparison point.

## Gate results (login node, GPU, functional/gate check only — no timing loop)

```
cd bench && LD_PRELOAD=/usr/lib64/libstdc++.so.6 \
  /pscratch/sd/c/cunyang/gnn/plexus_env/bin/python -m kernelbench.runner \
  --kernel ann-search --variant exact-spatial-knn-kernel \
  --impl clover-hubs-knn --smoke
```

```
  clover-hubs-knn smoke-exact-uniform3d-1500  ...  err 1.31e-07 <= 1.00e-04   PASS
  clover-hubs-knn smoke-exact-uniform3d-3000  ...  err 7.98e-08 <= 1.00e-04   PASS
2/2 runs valid
```

Gate: exact top-k distance-multiset match against this repo's own
independent brute-force fp64 ground truth (`kernelbench.domains.annsearch._
bruteforce_ground_truth`), via `to_host()`'s shared distance-canonicalization
helper — see `kernelbench/domains/annsearch.py`'s module docstring. CLOVER's
own reported distances (squared L2, no sqrt — `spatial.cuh`'s `l2dist`) are
not used for gating; only its returned neighbor INDICES are, recomputed
independently. Dataset: synthetic uniform `U(-5,5)^3` smoke sets (N=1500/
3000, D=3, k=10) — NOT spec-conforming (smoke protocol). No timing loop was
run on the login node; the numbers above are a correctness check only,
matching `--smoke`'s reduced (warmup=5, reps=20) protocol, and are explicitly
marked non-conforming by the runner.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge
  gcc 13.4.0-20), torch 2.8.0+cu128, Python 3.12.14; no cmake (direct nvcc
  compile, as on Perlmutter). `-arch=sm_80` (unchanged from the Perlmutter
  retarget already recorded above).
- Build: OK. Build-system changes: none (`build.sh`'s existing
  `${CUDA_HOME:-...}`/`${CXX:-...}` fallbacks are already satisfied by
  `bench/artifacts/toolchain.sh`'s zaratan-side exports; nothing to touch).
- Gate: `exact-spatial-knn-kernel`, fp32: PASS both —
  `smoke-exact-uniform3d-1500` err 1.31e-07 <= 1.00e-04;
  `smoke-exact-uniform3d-3000` err 7.98e-08 <= 1.00e-04. 2/2 runs valid.
- Deviation from the recorded ruling: none — both errors match the
  Perlmutter-recorded run to the printed digits.
- Verdict here: BUILT+GATED — same as recorded ruling.
