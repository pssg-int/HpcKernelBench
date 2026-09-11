# pathweaver (PathWeaver: multi-GPU graph-based ANNS) — ann-search

**Status: BUILT+GATED** (search kernel only, single GPU, `ann-highdim-recall-qps-pareto` variant)

- Paper: "PathWeaver: A High-Throughput Multi-GPU System for Graph-Based
  Approximate Nearest Neighbor Search" (USENIX ATC'25).
  `PAPER_KEY = conf/usenix/KimPNHKLL25`.
- Artifact: https://github.com/AIS-SNU/PathWeaver
- Commit cloned: `c0804a5f46fd603794f4ce6205956a35ca46890e` (2025-07-23),
  `git clone --depth 1`.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  host compiler `g++` 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`),
  `-arch=sm_80` (A100). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python` (torch 2.8.0+cu128).
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required at run time (CXXABI/
  GLIBCXX mismatch between the system libstdc++ and the one torch links,
  same fix used by every other CUDA artifact in this repo).

## Toolchain pin (host environment change mid-integration, not a PathWeaver bug)

This login node's default module set changed to `cudatoolkit/13.2` (HPC SDK
26.5) partway through this integration pass, while plexus_env's torch is
built against cu128. `build.sh` sources the repo-shared
`bench/artifacts/toolchain.sh`, which unconditionally pins
`CUDA_HOME`/`NVHPC_CUDA_HOME`/`PATH` to the 12.9 HPC-SDK prefix and
repoints `CPATH` at 12.9's own `cuda/include` **and** `math_libs/include`
(not merely unset — torch's own ATen/cuda headers need `cusparse.h`, which
on this SDK layout lives only under `math_libs`). Two real build failures
were hit and fixed before the shared `toolchain.sh` existed, both now
folded into it:

1. A naive `export CUDA_HOME="${CUDA_HOME:-/opt/.../12.9}"` does **not**
   override an already-nonempty inherited `CUDA_HOME` — the first build
   attempt silently compiled with nvcc 13.2 against cu128 torch (visible in
   the ninja command line). Fixed by an unconditional override.
2. `pathweaver.cu` `#include`s `<curand_kernel.h>`; on this HPC SDK layout
   cuRAND's headers live under `math_libs/<ver>/include`, **not**
   `cuda/<ver>/include` (which only has the compiler-core CUDA headers) —
   first build attempt failed with `curand_kernel.h: No such file or
   directory`. Fixed by `toolchain.sh`'s `CPATH`, which includes both
   directories (a build-system-only include-path fix, ARTIFACT_GUIDE
   rule 3 — zero lines of `source/` touched).

`kernelbench.runner` performs the same pin in-process
(`kernelbench/env.py::pin_cuda_toolchain`, printed as `[env] toolchain:
...` in every run below) so gate runs are covered automatically; `build.sh`
(a plain shell script, not run through the runner) sources `toolchain.sh`
directly for the same effect.

## What was built, and what was deliberately NOT built

`source/pathweaver/setup.py` declares TWO `CUDAExtension` targets:
`cpu_generate_sign_bit` (an OpenMP host-only helper for PathWeaver's
sign-bit pruning feature) and `pathweaver` (the actual GPU search kernel,
`csrc/pathweaver.cu`, 2202 lines). Per the task's explicit "build ONLY the
pathweaver extension" and ARTIFACT_GUIDE rule 1 (wrap the kernel, not the
whole artifact), `build.sh` does not invoke `setup.py` at all — it drives
`torch.utils.cpp_extension.load()` directly against
`source/pathweaver/csrc/pathweaver.cu` (zero lines of `source/` modified),
with the same `-O3 -Xptxas=-v -arch=sm_80` flags the artifact's own
`setup.py` already used (sm_80 was already PathWeaver's own choice,
matching this machine's A100 — no arch retarget needed, unlike CLOVER's
sm_89→sm_80). The compiled `.so` is vendored at
`artifacts/ann-search/pathweaver/build/pathweaver.so` — **not** under
`source/`.

`cpu_generate_sign_bit` is never built: it exists only to precompute the
per-edge sign bits PathWeaver's `SIGN_BIT_PRUNE` search-quality feature
consumes, which this integration's first, simplest-valid-configuration pass
does not exercise (see below).

Also **not built**: `source/pathweaver/csrc/cagra_graph/` — the RAPIDS
RAFT/cuVS CAGRA graph-CONSTRUCTION code (fetched at CMake-configure time via
`rapids-cmake` CPM, an hours-long, network-heavy build of an entire
third-party RAPIDS library). This is explicitly out of scope per the task
brief; see "The graph substitution" below for what replaces it.

## Import-order and build-process findings (real, worth recording)

1. `torch.utils.cpp_extension.load()` imports the freshly-linked module
   **inside its own build process** (right after linking, to hand back a
   module object) — without `LD_PRELOAD` set for that step too (not just
   the later verification step), this raised the project's usual
   `CXXABI_1.3.15` `ImportError`. Fixed by wrapping the entire `load()`
   invocation in `LD_PRELOAD=/usr/lib64/libstdc++.so.6` in `build.sh`, not
   only the standalone verification command.
2. The compiled `pathweaver.so` links against `-lc10 -lc10_cuda -ltorch*`
   by **SONAME only** — no baked-in rpath to torch's `lib/` directory
   (unlike a `setup.py`/`CUDAExtension`-built wheel, which typically gets
   one). Consequence: `import pathweaver` **before** `import torch` in a
   fresh process fails with `ImportError: libc10.so: cannot open shared
   object file` — confirmed experimentally in `build.sh`'s first
   verification attempt. Fixed by importing `torch` first, both in
   `build.sh`'s verification step and in `adapter.py`'s `_load_ext()` (the
   harness always imports torch before any CUDA adapter in normal use, so
   this is naturally satisfied there; `_load_ext()` still imports `torch`
   explicitly for standalone correctness).

Neither finding required touching `pathweaver.cu` itself — both are
build/import-environment issues.

## Hard kernel constraints discovered (compile-time / device-assert, not tunable)

Full derivation and file:line citations are in `adapter.py`'s module
docstring; summary:

1. **`VECTOR_DIM` is a C++ template parameter.** `search()`'s host-side
   dispatch (`pathweaver.cu:1732-2158`, a `switch(VECTOR_DIM)`) only
   instantiates `search_kernel<D,*>` for D in `{96, 100, 128, 256, 768,
   960}` — the dimensions of the ann-benchmarks datasets PathWeaver's own
   paper evaluates. Any other value falls to `default:`, which calls
   `assert(false)` **on the host**, aborting the whole Python process (not
   a catchable Python exception — confirmed by reading the code, not by
   deliberately crashing a process to check). This module's smoke workloads
   use D=32/64 (`kernelbench/domains/annsearch.py`'s `_make_recall_smoke`),
   neither of which is wired.
2. **`INTERNAL_TOPK` is also a template parameter**, wired for `{64, 128}`
   only (same switch).
3. **`CANDIDATE_BUFFER_SIZE` (= `SEARCH_WIDTH * GRAPH_DEGREE` at runtime)
   must equal exactly 64.** Three device functions
   (`candidate_by_bitonic_sort`, `candidate_by_bitonic_sort_inverse`,
   `topk_by_bitonic_sort` — `pathweaver.cu:255-258, 305-308, 358-361`)
   each contain `if (CANDIDATE_BUFFER_SIZE != 64) { printf(...);
   assert(false); }`, a **device-side** assert hit on every iteration of
   the search loop. This was found experimentally: a first attempt with
   `GRAPH_DEGREE=32` produced exactly this device `printf`
   (`"CANDIDATE_BUFFER_SIZE must be 64"`) on every launch. With
   `SEARCH_WIDTH=1` (the artifact's own driver's fixed value,
   `single_pathweaver_one.py:71`), `GRAPH_DEGREE` **must** be exactly 64 —
   not a tunable config despite nominally being a runtime `configs[]`
   entry. This conveniently matches PathWeaver's own paper's CAGRA
   `build_graph_degree=64` for `sift-128-euclidean` (survey.md's reading of
   `single_pathweaver_all.sh`).
4. **`TEAM_SIZE` must be exactly 8** (`search_kernel`'s own first check,
   `pathweaver.cu:1344-1348`).

None of these are patches to the kernel — they are discovered, documented
constraints this adapter's `prepare()` either satisfies exactly (3, 4,
fixed constants) or works around via zero-padding (1) or a fixed choice of
the smaller wired value (2). `_padded_dim()` in `adapter.py` raises a named
`NotImplementedError` (never silently) for any workload dimension exceeding
the largest wired VECTOR_DIM (960).

## The graph substitution (fairness-relevant, read this before citing recall numbers)

PathWeaver's own pipeline builds its search graph via RAFT/cuVS's CAGRA
graph-construction algorithm — an approximate, quality-optimized navigable
graph, not an exact k-NN graph. Building CAGRA's constructor is out of
scope here (hours-long RAPIDS build, network-heavy `rapids-cmake` CPM
fetch). Instead, `adapter.py`'s `prepare()` builds its own **independent
exact k-NN graph** over the workload's base set: for every point, its 64
(= the hard-required `GRAPH_DEGREE`) nearest OTHER base points, via a
chunked GPU Gram-matrix L2 computation (`_build_knn_graph`) — a genuinely
separate computation from the workload's cached query-vs-base ground truth
(`gt_idx`/`gt_dist`), touching only the base set `X`.

**Consequence**: what this integration measures is PathWeaver's SEARCH
KERNEL's recall/correctness behavior on a graph of matching degree, **not**
PathWeaver's (or CAGRA's) own graph-quality contribution. An exact k-NN
graph is not a CAGRA-optimized navigable small-world graph; recall achieved
here says nothing about how PathWeaver performs on its own paper's graphs,
and no QPS/timing comparison against another impl using its own index would
be an apples-to-apples index-quality comparison (index-BUILD cost is
reported separately from search per DOMAIN_GUIDE regardless, but the
build ALGORITHM itself differs from the paper's). This is disclosed
per ARTIFACT_GUIDE rule 1's "document the contamination" requirement.

Zero-dimension padding (item 1 above) is, separately, exactly
distance-preserving for L2 and does not affect recall at all — see
`adapter.py`'s module docstring for the derivation from the distance
kernel's own arithmetic (`compute_similarity_vector_load_full`,
`pathweaver.cu:510-567`).

## Search configuration used (simplest single-pass, no ghost stage, no sign-bit prune)

Per the task's own guidance, this first pass does **not** reproduce
PathWeaver's two-phase "ghost search" (a fast pre-search over a subsampled
graph, used only to pick a good entry point for the real search — itself
just two more calls to the same `search()` kernel already wrapped here) or
its `SIGN_BIT_PRUNE` direction-guided pruning (needs the un-built
`cpu_generate_sign_bit` extension's output). Instead, `run()` issues **one**
`search()` call per query batch, configured exactly like the driver's own
"ghost pass" (`single_pathweaver_one.py:140-166`):

- `PRUNE_CONFIG = NO_PRUNE`, `THRESHOLD_CONFIG = NO_THRESHOLD`
- `sign_bit`: empty `(0,0)` uint32 tensor
- `SEED_CONFIG = USE_SEED` with a **trivial** seed: all-zero
  `top10`/`initial_starting_point` + an **identity** `seed_map` — per
  `compute_distance_to_maped_top10_nodes` (`pathweaver.cu:766-850`), this
  means every query's greedy graph walk starts from node 0's neighbor list
  (`top1 = top10_ptr[query_id] == 0`, `parent_id = seed_map_ptr[0] == 0`) —
  a fixed, deterministic entry point (the same category of simplification
  CAGRA/HNSW-style search commonly uses).
- `HASH_TABLE_CONFIG = USE_HASH_TABLE` — visited-node dedup, a pure
  efficiency optimization with no effect on which candidates are found,
  matching the driver's own default.
- `GRAPH_DEGREE = 64`, `INTERNAL_TOPK = 64`, `TEAM_SIZE = 8`,
  `SEARCH_WIDTH = 1`, `BLOCK_SIZE = 32`, `BITLEN = 10`,
  `SMALL_HASH_RESET_INTERVAL = 16`, `MAX_ITER = 48` (tuned empirically
  against ONLY this integration's smoke workloads to clear the 0.9 recall
  floor with margin — NOT tuned or validated against any larger/real
  dataset; see "Not done" below).

`configs[]` slot layout (matching `pathweaver.cu:1680-1712` and the
driver's own array literal exactly) is documented field-by-field in
`adapter.py`'s module docstring. Index 22 (`SEED_TOPK_SIZE`) is
intentionally omitted (defaults to 1 via a caught out-of-range exception in
`search()` itself, `pathweaver.cu:1707-1712`), matching this adapter's
`top10_ptr[query_id * 1]` indexing.

## Gate results (login node, GPU, functional/gate check only — no timing sweep)

```
cd bench && LD_PRELOAD=/usr/lib64/libstdc++.so.6 \
  /pscratch/sd/c/cunyang/gnn/plexus_env/bin/python -m kernelbench.runner \
  --kernel ann-search --variant ann-highdim-recall-qps-pareto \
  --smoke --impl pathweaver-search
```

```
running pathweaver-search smoke-recall-gaussian-32d  ... 1.022-1.027 ms  ~62.3-62.6k QPS  (err 9.38e-03 <= 0.1)   PASS
running pathweaver-search smoke-recall-gaussian-64d  ... 1.044 ms        ~61.3k QPS       (err 2.34e-02 <= 0.1)   PASS
2/2 runs valid
```

Run twice (repeatability check, not a timing sweep): identical correctness
results both times (`err` exact to the printed digits). Decoded to
recall@10: **0.9906** (D=32, padded to 96) and **0.9766** (D=64, padded to
96) — both comfortably above the recall variant's spec-carried floor of
0.9 (encoded via the domain module's `max_abs_err(1-recall, 0) <=
0.1` mechanism, per `kernelbench/domains/annsearch.py`'s module docstring).
Gate: recall@k against this repo's own independent brute-force fp64 ground
truth (`kernelbench.domains.annsearch._bruteforce_ground_truth`), via the
shared `_recall_at_k` helper — never sharing code with the reference
builder. Result files:
`results/ann-search_ann-highdim-recall-qps-pareto_1788573437.json` and
`..._1788573450.json`.

As a guard-path sanity check (not a real gate run), `--variant
exact-spatial-knn-kernel --impl pathweaver-search --smoke` was also run
once: `prepare()` raised the named `NotImplementedError` documented above
("only wired for 'ann-highdim-recall-qps-pareto'...") on the very first
matrix, exactly as designed — the runner does not catch this (it is not
meant to gracefully skip a deliberately-wrong `--variant`/`--impl`
pairing), so it surfaces as an uncaught exception with a full traceback,
not a silent skip or a device-side crash. This is expected, correct
behavior for an intentionally out-of-scope variant, not a bug.

QPS numbers above are load-bearing for nothing (no timing sweep was run,
per ARTIFACT_GUIDE rule 5 — shared login-node GPU); they are printed by the
runner's smoke-mode output as a side effect of the correctness check and
are explicitly marked non-conforming (`conforming: false`, "synthetic
smoke matrices; reduced protocol; ... GPU clocks not locked" etc. — see the
result JSON's own `nonconformance_reasons`).

## Not done (explicit scope boundary)

- **No timing/performance sweep** of any kind — login node, shared GPU,
  functional gate only, per ARTIFACT_GUIDE rule 5.
- **No real dataset run** (SIFT1M or otherwise) — would need
  `kernelbench.domains.annsearch.load_workload("sift1m")`'s ~168MB texmex
  download, which this integration pass did not attempt (bounded-effort
  scope; the domain module's own loader already exists and is wired,
  should a later pass want to run it). At D=128 (SIFT1M's native
  dimension), no padding would even be needed — `128` is directly wired.
  `MAX_ITER=48` and the exact-kNN graph substitution are untuned/unvalidated
  at that scale; a real run needs its own parameter sweep.
- **No multi-GPU pipeline** — PathWeaver's actual headline contribution
  (pipelining-based path extension across GPUs via NVLink) is entirely out
  of this single-GPU integration's scope; see the module docstring's SCOPE
  section.
- **No ghost-stage seeding, no sign-bit pruning** — both are additional
  `search()` calls / a precomputed side-input to the SAME kernel already
  wrapped here, deliberately deferred per the task's "start with the
  simplest valid configuration" guidance; not attempted in this pass at
  all (not even as a follow-up experiment), since the simplest
  configuration already clears the recall gate with margin.
- **CAGRA/RAFT graph construction was not reproduced** — see "The graph
  substitution" above; this is the single most important caveat for
  interpreting any number from this adapter.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge
  gcc 13.4.0-20), torch 2.8.0+cu128, Python 3.12.14; no cmake.
  `-O3 -Xptxas=-v -arch=sm_80` (unchanged).
- Build: OK. Build-system changes: `build.sh`'s own toolchain sanity check
  (line 57) hardcoded `nvcc --version | grep -q 'release 12\.9'` — a
  Perlmutter-specific assumption (guarding against that login node's
  cudatoolkit/13.2 module shadowing the pinned 12.9 toolkit). zaratan's
  matching CUDA major is 12.8, not 12.9, so this literal check fails here
  even though the toolchain pin itself (via `toolchain.sh`/`KB_CUDA_HOME`)
  is correct. Relaxed to `grep -qE 'release 12\.'` (accept any CUDA 12.x),
  which preserves the check's actual purpose — guard against a
  mismatched-major nvcc — without hardcoding a specific minor version.
- Gate: `ann-highdim-recall-qps-pareto`, fp32: PASS both —
  `smoke-recall-gaussian-32d` err 9.38e-03 <= 0.1;
  `smoke-recall-gaussian-64d` err 2.34e-02 <= 0.1. 2/2 runs valid.
- Deviation from the recorded ruling: none — both errors match the
  Perlmutter-recorded run to the printed digits.
- Verdict here: BUILT+GATED (search kernel only, single GPU,
  `ann-highdim-recall-qps-pareto` variant) — same as recorded ruling.
