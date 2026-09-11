# tc-compare (GroupTC, from TC-Compare) — triangle-counting

**Status: BUILT+GATED**

- Paper: "A Comparative Study of Intersection-Based Triangle Counting
  Algorithms on GPUs" (IPDPS'24). `PAPER_KEY = conf/ipps/Li0PTZ24` (matched
  by title + artifact_url in `../../../output/included.json`).
- Artifact: https://github.com/Jangbao/TC-Compare
- Commit cloned: `149b90d99083e72f5e47ecc32bc228d0b4c67386`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9, host compiler `g++` 14.3.0
  (`/opt/cray/pe/gcc-native/14/bin/g++`), `-arch=sm_80` (A100). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.

## Track thinness

triangle-counting is one of two "thin tracks" in this integration pass
(2 GPU artifacts total in this track's survey; the other is `tot` above).
Both are integrated.

## Which kernel was wrapped, and why

TC-Compare ships **9 GPU triangle-counting kernels** behind one shared
CLI/timing harness (`approach/{Bisson,Fox,Green,GroupTC,H-INDEX,Hu,polak,
tricore,TRUST}`) — 8 re-collected prior implementations plus the paper's
own new algorithm. Per the task's instruction to wrap "its best/
representative GPU intersection kernel as one impl," **GroupTC**
(`approach/GroupTC/tc.cu`) was chosen: it is the paper's own proposed
contribution (not a re-collected baseline), the one the paper's abstract
and README foreground, and the one the survey (`benchspecs/
triangle-counting/survey.md` #2) code-verified in the most detail
(its `gpu_run()`'s exact 100-iteration/no-warmup/mean-only timing
protocol, and its kernel-only vs. H2D boundary). Wrapping a second variant
from the same repo (per the task's "wrapping 2 impls is fine if cheap")
was skipped — 8 of the other kernels are third-party re-implementations
the paper itself is comparing AGAINST, not its own contribution, and
GroupTC alone was sufficient to demonstrate the artifact integrates
correctly.

## The shim (rule 1: wrap the kernel, not the driver; rule 3: zero patches)

`tc.cu`'s only entry point is `main()`, which reads three raw binary files
(`begin.bin`/`source.bin`/`adjacent.bin`, a CSR-like oriented-DAG layout —
see `approach/comm/graph.hpp`) and drives `TC_gpu()` -> `gpu_run()`, which
hardcodes a 100-iteration mean-only timing loop
(`int iterator_count = 100;`) with H2D done once outside the loop.

`tcc_shim.cu` (this directory, **NOT** part of the artifact) `#include`s
`source/approach/GroupTC/tc.cu` **VERBATIM, zero patches** — its
`grouptc` `__global__` kernel, `graph` struct, and
`edge_count`/`vertex_count`/`grid_size`/`block_bucketnum` globals become
directly usable in the same translation unit. (`tc.cu`'s own `main()`
becomes dead, unreferenced code inside the resulting `.so` — a shared
library may contain an unused function literally named `main`; this is
not a patch, nothing in `tc.cu` was edited.) Two `extern "C"` entry points:

- `tcc_prepare(n, nnz, source, adj, offset)` — H2D copy of the oriented
  CSR arrays (mirrors `TC_gpu()`'s own H2D-once-outside-the-loop step) +
  allocation of the `results` reduction buffer.
- `tcc_count(handle)` — reproduces EXACTLY ONE iteration of `gpu_run()`'s
  own per-iteration body (`cudaMemset` -> `grouptc<<<>>>` ->
  `cudaDeviceSynchronize` -> `thrust::reduce`, tc.cu:159-171), returned as
  a value instead of accumulated into a printf'd mean. This IS the
  artifact's own kernel-only call, just invoked once per `run()` instead
  of always 100x with no return path.

## Preprocessing: what prepare() does, and the bug this integration found

TC-Compare's own pipeline for GroupTC's real datasets is
`SNAP2CSR` (orient by first-appearance vertex order, dedup, drop
self-loops) -> `CSR2RidDCSR` (**relabel** vertex IDs so `new_id` ascends
with degree, then re-orient `new_id(u) < new_id(v)`) — see
`preprocessing/cpu_preprocessing/CSR2XXX/CSR2RidDCSR.cpp`. This adapter
builds the same oriented-DAG + CSR-row-pointer layout directly from
`graph.csr` (via scipy) in `prepare()`, rather than shelling out to those
file-based preprocessing binaries (which read/write `.bin` files, not an
in-memory `Matrix`) — a legitimate "the artifact's own format conversion,
done in prepare()" per ARTIFACT_GUIDE rule 2.

**Finding (artifact-adjacent correctness trap, not a bug in the artifact
itself, but a real trap in wrapping it): vertex-ID relabeling is NOT
merely a cache-locality optimization for GroupTC — it is required for
correctness.** The first version of this adapter reused this project's
domain-level orientation (`kernelbench.domains.graph._orient_by_degree`,
which orients edges by degree-rank but leaves original vertex IDs in
place) and got a triangle count **exactly half** the true value on every
test graph (289 vs. 578 on `graph-smoke-uniform`; confirmed on a second
smoke graph too). Root cause: `grouptc`'s wedge search
(`tb_start = i + tid + 1; tb_len = beg_pos[src + 1] - tb_start;`) finds
"neighbors of `src` with larger RANK than `dst`" not via a real search but
by reading the very next array position after the current edge — which
only yields the correct candidate set when each row's stored column
*index* order coincides with rank order. Sorting a CSR row by raw vertex
ID (the domain's default) is a DIFFERENT ordering from sorting by degree
rank unless vertex IDs are first relabeled to equal rank — exactly what
`CSR2RidDCSR.cpp` does and what this adapter now replicates (`rank`
computed identically to `_orient_by_degree`'s own ascending-degree,
tie-broken-by-id ranking; then `new_id := rank`, oriented
`new_id(u) < new_id(v)`). After this fix the kernel matches the reference
exactly on every graph tested (see below). **This is not a bug in
TC-Compare's own code** (its own `CSR2RidDCSR` always does the relabel);
it is a trap for anyone wrapping `grouptc` with a differently-oriented
CSR that doesn't relabel — worth recording since it is easy to miss (the
kernel runs without error, produces a plausible-looking but silently
wrong count, and no assertion catches it).

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 -Xcompiler=-fopenmp,-fPIC -shared -o tcc_shim.so
tcc_shim.cu -lgomp`. Clean build, zero warnings. Idempotent.

## Gate verification (login node, functional check only)

Mandated smoke command:
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel triangle-counting --variant tc-gpu-kernel-exact \
    --impl tc-compare-grouptc --smoke
```
Result: **2/2 runs valid**, `err = 0.00e+00` (exact) on both synthetic
smoke graphs, AFTER the vertex-relabeling fix above (before the fix: 0/2,
`exact=1.0` on both — count off by exactly 2x).

Additionally cross-checked on a real cached SuiteSparse graph
(`ca-HepPh`, 12008 vertices / 236978 directed entries) against the same
independent CPU reference used for ToT:

```
CPU reference triangle count: 3358499
TC-Compare/GroupTC:           3358499   MATCH
```

Reduced-protocol numbers only (warmup=5, reps=20 via `--smoke`, shared
login-node GPU) — non-conforming, not a timing claim, per ARTIFACT_GUIDE.md
rule 5. No timing sweep was run.

## Not done

- Only GroupTC wrapped, not the other 8 algorithms in the repo (see
  "Which kernel was wrapped" above).
- No sweep across the spec's `recommended_subset` — functional/gate check
  only.
- Vertex IDs are relabeled to `rank` in this adapter's `prepare()`, unlike
  GroupTC's own on-disk `rid_dcsr_dataset` naming convention, but the
  RELABELING RULE ITSELF (`new_id := ascending-degree rank`) is identical
  to `CSR2RidDCSR.cpp`'s own; only the code path (direct numpy vs.
  file-based CLI tool) differs.

## Verdict

`tc-compare-grouptc: BUILT+GATED, exact match on 2 smoke graphs + 1 real
graph (ca-HepPh, 3,358,499 triangles) after fixing a vertex-relabeling
omission in this integration's own preprocessing (not an artifact bug —
see Finding above); wrapped GroupTC, the paper's own algorithm, out of the
suite's 9 kernels.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, NVIDIA A100-SXM4-40GB (sm_80, gres `a100_1g.5gb`
  request landed a full card this run), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14, no cmake involved
  (`nvcc -Xcompiler=-fopenmp,-fPIC -shared ... -lgomp`); `-arch=sm_80`.
- Build: OK. Build-system changes: none.
- Gate: `tc-gpu-kernel-exact`/`tc-compare-grouptc`: PASS, 2/2 runs valid,
  err 0.00e+00 (exact) on both smoke graphs (`graph-smoke-uniform`,
  `graph-smoke-powerlaw`); non-conforming (synthetic smoke, reduced
  protocol), as expected.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
