# stmatch (STMatch) — graph-pattern-mining

**Status: BUILT, gate FAILS — mechanical wrapping verified correct in
isolation (job-queue construction matches an independent reimplementation
exactly), but the artifact's own reported k-clique count is wrong on both a
brute-force-verified small controlled graph and the mandated `ca-HepPh`
check, under both optimization configs tried. A real off-by-one
(heap out-of-bounds read) was found in the artifact's own
`PatternPreprocessor::get_labels()`. Not patched (rule 3); reported as-is.**

- Paper: "STMatch: Accelerating Graph Pattern Matching on GPU with
  Stack-Based Loop Optimizations" (SC'22, `conf/sc/WeiJ22` in
  `../../../output/included.json`). Selected as a **core baseline** under
  the revised kernel-centrality rule (`output/kernel_centrality.json` key
  `graph-pattern-mining|conf/sc/WeiJ22`: centrality `core`, regime
  `matches` — "stack-based GPU pattern matching, 24 patterns (5-7 vertices)
  on wiki-Vote/Enron/youtube/LiveJournal"; see
  `output/baseline_selection.md`).
- Artifact: https://github.com/HPC-Research-Lab/STMatch
- Commit cloned: `f98462acfa59622b493f88130d943eafb92d598a`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9, host compiler `g++` 14.3.0
  (`/opt/cray/pe/gcc-native/14/bin/g++`), `-arch=sm_80` (A100), C++17,
  `-rdc=true -maxrregcount=64`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required at runtime.

## Pattern wrapped

Domain headline: exact **K4 (4-clique) COUNTING** (`params['clique_k']`,
default 4; see `bench/kernelbench/domains/graph.py`'s module docstring
deviation (4)). STMatch has no dedicated clique driver (unlike
GLumin/GraphFold/GraphSet); its general stack-based subgraph-matching
kernel (`_parallel_match`, `source/src/gpu_match.cu`) is driven with a
complete-graph K_k `Pattern` written in STMatch's own `v`/`e` text format
(`source/src/pattern.h`'s `PatternPreprocessor::readfile`). This adapter
exposes `clique_k` in `[3,7]` (`source/src/config.h`'s `PAT_SIZE=7`
compile-time array-sizing bound; `job_queue.h` additionally requires
`nnodes>=3` via `p.pat.degree[2]`).

## What the artifact actually is

`cu_test.cu`'s `main()` is monolithic exactly like every other artifact in
this track (GLumin's `CliqueSolver`, TileSpGEMM's `tilespgemm()`,
GraphSet's `pattern_matching_init()`): it builds the device
Graph/Pattern/JobQueue/CallStack representations AND launches
`_parallel_match` AND reads back the per-warp counts, with no public
sub-boundary. Unlike GraphSet, almost everything the kernel needs
(`GraphPreprocessor`, `PatternPreprocessor`, `JobQueuePreprocessor`,
`CallStack`) is **header-only** (`source/src/{graph,pattern,job_queue,
callstack}.h`) — only `_parallel_match` itself lives in a separately
compiled `.cu` file (`source/src/gpu_match.cu`), and the artifact's OWN
Makefile already compiles it with `-dc` (relocatable device code) and
links it against `cu_test.cu` as a second translation unit — i.e. this
integration's `-rdc=true` bridging approach is not a novel technique
invented for this project, it is the artifact's own intended build
pattern, just retargeted from two `.exe`s to one shared library.

## The shim (rule 1: wrap at the finest boundary available)

`st_shim.cu` (this directory, **NOT** part of the artifact) reproduces
`main()`'s own body **verbatim**, split into:

- `st_prepare(n, e_cnt, rowptr, colidx, pattern_path)` — builds the
  artifact's own `Graph` struct **directly from our CSR** (no file I/O;
  `GraphPreprocessor`'s binary `.meta.txt`/`.vertex.bin`/`.edge.bin` format
  is bypassed entirely). `vertex_label[i]` is set to `(1 << 1)` for every
  vertex, matching the UNLABELED convention `PatternPreprocessor::
  readfile()` itself uses (`vertex_labels.push_back(1)` when `!LABELED`) —
  deliberately NOT reproducing `GraphPreprocessor::read_bin_file()`'s own
  default-label path, which fills its `lb` array via
  `memset(lb, 1, n*sizeof(int))` (a byte-value-1 memset on an **int**
  array, i.e. every int becomes `0x01010101`, not the integer `1`) — that
  only "works" in the artifact's own code by relying on x86's
  shift-count-mod-32 behavior (`0x01010101 % 32 == 1`). This adapter
  constructs the INTENDED value directly instead of relying on that
  coincidence. Then calls the artifact's own `PatternPreprocessor`
  (reads a small auto-generated K_k pattern text file this adapter writes
  via a temp file — STMatch's own schedule/set-operation compiler,
  unmodified) and `JobQueuePreprocessor` (builds the initial 2-vertex job
  list from graph+pattern, unmodified), then reproduces `main()`'s own
  device setup (Graph/Pattern/JobQueue H2D copies, CallStack/
  slot_storage/idle_warps/global_mutex allocation+init) verbatim — exactly
  `main()`'s own pre-launch scope.
- `st_run(handle)` — resets EVERY piece of per-call mutable device state
  (CallStack array replayed from a saved pristine copy, job-queue
  cursor/mutex round-tripped through host memory to preserve the device
  job-array pointer, per-warp result buffer, idle-warp bookkeeping) to its
  prepare()-time value — needed because this harness reuses one handle
  across warmup+reps, unlike `main()`'s single-shot CLI — then launches
  `_parallel_match` (the artifact's own unmodified `__global__` kernel) and
  reads back the per-warp counts. **Verified deterministic across repeated
  calls** (identical raw count on 2 successive `st_run()` calls on the same
  handle, confirmed on the small controlled test below) — the reset logic
  itself is correct.
- `st_free(handle)` — releases everything.

Zero lines of `source/`'s kernel logic were modified; `_parallel_match` and
everything it calls (`match`, `trans_layer`, `trans_skt`) remain exclusively
in the unmodified `gpu_match.cu`.

## Build-system fixes (no kernel arithmetic touched)

1. **Arch retarget**: artifact's own Makefile targets `-arch=compute_86`
   (a consumer Ampere GPU) → `-arch=sm_80` for this machine's A100.
2. **`-maxrregcount=64`**: `_parallel_match` launches with `BLOCK_DIM=1024`
   threads/block (`config.h`, compile-time constant). sm_80's
   65536-register/SM budget hard-caps a 1024-thread block at 64
   registers/thread. Without this flag, the launch failed at runtime with
   `cudaErrorLaunchOutOfResources` ("too many resources requested for
   launch") under nvcc 12.9's optimizer — the artifact's own
   different-toolkit/different-arch build apparently fit without this
   flag; a compiler-version/arch-specific register-allocation difference,
   not a kernel logic bug. Standard build-system knob (trades a little
   occupancy/ILP for making the launch configuration valid at all — no
   arithmetic changed).
3. **Config selection** (recorded in `source.patch`): `source/src/config.h`
   hardcodes ONE `#include "config_for_ae/<variant>.h"` line; the
   artifact's own Makefile selects between the 8 `config_for_ae/*.h`
   variants per build target via a `sed -i` line-replace (`edit_config` in
   the Makefile) — there is no runtime switch. Swapped the checked-in
   default (`fig_local_global_unroll.h`, `LABELED=true` — meant for the
   paper's labeled-pattern figure reproduction) to `table_vertex_ulb.h`
   (`LABELED=false, EDGE_INDUCED=false` — STMatch's own "vertex-induced,
   unlabeled" config), the correct one for an unlabeled k-clique query,
   exactly the way the artifact's own Makefile would for a
   `table_vertex_ulb` build target.

## Real bug found in the artifact's own pattern compiler

`PatternPreprocessor::get_labels()` (`source/src/pattern.h`):
```cpp
for (int i = 0; i < pat.nnodes; i++) {
    slot_labels[i][0] = (1 << vertex_labels[i + 1]);
}
```
`vertex_labels` (a `std::vector<int>`) has exactly `pat.nnodes` elements
(filled one-per-`v`-line in `readfile()`). For `i = pat.nnodes - 1` (the
loop's last iteration), this reads `vertex_labels[pat.nnodes]` — **one
past the vector's last valid index**, undefined behavior via
`std::vector::operator[]` (no bounds check). This is a genuine,
reproducible artifact bug in the pattern compiler's own indexing, present
regardless of this integration's file-format bypass (the bug is in how
`vertex_labels` — itself correctly sized — is READ, not in how it's
populated). Whether this is the root cause of the wrong final counts below
was not conclusively proven within this integration's budget (would
require instrumenting the artifact's own kernel-side label-filtering path,
`slot_labels`'s consumption inside `_parallel_match`/`match()`), but it is
independent, direct evidence of a real correctness defect in the code path
this adapter's k-clique query exercises, found while investigating the
count mismatch below.

## Diagnosis: job-queue construction verified correct; final count is not

To isolate this adapter's own construction from the artifact's internal
search correctness, `JobQueuePreprocessor`'s output was cross-checked
against an independent Python reimplementation of its exact filter
(`job_queue.h`'s degree-threshold + label-match conditions) on a 30-vertex
random test graph (density 0.35, seed 1): **both computed exactly 159
qualifying starting edges** — this adapter's Graph/Pattern construction and
the artifact's own `JobQueuePreprocessor` are not the source of the error.
`PatternMultiplicity` (`PatternPreprocessor`'s own automorphism-count field)
was independently confirmed to be `24` (`= 4!`, correct for a fully
symmetric K4 pattern — every permutation of 4 vertices is a valid
automorphism).

Brute-force ground truth on the same 30-vertex graph (Python
`itertools.combinations`, all 4-subsets checked pairwise):
**61 distinct K4 subsets.**

STMatch's own raw per-warp sum (before any `PatternMultiplicity`
adjustment) on that same graph:
- `table_vertex_ulb.h` config (vertex-induced, unlabeled): **517** — neither
  `517` nor `517*24=12408` nor `517/24` is `61` or close to it.
- `self_defined.h` config (edge-induced, unlabeled): **1898** — also does
  not match `61` under any of the same candidate scalings.

Both configs are internally deterministic (repeated `st_run()` calls on the
same handle return the identical count), and job-queue length agrees with
the independent check above in both cases — the discrepancy is inside the
kernel's own stack-based traversal (`match()`/`trans_layer()`/`trans_skt()`
in `gpu_match.cu`) or the label-filtering path the bug above lives in, not
in this adapter's setup code.

### Real-graph cross-check (bypassing the harness's slow Python reference)

`kernelbench.domains.graph.reference_k_clique_count` (the harness's
correctness-gate reference) is too slow to complete on `ca-HepPh` within
this integration's login-node time budget (same combinatorial-
infeasibility issue `glumin`/`graphfold`'s own STATUS.md documents — those
two adapters cross-validate against each other instead of the slow
reference for real graphs). A direct ctypes call (bypassing the harness
entirely) on `ca-HepPh` (12,008 vertices / 236,978 undirected entries,
loaded via `kernelbench.domains.graph.load_workload`, identical to what the
harness itself loads) with the `table_vertex_ulb.h` config returned a raw
sum of **34,323,168** — far from the other three adapters' cross-validated
K4 count of **150,281,372** (see `../glumin/STATUS.md`), under either `x1`
or any small-integer scaling.

## Gate verification

### Harness smoke (`--smoke`, synthetic 4000x4000, reduced protocol)
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel graph-pattern-mining --variant gpm-kernel-small-f32 \
    --impl stmatch-stack-kclique --smoke
```
```
  running stmatch-stack-kclique graph-smoke-uniform   ... INVALID: exact=1.000e+00 vs tol None
  running stmatch-stack-kclique graph-smoke-powerlaw  ... INVALID: exact=1.000e+00 vs tol None
0/2 runs valid
```
(`exact=1.0` is `check_correctness`'s generic "not equal" signal for an
exact-mode gate, not a scaled error — the counts simply disagree.)

### Real-graph mandated command
```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel graph-pattern-mining --variant gpm-kernel-small-f32 \
    --impl stmatch-stack-kclique --matrices ca-HepPh --warmup 1 --reps 1
```
Does not complete within this integration's time budget — NOT because of
this adapter (a direct ctypes call on the same loaded graph returns a
result in well under a second, see above), but because the harness's own
correctness-gate reference (`reference_k_clique_count`, pure-Python
recursive bitmask counting) is combinatorially too slow on a 12,008-vertex/
236,978-edge real graph for K4, independent of which implementation is
under test — the exact same issue `glumin`/`graphfold`'s STATUS.md already
documents and works around via cross-validation instead. The direct
ctypes cross-check above (34,323,168 vs. the cross-validated 150,281,372)
already establishes the FAIL verdict without needing this command to
finish.

Per ARTIFACT_GUIDE.md rule 4 ("failures are results; never loosen a gate")
and rule 3 (no kernel-code patches): reported as-is. The off-by-one bug
found in `PatternPreprocessor::get_labels()` is left unpatched (fixing it
would change the artifact's own pattern-compiler output/semantics, not
merely a build-system/linkage knob).

## Not done

- Root-causing exactly which of (a) the `get_labels()` off-by-one, (b)
  another bug in the stack-based traversal, or (c) a subtlety in
  EDGE_INDUCED vs. VERTEX_INDUCED semantics this integration's reasoning
  ("shouldn't matter for a complete-graph pattern") got wrong, is the
  precise root cause — both tested configs are wrong by different, non-
  clean factors, so more than one issue may be at play; not resolved
  within the login-node integration budget.
- No sweep across the spec's `recommended_subset` — a single real-graph
  cross-check (`ca-HepPh`) already demonstrates the FAIL verdict.
- K5/K6/K7 not separately tested (K4's failure already establishes the
  verdict; the same code paths would be exercised).

## Verdict

`stmatch-stack-kclique: BUILT (compiles clean with 2 build-system fixes +
1 config-selection edit; runs without crashing; deterministic across
repeated calls), gate FAILS — wrong count on a brute-force-verified small
controlled graph (61 true vs. 517 / 1898 under two different optimization
configs) and on the mandated real-graph cross-check (34,323,168 vs. the
cross-validated 150,281,372 on ca-HepPh). This adapter's own Graph/
JobQueue construction is independently verified correct (exact match
against an independent Python reimplementation of the artifact's own
job-queue filter). A real, reproducible off-by-one (heap out-of-bounds
read) was found in the artifact's own `PatternPreprocessor::get_labels()`
while investigating the mismatch. Not patched, not loosened — reported as
a genuine artifact-correctness finding.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan. Two GPUs used (see below): A100 MIG 1g.5gb slice
  (sm_80) for the standard gate attempt, then a full A100-SXM4-40GB
  (sm_80, `-g a100`) to rule out a MIG-slice memory artifact. Login-node
  build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14; `-arch=sm_80` (via toolchain.sh's
  `KB_SM=80` default), no cmake used.
- Build: OK, clean, no new build-system changes needed (build.sh's existing
  `${CXX:-...}`/`${NVCC:-...}` knobs, `-maxrregcount=64`, and the recorded
  `config_for_ae/table_vertex_ulb.h` selection in `source.patch` already
  resolved correctly on this toolchain).
- Gate: gpm-kernel-small-f32, int64: FAIL, 0/2 runs valid, `INVALID:
  exact=1.000e+00 vs tol None` on both graph-smoke-uniform and
  graph-smoke-powerlaw — same mandated smoke command as recorded (`--impl
  stmatch-stack-kclique --smoke`).
  - On the default A100 MIG 1g.5gb slice, this run ALSO emitted a cascade of
    `st_shim CUDA error ...: out of memory` / `...: an illegal memory access
    was encountered` lines (not present in the originally recorded ruling)
    before settling into the same `0/2 runs valid` verdict. Per the
    MIG-slice guidance ("use `-g a100` only if a gate fails purely for lack
    of memory/SMs on the slice"), re-ran the identical command on a full
    A100 (`-g a100`, 40GB) to check whether this was a slice-memory
    artifact: on the full A100, the CUDA OOM/illegal-access messages
    disappeared entirely, and the gate reproduced the exact
    originally-recorded signature cleanly (`INVALID: exact=1.000e+00 vs tol
    None` on both graphs, 0/2 runs valid, no CUDA error lines) — confirming
    the MIG slice's ~5GB budget (vs. Perlmutter's 80GB card) was
    insufficient for this artifact's device-side CallStack/JobQueue
    workspace, and that once one CUDA call hit OOM the context went sticky
    (every subsequent CUDA call in the same process then also reports a
    generic error) — a slice-capacity artifact of this reproduction attempt,
    not a new artifact-correctness finding.
  - The real-graph mandated command (`--matrices ca-HepPh --warmup 1
    --reps 1`) was not re-run: STATUS.md already documents it as
    non-terminating on any machine (the harness's own pure-Python
    correctness-gate reference is combinatorially too slow on ca-HepPh,
    independent of this artifact or this machine), and the direct-ctypes
    cross-check that established the FAIL verdict for that graph is outside
    the runner and was not re-executed within this pass's budget.
- Deviation from the recorded ruling: none in the final verdict (gate FAILS,
  identical `INVALID: exact=1.0` signature on both smoke graphs, confirmed on
  a full A100 after ruling out a MIG-slice memory artifact on the default
  GPU).
- Verdict here: BUILT, gate FAILS — equals the recorded ruling.
