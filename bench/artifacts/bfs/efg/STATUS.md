# efg (Elias-Fano-compressed-Graph BFS) — bfs

**Status: BUILT+GATED**

- Paper: "Traversing Large Compressed Graphs on GPUs" (IPDPS'23).
  `PAPER_KEY = conf/ipps/GeraK23` (matched by title + artifact_url in
  `../../../output/included.json`).
- Artifact: https://github.com/pgera/efg
- Commit cloned: `10fdf311966504cdcaec2f70f6f21e618ed2c8c8`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  host compiler `g++` 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`),
  `-arch=sm_80` (A100), C++17 (`--std=c++17 --expt-relaxed-constexpr`,
  matching the artifact's own `Makefile`). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`. `nvcc` already on
  `PATH`. Torch extension ABI needs
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` at run time.

## Dependency problem and how it was resolved (the interesting part)

efg's own `Makefile` requires **Facebook folly** (built+installed from
source via folly's `getdeps.py`), **boost**, **glog**, **double-
conversion**, **fmt**, and **openssl** — none of these are available on
this machine (checked: no `module spider` hits for `boost`/`glog`/`folly`/
`double-conversion`; no system pkg-config entries) and none are
installable without sudo/system packages, which ARTIFACT_GUIDE.md forbids.
Building real folly from source (`getdeps.py build`) additionally compiles
a large chain of its own third-party dependencies from scratch — far
outside a login-node "build + a handful of gate launches" budget.

Rather than skip the artifact outright, the actual header-level dependency
graph of the **one** folly feature efg uses —
`folly::compression::EliasFanoEncoder` (`folly/experimental/
EliasFanoCoding.h`), plus its `Select64`/`Instructions` helpers — was
traced by fetching headers directly from folly's GitHub (pinned to folly's
**v2023.05.22.00** tag, the exact version efg's own README validates
against) and iteratively compiling a minimal test program, auto-resolving
each `#include` error. This closure turned out to be **bounded and
genuinely header-only-compatible**: ~30 folly headers (bit manipulation —
`Bits.h`, `Select64.h`; portability macros — `Portability.h`,
`CPortability.h`; a Spooky hash used by `Range.h`) plus 6 small `.cpp`
files (`detail/RangeCommon.cpp`, `detail/RangeSse42.cpp`,
`hash/SpookyHashV2.cpp`, `lang/SafeAssert.cpp`, `lang/CString.cpp`,
`lang/Exception.cpp`) and **one hand-written file**: a generated-at-
CMake-time `folly/folly-config.h` (feature-detection macros; written by
hand with safe defaults for a modern x86_64 Linux/glibc/GCC toolchain,
compression-library/gflags/glog/libunwind flags left OFF since this narrow
closure never exercises those paths — see the file's own header comment).
**No boost, no double-conversion, no fmt, no openssl were needed anywhere
in this closure.**

The one remaining external dependency was `<glog/logging.h>` (`CHECK_EQ`,
`CHECK_GE`, `CHECK_LE`, `CHECK_LT`, `DCHECK`, `DCHECK_{EQ,NE,LE,LT,GE,GT}`,
`VLOG` — confirmed by `grep` to be the **complete** set of macros used
anywhere in the closure, all as bare calls with no `<<` message chaining).
glog itself is not header-only (needs gflags + a compiled library).
`folly_vendor/glog/logging.h` (this directory, **NOT** part of folly or
efg) is a from-scratch, ~90-line macro shim reproducing glog's *observable*
behavior exactly: `CHECK*` always aborts with a message on failure
(release or debug), `DCHECK*` only checks when `NDEBUG` is undefined
(matching glog's own release-vs-debug contract — the guarded expression is
not evaluated at all when disabled), `VLOG(n)` is a no-op sink. **This
touches zero Elias-Fano encode/decode logic** — every algorithmic line in
`folly_vendor/folly/` is the real, unmodified upstream source; the shim
only replaces a debug-assertion/logging library, never called on any
success path.

`folly_vendor/` lives as a sibling of `source/` (not inside it), matching
the pattern of `tot_shim.cu` living alongside `tot/source/` rather than
inside it — `source/` stays a pristine, unmodified clone.

## What the artifact actually is

CSR -> Elias-Fano-compressed graph representation -> GPU top-down BFS.
`source/src/csr.{h,cpp}` loads a CSR from two binary files (`vertex.CSR`,
`edge.CSR`). `source/src/ef_layout.h`/`ef_graph.h` build the Elias-Fano
compressed representation **on the CPU** using folly's real
`EliasFanoEncoder` — per vertex, this is: a forward-pointer skip table
(every `kForwardQuantum`-th cumulative gap count, `512` by default, for
fast random access into the bitstream), Elias-Fano **upper bits** (unary-
coded gaps between sorted neighbor ids, laid out as a bitmap) and **lower
bits** (fixed-width remainder bits) — the classical Elias-Fano monotone-
sequence encoding applied independently to each vertex's sorted adjacency
list. `source/src/cu_ef_graph.cuh` uploads this compressed representation
to device memory. `source/src/bfs.cuh`/`bfs_kernels.cuh` run a top-down
frontier-expansion BFS that decodes the Elias-Fano bitstream **entirely
on-device** with a hand-rolled bit-select routine (`byte_select.cuh`'s
`dSelectInByte` lookup table + `cub::BFE`) — **folly is never used on the
device side**, confirmed by grep: no `folly::` symbol appears in any
`__device__`/`__global__` function in `bfs_kernels.cuh`.

## The shim (rule 1: wrap the kernel, not the driver)

`main.cu`'s only entry point is the CLI driver (reads CLI args via
`boost::program_options`, then does load+compress+upload+traverse in one
process). Per ARTIFACT_GUIDE.md rule 1, `efg_shim.cu` (this directory,
**NOT** part of the artifact) links directly against efg's own `CSR`,
`EFLayout`, `EFGraph`/`CUEFGraph`, `BFS` classes and exposes `extern "C"`
entry points that split exactly along `main.cu`'s own phase boundaries
(lines 68-128 of `source/src/main.cu`):

- `efg_prepare(csr_dir, use_uvm, sort_frontier)` — `CSR(dir)` [load] ->
  `EFLayout<0,512>(csr)` [compute EF byte sizes] ->
  `CUEFGraph<0,512>(ef_layout, alloc_mode)` [**the compression step**:
  `EFGraph`'s constructor Elias-Fano-encodes every vertex's neighbor list
  via folly's real `EliasFanoEncoder`, then uploads the compressed
  representation to device memory] -> `BFS<0,512>(cu_ef_graph,
  sort_frontier)` [allocate device scratch buffers]. ALL of this is
  preprocessing under the bfs spec, matching efg's own README/paper
  convention of reporting CSR size / EF size / compression ratio once,
  separately (this adapter exposes those same three numbers, see below).
- `efg_run(handle, source, out_distances)` — `BFS<0,512>::traverse(source)`
  **ONLY**: the GPU top-down frontier-expansion kernel — efg's own timed
  region (`BFS::traverse_impl` wraps exactly this in a cudaEvent pair,
  matching `bfs.get_last_elapsed_time()` in `main.cu`). Returns the
  **LEVEL/distance array directly** (`std::vector<size_t>`, `UINT64_MAX`
  sentinel) — efg was **already level-array-native**, unlike blest/
  bit-graphblas, which is exactly the representation this track's
  correctness gate now uses (see "bfs correctness gate" below).

`kSkipQuantum=0`, `kForwardQuantum=512` are `main.cu`'s own compile-time
template defaults (`BFS<>`/`EFLayout<>` are C++ templates on these two
values, so they cannot be adapter-level runtime parameters without
exploding the number of instantiations built) — using the paper's own CLI
defaults is the honest, unmodified choice.

`source/src/csr.cpp` and one small hand-written file (`efg_util_shim.cpp`,
providing `get_unsigned_type_size` — a deterministic byte-width-selector
utility, copied verbatim from `source/src/util.cpp`'s own body) are
compiled into the shim; `source/src/util.cpp` itself is **not** compiled,
since its other function (`get_sha_sum`, needed only for `main.cu`'s own
CLI self-consistency-hash printout) needs `<openssl/sha.h>` and is never
called by this adapter's `prepare()`/`run()` split. Zero lines of
`source/` were modified.

## CSR-loading caveat (documented, not a bug)

`CSR`'s only constructor is file-based (`source/src/csr.cpp`) — there is no
in-memory constructor in the artifact, and none was added (that would mean
patching a source file, ARTIFACT_GUIDE.md rule 3). `adapter.py`'s
`prepare()` writes the graph's CSR to a **fresh temp directory**
(`tempfile.mkdtemp()`) as `vertex.CSR`/`edge.CSR` (uint64, exactly the
format `source/README.md` documents) and hands `efg_prepare()` that
directory path — this IS still "the artifact's own format conversion"
inside `prepare()` (ARTIFACT_GUIDE.md rule 2), just routed through a temp
directory because that's the only loading path `CSR` offers. The temp
directory is removed in `free()`.

## bfs correctness gate: LEVEL array (domain fix — shared with blest)

Same domain-level fix described in `../blest/STATUS.md`: the bfs
correctness gate in `bench/kernelbench/domains/graph.py` was switched from
comparing PARENT arrays to comparing LEVEL/DISTANCE arrays (module
docstring deviation 5), since parent-array exact-match is not a
tie-break-order-invariant target across different BFS implementations,
while the level/distance array is unique per (graph, source) regardless of
traversal order — and the spec's own `operation` text explicitly sanctions
either representation. efg required no adapter-side workaround for this at
all: its `BFS::traverse()` was *already* returning distances natively.

## Gate verification (login node, functional check only)

Mandated smoke command:
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel bfs --variant bfs-kernel-real-graphs \
    --impl efg-elias-fano-bfs --smoke
```
Result: **2/2 runs valid**, `err = 0.00e+00` (exact level-array match) on
both synthetic smoke graphs (`graph-smoke-uniform` 1500 v / 23926 directed
entries; `graph-smoke-powerlaw` 1500 v / 17968 directed entries).

Additionally cross-checked on a real cached SuiteSparse graph (`ca-HepPh`,
12008 vertices / 236978 directed entries) against
`kernelbench.domains.graph.reference_bfs`:
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel bfs --variant bfs-kernel-real-graphs \
    --impl efg-elias-fano-bfs,scipy-bfs --matrices ca-HepPh \
    --warmup 1 --reps 3
```
Result: **2/2 runs valid**, `err = 0.00e+00` for both `efg-elias-fano-bfs`
and `scipy-bfs`.

Reduced-protocol numbers only (warmup=5, reps=20 via `--smoke`, or explicit
`--warmup 1 --reps 3`; shared login-node GPU) — explicitly non-conforming,
not a timing claim, per ARTIFACT_GUIDE.md rule 5. The per-call host-side
Elias-Fano CPU encode (inside `prepare()`, untimed as kernel time) plus a
temp-directory round-trip through disk make `prepare()` itself noticeably
slower than blest's on these tiny smoke graphs (seconds, not milliseconds)
— irrelevant to the kernel-only gate result, and not something a login-node
functional check needs to optimize away.

## Not done

- No sweep across the spec's `recommended_subset` real graphs beyond the
  one cached cross-check (`ca-HepPh`) — efg's own paper-scale graphs
  (twitter, gsh-h-15, frndster, uk-07-05, kron27_s) are hosted externally
  (p.gera.io) and were not fetched — out of scope for the login-node
  budget.
- UVM mode (`use_uvm=1`, wired through `efg_prepare`'s second argument) was
  not exercised — irrelevant at smoke-graph / cached-graph scale, only
  matters for graphs that don't fit in device memory.
- `-d`/`nosort` (disabling the frontier-sorting optimisation) was not
  exercised; this adapter always passes `sort_frontier=1` (main.cu's
  default).
- `bfs-e2e-preprocessing` variant (amortized compression cost) was not
  separately exercised, though `params["csr_storage_bytes"]`,
  `params["ef_storage_bytes"]`, and `params["compression_ratio_vs_optimal_
  csr"]` are already recorded by `prepare()` and would serve it directly.

## Artifact bugs found

None. The kernel produced exact-match BFS levels on every graph tested.

## Verdict

`efg-elias-fano-bfs: BUILT+GATED, exact level-array match on 2 smoke
graphs + 1 real graph (ca-HepPh); pure top-down frontier expansion (no
push/pull hybrid — confirmed no folly/push-pull code on the device side);
the interesting finding here is methodological, not algorithmic: efg's own
Makefile-declared dependency chain (folly+boost+glog+double-conversion+
fmt+openssl) looked like a hard blocker, but the artifact's ACTUAL use of
folly reduces to one class (EliasFanoEncoder) whose real header closure is
small and self-contained once glog's few used macros are shimmed --
avoiding a from-scratch folly build entirely while using folly's real,
unmodified Elias-Fano implementation.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge
  gcc 13.4.0-20), Python 3.12.14; no cmake. `-std=c++17
  --expt-relaxed-constexpr -arch=sm_80` (unchanged).
- Build: OK. Build-system changes: none (`build.sh`'s existing
  `${CUDA_HOME:-...}`/`${CXX:-...}` fallbacks already satisfied via
  `toolchain.sh`'s zaratan-side exports; `folly_vendor/` closure needed no
  further changes).
- Gate: `bfs-kernel-real-graphs`, int64: PASS both — `graph-smoke-uniform`
  err 0.00e+00; `graph-smoke-powerlaw` err 0.00e+00. 2/2 runs valid.
- Deviation from the recorded ruling: none in the gate itself. Scope note
  (shared with `../blest/STATUS.md`): the additional `ca-HepPh` real-graph
  cross-check was not repeated here — the SuiteSparse matrix fetch stalled
  under heavy multi-session login-node process contention
  (RLIMIT_NPROC=256 shared across concurrent sessions) and was abandoned
  per the bounded-effort rule; the smoke-gate result the ruling below is
  based on is unaffected.
- Verdict here: BUILT+GATED — same as recorded ruling.
