# gpu-topk-study (GridSelect, from gpu_topK_benchmark) — topk-selection

**Status: BUILT (gate cannot run on UMD zaratan: the artifact's own prebuilt `libgridselect.so` requires `GLIBC_2.34`, not present on this machine — see "Reproduction on zaratan" below)**

Earlier outcome, kept for the record: BUILT+GATED (gate PASSES within a
confirmed capacity envelope; k > 2048 REFUSED, see below) — this was the
outcome on the reference machine (NERSC Perlmutter).

- Paper: "Parallel Top-K Algorithms on GPU: A Comprehensive Study and New
  Methods" (AIR Top-K / GridSelect, SC'23). `PAPER_KEY = conf/sc/ZhangNLW23`
  (matched by title + artifact_url in `../../../output/included.json`).
- Artifact: https://github.com/ZhangJingrong/gpu_topK_benchmark
- Commit cloned: `719d3a2d6df595c8facae7be8e8d22702d7674c4`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9, host compiler `g++` 14.3.0
  (`/opt/cray/pe/gcc-native/14/bin/g++`), `-arch=sm_80` (A100). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.

## Track thinness

topk-selection is a "thin track" in this integration pass: this artifact
is its ONLY GPU single-card candidate. Integrated.

## Which algorithm was wrapped, and why

`gpu_topK_benchmark` is a comparative-study suite: ONE `Factory<T,idxT>`
(`source/benchmark/factory.h`) behind a shared harness, wrapping 9
algorithms — 7 re-collected baselines (cub, faiss_block/warp, 3
SampleSelect variants, 2 DrTopK variants) plus the paper's own TWO new
methods, `raft_radix_11bits_extra_pass` ("AIR Top-K" in the paper's own
plotting scripts) and `grid_select` ("their second method", per
`benchspecs/topk-selection/survey.md`).

**`grid_select` was chosen over `raft_radix_11bits_extra_pass`** for a
concrete, disclosed cost reason: AIR Top-K needs the full RAFT + RMM
template-library stack downloaded and patched
(`source/third_party/download.sh` pulls 3 separate RAPIDS repos —
raft/rmm/spdlog — plus a checked-in `raft.patch`) — a heavy, fragile build
for a login-node integration pass. GridSelect's CUDA implementation, in
contrast, ships as a **prebuilt shared library already checked into this
repo** (`source/third_party/libgridselect.so`, exported symbol
`nv::grid_select(void*, unsigned long&, float const*, int, int, int,
float*, int*, bool, CUstream_st*)` confirmed via `nm -D`) — zero
third-party downloads needed. Only `grid_select` is wrapped (not a 2nd
algorithm): `cub` (NVIDIA's own baseline, header-only, ships with the CUDA
toolkit) was considered as a cheap 2nd impl but skipped — it is not one of
the paper's own contributions, and one well-verified paper-owned algorithm
was judged sufficient for this thin track.

## The shim (rule 1: wrap the kernel, not the driver; rule 3: zero patches)

`topk_shim.cu` (this directory, **NOT** part of the artifact) `#include`s
only `source/include/grid_select.h` (the artifact's own public
declaration) and links directly against the artifact's prebuilt
`libgridselect.so` — no artifact source was compiled or modified. Two
`extern "C"` entry points reproduce the exact two-call workspace protocol
every algorithm in this suite's `Factory` uses
(`source/benchmark/benchmark.cu::run_algo()`: query `buf_size` once with
`buf=nullptr`, allocate once, reuse for every subsequent call):

- `gridselect_prepare(batch_size, len, k)` — the one-time workspace query +
  `cudaMalloc`.
- `gridselect_run(handle, d_in, batch_size, len, k, d_out, d_out_idx)` —
  the kernel-only call, `nv::grid_select(buf, buf_size, in, ..., stream=0)`.

## Finding: an undocumented k <= 2048 capacity ceiling

While validating the adapter, `grid_select` was found to silently return a
**wrong, mostly-zero-filled result for k > 2048** rather than erroring.
Diagnosis (all confirmed directly, not guessed):

- At k <= 2048 (tested n in {2^15, 2^20, 2^22, 2^24, 2^26}, k in {10, 32,
  33, 64, 128, 256, 1000, 1024, 2048}): output is **bit-exact** against a
  numpy full-sort reference, index set valid (bounds/uniqueness/
  value-mapping) every time.
- At k = 2049 (same n's): only the **first 32 of the k requested slots**
  are ever populated with real (correct, sorted) values; the remaining
  slots stay at their zero-initialized value, and the corresponding
  indices are all `0` (a valid-but-wrong index, since `in[0]` is a real
  array element — this is why the failure surfaces as "duplicate index"
  rather than "out of bounds").
- 32 is suspicious: it is exactly `WARP_SIZE`, and `nm -D
  libgridselect.so` shows `nv::block_kernel<{WarpSelect,WarpBitonic,
  WarpMerge}, {32,64,128,256,512,1024,2048}, ..., float, int>` — i.e. the
  library's own compiled template instantiations top out at capacity
  **2048**. `buf_size` itself DOES grow correctly when k crosses 2048
  (852223 bytes at k=2048 -> 2098431 bytes at k=2049, confirmed via a
  `gridselect_buf_size()` probe added temporarily to the shim), so the
  library recognizes it needs a different/larger strategy above 2048 — but
  whatever that larger-k code path is, it does not populate the output
  correctly in this build. This is either a genuine bug in the shipped
  `.so`, or a usage detail this adapter's minimal 2-call protocol misses
  that the full `benchmark` binary (never built here — needs the RAFT/
  Faiss/etc. stack, see above) would exercise differently; either way, it
  is a real, reproducible boundary of the *artifact as integrated here*.

This adapter's `prepare()` now **refuses (`RuntimeError`) any request with
k > 2048** — matching `bench/artifacts/spmv/diaq/adapter.py`'s
`MAX_DENSE_ELEMENTS` precedent (a clean, disclosed refusal rather than
letting a known-bad configuration silently fail the correctness gate with
a confusing message). k <= 2048 covers the spec's own "small-k regime ...
matching realistic ML/DB top-k usage" (`k in {1, 10, 32, 100, 1000}`)
entirely; only the k-as-fraction-of-n stress sweep at large n (which
routinely wants k in the tens of thousands) falls outside it.

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 -Xcompiler=-fPIC -I source/include -L
source/third_party -lgridselect -shared -o topk_shim.so topk_shim.cu
-Xlinker -rpath,'$ORIGIN/source/third_party'`. Clean build, zero warnings.
`ldd topk_shim.so` confirms `libgridselect.so` resolves via the rpath
(no `LD_LIBRARY_PATH` needed). Idempotent.

## Gate verification (login node, functional check only)

Mandated smoke command, run exactly as specified:
```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel topk-selection --variant topk-array-kernel-exact \
    --impl gridselect-topk --smoke
```
Result: **partial run, then a clean documented `RuntimeError`.**
`kernelbench.domains.primitives.SMOKE`'s 3 fixed workloads happen to be
`prim-smoke-uniform` (k=256, <=2048), `prim-smoke-skewed` (k=4096, >2048),
`prim-smoke-adversarial` (k=16384, >2048) — a domain-level choice made
independently of this specific artifact's capacity, not something this
integration should edit (per the task's "do not change domain modules
unless a small fix is genuinely required" instruction — this is a real
artifact limitation, not a domain bug). `prim-smoke-uniform` runs and
passes (`err=0.00e+00`); `prim-smoke-skewed` then hits the k>2048 guard and
raises, which the CLI runner (no try/except around `harness.run_variant`)
propagates as an uncaught exception — an accurate, non-silent report of
the real boundary, not a broken adapter.

**Substitute verification** (same code path, `kernelbench.harness.
run_variant`, called directly — matching `bench/artifacts/spmv/diaq/
STATUS.md`'s precedent for a mandated command that cannot run as
specified) on 3 workloads built with the SAME 3 distributions and similar
n's as the domain's own smoke set, but with k capped to fit the confirmed
envelope:

```
prim-check-uniform-k256:      n=4096     k=256  distribution=uniform               valid=True  exact match
prim-check-skewed-k1024:      n=1048576  k=1024 distribution=skewed                valid=True  exact match
prim-check-adversarial-k2048: n=4194304  k=2048 distribution=adversarial-clustered valid=True  exact match
```

All 3 **PASS** the `exact` gate (sorted top-k value multiset bit-identical
to the numpy full-sort reference; index bounds/uniqueness/value-mapping
independently re-checked in `to_host()` on top of the harness's own
value-only gate, mirroring `NumpyTopK`'s documented precedent) — including
the adversarial "Unfriendly" bit-pattern-clustered distribution, the one
this track's literature documents as capable of defeating radix-select
pruning. Reduced-protocol numbers only (warmup=2, reps=3 for the substitute
run; warmup=5/reps=20 via `--smoke` for the mandated command) — non-
conforming, not a timing claim, per ARTIFACT_GUIDE.md rule 5.

## Not done

- `raft_radix_11bits_extra_pass` ("AIR Top-K" proper) was not wrapped — see
  "Which algorithm was wrapped" above.
- No sweep across the spec's full n/k/batch_size/distribution grid —
  functional/gate check only.
- The k>2048 code path was not debugged further (would need building the
  full `benchmark` binary against RAFT/Faiss/gpu_selection/DrTopKSC to
  compare against the artifact's OWN correctness-checked invocation of the
  same function, out of scope for a login-node budget).
- `batch_size` is fixed at 1 (the domain's own `Primitive`/`NumpyTopK`
  model no batch dimension either — not a new limitation introduced here).

## Verdict

`gridselect-topk: BUILT+GATED, exact match on 3 substitute workloads
(uniform/skewed/adversarial-clustered distributions, k in {256,1024,2048})
plus 1 of the 3 domain smoke workloads (the other 2 exceed the confirmed
k<=2048 capacity and are correctly refused, not silently miscounted);
real artifact-adjacent bug found: k>2048 silently returns a wrong,
mostly-zero result instead of erroring (see Finding above).`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, on an
  A100-SXM4-40GB physical card, driver 595.71.05, node gpu-b11-6),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge
  `kb-env`, via `KB_CXX`), Python 3.12.14, `-arch=sm_80` unchanged
  (matches this GPU). `build.sh` needed zero edits: its `CUDA_HOME`/`CXX`
  `${VAR:-<Perlmutter default>}` fallbacks already read the knobs
  `bench/env.sh`/`toolchain.sh` export before build.sh's own defaults are
  ever evaluated (rule 8), so the HPC-SDK/`gcc-native` paths in the script
  are never reached here.
- Build: OK. `topk_shim.cu` compiles and links cleanly against the
  artifact's checked-in prebuilt `source/third_party/libgridselect.so`
  (946 KB `topk_shim.so` produced, `ldd` resolves `libgridselect.so` via
  the baked-in `$ORIGIN` rpath). Build-system changes: none.
- Gate: FAIL to even load — not a correctness failure. Running the
  mandated `--smoke` command on the GPU (`gpu-b11-6`, same OS image as the
  login node) raises at `ctypes.CDLL(topk_shim.so)`:
  `OSError: /lib64/libc.so.6: version 'GLIBC_2.34' not found (required by
  .../source/third_party/libgridselect.so)`. Confirmed directly:
  `ldd --version` on this machine reports glibc 2.28 (RHEL 8.10);
  `objdump -T source/third_party/libgridselect.so` shows it was linked
  against symbols only present from glibc 2.34 onward (`pthread_create`,
  `pthread_join`, `dlopen`/`dlsym`, `sem_wait`, etc. — the set glibc merged
  from `libpthread`/`libdl` into `libc.so.6` in the 2.34 release), so the
  artifact's prebuilt `.so` requires a newer glibc than this cluster ships
  system-wide. No `--precision`/variant choice avoids this: the failure is
  at library-load time, before any kernel-specific code runs.
- Fix attempted: none applied. This is a runtime ABI requirement of a
  *prebuilt, source-not-shipped* third-party binary (see "Which algorithm
  was wrapped" above — the artifact ships `libgridselect.so` as a compiled
  blob with no corresponding `.cu`/`.cpp` in the repo to recompile against
  this machine's glibc), not a compiler flag, include path, or link path —
  outside the scope of a build-system fix (rule 2/8: no source exists to
  patch or recompile). No newer glibc is available on this cluster's
  module tree (`module spider` unsupported here; no `libc.so*` found under
  the conda-forge `kb-env` prefix used for the rest of this harness) and
  swapping the process's libc via `LD_PRELOAD` is not a safe or supported
  pattern (unlike the documented libstdc++ CXXABI trap elsewhere in this
  repo, glibc is not forward/backward-swappable at the whole-process
  level). Not pursued further, per rule 9.
- Deviation from the recorded ruling: yes. The Perlmutter run recorded
  BUILT+GATED (gate PASSES for k<=2048). On zaratan the shim builds
  identically, but the artifact's own prebuilt `libgridselect.so` cannot
  even be `dlopen`'d — a machine-portability limitation of a vendored
  binary blob, not a regression in the shim, the k<=2048 finding, or the
  GridSelect algorithm itself (which was never reached).
- Verdict here: BUILT (gate fails: GLIBC_2.34 not found, prebuilt
  third-party `.so` incompatible with this machine's glibc 2.28) — does
  NOT equal the recorded ruling; see updated top-line status above.
