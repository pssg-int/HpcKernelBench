# sequence-alignment artifacts

Track has 6 papers (`output/benchmark_groups.json`'s `"sequence-alignment"`
group, all 6 surveyed in `benchspecs/sequence-alignment/survey.md`). Scope
ruling (ARTIFACT_GUIDE.md, 2026-08-07): NVIDIA GPU, single-card only.

## Newest-3 accounting

Sorted by year, newest first: **BWA-FastAlign (2026, CPU-only)**,
**futhark-mem-sc22 (2022)**, **MASA-CUDAlign (2021)**, then a **three-way
2020 tie** (LOGAN / AnySeq / SegAlign).

- BWA-FastAlign is the single newest paper but is CPU-only -- out of scope
  by the scope ruling above, a cheap SKIP regardless of recency (see below).
- The next two newest GPU-eligible papers, **futhark-mem-sc22** and
  **MASA-CUDAlign**, were both attempted in full (build attempted through
  to a definitive pass/fail, with evidence).
- That leaves ONE more "newest-3" slot inside the 2020 three-way tie. Per
  this integration's brief, the tie is broken as follows: **LOGAN** is
  attempted in full (it is also this track's own designated primary target
  for `seqalign-xdrop-heuristic-kernel`, so attempting it is doubly
  motivated, not an arbitrary tie-break). **AnySeq** and **SegAlign**, the
  other two 2020 papers, each get a quick feasibility check (module
  availability / dependency footprint) rather than a full build attempt --
  both come back cheap, well-evidenced SKIPs (see their rows below), so the
  tie-break costs nothing in coverage: neither would have been buildable in
  this task's bounded effort even if chosen instead of LOGAN.

Net: **3 artifacts attempted through to a definitive result**
(futhark-mem-sc22, MASA-CUDAlign, LOGAN); **3 cheap SKIPs with evidence**
(BWA-FastAlign, AnySeq, SegAlign).

## Per-paper outcomes

| paper | key | year | outcome |
|---|---|---|---|
| BWA-FastAlign | `conf/ppopp/ZhangLMZT26` | 2026 | **SKIP** -- CPU-only (AVX2 intra-query-parallel extension on commercial CPU servers); out of scope per the NVIDIA-GPU-single-card scope ruling. Newest paper in the track, but platform-ineligible, so it does not occupy a "newest-3" attempt slot. |
| futhark-mem-sc22 | `conf/sc/MunksgaardHSO22` | 2022 | **BUILD-FAILED** -- `benchmarks/nw/futhark/nw.fut` (Needleman-Wunsch/BLOSUM62) exists and its vendored Futhark 0.22.0 compiler (`source/bin/futhark`, already checked into the repo) DOES compile a working host-side CUDA driver after a build-system `cc` include/lib-path shim (this machine's system `cc` mis-resolves `<cuda.h>` to the login-default 13.2 module even with `CPATH` pinned to 12.9 -- isolated and fixed, see STATUS.md). The generated GPU kernel then fails NVRTC compilation: `identifier "mulhi"/"mul64hi" is undefined` -- Futhark 0.22.0's bundled CUDA runtime-system emits pre-CUDA-12 unprefixed intrinsic names no longer defined by this machine's NVRTC. A Futhark-internal (not `nw.fut`-specific) version-compatibility bug; fixing it would mean patching Futhark's own generated RTS code, out of scope. See `futhark-mem-sc22/STATUS.md`. |
| MASA-CUDAlign / MultiBP | `journals/tpds/FigueiredoNSTM21` | 2021 | **BUILD-FAILED** -- `./configure && make` (autotools, nested `libs/masa-core` subproject) succeeds through configure but fails compiling `src/CUDAligner.cu`: the codebase uses the CUDA **legacy texture reference API** (`texture<T,dim,mode>`, `cudaBindTexture`, `tex1Dfetch`) for sequence/score-row caching, and this machine's CUDA 12.9 toolkit has fully removed the header (`cuda_texture_types.h`) that declares it -- confirmed for both the 4.0.2.1028 and 3.9.1.1024 releases shipped in the repo. Fixing this means rewriting the kernel's device-memory-access mechanism to the texture OBJECT API -- a kernel-code change, out of scope per ARTIFACT_GUIDE.md rule 3. This was the primary target for `seqalign-exact-pairwise-kernel`; see `masa-cudalign/STATUS.md`. |
| LOGAN | `conf/ipps/ZeniGEDSHBOY20` | 2020 | **BUILT+GATED** -- builds cleanly on the first attempt (`make demo CUDAFLAGS="...-arch=sm_80"`); a new, additive driver (`source/src/kernelbench_driver.cu`, no upstream file modified) wraps LOGAN's own `extendSeedL()` library API (the shipped `demo` CLI discards its own scores without printing them, so it cannot be wrapped directly). The correctness gate **fails deterministically** -- and this is the correct, expected, well-evidenced outcome, not an adapter bug: reading (and then empirically confirming) LOGAN's own kernel source shows (1) its GPU kernel hardcodes match=+1/mismatch=-1/gap=-1 via preprocessor macros that are NEVER connected to the runtime `ScoringSchemeL` API parameter it otherwise accepts, and (2) the `direction` argument to `extendSeedL()` is dead code -- it always launches BOTH left- and right-extension kernels, i.e. LOGAN's real algorithm is an unconditionally BIDIRECTIONAL seed-anchored extension, not the single-direction-from-origin definition this benchmark's `seqalign-xdrop-heuristic-kernel` uses. The reference was NOT adjusted to LOGAN's behavior (per this task's own rule). This is this track's primary target for `seqalign-xdrop-heuristic-kernel`; see `logan/STATUS.md` for the full evidence, including an empirical confirmation of finding (1) (`match=2` requested, `match=1` observed in the returned score). |
| AnySeq | `conf/ipps/MullerS0MLKH20` | 2020 | **SKIP** (cheap, feasibility-only) -- AnySeq is built via AnyDSL's `impala` partial-evaluation DSL compiler, not a conventional CUDA/C++ build. `module spider anydsl` / `module spider impala` both return "Unable to find" on this machine, and neither `impala` nor `anydsl` is on `PATH`. AnyDSL is not distributed as a simple prebuilt tarball (unlike Futhark) -- building it means building AnyDSL's own LLVM-based Thorin-IR compiler stack from source, a materially larger undertaking than this task's bounded effort. One of the 2020 three-way tie; see "Newest-3 accounting" above for why LOGAN was attempted instead. |
| SegAlign | `conf/sc/GoenkaTPH20` | 2020 | **SKIP** (cheap, feasibility-only) -- an end-to-end whole-genome-alignment pipeline (GPU seed+filter, `src/seed_filter.cu`, 946 lines) that spawns the LASTZ binary as a CPU subprocess per HSP segment for the actual gapped extension. Dependencies (`module spider tbb` / `module spider boost` both "Unable to find", plus LASTZ and kentUtils' `faToTwoBit`/`twoBitToFa`, none present on this machine) are heavy and CUDA-10.2-targeted. Even if built, `seed_filter.cu`'s GPU kernel computes seed-hit filtering (a filter-stage throughput number, "million seed-hits/sec" per the paper itself -- explicitly NOT a GCUPS/DP-recurrence number, per `benchspecs/sequence-alignment/spec.yaml`'s own metric note), which does not correspond to either variant this integration's domain module implements (`seqalign-exact-pairwise-kernel`, `seqalign-xdrop-heuristic-kernel`); its natural home is `seqalign-e2e-whole-genome-alignment`, which `kernelbench/domains/alignment.py` marks PLANNED-in-module, not implemented. One of the 2020 three-way tie; see "Newest-3 accounting" above. |

## Layout

```
bench/artifacts/sequence-alignment/
  masa-cudalign/   source/ (git, unmodified), STATUS.md (BUILD-FAILED, no adapter)
  logan/           source/ (git + one additive file), build.sh, adapter.py, STATUS.md (BUILT+GATED)
  futhark-mem-sc22/ source/ (git, unmodified), cc_shim/, build.sh, STATUS.md (BUILD-FAILED)
  README.md        this file
```

BWA-FastAlign, AnySeq and SegAlign have no directory (per-ARTIFACT_GUIDE.md
"cheap SKIP, no build attempt" convention for platform-ineligible or
infeasible-toolchain candidates) -- their evidence lives entirely in this
README's table above.

## Gate commands (reproduce)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
  --kernel sequence-alignment --variant seqalign-xdrop-heuristic-kernel \
  --smoke --impl logan-xdrop
# -> 0/2 runs valid (expected; see logan/STATUS.md)
```

MASA-CUDAlign and futhark-mem-sc22 have no `--impl` to gate (build never
reached a runnable artifact); `*/build.sh` reproduces each failure
deterministically (`masa-cudalign` has no `build.sh` -- `./configure &&
make` in `source/masa-cudalign-4.0.2.1028/` per its own README reproduces
the failure directly; `futhark-mem-sc22/build.sh` reproduces both the fixed
host-compile step and the unfixed device-compile failure, exiting 1).
