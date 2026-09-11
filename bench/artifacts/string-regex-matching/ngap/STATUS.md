# ngap (ngAP) — string-regex-matching

**Status: BUILT+GATED**

- Paper: "ngAP: Non-blocking Large-scale Automata Processing on GPUs"
  (ASPLOS'24, `conf/asplos/GeZ024` in `../../../output/included.json`).
- Artifact: https://github.com/getianao/ngAP
- Commit cloned: `6fcab891ddcc1dbac79b533469f6ccbf3dd7845a`, `git clone --depth 1`
  (non-recursive; the `hscompile` submodule, Hyperscan for the CPU baseline,
  is not needed for the GPU `ngap`/`obat`/`ppopp12` targets built here).
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`,
  via `toolchain.sh`), host compiler `g++`/`gcc` 14.3.0
  (`/opt/cray/pe/gcc-native/14/bin/`, via `-DCMAKE_CXX_COMPILER`/
  `-DCMAKE_C_COMPILER`/`-DCMAKE_CUDA_HOST_COMPILER` — the system default
  g++ 7.5.0 lacks C++17 `<execution>`, needed by `src/commons/
  report_formatter.cpp`), `-arch=sm_80` / `-DCMAKE_CUDA_ARCHITECTURES=80`
  (A100; the artifact's own CMakeLists.txt hardcodes `-arch=sm_86` for its
  RTX 3090 dev box — PATCHED, see below), C++17, TBB (system
  `/usr/include/tbb`, `/usr/lib64/libtbb.so`, no module needed), OpenMP.
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required at run time.

## Patches (all build-system/portability, per ARTIFACT_GUIDE rule 3 — no
kernel logic touched)

1. `source/code/CMakeLists.txt`: `-arch=sm_86` -> `-arch=sm_80` (one line;
   A100 is sm_80, the artifact's own default targets its RTX 3090 dev box).
2. `source/code/include/commons/SymbolStream.h`: added `#include <cstdint>`
   — gcc-14's libstdc++ no longer transitively pulls this header in the way
   gcc-7 did, so `uint8_t`/`uint32_t` were undeclared.
3. `source/code/src/commons/vasim_helper.cpp`: same one-line `#include
   <cstdint>` fix (this file is literally VASim's own `parseSymbolSet`,
   the same function this project's `kernelbench/domains/automata.py`
   independently reimplements from the same upstream source — same root
   cause as fix 2).
4. `build.sh` skips the `asyncap` target (AsyncAP, SIGMETRICS'23 — one of
   this artifact's own CPU/GPU BASELINES, not the paper's own scheme): it
   uses a hand-rolled sub-Makefile hardcoded to `/usr/local/cuda/bin/nvcc`
   (no CMake toolchain plumbing, not present on this machine) — a real fix
   would mean patching that Makefile's nvcc path, judged not worth the
   time for a baseline the track doesn't need. `ngap`/`obat`/`ppopp12`
   (all proper CMake CUDA targets) build cleanly.

## What is wrapped

`ngap --algorithm=nonblockingallgroups` — the paper's full headline scheme
(non-blocking + Prefetching Always-Active-states + Prefix Memoization +
Work Privatization, per the README's own algorithm list). Like gpunfa's
`obat`, `ngap` is a monolithic CLI (ANML load, NFA/CC build, H2D, kernel
launch, stdout report, optional internal CPU cross-validation, all in one
process) with no separable library API, so this adapter wraps at the
CLI-invocation boundary (one subprocess per `run()` call), same
contamination note as `gpunfa/adapter.py`: the harness's own wall-clock for
this impl includes ANML parse + NFA/CC construction + a large internal
"precompute table" stage (observed ~15s per invocation, independent of
corpus size — this appears to be a fixed per-run setup cost of ngAP's own
precompute/memoization machinery, not proportional to work), not just the
matching kernel. The artifact's own printed `elapsed time` / `throughput`
(device-side, matching-kernel-only per its own README example) are
captured into `params['artifact_reported_elapsed_s']` /
`params['artifact_reported_throughput_mb_per_s']`.

**Required CLI flags**: `ngap` needs a specific, large flag set beyond
`-i/-a/-g` (taken verbatim from the artifact's own README "small dataset"
worked example) — omitting several of them (`--result-capacity`,
`--data-buffer-fetch-size`, `--add-aan-start`, etc, left at internal
defaults) **segfaulted** on every automaton tried, including the
artifact's own 5-state `apple.anml` and this module's own 16-state smoke
automaton. `--group-num` must equal the automaton's true connected-
component (CC) count — the default (10) only self-corrects with a printed
warning, and the "corrected" run still segfaults; passing the true CC
count (computed via `scipy.sparse.csgraph.connected_components` in
`prepare()`) from the start does not. All other flags are the artifact's
own documented example values, reused as-is (result-capacity=54619400 etc
— generous enough for these smoke-sized corpora).

**Exit-code quirk**: `ngap` returns exit code `1` even on a fully
successful run that prints `Validation PASS!` and `FINISHED!` — this
adapter treats `"FINISHED!"` appearing in stdout as the success signal
instead of the process return code (see `run()`).

## Report-convention mapping (empirically decoded, per ARTIFACT_GUIDE's own
"first rule out a convention mismatch on a tiny hand-checkable automaton")

`ngap` does **not** write the `--report-filename` file for this algorithm
family (confirmed empirically — the file was never created in any run,
unlike gpunfa's `obat`); results print to stdout as
`Result(N): \n0x.., 0x.., ...`. Each token is a packed 64-bit value
`(node << 32) | offset` (`getResult()`, `src/ngap/kernel_helper.h`), where
`node = local_state_id | (blockIdx.x << 22)` — the high bits encode which
CUDA thread block ("group", == which CC) produced the match, the low 22
bits are the STE's 0-based index WITHIN that CC in document order.

Verified on two hand-checkable cases:

1. **apple.anml** (5 states, 1 CC, pattern "apple", STE `__45__`
   reporting): `ngap`'s own README worked example reports token
   `0x400000005` for offsets `5, 47, 64, 75` (`node=4` = the 5th, and only
   CC's, state in document order) — the README's own prose confirms this
   ("ending positions ... at positions: 5, 47, 64, and 75 ... state index
   of 4"). Reproduced exactly here; identical to `_simulate_reference`'s
   output on the same automaton+stream (offsets 5/47/64/75, report_id
   2019 after this module's own id->report_id resolution).
2. **This module's own 16-state, 4-CC synthetic smoke automaton**
   (`smoke-regex-features`, patterns
   `["a[0-9]+","b.c","(foo|bar)x*","gr[ae]y"]`): token
   `0x4000020000012f` decodes to `node=0x400002` = group 1 (0x400002 >>
   22 = 1), local_index 2 (0x400002 & 0x3FFFFF = 2), offset `0x12f=303`.
   Group 1 (2nd CC discovered, 0-based) is pattern index 1 ("b.c");
   local_index 2 is that CC's 3rd state in document order (`p1_2`, that
   pattern's only reporting STE, canonical report_id 1) — matches
   `_simulate_reference`'s own `(303, 1)` entry exactly. All 18 reports on
   this automaton decoded and cross-checked by hand this way while writing
   `_group_local_index_map()`.

The adapter replicates this (group, local_index) numbering via
`scipy.sparse.csgraph.connected_components` (undirected) with same-
document-order tie-breaking (`_group_local_index_map` in `adapter.py`),
then maps `(group, local_index)` back to the STE's canonical `report_id`
via the SAME Automaton object serialized to ANML for `ngap` to read.

## Gate result

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 \
  /pscratch/sd/c/cunyang/gnn/plexus_env/bin/python -m kernelbench.runner \
  --kernel string-regex-matching --variant regex-kernel-throughput-precompiled \
  --smoke --impl ngap-nonblockingallgroups --warmup 1 --reps 1

running ngap-nonblockingallgroups smoke-literal-set     ... 4552.612 ms  0.00 Gbps  (err 0.00e+00 <= None)  [27.6s]
running ngap-nonblockingallgroups smoke-regex-features  ... 11466.510 ms  0.00 Gbps  (err 0.00e+00 <= None)  [32.9s]
2/2 runs valid
```

`--warmup 1 --reps 1` (an explicit protocol override, flagged non-
conforming by the runner as usual) was used for this gate check ONLY: each
`ngap` invocation pays a ~15s fixed "precompute table" setup cost
regardless of corpus size, so the spec's default smoke protocol
(warmup=5, reps=20 => 25 process launches per workload) would take
several minutes per workload — acceptable for a timed run on a compute
node later, not for a login-node correctness gate. Both smoke workloads
pass the exact structural gate with non-trivial match sets (report_count 6
and 8, identical to the CPU floor impl's own counts on the same
workloads) — not a vacuous zero-match pass. `Gbps` reads 0.00 for the same
harness-timer-contamination reason as gpunfa (see above); the artifact's
own reported numbers land in `params['artifact_reported_*']`.

## Not done

- `obat` (this repo's own copy of gpunfa's GPU-NFA scheme, reused verbatim
  as a baseline per the paper) and `ppopp12` (NFA-CG baseline) are built
  but not separately wrapped — lower priority than the paper's own
  `nonblockingallgroups` headline scheme given the time budget.
- `asyncap` (AsyncAP, SIGMETRICS'23 baseline) not built — see patch note 4.
- Full spec-conforming timing (1/10/20MB corpus sweep, default
  warmup/reps, compute-node allocation) not attempted — out of scope for
  this login-node pass, and impractical anyway given the ~15s/call fixed
  setup cost observed (would need investigation on a compute node, e.g.
  whether `--precompute-cutoff`/`--precompute-depth` can shrink it, before
  a real timing campaign).

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, full A100-SXM4-40GB (sm_80, driver 595.71), login-node build. (The A100 MIG 1g.5gb slice used for most gates was insufficient here — see deviation.)
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 host compiler, cmake 4.2.3, torch 2.8.0+cu128, Python 3.12.14; arch flag `-arch=sm_80` / `CMAKE_CUDA_ARCHITECTURES=80`.
- Build: OK, exit 0. Build-system changes: `build.sh` — the vendored oneTBB (from the earlier pass; only `/usr/lib64/libtbb.so.2` exists system-wide, no dev symlink or CMake package) needed its lib dir on the link search path: added `-DCMAKE_EXE_LINKER_FLAGS`/`-DCMAKE_SHARED_LINKER_FLAGS="-L$TBB_VENDOR/lib -Wl,-rpath,$TBB_VENDOR/lib"` to the cmake invocation (ngAP's CMakeLists links a plain `-ltbb`, not the `TBB::tbb` imported target, so `find_package`'s config alone did not put vendor/lib on the link line), plus an `rm -f CMakeCache.txt` before configure so the flags take effect on a re-run. Machine-neutral (vendor/ is artifact-local; harmless where a system TBB dev package exists).
- Gate: `regex-kernel-throughput-precompiled` / `ngap-nonblockingallgroups` / --smoke: smoke-literal-set PASS (err 0.00e+00), smoke-regex-features PASS (err 0.00e+00); 2/2 runs valid.
- Deviation from the recorded ruling: none in outcome — but the gate REQUIRES a full A100, not the 5 GB MIG slice: ngAP allocates fixed large device buffers at start (`buffer_capacity_per_block = 500000000`, `results_capacity = 54619400`) regardless of input size, so on the MIG slice it aborts with `ERROR: out of memory` (source/code/src/ngap/ngap_buffer.cu:559) before matching. Re-run on `-g a100` (40 GB): clean 2/2 pass.
- Verdict here: BUILT+GATED — equals the recorded ruling.
