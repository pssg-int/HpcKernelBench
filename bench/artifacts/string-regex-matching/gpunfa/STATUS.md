# gpunfa (gpunfa-artifact) — string-regex-matching

**Status: BUILT+GATED**

- Paper: "Why GPUs are Slow at Executing NFAs and How to Make them Faster"
  (ASPLOS'20, `conf/asplos/0002PJ20` in `../../../output/included.json`).
- Artifact: https://github.com/bigwater/gpunfa-artifact
- Commit cloned: `a2937d035bba88e7d1de8bbbf9c9918dbd86c0c2`, `git clone --depth 1`
  (repo also ships two large `.zip` data bundles, ~925MB total, not needed
  for this gate check -- see `source/small_dataset/` instead).
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`,
  via `bench/artifacts/toolchain.sh`), host compiler `g++` 7.5.0 (system
  default picked up by CMake's `find_package(CUDA)`), `-arch=sm_80` /
  `-DCMAKE_CUDA_ARCHITECTURES=80` (A100; the artifact's own CMakeLists.txt
  sets no arch, so this was added on the `cmake` command line), C++11
  (`std=c++11 --expt-extended-lambda -O3`, the artifact's own flags,
  unmodified). All dependencies (moderngpu, clara, pugixml) are vendored
  under `source/gpunfa_code/include/` -- no external CUB/etc. needed.
  Built via `cmake` 3.28.3 (system). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required at run time (torch import
  inside `available()`/the harness; the `obat` binary itself does not need
  it, only the Python process running the harness does).
- Build: `artifacts/string-regex-matching/gpunfa/build.sh` — plain CMake
  configure+build, no source patches needed, clean first-try build
  producing `build/bin/{obat,infant,ppopp12}`.

## What is wrapped

`obat -g obat2` — the paper's own "NewTran" scheme
(`one_byte_at_a_time::OBAT_baseline_2`, `src/obat/one_byte_at_a_time.cu`),
one of the artifact's three headline algorithms (obat2 / obat_MC /
hotstart*). `obat` is a **monolithic CLI**: ANML load, NFA construction,
H2D copy, kernel launch, D2H, and report-file write all happen inside one
process invocation with no exposed library API — there is no separable
prepare()/run() boundary to call into directly. Per ARTIFACT_GUIDE.md rule 1
("if the artifact only ships an end-to-end binary... wrap at the finest
boundary available and document the contamination"), this adapter wraps at
the CLI-invocation boundary: `prepare()` writes the ANML/input files to a
temp dir (no compute), `run()` launches `obat` as a subprocess (one
invocation per harness rep) and parses its report file. This means:

- **The harness's own wall-clock timer around run() is contaminated** —
  it includes ANML parse, NFA construction and H2D transfer, not just the
  matching kernel, so `stats_ms`/the harness's `Gbps` field for this impl
  UNDER THIS ADAPTER are NOT directly comparable to another impl's clean
  matching-only time. **The artifact's OWN internal timer is clean**: it
  wraps only the CUDA kernel launch with `cudaEvent_t` (device-side events,
  exactly the spec's own timer convention for GPU impls) and prints
  `Elapsed time : ... ms` / `throughput = ... ` (bytes/s); this adapter
  captures both into `params['artifact_reported_elapsed_ms']` /
  `params['artifact_reported_throughput_bytes_per_s']` every run, so the
  spec-conforming number is present in every result record even though the
  harness's own field is not, for this impl.
- Per ARTIFACT_GUIDE.md rule 5 (login node = build + minimal functional
  check only, no timing sweeps), the run below is a `--smoke` correctness
  gate, not a timing measurement.

## Report-convention mapping (empirically verified, per ARTIFACT_GUIDE rule
"first rule out a convention mismatch on a tiny hand-checkable automaton")

`obat` writes `report.txt` as `<offset>\t<original_ANML_id>` per line
(`report_formatter::print_to_file`, `src/commons/report_formatter.cpp`),
sorted + de-duplicated. `offset` is 0-based (the byte index that satisfies
the reporting STE); `original_ANML_id` is the STE's ANML `id` **string**
(not its `reportcode`). Verified on the artifact's own
`source/small_dataset/apple.anml` (pattern "apple", reportcode=2019, STE id
`__45__`) + `inputstream.txt` (81 bytes): `obat -g obat2` produced

```
5	__45__
47	__45__
64	__45__
75	__45__
```

This module's own `_simulate_reference`/`NumpyBitsetNFA` on the SAME
automaton+stream produced offsets `[5, 47, 64, 75]` with report_id `2019`
— **identical**, zero offset/base conversion needed. The adapter converts
`original_ANML_id` -> canonical `report_id` via
`automaton.states[id].report_id` (the same Automaton object it serialized
to ANML for `obat` to read). A small, CLI-flag-independent internal
alignment pad was observed (81-byte input -> "padding_input_stream = 4"
in `obat`'s own stdout); the adapter truncates reports at
`offset >= true_stream_length` before gating (same mechanism as real-
workload `gate_window_bytes`), so this padding cannot leak spurious
reports into the comparison.

## Gate result

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 \
  /pscratch/sd/c/cunyang/gnn/plexus_env/bin/python -m kernelbench.runner \
  --kernel string-regex-matching --variant regex-kernel-throughput-precompiled \
  --smoke --impl gpunfa-obat2

running gpunfa-obat2 smoke-literal-set     ... 862.494 ms  0.00 Gbps  (err 0.00e+00 <= None)  [22.6s]
running gpunfa-obat2 smoke-regex-features  ... 866.936 ms  0.00 Gbps  (err 0.00e+00 <= None)  [22.8s]
2/2 runs valid
```

Both smoke workloads pass the exact structural gate against the module's
independent reference simulator, with non-trivial match sets (report_count
6 and 8 respectively, matching the CPU floor impl's own counts exactly on
the same workloads) — not a vacuous zero-match pass. `Gbps` reads 0.00
because the harness's own timer for this impl includes CLI startup/ANML
parse (see contamination note above); the artifact's own clean
matching-only numbers are `artifact_reported_throughput_bytes_per_s`
278018.7 and 346330.8 (~2.2-2.8 Mbit/s on these ~2-3KB smoke corpora —
expected to be far higher on the spec's real MB-scale corpora; not
measured here per the login-node no-timing-sweep rule).

## Not done

- `infant` (iNFAnt reimplementation) and `ppopp12` (NFA-CG reimplementation)
  binaries are built but not wrapped by a separate adapter — `obat2` was
  judged the paper's own headline contribution; the other two are the
  paper's OWN baselines, lower integration priority given the 3h budget.
- Full spec-conforming timing (1/10/20MB corpus sweep, compute-node
  allocation) not attempted — out of scope for this login-node pass.
- No source patch was needed or made.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, on an
  A100-SXM4-40GB physical card, driver 595.71.05, node gpu-b11-6),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), cmake (system, via
  CMake's own `find_package`), torch 2.8.0+cu128, Python 3.12.14,
  `-DCMAKE_CUDA_ARCHITECTURES=80` unchanged. Two build-system changes
  needed (neither touches `source/`):
  1. **Host compiler pin.** Without an explicit `-DCMAKE_C(XX)_COMPILER`,
     CMake picked up this build env's own default g++ 13.4.0 (bundled
     alongside nvcc in the same conda env), and the vendored
     `gpunfa_code/src/commons/SymbolStream.cpp` failed to compile:
     `uint8_t`/`uint32_t` came up "not declared" (GCC 13's libstdc++ no
     longer transitively pulls `<cstdint>` in via `<string>`/`<set>`/
     `<vector>` the way the artifact's originally-recorded g++ 7.5.0 did,
     and `SymbolStream.h` has no direct `<cstdint>` include of its own).
     GCC's unknown-type error recovery then silently treated the
     `vector<uint8_t>`/`set<uint8_t>` members as plain `int` for the rest
     of the translation unit, cascading into "request for member 'insert'
     in ..., which is of non-class type 'int'" and similar. Fixed in
     `build.sh` by pinning `-DCMAKE_C_COMPILER`/`-DCMAKE_CXX_COMPILER`/
     `-DCMAKE_CUDA_HOST_COMPILER` to `${KB_GXX12:-g++-12}`/
     `${KB_GCC12:-gcc-12}` (exported by `toolchain.sh`, sourced above) --
     same class of nvcc-12.x/g++-13 friction, same fix pattern, documented
     elsewhere in this repo (`bench/artifacts/convolution/hidet/STATUS.md`,
     `grep -l ccbin bench/artifacts/*/*/build.sh`). No source file touched;
     `git diff` on `source/` remains empty.
  2. **`-j` lowered** from the artifact's original `-j8` to
     `-j"${KB_MAKE_JOBS:-4}"`: at `-j8`, concurrent `cc1plus`/`cicc`
     invocations from this build plus other builds sharing the login
     node's session-wide `RLIMIT_NPROC=256` hit
     `cc1plus: vfork: Resource temporarily unavailable` partway through;
     this is the documented login-node resource-contention gotcha, not an
     artifact defect (rebuilt clean at `-j4`).
  3. **Adapter fix (`adapter.py`, not `build.sh`)**: `run()` was
     unconditionally overwriting `env["LD_PRELOAD"]` with the hardcoded
     Perlmutter path `/usr/lib64/libstdc++.so.6` for the `obat` subprocess
     it launches, clobbering the correct, already-exported
     `LD_PRELOAD` (zaratan's `env.sh` points it at conda-forge's own newer
     libstdc++.so.6). `obat`, built here against the newer conda-forge
     libstdc++, failed at run time with `GLIBCXX_3.4.29 not found` when
     forced back onto the older `/usr/lib64` one. Fixed with
     `env.setdefault("LD_PRELOAD", _LD_PRELOAD)` -- keeps the calling
     shell's own value if already set, only falls back to the old
     hardcoded path if not (preserves Perlmutter behavior, where the
     parent shell did not already export `LD_PRELOAD` itself).
- Build: OK (after the two `build.sh` fixes above and one full `rm -rf
  build/` to drop a stale `CMakeCache.txt` from the failed first
  compiler). `infant`, `obat`, `ppopp12` all produced.
- Gate: `regex-kernel-throughput-precompiled` / `gpunfa-obat2` --
  **2/2 smoke workloads valid**: `smoke-literal-set` 376.113 ms, err
  0.00e+00; `smoke-regex-features` 394.297 ms, err 0.00e+00 (tolerance
  `None`, structural/exact-match gate, matches the recorded convention).
  `conforming: False` expected (`--smoke`, shared MIG slice, GPU clocks
  not locked) -- not a publishable timing run.
- Deviation from the recorded ruling: none in outcome (still 2/2 valid,
  exact structural match on both smoke workloads); the build-system fixes
  above were needed purely because of this machine's different default
  host compiler (g++ 13.4.0 here vs. g++ 7.5.0 on Perlmutter) and
  session-wide process-limit contention, not anything about the artifact
  itself. `Gbps` still reads 0.00 for the same harness-timer-contamination
  reason documented above (this adapter wraps a monolithic CLI); this
  reproduction did not separately re-capture
  `artifact_reported_throughput_bytes_per_s` into this file (visible in
  the JSON result record referenced in the gate log below).
- Verdict here: **BUILT+GATED** -- equals the recorded ruling.
