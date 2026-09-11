# SSpMV (LeSpMV) — STATUS

**Outcome: BUILT+GATED — gate PASSES**

Paper: "SSpMV: A Sparsity-aware SpMV Framework Empowered by Multimodal
Machine Learning", DAC 2025. PAPER_KEY = `conf/dac/LinLDZLY25`.
Repo: `https://github.com/lsl036/SSpMV`
(commit `3acff5afd8f375062e43d45097befc36a5f7647d`, 2025-11-24;
`git clone --depth 50` into `./source/`).

## Scope decision

The paper's actual contribution is `SmartAdpter/`, an ML model that picks
the best sparse format + kernel per matrix (trained weights not included in
the repo, inference pipeline is a separate Python/xgboost/pytorch stack).
The kernels it selects among live in `LeSpMV/`, a CPU SpMV kernel library
supporting CSR/BSR/DIA/ELL/S-ELL/SELL-C-sigma/SELL-C-R/CSR5. Per the "wrap
the kernel, not the paper's benchmark script" rule, this adapter wraps
`LeSpMV`'s always-available, default CSR kernel — `LeSpMV_csr<int,double>`
(`kernel_flag=1` → `__spmv_csr_omp_simple`, an OpenMP-parallel row-per-
thread CSR SpMV; `source/LeSpMV/src/spmv_csr.cpp`) — rather than
reimplementing the ML selector.

## Build-system patches (all in `source/LeSpMV/CMakeLists.txt`; full diff
via `git diff` in `source/`, commit not made — working tree only)

1. **Forced `icx` with a broken fallback guard.** The file did
   `set(CMAKE_CXX_COMPILER "icx")` unconditionally, immediately before an
   `if(NOT CMAKE_CXX_COMPILER) set(... "g++") endif()` block that can now
   never trigger (the variable is always already set) — this machine has no
   `icx`. Removed the unconditional `set()`, restoring the guard's evident
   intent (fall back to g++ when no compiler is pre-selected).
2. **`src/spmv_csr5.cpp` uses AVX-512 intrinsics (`_mm512_*`) with no
   `#ifdef` guard.** CSR5 is documented in the repo's own README as "This
   format only support the Intel CPU" (i.e. AVX-512). This login node's CPU
   (AMD EPYC 7713) has no AVX-512F (confirmed via DiaQ's own cmake probe
   during that build: "avx512f NOT present on this CPU"). We only need the
   CSR kernel (`spmv_csr.cpp`, unaffected), so `spmv_csr5.cpp` is excluded
   from `LIB_SOURCES` rather than forcing `-mavx512f` (which would compile
   but SIGILL if that codepath ever ran here, and would change codegen
   globally since `CMAKE_CXX_FLAGS` is shared across all files).
3. **`utils/test_spmv_csr5.cpp`** explicitly instantiates a helper
   (`test_csr5_matrix_kernels<int,uint32_t,double>`) that references
   `LeSpMV_csr5`; with (2) applied this became an unresolved symbol —
   harmless at `LeSPMV_shared`'s *build* time (shared libs allow undefined
   symbols by default) but fatal at `ctypes.CDLL` *load* time:
   `OSError: .../libLeSPMV.so: undefined symbol: _Z11LeSpMV_csr5IijdE...`.
   Excluded from `LIB_UTILS` for the same AVX-512-unavailable reason.
4. **`baseline_mkl_csr` test target unconditionally does
   `find_package(MKL CONFIG REQUIRED)`**, which fails cmake's *configure*
   step outright (MKL not installed/wired up in this environment) —
   independent of which target is later selected to build, since the
   `foreach(test_src ...)` loop runs at configure time. We don't need any
   test executable (our adapter calls the library directly via a shim), so
   the whole "build a test exe per `test/*.cpp`" block is now gated behind
   `option(LESPMV_BUILD_TESTS ... OFF)`, default off.
5. **Runtime `libstdc++` ABI mismatch.** `ctypes.CDLL` on the harness's
   Python failed with `OSError: .../libstdc++.so.6: version
   'CXXABI_1.3.15' not found` — this login node's Python resolves an older
   NERSC-conda `libstdc++.so.6` ahead of this g++-14's own at runtime.
   Fixed by linking both `libLeSPMV.so` and `csr_shim.so` with
   `-static-libstdc++ -static-libgcc` (in `build.sh`, not a source patch).

All five are build/link-environment fixes; **no kernel source
(`src/spmv_*.cpp` other than the CSR5 exclusion) was modified**, and
`LeSpMV_csr` itself is called completely unmodified.

## The shim

`LeSpMV_csr<int,double>` is a C++ template with no `extern "C"` entry
point. `csr_shim.cpp` (this directory, NOT part of the artifact) is a ~30
line wrapper that builds the artifact's own `CSR_Matrix<int,double>`
aggregate from raw CSR arrays and calls `LeSpMV_csr` directly — the same
role a pybind11 binding plays for a Python-first artifact. It is compiled
into `csr_shim.so`, linked against `libLeSPMV.so` (built by `source/LeSpMV`'s
own, lightly-patched CMake), and driven from `adapter.py` via ctypes.

## Build

```
./build.sh
```
Builds `source/LeSpMV/build/libLeSPMV.so` (target `LeSPMV_shared` only,
`cmake --build ... --target LeSPMV_shared`) then `csr_shim.so`. Toolchain:
g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`, actually SUSE system gcc),
cmake 3.28.3, `-O3 -march=native -fopenmp` (native = AVX2, no AVX-512).
Idempotent (`rm -rf build` at the top).

## Adapter

`IMPL_NAME = "sspmv-lespmv-csr"`, `PRECISIONS = ["fp64"]` (the
`<int,double>` instantiation; `<int,float>` also exists in the library but
is not wrapped here). `prepare()` builds contiguous int32/fp64 CSR arrays
plus a `U(-1,1)` seeded x-vector (same generation as
`kernelbench/impls/cpu_ref.py::reference_spmv`, so the correctness gate
compares against matching input). `run()` is exactly one
`sspmv_csr_f64(...)` call (→ one `LeSpMV_csr` call, `alpha=1, beta=0`).

## Gate verification (mandated command, run as specified)

```
$PY -m kernelbench.runner --kernel spmv --variant spmv-csr-kernel \
    --impl sspmv-lespmv-csr --matrices cant --warmup 1 --reps 3
```

```
  loading cant ...
  running sspmv-lespmv-csr cant  ... 99.721 ms  0.08 GFLOP/s  (err 5.57e-16 <= 1e-09)
1/1 runs valid
```

`max_scaled_err = 5.57e-16`, tolerance `1e-9` — **PASS**, five orders of
magnitude of margin. (The 0.08 GFLOP/s / ~100ms figure is a login-node,
warmup=1/reps=3, shared-CPU number — explicitly non-conforming per the
runner's own output — recorded here only as evidence the gate ran, not as a
timing claim.)

## Verdict

`sspmv-lespmv-csr: BUILT+GATED err=5.57e-16 (tol 1e-9)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, CPU-only artifact; login-node build, gate run through
  `bench/gpu_run.sh` (an A100 MIG 1g.5gb slice's compute-node CPU core, used
  purely for a clean, uncontended core allocation -- no GPU code path in
  this artifact).
- Toolchain: gcc/g++ 13.4.0 (conda-forge), cmake 4.2.3, Python 3.12.14,
  `-O3 -march=native -fopenmp` (AVX2, no AVX-512 on this CPU either).
- Build: OK, after one build-system fix. Build-system changes: `build.sh` --
  `cmake --build ... -j "$(nproc)"` (nproc=128 on this login node) exhausted
  the shared-session `RLIMIT_NPROC=256` under concurrent sibling builds
  (`cc1plus: ... vfork: Resource temporarily unavailable`, observed directly
  mid-build); capped to `-j "${KB_MAKE_J:-8}"`, overridable, per the
  documented login-node process-limit gotcha (resource contention, not an
  artifact defect).
- Gate: mandated command (`--matrices cant --warmup 1 --reps 3`): PASS, err
  5.57e-16 <= 1e-9 -- matches the recorded value exactly. (First attempted
  directly on the login node; the matrix load stalled for minutes under this
  session's shared-login-node I/O contention from concurrently running
  sibling reproduction agents -- confirmed via `ps`/`/proc` that the process
  was genuinely CPU/IO-starved, not deadlocked. Re-run through
  `gpu_run.sh`'s dedicated compute-node allocation completed in 0.1s, this
  is purely an environment workaround, not a change to what is measured or
  gated.)
- Deviation from the recorded ruling: none -- identical error value.
- Verdict here: BUILT+GATED -- same as the recorded ruling.

## Baseline role (2026-09-05 selection-rule revision)

**Competitor, not a SOTA baseline.** Under the revised rule (core kernel papers
evaluated on the track's own input regime first; `kernel-papers/output/
baseline_selection.md`), this artifact would not have been selected:
- no single-NVIDIA-GPU kernel path (current scope).
Rating rationale (`output/kernel_centrality.json`): SSpMV's headline is an ML-based algorithm/parameter selector specifically for SpMV, evaluated at kernel level against SOTA sparse libraries across multi-core platforms; CPU-only.
It stays in the registry and runs under the same gate as every other
implementation, but Phase 3 does not treat it as the human-SOTA reference for
`spmv`.
