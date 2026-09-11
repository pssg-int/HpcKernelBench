# MANS — STATUS

**Outcome: BUILT — correctness gate CRASHES (CUDA runtime fault inside
MANS's own GPU kernel, not this adapter's glue code) — recorded as a
result per ARTIFACT_GUIDE rule 4, not silently worked around**

Paper: "MANS: Efficient and Portable ANS Encoding for Multi-Byte Integer
Data on CPUs and GPUs", SC 2025. PAPER_KEY = `conf/sc/HuangYYLGLJWFHT25`
(matches `output/included.json`'s entry for
`https://github.com/hpdps-group/MANS` and
`benchspecs/lossless-compression/spec.yaml`'s own evidence key). Repo:
`https://github.com/hpdps-group/MANS`
(commit `7e9265f0d90027a7f4ea399feceba376d58b0b24`, 2026-04-04;
`git clone --depth 1` into `./source/`). `git status`/`git diff` inside
`source/` is empty — zero patches to the artifact's own code.

## Why CMake was bypassed (no kernel code touched)

MANS's own build is CMake (`source/CMakeLists.txt`, `cmake
-DTARGET_PLATFORM=nv ...`), but `source/nv/ans/CMakeLists.txt` (a nested
subproject for the `gpu_ans` shared library) hardcodes
`set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)` before its own
`project(... LANGUAGES CUDA ...)` call — that path does not exist on this
machine (nvcc lives under
`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/`), and CMake's
`project()` rejects a `CMAKE_CUDA_COMPILER` value that isn't itself a real
executable before this integration's own `-DCMAKE_CUDA_COMPILER=$(command -v
nvcc)` cache entry gets a chance to override the subproject's own hardcoded
`set()`. Confirmed by direct reproduction:
```
cmake -DTARGET_PLATFORM=nv -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" -DBUILD_HDF5_PLUGIN=OFF ..
...
CMake Error at nv/ans/CMakeLists.txt:9 (project):
  The CMAKE_CUDA_COMPILER: /usr/local/cuda/bin/nvcc
  is not a full path to an existing compiler tool.
```
Rather than patch that line (which would touch `source/`, and every other
GPU artifact integrated into this benchmark so far — fzgpu, cuszp, gpulz,
pfpl — has kept `source/` at zero patches by bypassing the artifact's own
build system instead), `build.sh` invokes `nvcc` directly on the exact
source-file list `CMakeLists.txt`'s own `BUILD_NV` section names for the
`mans_nv_core` target: `nv/mans_nv.cpp`, `nv/adm/mapping_uint{16,32}.cu`,
`nv/ans/mans_nv_ans.cu`, `nv/ans/src/GpuANS.cu` — same pattern this
benchmark's other GPU compression artifacts already use to sidestep a
broken/portability-limited build system without editing it.

`mans_api.cpp` (the CPU+GPU dispatch layer, `source/mans_api.cpp`) is
**not** compiled — it `#include`s the CPU backend header
(`cpu/mans_cpu.h`) unconditionally in a way that pulls in
`-march=native -mavx512f`/OpenMP requirements this integration does not
need; `bridge_mans.cu` calls MANS's lower-level, NV-only device-pointer
functions (`mans::nv::compress_internal_device` /
`decompress_internal_device`, declared in `source/nv/mans_nv.h`) directly
instead — the finest boundary available, and the same functions
`mans_api.cpp`'s own `compress_device`/`decompress_device` wrappers
ultimately forward to, unmodified.

## Adapter

`IMPL_NAME = "mans-compress"`, `PRECISIONS = ["fp32"]`. `direction =
"compress"` per this domain's contract, mirroring
`gpulz-compress`/`fzgpu-compress`/`cuszp-compress`. Device buffers are
allocated with torch (`ctypes.c_void_p(tensor.data_ptr())`), the cuszp-
compress pattern. `bridge_mans.cu` is a thin `extern "C"` forwarder around
MANS's own `mans::nv::compress_internal_device`/`decompress_internal_device`
(needed only because those are C++ functions taking a `std::size_t&`
out-parameter, which ctypes cannot call) — no kernel code touched; MANS's
dtype = U16, mode = P ("portable, GPU-consistent ANS", MANS's own README
term and `mans_api.cpp`'s own `default_params()` default).

## Build

```
./build.sh
```
Compiles clean, first attempt, no patches, no warnings requiring
investigation:
```
nvcc -O3 -arch=sm_80 -std=c++17 -Xcompiler -fPIC --shared \
  -I source -I source/nv -I source/nv/ans/include -I source/nv/ans/src \
  bridge_mans.cu source/nv/mans_nv.cpp source/nv/adm/mapping_uint16.cu \
  source/nv/adm/mapping_uint32.cu source/nv/ans/mans_nv_ans.cu \
  source/nv/ans/src/GpuANS.cu -o bridge.so
```
`nm -D bridge.so` confirms all 3 expected `extern "C"` symbols exported
(`mans_compress_device`, `mans_decompress_device`,
`mans_max_compress_bytes`). `available()` returns `(True, "")`.
`mans_max_compress_bytes(dtype=U16, mode=P, num_elements=27648)` returns a
sane bound (665160 bytes) — the library loads and its pure-sizing entry
point runs correctly; the failure below is specific to the actual compress
KERNEL launch, not to linking/loading.

## Gate verification — CRASHES, not a PASS/FAIL correctness result

Mandated command:
```
$PY -m kernelbench.runner --kernel lossless-compression \
    --variant lossless-comp-gpu-multibyte-dual-scope --impl mans-compress \
    --smoke --warmup 1 --reps 3
```
**This command hangs (no output, 2+ minute timeout, no progress) on the
shared login-node A100** — traced (see below) to MANS's own compress
kernel corrupting the CUDA context on the very first call, before the
runner even finishes timing the first of 3 smoke workloads. Killed via
external timeout; `nvidia-smi` confirmed clean afterward (no orphaned GPU
process).

### Isolated reproduction (single call, own short-lived process — does not hang, fails cleanly)

Two independent single-shot reproductions, run directly against
`bridge.so` (bypassing the runner, one call each, process exits
immediately after) to separate "this adapter's glue code is broken" from
"MANS's own kernel faults on this input":

1. Synthetic full-range random `uint16` (`np.random.randint(0, 65535,
   27648)`, avoiding this benchmark's synthetic Field content entirely) —
   `mans_compress_device` returns `rc=-1` after printing
   `mans_compress_device: map_values_kernel_decoupled sync: an illegal
   memory access was encountered` (MANS's own `check_cuda()` converts the
   CUDA fault to a caught `std::runtime_error`, which `bridge_mans.cu`'s
   `try`/`catch` catches — no process crash from OUR code, but the CUDA
   context is now sticky-corrupted for the rest of the process).
2. **The compression domain's own actual smoke workload**
   (`kernelbench.domains.compression.smoke_workloads()[0]`, i.e. the exact
   `Field` the mandated runner command feeds to `prepare()`/`run()`), raw
   bytes reinterpreted as `uint16` — `mans_compress_device` returns `rc=-1`
   after printing `mans_compress_device: cudaMemcpyAsync packed bits D2D:
   invalid argument` — a different symptom (a computed buffer size
   overflows/goes negative rather than an out-of-bounds write triggering
   an MMU fault), same root cause.

### Root cause (read from `source/nv/adm/mapping_uint16.cu`, unmodified)

MANS's ADM ("Adaptive Data Mapping") stage encodes every element as a
signed offset from a per-thread-block "center" (the mean of that block's
16 elements), using `output_idx = diff / 126` unary-coded "chunk" bits per
element. The per-thread scratch buffer that holds these bits is FIXED-SIZE:
`local_bit_signal[cmp_chunk * max_bytes_signal_per_ele_16b]` = `16 * 2` =
32 bytes (256 bits) for **all 16 elements a thread encodes, combined** —
i.e. the design implicitly budgets at most ~16 bits/element of "chunk"
overhead, which only holds if `|value - block_center| ≲ 2000` for every
element in every block. This is true by construction for MANS's own target
domain (SZ/cuSZ quantization-code residuals — small, tightly-clustered
integers by design of error-bounded prediction) but is **not checked or
enforced anywhere in `compress_u16_device`** for arbitrary multi-byte
integer input: this benchmark's raw-byte-reinterpretation of a `float32`
array (any content — the IEEE 754 exponent field alone varies across
enough values that per-block value ranges routinely span thousands to tens
of thousands) blows this budget on virtually every block. The result is
either (a) an out-of-bounds write inside `map_values_kernel_decoupled`'s
per-thread local array / the shared `d_concatenated_signals` global buffer
(construction (1) above), or (b) a `bit_signals_size`
(`source/nv/adm/mapping_uint16.cu:880-881`) computed from a corrupted
`signal_lengths`/`output_lengths` D2H readback that overflows/underflows
into an invalid `cudaMemcpyAsync` size (construction (2) above). Neither
path is bounds-checked or gracefully rejected before the kernel launch;
`get_max_u16_payload_bytes` (the function this integration calls to size
the compressed-output buffer, matching MANS's own `mans_api.cpp` usage)
provides no warning that the INPUT itself, not just the output buffer, has
an implicit narrow-dynamic-range precondition.

**This is a genuine artifact reliability bug, not an integration mistake**:
`bridge_mans.cu` calls MANS's own unmodified, documented device-pointer API
with straightforward arguments (dtype=U16, mode=P, a flat device buffer, an
output buffer sized via MANS's own `get_max_compress_bytes`); the crash
originates entirely inside MANS's own kernel and sizing arithmetic. It is
also the likely explanation for why the full runner invocation *hangs*
rather than erroring cleanly like the two isolated single-call
reproductions above: `map_values_kernel_decoupled`'s "decoupled look-back"
prefix-sum (`source/nv/adm/mapping_uint16.cu:684-714`) is a spin-wait
(`while(lookback>0) { do { status = prefix_state[lookback]; ... } while
(status==0); ... }`) across thread blocks — if an earlier block never
reaches its `prefix_state[...] = ...` write (e.g. because it faulted or
took a divergent early-exit path under the overflow condition above), a
later block's warp-0 thread can spin on that flag indefinitely; combined
with a CUDA context already left sticky-corrupted by the FIRST of the 5
compress calls the mandated smoke command issues per workload
(1 gate + 1 warmup + 3 reps), this plausibly explains the observed hang
rather than a clean per-call failure once inside the full runner loop.

## Byte semantics (why no "friendlier" input was substituted)

This benchmark's `--smoke` protocol always calls the compression domain's
own `smoke_workloads()` (`kernelbench/domains/compression.py`) — three
fixed synthetic `float32` `Field`s (smooth/turbulent/multiscale), the SAME
inputs every other lossless-compression impl in this track is gated
against. Per this task's own instruction not to modify the domain module,
and per ARTIFACT_GUIDE rule 4 ("do not loosen the gate to make it pass"),
this integration does **not** substitute a narrow-dynamic-range synthetic
integer stream MANS could actually digest cleanly — doing so would (a)
require a lossy quantization step to manufacture "clustered" values, which
cannot round-trip bit-exactly against the ORIGINAL `float32` `Field` the
domain's `reference_compression()` checks against (defeating the
correctness gate's purpose), and (b) mean this adapter is no longer being
gated on the same workload as every other implementation in this track.
The crash above is reported as the honest, reproducible result of running
MANS's actual GPU kernel through this benchmark's actual mandated protocol.

## Verdict

`mans-compress: BUILT (library loads, sizing API works, correctly wired to MANS's own unmodified NVIDIA device-pointer compress/decompress API) — GATE CRASHES: MANS's compress kernel (map_values_kernel_decoupled / map_values_kernel_thrust, source/nv/adm/mapping_uint16.cu) has an unbounded per-block dynamic-range assumption (~16 bits/element chunk-encoding budget) with no input validation, causing an illegal-memory-access or invalid cudaMemcpy size on this benchmark's generic byte-stream smoke workload, and a process hang across repeated calls in the full runner protocol -- a genuine artifact reliability bug (research finding), not an adapter defect.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge), torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80` unchanged from the recorded build.
- Build: OK. Build-system changes: none (idempotent rebuild, no diff to `source/`).
- Gate: mandated command run wrapped in `timeout 300` inside the srun allocation (per this task's
  instruction to reproduce the CRASH faithfully, not chase a possible hang). Result:
  `lossless-comp-gpu-multibyte-dual-scope` mans-compress, `smoke-smooth-3d`: the first compress call
  prints `mans_compress_device: cudaMemcpyAsync packed bits D2D: invalid argument`
  (same symptom as this file's own isolated single-call reproduction #2 above), the adapter's
  `run()` raises `RuntimeError: mans_compress_device failed (rc=-1)`, and the runner process exits
  uncaught with a Python traceback and exit code 1 — a clean CRASH, not the 2+ minute hang seen on
  the original (Perlmutter, shared login-node) reproduction. `0/3 runs` (only the first workload was
  reached before the process died).
- Deviation from the recorded ruling: minor, disclosed — on this machine's A100 MIG slice the gate
  terminates via an uncaught Python exception (traceback, exit code 1) immediately on the first
  smoke workload, rather than hanging with no output for 2+ minutes as originally observed on
  Perlmutter. Root cause is the same underlying MANS kernel/sizing defect (documented above,
  unchanged, `source/` still at zero patches) — same construction-2 symptom
  (`cudaMemcpyAsync ... invalid argument`) as the original isolated single-call repro; only the
  outer failure mode (hang vs. immediate crash) differs, plausibly because this run used a single
  `timeout`-wrapped process on an isolated MIG slice rather than the original's login-node
  environment. The correctness gate never reaches a PASS/FAIL comparison either way.
- Verdict here: BUILT (gate fails: CRASHES) — same as the recorded ruling (BUILT, gate crashes
  inside MANS's own kernel); failure MODE differs (immediate crash vs. hang) as disclosed above.
