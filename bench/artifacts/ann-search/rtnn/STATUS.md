# rtnn (RTNN: ray-tracing-accelerated neighbor search) — STATUS

**Outcome: BUILD-FAILED — OptiX itself is available on this machine (see
below), but the artifact's 2022-era Thrust usage produces a Thrust ABI-
versioned-namespace mismatch against CUDA 12.9's bundled Thrust that a
minimal, non-kernel-touching patch cannot resolve within this pass**

- Paper: "RTNN: accelerating neighbor search using hardware ray tracing"
  (PPoPP'22). `PAPER_KEY = conf/ppopp/000122`.
- Artifact: https://github.com/horizon-research/rtnn
- Commit cloned: `5532e7031d0c8268ffa555972f074f8882b379b5`, `git clone --depth 1`.

## OptiX availability — corrected the a-priori assumption

The task brief's default expectation was that OptiX would be unavailable and
this artifact should be cheaply SKIPPED without a build attempt. That
expectation does NOT hold on this machine, and it is worth recording why:

1. `module spider optix` finds nothing, and no `optix` headers exist under
   any `/opt/nvidia/hpc_sdk/Linux_x86_64/*/` prefix (checked, bounded, depth
   1 listing only) — so the CUDA HPC SDK itself indeed does not vendor
   OptiX, confirming the task brief's premise as far as *that* source goes.
2. **RTNN vendors the OptiX 7.1.0 SDK headers itself**, in-repo, under
   `include/` (`optix.h`, `optix_7_host.h`, `optix_stubs.h`,
   `optix_function_table_definition.h`, ...) — its own README says so
   explicitly: *"You do not have to install the OptiX SDK yourself. The
   code is developed using the SDK as a template and includes all the
   necessary headers."* This is standard practice for OptiX research code
   (the programming headers are freely redistributable; only the full SDK
   *installer* bundle requires a click-through EULA) and is NOT us fetching
   a license-gated SDK — the artifact already ships what it needs.
3. OptiX's actual RUNTIME dependency is the NVIDIA display driver's
   built-in RT support (`optixInit()` resolves function pointers from a
   driver-bundled library, `libnvoptix.so`, dynamically — no separate
   toolkit-external install). This library **is present** on this machine:
   `/usr/lib64/libnvoptix.so.1` and `/usr/lib64/libnvoptix.so.580.159.04`
   (checked via a bounded `ls` of the known driver library directory, not a
   filesystem scan), matching `nvidia-smi`'s reported driver version
   580.159.04 on the A100. `src/CMake/FindOptiX.cmake`'s own
   `OptiX_INSTALL_DIR` default (`${CMAKE_SOURCE_DIR}/../`, i.e. the repo's
   own top-level `include/`) confirmed this at configure time: `cmake
   -DKNN=5 ..` succeeded cleanly with zero OptiX-related errors.

**So OptiX itself was never the blocker here.** The build proceeded past
CMake configuration, past OptiX-header-dependent CUDA/PTX compilation
(`aabb.cu`, `grid.cu` compiled clean), and failed only at the final link
step, for an unrelated reason (below).

## What actually failed, and the patches applied

1. **GCC 14 (this machine's default `g++`) is too new for nvcc 12.9's
   internal libstdc++-header parsing** (`std::allocator_traits`-family
   internal template errors in `<bits/alloc_traits.h>`). Fixed by building
   with `g++-12` (`/usr/bin/g++-12`, also present on this machine) as both
   `CMAKE_CXX_COMPILER` and `CUDA_HOST_COMPILER` — a toolchain-version
   workaround, not a source change, same category of fix used elsewhere in
   this repo (e.g. `bench/artifacts/bfs/blest/build.sh`'s toolchain note).
2. **Two genuinely missing `#include`s**, applied as minimal patches per
   ARTIFACT_GUIDE.md rule 3 ("CUDA-version guards are fine; touching kernel
   code is not" — these add no new logic, they only make already-called
   library functions visible):
   - `src/optixNSearch/thrust_helper.cu`: `thrust::count`, `count_if`,
     `unique`, `unique_by_key` are called but `<thrust/count.h>` and
     `<thrust/unique.h>` were never included. Under CUDA 11.x's bundled
     Thrust (this artifact's original PPoPP'22 dev toolchain) one of the
     already-included headers transitively pulled these in; CUDA 12.9's
     much more modular Thrust no longer does.
   - `src/optixNSearch/sort.cpp`: `thrust::host_vector` is used throughout
     but `<thrust/host_vector.h>` was never included — same root cause.
   Both patches are two-line, comment-documented, `#include`-only additions
   (see the files themselves for the inline `KERNELBENCH PATCH` comments).
3. **The actual, unresolved blocker**: after both `#include` fixes, the
   build still fails, only now at the FINAL LINK step, with `undefined
   reference` errors for every Thrust-wrapping helper function defined in
   `thrust_helper.cu` (`fillByValue`, `sortByKey`, `genSeqDevice`,
   `gatherByKey`, `countUniq`, `thrustCopyD2D`, `thrustGenHist`,
   `uniqueByKey`, `exclusiveScan`, `countIfInRange`, `copyIfInRange`,
   `copyIfIdInRange`). The mangled symbol names reveal the cause: the
   plain-`.cpp` translation units (`sort.cpp`, `search.cpp`, `optix.cpp`,
   compiled directly by `g++-12`) resolve Thrust's ABI-versioned inline
   namespace to the literal, UN-expanded token
   `thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::...`, while
   `thrust_helper.cu` (compiled by `nvcc`, which DOES define
   `__CUDA_ARCH_LIST__` as a builtin during its host-compilation pass)
   resolves the SAME namespace macro to a different, correctly-expanded
   token — so the two sets of translation units define/reference the exact
   same C++ symbols under two DIFFERENT mangled names, and the linker can
   never match them. This is a genuine Thrust/CUB versioned-ABI-namespace
   incompatibility between an artifact written years before Thrust
   introduced this mechanism and CUDA 12.9's much newer bundled Thrust —
   confirmed NOT fixable by passing `-D__CUDA_ARCH_LIST__=800` to the plain
   `g++` compile (tried; no effect, the macro is evidently consulted by
   Thrust's namespace-generation logic in a way that only nvcc's own
   preprocessing pass satisfies). A real fix would need either an older
   CUDA toolkit (none is installed on this machine — only 12.9 is present)
   matching RTNN's original ~CUDA-11-era Thrust ABI, or restructuring which
   translation units nvcc vs. the host compiler builds (e.g. renaming every
   Thrust-touching `.cpp` to `.cu` and updating the CMake build rules to
   route them through `CUDA_COMPILE` too) — a source/build-system
   reorganization beyond a "minimal patch," so not attempted here.

## Reproduction

```
cd source/src && mkdir build && cd build
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin:$PATH
cmake -DKNN=5 -DCUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 ..
make -j4
# configure: clean. aabb.cu/grid.cu/thrust_helper.cu: compile clean (after
# the two #include patches). Final link: undefined reference (Thrust ABI
# namespace mismatch, see above).
```

No timing or correctness gate was reached — the binary was never produced.
Not re-attempted with a from-scratch translation-unit reorganization within
this integration pass's time budget; recorded here as a concrete, actionable
finding rather than a cheap OptiX-unavailability skip (which would have been
factually wrong on this machine).
