# futhark-mem-sc22 — sequence-alignment

**Status: BUILD-FAILED** (host driver compiles; the generated GPU kernel
does not compile on this machine's CUDA 12.9 NVRTC)

- Paper: "Memory Optimizations in an Array Language" (SC'22) --
  `PAPER_KEY = conf/sc/MunksgaardHSO22`. NOT an alignment-specific paper (a
  Futhark compiler/memory-layout-optimization paper); `benchmarks/nw` is
  ONE of 7 unrelated memory-bound kernels it evaluates, per
  `benchspecs/sequence-alignment/survey.md`'s own relevance caveat.
- Artifact: https://github.com/diku-dk/futhark-mem-sc22
- Commit: `8018f67085b153d5414747b6d0227e1f6c463d9d` (2022-06-02), cloned
  with `git clone --depth 1`.
- Toolchain: `nvcc`/CUDA 12.9 (via `bench/artifacts/toolchain.sh`); GPU:
  single NVIDIA A100-PCIE-40GB, sm_80 (shared login-node GPU); system `cc`
  = SUSE gcc (`/usr/bin/cc`).

## Does the kernel exist? Yes.

`benchmarks/nw/futhark/nw.fut` -- global Needleman-Wunsch, BLOSUM62 24x24
substitution matrix, linear gap penalty (`pen`, default 10), tiled
anti-diagonal wavefront (`block_size=64`), sizes `row_length` in
{8192,16384,32768} per its own compiled-script header (`-- ==` block).
`benchmarks/nw/reference/nw.c` is verbatim Rodinia's OpenMP `nw` kernel
(credited in a code comment), the CPU baseline this paper benchmarks
against.

## Can the compiler be obtained without network/root beyond a tarball? Yes,
## more directly: it is already vendored.

`source/bin/futhark` is a **statically-linked Linux x86_64 ELF binary
already checked into this repo's own git history** (`Futhark 0.22.0`,
66.7 MB) -- no separate GitHub-releases tarball fetch was needed; `git
status` on the cloned `source/` tree is clean (nothing added or modified to
obtain it).

## Attempt 1: `futhark cuda nw.fut` with the plain environment -- host compile fails

Futhark 0.22.0 supports a **native CUDA backend** (`futhark cuda`, distinct
from `futhark opencl`) that generates a C file calling the CUDA driver API
+ NVRTC. Run directly (with `bench/artifacts/toolchain.sh` sourced, i.e.
`CUDA_HOME`/`CPATH` pinned to 12.9), the generated `nw.c` fails to compile:

```
cc failed with code 1:
nw.c: In function 'cuda_setup':
/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/include/cuda.h:90:45: error: too few arguments to function 'cuCtxCreate_v4'
```

**Root cause, isolated**: this machine's system `cc` resolves
`#include <cuda.h>` to the **login-default module's CUDA 13.2** tree even
though `CPATH` names the toolchain-pinned 12.9 tree first (confirmed with
`cc -H`: with only `CPATH` set, `cc` opens
`.../26.5/cuda/13.2/include/cuda.h`; the exact same `cc` invocation with an
explicit `-I/opt/.../25.5/cuda/12.9/include` flag opens the 12.9 file and
compiles cleanly). This is a genuine quirk of this specific system image
(GCC's documented search order puts `-I` and `CPATH` ahead of any implicit
paths, so an explicit `-I` should not be *necessary* to beat a bare system
default -- but empirically here it is: `CPATH` alone does not override
whatever is giving `cc` a bare `<cuda.h>` resolution to 13.2). CUDA 13.2's
newer `cuCtxCreate_v4` driver-API signature is binary/source-incompatible
with the older call site Futhark's RTS emits, so this is exactly the kind
of CUDA-major-version drift `toolchain.sh` exists to avoid -- it just
doesn't reach `futhark`'s own internal `cc` invocation, which passes no
CUDA-specific flags at all and relies entirely on environment/default
search paths.

## Fix (build-system only, ARTIFACT_GUIDE.md rule 3): a `cc` shim

`cc_shim/cc` (checked into this artifact directory, not the system) is a
5-line wrapper that execs the real `/usr/bin/cc` with explicit
`-I$CUDA_HOME/include -L$CUDA_HOME/lib64 -Wl,-rpath,... -L.../stubs`
flags, then is prepended onto `PATH` before invoking `futhark cuda`. This
is an include/library-path fix, not a code change -- the same category
ARTIFACT_GUIDE.md rule 3 explicitly sanctions ("Build-system fixes (arch
flags, include paths, CUDA-version guards) are fine"). With the shim in
place, `futhark cuda nw.fut` compiles the host-side `nw` driver **cleanly,
zero errors** (only Futhark's own pre-existing `intrinsics.fut` warnings,
unrelated to this integration).

## Attempt 2: running the compiled binary -- device-side NVRTC compile fails

`./nw -e mk_input` (or any real run) triggers Futhark's runtime to JIT the
actual GPU kernel via NVRTC at first launch. This fails:

```
./nw: NVRTC compilation failed.

futhark-cuda(1130): error: identifier "mulhi" is undefined
    return mulhi(a, b);

futhark-cuda(1134): error: identifier "mul64hi" is undefined
    return mul64hi(a, b);

2 errors detected in the compilation of "futhark-cuda".
```

## Root cause

Futhark 0.22.0's CUDA runtime-system (RTS) header emits calls to the
**unprefixed** CUDA device intrinsics `mulhi`/`mul64hi` for 64-bit
multiply-high operations. These unprefixed names were only ever provided
by older CUDA headers as convenience macros; the correct, currently
supported device intrinsics are the double-underscore-prefixed
`__mulhi`/`__mul64hi`. NVRTC on this machine's CUDA 12.9 no longer defines
the unprefixed aliases (they were dropped between whatever CUDA version
Futhark 0.22.0 (released ~2023) was developed/tested against and CUDA
12.9). This is a version-drift bug **inside Futhark's own bundled RTS code
generation**, not in `nw.fut` itself and not in anything this integration
wrote -- the identical failure would occur for any Futhark CUDA-backend
program compiled with this Futhark release against this CUDA version, not
just `nw`.

## Why this is BUILD-FAILED, not a further build-system patch

Fixing this would require patching Futhark's own generated/RTS C code (the
`futhark-cuda(1130)`/`(1134)` lines are Futhark-emitted, not user code) --
either by modifying the vendored `bin/futhark` binary's embedded RTS
sources (not available separately from the binary; would require building
Futhark itself from source, a materially larger undertaking entirely
outside this task's scope) or by hand-patching the NVRTC-compiled string at
runtime (not a supported extension point). This is a toolchain-compatibility
failure of the SAME KIND as MASA-CUDAlign's legacy-texture-API removal
(`../masa-cudalign/STATUS.md`) and is treated the same way: **BUILD-FAILED
with evidence**, no attempt to patch compiler-internal/generated code.

## Disposition

BUILD-FAILED. `build.sh` reproduces both the (now-fixed) host-compile step
and the (unfixed, and not fixable within this task's scope)
device-compile failure deterministically, exiting 1. No adapter.py is
provided (nothing runnable to wrap: the binary exists but cannot execute
its one GPU kernel). This was the intended "third" GPU-eligible candidate
for `seqalign-exact-pairwise-kernel`'s protein/BLOSUM62 tier; with it
unavailable, that tier has no GPU paper-artifact competitor in this
integration pass -- see `../README.md` for the track-level accounting.

## Not done

- Building Futhark from source against this machine's CUDA 12.9 (to get an
  RTS that emits `__mulhi`/`__mul64hi`) -- a materially larger undertaking
  (Futhark is a Haskell project with its own toolchain/dependency
  footprint) that is out of scope for this integration pass; the task
  brief's own bar ("a prebuilt Futhark compiler binary... into the artifact
  dir") was met (the binary IS vendored and DOES work up to this
  CUDA-version-specific NVRTC failure), not exceeded.
- The OpenCL backend (`futhark opencl`, this repo's own default/documented
  path, using the system's `libOpenCL.so`/`libnvidia-opencl.so`, both
  present on this machine) was not attempted: the task brief specifically
  asks whether the kernel "compiles with the CUDA backend," and a
  cross-check via a materially different backend was judged out of scope
  given the CUDA-path failure is already fully diagnosed and is a Futhark-
  internal (not nw.fut-specific) version-compatibility issue that an
  OpenCL-backend success would not actually resolve for this task's
  CUDA-only integration scope (ARTIFACT_GUIDE.md's scope ruling: "NVIDIA
  GPU, single-card" -- OpenCL-via-NVIDIA-driver would technically satisfy
  "single-card NVIDIA GPU" but is a materially different code path from
  every other artifact in this track).
