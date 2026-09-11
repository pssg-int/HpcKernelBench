# flashattention-t (FlashAttention-T) — attention-kernel

**Status: BUILT+GATED**

- Paper: "FlashAttention-T: Towards Fully Tensorized Attention by Exploiting
  Tensor-Vector Parallelism", PPoPP'26. `PAPER_KEY = conf/ppopp/Xu0BXX00000C26`
  (matched by title in `../../../output/included.json`).
- Artifact: the paper's own AE reproduction package (repo name not otherwise
  published; cloned per task instructions from `guoqiao7/FlashAttention-T`).
- Commit cloned: `4a204b777f99606635a7a92886ee727d9043d32f`, `main` branch,
  `git clone --depth 1`.
- Toolchain: python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch
  `2.8.0+cu128`, nvcc `12.9` (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  host compiler `/opt/cray/pe/gcc-native/14/bin/g++` (GCC 14), GPU arch
  `sm_80` (A100, `-gencode=arch=compute_80,code=sm_80`).

## What the artifact actually is, and the wrapping boundary chosen

This repo is an artifact-EVALUATION reproduction package, not a library: top
level splits into `1-figure8-main-results/`, `2-figure11-synthetic-precision/`,
`3-table2-end2end-generative-accuracy/`. The Ampere kernel lives at
`source/1-figure8-main-results/flashattention-t/ampere/` (a `hopper/` sibling
exists for H100 and is unused — our GPU is an A100). There is **no
`setup.py`** under `ampere/` (only `hopper/` has one) — the artifact's own
workflow for Ampere compiles a standalone CLI benchmark binary via CMake
(`cxx-tests/fwd_bench/`), not a pip-installable Python extension.

Per ARTIFACT_GUIDE rule 1 ("wrap the kernel, not the paper's benchmark
script"), `cxx-tests/fwd_bench/` (the paper's own CSV-writing benchmark
binary) was **not** built. Instead this integration wraps
`cxx-tests/custom_api/flash_api_custom.cuh`'s `custom_mha_fwd_{causal,noncausal}<Dtype,
HeadDim, CtaM, CtaN, NWarps>(...)` — the artifact's own header-only C++ entry
point for its tensorized forward kernel (the same function
`cxx-tests/fwd_bench/benches/core.cuh`'s `bench_fwd_fp16` calls). This
function itself calls `FLASH_NAMESPACE::run_custom_mha_fwd<...>()`, a
template **defined in a header**
(`csrc/flash_attn/src/flash_fwd_launch_template.h:~386`) — unlike the
standard (non-tensorized) FlashAttention-2 comparison path, the custom
tensorized kernel is **not** split into the 114
`flash_{fwd,bwd}_hdim*_{fp16,bf16}_{,causal}_sm80.cu` explicit-instantiation
files, so none of those 114 files needed to be compiled (that 114-file tree
exists only for the artifact's own `fp16-fwd-bench-orig` non-tensorized
comparison baseline, out of scope here).

`wrapper.cu` (this directory, NOT under `source/`) is a **new** pybind11
boundary file — analogous to `bench/artifacts/spmm/rassm/wrapper.cpp` — that
`#include`s the artifact's unmodified `flash_api_custom.cuh` and instantiates
`custom_mha_fwd_{causal,noncausal}` for exactly `head_dim in {64,128}` x
`dtype in {fp16,bf16}`, forward-only (`(CtaM,CtaN,NWarps)` per head_dim copied
verbatim from the artifact's own `cxx-tests/fwd_bench/benches/hdim{64,128}-
{,non}causal.cu`, since those tile sizes are the artifact's own choice for
each head dim, not something this integration invented). The
`FLASHATTENTION_DISABLE_*` / `USE_*` / `LOOP{1,2}_USE_ACCS_SCALE_*` macros at
the top of `wrapper.cu` are copied **verbatim** from the artifact's own
`cxx-tests/fwd_bench/benches/core.cuh` — several gate `#error`s in
`softmax_mma.h` if undefined, so this is a required, not decorative, copy of
the artifact's own configuration, and select which ILP-softmax scheduling
code path compiles in (the paper's actual contribution). **No artifact
source file was edited.**

## Build (`build.sh`)

JIT-compiles `wrapper.cu` via `torch.utils.cpp_extension.load()` (ninja
backend), `build_directory=_build/`, idempotent (skips if the cached `.so`
already imports). Compiles in seconds (a handful of template instantiations,
not the full 114-file tree) — well inside budget. Known machine-specific
compiler workaround (same fix as `bench/artifacts/sddmm/fused3s/build.sh`):
`CXX`/`CC` pointed at `/opt/cray/pe/gcc-native/14/bin/g++`/`gcc` (bypasses a
broken `-B .../compiler_compat` the venv's distutils bakes in, which
otherwise picks up an ancient GCC-7-era `cc1plus`).

## Finding: a second, distinct libstdc++ ABI trap (not the fused3s one) — read before running the gate

`torch.utils.cpp_extension.load()`'s generated `build.ninja` link rule is a
plain `$cxx $in $ldflags -o $out` — **it does not read the `LDSHARED`
env var at all** (that mechanism is distutils-only, and only applies to
`setup.py build_ext`, e.g. PAT's and fused3s's build). So `wrapper.cu`'s
compiled `.so` has **no `RPATH`/`RUNPATH`** at all (confirmed via
`readelf -d`), unlike fused3s's `F3S.so`, which got one baked in via its
`setup.py`-driven `LDSHARED` override.

Reproduced and root-caused directly (not guessed):
- This venv's python (`/pscratch/.../plexus_env/bin/python`) is a **symlink**
  to `/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python`.
  A bare `import torch` (nothing else) causes the process to map
  `.../nersc-python/lib/libstdc++.so.6.0.32` — an **old** libstdc++ missing
  `CXXABI_1.3.15` (confirmed via `/proc/self/maps`; this is loaded as a side
  effect of torch's own C-extension import chain, before any of our code
  runs). The system's own `/usr/lib64/libstdc++.so.6.0.34` **does** have
  `CXXABI_1.3.15` (checked directly), but once the old copy is resident,
  glibc's loader will not silently load a second copy of the same SONAME —
  any subsequently-loaded object requiring `CXXABI_1.3.15` (our `wrapper.cu`,
  compiled with GCC 14) fails to import with exactly the error reproduced
  below.
- **This is not specific to our wrapper.** The *already-integrated*
  `bench/artifacts/sddmm/fused3s` extension (`F3S.so`) requires the identical
  `CXXABI_1.3.15` symbol (checked with `objdump -T`) and **also fails to
  import** if `import torch` happens first in the process — it only "works"
  in isolation because its own `RUNPATH` (torch's `lib/`, which has no
  `libstdc++.so.6` of its own) happens to let it be the first thing in the
  process to resolve `libstdc++.so.6` against the system's good copy, in the
  single-adapter smoke-test scenarios it's been run in so far. That ordering
  is incidental, not a real fix — flagging it here since it's a shared risk
  for every compiled-CUDA-extension adapter in this repo, not an
  FlashAttention-T-specific defect.
- Confirmed fix: `LD_PRELOAD=/usr/lib64/libstdc++.so.6` set **in the shell
  environment before the python process starts** (an `LD_PRELOAD` set via
  `os.environ` from inside an already-running interpreter is a documented
  no-op — the loader has already resolved everything by then, per the
  fused3s postmortem's own note on why its `RTLD_GLOBAL` dlopen workaround
  segfaulted). With `LD_PRELOAD` set at process start, only ONE copy of
  libstdc++ (the good, system one) ever gets mapped — no ABI clash, no
  segfault (verified: `torch.cuda.is_available()` still returns `True`, and
  `wrapper.cu`'s compiled extension imports cleanly with `fat_fwd` present).

**Environment requirement, like PAT's `CUTLASS_ROOT`**: every invocation of
this adapter (gate checks, and any future timed runs) must be prefixed with
`LD_PRELOAD=/usr/lib64/libstdc++.so.6`. This is a process-startup /
environment fix, not a kernel-code or harness-code patch.

## Shape coverage

Only `head_dim in {64, 128}` was compiled (the two head dims the paper's own
Figure 8 benchmark uses, and the `CtaM/CtaN` tile choice is head_dim-specific
in the artifact's own benchmark source). `PRECISIONS = ["fp16"]` — a `bf16`
template path exists in `wrapper.cu` (`cute::bfloat16_t`) but was not
exercised/gated in the time available; trimmed from `PRECISIONS` rather than
claiming untested coverage. `prepare()` raises `NotImplementedError` cleanly
for any other `d` (including `d=16`, every `kernelbench.domains.ml.
smoke_workloads(kernel="attention-kernel")` synthetic shape) and for `bf16`.

## Gate verification

The literal command
```
$PY -m kernelbench.runner --kernel attention-kernel --variant attn-prefill-kernel-fp16bf16 \
    --impl flashattention-t-fwd --smoke --warmup 1 --reps 3
```
cannot be used as-is: `--smoke` forces **all 4** of
`ml.smoke_workloads(kernel="attention-kernel")` regardless of `--variant`
(prefill-causal, prefill-bidir, prefill-gqa, decode-gqa — all at `d=16`), and
`runner.py`'s per-run loop has no try/except, so the first unsupported shape
(`d=16`, outside `{64,128}`) crashes the whole invocation before a result
file is written. Gated instead with a small standalone script exercising ONE
real, supported shape via `kernelbench.harness.run_variant` directly (this
artifact's home variant, `attn-prefill-kernel-fp16bf16`):

```python
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")
from kernelbench import domains, harness, spec
from kernelbench.artifact_registry import discover

domain = domains.load("attention-kernel")
sp = spec.load("attention-kernel")
variant = sp.variant("attn-prefill-kernel-fp16bf16")
impls = discover("attention-kernel")
factory = impls["flashattention-t-fwd"]["factory"]
impl = factory("fp16")

from kernelbench.domains.ml import AttentionWorkload
w = AttentionWorkload(name="fat-gate-tiny", variant="attn-prefill-kernel-fp16bf16",
                       variant_kind="prefill", B=2, H=4, Hkv=4, d=128, Sq=128, Sk=128, mask="causal")

params = {"seed": 42, "precision": "fp16"}
r = harness.run_variant(impl, w, variant, params,
                         reference=domain.REFERENCES["attention-kernel"],
                         correctness_mode=domain.CORRECTNESS_MODE["attention-kernel"],
                         warmup_override=1, reps_override=3)
print(r.to_dict())
```
Run as:
```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY gate_fat.py
```

Result: **valid: true**. `max_scaled_err = 3.009e-03 <= tol 1e-2`
(secondary views also recorded: `max_pointwise_rel_err=48.2` — expected,
softmax-cancellation-sensitive elements near zero, exactly why the harness's
default gate is the scaled-not-pointwise metric, see `harness.py`'s
`check_correctness` docstring; `l2_rel_err=8.15e-04`). `0.057 ms` median,
`588 GFLOP/s` — reduced-protocol numbers only (`warmup=1, reps=3`, shared
login-node GPU, high-variance warning triggered), explicitly non-conforming
per ARTIFACT_GUIDE rule 5; not a timing result. `--list`/`discover_status`
confirms `flashattention-t-fwd` is discovered as `available: True` for the
`attention-kernel` track (with `LD_PRELOAD` set).

## Not done

- No bf16 coverage (template path exists, untested — see Shape coverage).
- No GQA/bidirectional/decode coverage — `custom_mha_fwd_*` as wired here is
  MHA prefill only (`Hkv==H` in every workload actually gated); the
  artifact's custom API may support more, not explored within budget.
- No sweep across the spec's `recommended_subset` shapes — a single gate
  check only, per the task's login-node budget.

## Environment drift, 2026-09-04 (harness finding, not an artifact bug)

- The login environment's default module changed from `cudatoolkit/12.9` to
  `cudatoolkit/13.2` (HPC SDK 26.5) between integration batches. `CUDA_HOME`,
  `NVHPC_CUDA_HOME`, the `nvcc` on PATH **and `CPATH`** now point at 13.2,
  while plexus_env's torch is still `2.8.0+cu128`.
- Consequence for this adapter: its old `available()` called
  `torch.utils.cpp_extension.load(build_directory=_build)`, relying on ninja's
  no-op check to be cheap. With `CUDA_HOME` = 13.2, `load()` regenerated
  `build.ninja` with the 13.2 nvcc, judged the cache stale, deleted
  `wrapper.cuda.o` and started a ~10-minute recompile from inside a plain
  registry scan (`discover_status`). Killed by hand; stale `_build/lock`
  removed.
- A retry with `CUDA_HOME`/PATH pinned to 12.9 (but the module's `CPATH`
  still set) FAILED at compile time:
  `cutlass/cuda_host_adapter.hpp(150): error: identifier
  "PFN_cuTensorMapEncodeTiled" is undefined` — nvcc 12.9 picked up
  `cudaTypedefs.h` from `.../26.5/cuda/13.2/include` via `CPATH`; the 13.2
  header only defines the versioned `PFN_cuTensorMapEncodeTiled_v12000`,
  the 12.9 header additionally defines the unversioned alias CUTLASS uses.
  The prebuilt `.so` from 2026-08-07 was not touched by the failed compile.
- Fixes: (1) `adapter.py` now dlopens the prebuilt
  `_build/flashattention_t_custom_fwd.so` directly (importlib
  ExtensionFileLoader) and never compiles; `available()` is a file check plus
  one import (2 s for the whole attention-kernel registry scan, measured).
  Compilation lives only in `build.sh`, which now re-points `CPATH` to the
  12.9 `math_libs`+`cuda` include dirs (a first attempt that merely `unset
  CPATH` failed with `fatal error: cusparse.h: No such file or directory` --
  torch's `ATen/cuda/CUDAContextLight.h` includes cusparse.h, which lives
  only under `.../25.5/math_libs/12.9/include`; the 12.9 module used to
  export exactly that CPATH).
  (2) The runner pins `CUDA_HOME`/PATH to the toolkit matching torch's CUDA
  major and drops foreign-major `CPATH`/`LIBRARY_PATH` entries at startup
  (`kernelbench/env.py::pin_cuda_toolchain`, printed as `[env] toolchain:`).
  (3) ARTIFACT_GUIDE gained rule 8 ("available() must never compile") and the
  pin recipe. `discover_status('attention-kernel')` after the fix:
  `flashattention-t-fwd available=True` with the 2026-08-07 `.so`.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, NVIDIA A100-SXM4-40GB (sm_80, driver 595.71.05), gpu
  partition (`bench/gpu_run.sh`), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge,
  `$KB_CXX`/`$KB_CC`), torch 2.8.0+cu128, Python 3.12.14, no cmake used (JIT
  via `torch.utils.cpp_extension.load()`, ninja backend, `_build/`). Arch
  flag: `-gencode=arch=compute_80,code=sm_80` (unchanged).
- Build: OK. Build-system changes: none — `build.sh` was already
  machine-neutral (reads `$CXX`/`$CC`/`$CUDA_HOME` via env.sh, no
  Perlmutter-specific path baked in). The Perlmutter-era "second libstdc++
  ABI trap" (this artifact's `.so` has no RPATH, needs
  `LD_PRELOAD=<good libstdc++>` set before the python process starts) did
  not need a manual workaround here: `bench/env.sh`'s `KB_LD_PRELOAD`
  default already points at `$KB_CUDA_HOME/lib/libstdc++.so.6` (gcc 13,
  has `CXXABI_1.3.15`) and is exported automatically by every
  `gpu_run.sh` invocation — the gate ran clean with no extra prefix.
- Gate: `attn-prefill-kernel-fp16bf16`, `flashattention-t-fwd`, fp16,
  `B=2,H=4,Hkv=4,d=128,Sq=Sk=128,causal` (same reduced-protocol standalone
  script as recorded, `warmup=1, reps=3`): **PASS**,
  `max_scaled_err=3.009e-03 <= tol 1e-2` (`max_pointwise_rel_err=48.2`,
  `l2_rel_err=8.15e-04` — identical to the recorded run to the last printed
  digit).
- Deviation from the recorded ruling: none.
- Verdict: BUILT+GATED — same as recorded ruling.
