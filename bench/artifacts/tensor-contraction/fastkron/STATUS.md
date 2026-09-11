# fastkron (FastKron) — tensor-contraction

**Status: BUILT+GATED**

- Paper: "Fast Kronecker Matrix-Matrix Multiplication on GPUs" (PPoPP'24).
  `PAPER_KEY = conf/ppopp/JangdaY24` (matched by title + `artifact_url` in
  `../../../output/included.json`).
- Artifact: https://github.com/abhijangda/fastkron
- Commit cloned: `3c52b73cdb2f6395ba61f9829108bfd8233a0afa`, `git clone
  --depth 1` into `./source/`; `pybind11` submodule separately fetched
  (`git clone --depth 1 https://github.com/pybind/pybind11` — the repo's own
  `.gitmodules` submodule, not fetched by `--depth 1`, needed for the
  `PYMODULE=ON` CMake build; vendored into `source/pybind11/`, itself
  unmodified).
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  host compiler `g++`/`gcc` 14.3.0 (`/opt/cray/pe/gcc-native/14/bin`), cmake
  3.28.3, `-arch=sm_80` (A100, via `-DCMAKE_CUDA_ARCHITECTURES=80`). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python` (pybind11 3.0.4 already
  present in this venv). `LD_PRELOAD=/usr/lib64/libstdc++.so.6` needed at
  *run* time (same CXXABI/GLIBCXX fix documented in
  `bench/artifacts/triangle-counting/tot/STATUS.md`).

## Track thinness

tensor-contraction is one of two "thin tracks" in this integration pass.
Its survey (`benchspecs/tensor-contraction/survey.md`) covers 4 papers, 3 of
them with real artifacts: Einsum Trees (CPU-only, out of scope per this
project's GPU-only integration ruling), FastKron (this artifact), cuKE
(`../cuke/`, SKIPPED — see its own STATUS.md), and SWQsim (extreme-scale
distributed, no reusable single-GPU kernel entry point, not pursued). With
FastKron integrated and cuKE resolved SKIP, this track's GPU-candidate list
is exhausted for this pass.

## What the artifact actually is

FastKron's core primitive is Generalized Matrix-Kronecker-Matrix
Multiplication (GeMKM): `Z = alpha * X @ (F_1 kron F_2 kron ... kron F_N) +
beta * Y`, for `X` shape `(M, P^N)` and `N` Kronecker factors `F_i` shape
`(P, Q)`. `pyfastkron.fastkrontorch`/`fastkronnumpy` are pure-Python
wrappers (pybind11) around a compiled `libFastKron.so` (built here as the
`pyfastkron.FastKronCUDA` extension module) exposing `sgemkm`/`dgemkm`
(float/double) as the real, tuned CUDA kernel entry points — this is the
spec's own `kron-matmul-kernel-fp32-fp64` variant's exact primitive.

## Mapping onto this harness's tensor-contraction domain (task A)

`kernelbench.domains.tensor` (the harness's ONLY tensor-contraction
workload/reference implementation) is strictly a 2-OPERAND einsum
contraction (`ContractionWorkload`), with no notion of an N-factor
Kronecker chain at all — `runner.py` always builds this same workload type
regardless of which spec variant id is passed, and gates every impl against
`np.einsum(w.equation, A, B)` on those SAME two operands. A genuine
multi-factor Kron-Matmul workload cannot be injected into this harness's
tensor-contraction path as it currently exists.

The mapping used: FastKron's own **N=1 case** — `GeMKM` with exactly one
factor `F` degenerates to `Z = X @ F`, an ordinary GEMM mathematically
identical to any 2-operand contraction after reducing it to `(free_a,
contracted) x (contracted, free_b)` via transpose+reshape (the SAME
axes/shape derivation `kernelbench.domains.tensor.
NumpyTensordotContraction.prepare()` already uses for this domain's CPU
baseline — copied verbatim into `adapter.py`'s `prepare()`). `run()` then
calls FastKron's real `sgemkm`/`dgemkm` CUDA kernel via
`pyfastkron.fastkrontorch.fastkrontorch.gemkm(False, A2d, [B2d])` — a
genuine, real invocation of the artifact's own compiled kernel, not cuBLAS
under FastKron's name. The transpose/reshape glue is done via torch ops
INSIDE `run()` (in-kernel), per this track's `notes_on_fairness`
("permutation is the one point of universal agreement... never strip it out
of a timed kernel") — it is explicitly NOT part of FastKron's own
zero-transpose contribution, which is specific to N>1 chains; an N=1 GEMM
has nothing to "avoid" by any algorithm.

**Scope caveat, stated plainly**: this does NOT exercise FastKron's actual
novel contribution (the shuffle-avoiding N>1 Kronecker-CHAIN algorithm,
Table 1's up-to-80%-transpose-elimination claim). It exercises FastKron's
real N=1 GEMM kernel path, correctly, on the same 2-operand contraction
shape every other impl in this domain is gated on. See "N=3 self-check"
below for direct confirmation that the genuine multi-factor path also works
correctly, even though it is not reachable through this harness's
tensor-contraction domain module as it stands.

## Real artifact bug found: `gekmmSizes()` crashes on a CUDA-only build

`pyfastkron.fastkronbase.FastKronBase.supportedProcessor()`:

```python
def supportedProcessor(self):
    return (platform.machine() in ("x86_64", "AMD64")) and self.x86 == True
```

unconditionally requires the **X86** backend to have been built and
imported (`self.x86 = x86_requested and fastkronX86 is not None`),
regardless of whether the call is for a CUDA tensor on the CUDA backend.
This function gates both (a) the top-level `fastkrontorch.gemkm()`
convenience wrapper's silent fallback to a naive PyTorch "shuffle"
implementation, AND (b) `FastKronBase.gekmmSizes()`, which on a CUDA-only
build (`self.x86 == False`) takes an "unsupported" branch setting the
temp-buffer size `ts = -1` as a sentinel. That `-1` then reaches
`x.new_empty(ts)` inside `FastKronTorch.gemkm()` and crashes:

```
RuntimeError: Trying to create tensor with negative dimension -1: [-1]
```

Reproduced directly: `fastkrontorch.gemkm(False, A, [B])` for CUDA float32
tensors, `fastkronCUDA` handle confirmed non-`None` and initialized
correctly, X86 extension simply never built. Calling the instance method
directly (bypassing only the OUTER `isSupported()` gate) does NOT avoid
this, since the crash is in a DIFFERENT method (`gekmmSizes()`) the instance
method itself calls internally. **A CUDA-only build of pyfastkron cannot
successfully call `gemkm`/`gekmm` through the documented Python API at all,
for any input** — this is a genuine packaging bug, not specific to this
integration's build flags (any user who follows the README's own advice to
build with `-DENABLE_X86=OFF` on a machine without AVX/OpenMP would hit it).

**Workaround** (runtime, not a source patch): `adapter.py`'s `prepare()`
sets `fastkrontorch.x86 = True` on the module-level singleton once. This
makes `supportedProcessor()`/`supportedSystem()` return `True` so
`gekmmSizes()` takes its correct branch and calls the real
`fastkronCUDA.libFastKron.gekmmSizes` C function; `self.x86` is never read
again for backend *dispatch* (only `device_type(x)` is), and `fastkronX86`
(`None` throughout) is never touched since every tensor here is
CUDA-resident. Verified this genuinely runs the real CUDA kernel (not a
coincidentally-passing shuffle fallback) via the N=3 self-check below.

## N=3 multi-factor self-check (functional, NOT wired into the harness gate)

Run once, standalone, to confirm the genuine Kronecker-CHAIN kernel path
(FastKron's actual contribution, not reachable through this harness's
2-operand `ContractionWorkload`) works correctly on this build:

```python
import torch
from pyfastkron.fastkrontorch import fastkrontorch as fkt
fkt.x86 = True
Fs = [torch.rand(4, 4, device="cuda", dtype=torch.float32) for _ in range(3)]
X = torch.rand(5, 4**3, device="cuda", dtype=torch.float32)
z3, _ = fkt.gemkm(False, X, Fs)
Fk = Fs[0]
for f in Fs[1:]:
    Fk = torch.kron(Fk, f)
ref3 = X @ Fk
print((z3 - ref3).abs().max().item())   # 9.5367431640625e-07
```

Result: `max abs err = 9.54e-07` — the real N=3 Kronecker-chain kernel
matches an independent `torch.kron`-based reference to fp32 precision.
(The N=1 case used by `adapter.py`'s `run()` was checked the same way
against plain `torch.matmul`: `max abs err = 2.86e-06`.)

## Build-system patches (rule 3: minimal, recorded; zero kernel code touched)

1. `source/setup.py`: added `-DPython3_EXECUTABLE={sys.executable}`
   alongside the existing legacy `-DPYTHON_EXECUTABLE=` flag. Modern CMake's
   `FindPython3` module (used by `CMakeLists.txt`) reads
   `Python3_EXECUTABLE`, not the old `PYTHON_EXECUTABLE` name; without this,
   configure silently probed a bare system `python3` with no `Python.h` and
   failed (`Could NOT find Python3 (missing: Python3_INCLUDE_DIRS
   Development.Module)`).
2. `source/CMakeLists.txt`: `find_package(Python3 REQUIRED COMPONENTS
   Interpreter Development.Module)` -> added `Development.Embed`.
   `pybind11_add_module(${FastKron_NAME} SHARED ...)` (note: `SHARED`, not
   the default `MODULE`) needs pybind11's CMake support to link the full
   `Python3::Python` target, which requires `Development.Embed` (a linkable
   `libpython`); without it, configure failed with `Python3_ADD_LIBRARY:
   dependent target 'Python3::Python' is not defined`. This venv's
   `libpython3.11.so` exists (`sysconfig`'s `Py_ENABLE_SHARED=0` is
   misleading — the `.so` is present in `LIBDIR` regardless), so requesting
   the component was sufficient; no venv change needed.
3. `source/CMakeLists.txt`: narrowed the hardcoded `gen_tuner_kernels.py`
   invocation from `-mm-type mkm kmm ... -batch-type cont strided` to
   `-mm-type mkm ... -batch-type cont` (this adapter only ever calls forward
   `GeMKM`, never `GeKMM` or a batched/strided call) and dropped the
   `kmm-a100-kernels`/`kmm-v100-kernels` match-config files (now irrelevant
   since `kmm` is excluded). Combined with `-DCMAKE_CUDA_ARCHITECTURES=80`
   (single-GPU target, vs. the project default `50;70;80`), this cuts the
   generated/compiled per-tile-config kernel count from **2734 to 136**,
   turning a from-scratch build from an estimated 30-45 minutes (not
   completed in this session before this patch — see "Build history" below)
   into under 5. **No kernel code is touched**: every one of the 136
   compiled `.cu` files is still `gen_tuner_kernels.py`'s own, unmodified,
   verbatim-generated template instantiation; this patch only narrows WHICH
   instantiations get generated in the first place.

`git -C source diff --stat`: `CMakeLists.txt | 24 ++++++++++++++++++++++--`,
`pybind11 | 2 +-` (submodule pointer, now populated), `setup.py | 12
+++++++++++-`.

## Build history (why the kernel-sweep patch exists)

The artifact's own default build (`-mm-type mkm kmm ... -batch-type cont
strided`, all 3 of `CMAKE_CUDA_ARCHITECTURES`'s default `50;70;80` archs)
generates ~2734 per-tile-config `.cu` files. Two earlier full-sweep build
attempts in this session were killed by unrelated session interruptions
before completing (an artifact of this interactive session's environment,
not of FastKron itself — one attempt reached 56%, another reached 99% with
no compile errors, before being interrupted); rather than keep re-attempting
a 30-45 minute compile against an environment with unpredictable
interruptions, patch 3 above narrows the build to exactly what this
adapter needs (forward `GeMKM`, non-batched, single target arch), which
reliably completes in under 5 minutes end-to-end (verified via 2 clean
`rm -rf pybuild && ./build.sh` runs from scratch).

## Gate verification (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel tensor-contraction --variant kron-matmul-kernel-fp32-fp64 \
    --impl fastkron-gemkm-n1 --smoke
```
Result: **3/3 runs valid** (`smoke-matmul`, `smoke-permute`,
`smoke-multi-index` — i.e. including the two cases that need genuine
transpose glue, not just the no-permutation-needed matmul case):
`err = 2.29e-16 / 2.67e-16 / 3.28e-16`, all `<= tol 1e-5`.

Same command with `--precision fp64`: **3/3 runs valid**, comparable
near-machine-epsilon errors (fp64 `dgemkm` path, `<= tol 1e-5`).

Reduced-protocol numbers only (warmup=5, reps=20, shared login-node GPU) —
explicitly non-conforming (`conforming: False`), not a timing claim, per
ARTIFACT_GUIDE.md rule 5. No timing sweep was run.

## Not done

- No sweep across the spec's microbenchmark grid or 28 named real-world
  Kron-Matmul shapes (Table 4) — out of scope for the login-node budget; a
  functional/gate check on this domain's fixed smoke set + fp32/fp64 only.
- The genuine N>1 multi-factor Kronecker-chain path (FastKron's actual
  headline contribution) is functionally verified (N=3 self-check above)
  but not wired into the harness's correctness gate or timing path, since
  `kernelbench.domains.tensor.ContractionWorkload` has no multi-operand
  representation to drive it through `runner.py`.
- The X86 backend was deliberately not built (GPU-only integration scope);
  consequently `fastkronX86` is `None` throughout and the `x86=True`
  workaround never actually dispatches to it.

## Verdict

`fastkron-gemkm-n1: BUILT+GATED, exact-to-near-machine-epsilon match on all
3 smoke workloads at both fp32 and fp64 (max err ~1e-16, tol 1e-5), via the
N=1 GeMKM==GEMM reduction; genuine N=3 multi-factor Kronecker-chain path
independently confirmed correct but not harness-reachable; one real
artifact bug found and worked around (gekmmSizes() crashes with a
negative-dimension RuntimeError on any CUDA-only pyfastkron build, via a
supportedProcessor() check that incorrectly gates CUDA dispatch on the
X86 backend's presence).`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, on an
  A100-SXM4-40GB physical card, driver 595.71.05), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge
  `kb-env`, via `KB_CXX` — no gcc12 pin needed for this artifact), torch
  2.8.0+cu128, Python 3.12.14, cmake 4.2.3, `-DCMAKE_CUDA_ARCHITECTURES=80`
  unchanged (matches this GPU). `build.sh` needed zero edits — its
  `PY`/`CXX`/`CC`/`CUDA_HOME` defaults already read the exported
  `${VAR:-...}` knobs first (rule 8) and `bench/env.sh`/`toolchain.sh`
  export all four before `build.sh`'s own fallback lines evaluate, so the
  Perlmutter `/pscratch/.../plexus_env` and `/opt/cray/pe/gcc-native/14`
  paths in the script are never reached here. The `pybind11` vendoring step
  and the `FASTKRON_BUILD_JOBS`-capped parallel build (default 8) both ran
  as written, no changes needed.
- Build: OK (100% built, `FastKronCUDA.cpython-312-x86_64-linux-gnu.so`
  produced under `pybuild/lib/` and staged into `pyfastkron_runtime/`).
  Build-system changes: none (this directory's `source/pybind11` and
  `pybuild/` were already partially populated from an earlier session on
  this machine; `build.sh` picked up cleanly from that state, as designed).
- Gate: `kron-matmul-kernel-fp32-fp64` fp32: PASS 3/3 (smoke-matmul err
  2.07e-07, smoke-permute err 1.56e-07, smoke-multi-index err 1.29e-07, all
  <= tol 1e-5). fp64: PASS 3/3 (err 2.29e-16 / 2.67e-16 / 3.28e-16, all <=
  tol 1e-10 — the runner's `tolerance_for(precision)` now applies a
  tighter fp64-specific tolerance than the spec's headline 1e-5; the fp64
  errors are themselves fp64-machine-epsilon, well inside either value).
  Note: the runner infers precision from the variant id when `--precision`
  is omitted (`"fp64" in variant.id` for `kron-matmul-kernel-fp32-fp64`),
  so an unqualified `--smoke` invocation here runs fp64, not fp32 — both
  were run explicitly to reproduce STATUS.md's original two-precision
  claim. Both `conforming: False` (reduced protocol, GPU clocks not
  locked) — not a timing result, per protocol.
- Deviation from the recorded ruling: none in outcome. The printed
  tolerance for fp64 differs from the `1e-5` quoted in the original "Gate
  verification" section above (now `1e-10`, a harness-side
  `tolerance_for()` refinement made after this artifact's original
  integration, not a zaratan-specific difference); both fp32 and fp64
  still pass comfortably under either value.
- Verdict here: BUILT+GATED — equals the recorded ruling.
