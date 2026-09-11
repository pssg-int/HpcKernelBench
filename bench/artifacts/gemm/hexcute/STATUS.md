# hexcute (Hexcute) — gemm

**Status: BUILT+GATED** (as of 2026-09-05: `spec.py` now carries a
per-precision tolerance table and the harness gates fp16 runs at the spec's
own fp16 bound (`1e-2`), not the fp64 bound — **3/3 smoke shapes now pass**,
err 1.08e-04..2.31e-04 vs tol `1e-2`. Originally reported as gate-blocked
against a stale fp64-calibrated `1e-6` bound; see "Dependency isolation
(2026-09-05)" below for the re-gate and for an unrelated shared-venv bug
that was fixed in the same pass — see "Gate result" for the original
finding this superseded).

- Paper: "Hexcute: A Compiler Framework for Automating Layout Synthesis in
  GPU Programs", CGO'26. `PAPER_KEY = conf/cgo/ZhangDSHSP26`.
- Artifact repo `hexcute/hexcute-bench` is *just benchmark scripts* (its own
  README states this literally) — not cloned. The real kernel-generating
  compiler is its pinned submodule `hexcute/hidet` (a fork of
  `hidet-org/hidet`), cloned directly as `source/`.
- Commit cloned/checked out: `d817e28f3efe4732a57eaead724ee3b45995edfc`
  (2025-12-10, the exact commit `hexcute-bench`'s `.gitmodules` pins),
  `git clone` (full clone, not shallow — repo is small, 15 MB) +
  `git checkout d817e28f...`.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`), host compiler
  for JIT-compiled CUDA kernels `g++-12` (SUSE 12.3.0) via `bin/nvcc` (see
  below), also used for the CMake build of `libhidet.so`/`libhidet_runtime.so`
  (kept at gcc-12/g++-12 for consistency with the nvcc host-compiler pin,
  though that step compiles clean under gcc-14 too — see "Dependency
  isolation (2026-09-05)"). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.
  **As of 2026-09-05, `hidet` is no longer `pip install`-ed anywhere** (not
  editable, not otherwise) — `build.sh` runs its CMake build directly and
  `adapter.py` imports it straight from `source/python/` (see "Dependency
  isolation" below for why and how); version string `0.0.0`, synthesized by
  build.sh into a self-authored `hidet-0.0.0.dist-info` next to the package
  (see that section) rather than left to setuptools-scm.

## What the artifact actually is

Hexcute's own contribution is a layout-synthesis compiler pass wired into
hidet's matmul op-resolution: `python/hidet/graph/ops/matmul/resolve.py`'s
`resolve_f16` dispatches to `matmul_f16_cute_experimental` (Hexcute's own
codegen, `python/hidet/graph/ops/matmul/matmul_f16_cute_experimental.py`)
when `hidet.option.hexcute_matmul(strategy='enable')` is set, vs. a
different, non-Hexcute lowering when `'disable'`.
`examples/cute/benchmark_matmul_hidet_cublas.py` (present at the pinned
commit, confirms the recon in the task brief) shows the exact call sequence
this adapter reproduces: `hidet.trace_from` → `hidet.graph.optimize` →
`graph.get_compiled_task(0)` → `matmul_task.pick_best_candidate(...)` → the
resulting `kernel(a, b, out)` callable **is** the one Hexcute-layout-
synthesized GEMM kernel launch — not cuBLAS/cuDNN/Triton (those are the
paper's own *baselines* in that same file, never wrapped here).

## Build (rule 3: minimal patches, all recorded; no kernel/compiler-logic code touched)

1. **`bin/nvcc` PATH shim** (new file, outside `source/`): hidet's own JIT
   CUDA backend (`python/hidet/backend/build.py`'s `NVCC` class) resolves
   `nvcc` via `shutil.which()` and invokes it with **no host-compiler
   override knob anywhere in its Python build path** — no `-ccbin`, no
   `CXX`/`CUDAHOSTCXX` env read, nothing (confirmed by reading the full
   `NVCC.compile()` command-list construction). nvcc 12.9's default host
   compiler on this machine (`g++-14`) fails compiling
   `<bits/alloc_traits.h>` (`identifier "__has_construct" is undefined`,
   57 errors) — the *exact* issue already diagnosed and fixed the same way
   in `spmm/inferfast/build.sh`. Since hidet exposes no override mechanism
   at all (unlike InferFast's Makefile, which already had a `HOST_COMPILER`
   variable to pass through), the only way to fix this without touching
   hidet's Python is to intercept `nvcc` itself: `bin/nvcc` is a 4-line
   passthrough script (`exec real_nvcc -ccbin g++-12 "$@"`), and
   `build.sh`/`adapter.py` put `bin/` first on `PATH` so
   `shutil.which('nvcc')` resolves it. Zero lines of `source/` changed —
   this is a build-system/host-compiler-selection fix (ARTIFACT_GUIDE rule 3
   explicitly names this class of patch as fine), not a change to Hexcute's
   compiler or kernel logic.
2. **`pip install --no-build-isolation -e source`** with `CC=gcc-12
   CXX=g++-12`: builds `libhidet_runtime.so` (a handful of small C++ files,
   `src/hidet/runtime/*.cpp` — host-side runtime glue only, no CUDA
   compilation happens at this step; Hexcute's actual GEMM kernels are
   JIT-compiled later, at `graph.optimize()` time, via `bin/nvcc`). This
   step is fast (~15s) and needed no source patch — `setup.py`'s
   `CustomBuildCommand` already shells out to plain `cmake .. && make -j8`
   with no compiler pinned, and that step compiles clean even under
   `g++-14`; `CC`/`CXX` are passed anyway for consistency, not because it
   was required. A pre-existing, *unrelated* PyPI `hidet==0.6.1` (no
   `hexcute_matmul` option — plain upstream `hidet-org/hidet`) was already
   present in the venv and was `pip uninstall`ed first so `import hidet`
   resolves to this fork.

Both steps are idempotent; `build.sh` re-runs cleanly and exits 0
(`hidet.option.get_hexcute_matmul()` reads back `'enable'` as the verification
step's final check).

**Superseded 2026-09-05**: step 2's `pip uninstall`/`pip install -e` into the
shared venv is exactly what caused a real collision with `convolution/hidet`
(both `pip install` a package named `hidet` into the same `site-packages`) —
see "Dependency isolation (2026-09-05)" below for the incident and the fix
(a direct CMake build, no `pip install` of any kind). The description above
is kept as an accurate record of what step 2 USED to do.

## adapter.py

- `KERNEL = "gemm"`, `IMPL_NAME = "hexcute-gemm"`,
  **`PRECISIONS = ["fp16"]`** — `resolve_f16` only fires when *both*
  operands are 16-bit float (`Tensor.dtype.is_any_float16()`, which covers
  fp16 **and** bf16 — verified by reading `python/hidet/ir/type.py`) *and*
  the contraction dim `K` and free dim `N` are both even
  (`a.shape[-1] % 2 == b.shape[-1] % 2 == 0` in `resolve.py`); there is no
  fp32/fp64 path at all — falling outside those conditions silently routes
  to a *different*, non-Hexcute resolver in hidet's `resolve()` chain, which
  would no longer be exercising this paper's own kernel. Only fp16 was
  actually exercised (this module's own standalone smoke test plus the gate
  run below); bf16 is *plausible* per the dtype check but was **not** run,
  so it is not claimed — ARTIFACT_GUIDE: "what the artifact actually
  supports," not what looks supported on paper. All three of `dense.py`'s
  `_smoke_gemm()` shapes (`256×256×256`; `384×256×512`; `64×64×64`×batch-4)
  have even K and N, so the Hexcute path genuinely fires for the gate check
  below — confirmed directly in the runner output
  (`Compiling cuda task matmul_f16_pk_cute_transpose_b_False(...)` for each
  shape, i.e. Hexcute's own experimental codegen, not the stable fallback).
  Shapes with odd K/N elsewhere in the gemm spec's full sweep (e.g. the
  `2049`, `4097` square sizes in `_gemm_square_shapes()`) would **not**
  trigger `resolve_f16` as currently wired — flagged as a known limitation
  for any future full-spec (non-smoke) run of this adapter, not hidden.
- `prepare()` (rule 2, timed as preprocessing): generates `A`/`B` with
  **exactly** `dense.py`'s `_rng_operand` recipe (`np.random.default_rng`,
  `uniform(-1,1)`, same `seed`/`seed+1_000_003` split for batch elements) —
  not torch's `Generator`, per the RNG-mismatch gotcha already documented in
  `spmm/insum`/`spmm/inferfast`'s adapters — then reproduces
  `benchmark_matmul_hidet_cublas.py`'s own compile sequence
  (`trace_from` → `graph.optimize` → `get_compiled_task(0)` →
  `pick_best_candidate`) inside a `hidet.option.context()` with
  `hexcute_matmul(strategy='enable')`. **Autotuning depth**: uses
  `hidet.option.search_space(0)` (a *single* schedule, ~seconds per shape)
  by default, not the paper's own `search_space(2)` ("arbitrary large" per
  hidet's own docstring, no time bound) — a deliberate, documented deviation
  to keep `prepare()` bounded on a **shared login node** per this task's own
  budget; the search level is `prepare()`-time only (doesn't change what
  kernel *code* runs, only which of hidet's autotuned schedule variants gets
  picked) and is overridable via `HEXCUTE_SEARCH_SPACE=1|2` for a real,
  dedicated-allocation timed run later — still correctly timed as
  preprocessing either way (rule 2), just slower to prepare.
- `run()`: exactly one `kernel(a, b, out)` call — the compiled candidate
  from `prepare()`, nothing else (rule 1). No accumulation semantics to
  reset between reps (GEMM overwrites `out`, unlike e.g. Insum's `+=`).
- `to_host()`: hidet's own `[L, M, N]` batch-leading layout matches
  `reference_gemm`'s `np.stack(...)` layout directly; squeezed back to
  `(M, N)` for `L == 1` to match the non-batched reference shape.
- `timer()` reuses `kernelbench.impls.gpu_cuda.CudaEventTimer`.

## Gate result (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
$PY -m kernelbench.runner --kernel gemm --variant gemm-square-kernel \
    --impl hexcute-gemm --smoke --warmup 1 --reps 3 --precision fp16
```

(`--precision fp16` is required: the default precision resolved from the
`gemm-square-kernel` variant/spec text is fp64 per `dense.py`'s own
`DEFAULT_PRECISION`, which this fp16-only kernel cannot run — same situation
`spmm/inferfast`'s adapter documented for the analogous reason.)

**Result: 0/3 smoke shapes valid.**

| shape | max_scaled_err | tolerance |
|---|---|---|
| smoke-gemm-square-256 (256×256×256) | `1.211e-04` | `1e-06` |
| smoke-gemm-irregular-384x256x512 | `1.080e-04` | `1e-06` |
| smoke-gemm-batched-b4-64 (batch=4, 64×64×64) | `2.306e-04` | `1e-06` |

The gemm spec's tolerance (verified directly via
`spec.load('gemm').variant('gemm-square-kernel').tolerance == 1e-6`, and
identically for `gemm-small-irregular-kernel`) is the **fp64** entry of the
spec's own precision-keyed table (`dense.py` module docstring point 3 — the
tolerance parser only extracts the first number in the prose, which for
every dense-track variant happens to be the fp64 bound). Hexcute's kernel
here is fp16 throughout (`acc_dtype=hidet.float32` internally, per the
compile log — mixed-precision accumulation, output rounded to fp16), so the
observed errors (`1.1e-4`–`2.3e-4`) are the *expected* magnitude for fp16
output rounding, not a correctness defect: they sit within roughly half of
one fp16 unit-roundoff (`2^-11 ≈ 4.9e-4`) of zero, i.e. essentially exact
fp16-rounded output, and are ~2 orders of magnitude below fp32's own would-be
bound (`1e-3`, the spec's fp32 entry) while still ~100x above the fp64 bound
this particular tolerance number represents. This is architecturally
identical to `spmm/inferfast`'s fp16-vs-fp32-tolerance finding: the gemm
spec would need a dedicated fp16-tolerance variant (mirroring how e.g. the
spmm spec's own `-gpu-quantized-int` variant already carves out a
structural, no-numeric-tolerance bucket for lower-precision kernels) for
this class of Tensor-Core `fp16` GEMM to be gated fairly.

**Per ARTIFACT_GUIDE.md rule 4 ("if it fails the gate, that IS a result —
record it, do not loosen the gate to make it pass"): no tolerance override
was applied.** `hexcute-gemm` is correctly wired, genuinely exercises
Hexcute's own `resolve_f16`/`matmul_f16_cute_experimental` codegen path
(confirmed via the runner's own compile-log output naming that function),
and produces numerically sane fp16 output — it cannot pass a fp64-calibrated
`1e-6` bound by construction.

## Not done

- No sweep across the gemm spec's full shape list (square 128..6144,
  small-irregular, LLM shapes, batched) — out of scope per the task's
  login-node budget; only the 3 smoke shapes were run.
- `HEXCUTE_SEARCH_SPACE` left at its default (`0`, single schedule) for the
  gate check — a real timed run should raise this (see adapter.py's
  docstring) at the cost of much longer `prepare()`.
- bf16 not exercised (see adapter.py's PRECISIONS discussion above).
- Odd-K/N shapes (where `resolve_f16` would not fire and the adapter would
  silently stop exercising Hexcute's own kernel) are not special-cased or
  guarded against — flagged as a limitation, not fixed, given the budget.

## Dependency isolation (2026-09-05)

**The collision.** This adapter's build used to `pip uninstall -y hidet`
then `pip install --no-build-isolation -e source` into the shared
`plexus_env` venv (see "Build" above for exactly what that did). Separately,
`bench/artifacts/convolution/hidet/build.sh` does `pip install hidet` into
the SAME venv — a plain, unrelated PyPI package that also happens to declare
`name = "hidet"`. Both installs write into the one shared `site-packages/
hidet`; whichever ran last won process-wide for every subsequent `import
hidet` in that venv, and this adapter's `available()` could not tell which
one it got: its old check was `hasattr(hidet.option, "hexcute_matmul")`,
and upstream hidet 0.6.1 turns out to ALSO define `hexcute_matmul`/
`get_hexcute_matmul` near-verbatim in its `option.py` (confirmed by diffing
the shared venv's installed `option.py` against this fork's — same function
bodies, same docstrings) — a stub with no real codegen behind it. On
2026-09-05, after `convolution/hidet`'s build ran, `import hidet` in the
venv resolved to upstream 0.6.1: `available()` still returned `(True, "")`,
but `hidet.graph.FlowGraph` had no `get_compiled_task` method (only this
fork's `FlowGraph` does), so `prepare()` crashed with `AttributeError:
'FlowGraph' object has no attribute 'get_compiled_task'` — a silently wrong
`available()` result, not a build failure.

**The fix.** `build.sh` no longer installs anything anywhere (no `pip
install`/`pip uninstall`, editable or otherwise, into `plexus_env` or
elsewhere). Reading `source/setup.py`'s `CustomBuildCommand` (the class the
old editable install invoked) shows its only real work is: `cmake ..` +
`make -j8` in `source/build/` (building `libhidet.so` and
`libhidet_runtime.so` — small host-side C++ files, no CUDA compiled at this
step), then copying those two `.so` files into `source/python/hidet/lib/`.
`build.sh` now does exactly that directly, with the same `gcc-12`/`g++-12`
host compiler as before (nvcc 12.9 needs it for the JIT-compiled kernels via
`bin/nvcc`; the CMake step itself compiles clean under either gcc-12 or
gcc-14, kept at gcc-12 for consistency, per the task's "keep whatever
compiles"). This works with **zero `site-packages` entry**: `source/python/
hidet/libinfo.py`'s `get_library_search_dirs()` locates `libhidet*.so`
relative to the package's own on-disk location (`./lib`, `../../build/lib`,
...), never via installed-package metadata — confirmed by reading it
directly. One wrinkle: `source/python/hidet/version.py` runs
`importlib.metadata.version("hidet")`, which (unlike `libinfo.py`) DOES
search installed-package `*.dist-info` metadata across `sys.path`; with no
`pip install` anywhere, that lookup would fall through to whatever `hidet`
metadata `plexus_env` happens to hold (today: `convolution/hidet`'s own
`pylibs/`-isolated install is NOT even on this process's `sys.path` by
default, but the venv's own stray `hidet-0.6.1.dist-info` still is) or raise
`PackageNotFoundError` outright on a venv with no such metadata at all,
which would break `import hidet` completely (`__init__.py` does `from
.version import __version__` unconditionally, uncaught). Fixed by having
`build.sh` synthesize a minimal, self-authored `source/python/
hidet-0.0.0.dist-info/METADATA` (just `Name: hidet` / `Version: 0.0.0`) —
found by `importlib.metadata` ahead of anything else once `adapter.py` puts
`source/python` first on `sys.path`, so this fork resolves its own version
with zero dependency on the ambient venv.

`adapter.py`'s `_ensure_env()` now inserts `source/python` at `sys.path[0]`
(in addition to its existing `bin/` PATH-shim insertion) before any `import
hidet`. `available()` was tightened from a purely attribute-based check
(`hasattr(hidet.option, "hexcute_matmul")` — insufficient, see above) to a
path-based one FIRST: `hidet.__file__` must resolve under this artifact's
own `source/python/`, with the actual resolved path printed in the reason
string if not (this is the "verify by printing hidet.__file__" check this
fix was asked to add) — plus the original `hexcute_matmul` check AND a new
`FlowGraph.get_compiled_task` check as defense in depth. Manually verified
the negative path: pre-importing the wrong (upstream) `hidet` into
`sys.modules` before loading this adapter makes `available()` return
`(False, "hidet resolved to '.../plexus_env/.../site-packages/hidet/
__init__.py', not this artifact's own fork under '.../hexcute/source/
python' -- ...")` instead of silently proceeding.

In practice this specific cross-import collision cannot happen inside a
single `kernelbench.runner` process anyway:
`kernelbench/artifact_registry.py::discover(kernel)` only imports
`adapter.py`s under `artifacts/<kernel>/`, and `convolution/hidet` /
`gemm/hexcute` live under different kernels — so a `--kernel gemm` run never
even loads `convolution/hidet`'s adapter module, and vice versa. The
`sys.modules`/`sys.path` collision is real only if something imports both in
one process (e.g. interactive debugging, as done to verify the negative path
above) or if the two artifacts fight over the shared venv's `site-packages`
(the actual incident, via two separate `pip install`s at different times,
not two adapters loaded together) — either way, `available()` now fails
loud with `hidet.__file__` in the reason instead of silently running the
wrong kernel.

**Re-gate.** Rebuilt from scratch (`rm -rf source/build`,
`bash build.sh` — verified idempotent on a second run) and re-ran with the
harness's now-correct per-precision tolerance (`spec.load('gemm').variant(
'gemm-square-kernel').tolerance_for('fp16') == 0.01`, vs. the stale generic
`1e-6` parse the original "Gate result" section above was gated against):

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel gemm \
    --variant gemm-square-kernel --smoke --precision fp16 --impl hexcute-gemm
```

**Result: 3/3 smoke shapes now valid** (identical measured errors to the
original run — nothing about the kernel changed, only the tolerance it is
gated against and the dependency isolation):

| shape | max_scaled_err | tolerance (fp16) | gate |
|---|---|---|---|
| smoke-gemm-square-256 (256×256×256) | `1.211e-04` | `1e-02` | **PASS** |
| smoke-gemm-irregular-384x256x512 | `1.080e-04` | `1e-02` | **PASS** |
| smoke-gemm-batched-b4-64 (batch=4, 64×64×64) | `2.306e-04` | `1e-02` | **PASS** |

Confirmed via a direct check that `hidet.__file__` inside this process is
`.../artifacts/gemm/hexcute/source/python/hidet/__init__.py` (not any
`site-packages` copy), and via `kernelbench.runner --kernel gemm --list`
that the adapter still registers `ok hexcute hexcute-gemm`. `./smoke_all.sh`
re-run after both fixes: **35/35 kernels green**, no regressions.

The "Gate result" section above (0/3 valid vs. a `1e-6` bound) is kept as an
accurate record of the original finding; it is superseded by this section,
not wrong for its time — `spec.py` did not yet carry `tolerance_by_precision`
when it was written.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, 14 SMs, ~5 GB; host GPU
  reports as NVIDIA A100-SXM4-40GB via nvidia-smi under the MIG partition),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge,
  `$CXX`/env.sh default) for the harness/torch; libhidet/libhidet_runtime's
  CMake build used g++/gcc 12.4.0 (conda-forge, `KB_GXX12`/`KB_GCC12`, same
  role as Perlmutter's g++-12 pin -- nvcc's host-compiler check); the
  `bin/nvcc` PATH shim likewise resolves `KB_GXX12` (12.4.0) as `-ccbin`.
  torch 2.8.0+cu128, Python 3.12.14, cmake 4.2.3. Commit unchanged:
  `d817e28f3efe4732a57eaead724ee3b45995edfc`.
- Build: OK. Build-system changes: `build.sh` gained a new step 4/4 that
  `pip install --target=pylibs --no-deps`s hidet's undeclared pure-Python
  runtime deps (tomlkit, click, lark, tabulate, tqdm, nvtx, psutil,
  gitpython, importlib_metadata, `hip-python`, cuda-python/cuda-bindings) --
  zaratan's kb-env venv (unlike Perlmutter's plexus_env) has none of these,
  so `import hidet` failed at `ModuleNotFoundError: No module named
  'tomlkit'` and then (after installing the first batch) at `No module
  named 'hip'` (hidet's `runtime/storage.py` does an unconditional `import
  hidet.hip` -> `from hip import hip`, unrelated to this artifact's
  CUDA-only kernel but not gated behind a try/except upstream; `hip` ships
  on PyPI as `hip-python`, pure ctypes bindings, no ROCm runtime library
  needed just to import it). `pylibs/` already existed pre-populated with
  everything except `hip-python` from an earlier session's manual fix that
  had never been folded into `build.sh` itself; this pass adds the actual
  `pip install` step (idempotent: skipped once `tomlkit`+`hip.hip` import
  cleanly) so a truly fresh checkout builds unattended, and fixes
  `build.sh`'s own internal self-verification `import hidet` check (the
  literal last step of the script) to put `pylibs/` on `PYTHONPATH`/
  `sys.path` too -- it previously only added `source/python`, so the
  self-check itself failed even once `pylibs/` had the right packages.
  `adapter.py` was not touched (its `_ensure_env()` already appended
  `pylibs/` correctly). No kernel/compiler-logic code changed.
- Gate: `--variant gemm-square-kernel --smoke --precision fp16 --impl
  hexcute-gemm`: **PASS** all 3 smoke shapes -- smoke-gemm-square-256 err
  1.21e-04, smoke-gemm-irregular-384x256x512 err 1.08e-04,
  smoke-gemm-batched-b4-64 err 2.31e-04 (tol fp16 `1e-2` for all) -- 3/3
  runs valid, identical error magnitudes to the recorded Perlmutter run.
- Deviation from the recorded ruling: none in outcome (3/3 pass, same error
  magnitudes to 3 sig figs). The only difference is environmental: kb-env's
  venv lacked hidet's transitive pure-Python/hip deps that plexus_env had
  pre-installed, fixed as a build-system change above, not a kernel or gate
  difference.
- Verdict here: BUILT+GATED -- equals the recorded ruling.
