# hidet (Hidet) — convolution

**Status: BUILT+GATED** (2/3 smoke shapes pass; the 3rd fails the gate for a
real, documented precision reason — see "Gate result" below, not a broken
adapter).

- Paper: "Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor
  Programs", ASPLOS'23. `PAPER_KEY = conf/asplos/DingYZLWP23`.
- Artifact repo: https://github.com/hidet-org/hidet (archived/read-only on
  GitHub, but still `pip install`-able from PyPI).
- Installed: `pip install hidet` into
  `/pscratch/sd/c/cunyang/gnn/plexus_env`, wheel `hidet-0.6.1-py3-none-any.whl`
  (pinned; see "Shared-venv collision" below for why the version had to be
  force-reinstalled and re-verified before the gate ran).
- Toolchain: nvcc `12.9.41`
  (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`), host compiler
  `gcc-native/12` (g++ 12.3.0, `/opt/cray/pe/gcc-native/12/bin` — see "Build
  fix" below), python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`,
  torch `2.8.0+cu128`. GPU: NVIDIA A100-PCIE-40GB, compute capability 8.0
  (`sm_80`), driver 580.159.04.

## `source/` is intentionally absent

Hidet is a proper pip package; the archived GitHub repo's source was never
needed — `hidet.ops`, `hidet.graph`, etc. are all available directly from
the installed wheel. No `git clone` was performed.

## What entry point was used, and why

Explored the installed package's actual module tree first (per the task's
own suggested probe):
`[x for x in dir(hidet.ops) if 'conv' in x.lower()]` ->
`['conv1d', 'conv1d_gemm', 'conv1d_transpose', 'conv2d', 'conv2d_channel_last',
'conv2d_gemm', 'conv2d_gemm_fp16', 'conv2d_gemm_fp16_channel_last',
'conv2d_gemm_image_transform', 'conv2d_transpose', 'conv2d_transpose_gemm',
'conv2d_winograd', 'conv3d', 'conv3d_gemm', 'conv3d_transpose', 'conv_pad']`.

`hidet.ops.conv2d(data, weight, stride=(1,1), dilations=(1,1), groups=1,
padding=(0,0))` is a direct, well-supported op-level API — no need for the
`torch.compile(backend='hidet')` escape hatch. `adapter.py`'s `prepare()`
builds a one-op `FlowGraph` directly:

```python
x_sym = hidet.symbol(list(X.shape), dtype="float32", device="cuda")
w_sym = hidet.symbol(list(W.shape), dtype="float32", device="cuda")
y_sym = hidet.ops.conv2d(x_sym, w_sym, stride=(w.stride, w.stride),
                         padding=(w.pad_h, w.pad_w), groups=w.groups)
graph = hidet.graph.optimize(hidet.trace_from(y_sym, inputs=[x_sym, w_sym]))
```

This is the tightest wrap of the actual kernel available (ARTIFACT_GUIDE.md
rule 1) — a single conv2d op, traced and graph-optimized by hidet's own
compiler, nothing else in the graph.

## `prepare()` / JIT-compile placement

Hidet JIT-compiles its generated CUDA source lazily, on the FIRST invocation
of a given (task, shape, schedule) — confirmed empirically: calling the
optimized graph the first time triggers an nvcc build (`Compiling cuda task
...` log line), taking seconds; every subsequent call on the same graph
object is sub-millisecond, served from hidet's in-process + on-disk
compiled-kernel cache (`hidet.option.get_cache_dir()`, default
`/pscratch/sd/c/cunyang/.hidet_cache`). `prepare()` calls the graph once
after building/optimizing it, specifically to force this JIT-compile inside
the harness's timed *preprocessing* window, never inside the first measured
`run()` call. Measured `preprocessing_ms` from the actual gate run:
`10216 ms` (smoke-conv-3x3, cold cache), `4026 ms` (smoke-conv-1x1, a
different shape/task so still a cold compile), `2361 ms`
(smoke-conv-depthwise-s2, different task again — the direct-conv2d path, not
the GEMM path, cheaper to compile). These are real one-time compile costs,
not artifacts of a badly-placed timer; the task's own instructions call this
out explicitly as expected DL-compiler behavior, "duration should just be
documented, not hidden."

## Build fix (ARTIFACT_GUIDE.md rule 3 — environment fix, not a kernel patch)

This host's default `gcc`/`g++` on `PATH` is `gcc-native/14` (SUSE gcc
14.3.0). nvcc 12.9 cannot use it as a host compiler: compiling any hidet
CUDA task failed with ~57 parse errors inside `<bits/alloc_traits.h>` /
`<bits/hashtable.h>` (`identifier "__has_construct" is undefined`, etc.) —
libstdc++ internals from a gcc newer than nvcc 12.9's supported host-compiler
range. Hidet resolves nvcc's host compiler implicitly via whatever
`g++`/`gcc` `shutil.which` finds on `PATH` (installed package's
`hidet/backend/build.py`, `NVCC._resolve_nvcc_path` / compile-command
construction) — there is no hidet config knob to override this, so
`adapter.py` prepends `/opt/cray/pe/gcc-native/12/bin` (g++ 12.3.0, a
version nvcc 12.9 supports) to `os.environ["PATH"]` at import time and again
at the top of `prepare()`. No hidet or CUDA source touched — purely a host
PATH fix, same class of fix as an arch flag or an include path (rule 3).

## Shared-venv name collision (found and fixed while building this adapter)

`plexus_env` is shared across concurrently-running artifact-integration
agents. While this adapter was being built, a sibling task
(`bench/artifacts/gemm/hexcute/`, worked on in parallel — confirmed by its
directory mtime, `21:03` same day, no `STATUS.md` yet at the time) ran
`pip install -e` on its own vendored/forked compiler, whose package metadata
ALSO declares `name = "hidet"`. Whichever install lands last in the shared
`site-packages` wins process-wide for every subsequent `import hidet` in the
venv — plain `import hidet` cannot tell which one it got. Observed directly:
after this adapter's own `pip install hidet` completed (log said
`Successfully installed hidet-0.6.1`), a follow-up check showed
`hidet.__version__ == '0.0.0'` and `hidet-0.0.0.dist-info/direct_url.json`
pointing at `file:///.../bench/artifacts/gemm/hexcute/source` — the
sibling's editable install had clobbered the real PyPI package in the window
between my install finishing and my next check.

Fixed by `pip install --force-reinstall --no-deps hidet==0.6.1` immediately
before running the gate (verified `hidet.__version__ == '0.6.1'` and
`hidet.__file__` under real `site-packages`, not `artifacts/...`, right
before invoking the runner). `adapter.py`'s `available()` now also checks
`hidet.__file__` doesn't resolve into any `artifacts/` tree and that
`hidet.ops.conv2d` exists, so if this collision recurs on a later run, the
adapter reports a clear `available() -> (False, reason)` instead of silently
running against the wrong package. This is a real environment hazard worth
flagging for Phase 2 generally: distinct artifact integrations that
`pip install` same-named packages into one shared venv are not isolated
from each other.

## `to_host()` / `timer()`

- `to_host()`: `out.torch().detach().to("cpu", dtype=torch.float64).numpy()`
  — `hidet.Tensor.torch()` shares memory with the hidet tensor (no copy),
  the interop method the task's own instructions predicted would exist.
- `timer()`: reuses `kernelbench.impls.gpu_cuda.CudaEventTimer` directly, no
  reimplementation. Confirmed applicable: `hidet.option.is_use_torch_stream()`
  is `True` by default in 0.6.1 (hidet executes on torch's current CUDA
  stream unless told otherwise), so `torch.cuda.Event` correctly brackets
  hidet's kernel launches — same CUDA context/stream. Verified empirically
  before wiring it into the adapter: 5 timed calls through a compiled graph
  via this exact timer settled at a stable ~0.18 ms after one slightly
  elevated first sample — ordinary device-event timing behavior, not a
  stream-mismatch artifact (which would read as ~0 or wildly inconsistent).

## Gate result (login node, functional/gate check only — no timing sweep)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
$PY -m kernelbench.runner --kernel convolution --variant conv-dense-kernel-fp32 \
    --impl hidet-conv2d --smoke --warmup 1 --reps 3
```

Result: **2/3 runs valid** (tolerance `1e-3`, `max_scaled_err`) ->
`results/convolution_conv-dense-kernel-fp32_1786161967.json`.

| smoke shape | groups | algorithm path (from hidet's own compile log) | max_scaled_err | gate |
|---|---|---|---|---|
| smoke-conv-3x3 | 1 | im2col-GEMM (`conv2d_gemm_image_transform` + `batch_matmul`) | 5.456e-04 | **PASS** |
| smoke-conv-1x1 | 1 | im2col-GEMM (`conv2d_gemm_image_transform` + `batch_matmul`) | 1.007e-03 | **FAIL** (tol 1e-3) |
| smoke-conv-depthwise-s2 | 6 | direct `conv2d` kernel (no GEMM) | 1.266e-07 | **PASS** |

Timing (informational only — reduced protocol, shared login-node GPU, not
spec-conforming, `conforming: false` in the result JSON):
smoke-conv-3x3 median `0.179 ms` / `0.46 GFLOP/s`; smoke-conv-depthwise-s2
median `0.145 ms` / `0.074 GFLOP/s`. smoke-conv-1x1 has no timing (harness
correctness gate runs before timing; a failing gate means "no timing
reported," per `harness.run_variant`).

### Why smoke-conv-1x1 fails — a real precision finding, not an adapter bug

Read from hidet's own compile-log task names: dense (`groups=1`) convs lower
through `conv2d_gemm_image_transform` (im2col) into a tensor-core
`batch_matmul`. `hidet/graph/ops/matmul/cuda_batch_matmul.py`'s
`resolve_mma_type` maps ANY input dtype pair that isn't fp16/bf16 —
including genuine `float32` — to `'mma_tf32_f32'`: Ampere TF32 tensor-core
MMA (~10-bit mantissa), never full-precision FP32 CUDA-core
multiply-accumulate, for the default (non-cublas, non-autotuned) schedule
this adapter's `graph()` call selects. There is no flag at the
`hidet.ops.conv2d` level to opt out of this; it is baked into the matmul
schedule's dtype-rank resolution, so no kernel-code patch was attempted to
route around it (rule 3). Depthwise/grouped convs take hidet's separate
direct-`conv2d` code path (no GEMM at all) and are essentially unaffected —
measured error there is ~4 orders of magnitude smaller. This means: on this
hardware, with hidet's DEFAULT schedule, dense fp32 conv layers with a small
reduction dimension (`Cin*Kh*Kw`, here `8*1*1=8` for the 1x1 case) sit right
at or just over the spec's `1e-3` tolerance, while layers with a larger
reduction (the 3x3 case, `Cin*Kh*Kw=36`) or the non-GEMM grouped path stay
comfortably under it. Per rule 4, this is reported as-is; the gate was not
loosened, and no attempt was made to force hidet onto a non-TF32 schedule by
patching its kernel/schedule code.

## Not done

- No timing sweep across the full 30-shape `layer_shape_suite` — out of
  scope per the task's login-node budget; a single smoke gate check only.
- No investigation into whether hidet's autotuning search space
  (`hidet.option.search_space`, left at its default here) would pick a
  `use_cublas=True` or otherwise more accurate schedule for the 1x1 case —
  would need a real compute-node allocation to run a search budget large
  enough to matter; out of scope for a login-node gate check.

## Dependency isolation (2026-09-05)

**The collision.** `plexus_env` is shared with `bench/artifacts/gemm/hexcute/`,
whose own build (at the time) ran `pip install --no-build-isolation -e
source` on its fork of hidet — a package that ALSO declares `name = "hidet"`
in its metadata. `pip install`/`pip install -e` both write into the SAME
shared `site-packages/hidet`; whichever install ran last won process-wide for
every later `import hidet`. On 2026-09-05, `import hidet` in the venv
resolved to the hexcute fork content (masquerading under a stale
`hidet-0.6.1.dist-info` left by an earlier plain `pip install hidet` here),
lacking `hidet.ops.conv2d`'s expected upstream behavior in general — the
converse of what broke `gemm/hexcute` the same day (its `available()` saw
upstream 0.6.1 instead of the fork; see that artifact's own STATUS.md). Two
distinct artifact integrations `pip install`-ing same-named packages into one
shared venv are not isolated from each other — flagged as a real environment
hazard back when this adapter was first built (see "Shared-venv name
collision" above); this is that hazard actually recurring, now fixed for
real instead of merely detected-and-hoped-not-to-recur.

**The fix.** `build.sh` no longer touches `plexus_env` at all — no `pip
install hidet` into the shared venv. Instead: `pip install --no-deps
--target=pylibs hidet==0.6.1` installs ONLY the hidet package itself into an
artifact-local, git-ignored `pylibs/` directory (same pattern as
`attention-kernel/pat`'s own `pylibs/`). `--no-deps` is required: a first
attempt without it (plain `pip install --target=pylibs hidet==0.6.1`) did
NOT treat `plexus_env`'s already-installed torch/numpy/scipy/etc. as
satisfying hidet's requirements and instead re-resolved and downloaded a
second, unrelated ML stack (torch 2.14.0+cu13, triton, a full set of
`nvidia-cu13-*` wheels, ~5.2 GB) straight into `pylibs/` — since every actual
runtime dependency hidet needs is already present in `plexus_env` from
before this fix (nothing was uninstalled, per the task's own constraint),
`--no-deps` correctly leaves them to be resolved from there via `sys.path`,
and `pylibs/` ends up holding only `hidet/` itself (1.3 MB: the package plus
its own bundled `libhidet.so`/`libhidet_runtime.so`, exactly as the PyPI
wheel ships them).

`adapter.py` gained `PYLIBS = <this dir>/pylibs` and `_ensure_paths()`,
called at import time and again at the top of `prepare()`, which inserts
`PYLIBS` at `sys.path[0]` before any `import hidet` — so this adapter's
behavior no longer depends on install order in the shared venv at all.
`available()`'s collision guard was tightened from "not under some
`artifacts/` tree" (too loose — it also had to tolerate legitimate PyPI
installs) to "resolves under THIS artifact's own `pylibs/`, or fail with
`hidet.__file__` printed in the reason string" — a precise, positive check
instead of a negative one.

**Re-gate (login node, functional/gate check only — no timing sweep).**
Rebuilt (`bash build.sh`, verified idempotent on a second run) and re-ran the
exact same command as the original gate:

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel convolution \
    --variant conv-dense-kernel-fp32 --impl hidet-conv2d --smoke --warmup 1 --reps 3
```

Result: **2/3 runs valid**, identical numbers to the original gate above —
smoke-conv-3x3 `5.456e-04` **PASS**, smoke-conv-1x1 `1.007e-03` **FAIL** (tol
`1e-3`, the same documented TF32-batch_matmul finding, unchanged), smoke-conv-
depthwise-s2 `1.266e-07` **PASS**. Confirmed via a direct check that
`hidet.__file__` inside this process is
`.../artifacts/convolution/hidet/pylibs/hidet/__init__.py` (not
`plexus_env/.../site-packages/hidet`), and via `kernelbench.runner --kernel
convolution --list` that the adapter still registers `ok hidet
hidet-conv2d`. No outcome change from the original gate — this section
records the isolation fix, not a new finding.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver reported via
  `nvidia-smi` as "NVIDIA A100-SXM4-40GB, 8.0" on the parent card
  `gpu-b11-6`), login-node build (pip installs) + GPU-node functional
  check/gate (`bench/gpu_run.sh`).
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8, `$KB_CUDA_HOME`), host
  compiler for hidet's nvcc invocations: g++ 12.4.0 (conda-forge, the
  `kb-gcc12` env, forced via `-ccbin`, see "Build-system changes" below),
  torch 2.8.0+cu128, Python 3.12.14 (`kb-env` conda env), hidet 0.6.1
  (unchanged pin).
- Build: OK. `pylibs/` (git-ignored, artifact-local `pip install --target`,
  same pattern as before) needed real additions on this machine beyond
  hidet itself: `pip install --no-deps --target=pylibs` for `click`,
  `cuda-bindings==12.8.0` (the actual code-providing package;
  `cuda-python==13.3.1` from PyPI is now a no-code meta-package that just
  pins `cuda-bindings`/`cuda-core`/`cuda-pathfinder` — installing it alone
  leaves `import cuda` failing), `cuda-pathfinder`, `gitpython`,
  `hip-python-fork`, `importlib_metadata`, `lark`, `nvtx`, `tabulate`,
  `tomlkit`. These were all already present in the original Perlmutter
  `plexus_env` (hence never installed by this adapter before) but are
  absent from zaratan's `kb-env`, so `import hidet` failed cascading
  through `hidet/option.py` (tomlkit) and deeper modules until every one
  was supplied. None of this touches hidet's own code — purely filling in
  runtime deps `--no-deps` had deliberately left for the base venv to
  provide, on a venv where it turns out they aren't there.
- Build-system changes:
  - `build.sh`: added a loop installing the light runtime deps above into
    `pylibs/` (guarded per-package, only if not already importable — a
    no-op on a venv that already has them, e.g. the original plexus_env);
    and set `NVCC_APPEND_FLAGS` to force `-ccbin <kb-gcc12>/g++` (see next
    bullet for why the pre-existing PATH-prepend fix alone stopped being
    sufficient on this machine).
  - `adapter.py`: `_ensure_compatible_host_compiler()` now also sets
    `NVCC_APPEND_FLAGS` (idempotent, only if `-ccbin` isn't already
    present), so the gate run gets the same fix as the build check.
    Root cause (new on this machine, not seen on Perlmutter): hidet's
    installed `hidet/backend/build.py` `NVCC.compile()` never passes
    `-ccbin` to nvcc, so nvcc does its own internal host-compiler search —
    and conda-forge's nvcc binary (which lives inside `$KB_CUDA_HOME/bin`,
    a full conda env that itself ships gcc/g++ 13.4.0 as `KB_CC`/`KB_CXX`)
    prepends a few install-relative directories that resolve back to its
    own bin dir ahead of whatever we put in front of `$PATH` ourselves —
    confirmed empirically with `nvcc -v` (its printed `#$ PATH=` line lists
    a `.../kb-env/bin/../targets/x86_64-linux/../../bin` entry, which
    resolves to `kb-env/bin` again, before our prepended `kb-gcc12/bin`),
    so it always finds gcc 13.4.0 first regardless of `$PATH` ordering and
    fails with the same `<bits/hashtable.h>`/`type_traits`-class parse
    errors as the original gcc-14 incompatibility this adapter's PATH fix
    was written for. `NVCC_APPEND_FLAGS="-ccbin <g++>"` is a real,
    documented nvcc environment variable (extra flags appended to every
    nvcc invocation) that bypasses nvcc's own host-compiler auto-search
    entirely — no hidet/kernel code touched. Perlmutter's NVIDIA HPC SDK
    nvcc install dir has no competing gcc, so this was never triggered
    there; harmless no-op on a `$CUDA_HOME` without a bundled gcc.
- Gate (single `--variant conv-dense-kernel-fp32 --impl hidet-conv2d
  --smoke --warmup 1 --reps 3` invocation, same command STATUS.md has
  recorded twice before): **2/3 runs valid**, identical numbers to both
  prior gates — smoke-conv-3x3 `5.456e-04` **PASS**; smoke-conv-1x1
  `1.007e-03` **FAIL** (tol `1e-3`, the same documented TF32-batch_matmul
  precision finding, unchanged); smoke-conv-depthwise-s2 `1.266e-07`
  **PASS**.
- Deviation from the recorded ruling: none in outcome. Two new
  machine-specific build fixes were needed to get there (missing pylibs
  deps; `-ccbin` via `NVCC_APPEND_FLAGS` instead of PATH alone), documented
  above, but the functional/precision result is bit-for-bit the same as
  the Perlmutter runs.
- Verdict here: BUILT+GATED (2/3 smoke shapes pass; conv-dense-1x1 fails
  the gate for the same documented TF32 precision reason) — equals the
  recorded ruling.
