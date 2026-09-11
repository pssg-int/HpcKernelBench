# tetris (Tetris sparse conv) — convolution

**Status: BUILT+GATED**

- Paper: "Tetris: Accelerating Sparse Convolution by Exploiting Memory Reuse
  on GPU", PPoPP'24. `PAPER_KEY = conf/ppopp/LiuZYLQ24`.
- Artifact: https://github.com/XG-zheng/Tetris-artifact-evalution
- Commit cloned: `80ebe9be0bd09af1a4a7bd358e4f63d34ffc9fdc` (`git clone --depth 1`).
  Submodules (`sputnik`, `Unified-Convolution-Framework`) were **not** fetched
  — the actual Tetris kernel (`source/Tetris/*`) is plain source directly in
  this repo, no submodule needed for it (only the sputnik/TACO-UCF
  *baselines* the paper's own fig6.sh compares against live in the
  submodules; out of scope for this integration).
- Toolchain: `nvcc` 12.9.41 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch
  `2.8.0+cu128`. GPU: NVIDIA A100-PCIE-40GB, compute capability 8.0 (`sm_80`).
  cuDNN 9.5.0 headers (NERSC module `cudnn/9.5.0`,
  `/global/common/software/nersc9/cudnn/9.5.0-cuda12/include`) resolved for a
  compile-time-only include (`cuda_utils.h` declares, never calls,
  `CuDNNConv2d`) — no cuDNN library linked.

## What the artifact actually is

`source/Tetris/` is the real kernel source, separate from the paper's own
benchmark driver (`evalution/conv2d_timing.py` + `fig6.sh`, which
subprocess-invokes a compiled CLI binary — `spconv2d_test.o` — that runs an
internal `WARM=30`/`REPEAT=100` loop and prints one aggregate mean; not
wrapped here, per `ARTIFACT_GUIDE.md` rule 1). Two real, reusable entry
points live in `source/Tetris/spconv2d_utils.h` / `spconv2d_kernel.cu`:

- `SparseFilter<ValueType,OffsetType,PositionType>` (header-only struct,
  `spconv2d_utils.h`): CPU-side packer that walks a dense (Cout,Cin,Kh,Kw)
  filter and builds Tetris's "SPF" (stride-packed-filter) sparse format —
  `offsets`/`stage_len`/`position`/`values`/`oc_permutation` arrays, uploaded
  to the GPU inside the constructor. This IS the artifact's own
  preprocessing (weight-format construction), per rule 2.
- `SparseConv2d<ValueType,OffsetType,PositionType,KERNEL_SIZE,STRIDE,
  TILE_IC,TILE_H,TILE_W>` (`spconv2d_kernel.cu`): host launcher for
  `SparseConv2dKernel`, ONE GPU kernel launch per call — the actual sparse
  conv forward computation.

`source/Tetris/spf.cc` (the artifact's own "ablation" driver) demonstrates
calling both directly: `SparseFilter(...)` once, then `SparseConv2d<float,
int,int, 3,1,128,2,4>(...)` in its WARM/REPEAT loop. This adapter calls the
exact same two entry points, the exact same template instantiation, but only
ONCE per `run()` — the WARM/REPEAT looping is the harness's job
(`kernelbench.harness.run_variant`), not the artifact's.

## What was built (`csrc/tetris_wrap.cu`, new file — no artifact file edited)

A thin C-linkage wrapper, `bench/artifacts/convolution/tetris/csrc/
tetris_wrap.cu`, `#include`s `source/Tetris/spconv2d_utils.h` and
`spconv2d_kernel.cuh` and exposes three `extern "C"` functions:
`tetris_pack_filter` (constructs `SparseFilter`), `tetris_free_filter`,
`tetris_conv_forward` (calls `SparseConv2d<float,int,int,3,1,128,2,4>`
once). `build.sh` compiles `source/Tetris/spconv2d_kernel.cu` (unmodified)
+ `csrc/tetris_wrap.cu` and links `csrc/build/libtetris.so`, loaded via
`ctypes` — the same idiom `kernelbench/impls/gpu_cuda.py`'s `_load_lib()`
uses for this project's own `csrc/kernels.cu`.

**No line of `source/Tetris/*.{cc,cu,cuh,h}` was edited** (rule 3). The only
patch is the CUDA arch flag in `build.sh`: the artifact's own
`evalution/script/build_all.sh` hardcodes
`--generate-code=arch=compute_70,code=sm_70` (this paper's eval GPU was a
V100, per `Readme.md`'s "ENV: V100-PCIE-32GB"); this machine's GPU is an
A100, so `build.sh` instead passes `--generate-code=arch=compute_80,
code=sm_80` — a build-system fix, explicitly allowed by rule 3.

**Kernel-config constraint (documented limitation, not a patch):** the
wrapper calls exactly ONE of `spconv2d_kernel.cu`'s ~100 pre-instantiated
tile configs — `SparseConv2d<..., K=3, S=1, TILE_IC=128, TILE_H=2,
TILE_W=4>`, the same one `spf.cc`'s own driver uses. The artifact's real
benchmark binary (`spconv2d.cc`) instead autotunes over that whole ~100-way
tile-config table per layer shape at runtime; reimplementing that dispatch
was out of this integration's time budget. Consequence: `adapter.py`'s
`prepare()` raises `NotImplementedError` for any workload that isn't
`Kh=Kw=3, stride=1, groups=1` — confirmed by the smoke-test run below,
which crashes exactly there for `smoke-conv-1x1` (K=1).

## Pruning granularity — a finding, not a guess

`benchspecs/convolution/spec.yaml`'s `conv-sparse-pruned-kernel-fp32` text
says `"{50%, 75%, 90%, 95%} structured out-channel pruning"`. Reading
`SparseFilter`'s packer (`spconv2d_utils.h`) shows no such constraint: it
walks `(out_channel, in_channel, kh, kw)` and packs whichever *individual*
taps are nonzero — nothing requires a whole output channel to be all-zero.
To settle it empirically rather than guess, one of the artifact's own
shipped pruned checkpoints was loaded and inspected directly:

```
source/evalution/conv_weight_data/vgg19/vgg19-92-acc-71.7/
  module8_1_module3_0_conv2d_0_H_56_W_56_IC_256_OC_256_KS_3_Pad_1_S_1_G_1
  -> 256x256x3x3 filter: 90.00% of ALL individual elements are exactly 0.0;
     0 of 256 output channels are fully zero.
```

I.e. Tetris's own "pruned" weights are **global, unstructured, elementwise
magnitude-pruned**, not structured-out-channel-pruned — confirming the spec
text's "structured out-channel" description does not match what the
artifact's own SPF format/checkpoints actually use. Per this task's
instructions ("read their code/paper for the real constraint rather than
guessing"), `adapter.py`'s `_prune_unstructured_magnitude()` implements
**global unstructured magnitude pruning at 90% sparsity** — a spec-listed
level, and the exact level measured in the artifact's own checkpoint above
— rather than literally zeroing whole output channels.

## Correctness — two distinct results, per the task's instructions

**(1) Formal harness gate — `conv-dense-kernel-fp32`'s reference (EXPECTED
mismatch, explained).** This reference (`kernelbench.domains.ml.
reference_conv`) draws its own FRESH, unpruned W from the same seed — it has
no way to receive the adapter's pruned W, so comparing Tetris's
pruned-weight output against it is comparing two different computations.
Run exactly as specified:

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
$PY -m kernelbench.runner --kernel convolution --variant conv-dense-kernel-fp32 \
    --impl tetris-sparse-conv --smoke --warmup 1 --reps 3
```

Result: `smoke-conv-3x3` (K=3/S=1/groups=1, the one smoke shape this adapter
supports) → **INVALID, `max_scaled_err=7.110e-01` vs tol `1e-3`** — the
expected pruned-vs-unpruned mismatch, not a kernel bug (see below). The
command then **crashes** on the next smoke shape, `smoke-conv-1x1`
(`NotImplementedError`, uncaught by `runner.py`), because `--smoke` iterates
all 3 convolution smoke shapes and this adapter only supports K=3/stride=1
(documented limitation above) — an honest, reproducible result of running
the exact specified command, not hidden.

A second, cleaner run against this adapter's actual working shape
(`resnet18_1`, the full `layer_shape_suite` instance used for the real
correctness check below), via the runner's non-smoke `--matrices` path:

```
$PY -m kernelbench.runner --kernel convolution --variant conv-dense-kernel-fp32 \
    --impl tetris-sparse-conv --matrices resnet18_1 --warmup 1 --reps 3
```

Result: **INVALID, `max_scaled_err=2.562e-01` vs tol `1e-3`** — same expected
reason, cleanly isolated (no crash, single workload). Reduced protocol
(warmup=1, reps=3) and a shared login-node GPU either way; explicitly
non-conforming (`conforming: False`), no timing reported since the gate
failed first (harness.py's own ordering: correctness before timing).

**(2) Pruned-weight-matched dense fp64 reference (the ACTUAL correctness
evidence for this pruned-weight kernel).** Standalone check, checked into
this directory as `check_pruned_correctness.py` (run with
`/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python check_pruned_correctness.py`
from this directory): same X, same seed=42, workload=`resnet18_1`
(N=1,Cin=64,Cout=64,Hout=Wout=56, Kh=Kw=3,stride=1,groups=1); reference
computed with `kernelbench.domains.ml._direct_conv_fp64` on the **exact
pruned W the kernel was fed** (zeros and all),
`scale = _direct_conv_fp64(|X|,|W_pruned|)`, same `max_scaled_err`
convention `harness.check_correctness` uses:

```python
X64, W64 = X.astype(np.float64), W_pruned.astype(np.float64)
ref = ml._direct_conv_fp64(X64, W64, workload)
scale = ml._direct_conv_fp64(np.abs(X64), np.abs(W64), workload)
max_scaled_err = float((np.abs(got - ref) / np.maximum(scale, 1e-300)).max())
```

Result: **`max_scaled_err = 1.935e-07`, well within tolerance `1e-3`
(spec's `conv-sparse-pruned-kernel-fp32` correctness bound). PASS.** This is
the methodologically correct gate for a pruned-weight kernel and the real
evidence the SPF pack + `SparseConv2dKernel` launch are computing the right
thing.

## adapter.py

- `KERNEL = "convolution"`, `IMPL_NAME = "tetris-sparse-conv"`,
  `PRECISIONS = ["fp32"]`.
- `prepare()`: exact `kernelbench.domains.ml._make_conv_operands` numpy
  recipe (`rng = np.random.default_rng(seed); X = rng.uniform(...); W =
  rng.uniform(...)` — seed then X then W) so operands match every other
  impl/reference of this workload/seed; then global unstructured magnitude
  pruning to 90% (see above); then `tetris_pack_filter` (SPF packing, H2D
  upload) and an NCHW→NHWC transpose of X (the layout
  `SparseConv2dKernel` indexes with) — all timed as this run's
  preprocessing, never inside `run()`.
- `run()`: exactly one `tetris_conv_forward` call (one `SparseConv2dKernel`
  launch), no allocation, no H2D/D2H.
- `to_host()`: NHWC (Tetris's native output layout) → NCHW, matching
  `reference_conv`'s `(N,Cout,Hout,Wout)` convention.
- `timer()` reuses `kernelbench.impls.gpu_cuda.CudaEventTimer` (one event
  pair per iteration) — no artifact/harness timing code duplicated.

## Not done / out of scope for this integration

- Only the K=3/stride=1/groups=1 tile instantiation is wrapped (see
  "Kernel-config constraint" above) — `spconv2d.cc`'s full autotuning
  dispatch across ~100 tile configs (needed for K=1 and stride=2 shapes, and
  for per-shape *best* tile choice even at K=3/stride=1) was out of budget.
- No timing sweep (rule 5: shared login-node GPU, functional gate only). The
  numbers quoted above (`warmup=1, reps=3`) are explicitly non-conforming
  and are not throughput results.
- `sputnik`/`Unified-Convolution-Framework` submodules were not fetched
  (not needed for the Tetris kernel itself; only for the paper's own
  baseline comparisons in `fig6.sh`, out of scope here).
- Only batch N=1 was exercised (`resnet18_1`, N∈{1,4,8,16} per the spec's
  batch sweep is a `--dims`-equivalent the convolution domain doesn't wire
  through `runner.py`'s CLI the way spmm's `--dims` does — `params["N"]`
  works programmatically via `adapter.py`'s `prepare()`, just not exercised
  here beyond N=1 for the login-node functional check).

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, parent card reported by
  `nvidia-smi` as "NVIDIA A100-SXM4-40GB, 8.0" on `gpu-b11-6`), login-node
  build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8, `$KB_CUDA_HOME`), host
  compiler for all three `nvcc` invocations: g++ 12.4.0 (conda-forge
  `kb-gcc12` env, via `-ccbin`, added — see below), torch 2.8.0+cu128,
  Python 3.12.14 (`kb-env`). cudnn 9.x headers resolved from the
  torch-bundled `nvidia-cudnn-cu12` pip package
  (`.../site-packages/nvidia/cudnn/include/cudnn.h`) instead of the
  NERSC `cudnn/9.5.0` module (not applicable off Perlmutter) — header-only
  use, unchanged from the original integration, no library linked.
- Build: OK. Same source commit `80ebe9be0bd09af1a4a7bd358e4f63d34ffc9fdc`.
  Build-system changes:
  - `build.sh`: added `HOST_COMPILER="${HOST_COMPILER:-${KB_GXX12:-/usr/bin/g++-12}}"`
    and `-ccbin "$HOST_COMPILER"` on all three `nvcc` calls (compile
    `spconv2d_kernel.cu`, compile `tetris_wrap.cu`, link
    `libtetris.so`) — this artifact's `nvcc` calls previously had no
    `-ccbin` at all, so `nvcc` did its own host-compiler search rather
    than following `$PATH`, and on this machine that search finds
    `$KB_CUDA_HOME`'s own bundled g++ 13.4.0 first (nvcc 12.8 cannot parse
    its `<type_traits>`/`<bits/hashtable.h>` — same failure class
    documented in `bench/artifacts/convolution/hidet/STATUS.md`'s
    Reproduction section). This is the same `-ccbin "$HOST_COMPILER"` /
    `KB_GXX12` fix already used by most other artifacts in this repo
    (`grep -l ccbin bench/artifacts/*/*/build.sh`); tetris just hadn't
    needed it on Perlmutter's NVIDIA HPC SDK nvcc, whose install dir has no
    competing gcc.
  - `build.sh`: added a portable fallback for `CUDNN_INC` that asks the
    interpreter (`$PY`) where its own `nvidia.cudnn` package's headers live
    (`importlib.util.find_spec`), tried before the NERSC-specific
    `$CUDNN_DIR`/`module load cudnn/9.5.0` path (kept, now a fallback of a
    fallback, for Perlmutter). Needed because zaratan has no `module`-based
    cudnn and no system-wide cudnn install; torch's own pip dependency
    closure already ships cudnn 9.x headers, which is all this artifact
    needs (header-only, no library linked, unchanged from before).
- Gate: ran all three of STATUS.md's recorded checks in one GPU allocation.
  - `--variant conv-dense-kernel-fp32 --impl tetris-sparse-conv --smoke
    --warmup 1 --reps 3`: smoke-conv-3x3 **INVALID**,
    `max_scaled_err=7.110e-01` vs tol `1e-3` — identical number to the
    original gate (expected pruned-vs-unpruned mismatch, not a kernel bug,
    per "Correctness" above). smoke-conv-1x1 and smoke-conv-depthwise-s2:
    **UNSUPPORTED** (`0/1 runs valid (2 unsupported)`, exit 0) — see
    "Deviation" below.
  - `--variant conv-dense-kernel-fp32 --impl tetris-sparse-conv --matrices
    resnet18_1 --warmup 1 --reps 3`: **INVALID**,
    `max_scaled_err=2.562e-01` vs tol `1e-3` — identical number to the
    original gate, same expected reason.
  - `check_pruned_correctness.py` (the methodologically correct pruned-
    weight-matched check): `W_pruned sparsity: 0.9000`,
    `max_scaled_err (pruned-weight-matched) = 1.935068e-07` vs tol `1e-3`
    — **PASS**, matches the original `1.935e-07` to the printed precision.
- Deviation from the recorded ruling: the correctness numbers are
  unchanged, but the `--smoke` command's behavior on unsupported shapes
  differs from what the original STATUS.md recorded: originally, hitting
  `smoke-conv-1x1` (K=1, outside this adapter's K=3/stride=1/groups=1-only
  wrapper) raised an uncaught `NotImplementedError` that crashed the whole
  `--smoke` run after printing the first (valid, INVALID-result)
  `smoke-conv-3x3` line. Here, the harness (`bench/kernelbench/runner.py`,
  not touched by this reproduction pass) instead reports
  `smoke-conv-1x1`/`smoke-conv-depthwise-s2` as `UNSUPPORTED` and exits 0
  cleanly — a harness-side improvement (or the crash description was
  already stale) unrelated to anything in this artifact's own build/adapter
  code; not an artifact-outcome change and not something this reproduction
  pass touched (`bench/kernelbench/` is off-limits per rule 1).
- Verdict here: BUILT+GATED — equals the recorded ruling (build succeeds;
  the formal dense-reference gate correctly reports INVALID for the
  expected pruned-vs-unpruned mismatch; the methodologically correct
  pruned-weight-matched check PASSes, same as originally recorded).
