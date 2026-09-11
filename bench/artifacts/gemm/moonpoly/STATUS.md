# moonpoly (MoonPoly-dev) — gemm

**Status: BUILT+GATED at BOTH precisions — fp32 err=1.62e-07..2.55e-07 (tol
1e-3) and fp16 err=1.08e-04..2.31e-04 (tol 1e-2), 3/3 smoke shapes each
(re-gated on zaratan 2026-09-09 under the harness's per-precision tolerance
table, `tolerance_for(precision)`, added 2026-09-05 — see the "Reproduction
on zaratan" section below; supersedes the fp16-gate-blocked finding right
below, kept for the record).**

Earlier outcome, kept for the record: **Status: BUILT+GATED
err=1.62e-07..2.55e-07 at fp32 (all <= tol 1e-6); fp16
BUILT but gate blocked (0/3, ~1.1e-4..2.3e-4 vs a fp64-calibrated 1e-6
tolerance — expected, not a bug, same story as hexcute-gemm/inferfast-spmm)**

- Paper: "Optimizing Dynamic-Shape Neural Networks on Accelerators via
  On-the-Fly Micro-Kernel Polymerization", ASPLOS'24. `PAPER_KEY =
  conf/asplos/YuLZCFX24`.
- Artifact: https://github.com/LinkZyy/MoonPoly-dev
- Commit cloned: `22f638da0ba5c6b2bf4a01305cee0d4c95154242` (2026-07-07),
  `git clone --depth 50`.
- CUTLASS vendored at the pinned commit `76c96b0be35cb263debe3e3d8418b80911a544ab`
  (2025-09-04), fetched shallow directly by SHA into `source/3rdparty/cutlass/`
  (not a git submodule in this repo — `3rdparty/` is merely `.gitignore`d;
  the pin comes from MoonPoly-dev's own README + `integrations/cutlass/`).
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`), host compiler
  `g++-12` (SUSE 12.3.0), `TORCH_CUDA_ARCH_LIST=8.0`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.

## What the artifact actually is

MoonPoly's own README: "a two-stage micro-kernel polymerization framework
for dynamic-shape GEMM... FP16/FP32/INT8 row-column-column GEMM kernels,
checked-in runtime selectors." `moonpoly/core/moonpoly_gemm.cu` builds a
pip-installable pybind11/torch extension (`setup.py`'s `CUDAExtension`)
exposing `moonpoly.linear(input, weight)` — verified directly against
`source/tests/python/test_moonpoly_linear.py`, which checks it against
`torch.nn.functional.linear`. Internally `moonpoly.linear` dispatches on
input dtype (fp16/fp32/int8) through `moonpoly::moonpolyLinear ->
moonpoly::moonpoly_gemm -> run_fp16_gemm/run_fp32_gemm`, each of which
selects among ~40 CUTLASS row-column-column micro-kernel candidates via the
paper's own fitted/analytic cost model
(`moonpoly/core/{fp16,fp32}/*_rcc.cu`, `*_rcc_cost.cu`,
`moonpoly/generated/{fp16,fp32}/rcc/*.inc`) — that dispatch/selection step
**is** the paper's own contribution ("on-the-fly micro-kernel
polymerization"), so `moonpoly.linear` is the correct kernel-level entry
point to wrap (rule 1), not one fixed lower-level micro-kernel call.

## Build (rule 3/6: minimal patches, all recorded; MoonPoly's own kernel code untouched)

1. **CUTLASS vendoring**: not a submodule in this repo; shallow-fetched
   directly by the pinned SHA (`git fetch --depth 1 origin <SHA>`) into
   `source/3rdparty/cutlass/` — avoids a full-history clone of a
   multi-hundred-MB repo for one commit. Per ARTIFACT_GUIDE.md's "header-
   only deps: vendor into source/" allowance.
2. **`integrations/cutlass/cutlass_4_1_pattern2_twin_gemm.patch`** (MoonPoly-
   dev's own patch file, shipped in the repo) applied to the vendored
   CUTLASS tree — **required**, not optional/vLLM-only: `setup.py`'s own
   source list includes `core/fp16/fp16_rcc.cu`, which references the
   patch's `TwinGemm`/Pattern-2 additions (new methods on existing CUTLASS
   `device::Gemm` classes + one new header `twin_gemm.h`), and the pip build
   fails to compile without it. This is a build-system/vendoring change
   (patching a *third-party dependency*, not MoonPoly's own kernel code),
   in-scope per rule 3. Applied idempotently (`build.sh` checks for the
   patch's marker symbol before reapplying).
3. **Host compiler**: `g++-12`, not the system default `g++-14` — same
   nvcc-12.9/`__has_construct`-undefined issue already fixed identically in
   `spmm/inferfast` and `gemm/turbofno`. Passed via `CC`/`CXX`/`CUDAHOSTCXX`
   env vars (torch's `CUDAExtension` build reads these).
4. `python setup.py build_ext --inplace` builds `moonpoly.cpython-311-*.so`
   (~22 MB) directly under `source/` — no other source patches needed.

`build.sh` is idempotent (checks for the built `.so` / vendored CUTLASS SHA
before redoing either step, exits 0 either way).

## adapter.py

- `KERNEL = "gemm"`, `IMPL_NAME = "moonpoly-gemm"`, `PRECISIONS = ["fp16",
  "fp32"]` — int8 is supported by the artifact's own dtype dispatch but not
  wired here (no real-valued correctness reference without a quantization
  scheme, out of scope).
- **Shape mapping**: our workload is `C(M,N) = A(M,K) @ B(K,N)`.
  `moonpoly.linear(input, weight)` computes `input @ weight.T` with
  `weight.shape == (out_features, in_features) == (N, K)`, so `prepare()`
  builds `weight = B.T.contiguous()` — an explicit transpose+copy that IS
  the artifact's own required operand layout (its `linear` has no
  "B-already-transposed" variant reachable from Python), timed as
  preprocessing per rule 2 (same discipline as insum's CSR->COO or
  inferfast's densify+pad).
- **Batching**: `moonpoly.linear` has no native batched entry point
  (`torch_helpers::createOutputTensor` reads `A.sizes()[0]`/`B.sizes()[0]`
  positionally — a 3D input would silently misinterpret the batch dim as
  M/N — and no batched-GEMM kernel exists anywhere under `moonpoly/core`).
  Per rule 1 ("wrap at the finest boundary available"), `run()` loops the
  real, unmodified `moonpoly.linear` call once per batch element.
- **RNG**: `prepare()` regenerates `A`/`B` with `np.random.default_rng`,
  byte-for-byte matching `dense.py`'s own `_rng_operand`/`reference_gemm`
  (`seed`/`seed+1_000_003+b` split) — NOT torch's `Generator`-based
  `gpu_cuda.py::_dense`, per the RNG-mismatch gotcha independently
  rediscovered by `spmm/insum`, `spmm/inferfast`, and `gemm/hexcute`.
- `to_host()`/`timer()`/`free()` follow the standard CUDA-impl pattern
  (`kernelbench.impls.gpu_cuda.CudaEventTimer`).

## Gate result (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
$PY -m kernelbench.runner --kernel gemm --variant gemm-square-kernel \
    --impl moonpoly-gemm --smoke --warmup 1 --reps 3 --precision fp32
$PY -m kernelbench.runner --kernel gemm --variant gemm-square-kernel \
    --impl moonpoly-gemm --smoke --warmup 1 --reps 3 --precision fp16
```
(`--precision` must be given explicitly: the default resolved from
`gemm-square-kernel`'s spec text is fp64 per `dense.py`'s own
`DEFAULT_PRECISION`, and `moonpoly-gemm` raises `NotImplementedError`
cleanly for fp64 rather than silently mis-running — confirmed by running
with no `--precision` flag first.)

**fp32: 3/3 runs valid.**

| smoke shape | max_scaled_err | tolerance | GFLOP/s |
|---|---|---|---|
| smoke-gemm-square-256 (256x256x256) | 2.55e-07 | 1e-06 | 343.91 |
| smoke-gemm-irregular-384x256x512 | 2.22e-07 | 1e-06 | 829.79 |
| smoke-gemm-batched-b4-64 (batch=4, 64x64x64) | 1.62e-07 | 1e-06 | 24.39 |

**fp16: 0/3 runs valid** (`max_scaled_err` 1.080e-04 to 2.306e-04, vs.
tolerance `1e-06`). Same root cause already documented in
`gemm/hexcute/STATUS.md` and `spmm/inferfast/STATUS.md`: `gemm-square-
kernel`'s parsed tolerance is the spec's fp64-precision table entry (per
`dense.py`'s own module docstring, point 3); a correctly-rounded fp16
result is expected to sit around `2^-11 ≈ 4.9e-4` scaled error, and the
observed values (1.1e-4 to 2.3e-4) are consistent with that, not a defect.
**Per ARTIFACT_GUIDE.md rule 4, no tolerance override was applied for
either precision run.** `PRECISIONS` is left as `["fp16", "fp32"]` (both
genuinely work end-to-end; only fp32 currently clears this particular
spec variant's gate) rather than trimmed to fp32-only, since ARTIFACT_GUIDE
asks for "what the artifact actually supports," and fp16 does run
correctly at fp16-appropriate precision — it just cannot pass a
fp64-calibrated bound, exactly like Hexcute's and InferFast's fp16 kernels.

Both result JSONs are under `bench/results/gemm_gemm-square-kernel_*.json`
(fp32 run timestamp `1786193810`, fp16 run timestamp `1786193820`).

## Not done

- No sweep across the full `gemm-square-*`/`gemm-cubic-*`/`gemm-irregular-*`/
  `gemm-llm-*`/`gemm-batched-*` shape lists — only the harness's own 3-shape
  `--smoke` set was run, per the task's login-node/shared-GPU budget.
- int8 dtype path (`moonpoly.linear` supports it) not wired — no real-valued
  reference to gate it against without adding a quantization scheme, out of
  scope here.
- `moonpoly.linear_cpp_selector` (a separate fitted-predictor entry point
  for fp16 2D contiguous inputs) was traced but not used — it falls back to
  the identical `moonpoly.linear` dispatch for every case this adapter
  exercises (confirmed by reading `moonpoly_gemm.cu`), so using it directly
  would not change behavior, only add an unnecessary extra entry point.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, 14 SMs, ~5 GB; host GPU
  reports as NVIDIA A100-SXM4-40GB via nvidia-smi under the MIG partition),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 12.4.0 (conda-forge,
  `KB_GXX12`/`KB_GCC12`, same role as Perlmutter's g++-12 pin), torch
  2.8.0+cu128, Python 3.12.14. `TORCH_CUDA_ARCH_LIST=8.0`. Commit unchanged:
  MoonPoly-dev `22f638da0ba5c6b2bf4a01305cee0d4c95154242`, CUTLASS
  `76c96b0be35cb263debe3e3d8418b80911a544ab`.
- Build: OK, after one real fix. Build-system changes: `build.sh`'s CUTLASS
  patch-idempotency check (`grep -q "get_grid_shape" .../gemm.h`) is a false
  positive -- that substring already exists in STOCK, unpatched CUTLASS at
  this exact pin (`gemm.h:477`'s pre-existing `threadblock_swizzle.
  get_grid_shape(...)` call, unrelated to the patch's own new
  `get_grid_shape()` method additions), so on a truly fresh CUTLASS vendor
  fetch the check always read "already patched" and silently skipped
  `git apply`, surfacing as `namespace "cutlass::gemm::device" has no
  member "TwinGemm"` compiling `core/fp16/fp16_rcc.cu`. Replaced with the
  robust idempotency test `git apply --check --reverse <patch>` (succeeds
  only if the patch is already applied). Nothing under `source/` or the
  patch file itself changed. (Also hit one transient
  `fatal: unable to create thread: Resource temporarily unavailable` on the
  CUTLASS shallow `git fetch` -- the documented shared-login-node
  RLIMIT_NPROC=256 contention from concurrent sibling builds, not an
  artifact issue; resolved by retrying.)
- Gate: `--variant gemm-square-kernel --smoke --warmup 1 --reps 3 --precision
  fp32`: **PASS** 3/3 (err 2.55e-07 / 2.22e-07 / 1.62e-07, tol 1e-3 as
  reported by the harness here -- see deviation note below). `--precision
  fp16`: **PASS** 3/3 (err 1.21e-04 / 1.08e-04 / 2.31e-04, tol 1e-2) --
  identical error magnitudes to the original run at both precisions.
- Deviation from the recorded ruling: the fp16 variant now GATES (3/3
  valid) where the original STATUS.md recorded 0/3 valid against a
  fp64-calibrated `1e-6` bound -- this is the exact same harness fix
  documented in `gemm/hexcute/STATUS.md`'s "Dependency isolation
  (2026-09-05)" / "Re-gate" section (`spec.py` gained a
  `tolerance_by_precision` table after this artifact's STATUS.md was
  written; the kernel/adapter did not change, only what tolerance it is
  gated against). Also note: fp32's tolerance now reads `1e-3` here versus
  the `1e-6` this STATUS.md originally recorded -- same root cause
  (per-precision table vs. the old blanket fp64-text parse); the fp32
  errors (~1e-7) clear either bound by a wide margin, so this is a
  reported-tolerance change only, not a pass/fail change for fp32.
- Verdict here: BUILT+GATED at both precisions -- an improvement on the
  recorded ruling (fp16 now passes where it was previously gate-blocked);
  see the updated top-of-file ruling above.
