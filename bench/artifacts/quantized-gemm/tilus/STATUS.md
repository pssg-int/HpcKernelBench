# tilus (Tilus) — quantized-gemm

**Status: BUILT+GATED err=1.74e-04..2.68e-04 (all <= tol 1e-3)**

- Paper: "Tilus: A Tile-Level GPGPU Programming Language for Low-Precision
  Computation", ASPLOS'26. `PAPER_KEY = conf/asplos/DingHZL0Y0P26`.
- Artifact: https://github.com/NVIDIA/tilus
- Commit cloned: `9a22de0aac52beb8343c1f2511135f351848afbd`, `git clone --depth 1`
  (`git log -1` reports commit date `2026-07-05` — upstream's own timestamp).
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`), GPU
  `sm_80` (A100-PCIE-40GB). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python` (3.11.7), torch
  `2.8.0+cu128`. `apache-tvm-ffi==0.1.10` (pinned — see "Build" below),
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required (see "Environment note").

## What the artifact actually is

Tilus is a tile-level GPU-kernel DSL (Triton-family): kernels are Python
`Script` classes that Tilus's own nvcc-based backend JIT-compiles the first
time they are invoked, optionally autotuned across a declared parameter
space (`@tilus.autotune(...)`). `examples/quantization/matmul_a16wx.py`
ships a genuine fp16-activation x low-bit-weight (1–8 bits, uniform
per-group symmetric quantization) fused-dequant GEMM — `QuantizedLinear`,
wrapping three JIT kernels: a layout-shuffle pair
(`QuantizedMatmulChangeLayout`/`RestoreLayout`) and the compute kernel
(`QuantizedMatmul`). Wrapped **unmodified**, imported directly from
`source/examples/quantization/matmul_a16wx.py` — no copy, no edit (rule 1).

## Autotuning is NOT a multi-hour problem here (rule 1/task-brief guidance)

`QuantizedMatmul` is decorated `@tilus.autotune("warp_spatial", ...)` /
`("warp_repeat", ...)` / `("num_stages", ...)` / `("split_k_factor", ...)` —
a 3×5×2×3 = 90-point space if swept in full. But the example class also
ships its own class attribute:
```python
debug_schedule = dict(warp_spatial=[2, 2], warp_repeat=[2, 4, 2],
                      num_stages=3, split_k_factor=1)
```
and Tilus's own `generate_schedules()`
(`vendor/tilus/lang/instantiated_script.py:132`) does
`if script_cls.debug_schedule: spanned_space = [script_cls.debug_schedule]`
— i.e. **the example already pins ONE schedule**, unmodified, with no flag
needed on our side. Every `[Building] ... 1/1` line in the gate run below
confirms exactly one schedule was ever compiled. Measured cold-build time
(3 kernels: `cast`, `change_layout`, `quantized_matmul`, one JIT build each,
first invocation only) was ~13.7s and ~18.2s for the two non-cache-hit smoke
shapes below — comfortably inside a login-node "run the JIT once in
prepare()" budget (task brief's own explicit allowance), not the
hours-long-tuning case that would warrant SKIPPED.

## Build (rule 3/6: build-system fixes only, kernel untouched)

`build.sh`:
1. `pip install --target=vendor --no-deps ./source apache-tvm-ffi==0.1.10`
   — builds a wheel from the cloned repo's own `pyproject.toml`
   (setuptools_scm-versioned: installs as `tilus-0.1.dev1+g9a22de0aa`,
   tying the installed version string to this exact commit) into a
   **local** `vendor/` directory, not the shared venv's site-packages
   (this machine's venv is actively shared with other concurrent
   integration work this session observed running in parallel — keeping
   Tilus's dependency tree self-contained avoids touching it). Installing
   from `source/` rather than PyPI's `tilus` wheel matters: the API
   `matmul_a16wx.py` uses is pinned to this commit; PyPI's latest release
   can drift.
2. **`apache-tvm-ffi==0.1.10`, pinned (a real bug found, not a typo fix)**:
   Tilus's `pyproject.toml` lists a bare, unpinned `"apache-tvm-ffi"`.
   Installing the latest PyPI release at integration time
   (`apache-tvm-ffi==0.1.13.post2`) makes this commit's generated CUDA glue
   fail to compile:
   ```
   error: class "tvm::ffi::TypeTraits<DLTensor *, void>" has no member
   "TryCastFromAnyView"
   ```
   deep inside `tvm/ffi/type_traits.h`, triggered while building the
   auto-generated `cast` kernel (float32→int4b). Confirmed independent of
   this integration's own code: this is tvm-ffi's OWN header failing to
   instantiate its OWN function-call-unpacking template
   (`FallbackOnlyTraitsBase::TryFallbackTypes`) for `void_p` args, a genuine
   API-shape change between tvm-ffi releases that this July-2026 Tilus
   commit's codegen was never updated for (`TypeTraits<DLTensor*>` in
   0.1.13.post2 uses a different base-class structure than 0.1.10/0.1.9/
   0.1.8.post2, all three of which still compile clean — checked by
   diffing `type_traits.h` across versions). No pin exists anywhere in
   Tilus's own repo (no lockfile, no CI pin) — recorded here as a real,
   reproducible dependency-version-skew bug in the artifact's own release
   process, not something introduced by this integration.
3. No kernel code touched anywhere (build-system/dependency-pin fixes only,
   rule 3).

## Environment note

`import tilus` (via `tvm_ffi`) needs `LD_PRELOAD=/usr/lib64/libstdc++.so.6`
on this machine (same CXXABI/GLIBCXX symbol-version mismatch documented for
`fused3s` and required by this task's own brief) — without it, `tvm_ffi`'s
compiled extension fails to import. `available()` catches this and reports
the fix in its reason string; `build.sh`'s own verification step already
runs under `LD_PRELOAD`.

## adapter.py

- `KERNEL = "quantized-gemm"`, `IMPL_NAME = "tilus-quantized-matmul"`,
  `PRECISIONS = ["fp16"]` (`QuantizedMatmulCommon.__init__` asserts
  `a_dtype in [float16, bfloat16]`; only fp16 wired, matching
  `DEFAULT_PRECISION["quantized-gemm"]`).
- **Bit-widths**: `{3, 4, 6, 8}` via Tilus's `int3b`/`int4b`/`int6b`/`int8`
  dtypes (`QuantGemmWorkload.quantize()` only ever produces *integer*
  codes, so Tilus's floating quantized formats, e.g. `float4_e2m1`, are not
  wired — nothing in this domain generates a float-coded weight to feed
  them).
- **KEY FAIRNESS POINT, honored (not the documented-exception path)**:
  `QuantizedLinear.load_and_quantize(weight)` as shipped computes its OWN
  scale/codes from a floating weight tensor — calling it unmodified would
  re-quantize independently of `w.quantize()` and (correctly, but
  uninterestingly) fail the harness's single kernel-level `reference_qgemm`
  gate on a **quantization-scheme mismatch**, not a kernel bug — the harness
  has no per-impl reference override (`runner.py` always passes
  `domain.REFERENCES[kernel]`). Tilus's own scheme (symmetric per-group
  affine, weight stored as a packed low-bit tensor + per-group fp16 scale)
  is *structurally identical* to `QuantGemmWorkload`'s own, so this adapter
  bypasses `load_and_quantize`'s round/clamp/scale-compute steps and instead:
  1. reads the SAME `(codes, scale, group_idx)` from `w.quantize()` every
     other impl and the reference consume;
  2. casts `codes` (already-integral, within Tilus's dtype range — e.g.
     `[-7,7] ⊂ int4b's [-8,7]`) through Tilus's OWN
     `tilus.from_torch(...).to(w_dtype)` cast kernel — **lossless**, since
     rounding an already-integer float to its nearest integer is a no-op —
     reproducing bit-for-bit the packed buffer `load_and_quantize` would
     have produced from these same codes;
  3. feeds that buffer through Tilus's own `change_layout_kernel` (the
     exact call `load_and_quantize` makes) to shuffle it into the tile
     layout `QuantizedMatmul` expects;
  4. writes `scale` directly into `self.scales` (already the exact
     `(n_groups, N)` shape/role Tilus's own parameter expects — no reshape
     needed).
  This is the "do the packing from OUR quantized values in prepare()"
  default rule (not the documented exception `quantix`'s adapter needs) —
  confirmed correct by the gate result below (err ~2e-4, roughly fp16-
  storage-rounding scale, exactly what a correctly-implemented kernel on
  correctly-shared inputs should produce).
- **prepare()** builds `QuantizedLinear`, performs the value-sharing pack
  above (timed as preprocessing, per ARTIFACT_GUIDE rule 2), and generates
  the activation via the identical recipe `kernelbench.domains.ml.
  _make_activation` uses (replicated verbatim, not imported — same
  convention `turbofno`'s/`fused3s`'s adapters use for a private
  cross-module helper).
- **run()** calls `ql(A)` — `QuantizedLinear.forward()` unmodified, one
  timed call to the compute kernel. `forward()` allocates its own output
  tensor each call (the artifact's own code, not something this adapter
  hoists — the same "the GEMM's own output temporary is the one exception"
  precedent `dense.py`'s `NumpyGemm` docstring states, since patching
  `forward()` to accept a pre-allocated buffer would mean editing the
  artifact's own Python glue, out of scope per rule 3's "kernel itself must
  change" boundary — this is the module's Python wrapper, not its CUDA
  kernel, but treated with the same caution).
- `timer()` returns `kernelbench.impls.gpu_cuda.CudaEventTimer`.

## Gate result (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel quantized-gemm --variant qgemm-w4a16-decode-kernel \
    --impl tilus-quantized-matmul --smoke
```

Result: **3/3 runs valid**, `max_scaled_err` tolerance `1e-3` (parsed from
`qgemm-w4a16-decode-kernel`'s spec text, see `kernelbench/domains/ml.py`):

| smoke shape | bits/group | max_scaled_err | cold-build time |
|---|---|---|---|
| smoke-qgemm-w4-g128-decode (M=8,K=256,N=384) | INT4, g=128 | 1.74e-04 | 1.3s (cache-warm from an earlier dev run) |
| smoke-qgemm-w4-percol-decode (M=4,K=96,N=128) | INT4, g=96 (per-column fallback) | 2.68e-04 | 13.7s (cold) |
| smoke-qgemm-w3-g64-batched (M=64,K=192,N=256) | INT3, g=64 | 2.18e-04 | 18.2s (cold) |

All comfortably within tolerance — errors are roughly fp16-storage-rounding
scale, exactly what a correctly-implemented fused-dequant kernel consuming
the identical quantized values as the fp64 reference should produce (same
order of magnitude as `numpy-dequant-gemm`'s own ~1e-4 baseline, see
`kernelbench/domains/ml.py`'s `NumpyDequantGemm` docstring). Per
ARTIFACT_GUIDE rule 4, no tolerance override was applied or needed.
`conforming: False` is expected and correct — a `--smoke`, protocol-
overridden (`warmup=5, reps=20`) functional check on a shared login-node
GPU, not a publishable timing run. The high-variance warning on the second
shape reflects login-node GPU contention (shared device), not a kernel
issue.

## Real artifact bug found

`apache-tvm-ffi==0.1.13.post2` (latest PyPI release at integration time)
breaks compilation of this Tilus commit's generated CUDA glue — see "Build"
above for the full C++ template-instantiation error and the version pin
that fixes it. Not filed upstream (out of scope for this integration), but
reproducible: `pip install --no-deps apache-tvm-ffi` (unpinned, matching
Tilus's own `pyproject.toml`) then running any Tilus kernel fails identically.

## Not done (out of this integration's login-node budget)

- No sweep across the full `quantized-gemm` recommended_subset registry (23
  model/layer shapes × 13 M values) — only the harness's own 3-shape
  `--smoke` set was run, per the task's login-node/GPU-shared budget. Real
  timed sweeps belong on a compute-node allocation (DOMAIN_GUIDE.md).
  Cold-build cost (~15-40s per NEW (K,N) shape, one-time and cached under
  `cache/` thereafter) makes a full sweep's first pass slow but not
  infeasible off the login node.
- Tilus's own floating-point quantized formats (`float3_e1m1`,
  `float4_e2m1`, `float6_e3m2`, `float8_e5m2` — FP3–FP8, matching the
  spec's FP5/FP6 weight_formats entries) are not wired, since
  `QuantGemmWorkload` only generates integer-coded weights (see above) —
  would need a genuinely different `quantize()` variant to exercise them.
- Blackwell/Hopper-specific kernel variants (`examples/hopper_matmul`,
  `examples/blackwell_matmul`) are separate example files, not the
  quantized-GEMM one wrapped here — out of scope (this integration targets
  the sm_80 A100 available on this machine).


## Baseline role (2026-09-05 selection-rule revision)

**Competitor, not a SOTA baseline.** Under the revised rule (core kernel papers
evaluated on the track's own input regime first; `kernel-papers/output/
baseline_selection.md`), this artifact would not have been selected:
- kernel centrality rated `component` (the quantized-gemm kernel is not this paper's headline, kernel-level contribution).
Rating rationale (`output/kernel_centrality.json`): Tilus is a general tile-level GPGPU DSL (arbitrary 1-8 bit types) whose flagship showcase is fused-dequant GEMM, so it fits the compiler/DSL-generates-this-kernel component rule; its own protocol (CUDA events, 50 reps median, L2 cleared, Gemma-2/Qwen2.5/Llama-3.3 shapes) is close kin to the spec's own decode-kernel protocol.
It stays in the registry and runs under the same gate as every other
implementation, but Phase 3 does not treat it as the human-SOTA reference for
`quantized-gemm`.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100-SXM4-40GB (sm_80, full card, partition `gpu`),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14, `apache-tvm-ffi==0.1.10` (same pin, still
  needed — confirmed the same version-skew failure mode is latent upstream).
- Build: OK (idempotent — `vendor/`'s `pip install --target` of `./source
  apache-tvm-ffi==0.1.10 tabulate tqdm cuda-python==12.8.0
  cuda-bindings==12.8.0` already present in `build.sh` from an earlier
  session, ran as-is). First attempt failed transiently with `error:
  [Errno 11] Resource temporarily unavailable` inside pip's build-isolation
  subprocess (egg_info) — this login node's shared `RLIMIT_NPROC=256`
  exhausted by concurrent build/session load, not an artifact issue; a
  bare retry succeeded. Build-system changes: `build.sh` — same LD_PRELOAD
  fix as `fp6llm`/`qfactory` (`LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/
  libstdc++.so.6}"` on the import-verification line, which was still
  hardcoded to the Perlmutter-only value and would have overridden
  zaratan's correctly-exported knob).
- Gate: qgemm-w4a16-decode-kernel/fp16 (`--smoke`, run via `bench/
  gpu_run.sh -t 40` — this is a runtime-JIT DSL, kernels compile on the
  GPU node on first invocation, ~13-53s per new (K,N) shape, matching the
  cold-build times this file already documents): smoke-qgemm-w4-g128-decode
  PASS (err 1.74e-04 <= 1e-3); smoke-qgemm-w4-percol-decode PASS (err
  2.68e-04 <= 1e-3); smoke-qgemm-w3-g64-batched PASS (err 2.18e-04 <=
  1e-3). 3/3 runs valid. All three error values match the recorded numbers
  exactly.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
