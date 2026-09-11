# marlin (MARLIN) — gemv

**Status: BUILT+GATED err=3.60e-05..6.37e-04 (all <= tol 1e-3), 3/3 smoke shapes valid**

- Paper: "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large
  Language Models", PPoPP'25. `PAPER_KEY = conf/ppopp/FrantarCCHA25`.
- Artifact: https://github.com/IST-DASLab/marlin
- Commit cloned: `1f25790bdd49fba53106164a24666dade68d7c90`, `git clone --depth 1`
  (upstream's own commit date `2024-09-04`).
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`), GPU `sm_80`
  (A100-PCIE-40GB). Python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`
  (3.11.7), torch `2.8.0+cu128`. `CC=/opt/cray/pe/gcc-native/14/bin/gcc`,
  `CXX=/opt/cray/pe/gcc-native/14/bin/g++` (see "Real build-environment bug
  found" below). `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required at runtime
  (see "Environment note").

## What the artifact actually is

Marlin is a real ahead-of-time CUDA extension (`marlin/marlin_cuda.cpp` +
`marlin/marlin_cuda_kernel.cu`, built via `torch.utils.cpp_extension.
CUDAExtension`, unlike quantized-gemm's `tilus`/`qfactory`, both runtime-JIT
DSLs) implementing a highly optimized FP16-activation x INT4-weight
(symmetric, per-group) GEMM kernel purpose-built for LLM decode (batch
1..~32, the "thin-GEMM" regime the paper's title refers to). Wrapped here
via its own high-level `marlin.Layer` class (`Layer.pack()` /
`Layer.forward()`, `source/marlin/__init__.py`) — **unmodified**, no copy,
no kernel edit (rule 1/3).

## Shape mapping: gemv's M=1 IS marlin's own "batch=1 decode" entry

This domain's gemv workload (`dense.py`'s `DenseShape`) models
`y[M] = A[M,K] @ x[K]` with an implicit batch of 1 — every shape in
`_gemv_shapes()` sets `N=1` (one output vector). Marlin's own convention is
`C[m,n] = A_act[m,k] @ B_weight[k,n]` with `m` = batch (decode: `m=1`).
Mapping `m=1` (our single x), `k` = this domain's `K` (contraction), `n` =
this domain's `M` (marlin's weight `B` is the TRANSPOSE of this domain's
`A`) reproduces `y=A@x` exactly:
`C[0,n] = sum_k x[k]*B[k,n] = sum_k x[k]*A[n,k] = y[n]`. This **is** "wrap
at M=1 (decode)" per the task brief — it falls out of gemv's own N=1
convention, not a choice made for this integration. The task's "optionally
the spec's small-M sweep" (gemv-batched-decode) is **not** exercised: this
domain module has no batch-sweep GEMV workload registered (only
gemv-dense-kernel's fixed shapes exist in `_build_registry()`), so there is
nothing to sweep — noted, not silently skipped.

## Quantization / value-sharing (the task's "follow the quantized-gemm
fairness pattern")

Marlin's kernel computes INT4-weight x FP16-activation GEMM; it physically
cannot consume a continuous fp64 weight. Per ARTIFACT_GUIDE.md's fairness
rule (the same one `domains/ml.py`'s `QuantGemmWorkload` docstring states
for quantized-gemm), the gate must compare marlin's output against a
reference built from the **same quantized values** marlin consumes, not the
pre-quantization continuous matrix.

**Verified empirically why this matters, before writing any adapter code**
(`smoke-gemv-square-512`/`-tallskinny-4096x32`/`-shortfat-32x4096`-shaped
U(-1,1) operands, symmetric per-group INT4 quantization, `max_scaled_err`
against the CONTINUOUS-A reference):

| shape (M,K) | group_size | max_scaled_err (quantization noise only) |
|---|---|---|
| 512,512 | 128 | 1.52e-2 |
| 4096,32 | 32 | 6.66e-2 |
| 32,4096 | 128 | 2.64e-3 |

All exceed `gemv-quantized-weight-kernel`'s parsed tolerance (1e-3) from
quantization noise **alone**, regardless of kernel correctness — gating
against the continuous reference would fail every shape for the wrong
reason.

### Genuinely-required small fix to `kernelbench/domains/dense.py` (rule 3)

`dense.py`'s gemv workload has no built-in quantization concept (unlike
`ml.py`'s `QuantGemmWorkload`), and `runner.py` calls
`domain.REFERENCES[kernel]` uniformly for every impl — there is no per-impl
reference override (the same constraint `tilus`'s own STATUS.md documents
for quantized-gemm). Fixed directly, minimally, in `dense.py`:

1. **New public helper** `quantize_dequantize_groupwise(A, bits, group_size)`
   — symmetric, round-to-nearest, per-(row,group) quantize-then-dequantize.
   Returns `(codes, scale, dequantized)`. Unlike this module's private RNG
   helpers (`_rng_operand` etc., which every adapter replicates verbatim —
   this project's established convention), this one is exported
   (no leading underscore) and **imported directly** by this adapter,
   because it must be the literal same function on both sides for the
   codes/scale to be bit-identical, not two independently-written
   look-alikes.
2. **`reference_gemv` extended** with two OPTIONAL `params` keys,
   `"quant_bits"` / `"quant_group_size"`. Absent (every EXISTING caller of
   gemv-dense-kernel): behavior is bit-for-bit unchanged — verified by
   re-running `numpy-gemv --variant gemv-dense-kernel --smoke` after the
   fix: identical `err 0.00e+00` on all 3 shapes, same as before touching
   the file. Present (set by this adapter's `prepare()`, called BEFORE
   `reference_gemv` with the SAME `params` dict — see `harness.run_variant`):
   `A` is quantized-then-dequantized via the helper above before being used
   as ground truth. This is **not** a looser gate — an impl that computed
   the wrong output from the SAME dequantized weights still fails; it
   isolates fp16-accumulation error (what we actually want to measure) from
   quantization error (an operand-generation artifact, not a kernel
   property).

Both changes are additive and default-off; see the docstrings in
`dense.py` itself (on `quantize_dequantize_groupwise` and the extended
`reference_gemv`) for the full reasoning, cross-referenced back to this file.

### Feeding marlin's own packing API the shared codes

`Layer.pack(linear, scales)` does not take raw integer codes — it takes a
DEQUANTIZED fp16 `nn.Linear` weight (marlin's own term: "a fake-quantized
linear layer") plus a matching per-group fp16 scale, and internally
re-derives `round(w / s)` to get its packed representation. This adapter
builds that dequantized weight as `codes * scale` from the SHARED
codes/scale (not independently re-quantized), so marlin's own `round(w/s)`
recovers exactly those same integer codes (lossless — `w/s` is already
integral by construction). This is the exact pattern marlin's own
`test.py::gen_quant4` uses to build **its own** ground truth for `test.py`'s
correctness check — "feed it shared quantized codes/scales where its
packing API allows" per the task brief.

## Shape-compatibility padding (marlin's own hard tile floor)

`Layer.__init__` hard-asserts `infeatures % 128 == 0` and
`outfeatures % 256 == 0`. This domain's gemv shapes come from TLR-MVM's
bandwidth sweep / GPU-DPF's table shapes (see `benchspecs/gemv/spec.yaml`),
not an LLM-decode shape list, so:

- tall-skinny (K=16 or 32) violates the `infeatures%128==0` floor (M is
  always already %256==0 for these shapes — no M padding needed here).
- short-fat (M=32 or 128) violates the `outfeatures%256==0` floor (K is
  always already %128==0 for these shapes — no K padding needed here).
- square (both dims powers of 2, >= 512) needs no padding at all.

Neither shape class needs padding on BOTH axes simultaneously for this
domain's actual shape set (checked, not assumed — see the `assert` in
`adapter.py::prepare()`). The adapter zero-pads up to the nearest valid
tile: extra K columns of the weight are zero (contribute nothing to `y`
regardless of `x`), extra M rows are computed but sliced off before
`to_host()`. This is **exact** for the correctness gate — verified: the
tall-skinny/short-fat smoke shapes gate at 6.37e-4 / 3.60e-5, the same
order of magnitude as the unpadded square shape (1.72e-4), not inflated by
padding. **Caveat for any FUTURE timed run** at these shapes: the kernel is
doing padded-tile work (up to 8x on K for tall-skinny, 8x/2x on M for
short-fat) the workload's own declared `M*K*bytes` accounting does not
reflect — a GB/s number computed from the workload's TRUE byte count would
not represent what the kernel actually moved. Recommend restricting any
real timed sweep of this impl to the square recommended_subset
(1024/2048/4096/8192/16384, all already marlin-tile-aligned, no padding)
and flagging tall-skinny/short-fat numbers with this caveat if run anyway.

## Precision convention

`PRECISIONS = ["fp16"]` — weights are INT4 (marlin is hard-coded 4-bit only,
`maxq = 2**4-1` in `Layer.pack`), activation and output are FP16, matching
marlin's own "W4A16" convention. `dense.py`'s `DEFAULT_PRECISION["gemv"]`
is unconditionally `"fp64"` (see that module's docstring point 3 — every
tolerance in this spec is parsed at its fp64 entry), so **running this impl
requires an explicit `--precision fp16`** — omitting it makes `runner.py`
resolve `precision="fp64"` and `MarlinGemv.__init__` correctly raises
`NotImplementedError` rather than silently coercing. This is a real,
documented deviation from the task's literal generic test-command template;
the corrected command is below.

## Real build-environment bug found (not a marlin bug, but blocked the build)

`setuptools`/`torch.utils.cpp_extension` resolves the bare compiler name
`"c++"` for the `.cpp` half of the CUDA extension (the `.cu` half goes
through `nvcc`). On this login node, `/usr/bin/c++` is a symlink to
`g++-7` (SUSE Linux 7.5.0) — too old for torch's headers
(`c10/util/C++17.h` hard-`#error`s below GCC 9). This is easy to miss
interactively: this shell's own `.bashrc` defines `alias c++=g++`, and
`$PATH` resolves the bare name `g++` to
`/opt/cray/pe/gcc-native/14/bin/g++` (14.3.0) *ahead* of `/usr/bin` — so a
human typing `c++ --version` at an interactive prompt sees 14.3.0 and is
misled. `pip`'s build subprocess is non-interactive (no bash alias
expansion applies), and PATH-only lookup of the literal name `c++` (not
`g++`) finds `/usr/bin/c++` first, because `gcc-native/14/bin` ships a
`g++` binary but no binary literally named `c++`. Confirmed by instrumenting
a `CXX` wrapper script that logged its own resolved path/version from
inside the actual pip subprocess: `/usr/bin/c++` (SUSE 7.5.0), while the
SAME wrapper resolves to 14.3.0 when invoked directly in an interactive
shell. **Fix** (`build.sh`): explicitly `export CC=/opt/cray/pe/gcc-native/
14/bin/gcc CXX=/opt/cray/pe/gcc-native/14/bin/g++` before `pip install` —
`distutils.sysconfig.customize_compiler` reads `$CC`/`$CXX` before falling
back to its own sysconfig defaults, so this reliably overrides the stale
`/usr/bin/c++` resolution regardless of alias/PATH ordering quirks.
Build-system-only fix (rule 3), no kernel/artifact code touched.

## Environment note

`import marlin`/`marlin_cuda` needs `LD_PRELOAD=/usr/lib64/libstdc++.so.6`
on this machine (same CXXABI/GLIBCXX symbol-version mismatch documented for
`tilus`/`qfactory` and required by this task's own brief).

## adapter.py

- `KERNEL = "gemv"`, `IMPL_NAME = "marlin-w4a16-decode"`,
  `PRECISIONS = ["fp16"]`.
- **prepare()**: regenerates `A`,`x` via the identical seeded-RNG recipe
  `reference_gemv` uses (generate at fp16 first, THEN widen to fp64 — see
  `dense.py`'s own module docstring for why this order is load-bearing;
  skipping it would let fp16 rounding of `A` land on a different side of a
  quantization boundary than what `reference_gemv` independently computes),
  quantizes via the shared `quantize_dequantize_groupwise` helper, sets
  `params["quant_bits"]`/`params["quant_group_size"]` so `reference_gemv`
  (called right after, same `params` dict) gates against the matching
  dequantized values, zero-pads to marlin's tile floor, and packs via
  `marlin.Layer.pack()` (timed as preprocessing per ARTIFACT_GUIDE rule 2).
- **run()** calls `layer(A)` — `Layer.forward()` unmodified, one timed call.
- **to_host()** slices the padded output back down to the workload's true
  `M` elements (`self._M_true`, stashed in `prepare()` — the `Implementation`
  contract's `to_host(out)` only receives `run()`'s return value, not the
  handle, hence the instance attribute).
- `timer()` returns `kernelbench.impls.gpu_cuda.CudaEventTimer`.

## Gate result (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel gemv --variant gemv-quantized-weight-kernel \
    --impl marlin-w4a16-decode --precision fp16 --smoke
```

Result: **3/3 runs valid**, `max_scaled_err` tolerance `1e-3` (parsed from
`gemv-quantized-weight-kernel`'s spec text):

| smoke shape | (M,K) | padding | max_scaled_err | cold time |
|---|---|---|---|---|
| smoke-gemv-square-512 | (512,512) | none | 1.72e-04 | 0.6s |
| smoke-gemv-tallskinny-4096x32 | (4096,32) | K: 32->128 | 6.37e-04 | 0.1s |
| smoke-gemv-shortfat-32x4096 | (32,4096) | M: 32->256 | 3.60e-05 | 0.1s |

All comfortably within tolerance, at the same fp16-accumulation-error order
of magnitude tilus's/qfactory's own quantized-gemm gate numbers land at
(~1e-4-ish) — consistent with a correctly-implemented kernel consuming
correctly-shared quantized values, not luck. Per ARTIFACT_GUIDE rule 4, no
tolerance override was applied or needed. `conforming: False` is expected
and correct (`--smoke`, protocol-overridden `warmup=5, reps=20`, shared
login-node GPU — not a publishable timing run).

**Note on `CORRECTNESS_MODE`**: `dense.py` fixes `CORRECTNESS_MODE["gemv"]
= "max_scaled_err"` at the KERNEL level (every gemv variant, regardless of
what that variant's own spec prose says); `gemv-quantized-weight-kernel`'s
own spec text asks for a *mean*-based check (matching marlin's own
`test.py`), so the harness here enforces a strictly stronger (max, not
mean) version of marlin's own tolerance. This is a pre-existing,
domain-wide property (documented in `dense.py`'s own module docstring,
point 2, for a different variant) — not something this integration
introduced or should silently patch — noted here because it is the reason
the numbers above are directly comparable to a *max* bound even though the
spec text says "mean".

## Real artifact/build bugs found

1. **Build-environment**: see "Real build-environment bug found" above —
   `/usr/bin/c++` (SUSE default, GCC 7.5.0) is too old for torch 2.8's
   headers; `CC`/`CXX` must be pinned explicitly. Not a marlin bug per se
   (any CUDAExtension build on this login node would hit it), but it did
   block building marlin specifically until diagnosed here.
2. No kernel-code bugs found in marlin itself — `Layer.pack()`/
   `Layer.forward()` behaved exactly as `test.py`'s own usage predicts.

## Not done (out of this integration's login-node budget)

- No sweep across `gemv-quantized-weight-kernel`'s own 18-shape LLM
  recommended_subset (Llama7B-65B, Falcon180B linear-layer shapes) or
  `gemv-batched-decode`'s batch sweep — neither is registered as a
  `dense.py` workload (only `gemv-dense-kernel`'s fixed shapes exist in
  `_build_registry()`); adding that shape list is a `dense.py` change
  beyond this integration's "small fix only" scope and is left as a
  follow-up if the LLM-shape sweep is wanted for real timed runs.
- No 1-second anti-throttling sleep between configs (spec's own
  `anti_throttling` field) — not exercised, since this integration ran no
  multi-config sweep, only the fixed 3-shape `--smoke` set.
- Real timed sweep (compute-node allocation, per ARTIFACT_GUIDE rule 5) —
  this integration is login-node build+gate only.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, 14 SMs, ~5 GB; host GPU
  reports as NVIDIA A100-SXM4-40GB via nvidia-smi under the MIG partition),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge,
  env.sh's `$CC`/`$CXX` -- this build's own `CC`/`CXX` default fallback to
  a Perlmutter Cray path is never reached since env.sh already exports
  both), torch 2.8.0+cu128, Python 3.12.14. Commit unchanged:
  `1f25790bdd49fba53106164a24666dade68d7c90`.
- Build: OK, after one real fix. Build-system changes: `build.sh` did
  `cd "$(dirname "$0")"` BEFORE `source ".../toolchain.sh"` (the latter
  resolved via `$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh`) --
  `BASH_SOURCE[0]` keeps whatever form (relative/absolute) the script was
  invoked with, so resolving it against the ALREADY-CHANGED cwd doubles the
  relative path (e.g. invoked as `bench/artifacts/gemv/marlin/build.sh`
  from the repo root -- exactly how this reproduction pass invokes every
  build.sh -- the doubled path becomes `.../gemv/marlin/bench/artifacts/
  gemv/marlin/../../toolchain.sh`) and fails with "No such file or
  directory". This is a real invocation-order bug (present since the
  original baseline commit), not a machine difference -- it only stayed
  latent because the reference machine's interactive session happened to
  invoke build.sh by absolute path. Fixed by sourcing `toolchain.sh` BEFORE
  the `cd` (same fix applied to `quantized-gemm/marlin/build.sh` and
  `gemv/packkv/build.sh`, which had the identical bug -- see their own
  STATUS.md entries). No change to `source/` or marlin's own code. (Also
  hit one transient `[Errno 11] Resource temporarily unavailable` from pip
  during `Preparing metadata` -- the documented shared-login-node
  RLIMIT_NPROC=256 contention from concurrent sibling builds, not an
  artifact issue; resolved by retrying.)
- Gate: `--variant gemv-quantized-weight-kernel --impl marlin-w4a16-decode
  --precision fp16 --smoke`: **PASS** 3/3 -- smoke-gemv-square-512 err
  1.03e-04, smoke-gemv-tallskinny-4096x32 err 6.37e-04,
  smoke-gemv-shortfat-32x4096 err 2.04e-05 (tol `1e-3`). Same order of
  magnitude as the recorded run (1.72e-04 / 6.37e-04 / 3.60e-05); the small
  per-shape differences are consistent with this adapter's own quantization
  RNG rather than a kernel difference (GPU/driver-dependent floating-point
  rounding in the `A`/`x` generation feeding the packer), not a regression.
- Deviation from the recorded ruling: none -- 3/3 pass at the same
  tolerance (`1e-3`), same order-of-magnitude errors.
- Verdict here: BUILT+GATED -- equals the recorded ruling.
