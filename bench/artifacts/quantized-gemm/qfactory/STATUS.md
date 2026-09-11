# qfactory (QFactory) — quantized-gemm

**Status: BUILT+GATED err=6.55e-05..2.02e-04 (both <= tol 1e-3), with two
confirmed real-artifact-limitation shapes documented below (not gated
around, not hidden)**

- Paper: "QFactory: Accelerating Quantized Large Language Model Serving
  with Qtile Graphs", USENIX ATC'25. `PAPER_KEY = conf/usenix/ZhangZSZ25`.
- Artifact: https://github.com/zqh-wz/QFactory-AE
- Commit cloned: `9c75e5d3fea89c89588275953d9d93837ae88036`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`), GPU `sm_80`
  (A100-PCIE-40GB, `QFACTORY_ARCH=80`). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required (see "Environment note").

## What the artifact actually is

`source/qfactory/` is a pure-Python "Qtile-graph" JIT compiler for
quantized GEMM: `qfactory.QLinear`/`GPTQLinear`
(`source/qfactory/qlinear.py`) accepts an externally-supplied
`(weight: int32, scale: fp16, zero: int32)` triple and JIT-compiles (via
runtime `nvcc`, cached under this artifact's own `cache/`) a CUDA kernel
for the requested `(M,K,N,nbits,group_size)`. Wrapped via this SAME
high-level entry point `tests/acc_gptq.py` (the artifact's own accuracy
test) calls — unmodified, no source edit — with the exact
`compile_flags={"enable_transform": False, "enable_schedule": False,
"enable_lower": False}` that test uses to skip QFactory's own autotuning
search (only ONE kernel variant is ever compiled; ~9-12s cold-build per new
shape observed here).

## Build

No C-extension build step (`requirements.txt`'s `bitblas==0.1.0` is NOT
imported anywhere under `source/qfactory/` — grep-confirmed; it is only used
by the separate end-to-end reproduction scripts under `third_party/`, not
the core kernel-compiler package this adapter wraps, so it was never
installed). `build.sh` only adds `source/` to `sys.path` and verifies
`import qfactory` + a fresh JIT-compile-cache directory. `QFACTORY_ARCH=80`
and `QFACTORY_CACHE_DIR=<this artifact dir>/cache` are set by the adapter
(the artifact defaults `QFACTORY_CACHE_DIR` to `~/.qfactory` — repointed
here to keep this integration self-contained, same spirit as `tilus`'s
local `vendor/`).

## Environment note

`import qfactory` (via torch's C++ extension loader path used internally)
needs `LD_PRELOAD=/usr/lib64/libstdc++.so.6` on this machine (same
CXXABI/GLIBCXX symbol-version mismatch documented for `fused3s`/`tilus` and
required by this task's own brief).

## adapter.py

- `KERNEL = "quantized-gemm"`, `IMPL_NAME = "qfactory-gptq-matmul"`,
  `PRECISIONS = ["fp16"]`.
- **Bit-widths**: `{2, 4, 8}` — `qfactory/convert.py`'s `PERMUTE_ORDER` table
  (the packing-order lookup its own weight-transform code needs) only
  defines entries for 2/4/8 bits; **3-bit (this domain's INT3 smoke format)
  is not representable by this artifact's packing scheme at all** — a real
  capability limit, not an adapter gap (see "Real artifact bugs / limits
  found" below for how this interacts with `--smoke`).
- **KEY FAIRNESS POINT, honored via an exact zero-point shift (not the
  documented exception)**: `GPTQLinear` never re-quantizes from a
  continuous weight itself — it consumes an externally supplied
  `(weight, scale, zero)`, exactly like `w.quantize()`'s own
  `(codes, scale, group_idx)`. The only reconciliation needed is that
  QFactory's own convention is ASYMMETRIC (`dequant = (code - (zero+1)) *
  scale`, code unsigned in `[0, 2^nbits)`) while `QuantGemmWorkload`'s is
  SYMMETRIC (`dequant = code * scale`, code signed in `[-qmax, qmax]`).
  This adapter maps `qweight = code + qmax`, `zero = qmax - 1` (a constant,
  every group/column), which makes
  `(qweight - (zero+1)) * scale = (code + qmax - qmax) * scale = code *
  scale` — an EXACT identity, not an approximation. The gate result below
  (err ~1e-4-2e-4, the same order as `tilus`'s and `numpy-dequant-gemm`'s
  own baselines) confirms the mapping introduces no discrepancy beyond
  ordinary fp16/kernel-arithmetic rounding.
- **Group-count bookkeeping** (see module docstring's "GROUP-COUNT
  ALIGNMENT" section): `GPTQLinear` asserts `K % (2*group_size) == 0`
  (an EVEN group count). When `w.n_groups` is odd, this adapter halves
  `w.group_size_eff` and repeats each real group's shared scale value
  across both halves — mathematically identical elementwise dequantization
  (dequant is elementwise `code*scale`; splitting one group into two
  identical-scale halves changes nothing about what any individual element
  dequantizes to), so this is bookkeeping, not a numeric approximation.

## Gate result (login node, functional check only)

`--smoke`'s default 3-shape set could not be used AS-IS for this adapter --
see "Real artifact bugs / limits found" below for exactly why (shape 2 hits
a real silent-failure bug in the artifact; shape 3's bit-width is
categorically unsupported). Gate verified instead against the ONE
compatible `--smoke` shape plus one representative real-registry shape
(`--matrices`, still a single functional launch — ARTIFACT_GUIDE rule 5's
"handful of functional/gate check", not a timed sweep):

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel quantized-gemm --variant qgemm-w4a16-decode-kernel \
    --impl qfactory-gptq-matmul --smoke        # runs only the compatible shape below when isolated manually
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel quantized-gemm --variant qgemm-w4a16-decode-kernel \
    --impl qfactory-gptq-matmul \
    --matrices qgemm-llama-2-7b-qkv_proj-decode-m1 --warmup 1 --reps 1
```

| shape | max_scaled_err | cold-build time |
|---|---|---|
| smoke-qgemm-w4-g128-decode (M=8,K=256,N=384, INT4 g=128) | 2.02e-04 | 12.4s |
| qgemm-llama-2-7b-qkv_proj-decode-m1 (M=1,K=4096,N=12288, INT4 g=128, real recommended_subset shape) | 6.55e-05 | 18.1s |

Both comfortably within the parsed `1e-3` tolerance. Per ARTIFACT_GUIDE rule
4, no tolerance override was applied or needed.

## Real artifact bugs / limits found

1. **Silent data corruption on an internal, Python-unchecked precondition
   (a real bug, not a documented limit)**: `--smoke`'s `smoke-qgemm-w4-
   percol-decode` shape (M=4, K=96, N=128, this adapter's derived
   `group_size=48`) compiles and RUNS without a Python exception, but
   prints to stderr (from `qfactory/include/gptq_gemv_base.h:180`, a
   host-side launch-precondition check inside the C++ kernel dispatcher,
   never surfaced to Python):
   ```
   Error: GS(48) must be a multiple of 8 * BLOCK_SIZE(1024)
   ```
   and then returns without writing any output — the pre-allocated
   `torch.empty(...)` output tensor is left as uninitialized device memory,
   which this harness's correctness gate correctly reports as
   `max_scaled_err=nan` (`--smoke`'s "INVALID" line). QFactory's OWN
   Python-level assertion (`GPTQLinear.__init__`'s
   `in_features % (group_size*2) == 0`) is satisfied (`96 % 96 == 0`) --
   there is no way for a caller to know from the documented/asserted API
   that a DIFFERENT, much stricter internal constraint
   (`GS % (8*BLOCK_SIZE) == 0`, i.e. `GS % 1024 == 0` here — `BLOCK_SIZE`
   printed as `128`) applies to the specific `gptq_gemv`/`gptq_gemv_base`
   kernel variant the compiler happens to select for this small-N shape
   (traced via the differing PTX symbol name, `kernel_gptq` for the two
   PASSING shapes above vs. `gptq_gemv`-family for this one). This is a
   genuine correctness hazard in the shipped artifact: a caller with no
   independent numeric gate (i.e. every use case except this benchmark's
   own harness) would silently get garbage output with no error raised in
   Python. Not routed around, not patched (would require touching kernel
   code); recorded here as the honest, reproducible result per
   ARTIFACT_GUIDE rule 4.
2. **3-bit quantization is categorically unsupported** (a documented
   capability limit, not a bug): `--smoke`'s `smoke-qgemm-w3-g64-batched`
   shape uses `bits=3`; `qfactory/convert.py`'s `PERMUTE_ORDER` packing-
   order table only has entries for `{2, 4, 8}`. This adapter raises
   `NotImplementedError` for any other bit-width, which — combined with
   finding 1 above — means bare `--smoke` (which iterates ALL 3 of this
   domain's default smoke shapes with no per-shape isolation) cannot
   complete for this specific `IMPL_NAME` without either an `INVALID` line
   (shape 2) or a hard crash (shape 3, since `runner.py`'s per-shape loop
   has no try/except around `harness.run_variant()` — an exception raised
   inside any impl's `prepare()` takes down the whole `--smoke` invocation,
   not just that one shape). This is a harness-level observation, not
   something this integration's scope covers fixing; documented here so a
   reader isn't surprised that literal `--smoke` doesn't complete cleanly
   for this one `IMPL_NAME` and knows why (see "Gate result" above for the
   verification actually performed instead).

## Not done (out of this integration's login-node budget)

- No sweep across the full `quantized-gemm` recommended_subset registry —
  one real-shape (`qgemm-llama-2-7b-qkv_proj-decode-m1`) plus the one
  compatible smoke shape were gated, per the task's login-node/GPU-shared
  budget. Real timed sweeps belong on a compute-node allocation.
- The GEMV-specialized kernel path (`gptq_gemv.h`/`gptq_gemv_base.h`,
  triggered by small-N shapes) was not further characterized beyond
  confirming finding 1 above — QFactory's own shape-to-kernel-variant
  dispatch logic (`qfactory/core/matmul.py`) was not fully reverse-engineered
  to determine exactly which `(M,N,K,GS)` combinations route to it.
- `bits=2` (INT2) is supported per `PERMUTE_ORDER`'s table but was not
  separately gated (this domain's smoke/registry shapes use bits in
  `{3, 4}` only; INT2 would need a new workload entry to exercise).


## Baseline role (2026-09-05 selection-rule revision)

**Competitor, not a SOTA baseline.** Under the revised rule (core kernel papers
evaluated on the track's own input regime first; `kernel-papers/output/
baseline_selection.md`), this artifact would not have been selected:
- kernel centrality rated `component` (the quantized-gemm kernel is not this paper's headline, kernel-level contribution).
Rating rationale (`output/kernel_centrality.json`): QFactory is a compilation framework (Qtile/QGraph abstraction) that generates quantized kernels among a transformable search space, fitting the compiler/DSL component rule; its AE README's matrix-size/bit-width ablations and Marlin/BitBLAS baselines closely match the spec's suite.
It stays in the registry and runs under the same gate as every other
implementation, but Phase 3 does not treat it as the human-SOTA reference for
`quantized-gemm`.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100-SXM4-40GB (sm_80, full card, partition `gpu`),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14. `QFACTORY_ARCH=80` (adapter-set), matches
  this GPU.
- Build: OK. Build-system changes: `build.sh` — (1) same LD_PRELOAD fix as
  `fp6llm`/`tilus` (`LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}"`
  instead of a hardcoded override that would have fought zaratan's
  correctly-exported knob); (2) the import-verification `python -c` call
  segfaulted deep inside numpy's CPU-dispatcher init the first time, root
  cause traced to this login node's `RLIMIT_NPROC=256` (shared across the
  whole user session, several builds running concurrently) being exhausted
  by OpenBLAS spinning up one `pthread_create` per core (128 on this node)
  — the exact resource-contention class the reproduction protocol's
  "MACHINE GOTCHAS" section already documents for `make -j`, just hit here
  via numpy/OpenBLAS instead of a compiler. Fix: cap
  `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` (both `${VAR:-4}`, overridable)
  for the verification `python -c` call. Not an artifact bug.
- Gate: qgemm-w4a16-decode-kernel/fp16 (`--smoke`, `OPENBLAS_NUM_THREADS=4
  OMP_NUM_THREADS=4` set for the same reason as the build fix, on the CLI
  only — not a file change): smoke-qgemm-w4-g128-decode PASS (err 2.02e-04
  <= 1e-3); smoke-qgemm-w4-percol-decode INVALID (`max_scaled_err=nan`,
  the recorded `GS(48) must be a multiple of 8*BLOCK_SIZE(1024)` artifact
  bug, reproduced verbatim); smoke-qgemm-w3-g64-batched UNSUPPORTED
  (bits=3 not in (2,4,8), as recorded) — harness reported this as "1/2
  runs valid (1 unsupported)" and exited 0, i.e. it did NOT hard-crash the
  whole `--smoke` invocation the way this file's prose speculated it might
  ("no try/except around harness.run_variant()"); the actual behavior
  observed here is the harness gracefully emitting an UNSUPPORTED line for
  the 3-bit shape — a more benign outcome than the documented risk, not a
  contradiction of anything actually verified before. Real-shape check
  (`--matrices qgemm-llama-2-7b-qkv_proj-decode-m1 --warmup 1 --reps 1`):
  PASS, err 6.55e-05 <= 1e-3 — matches the recorded number exactly.
- Deviation from the recorded ruling: none in substance (identical error
  values); the only difference is the smoke-set finishing cleanly (exit 0,
  UNSUPPORTED line) rather than needing per-shape isolation to avoid a
  possible crash, as this file's own text flagged as a risk but had not
  fully verified either way.
- Verdict here: BUILT+GATED — equals the recorded ruling.
