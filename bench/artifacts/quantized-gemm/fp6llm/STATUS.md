# fp6llm (FP6-LLM / Quant-LLM) — quantized-gemm

**Status: BUILT+GATED err=2.68e-05..1.92e-04 (all <= tol 1e-3) — `partial`
regime match (FP6 minifloat vs. the spec's INT bit-widths; a genuinely
required, documented `ml.py` hook was added to gate it fairly — no
regression on tilus/qfactory/marlin/numpy-dequant-gemm, re-verified below).**

- Paper: "Quant-LLM: Accelerating the Serving of Large Language Models via
  FP6-Centric Algorithm-System Co-Design" (kernel/artifact name FP6-LLM /
  TC-FPx), USENIX ATC'24. `PAPER_KEY = conf/usenix/XiaZWCYYBWZZRHS24`.
- Artifact: https://github.com/usyd-fsalab/fp6_llm
- Commit cloned: `12e83379f16a4ee1494be00db6956aab56baf620` (2025-07-16),
  `git clone` (full, not `--depth 1` — see source.provenance).
- Toolchain: `nvcc` 12.9, GPU `sm_80` (A100-PCIE-40GB). `source/setup.py`
  already hardcodes `-gencode=arch=compute_80,code=sm_80` — no arch patch
  needed. Python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch
  `2.8.0+cu128`. `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required.

## Selection rationale

`kernel_centrality.json`'s `quantized-gemm|conf/usenix/XiaZWCYYBWZZRHS24`:
centrality `core`, regime `matches` — "one of only two papers the spec
calls out as treating model-level accuracy as first-class... its
batch-crossover finding (0.94x slowdown at batch=32) directly motivated
the spec's batched-kernel variant." Core baseline per
`output/baseline_selection.md`'s quantized-gemm section.

## What the artifact actually is

A real ahead-of-time CUDA extension (`fp6_llm/csrc/{pybind.cpp,
fp6_linear.cu}`) implementing a fused FPx-weight (E2M2/FP5 or E3M2/FP6
floating minifloat) x FP16-activation Tensor-Core GEMM. Built via torch's
`cpp_extension.CUDAExtension` (unmodified `setup.py`).

## Build (`build.sh`)

`pip install --target=vendor --no-deps ./source`, same local-vendor
isolation rationale as tilus/marlin. **One real, reproducible artifact
packaging bug found**: `fp6_llm/__init__.py` does
`from fp6_llm_cuda import ...` WITHOUT importing `torch` first (unlike
marlin's/tilus's own `__init__.py`, both of which `import torch` before
touching their compiled extension) — the bare extension import then fails
with `ImportError: libc10.so: cannot open shared object file: No such
file or directory`, because `fp6_llm_cuda.so`'s dependency on torch's
`libc10.so` is only resolvable once torch's own package machinery has
already loaded it into the process. Fix (build-system-adjacent, not a
kernel-code patch): `import torch` before `import fp6_llm`/
`fp6_llm_cuda`, both in `build.sh`'s own verification step and in this
adapter's `available()`/`prepare()` (which already do `import torch`
first as a matter of course, same as every other CUDA adapter here).

## Regime mismatch — why a new `ml.py` hook was needed (task brief's
explicit instruction)

`QuantGemmWorkload.quantize()` only ever produces symmetric per-GROUP
INTEGER codes (`ml.py`'s own docstring). FP6-LLM's kernel instead consumes
a genuinely different format: an IEEE-754-style floating MINIFLOAT
(sign + E exponent bits + M mantissa bits, non-uniform step size) with a
SINGLE scale per output channel (this track's N, spanning the entire K —
not a 128-column group). Neither the numeric format nor the scaling
granularity match `QuantGemmWorkload`'s scheme. Per the task brief, the
workload was **not** bent to fit; instead:

1. This adapter regenerates the SAME pre-quantization continuous weight
   `QuantGemmWorkload.quantize()` itself draws (`np.random.default_rng
   (w.seed + _W_OFFSET).standard_normal((K, N))`, `_W_OFFSET=1_000_003`
   copied verbatim from `ml.py` — private to that module, same
   cross-module-replication convention every adapter here already uses).
2. Quantizes it into FP6 (E3M2, the paper's dominant format) via a new,
   independently-written minifloat quantizer, `_quantize_fpx_code` (see
   "The quantizer" below) — legitimate "pack from our values" work per
   ARTIFACT_GUIDE rule 3, since the artifact ships **no** continuous-fp16-
   to-FPx quantizer anywhere: `csrc/utils/weight_quant.h::cast_fp16_fp6`
   is uncompiled, unbound dead code (not in `setup.py`'s `sources`, not in
   `pybind.cpp`), explicitly commented "To be used in the future as a
   tool", and the artifact's own benchmark script
   (`tests/python/kernel_test_fpx.py`) feeds the kernel `torch.randint(...)`
   RANDOM BITS, never a real quantized weight — confirmed by reading both
   files; no quantization path is exercised anywhere in this repo.
3. Gates against a reference built from those SAME dequantized FP6 values
   via **one new optional params key in `kernelbench/domains/ml.py`**:
   `reference_qgemm`'s `params["dequantized_W_override"]` — the exact
   `dense.py::reference_gemv`'s `dequantized_A_override` pattern (used by
   `gemv/packkv`'s own regime-mismatched quantization scheme), generalized
   to this track's weight side. Absent (every existing caller —
   `tilus`, `qfactory`, `marlin`, `numpy-dequant-gemm`): behavior is
   bit-for-bit unchanged (verified below). This does NOT loosen the gate:
   a kernel bug computing the wrong output from the identical dequantized
   FP6 weights still fails it.

### `ml.py` diff (only touched file, minimal per the task brief)

`reference_qgemm` gained: `W_override = params.pop("dequantized_W_override",
None)`; if set, `W_dequant = np.asarray(W_override, dtype=np.float64)`
(shape-checked against `(w.K, w.N)`) is used verbatim in place of the
`codes.astype(np.float64) * scale[group_idx]` inline expression, otherwise
the original code path runs unchanged. No other function touched.

### No-regression proof (re-ran every existing quantized-gemm gate after
the `ml.py` change, identical numbers to before)

```
$PY -m kernelbench.runner --kernel quantized-gemm --variant qgemm-w4a16-decode-kernel --impl numpy-dequant-gemm --smoke
  -> 1.22e-04, 1.52e-04, 1.26e-04   (unchanged)
LD_PRELOAD=... $PY -m kernelbench.runner ... --impl marlin-w4a16-gemm --smoke
  -> 2.14e-04, 3.65e-04, UNSUPPORTED(bits=3)   (unchanged, matches marlin/STATUS.md)
LD_PRELOAD=... $PY -m kernelbench.runner ... --impl tilus-quantized-matmul --smoke
  -> 1.74e-04, 2.68e-04, 2.18e-04   (unchanged, matches tilus/STATUS.md exactly)
LD_PRELOAD=... $PY -m kernelbench.runner ... --impl qfactory-gptq-matmul --matrices qgemm-llama-2-7b-qkv_proj-decode-m1
  -> 6.55e-05   (unchanged, matches qfactory/STATUS.md exactly)
```
All four impls reproduce their pre-change numbers bit-for-bit (same
`max_scaled_err` to the printed precision) — the new `dequantized_W_override`
path is dead code for every caller except this adapter, confirming the
fix is genuinely additive/optional as designed.

## The quantizer (`_quantize_fpx_code`)

A numpy port of `cast_fp16_fp6`'s own algorithm (sign copied, exponent
rebiased by a constant, mantissa TRUNCATED — not rounded — from fp16's 10
bits to M bits, subnormal handling), generalized from its hardcoded
(E=3,M=2) to arbitrary (E,M) via the exact bias-remap identity
`weight_dequant.h`'s own `BIAS_OFFSET=(1<<4)-(1<<(EXPONENT-1))` constant
encodes (E=3: `16-4=12=15-3`, i.e. fp16's bias(15) minus FPx's IEEE
bias(3) — confirms this adapter's encoder and the artifact's COMPILED
decoder agree on the format). **Verified empirically** (round-trip through
the artifact's own `weight_dequant_eXmY_cpu`, independent of the GEMM
kernel under test) before writing the adapter: E3M2 gives ~8-10% median
relative quantization error (consistent with a 2-bit mantissa: expected
step size ~2^-2 per octave), E2M2 saturates at |value|=7 vs. E3M2's 28,
both matching `cast_fp16_fp6`'s own `absmax_fp6=28` literal.

## Real bug found in THIS adapter's own development (not an artifact bug)

Bit-packing the raw FPx codes into the `int32` tensor
`weight_prepacking_eXmY_cpu`/`weight_dequant_eXmY_cpu` expect required
matching `Extract_X_Bits_To_A_Byte`'s (`csrc/utils/common.h`) big-endian
BIT-stream convention (row's first bits -> LOWEST byte address). A first
implementation built the packed value by naive bit-shifting into a Python
integer treating stream-bit-0 as the value's own MSB, then `.view(np.int32)`
— WRONG on this little-endian machine (numpy's native int32 view puts the
value's low BYTE at the lowest address, the opposite of what the C++ side
reads). Caught immediately by the pre-integration round-trip check above
(garbage decode, `median relerr > 100%`), fixed by building the actual
byte SEQUENCE first (byte 0 = the row's first 8 stream-bits, MSB-first
within the byte) and only THEN `.view(np.int32)`-ing that already-
correctly-ordered byte buffer — see `_pack_bits`'s docstring for detail.
Documented here as a real trap in bridging Python/numpy's native-endian
tensor views against a C++ extension's `reinterpret_cast<unsigned char*>`
bit-numbering convention, not an artifact defect.

## adapter.py

- `KERNEL="quantized-gemm"`, `IMPL_NAME="fp6llm-fpx-gemm"`,
  `PRECISIONS=["fp16"]`.
- Always quantizes into FP6 (E3M2) regardless of the workload's own
  `bits`/`group_size` fields (deliberately ignored — see "Regime
  mismatch" above; this is the task brief's explicit design, not an
  adapter gap).
- Shape mapping: FP6-LLM's own convention is weight `[OC,IC]`, activation
  `[B,IC]`, output `[B,OC]` — `OC<->N`, `IC<->K`, `B<->M` in this track's
  `C[M,N]=A[M,K]@W[K,N]`. Real kernel constraints (`fp6_linear.cu`'s own
  `assert`s inside `fpx_linear_kernel`, not just the test script's
  convenience checks): `N % 256 == 0`, `K % 64 == 0`. Every real
  `recommended_subset` N/K already satisfies this (checked exhaustively,
  same table `marlin/adapter.py` checked) — padding only fires on 2 of 3
  `--smoke` shapes, exact for the gate (zero-coded padding decodes to
  exactly 0.0).
- `prepare()`: regenerates `Wtrue`, quantizes to FP6, bit-packs, calls the
  artifact's own `weight_dequant_eXmY_cpu` (a function SEPARATE from
  `linear_forward_eXmY_cuda`, preserving DOMAIN_GUIDE's reference-
  independence rule) to build `params["dequantized_W_override"]`, then
  the artifact's own `weight_prepacking_eXmY_cpu` for the kernel's actual
  tile-interleaved input (this IS preprocessing, ARTIFACT_GUIDE rule 2,
  timed as such).
- `run()`: one call to `fp6_llm.linear_forward_eXmY_cuda(3, 2, A, W, s, 1)`
  (`splitK=1`, no autotuning of the split-K heuristic — a login-node
  functional check, not a timed sweep).

## Gate results (login node, functional check only)

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel quantized-gemm --variant qgemm-w4a16-decode-kernel \
    --impl fp6llm-fpx-gemm --smoke
```

**3/3 valid**, tolerance `1e-3` (parsed):

| shape | max_scaled_err |
|---|---|
| smoke-qgemm-w4-g128-decode (M=8,K=256,N=384) | 1.12e-04 |
| smoke-qgemm-w4-percol-decode (M=4,K=96,N=128) | 1.92e-04 |
| smoke-qgemm-w3-g64-batched (M=64,K=192,N=256) | 1.51e-04 |
| qgemm-llama-2-7b-qkv_proj-decode-m1 (real, M=1,K=4096,N=12288) | 2.68e-05 |
| qgemm-mistral-7b-gate_up_proj-decode-m16 (real, M=16,K=4096,N=28672) | 5.00e-05 |

All comfortably within tolerance — same order of magnitude as
`numpy-dequant-gemm`'s ~1e-4 baseline, appropriately larger than marlin's
INT4 error at the smoke shapes (FP6's 2-bit mantissa is coarser than a
well-chosen per-group INT4 scale, as expected).

`qgemm-w4a16-batched-kernel` spot-check (unparsed tolerance, same
track-wide gap tilus/qfactory/marlin already document): M=128 on the same
real shape gives `max_scaled_err=5.27e-05` — **no sign of marlin's
`par>1` precision regression** (see `marlin/STATUS.md`) at this or any
other M tested; FP6-LLM's own batch dispatch (`N_PowerOf2` rounding,
`fp6_linear.cu`) is structurally different from marlin's parallel-batch-
of-64 split and does not appear to share its fragility.

## Not done (out of this integration's login-node budget)

- No sweep across the full `recommended_subset` registry — a handful of
  representative shapes at both decode and batched M, per ARTIFACT_GUIDE
  rule 5.
- FP5 (E2M2) is implemented in `_quantize_fpx_code`/wired via
  `_EXPONENT`/`_MANTISSA` constants but not exercised by this integration
  (no workload-level switch exists to request it — see "always FP6"
  above); flipping the two constants and re-running the same smoke set
  would exercise it if a future spec revision adds an FP5-specific
  variant.
- `qgemm-model-accuracy` (perplexity delta) is out of scope per `ml.py`'s
  own module docstring — not attempted here either.
- The extreme Falcon-180B `fused_fc2` shape was not attempted (same
  fp64-host-memory concern documented in `marlin/STATUS.md` applies
  equally to `reference_qgemm`'s fp64 path here).

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100-SXM4-40GB (sm_80, full card, job on partition
  `gpu`), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14. No arch-flag override needed (`setup.py`
  already hardcodes `sm_80`, matching this GPU too).
- Build: OK. Build-system changes: `build.sh` — (1) the hardcoded
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` on the import-verification line
  was changed to `LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libstdc++.so.6}"`
  (reads the already-exported knob first; zaratan's `env.sh` exports a
  different, ABI-correct libstdc++ than Perlmutter's `/usr/lib64` one, so
  the old hardcoded line would have overridden it and broken the import
  check specifically on this machine); (2) `source .../toolchain.sh` and
  `cd "$(dirname ...)"` were swapped back to toolchain-then-cd order — the
  script had `cd` BEFORE sourcing toolchain.sh via a path relative to
  `BASH_SOURCE[0]`, which resolves against the pre-`cd` cwd, so once cwd
  changed the `../../toolchain.sh` lookup pointed at a nonexistent path
  when build.sh is invoked as `bench/artifacts/.../build.sh` from `$REPO`
  (as this reproduction protocol requires) — a pre-existing script-ordering
  bug, not a zaratan-specific issue, just never previously exercised with
  this invocation style.
- Gate: qgemm-w4a16-decode-kernel/fp16: PASS, 3/3 smoke shapes valid
  (err 1.12e-04, 1.92e-04, 1.51e-04, tol 1e-3) + 2 real recommended_subset
  shapes (`qgemm-llama-2-7b-qkv_proj-decode-m1` err 2.68e-05,
  `qgemm-mistral-7b-gate_up_proj-decode-m16` err 5.00e-05, tol 1e-3) — all
  5/5 valid, errors match the recorded numbers exactly.
- Deviation from the recorded ruling: none (identical error values to the
  Perlmutter run, same order of magnitude, same tolerance margin).
- Verdict here: BUILT+GATED — equals the recorded ruling.
