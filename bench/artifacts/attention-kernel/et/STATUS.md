# et (E.T.) — attention-kernel

**Status: BUILT+GATED (with a confirmed artifact bug — correctness gate FAILS)**

- Paper: "E.T.: Re-Thinking Self-Attention for Transformer Models on GPUs",
  SC'21. `PAPER_KEY = conf/sc/ChenHPLG0D021`.
- Selection rationale: core baseline under the revised kernel-centrality
  rule (`kernel-papers/output/baseline_selection.md`) — centrality `core`
  (the re-engineered self-attention operator is one of the paper's two
  headline contributions, evaluated against TensorRT/FasterTransformer at
  kernel level), regime `matches` (BERT/DistilBERT self-attention, fp16,
  GPU), single-NVIDIA-GPU path.
- Artifact: https://github.com/cctry/E.T.
- Commit cloned: `132579801559d3a32055e15da70cb381827964e6` (`main`),
  `git clone --depth 1`.
- Toolchain: python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`,
  torch `2.8.0+cu128`, nvcc `12.9`
  (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`), host compiler
  `/opt/cray/pe/gcc-native/14/bin/g++` (GCC 14), GPU arch `sm_80` (A100,
  `-gencode=arch=compute_80,code=sm_80`). The artifact's own README states
  it was tested on V100S/CUDA 11.4; this is the first time it has been run
  on an sm_80/Ampere part, per the "What was found" section below.
- No CUTLASS dependency — `kernels/attention.cu` only includes `utils.h`
  (`nvcuda::wmma` + `cooperative_groups`, header-only).

## What the artifact actually is, and the wrapping boundary chosen

E.T. ships **no python bindings at all** — only CMake C++ test binaries
(its own README: "a few examples of encoder"). It provides three
self-attention kernel launchers in `kernels/attention.cu`
(`kernels/kernels.h`):
`OTF_attention_kernelLauncher` (dense, "on-the-fly" self-attention),
`Prune_attention_kernelLauncher` (attention-aware structured-pruned V, a
separate contribution), and `sharedQK_attention_kernelLauncher` (consumes
an externally-precomputed QK product via a separate cuBLAS
strided-batched GEMM, used by the "sequence-aware" encoder). Per this
task's own instructions, only the **dense** path is wrapped:
`OTF_attention_kernelLauncher`.

`bench/artifacts/attention-kernel/et/attn_binding.cu` is a thin pybind11
wrapper WE wrote (E.T. has none of its own) calling
`OTF_attention_kernelLauncher` directly — ONE CUDA kernel launch per call
(`__kernel_multi_head_full_skew_warpSFM`), confirmed as the correct
boundary from `encoder/encoder.cu`'s `Encoder_tile::run()`, which calls
this same function directly (no wrapping class needed; going through
`Encoder_tile` would additionally run its tile-pruned Q/K/V/O/MLP linear
layers and LayerNorms — an encoder LAYER, not this kernel).

The kernel has no batch dimension in its own grid (`dim3(seq_len/16,
nhead)`) — `adapter.py::run()` loops over the workload's B in Python,
calling the binding once per batch item, matching the gpa/spfa_csr
adapter's identical "loop over independent per-item kernel calls"
precedent; the same fixed additive `(S,S)` mask is reused across every
batch item.

## Real, load-bearing constraints of the artifact (not this adapter), enforced as `NotImplementedError`

- `seq_len` must be a multiple of 16 (`grid.x = seq_len/16`, integer
  division, no remainder handling).
- `head_dim` (`d`) must be a multiple of 16 (WMMA 16x16x16 tiling).
- `H == Hkv` (no GQA — the launcher takes one `nhead`).
- dynamic shared memory (`16*(head_dim+16) + 16*(seq_len+16)` halfs) must
  fit CUDA's default 48KB limit — `OTF_attention_kernelLauncher` never
  calls `cudaFuncSetAttribute` to raise it (unlike, e.g., ByteTransformer's
  long-sequence kernel), so large `(head_dim, seq_len)` combinations would
  fail the launch itself.
- fp16 only — every buffer in `attention.cu` is `half`.
- prefill only (`Sq==Sk>1`) — no batch dimension, no KV-cache concept.

Unlike bytetransformer's `fused_rm` kernel, this kernel's masking IS fully
general (an additive `(S,S)` matrix, confirmed from the kernel's own
`temp_row2[...] += mask_base[...]` line) — both `"causal"` and
`"bidirectional"` are supported.

## PATCH recorded (header syntax only, not kernel logic)

`kernels/kernels.h`'s last declaration
(`sharedQK_attention_kernelLauncher(...)`) is **missing its terminating
semicolon** in the artifact's own upstream repo (confirmed: `git diff`
against the pristine clone shows `\ No newline at end of file` immediately
after the closing `)`, i.e. the file genuinely ends there in the checked-in
source). Any translation unit that includes this header AFTER that point
(our `attn_binding.cu`) fails to parse — the next declaration's tokens get
silently absorbed into this one, producing a cascade of unrelated-looking
errors (`"out" has already been declared`, `expression must have class
type`, ...). Fixed with a one-character `sed` in `build.sh` (add `;`),
applied idempotently on every build; diff recorded in `source.patch`. No
computation changed — a header-syntax-only, build-system-level fix
(ARTIFACT_GUIDE rule 3).

A second build-system fix (not a `source.patch` entry — a compiler-flag
override, no source file touched): torch's `CUDAExtension` unconditionally
injects `-D__CUDA_NO_HALF{,2}_{OPERATORS,CONVERSIONS}__`, which breaks
`attention.cu`'s own use of the default `half2` arithmetic operators
(`sum2.x + sum2.y`, `half2{(half)1.0f,(half)1.0f}`, `dst2[i] *
one_over_sum`, `wmma::fill_fragment(c_frag, (half)0.0f)`). `setup.py`
appends `-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__` to
`extra_compile_args["nvcc"]` (same fix as bytetransformer's `setup.py`).

## What was found: a confirmed correctness bug in the artifact's own `softmax_blk`

The correctness gate fails hard (`max_scaled_err` in the 0.47–1.7 range
against a `1e-2` tolerance, on THREE independent real shapes — see
"Verification" below). This is not a wrapping-boundary or layout mistake
in `adapter.py`: it was isolated to a specific, exactly-reproducible bug in
the artifact's own `softmax_blk()` helper (`kernels/attention.cu`, used by
`__kernel_multi_head_full_skew_warpSFM`, i.e. `OTF_attention_kernelLauncher`
itself — and also by `__kernel_multi_head_prune`'s `Prune_attention_
kernelLauncher` and, via its own separate inline copy of the same pattern,
`__kernel_multi_head_sharedQK`'s `sharedQK_attention_kernelLauncher` — all
THREE of the repo's attention kernels share it):

```cpp
const auto temp = h2exp2(__hsub2(row_ptr2[i], max2));
```

**`h2exp2` computes `2^x` (base-2 exponential), not `e^x`** (natural
exponential) — CUDA's `e^x` intrinsic is `h2exp`, a different function.
Softmax is defined as `exp(x_i)/sum(exp(x_j))` (Euler's number base); using
`2^x` instead computes a different, still row-normalized function that is
systematically FLATTER than the true softmax by an effective factor of
`ln(2) ≈ 0.693` in the logit scale (since `2^x = e^(x·ln 2)`), with no
compensating `×log2(e)` factor applied to the logits/scale anywhere in the
kernel to correct for this (the standard trick for deliberately using
`exp2` for speed is to pre-multiply the exponent by `log2(e)=1/ln(2)`
first, recovering the exact `e^x` result — that pre-multiplication is
simply absent here).

### Isolation (all steps confirmed independently against the compiled kernel via `et_attn_binding`, not inferred)

1. **Row-stochastic normalization is exact.** `V = ones(S,hidden)` ->
   kernel output ≈ `1.0` for every element (mean `0.9998`, std `6e-4`) —
   the sum-then-divide normalization itself is correct.
2. **Mask application and the P@V matmul are exact.** An "identity" mask
   (`-inf` off-diagonal, `0` on-diagonal, forcing each row's softmax weight
   entirely onto its own diagonal position regardless of logit values) ->
   kernel output matches `V` to `4.9e-4` max abs diff (pure fp16 rounding).
3. **Extreme-dominant-logit case is exact.** One key given a logit ~256
   larger than all others -> kernel output matches that key's `V` row
   EXACTLY (`0.0` max abs diff) — both `2^x` and `e^x` softmax converge to
   the same one-hot answer in this limit, so this test alone cannot
   distinguish the two, but confirms the QK^T/PV machinery is otherwise
   sound.
4. **Moderate, well-separated logits reveal the bug directly.** Using the
   "identity-V" trick (`V = eye(S)`, so `output[i,:] == P[i,:]` exactly,
   the kernel's actual softmax row read out directly) with **hand-computed,
   exact logits `[0, 1, 2, 3]`** (via one-hot `Q`/`K` columns, avoiding any
   ambiguity from fp16-rounded random dot products): the kernel's returned
   weights are `[0.0667, 0.1333, 0.2666, 0.5332]`, i.e. **exactly
   proportional to `[1, 2, 4, 8] = [2^0, 2^1, 2^2, 2^3]`**, normalized
   (`1+2+4+8=15`; `1/15=0.0667`, `8/15=0.5332` — matches to the last printed
   digit). The mathematically correct softmax of `[0,1,2,3]` is
   `[0.0321, 0.0871, 0.2369, 0.6439]` (`e^x`-based, ratios `[1, e, e^2,
   e^3]`). This is not "close to" `2^x` normalization — it is bit-for-bit
   what `2^x`-based softmax produces for this exact logit vector, which is
   as close to a mathematical proof as an empirical test can get without
   reading the compiled SASS.
5. Cross-checked against `random N(0,1)` Q/K (the harness's own operand
   convention) via the same identity-V readout: the kernel's row
   distribution is visibly flatter than the true softmax's (e.g. row 0's
   top weight `0.217` vs. the reference's `0.301` for the same inputs) —
   consistent with, not merely suggestive of, the `ln(2)`-scale-factor
   explanation from point 4.

This is a property of the artifact's own released kernel source (not a
build-flag, not GPU-architecture-dependent behavior of `h2exp2` itself —
`h2exp2` computes exactly what its name says on every architecture; the
bug is calling it where `h2exp`/`hexp` was needed), not something
introduced by this integration's wrapping, and not something patched here
(ARTIFACT_GUIDE rule 3: patch build systems, not kernel logic — a fix here
would mean rewriting the kernel's actual math, which rule 3 explicitly
reserves for a SKIPPED verdict, and rule 4 says a failing gate IS the
result, not something to work around).

## Verification (login node, correctness gate only, reduced protocol)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
$PY -m kernelbench.runner --kernel attention-kernel \
    --variant attn-prefill-kernel-fp16bf16 \
    --matrices bert-base-short,bert-large,gpt2-small \
    --impl et-otf-attn --precision fp16 --warmup 2 --reps 3
```

Result: **0/3 valid** —
- `bert-base-short` (B=32,H=12,d=64,S=128,bidirectional): `max_scaled_err = 4.893e-01`
- `bert-large` (B=16,H=16,d=64,S=384,bidirectional): `max_scaled_err = 4.703e-01`
- `gpt2-small` (B=8,H=12,d=64,S=1024,causal): `max_scaled_err = 1.733e+00`

All three fail against tolerance `1e-2`, consistent in magnitude with the
`ln(2)`-scale-factor explanation above (a systematic, not random, error —
note it does NOT shrink with more warmup/reps, since it is not a timing or
noise artifact). Per ARTIFACT_GUIDE rule 4, this failing gate IS the
recorded result — not loosened, not worked around. Not spec-conforming
regardless (login node, overridden warmup/reps, shared GPU).

## Not done

- `Prune_attention_kernelLauncher` and `sharedQK_attention_kernelLauncher`
  were not wrapped, per this task's instruction to wrap the dense
  self-attention path only — both share the same `softmax_blk`/inline
  `h2exp2` pattern and would be expected to fail the same gate for the
  same reason, but this was not independently verified for either.
- No attempt was made to "fix" the kernel's softmax (out of scope per
  ARTIFACT_GUIDE rule 3); the gate failure is reported as the finding.
- Timing sweeps and compute-node runs: none — login node is build +
  correctness gate only.

## Provenance

- Artifact commit: `132579801559d3a32055e15da70cb381827964e6`
- Patch: `source.patch` (one-line semicolon fix to `kernels/kernels.h`)
- nvcc: 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`)
- Host compiler: g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`)
- GPU arch: `sm_80` (A100, `-gencode=arch=compute_80,code=sm_80`)
- Python/torch: `/pscratch/sd/c/cunyang/gnn/plexus_env` — torch 2.8.0+cu128

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, NVIDIA A100-SXM4-40GB (sm_80, driver 595.71.05), gpu
  partition (`bench/gpu_run.sh`, MIG/full-card slice per job), login-node
  build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8, `$KB_CUDA_HOME`), g++
  13.4.0 (conda-forge, `$KB_CXX`), torch 2.8.0+cu128, Python 3.12.14, no
  cmake used (setup.py/CUDAExtension build, same as recorded). Arch flag:
  `-gencode=arch=compute_80,code=sm_80` (unchanged, `setup.py`'s own
  choice).
- Build: OK — extension was already built from an earlier pass on this
  machine; `build.sh` reported "extension already built, skipping" (idempotent
  no-op, per rule). No build-system changes needed beyond what's already in
  `source.patch` (the semicolon fix) and `setup.py` (the `-U__CUDA_NO_HALF*`
  flags), both already present and unchanged.
- Gate: `attn-prefill-kernel-fp16bf16`, `et-otf-attn`, fp16, `--matrices
  bert-base-short,bert-large,gpt2-small --warmup 2 --reps 3` (same reduced
  protocol as the recorded run): **0/3 valid** —
  `bert-base-short` FAIL `max_scaled_err=4.893e-01` vs tol `1e-2`;
  `bert-large` FAIL `max_scaled_err=4.703e-01` vs tol `1e-2`;
  `gpt2-small` FAIL `max_scaled_err=1.733e+00` vs tol `1e-2`.
- Deviation from the recorded ruling: none — the three `max_scaled_err`
  values reproduce the Perlmutter run to the last printed digit, confirming
  the `h2exp2`-vs-`h2exp` softmax bug is a property of the artifact's own
  kernel source, not of the Perlmutter A100/nvcc-12.9/gcc-14 toolchain (this
  run used a different A100 SKU, a different nvcc minor, and a different gcc
  major — same deterministic failure).
- Verdict: BUILT+GATED (gate fails: confirmed `h2exp2`/softmax bug in the
  artifact's own `softmax_blk`) — same as recorded ruling.
