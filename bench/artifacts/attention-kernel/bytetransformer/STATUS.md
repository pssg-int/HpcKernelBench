# bytetransformer (ByteTransformer) — attention-kernel

**Status: BUILT+GATED**

- Paper: "ByteTransformer: A High-Performance Transformer Boosted for
  Variable-Length Inputs", IPDPS'23. `PAPER_KEY = conf/ipps/ZhaiJWJZCLZ23`.
- Selection rationale: core baseline under the revised kernel-centrality
  rule (`kernel-papers/output/baseline_selection.md`) — centrality `core`
  (the fused padding-free MHA kernel IS the paper's headline contribution,
  evaluated at kernel level against PyTorch/FasterTransformer/TurboTransformers
  in the paper), regime `matches` (BERT-style bidirectional MHA, fp16, GPU),
  single-NVIDIA-GPU path.
- Artifact: https://github.com/bytedance/ByteTransformer.
- Commit cloned: `44c66ae8d88719f463679baf631d0f2cb0dadfff` (`main`),
  `git clone --depth 1`.
- Toolchain: python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`,
  torch `2.8.0+cu128`, nvcc `12.9`
  (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`), host compiler
  `/opt/cray/pe/gcc-native/14/bin/g++` (GCC 14), GPU arch `sm_80` (A100,
  `-gencode=arch=compute_80,code=sm_80`).
- **No CUTLASS dependency** — see "What was built" below for why. The task
  brief anticipated needing to fetch a pinned CUTLASS checkout (precedent:
  `pat/build.sh`); reading the actual translation-unit dependency graph of
  the specific kernel this adapter wraps showed CUTLASS is not on it.

## What the artifact actually is, and the wrapping boundary chosen

ByteTransformer's `th_op/` exposes a torch custom-class op
(`bytetransformer::torch_ths::BertTransformer`), but that wraps a whole BERT
**encoder layer** — QKV projection GEMM, the fused-attention kernel, output
projection GEMM, two LayerNorms, and the FFN block (`gemm.cu`/
`gemm_bias_act.cu`, CUTLASS-backed for the fused-bias-activation variant) —
not the attention kernel alone. Per ARTIFACT_GUIDE rule 1 ("wrap the kernel,
not the paper's benchmark script/pipeline"), this integration goes one level
finer: `bytetransformer::Attention<OperationType::HALF>` is itself a
standalone class (constructed once per call, no BertTransformer needed), and
its `fused_rm_infer(AttentionInferParam)` / `fused_long_rm_infer(...)` member
functions ARE the fused, PADDING-FREE (remove-padding) multi-head
self-attention kernel — one CUDA kernel launch per call (`wmma_attention_rm_
kernel` or `wmma_attention_long_rm_kernel`, `attention_fused.cu` /
`attention_fused_long.cu`), doing `softmax(QK^T/sqrt(d)) @ V` for a whole
packed batch in one launch.

`bench/artifacts/attention-kernel/bytetransformer/attn_binding.cu` is a thin
pybind11 wrapper WE wrote (not part of the artifact; lives outside `source/`
per rule 3) that constructs an `Attention<HALF>` object directly and calls
`fused_rm_infer`/`fused_long_rm_infer` — reproducing only the
seq_len<=80-vs->80 dispatch threshold `Attention<OpType>::infer()` itself
uses (see attn_binding.cu's own docstring for the full derivation of why we
call these two functions directly rather than `infer()`, and why that still
required compiling `attention_nofused.cu`/`attention_nofused_utils.cu`/
`gemm.cu` — none of which touch CUTLASS — to satisfy the linker: `Attention`
has a virtual destructor and two virtual methods defined inline in
`attention.h`, so instantiating/destroying an `Attention<HALF>` object emits
a full vtable, which references `nofused_infer`/`fused_infer`/
`fused_long_infer` too (`infer()`'s dead branches) even though none of them
ever execute for our objects — confirmed by a first build attempt failing
at LINK time with `undefined symbol ...nofused_infer...`).

Our own `setup.py` (NOT the artifact's CMake) compiles exactly:
`attn_binding.cu` + `source/bytetransformer/src/{attention_fused,
attention_fused_long,attention_nofused,attention_nofused_utils,gemm}.cu`.
Grepping the whole `bytetransformer/` tree for `cutlass` confirms none of
these five files (or the headers they pull in — `attention.h`, `reduce.h`,
`common.h`, `gemm.h`, `variety_attention_fused.h`, `attention_nofused_utils.h`)
reference it; only `bert_transformer.h`, `cutlass_attention*.h/.cu` and
`gemm_bias_act.*` do, and none of those are compiled here.

## Boundary and coverage limits (real constraints of the ARTIFACT, not this adapter)

- **`size_per_head` (d) must be exactly 64.** Every `WMMA_ATTENTION_RM` /
  `WMMA_ATTENTION_LONG_RM` macro invocation in `attention_fused.cu`/
  `attention_fused_long.cu` hardcodes the `SIZE_PER_HEAD` template argument
  to the literal `64`, independent of what's passed to the `Attention<>`
  constructor — confirmed by reading every macro call site.
- **`seq_len` (S) must be in [1, 352].** `fused_rm_infer`'s own `switch`
  only has cases for S in `{16,32,48,64,80}`; `fused_long_rm_infer`'s only
  for `split_count` 6..22, i.e. S in `(80,352]`. Outside that range NEITHER
  function launches any kernel — the switch falls through and `output`
  stays uninitialized. `adapter.py::prepare()` raises `NotImplementedError`
  for S outside `[1,352]` rather than let that happen.
- **No GQA** (`H == Hkv` required) — the kernel has one `head_num`, no
  separate KV head count.
- **Mask must be `"bidirectional"`.** `wmma_attention_rm_kernel`'s own
  softmax loop applies no masking beyond each sequence's own length — its
  `attention_mask` parameter is accepted but never dereferenced in the body
  (confirmed by reading the full kernel). A causal workload would silently
  get the wrong (fully bidirectional) answer if allowed through, so causal
  is refused rather than silently mismeasured.
- **Prefill only** (`variant_kind=="prefill"`, `Sq==Sk>1`) — this is a
  batched self-attention kernel over one packed token set, not a KV-cache
  decode kernel; PAT already covers this track's decode variant.
- **fp16 only.** `Traits<OperationType::HALF>::DataType = __half`; the
  artifact's own `Attention` constructor forces `use_fused_attention_=false`
  whenever `OpType==FP32`, i.e. FP32 never reaches this kernel family at
  all — not a gate loosened, a fact about the artifact.

Combined with `benchspecs/attention-kernel/spec.yaml`'s own
`attn-prefill-kernel-fp16bf16` `recommended_subset`, exactly ONE of the 12
listed shapes satisfies all of the above: `bert-base-short`
(B=32,H=Hkv=12,d=64,S=128,mask=bidirectional) — S=128 lands in the "long"
bucket (`112<S<=128`), so gating it also exercises `fused_long_rm_infer`,
not only the simpler short-sequence kernel.

## Padding-free / degenerate-input handling (per this task's own instruction)

`kernelbench.domains.ml.AttentionWorkload` models a dense `(B,H,S,d)` batch
with every sequence the SAME length — there is no padding to remove. Per
this task's instruction, this adapter feeds exactly that: `batch_idx =
[0, S, 2S, ..., B*S]`, the correct cu-seqlens-style offsets for "every
sequence already has its real, full length, nothing padded". This is a
real, artifact-supported input (the paper's own contribution is precisely
that variable-length batches need no padding to reach full utilization; a
batch where every length happens to be equal is the boundary case that
contribution must still handle correctly), not a synthetic workaround.

## Preprocessing done in `prepare()` (ARTIFACT_GUIDE rule 2)

1. **QKV packing.** The kernel wants one packed `qkv` tensor of shape
   `[B*S, 3*hidden_dim]` (`hidden_dim=H*d`), token-major, with Q/K/V
   concatenated along the last axis and heads concatenated within each
   segment — derived from `attention_fused.cu`'s own pointer arithmetic
   (`pos = token*(hidden_dim/2 half2 units)*3 + head*(d/2) + ...`), cross-
   checked against `th_op/ths_op.h`'s `qkv_kernel` weight-shape comment
   (`hidden_dim, hidden_dim*3`). `prepare()` builds this from the harness's
   independent `(B,H,S,d)` Q,K,V via `transpose(0,2,1,3).reshape(...)` +
   `concatenate`.
2. **Zero QKV bias.** The kernel always adds a QKV bias internally
   (`__hadd2(qkv_val, bias)`); `reference_attention`/this track's operand
   convention model no bias term, so a zero bias tensor is fed — this makes
   the kernel's computed math equal ours EXACTLY (not an approximation),
   the same technique gpa/pat use for parameters their own reference
   doesn't model.
3. Both are done ONCE per `prepare()` call and timed as preprocessing, not
   inside `run()`, per rule 2.

RNG: Q/K/V generated bit-identically to `kernelbench.domains.ml._qkv` (same
`np.random.default_rng` seed, shapes, call order) — independently re-typed
in `adapter.py`, per the RNG-discipline precedent in pat/gpa/fused3s's
adapters (trap documented in `bench/artifacts/spmm/insum/STATUS.md`).

## Build-system patch recorded (no kernel-source change)

Torch's `CUDAExtension` unconditionally injects
`-D__CUDA_NO_HALF{,2}_{OPERATORS,CONVERSIONS}__` ahead of any
`extra_compile_args` (to avoid ODR clashes between raw CUDA half ops and
ATen's own overloads when a `.cu` file also includes torch headers).
ByteTransformer's own `reduce.h` (unmodified) relies on the DEFAULT implicit
`__half`<->`float` conversion the artifact was written against
(`(float)__half2add(shared[lane])`, where `__half2add` returns `__half`),
which those macros disable — the first build attempt failed with "no
suitable conversion function from `__half` to `float` exists" in
`reduce.h`. Fix (`setup.py`'s `extra_compile_args["nvcc"]`): append
`-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__` — since
`-U` appears after `-D` on the same nvcc command line, this undoes exactly
those 4 macros for this build. A build-flag fix (ARTIFACT_GUIDE rule 3:
"CUDA-version guards are fine"), no kernel source file touched.

Two other machine-specific workarounds, identical derivation to every other
torch-CUDA-extension build on this machine (`pat`, `sddmm/fused3s`): this
venv's python bakes a `-B .../compiler_compat` into distutils' `CXX`
(routes g++ through an ancient `cc1plus`, fails torch's GCC-version check)
and bakes an old-libstdc++ rpath into `LDSHARED` — fixed by pointing
`CXX`/`CC` at the real system compiler and overriding `LDSHARED` with a
clean link line into this venv's own `torch/lib`.

**Running this adapter requires `LD_PRELOAD=/usr/lib64/libstdc++.so.6` in
the invoking shell** (same `CXXABI_1.3.15` trap as every other torch
extension on this machine — `available()` reports the fix in its reason
string).

## Verification (login node, correctness gate only, reduced protocol)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
$PY -m kernelbench.runner --kernel attention-kernel \
    --variant attn-prefill-kernel-fp16bf16 --matrices bert-base-short \
    --impl bytetransformer-fused-rm-attn --precision fp16 --warmup 2 --reps 3
```

Result: **1/1 valid**, `max_scaled_err = 2.61e-03` (tolerance `1e-2`) — a
clean pass with margin consistent with fp16 rounding, exercising
`fused_long_rm_infer` (S=128 falls in the `112<S<=128` bucket).

An additional direct-harness check (not through `runner.py`, same gate
machinery) exercised the OTHER kernel family, `fused_rm_infer`
(short-sequence, S<=80): `B=2,H=Hkv=4,d=64,S=64,mask=bidirectional` ->
**valid**, `max_scaled_err = 1.49e-03`. Both dispatch paths this adapter can
reach are therefore independently confirmed correct, not just the one path
the spec's own `recommended_subset` happens to exercise.

`--smoke` (this track's own synthetic shapes, all `d=16`) correctly reports
all 4 as `UNSUPPORTED` (`d=16 != 64`, causal mask on the rm kernel, or
`variant_kind="decode"`) rather than a silent wrong answer — see the
"Boundary and coverage limits" section above.

Not spec-conforming (login node, overridden warmup/reps, shared GPU) —
timing numbers from this run must not be published, only the correctness
result.

## Not done

- The `attention_nofused`/`gemm`/`variety_attention_fused` code paths
  (dead at runtime for this adapter's objects, needed only to satisfy the
  linker — see "What the artifact actually is" above) were not themselves
  gated; they are never executed by any call this adapter makes.
- `cutlass_attention.cu`'s CUTLASS-based attention variant (a THIRD
  attention implementation in this repo, used by `bert_transformer.cu` for
  non-BERT model types / other configurations) was not investigated —
  `fused_rm_infer`/`fused_long_rm_infer` are the paper's own headline
  "padding-free" contribution and were the more directly relevant target.
- Coverage is 1 of 12 `attn-prefill-kernel-fp16bf16` `recommended_subset`
  shapes (`bert-base-short`) plus one ad hoc short-bucket shape; the
  artifact's own real scope limit (d=64, S<=352) rules out the other 11
  registry entries (d=128 or S>352), not an integration shortcut.
- Timing sweeps and compute-node runs: none — per ARTIFACT_GUIDE rule 5,
  login node is build + correctness gate only.

## Provenance

- Artifact commit: `44c66ae8d88719f463679baf631d0f2cb0dadfff`
- nvcc: 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`)
- Host compiler: g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`)
- GPU arch: `sm_80` (A100, `-gencode=arch=compute_80,code=sm_80`)
- Python/torch: `/pscratch/sd/c/cunyang/gnn/plexus_env` — torch 2.8.0+cu128

## Reproduction on zaratan (2026-09-08)

- Machine: UMD zaratan, NVIDIA A100-SXM4-40GB (sm_80, driver 595.71),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8 toolkit, `$KB_CUDA_HOME`),
  g++ 13.4.0 (conda-forge, `$KB_CXX`), torch 2.8.0+cu128, Python 3.12.14,
  arch flags unchanged (`-gencode=arch=compute_80,code=sm_80`).
- Build: OK. No build-system changes needed beyond what `build.sh` already
  reads from `${PY:-...}`/`${CUDA_HOME:-...}`/`${CXX:-...}`/`${CC:-...}`
  (all supplied by `bench/env.sh`/`toolchain.sh` on this machine) — none.
- Gate: `attn-prefill-kernel-fp16bf16` / `bert-base-short` (fp16, impl
  `bytetransformer-fused-rm-attn`, `--warmup 2 --reps 3`): **PASS**,
  `max_scaled_err = 2.61e-03 <= tol 1e-2`, 1/1 runs valid. Identical to the
  originally recorded result.
- Deviation from the recorded ruling: none.
- Verdict here: **BUILT+GATED** — equals the recorded ruling.
