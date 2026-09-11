# packkv (PackKV) — gemv

**Status: BUILT+GATED err=7.87e-05..1.44e-03 (all <= tol 1e-2), 3/3 smoke shapes valid**

- Paper: "PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy
  Compression", IPDPS'26. `PAPER_KEY = conf/ipps/JiangYLHDJ26` (verified
  against `kernel-papers/output/included.json` by title).
- Artifact: https://github.com/BoJiang03/PackKV
- Commit cloned: `da3c00d986d1e37b5c3a7e2ce9403bbf2ab6397f`, `git clone --depth 1`
  (upstream's own commit date `2026-01-30`).
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`), GPU `sm_80`
  (A100-PCIE-40GB). Python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`
  (3.11.7), torch `2.8.0+cu128`. `CC=/opt/cray/pe/gcc-native/14/bin/gcc`,
  `CXX=/opt/cray/pe/gcc-native/14/bin/g++`. `LD_PRELOAD=/usr/lib64/
  libstdc++.so.6` required at runtime (same CXXABI/GLIBCXX symbol-version
  mismatch documented for marlin/tilus/qfactory).
  **Toolchain-pin hardening (2026-09-04):** this login node's default module
  changed to `cudatoolkit/13.2` (HPC SDK 26.5) partway through this track's
  integration work, which would silently leak CUDA 13.2's `nvcc`/headers
  into any (re)build against plexus_env's cu128 torch. `build.sh` now
  sources the shared `bench/artifacts/toolchain.sh` (pins `CUDA_HOME`/`PATH`/
  `CPATH` to the SDK's 12.9 tree, re-pointing rather than clearing `CPATH`
  since torch's ATen/cuda headers need `cusparse.h` from the matching
  `math_libs` tree) as a one-line hardening; this vendor/ build predates
  that change and was NOT rebuilt (it was already built correctly against
  12.9 in the prior session and still imports/runs correctly under
  `LD_PRELOAD` — verified below). `gemv/marlin/build.sh` received the same
  hardening line for consistency across this track.

## What the artifact actually is

PackKV is a real ahead-of-time CUDA extension
(`source/packkv_cuda_ext/`, built via `torch.utils.cpp_extension.
CUDAExtension`, same family as marlin's build, unlike quantized-gemm's
runtime-JIT `tilus`/`qfactory`) implementing two independent codec+kernel
pairs for LLM KV-cache compression: a K-cache path (`k_encode_cpu`/
`k_decode_cpu` + `kq_mat_vec_mul`/`fused_kq`, producing the attention
score `K^T@q`) and a V-cache path (`v_encode_cpu`/`v_decode_cpu` +
`wv_mat_vec_mul`/`fused_wv`, producing the attention-output contribution
`W^T@V`). This integration wraps **only the K path** — it is the one that
is structurally a plain GEMV (`y = A@x`, per the task brief's own mapping);
the V path computes a weighted-sum-of-rows operation over softmax
attention weights, a different operation not asked for here.

`k_encode_cpu`/`k_decode_cpu` are genuine CPU-side bit-packing (a per-tile
adaptive-bit-length scheme, `k_encode_cpu.cpp`: within each 64x128 tile it
further reduces the stored bit width per small 16-row "round" based on that
round's own local min/max — a real, additional compression layer beyond
the outer 4-bit affine quantization). `kq_mat_vec_mul` + `fused_kq` are the
GPU-side fused decompression+dot-product kernel: `kq_mat_vec_mul` reads
the COMPRESSED buffer directly (never materializing a decompressed dense K
matrix) and computes partial codes-dot-q products; `fused_kq` folds in the
per-row zero-point/scale to produce the final real-valued attention score.
Both called here **unmodified** — no copy, no kernel edit (rule 1/3).

## Variant: `gemv-e2e-compressed-format`, and why

`benchspecs/gemv/spec.yaml`'s `gemv-e2e-compressed-format` variant is
explicitly for "GEMV embedded in a compressed/structured storage format
where decompression is NOT separable from the per-call kernel by the
algorithm's own design" — `survey.md` names PackKV by name for exactly this
variant ("PackKV's kernel-vs-cuBLAS comparison times fused
decompression+compute together BY CONSTRUCTION... there is no
already-decompressed state to start the timer at"). This is precisely
PackKV's K path: `kq_mat_vec_mul` never produces a decompressed dense K at
any point, so there is nothing to "already have decoded" before the timed
call — matching this variant's parsed tolerance **1e-2** (the spec's
correctness field's first `<` number, from the TLR-MVM half of that field's
prose; verified via `spec.load("gemv").variant("gemv-e2e-compressed-
format").tolerance == 0.01`, confirmed live via `--list` below).
Preprocessing (timed once, in `prepare()`) is genuinely just: quantize K
into codes and encode those codes into PackKV's own compressed block
format (`k_encode_cpu`) — matching this variant's own protocol field
("PackKV has no separable preprocessing to report [beyond
quantize+encode]"). The decode+dot-product chain (`kq_mat_vec_mul` ->
`fused_kq`) is the timed kernel call in `run()`.

## Shape mapping

`y[M] = A[M,K] @ x[K]` maps onto PackKV's own K-cache dot product with
`A` = the (uncompressed, per-request) K cache `[ctx_len=M, hidden_dim=K]`
and `x` = the query vector `q[K]` — the literal mapping given in the task
brief, and exactly what `evaluation.py::k_cpu_compress_gpu_mat_vec_mul`
computes for one real attention head
(`target_kq[h,:] = K[h,:,:] @ q[h,:]`).

### The artifact's kernel only supports two exact `hidden_dim` values

Read directly from `kernels.cu`/`fused_kernels.cu` (NOT from
`evaluation.py`, which happens to never exercise an unsupported shape in
its own usage): both `kq_mat_vec_mul_impl` (`kernels.cu` ~L564-589) and
`fused_kq_launcher_impl` (`fused_kernels.cu` ~L108-150) are `if/else if`
ladders hardcoded to EXACTLY `hidden_dim/head_num in {1024/8, 5120/40}`
(matching two of the paper's own eval models' KV-head counts — Qwen3-8B
has 8 KV heads, Phi-4 has 40 attention heads, both `head_dim=128`), with
**no general case** — this is not a runtime parameter the C++ dispatches
generically on, it is two compiled template instantiations and nothing
else. `fused_kq_launcher_impl`'s `else` branch throws
`std::runtime_error("Unsupported kernel configuration")`; `kq_mat_vec_mul`'s
does not (see "Real artifact bug" below).

This domain's gemv shapes were not chosen with this floor in mind (none of
`dense.py`'s `_gemv_shapes()` K values equal 1024 or 5120), so an arbitrary
K is realized here by **chunking** it into fixed `HEADS_PER_CHUNK=8`
(`hidden_dim=1024`) segments and calling PackKV's own, unmodified per-chunk
kernel chain once per chunk; `run()` sums the chunks' partial dot products
to recover the full-K `sum_{k=0}^{K-1} A[ctx,k]*x[k]` — elementary linear
algebra (a column-block-partitioned matrix product is the sum of its block
partial products), never a kernel edit. In real PackKV usage there is only
ever ONE chunk per call (hidden_dim already equals the model's true
KV-head count); chunking an arbitrary K into several independent
1024-wide calls is this adapter's own boundary choice, documented rather
than silently assumed. `HEADS_PER_CHUNK=8` (not 40) is used throughout
since it is the smaller of the two supported sizes and divides more of
this domain's K values without padding.

## Quantization scheme replicated, and the fairness hook used

PackKV's own K-path quantizer (`evaluation.py`'s `quant_ints`,
`QuantMethod.PackKV = (TokenQuant, TokenQuant)`, `QUANT_DIM[TokenQuant] =
[1,4]`) is an **asymmetric, per-row (per ctx-len position), zero-point**
affine quantizer, reduced across the FULL per-call hidden-state width (in
this adapter: per chunk, shared across that chunk's 8 synthetic heads):

```
scale[row]         = (max(row) - min(row)) * quant_scale_rel
zero_int[row]      = round(min(row) / scale[row])
code_stored[row,c] = clip(round(A[row,c]/scale[row]) - zero_int[row], 0, qmax)
```

**Key algebraic fact this adapter relies on** (verified by direct
substitution, and independently confirmed by the GPU-vs-numpy check below):
PackKV's own reconstruction `(code_stored + zero_int) * scale` simplifies
to `round(A/scale) * scale` exactly — the zero-point is a CODE-STORAGE-only
offset (keeps stored codes non-negative for nibble-packing) that cancels
out of the reconstructed VALUE. `fused_kq`'s CUDA formula
(`result = scale * (t + zero_int * sum_c(q[c]))`) is exactly this identity
distributed across the codes-dot-q partial product `t` that
`kq_mat_vec_mul` computes — so the reconstructed dot product is provably
`dequant(A) @ x` with `dequant(A) = round(A/scale)*scale`, independent of
the zero-point storage convention.

This scheme is **not** expressible via the marlin-precedent
`quantize_dequantize_groupwise` helper in `dense.py` (SYMMETRIC,
zero-point-free, groups along COLUMNS; PackKV's scheme has an explicit
zero-point and groups along ROWS — a different quantization FAMILY, not
just a different axis). Per the task brief's explicit fallback, this
integration does **not** reuse that helper. Instead, `kernelbench/domains/
dense.py::reference_gemv` gained ONE new, minimal, optional params key:

```python
A_override = params.pop("dequantized_A_override", None)
if A_override is not None:
    A = np.asarray(A_override, dtype=np.float64)   # used verbatim, no quantization in dense.py
```

`reference_gemv` performs **no quantization logic at all** on this path —
it is handed finished VALUES. This adapter's own private `_quantize_row`
(never imported by, or shared with, `dense.py`) computes those values
independently, applied per chunk using only that chunk's real (unpadded)
columns. This preserves the reference-independence rule (DOMAIN_GUIDE.md):
`reference_gemv` never calls any function that computes PackKV's
quantization, and never touches `packkv_cuda`'s actual C++/CUDA-executed
output — that output (`run()`'s real, GPU-computed `our_kq`, summed across
chunks) is what gets compared against this independently-computed ground
truth. A bug in `k_encode_cpu`/`k_decode_cpu`/`kq_mat_vec_mul`/`fused_kq`
(wrong nibble unpacking, wrong scale/zero application, wrong q-transform
permutation) would change `run()`'s output relative to this ground truth,
exactly as a meaningful gate requires.

**Why the override is `pop()`ped, not left in `params`** (a real bug found
and fixed during this integration, distinct from marlin's `quant_bits`/
`quant_group_size` keys, which ARE left in place because they are small
JSON-serializable ints): the override is a full `(M,K)` numpy array.
`runner.py`'s `json.dump(doc, ...)` serializes `params` verbatim into the
result record (`harness.RunResult.to_dict`); the first working version of
this integration left the array in `params` and every run crashed at
`json.dump` with `TypeError: Object of type ndarray is not JSON
serializable` (reproduced, then fixed — see git history of this session).
Fix: `reference_gemv` now `params.pop("dequantized_A_override", None)`
instead of `params.get(...)`, removing the large array from the shared
dict once consumed. `reference_gemv` runs exactly once per `run_variant`
call (the correctness gate, before the warmup/timed-rep loop), so popping
cannot affect anything later in the same run. Verified: marlin's own
`quant_bits`/`quant_group_size` keys are untouched by this change (they
use a separate code path, never popped), and the packkv gate below now
produces a valid, dumped JSON result record.

## Padding (K-columns per chunk, and M-rows overall)

Same situation as `gemv/marlin`'s own tile padding, with one added
subtlety from PackKV's row-shared (not column-grouped) scale:

- **K columns within a chunk**: a chunk's scale/zero are shared across the
  WHOLE 1024-column chunk width, so naively zero-padding a short final
  chunk would let the padding value become that row's new min or max,
  corrupting the REAL columns' own scale/zero too. Fixed by computing
  scale/zero from the chunk's REAL columns ONLY, then padding the STORED
  CODE array (not the raw values) with a duplicate of one of those real
  columns' own code (already valid in `[0,qmax]` by construction) and
  padding `x` with an actual `0` in the matching lanes — the padded
  columns' contribution to the dot product is provably zero (multiplied
  by `x=0`) regardless of which real code was duplicated, while the real
  columns' codes are IDENTICAL to what an unpadded, exactly-1024-wide
  chunk would produce. Proven exact, not approximate.
- **M rows**: `ctx_len` must be an exact multiple of PackKV's own
  hardcoded `ctx_len_block_size=64`. Extra rows are filled with a
  DUPLICATE of the last real row (never all-zero — an all-zero row makes
  that row's own `max-min=0`, i.e. `scale=0`, a genuine divide-by-zero in
  PackKV's own formula); those rows are computed by the kernel like any
  other but sliced off in `to_host()`, exactly like marlin's own
  M-padding.

Only `smoke-gemv-shortfat-32x4096` (M=32) needed M-padding among the 3
smoke shapes (32 -> 64); `smoke-gemv-tallskinny-4096x32` (K=32) and
`smoke-gemv-square-512` (K=512) both needed K-column padding to 1024
within their single chunk; `smoke-gemv-shortfat-32x4096` (K=4096) needed
none (4096 = 4 x 1024 exactly).

## Precision

`PRECISIONS = ["fp16"]` — `fused_kq_launcher`/`kq_mat_vec_mul` dispatch on
`torch.float16`/`torch.bfloat16` only (`fused_kernels.cu`/`kernels.cu`),
never a continuous fp32/fp64 K cache — matching the spec's own
`gemv-e2e-compressed-format` `dense_operand` field ("PackKV: FP16/BF16").
`dense.py`'s `DEFAULT_PRECISION["gemv"]` is unconditionally `"fp64"`, so
running this impl requires an explicit `--precision fp16`, same as marlin.

## Gate result (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel gemv --variant gemv-e2e-compressed-format \
    --impl packkv-kq-compressed-gemv --precision fp16 --smoke
```

Result: **3/3 runs valid**, `max_scaled_err` tolerance `1e-2` (parsed from
`gemv-e2e-compressed-format`'s spec text):

| smoke shape | (M,K) | padding | max_scaled_err | cold time |
|---|---|---|---|---|
| smoke-gemv-square-512 | (512,512) | K: 512->1024 (1 chunk) | 2.69e-04 | 0.97ms |
| smoke-gemv-tallskinny-4096x32 | (4096,32) | K: 32->1024 (1 chunk) | 1.44e-03 | 0.49ms |
| smoke-gemv-shortfat-32x4096 | (32,4096) | M: 32->64; K: exact (4 chunks) | 7.87e-05 | 3.95ms |

All comfortably within the 1e-2 tolerance, and in fact all three also
clear `gemv-quantized-weight-kernel`'s stricter 1e-3 bound except the
tallskinny shape (1.44e-03, marginally over 1e-3 — not this variant's own
tolerance, noted only for context) — consistent with a correctly
implemented kernel chain consuming correctly-shared quantized values, not
luck. Timings above are cold-cache single-call latencies from the
`--smoke` harness path (`warmup=5, reps=20` override; NOT a timed sweep —
per ARTIFACT_GUIDE rule 5, login-node build+gate only). `conforming: False`
is expected and correct (synthetic smoke shapes, shared login-node GPU).

Independent standalone verification (before wiring the adapter, to
de-risk the chunking/padding design against a hand-checkable case, not
part of the harness): a throwaway script exercising the SAME
`k_encode_cpu`/`k_decode_cpu`/`kq_mat_vec_mul`/`fused_kq` chain directly at
`(M,K) in {(64,40),(128,128),(64,1024),(192,1500),(32,4096)}` against a
plain-numpy `dequant@x` ground truth produced `max_scaled_err` in
`[6.4e-05, 8.7e-04]` across all five shapes, including K values that are
neither a multiple of 1024 nor of 128 — confirming the chunking/padding
design generalizes correctly beyond just this domain's 3 smoke shapes.

## Real artifact bug found

`kernels.cu`'s `kq_mat_vec_mul_impl`'s `else` branch (any `hidden_dim`
other than 1024 or 5120) does:

```cpp
} else {
    std::cout << "no such case" << std::endl;
    return 0.0;
}
```

This neither throws nor sets any error flag on the tensors it was handed —
the caller's `kq_out` (`t`) buffer is left exactly as initialized (zeros),
and the immediately-following `fused_kq` call would then silently compute
a plausible-looking but WRONG result from that all-zero `t` (not even
detectably zero-valued output — `fused_kq`'s own formula
`scale*(t + zero_int*sum(q))` still produces a nonzero number from the
`zero_int*sum(q)` term alone even when `t` is entirely zero). Nothing in
the call chain — not `kq_mat_vec_mul`'s own return value (a timing float,
identical in shape to the SUCCESS case), not any exception, not any
sentinel value in `kq_out` — distinguishes this failure mode from a
legitimate result. `evaluation.py`'s own usage never triggers this branch
(its only tested models have exactly 8 or 40 KV heads), so it has
apparently never been observed upstream. This adapter defends against it
explicitly (`run()` hard-asserts `hidden_dim in (1024, 5120)` before every
`kq_mat_vec_mul` call, raising `RuntimeError` rather than trusting the
extension to fail loudly), but the underlying silent-no-op is a real,
unpatched robustness bug in the artifact itself — not touched here (rule 3:
patch build systems, not kernel code; this is a correctness/robustness
issue in kernel-adjacent C++ glue, left as-is and merely defended against
from the adapter side).

## Environment note

`import packkv_cuda` needs `LD_PRELOAD=/usr/lib64/libstdc++.so.6` on this
machine (same CXXABI/GLIBCXX symbol-version mismatch documented for
marlin/tilus/qfactory). Unlike marlin's package (whose `__init__.py`
imports `torch` itself before touching `marlin_cuda`), `packkv_cuda` is a
bare C extension with no Python wrapper — `import torch` MUST happen
before `import packkv_cuda` in any script/REPL using it (its `.so` depends
on `libc10.so`, torch's core lib, which only resolves once torch has
already loaded it into the process); `build.sh`'s own verification step
and this adapter's `available()`/`prepare()` both do this in the correct
order.

## Not done (out of this integration's login-node budget)

- No sweep across `gemv-e2e-compressed-format`'s own recommended datasets
  (PackKV's 6-model x {32K,128K}-context-length sweep) — that requires
  downloading real LLM weights/KV caches per model, well beyond a
  login-node functional gate; only this domain's 3 synthetic smoke shapes
  were run, per ARTIFACT_GUIDE rule 5.
- No V-cache path (`v_encode_cpu`/`wv_mat_vec_mul`/`fused_wv`) — a
  different operation (weighted row-sum over softmax attention weights,
  not a plain GEMV), out of scope for this track's `y=Ax` mapping per the
  task brief.
- No real-LLM head-count (8 or 40) exact-fit run without chunking — every
  smoke shape here happens to need at least K-column padding or chunking
  since none of `_gemv_shapes()`'s K values equal 1024 or 5120 exactly;
  not a gap in this integration, just worth noting the "zero chunks/no
  padding needed" case was never exercised by this domain's own shape set.
- Real timed sweep (compute-node allocation, per ARTIFACT_GUIDE rule 5) —
  this integration is login-node build+gate only.
- The `kq_mat_vec_mul` silent-no-op bug above was not reported upstream
  (out of scope for this integration) and not patched (rule 3: kernel-code
  changes are not "minimal build-system fixes").

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, 14 SMs, ~5 GB; host GPU
  reports as NVIDIA A100-SXM4-40GB via nvidia-smi under the MIG partition),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge,
  env.sh's `$CC`/`$CXX`), torch 2.8.0+cu128, Python 3.12.14. Commit
  unchanged: `da3c00d986d1e37b5c3a7e2ce9403bbf2ab6397f`.
- Build: OK, after one real fix (same bug class as `gemv/marlin`): `build.sh`
  did `cd "$(dirname "$0")"` BEFORE `source ".../toolchain.sh"` (resolved
  via `$(dirname "${BASH_SOURCE[0]}")`), so once cwd had already changed,
  the toolchain.sh path doubled and failed with "No such file or
  directory" -- the identical invocation-order bug documented in
  `gemv/marlin/STATUS.md`'s "Reproduction on zaratan" section (present
  since the original baseline commit, latent only because the reference
  machine invoked build.sh by absolute path). Fixed the same way: `source
  toolchain.sh` moved before the `cd`. The pre-existing `sm_80` gencode
  patch to `source/packkv_cuda_ext/setup.py` (idempotent, marker-based) and
  the CC/CXX pin were left untouched and worked as-is. No change to
  `source/` or PackKV's own code. This build also repeatedly hit the
  shared-login-node RLIMIT_NPROC=256 contention documented project-wide
  (`fork: retry: Resource temporarily unavailable` from the shell itself,
  OpenBLAS `pthread_create` failures, and — as a downstream symptom of the
  same thread/process-creation pressure — numpy's `RuntimeError: CPU
  dispatcher tracer already initlized` on `import torch` inside pip's
  build subprocess, reproduced even with a bare `$PY -c "import torch"`
  outside any packkv-specific context) far more severely than any other
  artifact in this pass; resolved purely by retrying until the shared
  node's load dropped, with `MAX_JOBS=1`. Not a packkv or build-system bug.
- Gate: `--variant gemv-e2e-compressed-format --impl
  packkv-kq-compressed-gemv --precision fp16 --smoke`: **PASS** 3/3 --
  smoke-gemv-square-512 err 2.69e-04, smoke-gemv-tallskinny-4096x32 err
  1.44e-03, smoke-gemv-shortfat-32x4096 err 7.87e-05 (tol `1e-2`) --
  identical errors to the recorded run.
- Deviation from the recorded ruling: none -- 3/3 pass, identical errors
  and tolerance.
- Verdict here: BUILT+GATED -- equals the recorded ruling.
