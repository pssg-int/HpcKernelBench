# gpu-dpf (GPU-DPF) — gemv

**Status: SKIPPED (no separable, gateable GEMV — see evidence below)**

- Paper: "GPU-based Private Information Retrieval for Uniform Messages: Kernel
  Level GPU Optimization and Concurrency", ASPLOS'24.
  `PAPER_KEY = conf/asplos/LamJ0MGLLLRLRW024`.
- Artifact: https://github.com/facebookresearch/GPU-DPF
- Commit cloned: `ce23a06af884ee54300b5bc5fd5350e445f10b0b`, `git clone --depth 1`
  (upstream's own commit `2023-01-26`, "Initial commit" — this repo has a
  single commit).

## Task ask, restated

"Its core compute is batched DPF evaluation + a large dot-product/
matrix-vector reduction over the database. Wrap the matrix-vector/batched-dot
kernel at the finest boundary IF it exists as a separable unit computing a
standard `y=Ax` (possibly over integer/modular arithmetic — if the arithmetic
is not plain fp/int GEMV, the gate cannot use the dense fp64 reference: SKIP
with evidence instead)."

## What was actually checked (not just the survey's prior claim)

The Python-facing PRF-evaluation path (`dpf.py`'s `DPF.eval_gpu` ->
`dpf_cpp.eval_gpu`) is confirmed compute-bound / not-a-materialized-table-read,
matching `benchspecs/gemv/spec.yaml`'s prior survey finding — but this
integration went further and searched specifically for the "matrix-vector
reduction over the database" component the task brief describes, since that
component, if separable and plain-arithmetic, WOULD be in scope.

**Found it**: `dpf_gpu/matmul/matmul.cu` (`GEMM128`/`GEMM128_kernel`/
`GEMM128_reduction_kernel`) plus its own dedicated benchmark driver,
`dpf_gpu/matmul_benchmark.cu` (own `main()`, own `alloc_test_matrix`/
`check_correct`, own README-less CLI). This is a genuine, standalone
`C[M,N] = A[M,K] @ B[K,N]` GEMM kernel (the task's "large dot-product/
matrix-vector reduction", here generalized to GEMM but trivially usable at
N=1 for GEMV, since `GEMM128`'s only shape constraints are `N%BLOCK_W==0`,
`M%BLOCK_H==0` with `BLOCK_W=BLOCK_H=4` — N=1 would need padding same as
`gemv/marlin`, not itself disqualifying).

**Why it is disqualified anyway — the arithmetic is not plain fp/int**:
`dpf_gpu/utils.h` defines `uint128_t_gpu` as a `uint4` (4x uint32) and
implements `add_uint128`/`mul_uint128` via **wraparound** (mod 2^128)
arithmetic using inline PTX carry-chain instructions
(`add.cc.u32`/`addc.cc.u32`/`addc.u32` for addition; a 4-limb schoolbook
multiply for `mul_uint128`, not shown in full here but confirmed present in
the same file). `GEMM128_kernel`'s inner reduction is literally
`C_frag_local[jj][kk] = add_uint128(C_frag_local[jj][kk],
mul_uint128(A_frag_local[jj], B_frag_local[kk]))` — i.e. genuine modular
(mod 2^128) integer matrix multiplication, not floating point and not
ordinary (non-wrapping) integer arithmetic. This is DEFINITIONALLY not
representable or comparable via this suite's fp64-reference `max_scaled_err`
gate (`dense.py`'s `reference_gemv`/`quantize_dequantize_groupwise`, and
`harness.check_correctness`'s `max_scaled_err` mode) — a 128-bit product of
two ~128-bit random values overflows fp64's 52-bit mantissa by roughly 90
orders of magnitude before even considering the wraparound, and "relative
error" is not a meaningful concept for wraparound modular arithmetic (the
result is either bit-exact or, after a single dropped carry, unboundedly
"wrong" with no continuum in between — matching `GPU-DPF`'s OWN correctness
convention throughout the rest of the repo, `np.linalg.norm(rec-gt)<=1e-8`
i.e. exact-match, never a relative-error tolerance). Per the task's own
explicit instruction, this disqualifies it: "if the arithmetic is not plain
fp/int GEMV, the gate cannot use the dense fp64 reference: SKIP with
evidence instead."

**Additional, corroborating evidence that this is not the artifact's actual
shipped kernel**: `setup.py`'s `CUDAExtension('dpf_cpp', sources=
['dpf_wrapper.cu'], ...)` is the ONLY thing built into the Python-importable
extension the paper's own `dpf.py`/README/`benchmark.py` actually use, and
`dpf_wrapper.cu` only `#include`s `dpf_base/dpf.h` and
`dpf_gpu/dpf/dpf_hybrid.cu` (the PRF-expansion DPF-eval kernel) — it does
**not** include `dpf_gpu/matmul/matmul.cu` or reference `GEMM128` anywhere.
`matmul_benchmark.cu` has its own free-standing `main()` and is not
referenced by `setup.py`, `install.sh`, `benchmark.py`, or any Python
binding — it is a standalone, uncompiled-by-default C++/CUDA source file,
apparently a development-time comparison/exploration artifact, never wired
into the paper's actual PIR interface. Wrapping a standalone `main()`-having
benchmark binary the paper itself never ships as a callable kernel would
also run against ARTIFACT_GUIDE rule 1 ("wrap the kernel, not the paper's
benchmark script") even before the arithmetic-type disqualification above.

## Verdict

**SKIPPED.** The one component of this artifact that IS a separable,
standard-shaped `C=A@B` reduction (`GEMM128`) fails the task's own explicit
arithmetic-type carve-out (uint128 wraparound modular arithmetic, not plain
fp/int) and is not even part of the artifact's actual shipped/tested Python
kernel interface. The artifact's actual shipped kernel (PRF-based DPF
evaluation, `dpf_hybrid.cu` et al.) is confirmed compute-bound
(AES-128/Salsa20/ChaCha20 PRF expansion) and never materializes or reads a
`y=Ax`-shaped table product from memory at all — consistent with
`benchspecs/gemv/spec.yaml`'s own prior survey finding (`notes_on_fairness`:
"GPU-DPF is included... but its own protocol is deliberately NOT used as a
template for any variant... because its kernel is compute-
(PRF-evaluation-) bound rather than memory-bandwidth-bound"). No build was
attempted (cheap skip, per ARTIFACT_GUIDE's scope-ruling precedent for
disqualified artifacts) — the disqualification is structural (arithmetic
type + not-actually-shipped-as-a-kernel), not something a build would
resolve.

## Not done

- No build (`build.sh` not written) — nothing to build toward; see verdict.
- No adapter.py — no gateable kernel exists per the evidence above.
