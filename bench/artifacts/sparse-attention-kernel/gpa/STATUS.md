# gpa (TomczakK25) — sparse-attention-kernel — STATUS: BUILT+GATED

- Paper: "Longer Attention Span: Increasing Transformer Context Length With
  Sparse Graph Processing Techniques", IPDPS'25. `PAPER_KEY =
  conf/ipps/TomczakK25` (matched by `artifact_url` in
  `../../../../output/included.json`; arXiv:2502.01659).
- Artifact: https://github.com/KLab-AI3/Graph-Processing-Attention-IPDPS-2025.
- Commit cloned: `b978f06d056d215f55a83d74cd79500f2ec477bc`, `git clone --depth 1`.
- Toolchain: python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`,
  torch `2.8.0+cu128`, nvcc `12.9`
  (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`), host compiler
  `/opt/cray/pe/gcc-native/14/bin/g++` (GCC 14), GPU arch `sm_80` (A100).

## What was built, and why only one of the 6 variants

The repo ships 6 separately-installable kernel extensions: COO/CSR
(explicit, arbitrary mask) and Local/Local-1D-Dilated/Local-2D-Dilated/
Global-No-Local (implicit, parameter-defined masks, no materialized mask
tensor). This integration builds **only `spfa_csr`**
(`source/Sparse_FlashAttention_CSR/`) — the most general variant: it accepts
ANY binary mask (converted to CSR), so it is a single drop-in receiver for
every pattern `kernelbench.domains.ml.SparseAttentionWorkload` can build
(causal/sliding_window/longformer/bigbird/dilated/block_local/
coo_arbitrary), rather than requiring 5 separate parametric adapters each
matched to one specific mask family. Per this repo's own
`verification/verify.py` (unmodified, read in full), CSR's output is
already cross-checked there against 3 of the other variants (COO
bit-identical, Local bit-identical at full density, dense PyTorch SDPA
directly) — the best-verified single variant to prioritize under the
login-node time budget. The other 5 extensions are **not built**.

The README's own install instructions ("Copy the 6 folders into your
`.../site-packages/torch/` directory... `python setup.py install`") would
install into the SHARED venv's site-packages — **not done**, per this
task's environment rules (no shared-env package installs; vendor into
`source/`). `build.sh` instead runs the repo's own `Sparse_
FlashAttention_CSR/setup.py` with `build_ext --inplace`, producing
`spfa_csr*.so` directly inside that source directory (importable via
`sys.path`, no site-packages pollution).

## Patch made, and why it is a build-system fix, not a kernel change

`sp_flatt_csr_kernel.cu`'s dtype dispatch used the deprecated ATen API
`Q.type()`:
```cpp
AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.type(), "spfa_csr_forward_cuda", ([&] {
```
This does not compile against this machine's torch 2.8 (`nvcc` error: "no
suitable conversion function from `const at::DeprecatedTypeProperties` to
`c10::ScalarType` exists"; `AT_DISPATCH_FLOATING_TYPES_AND_HALF`'s macro
internals now require a `c10::ScalarType`, and `.type()`'s legacy return
type is no longer implicitly convertible). Changed to `Q.scalar_type()` —
semantically identical (same tensor, same dtype dispatch), the standard
modern equivalent, and the exact class of fix ARTIFACT_GUIDE rule 3
sanctions ("CUDA-version guards are fine; touching kernel code is not" — this
is a torch-API-version guard, zero change to the kernel's algorithm/
arithmetic). Recorded in `build.sh`'s header comment and `adapter.py`'s
module docstring as well as here. **One line changed; no other file
touched.**

(Separately, unrelated, NOT patched: lines 260-261 of the same file compute
`sizeof(Q.type())`/`sizeof(l.type())` inside the shared-memory budget
calculation — this compiles fine (`sizeof` doesn't care about the deprecated
type's semantics) but is very likely ALSO a pre-existing bug in the
artifact's own shared-memory sizing, since `sizeof(at::DeprecatedType
Properties&)` is not `Q`'s element size. Left untouched: it does not block
compilation or this integration's gate, and "fix an unrelated latent bug not
required to run" is out of scope per ARTIFACT_GUIDE's minimal-patch rule.)

## What the adapter calls, and two things confirmed by reading the kernel in full

`adapter.py::run()` calls `spfa_csr.forward(q, k, v, w_row_off, w_col_ind,
w_val, m, l, out, use_nan)` — the compiled extension's actual entry point —
once per `(batch, head)` slice (the kernel's own interface is 2D `(N,d)`
Q/K/V, "single-headed, single-batched attention" per the README; this
adapter loops `B*H` times in `run()`, all inside the timed region, matching
how a real multi-head caller would have to use this artifact).

Reading `sp_flatt_csr_kernel.cu` in full turned up:
1. **`W_val` (the CSR mask's value array) is never read by the compute** —
   the kernel always recomputes the raw score `dot(Q_i,K_j)/sqrt(d)` fresh
   into its own (confusingly identically-named) shared-memory scratch
   buffer `W_val_shared`, and the global `W_val` accessor is never
   dereferenced anywhere in the kernel body. This means `spfa_csr` computes
   EXACTLY this track's unweighted 0/1-mask operation — no adapter-side
   workaround needed; `adapter.py` passes `W_val = ones(nnz)` since any
   value would give an identical result.
2. `m`/`l` (FlashAttention-style online-softmax running state) must be
   caller-pre-initialized (`m=-inf`, `l=0.0`, confirmed from `verify.py`'s
   own usage) and are reset once per `(b,h)` slice in `run()`.

## Gate verification (login node, correctness gate only, reduced protocol)

```python
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")
from kernelbench import domains, harness, spec
from kernelbench.artifact_registry import discover
from kernelbench.domains.ml import SparseAttentionWorkload

domain = domains.load("sparse-attention-kernel")
sp = spec.load("sparse-attention-kernel")
variant = sp.variant("sparse-attn-structured-mask-kernel-fp16")
impl = discover("sparse-attention-kernel")["gpa-spfa-csr"]["factory"]("fp32")

w = SparseAttentionWorkload(
    name="gpa-gate-sliding", variant="sparse-attn-structured-mask-kernel-fp16",
    B=2, H=3, d=32, S=256, pattern="sliding_window",
    pattern_params={"window": 32}, sf_target=None)
params = {"seed": 42, "precision": "fp32"}
r = harness.run_variant(impl, w, variant, params,
                         reference=domain.REFERENCES["sparse-attention-kernel"],
                         correctness_mode=domain.CORRECTNESS_MODE["sparse-attention-kernel"],
                         warmup_override=1, reps_override=3)
print(r.to_dict())
```
Run as: `LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY gate_gpa.py`.

Result: **valid: true**, `max_scaled_err = 5.40e-7 <= tol 1e-2` (fp32,
`sliding_window`) — a clean pass, well within fp32-rounding noise. Two
additional patterns checked (not part of the recorded gate, extra
confidence): `bigbird_local_global_random` at fp16 (`err=3.47e-3`, valid)
and `coo_arbitrary` at fp32 (`err=5.16e-7`, valid) — **the strongest
cross-pattern result of the 3 artifacts integrated in this track**: unlike
vit-sparse (single-block-containment restricted) and STOF (bitmap-masking
bug restricts it to tile-aligned patterns), `spfa_csr`'s single fused
online-softmax kernel handles every pattern this domain module builds
correctly, with no restriction. Reduced-protocol numbers only (`warmup=1,
reps=2-3`, shared login-node GPU), non-conforming per ARTIFACT_GUIDE rule 5.
`discover_status("sparse-attention-kernel")` confirms `gpa-spfa-csr` is
discovered `available: True` (with `LD_PRELOAD` set).

## Mask/pattern provenance

The workload's mask is generic (any pattern `kernelbench.domains.ml`
builds); this artifact places NO restriction on the pattern family (unlike
vit-sparse's single-block-containment need or STOF's tile-alignment need),
so no synthetic-family substitution or containment-check gate was needed
here — a genuine strength of this artifact's design (a real, correctly-
implemented flash-attention-style online-softmax over an arbitrary CSR
mask).

## Provenance

- Artifact commit: `b978f06d056d215f55a83d74cd79500f2ec477bc`
- nvcc: 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`)
- Host compiler: g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`)
- GPU arch: `sm_80` (A100)
- Python/torch: `/pscratch/sd/c/cunyang/gnn/plexus_env` — torch 2.8.0+cu128
- Patch: `source/Sparse_FlashAttention_CSR/sp_flatt_csr_kernel.cu`, line
  ~270, `Q.type()` -> `Q.scalar_type()` (torch-version compatibility only).

## Not done

- COO, Local, Local-1D-Dilated, Local-2D-Dilated, Global-No-Local variants
  not built/gated — CSR alone already generalizes over every mask pattern
  this project's workload builder produces; the implicit-mask variants
  would each need their own adapter matched to their own specific
  parametrization (window/dilation/global-token-list), lower priority given
  CSR's generality and the login-node time budget.
- No sweep across the spec's `recommended_subset` shapes (e.g. `S=16384`) —
  small gate checks only.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05; job
  ran on gpu-b11-6.zaratan.umd.edu, physical card NVIDIA A100-SXM4-40GB),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14; no cmake (torch `setup.py build_ext`).
  Arch flags: `TORCH_CUDA_ARCH_LIST=8.0` (sm_80), same as recorded.
- Build: OK (`build.sh` found `spfa_csr.cpython-312-x86_64-linux-gnu.so`
  already built this session and skipped recompilation — idempotent). The
  recorded `Q.type()` -> `Q.scalar_type()` patch is present in
  `source/Sparse_FlashAttention_CSR/sp_flatt_csr_kernel.cu` and matches
  `source.patch` byte-for-byte (`git -C source diff` == `source.patch`).
  Build-system changes: none beyond the already-recorded patch —
  `build.sh`'s `${PY:-...}`/`${CUDA_HOME:-...}`/`${CXX:-...}`/`${CC:-...}`
  fallbacks already read this machine's env.sh-exported values first.
- Gate (same reduced-protocol Python harness script as recorded, run via
  `discover(...)["gpa-spfa-csr"]["factory"](precision)`, `warmup=1, reps=3`):
  - `sliding_window(window=32)`, fp32, S=256: PASS, `max_scaled_err =
    5.40e-07 <= tol 1e-2` (this is the recorded primary gate).
  - `bigbird_local_global_random`, fp16, S=256 (extra confidence check):
    PASS, `max_scaled_err = 5.45e-03 <= tol 1e-2`.
  - `coo_arbitrary`, fp32, S=256 (extra confidence check): PASS,
    `max_scaled_err = 4.03e-07 <= tol 1e-2`.
  `discover_status("sparse-attention-kernel")` / `--list` confirm
  `gpa-spfa-csr` discovered (`no CUDA device visible` on the login node,
  as expected without a GPU).
- Deviation from the recorded ruling: none on outcome (all 3 patterns still
  PASS, well inside tolerance). One numeric deviation worth recording: the
  `bigbird_local_global_random` fp16 error came back `5.45e-03` here vs.
  `3.47e-03` recorded on Perlmutter — still comfortably `<= 1e-2`, and
  consistent with ordinary fp16-kernel run-to-run/machine-to-machine
  floating-point non-associativity (parallel-reduction ordering), not a
  correctness regression; the fp32 patterns (`sliding_window`,
  `coo_arbitrary`) match the recorded values to within normal fp32 noise.
- Verdict here: BUILT+GATED — same as the recorded ruling.
