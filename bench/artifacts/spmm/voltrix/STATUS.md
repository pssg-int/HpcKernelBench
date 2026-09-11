# Voltrix — STATUS

**Outcome: BUILT on H100 (sm_90a) — gate FAILS: `spmm-binary-adjacency-kernel` 0/12 valid, max_scaled_err 2.9e-4 vs the fp32 tol 1e-4 (TF32 tensor-core roundoff level; the kernel computes in TF32 but the adapter declares fp32). Earlier outcome, kept for the record: DEFERRED-HARDWARE (needs sm_90a / Hopper) — to be integrated on a Hopper-class machine; clone + provenance kept, see REQUIRES_GPU.**

> Original ruling (kept for the record): SKIPPED — requires NVIDIA Hopper (sm_90), this machine's GPU is
an A100 (sm_80). Confirmed empirically, not just from the README.**

- Paper: "Voltrix: Sparse Matrix-Matrix Multiplication on Tensor Cores with
  Asynchronous and Balanced Kernel Optimization", USENIX ATC'25.
  `PAPER_KEY = conf/usenix/XiaWY0C25`.
- Artifact: https://github.com/YaqiXia/Voltrix-SpMM
- Commit cloned: `ea7ab6566f3c97931ae9ffe27f4e6ad4c2580a6b` (2025-07-16),
  `git clone --depth 1`. See `source.provenance`.
- Selection rationale (per the revised kernel-centrality rule): core
  general-GPU-SpMM baseline, ATC'25, claims 1.7x over RoDe on tensor cores —
  in scope before the hardware check below ruled it out.

## Why this is a genuine hardware blocker, not a build-system fix

The README states the requirement explicitly under "Hardware":

> **GPU Architecture:** NVIDIA Hopper or newer (compute capability >= 90).
> Voltrix **requires NVIDIA Hopper-specific architectural features**,
> including **Tensor Memory Accelerator (TMA)** for high-throughput memory
> movement and **mbarrier** for efficient thread synchronization.

This is not merely an unset `-arch` flag defaulting to something newer than
necessary. Two independent pieces of evidence confirm the dependency is
architectural, not cosmetic:

1. **The JIT compiler hardcodes Hopper codegen.**
   `voltrix/jit/compiler.py:125`:
   `"-gencode=arch=compute_90a,code=sm_90a"` — not configurable via any of
   the documented `VOLTRIX_*` environment variables (checked all of them;
   none touches the arch flag).
2. **The kernel source uses genuine Hopper-only PTX instructions**, not an
   optional fast path. `voltrix/include/voltrix/spmm_kernels.cuh` contains
   real (non-commented) inline PTX for the warp-specialized async-copy
   pipeline:
   ```
   cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes ...
   mbarrier.init.shared::cta.b64 [%0], %1;
   mbarrier.arrive.shared::cta.b64 _, [%0];
   ```
   `cp.async.bulk` (TMA) and `mbarrier` are PTX ISA 8.0+ instructions valid
   only on `sm_90a` — there is no `sm_80` fallback path in this file; the
   whole load pipeline (not just the MMA instruction) is built on them.

Patching this to run on sm_80 would mean rewriting the warp-specialized
TMA/mbarrier pipeline into an Ampere `cp.async`-based one — a real kernel
algorithm rewrite, exactly what ARTIFACT_GUIDE.md rule 3 prohibits ("if the
kernel itself must change to run, mark SKIPPED").

## Empirical confirmation (login node, A100-PCIE-40GB, sm_80)

Rather than skip on documentation alone, ran the artifact's own
`tests/test_spmm.py` (unmodified) with a small synthetic input, toolchain
pinned per `bench/artifacts/toolchain.sh`:

```
source bench/artifacts/toolchain.sh
LD_PRELOAD=/usr/lib64/libstdc++.so.6 PYTHONPATH=<source> \
  /pscratch/sd/c/cunyang/gnn/plexus_env/bin/python tests/test_spmm.py \
  --num_nodes 512 --num_feats 32 --density 0.1
```

Result: `nvcc` (12.9, this machine's pinned toolkit) compiles the JIT
kernels for `sm_90a` without error — cross-compilation to a newer target
succeeds regardless of the physical GPU present, so a clean compile is not
evidence of compatibility. The failure is at kernel **launch**, and it
happens in the *preprocessing* kernel (`hmat_cuda_kernel`, part of
`voltrix.csr_preprocess()`), before the SpMM kernel proper is ever reached:

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  CUDA error in hmat_cuda_kernel: no kernel image is available for execution on the device
```

This is `cudaErrorNoKernelImageForDevice` — the CUDA driver refusing to run
an `sm_90a`-only cubin on an `sm_80` device. Since even the format-conversion
step (not just the tensor-core MMA kernel) fails this way, no part of this
artifact's GPU pipeline executes on this machine's hardware.

## Verdict

`voltrix: SKIPPED (Hopper-only artifact -- hardcoded compute_90a/sm_90a JIT
codegen, genuine TMA/mbarrier PTX in the async-copy pipeline (not an
optional fast path), and an empirical run of the artifact's own unmodified
test script reproduces cudaErrorNoKernelImageForDevice on this machine's
A100 sm_80, in the preprocessing kernel before the SpMM kernel is even
reached. No legitimate build-system patch exists -- fixing this requires
rewriting the kernel's memory pipeline for Ampere, which ARTIFACT_GUIDE.md
rule 3 prohibits.)`

No `build.sh`/`adapter.py` written (matches the `rassm`/spmv SKIPPED
precedent) — nothing to build toward; the diagnostic run above used the
artifact's own unmodified `tests/test_spmm.py` directly, no adapter shim
involved.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan. Unlike Perlmutter, zaratan HAS sm_90a hardware (H100 nodes, gpu-a6-*), so per ARTIFACT_GUIDE rule 9 this artifact was taken off DEFERRED status and set up to build+gate. Login-node build phase only (see gate note).
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch 2.8.0+cu128, Python 3.12.14. Voltrix JIT-compiles its kernels with its own hardcoded `-gencode=arch=compute_90a,code=sm_90a` (Hopper).
- Build: login-phase OK, exit 0 (import + nvcc presence/version checks pass; source + deps in place). The GPU warm-up compile+launch step (which needs a real sm_90a GPU) was NOT exercised — see gate.
- adapter.py + build.sh: NEW this pass (the artifact was DEFERRED and had no adapter). adapter.py wraps Voltrix's own top-level API unmodified (`voltrix.csr_preprocess` + `voltrix.spmm` -> `voltrix::voltrix_spmm_forward_cuda`), gating under `spmm-binary-adjacency-kernel` (pattern-only: `csr_preprocess` takes no values, same finding as dtcspmm/flashsparse); rectangular/unsupported shapes raise NotImplementedError. Machine-neutral (CUDA_HOME/gpu_run.sh, no hardcoded paths).
- Gate: NOT RUN. It requires an sm_90a H100; zaratan's `gpu` partition H100 nodes were saturated for the whole session and SLURM's estimated start for a 12-20 min `--gres=gpu:h100:1` job was 2026-09-12 (~3 days out), unschedulable within the session. The build.sh + adapter are ready: `bench/gpu_run.sh -g h100 -t 40 -- 'bench/artifacts/spmm/voltrix/build.sh'` then the `--impl voltrix-spmm --variant spmm-binary-adjacency-kernel --smoke` gate will run it on an idle H100.
- Deviation from the recorded ruling: the blocker changed from "no such hardware anywhere (deferred to another machine)" to "hardware present on this machine but not schedulable within the session window". The kernel/PTX requirement (sm_90a) is unchanged.
- Verdict here: DEFERRED-HARDWARE (sm_90a H100 present on zaratan but the H100 queue was ~3 days out; adapter+build now in place, gate pending an idle H100) — ruling unchanged.


## Reproduction on zaratan — H100 run (2026-09-10)

- Machine: UMD zaratan, NVIDIA **H100 80GB HBM3 (sm_90)**, node gpu-a6-4, via
  `sbatch -p gpu-h100 --gres=gpu:h100:1 -t 90:00` (a short 20/30-min job only
  backfilled but could not finish Voltrix's single compute_90a JIT compile; the
  compile was first warmed on the login node — nvcc cross-compiles compute_90a
  without a GPU — then a 90-min H100 job finished it in ~44 min and ran the gate).
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch 2.8.0+cu128,
  Python 3.12.14; Voltrix's own hardcoded `-gencode=arch=compute_90a,code=sm_90a`.
- Build: OK on H100 (the JIT warm-up compile completed; `VOLTRIX_CACHE_DIR` warmed).
- Gate: `spmm-binary-adjacency-kernel` / `voltrix-spmm` / --smoke, fp32: **0/12 runs
  valid — all INVALID at max_scaled_err ≈ 2.915e-4 vs tol 1e-4.** The kernel builds,
  launches and produces output on sm_90a (so the DEFERRED-HARDWARE blocker is
  resolved), but the numeric gate fails.
- Deviation / analysis: the 2.9e-4 error is TF32 unit-roundoff level (≈ 2^-11.5).
  Voltrix is a Hopper tensor-core SpMM that computes in TF32, but its adapter
  declares `PRECISIONS=["fp32"]`, so the runner holds it to the fp32 tolerance
  (1e-4) and it fails. The sibling pattern-only tensor-core kernels dtcspmm /
  flashsparse show the SAME error magnitude (1.95e-4–2.92e-4) on this exact
  variant but PASS because they are gated at the variant's TF32 tolerance (1e-2).
  Not adjusted here (changing the declared precision to make it pass would be
  gate-gaming; recorded as a finding for a deliberate precision-declaration
  decision by the maintainer). No gate/tolerance was loosened.
- Verdict here: **BUILT on H100 (sm_90a), gate FAILS at fp32 tol (TF32-level error).**
  Supersedes the recorded DEFERRED-HARDWARE ruling: the hardware blocker is gone,
  the numeric result is a genuine (precision-declaration) gate failure.
