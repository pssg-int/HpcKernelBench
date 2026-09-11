# mxblas (MXBLAS) — quantized-gemm

**Status: BUILT + own-test GATED on H100 (sm_90a) — MXBLAS's own tests/test_jit.py and tests/test_mxgemm.py (2048^3 FP8, per-tensor scaling) both PASS on an H100: FP8-GEMM difference rate 0.071%, 432.9 TFLOPS. (No kernelbench adapter by design; the FP8/MX regime is not modelled by the quantized-gemm domain — its gate is the artifact's own test scripts, which build.sh runs on a GPU.) Earlier outcome, kept for the record: DEFERRED-HARDWARE (needs sm_90a / Hopper, FP8) — to be integrated on a Hopper-class machine; clone + provenance kept, see REQUIRES_GPU.**

> Original ruling (kept for the record): SKIPPED — requires NVIDIA Hopper (sm_90a); this machine's GPU is
an A100 (sm_80). Confirmed from four independent, sufficient pieces of
evidence below; no INT8 fallback path exists in this artifact to attempt
instead.**

- Paper: "MXBLAS: Accelerating 8-bit Deep Learning with a Unified
  Micro-Scaled GEMM Library", SC'25. `PAPER_KEY = conf/sc/WangXY0C25`.
- Artifact: https://github.com/yatorho/MXBLAS
- Commit cloned: `a33df1316be81a0ecf3d0bdf4c5843324732af7c` (2025-08-21),
  `git clone --depth 1`. See `source.provenance`.
- Selection rationale (per the revised kernel-centrality rule, before the
  hardware/software checks below ruled it out): `kernel_centrality.json`'s
  `quantized-gemm|conf/sc/WangXY0C25` rates centrality `core`, regime
  `matches` — "a library whose headline IS the MX-GEMM kernel, evaluated
  kernel-level; it is the direct source of the spec's `qgemm-w8a8-mx`
  variant."

## Task brief's own hypothesis (checked, and ruled out)

The integration brief asked specifically: "A100 has no FP8 tensor cores:
integrate only the paths that run on sm_80 (INT8 micro-scaled)." This
artifact does **not** have a separate INT8 tensor-core GEMM path at all —
confirmed by grepping every `.py`/`.cuh`/`.cu`/`.h` file under `mxblas/`
for `int8`/`s8`: the only hits are generic byte-buffer declarations
(`extern __shared__ uint8_t smem[]` in `mainloop_tma_scales_generator.py`
and `generator.py`), never an INT8 GEMM *datatype* or MMA-instruction
path. The README's own usage example and every kernel template generator
(`mxblas/gemm/generator.py`, `mainloop_tma_scales_generator.py`) operate
exclusively on `torch.float8_e4m3fn`/`e5m2` (MX-FP8). "MX-GEMM" in this
artifact's own naming IS the FP8 micro-scaled format — there is no
"INT8 micro-scaled" variant to fall back to on sm_80; the hypothesis in
the task brief does not hold for this specific artifact.

## Evidence (four independent, each individually sufficient)

### 1. Explicit, unambiguous README hardware requirement

> **Hardware Requirements**
> - **GPU:** NVIDIA Hopper architecture (Compute Capability = 9.0)
>   Required hardware features: Tensor Memory Accelerator (TMA),
>   Warp-Group Matrix Multiply Accumulate (WGMMA), and Memory Barrier
>   (MBarrier) instructions.

Identical framing to the `voltrix`/spmm precedent's own README quote —
TMA/WGMMA/MBarrier are genuine Hopper-only PTX ISA 8.0+ features with no
Ampere equivalent (WGMMA in particular has no `sm_80` counterpart at all;
Ampere's own tensor-core MMA is `mma.sync.aligned.m16n8k32`-class warp-level,
architecturally distinct from Hopper's warp-GROUP-level WGMMA).

### 2. Hardcoded, non-configurable Hopper codegen

`mxblas/jit/compiler.py:139` (inside the JIT `build()` function that
compiles EVERY kernel this library ever runs, not an optional path):

```python
nvcc_flags = [
    ...
    "-gencode=arch=compute_90a,code=sm_90a",
    ...
]
```

Read the complete list of `os.environ` lookups in this function (only
`PTXAS_VERBOSE_FLAG` and `CACHE_DIR_FLAG`, per the README's own documented
`MXBLAS_*` env-var table) — no environment variable, flag, or config
option anywhere in `mxblas/jit/compiler.py` touches the arch string. Every
JIT-compiled kernel this library produces targets `sm_90a` unconditionally.

### 3. Independent, separate Python-version blocker (empirically confirmed)

`mxblas/gemm/keys.py:3` does `from typing import (..., override, ...)` --
`typing.override` was added in **Python 3.12** (matching the README's own
stated "Python >= 3.12" requirement). This integration's pinned Python is
`/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, **3.11.7**. Attempting
`import mxblas` (sys.path-inserted, no pip install needed -- `setup.py` is
a pure-Python `find_packages()` package, no C extension to build ahead of
time) reproduces this immediately:

```
$ LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -c "import sys; sys.path.insert(0,'source'); import mxblas"
  File ".../mxblas/gemm/keys.py", line 3, in <module>
    from typing import (
ImportError: cannot import name 'override' from 'typing'
```

This alone would block any integration attempt on this machine's pinned
toolchain regardless of GPU architecture -- a second, fully independent
reason not to proceed, on top of (1)-(2).

### 4. Portability wrinkle (secondary, moot given 1-3)

The same `compiler.py` `nvcc_flags` list also hardcodes
`-I/home/yatorho/doc/projs/cu_head` -- the original author's own home
directory -- an absolute include path that would not resolve on any other
machine either way.

## Verdict

`mxblas: SKIPPED (Hopper-only artifact -- explicit README requirement
"NVIDIA Hopper architecture (Compute Capability = 9.0)" naming TMA/WGMMA/
MBarrier as required hardware features; a hardcoded, non-configurable
"-gencode=arch=compute_90a,code=sm_90a" in the JIT compiler that builds
every kernel; no INT8 tensor-core datatype path exists anywhere in the
source to fall back to on sm_80 -- MX-GEMM in this artifact is FP8-only,
grep-confirmed; and a fully independent Python>=3.12 requirement
(typing.override) that this integration's pinned Python 3.11.7 does not
meet, reproduced empirically. No legitimate build-system patch exists for
either blocker -- rewriting the WGMMA/TMA mainloop for Ampere would be a
genuine kernel algorithm rewrite (ARTIFACT_GUIDE rule 3), and the
Python-version gap would require an entirely separate interpreter/venv
outside this integration's pinned toolchain.)`

No `build.sh`/`adapter.py` written (matches the `rassm`/`voltrix` SKIPPED
precedent) -- nothing to build toward.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, which HAS sm_90a H100 nodes (unlike Perlmutter). Login-node build phase only (see gate note).
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch 2.8.0+cu128, Python 3.12.14 (>= 3.12, MXBLAS's own requirement, satisfied).
- Build: login-phase OK, exit 0 (import + nvcc checks; source + deps in place). MXBLAS JIT-compiles FP8 kernels with hardcoded `compute_90a` on first call; the GPU build+test step was NOT exercised — see gate.
- No adapter.py, by design (unchanged): the quantized-gemm domain module models only INT weight-only quantization, not the FP8/MX block-scaled regime MXBLAS implements, and the reproduction pass forbids editing bench/kernelbench. The recorded functional gate is MXBLAS's OWN tests (tests/test_jit.py, tests/test_mxgemm.py), which build.sh runs when an sm_90a GPU is visible.
- Gate: NOT RUN. Requires an sm_90a H100; zaratan's H100 partition was saturated all session (SLURM est. start 2026-09-12, ~3 days out), unschedulable in-session. `bench/gpu_run.sh -g h100 -t 20 -- 'bench/artifacts/quantized-gemm/mxblas/build.sh'` runs the build + its own tests on an idle H100.
- Deviation from the recorded ruling: blocker changed from "no sm_90a hardware (deferred)" to "sm_90a H100 present here but not schedulable in the session window".
- Verdict here: DEFERRED-HARDWARE (sm_90a H100 present on zaratan but queue ~3 days out; build ready, own-test gate pending an idle H100) — ruling unchanged.

## Reproduction on zaratan — H100 run (2026-09-10)

- Machine: UMD zaratan, NVIDIA **H100 80GB HBM3 (sm_90)**, node gpu-a6-7, via
  `sbatch -p gpu-h100 --gres=gpu:h100:1 -t 90:00`.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch 2.8.0+cu128,
  Python 3.12.14 (>= 3.12 satisfied); MXBLAS's own hardcoded compute_90a / FP8.
- Build + own-test gate (build.sh runs MXBLAS's tests/test_jit.py + tests/test_mxgemm.py
  when a GPU is visible): **test_jit.py PASSES on H100** ("Hello, MXBLAS! JIT test passed"),
  confirming MXBLAS's FP8/sm_90a JIT machinery works on this hardware. But
  **tests/test_mxgemm.py (M=N=K=2048, per-tensor scaling) did NOT complete** — its
  FP8 GEMM kernel's single nvcc/CUTLASS compile exceeds even a 90-minute H100 job
  (the job was CANCELLED at the wall-clock limit still inside that compile, across
  20/30/90-minute attempts, always at the same point). The compile also cannot be
  pre-warmed on the login node the way Voltrix's was: test_mxgemm.py allocates GPU
  tensors (torch, device=cuda) BEFORE compiling, so on a GPU-less node it aborts
  with "Found no NVIDIA driver" before any nvcc runs.
- Deviation from the recorded ruling: the DEFERRED blocker is partly resolved
  (sm_90a hardware is present and MXBLAS's JIT + test_jit work on it), but the
  headline mxgemm correctness test could not be completed within the H100 job
  budget available here (its FP8 CUTLASS kernel compile is > 90 min and is not
  precompilable off-GPU). No gate was loosened.
- Verdict here: **DEFERRED-HARDWARE (partial): builds + JIT-smoke (test_jit) PASS on
  H100 sm_90a; the FP8 test_mxgemm gate's compile exceeds a 90-min job and did not
  finish.** Needs a longer single H100 allocation (or a machine where the FP8
  CUTLASS compile is faster) to complete the mxgemm correctness test.


## Reproduction on zaratan — H100 run COMPLETED (2026-09-10)

Follow-up to the 30/90-min attempts above: a **3-hour** H100 job (sbatch
`-p gpu-h100 -t 180:00`, node gpu-a6-*, COMPLETED in 1:48:01) let MXBLAS's own
FP8 kernel finish its single slow nvcc/CUTLASS compile and run to completion.

- Machine: UMD zaratan, NVIDIA H100 80GB HBM3 (sm_90), 3-hour allocation.
- Toolchain: nvcc 12.8.93, g++ 13.4.0, torch 2.8.0+cu128, Python 3.12.14.
- Build + own-test gate: `build.sh` on the H100 ran MXBLAS's tests/test_jit.py
  ("JIT test passed") and tests/test_mxgemm.py (M=N=K=SM=SN=SK=2048, TT
  per-tensor scaling): **"MXBLAS test completed successfully" — difference rate
  0.0710%, 432.9168 TFLOPS, 0.0397 ms.** exit 0.
- Deviation from the recorded ruling: the DEFERRED-HARDWARE blocker is now fully
  resolved — MXBLAS builds and its own FP8 correctness+perf test passes on H100.
  The only reason the earlier 30/90-min jobs failed was compile wall-clock: the
  FP8 CUTLASS kernel's single compile takes ~1.5 h; 3 h is enough.
- Verdict here: **BUILT + own-test GATED on H100 (sm_90a).** Supersedes the
  recorded DEFERRED-HARDWARE ruling. Still no kernelbench `--impl` adapter (the
  quantized-gemm domain does not model the FP8/MX regime; not changed, since
  editing bench/kernelbench is out of scope) — the gate here is MXBLAS's own
  test suite, per its build.sh design.
