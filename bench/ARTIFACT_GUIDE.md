# Integrating a paper artifact

How a paper's own kernel implementation becomes a competitor in this harness.
The point: the artifact runs under OUR protocol (spec-driven timing, correctness
gate before timing, preprocessing split) — not its own scripts — so its numbers
are comparable with every other implementation of the track.

## Layout

```
bench/artifacts/<kernel>/<shortname>/
  source/        the cloned repo (git clone --depth 1; keep .git for provenance)
  build.sh       idempotent build script; exit 0 = built. Records versions used.
  adapter.py     wraps the artifact as a kernelbench Implementation (contract below)
  STATUS.md      honest state: BUILT+GATED / BUILT / BUILD-FAILED / SKIPPED /
                 DEFERRED-HARDWARE, with exactly what was done, patched, observed
  REQUIRES_GPU   only for DEFERRED-HARDWARE: one line, the minimum architecture
                 and the feature that needs it (e.g. `sm_90a  cp.async.bulk + mbarrier`)
```

`<shortname>` = the paper's system name, lowercase (e.g. `rode`, `fused3s`).

## adapter.py contract

```python
KERNEL = "spmm"                  # track this competes in
IMPL_NAME = "rode-spmm"          # unique implementation name
PAPER_KEY = "conf/ppopp/..."     # dblp key, ties the result to the paper
PRECISIONS = ["fp32"]            # what the artifact actually supports

def available() -> tuple[bool, str]:
    """(True, "") if built and loadable; (False, reason) otherwise.
    MUST be cheap and never raise."""

def create(precision: str):
    """Return an object satisfying kernelbench.harness.Implementation:
    prepare(workload, params) / run(handle) / to_host(out) / timer() / free().
    prepare() may convert CSR into the artifact's native format — that IS the
    artifact's preprocessing and the harness times it as such."""
```

The registry (`kernelbench/artifact_registry.py`) discovers adapters
automatically; `runner.py --impl <IMPL_NAME>` then works like any built-in.

## Rules

1. **Wrap the kernel, not the paper's benchmark script.** Call the artifact's
   library/binary entry point for ONE kernel invocation from `run()`. If the
   artifact only ships an end-to-end binary that loads matrices itself, wrap at
   the finest boundary available and document the contamination in STATUS.md.
2. **The artifact's own format conversion goes in `prepare()`** — it is
   preprocessing under our specs, timed once and reported separately. Never
   pre-convert offline and hide it.
3. **Patch minimally, record every patch.** Build-system fixes (arch flags,
   include paths, CUDA-version guards) are fine; touching kernel code is not —
   if the kernel itself must change to run, mark SKIPPED and say why.
   `source/` is git-ignored repo-wide (5+ GB of clones); only
   `<short>/source.provenance` (remote + commit) and `<short>/source.patch`
   (our diff to TRACKED files) are versioned. Therefore every file WE write
   (shims, drivers, wrappers, compat headers) lives in `<short>/`, never only
   inside `source/`; if the build needs it in-tree, `build.sh` copies it in
   (precedent: `sequence-alignment/logan/build.sh`). Likewise anything the
   build fetches (a CUTLASS checkout, a pip `--target` tree) must be fetched
   by `build.sh`, not by hand.
4. **Correctness gate is non-negotiable.** The artifact's output goes through
   the same gate as every other impl. If it fails the gate, that IS a result —
   record it in STATUS.md; do not loosen the gate to make it pass.
5. **Login node: build + a minimal functional/gate check only.** A handful of
   kernel launches to verify correctness is acceptable; timing measurements are
   NOT (shared GPU — the runner marks them non-conforming anyway). Full timed
   runs happen later on a compute-node allocation.
6. Pin provenance in STATUS.md: commit hash, CUDA/toolchain versions, GPU arch
   flags used.
7. If the artifact genuinely does not implement the track's kernel (it happens:
   RASSM ships no SpMV despite the paper), mark SKIPPED with evidence and move
   to the next candidate.

8. **Unsupported workloads: raise `NotImplementedError` from `prepare()`**,
   naming the constraint (fixed head dims, power-of-two sizes only, k<=2048,
   ...). The runner records such (impl, workload) pairs under `"unsupported"`
   in the result file and prints `UNSUPPORTED: <reason>`, then continues with
   the next workload — they count neither as valid nor as failed runs. Never
   silently return garbage for a shape the kernel cannot handle.
9. **Newer-GPU-only artifacts are DEFERRED-HARDWARE, not SKIPPED** (user
   decision, 2026-09-05: they will be integrated on another machine). When the
   only blocker is the GPU architecture (hardcoded `sm_90a`/`sm_100`, Hopper
   PTX such as `cp.async.bulk`/`wgmma`/`mbarrier`, FP8 tensor cores, ...):
   keep the clone + `source.provenance`, write `REQUIRES_GPU` (minimum arch +
   the feature that needs it), start STATUS.md with
   `**Status: DEFERRED-HARDWARE (needs sm_90a) — ...**`, and leave a `build.sh`
   that should work on the right machine if you got that far. Cite the evidence
   (file:line of the arch flag or the PTX, or the runtime `cudaErrorNoKernelImageForDevice`).
10. **`available()` must never compile.** It may check files and dlopen a
   prebuilt library; compilation (including torch's JIT `load()` with a
   "fresh" cache) belongs in build.sh only. A JIT call inside `available()`
   turned every registry scan into a 10-minute rebuild the day the default
   toolkit module changed (flashattention-t, 2026-09-04).

## Build environment

**Machine facts and setup steps live in `bench/ENVIRONMENT.md`; the knobs in
`bench/env.sh` (`KB_*`).** Never hard-code a machine path in a build.sh or
adapter: use `${PY:-…}`, `${CC:-…}`, `$CUDA_HOME`, `$KB_MPI_ROOT`,
`$KB_BLAS_LIBDIR`, `$KB_HOST_COMPILER_BIN` with the Perlmutter value only as the
fallback default, and record the versions actually used in STATUS.md.

Reference machine (Perlmutter, 2026-09):

- nvcc 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`), GPU sm_80 (A100)
- Python: /pscratch/sd/c/cunyang/gnn/plexus_env (torch 2.8 cu128, numpy, scipy)
- No sudo; no system package installs. Header-only deps: vendor into source/.
- SuiteSparse matrices cached in bench/matrices/ (loader: kernelbench.matrices)
- **Toolchain pin (required since 2026-09-04).** The login environment's
  default module became `cudatoolkit/13.2` (HPC SDK 26.5): `CUDA_HOME`,
  `NVHPC_CUDA_HOME`, the `nvcc` on PATH and `CPATH` (headers!) all point at
  13.2 while torch is cu128. Every build.sh must therefore start with

  ```bash
  source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"   # bench/artifacts/toolchain.sh
  ```

  which is equivalent to what `module load cudatoolkit/12.9` exports in an
  interactive shell (the `module` function is not available inside a plain
  `#!/usr/bin/env bash` script, hence the file):

  ```bash
  export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
  export NVHPC_CUDA_HOME="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  # RE-POINT CPATH, do not unset it: 13.2's cudaTypedefs.h breaks CUTLASS
  # (no unversioned PFN_* aliases), but torch's ATen/cuda headers need
  # cusparse.h, which lives only in the SDK's math_libs tree.
  export CPATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/include:$CUDA_HOME/include"
  # link/run-time: cuBLAS/cuSOLVER/cuRAND/cuSPARSE/cuFFT live in math_libs, not cuda/12.9/lib64
  export LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64:$CUDA_HOME/lib64"
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  ```
  Note: CUDA 12.9's SDK ships no `libnvToolsExt.so` (NVTX v3 is header-only);
  a `-lnvToolsExt` in an artifact's build system is a build-system fix (drop it).

  The runner does the same automatically at startup (`env.pin_cuda_toolchain`,
  printed as `[env] toolchain: ...`; override with `KB_CUDA_HOME`), so gate
  runs and runtime-JIT DSLs (tilus, qfactory, hidet) get the right toolkit
  without shell setup — but shell builds do not go through the runner.
- torch CUDA extensions need `LD_PRELOAD=/usr/lib64/libstdc++.so.6` at run
  time (CXXABI_1.3.15; see any torch-extension STATUS.md).
```

## Baseline selection rule (user decision, 2026-09-05)

"Newest 3 open-source artifacts per track" is retired: applied blindly it
picked a DSL, an off-regime LLM-sparsity kernel and a CPU kernel as the SpMM
baselines while skipping the canonical GPU SpMM line (RoDe, DTC-SpMM,
FlashSparse, ...). Baselines are now chosen from
`kernel-papers/output/baseline_selection.md`, produced by
`select_baselines.py` from the per-(paper, track) ratings in
`output/kernel_centrality.json`:

1. **kernel centrality** `core` (the kernel IS the paper's headline
   contribution, evaluated at kernel level) > `component` (a part of a bigger
   system / pipeline / codegen output) — `tangential` is never a baseline;
2. **regime match** with the track spec's inputs: `matches` > `partial` >
   `mismatch`;
3. **single-NVIDIA-GPU path** required (current scope);
4. **recency** only as the tiebreak; up to 5 per track.

Already-integrated adapters that the rule would not have picked stay in the
registry as competitors (they still run under the same gate) but are labelled
`component` / off-regime in their STATUS.md and on the status page, and are not
the "human SOTA" reference in Phase 3.

## Scope ruling (user decision, 2026-08-07)

Integration targets **NVIDIA GPU, single-card** implementations only for now.
CPU-only, multi-GPU/distributed-only, and FPGA artifacts are SKIPPED with a
one-line reason (platform out of current scope) — cheap skip, no build attempt.
Already-integrated CPU adapters (rassm spmm/sddmm, sspmv, diaq) stay in place
as backups; do not delete them.

## NERSC Filesystem Safety (REQUIRED — admin warning received 2026-08-07)

Never recursively traverse `/`, `/global`, `/global/cfs`, `/global/homes`,
`/pscratch`, `/opt`, `/usr`, `/cvmfs`, or any other shared top-level directory.
This covers: `find`, `bfs`, `fd`, `tree`, recursive `du`, `rg --files`,
recursive `grep`, recursive `ls`, globstar expansion, and recursive traversal
in Python or any other language.

Before searching, identify a bounded root inside the current workspace or a
known project or data directory. Constrain depth and filename patterns.

To locate software use `command -v`, `type -a`, `module spider`, package
metadata, or known environment prefixes. Do not search mounted filesystems for
executables or libraries. A compute allocation is not permission for an
unbounded traversal of a shared filesystem.
