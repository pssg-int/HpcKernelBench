# Ansor-AF-DS — convolution — STATUS: SKIPPED

- Paper: "Accelerated Auto-Tuning of GPU Kernels for Tensor Computations"
  (ICS'24). `PAPER_KEY = conf/ics/LiXSS24`.
- Repo: https://github.com/HPCRL/Ansor-AF-DS, commit
  `bd6adf3a5463e3e7c2d6a2c581abc8ce7ede1797` (default branch `main`),
  `git clone --depth 1` (non-recursive — see below for why the submodules
  were never fetched).

## What the artifact actually is

Ansor-AF-DS is not a conv2d *kernel*; it is a modified TVM auto-scheduler
(a fork tracked via `.gitmodules` at `benchmarks/Ansor_AF_DS` ->
`git@github.com:HPCRL/ics24tvm.git@ics_1000trials_AF_DS`, i.e. the entire
TVM tree, not a small patch). The paper's contribution — and its own
`README.md`'s explicit request for "num_of_runs num_sm num_shared_mem
network num_trials num_init_states threshold pz_num" — is speed of the
*search* itself (fewer trials/wall-clock to reach near-oracle schedule
quality), not the steady-state throughput of a fixed kernel. This is
exactly why this project's own `benchspecs/convolution/spec.yaml` already
carves this artifact into a dedicated `conv-tuning-cost` variant ("target:
reach >=95% of a one-time 'oracle' reference GFLOP/s... search_budget_cap:
30 minutes wall-clock OR 1000 compile-execute trials... timer: host wall
clock around the ENTIRE SEARCH LOOP") — a fundamentally different track
from `conv-dense-kernel-fp32` (steady-state kernel throughput of one
already-tuned kernel), which is what this integration task gates against.

## Why SKIPPED rather than attempted

Per ARTIFACT_GUIDE.md / the task brief: "wrap ONE pre-tuned conv kernel if
runnable without hours of search; else SKIP with evidence." Checked both
halves:

1. **No pre-tuned kernel to load.** The README documents exactly one
   benchmark entry point, `run_tests_times_conv.sh conv2d cuda <num_runs>
   <num_sm> <num_shared_mem> <network> <num_trials> <num_init_states>
   <threshold> <problem_size>`, e.g. `bash run_tests_times_conv.sh conv2d
   cuda 3 128 48 yolo 5 64 0.6`. `num_trials=5` start points x
   `num_init_states=64` initial configurations is a live autotuning search,
   not a saved-schedule replay; nothing under `benchmarks/`,
   `default_ansor_benchmarks/`, or `cal_var/` in the shallow clone is a
   serialized Ansor tuning log (`*.json` schedule records) that could be
   loaded once and executed without re-running the search — confirmed by
   listing the shallow (non-recursive) clone's top level: `benchmarks/`,
   `cal_var/`, `default_ansor_benchmarks/`, `figures/`, `README.md`; the
   actual kernel-generating code lives inside the `Ansor_AF_DS`/`Ansor_AF`/
   `Ansor_DS`/`Ansor` submodules (each a full TVM checkout), which this
   clone never fetched (see next point) — so even the "did they ship a log"
   question is answered by the README's own instructions, which describe
   only the search command, never a `--replay-log` or equivalent flag.
2. **Build is structurally out of budget, independent of search time.**
   Each of the four submodules is `git@github.com:HPCRL/ics24tvm.git`
   (full TVM), built via `cmake .. && make -j8` after `set(USE_CUDA ON)` /
   `set(USE_LLVM ON)` in `config.cmake`. The README's own prerequisite
   check is `llvm-config --version` — this machine has **no `llvm-config`**
   (`command -v llvm-config` returns nothing) and no sudo / system package
   install is permitted (ARTIFACT_GUIDE.md build-environment rule), so the
   TVM build cannot even start without first building or locating an LLVM
   toolchain ourselves — a second multi-artifact undertaking on top of the
   TVM build itself, which independently is a well-known 20-60+ minute
   `make` even once LLVM is present. This alone exceeds the ~25 min/artifact
   budget before any tuning trial is run.

Both conditions in the task's SKIP clause are met, so no build was
attempted (`.gitmodules` submodules were never fetched — the depth-1
non-recursive clone above is sufficient to read the README and top-level
layout that support this decision). No `build.sh`/`adapter.py` written,
matching this repo's convention for a documented, evidence-based SKIP
(see `../../spmv/rassm/STATUS.md`, `../../lossy-compression/ffcz/STATUS.md`
for the same pattern: `source/` + `STATUS.md` only).
