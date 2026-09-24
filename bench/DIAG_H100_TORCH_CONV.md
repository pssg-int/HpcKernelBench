# Prompt: diagnose (and fix) the slow torch-conv-stencil baseline on H100

Paste everything below the line into a Claude Code session started in the repo
root on zaratan.

---

You are working in the HPC-KernelBench repo on zaratan (UMD). Read `CLAUDE.md`
and `bench/STENCIL_ON_ZARATAN.md` first. Run `source bench/env.sh` in every
shell. Login nodes have no GPU: every GPU command MUST go through
`bench/gpu_run.sh` (e.g. `bench/gpu_run.sh -g h100 -t 15 -- '$PY ...'`).

## The problem

`torch-conv-stencil` (class `TorchConvStencil`, `bench/kernelbench/impls/gpu_cuda.py`)
is the library baseline for the stencil track: per timestep it does a circular
`F.pad` then `F.conv2d`/`F.conv3d` in fp64, which dispatches to cuDNN. Its
results are implausible on H100:

| GPU  | workload                        | ms per step | GCUP/s |
|------|---------------------------------|-------------|--------|
| A100 | star2d1r @ 10240^2, T=10240     | ~2.8        | 37.28  |
| H100 | star2d1r @ 16384^2, T=1000      | ~150        | 1.79   |
| H100 | box2d1r  @ 16384^2, T=1000      | ~150        | 1.79   |
| H100 | star2d3r @ 10240^2, T=10240     | ~121        | 0.86   |
| H100 | star3d1r / box3d1r @ 512^3      | ~83         | 1.61   |

An fp64 single-sweep stencil on H100 should be bandwidth-bound near
~100-200 GCUP/s; 1.79 is <1% of that and ~20-40x slower than the same code on
A100. Star == box and radius-3 only ~2x slower than radius-1 (5.4x the FLOPs)
say the time is not spent on arithmetic. This inflates every H100 "speedup vs
baseline" in the stencil report (an5d 75-335x, lorastencil 195x, etc.).

Unconfirmed hypotheses, in order: (1) cuDNN's heuristic picks a slow fp64
one-channel algorithm on H100 (`cudnn.benchmark` is never set); (2) the H100
jobs use a different/broken torch or cuDNN build than the A100 jobs
(`gpu_cuda.py` docstrings already record a cuDNN install missing its engine
plugin libraries); (3) the per-step `F.pad` allocation + copy (worth ~2x,
cannot explain ~100x).

## Tasks

1. **Diagnose.** `bench/diag_torch_conv.py` already exists (one 16384^2 fp64
   3x3 step: versions, plain-copy bandwidth, pad alone, conv with
   cudnn.benchmark off/on, profiler kernel names). Run it on both GPUs:
   ```
   bench/gpu_run.sh -g h100 -t 15 -- '$PY diag_torch_conv.py' 2>&1 | tee bench/diag_h100.txt
   bench/gpu_run.sh -g a100 -t 15 -- '$PY diag_torch_conv.py' 2>&1 | tee bench/diag_a100.txt
   ```
   Extend the script if the output does not settle the cause (e.g. also try
   3D, radius 3, `torch.backends.cudnn.enabled=False`, fp32 for contrast,
   `CUDNN_LOGINFO_DBG=1`/`CUDNN_LOGDEST_DBG=stdout` to log the chosen engine).
   Also check which Python/torch/cuDNN the runner uses on each node
   (`$PY -c 'import torch; print(torch.__file__, torch.__version__, torch.backends.cudnn.version())'`
   through gpu_run.sh on both).

2. **Fix, if the cause is on our side.** Allowed fixes to `TorchConvStencil`:
   enable `torch.backends.cudnn.benchmark` (set it in `prepare()`, and make
   sure autotuning happens in warm-up, not inside the timed region);
   pre-allocate the padded buffer and refresh only its halo instead of calling
   `F.pad` into a new tensor every step; environment fixes (matching cuDNN
   install) documented in `bench/ENVIRONMENT.md`. Record the cuDNN version
   in the result if the result schema has a place for it. Keep the math identical
   (same weights, periodic boundary, cross-correlation layout).

3. **Verify.**
   - `cd bench && ./smoke_all.sh` must stay green.
   - Re-run the baseline smoke gate on H100:
     `bench/gpu_run.sh -g h100 -t 20 -- '$PY -m kernelbench.runner --kernel stencil --variant stencil-cpu-gpu-kernel-fp64 --impl torch-conv-stencil --smoke'`
   - Re-run the real baseline on H100 for `star2d1r` and `box3d1r` (variant
     defaults: 16384^2 / 512^3, T=1000), and on A100 for
     `star2d1r@10240x10240:T=10240`, so the before/after is comparable to the
     table above.

## Ground rules

- Never loosen a correctness gate, tolerance, or protocol
  (`benchspecs/stencil/spec.yaml`, `bench/kernelbench/`). A failed gate is a
  result.
- Never edit paper kernel code under `bench/artifacts/stencil/*/source/`.
- Do not change the research adapters to make them look better or worse;
  this task is only about the baseline.
- If the cause is cuDNN itself (no fast fp64 one-channel path on H100 even with
  autotuning and a correct install), say so plainly — do not hide it. Then
  propose, but do not implement, a hand-written CUDA stencil baseline.
- Do not commit or push; leave changes in the working tree.

## Deliverable

Write `bench/DIAG_H100_TORCH_CONV_RESULT.md` containing: the root cause, with
the evidence lines from `diag_h100.txt` / `diag_a100.txt`; what you changed
(files, one line each); a before/after table (ms per step and GCUP/s, per GPU
and workload); and anything still unexplained. Then print a 5-line summary.
