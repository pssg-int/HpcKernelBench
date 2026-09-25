# Running the stencil track on zaratan: prompt for a coding agent

> Scope: the `stencil` kernel track ONLY (5 paper adapters + the library
> baselines the papers compare against), not the full 98-artifact suite. For a
> from-scratch reproduction of everything, use `bench/REPRODUCE.md` instead.
> zaratan is already set up (see `bench/ENVIRONMENT.md` §9) — this prompt does
> not redo environment setup, it builds and runs one track.

Paste the block below into a coding agent (Claude Code or similar) started in
a fresh clone of this repository on zaratan, after `git pull` has brought in
the 2026-09-25 changes (cuDNN/cuFFT baselines, per-paper baseline report,
shared-configuration run plan).

---

```text
You are working in a clone of HPC-KernelBench, on zaratan. Run every
command below from the REPOSITORY ROOT unless it says otherwise.

GOAL. Measure every stencil paper against the baselines ITS OWN evaluation
used, and run every implementation on each paper's evaluation configuration
wherever it supports it, all on one GPU model (full A100). Report what you
get, including every failure, honestly. Do not "fix" the benchmark (spec,
harness, or a paper's own kernel source) to make a result pass. Exceptions:
bench/artifacts/stencil/cudnn/bridge.cu (new 2026-09-25) and the
convstencil/lorastencil bridge.cu + artifacts/stencil/periodic_halo.cuh are
OUR code; fixing a compile error there is allowed, but report exactly what
you changed. NOTE: on zaratan a normal user cannot lock GPU clocks
(`nvidia-smi -lgc` needs root), so the runner stamps every result
`conforming: false` with reason "GPU clocks not locked" even inside an
exclusive allocation -- per bench/REPRODUCTION_zaratan.md section 4, this is
expected on this machine and is NOT something to debug or work around; a
non-smoke, non-warmup/reps-overridden run with only that one
nonconformance_reason is the best attainable result here.

READ FIRST: CLAUDE.md; bench/CLAUDE.md; bench/ARTIFACT_GUIDE.md (rules 1-10
are binding); benchspecs/stencil/spec.yaml (especially the new
`paper_baselines` and `shared_configurations` sections);
benchspecs/stencil/survey.md (the "Baselines each paper ran" addendum at the
end); bench/artifacts/stencil/README.md and each adapter's own STATUS.md,
including the new bench/artifacts/stencil/cudnn/STATUS.md.

WHAT CHANGED ON 2026-09-25 (do not rediscover these the slow way):
- Each paper's own baselines were re-read from its paper and artifact
  scripts. Machine-readable list: kernelbench/domains/stencil.py
  PAPER_BASELINES. Integrated stand-ins:
    cudnn-stencil          ConvStencil's src/cudnn/conv_*.cu programs, which
                           ConvStencil, LoRAStencil and SPIDER report against:
                           cudnnConvolutionForward per step,
                           IMPLICIT_PRECOMP_GEMM, zero padding, fp64 (+ fp16).
                           NEW, never built or run on a GPU.
                           bench/artifacts/stencil/cudnn/{bridge.cu,build.sh}.
    cudnn-stencil-fastest  FlashFFTStencil's benchmarks/cudnn/cudnn-test.cpp:
                           same convolution, fastest of every forward
                           algorithm (searched once in prepare(); the pick is
                           recorded as params.cudnn_algo). Same bridge.so,
                           adapter in artifacts/stencil/cudnn_fastest/.
    torch-cufft-stencil    FlashFFTStencil's cuFFT baseline
                           (benchmarks/cufft-by-pytorch): irfftn(rfftn(U)*K)
                           per step, periodic. Pure torch, no build. Verified
                           on CPU only (err ~1e-15 vs reference).
    convstencil-tcu,       as baselines of the later papers.
    flashfftstencil-box2d1r
  Not integrated, with reasons in PAPER_BASELINES: AMOS, Brick, DRStencil,
  TCStencil, PPCG loop/hybrid tiling, STENCILGEN. Do not try to add them in
  this pass.
- torch-conv-stencil is NOT any paper's baseline (circular F.pad + conv per
  step). It stays as a fallback yardstick; the report's generic speedup
  column now prefers cudnn-stencil.
- Runner: each implementation now gets its OWN copy of every workload
  (FRESH_WORKLOAD_PER_IMPL), so SPIDER/FlashFFTStencil rewriting their
  workload no longer leaks into implementations listed after them -- the old
  "run SPIDER in its own job" rule is obsolete. New flag --keep-going records
  a crash (e.g. CUDA out of memory) under the result's `errors` and continues.
- The CPU reference is cached (one entry, KB_STENCIL_REF_CACHE), so the
  implementations gated under the same boundary convention in one job share
  it; that is why each job below runs ONE workload with all implementations.
- Report: groups results by the computation (shape, grid, T), not the
  workload's name; prints crashes; and adds a section "each paper vs the
  baselines its own evaluation used", including SPIDER's own cross-precision
  normalization (fp16 / 4 x 7/r vs fp64 box2d{1,2,3}r runs), labeled derived.

COVERAGE (unchanged adapters, as of 2026-09-23; an adapter that cannot run a
workload raises NotImplementedError and the runner records "unsupported"
with the reason -- a coverage fact, not a failure):
    an5d-stencil            star2d1r, box2d1r, star2d3r, box2d3r, star3d1r,
                            box3d1r; square/cube grids; AN5D's own
                            coefficients; fixed boundary.
    convstencil-tcu         any 2D stencil up to radius 3; grid a multiple of
                            32 x 64 (only star2d1r has ever been gated).
    lorastencil-star2d3r    star2d3r only; grid a multiple of 32 x 64.
    flashfftstencil-box2d1r box2d1r only; square, width a multiple of 6; T=1.
    spider-box2d7r-sptc     its own radius-7 box, one step, fp16; grid a
                            multiple of 64 x 128.
    cudnn-stencil(-fastest) any 1D/2D/3D linear stencil; fp64 and fp16.
    torch-cufft-stencil     any 1D/2D/3D linear stencil; fp64/fp32.

GPU CHOICE: A100 FOR EVERYTHING (decided 2026-09-24). Builds and --smoke
gates on the default MIG slice (no -g flag; a100_1g.5gb, sm_80); every REAL
timed run on a full card (-g a100). Do not use -g h100 in this pass. Older
result JSONs under bench/results/ are history: do not mix them into this
pass's report (use the file names listed in STEP 5).

GROUND RULES (same as bench/REPRODUCE.md)
- Never loosen a correctness gate, tolerance, or protocol
  (benchspecs/stencil/spec.yaml, bench/kernelbench/). A failed gate is a
  result, not a bug to work around.
- Never edit a paper's kernel code under any artifacts/stencil/*/source/
  checkout. Build-system fixes only (arch flags, include/link paths,
  CUDA-version guards); after such a fix run `git -C source diff >
  source.patch` in that artifact directory and set `patch=source.patch` in
  source.provenance.
- No machine-specific paths in scripts: use the KB_* variables from
  bench/env.sh.
- All GPU work goes through bench/gpu_run.sh; nothing GPU-related runs on
  the login node (it has no GPU). Each call is one blocking `srun` job; it
  sources env.sh and runs the quoted command with cwd = bench/, so paths
  INSIDE the quotes are relative to bench/. Keep the quotes single so $PY
  expands on the compute node. `bench/gpu_run.sh [-g <gres>] [-t MIN]
  [-m MEM] -- '<cmd>'`, no SLURM account flag; if srun rejects a job for a
  missing account or a memory request, that is a finding to report, not
  something to paper over by guessing.
- Bound every filesystem search to this repository (bench/CLAUDE.md).
- Read an artifact's STATUS.md before touching it; append to it, never
  rewrite its history (see step 4).

STEP 1: ENVIRONMENT (should already be set up on zaratan)
  source bench/env.sh
  $PY bench/check_env.py       # expect 0 MISSING
  (cd bench && ./smoke_all.sh) # CPU only; confirm stencil's lines are green
If check_env.py reports MISSING items or smoke_all.sh fails on stencil,
STOP and report that first.

STEP 2: BUILD (6 build scripts; cudnn_fastest has none, it uses cudnn's)
  for a in an5d convstencil lorastencil flashfftstencil spider cudnn; do
    bench/gpu_run.sh -t 30 -- "bash artifacts/stencil/$a/build.sh"
  done
Record exit code and the last lines per build. A build failure is a result
(BUILD-FAILED with the first real error line in that STATUS.md). cudnn's
build.sh finds cudnn.h and libcudnn.so.9 in $PY's nvidia-cudnn-cu12 wheel
(the same library torch loads); it prints the path and version -- record
them.
  (cd bench && $PY -m kernelbench.runner --kernel stencil --list)
    confirm every adapter shows as built ("ok"), including cudnn-stencil and
    cudnn-stencil-fastest.

STEP 3: SMOKE GATES (MIG slice), one job:
  bench/gpu_run.sh -t 30 -- '$PY -m kernelbench.runner --kernel stencil \
      --variant stencil-cpu-gpu-kernel-fp64 --keep-going \
      --impl an5d-stencil,cudnn-stencil,cudnn-stencil-fastest,convstencil-tcu,lorastencil-star2d3r,flashfftstencil-box2d1r,torch-conv-stencil,torch-cufft-stencil \
      --smoke'
  bench/gpu_run.sh -t 20 -- '$PY -m kernelbench.runner --kernel stencil \
      --variant stencil-tcu-matmul-kernel-fp16 --precision fp16 --keep-going \
      --impl spider-box2d7r-sptc,cudnn-stencil,cudnn-stencil-fastest --smoke'
Expect "unsupported" for smoke shapes an adapter cannot run. The new
baselines (cudnn-stencil, cudnn-stencil-fastest, torch-cufft-stencil) must
PASS on every linear smoke shape; if one fails its gate, stop and report
the error value -- do not proceed to the timed matrix with a failing
baseline. Note the algorithm cudnn-stencil-fastest picked (stderr lines
"cudnn_stencil: algo N ..." and params.cudnn_algo in the JSON).

STEP 4: THE SHARED-CONFIGURATION MATRIX (full A100 per job)
The plan is kernelbench/domains/stencil.py CROSS_PLAN; the driver prints or
runs it, one job per workload, all of that configuration's implementations
in one runner call, result files results/stencil_<sweep>_<workload>.json:
  $PY bench/artifacts/stencil/cross_jobs.py                 # print all 15 jobs
  $PY bench/artifacts/stencil/cross_jobs.py --run --sweep cross-tcu-2d
Run it on the LOGIN node (it only submits via gpu_run.sh, sequentially;
each job blocks until done). It skips jobs whose result file exists, so it
resumes after an interruption; run it in the background with a log (e.g.
nohup ... > cross.log 2>&1 &) and check the log. Order, most important first:
   1. cross-tcu-2d       5 jobs  ConvStencil Table 4 / LoRAStencil Table II /
                                 SPIDER Fig. 10 baselines: 10240^2, T=10240
   2. cross-flashfft     1 job   FlashFFTStencil Table 3 2D point, 16320^2, T=1
   3. cross-spider-fp16  1 job   SPIDER's radius-7 fp16 run vs cuDNN fp16
   4. cross-an5d         6 jobs  AN5D: 16384^2 / 512^3, T=1000
   5. cross-tcu-3d       2 jobs  LoRAStencil 3D: 1024^3, T=1024 (-m 160G)
Budgets are in CROSS_PLAN (minutes/mem per job; 3D asks 160G host memory
because the CPU reference of a 1024^3 grid needs ~60 GB). If a job hits its
time limit, record which implementation it was on (the log shows the
"running <impl> <workload>" line) and re-run that workload with the
remaining implementations only, using the runner command the driver printed
with a shorter --impl list and a different --out.
Expected, not failures: unsupported rows (shape/grid/dimension coverage);
torch-conv-stencil and torch-cufft-stencil running out of device memory at
1024^3 (recorded under `errors` by --keep-going; FlashFFTStencil's paper
reports FFT stencils' memory footprint as a result, so an OOM is data).
Unexpected, report with evidence: a gate failure of any implementation, a
crash that is not an OOM, torch-conv-stencil far below its earlier A100
number (star2d1r @10240^2, T=10240: 37.28 GCUP/s).
Record for every job the `[gpu_run] host=... gpu=... job=...` line, the
"N/M runs valid (k unsupported, c crashed)" line and the conforming line.

STEP 5: RECORD
Append to each artifact's STATUS.md (an5d, convstencil, lorastencil,
flashfftstencil, spider, cudnn) a section
  ## Shared-configuration runs on zaratan A100 (<YYYY-MM-DD>)
with: build outcome; smoke-gate outcome; per workload it ran: GCUP/s, the
paper's own metric, correctness error and gated step count
(protocol_used.correctness_gate), conforming; unsupported workloads with
the reason; crashes. Never delete or rewrite earlier STATUS.md text -- put
the new section first and keep the old text after "Earlier outcome, kept
for the record:".

STEP 6: REPORT
  (cd bench && $PY -m kernelbench.report results/stencil_cross-*.json \
      --html results/stencil_report_cross_a100.html)
Then write a short summary for the user with two tables:
  (a) each paper vs its own baselines, from the report's "each paper vs the
      baselines its own evaluation used" section: paper, workload, its
      GCUP/s, speedup over each integrated baseline, and the baselines not
      integrated (one line: see PAPER_BASELINES);
  (b) the same-configuration leaderboard per sweep: every implementation's
      GCUP/s on each workload it ran, with unsupported/crashed marked.
Point out where our measured speedup over a paper's baseline disagrees with
the paper's own claim by more than 2x (ConvStencil and SPIDER both report
"x over cuDNN"; see spec.yaml metrics.per_paper_native), without
speculating beyond the evidence. Keep the SPIDER derived comparison
labeled as the paper's normalization, not a measurement.
Commit in small commits, one per artifact directory touched, never
checkouts/build products/results JSON over 1 MB. Commit messages: what
changed and why, one paragraph.
```
