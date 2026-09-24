# Running the stencil track on zaratan: prompt for a coding agent

> Scope: the `stencil` kernel track ONLY (5 paper adapters), not the full
> 98-artifact suite. For a from-scratch reproduction of everything, use
> `bench/REPRODUCE.md` instead. zaratan is already set up (see
> `bench/ENVIRONMENT.md` §9) — this prompt does not redo environment setup,
> it builds and runs one track.

Paste the block below into a coding agent (Claude Code or similar) started in
a fresh clone of this repository on zaratan, after `git pull` has brought in
the 2026-09-23 changes (stencil spec, runner/report/harness, and the rewritten
convstencil/lorastencil bridges).

---

```text
You are working in a clone of HPC-KernelBench, on zaratan. Run every
command below from the REPOSITORY ROOT unless it says otherwise.

GOAL. Build and run the 5 paper adapters in the stencil track
(bench/artifacts/stencil/{an5d,convstencil,lorastencil,flashfftstencil,spider}),
get a real (non-smoke) performance number for each at its paper's own size
on a shape it supports, and report what you get, including any failure, honestly.
Do not "fix" the benchmark (spec, harness, or a paper's own kernel source) to
make a result pass. The one exception: the convstencil and lorastencil
bridge.cu files and artifacts/stencil/periodic_halo.cuh are OUR code, new on
2026-09-23 and never compiled on a GPU yet; fixing a compile error there is
allowed, but report exactly what you changed. NOTE: on zaratan a normal user cannot lock GPU clocks
(`nvidia-smi -lgc` needs root), so the runner will stamp every result
`conforming: false` with reason "GPU clocks not locked" even inside an
exclusive allocation -- per bench/REPRODUCTION_zaratan.md section 4, this is
expected on this machine and is NOT something to debug or work around; a
non-smoke, non-warmup/reps-overridden run with only that one
nonconformance_reason is the best attainable result here.

READ FIRST: CLAUDE.md; bench/CLAUDE.md; bench/ARTIFACT_GUIDE.md (rules 1-10
are binding); benchspecs/stencil/spec.yaml; benchspecs/stencil/survey.md;
bench/artifacts/stencil/README.md and each adapter's own STATUS.md.

GPU CHOICE: A100 FOR EVERYTHING (decided 2026-09-24). Every stencil
adapter is built for sm_80 (-arch=sm_80 in all 5 build.sh); ConvStencil and
LoRAStencil were designed and evaluated on A100, FlashFFTStencil and SPIDER
were evaluated on A100 and H100, and AN5D needs no special hardware. None
needs H100. Earlier H100 runs were A100 code JIT-compiled from PTX, and on
H100 the torch-conv-stencil baseline was ~20-40x slower than on A100 (being
diagnosed separately, bench/DIAG_H100_TORCH_CONV.md), which inflated every
H100 speedup. So:
  - builds and --smoke gates: the default MIG slice (no -g flag; a100_1g.5gb,
    sm_80, allocates in seconds);
  - every REAL timed run: a full card, `-g a100`, so all adapters and the
    baseline are measured on the same GPU model.
Do not use -g h100 in this pass. Earlier H100 result JSONs may exist under
bench/results/; they are not the headline -- report the A100 numbers.

IMPORTANT CONTEXT (do not rediscover these the slow way):
- Coverage is narrow and per-adapter (as of 2026-09-23):
    an5d-stencil            6 shapes: star2d1r, box2d1r, star2d3r, box2d3r,
                            star3d1r, box3d1r. Uses AN5D's own coefficients
                            and a fixed boundary. DECLINES AN5D's named
                            kernels (j2d5pt, j2d9pt, j2d9pt-gol, gradient2d,
                            j3d27pt): implemented in the domain, not bridged.
    convstencil-tcu         any 2D stencil up to radius 3; grid must be a
                            multiple of 32 x 64. Only star2d1r has ever been
                            gated; other 2D shapes are untested.
    lorastencil-star2d3r    star2d3r only (its other kernels divide by zero or
                            ignore the weights); grid a multiple of 32 x 64.
    flashfftstencil-box2d1r box2d1r only; square grid, width a multiple of 6;
                            one step per call (the adapter forces T=1).
    spider-box2d7r-sptc     always runs its own radius-7 box for one step, in
                            fp16, whatever shape you pass; 2D only; grid a
                            multiple of 64 x 128.
  An adapter that cannot run a workload raises NotImplementedError; the
  runner records it as "unsupported" with the reason. That is a coverage
  fact, not a failure.
- NEW BRIDGES (2026-09-23): convstencil and lorastencil now keep the grid on
  the GPU for the whole T-step run (bridge.cu launches the artifact's own
  kernel and refreshes the periodic halo on the device between steps). They
  have NEVER been built or gated on a GPU. Their earlier timings included a
  host round trip every step and are NOT comparable to the papers; do not
  quote them. Each adapter's STATUS.md has a "Bridge rewrite" section.
- PRECISION GROUPS: convstencil, lorastencil and flashfftstencil compute in
  fp64, so run them under --variant stencil-cpu-gpu-kernel-fp64 (tolerance
  1e-5) and pass each paper's own size explicitly (table below). Only
  spider (fp16) runs under --variant stencil-tcu-matmul-kernel-fp16
  (tolerance 1e-2). The report never ranks fp16 and fp64 results together.
- WORKLOAD NAMES: `<shape>[@<grid>][:T=<steps>]`, e.g.
  `star2d3r@10240x10240:T=10240`. Without @/:T the variant's defaults apply
  (fp64 variant: 16384^2 / 512^3 at T=1000; fp16 variant: 10240^2 at
  T=10240). `--matrices sweep:<id>` expands to a paper's whole table or
  figure (ids in kernelbench/domains/stencil.py SWEEPS); do not start sweeps
  in this pass.
- CORRECTNESS GATE AT FULL SIZE: the gate compares against a CPU numpy
  reference, which would take hours at paper sizes. So for large workloads
  the runner gates on the SAME grid for fewer steps (e.g. 22 instead of
  1000), then times the full run. The result records this under
  protocol_used.correctness_gate and in correctness.note. Budget roughly
  5-10 minutes of CPU per workload for the gate -- use `-t 60` per job.
- REPORT: `kernelbench.report` now prints each paper's own metric under its
  row (AN5D: GFLOP/s; the others: GStencil/s + execution time), lists
  unsupported workloads with reasons, ranks only within one GPU and
  precision, and shows a speedup column against torch-conv-stencil when that
  baseline ran on the same GPU, workload and precision.
- RUN SPIDER IN ITS OWN JOB. The runner shares one workload object across all
  --impl entries, and spider's prepare() rewrites that workload's shape to
  its radius-7 box, which would leak into any adapter listed after it.
- kernelbench/domains/stencil.py's synthetic weights changed on 2026-09-21
  (now sum to 1; see check10_stencil_gate_not_vacuous.py), so gate numbers in
  STATUS.md from before that date are stale for convstencil/lorastencil/
  flashfftstencil. an5d and spider use their papers' own coefficients.
- Last recorded H100 status (bench/H100_baselines.md, h100_status.json,
  BEFORE the 2026-09-21/23 changes): an5d PASS (4.76 GCUP/s), spider PASS,
  convstencil REGRESSION (CUDA error 700: illegal memory access; possibly a
  grid not aligned to 32 x 64, which the adapter now refuses -- check),
  flashfftstencil and lorastencil UNSUPPORTED by --smoke (--smoke now has a
  96x96 box2d1r and a 64x128 star2d3r for them). Investigate and report
  anything that recurs; do not paper over it.

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
  the login node (it has no GPU). This includes smoke gates, builds that
  need nvcc/a GPU, and any one-off Python that touches a kernel. Each call
  is one blocking `srun` job on the gpu partition; it sources env.sh and
  runs the quoted command with cwd = bench/, so paths INSIDE the quotes are
  relative to bench/ (artifacts/..., not bench/artifacts/...). Keep the
  quotes single so $PY expands on the compute node.
- Call gpu_run.sh exactly as documented in bench/ENVIRONMENT.md §9 /
  bench/REPRODUCTION_zaratan.md: `bench/gpu_run.sh [-g <gres>] [-t MIN] --
  '<cmd>'`, no SLURM account flag. Every recorded zaratan run in this repo
  used a bare `-p gpu --gres=gpu:...` allocation with no `-A`; if `srun`
  rejects a job for a missing/ambiguous account on this account, that is a
  new finding to report, not something to silently paper over by guessing
  an account.
- Bound every filesystem search to this repository (see bench/CLAUDE.md's
  NERSC-derived filesystem-safety note; the same discipline applies here:
  no unbounded find/grep -r/du/tree over a shared root).
- Read an artifact's STATUS.md before touching it; append to it, never
  rewrite its history (see step 4).

STEP 1: ENVIRONMENT (should already be set up on zaratan)
  source bench/env.sh
  $PY bench/check_env.py       # expect 0 MISSING
  (cd bench && ./smoke_all.sh) # CPU only; confirm stencil's lines are green
If check_env.py reports MISSING items or smoke_all.sh fails on stencil,
STOP and report that first -- do not proceed to GPU builds on a broken base.

STEP 2: BUILD the 5 adapters
For each of an5d, convstencil, lorastencil, flashfftstencil, spider:
  bench/gpu_run.sh -t 30 -- 'bash artifacts/stencil/<name>/build.sh'
(Default MIG slice is an A100, sm_80 -- the same arch the timed runs use.
convstencil/lorastencil STATUS.md show older `-g h100` build commands; those
are history, rebuild here on A100.)
Record exit code and the last few lines of output per adapter. A build
failure is a result (record BUILD-FAILED with the first real error line in
that adapter's STATUS.md, per the Reproduction-section format used
elsewhere in this repo) -- do not attempt to patch the paper's own kernel
source to fix it.

STEP 3: FUNCTIONAL GATE, then a REAL (non-smoke) run, per adapter
  (cd bench && $PY -m kernelbench.runner --kernel stencil --list)
    (CPU-only listing; confirm all 5 show as built, not "not built: ... missing")
  a. Smoke gate each adapter first, to catch a broken build cheaply:
     bench/gpu_run.sh -- \
       '$PY -m kernelbench.runner --kernel stencil --variant <variant> \
            --impl <IMPL_NAME> --smoke'
     Expect "unsupported" for the smoke shapes an adapter cannot run.
  b. Then one real run per adapter (no --smoke, no --warmup/--reps), each in
     its own job, at the paper's own size:

     IMPL_NAME                variant  --matrices                        source
     an5d-stencil             fp64     star2d1r (repeat for each of its  AN5D Sec. 6.1
                                       6 shapes; defaults 16384^2 /
                                       512^3, T=1000)
     convstencil-tcu          fp64     star2d1r@10240x10240:T=10240      ConvStencil Table 4
     lorastencil-star2d3r     fp64     star2d3r@10240x10240:T=10240      LoRAStencil Table II
     flashfftstencil-box2d1r  fp64     box2d1r@16380x16380:T=1           Table 3 is 16384^2;
                                                                         16380 is the nearest
                                                                         width divisible by 6
     spider-box2d7r-sptc      fp16     box2d1r (default 10240^2; runs    SPIDER Fig. 10
                                       its radius-7 box, T=1)

     (fp64 = stencil-cpu-gpu-kernel-fp64, fp16 = stencil-tcu-matmul-kernel-fp16)

     bench/gpu_run.sh -g a100 -t 60 -- \
       '$PY -m kernelbench.runner --kernel stencil --variant <variant> \
            --impl <IMPL_NAME>,torch-conv-stencil --matrices <workload>'

     A full A100 can queue for hours: submit each job and wait, do not
     fall back to h100 or to the MIG slice for timed runs (a 1g.5gb slice
     has ~5 GB and 1/7 of the SMs, and cannot hold 16384^2 fp64 buffers plus
     the baseline anyway). The earlier A100 baseline was healthy (star2d1r
     @10240^2:T=10240: 37.28 GCUP/s); if torch-conv-stencil now lands far
     below that on A100, stop and report it.
     Adding torch-conv-stencil (the library baseline) fills the report's
     speedup column; drop it for spider (fp16, and it must run alone).
     flashfftstencil also rewrites the shared workload (it forces T=1),
     which is why its row passes :T=1 explicitly -- that way the baseline
     runs the same single step.
  Record the `[gpu_run] host=... gpu=... job=...` line of every run (it is
  the evidence of which GPU was used). Check the runner's own
  `conforming: true/false` line and the
  `N/M runs valid (k unsupported)` line for each. Report `nonconformance_reasons`
  verbatim; "GPU clocks not locked" alone is expected here (see GOAL note) --
  flag it only if something ELSE appears alongside it. Also record the gated
  step count (protocol_used.correctness_gate) for each run.

STEP 4: RECORD
Append to each artifact's STATUS.md a section
  ## Reproduction on zaratan, post-2026-09-23 changes (<YYYY-MM-DD>)
with: GPU used (must be A100 for timed runs); build outcome; smoke-gate outcome; real-run outcome
(GCUP/s, the paper's own metric from the report, correctness error vs
tolerance and the gated step count, conforming true/false); anything that
deviates from the last recorded ruling and your evidence for why. For
convstencil/lorastencil, also say whether the new GPU-resident bridge built
and gated, and how its smoke time compares with the old 6-11 ms.
Never delete or rewrite earlier STATUS.md text -- put the new ruling first
and keep the old one after "Earlier outcome, kept for the record:".

STEP 5: REPORT
- (cd bench && $PY -m kernelbench.report results/stencil_*.json \
      --html results/stencil_report.html)
- The report ranks per GPU; quote the A100 group. If older H100 JSONs are
  in results/, say they exist but do not mix them into the summary.
- Write a short summary (5 papers, one row each): workload run, GCUP/s, the
  paper's own metric, speedup vs torch-conv-stencil where shown, correctness
  error and gated step count, conforming true/false, and for anything that
  did not produce a real number, the exact blocking reason (build failure,
  runtime crash, unsupported shape/grid, gate failure, job time limit).
- Commit in small commits, one per artifact directory touched, never
  checkouts/build products/results JSON over 1 MB. Commit messages: what
  changed and why, one paragraph.
```
