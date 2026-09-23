# Running the stencil track on zaratan: prompt for a coding agent

> Scope: the `stencil` kernel track ONLY (5 paper adapters), not the full
> 98-artifact suite. For a from-scratch reproduction of everything, use
> `bench/REPRODUCE.md` instead. zaratan is already set up (see
> `bench/ENVIRONMENT.md` §9) — this prompt does not redo environment setup,
> it builds and runs one track.

Paste the block below into a coding agent (Claude Code or similar) started in
a fresh clone of this repository on zaratan, after `git pull` has brought in
the latest `benchspecs/stencil/spec.yaml`, `bench/kernelbench/domains/stencil.py`,
and `bench/audit/scripts/check10_stencil_gate_not_vacuous.py`.

---

```text
You are working in a clone of HPC-KernelBench, on zaratan. Run every
command below from the REPOSITORY ROOT unless it says otherwise.

GOAL. Build and run the 5 paper adapters in the stencil track
(bench/artifacts/stencil/{an5d,convstencil,lorastencil,flashfftstencil,spider}),
get a real (non-smoke) conforming performance number for each on the one
shape it actually supports, and report what you get, including any failure,
honestly. Do not "fix" the benchmark (spec, harness, or a paper's own kernel
source) to make a result pass.

READ FIRST: CLAUDE.md; bench/CLAUDE.md; bench/ARTIFACT_GUIDE.md (rules 1-10
are binding); benchspecs/stencil/spec.yaml; benchspecs/stencil/survey.md;
bench/artifacts/stencil/README.md and each adapter's own STATUS.md.

IMPORTANT CONTEXT (do not rediscover these the slow way):
- Coverage is narrow and per-adapter, not uniform: an5d supports 6 shapes
  (star2d1r, box2d1r, star2d3r, box2d3r, star3d1r, box3d1r); convstencil
  supports star2d1r only (2D only); lorastencil supports star2d3r only (its
  other 3 kernels divide by zero or ignore caller-supplied weights); spider
  ignores whatever shape name you pass and substitutes its own fixed
  radius-7 2D box, forcing timesteps=1 (fp16 overflow risk beyond one
  sweep); flashfftstencil supports box2d1r only AND requires the grid width
  to be a multiple of 6 -- the domain's own default 16384 grid does NOT
  satisfy this, so it needs an explicit smaller grid_shape (see its
  adapter.py docstring "SHAPE CONSTRAINT" and its STATUS.md for the grid
  actually used in its last successful gate).
- IMPL_NAMEs: an5d-stencil, convstencil-tcu, lorastencil-star2d3r,
  flashfftstencil-box2d1r, spider-box2d7r-sptc.
- kernelbench/domains/stencil.py's synthetic weights changed recently
  (2026-09-21): they now sum to 1 with all-positive coefficients, replacing
  an earlier set that summed to 0.5 and made the correctness gate vacuous at
  T>=100 (see the module's _build_weights docstring and
  bench/audit/scripts/check10_stencil_gate_not_vacuous.py). This means every
  gate number recorded in each adapter's STATUS.md before that date was
  computed against the OLD weights and is stale for an5d/convstencil/
  lorastencil/flashfftstencil (which build their reference from the
  workload's own weights). an5d and spider use the PAPER's own hardcoded
  coefficients, not the domain's synthetic ones, so their gate numbers are
  unaffected by this change -- confirm this yourself by reading each
  adapter's prepare() rather than assuming it.
- Last recorded H100 status (bench/H100_baselines.md, h100_status.json,
  before the weights fix): an5d PASS (4.76 GCUP/s), spider PASS, convstencil
  REGRESSION (CUDA error 700: illegal memory access), flashfftstencil and
  lorastencil UNSUPPORTED by the standard --smoke invocation (each needs a
  workload its own STATUS.md documents, not the domain's default). Expect
  to hit the same convstencil/flashfftstencil/lorastencil issues again;
  investigate and report them, do not paper over them.
- The runner does not thread which spec variant you pass into
  load_workload(), so every workload you get is sized per variant 1's
  defaults (16384^2 2D, 512^3 3D) regardless of --variant. A true variant-2
  run (10240^2, T=10240) is not reachable through the CLI as of this
  writing -- note this limitation in your report rather than silently
  reporting a variant-1-sized run as if it were variant 2.

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
- Before any gpu_run.sh call, set the SLURM account (env.sh does not):
    export KB_SLURM_EXTRA="-A <your zaratan account>"
  If you do not know the account, run `sacctmgr -nP show assoc user=$USER
  format=account` and use the one with GPU access; if none is obvious,
  STOP and ask.
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
  bench/gpu_run.sh -g h100 -t 30 -- 'bash artifacts/stencil/<name>/build.sh'
Record exit code and the last few lines of output per adapter. A build
failure is a result (record BUILD-FAILED with the first real error line in
that adapter's STATUS.md, per the Reproduction-section format used
elsewhere in this repo) -- do not attempt to patch the paper's own kernel
source to fix it.

STEP 3: FUNCTIONAL GATE, then a REAL (non-smoke) run, per adapter
  (cd bench && $PY -m kernelbench.runner --kernel stencil --list)
    (CPU-only listing; confirm all 5 show as built, not "not built: ... missing")
For each adapter, on the ONE shape it supports (see IMPORTANT CONTEXT above):
  a. Smoke gate first, to catch a broken build cheaply (on a GPU node --
     the smoke run executes the kernel):
     bench/gpu_run.sh -g h100 -- \
       '$PY -m kernelbench.runner --kernel stencil --variant <its variant> \
            --impl <IMPL_NAME> --smoke'
  b. Then a real, conforming run (no --smoke, no --warmup/--reps override),
     under an EXCLUSIVE node allocation (gpu_run.sh does not add this by
     itself), at the shape it supports:
     KB_SLURM_EXTRA="$KB_SLURM_EXTRA --exclusive" bench/gpu_run.sh -g h100 -t 30 -- \
       '$PY -m kernelbench.runner --kernel stencil --variant <variant> \
            --impl <IMPL_NAME> --matrices <shape>'
     an5d and convstencil: --variant stencil-cpu-gpu-kernel-fp64
     lorastencil, flashfftstencil, spider: --variant stencil-tcu-matmul-kernel-fp16
       (flashfftstencil and lorastencil actually compute in fp64 despite this
       variant's fp16 label -- their own STATUS.md explains why they are
       gated against this variant anyway; do not "fix" this mismatch, report
       it if you think it is wrong)
     flashfftstencil specifically: its adapter's SHAPE CONSTRAINT means the
     domain's default 16384x16384 grid will raise NotImplementedError; you
     will need to call load_workload with an explicit, 6-divisible
     grid_shape (see its STATUS.md for the grid it was last successfully
     gated at) -- this is not reachable via --matrices alone; write a short
     one-off Python invocation if needed (run it through the same
     exclusive bench/gpu_run.sh command as above, e.g. as a script under
     bench/ invoked as '$PY <script>.py'), and say so in your report.
  Record the `[gpu_run] host=... gpu=... job=...` line of every run (it is
  the evidence of which GPU was used). Check the runner's own
  `conforming: true/false` line and the
  `N/M runs valid (k unsupported)` line for each. If `conforming: false`,
  read `nonconformance_reasons` and report it verbatim.

STEP 4: RECORD
Append to each artifact's STATUS.md a section
  ## Reproduction on zaratan, post-weights-fix (<YYYY-MM-DD>)
with: GPU used; build outcome; smoke-gate outcome; real-run outcome
(GCUP/s, correctness error vs tolerance, conforming true/false); anything
that deviates from the last recorded ruling and your evidence for why.
Never delete or rewrite earlier STATUS.md text -- put the new ruling first
and keep the old one after "Earlier outcome, kept for the record:".

STEP 5: REPORT
- (cd bench && $PY -m kernelbench.report results/stencil_*.json \
      --html results/stencil_report.html)
- Write a short summary (5 papers, one row each): shape tested, GCUP/s,
  correctness error, conforming true/false, and for anything that did not
  produce a real number, the exact blocking reason (build failure, runtime
  crash, shape/grid constraint not satisfiable via the CLI, etc.).
- Commit in small commits, one per artifact directory touched, never
  checkouts/build products/results JSON over 1 MB. Commit messages: what
  changed and why, one paragraph.
```
