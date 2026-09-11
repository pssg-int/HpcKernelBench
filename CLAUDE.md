# HPC-KernelBench — project context for Claude

Read this first. It encodes the project state, decisions already made with the
user, and traps already paid for. `README.md` documents the pipeline for humans;
this file adds the operational knowledge needed to CONTINUE the work.

## Mission (3 phases)

Build a kernel benchmark for **HPC** (not just ML/LLM kernels, which existing
KernelBench-style suites cover):

1. **Phase 1 — survey (DONE)**: find every main-track paper since 2020 whose
   primary contribution is an optimized computational-kernel implementation;
   classify into 12 kernel categories; big filterable table.
2. **Phase 2 — benchmark (IN PROGRESS)**: specs DONE (90 tracks, ~256 variants,
   `benchspecs/`), harness DONE (`bench/`, 25 kernels / 10 domains, CPU smoke
   25/25 green), GPU kernels compiled but NEVER RUN (login-node GPU is shared;
   user said no compute-node runs yet). User decisions: artifact must be
   open-source; **Perlmutter reproducibility NOT required**; selection by kernel
   coverage, NOT stars/activity.
3. **Phase 3 — agent study (NOT STARTED)**: have coding agents implement
   kernels from scratch; compare against the human SOTA measured in Phase 2.

## Current state (as of 2026-08-07)

- Corpus: **7029** main-track papers 2020+ from SC, IPDPS, ICS, HPDC, ASPLOS,
  PPoPP, CGO, USENIX ATC, DAC, TPDS (venue set confirmed by user; PPoPP+CGO
  added on top of their original 8).
- Included: **917 kernel papers**, classified into 12 categories
  (see `classify_instructions.md` for authoritative criteria), all verdicts
  either high-confidence or re-checked by a verify pass with uniform boundary
  rulings (`verify_instructions.md`).
- Artifacts: **309/917 open-source** (274 verified + 35 likely), with
  stars/license/last_push per paper. Two misattributions were retracted during
  the Phase-2 spec surveys (Shun20 -> jshun/ligra; Khanda TPDS'22 retracted).
- Kernel grouping: 309 papers -> 91 kernel groups, 47 strong tracks (>=3
  papers); `output/benchmark_candidates.html`, `output/benchmark_groups.json`.
- Benchmark specs: `benchspecs/<kernel>/{survey.md,spec.yaml}` x90+,
  aggregated in `output/specbook.html`. Grounded in paper fulltext AND artifact
  benchmark-script archaeology; `notes_on_fairness` lists where each track's
  own literature is unfair and how the spec diverges.
- Harness: `bench/` — spec-driven (spec.yaml is parsed and EXECUTED, see
  `bench/README.md` + `bench/DOMAIN_GUIDE.md`). `./smoke_all.sh` must stay
  green after any change.
- Deliverables in `output/`: `index.html` (self-contained interactive table,
  open in any browser), `kernel_table.csv`, `papers_flat.csv`,
  `included.json`, `papers.json` (all 4047 candidates incl. exclusion
  reasons), `summary.md`.
- `output/included.json` is the SOURCE OF TRUTH for downstream phases.

## Regeneration map (only needed if inputs change)

See README.md ("Pipeline" + "Artifact discovery") for the full script chain,
including which steps are run by classification/verify/judge agents.

All intermediate JSON is committed, so nothing needs re-running unless you
extend the corpus (new venues / refreshed DBLP dump).

## Machine-specific bits (adjust on a new machine)

- **Two machines are set up: Perlmutter (reference) and zaratan (UMD).** On
  **zaratan just `source bench/env.sh`** — its `KB_*` defaults point at the
  group-shared conda-forge toolchain + dataset caches under
  `/scratch/zt1/project/bhatele-lab/shared/kernel-bench/` (envs/kb-env,
  envs/kb-gcc12, data/); `smoke_all.sh` is 35/35 and all 99 adapters were
  built+gated there (2026-09-09). Setup/recreate steps: `bench/ENVIRONMENT.md`
  §9; full reproduction outcome + per-adapter deviations:
  `bench/REPRODUCTION_zaratan.md`; GPU work goes through `bench/gpu_run.sh`
  (login nodes have no GPU). On a THIRD machine, edit the `KB_*` lines in
  `env.sh` (they currently hold zaratan values) per `bench/ENVIRONMENT.md`.
- **DBLP dump**: `extract_corpus.py` expects
  `../HPCs/hpc-first-authors/data/dblp.xml` (5.2 GB, not in this repo).
  On a new machine either clone the user's HPCs repo + re-download the dump
  (`curl -O https://dblp.org/xml/dblp.xml.gz`; `dblp.dtd` must sit next to it),
  or edit the path constants at the top of the script. Only needed to extend
  the corpus.
- **Benchmark harness + artifact builds**: everything machine-specific is in
  `bench/ENVIRONMENT.md` (requirements table, setup steps, per-artifact
  extras, known traps) and parameterised through `bench/env.sh` (`KB_*`);
  `bench/check_env.py` audits a machine; `bench/requirements.txt` pins the
  Python layer (torch installed separately for your CUDA major).
  `bench/REPRODUCE.md` is the ready-to-paste prompt for a coding agent that
  has to rebuild and re-gate every artifact on a new machine.
- **Paper pipeline (Phase 1 scripts)**: any Python 3 with `lxml`, `requests`,
  `pypdf` (also in `bench/requirements.txt`). On Perlmutter the user's venv
  was `/pscratch/sd/c/cunyang/gnn/plexus_env`.
- **GitHub API**: artifact scripts get a token via `gh auth token` — needs an
  authenticated `gh` CLI.
- **API etiquette**: OpenAlex/Crossref calls send `mailto:weicunyang@gmail.com`.
  PapersWithCode API is dead (site sunset) — the stage is a no-op, skip it.

## Working agreements with the user (do not re-ask)

- **Model strategy: bulk subagent work (classification, judging) runs on
  Sonnet; the main session on Fable/Opus.** Opus for bulk was tried and
  reverted — too expensive. Batches of ~10 concurrent agents; ~40 Opus-sized
  chunks once exhausted a session limit, so pace accordingly and expect
  resets (classification chunks are resumable; completed files are skipped).
- Docs/replies to the user in Chinese; code and code comments in English.
- ML kernels: only HPC-crossing ones are in scope (category `ml_kernels`);
  pure LLM-serving systems are excluded.
- Codegen/autotuning papers count as kernel work, tagged via `approach`.

## Traps already paid for (do not relax)

- Fulltext-extracted URLs include **cited baselines** (transformers, zstd
  showed up as false "artifacts"). Never auto-trust a URL just because it
  appears in the paper PDF; abstract URLs are trustworthy.
- GitHub `in:readme` hits include **awesome-lists** that quote titles and DOIs.
  A repo with no detected code language is a paper list.
- **Substring matching of author surnames or venue names is worthless**
  ("Zhang", "sc" match everything). Require title-fragment or explicit
  self-identification, else send to an LLM judge.
- Same-name-different-project collisions are common (LEGO→go-acme/lego,
  TANGO→NetEase/tango). Third-party reimplementations and reproducibility
  -report forks are not artifacts.
- DAC papers are 6 pages (page filter!); ICS shares its booktitle with ITCS;
  main-track filtering needs booktitle ∧ crossref + TOC whitelist (method
  inherited from the user's HPCs repo).
- lxml on the 5 GB dump: iterparse + clear + prune previous siblings, parse by
  file path so the DTD resolves.

## Phase 2 plan (agreed direction, details open)

1. Selection from the 309 with-artifact papers. User-confirmed criteria:
   per-category kernel coverage ONLY (stars/activity explicitly excluded).
   Aim for a benchmark set of ~20-50 kernels
   across categories (dense_la 98, sparse_la 87, ml_kernels 81, graph 61,
   primitives 35, compression 26, stencil_pde 23, tensor 22, solvers 18,
   fft 11, nbody 3-5 available).
2. Harness: clone artifact, build, run on standard inputs (for sparse:
   SuiteSparse matrices — the user has an existing set from their SpMM/RoDe
   work), record wall/throughput vs. the paper's claimed baseline.
3. The user's related assets: SpMM-AKO4X project (matrix set + RoDe baseline
   recipes), Perlmutter profiling recipes (dcgmi pause/resume around ncu/nsys).
4. Measurement machine: NOT restricted to Perlmutter (user decision).

## Phase 3 plan (sketch)

For each benchmark kernel: give a coding agent the kernel spec (problem, input
format, target hardware) WITHOUT the paper's solution; let it implement and
optimize from scratch; measure where the agent lands relative to (a) vendor
library baseline, (b) the paper's artifact. Open questions: agent harness,
iteration budget, hardware access pattern — design when Phase 3 starts.

## Continuing on a new machine — checklist

**On zaratan (already set up):** clone the repo, `source bench/env.sh`,
`$PY bench/check_env.py` (expect 0 MISSING), `cd bench && ./smoke_all.sh`
(expect 35/35). For GPU/artifact work read `bench/ENVIRONMENT.md` §9 +
`bench/REPRODUCTION_zaratan.md`, and run every GPU command through
`bench/gpu_run.sh`. If the conda envs are gone, §9 has the recreate commands.
On any OTHER machine, the generic flow:

1. Clone this repo; everything except the DBLP dump travels with it.
2. `pip install lxml requests pypdf`; authenticate `gh` if artifact work is
   needed.
3. Open `output/index.html` to see the current table; `output/included.json`
   for programmatic access.
4. Phase 2 needs: GPU/CPU build environment per selected artifact. Start by
   confirming the selection criteria with the user, then filter
   `included.json` by `artifact_status != "none"`.
