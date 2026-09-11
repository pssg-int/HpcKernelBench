# kernel-papers — HPC kernel-optimization survey (2020+)

Phase 1 of the HPC-KernelBench project: find every main-track paper since 2020
whose primary contribution is an optimized computational-kernel implementation
(GEMM, SpMM, stencil, FFT, sort, ...) or a kernel-producing code generator, and
tabulate them by kernel category.

**Result: 917 papers** out of a 7029-paper corpus from SC, IPDPS, ICS, HPDC,
ASPLOS, PPoPP, CGO, USENIX ATC, DAC, TPDS.

## Deliverables (`output/`)

| File | What |
|---|---|
| `index.html` | Self-contained interactive table: filter by category/venue/year/platform/approach, full-text search, sortable, with per-category / per-year / per-venue charts. Open directly in a browser. |
| `kernel_table.csv` | One row per (category, paper) |
| `papers_flat.csv` | One row per included paper |
| `included.json` | Included papers, full metadata + verdicts |
| `papers.json` | All 4047 classified candidates incl. exclusion reasons |
| `summary.md` | Category / venue / year counts, most-cited per category |

## Pipeline

```
extract_corpus.py     DBLP dump -> data/corpus.json          (7029 papers, 2020+ main-track)
fetch_abstracts.py    OpenAlex/S2/Crossref -> abstracts.json  (99.5% coverage)
fetch_atc.py          S2 title-match + USENIX scrape for ATC (no DOIs)
prefilter.py          high-recall keyword taxonomy -> candidates.json (4047)
make_chunks.py        50-paper chunks for LLM classification
  (parallel Claude agents classify per classify_instructions.md -> data/classified/)
merge_classified.py   -> output/papers.json / included.json
  (verify agents re-check low-confidence + borderline per verify_instructions.md)
apply_verify.py       apply corrections (472 re-checked: 29 dropped, 10 rescued)
build_table.py        -> CSVs + summary.md
make_html.py          -> index.html
```

Venue filtering mirrors `../HPCs/hpc-first-authors/build.py` (booktitle AND
crossref-series must match; TOC whitelist; >=6 pages with SC exceptions).
The DBLP dump lives in `../HPCs/hpc-first-authors/data/dblp.xml`.

## Classification

12 categories (see `classify_instructions.md` for authoritative criteria):
dense_la, sparse_la, stencil_pde, fft_spectral, tensor, graph, primitives,
nbody_md, unstructured_amr, solver_components, compression, ml_kernels
(HPC-crossing only — pure LLM-serving systems are excluded).

Judged from title+abstract by parallel Claude agents (Opus 4.8 for chunks
000-013/016-020 and 033-042, Sonnet 4.5 for the rest), with a uniform-boundary
verify pass (Sonnet) over all low/medium-confidence verdicts. Boundary rulings
(NTT/crypto kernels included under fft_spectral; kernel *selectors* excluded as
runtime-sched; FPGA HLS in, RTL out; quantum-circuit *simulation* in,
compilation out) are codified in `verify_instructions.md`.

Excluded-paper counts by reason: other 906, hardware-accel 618, runtime-sched
~400, pure-ml-serving ~370, perf-model ~215, io-storage ~215, application ~205,
communication ~180.

## Artifact discovery

**309/917 papers have a confirmed open-source artifact** (274 verified + 35
likely, after two misattribution corrections found during the spec surveys), recorded per paper as `artifact_status/url/stars/license/last_push`
and filterable in `index.html`.

```
find_artifacts.py     multi-signal candidate search:
                        abstract regex | arXiv fulltext | OA PDF scan |
                        Crossref relations (empty in practice) |
                        GitHub name search | GitHub in:readme title search
verify_artifacts.py   GitHub API checks (repo alive, has code, README
                        cross-reference); circular evidence distrusted
make_judge_chunks.py  ambiguous matches -> LLM judge input
  (Sonnet judges decide verified/likely/no; paper-lists, cited baselines,
   same-name projects and third-party reimplementations are rejected)
apply_artifacts.py    judge > script precedence, writes into included.json
```

Traps encoded in the pipeline (do not relax them): fulltext-extracted URLs
include cited baselines (transformers/zstd showed up as false "artifacts");
`in:readme` hits include awesome-lists that quote titles/DOIs; author-surname
and venue substring matching is worthless (Zhang/sc match everything); repos
with no detected code language are paper lists.

## Benchmark tracks (Phase 2 grouping)

The 309 artifact papers are normalized to canonical kernel slugs
(`norm_instructions.md`, LLM-assisted) and grouped by concrete kernel:
**91 groups, 47 "strong tracks" with >=3 competing implementations**
(spmm 36, gemm 35, stencil 21, spmv 20, lossy-compression 20,
quantized-gemm 16, sddmm 13, spgemm 13, ...). Selection is by kernel
coverage only — stars/maintenance activity deliberately not weighted.

- `output/benchmark_candidates.html` — tracks grouped by domain, papers +
  artifact links per track
- `output/benchmark_groups.{json,csv}` — machine-readable grouping
- `benchmark_groups.py` / `make_benchmark_html.py` — regenerate

## Benchmark specifications (`benchspecs/`)

**90 kernel tracks, 257 benchmark variants.** For each track an agent read the
papers' own evaluation methodology — arXiv/OA fulltext where it exists,
otherwise the artifact's benchmark scripts and timing loops via `gh api` — and
wrote:

- `benchspecs/<kernel>/survey.md` — per-paper: input suite, warmup/repetition/
  statistic, what the timing window includes, correctness practice, metric,
  baselines; plus a *Divergences* section where the papers disagree
- `benchspecs/<kernel>/spec.yaml` — 1–5 variants making different legitimate
  claims comparable (kernel-only / preprocessing-amortized / end-to-end /
  precision- or scale-specific), each pinning inputs, protocol, timing scope,
  correctness gate and metric, plus `notes_on_fairness`, `open_questions` and an
  `evidence` map back to individual papers

Aggregated by `make_specbook.py` into `output/specbook.{html,json,md}`.

Reading the artifacts rather than only the papers is what made this worth doing —
recurring findings the spec book corrects:

| Finding | Example |
|---|---|
| Timing loop contradicts the paper | CB-SpMV code `ITER=10` vs paper's "1000 times" |
| Single measured iteration, no statistic | Cheddar, TileSpGEMM (`REPEAT_NUM=1`), ZipServ (`WARM_UP=0`) |
| best-of-N reported as the headline | modal practice in GEMM, MANS, sequence-alignment |
| Correctness check exists but is never called | RoDe, MEATTEN, SMaT (`enable_check=false`) |
| No correctness check at all | DBMPK; 5 of 8 graph-pattern-mining artifacts |
| Metric not comparable across algorithms | stencil GFLOP/s → spec uses GCell-updates/s; Winograd flop accounting |
| Preprocessing silently excluded | AlphaSparse's up-to-8h per-matrix search, unamortized |
| Convergence gate defeated | AmgT sets HYPRE tolerance to 1e-20 so early exit can't fire |
| Benchmark input isn't the real object | 4/5 SpTRSV papers slice a triangle out of an arbitrary matrix instead of using an LU/Cholesky factor |
| Artifact ≠ paper | GeraK23's repo has no PageRank; STM-QR's README says the release doesn't combine its two contributions |

Two artifact misattributions found during the surveys were corrected in
`output/included.json` (Shun20 → `jshun/ligra`; Khanda TPDS'22 retracted as a
third-party class project), leaving **309** papers with artifacts.

## The harness (`bench/`)

Spec-driven runner: `benchspecs/<kernel>/spec.yaml` is parsed at run time and
its protocol is what executes. 25 kernels across 10 domains have CPU
implementations (`bench/smoke_all.sh` — 25/25 green); CUDA baselines compile
(sm_80) but have never been executed (login-node GPU is shared). See
`bench/README.md` and `bench/DOMAIN_GUIDE.md`.

## Next

- Phase 2 (remaining): measure on a compute-node allocation; integrate paper
  artifacts as competing implementations per track.
- Phase 3: agent-from-scratch kernel implementations vs. human SOTA.
