# Survey — json-parsing (singleton track: 1 paper)

## conf/asplos/GargaryL026 — cuJSON: A Highly Parallel JSON Parser for GPUs (ASPLOS 2026)

- **What is measured**: end-to-end GPU JSON parsing, structured as three phases all offloaded to
  the GPU: (i) UTF-8 validation, (ii) tokenization, (iii) nesting-structure recognition (producing
  a "structural index" of byte positions + matching-bracket `pair_pos` array), followed by a
  post-parse query-iterator phase for extracting specific keys/values from the parsed structural
  index. The paper's stated contribution is a branch-minimal, maximally parallel redesign of the
  parsing algorithm (challenging the "GPUs are ill-suited for branch-heavy parsing" conventional
  wisdom).
- **workloads/inputs**: two input MODES with materially different parallelization opportunities,
  both required as separate sub-cases:
  - **Standard JSON**: one large, deeply-nested JSON document (`*_large_record.json` naming
    convention in the `dataset/` folder) — parallelism must come from within-document structure,
    since there is only one top-level record.
  - **JSON Lines**: many small independent JSON records, one per line (`*_small_records.json`) —
    the repo exposes THREE different chunking strategies as separate entry points/binaries
    (`main_jsonlines.cu`: split into a fixed chunk COUNT; `main_jsonlines_chunksize.cu` /
    `main_jsonlines_chunksize_MB.cu`: split by a maximum chunk SIZE in bytes/MB) — chunk-boundary
    strategy is itself a benchmarkable axis since inter-chunk dependency-breaking is core to the
    paper's parallelism claim (README: "cuJSON is able to break key dependencies in the parsing
    process, making it possible to accelerate the parsing of a single large JSON file effectively").
  Sample datasets are bundled small; large 1GB-class datasets used in the paper's actual
  performance evaluation are hosted externally (Google Drive), not in-repo — named source appears
  to be Twitter-sample-derived JSON (`twitter_sample_*.json` naming in the examples).
- **timing protocol**: not visible at the top-level README/build-command level (only `nvcc -O3
  -std=c++17 -arch=sm_80 main.cu -o cujson_standard.out` compile commands and a bare `./out
  file.json` run example are shown); a `paper_reproduced/` directory exists specifically "for
  research purposes and comparison" with its own `Makefile.compile`/`Makefile.run` and a
  `scripts/readme.md` claimed to hold "all the scripts" for reproducing the paper's figures — this
  is the authoritative source for the actual timing protocol but its contents were not fetched in
  this survey (see open_questions).
- **timing scope**: not stated whether the reported throughput includes host-to-device transfer of
  the raw JSON bytes and device-to-host transfer of the structural-index result, or is
  kernel(s)-only; the 3-phase pipeline (UTF-8 validate -> tokenize -> nesting-recognition) could be
  timed as one fused region or per-phase — not specified in the fetched top-level docs.
- **precision & correctness**: JSON parsing is an exact/discrete problem (no floating-point
  tolerance applies) — correctness means the recovered structural index (and any extracted
  key/value strings via the query iterator) exactly matches a reference parse. UTF-8 validation
  strictness (RFC 8259 compliance — how cuJSON handles malformed/invalid JSON, e.g. invalid UTF-8
  sequences, trailing commas, unescaped control characters) is not documented in the fetched
  README.
- **metric**: not stated numerically in the fetched abstract/README beyond qualitative "cuJSON
  absolutely flies... leaving other top-tier CPU and even existing GPU parsers in the dust"; the
  field's standard metric for this workload is throughput in **GB/s** (matching this track's brief).
- **baselines**: named in the abstract/README — CPU: simdjson, Pison; GPU: cuDF, GPJSON.
- **source**: abstract (`output/included.json`); repo README
  (`gh api repos/AutomataLab/cuJSON/contents/README.md`); `paper_reproduced/` directory listing
  (`gh api repos/AutomataLab/cuJSON/contents/paper_reproduced`).

## Divergences

- Single-paper track: no cross-paper divergence. The main internal tension this spec must resolve
  is Standard-JSON (one big nested document, parallelism from within-document structure) vs.
  JSON-Lines (many independent small records, parallelism from across-record independence, plus a
  chunking-strategy sub-choice) — these are different enough workload characters that this spec
  keeps them as two required sub-cases within one variant rather than pretending a single GB/s
  number covers both.
