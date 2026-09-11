# Survey — membership-filter (singleton track: 1 paper)

## conf/ppopp/McCoyHY023 — High-Performance Filters for GPUs (PPoPP 2023)

- **What is measured**: a family of approximate-membership-query (AMQ) filter data structures
  ported/redesigned for GPU: **GQF** (General Quotient Filter), **TCF** (Truly Concurrent Filter,
  including a cooperative-group "pog_warp" variant), **RSQF** (Rank-and-Select Quotient Filter),
  **SQF** (Shrinking Quotient Filter), **Bloom Filter**, **Blocked Bloom Filter** — plus a bulk-
  insert/warpcore-comparison path (`bulk-tcf`) and per-filter deletion support (GQF/SQF/TCF).
- **workloads/inputs**: filter capacity swept as powers of two, **2^22 to 2^30 items** (the
  README's own stated point-test range) for insert/query/random-query on GQF/TCF/RSQF/SQF/Bloom/
  Blocked-Bloom; the SEPARATE bulk-TCF test path (`bulk-tcf/build/batched_template_tests`) is run
  at capacities **2^22, 2^24, 2^26, 2^28** directly and **2^30 via a pre-sorted variant**
  (`presorted_template_tests 30`) specifically because sorting ~8.5GB of keys before insertion
  (doubled by Thrust's sort buffer) exceeds memory at that scale on some GPUs — i.e. the artifact
  itself documents a memory-scaling workaround as part of its own protocol, not an incidental
  detail. Query workload has THREE distinct sub-types tested per filter: (1) **insert** (build the
  filter from N random keys), (2) **positive query** (query keys known to be present), (3)
  **random/false-positive query** (query random keys NOT necessarily present, measuring the
  false-positive RATE as well as throughput) — and (4) **delete** for the filters supporting it
  (SQF, GQF, TCF). The README explicitly notes it does NOT repeat GQF's counting-filter tests
  because it lacks a Zipfian-distribution generator locally (author offers the dataset by email
  request) — i.e. **skewed/Zipfian key-frequency distributions are part of the paper's own full
  evaluation but not exercised by the released harness's default run**.
- **timing protocol**: `run_full_battery.sh` (chmod +x, run) orchestrates GQF/SQF/RSQF's own
  `scripts/run_all_tests.sh`, an `aggregate_local_data.py` post-processing step, a separate
  `run_deletes.sh`, then a compiled `pog_warp_tests/build/test` binary for TCF/Warpcore, then the
  `bulk-tcf` batched tests at the 5 capacities above — no single stated warmup/repeat/statistic
  convention is visible at this orchestration level (each sub-tool likely has its own internal
  convention, not inspected further in this survey). The README states point-GQF and SQF-delete
  tests "can take a while... 30 minutes" on Cori/Perlmutter, implying substantial per-configuration
  work (consistent with large-N sweeps, not a single quick timed call).
- **timing scope**: not stated whether reported throughput is insert/query kernel time only or
  includes filter-array allocation/initialization; deletion tests are timed and reported
  separately from insert/query (a natural, paper-native separation this spec preserves).
- **precision & correctness**: AMQ filters are INHERENTLY approximate for negative queries (a
  false-positive rate epsilon is a designed property, not a bug) but must be EXACT for positive
  queries (a key that was inserted must always be found — zero false negatives is a hard
  correctness requirement for every filter type here). No automated correctness/FPR-bound check is
  visible as a gate in the fetched top-level scripts (`run_tests.py`, `run_full_battery.sh`) — FPR
  appears to be an OUTPUT of the "random (false positive) queries" test, not a pre-timing gate.
- **metric**: throughput (ops/s, implied — not explicitly itemized in the fetched README) for
  insert/query/delete; false-positive rate for the random-query test, reported per-filter and
  presumably per-capacity (matching this track's brief of FPR jointly with throughput).
- **baselines**: the filters are compared against EACH OTHER within this same artifact (GQF vs TCF
  vs RSQF vs SQF vs Bloom vs Blocked-Bloom), plus an external **Warpcore** comparison specifically
  for the point-TCF path (`pog_warp_tests`, cooperative-group variants) — Warpcore is a known
  high-performance GPU hash-table library, a natural external AMQ/hash-table baseline.
- **source**: abstract (`output/included.json`); repo README
  (`gh api repos/huntermBerkeley/PPOPP_Artifacts/contents/README.md`); top-level file listing and
  `run_tests.py` (`gh api .../contents` and `.../contents/run_tests.py`).

## Divergences

- Single-paper track: no cross-paper divergence. The main internal tension is BREADTH — six
  filter data structures with different feature sets (only GQF/SQF/TCF support delete; only TCF has
  a documented Warpcore external-baseline comparison path) — this spec keeps insert/query as the
  REQUIRED common-denominator variant across all six filter types, and treats delete as an optional
  filter-type-specific extension rather than forcing Bloom/RSQF (which don't support delete) into
  an incomplete comparison.
- The artifact's own README explicitly flags that it does NOT reproduce the GQF counting/Zipfian
  test locally (dataset available on request only) — this spec's key-distribution axis therefore
  cannot be fully grounded in the released harness and is instead specified from field convention
  (uniform-random keys as the common denominator, with a Zipfian sub-case flagged as
  paper-claimed-but-not-independently-reproducible from the released artifact alone).
