# Hash-table track — evaluation-methodology survey

Surveyed 3/3 papers from `data/track_inputs/hash-table.json`. Full evaluation
sections were reachable for 2/3 papers (DEDUKT via its eScholarship PDF
mirror; Clevel-Hashing via the USENIX-hosted PDF, fetched with a
browser-User-Agent `curl` since USENIX blocks the WebFetch tool's default
UA). ShovonGMK23 has no arXiv/OA fulltext and no DOI in the input record; its
protocol facts below come entirely from the artifact repo's driver code
(`hashjoin.cu`, `tc_cuda.cu`, `kernels.cu`) and its README's reproducibility
table, which is unusually precise (per-dataset row counts, iteration counts,
and a default `load_factor = 0.4` read directly from `main()`).

A structural note up front: the three papers do NOT benchmark the same kind
of hash table. ShovonGMK23's hash table is a transient, rebuilt-every-call
**build-once/probe-once join table** with no delete and no single-op path.
DEDUKT's is a **counting (upsert) table**: every operation is "find-and-increment
or insert" — there is no plain lookup-only or delete-only operation. Only
Clevel-Hashing benchmarks a full, general-purpose, dynamically-mutable table
with independently invokable insert/read/update/delete. This three-way split
is itself the central finding for the spec (see Divergences).

---

## 1. ShovonGMK23 — Towards Iterative Relational Algebra on the GPU (USENIX ATC 2023)

Source: artifact repo `harp-lab/usenixATC23` — `README.md`,
`code/hashjoin.cu`, `code/tc_cuda.cu` (contains `main()` with the default
run parameters), `code/common/kernels.cu` (hash kernels). No arXiv/OA
fulltext; DOI/arXiv fields empty in the input record.

- **hash table used for**: a GPU hash **join** (the paper's contribution is
  an iterative-Datalog transitive-closure engine; the hash table itself is
  a standard equi-join build/probe table, rebuilt fresh on **every
  fixed-point iteration** of the TC loop — iteration counts per dataset
  range from 10 (wiki-Vote) to 606 (usroads), each iteration re-hashing the
  current `t_delta` relation).
- **hash table design**: open addressing, **linear probing**
  (`position = (position + 1) & (hash_table_row_size - 1)`), slot claimed via
  `atomicCAS(&hash_table[position].key, -1, key)` (empty sentinel = -1),
  32-bit Murmur3-style hash (`get_position`). Table size is forced to the
  next power of 2 so the probe step can use a bitmask instead of modulo.
  Keys and values are both 32-bit ints (`struct Entity { int key; int
  value; }`). No resizing — the table is sized once from `load_factor` and
  discarded after the call.
- **operation mix**: **bulk build (100% insert) then bulk probe (100%
  lookup)**, two separate kernels (`build_hash_table` then
  `get_join_result_size_ar` / `get_join_result_ar`). No delete kernel exists
  for this table; no single-key API — only whole-relation bulk kernels
  (`relation_rows` keys per call, `grid_size x block_size` = 3456x512 in the
  reproducibility runs).
- **load factor**: an explicit tunable **input parameter** to both
  `gpu_hashjoin()` and `gpu_tc()`:
  `hash_table_rows = pow2(ceil(relation_rows / load_factor))`. The shipped
  `main()` in `tc_cuda.cu` calls `run_benchmark(0, 0, 0.4)` — **default
  load_factor = 0.4** (40% target occupancy) for every reported result; the
  value is never swept in the artifact (no load-factor-vs-throughput curve
  reported anywhere in this track's paper, unlike Clevel-Hashing below).
- **key distribution**: real-world relations for the headline numbers — 18
  named datasets with exact row counts read from the README's reproducibility
  table and `tc_cuda.cu`'s `datasets[]` array: OL.cedge_initial (7,035
  rows), CA-HepTh (51,971), SF.cedge (223,001), ego-Facebook (88,234),
  wiki-Vote (103,689), p2p-Gnutella09 (26,013), p2p-Gnutella04 (39,994),
  cal.cedge (21,693), TG.cedge (23,874), OL.cedge (7,035, distinct file from
  `_initial`), luxembourg_osm (119,666), fe_sphere (49,152), fe_body
  (163,734), cti (48,232), fe_ocean (409,593), wing (121,544), loc-Brightkite
  (214,078), delaunay_n16 (196,575), usroads (165,435) — SNAP social/web
  graphs plus DIMACS/FEM mesh and road-network relations, i.e. genuinely
  skewed real-world key distributions (not characterized quantitatively by
  the paper, unlike DEDUKT). The standalone `hashjoin.cu` microbenchmark
  additionally supports synthetic keys via a `random_category` flag
  (`1` = uniform-random ints, `2` = random strings), but these synthetic
  modes are never used for the paper's reported numbers.
- **batch vs single-op**: batch-only; smallest reported dataset is 7,035
  keys, largest 409,593 (paper); the demo table goes up to `usroads`
  (165,435 rows, 871M resulting TC join pairs).
- **timing protocol**: **no warmup, no repeated trials** visible anywhere in
  the driver — each dataset is run exactly once per invocation.
  Timer: CUDA-event-based `KernelTimer` wraps each GPU kernel individually
  (used to compute "Hashtable rate"); `std::chrono::high_resolution_clock`
  wraps host-side phases (read, memory alloc/free). The README's own
  reproducibility table reports a **single reproduced value** side-by-side
  with the paper's single published value (no min/max/mean shown) —
  e.g. CA-HepTh: 2.65s reproduced vs 4.32s in-paper, a >60% discrepancy that
  the artifact does not explain, itself evidence that single-run timing on
  a shared/variable GPU is noisy for this workload.
- **timing scope / throughput metric**: the build kernel's rate is printed
  **in isolation**: `"Hashtable rate: <keys_per_sec>, time: <s>"` (sample
  values: 480,927,938 keys/s for CA-HepTh, 528,470,545 keys/s for
  OL.cedge_initial) — this is a **build-only** Mops/s number, separate from
  the reported join/dedup/union/total times. The paper's headline
  wall-clock numbers, by contrast, are **end-to-end per-TC-iteration totals**
  including read, hash-table build, join, GPU-side sort-based dedup, and
  set-union — i.e. two different, non-interchangeable throughput scopes
  coexist in the same artifact.
- **precision & correctness**: exact 32-bit integer keys; correctness is
  validated indirectly via **matching TC output cardinality** ("TC size"
  column must match the paper's published value) — no direct check that the
  hash table itself returns collision-free correct probes (e.g. no
  independent CPU-hash-map cross-check of the join result).
- **memory overhead**: not reported as an explicit quantity (bytes/key or
  bytes overhead vs a theoretical minimum); the only memory-consequence
  visible is the `load_factor = 0.4` sizing choice, which by construction
  allocates ~2.5x more slots than keys.
- **baselines**: cuDF (`code/cudf_related/transitive_closure.py`, Python/RAPIDS
  GPU DataFrame join) and Souffle (`code/datalog_related`, CPU Datalog
  engine) — per the input JSON's `one_liner`, up to 10.8x vs cuDF.
- **hardware**: NVIDIA A100 (README specifies >=40GB memory requirement;
  reproducibility results collected on an 80GB A100 + 128 CPU cores).

## 2. NisaPEOBY21 — Distributed-Memory k-mer Counting on GPUs / DEDUKT (IPDPS 2021)

Source: fulltext PDF via the `oa_url`
(`https://escholarship.org/content/qt6089p11k/qt6089p11k.pdf`), extracted
with `pypdf`; cross-checked against the artifact repo `PASSIONLab/DEDUKT`
(`kmercount/common_gpu.h`, `kmercount/kmrCnt_GPU.h`).

- **hash table used for**: a **counting (upsert) hash table** for k-mer
  frequency histograms — the fundamental operation is "increment if present,
  else insert with count 1," never a plain lookup or a delete. This is a
  materially different op profile from a CRUD table.
- **hash table design**: `struct KeyValue { keyType key /* uint64 */;
  uint32_t value; }`; open addressing with **linear probing**, collisions
  resolved by "seek[ing] for a free slot in a probe sequence (linear,
  quadratic, etc.). In this work, we use linear probing" (Sec. III-B,
  verbatim). Hashing via **MurmurHash3** (both 32-bit and 64-bit variants
  present in `common_gpu.h`). Fixed capacity per local GPU table:
  `kHashTableCapacity = 4 * 128 * 1024 * 1024` ≈ 536,870,912 slots — a
  compile-time constant, **not resized** dynamically; each MPI
  rank/GPU owns one such table for its post-hash partition of k-mers.
  Both insert and increment are performed **atomically** to avoid races
  under concurrent GPU threads.
- **operation mix**: 100% upsert (find-or-insert-then-increment), applied in
  bulk over an entire partition's worth of parsed k-mers or supermers per
  call — no isolated read-only or delete workload is ever run.
- **key distribution**: real genomic k-mer streams, explicitly flagged as
  **skewed**: "the frequency distribution of the k-mers is often skewed"
  (Sec. II-A). Skew is quantified in Table III as a **load-imbalance ratio**
  (max partition load / average partition load): 1.16 for k-mer-based
  partitioning on C. elegans 40X, rising to **2.37** for the
  supermer/minimizer-based partitioning on H. sapien 54X — i.e. the paper
  directly measures and reports how partition-key skew degrades balance,
  though it never compares against a synthetic uniform-key baseline for the
  hash table itself.
- **datasets** (Table I, exact sizes): E. coli 30X (792 MB), P. aeruginosa
  30X (360 MB), V. vulnificus 30X (297 MB), A. baumannii 30X (249 MB),
  C. elegans 40X (8.90 GB), H. sapien 54X (317 GB); k=17 for all runs.
- **batch vs single-op**: batch-only; entire read files are parsed into
  k-mer/supermer streams and bulk-inserted per pass; total keys processed
  range from ~129M (A. baumannii k-mer count) to **167 billion** (H. sapien
  54X k-mer count, Table II).
- **load factor**: not reported as an explicit percentage anywhere in the
  reachable text; the fixed 536M-slot capacity relative to per-partition
  k-mer counts (Table III: 12M-283M average per partition on 384 GPUs) puts
  the *effective* per-table load factor loosely in the 2-50% range depending
  on dataset and partition count, but this ratio is never computed or
  plotted by the paper — a track-level gap (see Divergences).
- **timing protocol**: **no warmup, no repeated-trial averaging, and no
  named statistic** appear anywhere in the reachable evaluation section —
  figures (Fig. 3, 6, 7, 9) appear to show single-run wall-clock
  measurements per configuration. Timer mechanism is not named explicitly
  (likely host-side wall clock bracketing each pipeline phase, consistent
  with the phase-by-phase breakdown reported, but not confirmed from the
  text alone).
- **timing scope**: results explicitly **exclude I/O** ("a reduction in
  overall runtime from approximately 50 minutes to just 30 seconds
  (excluding I/O)", Sec. III-C). The pipeline has 3 phases (parse, MPI
  exchange, count); the counting-hash-table step is isolated in the
  **scalability figure (Fig. 9)**, which reports "Scalability of k-mer
  insertion rate... using the computation kernels on GPUs (**excl. exchange
  module**)" — i.e. one figure gives hash-table-only throughput, while the
  headline speedup figures (Fig. 6) are end-to-end pipeline (parse +
  exchange + count) speedups vs. the CPU baseline.
- **metric**: primary throughput unit is **billions of k-mers/sec**
  (Fig. 9 y-axis), for the counting-kernel-only scope; separately, overall
  pipeline results are reported purely as **speedup vs. CPU baseline**
  (up to 100x on 384 GPUs vs. 2688 CPU cores; up to 150x with the
  supermer optimization on H. sapien 54X), never as an absolute end-to-end
  keys/sec number.
- **precision & correctness**: exact integer counts (no floating point, no
  tolerance); no explicit statement of a correctness-validation step (e.g.
  comparing final k-mer histograms bit-for-bit against the CPU baseline) was
  found in the reachable text, though the CPU-derived baseline (same
  algorithm, no GPU/supermer) implies the intended ground truth.
- **memory overhead**: not reported as bytes/key or as overhead vs an ideal
  packing; only the fixed slot-count constant is disclosed.
- **baselines**: a CPU-only k-mer counter, explicitly the *same algorithm*
  derived from diBELLA's k-mer-analysis component **without** GPU
  acceleration or the supermer optimization (an internal ablation, not a
  third-party library) — Gerbil and KMC3 are discussed as related GPU/CPU
  k-mer counters in Sec. VI but not directly benchmarked against.
- **hardware**: Summit (ORNL) — dual-socket IBM Power9 (22 cores/socket,
  42 cores/node) + 6x NVIDIA V100 (16GB HBM2) per node, NVLink (25 GB/s/link),
  EDR InfiniBand (23 GB/s/node injection); scaled from 4 nodes (24 GPUs) to
  128 nodes (768 GPUs).

## 3. ChenHDZ20 — Lock-free Concurrent Level Hashing for Persistent Memory / Clevel Hashing (USENIX ATC 2020)

Source: fulltext PDF fetched directly from the USENIX-hosted mirror
(`https://www.usenix.org/system/files/atc20-chen.pdf`; WebFetch returned
HTTP 403 for this host, so the PDF was retrieved with `curl` using a
browser User-Agent and then text-extracted with `pypdf`), cross-checked
against the artifact repo `chenzhangyu/Clevel-Hashing`
(`tests/clevel_hash/clevel_hash_ycsb.cpp`, `tests/clevel_hash/README.md`).
This is the only paper in the track with a genuine, independently
invokable insert/read/update/delete API and the only one whose evaluation
matches "workload-mix + load-factor-sweep" benchmarking in the classical
hash-table literature sense.

- **hash table used for**: general-purpose, dynamically-resizable,
  lock-free, multi-threaded key-value index for persistent memory (Intel
  Optane DC PMM); supports independent search/insert/update/delete.
- **hash table design**: level hashing with 8 slots/bucket (chosen via a
  slot-count sweep, Fig. 5: 4/8/16-slot buckets compared; 8 is the reported
  sweet spot), asynchronous background resizing (one dedicated rehashing
  thread by default — the artifact enforces `thread_num >= 2` and reserves
  one thread for this), lock-free concurrent operations for all four op
  types. Hash function: `std::hash` from the C++ STL, used identically
  across ALL compared schemes for fairness.
- **operation mix — exact YCSB workload table (Table 2 of the paper)**:
  | Workload | Read % | Write % |
  |---|---|---|
  | Load A | 0 | 100 (pure insert, populates the table) |
  | A | 50 | 50 |
  | B | 95 | 5 |
  | C | 100 | 0 |

  plus separate **micro-benchmarks** that isolate individual op types:
  insertion (unique keys), positive search (key present), negative search
  (key absent), update, and deletion (target keys pre-populated) — each run
  in isolation with 36 threads to measure per-operation-type latency without
  interference from resizing.
- **key distribution**: micro-benchmarks generated by **YCSB in Zipfian
  distribution with the YCSB default skewness parameter 0.99**; the paper
  states "results using uniformly distributed workloads are similar due to
  the randomness of hash functions" but does not report the uniform numbers
  in the reachable text — Zipfian is what is actually plotted.
- **key/value size**: fixed **15-byte keys, 15-byte values** for every
  experiment (`KEY_LEN 15` in the artifact matches exactly).
- **batch vs single-op**: the YCSB macro-benchmarks are batch-driven (a
  pre-generated trace file consumed in a tight per-thread loop — this
  matches the artifact's `clevel_hash_ycsb[_macro]` binaries exactly), but
  the artifact additionally ships a genuine **single-op CLI**
  (`clevel_hash_cli <pool> <print|alloc|free> <key>`) for one query at a
  time — the only paper in the track offering a true single-op interface
  distinct from its bulk-throughput harness.
- **scale (load + run phase, exact from paper and cross-confirmed in code)**:
  micro-benchmarks — initial capacity 64K, load phase inserts 16,000,000
  items, run phase issues 16,000,000 queries (matches `READ_WRITE_NUM
  16000000` under the non-macro `#define` branch in
  `clevel_hash_ycsb.cpp`). Macro-benchmarks — initial capacity 256K, load
  phase inserts 64,000,000 items, run phase issues 64,000,000 queries
  (matches `READ_WRITE_NUM 64000000` under `MACRO_TEST_FOR_CLEVEL_HASH`).
- **load factor — dedicated first-class evaluation (Sec. 4.2, Fig. 6)**: an
  insert-only workload records load factor (inserted items / total slots)
  **after every 10K insertions** (this exact protocol is implemented
  verbatim in the artifact's `clevel_hash_resize` test). Reported maximum
  load factors before resizing: **CCEH <=45%** (limited by a fixed 16-slot
  linear-probe distance), **P-CLHT up to 84%** (3-slot bucket + linked
  list), **CMAP always 100%** (slots allocated on demand, i.e. no
  preallocation overhead but no probing benefit either), **LEVEL and
  CLEVEL (this paper) ~86%**. This is the most rigorous load-factor
  treatment in the track — the natural definition to standardize the
  spec on.
- **timing protocol**: **5 runs, reported as the average** ("The reported
  latency and throughput are the average values of 5 runs," Sec. 4.1) — the
  only paper in the track with an explicit repeated-trial statistic. No
  separate warmup phase is described (the load phase, itself unmeasured for
  throughput purposes, effectively warms caches before the timed run phase).
  Timer: `clock_gettime(CLOCK_MONOTONIC)` wraps the entire multi-threaded
  run phase (matches the artifact exactly); an optional
  `LATENCY_ENABLE` compile-time mode additionally records **per-operation
  latency** timestamps for percentile/average-latency analysis.
- **metric**: primary is **throughput in ops/sec**, computed as
  `total_run_phase_ops / elapsed_seconds`, a single **blended** number
  across whatever read/write mix the workload defines (not decomposed
  per-op-type in the throughput figure, though the op-type *counts* are
  tracked separately); secondary is **average per-op latency in
  microseconds** from the micro-benchmarks (isolated per op type: positive
  search, negative search, update, deletion, insertion).
- **precision & correctness**: exact string-key equality (`std::equal_to`);
  correctness is enforced structurally (a bug in a competitor, CCEH,
  was found and *patched* by the authors during porting — a missing reader
  lock causing search-after-concurrent-directory-doubling failures — showing
  correctness was actively verified during evaluation, not merely assumed).
- **memory overhead**: not reported as absolute bytes/key, but the
  **load-factor curve is explicitly used as the memory-efficiency metric**
  (Sec. 4.2: "In order to evaluate the memory efficiency of different
  schemes, we use an insert-only workload to record the load factor...").
  This is the track's only quantified memory-overhead proxy (overhead ∝
  1/load_factor − 1).
- **baselines**: LEVEL (original concurrent level hashing), CCEH
  (cacheline-conscious extendible hashing, lazy-deletion variant), CMAP
  (pmemkv's `concurrent_hash_map`), P-CLHT (RECIPE's cache-line hash
  table); TBB-spinlock variants of LEVEL/CCEH/CMAP additionally compared.
  All ported to the same PMDK 1.6 platform with the same STL hash function
  and same reader/writer lock implementation, specifically to remove
  lock/hash-function choice as a confound — a fairness practice worth
  adopting in the spec.
- **hardware**: 6x Intel Optane DC PMM (1.5 TB total) in App Direct mode,
  ext4-DAX filesystem, 128 GB DRAM, 24.75 MB L3, 2 sockets x 36
  threads/socket; all experiments pinned to a **single NUMA node** (max 36
  threads) to avoid cross-socket latency noise.

---

## Divergences

1. **Fundamentally different operation profiles, not just different
   ratios.** ShovonGMK23's table supports only bulk-insert-then-bulk-probe
   (no delete, no single-op); DEDUKT's table supports only bulk upsert
   (find-or-increment-or-insert, no plain read, no delete); Clevel-Hashing
   is the only one with independently invokable insert/read/update/delete
   and the only one following the classical YCSB-workload-letter convention
   (Load A / A / B / C with explicit read:write ratios). A spec that only
   offers a single "insert/lookup/delete ratio" knob would misrepresent
   two-thirds of this track's actual workloads. **Resolution**: the spec
   defines three separate variants matching these three op-profile
   families (build-probe join table, counting/upsert table, general CRUD
   table) rather than forcing one universal op-mix parameter.

2. **Load factor: swept and reported vs. fixed and undisclosed.**
   Clevel-Hashing treats load factor as a first-class evaluated curve
   (Fig. 6, sampled every 10K insertions, competitor max load factors
   reported to 2 significant figures: CCEH<=45%, P-CLHT<=84%, CLEVEL~86%).
   ShovonGMK23 fixes load_factor=0.4 as a hardcoded default and never
   varies or reports it as a curve. DEDUKT never computes or reports a
   load-factor number at all despite having a fixed-capacity table where
   the ratio is directly computable from disclosed numbers (Table
   III's per-partition k-mer counts vs. the 536M-slot constant).
   **Resolution**: the spec makes a load-factor sweep (a fixed target-set,
   e.g. {0.25, 0.5, 0.75, 0.9}) a **required** protocol axis in every
   variant, following Clevel-Hashing's methodology, specifically because
   two of the three papers under-report this axis despite it being cheap
   to measure and central to hash-table design tradeoffs.

3. **Key distribution disclosure is uneven.** Clevel-Hashing is explicit
   and quantitative (YCSB Zipfian, skewness=0.99, with a passing claim that
   uniform gives "similar" results but no uniform numbers shown). DEDUKT
   is explicit that real genomic data is skewed and quantifies the
   resulting load imbalance (1.16x-2.37x) but never runs a synthetic
   uniform-key control to isolate the hash table's own behavior from the
   genomic data's inherent skew. ShovonGMK23 uses only real-world relations
   (implicitly skewed, e.g. power-law social graphs) and never reports a
   skewness metric, though its standalone `hashjoin.cu` microbenchmark does
   support synthetic uniform-random keys as an option that the paper never
   exercises. **Resolution**: the spec requires BOTH a synthetic uniform
   and a synthetic Zipfian(0.99) key generator (matching Clevel-Hashing's
   own choice of skewness, for direct comparability) as mandatory
   variant inputs, with real-world key sets (SNAP-style graphs for
   join-style workloads, real k-mer corpora for counting-style workloads)
   as an additional, separately-reported "real-world" condition — since
   none of the three papers isolates synthetic-distribution effects from
   real-dataset effects cleanly on its own.

4. **Timing protocol: only one paper reports a repeated-trial statistic.**
   Clevel-Hashing explicitly averages 5 runs (Sec. 4.1) — the only paper in
   the track to name a repetition count and a statistic. ShovonGMK23's
   README reproducibility table shows single "Reproduced Results" next to
   single "Results in Paper" values that differ by tens of percent on some
   datasets (e.g. CA-HepTh: 2.65s vs 4.32s) with no explanation, itself
   evidence single-run timing is unreliable for this class of GPU kernel.
   DEDUKT's reachable text names no warmup/repetition/statistic at all.
   **Resolution**: the spec fixes a uniform protocol (warmup + >=5
   measured repetitions + median, min/max reported) across all three
   variants, adopting Clevel-Hashing's repetition count as the floor since
   it is the only literature precedent in this track, and explicitly
   flagging that ShovonGMK23's and DEDUKT's single-run numbers should not
   be taken as reproducible without re-measurement under this protocol.

5. **Throughput metric scope: build-only, counting-only, and full-mix
   blended are three incompatible numbers.** ShovonGMK23 prints a
   build-only "Hashtable rate" (keys/s) alongside separate join/dedup/union
   timings, but never gives a single combined "hash table ops/s" figure
   comparable across papers. DEDUKT's counting-kernel throughput (Gk-mers/s,
   Fig. 9) explicitly **excludes** the MPI exchange phase, isolating exactly
   the hash-table-relevant cost, but its headline pipeline speedups (Fig. 6)
   are end-to-end and not decomposable back into a hash-table-only number
   from the paper text alone. Clevel-Hashing's ops/sec is blended across
   whatever read/write ratio the workload letter specifies, not split by
   op type in the throughput number (only in the separate micro-benchmark
   latencies). **Resolution**: the spec requires every variant to report
   **both** a kernel-only (table-operations-only, excluding any
   partitioning/communication/parsing) throughput number AND, if
   applicable, the surrounding pipeline's end-to-end number as a distinct,
   separately-labeled figure — never blending the two silently, mirroring
   DEDUKT's own good practice of isolating Fig. 9 from Fig. 6.

6. **Memory overhead is reported by only one paper, and only as load
   factor.** Clevel-Hashing's load-factor curve (item 2 above) is the only
   direct memory-overhead evidence in the track; ShovonGMK23's
   load_factor=0.4 constant implies ~2.5x slot overallocation but this is
   never stated as a memory cost; DEDUKT's fixed capacity constant is never
   related back to memory footprint at all. **Resolution**: the spec
   defines memory overhead explicitly as `bytes_allocated / (num_live_keys *
   (key_size + value_size)) - 1`, computed at each swept load-factor point,
   since no paper in the track provides this as an absolute figure and it
   is directly derivable from the load-factor protocol adopted in item 2.

## Open questions

- DEDUKT's effective load factor per dataset/partition-count combination
  could be computed from Table III's disclosed numbers (average k-mers per
  partition ÷ the 536,870,912-slot constant) but is never done by the paper
  itself; the spec's load-factor-sweep variant should be checked against
  this derived range (roughly 2%-53% across the disclosed configurations)
  once real GPUs are available to re-measure rather than trusting the
  back-of-envelope derivation alone.
- ShovonGMK23's `hashjoin.cu` reports a "Hashtable rate" that (from the
  kernel code) times only the `build_hash_table` kernel via CUDA events —
  but it is not confirmed from the reachable README/paper text whether the
  in-paper published numbers for the full TC pipeline (`tc_cuda.cu`) used
  the *same* `load_factor=0.4` default, since `tc_cuda.cu`'s hash table is
  rebuilt with a fresh `load_factor` argument on every one of up to 606
  iterations and only the top-level `main()` default was inspected.
- Clevel-Hashing's claim that uniform-key-distribution YCSB results are
  "similar" to the reported Zipfian(0.99) results is not backed by any
  visible numbers in the reachable text (only asserted); the spec's
  requirement for both distributions (Divergence 3) should be treated as
  testing this claim, not assuming it holds for competing hash-table
  designs in general.
- None of the three papers' artifacts expose GPU-side (ShovonGMK23,
  DEDUKT) vs CPU-PM-side (Clevel-Hashing) results on the same hardware, so
  the spec cannot fix a single hardware baseline across all three variants
  — each variant instead specifies its own natural platform (GPU for
  build-probe and counting variants, following the two GPU papers; CPU+PM
  or CPU+DRAM for the CRUD variant, following Clevel-Hashing) rather than
  forcing an artificial cross-platform comparison the literature itself
  does not attempt.
