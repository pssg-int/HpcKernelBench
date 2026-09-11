# String/regex-matching track — evaluation-methodology survey

Surveyed 5/5 papers from `data/track_inputs/string-regex-matching.json`. None
have an arXiv preprint; ACM DL (`dl.acm.org`) returned HTTP 403 to WebFetch
for both `doi.org`-resolved links (GeZ024, 0002PJ20). Only CicoliniCSC24's
`oa_url` (an author-hosted Politecnico di Milano institutional-repository
PDF, not paywalled) yielded full fulltext via `pypdf` extraction. For the
remaining 4 papers, evidence comes from their public artifact repositories —
READMEs, artifact-evaluation (`AE.md`) documents, benchmark config files
(`app_spec` JSON/Python listing every automaton+input pair), and driver
source code (`kernels.cu`, `framework.cu`) — which for this track are
unusually rich because 3 of the 5 (`getianao/ngAP`, `bigwater/gpunfa-artifact`,
`necst/iMFAnt`) are artifact-evaluation-badged repositories with reproduction
scripts that name every benchmark, input size, and metric explicitly.

A track-defining fact discovered during the survey: **three of the five
papers (GeZ024/ngAP, 0002PJ20/GPU-NFA, and by direct citation MFSA) sit on a
shared benchmark-corpus lineage** — the ANMLZoo (Wadden et al.) + AutomataZoo
(Angstadt et al.) + "Regex" (Becchi et al., regex.wustl.edu) suites — and
**ngAP's own artifact directly reimplements 0002PJ20's GPU-NFA scheme as a
baseline** (labelled `obat` in ngAP's build, citing "GPU-NFA (ASPLOS 2020)"
verbatim). GSpecPal, by contrast, draws from a separately-sourced Snort/
ClamAV/PowerEN DFA-table corpus that is NOT ANMLZoo-packaged. Cicero targets
an FPGA domain-specific architecture and reports fundamentally different
units (microseconds-per-match, energy) rather than a corpus-throughput
number.

---

## 1. SomainiCASC25 — Combining MLIR Dialects with Domain-Specific Architecture for Efficient Regular Expression Matching / Cicero (CGO 2025)

Source: artifact repo `necst/cicero` — `README.md` and
`scripts/artifact_evaluation/AE.md` (the paper's own `oa_url` is an
ACM-DL `doi.org` redirect; not fetched — paywalled). Tier-2 source
(artifact-only, no fulltext).

- **what is being measured**: an FPGA domain-specific architecture (DSA)
  that executes regular-expression matching **without backtracking**
  (explicit design goal, contrasted with RE2-style backtracking engines in
  the README prose), fed by an MLIR-based compiler (`regex` dialect ->
  `cicero` dialect -> Cicero ISA binary).
- **automata family disclosure**: NOT a classical enumerated NFA/DFA table
  at all — REs are compiled to a custom instruction-set binary executed by
  a hardware "sliding input window" architecture with per-character
  FIFO+core units; the architecture is functionally NFA-equivalent
  (explores multiple options in one pass, no backtracking) but is not
  represented or benchmarked as a DFA/NFA state-transition table the way
  the other 4 papers in this track are.
- **pattern sets / input corpus**: **AutomataZoo**
  (`https://github.com/tjt7a/AutomataZoo`), the same upstream dataset
  0002PJ20 and GeZ024 draw from (see below); the AE checklist does not
  enumerate which specific AutomataZoo sub-benchmarks are used or their
  RE counts, only that "data sets are already included with the rest of
  the provided artifact."
- **compilation/preprocessing vs matching, explicitly separated**: the
  AE checklist's own metrics list keeps these strictly apart: **average
  compilation time (microsecond)**, **code size**, and **code locality**
  are one group of static-compiler metrics (measured on an x86 host,
  comparing OLD vs NEW compiler implementations); **average runtime to
  match a RE (microsecond)** and **average energy used to match a RE**
  (W*microsecond) are a second, disjoint group measured on the FPGA board
  itself. FPGA resource usage (%) and total on-chip power (W) form a third
  group (bitstream/hardware-synthesis metrics, not matching performance at
  all).
- **throughput metric**: NOT a corpus-wide Gbps/MB/s number — Cicero
  reports **per-match latency in microseconds** and **per-match energy**,
  a fundamentally different unit family from the other 4 papers' aggregate
  corpus-throughput numbers (a genuine, reportable divergence, see below).
- **timing protocol**: not disclosed in the reachable README/AE.md (no
  warmup/repetition/statistic named); the AE.md states RE-matching on the
  FPGA "should have near-identical processing times thanks to the
  stability of the hardware environment," implying low run-to-run variance
  is expected/assumed rather than empirically bounded via repeated trials.
- **correctness**: not described in the reachable sources (the paper's
  entire premise — non-backtracking exact matching — is a correctness
  property by construction, but no independent-reference cross-check
  procedure is documented in the artifact).
- **hardware**: Avnet Ultra96-V2 FPGA board (PYNQ 3.0.1) + an x86 host for
  the compiler-only metrics; MLIR/LLVM 16, Antlr4 toolchain.
- **baselines**: OLD (CASES21 Parravicini et al. predecessor) vs NEW
  (this paper's MLIR-based) compiler, compared on static metrics; no
  external regex-matching system (Hyperscan, RE2, an NFA/DFA engine) is
  benchmarked head-to-head in the reachable sources, only the paper's own
  prior architecture.

## 2. GeZ024 — ngAP: Non-blocking Large-scale Automata Processing on GPUs (ASPLOS 2024)

Source: artifact repo `getianao/ngAP` — `README.md`,
`code/scripts/configs/app_spec_ngap_new` (full benchmark listing). The
`oa_url` (`dl.acm.org/doi/pdf/...`) returned HTTP 403 to WebFetch. Tier-2
source.

- **automata family disclosure**: **NFA**, explicit. ngAP's own baselines,
  built from the same codebase, are all NFA engines: `ppopp12` (NFA-CG,
  PPoPP 2012), `asyncap` (AsyncAP, SIGMETRICS 2023), **`obat` (GPU-NFA,
  ASPLOS 2020 — i.e. this track's own paper 0002PJ20, reused verbatim as a
  baseline)**, and `ngap` (this paper). A CPU-side Hyperscan (`hsrun`,
  NSDI 2019) build is additionally provided as a production DFA/hybrid CPU
  baseline for cross-checking.
- **pattern sets / input corpus**: the `app_spec_ngap_new` config lists
  named benchmarks each pointing to an automaton + a fixed input file:
  **ANMLZoo** benchmarks (Dotstar — `backdoor_10MB.input`; PowerEN —
  `poweren_10MB.input`) and **AutomataZoo** benchmarks (Brill —
  `brown_corpus.txt`; ClamAV — `clamav.input`; EntityResolution —
  `1m_names.input`; FileCarving — `fat32_files.input`; CRISPR_CasOFFinder /
  CRISPR_CasOT — `10MB_G.dna` / `10MB_H.dna`; RandomForest —
  `input_features_large.bin`; and more). Every entry also carries
  precompiled representations in multiple formats (`.anml`, `.mnrl`, `.hs`
  for Hyperscan) plus a `quick_validation` field, i.e. the artifact
  ships one automaton per benchmark **name** (not a size-sweep family the
  way GSpecPal does).
- **compilation/preprocessing vs matching**: automata are supplied
  pre-compiled (`.anml`/`.mnrl`/`.hs` files ship in the downloaded 2.5 GB
  benchmark archive); the artifact's `run-breakdown.sh` (~8 hrs, generates
  Fig. 14) is explicitly a **breakdown** experiment separate from
  `run-throughput.sh` (~16 hrs, Fig. 13/Table 4), implying the paper
  itself decomposes total time into phases (though the exact phase names
  were not confirmed from the reachable README text alone — flagged as an
  open question).
- **throughput metric**: **MB/s**, confirmed directly from the shipped
  demo-run output: `"ngap elapsed time: 3.6864e-05 seconds, throughput =
  2.19727 MB/s"` — throughput is computed as bytes of input stream
  processed divided by elapsed wall/device time.
- **correctness gate**: every run is validated against **a serial CPU
  reference automata-processing pass**, printed explicitly
  (`"############ Validate result ############\nValidation PASS!"`),
  comparing exact match-position lists before a run's throughput is
  presumably trusted; a `-quick-validation` flag exists to skip this for
  faster iteration, implying full validation is the default/paper-used
  mode.
- **algorithm variants swept** (a within-paper ablation axis): `ngap`
  itself exposes 5 selectable algorithms via `--algorithm`:
  `blockinggroups` (baseline blocking automata processing, BAP),
  `NAPgroups` (ngAP core), `nonblockinggroups` (+ prefetch always-active
  states), `nonblockingpcgroups` (+ prefix memoization), and
  `nonblockingallgroups` (+ work privatization) — i.e. the paper's 3
  claimed optimizations are individually turn-on-able and presumably
  individually measured.
- **timing protocol**: not confirmed in detail from the reachable README
  (no explicit warmup/repetition/statistic named at the artifact-README
  level; the config-driven `launch_exps.py` harness likely encodes this,
  but was not inspected file-by-file).
- **hardware**: tested on NVIDIA RTX 3090 (Ampere, 24 GB) and Tesla V100
  SXM2 (Volta, 32 GB); CUDA >=12.0, requires >=24 GB device memory.

## 3. CicoliniCSC24 — One Automaton to Rule Them All: Beyond Multiple Regular Expressions Execution / MFSA + iMFAnt (CGO 2024)

Source: **fulltext PDF** via the paper's own `oa_url`
(`https://re.public.polimi.it/bitstream/11311/1262662/1/mfse_midend_cgo24.pdf`,
a Politecnico di Milano institutional-repository mirror, not paywalled),
extracted with `pypdf`. Tier-1 source — the only paper in this track with
full evaluation-section fulltext reached.

- **automata family disclosure**: **NFA**, explicit and discussed at
  length (Sec. II-B contrasts DFA-vs-NFA tradeoffs directly: "NFAs have a
  significantly lower memory footprint compared to DFAs. However, their
  execution requires the simultaneous activation of multiple transitions
  ... A set of works addressing this issue targets NFAs execution on
  hardware accelerators"). The paper's own contribution (MFSA) is an
  **NFA extension** built via Thompson-construction from REs; execution
  uses **iMFAnt**, an extension of the pre-existing **iNFAnt** NFA
  algorithm (the same iNFAnt that 0002PJ20's artifact also reimplements as
  a baseline — a second cross-paper lineage point in this track).
- **pattern sets — exact table (Table I of the paper)**, drawn from two
  standard benchmark-suite lineages also used elsewhere in this track:
  | Abbr. | Dataset | # REs | Source |
  |---|---|---|---|
  | BRO | Bro217 | 217 | Becchi et al. "Regex" suite |
  | DS9 | Dotstar09 | 299 | Wadden et al. (ANMLZoo) |
  | PEN | PowerEN | 300 | Becchi et al. "Regex" suite |
  | PRO | Protomata | 300 | Becchi et al. "Regex" suite |
  | RG1 | Ranges1 | 299 | Becchi et al. "Regex" suite |
  | TCP | TCP-ext.homenet | 300 | Wadden et al. (ANMLZoo) |

  i.e. **PowerEN is directly shared** with 0002PJ20 and (via ANMLZoo)
  GeZ024's benchmark suites; Bro217/Ranges1 match the exact benchmark
  **names** that appear in 0002PJ20's `app_spec` config (`Bro217`,
  `Ranges05`), confirming these three papers draw from a genuinely shared
  benchmark family, not merely similarly-named-but-different sets.
- **input corpus**: a **fixed 1 MB data input stream** used for every
  dataset in the throughput evaluation (Sec. VI-C) — a single corpus size,
  no size sweep (a divergence from GSpecPal's 1MB/10MB/20MB and ngAP's
  10MB/20MB conventions, see below).
- **compilation/preprocessing vs matching — the most detailed breakdown in
  the track (Sec. VI-B, Fig. 8)**: explicitly measures, per compilation
  stage, **averaged over 30 executions**: front-end (lexing/parsing,
  1.29 ms avg), AST-to-FSA conversion (1.33 ms avg), single-FSA
  optimization (2.03 ms avg) — all three independent of merging factor M —
  plus the merging stage itself, which dominates and scales with M (up to
  **6.65 s** at M=all, the most computationally-intensive configuration),
  yielding a total average compilation time of **6.66 s**. This is
  reported as **strictly separate** from the execution-throughput numbers
  (Sec. VI-C) — a clean, well-isolated preprocessing/matching split.
- **throughput metric**: a **custom RE-throughput definition**, not
  Gbps/MB/s: `throughput_MFSA = (#MFSA * M * D_size) / Exe_time_tot`
  (Eq. 11) — data processed multiplied by the number of REs simultaneously
  represented, divided by total execution time; reported primarily as a
  **relative fold-improvement** ("Throughput Imp. [x]", Fig. 9) against the
  M=1 (single-FSA, no merging) configuration, not as an absolute
  bytes/second figure comparable across papers.
- **timing protocol**: execution timing is **averaged over 15 iMFAnt
  runs** ("To increase the reliability of the results, we averaged the
  execution time of 15 iMFAnt runs," Sec. VI-C) — a DIFFERENT repetition
  count from the 30-execution average used for compilation-time
  measurement in the same paper (two distinct, paper-disclosed repetition
  counts for two distinct measured quantities). No explicit warmup phase
  is described for either.
- **correctness**: enforced structurally via the **activation-function**
  formal model (Sec. III-B), which provably prevents an MFSA from
  recognizing strings outside the union of its constituent REs' languages
  (a design-time correctness guarantee, not a runtime cross-check against
  an independent reference implementation).
- **hardware**: single machine, **Intel i7-6700 CPU (4 cores, 8 threads)**
  — a CPU-only paper (the only CPU-only automata-matching paper in this
  track; the other 4 all target GPU or FPGA). Thread scaling swept 1 to
  **128 threads** (i.e. well beyond the 8 physical hardware threads, to
  study oversubscription).
- **merging factor swept**: M in {1, 2, 5, 10, 20, 50, 100, all}
  (all = merge the entire dataset into one MFSA) — the paper's own
  primary experimental axis, analogous to a "compression ratio vs
  throughput" tradeoff curve.
- **baselines**: single-FSA (M=1, no merging) and naive multi-threaded
  parallel-FSA execution (one thread per RE) — both are the paper's own
  un-merged ablations, not external third-party regex engines (RE2,
  Hyperscan, PCRE are cited in the introduction as related work but not
  directly benchmarked in the reachable evaluation section).

## 4. WangWQW22 — GSpecPal: Speculation-Centric Finite State Machine Parallelization on GPUs (IPDPS 2022)

Source: artifact repo `l1ghtWang/GSpecPal` — `README.md` (minimal),
`script_release/runList_23clamAV_20input_exeTime.sh` (the actual run
driver, most informative source), `src/framework.cu` (timing/output
format), `data/{clamAV,newSnort,powerEN}` directory listings. No
arXiv/OA fulltext; IPDPS is IEEE-paywalled. Tier-2/3 source
(artifact-only).

- **automata family disclosure**: **DFA**, explicit — the artifact's core
  directory is literally named `DFA_GPU`, and every pattern set ships as
  a compiled **DFA transition table** (`*.table` binary files) alongside
  an Aho-Corasick text representation (`*.txt`). This is the track's
  **only DFA-based** paper (the other 4 are NFA-based or a non-classical
  FPGA ISA) — GSpecPal's "speculation-centric" contribution is specifically
  about parallelizing sequential DFA-table traversal on GPUs via
  speculative execution + 4 recovery-policy heuristics for misspeculation,
  not about NFA multi-state activation the way the rest of the track is.
- **pattern sets — exact match to the task's named axis**: three real
  rulesets, each shipped as **multiple distinctly-sized DFA compilations**
  (the numeric suffix in each filename, e.g. `clamAV_DFA1104.table`
  through `clamAV_DFA2042.table`, is a distinct automaton-size identifier
  — i.e. GSpecPal sweeps **automaton size** as its own axis, not just
  corpus content):
  - **ClamAV**: 23 distinct DFA-table sizes (`clamAV_list` in the run
    script: 1976, 1997, 1798, 1447, 1104, 1501, 1133, 1319, 1609, 1717,
    2042, 1777, 1593, 1374, 1171, 1487, 1623, 1455, 1732, 1744, 1563,
    1701, 1342).
  - **newSnort**: 20 distinct DFA-table sizes (12038-12974 range;
    `snort_regex*.regex` files additionally ship the raw regex text for a
    subset).
  - **powerEN**: 21 distinct DFA-table sizes (841-9841 range).
- **input corpus / sizes**: **20 distinct 10 MB input files** per pattern
  set for the main exeTime experiments (`clamAV_input_10MB_1.in` through
  `_20.in`, same convention for newSnort); powerEN additionally ships
  **20 distinct 20 MB input files**; a separate, smaller **20x 1 MB
  "profilingInput"** set exists per pattern set for a distinct
  runtime-profiling experiment (`runList_*_runtimeProfiling.sh`,
  `offline_dfa_profiler`), i.e. GSpecPal explicitly separates a
  throughput-measurement corpus (10-20 MB) from a profiling corpus (1 MB)
  — two different input scales for two different purposes.
- **compilation/preprocessing vs matching**: DFA tables ship
  **pre-compiled** (as `.table` binaries) — RE-to-DFA compilation is
  entirely **out of scope** of the runtime-matching benchmark; a separate
  `offline_dfa_profiler` tool (`run_offline_dfa_profiler_{clamAV,powerEN,
  snort}_20input.sh`) exists specifically to profile DFA characteristics
  offline, confirming preprocessing is treated as a genuinely separate,
  one-time step never mixed into the matching-throughput number.
- **timing protocol — explicit warmup, confirmed in the code**: the run
  script issues **3 discarded warmup invocations** on a fixed Snort
  automaton (`snort_DFA12771.table` against `newSnort_input_10MB_1.in`)
  BEFORE the main experiment sweep begins, with output not captured. The
  binary itself (`framework.cu`) separately prints `"The Total WarmUp time
  (in s) is: ..."` and `"The Total Elapsed time (in s) is: ..."` as two
  DISTINCT timed quantities inside a single invocation too — i.e. warmup
  is measured (not merely discarded) at both the shell-script level
  (3 throwaway runs) and the in-binary level (a named, reported warmup
  phase within each run).
- **repetition**: for each (recovery-policy, automaton-size) pair, all
  **20 distinct 10MB input files** are run once each and appended to a
  per-configuration output file (`raw_clamAV<id>_<policy>.txt`) — this is
  repetition-over-distinct-corpus-samples, not repeated-trials-of-the-same
  input; no explicit statistic (mean/median) computation is visible in the
  shell driver itself (would happen in unreached post-processing).
- **throughput metric**: **NOT normalized to Gbps/MB/s in the reachable
  driver code** — `framework.cu` prints raw elapsed **seconds**, not a
  throughput figure; if the paper itself reports a Gbps number, the
  conversion (from 10-20MB-per-fixed-elapsed-time) would happen in
  unreached post-processing/plotting scripts — flagged as an open
  question below, and itself a real divergence from every other paper in
  this track, all of which report normalized throughput directly.
- **correctness gate**: explicit CPU cross-check
  (`"Use CPU results to verify the correctness..."` printed, followed by
  per-block `"Pass Block i"` / `"Errors in Block i"` output) — a genuine
  independent-reference validation step, run as part of each invocation.
- **recovery-policy axis swept**: 4 policies (`merge`, `end`, `round`,
  `first` — corresponding to `P_MERGE`, `P_ENDSTATE`, `P_ROUNDROBIN`,
  `P_NEARESTFIRST` in the code) x 23 ClamAV automaton sizes x 20 input
  files = 1,840 individual runs for the ClamAV sweep alone — this
  speculative-recovery-heuristic axis is GSpecPal's own primary
  contribution and has no analogue in the other 4 papers.
- **hardware**: CUDA/GPU (implied by `DFA_GPU` directory, `.cu` kernel
  files `kernels_full.cu`/`kernels_hash.cu`/`kernels_transform.cu`);
  specific GPU model not disclosed in the reachable README.

## 5. 0002PJ20 — Why GPUs are Slow at Executing NFAs and How to Make them Faster / "GPU-NFA" (ASPLOS 2020)

Source: artifact repo `bigwater/gpunfa-artifact` — `README.md`,
`run_experiments_get_table3.sh`, `gpunfa_code/scripts/configs/app_spec`
(full benchmark listing), `gpunfa_code/scripts/ploting/abs_throughput_table.py`
(baseline/config name mapping). The `oa_url`
(`dl.acm.org/doi/pdf/10.1145/3373376.3378471`) returned HTTP 403 to
WebFetch. Tier-2 source.

- **automata family disclosure**: **NFA**, explicit in the title itself.
  This paper IS the origin of the `GPU-NFA` scheme that GeZ024/ngAP later
  reimplements as its `obat` baseline — direct cross-paper lineage
  confirmed from ngAP's own README citation.
- **pattern sets / input corpus**: `app_spec` config lists named
  benchmarks from the SAME two suite lineages as MFSA and ngAP:
  **AutomataZoo** (Brill/`brown_corpus.txt`; ClamAV/`clamav.input`;
  CRISPR_CasOFFinder/`10MB_G.dna`; CRISPR_CasOT/`10MB_H.dna`;
  EntityResolution/`1m_names.input`; Hamming_l18d3/`10MB_D.DNA`;
  Levenshtein_l19d3/`10MB_A.dna`; Protomata/`30k_prots.input`;
  Snort/`wrccdc2012.pcap`, a real network-traffic trace; YARA/
  `malware_malz2.input`) and **"Regex"** (Becchi et al.,
  regex.wustl.edu — Bro217/`Bro217_10MB_...trace.input`; ExactMath;
  Ranges05; plus more not fully enumerated in the excerpt read). **Snort**
  is named directly (matching GSpecPal's Snort ruleset, though sourced
  from a different upstream packaging — GSpecPal's is a raw Aho-Corasick/
  DFA-table export, this paper's is an ANML-format automaton matched
  against a real pcap trace) and **Bro217** is shared verbatim with MFSA's
  BRO dataset.
- **compilation/preprocessing vs matching**: not confirmed in detail from
  the reachable README (automata ship pre-converted to ANML format per
  the README's own statement: "we convert the automata files of them to
  ANML format... we provide the data set that is ready to use" — i.e.
  format conversion is a one-shot artifact-preparation step, done once
  and shipped, not part of the timed matching benchmark).
- **throughput metric**: the artifact's headline reproduction target is
  explicitly named **"Table 3"** and generated via
  `abs_throughput_table.py` into `abs_throughput.csv` — i.e. the paper's
  primary reported metric is **absolute throughput** (as opposed to a
  relative-speedup-only table); the exact unit (Gbps vs MB/s) was not
  directly confirmed from the reachable script header, but given this
  paper is the direct ancestor of ngAP's MB/s convention and shares the
  same benchmark/config-file lineage, MB/s is the most likely unit
  (flagged as an open question rather than asserted).
- **baselines — the richest baseline set in the track**: the plotting
  script's own config-name mapping reveals the full comparison set: **AP**
  (Micron's Automata Processor hardware, `AP` / `AP_ideal` — a real
  hardware NFA/DFA accelerator, not another software scheme), **iNFAnt**
  (their own reimplementation of the classic iNFAnt GPU-NFA algorithm —
  the same iNFAnt that MFSA's iMFAnt extends), **NFA-CG** (their
  reimplementation of PPoPP 2012's NFA-CG), and 4 of their OWN ablations:
  **NT** (NewTran), **NT-Mac** (NewTran + matchset compression),
  **HotStart**, **HotStart-Mac**, **HotStartTT**. This confirms the paper
  benchmarks against BOTH prior software GPU-NFA engines AND real
  dedicated hardware (AP), a baseline breadth no other paper in this
  track matches.
- **timing protocol**: not confirmed in detail (no warmup/repetition/
  statistic named in the reachable README; `run_experiments_get_table3.sh`
  reports "several hours" total runtime across the whole benchmark suite,
  implying substantial per-configuration measurement time, but the
  internal repetition count per configuration was not inspected at the
  Python-harness level).
- **correctness**: not confirmed from the reachable README text (given
  the shared lineage with ngAP, which DOES validate against a serial CPU
  reference, a similar practice is plausible but not directly evidenced
  here).
- **hardware**: NVIDIA Quadro P6000 and Tesla V100 (compute capability
  >=5.0 expected to work generally); CUDA 9.2 SDK, tested on Ubuntu 18.04.

---

## Divergences

1. **Automata family: DFA vs NFA vs non-classical custom ISA.** GSpecPal
   is the track's only DFA-based paper (GPU-parallelized DFA-table
   traversal via speculation). ngAP, 0002PJ20, and MFSA are all
   NFA-based, with a DIRECT code/algorithm lineage: 0002PJ20 defines
   GPU-NFA and reimplements iNFAnt/NFA-CG; ngAP reuses 0002PJ20's own
   GPU-NFA scheme as a baseline (`obat`) and adds AsyncAP; MFSA extends
   iNFAnt (the same algorithm 0002PJ20 reimplements) into iMFAnt.
   Cicero is neither — REs compile to a custom non-backtracking hardware
   ISA that is functionally NFA-equivalent but never represented as an
   enumerated state-transition table at all. **Resolution**: the spec's
   `automata_family_disclosure` field is REQUIRED and open-ended (not a
   DFA/NFA enum alone) precisely because Cicero's architecture doesn't
   fit either classical label; DFA and NFA implementations get separate
   recommended input encodings (precompiled `.table` vs `.anml`/`.mnrl`)
   but share the same throughput-variant protocol, since both ultimately
   consume a byte stream and report match positions.

2. **Throughput metric: four different unit families, not
   interchangeable.** ngAP reports **MB/s** directly (confirmed from
   shipped example output). MFSA reports a **custom RE-count-weighted
   throughput** (`#REs * corpus_size / time`, Eq. 11), presented mostly as
   a **relative fold-improvement** against its own M=1 baseline rather
   than an absolute figure. GSpecPal's reachable driver code prints only
   **raw elapsed seconds**, with no throughput normalization visible in
   the artifact at all. Cicero reports **microseconds-per-single-match**
   plus **energy-per-match** — a latency/energy framing, not a
   corpus-throughput framing. 0002PJ20's "Table 3" is named "absolute
   throughput" but its exact unit was not confirmed from reachable
   sources. **Resolution**: the spec defines ONE canonical throughput
   unit (Gbps = corpus_bits_processed / matching_time_seconds / 1e9,
   matching the task's own requested convention) that every variant must
   report in ADDITION to whatever native unit an implementation prefers,
   with an explicit conversion note for MB/s (x8), for MFSA's relative
   fold-numbers (multiply by the M=1 baseline's own converted Gbps, which
   must ALSO be reported in absolute terms — not left as only a fold
   number), and for Cicero's per-match latency (report separately as a
   distinct, non-fungible "single-match latency" variant rather than
   forcing a corpus-Gbps conversion that would misrepresent an
   architecture never designed for streaming-corpus throughput).

3. **Pattern-set corpus lineage: shared union vs independently-sourced.**
   ANMLZoo + AutomataZoo + "Regex" (Becchi et al.) form a genuinely shared
   corpus across ngAP, 0002PJ20, and (for PowerEN/Bro217/Ranges1
   specifically) MFSA — confirmed by exact benchmark-name matches across
   their three separately-read config files/tables, not merely inferred
   from similar naming. GSpecPal's Snort/ClamAV/PowerEN DFA tables are a
   SEPARATE, non-ANML-packaged corpus (raw Aho-Consortium/DFA-table
   exports with paper-internal numeric size identifiers, no ANMLZoo/
   AutomataZoo file paths anywhere in its data directory). Cicero uses
   AutomataZoo per its AE checklist but does not enumerate which
   sub-benchmarks. **Resolution**: the spec's `recommended_subset`
   standardizes on the ANMLZoo+AutomataZoo+Regex union (naming the exact
   overlapping benchmarks: PowerEN, Bro217, Snort, ClamAV, CRISPR x2) as
   the primary suite for the NFA-family variants, since it is the only
   union with confirmed multi-paper reuse in this track, while keeping
   GSpecPal's Snort/ClamAV/PowerEN DFA-table corpus as an EXPLICIT
   additional named source for the DFA-family path (not silently merged,
   since despite sharing ruleset names, the two corpora are packaged and
   sized completely differently and are not bit-for-bit the same
   automata).

4. **Compilation/preprocessing scope: first-class metric vs shipped
   pre-compiled vs unconfirmed.** MFSA gives the track's only genuinely
   detailed, stage-by-stage compilation-time breakdown (front-end/AST/
   single-FSA-opt/merging, each independently averaged over 30 runs) and
   explicitly separates it from execution throughput (averaged over a
   DIFFERENT count, 15 runs). Cicero also treats compilation time as a
   first-class, separately-reported metric (part of its AE checklist).
   GSpecPal and ngAP both ship PRE-COMPILED automata (DFA `.table` files;
   `.anml`/`.mnrl` files respectively) with compilation/conversion
   entirely out of the timed matching loop — a "given, not timed" stance
   equivalent to how sparse-kernel papers often exclude format
   conversion. 0002PJ20 similarly ships pre-converted ANML. **Resolution**:
   the spec's end-to-end variant makes one-shot compilation/merging cost a
   REQUIRED separately-reported number (following MFSA's own excellent
   practice) with an explicit amortization count k, while the kernel-only
   variant explicitly documents "automaton supplied pre-compiled, given"
   as its scope — matching what GSpecPal, ngAP, and 0002PJ20 already do in
   practice, but stating it as policy rather than a silent omission.

5. **Warmup and repetition: named but inconsistent across and even within
   papers.** GSpecPal has warmup at TWO levels (3 discarded shell-level
   runs, plus an in-binary named "Total WarmUp time" phase) and repeats
   over 20 distinct input files per configuration (corpus-sample
   repetition, not same-input repeated trials). MFSA uses 30 repetitions
   for compilation-time measurement and a DIFFERENT count, 15 repetitions,
   for execution-throughput measurement — i.e. even a single paper isn't
   internally consistent about repetition count across its own two
   measured quantities. ngAP validates every single run against a CPU
   reference before counting it (a correctness-gate practice, not a
   repetition-count practice) but its throughput-run repetition count
   wasn't confirmed. Cicero and 0002PJ20 disclose neither in reachable
   sources. **Resolution**: the spec fixes ONE repetition count (10,
   between MFSA's two disclosed counts of 15 and 30, and comfortably above
   GSpecPal's 3-run shell-level warmup) with an explicit discarded-warmup
   requirement (at least 1 full pass, following GSpecPal's own two-level
   practice) as the floor for every variant, rather than inheriting any
   single paper's inconsistent choice.

6. **Correctness validation: present in some artifacts, absent from
   others' reachable evidence.** ngAP validates every run against a
   serial CPU reference (explicit "Validation PASS!" gate). GSpecPal
   validates against CPU results per-block with explicit pass/fail
   reporting. MFSA's correctness is a formal, design-time guarantee (the
   activation-function model provably prevents cross-language false
   matches) rather than a runtime empirical check. Cicero's and
   0002PJ20's correctness-validation procedures were not found in
   reachable sources. **Resolution**: the spec requires an explicit
   correctness gate — exact match-position-list equality against an
   independent single-threaded CPU/NFA-simulation reference — before any
   run's throughput counts, following ngAP's and GSpecPal's own practice,
   regardless of whether the paper under test documents one.

## Open questions

- 0002PJ20's exact "Table 3" throughput unit (Gbps vs MB/s vs another
  convention) could not be confirmed from the reachable README/config
  files; `abs_throughput_table.py`'s column headers were not inspected
  beyond the config-name mapping shown. Given the direct code lineage to
  ngAP (which uses MB/s), MB/s is the most likely convention but this is
  an inference, not a confirmed fact.
- GSpecPal's paper-reported throughput unit (if any — the reachable driver
  code prints only raw elapsed seconds) could not be confirmed; it is
  possible the IPDPS paper itself reports a derived Gbps/MB/s figure
  computed in post-processing scripts that were not part of the reachable
  artifact excerpt (only `script_release/*.sh` and `src/framework.cu` were
  inspected, not any Python/plotting post-processing).
- Cicero's paper-level fulltext (behind the ACM DL `doi.org` link) was not
  reached; whether the paper reports ANY aggregate corpus-throughput
  number alongside its per-match-latency framing is unconfirmed — the
  survey above relies entirely on the AE.md checklist's metric list,
  which may not be exhaustive of everything the paper itself reports.
- ngAP's exact per-configuration repetition count and warmup protocol for
  its `run-throughput.sh` experiment (Fig. 13/Table 4) were not confirmed;
  only the top-level README and one `app_spec` config file were read, not
  the `launch_exps.py` harness or `code/scripts/configs/exec_config_*`
  files that likely encode this.
- Whether MFSA's paper reports an absolute Gbps/MB/s number ANYWHERE
  (beyond the relative fold-improvement figures reached in the extracted
  text) is not fully confirmed — pages beyond the evaluation section
  (which was fully read) were not all inspected for a possible
  appendix/table with absolute throughput values.
- The exact relationship between GSpecPal's per-ruleset numeric size
  identifiers (e.g. `clamAV_DFA1104` through `clamAV_DFA2042`) and a
  human-meaningful quantity (number of DFA states? number of source
  regex patterns compiled together?) was not confirmed from the reachable
  filenames alone.
