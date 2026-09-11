# Sequence-alignment track — evaluation methodology survey

Track input: `data/track_inputs/sequence-alignment.json`, 6 papers (all
surveyed — below the 6-paper track total). 3 via true arXiv/PDF fulltext
(AnySeq — arXiv PDF read directly; LOGAN — ar5iv fulltext; SegAlign — PDF
fetched from the authors' own hosting and text-extracted with `pypdf`, since
it has no arXiv id), 1 via artifact-repo README that itself reproduces the
paper's headline evaluation tables with citations (MASA-CUDAlign/MultiBP), 1
via artifact-repo benchmark scripts + kernel source read with `gh api`
(futhark-mem-sc22's `nw` benchmark), 1 via artifact-repo README only — no
arXiv/OA and the ACM DOI page is paywalled/forthcoming (FastAlign, PPoPP'26).

Operation surveyed: pairwise dynamic-programming sequence alignment
(Smith-Waterman local / Needleman-Wunsch global / semi-global, exact or
X-drop-approximate), plus two end-to-end genome-mapping pipelines that use a
pairwise-alignment kernel as one stage of a larger seed→filter/index→extend
pipeline.

---

## 1. Müller, Schmidt, Hildebrandt, Membarth, Leißa, Kruse, Hack — "AnySeq: A High Performance Sequence Alignment Library based on Partial Evaluation" (IPDPS'20)

- key: `conf/ipps/MullerS0MLKH20`, arXiv:2002.04561 (fulltext PDF read directly)
- **algorithm/scoring disclosure**: generic exact pairwise DP (Smith-Waterman/
  Needleman-Wunsch family) with compile-time-selectable alignment type
  (global/local/semi-global), gap scheme (linear or affine), and output mode
  (score-only vs full traceback). Scoring scheme explicitly given: match =
  +2, mismatch = -1, linear gap = -1; affine gap Go = -2, Ge = -1 "if
  supported" by the compared library. This is the most fully disclosed
  scoring scheme of the 6 papers.
- **workloads**: (a) 6 named real GenBank accessions, aligned as 3 pairs of
  "roughly similar length," reused from prior literature — M. tuberculosis
  H37Rv (4,411,532 bp), E. coli K12 MG1655 (4,641,652 bp), D. melanogaster
  chr2L (23,011,544 bp), P. troglodytes chr22 (32,799,110 bp), O. aries
  chr24 (42,034,648 bp), O. aries chr21 (50,073,674 bp); (b) 12.5 million
  pairwise alignments of 150 bp Illumina reads, simulated with Mason using
  GRCh38 chromosome 10 as reference.
- **timing protocol**: "median performance" reported (Figures 5-6) —
  repetition count not stated, but this is the only one of the 6 papers to
  name a statistic at all.
- **timing scope**: compute-only; AnyDSL specialization happens at compile
  time and is not counted in the runtime GCUPS numbers. GPU alignment uses
  32-bit score arithmetic (no native short-int support); CPU AVX
  variants use 16-bit scores per SIMD lane.
- **precision & correctness**: integer DP scores; no explicit tolerance
  section — correctness is implicit (all compared libraries compute the same
  well-defined recurrence, so an exact score match is the only outcome
  reported, not stated as an explicit test).
- **metric**: median GCUPS (giga cell updates/second, `n*m` cells) for
  score-only and traceback, separately for linear/affine gap; also
  GCUPS/watt energy efficiency (device TDP from spec sheet, not measured
  power) comparing 2xXeon Gold 6130 (125W), Titan V (250W), Xilinx ZCU104
  (6.181W per synthesis report).
- **baselines**: SeqAn 2.4.0, Parasail 2.0 (CPU); NVBio 1.1 (GPU). AnySeq is
  "at most 7% slower, up to 12% faster" than SeqAn/NVBio.
- **hardware**: 2x Intel Xeon Gold 6130 (32 threads total), Titan V GPU,
  Xilinx Zynq UltraScale+ ZCU104 FPGA (187.5 MHz).
- source: arXiv fulltext PDF (arxiv.org/pdf/2002.04561), full paper text
  including Section V (Performance Evaluation) read directly.

## 2. Zeni, Guidi, Ellis, Ding, Santambrogio, Hofmeyr, Buluç, Oliker, Yelick — "LOGAN: High-Performance GPU-Based X-Drop Long-Read Alignment" (IPDPS'20)

- key: `conf/ipps/ZeniGEDSHBOY20`, arXiv:2002.05200 (fulltext via
  ar5iv.labs.arxiv.org)
- **algorithm/scoring disclosure**: X-drop — an APPROXIMATE, pruned banded
  local-alignment heuristic (not exact Smith-Waterman); first GPU port of
  X-drop. Exact substitution-matrix and gap-penalty values not found in the
  accessible text (open question). X-drop threshold swept over X ∈ {10, 20,
  50, 100, 500, 1000, 2500, 5000}.
- **workloads**: synthetic — 100K read pairs, length 2,500-7,500 characters,
  ~15% simulated error rate; real — integrated into the BELLA long-read
  overlapper pipeline and run on E. coli (1.8M alignments required) and
  C. elegans (235M alignments required) datasets.
- **timing protocol**: no warmup/repetition/statistic disclosed in the
  accessible text; wall-clock seconds reported per X value for the 100K-pair
  benchmark (single-run numbers in the results tables).
- **timing scope**: ambiguous — the pairwise-alignment step is measured, but
  it is unclear whether CPU-side sequence reversal/buffering before the GPU
  call is included in the reported GPU time.
- **precision & correctness**: "equivalent accuracy"/"equivalent results"
  claimed vs. SeqAn's and ksw2's own X-drop implementations run with the same
  parameters — NOT compared against the exact/unbounded Smith-Waterman
  optimum, since X-drop is a deliberate approximation.
- **metric**: GCUPS ("up to 181.6 GCUPS"); the exact cell-counting convention
  (nominal full band vs. actually-visited cells) is not disclosed.
- **baselines**: SeqAn X-drop (CPU, 168 threads on 2x POWER9), ksw2/minimap2
  X-drop (CPU, 80 threads on 2x Skylake Xeon Gold 6148); related-work GPU
  competitors CUDASW++3, manymap. Reports 6.6x (1 GPU) / 30.7x (6 GPUs)
  speedup vs. the CPU baseline, and 2.3x vs. ksw2, improving overall BELLA
  runtime by up to 10.6x.
- **hardware**: NVIDIA Tesla V100 (16GB HBM2), 1-8 GPUs.
- source: ar5iv fulltext (ar5iv.labs.arxiv.org/html/2002.05200) plus repo
  README (github.com/albertozeni/LOGAN) which independently republishes the
  SeqAn/ksw2 comparison tables.

## 3. Goenka, Turakhia, Paten, Horowitz — "SegAlign: a scalable GPU-based whole genome aligner" (SC'20)

- key: `conf/sc/GoenkaTPH20`, no arXiv id
- **algorithm/scoring disclosure**: GPU-accelerated seed-filter-extend
  whole-genome aligner that reimplements LASTZ's seeding/filtering stage on
  GPU (98% of LASTZ's own runtime for human-mouse WGA) while the gapped
  EXTEND stage still spawns the original LASTZ binary as a CPU subprocess per
  HSP segment file, pipelined with the GPU stage. Scoring is inherited from
  LASTZ's default configuration, not independently re-disclosed.
- **workloads**: real UCSC genome assemblies only — human hg38 (3.08 Gbp),
  mouse mm10 (2.72 Gbp), chicken galGal6 (1.05 Gbp), zebra finch taeGut2
  (1.02 Gbp), pre-processed to drop mitochondrial/unplaced/ALT sequences.
  3 whole-genome-alignment pairs evaluated: human-mouse, human-chicken,
  chicken-zebra finch (spans a range of evolutionary distances and genome
  size products).
- **timing protocol**: single-run wall-clock time per genome pair per AWS
  instance type; no repetition count or variance reported.
- **timing scope**: preprocessing (per-target-chromosome seed-position-table
  / index-table construction, reused across all query chromosomes aligned to
  that target) is explicitly profiled and reported SEPARATELY from the
  steady-state seed+filter+extend pipeline — the paper even reports a 2-3x
  speedup on index construction itself from an atomic-add optimization.
- **precision & correctness**: NOT a bit-exact match to the baseline. Verified
  via an R dot-plot comparison of SegAlign vs. LASTZ alignments on human-mouse
  chr1: SegAlign recovers EVERY LASTZ alignment (no missed/"green" points)
  plus additional ones LASTZ misses, because SegAlign does not implement
  LASTZ's diagonal-hashing heuristic. The correctness bar is therefore
  "superset of the baseline's output," not equality.
- **metric**: (1) filter-stage throughput in million seed-hits/sec (305M/s on
  1 V100, ~180x a single LASTZ CPU core — no GCUPS used for this stage since
  it is not a DP recurrence); (2) end-to-end wall-clock speedup vs. LASTZ,
  13.5-14x across all 3 species pairs; (3) cost ($/instance-hour)-normalized
  speedup, 2.25-2.33x, tied to specific 2020 AWS on-demand pricing.
- **baselines**: LASTZ v1.04.03, default settings, parallelized across cores
  with GNU `parallel` on 10 Mbp chunks (10 Kbp overlap) on a 96-vCPU AWS
  c5.24xlarge instance ($4.08/hr — the instance type that gave LASTZ its best
  throughput among those tested).
- **hardware**: AWS p3.2xlarge/p3.8xlarge/p3.16xlarge (1/4/8 V100 GPUs,
  8/32/64 vCPUs); multi-node Apache Spark cluster for weak/strong scaling.
- source: PDF fetched from public.gi.ucsc.edu/~yatisht/files/segalign-sc20.pdf
  (author-hosted full paper, no arXiv id exists), text-extracted with
  `pypdf`; Sections IV (Methodology) and V (Results) read directly. Repo
  README (github.com/gsneha26/SegAlign) cross-checked for usage/dependencies.

## 4. Figueiredo, Navarro, Sandes, Teodoro, Melo — "Parallel Fine-Grained Comparison of Long DNA Sequences in Homogeneous and Heterogeneous GPU Platforms With Pruning" (TPDS'21) — MultiBP / MASA-CUDAlign

- key: `journals/tpds/FigueiredoNSTM21`, no arXiv id (IEEE Xplore paywalled;
  abstract + artifact README used)
- **algorithm/scoring disclosure**: EXACT Smith-Waterman (local) /
  Needleman-Wunsch (global) combined with Myers-Miller linear-space
  traceback (the MASA architecture); this paper adds "MultiBP" block
  pruning across multiple GPUs — static score-sharing (workload statically
  distributed, best score broadcast to neighbor GPUs to simulate a global
  view) and dynamic (execution divided into cycles, workload reassigned by
  GPU processing rate). Block pruning is provably safe (cannot discard the
  optimal-score region), so it does not change the reported alignment score
  versus the unpruned CUDAlign. Exact match/mismatch/gap-penalty values used
  by this specific paper were not found in the available sources (open
  question — the MASA-Core command-line accepts configurable scoring but the
  paper's chosen values are not visible from the README).
- **workloads**: real, unrestricted-size DNA chromosome pairs, e.g.
  NC_000001.9 (human chr1, 249 Mbp) vs. NC_006468.3 (228 Mbp) as the
  headline TPDS'21 result.
- **timing protocol**: single-run wall-clock time to solution per
  configuration (e.g. "11m", "53m8s") — no repetitions/warmup/variance
  disclosed, presumably because full chromosome-scale, 512-GPU-cluster runs
  are too expensive to repeat.
- **timing/hardware scope — a fairness flag**: the artifact README's
  "Performance Benchmarks" table aggregates the "best results in different
  scenarios" (the README's own words) from SEVEN different published papers
  spanning 2010-2021 and five distinct GPU generations/cluster scales — from
  a single GTX 560 Ti (25.82-50.70 GCUPS) up to a 512x V100 cluster (82,822
  GCUPS, this paper's own number) — explicitly framed as showcasing the best
  number per scenario rather than a single fixed, reproducible hardware
  configuration. This is a best-of-N-across-incomparable-hardware pattern.
- **precision & correctness**: implicit exact-DP bit-exact score (pruning is
  provably lossless by construction); no separate correctness section beyond
  algorithmic argument.
- **metric**: GCUPS (`n*m` cells / time), the headline number for this
  paper's own contribution being 82,822 GCUPS in 11 minutes on 512x V100 for
  the 249 Mbp human-chr1-vs-chicken pair.
- **baselines**: implicitly the prior single/few-GPU CUDAlign versions
  (CUDAlign 3.0/4.0) and the paper's own predecessor result of 10,370 GCUPS
  on a 384x Tesla M2090 cluster (from the TPDS'16 CUDAlign 4.0 paper, cited
  in the same README table but NOT this paper's own contribution).
- source: artifact repo README (github.com/edanssandes/MASA-CUDAlign), which
  is maintained by the same author group and directly republishes this
  paper's headline table with a citation; abstract (from track input JSON)
  cross-checked. Full paper text (IEEE Xplore) was not accessible.

## 5. Munksgaard, Henriksen, Sivertsen, Oancea — "Memory Optimizations in an Array Language" (SC'22)

- key: `conf/sc/MunksgaardHSO22`, no arXiv id (compiler paper, not
  sequence-alignment-specific)
- **relevance caveat**: this is NOT a sequence-alignment paper. It is a
  compiler/memory-layout-optimization paper for the Futhark functional array
  language, evaluated on 7 unrelated benchmark kernels (LocVolCalib,
  OptionPricing, hotspot, lbm, lud, nn, and `nw` — Needleman-Wunsch). NW is
  used purely as one memory-bound dynamic-programming stress test among
  seven, not as a bioinformatics contribution; the paper's claim (1.1x-2x
  speedup from memory optimizations) is about compiler-generated code
  quality, not alignment algorithm novelty.
- **algorithm/scoring disclosure**: read directly from the artifact
  (`benchmarks/nw/futhark/nw.fut` and `benchmarks/nw/reference/nw.c`, the
  latter being verbatim Rodinia's OpenMP/OpenCL `nw` kernel, credited in a
  code comment: "based on
  https://github.com/kkushagra/rodinia/blob/master/openmp/nw"). Global
  Needleman-Wunsch alignment, BLOSUM62 24x24 protein substitution matrix
  (not DNA), single-parameter LINEAR gap penalty (`pen`, default 10 per
  Rodinia convention), tiled anti-diagonal wavefront with block_size=64.
- **workloads**: purely synthetic — random sequences, sizes given directly
  in the `.fut` compiled-script header as `row_length` ∈ {8192, 16384,
  32768}. No real biological data, no correctness claim beyond matching the
  Rodinia reference C/OpenCL implementation bit-for-bit.
- **timing protocol**: the top-level `benchmarks/Makefile` supports a
  `RUNS=N` override ("you can specify how many executions of each benchmark
  you want to use with e.g. `RUNS=10`"), default value not visible in the
  fetched files; results are cached per-benchmark to avoid re-running.
- **timing scope**: pure DP-kernel compute; no indexing/format
  conversion concept applies (protein string data only).
- **precision & correctness**: implicit exact match to the Rodinia C
  reference (both are exact DP with no approximation).
- **metric**: whichever `result-table.py` reports (not fetched in detail;
  likely runtime/throughput ratio Futhark-vs-reference, since the paper's
  point is optimized-vs-baseline speedup, not an absolute GCUPS number).
- **baselines**: the Rodinia OpenMP/OpenCL `nw.c`/`nw.cl` reference
  implementation, run on the same GPU/CPU.
- source: repo file contents via `gh api`
  (diku-dk/futhark-mem-sc22/benchmarks/nw/{futhark/nw.fut,reference/nw.c},
  benchmarks/Makefile, root README.md).

## 6. Zhang, Li, Meng, Zhang, Tan — "Faster and Cheaper: Pushing the Sequence Alignment Throughput with Commercial CPUs" (PPoPP'26) — FastAlign / BWA-FastAlign

- key: `conf/ppopp/ZhangLMZT26`, no arXiv id; ACM DOI (10.1145/3774934.3786421)
  points to a forthcoming/paywalled PPoPP'26 proceedings entry — full paper
  text was not accessible from any source tried (arXiv, ar5iv, web search,
  author lab pages). All facts below are sourced from the artifact repo
  README, which itself restates several evaluation claims from the abstract.
- **algorithm/scoring disclosure**: drop-in replacement for BWA-MEM's
  seed-and-extend pipeline. Two contributions: (1) Multi-stage Seeding /
  "Hybrid Index" combining a Kmer-Index, an FMT-Index (enhanced FM-Index
  with prefetching), and a Direct-Index, dynamically switched by seed length
  and match density; (2) intra-query parallel seed-extension — Smith-
  Waterman-based extension parallelized WITHIN a single query (vs.
  BWA-MEM2's inter-query parallelism, which the paper argues load-imbalances
  on variable-length reads), using AVX2 SIMD, dynamic pruning of
  zero-alignment-score branches, and a sliding-window scheme to reduce
  memory-gather cost. Scoring scheme is inherited from BWA-MEM by
  construction (guaranteed-identical-output requirement, see correctness
  below).
- **workloads**: repo's quick-test example downloads a real E. coli
  reference genome (GCA_000005845.2_ASM584v2) and real SRA reads
  (SRR2584863); this is presented as a minimal usage demo, not necessarily
  the paper's actual benchmark scale. The abstract states results hold for
  both WGS and WES data and "commercial CPU servers," implying human-scale
  genome data, but the exact dataset(s)/sizes/read counts used in the
  paper's own evaluation are an OPEN QUESTION — not recoverable from any
  source tried.
- **timing protocol**: not disclosed in any accessible source.
- **timing scope**: index build (`./fastalign index ref.fa`) is an explicit,
  separate command from alignment (`./fastalign mem ...`) — i.e. the tool's
  own interface already treats indexing as a one-shot, amortized step
  distinct from per-run mapping throughput.
- **precision & correctness**: the strongest correctness bar of all 6 papers
  — "100% identical output (SAM/BAM)" to BWA-MEM, i.e. bit-exact match to a
  reference tool's output, not just to a theoretical DP optimum. (Note: since
  BWA-MEM's own pipeline already uses heuristic seeding+chaining ahead of an
  exact SW extension, "identical to BWA-MEM" means matching a heuristic
  pipeline's output, not the globally optimal alignment.)
- **metric**: throughput speedup 2.27x-3.28x vs. BWA-MEM; cost reduction
  2.54x-5.65x vs. "state-of-the-art CPU and GPU baselines" (BWA-MEM2,
  BWA-GPU, and ERT-BWA-MEM2 named as compared systems); memory-efficiency
  metric "bases processed per GB per second" improved 18.92x by the
  Hybrid-Index seeding stage; SIMD utilization improved 3.45x by the
  intra-query extension stage.
- **baselines**: BWA-MEM (correctness reference and primary throughput
  baseline), BWA-MEM2, BWA-GPU, ERT-BWA-MEM2 (cost/throughput baselines).
- **hardware**: "standard CPU servers"/"commercial CPUs," AVX2-capable,
  tested on Ubuntu 22.04 per build instructions; exact core count/model used
  in the paper's evaluation not disclosed in the repo.
- source: repo README (github.com/zzhofict/BWA-FastAlign), ACM DOI page
  title/venue metadata (dl.acm.org/doi/10.1145/3774934.3786421, abstract
  only), track input JSON abstract.

---

## Divergences

1. **Exact vs. heuristic/approximate correctness class.** AnySeq,
   MASA-CUDAlign/MultiBP, and the futhark-mem-sc22 `nw` benchmark all compute
   the EXACT optimal DP score (block pruning in MultiBP is provably lossless,
   not an approximation). LOGAN computes a deliberately approximate X-drop
   score and validates only against another X-drop implementation, not
   against full Smith-Waterman. SegAlign's filter stage is a heuristic
   seed-matching filter feeding an exact LASTZ extension, validated as
   "superset of baseline output" rather than equality. FastAlign is exact
   -equivalent to BWA-MEM's own (partly heuristic) pipeline output. A single
   correctness gate cannot cover all three classes — the spec below splits
   variants by correctness class rather than forcing one tolerance rule.
2. **Real vs. synthetic data.** SegAlign and MASA-CUDAlign/MultiBP use
   exclusively real reference genomes/chromosomes. AnySeq mixes real long
   genomes with Mason-simulated short reads. LOGAN mixes synthetic
   error-injected long reads with real E. coli/C. elegans data (used inside
   BELLA, not as standalone alignment benchmarks). futhark-mem-sc22 uses only
   synthetic random protein-alphabet sequences. FastAlign's paper-scale
   dataset is unknown; its repo demo uses real but tiny (E. coli-scale) data.
3. **Scoring-scheme disclosure.** AnySeq fully discloses match/mismatch/
   gap-open/gap-extend values (DNA). futhark-mem-sc22 uses BLOSUM62 + linear
   gap (PROTEIN, not DNA — a materially different scoring domain).
   MASA-CUDAlign/MultiBP's and LOGAN's exact scoring values used in these
   specific papers were not recoverable from available sources. SegAlign and
   FastAlign both inherit scoring from an external reference tool (LASTZ,
   BWA-MEM respectively) rather than disclosing independent values.
4. **GCUPS metric definition is not standardized across the exact/heuristic
   boundary.** GCUPS = `n*m`/time is unambiguous for exact, unpruned DP
   (AnySeq). It becomes ambiguous once pruning or banding is introduced:
   MASA-CUDAlign/MultiBP's GCUPS number is not clearly stated as nominal
   (full `n*m`) or effective (only computed cells after pruning) in the
   available sources; LOGAN's X-drop GCUPS has the same ambiguity, since the
   band searched is parameter- and data-dependent. SegAlign does not use
   GCUPS at all for its dominant filter stage — it reports "seed hits/sec"
   because that stage is not a DP recurrence.
5. **Preprocessing/index scope.** FastAlign and SegAlign both have a
   genuine, explicit one-shot index/table-construction phase, separated from
   per-alignment work, and both papers report it distinctly (SegAlign even
   optimizes and profiles it as its own contribution). AnySeq, LOGAN, and
   MASA-CUDAlign/MultiBP have no indexing concept at all — pure pairwise DP
   over given sequences. futhark-mem-sc22 likewise has no indexing concept.
6. **Cost-normalized ($/hour) metrics are not reproducible over time.**
   SegAlign and FastAlign both report cost-efficiency improvements tied to
   specific (2020- and unstated-date, respectively) cloud instance pricing.
   Prices change; this metric cannot be independently re-verified by a
   benchmark harness running today without re-pricing against current rates,
   and comparing a 2020-priced number to a 2026-priced number would be
   invalid. The spec treats cost-normalized numbers as non-gating appendix
   data only.
7. **Repetition/statistic disclosure is a near-universal gap.** Only AnySeq
   explicitly names a statistic ("median performance," repetition count still
   unstated). The other five papers report what appear to be single-run
   wall-clock/GCUPS numbers with no stated repetition count or variance —
   understandable for the largest chromosome-/genome-scale runs (hours on a
   512-GPU cluster) but not disclosed even for the smaller-scale results
   (e.g. LOGAN's 100K-pair synthetic benchmark, cheap enough to repeat).
8. **Best-of-N across incomparable hardware within a single reported table.**
   The MASA-CUDAlign README (which restates the MultiBP/TPDS'21 paper's own
   headline result) aggregates "best results in different scenarios" spanning
   five hardware generations and 2010-2021 publication dates in one table,
   explicitly per its own text. This is the kind of best-of-N presentation
   the benchmark spec must not inherit — variant 1 below fixes a single named
   hardware class and requires the full pair-vs-GPU-count table, not a
   cherry-picked best cell.
