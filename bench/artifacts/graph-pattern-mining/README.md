# graph-pattern-mining — artifact index

`benchspecs/graph-pattern-mining/spec.yaml`: exact k-clique COUNTING
(`params['clique_k']`, default 4 = K4), the spec's own cross-verified
common denominator across the surveyed pattern-mining literature (see
`bench/kernelbench/domains/graph.py`'s module docstring deviation (4)).
Metric: GEPS (edges/s, Fringe-SGC's own convention). CPU reference/impl:
`dag-kclique-count` (bitmask DAG-orientation counting,
`kernelbench/domains/graph.py`).

## Selection rule (revised 2026-09-05)

"Newest 3 open-source artifacts per track" is retired. Baselines are now
chosen from `kernel-papers/output/baseline_selection.md`
(`select_baselines.py`, driven by per-(paper,track) ratings in
`output/kernel_centrality.json`):

1. **kernel centrality** `core` (the kernel IS the paper's headline
   contribution, evaluated at kernel level) > `component` (a part of a
   bigger system/pipeline) — `tangential` is never a baseline;
2. **regime match** with the track spec's inputs (exact k-clique counting,
   single NVIDIA GPU): `matches` > `partial` > `mismatch`;
3. **single-NVIDIA-GPU path** required (current scope);
4. **recency** only as the tiebreak; up to 5 per track.

For graph-pattern-mining, `output/baseline_selection.md` names 5 core/
matches papers total, 3 already integrated in an earlier session
(`glumin`, `fringe-sgc`, `graphfold`) and 2 integrated in this session
(`graphset`, `stmatch`) — all 5 of the track's `[integrated]`/`to-integrate`
entries are now covered; none were skipped or demoted.

## Artifacts

| short | paper / venue | centrality / regime | status | notes |
|---|---|---|---|---|
| **glumin** | GLumin, PPoPP'25 (`conf/ppopp/CaoMLT25`) | core / matches | BUILT+GATED — exact K4/K5 on 2 smoke graphs + `ca-HepPh` (150281372, cross-validated) | k=4..7 (LUT-accelerated G2Miner CliqueSolver) |
| **fringe-sgc** | Fringe-SGC, SC'25 (`conf/sc/BradleyAB25`) | core / matches | BUILT+GATED — exact K4 on 2 smoke graphs + `ca-HepPh` (150281372, 3-way cross-validated) | K4 ONLY — core/fringe decomposition provably can't reach K5+ |
| **graphfold** | GraphFold, PPoPP'24 (`conf/ppopp/LinMSZXT24`) | core / matches | BUILT+GATED — exact K4/K5 on 2 smoke graphs + `ca-HepPh` (150281372) | k=4..7 (CFSolver) |
| **graphset** | GraphSet, SC'23 (`conf/sc/ShiZWCZHYC23`) | core / matches | BUILT+GATED — exact K4 on 2 smoke graphs + `ca-HepPh` (150281372, cross-validated) | equivalent-set-transformation (Schedule_IEP) driving a K_k `Pattern`; dedicated clique CLI in the artifact |
| **stmatch** | STMatch, SC'22 (`conf/sc/WeiJ22`) | core / matches | BUILT, gate FAILS — wrong count on a brute-force-verified controlled graph AND on `ca-HepPh` (34323168 vs. 150281372); real off-by-one found in the artifact's own `PatternPreprocessor::get_labels()` | general stack-based matcher driven by a K_k pattern file; not fixed (rule 3) |

`graphset` and `stmatch` were integrated in this session (full provenance,
patches, and gate evidence in each `STATUS.md`); `glumin`/`fringe-sgc`/
`graphfold` were integrated in an earlier session.

## Cross-artifact K4 ground truth

`glumin`, `fringe-sgc`, `graphfold`, and `graphset` — 4 independent
implementations, 4 different codebases — all report **K4(ca-HepPh) =
150,281,372**, each gated against the harness's own independent Python
reference on synthetic graphs and cross-validated against each other on
`ca-HepPh` (the domain's own combinatorial reference is too slow to
complete on a real 12K-vertex graph for K4 — see any of these `STATUS.md`
files' "Real-graph cross-validation" section). This 4-way agreement is
strong, artifact-independent evidence of the correct K4 count and is what
exposed `stmatch`'s wrong count as a genuine finding rather than a
reference-side issue.

## Cross-artifact findings from this session

**GraphSet**: one tracked-file linkage patch (`source/gpu/component/
utils.cuh`: `dev_alloc_and_copy`/`lower_bound` marked `static` — neither
was `inline`/`static` in the original header, harmless for the artifact's
own single-`.cu`-file builds but an ODR "multiple definition" nvlink error
once a second `.cu` file needs to launch its kernel via `-rdc=true`). Zero
behavior change; see `graphset/STATUS.md`.

**STMatch**: a genuine, reproducible off-by-one (heap out-of-bounds read)
in `PatternPreprocessor::get_labels()` (`source/src/pattern.h`:
`vertex_labels[i + 1]` reads one past the vector's last valid index on the
final loop iteration). Found while root-causing a wrong k-clique count that
this adapter's own Graph/JobQueue construction was independently ruled out
as the cause of (job-queue length matches an independent Python
reimplementation of the artifact's own filter, exactly). Not patched
(rule 3 — fixing it would change the artifact's own pattern-compiler
semantics, not merely a linkage/build knob); reported as a real
correctness finding, gate legitimately FAILS per rule 4.
