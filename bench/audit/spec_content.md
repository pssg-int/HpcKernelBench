# Benchmark spec content audit

Adversarial review of 12 `benchspecs/*/spec.yaml` + `survey.md` pairs spanning
domains and authorship waves: spmm, gemm, stencil, fft, bfs, lossy-compression,
cg-krylov, attention-kernel, mttkrp, sptrsv, ann-search, qr.

**Overall**: this is a very high-quality batch. Every spec traces its design
choices to specific, quoted survey evidence; `notes_on_fairness` sections are
concrete (not boilerplate) in all 12; `open_questions` sections honestly flag
what wasn't confirmed rather than papering over gaps. All SuiteSparse matrix
names checked against `bench/matrices/ssstats.csv` (format
`group,name,rows,cols,nnz,...`) exist — **zero hallucinated/typo'd matrix
names found** across spmm, cg-krylov, sptrsv, qr, and bfs's SNAP-overlapping
real-graph list (spot-checked 60+ names total, all resolved). That said,
adversarial reading surfaced concrete, fixable defects in about half the
specs: two direct self-contradictions between prose and data
(sptrsv, spmm), one cross-track citation that doesn't exist in its own
track's survey (stencil/CLAIRE), one internal claim-vs-protocol conflict
(attention-kernel's mask axis), and several untraceable-to-survey numeric
specifics (bfs, lossy-compression, mttkrp).

Verdict counts: **SOLID 5, MINOR ISSUES 7, NEEDS WORK 0.**

---

## spmm — MINOR ISSUES

**Consistency violation (concrete):** `spmm-cpu-kernel-f32`'s `selection`
field states the CPU variant is "restricted to nnz <= 2e7 (CPU-appropriate
scale...)", but its own `recommended_subset` includes `circuit5M`, whose
actual nnz is **59,524,291** (confirmed in `ssstats.csv`, Freescale group) —
**~3x over the stated cap**. This looks like `circuit5M` was carried over
from the GPU variant's list (where it's correctly justified via SparseLNR)
without re-checking the CPU size cap.

**Survey traceability:** spot-checked 3 claims (GE-SpMM's disabled
`VALIDATE` macro, RoDe's separate-preprocessing-CSV practice, Magicube's
per-iteration CUDA events) — all confirmed verbatim in survey.md. The
`recommended_subset`'s "orkut" entry isn't restated in survey.md's final
"Recommended-subset provenance" summary paragraph (which only lists
cora/citeseer/pubmed/amazon/youtube from FusedMM), but IS traceable to
FusedMM's own workload list earlier in the same survey section — a
documentation gap, not a fabrication.

**Fairness notes:** honest and specific (e.g. the componentwise
backward-error-denominator finding, explicitly dated "2026-08-06," reads as
a genuine implementation discovery, not boilerplate).

**Actionability — top ambiguity:** the correctness gate's denominator
`|out-ref| / (|A|*|B|)` uses matrix-elementwise-absolute-value notation
(`|A|` = elementwise `abs`, then matrix-multiplied) without spelling this
out; an implementer unfamiliar with the componentwise-backward-error
convention could easily misread it as a scalar norm.

**SuiteSparse names:** mip1, shipsec1, pdb1HYS, consph, cant, cop20k_A, dc2,
rma10, conf5_4-8x8-10, circuit5M, web-BerkStan — all verified present and
correctly sized.

---

## gemm — SOLID

No internal contradictions found. Reps/warmup rules are explicit
generalizations of specific papers' own practice (MPGEMM's manual
500/100/20 nreps scaling → an explicit min/max-bounded rule), correctly
cited. Spot-checked 3 claims (WuZLHJWC23's 1024-6144 step-512 sweep,
CL-DB-GEMM's "multiples of 128/129," MPGEMM's per-size nreps values) — all
match survey.md exactly. `notes_on_fairness` is specific (e.g. min-vs-median
statistic critique, cache-state fairness fix for LibShalom's finding).
`gemm-distributed-e2e` honestly keeps CA3DMM's native mean statistic
(diverging from the other 3 variants' median) because the tool doesn't
expose raw per-run samples — explicitly flagged rather than silently forced
into the median convention. Checked all 4 variants' `recommended_subset`
entries against their own numeric `selection` inclusion rules — no
boundary violations found (e.g. `gemm-small-irregular-kernel`'s LLM-shape
entries with N/K in the tens of thousands still satisfy "one of M,N <= 512"
via M=64).

**Actionability — top ambiguity:** `gemm-small-irregular-kernel`'s "cold"
cache-state protocol says to touch "a companion buffer larger than LLC...
between each timed call to evict the operands" without specifying how much
larger or what access pattern guarantees eviction on modern
set-associative/prefetching caches — two implementations could produce
different "cold" numbers.

---

## stencil — MINOR ISSUES

**Traceability failure (concrete):** `stencil-cpu-gpu-kernel-fp64`'s
`grid_sizes` field justifies the 512³ 3D grid size as "AN5D's 3D convention
== Devito's 512³ velocity model, == CLAIRE's 256³-512³ range." **CLAIRE does
not appear anywhere in `stencil/survey.md`** — grepped for "claire"
case-insensitively, zero hits. CLAIRE (Brunn et al., SC'20, diffeomorphic
image registration) is a real paper, but it belongs to this project's
**fft** and **cg-krylov** tracks (its own survey.md files in both cite it
in depth), not stencil. AN5D and Devito are legitimately stencil-track
papers, so the underlying 512³ choice is still independently grounded — but
the CLAIRE citation is a fabricated/misattributed piece of evidence
smuggled into the stencil spec's justification chain, most likely from an
author who also wrote the fft/cg-krylov specs and cross-contaminated a
grid-size convention across tracks without checking whether CLAIRE was
actually surveyed here.

**Survey traceability (otherwise solid):** spot-checked 3 further claims
(SPIDER's per-launch `cudaDeviceSynchronize()` sync-granularity flaw,
ConvStencil's `CHECK_ERROR` compiled out by default, WindStencil's
kernel-only-vs-full-application split) — all confirmed verbatim.

**Fairness notes:** specific and evidence-tied throughout (8 divergences,
each with a named paper and mechanism).

**Actionability — top ambiguity:** the `operation` and protocol both allow
`boundary: periodic wrap or fixed halo of width r` without specifying which
to use for the `recommended_subset` points — an implementer has no way to
know which boundary condition a given benchmark run should use, and the two
choices are not performance-equivalent (fixed halo requires actual boundary
handling logic inside the timed region's assumptions, periodic doesn't).

---

## fft — SOLID

Extremely well-grounded; every protocol departure from surveyed practice is
explicitly labeled "THIS IS A DELIBERATE FIX" with the specific paper/file
cited. Spot-checked 3 claims (TurboFFT's per-launch device-sync inside its
`ntest` loop, FLUPS's three-way one-shot-cost separation, cuHPX's
`ellmax=3*nside-1≈3071` mixed-radix figure) — all confirmed.
`fft-distributed-alltoall`'s GPU sub-case assumes CLAIRE's single-GPU 256³
headline number as the "per-GPU" weak-scaling unit; this is a reasonable
inference from a genuinely ambiguous source (open_questions explicitly
flags CLAIRE's timing protocol/precision/comm-primitive as unconfirmed,
though it doesn't separately flag this specific per-GPU-size inference).

**Actionability — top ambiguity:** the GPU sub-case's "device-direct
communication (GPU-aware MPI / NCCL-style transport, CLAIRE's own described
but not further specified mechanism — flagged open question on the exact
primitive)" leaves an implementer no way to choose among genuinely
different communication implementations that would produce different
numbers — honestly flagged as open, but still blocks a literal reproduction.

---

## bfs — MINOR ISSUES

**Data inconsistency (concrete):** the `bfs-kernel-real-graphs`
`recommended_subset` lists `"com-LiveJournal (SNAP, ~4.8M v / 69M e,
undirected)"`. SuiteSparse's actual `com-LiveJournal` (SNAP group) has
**3,997,962 rows** (~4.0M, not 4.8M) with nnz 69,362,378 (edge count matches).
The ~4.8M vertex figure instead matches **`soc-LiveJournal1`** (4,847,571
rows) — a *different* matrix already listed separately three lines below in
the same `recommended_subset`. This reads as the vertex count for one
dataset being copy-pasted onto the other's description. Minor knock-on: at
its real ~4.85M count, `soc-LiveJournal1` sits just under the spec's own
`size_classes` "small: <5M vertices" boundary but is bucketed as "medium"
— a boundary-labeling nit, not a real defect.

**Survey traceability:** spot-checked 3 claims (efg's `frndster`
65.61M v/3.61B e, Corder's single-max-degree-root convention and its own
"not skewed" `urand` exclusion, RTNN's measured <0.001% approximation error)
— all confirmed verbatim, including the 65.6M/3.61B figures matching
SuiteSparse's `com-Friendster` exactly.

**Fairness notes:** unusually strong — each of the 5 bullets names the
specific paper practice being corrected (Corder's max-degree root singled
out as "the single most favorable and least representative choice
possible," with a stated mechanism, not just an assertion).

**Actionability — top ambiguity:** "uniform-random from the giant
component, fixed RNG seed = 0... same 64 roots reused across all
implementations" doesn't specify which PRNG algorithm to seed — different
languages/libraries seeded with "0" produce different root sets, undermining
the stated cross-implementation comparability goal that this exact clause
exists to guarantee.

**SuiteSparse names:** europe_osm, com-DBLP, com-LiveJournal, com-Orkut,
com-Friendster, soc-LiveJournal1, wiki-topcats, webbase-2001, it-2004,
uk-2005 all verified present (all 10 spot-checked resolve, both edge counts
and — except com-LiveJournal above — vertex counts match).

---

## lossy-compression — MINOR ISSUES

**Traceability gap:** several `lossy-comp-kernel-cpu-ebound`
`recommended_subset` entries state precise dimensions/element-counts not
recoverable from survey.md's text — e.g. `"EXAALT (molecular dynamics, 1D
2,869,440 elements, fp64)"` and `"SCALE-LETKF (climate DA, 3D
98x1200x1200, fp32)"`. survey.md's SDRBench catalog section only gives
size ranges in GB/MB for these two datasets ("EXAALT (MD, 60 MB–2.4 GB)",
"SCALE-LETKF (climate DA, 4.9 GB)"), with no per-field dimension anywhere
in the document. These specific numbers may well be correct (likely pulled
from SDRBench's own per-dataset pages during spec authoring) but are not
independently verifiable from this project's own survey artifact, unlike
most other entries in the same list (HACC/CESM-ATM/NYX dimensions do match
survey.md's FZ-GPU section verbatim).

**Design strength worth noting explicitly:** `lossy-comp-gpu-dual-scope`
is a genuinely good fix for exactly the "e2e variant that silently mismatches
its claim" trap this audit was told to look for — cuSZp/FZ-GPU's two
incompatible senses of "end-to-end" (fused-kernel-only vs.
PCIe-transfer-inclusive) are resolved by naming BOTH scopes explicitly and
banning the bare term "end-to-end" from the spec, rather than picking one
and silently contradicting the other paper family's usage.

**Survey traceability:** spot-checked 3 claims (TAC vs. TAC+'s
preprocessing-amortization divergence, FAZ's `-F 0` ablation mirroring
QoZ's `-q 0`, lsCOMP's confirmed 3-iteration GPU warmup) — all verified.

**Actionability — top ambiguity:** SDRBench datasets ship as specific
files/timesteps on the SDRBench portal, often with multiple snapshots or
precision variants per named field; the spec names fields
(`NYX/baryon_density`, etc.) but not a specific file/URL/checksum, so two
implementers downloading "the same" named dataset could get different
files.

---

## cg-krylov — SOLID

The strongest-grounded spec in the batch. Every protocol number traces to
a specific, quoted artifact fact, including two real bugs found by reading
code (PERKS's single-event-bracket-over-all-iterations, and Mille-feuille's
hardcoded `while(iterations<10000)` loop that ignores its own `maxiter=10`
argument plus a `/100` divisor bug) — both bugs are accurately reflected in
the spec's design rationale (mandating genuine per-iteration timestamps and
a loop-counter readback specifically to catch this class of error).
Spot-checked 3 claims against survey.md (SPCG's separate CUDA-event
brackets for "Preconditioning Time" vs "PCG Time," BootCMatchGX's
`rtol=1e-6`/`itnlim=2000` relative-residual convention, F3R's fp64/fp32/fp16
three-way sweep) — all confirmed verbatim. `notes_on_fairness` correctly
distinguishes SPCG's absolute/unnormalized tolerance (rejected, not
scale-invariant) from BootCMatchGX's relative tolerance (adopted) with a
clear rationale. Checked the recommended_subset's nnz values (largest:
Queen_4147 ~3.17e8, smallest: bcsstk18 ~1.5e5) against the stated `[1e4,
4e8]` range — no boundary violations.

**Actionability — top ambiguity:** `cg-hpcg-generator-amg`'s IPU
(Graphene) row assumes BootCMatchGX's `rtol=1e-6`/`itnlim=2000` defaults
"by analogy... unconfirmed for Graphene specifically" (explicitly flagged
in open_questions) — an implementer running the actual Graphene JSON-config
solver has no confirmed default to target and might silently use different
convergence criteria than every other platform row in the same variant.

---

## attention-kernel — MINOR ISSUES

**Internal contradiction (concrete):** `attn-prefill-kernel-fp16bf16`'s
`inputs.mask` field states `"causal AND bidirectional both required as
separate sub-runs per shape"`, and the `metric.primary` FLOP formula is
explicitly mask-conditional (`(mask==bidirectional ? 4 : 2) * ...`),
implying every shape must be run under both masks. But the
`recommended_subset` table assigns exactly **one** fixed `mask:` value per
named shape (e.g. `bert-base-short` → `mask: bidirectional` only,
`llama2-7b-mha` → `mask: causal` only) with no shape appearing twice under
different masks. A literal reader can't tell whether to (a) run each of the
12 shapes once, under its single labeled mask, or (b) run all 12 shapes
under both masks (24 sub-runs) as the prose and metric formula imply. This
directly undercuts the stated purpose (`notes_on_fairness`: "this spec adds
mask as an explicit required sub-case... so the ~2x FLOP difference is
visible") since the table as written never actually exercises the same
shape under both masks.

**Survey traceability:** spot-checked 3 claims (PAT's `batch=1134`
GPU-saturation rationale, MEATTEN's `compareTensor()` existing but never
called from the perf driver, FlashAttention-T's unconfirmed warmup/repeat
count inside its compiled binary) — all confirmed, including correctly
carrying an "unconfirmed" fact through as an open question rather than
asserting it.

**Fairness notes:** strong and specific throughout (7 bullets, each citing
a concrete artifact behavior).

**Actionability — top ambiguity:** the mask contradiction above is also
the actionability blocker — an implementer cannot determine the actual
run matrix (12 vs. 24 configurations) from the spec alone.

---

## mttkrp — MINOR ISSUES

**Traceability gap:** `mttkrp-symmetric-kernel-fp64`'s `recommended_subset`
gives precise per-dataset stats for 3 of 5 hypergraph tensors —
`"contact-school (order 5, dim 245, unnz 12704, rank 12)"`,
`"trivago-clicks (order 6, dim 154987, unnz 208076, rank 4)"`,
`"walmart-trips (order 8, dim 62240, unnz 47560, rank 10)"` — but
survey.md's SymProp section only gives aggregate ranges for the whole
5-dataset set ("order 5 to 12, dims up to 2.5M, UNNZ up to 740K"), never
breaking out individual dataset numbers. As with lossy-compression's
EXAALT entry, these specific figures are plausible (likely read directly
from SymProp's own Table III, not fully summarized in survey.md) but are
not independently verifiable from this project's survey artifact.

**Survey traceability (otherwise strong):** spot-checked 3 further claims
(BLCO's ~12-iteration format-construction amortization, WACO's
fastest-of-top-10-candidates practice flagged as a mild best-of-N issue,
BLCO's up-to-12x per-mode variance motivating all-mode reporting) — all
confirmed verbatim.

**Fairness notes:** specific and well-reasoned (correctly distinguishes
SySTeC/SymProp's mode-agnosticism as a genuine mathematical property of
symmetric input, not an evaluation shortcut, from WACO's mode-0-only choice
which IS flagged as insufficient evidence for a general MTTKRP claim).

**Actionability — top ambiguity:** no fixed random seed is specified for
the dense factor-matrix initialization (`"R (rank) in {16, 32, 64}, fp64,
initialized U(-1,1)"`, no seed given) — unlike spmm/cg-krylov/sptrsv in
this same batch, which all pin an explicit seed for their random operands.

---

## sptrsv — MINOR ISSUES (worst finding of the batch)

**Direct self-contradiction (concrete, the most severe single defect
found in this audit):** `sptrsv-solve-kernel`'s `selection` field states
verbatim: *"Matrices used ONLY by the distributed variant's 1e9-nnz-class
LU factors (nlpkkt80, Ga19As19H42, dielFilterV3real) are **deliberately
excluded here**..."* — but the very next field, `recommended_subset`, ends
with exactly those three names: `[...  ldoor, nlpkkt80, dielFilterV3real,
Ga19As19H42]`. The prose says excluded; the list includes them. This isn't
a subtle inference gap — it's a same-block, adjacent-field contradiction an
implementer would hit immediately. It propagates: both
`sptrsv-analysis-phase` and `sptrsv-e2e-amortized` inherit
`recommended_subset: "same as sptrsv-solve-kernel"`, so the contradiction
affects 3 of the spec's 4 variants. Net effect: an implementer following
the list literally will benchmark 3 large (~1e9-nnz-LU-factor-class)
matrices using the CANONICAL structural-proxy triangular-factor method
(unit diagonal, values from A's own lower triangle) — a construction the
spec's own text says is inappropriate for exactly these 3 matrices, since
their whole reason for being in the track at all is real LU fill-in, which
the structural proxy doesn't produce.

**Minor secondary nuance:** the same `triangular_factor_derivation` field
also states YuenyeungSpTRSV is "the only one of the three [surveyed
extraction methods] that preserves the input matrix's actual numeric
structure" — but survey.md documents that `partially-strided-codelet` (a
4th, separately-surveyed paper) *also* keeps original values ("kept as-is
in #3/#5"), so YuenyeungSpTRSV is not actually unique in this respect
among the papers survey.md covers; it's one of (at least) two.

**Survey traceability (otherwise excellent):** spot-checked 3 further
claims (SC'23's `berr` computed-but-never-gated flaw, Split_SpTRSV's
analysis phase being re-measured every one of its 10 timing iterations,
the 4-of-5-papers structural-extraction-not-real-factorization finding
that is this track's central methodological discovery) — all confirmed
verbatim, and the underlying discovery (most SpTRSV papers silently
benchmark on a fake "L" that was never actually factorized) is a genuinely
valuable, well-evidenced catch this spec gets right everywhere except the
one contradiction above.

**Fairness notes:** strong, specific, correctly distinguishes the 3
single-device structural-proxy variants from the 1 distributed variant's
genuine-LU-factor requirement.

**Actionability — secondary concern:** the correctness gate for
`sptrsv-solve-kernel` requires `< 1e-9` relative error (fp64) against the
structural-proxy `L` (unit diagonal, off-diagonal values taken as-is from
`A`). Because this `L` was never produced by an actual factorization, its
condition number is uncontrolled — for a SuiteSparse matrix whose
lower-triangular entries are large relative to 1 (the unit-diagonal
value), the resulting triangular system could be poorly conditioned enough
that no correct implementation achieves `1e-9` in fp64. This is a plausible
risk given the spec's own construction choice, not a confirmed failure,
but worth flagging before treating the tolerance as universally achievable
across the full `recommended_subset`.

**SuiteSparse names:** apache2, ecology2, thermal2, parabolic_fem,
offshore, G3_circuit, StocF-1465, Serena, Hook_1498, Geo_1438, audikw_1,
boneS10, af_shell8, cage14, torso3, crashbasis, Bump_2911, cant, pwtk,
bcsstk36, ldoor, nlpkkt80, dielFilterV3real, Ga19As19H42 — all 24 verified
present in ssstats.csv (the names themselves are fine; the problem is
which variant's list they should be in).

---

## ann-search — SOLID

Correctly identifies and preserves the track's core structural split
(high-dim approximate SIFT/DEEP/SPACEV vs. low-dim exact spatial k-NN)
rather than forcing a shared metric onto both. Spot-checked 3 claims
(CLOVER's `run_test()` unconditional `return true;` before its real
assertions, X-HD's `-check=false` on every reproducibility run, DRIM-ANN's
explicit `recall@10 >= 0.8` floor being the only named recall gate in the
track) — all confirmed verbatim. The mandatory recall-vs-QPS-curve
requirement directly closes UpANNS's documented gap (exposes
`NPROBS`/`TOPK` as tunable but never surfaces resulting recall in the same
script). Point-set sizes in `exact-spatial-knn-kernel`'s recommended_subset
(Stanford Bunny 360K, Asian Dragon 3.6M, Millennium N-body 9M) match
survey.md's RTNN section exactly.

**Actionability — top ambiguity:** the `exact-spatial-knn-kernel`
`recommended_subset`'s `clover-mesh` entry has `N: "per-mesh vertex
count"` — a placeholder, not an actual value — and open_questions honestly
admits "the actual mesh identities/sizes were not enumerated... uses a
generic placeholder rather than named files." Honestly flagged, but still
means this one recommended-subset entry cannot be implemented as written;
an implementer has no way to know which CLOVER mesh files to fetch.

---

## qr — SOLID

The correctness-gate design is the standout: NO paper in this 4-paper
track computes `||Q^T*Q - I||` orthogonality anywhere in its own
performance-measurement path, and the spec makes this a required, gating
check in every variant rather than treating the gap as optional — flagged
explicitly and correctly as "the single largest gap this survey found."
Spot-checked 3 claims (PAQR's ScaLAPACK-arm two-warmup-then-single-shot
convention vs. its LAPACK arm's zero warmup, GPTuneCrowd's strict-minimum-
over-`niter` statistic, STM-QR's forward-error check computed but never
gated against its own `tol`) — all confirmed verbatim, including the exact
code-level mechanism (`if (mytime < times[idxparam])`) for GPTuneCrowd's
best-of-N practice. Honestly documents a genuine literature gap (no paper
in the track implements TSQR/CAQR) rather than fabricating a baseline for
it, and states this explicitly in both `notes_on_fairness` and
`open_questions` rather than only one place.

**Actionability — top ambiguity:** `qr-sparse-multifrontal-kernel`'s
`adaptive/GCN-selected` reordering strategy requires LinYWT021's trained
GCN classifier to actually select a strategy at run time, but the spec's
own `recommended_subset` is the 16-matrix reproducible demo list, while the
GCN classifier's train/test data is a *separate* 408-matrix set
(`GCNdata_408.txt`, per survey.md) not included anywhere in the spec — an
implementer wanting to reproduce the "adaptive" arm has no path to obtain
or train the classifier the spec asks them to disclose the choice of.

---

## Summary table

| spec | verdict | headline concrete issue |
|---|---|---|
| spmm | MINOR ISSUES | `circuit5M` (nnz 5.95e7) in CPU variant's `recommended_subset` violates that variant's own stated `nnz <= 2e7` cap |
| gemm | SOLID | — |
| stencil | MINOR ISSUES | cites "CLAIRE" as grid-size evidence; CLAIRE never appears in stencil/survey.md (it's an fft/cg-krylov paper) |
| fft | SOLID | — |
| bfs | MINOR ISSUES | `com-LiveJournal` mislabeled ~4.8M v (actual ~4.0M; the 4.8M figure belongs to the separately-listed `soc-LiveJournal1`) |
| lossy-compression | MINOR ISSUES | several dataset dims (EXAALT, SCALE-LETKF) not traceable to survey.md text |
| cg-krylov | SOLID | — |
| attention-kernel | MINOR ISSUES | `mask` field claims both causal+bidirectional required per shape; `recommended_subset` assigns only one mask per shape |
| mttkrp | MINOR ISSUES | 3 symmetric-tensor per-dataset stats not traceable to survey.md's aggregate-only description |
| sptrsv | MINOR ISSUES (worst overall) | `selection` text says 3 named matrices are "deliberately excluded," `recommended_subset` includes exactly those 3, propagated into 2 more variants |
| ann-search | SOLID | — |
| qr | SOLID | — |

**Totals: SOLID 5 (gemm, fft, cg-krylov, ann-search, qr), MINOR ISSUES 7
(spmm, stencil, bfs, lossy-compression, attention-kernel, mttkrp, sptrsv),
NEEDS WORK 0.**
