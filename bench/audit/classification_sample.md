# Phase-1 classification audit — independent re-judge, n=60

Method: fixed reproducible sample (seed 20260807, see command below), 30 papers
drawn from `is_kernel_opt=true` and 30 from `is_kernel_opt=false` in
`output/papers.json`. For each, I read title+abstract+categories/kernels/
excluded_reason/one_liner, formed my own verdict from `classify_instructions.md`
+ `verify_instructions.md` FIRST, then compared to the recorded verdict.

Sampling command (for reproducibility):
```
/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python -c "
import json, random
papers = json.load(open('output/papers.json'))
rng = random.Random(20260807)
inc = [p for p in papers if p['is_kernel_opt']]
exc = [p for p in papers if not p['is_kernel_opt']]
for p in rng.sample(inc, 30): print('INC', p['key'])
for p in rng.sample(exc, 30): print('EXC', p['key'])
"
```

## Disagreements

### 1. `conf/ipps/GoodrichG26` — recorded INCLUDE (high conf, primitives/sort), my verdict EXCLUDE (or at best `unclear`/low-conf)

"Parallel Integer and Learning-Augmented Sorting Algorithms in the
Binary-Forking Model." The abstract is purely asymptotic-complexity work —
`O(log n)` span / `O(n)` work bounds "in the binary-forking model," with **no
mention of any implementation, benchmark, hardware target, or measured
speedup**. INCLUDE criterion #1 requires "a faster/better software
implementation of a computational kernel **on real hardware**"; nothing in the
abstract establishes that this paper contains a runnable implementation rather
than a pure theory-of-parallel-algorithms contribution (bounds proofs in an
abstract cost model, in the vein of PRAM-style papers). Sorting is a
legitimate `primitives` kernel in general, but this specific paper's
described contribution doesn't clear the "real hardware, measurable" bar the
other 29 sampled INCLUDE papers all clear. I'd have called this `unclear`
(low confidence) at best, more likely EXCLUDE — a benchmark can't "re-measure"
a paper that never reports a measurement.

### 2. `conf/cgo/SommerA022` — recorded INCLUDE (medium conf, ml_kernels), my verdict leans EXCLUDE (borderline)

"SPNC: An Open-Source MLIR-Based Compiler for Fast Sum-Product Network
Inference on CPUs and GPUs." This does satisfy the codegen/compiler INCLUDE
criterion mechanically (MLIR compiler generating vectorized native code,
evaluated by kernel speed, "multiple orders of magnitude" over existing SPN
libraries) — that's why I'm not fully confident in the flip. But Sum-Product
Networks are a probabilistic graphical model, in the same family as the
Bayesian networks that `verify_instructions.md` boundary ruling #5 says to
exclude "unless recast as one of the canonical kernel categories." SPN
inference (weighted sum-product DAG evaluation) doesn't match any of the
`ml_kernels` examples given (tensor-core GEMM, GNN aggregation, scientific
convolution, sparse attention), nor any other category cleanly. The recorded
"medium" confidence suggests the original classifier felt the same tension.
On balance I'd lean EXCLUDE (`other`) by analogy to the Bayesian-network
ruling, but flag this as a genuinely close call either way — not a clear
miss like #1.

## Non-disagreements worth flagging as close calls (did NOT flip, for transparency)

- `conf/ics/0036SMLS23` (EXCLUDE, `application`, low conf) — "kernel
  [statistical] regression" software for million-scale datasets. The word
  "kernel" here is a false-friend (RBF/statistical kernel methods, not a
  computational kernel), and the abstract genuinely gives no algorithm name,
  platform, or speedup number to hang an INCLUDE verdict on. Low-confidence
  EXCLUDE / `unclear` is the defensible default when evidence is this thin;
  agreed, not counted as a disagreement.
- `conf/sc/MosesCPHNSD21` (EXCLUDE, `other`) — Enzyme, an LLVM AD plugin that
  generates gradient GPU kernels. Tempting to read as codegen-produces-kernels
  INCLUDE, but the paper's own framing is about correctness/feasibility of
  automatic differentiation with bounded overhead ("within an order of
  magnitude of the original"), not about producing a *faster* kernel of any
  named canonical type. Agreed with EXCLUDE.
- `conf/cgo/SuiLLYLZZXZX26` (EXCLUDE, `other`) — FHEFusion, an FHE-DNN
  operator-fusion compiler with concrete kernel-level speedups (up to 3.02x).
  Close to the NTT/crypto-kernel boundary ruling #1, but the contribution is
  graph-level operator fusion (reducing multiplicative depth), explicitly
  benchmarked against another *graph-fusion* framework (NGRAPH) — i.e. a
  systems/compiler-pass contribution analogous to ML-serving graph
  optimization, not a direct NTT/poly-mult kernel implementation. Agreed with
  EXCLUDE, but a closer call than most.

## All 60 verdicts (verbatim summary)

Included sample (30) — 28/30 agreed, 2 disagreements (`GoodrichG26`,
`SommerA022`, both over-inclusions/false positives).

Excluded sample (30) — 30/30 agreed (0 disagreements), with 3 close calls
noted above that I ultimately did not flip.

## Rate estimates (n=30 each, caveat: small sample, wide CI)

- **Included-set false-positive rate**: 2/30 ≈ **6.7%** (95% CI roughly
  1–22%, Wilson interval — n=30 is too small to pin down precisely). Both
  flagged cases are genuinely-close boundary calls, not obvious errors: one
  is a theory paper with no reported implementation that shouldn't have
  cleared the "real hardware" bar; the other is a defensible-but-arguable
  category stretch (SPN inference under `ml_kernels`).
- **Excluded-set false-negative rate**: 0/30 = **0%** (95% CI roughly
  0–11%). No excluded paper in the sample should have been flipped to
  INCLUDE. The exclusion machinery (12 excluded_reason buckets, the 6
  boundary rulings for NTT/GNN/FPGA/DB/Bayesian-SAT/quantum) is being
  applied consistently and correctly across a wide variety of adjacent-looking
  systems papers (storage, networking, EDA, security, ML-serving) that
  mention "kernel"-adjacent language without a real kernel contribution.

## Overall assessment

The Phase-1 classification holds up well under independent re-judging. The
only systematic risk visible in this sample is a slight looseness on the
INCLUDE side at the edges of `ml_kernels`/`primitives` — i.e. very rare
theory-only or hard-to-categorize contributions being swept in — not any
weakness on the EXCLUDE side (no evidence of missed kernel papers among 30
excluded samples). This is the safer direction to err in for Phase 2 (an
over-included candidate is filtered out again at the artifact/build stage;
an under-included one is invisible and permanently lost), so no corrective
action is strictly required, but the two flagged INCLUDE keys are worth a
manual second look before they're used to anchor Phase-2 benchmark selection.
