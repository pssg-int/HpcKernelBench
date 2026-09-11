# Domain-module audit: graph.py / primitives.py / solvers.py

Adversarial review of `kernelbench/domains/{graph.py,primitives.py,solvers.py}`
against their kernels' `benchspecs/<kernel>/spec.yaml`. Every finding below was
either read against the actual `harness.py` call order (`prepare()` →
correctness gate on the FIRST `run()` → warmup → timed reps, cost computed
from `params` after all reps) or verified empirically with
`/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python` (CPU only, scripts kept
under `/tmp/claude-106793/.../scratchpad/audit_check*.py`).

**Overall**: the reference-independence discipline is genuinely strong for
triangle-counting, connected-components and hash-table (real algorithm-family
diversity, empirically confirmed — see below), and the BFS/CC/CG correctness
gates hold up under adversarial testing (20k-trial canonicalization fuzz, a
positive-control corrupted-hash-slot test, an orientation-invariance test, a
no-early-exit CG test — all passed). The two BUGs found are both
**disclosure/self-description defects, not silent wrong numbers**: a
mislabeled push/pull axis on PageRank, and a GTEPS formula string that
doesn't match the number it's attached to. Both are one-line fixes.

Verdict counts: **2 BUG, 3 FAIRNESS, 4 NIT.**

---

## BUG 1 — PageRank `direction="push"` is actually a pull/gather implementation

`kernelbench/domains/graph.py:312-330` (class `NumpyPageRank`, `direction =
"push"`), `graph.py:342,346` (`AT = A.T.tocsr()` then `params["direction"] =
self.direction`), `graph.py:352-361` (`run()`: `AT @ (r * inv_outdeg)`).

The pagerank spec (`benchspecs/pagerank/spec.yaml`, `direction_disclosure`,
all 3 variants) defines the two options operationally, not just by name:

> "push (CSR, source scatters to destinations, typically atomic destination
> updates ...), pull (CSC, destination gathers from in-neighbors, **no
> atomics needed** since PageRank has no frontier ...)"

and its `notes_on_fairness` calls this exact axis out as high-stakes://
undisclosed push/pull choice can be "the kind of implementation-choice gap
that inflates apparent 'algorithmic' speedups if left undisclosed."

`NumpyPageRank.prepare()` materializes `AT = A.T.tocsr()` **once**, and
`run()` computes `AT @ v` every iteration. Structurally this is a CSC-over-A
gather: row *i* of `AT` is exactly *i*'s in-neighbor list, so the SpMV reads
(gathers) from in-neighbors and writes once per destination — zero atomics,
zero write conflicts, textbook "pull" per the spec's own definition above.
The class nonetheless declares `direction = "push"` and its docstring calls
it "CSR source-scatter via a one-shot transpose." Both are backwards: there
is no scatter and no atomic anywhere in this implementation.

This isn't cosmetic — it's the literal disclosure field the spec's own
fairness section warns readers to check before trusting a "speedup," and a
consumer of `RunResult.params["direction"]` would draw the wrong conclusion
from it every time this impl runs.

**Fix** (one line + a docstring edit, no algorithm change): `direction =
"pull"`, and reword the docstring's "CSR source-scatter" phrase to "CSC-of-A
(materialized transpose) gather, no atomics."

---

## BUG 2 — BFS's disclosed `gteps_definition` string doesn't match what `_cost_bfs` actually counts

`graph.py:264-266` (`params["gteps_definition"] = "edges connecting BFS-tree
vertices to their parents (both directions counted if undirected) /
search_time_s / 1e9"`) vs. `graph.py:183-193` (`_largest_weak_component`,
`edges = deg[idx].sum()` over **every** vertex in the giant component) and
`graph.py:300-308` (`_cost_bfs` uses that same `giant_component_edges` as the
GTEPS numerator).

The bfs spec's own `metric.primary` text is almost verbatim what's written
into `params["gteps_definition"]`: "GTEPS = (# edges connecting BFS-tree
vertices to their parents, summed over both directions if undirected) /
search_time_seconds / 1e9." Read literally, that's **spanning-tree edges**
(one edge per reached non-root vertex, `n_reached - 1`), not "every edge
incident to every reached vertex."

What `_cost_bfs` actually uses is `giant_component_edges`: the sum of
out-degree over **all** vertices in the giant weakly-connected component —
i.e. the total induced-subgraph edge count, independent of which specific
tree a BFS run happened to build. Empirically, on the `graph-smoke-powerlaw`
smoke graph (single connected component, 1500 vertices):

```
tree edges (root #0, n_reached-1):        1498
giant_component_edges used for GTEPS:    48082
ratio:                                     32.1x
```

The GTEPS number the harness reports is ~32x the value the disclosed formula
string would predict. (This is the standard, defensible Graph500/GAP-suite
GTEPS convention — total edges of the searched component, independent of
implementation-specific tree shape — so the *number itself* is not
indefensible. The bug is that `params["gteps_definition"]` — a field whose
entire purpose, per `harness.py`'s own docstring, is "every constant used is
copied into the result record ... so a result can be audited against its
spec without rerunning it" — describes a different, much smaller quantity
than the one actually reported.)

**Fix**: reword the disclosed string to describe what's computed (e.g. "sum
of degree over every vertex in the searched weakly-connected component, both
directions counted if undirected — NOT spanning-tree-only edges"), or, if the
project wants the literal spec text honored, switch `_cost_bfs` to use
`n_reached - 1` (cheap: `reference_bfs`'s own `parent` array already has this
for root #0; a per-root count would need one popcount over the tree output,
not a full re-traversal).

---

## FAIRNESS 3 — PageRank's reference is the same formula at higher precision, not an independent code path

`graph.py:352-361` (`NumpyPageRank.run`) vs. `graph.py:373-395`
(`reference_pagerank`).

Contrast with `SpgemmTriangleCount`/`reference_triangle_count`
(`graph.py:429-506`), whose own docstring states the point explicitly: the
reference uses "Python sets, not scipy sparse matmul, so a bug in scipy's
sparse product isn't silently replicated in both sides of the gate." PageRank
gets no equivalent protection — `reference_pagerank` is a line-for-line copy
of `NumpyPageRank.run()`'s push/pull formula (same `AT @ (r * inv_outdeg)`,
same dangling-mass term, same `base` term), just re-typed at fp64. A
*design-level* misunderstanding shared by both hand-written copies (wrong
dangling-redistribution divisor, damping factor applied to the wrong term,
etc.) would pass the gate on both sides, undetected — the class of bug this
project's own house style (triangle-counting, CC's Kosaraju-vs-Tarjan,
hash-table's dict-vs-open-addressing) is normally built to catch.

Empirically, I could not find an actual bug this way: I wrote a **third**,
genuinely independent implementation (Python dict/adjacency-list, pull-style,
no scipy sparse ops at all) and it agrees with `reference_pagerank` to
`6.9e-18` after k=20 iterations and `9.5e-18` after k=2000 (full
convergence) on the smoke graph — the shared formula is mathematically
correct. So this is a real gap in the gate's power, not an active bug.

**Cheap fix**: swap in something like the dict-based pull reference used for
this verification (or `networkx.pagerank` with `tol` set tight, run for
exactly k iterations by catching/absorbing its convergence check) as the
`reference_pagerank` implementation.

---

## FAIRNESS 4 — BFS single-root gate: severity assessment + cheap partial fix

`graph.py:18-32` (module docstring, deviation 1) already discloses this at
length, so it is not a new finding — but the task asked for a severity
assessment and a cheap-fix proposal, so both follow.

**How bad is it**: `harness.run_variant()` calls `impl.run(handle)` exactly
once for the correctness gate, then `warmup` times, then `reps` times — for
`bfs-kernel-real-graphs` (`warmup=1, reps=64`), that's 66 total calls against
a root list that `ScipyBFS` cycles through via `h["cursor"]`. Only call #0
(root `roots[0]`) is ever diffed against `reference_bfs`. Traced the cursor
arithmetic: because 66 calls against 64 roots wrap around, all 64 roots *do*
get exercised somewhere across the run (just shifted, since calls 0-1 are
consumed by the correctness check + warmup) — but only `roots[0]`'s output is
ever independently checked. **63 of the 64 timed searches are validated
zero times.** A direction-optimizing implementation that e.g. mishandles the
bottom-up→top-down switch threshold on some roots but not others (a real,
plausible bug class for exactly this kind of code) would produce wrong
*timed* GTEPS with no correctness signal at all.

**Cheap fix that doesn't require the harness change the module docstring
explicitly rules out** (re-validating all 64 roots inside `prepare()`, which
would inflate `preprocessing_ms` by ~64x): validate a small, fixed-size
**rotating subsample** inside `prepare()` instead of either extreme — e.g. 4
roots spread across the cycle (indices `0, 16, 32, 48`, or a
`hash(graph.name) % 64` offset so different graphs in a batch exercise
different roots over a full run). This is O(4) extra BFS traversals, not
O(64), and would have caught a root-dependent bug with much higher
probability than the current O(1) check, at a preprocessing-time cost the
module's own stated objection doesn't apply to.

---

## FAIRNESS 5 — preconditioner's mandatory 3rd axis (iteration-count effect) is vacuous on every smoke workload

`solvers.py:191-195` (`SMOKE`, all 3 entries build via `poisson_matrix`) vs.
`solvers.py:371-410` (`_measure_iteration_effect`, the spec's mandatory
preconditioned-vs-unpreconditioned iteration-count comparison).

All three Poisson smoke matrices have an **exactly constant diagonal** (4.0
for both 2D grids, 6.0 for the 3D grid — verified empirically). For
`JacobiPreconditioner`, `M = diag(A)^-1` is therefore a scalar multiple of
the identity, which is provably indistinguishable from no preconditioner at
all to any Krylov method (BiCGSTAB's search directions are invariant to a
constant rescaling). Confirmed empirically on `smoke-poisson2d-24`:

```
outer_iters_preconditioned:            48
outer_iters_baseline_unpreconditioned: 48
outer_relres_preconditioned:  9.437722618431714e-07
outer_relres_baseline:        9.437722618431714e-07   <- bit-identical
```

So the spec's mandatory "does this preconditioner buy iterations" axis
always reports `outer_iteration_delta_vs_baseline == 0` for
`JacobiPreconditioner` on every smoke input, regardless of whether the
implementation is correct — a wrong Jacobi impl that e.g. used `1/diag^2`
would *also* show delta=0 here (still a scalar rescale), so this axis
provides zero implementation-bug protection on smoke data. `ILU0Preconditioner`
is unaffected (ILU(0) fill-in is not diagonal-only, and the same script
confirms a real 33-vs-48 iteration win, matching the module's own docstring
example). This is a smoke-suite-only limitation — real (non-smoke)
SuiteSparse SPD matrices from the recommended_subset lists have non-constant
diagonals and would exercise this axis properly — so it does not affect
spec-conforming runs, only local smoke-test coverage of `JacobiPreconditioner`
specifically.

---

## NIT 6 — PageRank's `GFLOP/s` cost rule is a self-invented metric, not "per the specs" as claimed, and silently drops the spec's own required GTEPS number

`graph.py:11-12` (module docstring: "GFLOP/s (2*nnz*iterations) for
pagerank, per the specs") vs. `graph.py:398-408` (`_cost_pagerank`,
`flops = 2 * g.nnz * k`, `workload.register_cost("pagerank", _cost_pagerank,
"GFLOP/s")`).

`benchspecs/pagerank/spec.yaml` never defines a `GFLOP/s` metric for
PageRank anywhere. Its actual metric fields are `primary: "ms per
iteration"` and `secondary: "GTEPS = (nnz * k) / total_time_seconds / 1e9"`
— note **no factor of 2**, and named GTEPS, not GFLOP/s. `2*nnz*k` (mul+add
per nonzero, the project's general SpMV-flop convention used elsewhere) is a
reasonable additional metric to report, but it is not what the spec calls
for, and the module never computes or reports the spec's own required GTEPS
number at all. "Per the specs" overclaims; the actual convention is this
module's own choice, undisclosed as a deviation the way the file's other 3
numbered deviations are.

## NIT 7 — scan-reduction reference is the same `np.sum`/`np.cumsum` call at higher precision

`primitives.py:408-431` (`reference_scan_reduction`) vs. `primitives.py:434-509`
(`NumpyReduction`/`NumpyScan`). Same "compare same formula at two
precisions" pattern as BUG/FAIRNESS 3 above, but far lower materiality: a
flat sum/cumsum has no design space to get wrong beyond precision/reassociation,
which is exactly what the `max_scaled_err` tolerance is built to tolerate
(well-argued in the module docstring). Noted for completeness, not treated as
a real risk.

## NIT 8 — hash-table's Zipf access pattern is generated via `zipf() % num_keys`, which distorts the intended skew

`primitives.py:260-269` (`_gen_trace_idx`): `raw = rng.zipf(a=1.5, size=...) -
1; return raw % num_keys`. Values of the Zipf draw that exceed `num_keys`
alias back onto low indices via modulo (e.g. draws of `num_keys+1`,
`2*num_keys+1`, ... all wrap onto index 0), which is a well-known anti-pattern
for turning an unbounded Zipf variable into a bounded key-index — it can
inflate the apparent hotness of low indices beyond what a properly
truncated/rejection-sampled Zipf-over-`[0,num_keys)` would produce. Still
correctly labeled `key_distribution="zipf"` (not silently mislabeled), and
for the `a=1.5` shape parameter most mass is already on very small values (P
(v=1)≈0.38), so the practical distortion is modest — flagged as a NIT, not a
correctness violation.

## NIT 9 — Jacobi preconditioner's reference re-derives the same trivial divide with the same seeded `r`

`solvers.py:283-291` (`_reference_jacobi_apply`) vs. `solvers.py:550-564`
(`JacobiPreconditioner.prepare`/`run`). Same "self-comparison" shape as NIT
7/FAIRNESS 3, again with low materiality since `z = r/diag` has no
meaningful alternate algorithm to independently verify against — this one is
closer to "verifying the wiring" (right `A`, right `r`) than "verifying an
algorithm," which the module's `ILU0Preconditioner` reference (a genuinely
fresh `spla.spilu()` factorization object, independent SuperLU instance) does
correctly for the one case here that actually has algorithmic room to be
wrong.

---

## What checked out clean (adversarial testing, no counterexample found)

- **CC partition canonicalization** (`_canonical_labels`, `graph.py:524-536`):
  20,000 random-trial fuzz (same-partition relabelings, and single-vertex
  partition perturbations checked against true partition-equality) found
  **zero** cases of two different partitions canonicalizing identically, or
  the same partition canonicalizing differently. Also cross-checked
  `_kosaraju_scc`/`_union_find_components` against `networkx`'s
  `strongly_connected_components`/`weakly_connected_components` on a random
  directed graph (200 vertices, density 0.02): exact partition match both
  ways.
- **Triangle-counting DAG orientation invariance** (`_orient_by_degree`,
  `graph.py:412-426`): confirmed the Cohen/forward-algorithm count is
  provably invariant to which total order is used — verified empirically by
  re-running the SAME graph with a uniform-random total order instead of
  degree-ascending: both give 578, matching an independent `networkx.triangles`
  ground truth and the module's own set-intersection reference.
- **Topk tie rule**: `argpartition`-based selection is invariant to which
  tied index is chosen (only the *value multiset* is gated, matching the
  spec's own correctness text), and the index-validity assertion
  (bounds/uniqueness) runs on every call including the correctness-gate call.
- **Hash-table gate genuinely exercises the probed table**: positive-control
  test — corrupting a slot in the live `tk`/`tv` arrays whose key IS touched
  by the op trace flips the gate to fail (confirmed); an earlier attempt that
  corrupted an untouched slot correctly showed no effect (the trace simply
  never probes that key, not a vacuous-gate bug).
- **CG fixed-iter really is fixed**: `rtol=atol=0.0` forces `maxiter=5/50/200`
  to run exactly that many iterations every time regardless of achieved
  residual (verified directly against `scipy.sparse.linalg.cg`), confirming
  no early exit is possible on the "fixed-iter" path.
- **Residual computed fresh, not scipy's internal tracking**: confirmed by
  reading `ScipyCG.to_host()` — recomputes `b64 - A64@x` from scratch every
  time, never touches scipy's `info`/callback residual estimate.
- **Preconditioner outer-iteration-effect baseline is fair**: same solver
  (BiCGSTAB), same `maxiter`/`rtol`, same seeded RHS, differing only in
  `M=`; `params["outer_solver"]="bicgstab"` is disclosed in the result
  record as the task requested. Also reproduced the module docstring's own
  claim that CG+ILU0 is a real (not hypothetical) failure mode: plain
  `scipy.sparse.linalg.cg` with the nonsymmetric ILU(0) factorization as `M`
  hits the 500-iteration cap without converging (relres ≈ 7.97e-2),
  confirming BiCGSTAB is the correct, necessary choice for this probe.
- **Preprocessing placement/determinism**: all synthetic generators
  (`matrices.synthetic`, `primitives._make_primitive`/`_gen_keys`/
  `_gen_trace_idx`, `solvers.poisson_matrix`) use fixed, explicit seeds
  (project-wide default `20260806`, or a workload's own `seed` field) with no
  reliance on global numpy RNG state — reran several of these to confirm
  bit-identical outputs across calls.
- **BFS root-cursor mechanics**: `reference_bfs` reconstructs the identical
  64-root list (same seed, deterministic `np.random.default_rng`) and checks
  exactly the root that `ScipyBFS`'s first `run()` call actually searches
  (empirically confirmed: root #0 match = True).
